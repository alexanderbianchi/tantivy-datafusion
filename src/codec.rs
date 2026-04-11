//! Codec for serializing tantivy-df [`DataSourceExec`] nodes across
//! distributed executors.
//!
//! The codec is **pure serialization** — it encodes/decodes plan nodes
//! as protobuf bytes. The runtime resource (opener factory) lives on
//! the [`SessionConfig`] as an extension, NOT in the codec.
//!
//! # Setup
//!
//! ```ignore
//! use tantivy_datafusion::{TantivyCodec, OpenerFactoryExt};
//!
//! // On each worker:
//! let mut builder = SessionStateBuilder::new();
//! builder.set_opener_factory(Arc::new(|meta| { ... }));
//! builder.set_distributed_user_codec(TantivyCodec);
//! ```

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use prost::Message;

use crate::index_opener::{IndexOpener, OpenerMetadata};
use crate::schema_mapping::tantivy_schema_to_arrow_with_multi_valued;
use crate::type_coercion::plan_fast_field_projection;
use crate::unified::agg_data_source::{AggDataSource, AggOutputMode};
use crate::unified::single_table_provider::{
    deserialize_fast_field_filter_exprs, deserialize_fast_field_filters,
    serialize_fast_field_filters, PartitionSpec, ScanSchema, SingleTableDataSource,
    SplitExecutionPlan,
};

// ── Opener factory as session extension ─────────────────────────────

/// Function that reconstructs an [`IndexOpener`] on a remote worker
/// from serialized metadata.
pub type OpenerFactory = Arc<dyn Fn(OpenerMetadata) -> Arc<dyn IndexOpener> + Send + Sync>;

/// Wrapper for storing the [`OpenerFactory`] on the [`SessionConfig`].
struct OpenerFactoryExtension(OpenerFactory);

/// Extension trait for registering/retrieving the opener factory on a
/// [`SessionConfig`] (or anything that wraps one like `SessionStateBuilder`).
pub trait OpenerFactoryExt {
    fn set_opener_factory(&mut self, factory: OpenerFactory);
    fn get_opener_factory(&self) -> Option<OpenerFactory>;
}

impl OpenerFactoryExt for SessionConfig {
    fn set_opener_factory(&mut self, factory: OpenerFactory) {
        self.set_extension(Arc::new(OpenerFactoryExtension(factory)));
    }

    fn get_opener_factory(&self) -> Option<OpenerFactory> {
        self.get_extension::<OpenerFactoryExtension>()
            .map(|ext| ext.0.clone())
    }
}

// ── Proto ───────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, prost::Message)]
struct SplitOpenerProto {
    #[prost(string, tag = "1")]
    identifier: String,
    #[prost(string, tag = "2")]
    tantivy_schema_json: String,
    #[prost(uint32, repeated, tag = "3")]
    segment_sizes: Vec<u32>,
    #[prost(uint64, tag = "4")]
    footer_start: u64,
    #[prost(uint64, tag = "5")]
    footer_end: u64,
    #[prost(string, repeated, tag = "6")]
    multi_valued_fields: Vec<String>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TantivyScanProto {
    #[prost(string, tag = "1")]
    identifier: String,
    #[prost(string, tag = "2")]
    tantivy_schema_json: String,
    #[prost(uint32, repeated, tag = "3")]
    segment_sizes: Vec<u32>,
    #[prost(uint32, repeated, tag = "4")]
    projection: Vec<u32>,
    #[prost(bool, tag = "5")]
    has_projection: bool,
    #[prost(uint32, tag = "6")]
    output_partitions: u32,
    /// 0 = FastField, 1 = InvertedIndex, 2 = Document
    #[prost(uint32, tag = "7")]
    provider_type: u32,
    #[prost(string, tag = "8")]
    raw_queries_json: String,
    #[prost(uint32, tag = "9")]
    topk: u32,
    #[prost(bool, tag = "10")]
    has_topk: bool,
    /// Serialized pushed filters (PhysicalExprNode protobuf bytes, one per filter).
    #[prost(bytes = "vec", repeated, tag = "11")]
    pushed_filters: Vec<Vec<u8>>,
    /// Footer byte range for storage-backed openers.
    #[prost(uint64, tag = "12")]
    footer_start: u64,
    #[prost(uint64, tag = "13")]
    footer_end: u64,
    /// Names of fields that are multi-valued in at least one segment.
    #[prost(string, repeated, tag = "14")]
    multi_valued_fields: Vec<String>,
    /// JSON-serialized tantivy aggregation specification (used by AGG_DATA_SOURCE).
    #[prost(string, tag = "16")]
    aggregations_json: String,
    /// Protobuf-encoded Arrow output schema (used by AGG_DATA_SOURCE).
    #[prost(bytes = "vec", tag = "17")]
    output_schema_bytes: Vec<u8>,
    /// JSON-serialized fast field filter Exprs (for SINGLE_TABLE / AGG_DATA_SOURCE).
    /// Workers re-derive `pre_built_query` from these on execution.
    #[prost(string, tag = "18")]
    fast_field_filters_json: String,
    /// Split descriptors for SINGLE_TABLE scans spanning multiple split openers.
    #[prost(message, repeated, tag = "19")]
    split_openers: Vec<SplitOpenerProto>,
    /// Protobuf-encoded canonical fast field schema for SINGLE_TABLE scans.
    #[prost(bytes = "vec", tag = "20")]
    canonical_ff_schema_bytes: Vec<u8>,
    /// 0 = final aggregate rows, 1 = partial aggregate state rows.
    #[prost(uint32, tag = "21")]
    agg_output_mode: u32,
    /// Optional planner-supplied per-partition row limit for SINGLE_TABLE scans.
    #[prost(uint32, tag = "22")]
    row_limit: u32,
    #[prost(bool, tag = "23")]
    has_row_limit: bool,
}

const FAST_FIELD: u32 = 0;
const INVERTED_INDEX: u32 = 1;
const DOCUMENT: u32 = 2;
const SINGLE_TABLE: u32 = 3;
const AGG_DATA_SOURCE: u32 = 4;
const AGG_OUTPUT_FINAL_MERGED: u32 = 0;
const AGG_OUTPUT_PARTIAL_STATES: u32 = 1;

struct OpenerProtoMetadata {
    identifier: String,
    tantivy_schema_json: String,
    segment_sizes: Vec<u32>,
    footer_start: u64,
    footer_end: u64,
    multi_valued_fields: Vec<String>,
}

// ── TantivyCodec — pure serialization, no runtime state ────────────

/// A [`PhysicalExtensionCodec`] for tantivy-df [`DataSourceExec`] nodes.
///
/// This codec is **stateless** — it only converts between plan nodes
/// and bytes. The opener factory that actually opens indexes lives on
/// the [`SessionConfig`] as an extension (see [`OpenerFactoryExt`]).
///
/// Register the codec on both coordinator and workers:
/// ```ignore
/// builder.set_distributed_user_codec(TantivyCodec);
/// ```
#[derive(Debug, Clone)]
pub struct TantivyCodec;

/// Rebuild a tantivy `pre_built_query` from serialized fast field filter JSON.
///
/// Returns `None` when the JSON is empty (no fast field filters were pushed).
fn reconstruct_pre_built_query(
    json: &str,
    tantivy_schema: &tantivy::schema::Schema,
) -> Result<Option<Arc<dyn tantivy::query::Query>>> {
    let queries = deserialize_fast_field_filters(json, tantivy_schema)?;
    if queries.is_empty() {
        return Ok(None);
    }
    if queries.len() == 1 {
        Ok(queries.into_iter().next().map(Arc::from))
    } else {
        Ok(Some(Arc::new(tantivy::query::BooleanQuery::intersection(
            queries,
        ))))
    }
}

/// Build a [`ScanSchema`] for `SingleTableDataSource` from the canonical fast
/// field schema and projection indices.
///
/// This mirrors the logic in `SingleTableProvider::scan` but avoids
/// constructing a full `SessionContext` and `TableProvider`.
fn build_single_table_scan_schema(
    canonical_ff_schema: &SchemaRef,
    projection: &Option<Vec<usize>>,
) -> Result<ScanSchema> {
    use arrow::datatypes::{DataType, Field, Schema};

    // Build unified schema: fast fields + _score + _document.
    let mut unified_fields: Vec<Arc<Field>> = canonical_ff_schema.fields().to_vec();
    let score_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_score", DataType::Float32, true)));
    let document_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_document", DataType::Utf8, false)));
    let unified_schema = Arc::new(Schema::new(unified_fields));

    // Determine projected indices.
    let actual_indices: Vec<usize> = match projection {
        Some(indices) => indices.clone(),
        None => (0..unified_schema.fields().len()).collect(),
    };

    // Classify projected columns into score, document, and fast field indices.
    let mut needs_score = false;
    let mut needs_document = false;
    let mut ff_indices = Vec::new();

    for &idx in &actual_indices {
        if idx == score_idx {
            needs_score = true;
        } else if idx == document_idx {
            needs_document = true;
        } else {
            ff_indices.push(idx);
        }
    }

    ff_indices.sort();
    ff_indices.dedup();
    if ff_indices.is_empty() {
        ff_indices.push(canonical_ff_schema.index_of("_doc_id").map_err(|_| {
            DataFusionError::Internal(
                "canonical fast field schema missing required _doc_id column".into(),
            )
        })?);
    }

    let ff_projected = {
        let fields: Vec<Field> = ff_indices
            .iter()
            .map(|&i| canonical_ff_schema.field(i).clone())
            .collect();
        Arc::new(Schema::new(fields))
    };

    let projected = {
        let fields: Vec<Field> = actual_indices
            .iter()
            .map(|&i| unified_schema.field(i).clone())
            .collect();
        Arc::new(Schema::new(fields))
    };

    Ok(ScanSchema {
        unified: unified_schema,
        projected,
        ff_projected,
        projection: projection.clone(),
        score_idx,
        document_idx,
        needs_score,
        needs_document,
    })
}

/// Extract serializable metadata from an opener.
fn opener_to_proto(opener: &Arc<dyn IndexOpener>) -> Result<OpenerProtoMetadata> {
    let tantivy_schema_json = serde_json::to_string(&opener.schema()).map_err(|e| {
        DataFusionError::Internal(format!("failed to serialize tantivy schema: {e}"))
    })?;
    let (footer_start, footer_end) = opener.footer_range();
    Ok(OpenerProtoMetadata {
        identifier: opener.identifier().to_string(),
        tantivy_schema_json,
        segment_sizes: opener.segment_sizes(),
        footer_start,
        footer_end,
        multi_valued_fields: opener.multi_valued_fields(),
    })
}

fn split_opener_to_proto(opener: &Arc<dyn IndexOpener>) -> Result<SplitOpenerProto> {
    let meta = opener_to_proto(opener)?;
    Ok(SplitOpenerProto {
        identifier: meta.identifier,
        tantivy_schema_json: meta.tantivy_schema_json,
        segment_sizes: meta.segment_sizes,
        footer_start: meta.footer_start,
        footer_end: meta.footer_end,
        multi_valued_fields: meta.multi_valued_fields,
    })
}

fn encode_schema_bytes(schema: &arrow::datatypes::Schema) -> Result<Vec<u8>> {
    let proto_schema = datafusion_proto::protobuf::Schema::try_from(schema)
        .map_err(|e| DataFusionError::Internal(format!("encode schema: {e}")))?;
    let mut schema_buf = Vec::new();
    prost::Message::encode(&proto_schema, &mut schema_buf)
        .map_err(|e| DataFusionError::Internal(format!("encode schema bytes: {e}")))?;
    Ok(schema_buf)
}

fn decode_schema_bytes(bytes: &[u8]) -> Result<Arc<arrow::datatypes::Schema>> {
    let proto_schema = datafusion_proto::protobuf::Schema::decode(bytes)
        .map_err(|e| DataFusionError::Internal(format!("decode schema: {e}")))?;
    Ok(Arc::new(
        arrow::datatypes::Schema::try_from(&proto_schema)
            .map_err(|e| DataFusionError::Internal(format!("convert schema: {e}")))?,
    ))
}

impl PhysicalExtensionCodec for TantivyCodec {
    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        if let Some(ds_exec) = node.as_any().downcast_ref::<DataSourceExec>() {
            let ds = ds_exec.data_source();
            let output_partitions = ds_exec.properties().partitioning.partition_count() as u32;

            if let Some(st) = ds.as_any().downcast_ref::<SingleTableDataSource>() {
                let split_openers: Vec<SplitOpenerProto> = st
                    .split_openers()
                    .iter()
                    .map(split_opener_to_proto)
                    .collect::<Result<_>>()?;
                let (proj, has_proj) = match st.projection() {
                    Some(p) => (p.iter().map(|&i| i as u32).collect(), true),
                    None => (Vec::new(), false),
                };
                let rq_json = serde_json::to_string(st.raw_queries()).map_err(|e| {
                    DataFusionError::Internal(format!("serialize raw_queries: {e}"))
                })?;
                let (topk, has_topk) = match st.topk() {
                    Some(k) => (k as u32, true),
                    None => (0, false),
                };
                let ff_filters_json = serialize_fast_field_filters(st.fast_field_filter_exprs())?;
                let canonical_ff_schema_bytes =
                    encode_schema_bytes(st.canonical_fast_field_schema().as_ref())?;
                let (row_limit, has_row_limit) = match st.row_limit() {
                    Some(limit) => (limit as u32, true),
                    None => (0, false),
                };
                return TantivyScanProto {
                    identifier: String::new(),
                    tantivy_schema_json: String::new(),
                    segment_sizes: Vec::new(),
                    projection: proj,
                    has_projection: has_proj,
                    output_partitions,
                    provider_type: SINGLE_TABLE,
                    raw_queries_json: rq_json,
                    topk,
                    has_topk,
                    pushed_filters: Vec::new(),
                    footer_start: 0,
                    footer_end: 0,
                    multi_valued_fields: Vec::new(),
                    aggregations_json: String::new(),
                    output_schema_bytes: Vec::new(),
                    fast_field_filters_json: ff_filters_json,
                    split_openers,
                    canonical_ff_schema_bytes,
                    agg_output_mode: AGG_OUTPUT_FINAL_MERGED,
                    row_limit,
                    has_row_limit,
                }
                .encode(buf)
                .map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }

            if let Some(agg_ds) = ds.as_any().downcast_ref::<AggDataSource>() {
                let opener_meta = agg_ds
                    .single_split_opener()
                    .map(opener_to_proto)
                    .transpose()?;
                let split_openers: Vec<SplitOpenerProto> = agg_ds
                    .split_openers()
                    .iter()
                    .map(split_opener_to_proto)
                    .collect::<Result<_>>()?;
                let agg_json =
                    serde_json::to_string(agg_ds.aggregations().as_ref()).map_err(|e| {
                        DataFusionError::Internal(format!("serialize aggregations: {e}"))
                    })?;
                let rq_json = serde_json::to_string(agg_ds.raw_queries()).map_err(|e| {
                    DataFusionError::Internal(format!("serialize raw_queries: {e}"))
                })?;
                let output_schema_bytes = encode_schema_bytes(agg_ds.output_schema().as_ref())?;
                let ff_filters_json =
                    serialize_fast_field_filters(agg_ds.fast_field_filter_exprs())?;
                let agg_output_mode = match agg_ds.output_mode() {
                    AggOutputMode::FinalMerged => AGG_OUTPUT_FINAL_MERGED,
                    AggOutputMode::PartialStates => AGG_OUTPUT_PARTIAL_STATES,
                };
                return TantivyScanProto {
                    identifier: opener_meta
                        .as_ref()
                        .map(|meta| meta.identifier.clone())
                        .unwrap_or_default(),
                    tantivy_schema_json: opener_meta
                        .as_ref()
                        .map(|meta| meta.tantivy_schema_json.clone())
                        .unwrap_or_default(),
                    segment_sizes: opener_meta
                        .as_ref()
                        .map(|meta| meta.segment_sizes.clone())
                        .unwrap_or_default(),
                    projection: Vec::new(),
                    has_projection: false,
                    output_partitions,
                    provider_type: AGG_DATA_SOURCE,
                    raw_queries_json: rq_json,
                    topk: 0,
                    has_topk: false,
                    pushed_filters: Vec::new(),
                    footer_start: opener_meta
                        .as_ref()
                        .map(|meta| meta.footer_start)
                        .unwrap_or(0),
                    footer_end: opener_meta
                        .as_ref()
                        .map(|meta| meta.footer_end)
                        .unwrap_or(0),
                    multi_valued_fields: opener_meta
                        .as_ref()
                        .map(|meta| meta.multi_valued_fields.clone())
                        .unwrap_or_default(),
                    aggregations_json: agg_json,
                    output_schema_bytes,
                    fast_field_filters_json: ff_filters_json,
                    split_openers,
                    canonical_ff_schema_bytes: Vec::new(),
                    agg_output_mode,
                    row_limit: 0,
                    has_row_limit: false,
                }
                .encode(buf)
                .map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }
        }

        Err(DataFusionError::Internal(format!(
            "TantivyCodec: unsupported node {}",
            node.name()
        )))
    }

    fn try_decode(
        &self,
        buf: &[u8],
        _inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let proto = TantivyScanProto::decode(buf)
            .map_err(|e| DataFusionError::Internal(format!("decode: {e}")))?;

        // Read the opener factory from the session config — set at worker startup.
        let opener_factory = ctx.session_config().get_opener_factory().ok_or_else(|| {
            DataFusionError::Internal(
                "no OpenerFactory registered on session config; \
                 call config.set_opener_factory() in the worker session builder"
                    .to_string(),
            )
        })?;

        let projection = if proto.has_projection {
            Some(
                proto
                    .projection
                    .iter()
                    .map(|&i| i as usize)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        match proto.provider_type {
            FAST_FIELD | INVERTED_INDEX | DOCUMENT => Err(DataFusionError::Internal(format!(
                "decomposed provider type {} is no longer supported; \
                     use SINGLE_TABLE or AGG_DATA_SOURCE",
                proto.provider_type
            ))),
            SINGLE_TABLE => decode_single_table(opener_factory, projection, &proto),
            AGG_DATA_SOURCE => decode_agg(opener_factory, &proto),
            other => Err(DataFusionError::Internal(format!(
                "unknown provider type: {other}"
            ))),
        }
    }
}

/// Decode a `SINGLE_TABLE` proto into a [`DataSourceExec`].
fn decode_single_table(
    opener_factory: OpenerFactory,
    projection: Option<Vec<usize>>,
    proto: &TantivyScanProto,
) -> Result<Arc<dyn ExecutionPlan>> {
    let canonical_ff_schema = if proto.canonical_ff_schema_bytes.is_empty() {
        let tantivy_schema: tantivy::schema::Schema =
            serde_json::from_str(&proto.tantivy_schema_json)
                .map_err(|e| DataFusionError::Internal(format!("parse tantivy schema: {e}")))?;
        tantivy_schema_to_arrow_with_multi_valued(&tantivy_schema, &proto.multi_valued_fields)
    } else {
        decode_schema_bytes(&proto.canonical_ff_schema_bytes)?
    };
    let scan_schema = build_single_table_scan_schema(&canonical_ff_schema, &projection)?;

    let raw_queries: Vec<(String, String)> = if proto.raw_queries_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&proto.raw_queries_json)
            .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?
    };

    let fast_field_filter_exprs =
        deserialize_fast_field_filter_exprs(&proto.fast_field_filters_json)?;

    let topk = if proto.has_topk {
        Some(proto.topk as usize)
    } else {
        None
    };
    let row_limit = if proto.has_row_limit {
        Some(proto.row_limit as usize)
    } else {
        None
    };

    let split_openers = if proto.split_openers.is_empty() {
        let tantivy_schema: tantivy::schema::Schema =
            serde_json::from_str(&proto.tantivy_schema_json)
                .map_err(|e| DataFusionError::Internal(format!("parse tantivy schema: {e}")))?;
        vec![opener_factory(OpenerMetadata {
            identifier: proto.identifier.clone(),
            tantivy_schema,
            segment_sizes: proto.segment_sizes.clone(),
            footer_start: proto.footer_start,
            footer_end: proto.footer_end,
            multi_valued_fields: proto.multi_valued_fields.clone(),
        })]
    } else {
        proto
            .split_openers
            .iter()
            .map(|split| {
                let tantivy_schema: tantivy::schema::Schema =
                    serde_json::from_str(&split.tantivy_schema_json).map_err(|e| {
                        DataFusionError::Internal(format!("parse tantivy schema: {e}"))
                    })?;
                Ok(opener_factory(OpenerMetadata {
                    identifier: split.identifier.clone(),
                    tantivy_schema,
                    segment_sizes: split.segment_sizes.clone(),
                    footer_start: split.footer_start,
                    footer_end: split.footer_end,
                    multi_valued_fields: split.multi_valued_fields.clone(),
                }))
            })
            .collect::<Result<Vec<_>>>()?
    };

    let splits: Vec<SplitExecutionPlan> = split_openers
        .iter()
        .map(|opener| {
            let split_ff_schema = tantivy_schema_to_arrow_with_multi_valued(
                &opener.schema(),
                &opener.multi_valued_fields(),
            );
            Ok(SplitExecutionPlan {
                opener: Arc::clone(opener),
                fast_field_projection: plan_fast_field_projection(
                    &split_ff_schema,
                    &scan_schema.ff_projected,
                )?,
            })
        })
        .collect::<Result<_>>()?;

    let mut partition_map = Vec::new();
    for (split_idx, opener) in split_openers.iter().enumerate() {
        let num_segments = opener.segment_sizes().len().max(1);
        for segment_idx in 0..num_segments {
            partition_map.push(PartitionSpec {
                split_idx,
                segment_idx,
            });
        }
    }

    let ds = SingleTableDataSource::new_from_codec(
        splits,
        scan_schema,
        raw_queries,
        fast_field_filter_exprs,
        topk,
        row_limit,
        partition_map,
    );
    Ok(Arc::new(DataSourceExec::new(Arc::new(ds))))
}

/// Decode an `AGG_DATA_SOURCE` proto into a [`DataSourceExec`].
fn decode_agg(
    opener_factory: OpenerFactory,
    proto: &TantivyScanProto,
) -> Result<Arc<dyn ExecutionPlan>> {
    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_str(&proto.aggregations_json)
            .map_err(|e| DataFusionError::Internal(format!("parse aggregations: {e}")))?;

    let raw_queries: Vec<(String, String)> = if proto.raw_queries_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&proto.raw_queries_json)
            .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?
    };

    if proto.output_schema_bytes.is_empty() {
        return Err(DataFusionError::Internal(
            "missing output schema for AGG_DATA_SOURCE".into(),
        ));
    }
    let output_schema = {
        let proto_schema =
            datafusion_proto::protobuf::Schema::decode(proto.output_schema_bytes.as_slice())
                .map_err(|e| DataFusionError::Internal(format!("decode schema: {e}")))?;
        Arc::new(
            arrow::datatypes::Schema::try_from(&proto_schema)
                .map_err(|e| DataFusionError::Internal(format!("convert schema: {e}")))?,
        )
    };

    let split_openers = if proto.split_openers.is_empty() {
        let tantivy_schema: tantivy::schema::Schema =
            serde_json::from_str(&proto.tantivy_schema_json)
                .map_err(|e| DataFusionError::Internal(format!("parse tantivy schema: {e}")))?;
        vec![opener_factory(OpenerMetadata {
            identifier: proto.identifier.clone(),
            tantivy_schema,
            segment_sizes: proto.segment_sizes.clone(),
            footer_start: proto.footer_start,
            footer_end: proto.footer_end,
            multi_valued_fields: proto.multi_valued_fields.clone(),
        })]
    } else {
        proto
            .split_openers
            .iter()
            .map(|split| {
                let tantivy_schema: tantivy::schema::Schema =
                    serde_json::from_str(&split.tantivy_schema_json).map_err(|e| {
                        DataFusionError::Internal(format!("parse tantivy schema: {e}"))
                    })?;
                Ok(opener_factory(OpenerMetadata {
                    identifier: split.identifier.clone(),
                    tantivy_schema,
                    segment_sizes: split.segment_sizes.clone(),
                    footer_start: split.footer_start,
                    footer_end: split.footer_end,
                    multi_valued_fields: split.multi_valued_fields.clone(),
                }))
            })
            .collect::<Result<Vec<_>>>()?
    };

    let pre_built_query = if split_openers.len() == 1 {
        reconstruct_pre_built_query(&proto.fast_field_filters_json, &split_openers[0].schema())?
    } else {
        None
    };
    let fast_field_filter_exprs =
        deserialize_fast_field_filter_exprs(&proto.fast_field_filters_json)?;

    let aggregations = Arc::new(aggs);
    let ds = match proto.agg_output_mode {
        AGG_OUTPUT_PARTIAL_STATES => AggDataSource::from_split_openers_partial_states(
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        ),
        _ => AggDataSource::from_split_openers(
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        ),
    };
    Ok(Arc::new(DataSourceExec::new(Arc::new(ds))))
}
