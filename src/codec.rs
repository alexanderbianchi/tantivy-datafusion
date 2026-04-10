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
use crate::unified::agg_data_source::AggDataSource;
use crate::unified::single_table_provider::{
    deserialize_fast_field_filters, serialize_fast_field_filters, ScanSchema, SingleTableDataSource,
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
}

const FAST_FIELD: u32 = 0;
const INVERTED_INDEX: u32 = 1;
const DOCUMENT: u32 = 2;
const SINGLE_TABLE: u32 = 3;
const AGG_DATA_SOURCE: u32 = 4;

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
        Ok(Some(Arc::from(queries.into_iter().next().unwrap())))
    } else {
        Ok(Some(Arc::new(tantivy::query::BooleanQuery::intersection(
            queries,
        ))))
    }
}

/// Build a [`ScanSchema`] for `SingleTableDataSource` from the opener's
/// tantivy schema, multi-valued field names, and projection indices.
///
/// This mirrors the logic in `SingleTableProvider::scan` but avoids
/// constructing a full `SessionContext` and `TableProvider`.
fn build_single_table_scan_schema(
    opener: &Arc<dyn IndexOpener>,
    multi_valued_fields: &[String],
    projection: &Option<Vec<usize>>,
) -> ScanSchema {
    use arrow::datatypes::{DataType, Field, Schema};

    let ff_schema =
        tantivy_schema_to_arrow_with_multi_valued(&opener.schema(), multi_valued_fields);

    // Build unified schema: fast fields + _score + _document.
    let mut unified_fields: Vec<Arc<Field>> = ff_schema.fields().to_vec();
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

    // Ensure _doc_id and _segment_ord are included (needed internally).
    if let Ok(doc_id_idx) = ff_schema.index_of("_doc_id") {
        if !ff_indices.contains(&doc_id_idx) {
            ff_indices.push(doc_id_idx);
        }
    }
    if let Ok(seg_ord_idx) = ff_schema.index_of("_segment_ord") {
        if !ff_indices.contains(&seg_ord_idx) {
            ff_indices.push(seg_ord_idx);
        }
    }
    ff_indices.sort();
    ff_indices.dedup();

    let ff_projected = {
        let fields: Vec<Field> = ff_indices
            .iter()
            .map(|&i| ff_schema.field(i).clone())
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

    ScanSchema {
        unified: unified_schema,
        projected,
        ff_projected,
        projection: projection.clone(),
        score_idx,
        document_idx,
        needs_score,
        needs_document,
    }
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

impl PhysicalExtensionCodec for TantivyCodec {
    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        if let Some(ds_exec) = node.as_any().downcast_ref::<DataSourceExec>() {
            let ds = ds_exec.data_source();
            let output_partitions = ds_exec.properties().partitioning.partition_count() as u32;

            if let Some(st) = ds.as_any().downcast_ref::<SingleTableDataSource>() {
                let opener_meta = opener_to_proto(st.opener())?;
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
                return TantivyScanProto {
                    identifier: opener_meta.identifier,
                    tantivy_schema_json: opener_meta.tantivy_schema_json,
                    segment_sizes: opener_meta.segment_sizes,
                    projection: proj,
                    has_projection: has_proj,
                    output_partitions,
                    provider_type: SINGLE_TABLE,
                    raw_queries_json: rq_json,
                    topk,
                    has_topk,
                    pushed_filters: Vec::new(),
                    footer_start: opener_meta.footer_start,
                    footer_end: opener_meta.footer_end,
                    multi_valued_fields: opener_meta.multi_valued_fields,
                    aggregations_json: String::new(),
                    output_schema_bytes: Vec::new(),
                    fast_field_filters_json: ff_filters_json,
                }
                .encode(buf)
                .map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }

            if let Some(agg_ds) = ds.as_any().downcast_ref::<AggDataSource>() {
                let opener_meta = opener_to_proto(agg_ds.opener())?;
                let agg_json =
                    serde_json::to_string(agg_ds.aggregations().as_ref()).map_err(|e| {
                        DataFusionError::Internal(format!("serialize aggregations: {e}"))
                    })?;
                let rq_json = serde_json::to_string(agg_ds.raw_queries()).map_err(|e| {
                    DataFusionError::Internal(format!("serialize raw_queries: {e}"))
                })?;
                let output_schema_bytes = {
                    let proto_schema = datafusion_proto::protobuf::Schema::try_from(
                        agg_ds.output_schema().as_ref(),
                    )
                    .map_err(|e| DataFusionError::Internal(format!("encode schema: {e}")))?;
                    let mut schema_buf = Vec::new();
                    prost::Message::encode(&proto_schema, &mut schema_buf).map_err(|e| {
                        DataFusionError::Internal(format!("encode schema bytes: {e}"))
                    })?;
                    schema_buf
                };
                let ff_filters_json =
                    serialize_fast_field_filters(agg_ds.fast_field_filter_exprs())?;
                return TantivyScanProto {
                    identifier: opener_meta.identifier,
                    tantivy_schema_json: opener_meta.tantivy_schema_json,
                    segment_sizes: opener_meta.segment_sizes,
                    projection: Vec::new(),
                    has_projection: false,
                    output_partitions,
                    provider_type: AGG_DATA_SOURCE,
                    raw_queries_json: rq_json,
                    topk: 0,
                    has_topk: false,
                    pushed_filters: Vec::new(),
                    footer_start: opener_meta.footer_start,
                    footer_end: opener_meta.footer_end,
                    multi_valued_fields: opener_meta.multi_valued_fields,
                    aggregations_json: agg_json,
                    output_schema_bytes,
                    fast_field_filters_json: ff_filters_json,
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

        let tantivy_schema: tantivy::schema::Schema =
            serde_json::from_str(&proto.tantivy_schema_json)
                .map_err(|e| DataFusionError::Internal(format!("parse tantivy schema: {e}")))?;

        let opener = opener_factory(OpenerMetadata {
            identifier: proto.identifier.clone(),
            tantivy_schema,
            segment_sizes: proto.segment_sizes.clone(),
            footer_start: proto.footer_start,
            footer_end: proto.footer_end,
            multi_valued_fields: proto.multi_valued_fields.clone(),
        });

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
            SINGLE_TABLE => decode_single_table(opener, projection, &proto),
            AGG_DATA_SOURCE => decode_agg(opener, &proto),
            other => Err(DataFusionError::Internal(format!(
                "unknown provider type: {other}"
            ))),
        }
    }
}

/// Decode a `SINGLE_TABLE` proto into a [`DataSourceExec`].
fn decode_single_table(
    opener: Arc<dyn IndexOpener>,
    projection: Option<Vec<usize>>,
    proto: &TantivyScanProto,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mv_fields = opener.multi_valued_fields();
    let scan_schema = build_single_table_scan_schema(&opener, &mv_fields, &projection);

    let raw_queries: Vec<(String, String)> = if proto.raw_queries_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&proto.raw_queries_json)
            .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?
    };

    let pre_built_query =
        reconstruct_pre_built_query(&proto.fast_field_filters_json, &opener.schema())?;

    let topk = if proto.has_topk {
        Some(proto.topk as usize)
    } else {
        None
    };
    let num_segments = (proto.output_partitions as usize).max(1);

    let ds = SingleTableDataSource::new_from_codec(
        opener,
        scan_schema,
        raw_queries,
        pre_built_query,
        topk,
        num_segments,
    );
    Ok(Arc::new(DataSourceExec::new(Arc::new(ds))))
}

/// Decode an `AGG_DATA_SOURCE` proto into a [`DataSourceExec`].
fn decode_agg(
    opener: Arc<dyn IndexOpener>,
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

    let pre_built_query =
        reconstruct_pre_built_query(&proto.fast_field_filters_json, &opener.schema())?;

    let ds = AggDataSource::new(
        opener,
        Arc::new(aggs),
        output_schema,
        raw_queries,
        pre_built_query,
        Vec::new(), // exprs not needed on worker — query already reconstructed
    );
    Ok(Arc::new(DataSourceExec::new(Arc::new(ds))))
}
