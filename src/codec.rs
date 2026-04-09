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

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::datasource::TableProvider;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, lit};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::SessionConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use futures::stream::{self, StreamExt};
use prost::Message;

use crate::decomposed::document_provider::{DocumentDataSource, TantivyDocumentProvider};
use crate::full_text_udf::full_text_udf;
use crate::index_opener::{IndexOpener, OpenerMetadata};
use crate::decomposed::inverted_index_provider::{InvertedIndexDataSource, TantivyInvertedIndexProvider};
use crate::schema_mapping::tantivy_schema_to_arrow_with_multi_valued;
use crate::unified::single_table_provider::SingleTableDataSource;
use crate::decomposed::table_provider::{FastFieldDataSource, TantivyTableProvider};

// ── Opener factory as session extension ─────────────────────────────

/// Function that reconstructs an [`IndexOpener`] on a remote worker
/// from serialized metadata.
pub type OpenerFactory =
    Arc<dyn Fn(OpenerMetadata) -> Arc<dyn IndexOpener> + Send + Sync>;

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
}

const FAST_FIELD: u32 = 0;
const INVERTED_INDEX: u32 = 1;
const DOCUMENT: u32 = 2;
const SINGLE_TABLE: u32 = 3;

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

/// Serialize a list of `PhysicalExpr`s into protobuf bytes.
fn serialize_pushed_filters(
    filters: &[Arc<dyn PhysicalExpr>],
    codec: &TantivyCodec,
) -> Result<Vec<Vec<u8>>> {
    filters
        .iter()
        .map(|f| {
            let proto = datafusion_proto::physical_plan::to_proto::serialize_physical_expr(f, codec)?;
            let mut buf = Vec::new();
            proto
                .encode(&mut buf)
                .map_err(|e| DataFusionError::Internal(format!("encode filter: {e}")))?;
            Ok(buf)
        })
        .collect()
}

/// Deserialize pushed filters from protobuf bytes.
fn deserialize_pushed_filters(
    filter_bytes: &[Vec<u8>],
    ctx: &TaskContext,
    schema: &arrow::datatypes::Schema,
    codec: &TantivyCodec,
) -> Result<Vec<Arc<dyn PhysicalExpr>>> {
    use prost::Message as ProstMessage;
    filter_bytes
        .iter()
        .map(|buf| {
            let proto =
                datafusion_proto::protobuf::PhysicalExprNode::decode(buf.as_slice())
                    .map_err(|e| DataFusionError::Internal(format!("decode filter: {e}")))?;
            datafusion_proto::physical_plan::from_proto::parse_physical_expr(
                &proto, ctx, schema, codec,
            )
        })
        .collect()
}

/// Extract serializable metadata from an opener.
fn opener_to_proto(opener: &Arc<dyn IndexOpener>) -> Result<(String, String, Vec<u32>, u64, u64, Vec<String>)> {
    let identifier = opener.identifier().to_string();
    let tantivy_schema_json =
        serde_json::to_string(&opener.schema()).map_err(|e| {
            DataFusionError::Internal(format!("failed to serialize tantivy schema: {e}"))
        })?;
    let segment_sizes = opener.segment_sizes();
    let (footer_start, footer_end) = opener.footer_range();
    let multi_valued_fields = opener.multi_valued_fields();
    Ok((identifier, tantivy_schema_json, segment_sizes, footer_start, footer_end, multi_valued_fields))
}

impl PhysicalExtensionCodec for TantivyCodec {
    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if let Some(ds_exec) = node.as_any().downcast_ref::<DataSourceExec>() {
            let ds = ds_exec.data_source();
            let output_partitions =
                ds_exec.properties().partitioning.partition_count() as u32;

            if let Some(ff) = ds.as_any().downcast_ref::<FastFieldDataSource>() {
                let (id, schema_json, seg, footer_start, footer_end, mv) = opener_to_proto(ff.opener())?;
                let (proj, has_proj) = match ff.projection() {
                    Some(p) => (p.iter().map(|&i| i as u32).collect(), true),
                    None => (Vec::new(), false),
                };
                let pushed_filters = serialize_pushed_filters(ff.pushed_filters(), self)?;
                return TantivyScanProto {
                    identifier: id, tantivy_schema_json: schema_json, segment_sizes: seg,
                    projection: proj, has_projection: has_proj, output_partitions,
                    provider_type: FAST_FIELD, raw_queries_json: String::new(), topk: 0, has_topk: false,
                    pushed_filters,
                    footer_start, footer_end, multi_valued_fields: mv,
                }.encode(buf).map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }

            if let Some(inv) = ds.as_any().downcast_ref::<InvertedIndexDataSource>() {
                let (id, schema_json, seg, footer_start, footer_end, mv) = opener_to_proto(inv.opener())?;
                let (proj, has_proj) = match inv.projection() {
                    Some(p) => (p.iter().map(|&i| i as u32).collect(), true),
                    None => (Vec::new(), false),
                };
                let rq_json = serde_json::to_string(inv.raw_queries())
                    .map_err(|e| DataFusionError::Internal(format!("serialize raw_queries: {e}")))?;
                let (topk, has_topk) = match inv.topk() {
                    Some(k) => (k as u32, true),
                    None => (0, false),
                };
                return TantivyScanProto {
                    identifier: id, tantivy_schema_json: schema_json, segment_sizes: seg,
                    projection: proj, has_projection: has_proj, output_partitions,
                    provider_type: INVERTED_INDEX, raw_queries_json: rq_json, topk, has_topk,
                    pushed_filters: Vec::new(),
                    footer_start, footer_end, multi_valued_fields: mv,
                }.encode(buf).map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }

            if let Some(doc) = ds.as_any().downcast_ref::<DocumentDataSource>() {
                let (id, schema_json, seg, footer_start, footer_end, mv) = opener_to_proto(doc.opener())?;
                let (proj, has_proj) = match doc.projection() {
                    Some(p) => (p.iter().map(|&i| i as u32).collect(), true),
                    None => (Vec::new(), false),
                };
                let pushed_filters = serialize_pushed_filters(doc.pushed_filters(), self)?;
                return TantivyScanProto {
                    identifier: id, tantivy_schema_json: schema_json, segment_sizes: seg,
                    projection: proj, has_projection: has_proj, output_partitions,
                    provider_type: DOCUMENT, raw_queries_json: String::new(), topk: 0, has_topk: false,
                    pushed_filters,
                    footer_start, footer_end, multi_valued_fields: mv,
                }.encode(buf).map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }

            if let Some(st) = ds.as_any().downcast_ref::<SingleTableDataSource>() {
                let (id, schema_json, seg, footer_start, footer_end, mv) = opener_to_proto(st.opener())?;
                let (proj, has_proj) = match st.projection() {
                    Some(p) => (p.iter().map(|&i| i as u32).collect(), true),
                    None => (Vec::new(), false),
                };
                let rq_json = serde_json::to_string(st.raw_queries())
                    .map_err(|e| DataFusionError::Internal(format!("serialize raw_queries: {e}")))?;
                let (topk, has_topk) = match st.topk() {
                    Some(k) => (k as u32, true),
                    None => (0, false),
                };
                let pushed_filters = serialize_pushed_filters(st.pushed_filters(), self)?;
                return TantivyScanProto {
                    identifier: id, tantivy_schema_json: schema_json, segment_sizes: seg,
                    projection: proj, has_projection: has_proj, output_partitions,
                    provider_type: SINGLE_TABLE, raw_queries_json: rq_json, topk, has_topk,
                    pushed_filters,
                    footer_start, footer_end, multi_valued_fields: mv,
                }.encode(buf).map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
            }
        }

        if let Some(lazy) = node.as_any().downcast_ref::<LazyScanExec>() {
            return TantivyScanProto {
                identifier: lazy.identifier.clone(),
                tantivy_schema_json: lazy.tantivy_schema_json.clone(),
                segment_sizes: lazy.segment_sizes.clone(),
                projection: lazy.projection.as_ref()
                    .map(|p| p.iter().map(|&i| i as u32).collect()).unwrap_or_default(),
                has_projection: lazy.projection.is_some(),
                output_partitions: lazy.output_partitions,
                provider_type: lazy.provider_type,
                raw_queries_json: lazy.raw_queries_json.clone(),
                topk: lazy.topk.map(|k| k as u32).unwrap_or(0),
                has_topk: lazy.topk.is_some(),
                pushed_filters: lazy.pushed_filter_bytes.clone(),
                footer_start: lazy.footer_start,
                footer_end: lazy.footer_end,
                multi_valued_fields: lazy.multi_valued_fields.clone(),
            }.encode(buf).map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
        }

        Err(DataFusionError::Internal(format!(
            "TantivyCodec: unsupported node {}", node.name()
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
        let opener_factory = ctx
            .session_config()
            .get_opener_factory()
            .ok_or_else(|| DataFusionError::Internal(
                "no OpenerFactory registered on session config; \
                 call config.set_opener_factory() in the worker session builder"
                    .to_string(),
            ))?;

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
            Some(proto.projection.iter().map(|&i| i as usize).collect::<Vec<_>>())
        } else {
            None
        };

        let arrow_schema: SchemaRef = match proto.provider_type {
            FAST_FIELD => tantivy_schema_to_arrow_with_multi_valued(
                &opener.schema(),
                &proto.multi_valued_fields,
            ),
            INVERTED_INDEX => TantivyInvertedIndexProvider::from_opener(opener.clone()).schema(),
            DOCUMENT => TantivyDocumentProvider::from_opener(opener.clone()).schema(),
            SINGLE_TABLE => {
                // Build unified schema: fast fields + _score + _document
                let ff_schema = tantivy_schema_to_arrow_with_multi_valued(
                    &opener.schema(),
                    &proto.multi_valued_fields,
                );
                let mut unified_fields: Vec<arrow::datatypes::Field> = ff_schema.fields().iter().map(|f| f.as_ref().clone()).collect();
                unified_fields.push(arrow::datatypes::Field::new("_score", arrow::datatypes::DataType::Float32, true));
                unified_fields.push(arrow::datatypes::Field::new("_document", arrow::datatypes::DataType::Utf8, false));
                Arc::new(arrow::datatypes::Schema::new(unified_fields))
            }
            other => return Err(DataFusionError::Internal(format!("unknown provider type: {other}"))),
        };

        let projected_schema = match &projection {
            Some(indices) => {
                let fields: Vec<_> = indices.iter().map(|&i| arrow_schema.field(i).clone()).collect();
                Arc::new(arrow::datatypes::Schema::new(fields))
            }
            None => arrow_schema,
        };

        let output_partitions = (proto.output_partitions as usize).max(1);
        let plan_properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
            Partitioning::UnknownPartitioning(output_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Arc::new(LazyScanExec {
            identifier: proto.identifier,
            tantivy_schema_json: proto.tantivy_schema_json,
            segment_sizes: proto.segment_sizes,
            projection,
            provider_type: proto.provider_type,
            raw_queries_json: proto.raw_queries_json,
            topk: if proto.has_topk { Some(proto.topk as usize) } else { None },
            pushed_filter_bytes: proto.pushed_filters,
            footer_start: proto.footer_start,
            footer_end: proto.footer_end,
            multi_valued_fields: proto.multi_valued_fields,
            plan_properties,
            projected_schema,
            output_partitions: proto.output_partitions,
        }))
    }
}

// ── LazyScanExec ────────────────────────────────────────────────────

/// Execution plan reconstructed on the worker from serialized metadata.
/// At execution time, reads the [`OpenerFactory`] from the
/// [`TaskContext`]'s session config to create the opener.
#[derive(Debug)]
struct LazyScanExec {
    identifier: String,
    tantivy_schema_json: String,
    segment_sizes: Vec<u32>,
    projection: Option<Vec<usize>>,
    provider_type: u32,
    raw_queries_json: String,
    topk: Option<usize>,
    pushed_filter_bytes: Vec<Vec<u8>>,
    footer_start: u64,
    footer_end: u64,
    multi_valued_fields: Vec<String>,
    plan_properties: PlanProperties,
    projected_schema: SchemaRef,
    output_partitions: u32,
}

impl DisplayAs for LazyScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let kind = match self.provider_type {
            FAST_FIELD => "FastField",
            INVERTED_INDEX => "InvertedIndex",
            DOCUMENT => "Document",
            SINGLE_TABLE => "SingleTable",
            _ => "Unknown",
        };
        write!(f, "LazyScanExec(id={}, type={})", self.identifier, kind)
    }
}

impl ExecutionPlan for LazyScanExec {
    fn name(&self) -> &str { "LazyScanExec" }
    fn as_any(&self) -> &dyn Any { self }
    fn properties(&self) -> &PlanProperties { &self.plan_properties }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> { vec![] }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Read the opener factory from session config (set at worker startup).
        let opener_factory = context
            .session_config()
            .get_opener_factory()
            .ok_or_else(|| DataFusionError::Internal(
                "no OpenerFactory on session config".to_string(),
            ))?;

        let tantivy_schema_json = self.tantivy_schema_json.clone();
        let segment_sizes = self.segment_sizes.clone();
        let identifier = self.identifier.clone();
        let projection = self.projection.clone();
        let output_partitions = self.output_partitions as usize;
        let provider_type = self.provider_type;
        let raw_queries_json = self.raw_queries_json.clone();
        let pushed_filter_bytes = self.pushed_filter_bytes.clone();
        let topk = self.topk;
        let footer_start = self.footer_start;
        let footer_end = self.footer_end;
        let multi_valued_fields = self.multi_valued_fields.clone();

        let schema_clone = self.projected_schema.clone();
        let stream = stream::once(async move {
            let tantivy_schema: tantivy::schema::Schema =
                serde_json::from_str(&tantivy_schema_json)
                    .map_err(|e| DataFusionError::Internal(format!("parse schema: {e}")))?;

            let opener = opener_factory(OpenerMetadata {
                identifier, tantivy_schema, segment_sizes, footer_start, footer_end,
                multi_valued_fields,
            });

            let target_partitions = output_partitions.max(1);
            let config = context.session_config().clone()
                .with_target_partitions(target_partitions);
            let session = datafusion::prelude::SessionContext::new_with_config(config);

            let filters: Vec<Expr> =
                if (provider_type == INVERTED_INDEX || provider_type == SINGLE_TABLE) && !raw_queries_json.is_empty() {
                    let rq: Vec<(String, String)> = serde_json::from_str(&raw_queries_json)
                        .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?;
                    session.register_udf(full_text_udf());
                    rq.into_iter()
                        .map(|(field, query)| {
                            Expr::ScalarFunction(
                                datafusion::logical_expr::expr::ScalarFunction::new_udf(
                                    Arc::new(full_text_udf()),
                                    vec![datafusion::prelude::col(&field), lit(query)],
                                ),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            let state = session.state();
            let exec: Arc<dyn ExecutionPlan> = match provider_type {
                FAST_FIELD => {
                    TantivyTableProvider::from_opener(opener)
                        .scan(&state, projection.as_ref().map(|v| v as &Vec<usize>), &filters, None)
                        .await?
                }
                INVERTED_INDEX => {
                    let inv_exec = TantivyInvertedIndexProvider::from_opener(opener)
                        .scan(&state, projection.as_ref().map(|v| v as &Vec<usize>), &filters, None)
                        .await?;
                    // Re-apply topk if it was serialized.
                    if let Some(k) = topk {
                        if let Some(ds_exec) = inv_exec.as_any().downcast_ref::<DataSourceExec>() {
                            if let Some(inv_ds) = ds_exec.data_source().as_any().downcast_ref::<InvertedIndexDataSource>() {
                                let updated = inv_ds.with_topk(k);
                                Arc::new(DataSourceExec::new(Arc::new(updated)))
                            } else {
                                inv_exec
                            }
                        } else {
                            inv_exec
                        }
                    } else {
                        inv_exec
                    }
                }
                DOCUMENT => {
                    TantivyDocumentProvider::from_opener(opener)
                        .scan(&state, projection.as_ref().map(|v| v as &Vec<usize>), &[], None)
                        .await?
                }
                SINGLE_TABLE => {
                    use crate::unified::single_table_provider::SingleTableProvider;
                    let provider = SingleTableProvider::from_opener(opener);
                    // Register full_text UDF if there are queries
                    if !raw_queries_json.is_empty() {
                        session.register_udf(full_text_udf());
                    }
                    let st_exec = provider.scan(&state, projection.as_ref().map(|v| v as &Vec<usize>), &filters, None).await?;
                    // Re-apply topk if present
                    if let Some(k) = topk {
                        if let Some(ds_exec) = st_exec.as_any().downcast_ref::<DataSourceExec>() {
                            if let Some(st_ds) = ds_exec.data_source().as_any().downcast_ref::<SingleTableDataSource>() {
                                let updated = st_ds.with_topk(k);
                                Arc::new(DataSourceExec::new(Arc::new(updated)))
                            } else {
                                st_exec
                            }
                        } else {
                            st_exec
                        }
                    } else {
                        st_exec
                    }
                }
                other => return Err(DataFusionError::Internal(format!("unknown provider type: {other}"))),
            };

            // Re-apply pushed filters that survived serialization.
            let exec: Arc<dyn ExecutionPlan> =
                if !pushed_filter_bytes.is_empty() {
                    if let Some(ds_exec) =
                        exec.as_any().downcast_ref::<DataSourceExec>()
                    {
                        let ds_schema = ds_exec.schema();
                        if let Some(ff_ds) = ds_exec
                            .data_source()
                            .as_any()
                            .downcast_ref::<FastFieldDataSource>()
                        {
                            let deserialized = deserialize_pushed_filters(
                                &pushed_filter_bytes,
                                &context,
                                ds_schema.as_ref(),
                                &TantivyCodec,
                            )?;
                            let updated = ff_ds.with_pushed_filters(deserialized);
                            Arc::new(DataSourceExec::new(Arc::new(updated)))
                        } else if let Some(doc_ds) = ds_exec
                            .data_source()
                            .as_any()
                            .downcast_ref::<DocumentDataSource>()
                        {
                            let deserialized = deserialize_pushed_filters(
                                &pushed_filter_bytes,
                                &context,
                                ds_schema.as_ref(),
                                &TantivyCodec,
                            )?;
                            let updated = doc_ds.with_pushed_filters(deserialized);
                            Arc::new(DataSourceExec::new(Arc::new(updated)))
                        } else if let Some(st_ds) = ds_exec
                            .data_source()
                            .as_any()
                            .downcast_ref::<SingleTableDataSource>()
                        {
                            let deserialized = deserialize_pushed_filters(
                                &pushed_filter_bytes,
                                &context,
                                ds_schema.as_ref(),
                                &TantivyCodec,
                            )?;
                            let updated = st_ds.with_pushed_filters(deserialized);
                            Arc::new(DataSourceExec::new(Arc::new(updated)))
                        } else {
                            exec
                        }
                    } else {
                        exec
                    }
                } else {
                    exec
                };

            // Forward the inner stream directly — no collecting into Vec.
            exec.execute(partition, context)
        })
        // Flatten Result<Stream<Result<Batch>>> into Stream<Result<Batch>>
        .flat_map(move |result: Result<SendableRecordBatchStream>| match result {
            Ok(inner_stream) => inner_stream.left_stream(),
            Err(e) => stream::once(async move { Err(e) }).right_stream(),
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema_clone,
            stream,
        )))
    }
}
