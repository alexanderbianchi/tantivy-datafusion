//! Codec for serializing tantivy [`DataSourceExec`] nodes across distributed
//! executors.
//!
//! The codec is pure serialization. Runtime split preparation happens on the
//! worker via [`crate::split_runtime::SplitRuntimeFactoryExt`].

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use prost::Message;

use crate::split_runtime::SplitDescriptor;
use crate::unified::agg_data_source::{AggDataSource, AggOutputMode};
use crate::unified::single_table_provider::{
    deserialize_fast_field_filter_exprs, deserialize_fast_field_filters,
    serialize_fast_field_filters, PartitionSpec, ScanSchema, SingleTableDataSource,
    SplitExecutionPlan,
};

#[derive(Clone, PartialEq, prost::Message)]
struct SplitDescriptorProto {
    #[prost(string, tag = "1")]
    split_id: String,
    #[prost(bytes = "vec", tag = "2")]
    payload: Vec<u8>,
    #[prost(string, tag = "3")]
    tantivy_schema_json: String,
    #[prost(string, repeated, tag = "4")]
    multi_valued_fields: Vec<String>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct TantivyPlanProto {
    /// 3 = SINGLE_TABLE, 4 = AGG_DATA_SOURCE
    #[prost(uint32, tag = "1")]
    provider_type: u32,
    #[prost(uint32, repeated, tag = "2")]
    projection: Vec<u32>,
    #[prost(bool, tag = "3")]
    has_projection: bool,
    #[prost(string, tag = "4")]
    raw_queries_json: String,
    #[prost(uint32, tag = "5")]
    topk: u32,
    #[prost(bool, tag = "6")]
    has_topk: bool,
    #[prost(string, tag = "7")]
    fast_field_filters_json: String,
    #[prost(message, repeated, tag = "8")]
    split_descriptors: Vec<SplitDescriptorProto>,
    #[prost(bytes = "vec", tag = "9")]
    canonical_ff_schema_bytes: Vec<u8>,
    #[prost(string, tag = "10")]
    aggregations_json: String,
    #[prost(bytes = "vec", tag = "11")]
    output_schema_bytes: Vec<u8>,
    /// 0 = final aggregate rows, 1 = partial aggregate state rows.
    #[prost(uint32, tag = "12")]
    agg_output_mode: u32,
    #[prost(uint32, tag = "13")]
    row_limit: u32,
    #[prost(bool, tag = "14")]
    has_row_limit: bool,
}

const SINGLE_TABLE: u32 = 3;
const AGG_DATA_SOURCE: u32 = 4;
const AGG_OUTPUT_FINAL_MERGED: u32 = 0;
const AGG_OUTPUT_PARTIAL_STATES: u32 = 1;

#[derive(Debug, Clone)]
pub struct TantivyCodec;

fn encode_split_descriptor(descriptor: &SplitDescriptor) -> Result<SplitDescriptorProto> {
    let tantivy_schema_json = serde_json::to_string(&descriptor.tantivy_schema)
        .map_err(|e| DataFusionError::Internal(format!("serialize tantivy schema: {e}")))?;
    Ok(SplitDescriptorProto {
        split_id: descriptor.split_id.clone(),
        payload: descriptor.payload.clone(),
        tantivy_schema_json,
        multi_valued_fields: descriptor.multi_valued_fields.clone(),
    })
}

fn decode_split_descriptor(proto: &SplitDescriptorProto) -> Result<SplitDescriptor> {
    let tantivy_schema = serde_json::from_str(&proto.tantivy_schema_json)
        .map_err(|e| DataFusionError::Internal(format!("parse tantivy schema: {e}")))?;
    Ok(SplitDescriptor::new(
        proto.split_id.clone(),
        proto.payload.clone(),
        tantivy_schema,
        proto.multi_valued_fields.clone(),
    ))
}

fn encode_schema_bytes(schema: &arrow::datatypes::Schema) -> Result<Vec<u8>> {
    let proto_schema = datafusion_proto::protobuf::Schema::try_from(schema)
        .map_err(|e| DataFusionError::Internal(format!("encode schema: {e}")))?;
    let mut buf = Vec::new();
    proto_schema
        .encode(&mut buf)
        .map_err(|e| DataFusionError::Internal(format!("encode schema bytes: {e}")))?;
    Ok(buf)
}

fn decode_schema_bytes(bytes: &[u8]) -> Result<Arc<arrow::datatypes::Schema>> {
    let proto_schema = datafusion_proto::protobuf::Schema::decode(bytes)
        .map_err(|e| DataFusionError::Internal(format!("decode schema: {e}")))?;
    Ok(Arc::new(
        arrow::datatypes::Schema::try_from(&proto_schema)
            .map_err(|e| DataFusionError::Internal(format!("convert schema: {e}")))?,
    ))
}

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

fn build_single_table_scan_schema(
    canonical_ff_schema: &SchemaRef,
    projection: &Option<Vec<usize>>,
) -> Result<ScanSchema> {
    use arrow::datatypes::{DataType, Field, Schema};

    let mut unified_fields: Vec<Arc<Field>> = canonical_ff_schema.fields().to_vec();
    let score_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_score", DataType::Float32, true)));
    let document_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_document", DataType::Utf8, true)));
    let unified_schema = Arc::new(Schema::new(unified_fields));

    let actual_indices: Vec<usize> = match projection {
        Some(indices) => indices.clone(),
        None => (0..unified_schema.fields().len()).collect(),
    };

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
    let doc_id_idx = canonical_ff_schema.index_of("_doc_id").map_err(|_| {
        DataFusionError::Internal(
            "canonical fast field schema missing required _doc_id column".into(),
        )
    })?;
    let segment_ord_idx = canonical_ff_schema.index_of("_segment_ord").map_err(|_| {
        DataFusionError::Internal(
            "canonical fast field schema missing required _segment_ord column".into(),
        )
    })?;
    if ff_indices.is_empty() || (needs_document && !ff_indices.contains(&doc_id_idx)) {
        ff_indices.push(doc_id_idx);
    }
    if needs_document && !ff_indices.contains(&segment_ord_idx) {
        ff_indices.push(segment_ord_idx);
    }
    ff_indices.sort();
    ff_indices.dedup();

    let ff_projected = {
        let fields: Vec<arrow::datatypes::Field> = ff_indices
            .iter()
            .map(|&i| canonical_ff_schema.field(i).clone())
            .collect();
        Arc::new(Schema::new(fields))
    };

    let projected = {
        let fields: Vec<arrow::datatypes::Field> = actual_indices
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

impl PhysicalExtensionCodec for TantivyCodec {
    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        let ds_exec = node
            .as_any()
            .downcast_ref::<DataSourceExec>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!("TantivyCodec: unsupported node {}", node.name()))
            })?;
        let ds = ds_exec.data_source();

        if let Some(st) = ds.as_any().downcast_ref::<SingleTableDataSource>() {
            let split_descriptors = st
                .split_descriptors()
                .iter()
                .map(encode_split_descriptor)
                .collect::<Result<Vec<_>>>()?;
            let (projection, has_projection) = match st.projection() {
                Some(indices) => (indices.iter().map(|&idx| idx as u32).collect(), true),
                None => (Vec::new(), false),
            };
            let raw_queries_json = serde_json::to_string(st.raw_queries())
                .map_err(|e| DataFusionError::Internal(format!("serialize raw_queries: {e}")))?;
            let fast_field_filters_json =
                serialize_fast_field_filters(st.fast_field_filter_exprs())?;
            let canonical_ff_schema_bytes =
                encode_schema_bytes(st.canonical_fast_field_schema().as_ref())?;
            let (topk, has_topk) = match st.topk() {
                Some(value) => (value as u32, true),
                None => (0, false),
            };
            let (row_limit, has_row_limit) = match st.row_limit() {
                Some(value) => (value as u32, true),
                None => (0, false),
            };

            return TantivyPlanProto {
                provider_type: SINGLE_TABLE,
                projection,
                has_projection,
                raw_queries_json,
                topk,
                has_topk,
                fast_field_filters_json,
                split_descriptors,
                canonical_ff_schema_bytes,
                aggregations_json: String::new(),
                output_schema_bytes: Vec::new(),
                agg_output_mode: AGG_OUTPUT_FINAL_MERGED,
                row_limit,
                has_row_limit,
            }
            .encode(buf)
            .map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
        }

        if let Some(agg_ds) = ds.as_any().downcast_ref::<AggDataSource>() {
            let split_descriptors = agg_ds
                .split_descriptors()
                .iter()
                .map(encode_split_descriptor)
                .collect::<Result<Vec<_>>>()?;
            let aggregations_json = serde_json::to_string(agg_ds.aggregations().as_ref())
                .map_err(|e| DataFusionError::Internal(format!("serialize aggregations: {e}")))?;
            let raw_queries_json = serde_json::to_string(agg_ds.raw_queries())
                .map_err(|e| DataFusionError::Internal(format!("serialize raw_queries: {e}")))?;
            let output_schema_bytes = encode_schema_bytes(agg_ds.output_schema().as_ref())?;
            let fast_field_filters_json =
                serialize_fast_field_filters(agg_ds.fast_field_filter_exprs())?;
            let agg_output_mode = match agg_ds.output_mode() {
                AggOutputMode::FinalMerged => AGG_OUTPUT_FINAL_MERGED,
                AggOutputMode::PartialStates => AGG_OUTPUT_PARTIAL_STATES,
            };

            return TantivyPlanProto {
                provider_type: AGG_DATA_SOURCE,
                projection: Vec::new(),
                has_projection: false,
                raw_queries_json,
                topk: 0,
                has_topk: false,
                fast_field_filters_json,
                split_descriptors,
                canonical_ff_schema_bytes: Vec::new(),
                aggregations_json,
                output_schema_bytes,
                agg_output_mode,
                row_limit: 0,
                has_row_limit: false,
            }
            .encode(buf)
            .map_err(|e| DataFusionError::Internal(format!("encode: {e}")));
        }

        Err(DataFusionError::Internal(format!(
            "TantivyCodec: unsupported data source {}",
            ds_exec.name()
        )))
    }

    fn try_decode(
        &self,
        buf: &[u8],
        _inputs: &[Arc<dyn ExecutionPlan>],
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let proto = TantivyPlanProto::decode(buf)
            .map_err(|e| DataFusionError::Internal(format!("decode: {e}")))?;
        let projection = if proto.has_projection {
            Some(
                proto
                    .projection
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        match proto.provider_type {
            SINGLE_TABLE => decode_single_table(&proto, projection),
            AGG_DATA_SOURCE => decode_agg(&proto),
            other => Err(DataFusionError::Internal(format!(
                "unknown tantivy provider type: {other}"
            ))),
        }
    }
}

fn decode_single_table(
    proto: &TantivyPlanProto,
    projection: Option<Vec<usize>>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if proto.canonical_ff_schema_bytes.is_empty() {
        return Err(DataFusionError::Internal(
            "missing canonical fast field schema for SINGLE_TABLE".into(),
        ));
    }
    let canonical_ff_schema = decode_schema_bytes(&proto.canonical_ff_schema_bytes)?;
    let scan_schema = build_single_table_scan_schema(&canonical_ff_schema, &projection)?;
    let raw_queries: Vec<(String, String)> = if proto.raw_queries_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&proto.raw_queries_json)
            .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?
    };
    let fast_field_filter_exprs =
        deserialize_fast_field_filter_exprs(&proto.fast_field_filters_json)?;
    let split_descriptors = proto
        .split_descriptors
        .iter()
        .map(decode_split_descriptor)
        .collect::<Result<Vec<_>>>()?;
    let splits = split_descriptors
        .into_iter()
        .map(|descriptor| SplitExecutionPlan {
            descriptor,
            opener: None,
        })
        .collect();
    let partition_map = (0..proto.split_descriptors.len())
        .map(|split_idx| PartitionSpec { split_idx })
        .collect();
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

fn decode_agg(proto: &TantivyPlanProto) -> Result<Arc<dyn ExecutionPlan>> {
    let aggregations = serde_json::from_str(&proto.aggregations_json)
        .map_err(|e| DataFusionError::Internal(format!("parse aggregations: {e}")))?;
    if proto.output_schema_bytes.is_empty() {
        return Err(DataFusionError::Internal(
            "missing output schema for AGG_DATA_SOURCE".into(),
        ));
    }
    let output_schema = decode_schema_bytes(&proto.output_schema_bytes)?;
    let raw_queries: Vec<(String, String)> = if proto.raw_queries_json.is_empty() {
        Vec::new()
    } else {
        serde_json::from_str(&proto.raw_queries_json)
            .map_err(|e| DataFusionError::Internal(format!("parse raw_queries: {e}")))?
    };
    let split_descriptors = proto
        .split_descriptors
        .iter()
        .map(decode_split_descriptor)
        .collect::<Result<Vec<_>>>()?;
    let pre_built_query = if split_descriptors.len() == 1 {
        reconstruct_pre_built_query(
            &proto.fast_field_filters_json,
            &split_descriptors[0].tantivy_schema,
        )?
    } else {
        None
    };
    let fast_field_filter_exprs =
        deserialize_fast_field_filter_exprs(&proto.fast_field_filters_json)?;

    let aggregations = Arc::new(aggregations);
    let ds = match proto.agg_output_mode {
        AGG_OUTPUT_PARTIAL_STATES => AggDataSource::from_split_descriptors_partial_states(
            split_descriptors,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        ),
        _ => AggDataSource::from_split_descriptors(
            split_descriptors,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        ),
    };
    Ok(Arc::new(DataSourceExec::new(Arc::new(ds))))
}
