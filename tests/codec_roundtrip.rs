use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow::datatypes::{Field, Schema};
use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::common::ScalarValue;
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use datafusion_datasource::source::DataSourceExec;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use tantivy::schema::{SchemaBuilder, FAST, STORED, STRING, TEXT};
use tantivy::{DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::unified::agg_data_source::{AggDataSource, AggOutputMode};
use tantivy_datafusion::unified::single_table_provider::SingleTableDataSource;
use tantivy_datafusion::{
    full_text_udf, PreparedSplit, SingleTableProvider, SplitDescriptor, SplitRuntimeFactory,
    SplitRuntimeFactoryExt, TantivyCodec,
};

#[derive(Debug)]
struct StaticSplitRuntimeFactory {
    indices: Arc<HashMap<String, Index>>,
    fallback: Option<Index>,
}

#[async_trait]
impl SplitRuntimeFactory for StaticSplitRuntimeFactory {
    async fn prepare_split(&self, descriptor: &SplitDescriptor) -> Result<Arc<PreparedSplit>> {
        let index = self
            .indices
            .get(&descriptor.split_id)
            .cloned()
            .or_else(|| self.fallback.clone())
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Internal(format!(
                    "missing test split {}",
                    descriptor.split_id
                ))
            })?;
        Ok(Arc::new(PreparedSplit::new(index, Arc::new(()))?))
    }
}

/// Create a simple in-memory tantivy index for testing.
fn create_test_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_u64_field("id", FAST | STORED);
    builder.add_i64_field("score_i64", FAST);
    builder.add_text_field("body", TEXT | STORED);
    builder.add_f64_field("price", FAST);
    builder.add_bool_field("active", FAST);
    builder.add_date_field("created_at", FAST);
    builder.add_text_field("category", STRING | FAST | STORED);
    builder.add_text_field("tags", STRING | FAST | STORED);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let id_field = schema.get_field("id").unwrap();
    let score_i64_field = schema.get_field("score_i64").unwrap();
    let body_field = schema.get_field("body").unwrap();
    let price_field = schema.get_field("price").unwrap();
    let active_field = schema.get_field("active").unwrap();
    let created_at_field = schema.get_field("created_at").unwrap();
    let category_field = schema.get_field("category").unwrap();
    let tags_field = schema.get_field("tags").unwrap();

    for i in 0..5u64 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id_field, i);
        doc.add_i64(score_i64_field, (i as i64) * 10);
        doc.add_text(
            body_field,
            format!("document number {i} about rust programming"),
        );
        doc.add_f64(price_field, (i as f64) * 1.5 + 1.0);
        doc.add_bool(active_field, i % 2 == 0);
        doc.add_date(
            created_at_field,
            DateTime::from_timestamp_micros((i as i64 + 1) * 1_000_000),
        );
        doc.add_text(
            category_field,
            match i % 3 {
                0 => "books",
                1 => "electronics",
                _ => "clothing",
            },
        );
        doc.add_text(tags_field, format!("tag-{}", i % 2));
        if i % 2 == 0 {
            doc.add_text(tags_field, "shared");
        }
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
    index
}

/// Build a `SessionContext` with a split runtime factory that returns the
/// given index for single-split decode tests.
fn session_with_index(index: Index) -> SessionContext {
    let mut config = SessionConfig::new();
    config.set_split_runtime_factory(Arc::new(StaticSplitRuntimeFactory {
        indices: Arc::new(HashMap::new()),
        fallback: Some(index),
    }));
    SessionContext::new_with_config(config)
}

fn session_with_named_openers(indices: HashMap<String, Index>) -> SessionContext {
    let mut config = SessionConfig::new();
    config.set_split_runtime_factory(Arc::new(StaticSplitRuntimeFactory {
        indices: Arc::new(indices),
        fallback: None,
    }));
    SessionContext::new_with_config(config)
}

fn roundtrip_exec(exec: Arc<dyn ExecutionPlan>, index: Index) -> Arc<dyn ExecutionPlan> {
    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec, &mut buf).unwrap();

    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    codec.try_decode(&buf, &[], &task_ctx).unwrap()
}

fn data_source_exec(plan: &Arc<dyn ExecutionPlan>) -> &DataSourceExec {
    plan.as_any().downcast_ref::<DataSourceExec>().unwrap()
}

fn single_table_ds(plan: &Arc<dyn ExecutionPlan>) -> &SingleTableDataSource {
    data_source_exec(plan)
        .data_source()
        .as_any()
        .downcast_ref::<SingleTableDataSource>()
        .unwrap()
}

fn agg_ds(plan: &Arc<dyn ExecutionPlan>) -> &AggDataSource {
    data_source_exec(plan)
        .data_source()
        .as_any()
        .downcast_ref::<AggDataSource>()
        .unwrap()
}

fn create_int_score_index(start_id: u64) -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let score = builder.add_i64_field("score", FAST);
    let schema = builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (offset, score_value) in [(0u64, 10i64), (1, 20)] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, start_id + offset);
        doc.add_i64(score, score_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn create_float_score_index(start_id: u64) -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let score = builder.add_f64_field("score", FAST);
    let schema = builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (offset, score_value) in [(0u64, 30.5f64), (1, 40.0)] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, start_id + offset);
        doc.add_f64(score, score_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

// ── SingleTable provider roundtrip ───────────────────────────────

#[tokio::test]
async fn test_single_table_roundtrip() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    // Encode
    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();
    assert!(!buf.is_empty(), "encoded bytes should be non-empty");

    // Decode
    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    // The decoded plan is a LazyScanExec — verify schema matches.
    assert_eq!(
        decoded.schema(),
        exec.schema(),
        "decoded schema must match original"
    );
    // Partition count should survive the roundtrip.
    assert_eq!(
        decoded.properties().partitioning.partition_count(),
        exec.properties().partitioning.partition_count(),
        "partition count must match"
    );
}

#[tokio::test]
async fn test_single_table_with_projection_roundtrip() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    let state = session.state();
    let full_schema = provider.schema();
    let id_idx = full_schema.index_of("id").unwrap();
    let price_idx = full_schema.index_of("price").unwrap();
    let projection = vec![id_idx, price_idx];
    let exec = provider
        .scan(&state, Some(&projection), &[], None)
        .await
        .unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema().fields().len(), 2);
    assert_eq!(decoded.schema().field(0).name(), "id");
    assert_eq!(decoded.schema().field(1).name(), "price");
}

#[tokio::test]
async fn test_single_table_with_query_roundtrip() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();

    let filter = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        Arc::new(full_text_udf()),
        vec![col("body"), lit("rust")],
    ));
    let exec = provider.scan(&state, None, &[filter], None).await.unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec.schema());
}

#[tokio::test]
async fn test_single_table_with_topk_roundtrip() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();

    let filter = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        Arc::new(full_text_udf()),
        vec![col("body"), lit("rust")],
    ));
    let exec = provider.scan(&state, None, &[filter], None).await.unwrap();

    // Manually set topk on the SingleTableDataSource.
    let ds_exec = exec.as_any().downcast_ref::<DataSourceExec>().unwrap();
    let st_ds = ds_exec
        .data_source()
        .as_any()
        .downcast_ref::<SingleTableDataSource>()
        .unwrap();
    let updated_ds = st_ds.with_topk(10);
    assert_eq!(updated_ds.topk(), Some(10));
    let exec_with_topk = Arc::new(DataSourceExec::new(Arc::new(updated_ds)));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec_with_topk.clone(), &mut buf).unwrap();

    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec_with_topk.schema());
    assert_eq!(single_table_ds(&decoded).topk(), Some(10));
}

#[tokio::test]
async fn test_single_table_with_row_limit_roundtrip() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());
    let session = SessionContext::new();
    let state = session.state();

    let exec = provider.scan(&state, None, &[], Some(7)).await.unwrap();
    assert_eq!(single_table_ds(&exec).row_limit(), Some(7));

    let decoded = roundtrip_exec(exec, index);
    assert_eq!(single_table_ds(&decoded).row_limit(), Some(7));
}

#[tokio::test]
async fn test_multi_split_single_table_roundtrip() {
    let left = create_int_score_index(0);
    let right = create_float_score_index(100);

    let canonical_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, true),
        Field::new("score", DataType::Float64, true),
    ]));

    let provider = SingleTableProvider::from_local_splits_with_fast_field_schema(
        vec![left.clone(), right.clone()],
        canonical_schema,
    )
    .unwrap();

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let mut split_indices = HashMap::new();
    split_indices.insert("local-split-0".to_string(), left);
    split_indices.insert("local-split-1".to_string(), right);
    let decode_session = session_with_named_openers(split_indices);

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();
    let decoded = codec
        .try_decode(&buf, &[], &decode_session.state().task_ctx())
        .unwrap();

    let decoded_ds = single_table_ds(&decoded);
    assert_eq!(decoded_ds.split_descriptors().len(), 2);
    assert!(decoded_ds.local_runtime_factory().is_none());
    assert_eq!(
        decoded.properties().partitioning.partition_count(),
        exec.properties().partitioning.partition_count()
    );
}

#[tokio::test]
async fn test_double_roundtrip_single_table() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let codec = TantivyCodec;

    // First roundtrip
    let mut buf1 = Vec::new();
    codec.try_encode(exec.clone(), &mut buf1).unwrap();

    let decode_session = session_with_index(index.clone());
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf1, &[], &task_ctx).unwrap();

    // Re-encode and verify bytes are identical.
    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).unwrap();
    assert_eq!(buf1, buf2, "double roundtrip must produce identical bytes");
}

// ── AggDataSource roundtrip ───────────────────────────────────────

#[tokio::test]
async fn test_agg_data_source_roundtrip() {
    let index = create_test_index();

    // Build a simple terms aggregation on "body" field.
    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "terms_body": { "terms": { "field": "body" } }
        }))
        .unwrap();

    // Build an output schema matching what a terms agg would produce.
    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("body", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", arrow::datatypes::DataType::Int64, false),
    ]));

    let agg_ds = AggDataSource::new(
        index.clone(),
        Arc::new(aggs),
        output_schema.clone(),
        Vec::new(),
        None,
        Vec::new(),
    );
    let exec = Arc::new(DataSourceExec::new(Arc::new(agg_ds)));

    // Encode
    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();
    assert!(!buf.is_empty(), "encoded bytes should be non-empty");

    // Decode
    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(
        decoded.schema(),
        exec.schema(),
        "decoded schema must match original"
    );
    assert_eq!(
        decoded.properties().partitioning.partition_count(),
        exec.properties().partitioning.partition_count(),
        "partition count must match"
    );
}

#[tokio::test]
async fn test_agg_data_source_with_query_roundtrip() {
    let index = create_test_index();

    // Build a terms aggregation with a FTS filter.
    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "terms_body": { "terms": { "field": "body" } }
        }))
        .unwrap();

    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("body", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", arrow::datatypes::DataType::Int64, false),
    ]));

    // raw_queries simulates a full_text(body, 'rust') filter.
    let raw_queries = vec![("body".to_string(), "rust".to_string())];

    let agg_ds = AggDataSource::new(
        index.clone(),
        Arc::new(aggs),
        output_schema.clone(),
        raw_queries,
        None,
        Vec::new(),
    );
    let exec = Arc::new(DataSourceExec::new(Arc::new(agg_ds)));

    // Encode
    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();
    assert!(!buf.is_empty(), "encoded bytes should be non-empty");

    // Decode
    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(
        decoded.schema(),
        exec.schema(),
        "decoded schema must match original after query roundtrip"
    );
}

#[tokio::test]
async fn test_multi_split_agg_data_source_roundtrip() {
    let left = create_int_score_index(0);
    let right = create_float_score_index(100);

    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "terms_id": { "terms": { "field": "id" } }
        }))
        .unwrap();

    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", arrow::datatypes::DataType::Int64, false),
    ]));

    let exec = Arc::new(DataSourceExec::new(Arc::new(
        AggDataSource::from_local_splits(
            vec![left.clone(), right.clone()],
            Arc::new(aggs),
            output_schema,
            Vec::new(),
            None,
            Vec::new(),
        ),
    )));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let mut split_indices = HashMap::new();
    split_indices.insert("local-split-0".to_string(), left);
    split_indices.insert("local-split-1".to_string(), right);
    let decode_session = session_with_named_openers(split_indices);
    let decoded = codec
        .try_decode(&buf, &[], &decode_session.state().task_ctx())
        .unwrap();

    let decoded_ds = agg_ds(&decoded);
    assert_eq!(decoded_ds.split_descriptors().len(), 2);
    assert!(decoded_ds.local_runtime_factory().is_none());
}

#[tokio::test]
async fn test_multi_split_partial_state_agg_data_source_roundtrip() {
    let left = create_int_score_index(0);
    let right = create_float_score_index(100);

    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "group": { "terms": { "field": "id" } }
        }))
        .unwrap();

    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new(
            "count(Int64(1))[count]",
            arrow::datatypes::DataType::Int64,
            false,
        ),
    ]));

    let exec = Arc::new(DataSourceExec::new(Arc::new(
        AggDataSource::from_local_splits_partial_states(
            vec![left.clone(), right.clone()],
            Arc::new(aggs),
            output_schema,
            Vec::new(),
            None,
            Vec::new(),
        ),
    )));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let mut split_indices = HashMap::new();
    split_indices.insert("local-split-0".to_string(), left);
    split_indices.insert("local-split-1".to_string(), right);
    let decode_session = session_with_named_openers(split_indices);
    let decoded = codec
        .try_decode(&buf, &[], &decode_session.state().task_ctx())
        .unwrap();

    let decoded_ds = agg_ds(&decoded);
    assert_eq!(decoded_ds.split_descriptors().len(), 2);
    assert_eq!(decoded_ds.output_mode(), AggOutputMode::PartialStates);
}

#[tokio::test]
async fn test_multi_split_partial_state_agg_data_source_with_fast_field_filters_roundtrip() {
    let left = create_test_index();
    let right = create_test_index();
    let filter = col("price").gt(lit(2.0));

    let provider =
        SingleTableProvider::from_local_splits(vec![left.clone(), right.clone()]).unwrap();
    let session = SessionContext::new();
    let state = session.state();
    let scan_exec = provider.scan(&state, None, &[filter], None).await.unwrap();
    let scan_ds = single_table_ds(&scan_exec);

    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "category_terms": {
                "terms": { "field": "category" }
            }
        }))
        .unwrap();
    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("category", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", arrow::datatypes::DataType::Int64, false),
    ]));
    let exec = Arc::new(DataSourceExec::new(Arc::new(
        AggDataSource::from_local_splits_partial_states(
            vec![left.clone(), right.clone()],
            Arc::new(aggs),
            output_schema,
            scan_ds.raw_queries().to_vec(),
            scan_ds.pre_built_query().cloned(),
            scan_ds.fast_field_filter_exprs().to_vec(),
        ),
    )));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let mut split_indices = HashMap::new();
    split_indices.insert("local-split-0".to_string(), left);
    split_indices.insert("local-split-1".to_string(), right);
    let decode_session = session_with_named_openers(split_indices);
    let decoded = codec
        .try_decode(&buf, &[], &decode_session.state().task_ctx())
        .unwrap();

    let decoded_ds = agg_ds(&decoded);
    assert_eq!(decoded_ds.output_mode(), AggOutputMode::PartialStates);
    assert!(!decoded_ds.fast_field_filter_exprs().is_empty());
    assert!(decoded_ds.pre_built_query().is_none());
}

#[tokio::test]
async fn test_codec_roundtrip_with_fast_field_filters() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());
    let session = SessionContext::new();
    let state = session.state();

    let timestamp_filter = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
        left: Box::new(col("created_at")),
        op: datafusion::logical_expr::Operator::GtEq,
        right: Box::new(Expr::Literal(
            ScalarValue::TimestampMicrosecond(Some(3_000_000), Some(Arc::<str>::from("UTC"))),
            None,
        )),
    });

    let filters = vec![
        col("id").gt(lit(1_u64)),
        col("score_i64").gt_eq(lit(10_i64)),
        col("price").lt(lit(6.0_f64)),
        col("active").eq(lit(true)),
        col("category").eq(lit("electronics")),
        timestamp_filter,
    ];

    for filter in filters {
        let exec = provider.scan(&state, None, &[filter], None).await.unwrap();
        let decoded = roundtrip_exec(exec, index.clone());
        assert!(
            single_table_ds(&decoded).pre_built_query().is_some(),
            "decoded plan should retain a tantivy fast-field query",
        );
    }
}

#[tokio::test]
async fn test_codec_roundtrip_fts_plus_fast_field_filter() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();

    let filters = vec![
        Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
            Arc::new(full_text_udf()),
            vec![col("body"), lit("rust")],
        )),
        col("price").gt(lit(2.0_f64)),
    ];

    let exec = provider.scan(&state, None, &filters, None).await.unwrap();
    let decoded = roundtrip_exec(exec, index);
    let decoded_ds = single_table_ds(&decoded);

    assert_eq!(
        decoded_ds.raw_queries(),
        [("body".to_string(), "rust".to_string())]
    );
    assert!(decoded_ds.pre_built_query().is_some());
}

#[tokio::test]
async fn test_codec_roundtrip_agg_with_fast_field_filters() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());
    let session = SessionContext::new();
    let state = session.state();

    let scan_exec = provider
        .scan(&state, None, &[col("price").gt(lit(2.0_f64))], None)
        .await
        .unwrap();
    let scan_ds = scan_exec
        .as_any()
        .downcast_ref::<DataSourceExec>()
        .unwrap()
        .data_source()
        .as_any()
        .downcast_ref::<SingleTableDataSource>()
        .unwrap();

    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "terms_category": { "terms": { "field": "category" } }
        }))
        .unwrap();
    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("category", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", arrow::datatypes::DataType::Int64, false),
    ]));
    let agg_exec = Arc::new(DataSourceExec::new(Arc::new(AggDataSource::new(
        index.clone(),
        Arc::new(aggs),
        output_schema,
        scan_ds.raw_queries().to_vec(),
        scan_ds.pre_built_query().cloned(),
        scan_ds.fast_field_filter_exprs().to_vec(),
    ))));

    let decoded = roundtrip_exec(agg_exec, index);
    assert!(agg_ds(&decoded).pre_built_query().is_some());
}

#[tokio::test]
async fn test_codec_roundtrip_multi_valued_field_schema() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index.clone());
    let session = SessionContext::new();
    let state = session.state();

    let projection = vec![provider.schema().index_of("tags").unwrap()];
    let exec = provider
        .scan(&state, Some(&projection), &[], None)
        .await
        .unwrap();
    let decoded = roundtrip_exec(exec, index);
    let schema = decoded.schema();
    let field = schema.field(0);

    assert_eq!(field.name(), "tags");
    assert_eq!(
        field.data_type(),
        &DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Utf8,
            true,
        ))),
    );
}

#[test]
fn test_new_preserves_multi_valued_schema_for_local_index() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);
    let schema = provider.schema();
    let field = schema.field(schema.index_of("tags").unwrap());

    assert_eq!(
        field.data_type(),
        &DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Utf8,
            true,
        ))),
    );
}

#[tokio::test]
async fn test_double_roundtrip_agg_data_source() {
    let index = create_test_index();
    let aggs: tantivy::aggregation::agg_req::Aggregations =
        serde_json::from_value(serde_json::json!({
            "terms_category": { "terms": { "field": "category" } }
        }))
        .unwrap();
    let output_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("category", DataType::Utf8, false),
        arrow::datatypes::Field::new("doc_count", DataType::Int64, false),
    ]));
    let exec = Arc::new(DataSourceExec::new(Arc::new(AggDataSource::new(
        index.clone(),
        Arc::new(aggs),
        output_schema,
        Vec::new(),
        None,
        Vec::new(),
    ))));

    let codec = TantivyCodec;
    let mut buf1 = Vec::new();
    codec.try_encode(exec, &mut buf1).unwrap();

    let decode_session = session_with_index(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf1, &[], &task_ctx).unwrap();

    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).unwrap();
    assert_eq!(buf1, buf2, "double roundtrip must produce identical bytes");
}
