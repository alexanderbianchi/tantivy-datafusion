use std::sync::Arc;

use datafusion::datasource::TableProvider;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use tantivy::schema::{SchemaBuilder, FAST, STORED, TEXT};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{
    DirectIndexOpener, IndexOpener, full_text_udf,
    OpenerFactoryExt, OpenerMetadata, TantivyCodec,
    SingleTableProvider,
};
use tantivy_datafusion::unified::agg_data_source::AggDataSource;
use tantivy_datafusion::unified::single_table_provider::SingleTableDataSource;

/// Create a simple in-memory tantivy index for testing.
fn create_test_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_u64_field("id", FAST | STORED);
    builder.add_text_field("body", TEXT | STORED);
    builder.add_f64_field("price", FAST);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let id_field = schema.get_field("id").unwrap();
    let body_field = schema.get_field("body").unwrap();
    let price_field = schema.get_field("price").unwrap();

    for i in 0..5u64 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id_field, i);
        doc.add_text(body_field, format!("document number {i} about rust programming"));
        doc.add_f64(price_field, (i as f64) * 1.5 + 1.0);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
    index
}

/// Build a `SessionContext` with an opener factory that returns a
/// `DirectIndexOpener` wrapping the given index.
fn session_with_opener(index: Index) -> SessionContext {
    let mut config = SessionConfig::new();
    let test_index = index.clone();
    config.set_opener_factory(Arc::new(move |_meta: OpenerMetadata| -> Arc<dyn IndexOpener> {
        Arc::new(DirectIndexOpener::new(test_index.clone()))
    }));
    SessionContext::new_with_config(config)
}

// ── SingleTable provider roundtrip ───────────────────────────────

#[tokio::test]
async fn test_single_table_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = SingleTableProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    // Encode
    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();
    assert!(!buf.is_empty(), "encoded bytes should be non-empty");

    // Decode
    let decode_session = session_with_opener(index);
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
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = SingleTableProvider::from_opener(opener.clone());

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

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema().fields().len(), 2);
    assert_eq!(decoded.schema().field(0).name(), "id");
    assert_eq!(decoded.schema().field(1).name(), "price");
}

#[tokio::test]
async fn test_single_table_with_query_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = SingleTableProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();

    let filter = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        Arc::new(full_text_udf()),
        vec![col("body"), lit("rust")],
    ));
    let exec = provider
        .scan(&state, None, &[filter], None)
        .await
        .unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec.schema());
}

#[tokio::test]
async fn test_single_table_with_topk_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = SingleTableProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();

    let filter = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
        Arc::new(full_text_udf()),
        vec![col("body"), lit("rust")],
    ));
    let exec = provider
        .scan(&state, None, &[filter], None)
        .await
        .unwrap();

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

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec_with_topk.schema());
}


#[tokio::test]
async fn test_double_roundtrip_single_table() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = SingleTableProvider::from_opener(opener);

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let codec = TantivyCodec;

    // First roundtrip
    let mut buf1 = Vec::new();
    codec.try_encode(exec.clone(), &mut buf1).unwrap();

    let decode_session = session_with_opener(index.clone());
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
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));

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
        opener,
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
    let decode_session = session_with_opener(index);
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
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));

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
        opener,
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
    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(
        decoded.schema(),
        exec.schema(),
        "decoded schema must match original after query roundtrip"
    );
}
