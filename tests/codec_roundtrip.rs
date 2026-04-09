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
    DirectIndexOpener, FastFieldDataSource, IndexOpener, InvertedIndexDataSource,
    OpenerFactoryExt, OpenerMetadata, TantivyCodec, TantivyDocumentProvider,
    TantivyInvertedIndexProvider, TantivyTableProvider, DocumentDataSource, full_text_udf,
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

// ── FastField provider roundtrip ───────────────────────────────────

#[tokio::test]
async fn test_fast_field_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyTableProvider::from_opener(opener.clone());

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
async fn test_fast_field_with_projection_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyTableProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    // Project only "id" and "price" (indices 0 and 2 in the arrow schema
    // derived from tantivy fast fields).
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
async fn test_fast_field_with_pushed_filters_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyTableProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    // Manually push a filter: price > 3.0
    let ds_exec = exec.as_any().downcast_ref::<DataSourceExec>().unwrap();
    let ff_ds = ds_exec
        .data_source()
        .as_any()
        .downcast_ref::<FastFieldDataSource>()
        .unwrap();
    let price_idx = exec.schema().index_of("price").unwrap();
    let filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("price", price_idx)),
        datafusion::logical_expr::Operator::Gt,
        Arc::new(Literal::new(datafusion::common::ScalarValue::Float64(Some(
            3.0,
        )))),
    ));
    let updated_ds = ff_ds.with_pushed_filters(vec![filter.clone()]);
    let exec_with_filter = Arc::new(DataSourceExec::new(Arc::new(updated_ds)));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec
        .try_encode(exec_with_filter.clone(), &mut buf)
        .unwrap();
    assert!(!buf.is_empty());

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(
        decoded.schema(),
        exec_with_filter.schema(),
        "schema must match after filter roundtrip"
    );
}

// ── InvertedIndex provider roundtrip ───────────────────────────────

#[tokio::test]
async fn test_inverted_index_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyInvertedIndexProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    session.register_udf(full_text_udf());
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec.schema());
    assert_eq!(
        decoded.properties().partitioning.partition_count(),
        exec.properties().partitioning.partition_count(),
    );
}

#[tokio::test]
async fn test_inverted_index_with_query_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyInvertedIndexProvider::from_opener(opener.clone());

    // Push a full_text filter through the scan.
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
async fn test_inverted_index_with_topk_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyInvertedIndexProvider::from_opener(opener.clone());

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

    // Manually set topk on the InvertedIndexDataSource.
    let ds_exec = exec.as_any().downcast_ref::<DataSourceExec>().unwrap();
    let inv_ds = ds_exec
        .data_source()
        .as_any()
        .downcast_ref::<InvertedIndexDataSource>()
        .unwrap();
    let updated_ds = inv_ds.with_topk(10);
    assert_eq!(updated_ds.topk(), Some(10));
    let exec_with_topk = Arc::new(DataSourceExec::new(Arc::new(updated_ds)));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec_with_topk.clone(), &mut buf).unwrap();

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec_with_topk.schema());
    // The decoded plan is a LazyScanExec. We cannot directly inspect its topk
    // field (it is private), but we verify the roundtrip did not error,
    // meaning has_topk + topk were correctly serialized and the plan was
    // reconstructed without an error.
}

// ── Document provider roundtrip ────────────────────────────────────

#[tokio::test]
async fn test_document_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyDocumentProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec.clone(), &mut buf).unwrap();

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(decoded.schema(), exec.schema());
    assert_eq!(
        decoded.properties().partitioning.partition_count(),
        exec.properties().partitioning.partition_count(),
    );
}

#[tokio::test]
async fn test_document_with_pushed_filters_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyDocumentProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    // Push a filter on _doc_id: _doc_id < 3
    let ds_exec = exec.as_any().downcast_ref::<DataSourceExec>().unwrap();
    let doc_ds = ds_exec
        .data_source()
        .as_any()
        .downcast_ref::<DocumentDataSource>()
        .unwrap();
    let doc_id_idx = exec.schema().index_of("_doc_id").unwrap();
    let filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
        Arc::new(Column::new("_doc_id", doc_id_idx)),
        datafusion::logical_expr::Operator::Lt,
        Arc::new(Literal::new(datafusion::common::ScalarValue::UInt32(Some(3)))),
    ));
    let updated_ds = doc_ds.with_pushed_filters(vec![filter.clone()]);
    let exec_with_filter = Arc::new(DataSourceExec::new(Arc::new(updated_ds)));

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec
        .try_encode(exec_with_filter.clone(), &mut buf)
        .unwrap();
    assert!(!buf.is_empty());

    let decode_session = session_with_opener(index);
    let task_ctx = decode_session.state().task_ctx();
    let decoded = codec.try_decode(&buf, &[], &task_ctx).unwrap();

    assert_eq!(
        decoded.schema(),
        exec_with_filter.schema(),
        "document schema must match after filter roundtrip"
    );
}

#[tokio::test]
async fn test_document_with_projection_roundtrip() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyDocumentProvider::from_opener(opener.clone());

    let session = SessionContext::new();
    let state = session.state();
    // Project only _document (index 2 in the document provider schema).
    let projection = vec![2usize];
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

    assert_eq!(decoded.schema().fields().len(), 1);
    assert_eq!(decoded.schema().field(0).name(), "_document");
}

// ── Error case: missing opener factory ─────────────────────────────

#[tokio::test]
async fn test_decode_without_opener_factory_fails() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyTableProvider::from_opener(opener);

    let session = SessionContext::new();
    let state = session.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    let codec = TantivyCodec;
    let mut buf = Vec::new();
    codec.try_encode(exec, &mut buf).unwrap();

    // Decode with a plain SessionContext — no opener factory registered.
    let plain_session = SessionContext::new();
    let task_ctx = plain_session.state().task_ctx();
    let result = codec.try_decode(&buf, &[], &task_ctx);
    assert!(result.is_err(), "decode should fail without opener factory");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("OpenerFactory"),
        "error should mention OpenerFactory, got: {err_msg}"
    );
}

// ── Roundtrip stability: encode -> decode -> re-encode gives same bytes ─

#[tokio::test]
async fn test_double_roundtrip_fast_field() {
    let index = create_test_index();
    let opener = Arc::new(DirectIndexOpener::new(index.clone()));
    let provider = TantivyTableProvider::from_opener(opener);

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

    // The decoded plan is a LazyScanExec which also implements try_encode.
    // Re-encode it and verify the bytes are identical.
    let mut buf2 = Vec::new();
    codec.try_encode(decoded, &mut buf2).unwrap();
    assert_eq!(buf1, buf2, "double roundtrip must produce identical bytes");
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
