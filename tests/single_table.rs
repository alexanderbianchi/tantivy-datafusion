use std::net::Ipv4Addr;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, UInt64Type};
use datafusion::prelude::*;
use tantivy::schema::{Field, SchemaBuilder, FAST, STORED, TEXT};
use tantivy::{DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{full_text_udf, SingleTableProvider};

// ---------------------------------------------------------------------------
// Shared helpers (mirrored from end_to_end.rs)
// ---------------------------------------------------------------------------

fn plan_to_string(batches: &[RecordBatch]) -> String {
    let batch = collect_batches(batches);
    let plan_col = batch.column(1).as_string::<i32>();
    (0..batch.num_rows())
        .map(|i| plan_col.value(i))
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_test_schema() -> (
    tantivy::schema::Schema,
    Field, // id (u64)
    Field, // score (i64)
    Field, // price (f64)
    Field, // active (bool)
    Field, // category (text)
    Field, // created_at (date)
    Field, // ip_address (ip)
    Field, // data (bytes)
) {
    let mut builder = SchemaBuilder::new();
    let u64_field = builder.add_u64_field("id", FAST | STORED);
    let i64_field = builder.add_i64_field("score", FAST);
    let f64_field = builder.add_f64_field("price", FAST);
    let bool_field = builder.add_bool_field("active", FAST);
    let text_field = builder.add_text_field("category", TEXT | FAST | STORED);
    let date_field = builder.add_date_field("created_at", FAST);
    let ip_field = builder.add_ip_addr_field("ip_address", FAST);
    let bytes_field = builder.add_bytes_field("data", FAST);
    let schema = builder.build();
    (
        schema,
        u64_field,
        i64_field,
        f64_field,
        bool_field,
        text_field,
        date_field,
        ip_field,
        bytes_field,
    )
}

fn add_test_documents(
    writer: &IndexWriter,
    fields: (Field, Field, Field, Field, Field, Field, Field, Field),
) {
    let (u64_field, i64_field, f64_field, bool_field, text_field, date_field, ip_field, bytes_field) =
        fields;

    let timestamps = [1_000_000i64, 2_000_000, 3_000_000, 4_000_000, 5_000_000];
    let ips: [Ipv4Addr; 5] = [
        Ipv4Addr::new(192, 168, 1, 1),
        Ipv4Addr::new(10, 0, 0, 1),
        Ipv4Addr::new(192, 168, 1, 2),
        Ipv4Addr::new(10, 0, 0, 2),
        Ipv4Addr::new(172, 16, 0, 1),
    ];
    let data_payloads: [&[u8]; 5] = [b"aaa", b"bbb", b"ccc", b"ddd", b"eee"];

    let ids = [1u64, 2, 3, 4, 5];
    let scores = [10i64, 20, 30, 40, 50];
    let prices = [1.5f64, 2.5, 3.5, 4.5, 5.5];
    let actives = [true, false, true, false, true];
    let categories = ["electronics", "books", "electronics", "books", "clothing"];

    for i in 0..5 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(u64_field, ids[i]);
        doc.add_i64(i64_field, scores[i]);
        doc.add_f64(f64_field, prices[i]);
        doc.add_bool(bool_field, actives[i]);
        doc.add_text(text_field, categories[i]);
        doc.add_date(date_field, DateTime::from_timestamp_micros(timestamps[i]));
        doc.add_ip_addr(ip_field, ips[i].to_ipv6_mapped());
        doc.add_bytes(bytes_field, data_payloads[i]);
        writer.add_document(doc).unwrap();
    }
}

fn create_test_index() -> Index {
    let (schema, u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f) =
        build_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    add_test_documents(
        &writer,
        (u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f),
    );
    writer.commit().unwrap();
    index
}

fn collect_batches(batches: &[RecordBatch]) -> RecordBatch {
    arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_single_fast_fields_only() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id, price FROM t WHERE price > 2.0 ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    // price > 2.0 -> ids {2, 3, 4, 5}
    assert_eq!(batch.num_rows(), 4);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 4);
    assert_eq!(ids.value(3), 5);
}

#[tokio::test]
async fn test_single_full_text_with_score() {
    use arrow::datatypes::Float32Type;

    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id, _score FROM t WHERE full_text(category, 'electronics') ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    // electronics -> ids {1, 3}
    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);

    // Scores should be positive
    let scores = batch.column(1).as_primitive::<Float32Type>();
    assert!(scores.value(0) > 0.0);
    assert!(scores.value(1) > 0.0);
}

#[tokio::test]
async fn test_single_full_text_with_filter() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql(
            "SELECT id, price FROM t \
             WHERE full_text(category, 'electronics') AND price > 2.0 \
             ORDER BY id",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    // electronics AND price > 2.0 -> only id=3
    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 3);
}

#[tokio::test]
async fn test_single_with_document() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql(
            "SELECT id, _document FROM t \
             WHERE full_text(category, 'electronics') \
             ORDER BY id",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);

    // _document should be valid JSON
    let docs = batch.column(1).as_string::<i32>();
    let doc0: serde_json::Value = serde_json::from_str(docs.value(0)).unwrap();
    assert_eq!(doc0["id"][0], 1);
    let doc1: serde_json::Value = serde_json::from_str(docs.value(1)).unwrap();
    assert_eq!(doc1["id"][0], 3);
}

#[tokio::test]
async fn test_single_three_way() {
    use arrow::datatypes::Float32Type;

    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql(
            "SELECT id, price, _score, _document FROM t \
             WHERE full_text(category, 'electronics') AND price > 2.0 \
             ORDER BY _score DESC LIMIT 10",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    // electronics AND price > 2.0 -> only id=3
    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 3);
    let price = batch.column(1).as_primitive::<Float64Type>().value(0);
    assert!((price - 3.5).abs() < 1e-10);
    let score = batch.column(2).as_primitive::<Float32Type>().value(0);
    assert!(score > 0.0);
    let doc: serde_json::Value =
        serde_json::from_str(batch.column(3).as_string::<i32>().value(0)).unwrap();
    assert_eq!(doc["id"][0], 3);
}

#[tokio::test]
async fn test_single_score_null_without_query() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id, _score FROM t ORDER BY id LIMIT 3")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);
    // _score should be null when no full_text() filter
    let scores = batch.column(1);
    assert!(scores.is_null(0), "_score should be null without a query");
}

#[tokio::test]
async fn test_single_document_without_inverted_index() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id, _document FROM t WHERE id = 1")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 1);

    let doc: serde_json::Value =
        serde_json::from_str(batch.column(1).as_string::<i32>().value(0)).unwrap();
    assert_eq!(doc["id"][0], 1);
}

#[tokio::test]
async fn test_single_plan_no_joins() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("EXPLAIN SELECT id, price FROM t WHERE price > 2.0")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let plan = plan_to_string(&batches);

    // Should NOT contain HashJoinExec -- single-table approach doesn't use joins
    assert!(
        !plan.contains("HashJoinExec"),
        "Single-table provider should not have joins.\n\nPlan:\n{plan}"
    );
    assert!(
        plan.contains("SingleTableDataSource"),
        "Plan should contain SingleTableDataSource.\n\nPlan:\n{plan}"
    );
}

#[tokio::test]
async fn test_single_plan_with_query() {
    let index = create_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql(
            "EXPLAIN SELECT id, _score FROM t \
             WHERE full_text(category, 'electronics')",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let plan = plan_to_string(&batches);

    // Key difference: even with FTS query, should NOT contain HashJoinExec
    assert!(
        !plan.contains("HashJoinExec"),
        "Single-table provider should not have joins even with FTS query.\n\nPlan:\n{plan}"
    );
    assert!(
        plan.contains("SingleTableDataSource"),
        "Plan should contain SingleTableDataSource.\n\nPlan:\n{plan}"
    );
}
