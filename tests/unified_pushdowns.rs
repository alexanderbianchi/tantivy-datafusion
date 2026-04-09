use std::net::Ipv4Addr;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float32Type, Float64Type, Int64Type, UInt64Type};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use tantivy::schema::{Field, SchemaBuilder, FAST, STORED, TEXT};
use tantivy::{DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{
    full_text_udf, AggPushdown, SingleTableProvider,
};

// ---------------------------------------------------------------------------
// Shared helpers (mirrored from single_table.rs / end_to_end.rs)
// ---------------------------------------------------------------------------

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

fn plan_to_string(batches: &[RecordBatch]) -> String {
    let batch = collect_batches(batches);
    let plan_col = batch.column(1).as_string::<i32>();
    (0..batch.num_rows())
        .map(|i| plan_col.value(i))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create a session context with the unified optimizer rules and the
/// SingleTableProvider registered as table "t".
fn setup_ctx(index: Index) -> SessionContext {
    let provider = SingleTableProvider::new(index);
    let config = SessionConfig::new().with_target_partitions(1);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(AggPushdown::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();
    ctx
}

// =========================================================================
// Aggregation pushdown tests
// =========================================================================

/// Helper: collect GROUP BY category results into a sorted Vec of (category, count)
/// from a two-column batch [category: Utf8/Dict, count: Int64].
fn collect_category_counts(batch: &RecordBatch) -> Vec<(String, i64)> {
    let cat_col = batch.column(0);
    let count_col = batch.column(1).as_primitive::<Int64Type>();
    let mut results: Vec<(String, i64)> = (0..batch.num_rows())
        .map(|i| {
            // Category may be Dictionary or Utf8 depending on optimiser path.
            let cat = if let Some(dict) = cat_col
                .as_any()
                .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
            {
                let values = dict.values().as_string::<i32>();
                values.value(dict.keys().value(i) as usize).to_string()
            } else {
                cat_col.as_string::<i32>().value(i).to_string()
            };
            (cat, count_col.value(i))
        })
        .collect();
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

#[tokio::test]
async fn test_agg_pushdown_group_by_count() {
    // SELECT category, COUNT(*) FROM t GROUP BY category
    // Expected: books=2, clothing=1, electronics=2
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT category, COUNT(*) as cnt FROM t GROUP BY category")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 category groups");
    let results = collect_category_counts(&batch);
    assert_eq!(results[0], ("books".to_string(), 2));
    assert_eq!(results[1], ("clothing".to_string(), 1));
    assert_eq!(results[2], ("electronics".to_string(), 2));
}

#[tokio::test]
async fn test_agg_pushdown_group_by_sum_avg() {
    // SELECT category, SUM(price), AVG(price) FROM t GROUP BY category
    // Expected:
    //   books:       SUM=7.0  AVG=3.5
    //   clothing:    SUM=5.5  AVG=5.5
    //   electronics: SUM=5.0  AVG=2.5
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT category, SUM(price) as s, AVG(price) as a FROM t GROUP BY category")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 category groups");

    // Build a sorted lookup: category -> (sum, avg)
    let cat_col = batch.column(0);
    let sum_col = batch.column(1).as_primitive::<Float64Type>();
    let avg_col = batch.column(2).as_primitive::<Float64Type>();

    let mut results: Vec<(String, f64, f64)> = (0..batch.num_rows())
        .map(|i| {
            let cat = if let Some(dict) = cat_col
                .as_any()
                .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
            {
                let values = dict.values().as_string::<i32>();
                values.value(dict.keys().value(i) as usize).to_string()
            } else {
                cat_col.as_string::<i32>().value(i).to_string()
            };
            (cat, sum_col.value(i), avg_col.value(i))
        })
        .collect();
    results.sort_by(|a, b| a.0.cmp(&b.0));

    let eps = 1e-10;
    assert_eq!(results[0].0, "books");
    assert!((results[0].1 - 7.0).abs() < eps, "books SUM: {}", results[0].1);
    assert!((results[0].2 - 3.5).abs() < eps, "books AVG: {}", results[0].2);

    assert_eq!(results[1].0, "clothing");
    assert!((results[1].1 - 5.5).abs() < eps, "clothing SUM: {}", results[1].1);
    assert!((results[1].2 - 5.5).abs() < eps, "clothing AVG: {}", results[1].2);

    assert_eq!(results[2].0, "electronics");
    assert!((results[2].1 - 5.0).abs() < eps, "electronics SUM: {}", results[2].1);
    assert!((results[2].2 - 2.5).abs() < eps, "electronics AVG: {}", results[2].2);
}

#[tokio::test]
async fn test_agg_pushdown_with_fts_filter() {
    // SELECT category, COUNT(*) FROM t WHERE full_text(category, 'electronics') GROUP BY category
    // Should only see electronics=2
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             WHERE full_text(category, 'electronics') \
             GROUP BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1, "Expected 1 category group");
    let results = collect_category_counts(&batch);
    assert_eq!(results[0], ("electronics".to_string(), 2));
}

// =========================================================================
// Filter conversion tests
// =========================================================================

#[tokio::test]
async fn test_fast_field_filter_u64_eq() {
    // SELECT * FROM t WHERE id = 3
    // Verify only doc with id=3 returned
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT id, price FROM t WHERE id = 3")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 3);
    let price = batch.column(1).as_primitive::<Float64Type>().value(0);
    assert!((price - 3.5).abs() < 1e-10);
}

#[tokio::test]
async fn test_fast_field_filter_f64_range() {
    // SELECT * FROM t WHERE price > 2.0 AND price < 5.0
    // prices: 1.5, 2.5, 3.5, 4.5, 5.5 => matches 2.5, 3.5, 4.5 (ids 2,3,4)
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT id FROM t WHERE price > 2.0 AND price < 5.0 ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 rows in range (2.0, 5.0)");
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 4);
}

#[tokio::test]
async fn test_fast_field_filter_combined_with_fts() {
    // SELECT * FROM t WHERE full_text(category, 'electronics') AND price > 2.0
    // electronics: ids {1 (price=1.5), 3 (price=3.5)} => only id=3 matches
    let ctx = setup_ctx(create_test_index());

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

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 3);
    let price = batch.column(1).as_primitive::<Float64Type>().value(0);
    assert!((price - 3.5).abs() < 1e-10);
}

#[tokio::test]
async fn test_fast_field_filter_bool() {
    // SELECT * FROM t WHERE active = true
    // actives: [true, false, true, false, true] => ids {1, 3, 5}
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT id FROM t WHERE active = true ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 active=true rows");
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 5);
}

// =========================================================================
// TopK / LIMIT tests
// =========================================================================

#[tokio::test]
async fn test_topk_with_score_limit() {
    // SELECT id, _score FROM t WHERE full_text(category, 'electronics') ORDER BY _score DESC LIMIT 1
    // Should return 1 row with the highest score
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql(
            "SELECT id, _score FROM t \
             WHERE full_text(category, 'electronics') \
             ORDER BY _score DESC LIMIT 1",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1, "Expected exactly 1 row with LIMIT 1");
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    // Both electronics docs (1 and 3) have the same query; either is valid
    assert!(id == 1 || id == 3, "Expected id 1 or 3, got {id}");
    let score = batch.column(1).as_primitive::<Float32Type>().value(0);
    assert!(score > 0.0, "_score should be positive");
}

#[tokio::test]
async fn test_limit_without_score() {
    // SELECT id FROM t ORDER BY id LIMIT 2
    // Should return only 2 rows: ids 1 and 2
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT id FROM t ORDER BY id LIMIT 2")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2, "Expected 2 rows with LIMIT 2");
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
}

// =========================================================================
// Aggregation result correctness
// =========================================================================

#[tokio::test]
async fn test_agg_empty_result() {
    // SELECT category, COUNT(*) FROM t WHERE full_text(category, 'nonexistent') GROUP BY category
    // Should return empty result set
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             WHERE full_text(category, 'nonexistent') \
             GROUP BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "Expected 0 rows for nonexistent term");
}

// =========================================================================
// Plan shape tests
// =========================================================================

#[tokio::test]
async fn test_plan_has_no_joins() {
    // EXPLAIN SELECT category, COUNT(*) FROM t GROUP BY category
    // Verify no HashJoinExec; should use SingleTableDataSource or AggDataSource
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("EXPLAIN SELECT category, COUNT(*) FROM t GROUP BY category")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let plan = plan_to_string(&batches);

    assert!(
        !plan.contains("HashJoinExec"),
        "Plan should not contain HashJoinExec.\n\nPlan:\n{plan}"
    );
    // Should contain one of our custom data sources
    let has_custom_ds = plan.contains("SingleTableDataSource")
        || plan.contains("AggDataSource")
        || plan.contains("TantivyAggregateExec")
        || plan.contains("DenseOrdinalAggExec");
    assert!(
        has_custom_ds,
        "Plan should contain a tantivy-specific execution node.\n\nPlan:\n{plan}"
    );
}
