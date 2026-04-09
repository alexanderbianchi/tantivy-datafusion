use std::net::Ipv4Addr;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, Int64Type};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use tantivy::schema::{Field, SchemaBuilder, FAST, STORED, TEXT};
use tantivy::{DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{
    full_text_udf, AggPushdown, SingleTableProvider,
};

// ---------------------------------------------------------------------------
// Shared helpers
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

/// Read a string value from a column that may be StringArray or DictionaryArray.
fn string_val(col: &dyn arrow::array::Array, idx: usize) -> String {
    if let Some(s) = col
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
    {
        return s.value(idx).to_string();
    }
    if let Some(dict) = col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
    {
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        let key = dict.keys().value(idx) as usize;
        return values.value(key).to_string();
    }
    let cast = arrow::compute::cast(col, &arrow::datatypes::DataType::Utf8).unwrap();
    cast.as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap()
        .value(idx)
        .to_string()
}

// =========================================================================
// Tests
// =========================================================================

/// Test 1: Terms aggregation with sub-aggs
///
/// Data:
///   electronics: prices 1.5, 3.5 -> count=2, sum=5.0, avg=2.5, min=1.5, max=3.5
///   books:       prices 2.5, 4.5 -> count=2, sum=7.0, avg=3.5, min=2.5, max=4.5
///   clothing:    prices 5.5      -> count=1, sum=5.5, avg=5.5, min=5.5, max=5.5
#[tokio::test]
async fn test_terms_agg_with_sub_aggs() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt, SUM(price) as s, AVG(price) as a, \
             MIN(price) as mn, MAX(price) as mx \
             FROM t GROUP BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 category groups");

    let schema = batch.schema();
    let cat_col = batch.column(schema.index_of("category").unwrap());
    let cnt_col = batch
        .column(schema.index_of("cnt").unwrap())
        .as_primitive::<Int64Type>();
    let sum_col = batch
        .column(schema.index_of("s").unwrap())
        .as_primitive::<Float64Type>();
    let avg_col = batch
        .column(schema.index_of("a").unwrap())
        .as_primitive::<Float64Type>();
    let min_col = batch
        .column(schema.index_of("mn").unwrap())
        .as_primitive::<Float64Type>();
    let max_col = batch
        .column(schema.index_of("mx").unwrap())
        .as_primitive::<Float64Type>();

    // Collect and sort by category for deterministic assertions
    let mut rows: Vec<(String, i64, f64, f64, f64, f64)> = (0..batch.num_rows())
        .map(|i| {
            (
                string_val(cat_col.as_ref(), i),
                cnt_col.value(i),
                sum_col.value(i),
                avg_col.value(i),
                min_col.value(i),
                max_col.value(i),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let eps = 1e-10;

    // books: count=2, sum=7.0, avg=3.5, min=2.5, max=4.5
    assert_eq!(rows[0].0, "books");
    assert_eq!(rows[0].1, 2);
    assert!((rows[0].2 - 7.0).abs() < eps, "books SUM: {}", rows[0].2);
    assert!((rows[0].3 - 3.5).abs() < eps, "books AVG: {}", rows[0].3);
    assert!((rows[0].4 - 2.5).abs() < eps, "books MIN: {}", rows[0].4);
    assert!((rows[0].5 - 4.5).abs() < eps, "books MAX: {}", rows[0].5);

    // clothing: count=1, sum=5.5, avg=5.5, min=5.5, max=5.5
    assert_eq!(rows[1].0, "clothing");
    assert_eq!(rows[1].1, 1);
    assert!((rows[1].2 - 5.5).abs() < eps, "clothing SUM: {}", rows[1].2);
    assert!((rows[1].3 - 5.5).abs() < eps, "clothing AVG: {}", rows[1].3);
    assert!((rows[1].4 - 5.5).abs() < eps, "clothing MIN: {}", rows[1].4);
    assert!((rows[1].5 - 5.5).abs() < eps, "clothing MAX: {}", rows[1].5);

    // electronics: count=2, sum=5.0, avg=2.5, min=1.5, max=3.5
    assert_eq!(rows[2].0, "electronics");
    assert_eq!(rows[2].1, 2);
    assert!(
        (rows[2].2 - 5.0).abs() < eps,
        "electronics SUM: {}",
        rows[2].2
    );
    assert!(
        (rows[2].3 - 2.5).abs() < eps,
        "electronics AVG: {}",
        rows[2].3
    );
    assert!(
        (rows[2].4 - 1.5).abs() < eps,
        "electronics MIN: {}",
        rows[2].4
    );
    assert!(
        (rows[2].5 - 3.5).abs() < eps,
        "electronics MAX: {}",
        rows[2].5
    );
}

/// Test 2: Terms aggregation ordering
///
/// Results ordered by count DESC: electronics=2, books=2, clothing=1.
/// (Ties between electronics and books resolved alphabetically by tantivy.)
#[tokio::test]
async fn test_terms_agg_ordering() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT category, COUNT(*) as cnt FROM t GROUP BY category ORDER BY cnt DESC")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);

    let schema = batch.schema();
    let cnt_col = batch
        .column(schema.index_of("cnt").unwrap())
        .as_primitive::<Int64Type>();

    // Verify descending order: first two counts >= last count
    assert!(
        cnt_col.value(0) >= cnt_col.value(1),
        "row 0 cnt ({}) should be >= row 1 cnt ({})",
        cnt_col.value(0),
        cnt_col.value(1)
    );
    assert!(
        cnt_col.value(1) >= cnt_col.value(2),
        "row 1 cnt ({}) should be >= row 2 cnt ({})",
        cnt_col.value(1),
        cnt_col.value(2)
    );
    assert_eq!(cnt_col.value(2), 1, "clothing should have count 1");
}

/// Test 3: Numeric (bool) GROUP BY
///
/// Data: active = [true, false, true, false, true]
///   true  -> count=3
///   false -> count=2
#[tokio::test]
async fn test_bool_group_by() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT active, COUNT(*) as cnt FROM t GROUP BY active")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2, "Expected 2 groups for bool");

    let schema = batch.schema();
    let active_col = batch.column(schema.index_of("active").unwrap());
    let cnt_col = batch
        .column(schema.index_of("cnt").unwrap())
        .as_primitive::<Int64Type>();

    // Collect results, converting the active column to string for easy matching
    let mut rows: Vec<(String, i64)> = (0..batch.num_rows())
        .map(|i| (string_val(active_col.as_ref(), i), cnt_col.value(i)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    // "false" sorts before "true" alphabetically
    assert_eq!(rows[0].1, 2, "false group should have count 2");
    assert_eq!(rows[1].1, 3, "true group should have count 3");
}

/// Test 4: GROUP BY with WHERE filter
///
/// Data after WHERE price > 2.0:
///   books: 2.5, 4.5 -> count=2
///   electronics: 3.5 -> count=1
///   clothing: 5.5 -> count=1
#[tokio::test]
async fn test_group_by_with_where() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT category, COUNT(*) as cnt FROM t WHERE price > 2.0 GROUP BY category")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    // 3 categories remain after filtering (electronics loses id=1 with price=1.5)
    assert_eq!(batch.num_rows(), 3, "Expected 3 category groups after filter");

    let schema = batch.schema();
    let cat_col = batch.column(schema.index_of("category").unwrap());
    let cnt_col = batch
        .column(schema.index_of("cnt").unwrap())
        .as_primitive::<Int64Type>();

    let mut rows: Vec<(String, i64)> = (0..batch.num_rows())
        .map(|i| (string_val(cat_col.as_ref(), i), cnt_col.value(i)))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    assert_eq!(rows[0], ("books".to_string(), 2));
    assert_eq!(rows[1], ("clothing".to_string(), 1));
    assert_eq!(rows[2], ("electronics".to_string(), 1));
}

/// Test 5: COUNT(*) without GROUP BY (metric-only)
///
/// 5 documents total.
#[tokio::test]
async fn test_count_without_group_by() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx.sql("SELECT COUNT(*) as cnt FROM t").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let cnt = batch
        .column(batch.schema().index_of("cnt").unwrap())
        .as_primitive::<Int64Type>()
        .value(0);
    assert_eq!(cnt, 5, "Expected total count of 5");
}

/// Test 6: SUM/AVG without GROUP BY (metric-only)
///
/// prices = [1.5, 2.5, 3.5, 4.5, 5.5]
///   SUM = 17.5
///   AVG = 3.5
#[tokio::test]
async fn test_sum_avg_without_group_by() {
    let ctx = setup_ctx(create_test_index());

    let df = ctx
        .sql("SELECT SUM(price) as s, AVG(price) as a FROM t")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);

    let schema = batch.schema();
    let sum_val = batch
        .column(schema.index_of("s").unwrap())
        .as_primitive::<Float64Type>()
        .value(0);
    let avg_val = batch
        .column(schema.index_of("a").unwrap())
        .as_primitive::<Float64Type>()
        .value(0);

    let eps = 1e-10;
    assert!((sum_val - 17.5).abs() < eps, "SUM should be 17.5, got {sum_val}");
    assert!((avg_val - 3.5).abs() < eps, "AVG should be 3.5, got {avg_val}");
}




