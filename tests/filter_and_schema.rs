use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, UInt64Type};
use datafusion::prelude::*;
use tantivy::schema::{SchemaBuilder, FAST, STORED, TEXT};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::fast_field_reader::read_segment_fast_fields_to_batch;
use tantivy_datafusion::{full_text_udf, SingleTableProvider};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a test schema with typed fast fields for filter conversion tests.
fn build_filter_test_schema() -> (
    tantivy::schema::Schema,
    tantivy::schema::Field, // id (u64)
    tantivy::schema::Field, // score (i64)
    tantivy::schema::Field, // price (f64)
    tantivy::schema::Field, // active (bool)
    tantivy::schema::Field, // category (text)
) {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let score = builder.add_i64_field("score", FAST);
    let price = builder.add_f64_field("price", FAST);
    let active = builder.add_bool_field("active", FAST);
    let category = builder.add_text_field("category", TEXT | FAST | STORED);
    let schema = builder.build();
    (schema, id, score, price, active, category)
}

/// Create a RAM index with 5 documents for filter tests.
///
/// | id | score | price | active | category    |
/// |----|-------|-------|--------|-------------|
/// |  1 |    10 |  1.5  | true   | electronics |
/// |  2 |    20 |  2.5  | false  | books       |
/// |  3 |    30 |  3.5  | true   | electronics |
/// |  4 |    40 |  4.5  | false  | books       |
/// |  5 |    50 |  5.5  | true   | clothing    |
fn create_filter_test_index() -> Index {
    let (schema, id_f, score_f, price_f, active_f, cat_f) = build_filter_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let ids = [1u64, 2, 3, 4, 5];
    let scores = [10i64, 20, 30, 40, 50];
    let prices = [1.5f64, 2.5, 3.5, 4.5, 5.5];
    let actives = [true, false, true, false, true];
    let categories = ["electronics", "books", "electronics", "books", "clothing"];

    for i in 0..5 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id_f, ids[i]);
        doc.add_i64(score_f, scores[i]);
        doc.add_f64(price_f, prices[i]);
        doc.add_bool(active_f, actives[i]);
        doc.add_text(cat_f, categories[i]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
    index
}

fn collect_batches(batches: &[RecordBatch]) -> RecordBatch {
    arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
}

/// Execute a SQL query and return the concatenated result batch.
async fn run_sql(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    if batches.is_empty() {
        return RecordBatch::new_empty(Arc::new(Schema::empty()));
    }
    collect_batches(&batches)
}

/// Set up a session with a filter-test index registered as table "t".
fn setup_filter_session() -> SessionContext {
    let index = create_filter_test_index();
    let provider = SingleTableProvider::new(index);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();
    ctx
}

// ===========================================================================
// Filter conversion tests — equality on each type
// ===========================================================================

#[tokio::test]
async fn test_eq_u64() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE id = 3").await;

    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
}

#[tokio::test]
async fn test_eq_i64() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE score = 30").await;

    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
}

#[tokio::test]
async fn test_eq_f64() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE price = 3.5").await;

    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
}

#[tokio::test]
async fn test_eq_bool() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE active = true ORDER BY id").await;

    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 5);
}

// ===========================================================================
// NotEq
// ===========================================================================

#[tokio::test]
async fn test_not_eq() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE id != 3 ORDER BY id").await;

    assert_eq!(batch.num_rows(), 4);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 4);
    assert_eq!(ids.value(3), 5);
}

// ===========================================================================
// Range combinations
// ===========================================================================

#[tokio::test]
async fn test_range_combination() {
    let ctx = setup_filter_session();
    let batch = run_sql(
        &ctx,
        "SELECT id FROM t WHERE price >= 2.5 AND price <= 4.5 ORDER BY id",
    )
    .await;

    // price in [2.5, 4.5] -> ids 2, 3, 4
    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 4);
}

// ===========================================================================
// Operator flipping (literal on left)
// ===========================================================================

#[tokio::test]
async fn test_operator_flipping() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE 3.0 < price ORDER BY id").await;

    // 3.0 < price  =>  price > 3.0  =>  ids 3 (3.5), 4 (4.5), 5 (5.5)
    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
    assert_eq!(ids.value(1), 4);
    assert_eq!(ids.value(2), 5);
}

// ===========================================================================
// Combined full-text search + fast field filter
// ===========================================================================

#[tokio::test]
async fn test_combined_fts_and_fast_field() {
    let ctx = setup_filter_session();
    let batch = run_sql(
        &ctx,
        "SELECT id FROM t WHERE full_text(category, 'electronics') AND price > 2.0 ORDER BY id",
    )
    .await;

    // electronics -> ids {1, 3}, price > 2.0 -> ids {2, 3, 4, 5}
    // intersection -> id 3
    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
}

// ===========================================================================
// Filter that matches nothing
// ===========================================================================

#[tokio::test]
async fn test_filter_matches_nothing() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE price > 100.0").await;

    assert_eq!(batch.num_rows(), 0);
}

// ===========================================================================
// Filter that matches everything
// ===========================================================================

#[tokio::test]
async fn test_filter_matches_everything() {
    let ctx = setup_filter_session();
    let batch = run_sql(&ctx, "SELECT id FROM t WHERE price > 0.0 ORDER BY id").await;

    assert_eq!(batch.num_rows(), 5);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    for (i, expected) in [1u64, 2, 3, 4, 5].iter().enumerate() {
        assert_eq!(ids.value(i), *expected);
    }
}

// ===========================================================================
// Schema evolution tests — null padding for missing fast fields
// ===========================================================================

/// Test that `read_segment_fast_fields_to_batch` returns null arrays for
/// columns in the projected schema that do not exist as fast fields in the
/// segment. This simulates the schema evolution case where a new column was
/// added to the Arrow schema but the tantivy segment was created before that
/// column existed.
#[test]
fn test_null_padding_for_missing_fast_field() {
    let mut builder = SchemaBuilder::new();
    let id_field = builder.add_u64_field("id", FAST);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let mut doc = TantivyDocument::default();
    doc.add_u64(id_field, 42);
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let segment_reader = &searcher.segment_readers()[0];

    // Project a schema that includes _doc_id, _segment_ord, the real "id"
    // field, AND a nonexistent "nonexistent_price" field.
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("_doc_id", DataType::UInt32, false),
        Field::new("_segment_ord", DataType::UInt32, false),
        Field::new("id", DataType::UInt64, true),
        Field::new("nonexistent_price", DataType::Float64, true),
    ]));

    let batch = read_segment_fast_fields_to_batch(
        segment_reader,
        &projected_schema,
        None,
        None,
        None,
        0,
        None,
    )
    .unwrap();

    assert_eq!(batch.num_rows(), 1);

    // _doc_id
    let doc_ids = batch
        .column(0)
        .as_primitive::<arrow::datatypes::UInt32Type>();
    assert_eq!(doc_ids.value(0), 0);

    // _segment_ord
    let seg_ords = batch
        .column(1)
        .as_primitive::<arrow::datatypes::UInt32Type>();
    assert_eq!(seg_ords.value(0), 0);

    // Real "id" field
    let ids = batch.column(2).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 42);

    // Nonexistent field -> null array
    assert!(batch.column(3).is_null(0));
}

/// Test null padding for multiple missing field types: i64, bool, u64.
#[test]
fn test_null_padding_multiple_missing_types() {
    let mut builder = SchemaBuilder::new();
    let id_field = builder.add_u64_field("id", FAST);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for v in [10u64, 20, 30] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id_field, v);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let segment_reader = &searcher.segment_readers()[0];

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("_doc_id", DataType::UInt32, false),
        Field::new("_segment_ord", DataType::UInt32, false),
        Field::new("id", DataType::UInt64, true),
        Field::new("missing_score", DataType::Int64, true),
        Field::new("missing_flag", DataType::Boolean, true),
        Field::new("missing_count", DataType::UInt64, true),
    ]));

    let batch = read_segment_fast_fields_to_batch(
        segment_reader,
        &projected_schema,
        None,
        None,
        None,
        0,
        None,
    )
    .unwrap();

    assert_eq!(batch.num_rows(), 3);

    // Real "id" field has values
    let ids = batch.column(2).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 10);
    assert_eq!(ids.value(1), 20);
    assert_eq!(ids.value(2), 30);

    // All missing columns are entirely null
    for col_idx in 3..6 {
        let col = batch.column(col_idx);
        assert_eq!(col.null_count(), 3, "column {col_idx} should be all nulls");
    }
}

/// SQL-level test: UNION two SingleTableProviders with different schemas.
/// Each provider returns nulls for columns it doesn't have.
#[tokio::test]
async fn test_schema_evolution_via_union() {
    // Index 1: has id + price
    let mut b1 = SchemaBuilder::new();
    let id1 = b1.add_u64_field("id", FAST | STORED);
    let price1 = b1.add_f64_field("price", FAST);
    let schema1 = b1.build();

    let index1 = Index::create_in_ram(schema1);
    let mut w1: IndexWriter = index1.writer_with_num_threads(1, 15_000_000).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_u64(id1, 1);
    doc.add_f64(price1, 10.0);
    w1.add_document(doc).unwrap();
    w1.commit().unwrap();

    // Index 2: has id + score (no price)
    let mut b2 = SchemaBuilder::new();
    let id2 = b2.add_u64_field("id", FAST | STORED);
    let score2 = b2.add_i64_field("score", FAST);
    let schema2 = b2.build();

    let index2 = Index::create_in_ram(schema2);
    let mut w2: IndexWriter = index2.writer_with_num_threads(1, 15_000_000).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_u64(id2, 2);
    doc.add_i64(score2, 99);
    w2.add_document(doc).unwrap();
    w2.commit().unwrap();

    let provider1 = SingleTableProvider::new(index1);
    let provider2 = SingleTableProvider::new(index2);

    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t1", Arc::new(provider1)).unwrap();
    ctx.register_table("t2", Arc::new(provider2)).unwrap();

    // Query: select id + price from t1, and id + NULL price from t2 via UNION ALL
    let sql = "\
        SELECT id, price FROM t1 \
        UNION ALL \
        SELECT id, CAST(NULL AS DOUBLE) AS price FROM t2 \
        ORDER BY id";
    let batch = run_sql(&ctx, sql).await;

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);

    let prices = batch.column(1).as_primitive::<Float64Type>();
    // id=1 has price 10.0
    assert!((prices.value(0) - 10.0).abs() < 1e-10);
    // id=2 has no price -> null
    assert!(prices.is_null(1));
}
