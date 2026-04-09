use std::net::Ipv4Addr;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float32Type, Float64Type, Int64Type, UInt64Type};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use tantivy::schema::{Field, SchemaBuilder, Term, FAST, STORED, TEXT};
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

/// Create a session context with optimizer rules and SingleTableProvider.
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

fn collect_batches(batches: &[RecordBatch]) -> RecordBatch {
    arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// ---------------------------------------------------------------------------
// Index builders
// ---------------------------------------------------------------------------

/// Standard 5-doc index, then delete docs with id=2 and id=4 (leaves 1, 3, 5).
fn create_index_with_deletes() -> Index {
    let (schema, u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f) =
        build_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    add_test_documents(
        &writer,
        (u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f),
    );
    writer.commit().unwrap();

    writer.delete_term(Term::from_field_u64(u64_f, 2u64));
    writer.delete_term(Term::from_field_u64(u64_f, 4u64));
    writer.commit().unwrap();

    index
}

/// Empty index with schema but no documents.
fn create_empty_index() -> Index {
    let (schema, ..) = build_test_schema();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
    writer.commit().unwrap();
    index
}

/// Two-segment index: first 3 docs in segment 1, remaining 2 in segment 2.
fn create_multi_segment_index() -> Index {
    let (schema, u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f) =
        build_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

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

    // Segment 1: docs 0..3
    for i in 0..3 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(u64_f, ids[i]);
        doc.add_i64(i64_f, scores[i]);
        doc.add_f64(f64_f, prices[i]);
        doc.add_bool(bool_f, actives[i]);
        doc.add_text(text_f, categories[i]);
        doc.add_date(date_f, DateTime::from_timestamp_micros(timestamps[i]));
        doc.add_ip_addr(ip_f, ips[i].to_ipv6_mapped());
        doc.add_bytes(bytes_f, data_payloads[i]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();

    // Segment 2: docs 3..5
    for i in 3..5 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(u64_f, ids[i]);
        doc.add_i64(i64_f, scores[i]);
        doc.add_f64(f64_f, prices[i]);
        doc.add_bool(bool_f, actives[i]);
        doc.add_text(text_f, categories[i]);
        doc.add_date(date_f, DateTime::from_timestamp_micros(timestamps[i]));
        doc.add_ip_addr(ip_f, ips[i].to_ipv6_mapped());
        doc.add_bytes(bytes_f, data_payloads[i]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();

    index
}

/// Two-segment index with doc id=2 deleted from segment 1.
fn create_multi_segment_index_with_deletes() -> Index {
    let (schema, u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f) =
        build_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

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

    // Segment 1: docs 0..3 (ids 1, 2, 3)
    for i in 0..3 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(u64_f, ids[i]);
        doc.add_i64(i64_f, scores[i]);
        doc.add_f64(f64_f, prices[i]);
        doc.add_bool(bool_f, actives[i]);
        doc.add_text(text_f, categories[i]);
        doc.add_date(date_f, DateTime::from_timestamp_micros(timestamps[i]));
        doc.add_ip_addr(ip_f, ips[i].to_ipv6_mapped());
        doc.add_bytes(bytes_f, data_payloads[i]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();

    // Segment 2: docs 3..5 (ids 4, 5)
    for i in 3..5 {
        let mut doc = TantivyDocument::default();
        doc.add_u64(u64_f, ids[i]);
        doc.add_i64(i64_f, scores[i]);
        doc.add_f64(f64_f, prices[i]);
        doc.add_bool(bool_f, actives[i]);
        doc.add_text(text_f, categories[i]);
        doc.add_date(date_f, DateTime::from_timestamp_micros(timestamps[i]));
        doc.add_ip_addr(ip_f, ips[i].to_ipv6_mapped());
        doc.add_bytes(bytes_f, data_payloads[i]);
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();

    // Delete id=2 from segment 1
    writer.delete_term(Term::from_field_u64(u64_f, 2u64));
    writer.commit().unwrap();

    index
}

/// Index with exactly 1 document.
fn create_single_doc_index() -> Index {
    let (schema, u64_f, i64_f, f64_f, bool_f, text_f, date_f, ip_f, bytes_f) =
        build_test_schema();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let mut doc = TantivyDocument::default();
    doc.add_u64(u64_f, 42u64);
    doc.add_i64(i64_f, 100i64);
    doc.add_f64(f64_f, 9.99f64);
    doc.add_bool(bool_f, true);
    doc.add_text(text_f, "electronics");
    doc.add_date(date_f, DateTime::from_timestamp_micros(1_000_000));
    doc.add_ip_addr(ip_f, Ipv4Addr::new(127, 0, 0, 1).to_ipv6_mapped());
    doc.add_bytes(bytes_f, b"solo");
    writer.add_document(doc).unwrap();
    writer.commit().unwrap();

    index
}

// =========================================================================
// 1. Deleted documents
// =========================================================================

#[tokio::test]
async fn test_deleted_docs_select_all() {
    let ctx = setup_ctx(create_index_with_deletes());

    let df = ctx.sql("SELECT id FROM t ORDER BY id").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3, "Expected 3 alive docs after deleting 2");
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 5);
}

#[tokio::test]
async fn test_deleted_docs_count_group_by() {
    let ctx = setup_ctx(create_index_with_deletes());

    // Alive docs: id=1 electronics, id=3 electronics, id=5 clothing
    // Deleted: id=2 books, id=4 books
    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             GROUP BY category ORDER BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2, "Expected 2 categories (books deleted)");

    let cat_col =
        arrow::compute::cast(batch.column(0), &arrow::datatypes::DataType::Utf8).unwrap();
    let categories = cat_col.as_string::<i32>();
    let counts = batch.column(1).as_primitive::<Int64Type>();

    assert_eq!(categories.value(0), "clothing");
    assert_eq!(counts.value(0), 1);
    assert_eq!(categories.value(1), "electronics");
    assert_eq!(counts.value(1), 2);
}

#[tokio::test]
async fn test_deleted_docs_full_text_excludes_deleted() {
    let ctx = setup_ctx(create_index_with_deletes());

    // "books" docs (id=2, id=4) are deleted, so full_text for "books" should return 0
    let df = ctx
        .sql("SELECT id FROM t WHERE full_text(category, 'books')")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(total_rows(&batches), 0, "Deleted 'books' docs should not appear");
}

#[tokio::test]
async fn test_deleted_docs_topk_excludes_deleted() {
    let ctx = setup_ctx(create_index_with_deletes());

    // TopK on electronics; alive docs with electronics: id=1 and id=3
    let df = ctx
        .sql(
            "SELECT id, _score FROM t \
             WHERE full_text(category, 'electronics') \
             ORDER BY _score DESC LIMIT 10",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let mut ids: Vec<u64> = batch
        .column(0)
        .as_primitive::<UInt64Type>()
        .iter()
        .map(|v| v.unwrap())
        .collect();
    ids.sort();
    assert_eq!(ids, vec![1, 3]);
}

#[tokio::test]
async fn test_deleted_docs_document_excludes_deleted() {
    let ctx = setup_ctx(create_index_with_deletes());

    let df = ctx
        .sql("SELECT id, _document FROM t ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 5);

    // Each _document should be valid JSON matching the surviving doc
    let docs = batch.column(1).as_string::<i32>();
    for i in 0..3 {
        let doc: serde_json::Value = serde_json::from_str(docs.value(i)).unwrap();
        assert_eq!(doc["id"][0], ids.value(i));
    }
}

// =========================================================================
// 2. Empty index
// =========================================================================

#[tokio::test]
async fn test_empty_index_select_all() {
    let ctx = setup_ctx(create_empty_index());

    let df = ctx.sql("SELECT * FROM t").await.unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(total_rows(&batches), 0, "Empty index should return 0 rows");
}

#[tokio::test]
async fn test_empty_index_count() {
    let ctx = setup_ctx(create_empty_index());

    let df = ctx.sql("SELECT COUNT(*) as cnt FROM t").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    let count = batch.column(0).as_primitive::<Int64Type>().value(0);
    assert_eq!(count, 0, "COUNT(*) on empty index should be 0");
}

#[tokio::test]
async fn test_empty_index_explain_no_crash() {
    let ctx = setup_ctx(create_empty_index());

    let df = ctx
        .sql("EXPLAIN SELECT id, price FROM t WHERE price > 1.0")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    // Just verify it succeeds and returns a plan
    assert!(!batches.is_empty(), "EXPLAIN should produce output");
    assert!(total_rows(&batches) > 0, "EXPLAIN should have plan rows");
}

// =========================================================================
// 3. Multi-segment index
// =========================================================================

#[tokio::test]
async fn test_multi_segment_select_all() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx.sql("SELECT id FROM t ORDER BY id").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 5, "All 5 docs across 2 segments");
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    for i in 0..5 {
        assert_eq!(ids.value(i), (i + 1) as u64);
    }
}

#[tokio::test]
async fn test_multi_segment_aggregation() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             GROUP BY category ORDER BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);

    let cat_col =
        arrow::compute::cast(batch.column(0), &arrow::datatypes::DataType::Utf8).unwrap();
    let categories = cat_col.as_string::<i32>();
    let counts = batch.column(1).as_primitive::<Int64Type>();

    assert_eq!(categories.value(0), "books");
    assert_eq!(counts.value(0), 2);
    assert_eq!(categories.value(1), "clothing");
    assert_eq!(counts.value(1), 1);
    assert_eq!(categories.value(2), "electronics");
    assert_eq!(counts.value(2), 2);
}

#[tokio::test]
async fn test_multi_segment_full_text() {
    let ctx = setup_ctx(create_multi_segment_index());

    // "books" appears in segment 1 (id=2) and segment 2 (id=4)
    let df = ctx
        .sql("SELECT id FROM t WHERE full_text(category, 'books') ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 4);
}

#[tokio::test]
async fn test_multi_segment_score_comparable() {
    let ctx = setup_ctx(create_multi_segment_index());

    // "electronics" spans both segments: id=1 (seg1) and id=3 (seg1).
    // BM25 scores should be comparable across segments (global IDF).
    let df = ctx
        .sql(
            "SELECT id, _score FROM t \
             WHERE full_text(category, 'electronics') \
             ORDER BY id",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let scores = batch.column(1).as_primitive::<Float32Type>();
    let s0 = scores.value(0);
    let s1 = scores.value(1);
    assert!(s0 > 0.0 && s1 > 0.0, "Scores should be positive");
    // Both docs contain exactly one occurrence of "electronics" in the same
    // field, so with global IDF the scores should be identical.
    assert!(
        (s0 - s1).abs() < 1e-5,
        "Scores should be equal for identical term frequency: {s0} vs {s1}"
    );
}

#[tokio::test]
async fn test_multi_segment_document() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx
        .sql(
            "SELECT id, _document FROM t \
             WHERE full_text(category, 'books') \
             ORDER BY id",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    let docs = batch.column(1).as_string::<i32>();

    assert_eq!(ids.value(0), 2);
    let doc0: serde_json::Value = serde_json::from_str(docs.value(0)).unwrap();
    assert_eq!(doc0["id"][0], 2);

    assert_eq!(ids.value(1), 4);
    let doc1: serde_json::Value = serde_json::from_str(docs.value(1)).unwrap();
    assert_eq!(doc1["id"][0], 4);
}

// =========================================================================
// 4. Multi-segment with deleted docs
// =========================================================================

#[tokio::test]
async fn test_multi_segment_with_deletes_select() {
    // Segment 1: ids 1, 2, 3 (id=2 deleted) -> alive: 1, 3
    // Segment 2: ids 4, 5 -> alive: 4, 5
    // Total alive: 1, 3, 4, 5
    let ctx = setup_ctx(create_multi_segment_index_with_deletes());

    let df = ctx.sql("SELECT id FROM t ORDER BY id").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 4);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 3);
    assert_eq!(ids.value(2), 4);
    assert_eq!(ids.value(3), 5);
}

#[tokio::test]
async fn test_multi_segment_with_deletes_aggregation() {
    let ctx = setup_ctx(create_multi_segment_index_with_deletes());

    // Alive: id=1 electronics, id=3 electronics, id=4 books, id=5 clothing
    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             GROUP BY category ORDER BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);
    let cat_col =
        arrow::compute::cast(batch.column(0), &arrow::datatypes::DataType::Utf8).unwrap();
    let categories = cat_col.as_string::<i32>();
    let counts = batch.column(1).as_primitive::<Int64Type>();

    assert_eq!(categories.value(0), "books");
    assert_eq!(counts.value(0), 1);
    assert_eq!(categories.value(1), "clothing");
    assert_eq!(counts.value(1), 1);
    assert_eq!(categories.value(2), "electronics");
    assert_eq!(counts.value(2), 2);
}

#[tokio::test]
async fn test_multi_segment_with_deletes_full_text() {
    let ctx = setup_ctx(create_multi_segment_index_with_deletes());

    // "books" docs: id=2 (deleted), id=4 (alive) -> only id=4
    let df = ctx
        .sql("SELECT id FROM t WHERE full_text(category, 'books') ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 4);
}

#[tokio::test]
async fn test_multi_segment_with_deletes_document() {
    let ctx = setup_ctx(create_multi_segment_index_with_deletes());

    let df = ctx
        .sql("SELECT id, _document FROM t ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 4);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    let docs = batch.column(1).as_string::<i32>();

    let expected_ids = [1u64, 3, 4, 5];
    for (i, &expected) in expected_ids.iter().enumerate() {
        assert_eq!(ids.value(i), expected);
        let doc: serde_json::Value = serde_json::from_str(docs.value(i)).unwrap();
        assert_eq!(doc["id"][0], expected);
    }
}

// =========================================================================
// 5. Zero-match query
// =========================================================================

#[tokio::test]
async fn test_zero_match_full_text() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx
        .sql("SELECT id FROM t WHERE full_text(category, 'nonexistent_term_xyz')")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(total_rows(&batches), 0, "Nonexistent term should match 0 rows");
}

#[tokio::test]
async fn test_zero_match_aggregation() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) as cnt FROM t \
             WHERE full_text(category, 'nonexistent_term_xyz') \
             GROUP BY category",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    assert_eq!(
        total_rows(&batches),
        0,
        "Aggregation on zero-match query should return empty"
    );
}

// =========================================================================
// 6. Single doc index
// =========================================================================

#[tokio::test]
async fn test_single_doc_select() {
    let ctx = setup_ctx(create_single_doc_index());

    let df = ctx.sql("SELECT id, price FROM t").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 42);
    let price = batch.column(1).as_primitive::<Float64Type>().value(0);
    assert!((price - 9.99).abs() < 1e-10);
}

#[tokio::test]
async fn test_single_doc_aggregation() {
    let ctx = setup_ctx(create_single_doc_index());

    let df = ctx
        .sql("SELECT COUNT(*) as cnt, SUM(price) as total FROM t")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let count = batch.column(0).as_primitive::<Int64Type>().value(0);
    assert_eq!(count, 1);
    let total = batch.column(1).as_primitive::<Float64Type>().value(0);
    assert!((total - 9.99).abs() < 1e-10);
}

#[tokio::test]
async fn test_single_doc_document() {
    let ctx = setup_ctx(create_single_doc_index());

    let df = ctx.sql("SELECT id, _document FROM t").await.unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 42);

    let doc: serde_json::Value =
        serde_json::from_str(batch.column(1).as_string::<i32>().value(0)).unwrap();
    assert_eq!(doc["id"][0], 42);
}

#[tokio::test]
async fn test_single_doc_full_text() {
    let ctx = setup_ctx(create_single_doc_index());

    let df = ctx
        .sql(
            "SELECT id, _score FROM t \
             WHERE full_text(category, 'electronics')",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 1);
    let id = batch.column(0).as_primitive::<UInt64Type>().value(0);
    assert_eq!(id, 42);
    let score = batch.column(1).as_primitive::<Float32Type>().value(0);
    assert!(score > 0.0, "_score should be positive");
}

// =========================================================================
// 7. All types query -- fast field filters on every supported type
// =========================================================================

#[tokio::test]
async fn test_filter_u64() {
    let ctx = setup_ctx(create_multi_segment_index());

    let df = ctx
        .sql("SELECT id FROM t WHERE id >= 3 ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
    assert_eq!(ids.value(1), 4);
    assert_eq!(ids.value(2), 5);
}

#[tokio::test]
async fn test_filter_i64() {
    let ctx = setup_ctx(create_multi_segment_index());

    // scores: 10, 20, 30, 40, 50 -> score > 25 means ids {3,4,5}
    let df = ctx
        .sql("SELECT id FROM t WHERE score > 25 ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 3);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 3);
    assert_eq!(ids.value(1), 4);
    assert_eq!(ids.value(2), 5);
}

#[tokio::test]
async fn test_filter_f64() {
    let ctx = setup_ctx(create_multi_segment_index());

    // prices: 1.5, 2.5, 3.5, 4.5, 5.5 -> price < 3.0 means ids {1, 2}
    let df = ctx
        .sql("SELECT id FROM t WHERE price < 3.0 ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
}

#[tokio::test]
async fn test_filter_bool() {
    let ctx = setup_ctx(create_multi_segment_index());

    // actives: true, false, true, false, true -> active = false means ids {2, 4}
    let df = ctx
        .sql("SELECT id FROM t WHERE active = false ORDER BY id")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 4);
}

#[tokio::test]
async fn test_filter_date() {
    let ctx = setup_ctx(create_multi_segment_index());

    // Timestamps: 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000 microseconds
    // 3_000_000 us = 3 seconds from epoch = '1970-01-01T00:00:03Z'
    // created_at > that threshold means ids with timestamps 4_000_000 and 5_000_000 -> ids {4, 5}
    let df = ctx
        .sql(
            "SELECT id FROM t \
             WHERE created_at > TIMESTAMP '1970-01-01T00:00:03' \
             ORDER BY id",
        )
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();
    let batch = collect_batches(&batches);

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 4);
    assert_eq!(ids.value(1), 5);
}
