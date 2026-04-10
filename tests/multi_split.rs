use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray, ListArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, UInt64Type};
use datafusion::datasource::TableProvider;
use datafusion::prelude::*;
use tantivy::schema::{SchemaBuilder, FAST, STORED, STRING, TEXT};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{full_text_udf, DirectIndexOpener, SingleTableProvider};

fn collect_batches(batches: &[RecordBatch]) -> RecordBatch {
    arrow::compute::concat_batches(&batches[0].schema(), batches).unwrap()
}

async fn run_sql(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    collect_batches(&batches)
}

fn create_int_score_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let score = builder.add_i64_field("score", FAST);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (id_value, score_value) in [(1u64, 10i64), (2, 25)] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, id_value);
        doc.add_i64(score, score_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn create_float_score_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let score = builder.add_f64_field("score", FAST);
    let active = builder.add_bool_field("active", FAST);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (id_value, score_value, active_value) in [(3u64, 30.5f64, true), (4, 42.0, false)] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, id_value);
        doc.add_f64(score, score_value);
        doc.add_bool(active, active_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn create_missing_score_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let active = builder.add_bool_field("active", FAST);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (id_value, active_value) in [(10u64, true), (11, false)] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, id_value);
        doc.add_bool(active, active_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn create_category_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let category = builder.add_text_field("category", TEXT | FAST | STORED);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for (id_value, category_value) in [(20u64, "books"), (21, "electronics")] {
        let mut doc = TantivyDocument::default();
        doc.add_u64(id, id_value);
        doc.add_text(category, category_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn create_scalar_tags_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let tags = builder.add_text_field("tags", STRING | FAST | STORED);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let mut first = TantivyDocument::default();
    first.add_u64(id, 1);
    first.add_text(tags, "alpha");
    writer.add_document(first).unwrap();

    writer.commit().unwrap();
    index
}

fn create_list_tags_split() -> Index {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let tags = builder.add_text_field("tags", STRING | FAST | STORED);
    let schema = builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let mut second = TantivyDocument::default();
    second.add_u64(id, 2);
    second.add_text(tags, "beta");
    second.add_text(tags, "shared");
    writer.add_document(second).unwrap();

    writer.commit().unwrap();
    index
}

#[tokio::test]
async fn test_multi_split_explicit_schema_casts_and_null_pads() {
    let split_a = Arc::new(DirectIndexOpener::new(create_int_score_split()));
    let split_b = Arc::new(DirectIndexOpener::new(create_float_score_split()));

    let canonical_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("active", DataType::Boolean, true),
    ]));

    let provider = SingleTableProvider::from_splits_with_fast_field_schema(
        vec![split_a, split_b],
        canonical_schema,
    )
    .unwrap();

    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batch = run_sql(&ctx, "SELECT id, score, active FROM t ORDER BY id").await;

    assert_eq!(batch.num_rows(), 4);

    let ids = batch.column(0).as_primitive::<UInt64Type>();
    let scores = batch.column(1).as_primitive::<Float64Type>();
    let active = batch
        .column(2)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    assert_eq!(ids.values(), &[1, 2, 3, 4]);
    assert_eq!(scores.value(0), 10.0);
    assert_eq!(scores.value(1), 25.0);
    assert!((scores.value(2) - 30.5).abs() < 1e-10);
    assert!((scores.value(3) - 42.0).abs() < 1e-10);

    assert!(active.is_null(0));
    assert!(active.is_null(1));
    assert!(active.value(2));
    assert!(!active.value(3));
}

#[tokio::test]
async fn test_multi_split_missing_field_filter_returns_only_matching_split_rows() {
    let split_a = Arc::new(DirectIndexOpener::new(create_int_score_split()));
    let split_b = Arc::new(DirectIndexOpener::new(create_missing_score_split()));

    let canonical_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, true),
        Field::new("score", DataType::Float64, true),
        Field::new("active", DataType::Boolean, true),
    ]));

    let provider = SingleTableProvider::from_splits_with_fast_field_schema(
        vec![split_a, split_b],
        canonical_schema,
    )
    .unwrap();

    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batch = run_sql(&ctx, "SELECT id FROM t WHERE score > 20 ORDER BY id").await;

    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 2);
}

#[tokio::test]
async fn test_multi_split_scalar_to_list_promotion() {
    let split_a = Arc::new(DirectIndexOpener::new(create_scalar_tags_split()));
    let split_b = Arc::new(DirectIndexOpener::new(create_list_tags_split()));

    let provider = SingleTableProvider::from_splits(vec![split_a, split_b]).unwrap();

    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batch = run_sql(&ctx, "SELECT id, tags FROM t ORDER BY id").await;

    assert_eq!(batch.num_rows(), 2);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.values(), &[1, 2]);

    let tags = batch
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let first = tags.value(0);
    let first_tags = first.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(first_tags.value(0), "alpha");
    assert_eq!(first_tags.len(), 1);

    let second = tags.value(1);
    let second_tags = second.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(second_tags.value(0), "beta");
    assert_eq!(second_tags.value(1), "shared");
    assert_eq!(second_tags.len(), 2);
}

#[tokio::test]
async fn test_multi_split_full_text_ignores_splits_missing_field() {
    let split_a = Arc::new(DirectIndexOpener::new(create_category_split()));
    let split_b = Arc::new(DirectIndexOpener::new(create_missing_score_split()));

    let provider = SingleTableProvider::from_splits(vec![split_a, split_b]).unwrap();

    let config = SessionConfig::new().with_target_partitions(4);
    let ctx = SessionContext::new_with_config(config);
    ctx.register_udf(full_text_udf());
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batch = run_sql(
        &ctx,
        "SELECT id FROM t WHERE full_text(category, 'books') ORDER BY id",
    )
    .await;

    assert_eq!(batch.num_rows(), 1);
    let ids = batch.column(0).as_primitive::<UInt64Type>();
    assert_eq!(ids.value(0), 20);
}

#[tokio::test]
async fn test_multi_split_partition_count_flattens_split_segments() {
    let mut builder = SchemaBuilder::new();
    let id = builder.add_u64_field("id", FAST | STORED);
    let schema = builder.build();

    let split_a = Index::create_in_ram(schema.clone());
    let mut writer_a: IndexWriter = split_a.writer_with_num_threads(1, 15_000_000).unwrap();
    let mut first = TantivyDocument::default();
    first.add_u64(id, 1);
    writer_a.add_document(first).unwrap();
    writer_a.commit().unwrap();
    let mut second = TantivyDocument::default();
    second.add_u64(id, 2);
    writer_a.add_document(second).unwrap();
    writer_a.commit().unwrap();

    let split_b = Index::create_in_ram(schema);
    let mut writer_b: IndexWriter = split_b.writer_with_num_threads(1, 15_000_000).unwrap();
    let mut third = TantivyDocument::default();
    third.add_u64(id, 3);
    writer_b.add_document(third).unwrap();
    writer_b.commit().unwrap();

    let provider = SingleTableProvider::from_splits(vec![
        Arc::new(DirectIndexOpener::new(split_a)),
        Arc::new(DirectIndexOpener::new(split_b)),
    ])
    .unwrap();

    let ctx = SessionContext::new();
    let state = ctx.state();
    let exec = provider.scan(&state, None, &[], None).await.unwrap();

    assert_eq!(exec.properties().partitioning.partition_count(), 3);
}
