use std::net::Ipv4Addr;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{AsArray, RecordBatch};
use arrow::datatypes::{Float64Type, Int64Type};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::*;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use tantivy::schema::{
    Field, IndexRecordOption, SchemaBuilder, TextFieldIndexing, FAST, STORED, STRING, TEXT,
};
use tantivy::{doc, DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::{full_text_udf, AggPushdown, SingleTableProvider};

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

// ---------------------------------------------------------------------------
// Runtime configuration profiling
// ---------------------------------------------------------------------------

/// Build a moderately large index (1M docs) with the same schema used in the
/// bench suite so the terms_few regression pattern is visible.
fn build_large_index(num_docs: usize) -> Index {
    let mut schema_builder = tantivy::schema::Schema::builder();
    let text_fieldtype = tantivy::schema::TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqs),
        )
        .set_stored();
    let text_field = schema_builder.add_text_field("text", text_fieldtype);
    let text_field_many_terms = schema_builder.add_text_field("text_many_terms", STRING | FAST);
    let text_field_few_terms_status =
        schema_builder.add_text_field("text_few_terms_status", STRING | FAST);
    let score_fieldtype = tantivy::schema::NumericOptions::default().set_fast();
    let score_field = schema_builder.add_u64_field("score", score_fieldtype.clone());
    let score_field_f64 = schema_builder.add_f64_field("score_f64", score_fieldtype);

    let index = Index::create_from_tempdir(schema_builder.build()).unwrap();

    let status_labels = ["INFO", "ERROR", "WARN", "DEBUG", "OK", "CRITICAL", "EMERGENCY"];
    let status_weights = [8000u32, 300, 1200, 500, 500, 20, 1];
    let status_dist =
        rand::distr::weighted::WeightedIndex::new(status_weights.iter().copied()).unwrap();

    let lg_norm = rand_distr::LogNormal::new(2.996f64, 0.979f64).unwrap();
    let many_terms: Vec<String> = (0..150_000).map(|n| format!("author{n}")).collect();

    let mut rng = StdRng::from_seed([1u8; 32]);
    let mut writer = index.writer_with_num_threads(1, 200_000_000).unwrap();

    for _ in 0..num_docs {
        let val: f64 = rng.random_range(0.0..1_000_000.0);
        writer
            .add_document(doc!(
                text_field => "cool",
                text_field_many_terms => many_terms.choose(&mut rng).unwrap().to_string(),
                text_field_few_terms_status => status_labels[status_dist.sample(&mut rng)],
                score_field => val as u64,
                score_field_f64 => lg_norm.sample(&mut rng),
            ))
            .unwrap();
    }
    writer.commit().unwrap();
    index
}

/// Build a SessionContext with the AggPushdown rule and a given target_partitions.
fn build_ctx(index: &Index, target_partitions: usize) -> SessionContext {
    let provider = SingleTableProvider::new(index.clone());
    let config = SessionConfig::new().with_target_partitions(target_partitions);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(AggPushdown::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(provider)).unwrap();
    ctx
}

/// Profile a single configuration. Re-creates the physical plan each iteration
/// because DataFusion's `RepartitionExec` has internal state that is consumed on
/// execution and cannot be reused across `collect()` calls.
async fn profile_one(label: &str, ctx: &SessionContext, sql: &str, iterations: usize) {
    // Warm up (also verifies correctness)
    let warm_plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let _ = datafusion::physical_plan::collect(warm_plan, ctx.task_ctx())
        .await
        .unwrap();

    let t = Instant::now();
    for _ in 0..iterations {
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let _ = datafusion::physical_plan::collect(plan, ctx.task_ctx())
            .await
            .unwrap();
    }
    let elapsed = t.elapsed();
    eprintln!(
        "{label}: {:.0}us avg  ({iterations} iters, {:.1}ms total)",
        elapsed.as_micros() as f64 / iterations as f64,
        elapsed.as_secs_f64() * 1000.0,
    );
}

/// Compare the DataFusion pushdown path under different tokio runtime
/// configurations and `target_partitions` settings.
///
/// Run with:
///   cargo test --release --test agg_correctness profile_runtime -- --nocapture
///
/// The test prints timing data to stderr. Look for:
/// - current_thread vs multi_thread overhead
/// - target_partitions=1 vs default (num_cpus)
#[test]
fn profile_runtime_configs() {
    const NUM_DOCS: usize = 1_000_000;
    const ITERATIONS: usize = 50;

    eprintln!("Building index with {NUM_DOCS} docs...");
    let index = build_large_index(NUM_DOCS);
    {
        let r = index.reader().unwrap();
        let s = r.searcher();
        eprintln!(
            "Index: {} segments, {} total docs",
            s.segment_readers().len(),
            s.segment_readers().iter().map(|sr| sr.max_doc()).sum::<u32>(),
        );
    }

    let sql_queries = [
        (
            "terms_few",
            "SELECT text_few_terms_status, COUNT(*) FROM t GROUP BY text_few_terms_status",
        ),
        (
            "terms_few_with_avg",
            "SELECT text_few_terms_status, COUNT(*), AVG(score_f64) FROM t GROUP BY text_few_terms_status",
        ),
        ("avg_f64", "SELECT AVG(score_f64) FROM t"),
    ];

    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Runtime configurations to test
    let runtime_configs: Vec<(&str, tokio::runtime::Runtime)> = vec![
        (
            "current_thread",
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        ),
        (
            "multi_thread(1)",
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
        ),
        (
            "multi_thread(default)",
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap(),
        ),
    ];

    // target_partitions settings to test
    let partitions_label = format!("partitions={num_cpus}");
    let partition_configs: Vec<(&str, usize)> =
        vec![("partitions=1", 1), (&partitions_label, num_cpus)];

    eprintln!("\n=== Runtime x Partitions x Query ===\n");

    for (query_name, sql) in &sql_queries {
        eprintln!("--- {query_name} ---");
        for (rt_name, rt) in &runtime_configs {
            for (part_name, partitions) in &partition_configs {
                let label = format!("{rt_name} / {part_name}");
                rt.block_on(async {
                    let ctx = build_ctx(&index, *partitions);
                    profile_one(&label, &ctx, sql, ITERATIONS).await;
                });
            }
        }
        eprintln!();
    }
}

/// Investigate whether `TermsAggregation { size: Some(429_496_729) }` causes
/// tantivy to pre-allocate a massive internal buffer, explaining the 3x
/// regression at 10M docs.
///
/// Compares three configurations at the raw tantivy level (no DataFusion):
/// - `size: None` (defaults to 10, segment_size defaults to 100)
/// - `size: Some(100)` (reasonable upper bound)
/// - `size: Some(429_496_729), segment_size: Some(429_496_729)` (current value)
///
/// Run with:
///   cargo test --release -p tantivy-datafusion investigate_terms_size_perf -- --ignored --nocapture
#[test]
#[ignore] // manual perf investigation
fn investigate_terms_size_perf() {
    use serde_json::json;
    use tantivy::aggregation::agg_req::Aggregations;
    use tantivy::aggregation::AggregationCollector;
    use tantivy::query::AllQuery;

    const NUM_DOCS: usize = 10_000_000;
    const ITERATIONS: usize = 20;

    eprintln!("Building {NUM_DOCS}-doc index (7 status categories)...");
    let index = build_large_index(NUM_DOCS);

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    eprintln!(
        "Index: {} segments, {} total docs",
        searcher.segment_readers().len(),
        searcher.num_docs(),
    );

    // Three size configurations to compare
    let configs: Vec<(&str, serde_json::Value)> = vec![
        (
            "size=default(10)",
            json!({
                "group": {
                    "terms": { "field": "text_few_terms_status" },
                    "aggs": { "avg_score": { "avg": { "field": "score_f64" } } }
                }
            }),
        ),
        (
            "size=100",
            json!({
                "group": {
                    "terms": { "field": "text_few_terms_status", "size": 100, "shard_size": 100 },
                    "aggs": { "avg_score": { "avg": { "field": "score_f64" } } }
                }
            }),
        ),
        (
            "size=429_496_729",
            json!({
                "group": {
                    "terms": { "field": "text_few_terms_status", "size": 429496729, "shard_size": 429496729 },
                    "aggs": { "avg_score": { "avg": { "field": "score_f64" } } }
                }
            }),
        ),
    ];

    // Warm up
    for (label, agg_json) in &configs {
        let aggs: Aggregations = serde_json::from_value(agg_json.clone()).unwrap();
        let collector = AggregationCollector::from_aggs(aggs, Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
        eprintln!("Warmup done: {label}");
    }

    // Benchmark
    eprintln!("\n=== TermsAggregation size investigation ({NUM_DOCS} docs, {ITERATIONS} iters) ===\n");
    for (label, agg_json) in &configs {
        let mut durations = Vec::with_capacity(ITERATIONS);
        for _ in 0..ITERATIONS {
            let aggs: Aggregations = serde_json::from_value(agg_json.clone()).unwrap();
            let collector = AggregationCollector::from_aggs(aggs, Default::default());
            let start = Instant::now();
            let result = searcher.search(&AllQuery, &collector).unwrap();
            let elapsed = start.elapsed();
            durations.push(elapsed);
            std::hint::black_box(result);
        }

        let total: std::time::Duration = durations.iter().sum();
        let mean = total / ITERATIONS as u32;
        durations.sort();
        let median = durations[ITERATIONS / 2];
        let min_d = durations[0];
        let max_d = durations[ITERATIONS - 1];

        eprintln!(
            "{label:30} | mean={mean:>10?} | median={median:>10?} | min={min_d:>10?} | max={max_d:>10?}",
        );
    }
}

// =========================================================================
// Profiling: 10M terms_few overhead breakdown (size parameter impact)
// =========================================================================

fn build_10m_index() -> Index {
    let num_docs = 10_000_000;
    let mut schema_builder = tantivy::schema::Schema::builder();
    let text_fieldtype = tantivy::schema::TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqs),
        )
        .set_stored();
    let text_field = schema_builder.add_text_field("text", text_fieldtype);
    let text_field_few = schema_builder.add_text_field("text_few_terms_status", STRING | FAST);
    let score_fieldtype = tantivy::schema::NumericOptions::default().set_fast();
    let score_field_f64 = schema_builder.add_f64_field("score_f64", score_fieldtype);
    let index = Index::create_from_tempdir(schema_builder.build()).unwrap();

    let status_labels = [
        "INFO", "ERROR", "WARN", "DEBUG", "OK", "CRITICAL", "EMERGENCY",
    ];
    let status_weights = [8000u32, 300, 1200, 500, 500, 20, 1];
    let status_dist =
        rand::distr::weighted::WeightedIndex::new(status_weights.iter().copied()).unwrap();
    let lg_norm = rand_distr::LogNormal::new(2.996f64, 0.979f64).unwrap();
    let mut rng = StdRng::from_seed([1u8; 32]);
    let mut writer = index.writer_with_num_threads(1, 200_000_000).unwrap();

    for _ in 0..num_docs {
        let mut doc = TantivyDocument::default();
        doc.add_text(text_field, "cool");
        doc.add_text(
            text_field_few,
            status_labels[status_dist.sample(&mut rng)],
        );
        doc.add_f64(score_field_f64, lg_norm.sample(&mut rng));
        writer.add_document(doc).unwrap();
    }
    writer.commit().unwrap();
    index
}

/// Isolate where the ~70ms overhead comes from in the 10M terms_few bench
/// (36ms native vs ~110ms pushdown).
///
/// The key hypothesis: `AggPushdown` sets `size = u32::MAX / 10` (429_496_729)
/// to avoid truncating results, while the native bench uses the default
/// `size = 10`. tantivy's TermsAggregation uses `segment_size = size * 10`
/// internally to allocate per-segment hash maps, and this massive allocation
/// is the source of the overhead.
///
/// Run with:
///   cargo test --release --test agg_correctness profile_10m -- --nocapture --ignored
#[ignore]
#[tokio::test]
async fn profile_10m_terms_overhead() {
    use tantivy::aggregation::agg_req::Aggregations;
    use tantivy::aggregation::AggregationCollector;
    use tantivy::query::AllQuery;

    eprintln!("Building 10M doc index...");
    let t_build = Instant::now();
    let index = build_10m_index();
    eprintln!("Index built in {}s", t_build.elapsed().as_secs());

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    eprintln!("Segments: {}", searcher.segment_readers().len());
    for (i, sr) in searcher.segment_readers().iter().enumerate() {
        eprintln!("  seg {i}: {} docs", sr.max_doc());
    }

    let iters: u128 = 10;

    // -----------------------------------------------------------------
    // 1. Native tantivy with default size (10)
    // -----------------------------------------------------------------
    let agg_small_json =
        serde_json::json!({ "t": { "terms": { "field": "text_few_terms_status" } } });
    let aggs_small: Aggregations = serde_json::from_value(agg_small_json).unwrap();

    // Warm
    let _ = searcher
        .search(
            &AllQuery,
            &AggregationCollector::from_aggs(aggs_small.clone(), Default::default()),
        )
        .unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        let collector = AggregationCollector::from_aggs(aggs_small.clone(), Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
    }
    let native_small_ms = t1.elapsed().as_millis() / iters;
    eprintln!("1. Native tantivy (size=default/10):  {native_small_ms}ms");

    // -----------------------------------------------------------------
    // 2. Native tantivy with size=429_496_729 (what pushdown uses)
    // -----------------------------------------------------------------
    let max_buckets = u32::MAX / 10;
    let agg_big_json = serde_json::json!({
        "t": {
            "terms": {
                "field": "text_few_terms_status",
                "size": max_buckets,
                "segment_size": max_buckets
            }
        }
    });
    let aggs_big: Aggregations = serde_json::from_value(agg_big_json).unwrap();

    // Warm
    let _ = searcher
        .search(
            &AllQuery,
            &AggregationCollector::from_aggs(aggs_big.clone(), Default::default()),
        )
        .unwrap();
    let t2 = Instant::now();
    for _ in 0..iters {
        let collector = AggregationCollector::from_aggs(aggs_big.clone(), Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
    }
    let native_big_ms = t2.elapsed().as_millis() / iters;
    eprintln!("2. Native tantivy (size={max_buckets}): {native_big_ms}ms");

    // -----------------------------------------------------------------
    // 3. Native tantivy with size=100
    // -----------------------------------------------------------------
    let agg_100_json = serde_json::json!({
        "t": {
            "terms": {
                "field": "text_few_terms_status",
                "size": 100
            }
        }
    });
    let aggs_100: Aggregations = serde_json::from_value(agg_100_json).unwrap();

    let _ = searcher
        .search(
            &AllQuery,
            &AggregationCollector::from_aggs(aggs_100.clone(), Default::default()),
        )
        .unwrap();
    let t3 = Instant::now();
    for _ in 0..iters {
        let collector = AggregationCollector::from_aggs(aggs_100.clone(), Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
    }
    let native_100_ms = t3.elapsed().as_millis() / iters;
    eprintln!("3. Native tantivy (size=100):          {native_100_ms}ms");

    // -----------------------------------------------------------------
    // 4. Native tantivy with size=1000
    // -----------------------------------------------------------------
    let agg_1k_json = serde_json::json!({
        "t": {
            "terms": {
                "field": "text_few_terms_status",
                "size": 1000
            }
        }
    });
    let aggs_1k: Aggregations = serde_json::from_value(agg_1k_json).unwrap();

    let _ = searcher
        .search(
            &AllQuery,
            &AggregationCollector::from_aggs(aggs_1k.clone(), Default::default()),
        )
        .unwrap();
    let t4 = Instant::now();
    for _ in 0..iters {
        let collector = AggregationCollector::from_aggs(aggs_1k.clone(), Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
    }
    let native_1k_ms = t4.elapsed().as_millis() / iters;
    eprintln!("4. Native tantivy (size=1000):         {native_1k_ms}ms");

    // -----------------------------------------------------------------
    // 5. Native tantivy with size=10000
    // -----------------------------------------------------------------
    let agg_10k_json = serde_json::json!({
        "t": {
            "terms": {
                "field": "text_few_terms_status",
                "size": 10000
            }
        }
    });
    let aggs_10k: Aggregations = serde_json::from_value(agg_10k_json).unwrap();

    let _ = searcher
        .search(
            &AllQuery,
            &AggregationCollector::from_aggs(aggs_10k.clone(), Default::default()),
        )
        .unwrap();
    let t5 = Instant::now();
    for _ in 0..iters {
        let collector = AggregationCollector::from_aggs(aggs_10k.clone(), Default::default());
        let _ = searcher.search(&AllQuery, &collector).unwrap();
    }
    let native_10k_ms = t5.elapsed().as_millis() / iters;
    eprintln!("5. Native tantivy (size=10000):        {native_10k_ms}ms");

    // -----------------------------------------------------------------
    // 6. Full DataFusion pushdown path (uses size=429M internally)
    // -----------------------------------------------------------------
    let provider = SingleTableProvider::new(index.clone());
    let config = SessionConfig::new().with_target_partitions(1);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(AggPushdown::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let plan = ctx
        .sql("SELECT text_few_terms_status, COUNT(*) FROM t GROUP BY text_few_terms_status")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    // Warm
    let _ = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx())
        .await
        .unwrap();
    let t6 = Instant::now();
    for _ in 0..iters {
        let _ = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx())
            .await
            .unwrap();
    }
    let df_pushdown_ms = t6.elapsed().as_millis() / iters;
    eprintln!("6. Full DF pushdown:                   {df_pushdown_ms}ms");

    // -----------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------
    eprintln!();
    eprintln!("=== OVERHEAD BREAKDOWN ===");
    eprintln!(
        "  size=10 -> size=429M (native only): +{}ms",
        native_big_ms as i128 - native_small_ms as i128
    );
    eprintln!(
        "  size=10 -> size=100:                +{}ms",
        native_100_ms as i128 - native_small_ms as i128
    );
    eprintln!(
        "  size=10 -> size=1000:               +{}ms",
        native_1k_ms as i128 - native_small_ms as i128
    );
    eprintln!(
        "  size=10 -> size=10000:              +{}ms",
        native_10k_ms as i128 - native_small_ms as i128
    );
    eprintln!(
        "  size=429M native -> DF pushdown:    +{}ms",
        df_pushdown_ms as i128 - native_big_ms as i128
    );
    eprintln!(
        "  size=10  native -> DF pushdown:     +{}ms",
        df_pushdown_ms as i128 - native_small_ms as i128
    );
}
