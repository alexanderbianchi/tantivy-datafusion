//! Benchmark: tantivy native aggregation vs unified SingleTableProvider.
//!
//! Compares three execution paths:
//! - **native**: tantivy AggregationCollector directly (baseline)
//! - **unified_pushdown**: SingleTableProvider + AggPushdown optimizer rule
//!   (should match native — AggDataSource calls the same collector)
//! - **unified_sql**: SingleTableProvider without AggPushdown (DataFusion
//!   hash GROUP BY on Arrow batches — measures the overhead)
//!
//! Run with: `cargo bench -p tantivy-datafusion`

use std::sync::Arc;

use binggan::{black_box, InputGroup};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::*;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_distr::Distribution;
use serde_json::json;
use tantivy::aggregation::agg_req::Aggregations;
use tantivy::aggregation::AggregationCollector;
use tantivy::query::AllQuery;
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, FAST, STRING};
use tantivy::{doc, Index};
use tantivy_datafusion::{AggPushdown, SingleTableProvider};

// ---------------------------------------------------------------------------
// Index builder (1M docs, mixed cardinality)
// ---------------------------------------------------------------------------

fn build_bench_index(num_docs: usize) -> Index {
    let mut schema_builder = Schema::builder();
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

// ---------------------------------------------------------------------------
// Aggregation cases
// ---------------------------------------------------------------------------

struct AggCase {
    name: &'static str,
    json: serde_json::Value,
    /// SQL equivalent for the unified_sql path (no pushdown).
    sql: &'static str,
}

fn agg_cases() -> Vec<AggCase> {
    vec![
        AggCase {
            name: "terms_few",
            json: json!({ "t": { "terms": { "field": "text_few_terms_status" } } }),
            sql: "SELECT text_few_terms_status, COUNT(*) FROM t GROUP BY text_few_terms_status",
        },
        AggCase {
            name: "terms_few_with_avg",
            json: json!({
                "t": {
                    "terms": { "field": "text_few_terms_status" },
                    "aggs": { "avg_score": { "avg": { "field": "score_f64" } } }
                }
            }),
            sql: "SELECT text_few_terms_status, COUNT(*), AVG(score_f64) FROM t GROUP BY text_few_terms_status",
        },
        AggCase {
            name: "avg_f64",
            json: json!({ "average": { "avg": { "field": "score_f64" } } }),
            sql: "SELECT AVG(score_f64) FROM t",
        },
        AggCase {
            name: "stats_f64",
            json: json!({ "s": { "stats": { "field": "score_f64" } } }),
            sql: "SELECT MIN(score_f64), MAX(score_f64), SUM(score_f64), COUNT(score_f64), AVG(score_f64) FROM t",
        },
    ]
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

/// Tantivy native: AggregationCollector directly.
fn run_native(index: &Index, agg_req: &serde_json::Value) {
    let aggs: Aggregations = serde_json::from_value(agg_req.clone()).unwrap();
    let collector = AggregationCollector::from_aggs(aggs, Default::default());
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    black_box(searcher.search(&AllQuery, &collector).unwrap());
}

/// Unified path with AggPushdown: SQL through SingleTableProvider.
/// AggPushdown rewrites AggregateExec → AggDataSource which calls
/// tantivy's AggregationSegmentCollector. Should match native perf.
struct UnifiedPushdownCtx {
    rt: tokio::runtime::Runtime,
    plan: Arc<dyn ExecutionPlan>,
    ctx: SessionContext,
}

impl UnifiedPushdownCtx {
    fn new(index: &Index, sql: &str) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let config = SessionConfig::new().with_target_partitions(1);
        let state = SessionStateBuilder::new()
            .with_config(config)
            .with_default_features()
            .with_physical_optimizer_rule(Arc::new(AggPushdown::new()))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table("t", Arc::new(SingleTableProvider::new(index.clone())))
            .unwrap();
        let plan = rt.block_on(async {
            ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
        });
        Self { rt, plan, ctx }
    }

    fn run(&self) {
        self.rt.block_on(async {
            let task_ctx = self.ctx.task_ctx();
            let batches =
                datafusion::physical_plan::collect(self.plan.clone(), task_ctx).await.unwrap();
            black_box(batches);
        });
    }
}

/// Unified path WITHOUT AggPushdown: SQL goes through DataFusion's
/// hash GROUP BY on Arrow batches. Measures the overhead vs native.
struct UnifiedSqlCtx {
    rt: tokio::runtime::Runtime,
    plan: Arc<dyn ExecutionPlan>,
    ctx: SessionContext,
}

impl UnifiedSqlCtx {
    fn new(index: &Index, sql: &str) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let config = SessionConfig::new().with_target_partitions(1);
        let ctx = SessionContext::new_with_config(config);
        ctx.register_table("t", Arc::new(SingleTableProvider::new(index.clone())))
            .unwrap();
        let plan = rt.block_on(async {
            ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
        });
        Self { rt, plan, ctx }
    }

    fn run(&self) {
        self.rt.block_on(async {
            let task_ctx = self.ctx.task_ctx();
            let batches =
                datafusion::physical_plan::collect(self.plan.clone(), task_ctx).await.unwrap();
            black_box(batches);
        });
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cases = agg_cases();
    let index_1m = build_bench_index(1_000_000);
    let index_10m = build_bench_index(10_000_000);

    // Print segment info
    for (name, idx) in &[("1M", &index_1m), ("10M", &index_10m)] {
        let r = idx.reader().unwrap();
        let s = r.searcher();
        eprintln!("{name}: {} segments", s.segment_readers().len());
        for (i, sr) in s.segment_readers().iter().enumerate() {
            eprintln!("  seg {i}: {} docs", sr.max_doc());
        }
    }

    let inputs: Vec<(&str, Index)> = vec![
        ("1M_docs", index_1m.clone()),
        ("10M_docs", index_10m.clone()),
    ];

    // Pre-build contexts for each index size × case
    let pushdown_1m: Vec<Arc<UnifiedPushdownCtx>> = cases.iter()
        .map(|c| Arc::new(UnifiedPushdownCtx::new(&index_1m, c.sql))).collect();
    let pushdown_10m: Vec<Arc<UnifiedPushdownCtx>> = cases.iter()
        .map(|c| Arc::new(UnifiedPushdownCtx::new(&index_10m, c.sql))).collect();

    // Run each case × each index size separately to avoid per-iteration index detection
    for (i, case) in cases.iter().enumerate() {
        // 1M benchmark
        {
            let pd = pushdown_1m[i].clone();
            let mut group = InputGroup::new_with_inputs(vec![("1M_docs", index_1m.clone())]);
            let json = case.json.clone();
            group.register(&format!("native/{}", case.name), move |index| {
                run_native(index, &json);
            });
            let pd_clone = pd.clone();
            group.register(&format!("unified_pushdown/{}", case.name), move |_| {
                pd_clone.run();
            });
            group.run();
        }

        // 10M benchmark
        {
            let pd = pushdown_10m[i].clone();
            let mut group = InputGroup::new_with_inputs(vec![("10M_docs", index_10m.clone())]);
            let json = case.json.clone();
            group.register(&format!("native/{}", case.name), move |index| {
                run_native(index, &json);
            });
            let pd_clone = pd.clone();
            group.register(&format!("unified_pushdown/{}", case.name), move |_| {
                pd_clone.run();
            });
            group.run();
        }
    }
}
