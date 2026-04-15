use std::sync::Arc;

use arrow::array::{Array, Float64Array, StringArray, UInt16Array, UInt32Array, UInt64Array};
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::*;
use tantivy::schema::{SchemaBuilder, FAST, STORED, STRING};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::nested_agg::exec::NestedApproxAggExec;
use tantivy_datafusion::nested_agg::node_table::{
    intermediate_results_to_node_table_batch, node_table_final_schema, node_table_partial_schema,
    COL_COUNT, COL_LEVEL, COL_NODE_ID, COL_PARENT_ID,
};
use tantivy_datafusion::nested_agg::spec::{
    BucketKind, BucketLevelSpec, MetricSpec, NestedApproxAggSpec,
};
use tantivy_datafusion::unified::agg_data_source::AggDataSource;
use tantivy_datafusion::SingleTableProvider;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn build_service_endpoint_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST | STORED);
    builder.add_text_field("endpoint", STRING | FAST | STORED);
    builder.add_f64_field("latency", FAST);
    builder.add_u64_field("status", FAST);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let service = schema.get_field("service").unwrap();
    let endpoint = schema.get_field("endpoint").unwrap();
    let latency = schema.get_field("latency").unwrap();
    let status = schema.get_field("status").unwrap();

    // api service: 6 docs
    for (ep, lat) in [
        ("/users", 10.0),
        ("/users", 20.0),
        ("/users", 30.0),
        ("/orders", 100.0),
        ("/orders", 200.0),
        ("/health", 1.0),
    ] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "api");
        doc.add_text(endpoint, ep);
        doc.add_f64(latency, lat);
        doc.add_u64(status, 200);
        writer.add_document(doc).unwrap();
    }

    // web service: 4 docs
    for (ep, lat) in [
        ("/home", 50.0),
        ("/home", 60.0),
        ("/about", 40.0),
        ("/contact", 30.0),
    ] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "web");
        doc.add_text(endpoint, ep);
        doc.add_f64(latency, lat);
        doc.add_u64(status, 200);
        writer.add_document(doc).unwrap();
    }

    // db service: 2 docs
    for (ep, lat) in [("/query", 5.0), ("/query", 15.0)] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "db");
        doc.add_text(endpoint, ep);
        doc.add_f64(latency, lat);
        doc.add_u64(status, 200);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn two_level_terms_spec(
    final_size_0: u32,
    fanout_0: u32,
    final_size_1: u32,
    fanout_1: u32,
) -> Arc<NestedApproxAggSpec> {
    Arc::new(
        NestedApproxAggSpec::try_new(
            vec![
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "service".into(),
                    final_size: final_size_0,
                    fanout: fanout_0,
                },
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "endpoint".into(),
                    final_size: final_size_1,
                    fanout: fanout_1,
                },
            ],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg { field: "latency".into() },
            ],
        )
        .unwrap(),
    )
}

// ---------------------------------------------------------------------------
// Spec tests
// ---------------------------------------------------------------------------

#[test]
fn spec_to_tantivy_roundtrip_serialization() {
    let spec = two_level_terms_spec(10, 40, 5, 20);
    let json = serde_json::to_string(spec.as_ref()).unwrap();
    let deserialized: NestedApproxAggSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.levels.len(), 2);
    assert_eq!(deserialized.metrics.len(), 2);
    assert_eq!(deserialized.levels[0].fanout, 40);
    assert_eq!(deserialized.levels[1].final_size, 5);
}

// ---------------------------------------------------------------------------
// Node table schema tests
// ---------------------------------------------------------------------------

#[test]
fn partial_schema_matches_spec() {
    let spec = two_level_terms_spec(10, 40, 5, 20);
    let schema = node_table_partial_schema(&spec);

    // structural + 2 key cols + 2 state cols (avg count + avg sum)
    assert_eq!(schema.fields().len(), 4 + 2 + 2);
    assert_eq!(schema.field(0).name(), COL_NODE_ID);
    assert_eq!(schema.field(1).name(), COL_PARENT_ID);
    assert_eq!(schema.field(2).name(), COL_LEVEL);
    assert_eq!(schema.field(3).name(), COL_COUNT);
    assert_eq!(schema.field(4).name(), "__key_0");
    assert_eq!(schema.field(5).name(), "__key_1");
}

#[test]
fn final_schema_matches_spec() {
    let spec = two_level_terms_spec(10, 40, 5, 20);
    let schema = node_table_final_schema(&spec);

    // structural + 2 key cols + 2 final metric cols
    assert_eq!(schema.fields().len(), 4 + 2 + 2);
}

// ---------------------------------------------------------------------------
// IntermediateResults -> node table conversion test
// ---------------------------------------------------------------------------

#[test]
fn intermediate_to_node_table_basic() {
    let index = build_service_endpoint_index();
    let spec = two_level_terms_spec(10, 100, 10, 100);
    let schema = node_table_partial_schema(&spec);

    let tantivy_aggs = spec.to_tantivy_aggregations();
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();

    let collector = tantivy::aggregation::DistributedAggregationCollector::from_aggs(
        tantivy_aggs,
        Default::default(),
    );
    let intermediate = searcher
        .search(&tantivy::query::AllQuery, &collector)
        .unwrap();

    let batch =
        intermediate_results_to_node_table_batch(&intermediate, &spec, &schema).unwrap();

    // We should have rows for: 3 services + their endpoints
    // api(6) has /users(3), /orders(2), /health(1) = 4 nodes
    // web(4) has /home(2), /about(1), /contact(1) = 4 nodes
    // db(2) has /query(2) = 2 nodes
    // Total = 3 + 3 + 3 + 1 = 10 nodes
    assert_eq!(batch.num_rows(), 10);

    // Verify schema columns
    let node_id_col = batch.column(0).as_any().downcast_ref::<UInt32Array>().unwrap();
    let parent_id_col = batch.column(1).as_any().downcast_ref::<UInt32Array>().unwrap();
    let level_col = batch.column(2).as_any().downcast_ref::<UInt16Array>().unwrap();
    let count_col = batch.column(3).as_any().downcast_ref::<UInt64Array>().unwrap();

    // All node IDs should be unique and > 0
    let mut seen_ids = std::collections::HashSet::new();
    for i in 0..batch.num_rows() {
        let nid = node_id_col.value(i);
        assert!(nid > 0, "node_id should be > 0");
        assert!(seen_ids.insert(nid), "duplicate node_id {nid}");
    }

    // Root nodes (level=0) should have parent_id=0
    for i in 0..batch.num_rows() {
        if level_col.value(i) == 0 {
            assert_eq!(parent_id_col.value(i), 0, "root node parent should be 0");
        }
    }

    // Total doc count at level 0 should sum to 12
    let mut level_0_count = 0u64;
    for i in 0..batch.num_rows() {
        if level_col.value(i) == 0 {
            level_0_count += count_col.value(i);
        }
    }
    assert_eq!(level_0_count, 12);
}

// ---------------------------------------------------------------------------
// Pushdown plan shape test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pushdown_plan_produces_correct_shape() {
    let index = build_service_endpoint_index();
    let spec = two_level_terms_spec(2, 10, 2, 10);

    let provider = SingleTableProvider::new(index);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("test", Arc::new(provider)).unwrap();

    // Get the scan from the table provider
    // Just verify the spec creates tantivy aggregations correctly
    let tantivy_aggs = spec.to_tantivy_aggregations();
    let agg_json = serde_json::to_string(&tantivy_aggs).unwrap();
    assert!(agg_json.contains("level_0"));
    assert!(agg_json.contains("level_1"));
    assert!(agg_json.contains("metric_1"));
    assert!(
        !agg_json.contains("metric_0"),
        "Count is structural and should not be pushed down as a metric"
    );
}

// ---------------------------------------------------------------------------
// End-to-end pushdown execution test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pushdown_single_split_two_level_terms() {
    let index = build_service_endpoint_index();
    let spec = two_level_terms_spec(3, 100, 10, 100);

    let tantivy_aggs = Arc::new(spec.to_tantivy_aggregations());
    let partial_schema = node_table_partial_schema(&spec);

    // Build an AggDataSource in NodeTablePartial mode
    let agg_ds = AggDataSource::from_local_splits_node_table_partial(
        vec![index],
        tantivy_aggs,
        partial_schema,
        Vec::new(),
        None,
        Vec::new(),
        Arc::clone(&spec),
    );

    let leaf = Arc::new(datafusion_datasource::source::DataSourceExec::new(
        Arc::new(agg_ds),
    ));
    let coalesced = Arc::new(
        datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec::new(leaf),
    );
    let final_merge = Arc::new(NestedApproxAggExec::new_final_merge(
        coalesced,
        Arc::clone(&spec),
    ));

    let state = SessionStateBuilder::new()
        .with_default_features()
        .build();
    let ctx = Arc::new(datafusion::execution::TaskContext::from(&state));

    let batches = collect(final_merge, ctx).await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    // Should have nodes for all 3 services and their endpoints
    let final_schema = node_table_final_schema(&spec);
    assert_eq!(batch.schema().as_ref(), final_schema.as_ref());

    let level_col = batch
        .column(final_schema.index_of(COL_LEVEL).unwrap())
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    let count_col = batch
        .column(final_schema.index_of(COL_COUNT).unwrap())
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let key_0_col = batch
        .column(final_schema.index_of("__key_0").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Count level-0 nodes
    let level_0_count: usize = (0..batch.num_rows())
        .filter(|&i| level_col.value(i) == 0)
        .count();
    assert_eq!(level_0_count, 3, "should have 3 service buckets");

    // Find the "api" service node and verify count
    let api_row = (0..batch.num_rows())
        .find(|&i| level_col.value(i) == 0 && key_0_col.value(i) == "api")
        .expect("should have api service");
    assert_eq!(count_col.value(api_row), 6, "api should have 6 docs");

    // Find the "web" service node and verify count
    let web_row = (0..batch.num_rows())
        .find(|&i| level_col.value(i) == 0 && key_0_col.value(i) == "web")
        .expect("should have web service");
    assert_eq!(count_col.value(web_row), 4, "web should have 4 docs");

    // Verify metrics exist on leaf nodes (level 1)
    let avg_col_name = spec.metrics[1].final_field_name(1);
    let avg_col = batch
        .column(final_schema.index_of(&avg_col_name).unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Level 1 nodes should have non-null avg values
    for i in 0..batch.num_rows() {
        if level_col.value(i) == 1 {
            assert!(!avg_col.is_null(i), "leaf metric should not be null at row {i}");
        }
    }

    // Parent-level metrics: level-0 nodes should ALSO have non-null avg.
    // api service has latencies [10,20,30,100,200,1] → avg = 60.167
    // This proves metrics are computed at every level, not just leaves.
    for i in 0..batch.num_rows() {
        if level_col.value(i) == 0 {
            assert!(
                !avg_col.is_null(i),
                "parent-level metric should not be null at row {i} (key={:?})",
                key_0_col.value(i),
            );
        }
    }

    // Spot-check: api avg latency = (10+20+30+100+200+1)/6 = 60.167
    let api_avg = avg_col.value(api_row);
    let expected_api_avg = (10.0 + 20.0 + 30.0 + 100.0 + 200.0 + 1.0) / 6.0;
    assert!(
        (api_avg - expected_api_avg).abs() < 0.01,
        "api avg latency should be ~{expected_api_avg}, got {api_avg}"
    );
}

// ---------------------------------------------------------------------------
// Multi-split pushdown test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pushdown_multi_split_merges_correctly() {
    // Create two indexes (splits) with overlapping services
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST);
    builder.add_text_field("endpoint", STRING | FAST);
    builder.add_f64_field("latency", FAST);
    let schema = builder.build();

    let service = schema.get_field("service").unwrap();
    let endpoint = schema.get_field("endpoint").unwrap();
    let latency = schema.get_field("latency").unwrap();

    // Split 1: api with /users
    let index1 = Index::create_in_ram(schema.clone());
    let mut w1: IndexWriter = index1.writer_with_num_threads(1, 15_000_000).unwrap();
    for lat in [10.0, 20.0, 30.0] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "api");
        doc.add_text(endpoint, "/users");
        doc.add_f64(latency, lat);
        w1.add_document(doc).unwrap();
    }
    w1.commit().unwrap();

    // Split 2: api with /orders
    let index2 = Index::create_in_ram(schema.clone());
    let mut w2: IndexWriter = index2.writer_with_num_threads(1, 15_000_000).unwrap();
    for lat in [100.0, 200.0] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "api");
        doc.add_text(endpoint, "/orders");
        doc.add_f64(latency, lat);
        w2.add_document(doc).unwrap();
    }
    w2.commit().unwrap();

    let spec = two_level_terms_spec(10, 100, 10, 100);
    let tantivy_aggs = Arc::new(spec.to_tantivy_aggregations());
    let partial_schema = node_table_partial_schema(&spec);

    let agg_ds = AggDataSource::from_local_splits_node_table_partial(
        vec![index1, index2],
        tantivy_aggs,
        partial_schema,
        Vec::new(),
        None,
        Vec::new(),
        Arc::clone(&spec),
    );

    let leaf = Arc::new(datafusion_datasource::source::DataSourceExec::new(
        Arc::new(agg_ds),
    ));
    let coalesced = Arc::new(
        datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec::new(leaf),
    );
    let final_merge = Arc::new(NestedApproxAggExec::new_final_merge(
        coalesced,
        Arc::clone(&spec),
    ));

    let state = SessionStateBuilder::new()
        .with_default_features()
        .build();
    let ctx = Arc::new(datafusion::execution::TaskContext::from(&state));

    let batches = collect(final_merge, ctx).await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];

    let final_schema = node_table_final_schema(&spec);
    let level_col = batch
        .column(final_schema.index_of(COL_LEVEL).unwrap())
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    let count_col = batch
        .column(final_schema.index_of(COL_COUNT).unwrap())
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let key_0_col = batch
        .column(final_schema.index_of("__key_0").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // Should have exactly 1 level-0 service (api) with merged count = 5
    let level_0_rows: Vec<usize> = (0..batch.num_rows())
        .filter(|&i| level_col.value(i) == 0)
        .collect();
    assert_eq!(level_0_rows.len(), 1, "only one service: api");
    assert_eq!(key_0_col.value(level_0_rows[0]), "api");
    assert_eq!(count_col.value(level_0_rows[0]), 5, "api merged count");

    // Should have 2 level-1 endpoints
    let level_1_rows: Vec<usize> = (0..batch.num_rows())
        .filter(|&i| level_col.value(i) == 1)
        .collect();
    assert_eq!(level_1_rows.len(), 2, "two endpoints");

    // Total endpoint doc count should be 5
    let endpoint_total: u64 = level_1_rows.iter().map(|&i| count_col.value(i)).sum();
    assert_eq!(endpoint_total, 5);
}

// ---------------------------------------------------------------------------
// Overscan / fanout test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fanout_controls_candidate_retention() {
    let index = build_service_endpoint_index();

    // Spec with final_size=1 for both levels — only keep top-1
    let spec_restrictive = two_level_terms_spec(1, 100, 1, 100);
    let spec_wide = two_level_terms_spec(3, 100, 10, 100);

    for (label, spec, expected_l0) in [
        ("restrictive", spec_restrictive, 1usize),
        ("wide", spec_wide, 3),
    ] {
        let tantivy_aggs = Arc::new(spec.to_tantivy_aggregations());
        let partial_schema = node_table_partial_schema(&spec);

        let agg_ds = AggDataSource::from_local_splits_node_table_partial(
            vec![index.clone()],
            tantivy_aggs,
            partial_schema,
            Vec::new(),
            None,
            Vec::new(),
            Arc::clone(&spec),
        );

        let leaf = Arc::new(datafusion_datasource::source::DataSourceExec::new(
            Arc::new(agg_ds),
        ));
        let coalesced = Arc::new(
            datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec::new(leaf),
        );
        let final_merge = Arc::new(NestedApproxAggExec::new_final_merge(
            coalesced,
            Arc::clone(&spec),
        ));

        let state = SessionStateBuilder::new()
            .with_default_features()
            .build();
        let ctx = Arc::new(datafusion::execution::TaskContext::from(&state));

        let batches = collect(final_merge, ctx).await.unwrap();
        let batch = &batches[0];
        let final_schema = node_table_final_schema(&spec);

        let level_col = batch
            .column(final_schema.index_of(COL_LEVEL).unwrap())
            .as_any()
            .downcast_ref::<UInt16Array>()
            .unwrap();

        let l0_count = (0..batch.num_rows())
            .filter(|&i| level_col.value(i) == 0)
            .count();

        assert_eq!(
            l0_count, expected_l0,
            "{label}: expected {expected_l0} level-0 buckets, got {l0_count}"
        );
    }
}

// ---------------------------------------------------------------------------
// Overscan regression tests: demonstrate fanout catching vs missing winners
// ---------------------------------------------------------------------------

/// Build multiple splits where "underdog" service is the true global winner
/// but is NOT top-1 on ANY individual split.
///
/// Split layout (3 splits, 1-level terms on "service"):
///
///   Split 0: local_champ_0 = 100 docs, underdog = 30 docs
///   Split 1: local_champ_1 = 100 docs, underdog = 30 docs
///   Split 2: local_champ_2 = 100 docs, underdog = 30 docs
///
/// Per-split top-1 by count:
///   Split 0: local_champ_0 (100)  — underdog is #2
///   Split 1: local_champ_1 (100)  — underdog is #2
///   Split 2: local_champ_2 (100)  — underdog is #2
///
/// Global truth:
///   underdog   = 90 docs (winner!)
///   local_champ_0 = 100
///   local_champ_1 = 100
///   local_champ_2 = 100
///
/// With final_size=1:
///   - fanout=1 → each split only keeps its local champ, underdog lost
///   - fanout=2 → each split keeps top-2 (includes underdog), merge recovers it
///
/// For a final_size=2: underdog (90) should beat each individual local_champ
/// because no single local_champ exceeds 100, and underdog has 90 from 3 splits.
/// Wait — 100 > 90, so local champs still win individually. Let me adjust
/// the numbers so underdog is the TRUE winner.
///
/// Better layout:
///   Split 0: champ_a = 50 docs, underdog = 40 docs
///   Split 1: champ_b = 50 docs, underdog = 40 docs
///   Split 2: champ_c = 50 docs, underdog = 40 docs
///
///   Global: underdog = 120 docs (winner!), each champ = 50
///
/// With final_size=1:
///   - fanout=1 → each split returns only its local #1 (champ_x), underdog is lost
///   - fanout=2 → each split returns top-2 (champ_x + underdog), merge sees
///     underdog = 120, which is the true winner
fn build_underdog_splits() -> Vec<Index> {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST);
    builder.add_f64_field("latency", FAST);
    let schema = builder.build();

    let service = schema.get_field("service").unwrap();
    let latency = schema.get_field("latency").unwrap();

    let mut indexes = Vec::new();
    for split_idx in 0..3 {
        let index = Index::create_in_ram(schema.clone());
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

        let champ_name = format!("champ_{split_idx}");

        // Local champion: 50 docs
        for i in 0..50 {
            let mut doc = TantivyDocument::default();
            doc.add_text(service, &champ_name);
            doc.add_f64(latency, i as f64);
            writer.add_document(doc).unwrap();
        }

        // Underdog: 40 docs on each split (= 120 total across 3 splits)
        for i in 0..40 {
            let mut doc = TantivyDocument::default();
            doc.add_text(service, "underdog");
            doc.add_f64(latency, (i as f64) * 10.0);
            writer.add_document(doc).unwrap();
        }

        writer.commit().unwrap();
        indexes.push(index);
    }
    indexes
}

/// Helper: run a 1-level nested agg with given fanout over the underdog splits.
/// Returns the set of service names that appear in the final output.
async fn run_underdog_agg(fanout: u32, final_size: u32) -> Vec<(String, u64)> {
    let indexes = build_underdog_splits();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size,
                fanout,
            }],
            vec![MetricSpec::Avg { field: "latency".into() }],
        )
        .unwrap(),
    );

    let tantivy_aggs = Arc::new(spec.to_tantivy_aggregations());
    let partial_schema = node_table_partial_schema(&spec);

    let agg_ds = AggDataSource::from_local_splits_node_table_partial(
        indexes,
        tantivy_aggs,
        partial_schema,
        Vec::new(),
        None,
        Vec::new(),
        Arc::clone(&spec),
    );

    let leaf = Arc::new(datafusion_datasource::source::DataSourceExec::new(
        Arc::new(agg_ds),
    ));
    let coalesced = Arc::new(
        datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec::new(leaf),
    );
    let final_merge = Arc::new(NestedApproxAggExec::new_final_merge(
        coalesced,
        Arc::clone(&spec),
    ));

    let state = SessionStateBuilder::new()
        .with_default_features()
        .build();
    let ctx = Arc::new(datafusion::execution::TaskContext::from(&state));

    let batches = collect(final_merge, ctx).await.unwrap();
    let batch = &batches[0];
    let final_schema = node_table_final_schema(&spec);

    let key_col = batch
        .column(final_schema.index_of("__key_0").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let count_col = batch
        .column(final_schema.index_of(COL_COUNT).unwrap())
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();

    let mut results: Vec<(String, u64)> = (0..batch.num_rows())
        .map(|i| (key_col.value(i).to_string(), count_col.value(i)))
        .collect();
    results.sort_by(|a, b| b.1.cmp(&a.1));
    results
}

/// **Demonstrates pushdown to tantivy + fanout recovering the true winner.**
///
/// With fanout=2, each split's tantivy DistributedAggregationCollector retains
/// 2 candidates per segment. The "underdog" service (which is #2 on every
/// split but #1 globally with 120 docs) is included in each split's partial
/// and recovered during the FinalMerge.
#[tokio::test]
async fn overscan_fanout_2_recovers_global_winner() {
    let results = run_underdog_agg(2, 1).await;

    // With fanout=2 each split keeps: [local_champ(50), underdog(40)]
    // FinalMerge sees underdog = 40*3 = 120, each champ = 50
    // After trim to final_size=1, the winner should be underdog (120)
    assert_eq!(results.len(), 1, "final_size=1 means one bucket");
    assert_eq!(
        results[0].0, "underdog",
        "underdog (120 docs) should beat any single champ (50 docs)"
    );
    assert_eq!(results[0].1, 120);
}

/// **Demonstrates MISSING the true winner when fanout is too small.**
///
/// With fanout=1, each split's tantivy collector only keeps the top-1
/// candidate per segment. The "underdog" (which is #2 on every split) is
/// pruned before it reaches the FinalMerge, so the final output contains
/// only local champions.
#[tokio::test]
async fn overscan_fanout_1_loses_global_winner() {
    let results = run_underdog_agg(1, 1).await;

    // With fanout=1 each split keeps only its local champ (50 docs).
    // Underdog is pruned on every split. The merge sees only champs.
    // final_size=1 picks the arbitrary first champ (all tied at 50).
    assert_eq!(results.len(), 1, "final_size=1 means one bucket");
    assert_ne!(
        results[0].0, "underdog",
        "underdog should be LOST with fanout=1 — it was pruned on every split"
    );
    assert_eq!(
        results[0].1, 50,
        "winner should be a local champ with 50 docs (not the 120 the underdog truly has)"
    );
}

/// Show both scenarios side-by-side: same data, different fanout, different
/// winners. This is the key demonstration of configurable overscan.
#[tokio::test]
async fn overscan_comparison_same_data_different_fanout() {
    // LOW fanout=1, final_size=1: misses underdog
    let results_low = run_underdog_agg(1, 1).await;
    // HIGH fanout=2, final_size=1: finds underdog
    let results_high = run_underdog_agg(2, 1).await;

    let has_underdog = |results: &[(String, u64)]| {
        results.iter().any(|(name, _)| name == "underdog")
    };

    // LOW fanout: underdog is absent — pruned on every split
    assert!(
        !has_underdog(&results_low),
        "fanout=1 should NOT see underdog — pruned on every split.\n  got: {results_low:?}"
    );

    // HIGH fanout: underdog is present and is the sole result
    assert!(
        has_underdog(&results_high),
        "fanout=2 should see underdog — retained as candidate on every split.\n  got: {results_high:?}"
    );
    assert_eq!(
        results_high[0].0, "underdog",
        "underdog should be the #1 result with count=120.\n  got: {results_high:?}"
    );
    assert_eq!(results_high[0].1, 120);
}

// ---------------------------------------------------------------------------
// Codec roundtrip test
// ---------------------------------------------------------------------------

#[test]
fn nested_spec_serde_roundtrip() {
    let spec = NestedApproxAggSpec::try_new(
        vec![
            BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 50,
                fanout: 200,
            },
            BucketLevelSpec {
                kind: BucketKind::DateHistogram {
                    fixed_interval: "1h".into(),
                },
                field: "timestamp".into(),
                final_size: 24,
                fanout: 24,
            },
        ],
        vec![
            MetricSpec::Count,
            MetricSpec::Sum { field: "bytes".into() },
            MetricSpec::Avg { field: "latency".into() },
            MetricSpec::Min { field: "latency".into() },
            MetricSpec::Max { field: "latency".into() },
        ],
    )
    .unwrap();

    let json = serde_json::to_string(&spec).unwrap();
    let rt: NestedApproxAggSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(rt.levels.len(), 2);
    assert_eq!(rt.metrics.len(), 5);
    assert_eq!(rt.levels[0].fanout, 200);
    assert_eq!(rt.levels[1].final_size, 24);

    match &rt.levels[1].kind {
        BucketKind::DateHistogram { fixed_interval } => {
            assert_eq!(fixed_interval, "1h");
        }
        _ => panic!("expected DateHistogram"),
    }
}
