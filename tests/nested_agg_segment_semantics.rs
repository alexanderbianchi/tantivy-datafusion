use std::sync::Arc;

use arrow::array::{StringArray, UInt64Array};
use datafusion::datasource::TableProvider;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::*;
use tantivy::schema::{SchemaBuilder, FAST, STRING};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::nested_agg::node_table::{node_table_final_schema, COL_COUNT};
use tantivy_datafusion::nested_agg::plan_builder::build_nested_approx_plan;
use tantivy_datafusion::nested_agg::spec::{
    BucketKind, BucketLevelSpec, MetricSpec, NestedApproxAggSpec,
};
use tantivy_datafusion::SingleTableProvider;

fn build_segmented_underdog_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST);
    let schema = builder.build();

    let service = schema.get_field("service").unwrap();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    for segment_idx in 0..3 {
        let champ = format!("champ_{segment_idx}");

        for _ in 0..50 {
            let mut doc = TantivyDocument::default();
            doc.add_text(service, &champ);
            writer.add_document(doc).unwrap();
        }

        for _ in 0..40 {
            let mut doc = TantivyDocument::default();
            doc.add_text(service, "underdog");
            writer.add_document(doc).unwrap();
        }

        writer.commit().unwrap();
    }

    index
}

fn session_context() -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(1);
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .build();
    SessionContext::new_with_state(state)
}

async fn raw_scan_plan(provider: &Arc<SingleTableProvider>, ctx: &SessionContext) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    provider.scan(&ctx.state(), None, &[], None).await.unwrap()
}

async fn projected_plan(ctx: &SessionContext) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    ctx.sql("SELECT service AS __na_key_0, _segment_ord FROM t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

async fn execute_results(
    plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ctx: &SessionContext,
    spec: &NestedApproxAggSpec,
) -> Vec<(String, u64)> {
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    let schema = node_table_final_schema(spec);
    let key_col = batch
        .column(schema.index_of("__key_0").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let count_col = batch
        .column(schema.index_of(COL_COUNT).unwrap())
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();

    (0..batch.num_rows())
        .map(|row| (key_col.value(row).to_string(), count_col.value(row)))
        .collect()
}

fn one_level_terms_spec(fanout: u32) -> Arc<NestedApproxAggSpec> {
    Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 1,
                fanout,
            }],
            vec![MetricSpec::Count],
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn segment_aware_fallback_matches_pushdown_when_fanout_is_one() {
    let index = build_segmented_underdog_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = one_level_terms_spec(1);
    let pushdown = build_nested_approx_plan(raw_scan_plan(&provider, &ctx).await, Arc::clone(&spec))
        .unwrap();
    let fallback = build_nested_approx_plan(projected_plan(&ctx).await, Arc::clone(&spec)).unwrap();

    let pushdown_rows = execute_results(pushdown, &ctx, &spec).await;
    let fallback_rows = execute_results(fallback, &ctx, &spec).await;
    assert_eq!(pushdown_rows, fallback_rows);
    assert_eq!(pushdown_rows, vec![("champ_0".to_string(), 50)]);
}

#[tokio::test]
async fn segment_aware_fallback_matches_pushdown_when_fanout_is_two() {
    let index = build_segmented_underdog_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = one_level_terms_spec(2);
    let pushdown = build_nested_approx_plan(raw_scan_plan(&provider, &ctx).await, Arc::clone(&spec))
        .unwrap();
    let fallback = build_nested_approx_plan(projected_plan(&ctx).await, Arc::clone(&spec)).unwrap();

    let pushdown_rows = execute_results(pushdown, &ctx, &spec).await;
    let fallback_rows = execute_results(fallback, &ctx, &spec).await;
    assert_eq!(pushdown_rows, fallback_rows);
    assert_eq!(pushdown_rows, vec![("underdog".to_string(), 120)]);
}
