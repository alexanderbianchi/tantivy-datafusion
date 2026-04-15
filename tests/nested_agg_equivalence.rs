use std::sync::Arc;

use arrow::array::{Array, Float64Array, StringArray, UInt16Array, UInt64Array};
use datafusion::datasource::TableProvider;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::*;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::Partitioning;
use tantivy::schema::{SchemaBuilder, FAST, STORED, STRING};
use tantivy::{DateTime, Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::nested_agg::exec::{NestedApproxAggExec, NestedApproxAggMode};
use tantivy_datafusion::nested_agg::node_table::{node_table_final_schema, COL_COUNT, COL_LEVEL};
use tantivy_datafusion::nested_agg::plan_builder::build_nested_approx_plan;
use tantivy_datafusion::nested_agg::spec::{
    BucketKind, BucketLevelSpec, MetricSpec, NestedApproxAggSpec,
};
use tantivy_datafusion::SingleTableProvider;

fn build_service_endpoint_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST | STORED);
    builder.add_text_field("endpoint", STRING | FAST | STORED);
    builder.add_f64_field("latency", FAST);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let service = schema.get_field("service").unwrap();
    let endpoint = schema.get_field("endpoint").unwrap();
    let latency = schema.get_field("latency").unwrap();

    for (service_value, endpoint_value, latency_value) in [
        ("api", "/users", 10.0),
        ("api", "/users", 20.0),
        ("api", "/users", 30.0),
        ("api", "/orders", 100.0),
        ("api", "/orders", 200.0),
        ("api", "/health", 1.0),
        ("web", "/home", 50.0),
        ("web", "/home", 60.0),
        ("web", "/about", 40.0),
        ("web", "/contact", 30.0),
        ("db", "/query", 5.0),
        ("db", "/query", 15.0),
    ] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, service_value);
        doc.add_text(endpoint, endpoint_value);
        doc.add_f64(latency, latency_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
    index
}

fn build_date_histogram_index() -> Index {
    let mut builder = SchemaBuilder::new();
    builder.add_date_field("ts", FAST);
    builder.add_text_field("service", STRING | FAST | STORED);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let ts = schema.get_field("ts").unwrap();
    let service = schema.get_field("service").unwrap();

    for (ts_micros, service_value) in [
        (5 * 60 * 1_000_000_i64, "api"),
        (15 * 60 * 1_000_000_i64, "web"),
        (70 * 60 * 1_000_000_i64, "api"),
        (80 * 60 * 1_000_000_i64, "api"),
        (120 * 60 * 1_000_000_i64, "web"),
    ] {
        let mut doc = TantivyDocument::default();
        doc.add_date(ts, DateTime::from_timestamp_micros(ts_micros));
        doc.add_text(service, service_value);
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();
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

async fn projected_plan(ctx: &SessionContext, sql: &str) -> Arc<dyn datafusion::physical_plan::ExecutionPlan> {
    ctx.sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}

async fn collect_single_batch(
    plan: Arc<dyn datafusion::physical_plan::ExecutionPlan>,
    ctx: &SessionContext,
) -> arrow::record_batch::RecordBatch {
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.len(), 1);
    batches.into_iter().next().unwrap()
}

fn render_final_rows(
    batch: &arrow::record_batch::RecordBatch,
    spec: &NestedApproxAggSpec,
) -> Vec<String> {
    let schema = node_table_final_schema(spec);
    let level_col = batch
        .column(schema.index_of(COL_LEVEL).unwrap())
        .as_any()
        .downcast_ref::<UInt16Array>()
        .unwrap();
    let count_col = batch
        .column(schema.index_of(COL_COUNT).unwrap())
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    let key_cols: Vec<&StringArray> = (0..spec.levels.len())
        .map(|level| {
            batch.column(schema.index_of(&format!("__key_{level}")).unwrap())
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        })
        .collect();
    let metric_cols: Vec<&Float64Array> = spec
        .metrics
        .iter()
        .enumerate()
        .map(|(metric_idx, metric)| {
            batch.column(schema.index_of(&metric.final_field_name(metric_idx)).unwrap())
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
        })
        .collect();

    (0..batch.num_rows())
        .map(|row| {
            let keys: Vec<String> = key_cols
                .iter()
                .map(|column| {
                    if column.is_null(row) {
                        "null".to_string()
                    } else {
                        column.value(row).to_string()
                    }
                })
                .collect();
            let metrics: Vec<String> = metric_cols
                .iter()
                .map(|column| {
                    if column.is_null(row) {
                        "null".to_string()
                    } else {
                        format!("{:.6}", column.value(row))
                    }
                })
                .collect();
            format!(
                "level={} count={} keys={:?} metrics={:?}",
                level_col.value(row),
                count_col.value(row),
                keys,
                metrics
            )
        })
        .collect()
}

fn two_level_terms_spec() -> Arc<NestedApproxAggSpec> {
    Arc::new(
        NestedApproxAggSpec::try_new(
            vec![
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "service".into(),
                    final_size: 10,
                    fanout: 40,
                },
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "endpoint".into(),
                    final_size: 10,
                    fanout: 40,
                },
            ],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg {
                    field: "latency".into(),
                },
            ],
        )
        .unwrap(),
    )
}

#[tokio::test]
async fn pushdown_and_projected_fallback_match_for_terms_terms_avg() {
    let index = build_service_endpoint_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = two_level_terms_spec();

    let pushdown_plan = build_nested_approx_plan(raw_scan_plan(&provider, &ctx).await, Arc::clone(&spec))
        .unwrap();
    let fallback_plan = build_nested_approx_plan(
        projected_plan(
            &ctx,
            "SELECT \
               service AS __na_key_0, \
               endpoint AS __na_key_1, \
               CAST(latency AS DOUBLE) AS __na_metric_1, \
               _segment_ord \
             FROM t",
        )
        .await,
        Arc::clone(&spec),
    )
    .unwrap();

    let pushdown_rows = render_final_rows(&collect_single_batch(pushdown_plan, &ctx).await, &spec);
    let fallback_rows = render_final_rows(&collect_single_batch(fallback_plan, &ctx).await, &spec);
    assert_eq!(pushdown_rows, fallback_rows);
}

#[tokio::test]
async fn projected_expression_bucket_uses_partial_split_local() {
    let index = build_service_endpoint_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "latency_band".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg {
                    field: "latency".into(),
                },
            ],
        )
        .unwrap(),
    );

    let projected = projected_plan(
        &ctx,
        "SELECT \
           CASE WHEN latency > 50 THEN 'slow' ELSE 'fast' END AS __na_key_0, \
           CAST(latency AS DOUBLE) AS __na_metric_1, \
           _segment_ord \
         FROM t",
    )
    .await;
    let plan = build_nested_approx_plan(projected, Arc::clone(&spec)).unwrap();

    let final_merge = plan
        .as_any()
        .downcast_ref::<NestedApproxAggExec>()
        .expect("top node should be NestedApproxAggExec");
    assert_eq!(final_merge.mode(), NestedApproxAggMode::FinalMerge);
    let coalesced = final_merge.children()[0]
        .as_any()
        .downcast_ref::<CoalescePartitionsExec>()
        .expect("final merge child should be CoalescePartitionsExec");
    let partial = coalesced.children()[0]
        .as_any()
        .downcast_ref::<NestedApproxAggExec>()
        .expect("coalesce child should be NestedApproxAggExec");
    assert_eq!(partial.mode(), NestedApproxAggMode::PartialSplitLocal);

    let batch = collect_single_batch(plan, &ctx).await;
    let rows = render_final_rows(&batch, &spec);
    assert_eq!(
        rows,
        vec![
            "level=0 count=9 keys=[\"fast\"] metrics=[\"9.000000\", \"22.333333\"]",
            "level=0 count=3 keys=[\"slow\"] metrics=[\"3.000000\", \"120.000000\"]",
        ]
    );
}

#[tokio::test]
async fn projected_expression_metric_is_merged_correctly() {
    let index = build_service_endpoint_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg {
                    field: "slow_latency".into(),
                },
            ],
        )
        .unwrap(),
    );

    let plan = build_nested_approx_plan(
        projected_plan(
            &ctx,
            "SELECT \
               service AS __na_key_0, \
               CAST(CASE WHEN latency > 50 THEN latency ELSE NULL END AS DOUBLE) AS __na_metric_1, \
               _segment_ord \
             FROM t",
        )
        .await,
        Arc::clone(&spec),
    )
    .unwrap();

    let rows = render_final_rows(&collect_single_batch(plan, &ctx).await, &spec);
    assert_eq!(
        rows,
        vec![
            "level=0 count=6 keys=[\"api\"] metrics=[\"6.000000\", \"150.000000\"]",
            "level=0 count=4 keys=[\"web\"] metrics=[\"4.000000\", \"60.000000\"]",
            "level=0 count=2 keys=[\"db\"] metrics=[\"2.000000\", \"null\"]",
        ]
    );
}

#[tokio::test]
async fn pushdown_and_projected_fallback_match_for_date_histogram_terms() {
    let index = build_date_histogram_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![
                BucketLevelSpec {
                    kind: BucketKind::DateHistogram {
                        fixed_interval: "1h".into(),
                    },
                    field: "ts".into(),
                    final_size: 24,
                    fanout: 24,
                },
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "service".into(),
                    final_size: 10,
                    fanout: 40,
                },
            ],
            vec![MetricSpec::Count],
        )
        .unwrap(),
    );

    let pushdown_plan = build_nested_approx_plan(raw_scan_plan(&provider, &ctx).await, Arc::clone(&spec))
        .unwrap();
    let fallback_plan = build_nested_approx_plan(
        projected_plan(
            &ctx,
            "SELECT ts AS __na_key_0, service AS __na_key_1, _segment_ord FROM t",
        )
        .await,
        Arc::clone(&spec),
    )
    .unwrap();

    let pushdown_rows = render_final_rows(&collect_single_batch(pushdown_plan, &ctx).await, &spec);
    let fallback_rows = render_final_rows(&collect_single_batch(fallback_plan, &ctx).await, &spec);
    assert_eq!(pushdown_rows, fallback_rows);
}

#[tokio::test]
async fn fallback_builder_rejects_repartitioned_normalized_input() {
    let index = build_service_endpoint_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider).unwrap();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg {
                    field: "latency".into(),
                },
            ],
        )
        .unwrap(),
    );

    let projected = projected_plan(
        &ctx,
        "SELECT \
           service AS __na_key_0, \
           CAST(latency AS DOUBLE) AS __na_metric_1, \
           _segment_ord \
         FROM t",
    )
    .await;
    let repartitioned: Arc<dyn ExecutionPlan> = Arc::new(
        RepartitionExec::try_new(projected, Partitioning::RoundRobinBatch(2)).unwrap(),
    );

    let err = build_nested_approx_plan(repartitioned, spec).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("must remain split-local") && message.contains("repartition/exchange"),
        "unexpected error: {message}"
    );
}

#[tokio::test]
async fn fallback_builder_rejects_non_float_metric_inputs_at_plan_time() {
    let index = build_service_endpoint_index();
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider).unwrap();

    let spec = Arc::new(
        NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg {
                    field: "latency".into(),
                },
            ],
        )
        .unwrap(),
    );

    let projected = projected_plan(
        &ctx,
        "SELECT \
           service AS __na_key_0, \
           service AS __na_metric_1, \
           _segment_ord \
         FROM t",
    )
    .await;

    let err = build_nested_approx_plan(projected, spec).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("__na_metric_1")
            && message.contains("Float64")
            && message.contains("List<Float64>"),
        "unexpected error: {message}"
    );
}
