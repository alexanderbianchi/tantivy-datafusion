use std::sync::Arc;

use arrow::array::{Float64Array, StringArray, UInt16Array, UInt64Array};
use datafusion::datasource::TableProvider;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::*;
use tantivy::schema::{SchemaBuilder, FAST, STORED, STRING};
use tantivy::{Index, IndexWriter, TantivyDocument};
use tantivy_datafusion::nested_agg::node_table::{node_table_final_schema, COL_COUNT, COL_LEVEL};
use tantivy_datafusion::nested_agg::plan_builder::build_nested_approx_plan;
use tantivy_datafusion::nested_agg::spec::{
    BucketKind, BucketLevelSpec, MetricSpec, NestedApproxAggSpec,
};
use tantivy_datafusion::SingleTableProvider;

fn session_context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .build();
    SessionContext::new_with_state(state)
}

fn two_level_count_spec() -> Arc<NestedApproxAggSpec> {
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
            vec![MetricSpec::Count],
        )
        .unwrap(),
    )
}

async fn execute_plan(index: Index, spec: Arc<NestedApproxAggSpec>) -> arrow::record_batch::RecordBatch {
    let provider = Arc::new(SingleTableProvider::new(index));
    let ctx = session_context();
    ctx.register_table("t", provider.clone()).unwrap();

    let raw_scan = provider.scan(&ctx.state(), None, &[], None).await.unwrap();
    let plan = build_nested_approx_plan(raw_scan, Arc::clone(&spec)).unwrap();
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.len(), 1);
    batches.into_iter().next().unwrap()
}

fn service_row_counts(
    batch: &arrow::record_batch::RecordBatch,
    spec: &NestedApproxAggSpec,
    service_name: &str,
) -> (u64, f64) {
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
    let key_col = batch
        .column(schema.index_of("__key_0").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let metric_col = batch
        .column(schema.index_of("__mf_0_count").unwrap())
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    let row = (0..batch.num_rows())
        .find(|&row| level_col.value(row) == 0 && key_col.value(row) == service_name)
        .unwrap();
    (count_col.value(row), metric_col.value(row))
}

#[tokio::test]
async fn count_ignores_missing_deepest_key_values() {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST | STORED);
    builder.add_text_field("endpoint", STRING | FAST | STORED);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let service = schema.get_field("service").unwrap();
    let endpoint = schema.get_field("endpoint").unwrap();

    for endpoint_value in ["a", "a"] {
        let mut doc = TantivyDocument::default();
        doc.add_text(service, "api");
        doc.add_text(endpoint, endpoint_value);
        writer.add_document(doc).unwrap();
    }

    let mut missing_endpoint = TantivyDocument::default();
    missing_endpoint.add_text(service, "api");
    writer.add_document(missing_endpoint).unwrap();
    writer.commit().unwrap();

    let spec = two_level_count_spec();
    let batch = execute_plan(index, Arc::clone(&spec)).await;
    let (structural_count, metric_count) = service_row_counts(&batch, &spec, "api");
    assert_eq!(structural_count, 3);
    assert_eq!(metric_count, 3.0);
}

#[tokio::test]
async fn count_is_not_inflated_by_multivalued_deepest_key() {
    let mut builder = SchemaBuilder::new();
    builder.add_text_field("service", STRING | FAST | STORED);
    builder.add_text_field("endpoint", STRING | FAST | STORED);
    let schema = builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();

    let service = schema.get_field("service").unwrap();
    let endpoint = schema.get_field("endpoint").unwrap();

    let mut multi_endpoint = TantivyDocument::default();
    multi_endpoint.add_text(service, "api");
    multi_endpoint.add_text(endpoint, "a");
    multi_endpoint.add_text(endpoint, "b");
    writer.add_document(multi_endpoint).unwrap();

    let mut single_endpoint = TantivyDocument::default();
    single_endpoint.add_text(service, "api");
    single_endpoint.add_text(endpoint, "c");
    writer.add_document(single_endpoint).unwrap();

    writer.commit().unwrap();

    let spec = two_level_count_spec();
    let batch = execute_plan(index, Arc::clone(&spec)).await;
    let (structural_count, metric_count) = service_row_counts(&batch, &spec, "api");
    assert_eq!(structural_count, 2);
    assert_eq!(metric_count, 2.0);
}
