//! Plan builder for nested approximate aggregations.
//!
//! Pushdown uses the underlying [`SingleTableDataSource`] directly. Fallback
//! uses a split-local normalized projection directly over a split-partitioned
//! tantivy scan. The fallback child must expose:
//!
//! - `__na_key_0 .. __na_key_n`
//! - `__na_metric_0 .. __na_metric_m` for non-`Count` metrics
//! - `_segment_ord`

use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;

use super::exec::NestedApproxAggExec;
use super::node_table::node_table_partial_schema;
use super::spec::{BucketKind, MetricSpec, NestedApproxAggSpec};
use crate::unified::agg_data_source::AggDataSource;
use crate::unified::plan_traversal::find_single_table_datasource;
use crate::unified::single_table_provider::SingleTableDataSource;

/// Build a physical plan for a nested approximate aggregation.
///
/// ## Pushdown input
///
/// Pass a raw scan (or other transparent wrapper around a raw scan) when every
/// bucket level and metric input is storage-native.
///
/// ## Fallback input
///
/// Pass a split-local child that already projects the normalized fallback
/// aliases listed in the module-level docs directly over a split-partitioned
/// `SingleTableDataSource` scan when any bucket key or metric input requires
/// DataFusion expression evaluation.
pub fn build_nested_approx_plan(
    input: Arc<dyn ExecutionPlan>,
    spec: Arc<NestedApproxAggSpec>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if input_has_normalized_projection(input.as_ref(), &spec) {
        return build_fallback_plan(input, spec);
    }

    if plan_contains_projection(input.as_ref()) {
        return Err(DataFusionError::Plan(
            "non-pushdown nested agg input must expose normalized fallback aliases \
             (__na_key_*, __na_metric_*, _segment_ord)"
                .into(),
        ));
    }

    build_pushdown_plan(input, spec)
}

fn build_pushdown_plan(
    input: Arc<dyn ExecutionPlan>,
    spec: Arc<NestedApproxAggSpec>,
) -> Result<Arc<dyn ExecutionPlan>> {
    ensure_pushdownable(&spec)?;
    let scan_info = find_single_table_datasource(&input).ok_or_else(|| {
        DataFusionError::Plan(
            "nested agg pushdown requires a SingleTableDataSource input".into(),
        )
    })?;

    let tantivy_aggs = Arc::new(spec.to_tantivy_aggregations());
    let output_schema = node_table_partial_schema(&spec);

    let agg_ds = AggDataSource::from_split_descriptors_node_table_partial_with_runtime_factory(
        scan_info.split_descriptors(),
        tantivy_aggs,
        output_schema,
        scan_info.raw_queries().to_vec(),
        scan_info.pre_built_query().cloned(),
        scan_info.fast_field_filter_exprs().to_vec(),
        Arc::clone(&spec),
        scan_info.local_runtime_factory(),
    );

    let leaf: Arc<dyn ExecutionPlan> = Arc::new(DataSourceExec::new(Arc::new(agg_ds)));
    let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(leaf));
    Ok(Arc::new(NestedApproxAggExec::new_final_merge(coalesced, spec)))
}

fn build_fallback_plan(
    input: Arc<dyn ExecutionPlan>,
    spec: Arc<NestedApproxAggSpec>,
) -> Result<Arc<dyn ExecutionPlan>> {
    ensure_fallback_input_contract(input.as_ref(), &spec)?;
    let partial: Arc<dyn ExecutionPlan> =
        Arc::new(NestedApproxAggExec::new_partial_split_local(input, Arc::clone(&spec)));
    let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(partial));
    Ok(Arc::new(NestedApproxAggExec::new_final_merge(coalesced, spec)))
}

fn ensure_pushdownable(spec: &NestedApproxAggSpec) -> Result<()> {
    for (level_idx, level) in spec.levels.iter().enumerate() {
        if !matches!(level.kind, BucketKind::Terms | BucketKind::DateHistogram { .. }) {
            return Err(DataFusionError::Plan(format!(
                "nested agg level {level_idx} is not pushdownable: {:?}",
                level.kind
            )));
        }
        if level.field.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "nested agg level {level_idx} is missing its pushdown field"
            )));
        }
    }

    for (metric_idx, metric) in spec.metrics.iter().enumerate() {
        match metric {
            MetricSpec::Count => {}
            MetricSpec::Sum { field }
            | MetricSpec::Avg { field }
            | MetricSpec::Min { field }
            | MetricSpec::Max { field } if !field.is_empty() => {}
            _ => {
                return Err(DataFusionError::Plan(format!(
                    "nested agg metric {metric_idx} is missing its pushdown field"
                )))
            }
        }
    }

    Ok(())
}

fn ensure_fallback_input_contract(
    input: &dyn ExecutionPlan,
    spec: &NestedApproxAggSpec,
) -> Result<()> {
    ensure_fallback_child_shape(input)?;

    let schema = input.schema();
    schema.index_of("_segment_ord").map_err(|_| {
        DataFusionError::Plan(
            "fallback nested agg input must include the hidden _segment_ord column".into(),
        )
    })?;

    for level_idx in 0..spec.levels.len() {
        let name = NestedApproxAggSpec::normalized_key_column_name(level_idx);
        schema.index_of(&name).map_err(|_| {
            DataFusionError::Plan(format!(
                "fallback nested agg input is missing normalized key column {name}"
            ))
        })?;
    }

    for (metric_idx, metric) in spec.metrics.iter().enumerate() {
        if matches!(metric, MetricSpec::Count) {
            continue;
        }
        let name = NestedApproxAggSpec::normalized_metric_column_name(metric_idx);
        let idx = schema.index_of(&name).map_err(|_| {
            DataFusionError::Plan(format!(
                "fallback nested agg input is missing normalized metric column {name}"
            ))
        })?;
        let data_type = schema.field(idx).data_type();
        if !is_supported_normalized_metric_type(data_type) {
            return Err(DataFusionError::Plan(format!(
                "fallback nested agg metric column {name} must be Float64, List<Float64>, \
                 or LargeList<Float64>; got {data_type:?}"
            )));
        }
    }

    Ok(())
}

fn ensure_fallback_child_shape(input: &dyn ExecutionPlan) -> Result<()> {
    let mut saw_projection = false;
    let mut plan = input;

    loop {
        if let Some(data_source) = plan.as_any().downcast_ref::<DataSourceExec>() {
            if !saw_projection {
                return Err(DataFusionError::Plan(
                    "fallback nested agg input must be a normalized ProjectionExec over \
                     SingleTableDataSource"
                        .into(),
                ));
            }

            if data_source
                .data_source()
                .as_any()
                .downcast_ref::<SingleTableDataSource>()
                .is_none()
            {
                return Err(DataFusionError::Plan(
                    "fallback nested agg input must sit directly over SingleTableDataSource"
                        .into(),
                ));
            }

            return Ok(());
        }

        if plan.as_any().downcast_ref::<RepartitionExec>().is_some() {
            return Err(DataFusionError::Plan(
                "fallback nested agg input must remain split-local; repartition/exchange \
                 between the tantivy scan and PartialSplitLocal is not allowed"
                    .into(),
            ));
        }

        let children = plan.children();
        if children.len() != 1 {
            return Err(DataFusionError::Plan(
                "fallback nested agg input must be a unary normalized projection over \
                 SingleTableDataSource"
                    .into(),
            ));
        }

        if plan.as_any().downcast_ref::<ProjectionExec>().is_some() {
            if saw_projection {
                return Err(DataFusionError::Plan(
                    "fallback nested agg input must use a single normalized ProjectionExec over \
                     SingleTableDataSource"
                        .into(),
                ));
            }
            saw_projection = true;
            plan = children[0].as_ref();
            continue;
        }

        if plan.as_any().downcast_ref::<CooperativeExec>().is_some()
            || plan.as_any().downcast_ref::<CoalesceBatchesExec>().is_some()
        {
            plan = children[0].as_ref();
            continue;
        }

        if plan.as_any().downcast_ref::<CoalescePartitionsExec>().is_some() {
            return Err(DataFusionError::Plan(
                "fallback nested agg input must remain split-local; partition coalescing between \
                 the tantivy scan and PartialSplitLocal is not allowed"
                    .into(),
            ));
        }

        return Err(DataFusionError::Plan(format!(
            "fallback nested agg input must be a normalized ProjectionExec over \
             SingleTableDataSource, found {}",
            plan.name()
        )));
    }
}

fn is_supported_normalized_metric_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float64)
        || matches!(data_type, DataType::List(field) if field.data_type() == &DataType::Float64)
        || matches!(data_type, DataType::LargeList(field) if field.data_type() == &DataType::Float64)
}

fn input_has_normalized_projection(input: &dyn ExecutionPlan, spec: &NestedApproxAggSpec) -> bool {
    let schema = input.schema();
    schema.index_of("_segment_ord").is_ok()
        && (0..spec.levels.len()).all(|level_idx| {
            schema
                .index_of(&NestedApproxAggSpec::normalized_key_column_name(level_idx))
                .is_ok()
        })
        && spec.metrics.iter().enumerate().all(|(metric_idx, metric)| {
            matches!(metric, MetricSpec::Count)
                || schema
                    .index_of(&NestedApproxAggSpec::normalized_metric_column_name(metric_idx))
                    .is_ok()
        })
}

fn plan_contains_projection(plan: &dyn ExecutionPlan) -> bool {
    if plan.as_any().downcast_ref::<ProjectionExec>().is_some() {
        return true;
    }
    let children = plan.children();
    children.len() == 1 && plan_contains_projection(children[0].as_ref())
}
