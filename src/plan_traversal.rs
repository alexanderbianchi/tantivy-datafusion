use std::sync::Arc;

use datafusion::common::Result;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;

use crate::table_provider::FastFieldDataSource;

/// Check if a plan node is a single-child, row-preserving operator
/// that is safe to traverse through during optimization.
///
/// Standard set: `CoalesceBatchesExec`, `CooperativeExec`, `ProjectionExec`.
pub(crate) fn is_transparent_operator(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.as_any().downcast_ref::<CoalesceBatchesExec>().is_some()
        || plan.as_any().downcast_ref::<CooperativeExec>().is_some()
        || plan.as_any().downcast_ref::<ProjectionExec>().is_some()
}

/// Like [`is_transparent_operator`] but also includes `RepartitionExec`.
/// Used between aggregation phases where repartitioning is expected.
pub(crate) fn is_transparent_operator_or_repartition(plan: &Arc<dyn ExecutionPlan>) -> bool {
    is_transparent_operator(plan)
        || plan.as_any().downcast_ref::<RepartitionExec>().is_some()
}

/// Walk down through safe, single-child operators to find a `DataSourceExec`
/// wrapping a specific `DataSource` type `T`.
///
/// Returns `(datasource_ref, datasource_exec_ref, path_of_indices)` where
/// the path records which child index was taken at each step (always 0 for
/// single-child operators).
pub(crate) fn find_data_source<T: 'static>(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(&T, &DataSourceExec, Vec<usize>)> {
    find_data_source_inner::<T>(plan, vec![])
}

fn find_data_source_inner<T: 'static>(
    plan: &Arc<dyn ExecutionPlan>,
    path: Vec<usize>,
) -> Option<(&T, &DataSourceExec, Vec<usize>)> {
    if let Some(dse) = plan.as_any().downcast_ref::<DataSourceExec>() {
        if let Some(ds) = dse.data_source().as_any().downcast_ref::<T>() {
            return Some((ds, dse, path));
        }
        return None;
    }

    // Safe to traverse through single-child, row-preserving operators
    if is_transparent_operator(plan) {
        let children = plan.children();
        if children.len() == 1 {
            let mut new_path = path;
            new_path.push(0);
            return find_data_source_inner::<T>(children[0], new_path);
        }
    }

    None
}

/// Rebuild the plan tree along the recorded path, replacing the leaf with
/// `new_leaf`.
pub(crate) fn rebuild_path(
    root: &Arc<dyn ExecutionPlan>,
    path: &[usize],
    new_leaf: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if path.is_empty() {
        return Ok(new_leaf);
    }

    let children = root.children();
    let child_idx = path[0];
    let new_child = rebuild_path(children[child_idx], &path[1..], new_leaf)?;

    let mut new_children: Vec<Arc<dyn ExecutionPlan>> =
        children.iter().map(|c| Arc::clone(c)).collect();
    new_children[child_idx] = new_child;

    Arc::clone(root).with_new_children(new_children)
}

/// Walk through transparent operators (including `RepartitionExec`) to find
/// a `FastFieldDataSource`.
pub(crate) fn find_fast_field_datasource(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<&FastFieldDataSource> {
    if let Some(dse) = plan.as_any().downcast_ref::<DataSourceExec>() {
        return dse
            .data_source()
            .as_any()
            .downcast_ref::<FastFieldDataSource>();
    }
    if is_transparent_operator_or_repartition(plan) {
        let children = plan.children();
        if children.len() == 1 {
            return find_fast_field_datasource(children[0]);
        }
    }
    None
}

/// Walk through transparent operators (including `RepartitionExec`) between
/// a Final and Partial aggregate to find the Partial `AggregateExec`.
pub(crate) fn find_partial_aggregate(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<&AggregateExec>> {
    if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
        if matches!(agg.mode(), AggregateMode::Partial) {
            return Ok(Some(agg));
        }
        return Ok(None);
    }

    if is_transparent_operator_or_repartition(plan) {
        let children = plan.children();
        if children.len() == 1 {
            return find_partial_aggregate(children[0]);
        }
    }

    Ok(None)
}

/// Find a `DataSourceExec` anywhere through transparent operators
/// (including `RepartitionExec`), returning its `Arc`.
pub(crate) fn find_data_source_exec(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.as_any().downcast_ref::<DataSourceExec>().is_some() {
        return Some(plan.clone());
    }
    if is_transparent_operator_or_repartition(plan) {
        let children = plan.children();
        if children.len() == 1 {
            return find_data_source_exec(children[0]);
        }
    }
    None
}
