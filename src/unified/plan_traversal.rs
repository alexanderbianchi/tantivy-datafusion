use std::sync::Arc;

use datafusion::common::Result;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;

use crate::unified::single_table_provider::SingleTableDataSource;

/// Like [`is_transparent_operator`] but also includes `RepartitionExec`.
/// Used between aggregation phases where repartitioning is expected.
pub(crate) fn is_transparent_operator_or_repartition(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.as_any()
        .downcast_ref::<CoalesceBatchesExec>()
        .is_some()
        || plan
            .as_any()
            .downcast_ref::<CoalescePartitionsExec>()
            .is_some()
        || plan.as_any().downcast_ref::<CooperativeExec>().is_some()
        || plan.as_any().downcast_ref::<ProjectionExec>().is_some()
        || plan.as_any().downcast_ref::<RepartitionExec>().is_some()
}

/// Walk through transparent operators to find a `SingleTableDataSource`.
pub(crate) fn find_single_table_datasource(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<&SingleTableDataSource> {
    if let Some(dse) = plan.as_any().downcast_ref::<DataSourceExec>() {
        return dse
            .data_source()
            .as_any()
            .downcast_ref::<SingleTableDataSource>();
    }
    if is_transparent_operator_or_repartition(plan) {
        let children = plan.children();
        if children.len() == 1 {
            return find_single_table_datasource(children[0]);
        }
    }
    None
}

/// Walk through transparent operators between a Final and Partial aggregate
/// to find the Partial `AggregateExec`.
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
