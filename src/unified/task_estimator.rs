use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;

use crate::unified::single_table_provider::SingleTableDataSource;
use crate::decomposed::table_provider::FastFieldDataSource;

/// Estimate the number of tasks (distributed workers) needed for a tantivy scan.
///
/// Returns the partition count of the leaf DataSource, which equals the number
/// of tantivy segments. Each segment can be executed independently on a separate
/// worker node.
///
/// Note: This is a standalone function rather than implementing a trait, because
/// we don't have access to `dd-datafusion-runtime`'s `TaskEstimator` trait in
/// the OSS crate. When integrating with the DD stack, this function would be
/// wrapped in a `TaskEstimator` impl.
pub fn estimate_task_count(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    // Check leaf nodes for tantivy data sources
    if let Some(ds_exec) = plan.as_any().downcast_ref::<DataSourceExec>() {
        let ds = ds_exec.data_source();
        if let Some(st) = ds.as_any().downcast_ref::<SingleTableDataSource>() {
            return Some(st.num_segments());
        }
        if let Some(ff) = ds.as_any().downcast_ref::<FastFieldDataSource>() {
            return Some(ff.num_partitions());
        }
    }

    // Recurse into children
    for child in plan.children() {
        if let Some(count) = estimate_task_count(child) {
            return Some(count);
        }
    }
    None
}
