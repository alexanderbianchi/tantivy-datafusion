use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::Result;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_optimizer::PhysicalOptimizerRule;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use tantivy::aggregation::agg_req::{Aggregation, AggregationVariants, Aggregations};
use tantivy::aggregation::bucket::TermsAggregation;
use tantivy::aggregation::metric::{
    AverageAggregation, CountAggregation, MaxAggregation, MinAggregation, SumAggregation,
};

use crate::unified::agg_exec::TantivyAggregateExec;
use crate::unified::plan_traversal::{
    find_fast_field_datasource, find_partial_aggregate, find_single_table_datasource,
};
use datafusion_datasource::source::DataSourceExec;

/// A physical optimizer rule that replaces DataFusion's `AggregateExec`
/// with tantivy's native `AggregationSegmentCollector` when the
/// `FastFieldDataSource` has tantivy `Aggregations` stashed.
///
/// This eliminates the overhead of DataFusion's hash-based GROUP BY and
/// Arrow materialization, achieving near-native tantivy aggregation
/// performance.
///
/// The rule only fires for **bucket aggregations** (terms, histogram, range)
/// where the hash GROUP BY overhead is significant. Simple metric-only
/// aggregations (avg, stats, count) are left to DataFusion's optimized
/// vectorized Arrow path, which is already efficient for single-pass scans.
///
/// The rule only fires when `FastFieldDataSource.aggregations` is `Some`,
/// which is set by `execute_aggregations`. Regular SQL queries (without
/// `execute_aggregations`) are unaffected.
#[derive(Debug)]
pub struct AggPushdown;

impl AggPushdown {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for AggPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(try_rewrite).map(|t| t.data)
    }

    fn name(&self) -> &str {
        "AggPushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Attempt to replace an `AggregateExec` subtree with `TantivyAggregateExec`.
fn try_rewrite(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() else {
        return Ok(Transformed::no(plan));
    };

    // Handle both single-phase and two-phase aggregation patterns.
    match agg.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            try_rewrite_single(agg, &plan)
        }
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            try_rewrite_two_phase(agg, &plan)
        }
        AggregateMode::Partial => {
            // Partial on its own — not the top-level, skip
            Ok(Transformed::no(plan))
        }
    }
}

/// Check if the aggregate has GROUP BY expressions (bucket aggregation).
/// We only push down bucket aggs; metric-only aggs (no GROUP BY) are
/// faster via DataFusion's native vectorized Arrow path.
fn has_group_by(agg: &AggregateExec) -> bool {
    !agg.group_expr().is_empty()
}

/// Rewrite single-phase: AggregateExec(Single) → [safe ops] → DataSourceExec.
fn try_rewrite_single(
    agg: &AggregateExec,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if !has_group_by(agg) {
        return Ok(Transformed::no(plan.clone()));
    }

    let input = agg.input();

    // Try FastFieldDataSource first
    if let Some(ff_ds) = find_fast_field_datasource(input) {
        // Check stashed aggregations (backward compat)
        if let Some(tantivy_aggs) = ff_ds.aggregations() {
            let new_exec = TantivyAggregateExec::new(
                ff_ds.opener().clone(),
                tantivy_aggs.clone(),
                ff_ds.query().map(|q| Arc::from(q.box_clone())),
                agg.schema(),
            );
            return Ok(Transformed::yes(Arc::new(new_exec)));
        }

        // Fallback: derive from AggregateExec expressions
        if let Some(tantivy_aggs) = derive_tantivy_aggregations(agg) {
            let new_exec = TantivyAggregateExec::new(
                ff_ds.opener().clone(),
                Arc::new(tantivy_aggs),
                ff_ds.query().map(|q| Arc::from(q.box_clone())),
                agg.schema(),
            );
            return Ok(Transformed::yes(Arc::new(new_exec)));
        }
    }

    // Try SingleTableDataSource: push aggs into the data source
    if let Some(st_ds) = find_single_table_datasource(input) {
        // Check stashed aggregations first
        if let Some(tantivy_aggs) = st_ds.aggregations() {
            let updated = st_ds.with_agg_mode(tantivy_aggs.clone(), agg.schema());
            return Ok(Transformed::yes(Arc::new(DataSourceExec::new(Arc::new(updated)))));
        }

        // Fallback: derive from AggregateExec expressions
        if let Some(tantivy_aggs) = derive_tantivy_aggregations(agg) {
            let updated = st_ds.with_agg_mode(Arc::new(tantivy_aggs), agg.schema());
            return Ok(Transformed::yes(Arc::new(DataSourceExec::new(Arc::new(updated)))));
        }
    }

    Ok(Transformed::no(plan.clone()))
}

/// Rewrite two-phase: AggregateExec(Final) → ... → AggregateExec(Partial) → [safe ops] → DataSourceExec.
fn try_rewrite_two_phase(
    _final_agg: &AggregateExec,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let final_agg = plan.as_any().downcast_ref::<AggregateExec>().unwrap();

    if !has_group_by(final_agg) {
        return Ok(Transformed::no(plan.clone()));
    }

    // Walk through safe operators between Final and Partial
    let partial_agg = find_partial_aggregate(final_agg.input())?;
    let Some(partial_agg) = partial_agg else {
        return Ok(Transformed::no(plan.clone()));
    };

    let partial_input = partial_agg.input();

    // Try FastFieldDataSource first
    if let Some(ff_ds) = find_fast_field_datasource(partial_input) {
        // Check stashed aggregations (backward compat)
        if let Some(tantivy_aggs) = ff_ds.aggregations() {
            let new_exec = TantivyAggregateExec::new(
                ff_ds.opener().clone(),
                tantivy_aggs.clone(),
                ff_ds.query().map(|q| Arc::from(q.box_clone())),
                final_agg.schema(),
            );
            return Ok(Transformed::yes(Arc::new(new_exec)));
        }

        // Fallback: derive from the Final AggregateExec expressions
        if let Some(tantivy_aggs) = derive_tantivy_aggregations(final_agg) {
            let new_exec = TantivyAggregateExec::new(
                ff_ds.opener().clone(),
                Arc::new(tantivy_aggs),
                ff_ds.query().map(|q| Arc::from(q.box_clone())),
                final_agg.schema(),
            );
            return Ok(Transformed::yes(Arc::new(new_exec)));
        }
    }

    // Try SingleTableDataSource: push aggs into the data source
    if let Some(st_ds) = find_single_table_datasource(partial_input) {
        // Check stashed aggregations first
        if let Some(tantivy_aggs) = st_ds.aggregations() {
            let updated = st_ds.with_agg_mode(tantivy_aggs.clone(), final_agg.schema());
            return Ok(Transformed::yes(Arc::new(DataSourceExec::new(Arc::new(updated)))));
        }

        // Fallback: derive from the Final AggregateExec expressions
        if let Some(tantivy_aggs) = derive_tantivy_aggregations(final_agg) {
            let updated = st_ds.with_agg_mode(Arc::new(tantivy_aggs), final_agg.schema());
            return Ok(Transformed::yes(Arc::new(DataSourceExec::new(Arc::new(updated)))));
        }
    }

    Ok(Transformed::no(plan.clone()))
}

// ---------------------------------------------------------------------------
// Derive tantivy Aggregations from DataFusion's AggregateExec
// ---------------------------------------------------------------------------

/// Try to derive tantivy `Aggregations` from an `AggregateExec`'s group-by
/// and aggregate expressions. Returns `None` if the expressions cannot be
/// mapped to supported tantivy aggregations.
///
/// This enables the BYOC/Substrait path where `AggregateExec` is built by
/// DataFusion's Substrait consumer and no pre-stashed aggregations exist.
fn derive_tantivy_aggregations(agg: &AggregateExec) -> Option<Aggregations> {
    let group_exprs = agg.group_expr();
    if group_exprs.is_empty() {
        // Metric-only aggregation (no GROUP BY) — not pushed down by this rule
        return None;
    }

    // For bucket aggregations, we only support a single GROUP BY column
    // (maps to a tantivy TermsAggregation). Multi-column GROUP BY is not
    // supported by tantivy.
    if group_exprs.expr().len() != 1 {
        return None;
    }

    let (group_expr, _alias) = &group_exprs.expr()[0];
    let group_col = group_expr.as_any().downcast_ref::<Column>()?;
    let group_field = group_col.name().to_string();

    // Build sub-aggregations from the aggregate expressions
    let mut sub_aggs = Aggregations::default();
    for agg_fn in agg.aggr_expr() {
        let func_name = agg_fn.fun().name();
        let args = agg_fn.expressions();

        // Get the field name from the first argument.
        // COUNT(*) becomes COUNT(1) with a Literal expression, not a Column.
        // Skip it — tantivy's TermsAggregation includes doc_count automatically.
        let field_name = if let Some(col) = args
            .first()
            .and_then(|e| e.as_any().downcast_ref::<Column>())
        {
            col.name().to_string()
        } else {
            // COUNT(1) / COUNT(*) or other non-column expression — skip
            continue;
        };

        let agg_name = agg_fn.name().to_string();
        let variant = match func_name {
            "sum" => AggregationVariants::Sum(SumAggregation::from_field_name(field_name)),
            "avg" => AggregationVariants::Average(AverageAggregation::from_field_name(field_name)),
            "min" => AggregationVariants::Min(MinAggregation::from_field_name(field_name)),
            "max" => AggregationVariants::Max(MaxAggregation::from_field_name(field_name)),
            "count" => AggregationVariants::Count(CountAggregation::from_field_name(field_name)),
            _ => return None, // Unsupported aggregate function
        };

        sub_aggs.insert(
            agg_name,
            Aggregation {
                agg: variant,
                sub_aggregation: Default::default(),
            },
        );
    }

    // Build the top-level terms aggregation
    let terms = TermsAggregation {
        field: group_field,
        size: Some(65535), // Return all terms by default
        ..Default::default()
    };

    let mut aggs = Aggregations::default();
    aggs.insert(
        "group".to_string(),
        Aggregation {
            agg: AggregationVariants::Terms(terms),
            sub_aggregation: sub_aggs,
        },
    );

    Some(aggs)
}

