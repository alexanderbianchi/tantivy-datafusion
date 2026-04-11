//! Physical aggregation pushdown from DataFusion `AggregateExec` to tantivy.
//!
//! ## Rule Ordering
//!
//! Register [`AggPushdown`] before any distributed physical optimizer rule.
//! For two-phase aggregates, this rule rewrites only the partial side into a
//! partitioned tantivy-native aggregation source and leaves DataFusion's final
//! aggregate in place, which lets distributed planners keep exchange /
//! re-aggregation boundaries above the pushdown. If a distributed optimizer
//! inserts network boundaries first, this rule can no longer see through the
//! aggregate subtree and pushdown will not fire.

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

use crate::unified::agg_data_source::AggDataSource;
use crate::unified::plan_traversal::{find_partial_aggregate, find_single_table_datasource};
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
///
/// Register this rule before distributed physical optimizer rules. Two-phase
/// pushdown preserves DataFusion's final aggregate so distributed execution can
/// still repartition and merge across node boundaries, but the rule still
/// needs to see the original partial-aggregate subtree before network nodes are
/// inserted.
#[derive(Debug)]
pub struct AggPushdown;

impl AggPushdown {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AggPushdown {
    fn default() -> Self {
        Self::new()
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

/// Attempt to replace an `AggregateExec` subtree with a `DataSourceExec(AggDataSource)`.
fn try_rewrite(plan: Arc<dyn ExecutionPlan>) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() else {
        return Ok(Transformed::no(plan));
    };

    // Handle both single-phase and two-phase aggregation patterns.
    match agg.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            Ok(try_rewrite_single(agg, &plan))
        }
        AggregateMode::Final | AggregateMode::FinalPartitioned => try_rewrite_two_phase(agg, &plan),
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
) -> Transformed<Arc<dyn ExecutionPlan>> {
    if !has_group_by(agg) {
        return Transformed::no(plan.clone());
    }
    if agg.filter_expr().iter().any(|expr| expr.is_some()) {
        return Transformed::no(plan.clone());
    }

    let input = agg.input();

    if let Some(st_ds) = find_single_table_datasource(input) {
        if !agg_fields_exist_on_all_splits(agg, st_ds) {
            return Transformed::no(plan.clone());
        }
        if let Some(tantivy_aggs) = derive_tantivy_aggregations(agg).map(Arc::new) {
            let agg_ds = AggDataSource::from_split_openers(
                st_ds.split_openers(),
                tantivy_aggs,
                agg.schema(),
                st_ds.raw_queries().to_vec(),
                st_ds.pre_built_query().cloned(),
                st_ds.fast_field_filter_exprs().to_vec(),
            );
            return Transformed::yes(Arc::new(DataSourceExec::new(Arc::new(agg_ds))));
        }
    }

    Transformed::no(plan.clone())
}

/// Rewrite two-phase: keep `AggregateExec(Final*)` and replace the partial side
/// with a partitioned `AggDataSource` that emits DataFusion-compatible partial
/// aggregate state rows.
fn try_rewrite_two_phase(
    final_agg: &AggregateExec,
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    if !has_group_by(final_agg) {
        return Ok(Transformed::no(plan.clone()));
    }
    if final_agg.filter_expr().iter().any(|expr| expr.is_some()) {
        return Ok(Transformed::no(plan.clone()));
    }

    // Walk through safe operators between Final and Partial
    let partial_agg = find_partial_aggregate(final_agg.input())?;
    let Some(partial_agg) = partial_agg else {
        return Ok(Transformed::no(plan.clone()));
    };
    if partial_agg.filter_expr().iter().any(|expr| expr.is_some()) {
        return Ok(Transformed::no(plan.clone()));
    }

    let partial_input = partial_agg.input();

    if let Some(st_ds) = find_single_table_datasource(partial_input) {
        if !agg_fields_exist_on_all_splits(final_agg, st_ds) {
            return Ok(Transformed::no(plan.clone()));
        }
        if let Some(tantivy_aggs) = derive_tantivy_partial_aggregations(partial_agg).map(Arc::new) {
            let agg_ds = AggDataSource::from_split_openers_partial_states(
                st_ds.split_openers(),
                tantivy_aggs,
                partial_agg.schema(),
                st_ds.raw_queries().to_vec(),
                st_ds.pre_built_query().cloned(),
                st_ds.fast_field_filter_exprs().to_vec(),
            );
            let replacement: Arc<dyn ExecutionPlan> =
                Arc::new(DataSourceExec::new(Arc::new(agg_ds)));
            let rewritten_input =
                replace_partial_aggregate(Arc::clone(final_agg.input()), replacement)?;
            return Ok(Transformed::yes(
                plan.clone().with_new_children(vec![rewritten_input])?,
            ));
        }
    }

    Ok(Transformed::no(plan.clone()))
}

fn replace_partial_aggregate(
    plan: Arc<dyn ExecutionPlan>,
    replacement: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    plan.transform_down(|node| {
        let Some(agg) = node.as_any().downcast_ref::<AggregateExec>() else {
            return Ok(Transformed::no(node));
        };
        if matches!(agg.mode(), AggregateMode::Partial) {
            return Ok(Transformed::yes(Arc::clone(&replacement)));
        }
        Ok(Transformed::no(node))
    })
    .map(|transformed| transformed.data)
}

fn agg_fields_exist_on_all_splits(
    agg: &AggregateExec,
    data_source: &crate::unified::single_table_provider::SingleTableDataSource,
) -> bool {
    let referenced_fields = referenced_agg_fields(agg);
    data_source.split_openers().into_iter().all(|opener| {
        referenced_fields
            .iter()
            .all(|field| opener.schema().get_field(field).is_ok())
    })
}

fn referenced_agg_fields(agg: &AggregateExec) -> Vec<String> {
    let mut fields = Vec::new();

    for (expr, _) in agg.group_expr().expr() {
        if let Some(column) = expr.as_any().downcast_ref::<Column>() {
            fields.push(column.name().to_string());
        }
    }

    for agg_fn in agg.aggr_expr() {
        for expr in agg_fn.expressions() {
            if let Some(column) = expr.as_any().downcast_ref::<Column>() {
                fields.push(column.name().to_string());
            }
        }
    }

    fields.sort();
    fields.dedup();
    fields
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
        if agg_fn.is_distinct() {
            return None;
        }

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
    // Use the maximum safe size that avoids overflow in tantivy's
    // `segment_size = size * 10` default calculation.
    let max_buckets = u32::MAX / 10;
    let terms = TermsAggregation {
        field: group_field,
        size: Some(max_buckets),
        segment_size: Some(max_buckets),
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

fn derive_tantivy_partial_aggregations(agg: &AggregateExec) -> Option<Aggregations> {
    let group_exprs = agg.group_expr();
    if group_exprs.is_empty() || group_exprs.expr().len() != 1 {
        return None;
    }

    let (group_expr, _alias) = &group_exprs.expr()[0];
    let group_col = group_expr.as_any().downcast_ref::<Column>()?;
    let group_field = group_col.name().to_string();

    let mut sub_aggs = Aggregations::default();
    for agg_fn in agg.aggr_expr() {
        if agg_fn.is_distinct() {
            return None;
        }

        let func_name = agg_fn.fun().name();
        let args = agg_fn.expressions();
        let state_fields = agg_fn.state_fields().ok()?;
        let column_arg = args
            .first()
            .and_then(|expr| expr.as_any().downcast_ref::<Column>());

        match func_name {
            "count" => {
                if let Some(col) = column_arg {
                    insert_state_sub_agg(
                        &mut sub_aggs,
                        state_fields.first()?.name().to_string(),
                        AggregationVariants::Count(CountAggregation::from_field_name(
                            col.name().to_string(),
                        )),
                    );
                }
            }
            "sum" => insert_state_sub_agg(
                &mut sub_aggs,
                state_fields.first()?.name().to_string(),
                AggregationVariants::Sum(SumAggregation::from_field_name(
                    column_arg?.name().to_string(),
                )),
            ),
            "min" => insert_state_sub_agg(
                &mut sub_aggs,
                state_fields.first()?.name().to_string(),
                AggregationVariants::Min(MinAggregation::from_field_name(
                    column_arg?.name().to_string(),
                )),
            ),
            "max" => insert_state_sub_agg(
                &mut sub_aggs,
                state_fields.first()?.name().to_string(),
                AggregationVariants::Max(MaxAggregation::from_field_name(
                    column_arg?.name().to_string(),
                )),
            ),
            "avg" => {
                let col = column_arg?;
                let [count_field, sum_field] = state_fields.as_slice() else {
                    return None;
                };
                insert_state_sub_agg(
                    &mut sub_aggs,
                    count_field.name().to_string(),
                    AggregationVariants::Count(CountAggregation::from_field_name(
                        col.name().to_string(),
                    )),
                );
                insert_state_sub_agg(
                    &mut sub_aggs,
                    sum_field.name().to_string(),
                    AggregationVariants::Sum(SumAggregation::from_field_name(
                        col.name().to_string(),
                    )),
                );
            }
            _ => return None,
        }
    }

    let max_buckets = u32::MAX / 10;
    let terms = TermsAggregation {
        field: group_field,
        size: Some(max_buckets),
        segment_size: Some(max_buckets),
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

fn insert_state_sub_agg(sub_aggs: &mut Aggregations, name: String, variant: AggregationVariants) {
    sub_aggs.insert(
        name,
        Aggregation {
            agg: variant,
            sub_aggregation: Default::default(),
        },
    );
}
