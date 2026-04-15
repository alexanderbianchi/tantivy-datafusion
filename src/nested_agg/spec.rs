//! Specification types for nested approximate top-k aggregations.
//!
//! A [`NestedApproxAggSpec`] describes a tree of bucket levels with metric
//! state carried on every retained bucket node and explicit overscan
//! (`fanout`) per level. The spec is the single source of truth for:
//!
//! - converting to tantivy [`Aggregations`] for pushdown
//! - deriving node-table Arrow schemas
//! - driving the final merge and trim algorithm

use serde::{Deserialize, Serialize};
use tantivy::aggregation::agg_req::{Aggregation, AggregationVariants, Aggregations};
use tantivy::aggregation::bucket::{DateHistogramAggregationReq, TermsAggregation};
use tantivy::aggregation::metric::{
    AverageAggregation, MaxAggregation, MinAggregation, SumAggregation,
};

/// Top-level specification for a nested approximate aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedApproxAggSpec {
    /// Ordered bucket levels from outermost to innermost.
    pub levels: Vec<BucketLevelSpec>,
    /// Metrics tracked for every retained bucket node in the tree.
    pub metrics: Vec<MetricSpec>,
}

/// Specification for a single bucket level in the nesting tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketLevelSpec {
    /// The kind of bucket aggregation.
    pub kind: BucketKind,
    /// The tantivy field name to aggregate on.
    pub field: String,
    /// Number of buckets to return in the final output.
    pub final_size: u32,
    /// Number of candidate buckets to retain per segment/split before final
    /// merge. Must be >= `final_size`. Larger values improve approximation
    /// quality at the cost of more intermediate data.
    pub fanout: u32,
}

/// The kind of bucket aggregation at a single level.
///
/// Range aggregations are intentionally excluded from v1. Tantivy's
/// `IntermediateRangeBucketResult` does not expose its buckets publicly,
/// so the conversion to node-table format is not possible without a
/// tantivy fork change. Add `Range` here once the accessor is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BucketKind {
    Terms,
    DateHistogram {
        /// Fixed interval string, e.g. `"1d"`, `"1h"`.
        fixed_interval: String,
    },
}

/// A metric specification merged onto every retained bucket node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricSpec {
    Count,
    Sum { field: String },
    Avg { field: String },
    Min { field: String },
    Max { field: String },
}

// ---------------------------------------------------------------------------
// State-field helpers
// ---------------------------------------------------------------------------

impl MetricSpec {
    /// Number of intermediate state values this metric produces.
    pub fn state_field_count(&self) -> usize {
        match self {
            // Bucket document count already travels in the structural `__count`
            // column, so `Count` carries no separate partial state.
            MetricSpec::Count => 0,
            MetricSpec::Sum { .. } => 1,
            MetricSpec::Min { .. } => 1,
            MetricSpec::Max { .. } => 1,
            MetricSpec::Avg { .. } => 2, // count + sum
        }
    }

    /// Names for intermediate state columns, indexed by metric position.
    pub fn state_field_names(&self, metric_idx: usize) -> Vec<String> {
        match self {
            MetricSpec::Count => Vec::new(),
            MetricSpec::Sum { .. } => vec![format!("__ms_{metric_idx}_sum")],
            MetricSpec::Min { .. } => vec![format!("__ms_{metric_idx}_min")],
            MetricSpec::Max { .. } => vec![format!("__ms_{metric_idx}_max")],
            MetricSpec::Avg { .. } => vec![
                format!("__ms_{metric_idx}_avg_count"),
                format!("__ms_{metric_idx}_avg_sum"),
            ],
        }
    }

    /// Name for the finalized output column.
    pub fn final_field_name(&self, metric_idx: usize) -> String {
        match self {
            // The finalized Count metric is derived from the structural
            // `__count` column, but we still expose a metric column so the
            // output shape tracks `spec.metrics`.
            MetricSpec::Count => format!("__mf_{metric_idx}_count"),
            MetricSpec::Sum { .. } => format!("__mf_{metric_idx}_sum"),
            MetricSpec::Min { .. } => format!("__mf_{metric_idx}_min"),
            MetricSpec::Max { .. } => format!("__mf_{metric_idx}_max"),
            MetricSpec::Avg { .. } => format!("__mf_{metric_idx}_avg"),
        }
    }
}

impl NestedApproxAggSpec {
    /// Total number of intermediate metric state columns across all metrics.
    pub fn total_metric_state_fields(&self) -> usize {
        self.metrics.iter().map(|m| m.state_field_count()).sum()
    }

    /// All intermediate metric state column names in order.
    pub fn all_state_field_names(&self) -> Vec<String> {
        self.metrics
            .iter()
            .enumerate()
            .flat_map(|(idx, m)| m.state_field_names(idx))
            .collect()
    }

    /// All finalized metric column names in order.
    pub fn all_final_field_names(&self) -> Vec<String> {
        self.metrics
            .iter()
            .enumerate()
            .map(|(idx, m)| m.final_field_name(idx))
            .collect()
    }

    /// Key column name for a given level index.
    pub fn key_column_name(level: usize) -> String {
        format!("__key_{level}")
    }

    /// Normalized projection alias for a fallback key column.
    pub fn normalized_key_column_name(level: usize) -> String {
        format!("__na_key_{level}")
    }

    /// Normalized projection alias for a fallback metric column.
    pub fn normalized_metric_column_name(metric_idx: usize) -> String {
        format!("__na_metric_{metric_idx}")
    }
}

// ---------------------------------------------------------------------------
// Conversion to tantivy Aggregations
// ---------------------------------------------------------------------------

impl NestedApproxAggSpec {
    /// Convert this spec into a tantivy [`Aggregations`] tree suitable for
    /// pushdown execution via `DistributedAggregationCollector`.
    ///
    /// Each level's `fanout` maps to tantivy's `size` and `segment_size` so
    /// that the leaf retains enough candidates for the final merge to trim.
    pub fn to_tantivy_aggregations(&self) -> Aggregations {
        self.build_level_agg(0)
    }

    fn build_level_agg(&self, level_idx: usize) -> Aggregations {
        if level_idx >= self.levels.len() {
            return self.build_metric_sub_aggs();
        }

        let level = &self.levels[level_idx];
        // Each bucket level gets the next bucket level AND metric sub-aggs
        // so that every bucket in the tree carries its own metric state.
        let mut sub_aggregation = self.build_level_agg(level_idx + 1);
        for (name, agg) in self.build_metric_sub_aggs() {
            sub_aggregation.entry(name).or_insert(agg);
        }

        let variant = match &level.kind {
            BucketKind::Terms => AggregationVariants::Terms(TermsAggregation {
                field: level.field.clone(),
                size: Some(level.fanout),
                segment_size: Some(level.fanout),
                ..Default::default()
            }),
            BucketKind::DateHistogram { fixed_interval } => {
                AggregationVariants::DateHistogram(DateHistogramAggregationReq {
                    field: level.field.clone(),
                    fixed_interval: Some(fixed_interval.clone()),
                    ..Default::default()
                })
            }
        };

        let mut aggs = Aggregations::default();
        aggs.insert(
            format!("level_{level_idx}"),
            Aggregation {
                agg: variant,
                sub_aggregation,
            },
        );
        aggs
    }

    fn build_metric_sub_aggs(&self) -> Aggregations {
        let mut aggs = Aggregations::default();
        for (idx, metric) in self.metrics.iter().enumerate() {
            if matches!(metric, MetricSpec::Count) {
                continue;
            }
            let name = format!("metric_{idx}");
            let variant = match metric {
                MetricSpec::Sum { field } => {
                    AggregationVariants::Sum(SumAggregation::from_field_name(field.clone()))
                }
                MetricSpec::Avg { field } => {
                    AggregationVariants::Average(AverageAggregation::from_field_name(field.clone()))
                }
                MetricSpec::Min { field } => {
                    AggregationVariants::Min(MinAggregation::from_field_name(field.clone()))
                }
                MetricSpec::Max { field } => {
                    AggregationVariants::Max(MaxAggregation::from_field_name(field.clone()))
                }
                MetricSpec::Count => unreachable!("Count is structural and not pushed down"),
            };
            aggs.insert(
                name,
                Aggregation {
                    agg: variant,
                    sub_aggregation: Default::default(),
                },
            );
        }
        aggs
    }

    /// Collect all tantivy field names referenced by this spec (for warmup).
    pub fn referenced_field_names(&self) -> Vec<String> {
        let mut fields: Vec<String> = self.levels.iter().map(|l| l.field.clone()).collect();
        for metric in &self.metrics {
            match metric {
                MetricSpec::Count => {}
                MetricSpec::Sum { field }
                | MetricSpec::Avg { field }
                | MetricSpec::Min { field }
                | MetricSpec::Max { field } => {
                    fields.push(field.clone());
                }
            }
        }
        fields.sort();
        fields.dedup();
        fields
    }
}

impl NestedApproxAggSpec {
    /// Create a new spec.
    ///
    /// Returns an error if any level has `fanout < final_size`, which would
    /// silently degrade approximation quality.
    pub fn try_new(
        levels: Vec<BucketLevelSpec>,
        metrics: Vec<MetricSpec>,
    ) -> datafusion::common::Result<Self> {
        for (i, level) in levels.iter().enumerate() {
            if level.fanout < level.final_size {
                return Err(datafusion::error::DataFusionError::Plan(format!(
                    "nested agg level {i} ({:?}): fanout ({}) must be >= final_size ({})",
                    level.field, level.fanout, level.final_size,
                )));
            }
        }
        Ok(Self { levels, metrics })
    }

    pub(crate) fn parse_fixed_interval_millis(interval: &str) -> datafusion::common::Result<i64> {
        use datafusion::error::DataFusionError;

        let split_at = interval
            .find(|ch: char| !ch.is_ascii_digit())
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "fixed_interval '{interval}' is missing a time unit"
                ))
            })?;
        let (count_str, unit) = interval.split_at(split_at);
        if count_str.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "fixed_interval '{interval}' is missing the numeric component"
            )));
        }
        let count: i64 = count_str.parse().map_err(|err| {
            DataFusionError::Plan(format!("invalid fixed_interval '{interval}': {err}"))
        })?;
        let multiplier = match unit {
            "ms" => 1_i64,
            "s" => 1_000,
            "m" => 60_000,
            "h" => 3_600_000,
            "d" => 86_400_000,
            _ => {
                return Err(DataFusionError::Plan(format!(
                    "unsupported fixed_interval unit '{unit}' in '{interval}'"
                )))
            }
        };
        count.checked_mul(multiplier).ok_or_else(|| {
            DataFusionError::Plan(format!("fixed_interval '{interval}' overflows i64 milliseconds"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_to_tantivy_aggs_two_level_terms() {
        let spec = NestedApproxAggSpec::try_new(
            vec![
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "service".into(),
                    final_size: 50,
                    fanout: 200,
                },
                BucketLevelSpec {
                    kind: BucketKind::Terms,
                    field: "endpoint".into(),
                    final_size: 20,
                    fanout: 80,
                },
            ],
            vec![MetricSpec::Count, MetricSpec::Avg { field: "latency".into() }],
        )
        .unwrap();

        let aggs = spec.to_tantivy_aggregations();
        // Top level should have "level_0"
        assert!(aggs.get("level_0").is_some());
        let level_0 = aggs.get("level_0").unwrap();
        match &level_0.agg {
            AggregationVariants::Terms(t) => {
                assert_eq!(t.field, "service");
                assert_eq!(t.size, Some(200));
                assert_eq!(t.segment_size, Some(200));
            }
            _ => panic!("expected terms"),
        }
        // Sub-agg should have "level_1"
        let level_1 = level_0.sub_aggregation.get("level_1").unwrap();
        match &level_1.agg {
            AggregationVariants::Terms(t) => {
                assert_eq!(t.field, "endpoint");
                assert_eq!(t.size, Some(80));
            }
            _ => panic!("expected terms"),
        }
        // Deepest sub-agg should have metrics
        assert!(
            level_1.sub_aggregation.get("metric_0").is_none(),
            "Count is structural and should not be pushed down"
        );
        assert!(level_1.sub_aggregation.get("metric_1").is_some());
    }

    #[test]
    fn metric_state_field_counts() {
        assert_eq!(MetricSpec::Count.state_field_count(), 0);
        assert_eq!(
            MetricSpec::Avg { field: "x".into() }.state_field_count(),
            2
        );
        assert_eq!(
            MetricSpec::Sum { field: "x".into() }.state_field_count(),
            1
        );
    }

    #[test]
    fn referenced_fields() {
        let spec = NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![
                MetricSpec::Count,
                MetricSpec::Sum { field: "latency".into() },
            ],
        )
        .unwrap();
        let fields = spec.referenced_field_names();
        assert_eq!(fields, vec!["latency", "service"]);
    }

    #[test]
    fn count_is_not_pushed_down() {
        let spec = NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![MetricSpec::Count, MetricSpec::Avg { field: "latency".into() }],
        )
        .unwrap();

        let aggs = spec.to_tantivy_aggregations();
        let level_0 = aggs.get("level_0").unwrap();
        assert!(
            level_0.sub_aggregation.get("metric_0").is_none(),
            "Count must not be emitted as a tantivy sub-aggregation"
        );
        assert!(level_0.sub_aggregation.get("metric_1").is_some());
    }

    #[test]
    fn parse_fixed_interval_millis_supports_supported_units() {
        assert_eq!(NestedApproxAggSpec::parse_fixed_interval_millis("5ms").unwrap(), 5);
        assert_eq!(NestedApproxAggSpec::parse_fixed_interval_millis("2s").unwrap(), 2_000);
        assert_eq!(NestedApproxAggSpec::parse_fixed_interval_millis("3m").unwrap(), 180_000);
        assert_eq!(
            NestedApproxAggSpec::parse_fixed_interval_millis("4h").unwrap(),
            14_400_000
        );
        assert_eq!(
            NestedApproxAggSpec::parse_fixed_interval_millis("1d").unwrap(),
            86_400_000
        );
    }

    #[test]
    fn try_new_rejects_fanout_below_final_size() {
        let result = NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 100,
                fanout: 10, // invalid: fanout < final_size
            }],
            vec![MetricSpec::Count],
        );
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("fanout"), "error should mention fanout: {msg}");
    }
}
