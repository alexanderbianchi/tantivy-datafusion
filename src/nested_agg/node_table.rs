//! Node-table Arrow schema derivation and conversion from tantivy
//! [`IntermediateAggregationResults`] to Arrow [`RecordBatch`].
//!
//! The node table is the shared intermediate format for nested approximate
//! aggregations. Each row represents one bucket node in the aggregation tree.
//!
//! ## Intermediate schema
//!
//! ```text
//! __node_id       UInt32
//! __parent_id     UInt32   (0 for root-level nodes)
//! __level         UInt16
//! __count         UInt64
//! __key_0 .. n    Utf8     (one per level, only the node's own level is non-null)
//! [metric state]  Float64  (one column per metric state field; `Count`
//!                          remains structural and carries no state column)
//! ```
//!
//! ## Final output schema
//!
//! ```text
//! __node_id       UInt32
//! __parent_id     UInt32
//! __level         UInt16
//! __count         UInt64
//! __key_0 .. n    Utf8
//! [finalized]     Float64  (one column per metric; `Count` is derived from
//!                          `__count`)
//! ```

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray, UInt16Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults, IntermediateBucketResult,
    IntermediateKey, IntermediateMetricResult,
};

use super::spec::{MetricSpec, NestedApproxAggSpec};

// ---------------------------------------------------------------------------
// Column name constants
// ---------------------------------------------------------------------------

pub const COL_NODE_ID: &str = "__node_id";
pub const COL_PARENT_ID: &str = "__parent_id";
pub const COL_LEVEL: &str = "__level";
pub const COL_COUNT: &str = "__count";

// ---------------------------------------------------------------------------
// Schema derivation
// ---------------------------------------------------------------------------

/// Derive the intermediate (partial) node-table schema from a spec.
pub fn node_table_partial_schema(spec: &NestedApproxAggSpec) -> SchemaRef {
    let mut fields = structural_fields(spec);
    for name in spec.all_state_field_names() {
        fields.push(Field::new(name, DataType::Float64, true));
    }
    Arc::new(Schema::new(fields))
}

/// Derive the final output node-table schema from a spec.
pub fn node_table_final_schema(spec: &NestedApproxAggSpec) -> SchemaRef {
    let mut fields = structural_fields(spec);
    for name in spec.all_final_field_names() {
        fields.push(Field::new(name, DataType::Float64, true));
    }
    Arc::new(Schema::new(fields))
}

fn structural_fields(spec: &NestedApproxAggSpec) -> Vec<Field> {
    let mut fields = vec![
        Field::new(COL_NODE_ID, DataType::UInt32, false),
        Field::new(COL_PARENT_ID, DataType::UInt32, false),
        Field::new(COL_LEVEL, DataType::UInt16, false),
        Field::new(COL_COUNT, DataType::UInt64, false),
    ];
    for level_idx in 0..spec.levels.len() {
        fields.push(Field::new(
            NestedApproxAggSpec::key_column_name(level_idx),
            DataType::Utf8,
            true,
        ));
    }
    fields
}

// ---------------------------------------------------------------------------
// IntermediateAggregationResults -> node-table RecordBatch
// ---------------------------------------------------------------------------

/// Convert tantivy intermediate aggregation results into a node-table
/// [`RecordBatch`].
///
/// This is the critical pushdown conversion: tantivy's distributed collector
/// produces [`IntermediateAggregationResults`] per split, and this function
/// flattens the nested tree into a columnar node table for transport.
pub fn intermediate_results_to_node_table_batch(
    partial: &IntermediateAggregationResults,
    spec: &NestedApproxAggSpec,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let num_levels = spec.levels.len();
    let num_state_cols = spec.total_metric_state_fields();

    let mut builder = NodeTableBuilder::new(num_levels, num_state_cols);
    walk_intermediate_level(partial, spec, 0, 0, &mut builder)?;

    builder.to_record_batch(schema)
}

// ---------------------------------------------------------------------------
// Recursive tree walk
// ---------------------------------------------------------------------------

struct NodeTableBuilder {
    node_ids: Vec<u32>,
    parent_ids: Vec<u32>,
    levels: Vec<u16>,
    counts: Vec<u64>,
    /// One Vec<Option<String>> per level.
    keys: Vec<Vec<Option<String>>>,
    /// One Vec<Option<f64>> per metric state column.
    metric_states: Vec<Vec<Option<f64>>>,
    next_node_id: u32,
}

impl NodeTableBuilder {
    fn new(num_levels: usize, num_state_cols: usize) -> Self {
        Self {
            node_ids: Vec::new(),
            parent_ids: Vec::new(),
            levels: Vec::new(),
            counts: Vec::new(),
            keys: (0..num_levels).map(|_| Vec::new()).collect(),
            metric_states: (0..num_state_cols).map(|_| Vec::new()).collect(),
            next_node_id: 1, // 0 is reserved for "no parent"
        }
    }

    fn alloc_node_id(&mut self) -> u32 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    fn push_row(
        &mut self,
        node_id: u32,
        parent_id: u32,
        level: u16,
        count: u64,
        key_level: usize,
        key_value: String,
        state_values: &[Option<f64>],
    ) {
        self.node_ids.push(node_id);
        self.parent_ids.push(parent_id);
        self.levels.push(level);
        self.counts.push(count);
        for (lvl, col) in self.keys.iter_mut().enumerate() {
            if lvl == key_level {
                col.push(Some(key_value.clone()));
            } else {
                col.push(None);
            }
        }
        for (i, col) in self.metric_states.iter_mut().enumerate() {
            col.push(state_values.get(i).copied().unwrap_or(None));
        }
    }

    fn to_record_batch(self, schema: &SchemaRef) -> Result<RecordBatch> {
        let num_rows = self.node_ids.len();
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

        columns.push(Arc::new(UInt32Array::from(self.node_ids)));
        columns.push(Arc::new(UInt32Array::from(self.parent_ids)));
        columns.push(Arc::new(UInt16Array::from(self.levels)));
        columns.push(Arc::new(UInt64Array::from(self.counts)));

        for key_col in self.keys {
            let arr: StringArray = key_col.into_iter().collect();
            columns.push(Arc::new(arr));
        }

        for state_col in self.metric_states {
            columns.push(Arc::new(Float64Array::from(state_col)));
        }

        if columns.len() != schema.fields().len() {
            return Err(DataFusionError::Internal(format!(
                "node table column count mismatch: built {} but schema has {}",
                columns.len(),
                schema.fields().len(),
            )));
        }
        if num_rows == 0 {
            return RecordBatch::try_new_with_options(
                Arc::clone(schema),
                columns,
                &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(0)),
            )
            .map_err(|e| DataFusionError::Internal(format!("build empty node table: {e}")));
        }

        RecordBatch::try_new(Arc::clone(schema), columns)
            .map_err(|e| DataFusionError::Internal(format!("build node table batch: {e}")))
    }
}

/// Recursively walk the intermediate aggregation tree and emit rows.
fn walk_intermediate_level(
    results: &IntermediateAggregationResults,
    spec: &NestedApproxAggSpec,
    level_idx: usize,
    parent_id: u32,
    builder: &mut NodeTableBuilder,
) -> Result<()> {
    let agg_name = format!("level_{level_idx}");
    let agg_result = match results.get(&agg_name) {
        Some(r) => r,
        None => return Ok(()), // no results at this level
    };

    match agg_result {
        IntermediateAggregationResult::Bucket(bucket) => {
            walk_bucket(bucket, spec, level_idx, parent_id, builder)
        }
        IntermediateAggregationResult::Metric(_) => Err(DataFusionError::Internal(format!(
            "expected bucket at level {level_idx}, got metric"
        ))),
    }
}

fn walk_bucket(
    bucket: &IntermediateBucketResult,
    spec: &NestedApproxAggSpec,
    level_idx: usize,
    parent_id: u32,
    builder: &mut NodeTableBuilder,
) -> Result<()> {
    match bucket {
        IntermediateBucketResult::Terms { buckets } => {
            for (key, entry) in buckets.entries() {
                let key_str = intermediate_key_to_string(key);
                let node_id = builder.alloc_node_id();
                let is_leaf = level_idx + 1 >= spec.levels.len();

                let state_values = extract_metric_states(&entry.sub_aggregation, spec)?;

                builder.push_row(
                    node_id,
                    parent_id,
                    level_idx as u16,
                    entry.doc_count as u64,
                    level_idx,
                    key_str,
                    &state_values,
                );

                if !is_leaf {
                    walk_intermediate_level(
                        &entry.sub_aggregation,
                        spec,
                        level_idx + 1,
                        node_id,
                        builder,
                    )?;
                }
            }
        }
        IntermediateBucketResult::Histogram {
            buckets,
            is_date_agg: _,
        } => {
            for entry in buckets {
                let key_str = histogram_key_to_string(entry.key);
                let node_id = builder.alloc_node_id();
                let is_leaf = level_idx + 1 >= spec.levels.len();

                let state_values = extract_metric_states(&entry.sub_aggregation, spec)?;

                builder.push_row(
                    node_id,
                    parent_id,
                    level_idx as u16,
                    entry.doc_count,
                    level_idx,
                    key_str,
                    &state_values,
                );

                if !is_leaf {
                    walk_intermediate_level(
                        &entry.sub_aggregation,
                        spec,
                        level_idx + 1,
                        node_id,
                        builder,
                    )?;
                }
            }
        }
        _ => {
            return Err(DataFusionError::NotImplemented(
                "node table conversion only supports terms and histogram buckets".into(),
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Metric state extraction
// ---------------------------------------------------------------------------

/// Extract metric state values from the sub-aggregation results at a leaf
/// bucket node. Returns one `Option<f64>` per metric state column.
fn extract_metric_states(
    sub_aggs: &IntermediateAggregationResults,
    spec: &NestedApproxAggSpec,
) -> Result<Vec<Option<f64>>> {
    let mut states = Vec::with_capacity(spec.total_metric_state_fields());

    for (idx, metric) in spec.metrics.iter().enumerate() {
        match metric {
            MetricSpec::Count => {}
            MetricSpec::Sum { .. } => {
                let metric = extract_metric_result(sub_aggs, idx)?;
                states.push(Some(extract_intermediate_sum(metric)));
            }
            MetricSpec::Min { .. } => {
                let metric = extract_metric_result(sub_aggs, idx)?;
                states.push(extract_intermediate_min(metric));
            }
            MetricSpec::Max { .. } => {
                let metric = extract_metric_result(sub_aggs, idx)?;
                states.push(extract_intermediate_max(metric));
            }
            MetricSpec::Avg { .. } => {
                let metric = extract_metric_result(sub_aggs, idx)?;
                let (count, sum) = extract_intermediate_avg_state(metric);
                states.push(Some(count));
                states.push(Some(sum));
            }
        }
    }

    Ok(states)
}

fn extract_metric_result<'a>(
    sub_aggs: &'a IntermediateAggregationResults,
    metric_idx: usize,
) -> Result<&'a IntermediateMetricResult> {
    let agg_name = format!("metric_{metric_idx}");
    match sub_aggs.get(&agg_name) {
        Some(IntermediateAggregationResult::Metric(metric)) => Ok(metric),
        Some(IntermediateAggregationResult::Bucket(_)) => Err(DataFusionError::Internal(format!(
            "expected metric for {agg_name}, got bucket"
        ))),
        None => Err(DataFusionError::Internal(format!(
            "missing intermediate metric result for {agg_name}"
        ))),
    }
}

fn extract_intermediate_sum(metric: &IntermediateMetricResult) -> f64 {
    match metric {
        IntermediateMetricResult::Sum(s) => s.finalize().unwrap_or(0.0),
        IntermediateMetricResult::Stats(s) => s.sum(),
        _ => 0.0,
    }
}

fn extract_intermediate_min(metric: &IntermediateMetricResult) -> Option<f64> {
    match metric {
        IntermediateMetricResult::Min(m) => m.finalize(),
        IntermediateMetricResult::Stats(s) => s.finalize().min,
        _ => None,
    }
}

fn extract_intermediate_max(metric: &IntermediateMetricResult) -> Option<f64> {
    match metric {
        IntermediateMetricResult::Max(m) => m.finalize(),
        IntermediateMetricResult::Stats(s) => s.finalize().max,
        _ => None,
    }
}

fn extract_intermediate_avg_state(metric: &IntermediateMetricResult) -> (f64, f64) {
    match metric {
        IntermediateMetricResult::Average(a) => (a.stats().count() as f64, a.stats().sum()),
        IntermediateMetricResult::Stats(s) => (s.count() as f64, s.sum()),
        _ => (0.0, 0.0),
    }
}

pub(crate) fn intermediate_key_to_string(key: &IntermediateKey) -> String {
    match key {
        IntermediateKey::Str(s) => s.clone(),
        IntermediateKey::F64(v) => numeric_key_to_string(*v),
        IntermediateKey::I64(v) => v.to_string(),
        IntermediateKey::U64(v) => v.to_string(),
        IntermediateKey::Bool(v) => v.to_string(),
        IntermediateKey::IpAddr(v) => v.to_string(),
    }
}

pub(crate) fn histogram_key_to_string(key: f64) -> String {
    numeric_key_to_string(key)
}

fn numeric_key_to_string(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nested_agg::spec::{BucketKind, BucketLevelSpec, MetricSpec};

    #[test]
    fn partial_schema_has_expected_columns() {
        let spec = NestedApproxAggSpec::try_new(
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
                    final_size: 5,
                    fanout: 20,
                },
            ],
            vec![
                MetricSpec::Count,
                MetricSpec::Avg { field: "latency".into() },
            ],
        )
        .unwrap();

        let schema = node_table_partial_schema(&spec);
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "__node_id",
                "__parent_id",
                "__level",
                "__count",
                "__key_0",
                "__key_1",
                "__ms_1_avg_count",
                "__ms_1_avg_sum",
            ]
        );
    }

    #[test]
    fn final_schema_has_expected_columns() {
        let spec = NestedApproxAggSpec::try_new(
            vec![BucketLevelSpec {
                kind: BucketKind::Terms,
                field: "service".into(),
                final_size: 10,
                fanout: 40,
            }],
            vec![MetricSpec::Sum { field: "bytes".into() }],
        )
        .unwrap();

        let schema = node_table_final_schema(&spec);
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "__node_id",
                "__parent_id",
                "__level",
                "__count",
                "__key_0",
                "__mf_0_sum",
            ]
        );
    }

    #[test]
    fn histogram_key_strings_match_integer_bucket_boundaries() {
        assert_eq!(histogram_key_to_string(1_710_000_000_000.0), "1710000000000");
        assert_eq!(histogram_key_to_string(1.5), "1.5");
    }
}
