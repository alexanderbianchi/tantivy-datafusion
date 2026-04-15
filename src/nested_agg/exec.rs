//! Custom [`ExecutionPlan`] for nested approximate aggregation execution.
//!
//! ## Modes
//!
//! - **`PartialSplitLocal`** reads normalized Arrow rows from a split-local
//!   child, respects `_segment_ord` boundaries, trims to `fanout`, and emits a
//!   node-table partial batch.
//! - **`FinalMerge`** merges one coalesced stream of node-table partial batches,
//!   trims each `terms` level to `final_size`, and emits the final node table.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Float64Array, LargeListArray, ListArray, RecordBatch, StringArray,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatchOptions;
use datafusion::common::{Result, ScalarValue, Statistics};
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::Partitioning;
use datafusion_physical_plan::ExecutionPlanProperties;
use futures::StreamExt;

use super::node_table::{
    histogram_key_to_string, node_table_final_schema, node_table_partial_schema, COL_COUNT,
    COL_LEVEL, COL_NODE_ID, COL_PARENT_ID,
};
use super::spec::{BucketKind, MetricSpec, NestedApproxAggSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NestedApproxAggMode {
    PartialSplitLocal,
    FinalMerge,
}

pub struct NestedApproxAggExec {
    mode: NestedApproxAggMode,
    spec: Arc<NestedApproxAggSpec>,
    input: Arc<dyn ExecutionPlan>,
    output_schema: arrow::datatypes::SchemaRef,
    properties: PlanProperties,
}

impl fmt::Debug for NestedApproxAggExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NestedApproxAggExec")
            .field("mode", &self.mode)
            .field("levels", &self.spec.levels.len())
            .field("metrics", &self.spec.metrics.len())
            .finish()
    }
}

impl NestedApproxAggExec {
    pub fn new_partial_split_local(
        input: Arc<dyn ExecutionPlan>,
        spec: Arc<NestedApproxAggSpec>,
    ) -> Self {
        let output_schema = node_table_partial_schema(&spec);
        let properties = make_properties(&output_schema, input.output_partitioning().partition_count());
        Self {
            mode: NestedApproxAggMode::PartialSplitLocal,
            spec,
            input,
            output_schema,
            properties,
        }
    }

    pub fn new_final_merge(input: Arc<dyn ExecutionPlan>, spec: Arc<NestedApproxAggSpec>) -> Self {
        let output_schema = node_table_final_schema(&spec);
        let properties = make_properties(&output_schema, 1);
        Self {
            mode: NestedApproxAggMode::FinalMerge,
            spec,
            input,
            output_schema,
            properties,
        }
    }

    pub fn from_codec(
        mode: NestedApproxAggMode,
        spec: Arc<NestedApproxAggSpec>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Self {
        match mode {
            NestedApproxAggMode::PartialSplitLocal => Self::new_partial_split_local(input, spec),
            NestedApproxAggMode::FinalMerge => Self::new_final_merge(input, spec),
        }
    }

    pub fn mode(&self) -> NestedApproxAggMode {
        self.mode
    }

    pub fn spec(&self) -> &Arc<NestedApproxAggSpec> {
        &self.spec
    }
}

fn make_properties(
    output_schema: &arrow::datatypes::SchemaRef,
    partitions: usize,
) -> PlanProperties {
    PlanProperties::new(
        EquivalenceProperties::new(Arc::clone(output_schema)),
        Partitioning::UnknownPartitioning(partitions.max(1)),
        EmissionType::Final,
        Boundedness::Bounded,
    )
}

impl DisplayAs for NestedApproxAggExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "NestedApproxAggExec(mode={:?}, levels={}, metrics={})",
            self.mode,
            self.spec.levels.len(),
            self.spec.metrics.len(),
        )
    }
}

impl ExecutionPlan for NestedApproxAggExec {
    fn name(&self) -> &str {
        "NestedApproxAggExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "NestedApproxAggExec expects exactly one child".into(),
            ));
        }
        Ok(Arc::new(Self::from_codec(
            self.mode,
            Arc::clone(&self.spec),
            Arc::clone(&children[0]),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        match self.mode {
            NestedApproxAggMode::PartialSplitLocal => {
                self.execute_partial_split_local(partition, context)
            }
            NestedApproxAggMode::FinalMerge => self.execute_final_merge(partition, context),
        }
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.output_schema))
    }
}

impl NestedApproxAggExec {
    fn execute_partial_split_local(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        let input_partitions = self.input.output_partitioning().partition_count();
        if partition >= input_partitions {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.output_schema),
                futures::stream::empty(),
            )));
        }

        let mut input_stream = self.input.execute(partition, context)?;
        let spec = Arc::clone(&self.spec);
        let schema = Arc::clone(&self.output_schema);

        let stream = futures::stream::once(async move {
            let mut state = PartialSplitState::new(&spec);
            while let Some(batch_result) = input_stream.next().await {
                let batch = batch_result?;
                ingest_projected_batch(&batch, &spec, &mut state)?;
            }
            state.finish(&spec);
            partial_batch_from_tree(&state.split_tree, &spec, &schema)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        )))
    }

    fn execute_final_merge(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<datafusion::physical_plan::SendableRecordBatchStream> {
        if partition != 0 {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.output_schema),
                futures::stream::empty(),
            )));
        }

        let mut input_stream = self.input.execute(0, context)?;
        let spec = Arc::clone(&self.spec);
        let schema = Arc::clone(&self.output_schema);

        let stream = futures::stream::once(async move {
            let mut merge_tree = MergeTree::new(spec.total_metric_state_fields());
            while let Some(batch_result) = input_stream.next().await {
                let batch = batch_result?;
                ingest_partial_batch(&batch, &spec, &mut merge_tree)?;
            }
            trim_tree(&mut merge_tree, &spec, TrimLimit::FinalSize);
            final_batch_from_tree(&merge_tree, &spec, &schema)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        )))
    }
}

struct PartialSplitState {
    current_segment_ord: Option<u32>,
    segment_tree: MergeTree,
    split_tree: MergeTree,
}

impl PartialSplitState {
    fn new(spec: &NestedApproxAggSpec) -> Self {
        let state_cols = spec.total_metric_state_fields();
        Self {
            current_segment_ord: None,
            segment_tree: MergeTree::new(state_cols),
            split_tree: MergeTree::new(state_cols),
        }
    }

    fn finish(&mut self, spec: &NestedApproxAggSpec) {
        self.flush_segment(spec);
        trim_tree(&mut self.split_tree, spec, TrimLimit::Fanout);
    }

    fn flush_segment(&mut self, spec: &NestedApproxAggSpec) {
        if self.segment_tree.is_empty() {
            return;
        }
        trim_tree(&mut self.segment_tree, spec, TrimLimit::Fanout);
        let segment_tree =
            std::mem::replace(&mut self.segment_tree, MergeTree::new(spec.total_metric_state_fields()));
        merge_tree_into(&mut self.split_tree, segment_tree, spec);
    }
}

struct MergeTree {
    root_children: HashMap<String, MergeNode>,
    num_state_cols: usize,
}

struct MergeNode {
    count: u64,
    metric_states: Vec<Option<f64>>,
    children: HashMap<String, MergeNode>,
}

impl MergeTree {
    fn new(num_state_cols: usize) -> Self {
        Self {
            root_children: HashMap::new(),
            num_state_cols,
        }
    }

    fn is_empty(&self) -> bool {
        self.root_children.is_empty()
    }
}

impl MergeNode {
    fn new(num_state_cols: usize) -> Self {
        Self {
            count: 0,
            metric_states: vec![None; num_state_cols],
            children: HashMap::new(),
        }
    }

    fn merge_count(&mut self, count: u64) {
        self.count += count;
    }

    fn merge_states(&mut self, other_states: &[Option<f64>], spec: &NestedApproxAggSpec) {
        let mut col_offset = 0;
        for metric in &spec.metrics {
            let width = metric.state_field_count();
            for state_idx in 0..width {
                let idx = col_offset + state_idx;
                self.metric_states[idx] = merge_metric_state_value(
                    metric,
                    state_idx,
                    self.metric_states[idx],
                    other_states.get(idx).copied().unwrap_or(None),
                );
            }
            col_offset += width;
        }
    }
}

fn merge_metric_state_value(
    metric: &MetricSpec,
    state_idx: usize,
    left: Option<f64>,
    right: Option<f64>,
) -> Option<f64> {
    match (left, right) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(left), Some(right)) => Some(match metric {
            MetricSpec::Count => left + right,
            MetricSpec::Sum { .. } => left + right,
            MetricSpec::Min { .. } => left.min(right),
            MetricSpec::Max { .. } => left.max(right),
            MetricSpec::Avg { .. } => match state_idx {
                0 | 1 => left + right,
                _ => left,
            },
        }),
    }
}

fn ingest_partial_batch(
    batch: &RecordBatch,
    spec: &NestedApproxAggSpec,
    tree: &mut MergeTree,
) -> Result<()> {
    if batch.num_rows() == 0 {
        return Ok(());
    }

    let schema = batch.schema();
    let node_id_col = batch
        .column(schema.index_of(COL_NODE_ID)?)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| DataFusionError::Internal("bad __node_id type".into()))?;
    let parent_id_col = batch
        .column(schema.index_of(COL_PARENT_ID)?)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| DataFusionError::Internal("bad __parent_id type".into()))?;
    let level_col = batch
        .column(schema.index_of(COL_LEVEL)?)
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| DataFusionError::Internal("bad __level type".into()))?;
    let count_col = batch
        .column(schema.index_of(COL_COUNT)?)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal("bad __count type".into()))?;

    let key_cols: Vec<&StringArray> = (0..spec.levels.len())
        .map(|level| {
            let name = NestedApproxAggSpec::key_column_name(level);
            let idx = schema.index_of(&name).map_err(|_| {
                DataFusionError::Internal(format!("missing key column {name}"))
            })?;
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| DataFusionError::Internal(format!("bad type for {name}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let state_names = spec.all_state_field_names();
    let state_cols: Vec<&Float64Array> = state_names
        .iter()
        .map(|name| {
            let idx = schema.index_of(name).map_err(|_| {
                DataFusionError::Internal(format!("missing state column {name}"))
            })?;
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| DataFusionError::Internal(format!("bad type for {name}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut rows_by_level: Vec<Vec<usize>> = vec![Vec::new(); spec.levels.len()];
    for row in 0..batch.num_rows() {
        let level = level_col.value(row) as usize;
        if let Some(rows) = rows_by_level.get_mut(level) {
            rows.push(row);
        }
    }

    let mut paths_by_node_id: HashMap<u32, Vec<String>> = HashMap::new();

    for level in 0..spec.levels.len() {
        for &row in &rows_by_level[level] {
            let node_id = node_id_col.value(row);
            let parent_id = parent_id_col.value(row);
            let count = count_col.value(row);
            if key_cols[level].is_null(row) {
                return Err(DataFusionError::Internal(format!(
                    "row {row} at level {level} is missing its key column"
                )));
            }
            let key = key_cols[level].value(row).to_string();

            let mut path = if parent_id == 0 {
                Vec::new()
            } else {
                paths_by_node_id.get(&parent_id).cloned().ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "row {row} references unknown parent node_id {parent_id}"
                    ))
                })?
            };
            path.push(key);
            paths_by_node_id.insert(node_id, path.clone());

            let states: Vec<Option<f64>> = state_cols
                .iter()
                .map(|column| {
                    if column.is_null(row) {
                        None
                    } else {
                        Some(column.value(row))
                    }
                })
                .collect();

            insert_partial_node_into_tree(tree, &path, level, count, &states, spec);
        }
    }

    Ok(())
}

fn insert_partial_node_into_tree(
    tree: &mut MergeTree,
    path: &[String],
    level: usize,
    count: u64,
    states: &[Option<f64>],
    spec: &NestedApproxAggSpec,
) {
    if path.is_empty() {
        return;
    }

    let mut current = tree
        .root_children
        .entry(path[0].clone())
        .or_insert_with(|| MergeNode::new(tree.num_state_cols));

    if level == 0 {
        current.merge_count(count);
        current.merge_states(states, spec);
        return;
    }

    for (depth, key) in path.iter().enumerate().skip(1) {
        current = current
            .children
            .entry(key.clone())
            .or_insert_with(|| MergeNode::new(tree.num_state_cols));
        if depth == level {
            current.merge_count(count);
            current.merge_states(states, spec);
            return;
        }
    }
}

fn ingest_projected_batch(
    batch: &RecordBatch,
    spec: &NestedApproxAggSpec,
    state: &mut PartialSplitState,
) -> Result<()> {
    if batch.num_rows() == 0 {
        return Ok(());
    }

    let schema = batch.schema();
    let segment_ord_col = batch
        .column(schema.index_of("_segment_ord")?)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| DataFusionError::Internal("bad _segment_ord type".into()))?;

    let key_columns: Vec<ArrayRef> = (0..spec.levels.len())
        .map(|level| {
            let name = NestedApproxAggSpec::normalized_key_column_name(level);
            let idx = schema.index_of(&name).map_err(|_| {
                DataFusionError::Internal(format!("missing normalized key column {name}"))
            })?;
            Ok(batch.column(idx).clone())
        })
        .collect::<Result<Vec<_>>>()?;

    let metric_columns: Vec<Option<ArrayRef>> = spec
        .metrics
        .iter()
        .enumerate()
        .map(|(metric_idx, metric)| match metric {
            MetricSpec::Count => Ok(None),
            _ => {
                let name = NestedApproxAggSpec::normalized_metric_column_name(metric_idx);
                let idx = schema.index_of(&name).map_err(|_| {
                    DataFusionError::Internal(format!(
                        "missing normalized metric column {name}"
                    ))
                })?;
                Ok(Some(batch.column(idx).clone()))
            }
        })
        .collect::<Result<Vec<_>>>()?;

    for row in 0..batch.num_rows() {
        let segment_ord = segment_ord_col.value(row);
        if let Some(current) = state.current_segment_ord {
            if current != segment_ord {
                state.flush_segment(spec);
            }
        }
        state.current_segment_ord = Some(segment_ord);

        let keys_by_level: Vec<Vec<String>> = spec
            .levels
            .iter()
            .enumerate()
            .map(|(level_idx, level_spec)| {
                extract_row_keys(key_columns[level_idx].as_ref(), row, &level_spec.kind)
            })
            .collect::<Result<Vec<_>>>()?;

        let row_states = row_metric_states(row, spec, &metric_columns)?;
        insert_row_into_tree(&mut state.segment_tree, &keys_by_level, &row_states, spec);
    }

    Ok(())
}

fn row_metric_states(
    row: usize,
    spec: &NestedApproxAggSpec,
    metric_columns: &[Option<ArrayRef>],
) -> Result<Vec<Option<f64>>> {
    let mut states = Vec::with_capacity(spec.total_metric_state_fields());

    for (metric_idx, metric) in spec.metrics.iter().enumerate() {
        let Some(column) = metric_columns
            .get(metric_idx)
            .ok_or_else(|| DataFusionError::Internal("metric column index out of range".into()))?
            .as_ref()
        else {
            continue;
        };

        let summary = summarize_metric_input(column.as_ref(), row)?;
        match metric {
            MetricSpec::Count => {}
            MetricSpec::Sum { .. } => states.push(Some(summary.sum)),
            MetricSpec::Min { .. } => states.push(summary.min),
            MetricSpec::Max { .. } => states.push(summary.max),
            MetricSpec::Avg { .. } => {
                states.push(Some(summary.count as f64));
                states.push(Some(summary.sum));
            }
        }
    }

    Ok(states)
}

#[derive(Debug, Clone, Copy, Default)]
struct MetricInputSummary {
    count: usize,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl MetricInputSummary {
    fn record(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min = Some(match self.min {
            Some(current) => current.min(value),
            None => value,
        });
        self.max = Some(match self.max {
            Some(current) => current.max(value),
            None => value,
        });
    }
}

fn summarize_metric_input(array: &dyn Array, row: usize) -> Result<MetricInputSummary> {
    match array.data_type() {
        DataType::Float64 => {
            let typed = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| DataFusionError::Internal("bad Float64 metric input".into()))?;
            let mut summary = MetricInputSummary::default();
            if !typed.is_null(row) {
                summary.record(typed.value(row));
            }
            Ok(summary)
        }
        DataType::List(field) if field.data_type() == &DataType::Float64 => {
            let typed = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| DataFusionError::Internal("bad List<Float64> metric input".into()))?;
            summarize_float64_values(typed.value(row).as_ref())
        }
        DataType::LargeList(field) if field.data_type() == &DataType::Float64 => {
            let typed = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("bad LargeList<Float64> metric input".into())
                })?;
            summarize_float64_values(typed.value(row).as_ref())
        }
        other => Err(DataFusionError::Internal(format!(
            "normalized metric inputs must be Float64 or List<Float64>, got {other:?}"
        ))),
    }
}

fn summarize_float64_values(array: &dyn Array) -> Result<MetricInputSummary> {
    let typed = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataFusionError::Internal("bad nested Float64 metric input".into()))?;
    let mut summary = MetricInputSummary::default();
    for index in 0..typed.len() {
        if !typed.is_null(index) {
            summary.record(typed.value(index));
        }
    }
    Ok(summary)
}

fn extract_row_keys(array: &dyn Array, row: usize, kind: &BucketKind) -> Result<Vec<String>> {
    match kind {
        BucketKind::Terms => extract_terms_keys(array, row),
        BucketKind::DateHistogram { fixed_interval } => {
            let interval = NestedApproxAggSpec::parse_fixed_interval_millis(fixed_interval)?;
            extract_histogram_keys(array, row, interval)
        }
    }
}

fn extract_terms_keys(array: &dyn Array, row: usize) -> Result<Vec<String>> {
    let scalar = ScalarValue::try_from_array(array, row)?;
    let mut seen = HashSet::new();
    let mut keys = Vec::new();
    collect_terms_keys_from_scalar(&scalar, &mut seen, &mut keys)?;
    Ok(keys)
}

fn collect_terms_keys_from_scalar(
    scalar: &ScalarValue,
    seen: &mut HashSet<String>,
    keys: &mut Vec<String>,
) -> Result<()> {
    match scalar {
        ScalarValue::Null => {}
        ScalarValue::List(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_terms_keys_from_scalar(&scalar, seen, keys)?;
            }
        }
        ScalarValue::LargeList(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_terms_keys_from_scalar(&scalar, seen, keys)?;
            }
        }
        ScalarValue::FixedSizeList(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_terms_keys_from_scalar(&scalar, seen, keys)?;
            }
        }
        _ => {
            let key = scalar_to_terms_key(scalar)?;
            if seen.insert(key.clone()) {
                keys.push(key);
            }
        }
    }
    Ok(())
}

fn scalar_to_terms_key(scalar: &ScalarValue) -> Result<String> {
    match scalar {
        ScalarValue::Dictionary(_, value) => scalar_to_terms_key(value),
        ScalarValue::Boolean(Some(value)) => Ok(value.to_string()),
        ScalarValue::Float16(Some(value)) => Ok(histogram_key_to_string(f64::from(*value))),
        ScalarValue::Float32(Some(value)) => Ok(histogram_key_to_string(*value as f64)),
        ScalarValue::Float64(Some(value)) => Ok(histogram_key_to_string(*value)),
        ScalarValue::Int8(Some(value)) => Ok(value.to_string()),
        ScalarValue::Int16(Some(value)) => Ok(value.to_string()),
        ScalarValue::Int32(Some(value)) => Ok(value.to_string()),
        ScalarValue::Int64(Some(value)) => Ok(value.to_string()),
        ScalarValue::UInt8(Some(value)) => Ok(value.to_string()),
        ScalarValue::UInt16(Some(value)) => Ok(value.to_string()),
        ScalarValue::UInt32(Some(value)) => Ok(value.to_string()),
        ScalarValue::UInt64(Some(value)) => Ok(value.to_string()),
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => Ok(value.clone()),
        ScalarValue::TimestampSecond(Some(value), _) => Ok(value.to_string()),
        ScalarValue::TimestampMillisecond(Some(value), _) => Ok(value.to_string()),
        ScalarValue::TimestampMicrosecond(Some(value), _) => Ok(value.to_string()),
        ScalarValue::TimestampNanosecond(Some(value), _) => Ok(value.to_string()),
        ScalarValue::Date32(Some(value)) => Ok(value.to_string()),
        ScalarValue::Date64(Some(value)) => Ok(value.to_string()),
        ScalarValue::Null => Err(DataFusionError::Internal(
            "unexpected null scalar while stringifying term key".into(),
        )),
        other => Err(DataFusionError::NotImplemented(format!(
            "unsupported terms key scalar in nested fallback: {other:?}"
        ))),
    }
}

fn extract_histogram_keys(array: &dyn Array, row: usize, interval_millis: i64) -> Result<Vec<String>> {
    let scalar = ScalarValue::try_from_array(array, row)?;
    let mut seen = HashSet::new();
    let mut keys = Vec::new();
    collect_histogram_keys_from_scalar(&scalar, interval_millis, &mut seen, &mut keys)?;
    Ok(keys)
}

fn collect_histogram_keys_from_scalar(
    scalar: &ScalarValue,
    interval_millis: i64,
    seen: &mut HashSet<String>,
    keys: &mut Vec<String>,
) -> Result<()> {
    match scalar {
        ScalarValue::Null => {}
        ScalarValue::List(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_histogram_keys_from_scalar(&scalar, interval_millis, seen, keys)?;
            }
        }
        ScalarValue::LargeList(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_histogram_keys_from_scalar(&scalar, interval_millis, seen, keys)?;
            }
        }
        ScalarValue::FixedSizeList(list) => {
            let values = list.value(0);
            for row in 0..values.len() {
                if values.is_null(row) {
                    continue;
                }
                let scalar = ScalarValue::try_from_array(values.as_ref(), row)?;
                collect_histogram_keys_from_scalar(&scalar, interval_millis, seen, keys)?;
            }
        }
        _ => {
            let (value, interval) = scalar_to_histogram_value_and_interval(scalar, interval_millis)?;
            let bucket_start = value.div_euclid(interval) * interval;
            let key = histogram_key_to_string(bucket_start as f64);
            if seen.insert(key.clone()) {
                keys.push(key);
            }
        }
    }
    Ok(())
}

fn scalar_to_histogram_value_and_interval(
    scalar: &ScalarValue,
    interval_millis: i64,
) -> Result<(i64, i64)> {
    match scalar {
        ScalarValue::TimestampSecond(Some(value), _) => Ok((
            value.saturating_mul(1_000_000_000),
            interval_millis.saturating_mul(1_000_000),
        )),
        ScalarValue::TimestampMillisecond(Some(value), _) => Ok((
            value.saturating_mul(1_000_000),
            interval_millis.saturating_mul(1_000_000),
        )),
        ScalarValue::TimestampMicrosecond(Some(value), _) => {
            Ok((value.saturating_mul(1_000), interval_millis.saturating_mul(1_000_000)))
        }
        ScalarValue::TimestampNanosecond(Some(value), _) => Ok((
            *value,
            interval_millis.saturating_mul(1_000_000),
        )),
        ScalarValue::Date32(Some(value)) => Ok((
            (*value as i64).saturating_mul(86_400_000_000_000),
            interval_millis.saturating_mul(1_000_000),
        )),
        ScalarValue::Date64(Some(value)) => Ok((
            value.saturating_mul(1_000_000),
            interval_millis.saturating_mul(1_000_000),
        )),
        ScalarValue::Int8(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::Int16(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::Int32(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::Int64(Some(value)) => Ok((*value, interval_millis)),
        ScalarValue::UInt8(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::UInt16(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::UInt32(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::UInt64(Some(value)) => i64::try_from(*value).map(|value| (value, interval_millis)).map_err(|_| {
            DataFusionError::Internal(format!(
                "histogram timestamp value {value} does not fit in i64 milliseconds"
            ))
        }),
        ScalarValue::Float32(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::Float64(Some(value)) => Ok((*value as i64, interval_millis)),
        ScalarValue::Null => Err(DataFusionError::Internal(
            "unexpected null scalar while computing histogram key".into(),
        )),
        other => Err(DataFusionError::NotImplemented(format!(
            "unsupported date_histogram input scalar in nested fallback: {other:?}"
        ))),
    }
}

fn insert_row_into_tree(
    tree: &mut MergeTree,
    keys_by_level: &[Vec<String>],
    states: &[Option<f64>],
    spec: &NestedApproxAggSpec,
) {
    insert_row_into_children(
        &mut tree.root_children,
        keys_by_level,
        0,
        states,
        spec,
        tree.num_state_cols,
    );
}

fn insert_row_into_children(
    children: &mut HashMap<String, MergeNode>,
    keys_by_level: &[Vec<String>],
    level: usize,
    states: &[Option<f64>],
    spec: &NestedApproxAggSpec,
    num_state_cols: usize,
) {
    let Some(level_keys) = keys_by_level.get(level) else {
        return;
    };
    if level_keys.is_empty() {
        return;
    }

    for key in level_keys {
        let node = children
            .entry(key.clone())
            .or_insert_with(|| MergeNode::new(num_state_cols));
        node.merge_count(1);
        node.merge_states(states, spec);
        insert_row_into_children(&mut node.children, keys_by_level, level + 1, states, spec, num_state_cols);
    }
}

fn merge_tree_into(target: &mut MergeTree, source: MergeTree, spec: &NestedApproxAggSpec) {
    merge_children_into(
        &mut target.root_children,
        source.root_children,
        spec,
        target.num_state_cols,
    );
}

fn merge_children_into(
    target: &mut HashMap<String, MergeNode>,
    source: HashMap<String, MergeNode>,
    spec: &NestedApproxAggSpec,
    num_state_cols: usize,
) {
    for (key, source_node) in source {
        let target_node = target
            .entry(key)
            .or_insert_with(|| MergeNode::new(num_state_cols));
        merge_node_into(target_node, source_node, spec, num_state_cols);
    }
}

fn merge_node_into(
    target: &mut MergeNode,
    source: MergeNode,
    spec: &NestedApproxAggSpec,
    num_state_cols: usize,
) {
    target.merge_count(source.count);
    target.merge_states(&source.metric_states, spec);
    merge_children_into(&mut target.children, source.children, spec, num_state_cols);
}

#[derive(Debug, Clone, Copy)]
enum TrimLimit {
    Fanout,
    FinalSize,
}

fn trim_tree(tree: &mut MergeTree, spec: &NestedApproxAggSpec, limit: TrimLimit) {
    trim_children_map(&mut tree.root_children, spec, 0, limit);
}

fn trim_children_map(
    children: &mut HashMap<String, MergeNode>,
    spec: &NestedApproxAggSpec,
    level: usize,
    limit: TrimLimit,
) {
    if level >= spec.levels.len() {
        return;
    }

    let level_spec = &spec.levels[level];
    if matches!(level_spec.kind, BucketKind::Terms) {
        let keep = match limit {
            TrimLimit::Fanout => level_spec.fanout as usize,
            TrimLimit::FinalSize => level_spec.final_size as usize,
        };
        if children.len() > keep {
            let mut candidates: Vec<(String, u64)> = children
                .iter()
                .map(|(key, node)| (key.clone(), node.count))
                .collect();
            candidates.sort_by(|left, right| {
                right
                    .1
                    .cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let keep_keys: HashSet<String> = candidates
                .into_iter()
                .take(keep)
                .map(|(key, _)| key)
                .collect();
            children.retain(|key, _| keep_keys.contains(key));
        }
    }

    for node in children.values_mut() {
        trim_children_map(&mut node.children, spec, level + 1, limit);
    }
}

fn partial_batch_from_tree(
    tree: &MergeTree,
    spec: &NestedApproxAggSpec,
    schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch> {
    let num_levels = spec.levels.len();
    let num_state_cols = spec.total_metric_state_fields();

    let mut node_ids = Vec::new();
    let mut parent_ids = Vec::new();
    let mut levels = Vec::new();
    let mut counts = Vec::new();
    let mut keys: Vec<Vec<Option<String>>> = (0..num_levels).map(|_| Vec::new()).collect();
    let mut state_cols: Vec<Vec<Option<f64>>> =
        (0..num_state_cols).map(|_| Vec::new()).collect();
    let mut next_id: u32 = 1;

    for (key, node) in sorted_children(&tree.root_children) {
        emit_partial_node(
            node,
            key,
            0,
            0,
            &mut next_id,
            &mut node_ids,
            &mut parent_ids,
            &mut levels,
            &mut counts,
            &mut keys,
            &mut state_cols,
        );
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    columns.push(Arc::new(UInt32Array::from(node_ids)));
    columns.push(Arc::new(UInt32Array::from(parent_ids)));
    columns.push(Arc::new(UInt16Array::from(levels)));
    columns.push(Arc::new(UInt64Array::from(counts)));
    for key_column in keys {
        columns.push(Arc::new(StringArray::from(key_column)) as ArrayRef);
    }
    for state_column in state_cols {
        columns.push(Arc::new(Float64Array::from(state_column)) as ArrayRef);
    }

    record_batch_from_columns(schema, columns)
}

fn emit_partial_node(
    node: &MergeNode,
    key: &str,
    level: usize,
    parent_id: u32,
    next_id: &mut u32,
    node_ids: &mut Vec<u32>,
    parent_ids: &mut Vec<u32>,
    levels: &mut Vec<u16>,
    counts: &mut Vec<u64>,
    keys: &mut [Vec<Option<String>>],
    states: &mut [Vec<Option<f64>>],
) {
    let my_id = *next_id;
    *next_id += 1;

    node_ids.push(my_id);
    parent_ids.push(parent_id);
    levels.push(level as u16);
    counts.push(node.count);

    for (idx, column) in keys.iter_mut().enumerate() {
        if idx == level {
            column.push(Some(key.to_string()));
        } else {
            column.push(None);
        }
    }

    for (idx, column) in states.iter_mut().enumerate() {
        column.push(node.metric_states.get(idx).copied().unwrap_or(None));
    }

    for (child_key, child_node) in sorted_children(&node.children) {
        emit_partial_node(
            child_node,
            child_key,
            level + 1,
            my_id,
            next_id,
            node_ids,
            parent_ids,
            levels,
            counts,
            keys,
            states,
        );
    }
}

fn final_batch_from_tree(
    tree: &MergeTree,
    spec: &NestedApproxAggSpec,
    schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch> {
    let num_levels = spec.levels.len();
    let num_metrics = spec.metrics.len();

    let mut node_ids = Vec::new();
    let mut parent_ids = Vec::new();
    let mut levels = Vec::new();
    let mut counts = Vec::new();
    let mut keys: Vec<Vec<Option<String>>> = (0..num_levels).map(|_| Vec::new()).collect();
    let mut metric_cols: Vec<Vec<Option<f64>>> = (0..num_metrics).map(|_| Vec::new()).collect();
    let mut next_id: u32 = 1;

    for (key, node) in sorted_children(&tree.root_children) {
        emit_final_node(
            node,
            key,
            0,
            0,
            spec,
            &mut next_id,
            &mut node_ids,
            &mut parent_ids,
            &mut levels,
            &mut counts,
            &mut keys,
            &mut metric_cols,
        );
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    columns.push(Arc::new(UInt32Array::from(node_ids)));
    columns.push(Arc::new(UInt32Array::from(parent_ids)));
    columns.push(Arc::new(UInt16Array::from(levels)));
    columns.push(Arc::new(UInt64Array::from(counts)));
    for key_column in keys {
        columns.push(Arc::new(StringArray::from(key_column)) as ArrayRef);
    }
    for metric_column in metric_cols {
        columns.push(Arc::new(Float64Array::from(metric_column)) as ArrayRef);
    }

    record_batch_from_columns(schema, columns)
}

fn emit_final_node(
    node: &MergeNode,
    key: &str,
    level: usize,
    parent_id: u32,
    spec: &NestedApproxAggSpec,
    next_id: &mut u32,
    node_ids: &mut Vec<u32>,
    parent_ids: &mut Vec<u32>,
    levels: &mut Vec<u16>,
    counts: &mut Vec<u64>,
    keys: &mut [Vec<Option<String>>],
    metrics: &mut [Vec<Option<f64>>],
) {
    let my_id = *next_id;
    *next_id += 1;

    node_ids.push(my_id);
    parent_ids.push(parent_id);
    levels.push(level as u16);
    counts.push(node.count);

    for (idx, column) in keys.iter_mut().enumerate() {
        if idx == level {
            column.push(Some(key.to_string()));
        } else {
            column.push(None);
        }
    }

    let finalized = finalize_metric_states(&node.metric_states, spec, node.count);
    for (idx, column) in metrics.iter_mut().enumerate() {
        column.push(finalized.get(idx).copied().unwrap_or(None));
    }

    for (child_key, child_node) in sorted_children(&node.children) {
        emit_final_node(
            child_node,
            child_key,
            level + 1,
            my_id,
            spec,
            next_id,
            node_ids,
            parent_ids,
            levels,
            counts,
            keys,
            metrics,
        );
    }
}

fn finalize_metric_states(
    states: &[Option<f64>],
    spec: &NestedApproxAggSpec,
    count: u64,
) -> Vec<Option<f64>> {
    let mut finalized = Vec::with_capacity(spec.metrics.len());
    let mut state_offset = 0;

    for metric in &spec.metrics {
        let value = match metric {
            MetricSpec::Count => Some(count as f64),
            MetricSpec::Sum { .. } | MetricSpec::Min { .. } | MetricSpec::Max { .. } => {
                states.get(state_offset).copied().unwrap_or(None)
            }
            MetricSpec::Avg { .. } => {
                let count = states.get(state_offset).copied().unwrap_or(None);
                let sum = states.get(state_offset + 1).copied().unwrap_or(None);
                match (count, sum) {
                    (Some(count), Some(sum)) if count > 0.0 => Some(sum / count),
                    _ => None,
                }
            }
        };
        finalized.push(value);
        state_offset += metric.state_field_count();
    }

    finalized
}

fn sorted_children(children: &HashMap<String, MergeNode>) -> Vec<(&str, &MergeNode)> {
    let mut sorted: Vec<(&str, &MergeNode)> =
        children.iter().map(|(key, node)| (key.as_str(), node)).collect();
    sorted.sort_by(|left, right| {
        right
            .1
            .count
            .cmp(&left.1.count)
            .then_with(|| left.0.cmp(right.0))
    });
    sorted
}

fn record_batch_from_columns(
    schema: &arrow::datatypes::SchemaRef,
    columns: Vec<ArrayRef>,
) -> Result<RecordBatch> {
    if columns.is_empty() || columns.first().map(|column| column.len()) == Some(0) {
        return RecordBatch::try_new_with_options(
            Arc::clone(schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(0)),
        )
        .map_err(|err| DataFusionError::Internal(format!("build empty nested agg batch: {err}")));
    }

    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|err| DataFusionError::Internal(format!("build nested agg batch: {err}")))
}
