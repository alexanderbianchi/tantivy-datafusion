/// A DataSource that runs tantivy's native aggregation on query-filtered docs.
///
/// Created by the `AggPushdown` optimizer rule when it detects an
/// `AggregateExec` above a `SingleTableDataSource`. Preserves the full
/// query context (FTS + fast field filters) from the original scan.
use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, Statistics};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::Expr;
use datafusion_datasource::source::DataSource;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::projection::ProjectionExprs;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, Partitioning, SendableRecordBatchStream};
use futures::stream::{self, StreamExt};
use tantivy::aggregation::agg_req::Aggregations;
use tantivy::aggregation::intermediate_agg_result::IntermediateAggregationResults;
use tokio::sync::OnceCell;

use crate::index_opener::{IndexOpener, OpenerSplitRuntimeFactory};
use crate::nested_agg::spec::NestedApproxAggSpec;
use crate::split_runtime::{
    PreparedSplit, SplitDescriptor, SplitRuntimeFactoryExt, SplitRuntimeFactoryRef,
};
use crate::unified::single_table_provider::build_split_fast_field_query;
use crate::util::build_combined_query;

/// Guard that calls `BaselineMetrics::done()` on drop so elapsed time is
/// recorded even when the stream is cancelled.
struct MetricsGuard(BaselineMetrics);
impl Drop for MetricsGuard {
    fn drop(&mut self) {
        self.0.done();
    }
}

/// Guard that aborts a spawned task when dropped.
struct AbortOnDrop {
    handle: tokio::task::JoinHandle<()>,
    cancelled: Arc<AtomicBool>,
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.handle.abort();
    }
}

#[derive(Debug, Clone)]
struct AggSplitPlan {
    descriptor: SplitDescriptor,
    needs_warmup: bool,
}

pub struct AggDataSource {
    splits: Vec<AggSplitPlan>,
    /// Tantivy aggregation specification (terms + metric sub-aggs).
    aggregations: Arc<Aggregations>,
    /// Output schema matching the AggregateExec this replaces.
    output_schema: SchemaRef,
    /// Raw full-text queries deferred to execution time.
    raw_queries: Vec<(String, String)>,
    /// Pre-built tantivy queries from fast field filter conversion.
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    /// Source logical `Expr`s that produced `pre_built_query`. Stored for
    /// codec serialization so workers can re-derive the tantivy query.
    fast_field_filter_exprs: Vec<Expr>,
    /// Whether this source emits final aggregate rows, partial aggregate
    /// state rows, or node-table partial rows for nested aggregations.
    output_mode: AggOutputMode,
    /// Nested aggregation spec — present only for `NodeTablePartial` mode.
    nested_spec: Option<Arc<NestedApproxAggSpec>>,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    /// Ensures warmup runs at most once per split.
    warmup_done: Vec<Arc<OnceCell<()>>>,
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

impl fmt::Debug for AggDataSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AggDataSource")
            .field("splits", &self.splits.len())
            .field("output_mode", &self.output_mode)
            .field("schema", &self.output_schema)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOutputMode {
    FinalMerged,
    PartialStates,
    /// Emit node-table partial rows for nested approximate aggregations.
    /// Requires `nested_spec` to be set on the data source.
    NodeTablePartial,
}

#[derive(Clone)]
struct AggExecutionContext {
    aggs: Arc<Aggregations>,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
    cancelled: Arc<AtomicBool>,
}

struct AggSourceConfig {
    aggregations: Arc<Aggregations>,
    output_schema: SchemaRef,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
    output_mode: AggOutputMode,
    nested_spec: Option<Arc<NestedApproxAggSpec>>,
}

fn local_split_descriptor(
    opener: &Arc<dyn IndexOpener>,
    split_id: impl Into<String>,
) -> SplitDescriptor {
    SplitDescriptor::new(
        split_id.into(),
        Vec::new(),
        opener.schema(),
        opener.multi_valued_fields(),
    )
}

fn build_local_split_plans(split_openers: &[Arc<dyn IndexOpener>]) -> Vec<AggSplitPlan> {
    split_openers
        .iter()
        .enumerate()
        .map(|(idx, opener)| AggSplitPlan {
            descriptor: local_split_descriptor(opener, format!("local-split-{idx}")),
            needs_warmup: opener.needs_warmup(),
        })
        .collect()
}

async fn prepare_split(
    split: &AggSplitPlan,
    local_runtime_factory: Option<&SplitRuntimeFactoryRef>,
    context: &datafusion::execution::TaskContext,
) -> Result<Arc<PreparedSplit>> {
    let factory = context
        .session_config()
        .get_split_runtime_factory()
        .or_else(|| local_runtime_factory.cloned())
        .ok_or_else(|| {
            DataFusionError::Internal(
                "no SplitRuntimeFactory registered on session config; \
                 remote split execution requires config.set_split_runtime_factory(...)"
                    .into(),
            )
        })?;
    factory.prepare_split(&split.descriptor).await
}

fn split_needs_warmup(split: &AggSplitPlan) -> bool {
    split.needs_warmup
}

impl AggDataSource {
    pub fn new(
        index: tantivy::Index,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        Self::from_local_splits(
            vec![index],
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        )
    }

    pub fn from_local_splits(
        local_indexes: Vec<tantivy::Index>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let split_openers: Vec<Arc<dyn IndexOpener>> = local_indexes
            .into_iter()
            .map(|index| {
                Arc::new(crate::index_opener::DirectIndexOpener::new(index)) as Arc<dyn IndexOpener>
            })
            .collect();
        Self::from_local_split_openers(
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        )
    }

    pub fn from_local_splits_partial_states(
        local_indexes: Vec<tantivy::Index>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let split_openers: Vec<Arc<dyn IndexOpener>> = local_indexes
            .into_iter()
            .map(|index| {
                Arc::new(crate::index_opener::DirectIndexOpener::new(index)) as Arc<dyn IndexOpener>
            })
            .collect();
        Self::from_local_split_openers_partial_states(
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        )
    }

    fn from_local_split_openers(
        split_openers: Vec<Arc<dyn IndexOpener>>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let opener_map: std::collections::HashMap<String, Arc<dyn IndexOpener>> = split_openers
            .iter()
            .enumerate()
            .map(|(idx, opener)| (format!("local-split-{idx}"), Arc::clone(opener)))
            .collect();
        Self::from_split_descriptors_with_runtime_factory(
            build_local_split_plans(&split_openers)
                .into_iter()
                .map(|split| split.descriptor)
                .collect(),
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            Some(Arc::new(OpenerSplitRuntimeFactory::new(opener_map))),
        )
    }

    pub fn from_split_descriptors(
        split_descriptors: Vec<SplitDescriptor>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        Self::from_split_descriptors_with_runtime_factory(
            split_descriptors,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            None,
        )
    }

    pub(crate) fn from_split_descriptors_with_runtime_factory(
        split_descriptors: Vec<SplitDescriptor>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
        local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    ) -> Self {
        let splits = split_descriptors
            .into_iter()
            .map(|descriptor| AggSplitPlan {
                descriptor,
                needs_warmup: true,
            })
            .collect();
        Self::from_split_plans(
            splits,
            AggSourceConfig {
                aggregations,
                output_schema,
                raw_queries,
                pre_built_query,
                fast_field_filter_exprs,
                output_mode: AggOutputMode::FinalMerged,
                nested_spec: None,
            },
            local_runtime_factory,
        )
    }

    fn from_local_split_openers_partial_states(
        split_openers: Vec<Arc<dyn IndexOpener>>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let opener_map: std::collections::HashMap<String, Arc<dyn IndexOpener>> = split_openers
            .iter()
            .enumerate()
            .map(|(idx, opener)| (format!("local-split-{idx}"), Arc::clone(opener)))
            .collect();
        Self::from_split_descriptors_partial_states_with_runtime_factory(
            build_local_split_plans(&split_openers)
                .into_iter()
                .map(|split| split.descriptor)
                .collect(),
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            Some(Arc::new(OpenerSplitRuntimeFactory::new(opener_map))),
        )
    }

    pub fn from_split_descriptors_partial_states(
        split_descriptors: Vec<SplitDescriptor>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        Self::from_split_descriptors_partial_states_with_runtime_factory(
            split_descriptors,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            None,
        )
    }

    pub(crate) fn from_split_descriptors_partial_states_with_runtime_factory(
        split_descriptors: Vec<SplitDescriptor>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
        local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    ) -> Self {
        let splits = split_descriptors
            .into_iter()
            .map(|descriptor| AggSplitPlan {
                descriptor,
                needs_warmup: true,
            })
            .collect();
        Self::from_split_plans(
            splits,
            AggSourceConfig {
                aggregations,
                output_schema,
                raw_queries,
                pre_built_query,
                fast_field_filter_exprs,
                output_mode: AggOutputMode::PartialStates,
                nested_spec: None,
            },
            local_runtime_factory,
        )
    }

    /// Create an `AggDataSource` that emits node-table partial rows for
    /// nested approximate aggregations.
    pub fn from_split_descriptors_node_table_partial_with_runtime_factory(
        split_descriptors: Vec<SplitDescriptor>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
        nested_spec: Arc<NestedApproxAggSpec>,
        local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    ) -> Self {
        let splits = split_descriptors
            .into_iter()
            .map(|descriptor| AggSplitPlan {
                descriptor,
                needs_warmup: true,
            })
            .collect();
        Self::from_split_plans(
            splits,
            AggSourceConfig {
                aggregations,
                output_schema,
                raw_queries,
                pre_built_query,
                fast_field_filter_exprs,
                output_mode: AggOutputMode::NodeTablePartial,
                nested_spec: Some(nested_spec),
            },
            local_runtime_factory,
        )
    }

    /// Create a node-table partial source from local indexes.
    pub fn from_local_splits_node_table_partial(
        local_indexes: Vec<tantivy::Index>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
        nested_spec: Arc<NestedApproxAggSpec>,
    ) -> Self {
        let split_openers: Vec<Arc<dyn IndexOpener>> = local_indexes
            .into_iter()
            .map(|index| {
                Arc::new(crate::index_opener::DirectIndexOpener::new(index)) as Arc<dyn IndexOpener>
            })
            .collect();
        let opener_map: std::collections::HashMap<String, Arc<dyn IndexOpener>> = split_openers
            .iter()
            .enumerate()
            .map(|(idx, opener)| (format!("local-split-{idx}"), Arc::clone(opener)))
            .collect();
        Self::from_split_descriptors_node_table_partial_with_runtime_factory(
            build_local_split_plans(&split_openers)
                .into_iter()
                .map(|split| split.descriptor)
                .collect(),
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            nested_spec,
            Some(Arc::new(OpenerSplitRuntimeFactory::new(opener_map))),
        )
    }

    fn from_split_plans(
        splits: Vec<AggSplitPlan>,
        config: AggSourceConfig,
        local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    ) -> Self {
        let warmup_done = splits.iter().map(|_| Arc::new(OnceCell::new())).collect();
        Self {
            splits,
            aggregations: config.aggregations,
            output_schema: config.output_schema,
            raw_queries: config.raw_queries,
            pre_built_query: config.pre_built_query,
            fast_field_filter_exprs: config.fast_field_filter_exprs,
            output_mode: config.output_mode,
            nested_spec: config.nested_spec,
            local_runtime_factory,
            warmup_done,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn split_descriptors(&self) -> Vec<SplitDescriptor> {
        self.splits
            .iter()
            .map(|split| split.descriptor.clone())
            .collect()
    }

    pub fn local_runtime_factory(&self) -> Option<SplitRuntimeFactoryRef> {
        self.local_runtime_factory.clone()
    }

    /// Access the tantivy aggregation specification.
    pub fn aggregations(&self) -> &Arc<Aggregations> {
        &self.aggregations
    }

    /// Access the output schema.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }

    /// Access the raw full-text queries.
    pub fn raw_queries(&self) -> &[(String, String)] {
        &self.raw_queries
    }

    /// Access the source logical `Expr`s that produced `pre_built_query`.
    /// Used by the codec for serialization.
    pub fn fast_field_filter_exprs(&self) -> &[Expr] {
        &self.fast_field_filter_exprs
    }

    /// Access the pre-built tantivy query reconstructed from fast field filters.
    pub fn pre_built_query(&self) -> Option<&Arc<dyn tantivy::query::Query>> {
        self.pre_built_query.as_ref()
    }

    pub fn output_mode(&self) -> AggOutputMode {
        self.output_mode
    }

    /// Access the nested aggregation spec (present only for `NodeTablePartial`).
    pub fn nested_spec(&self) -> Option<&Arc<NestedApproxAggSpec>> {
        self.nested_spec.as_ref()
    }
}

impl DataSource for AggDataSource {
    fn open(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics_guard = MetricsGuard(BaselineMetrics::new(&self.metrics, partition));
        let sync_pool = crate::sync_exec::get_or_default_pool(context.as_ref());
        let schema = self.output_schema.clone();
        match self.output_mode {
            AggOutputMode::FinalMerged if partition != 0 => {
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::empty(),
                )));
            }
            AggOutputMode::PartialStates | AggOutputMode::NodeTablePartial
                if partition >= self.splits.len() =>
            {
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::empty(),
                )));
            }
            _ => {}
        }

        let splits = self.splits.clone();
        let warmup_done = self.warmup_done.clone();
        let output_mode = self.output_mode;
        let nested_spec = self.nested_spec.clone();
        let local_runtime_factory = self.local_runtime_factory.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(1);

        let cancelled_task = Arc::clone(&cancelled);
        let exec_ctx = AggExecutionContext {
            aggs: Arc::clone(&self.aggregations),
            raw_queries: self.raw_queries.clone(),
            pre_built_query: if self.splits.len() == 1 {
                self.pre_built_query
                    .as_ref()
                    .map(|q| Arc::from(q.box_clone()))
            } else {
                None
            },
            fast_field_filter_exprs: self.fast_field_filter_exprs.clone(),
            cancelled: Arc::clone(&cancelled_task),
        };
        let handle = tokio::spawn(async move {
            let result = match output_mode {
                AggOutputMode::FinalMerged => {
                    execute_final_agg_batch(
                        splits,
                        warmup_done,
                        schema.clone(),
                        exec_ctx,
                        local_runtime_factory.clone(),
                        context,
                        sync_pool,
                    )
                    .await
                }
                AggOutputMode::PartialStates => {
                    let split = splits.get(partition).cloned().ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "invalid agg split partition {partition}"
                        ))
                    });
                    match split {
                        Ok(split) => {
                            let warmup_done = Arc::clone(&warmup_done[partition]);
                            execute_partial_state_agg_batch(
                                split,
                                warmup_done,
                                schema.clone(),
                                exec_ctx,
                                local_runtime_factory.clone(),
                                context,
                                sync_pool,
                            )
                            .await
                        }
                        Err(err) => Err(err),
                    }
                }
                AggOutputMode::NodeTablePartial => {
                    let split = splits.get(partition).cloned().ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "invalid agg split partition {partition}"
                        ))
                    });
                    match split {
                        Ok(split) => {
                            let warmup_done = Arc::clone(&warmup_done[partition]);
                            let nested_spec = nested_spec.as_ref().cloned().ok_or_else(|| {
                                DataFusionError::Internal(
                                    "NodeTablePartial requires nested_spec".into(),
                                )
                            });
                            match nested_spec {
                                Ok(spec) => {
                                    execute_node_table_partial_batch(
                                        split,
                                        warmup_done,
                                        schema.clone(),
                                        exec_ctx,
                                        spec,
                                        local_runtime_factory.clone(),
                                        context,
                                        sync_pool,
                                    )
                                    .await
                                }
                                Err(err) => Err(err),
                            }
                        }
                        Err(err) => Err(err),
                    }
                }
            };

            if cancelled_task.load(Ordering::Relaxed) {
                return;
            }

            match result {
                Ok(Some(batch)) => {
                    let _ = tx.send(Ok(batch)).await;
                }
                Ok(None) => {}
                Err(err) => {
                    let _ = tx.send(Err(err)).await;
                }
            }
        });
        let guard = AbortOnDrop { handle, cancelled };

        let stream = futures::stream::unfold((rx, guard), |(mut rx, guard)| async move {
            rx.recv().await.map(|batch| (batch, (rx, guard)))
        })
        .map(move |result| {
            if let Ok(ref batch) = result {
                metrics_guard.0.record_output(batch.num_rows());
            }
            result
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.output_schema.clone(),
            stream,
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "AggDataSource(mode={:?}, aggs={}, splits={}, query={})",
            self.output_mode,
            self.aggregations.len(),
            self.splits.len(),
            !self.raw_queries.is_empty()
                || self.pre_built_query.is_some()
                || !self.fast_field_filter_exprs.is_empty(),
        )
    }

    fn output_partitioning(&self) -> Partitioning {
        let partitions = match self.output_mode {
            AggOutputMode::FinalMerged => 1,
            AggOutputMode::PartialStates | AggOutputMode::NodeTablePartial => {
                self.splits.len().max(1)
            }
        };
        Partitioning::UnknownPartitioning(partitions)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.output_schema.clone())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.output_schema))
    }

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        None
    }

    fn fetch(&self) -> Option<usize> {
        None
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.metrics.clone()
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        let results: Vec<PushedDown> = filters.iter().map(|_| PushedDown::No).collect();
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            results,
        ))
    }
}

fn cancelled_error() -> DataFusionError {
    DataFusionError::Execution("aggregation cancelled".into())
}

fn ensure_not_cancelled(cancelled: &AtomicBool) -> Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        return Err(cancelled_error());
    }
    Ok(())
}

async fn execute_final_agg_batch(
    splits: Vec<AggSplitPlan>,
    warmup_done: Vec<Arc<OnceCell<()>>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    context: Arc<datafusion::execution::TaskContext>,
    sync_pool: crate::sync_exec::SyncExecutionPoolRef,
) -> Result<Option<RecordBatch>> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;

    if splits.len() == 1 {
        let split = splits
            .into_iter()
            .next()
            .ok_or_else(|| DataFusionError::Internal("AggDataSource missing split".into()))?;
        let warmup_done = warmup_done.into_iter().next().ok_or_else(|| {
            DataFusionError::Internal("AggDataSource missing warmup state".into())
        })?;
        let batch = execute_single_split_agg_batch(
            split,
            warmup_done,
            schema,
            exec_ctx,
            local_runtime_factory,
            context,
            Arc::clone(&sync_pool),
        )
        .await?;
        return Ok(Some(batch));
    }

    let mut partials = Vec::with_capacity(splits.len());
    for (split, warmup_done) in splits.into_iter().zip(warmup_done) {
        ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
        partials.push(
            execute_split_intermediate_agg(
                split,
                warmup_done,
                exec_ctx.clone(),
                local_runtime_factory.as_ref(),
                context.as_ref(),
                &*sync_pool,
            )
            .await?,
        );
    }

    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let aggs = Arc::clone(&exec_ctx.aggs);
    let batch = crate::sync_exec::run_sync(&*sync_pool, move || {
        let results = crate::unified::agg_exec::merge_intermediate_agg_results(partials, &aggs)?;
        crate::unified::agg_exec::agg_results_to_output_batch(&results, &aggs, &schema)
    })
    .await?;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    Ok(Some(batch))
}

async fn execute_partial_state_agg_batch(
    split: AggSplitPlan,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    context: Arc<datafusion::execution::TaskContext>,
    sync_pool: crate::sync_exec::SyncExecutionPoolRef,
) -> Result<Option<RecordBatch>> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let batch = execute_single_split_partial_state_batch(
        split,
        warmup_done,
        schema,
        exec_ctx,
        local_runtime_factory,
        context,
        sync_pool,
    )
    .await?;
    Ok(Some(batch))
}

async fn warmup_for_agg(
    split: &AggSplitPlan,
    prepared: &PreparedSplit,
    raw_queries: &[(String, String)],
    fast_field_filter_exprs: &[Expr],
    aggs: &Aggregations,
    warmup_done: &OnceCell<()>,
    cancelled: &AtomicBool,
) -> Result<()> {
    if !split_needs_warmup(split) {
        return Ok(());
    }

    ensure_not_cancelled(cancelled)?;
    let searcher = prepared.searcher().clone();
    let tantivy_schema = prepared.index().schema();
    let raw_queries = raw_queries.to_vec();
    let fast_field_filter_exprs = fast_field_filter_exprs.to_vec();
    let agg_fields = extract_agg_field_names(aggs);

    warmup_done
        .get_or_try_init(|| async move {
            let queried_fields: Vec<tantivy::schema::Field> = raw_queries
                .iter()
                .filter_map(|(field_name, _)| tantivy_schema.get_field(field_name).ok())
                .collect();
            if !queried_fields.is_empty() {
                crate::warmup::warmup_inverted_index(&searcher, &queried_fields).await?;
            }

            let mut fast_field_names: std::collections::BTreeSet<String> = agg_fields
                .into_iter()
                .filter(|field_name| tantivy_schema.get_field(field_name).is_ok())
                .collect();
            fast_field_names.extend(crate::warmup::fast_field_filter_field_names(
                &tantivy_schema,
                &fast_field_filter_exprs,
            )?);
            if !fast_field_names.is_empty() {
                let fast_field_names: Vec<String> = fast_field_names.into_iter().collect();
                let fast_field_name_refs: Vec<&str> =
                    fast_field_names.iter().map(String::as_str).collect();
                crate::warmup::warmup_fast_fields_by_name(&searcher, &fast_field_name_refs).await?;
            }

            Ok::<(), DataFusionError>(())
        })
        .await?;

    ensure_not_cancelled(cancelled)?;
    Ok(())
}

async fn execute_split_intermediate_agg(
    split: AggSplitPlan,
    warmup_done: Arc<OnceCell<()>>,
    exec_ctx: AggExecutionContext,
    local_runtime_factory: Option<&SplitRuntimeFactoryRef>,
    context: &datafusion::execution::TaskContext,
    sync_pool: &dyn crate::sync_exec::SyncExecutionPool,
) -> Result<IntermediateAggregationResults> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let prepared = prepare_split(&split, local_runtime_factory, context).await?;
    warmup_for_agg(
        &split,
        prepared.as_ref(),
        &exec_ctx.raw_queries,
        &exec_ctx.fast_field_filter_exprs,
        &exec_ctx.aggs,
        warmup_done.as_ref(),
        exec_ctx.cancelled.as_ref(),
    )
    .await?;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let AggExecutionContext {
        aggs,
        raw_queries,
        pre_built_query,
        fast_field_filter_exprs,
        cancelled: _,
    } = exec_ctx;

    crate::sync_exec::run_sync(sync_pool, move || {
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => {
                build_split_fast_field_query(&fast_field_filter_exprs, &prepared.index().schema())
            }
        };
        let query = build_combined_query(
            prepared.index(),
            split_fast_field_query.as_ref(),
            &raw_queries,
        )?;
        crate::unified::agg_exec::execute_tantivy_intermediate_agg_with_reader(
            prepared.index(),
            &aggs,
            query.as_ref(),
            Some(prepared.reader()),
        )
    })
    .await
}

async fn execute_single_split_agg_batch(
    split: AggSplitPlan,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    context: Arc<datafusion::execution::TaskContext>,
    sync_pool: crate::sync_exec::SyncExecutionPoolRef,
) -> Result<RecordBatch> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let prepared = prepare_split(&split, local_runtime_factory.as_ref(), context.as_ref()).await?;
    warmup_for_agg(
        &split,
        prepared.as_ref(),
        &exec_ctx.raw_queries,
        &exec_ctx.fast_field_filter_exprs,
        &exec_ctx.aggs,
        warmup_done.as_ref(),
        exec_ctx.cancelled.as_ref(),
    )
    .await?;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let AggExecutionContext {
        aggs,
        raw_queries,
        pre_built_query,
        fast_field_filter_exprs,
        cancelled: _,
    } = exec_ctx;

    crate::sync_exec::run_sync(&*sync_pool, move || {
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => {
                build_split_fast_field_query(&fast_field_filter_exprs, &prepared.index().schema())
            }
        };
        let query = build_combined_query(
            prepared.index(),
            split_fast_field_query.as_ref(),
            &raw_queries,
        )?;
        crate::unified::agg_exec::execute_tantivy_agg_with_reader(
            prepared.index(),
            &aggs,
            query.as_ref(),
            &schema,
            Some(prepared.reader()),
        )
    })
    .await
}

async fn execute_single_split_partial_state_batch(
    split: AggSplitPlan,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    context: Arc<datafusion::execution::TaskContext>,
    sync_pool: crate::sync_exec::SyncExecutionPoolRef,
) -> Result<RecordBatch> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let prepared = prepare_split(&split, local_runtime_factory.as_ref(), context.as_ref()).await?;
    warmup_for_agg(
        &split,
        prepared.as_ref(),
        &exec_ctx.raw_queries,
        &exec_ctx.fast_field_filter_exprs,
        &exec_ctx.aggs,
        warmup_done.as_ref(),
        exec_ctx.cancelled.as_ref(),
    )
    .await?;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let AggExecutionContext {
        aggs,
        raw_queries,
        pre_built_query,
        fast_field_filter_exprs,
        cancelled: _,
    } = exec_ctx;

    crate::sync_exec::run_sync(&*sync_pool, move || {
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => {
                build_split_fast_field_query(&fast_field_filter_exprs, &prepared.index().schema())
            }
        };
        let query = build_combined_query(
            prepared.index(),
            split_fast_field_query.as_ref(),
            &raw_queries,
        )?;
        let results = crate::unified::agg_exec::execute_tantivy_agg_results_with_reader(
            prepared.index(),
            &aggs,
            query.as_ref(),
            Some(prepared.reader()),
        )?;
        crate::unified::agg_exec::agg_results_to_partial_state_batch(&results, &aggs, &schema)
    })
    .await
}

async fn execute_node_table_partial_batch(
    split: AggSplitPlan,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
    nested_spec: Arc<NestedApproxAggSpec>,
    local_runtime_factory: Option<SplitRuntimeFactoryRef>,
    context: Arc<datafusion::execution::TaskContext>,
    sync_pool: crate::sync_exec::SyncExecutionPoolRef,
) -> Result<Option<RecordBatch>> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let prepared = prepare_split(&split, local_runtime_factory.as_ref(), context.as_ref()).await?;
    warmup_for_agg(
        &split,
        prepared.as_ref(),
        &exec_ctx.raw_queries,
        &exec_ctx.fast_field_filter_exprs,
        &exec_ctx.aggs,
        warmup_done.as_ref(),
        exec_ctx.cancelled.as_ref(),
    )
    .await?;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let AggExecutionContext {
        aggs,
        raw_queries,
        pre_built_query,
        fast_field_filter_exprs,
        cancelled: _,
    } = exec_ctx;

    crate::sync_exec::run_sync(&*sync_pool, move || {
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => {
                build_split_fast_field_query(&fast_field_filter_exprs, &prepared.index().schema())
            }
        };
        let query = build_combined_query(
            prepared.index(),
            split_fast_field_query.as_ref(),
            &raw_queries,
        )?;
        let intermediate = crate::unified::agg_exec::execute_tantivy_intermediate_agg_with_reader(
            prepared.index(),
            &aggs,
            query.as_ref(),
            Some(prepared.reader()),
        )?;
        let batch = crate::nested_agg::node_table::intermediate_results_to_node_table_batch(
            &intermediate,
            &nested_spec,
            &schema,
        )?;
        Ok(Some(batch))
    })
    .await
}

/// Extract all field names referenced by an `Aggregations` tree.
fn extract_agg_field_names(aggs: &Aggregations) -> Vec<String> {
    let mut fields: Vec<String> = tantivy::aggregation::agg_req::get_fast_field_names(aggs)
        .into_iter()
        .collect();
    fields.sort();
    fields
}
