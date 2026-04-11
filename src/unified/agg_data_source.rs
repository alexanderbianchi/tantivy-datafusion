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

use datafusion::logical_expr::Expr;

use crate::index_opener::IndexOpener;
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

#[derive(Debug)]
pub struct AggDataSource {
    split_openers: Vec<Arc<dyn IndexOpener>>,
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
    /// Whether this source emits final aggregate rows or partial aggregate
    /// state rows for a downstream `AggregateExec(Final*)`.
    output_mode: AggOutputMode,
    /// Ensures warmup runs at most once per split.
    warmup_done: Vec<Arc<OnceCell<()>>>,
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOutputMode {
    FinalMerged,
    PartialStates,
}

#[derive(Clone)]
struct AggExecutionContext {
    aggs: Arc<Aggregations>,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
    cancelled: Arc<AtomicBool>,
}

impl AggDataSource {
    pub fn new(
        opener: Arc<dyn IndexOpener>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        Self::from_split_openers(
            vec![opener],
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
        )
    }

    pub fn from_split_openers(
        split_openers: Vec<Arc<dyn IndexOpener>>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let warmup_done = split_openers
            .iter()
            .map(|_| Arc::new(OnceCell::new()))
            .collect();
        Self {
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            output_mode: AggOutputMode::FinalMerged,
            warmup_done,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    pub fn from_split_openers_partial_states(
        split_openers: Vec<Arc<dyn IndexOpener>>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        fast_field_filter_exprs: Vec<Expr>,
    ) -> Self {
        let warmup_done = split_openers
            .iter()
            .map(|_| Arc::new(OnceCell::new()))
            .collect();
        Self {
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            output_mode: AggOutputMode::PartialStates,
            warmup_done,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// Access the sole index opener.
    pub fn opener(&self) -> &Arc<dyn IndexOpener> {
        self.single_split_opener()
            .expect("AggDataSource spans multiple splits; use split_openers() instead")
    }

    /// Access all split openers.
    pub fn split_openers(&self) -> &[Arc<dyn IndexOpener>] {
        &self.split_openers
    }

    /// Access the sole opener when this aggregation spans a single split.
    pub fn single_split_opener(&self) -> Option<&Arc<dyn IndexOpener>> {
        match self.split_openers.as_slice() {
            [opener] => Some(opener),
            _ => None,
        }
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
}

impl DataSource for AggDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics_guard = MetricsGuard(BaselineMetrics::new(&self.metrics, partition));
        let schema = self.output_schema.clone();
        match self.output_mode {
            AggOutputMode::FinalMerged if partition != 0 => {
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::empty(),
                )));
            }
            AggOutputMode::PartialStates if partition >= self.split_openers.len() => {
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    schema,
                    stream::empty(),
                )));
            }
            _ => {}
        }

        let raw_queries = self.raw_queries.clone();
        let split_openers = self.split_openers.clone();
        let warmup_done = self.warmup_done.clone();
        let pre_built_query = if self.split_openers.len() == 1 {
            self.pre_built_query
                .as_ref()
                .map(|q| Arc::from(q.box_clone()))
        } else {
            None
        };
        let fast_field_filter_exprs = self.fast_field_filter_exprs.clone();
        let output_mode = self.output_mode;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(1);

        let cancelled_task = Arc::clone(&cancelled);
        let exec_ctx = AggExecutionContext {
            aggs: Arc::clone(&self.aggregations),
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            cancelled: Arc::clone(&cancelled_task),
        };
        let handle = tokio::spawn(async move {
            let result = match output_mode {
                AggOutputMode::FinalMerged => {
                    execute_final_agg_batch(split_openers, warmup_done, schema.clone(), exec_ctx)
                        .await
                }
                AggOutputMode::PartialStates => {
                    let opener = Arc::clone(&split_openers[partition]);
                    let warmup_done = Arc::clone(&warmup_done[partition]);
                    execute_partial_state_agg_batch(opener, warmup_done, schema.clone(), exec_ctx)
                        .await
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
            self.split_openers.len(),
            !self.raw_queries.is_empty()
                || self.pre_built_query.is_some()
                || !self.fast_field_filter_exprs.is_empty(),
        )
    }

    fn output_partitioning(&self) -> Partitioning {
        let partitions = match self.output_mode {
            AggOutputMode::FinalMerged => 1,
            AggOutputMode::PartialStates => self.split_openers.len().max(1),
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
        // No filter pushdown support — aggregation runs on the full query result.
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
    split_openers: Vec<Arc<dyn IndexOpener>>,
    warmup_done: Vec<Arc<OnceCell<()>>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
) -> Result<Option<RecordBatch>> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;

    if split_openers.len() == 1 {
        let opener = split_openers.into_iter().next().ok_or_else(|| {
            DataFusionError::Internal("AggDataSource missing split opener".into())
        })?;
        let warmup_done = warmup_done.into_iter().next().ok_or_else(|| {
            DataFusionError::Internal("AggDataSource missing warmup state".into())
        })?;
        let batch = execute_single_split_agg_batch(opener, warmup_done, schema, exec_ctx).await?;
        return Ok(Some(batch));
    }

    let mut partials = Vec::with_capacity(split_openers.len());
    for (opener, warmup_done) in split_openers.into_iter().zip(warmup_done) {
        ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
        partials.push(execute_split_intermediate_agg(opener, warmup_done, exec_ctx.clone()).await?);
    }

    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let aggs = Arc::clone(&exec_ctx.aggs);
    let batch = tokio::task::spawn_blocking(move || {
        let results = crate::unified::agg_exec::merge_intermediate_agg_results(partials, &aggs)?;
        crate::unified::agg_exec::agg_results_to_output_batch(&results, &aggs, &schema)
    })
    .await
    .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))??;
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    Ok(Some(batch))
}

async fn execute_partial_state_agg_batch(
    opener: Arc<dyn IndexOpener>,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
) -> Result<Option<RecordBatch>> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let batch =
        execute_single_split_partial_state_batch(opener, warmup_done, schema, exec_ctx).await?;
    Ok(Some(batch))
}

async fn warmup_for_agg(
    opener: &Arc<dyn IndexOpener>,
    index: &tantivy::Index,
    raw_queries: &[(String, String)],
    fast_field_filter_exprs: &[Expr],
    aggs: &Aggregations,
    warmup_done: &OnceCell<()>,
    cancelled: &AtomicBool,
) -> Result<()> {
    if !opener.needs_warmup() {
        return Ok(());
    }

    ensure_not_cancelled(cancelled)?;
    let index_ref = index.clone();
    let raw_queries = raw_queries.to_vec();
    let fast_field_filter_exprs = fast_field_filter_exprs.to_vec();
    let agg_fields = extract_agg_field_names(aggs);

    warmup_done
        .get_or_try_init(|| async move {
            let queried_fields: Vec<tantivy::schema::Field> = raw_queries
                .iter()
                .filter_map(|(field_name, _)| index_ref.schema().get_field(field_name).ok())
                .collect();
            if !queried_fields.is_empty() {
                crate::warmup::warmup_inverted_index(&index_ref, &queried_fields).await?;
            }

            let mut fast_field_names: std::collections::BTreeSet<String> = agg_fields
                .into_iter()
                .filter(|field_name| index_ref.schema().get_field(field_name).is_ok())
                .collect();
            fast_field_names.extend(crate::warmup::fast_field_filter_field_names(
                &index_ref.schema(),
                &fast_field_filter_exprs,
            )?);
            if !fast_field_names.is_empty() {
                let fast_field_names: Vec<String> = fast_field_names.into_iter().collect();
                let fast_field_name_refs: Vec<&str> =
                    fast_field_names.iter().map(String::as_str).collect();
                crate::warmup::warmup_fast_fields_by_name(&index_ref, &fast_field_name_refs)
                    .await?;
            }

            Ok::<(), DataFusionError>(())
        })
        .await?;

    ensure_not_cancelled(cancelled)?;
    Ok(())
}

async fn execute_split_intermediate_agg(
    opener: Arc<dyn IndexOpener>,
    warmup_done: Arc<OnceCell<()>>,
    exec_ctx: AggExecutionContext,
) -> Result<IntermediateAggregationResults> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let index = opener.open().await?;
    warmup_for_agg(
        &opener,
        &index,
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

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema()),
        };
        let query = build_combined_query(&index, split_fast_field_query.as_ref(), &raw_queries)?;
        crate::unified::agg_exec::execute_tantivy_intermediate_agg_with_reader(
            &index,
            &aggs,
            query.as_ref(),
            Some(&reader),
        )
    })
    .await
    .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
}

async fn execute_single_split_agg_batch(
    opener: Arc<dyn IndexOpener>,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
) -> Result<arrow::record_batch::RecordBatch> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let index = opener.open().await?;
    warmup_for_agg(
        &opener,
        &index,
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

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema()),
        };
        let query = build_combined_query(&index, split_fast_field_query.as_ref(), &raw_queries)?;
        crate::unified::agg_exec::execute_tantivy_agg_with_reader(
            &index,
            &aggs,
            query.as_ref(),
            &schema,
            Some(&reader),
        )
    })
    .await
    .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
}

async fn execute_single_split_partial_state_batch(
    opener: Arc<dyn IndexOpener>,
    warmup_done: Arc<OnceCell<()>>,
    schema: SchemaRef,
    exec_ctx: AggExecutionContext,
) -> Result<arrow::record_batch::RecordBatch> {
    ensure_not_cancelled(exec_ctx.cancelled.as_ref())?;
    let index = opener.open().await?;
    warmup_for_agg(
        &opener,
        &index,
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

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema()),
        };
        let query = build_combined_query(&index, split_fast_field_query.as_ref(), &raw_queries)?;
        let results = crate::unified::agg_exec::execute_tantivy_agg_results_with_reader(
            &index,
            &aggs,
            query.as_ref(),
            Some(&reader),
        )?;
        crate::unified::agg_exec::agg_results_to_partial_state_batch(&results, &aggs, &schema)
    })
    .await
    .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
}

/// Extract all field names referenced by an `Aggregations` tree.
fn extract_agg_field_names(aggs: &Aggregations) -> Vec<String> {
    let mut fields: Vec<String> = tantivy::aggregation::agg_req::get_fast_field_names(aggs)
        .into_iter()
        .collect();
    fields.sort();
    fields
}
