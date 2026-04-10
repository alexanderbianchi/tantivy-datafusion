/// A DataSource that runs tantivy's native aggregation on query-filtered docs.
///
/// Created by the `AggPushdown` optimizer rule when it detects an
/// `AggregateExec` above a `SingleTableDataSource`. Preserves the full
/// query context (FTS + fast field filters) from the original scan.
use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
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
use futures::future::try_join_all;
use futures::stream::{self, StreamExt};
use tantivy::aggregation::agg_req::Aggregations;
use tantivy::aggregation::intermediate_agg_result::IntermediateAggregationResults;

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
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOutputMode {
    FinalMerged,
    PartialStates,
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
        Self {
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            output_mode: AggOutputMode::FinalMerged,
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
        Self {
            split_openers,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            output_mode: AggOutputMode::PartialStates,
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
        let split_openers = self.split_openers.clone();
        let raw_queries = self.raw_queries.clone();
        let pre_built_query = if self.split_openers.len() == 1 {
            self.pre_built_query
                .as_ref()
                .map(|q| Arc::from(q.box_clone()))
        } else {
            None
        };
        let fast_field_filter_exprs = self.fast_field_filter_exprs.clone();
        let aggs = self.aggregations.clone();
        let schema = self.output_schema.clone();

        let stream = match self.output_mode {
            AggOutputMode::FinalMerged => {
                if partition != 0 {
                    return Ok(Box::pin(RecordBatchStreamAdapter::new(
                        schema,
                        stream::empty(),
                    )));
                }

                stream::once(async move {
                    if split_openers.len() == 1 {
                        return execute_single_split_agg_batch(
                            split_openers.into_iter().next().unwrap(),
                            aggs,
                            schema,
                            raw_queries,
                            pre_built_query,
                            fast_field_filter_exprs,
                        )
                        .await;
                    }

                    let partials = try_join_all(split_openers.into_iter().map(|opener| {
                        let raw_queries = raw_queries.clone();
                        let pre_built_query = pre_built_query.clone();
                        let fast_field_filter_exprs = fast_field_filter_exprs.clone();
                        let aggs = aggs.clone();
                        async move {
                            execute_split_intermediate_agg(
                                opener,
                                aggs,
                                raw_queries,
                                pre_built_query,
                                fast_field_filter_exprs,
                            )
                            .await
                        }
                    }))
                    .await?;

                    tokio::task::spawn_blocking(move || {
                        let results = crate::unified::agg_exec::merge_intermediate_agg_results(
                            partials, &aggs,
                        )?;
                        crate::unified::agg_exec::agg_results_to_output_batch(
                            &results, &aggs, &schema,
                        )
                    })
                    .await
                    .map_err(|e| {
                        DataFusionError::Internal(format!("spawn_blocking join error: {e}"))
                    })?
                })
                .boxed()
            }
            AggOutputMode::PartialStates => {
                if partition >= split_openers.len() {
                    return Ok(Box::pin(RecordBatchStreamAdapter::new(
                        schema,
                        stream::empty(),
                    )));
                }

                let opener = Arc::clone(&split_openers[partition]);
                stream::once(async move {
                    execute_single_split_partial_state_batch(
                        opener,
                        aggs,
                        schema,
                        raw_queries,
                        pre_built_query,
                        fast_field_filter_exprs,
                    )
                    .await
                })
                .boxed()
            }
        }
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

async fn execute_split_intermediate_agg(
    opener: Arc<dyn IndexOpener>,
    aggs: Arc<Aggregations>,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
) -> Result<IntermediateAggregationResults> {
    let index = opener.open().await?;

    if opener.needs_warmup() {
        let queried_fields: Vec<tantivy::schema::Field> = raw_queries
            .iter()
            .filter_map(|(field_name, _)| index.schema().get_field(field_name).ok())
            .collect();
        if !queried_fields.is_empty() {
            crate::warmup::warmup_inverted_index(&index, &queried_fields).await?;
        }
        let agg_fields = extract_agg_field_names(&aggs);
        let agg_field_refs: Vec<&str> = agg_fields
            .iter()
            .filter(|field_name| index.schema().get_field(field_name).is_ok())
            .map(|field_name| field_name.as_str())
            .collect();
        if !agg_field_refs.is_empty() {
            crate::warmup::warmup_fast_fields_by_name(&index, &agg_field_refs).await?;
        }
    }

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema())?,
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
    aggs: Arc<Aggregations>,
    schema: SchemaRef,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
) -> Result<arrow::record_batch::RecordBatch> {
    let index = opener.open().await?;

    if opener.needs_warmup() {
        let queried_fields: Vec<tantivy::schema::Field> = raw_queries
            .iter()
            .filter_map(|(field_name, _)| index.schema().get_field(field_name).ok())
            .collect();
        if !queried_fields.is_empty() {
            crate::warmup::warmup_inverted_index(&index, &queried_fields).await?;
        }
        let agg_fields = extract_agg_field_names(&aggs);
        let agg_field_refs: Vec<&str> = agg_fields
            .iter()
            .filter(|field_name| index.schema().get_field(field_name).is_ok())
            .map(|field_name| field_name.as_str())
            .collect();
        if !agg_field_refs.is_empty() {
            crate::warmup::warmup_fast_fields_by_name(&index, &agg_field_refs).await?;
        }
    }

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema())?,
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
    aggs: Arc<Aggregations>,
    schema: SchemaRef,
    raw_queries: Vec<(String, String)>,
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    fast_field_filter_exprs: Vec<Expr>,
) -> Result<arrow::record_batch::RecordBatch> {
    let index = opener.open().await?;

    if opener.needs_warmup() {
        let queried_fields: Vec<tantivy::schema::Field> = raw_queries
            .iter()
            .filter_map(|(field_name, _)| index.schema().get_field(field_name).ok())
            .collect();
        if !queried_fields.is_empty() {
            crate::warmup::warmup_inverted_index(&index, &queried_fields).await?;
        }
        let agg_fields = extract_agg_field_names(&aggs);
        let agg_field_refs: Vec<&str> = agg_fields
            .iter()
            .filter(|field_name| index.schema().get_field(field_name).is_ok())
            .map(|field_name| field_name.as_str())
            .collect();
        if !agg_field_refs.is_empty() {
            crate::warmup::warmup_fast_fields_by_name(&index, &agg_field_refs).await?;
        }
    }

    tokio::task::spawn_blocking(move || {
        let reader = index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        let split_fast_field_query = match pre_built_query {
            Some(query) => Some(query),
            None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema())?,
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
