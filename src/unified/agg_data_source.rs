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
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::projection::ProjectionExprs;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, Partitioning, SendableRecordBatchStream};
use datafusion_physical_expr::PhysicalExpr;
use futures::stream::{self, StreamExt};
use tantivy::aggregation::agg_req::Aggregations;

use crate::index_opener::IndexOpener;
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
    opener: Arc<dyn IndexOpener>,
    /// Tantivy aggregation specification (terms + metric sub-aggs).
    aggregations: Arc<Aggregations>,
    /// Output schema matching the AggregateExec this replaces.
    output_schema: SchemaRef,
    /// Raw full-text queries deferred to execution time.
    raw_queries: Vec<(String, String)>,
    /// Pre-built tantivy queries from fast field filter conversion.
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

impl AggDataSource {
    pub fn new(
        opener: Arc<dyn IndexOpener>,
        aggregations: Arc<Aggregations>,
        output_schema: SchemaRef,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    ) -> Self {
        Self {
            opener,
            aggregations,
            output_schema,
            raw_queries,
            pre_built_query,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DataSource for AggDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics_guard = MetricsGuard(BaselineMetrics::new(&self.metrics, partition));
        let opener = self.opener.clone();
        let raw_queries = self.raw_queries.clone();
        let pre_built_query = self
            .pre_built_query
            .as_ref()
            .map(|q| Arc::from(q.box_clone()));
        let aggs = self.aggregations.clone();
        let schema = self.output_schema.clone();

        // Only partition 0 runs the aggregation (single output partition)
        if partition != 0 {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                schema,
                stream::empty(),
            )));
        }

        let stream = stream::once(async move {
            let index = opener.open().await?;

            // Warm up inverted index for queried text fields
            let queried_fields: Vec<tantivy::schema::Field> = raw_queries
                .iter()
                .filter_map(|(field_name, _)| index.schema().get_field(field_name).ok())
                .collect();
            if !queried_fields.is_empty() {
                crate::warmup::warmup_inverted_index(&index, &queried_fields).await?;
            }

            // Warm fast fields for aggregation (sum, avg, min, max read columnar data)
            crate::warmup::warmup_fast_fields(&index).await?;

            // Move sync work (including query building) to blocking thread.
            tokio::task::spawn_blocking(move || {
                let query = build_combined_query(&index, pre_built_query.as_ref(), &raw_queries)?;
                crate::unified::agg_exec::execute_tantivy_agg(
                    &index,
                    &aggs,
                    query.as_ref(),
                    &schema,
                )
            })
            .await
            .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
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
            "AggDataSource(aggs={}, query={})",
            self.aggregations.len(),
            !self.raw_queries.is_empty() || self.pre_built_query.is_some(),
        )
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
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
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(results))
    }
}
