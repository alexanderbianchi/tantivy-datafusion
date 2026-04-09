use std::any::Any;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result, Statistics};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::projection::ProjectionExprs;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, Partitioning, SendableRecordBatchStream};
use futures::stream::StreamExt;
use tantivy::query::RangeQuery;
use tantivy::schema::{FieldType, IndexRecordOption, Schema as TantivySchema, Term};
use tantivy::{DateTime, Document, Index};

use crate::fast_field_reader::{DictCache, read_segment_fast_fields_to_batch};
use crate::full_text_udf::extract_full_text_call;
use crate::index_opener::{DirectIndexOpener, IndexOpener};
use crate::schema_mapping::{tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index};
use crate::util::{build_combined_query, collect_matching_docs};

/// Guard that calls `BaselineMetrics::done()` on drop so elapsed time is
/// recorded even when the stream is cancelled.
struct MetricsGuard(BaselineMetrics);
impl Drop for MetricsGuard {
    fn drop(&mut self) {
        self.0.done();
    }
}

/// Guard that aborts a spawned task when dropped.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

// ---------------------------------------------------------------------------
// Per-partition statistics for partition pruning
// ---------------------------------------------------------------------------

/// Per-segment (partition) statistics derived from tantivy fast field metadata.
///
/// Used by `partition_statistics()` to report column min/max values so that
/// DataFusion's partition pruning can skip segments whose value ranges do not
/// overlap the query's WHERE clause.
#[derive(Debug, Clone)]
struct PartitionStat {
    num_rows: usize,
    /// Whether this segment has deleted documents. When true, `num_rows` is
    /// an estimate (alive count) rather than an exact value.
    has_deletes: bool,
    /// Column name -> (min, max) as `ScalarValue`.
    column_stats: Vec<(String, Option<ScalarValue>, Option<ScalarValue>)>,
}

/// Eagerly compute per-segment partition statistics from a local tantivy index.
///
/// Reads fast field min/max from each segment's columnar data. This is cheap:
/// it only reads column metadata, not document values.
fn compute_partition_stats(
    index: &Index,
    ff_schema: &SchemaRef,
) -> Result<Vec<Option<PartitionStat>>> {
    let reader = index
        .reader()
        .map_err(|e| DataFusionError::Internal(format!("open reader for stats: {e}")))?;
    let searcher = reader.searcher();

    let mut stats = Vec::with_capacity(searcher.segment_readers().len());
    for seg_reader in searcher.segment_readers() {
        let num_rows = seg_reader.max_doc() as usize;
        // Subtract deleted docs for an accurate alive count.
        let alive = seg_reader
            .alive_bitset()
            .map_or(num_rows, |b| b.num_alive_docs());

        let fast_fields = seg_reader.fast_fields();
        let mut column_stats = Vec::new();

        for field in ff_schema.fields() {
            let name = field.name();
            if name == "_doc_id" || name == "_segment_ord" {
                continue;
            }

            let (min_val, max_val) = match field.data_type() {
                DataType::UInt64 => {
                    match fast_fields.u64(name) {
                        Ok(col) if col.values.num_vals() > 0 => (
                            Some(ScalarValue::UInt64(Some(col.min_value()))),
                            Some(ScalarValue::UInt64(Some(col.max_value()))),
                        ),
                        _ => (None, None),
                    }
                }
                DataType::Int64 => {
                    match fast_fields.i64(name) {
                        Ok(col) if col.values.num_vals() > 0 => (
                            Some(ScalarValue::Int64(Some(col.min_value()))),
                            Some(ScalarValue::Int64(Some(col.max_value()))),
                        ),
                        _ => (None, None),
                    }
                }
                DataType::Float64 => {
                    match fast_fields.f64(name) {
                        Ok(col) if col.values.num_vals() > 0 => (
                            Some(ScalarValue::Float64(Some(col.min_value()))),
                            Some(ScalarValue::Float64(Some(col.max_value()))),
                        ),
                        _ => (None, None),
                    }
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    match fast_fields.date(name) {
                        Ok(col) if col.values.num_vals() > 0 => (
                            Some(ScalarValue::TimestampMicrosecond(
                                Some(col.min_value().into_timestamp_micros()),
                                None,
                            )),
                            Some(ScalarValue::TimestampMicrosecond(
                                Some(col.max_value().into_timestamp_micros()),
                                None,
                            )),
                        ),
                        _ => (None, None),
                    }
                }
                DataType::Boolean => {
                    match fast_fields.bool(name) {
                        Ok(col) if col.values.num_vals() > 0 => (
                            Some(ScalarValue::Boolean(Some(col.min_value()))),
                            Some(ScalarValue::Boolean(Some(col.max_value()))),
                        ),
                        _ => (None, None),
                    }
                }
                _ => (None, None),
            };

            column_stats.push((name.to_string(), min_val, max_val));
        }

        let has_deletes = seg_reader.alive_bitset().is_some();
        stats.push(Some(PartitionStat {
            num_rows: alive,
            has_deletes,
            column_stats,
        }));
    }
    Ok(stats)
}

/// A single-table DataFusion provider for tantivy indexes.
///
/// Unlike [`UnifiedTantivyTableProvider`](crate::unified_provider::UnifiedTantivyTableProvider),
/// which composes three separate data sources joined by `HashJoinExec`, this
/// provider handles FTS queries, fast field reading, scoring, and document
/// retrieval in a single pass per segment.
///
/// The schema is identical: `[_doc_id, _segment_ord, fast_field_1, ..., fast_field_n, _score, _document]`
///
/// Optimizations:
/// - Skips scoring when `_score` is not projected and no FTS query is used
/// - Skips document store reads when `_document` is not projected
/// - Returns `_score` as null when no FTS query is active
///
/// # Example
/// ```sql
/// SELECT id, price, _score, _document
/// FROM my_index
/// WHERE full_text(category, 'books') AND price > 2
/// ORDER BY _score DESC LIMIT 10
/// ```
pub struct SingleTableProvider {
    opener: Arc<dyn IndexOpener>,
    unified_schema: SchemaRef,
    fast_field_schema: SchemaRef,
    score_column_idx: usize,
    document_column_idx: usize,
}

impl SingleTableProvider {
    /// Create a provider from an already-opened tantivy index.
    pub fn new(index: Index) -> Self {
        let ff_schema = tantivy_schema_to_arrow_from_index(&index);
        Self::from_opener_with_ff_schema(Arc::new(DirectIndexOpener::new(index)), ff_schema)
    }

    /// Create a provider from an [`IndexOpener`] for deferred index opening.
    ///
    /// Uses `tantivy_schema_to_arrow` which cannot detect multi-valued fields
    /// (those require segment inspection). For multi-valued support, use `new()`.
    pub fn from_opener(opener: Arc<dyn IndexOpener>) -> Self {
        let ff_schema = tantivy_schema_to_arrow(&opener.schema());
        Self::from_opener_with_ff_schema(opener, ff_schema)
    }

    pub(crate) fn from_opener_with_ff_schema(
        opener: Arc<dyn IndexOpener>,
        fast_field_schema: SchemaRef,
    ) -> Self {
        let mut unified_fields: Vec<Arc<Field>> = fast_field_schema.fields().to_vec();
        let score_column_idx = unified_fields.len();
        unified_fields.push(Arc::new(Field::new("_score", DataType::Float32, true)));
        let document_column_idx = unified_fields.len();
        unified_fields.push(Arc::new(Field::new("_document", DataType::Utf8, false)));
        let unified_schema = Arc::new(Schema::new(unified_fields));

        Self {
            opener,
            unified_schema,
            fast_field_schema,
            score_column_idx,
            document_column_idx,
        }
    }

}

impl fmt::Debug for SingleTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleTableProvider")
            .field("unified_schema", &self.unified_schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for SingleTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.unified_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let tantivy_schema = self.opener.schema();
        Ok(filters
            .iter()
            .map(|f| {
                if extract_full_text_call(f).is_some() {
                    TableProviderFilterPushDown::Exact
                } else if logical_expr_to_tantivy_query(f, &tantivy_schema).is_some() {
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. Extract full_text() calls and fast field filters from pushed-down filters.
        let tantivy_schema = self.opener.schema();
        let mut raw_queries: Vec<(String, String)> = Vec::new();
        let mut tantivy_queries: Vec<Box<dyn tantivy::query::Query>> = Vec::new();
        let mut fast_field_filter_exprs: Vec<Expr> = Vec::new();
        for filter in filters {
            if let Some((field_name, query_string)) = extract_full_text_call(filter) {
                tantivy_schema.get_field(&field_name).map_err(|e| {
                    DataFusionError::Plan(format!(
                        "full_text: field '{field_name}' not found: {e}"
                    ))
                })?;
                raw_queries.push((field_name, query_string));
            } else if let Some(q) = logical_expr_to_tantivy_query(filter, &tantivy_schema) {
                tantivy_queries.push(q);
                fast_field_filter_exprs.push(filter.clone());
            }
        }
        // Combine fast field tantivy queries into a single pre-built query.
        let pre_built_query: Option<Arc<dyn tantivy::query::Query>> =
            if tantivy_queries.is_empty() {
                None
            } else if tantivy_queries.len() == 1 {
                Some(Arc::from(tantivy_queries.into_iter().next().unwrap()))
            } else {
                Some(Arc::new(tantivy::query::BooleanQuery::intersection(
                    tantivy_queries,
                )))
            };

        // 2. Analyze projection.
        let projected_indices: Vec<usize> = match projection {
            Some(indices) => indices.clone(),
            None => (0..self.unified_schema.fields().len()).collect(),
        };

        let mut needs_score = false;
        let mut needs_document = false;
        let mut ff_indices = Vec::new();

        for &idx in &projected_indices {
            if idx == self.score_column_idx {
                needs_score = true;
            } else if idx == self.document_column_idx {
                needs_document = true;
            } else {
                ff_indices.push(idx);
            }
        }

        // Ensure _doc_id and _segment_ord are in ff_indices (needed internally).
        let doc_id_idx = self.fast_field_schema.index_of("_doc_id").unwrap();
        let seg_ord_idx = self.fast_field_schema.index_of("_segment_ord").unwrap();
        if !ff_indices.contains(&doc_id_idx) {
            ff_indices.push(doc_id_idx);
        }
        if !ff_indices.contains(&seg_ord_idx) {
            ff_indices.push(seg_ord_idx);
        }
        ff_indices.sort();
        ff_indices.dedup();

        let ff_projected_schema = {
            let fields: Vec<Field> = ff_indices
                .iter()
                .map(|&i| self.fast_field_schema.field(i).clone())
                .collect();
            Arc::new(Schema::new(fields))
        };

        let projected_schema = {
            let fields: Vec<Field> = projected_indices
                .iter()
                .map(|&i| self.unified_schema.field(i).clone())
                .collect();
            Arc::new(Schema::new(fields))
        };

        // 3. Compute segments — 1 partition per segment.
        let segment_sizes = self.opener.segment_sizes();
        let num_segments = segment_sizes.len().max(1);

        // 4. Compute per-partition statistics for partition pruning.
        // For DirectIndexOpener (local), we can cheaply read fast field
        // min/max from columnar metadata. For remote openers, stats are
        // unavailable and we fall back to unknown.
        let partition_stats: Vec<Option<PartitionStat>> = if let Some(direct) =
            self.opener.as_any().downcast_ref::<DirectIndexOpener>()
        {
            compute_partition_stats(direct.index(), &self.fast_field_schema)?
        } else {
            vec![None; num_segments]
        };

        let data_source = SingleTableDataSource {
            opener: self.opener.clone(),
            schema: ScanSchema {
                unified: self.unified_schema.clone(),
                projected: projected_schema,
                ff_projected: ff_projected_schema,
                projection: projection.cloned(),
                score_idx: self.score_column_idx,
                document_idx: self.document_column_idx,
                needs_score,
                needs_document,
            },
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs,
            topk: None,
            num_segments,
            partition_stats,
            warmup_done: Arc::new(tokio::sync::OnceCell::new()),
            metrics: ExecutionPlanMetricsSet::new(),
        };

        Ok(Arc::new(DataSourceExec::new(Arc::new(data_source))))
    }
}

// ---------------------------------------------------------------------------
// DataSource implementation
// ---------------------------------------------------------------------------

/// Bundles the eight schema-related fields that travel together.
#[derive(Debug, Clone)]
pub(crate) struct ScanSchema {
    pub(crate) unified: SchemaRef,
    pub(crate) projected: SchemaRef,
    pub(crate) ff_projected: SchemaRef,
    pub(crate) projection: Option<Vec<usize>>,
    pub(crate) score_idx: usize,
    pub(crate) document_idx: usize,
    pub(crate) needs_score: bool,
    pub(crate) needs_document: bool,
}

#[derive(Debug)]
pub struct SingleTableDataSource {
    opener: Arc<dyn IndexOpener>,
    schema: ScanSchema,
    raw_queries: Vec<(String, String)>,
    /// Pre-built tantivy queries from fast field filters converted at scan time.
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    /// Source logical `Expr`s that were successfully converted to tantivy
    /// queries. Stored for serialization: the codec encodes these and the
    /// worker re-derives `pre_built_query` from them.
    fast_field_filter_exprs: Vec<Expr>,
    pub(crate) topk: Option<usize>,
    num_segments: usize,
    /// Per-partition (segment) statistics for partition pruning.
    /// Indexed by partition number. `None` means stats are unavailable for
    /// that partition (e.g. remote opener without metadata).
    partition_stats: Vec<Option<PartitionStat>>,
    /// Ensures warmup runs at most once across all partitions.
    warmup_done: Arc<tokio::sync::OnceCell<()>>,
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

impl SingleTableDataSource {
    /// Construct a `SingleTableDataSource` directly from deserialized codec
    /// fields, bypassing `TableProvider::scan` and `SessionContext`. Used by
    /// `TantivyCodec::try_decode` to reconstruct a `DataSourceExec` on workers.
    pub(crate) fn new_from_codec(
        opener: Arc<dyn IndexOpener>,
        schema: ScanSchema,
        raw_queries: Vec<(String, String)>,
        pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
        topk: Option<usize>,
        num_segments: usize,
    ) -> Self {
        Self {
            opener,
            schema,
            raw_queries,
            pre_built_query,
            fast_field_filter_exprs: Vec::new(),
            topk,
            num_segments,
            partition_stats: vec![None; num_segments],
            warmup_done: Arc::new(tokio::sync::OnceCell::new()),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    fn clone_with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut new = SingleTableDataSource {
            opener: self.opener.clone(),
            schema: self.schema.clone(),
            raw_queries: self.raw_queries.clone(),
            pre_built_query: self.pre_built_query.clone(),
            fast_field_filter_exprs: self.fast_field_filter_exprs.clone(),
            topk: self.topk,
            num_segments: self.num_segments,
            partition_stats: self.partition_stats.clone(),
            warmup_done: self.warmup_done.clone(),
            metrics: self.metrics.clone(),
        };
        f(&mut new);
        new
    }

    /// Access the index opener.
    pub fn opener(&self) -> &Arc<dyn IndexOpener> {
        &self.opener
    }

    /// Access the raw full-text queries.
    pub fn raw_queries(&self) -> &[(String, String)] {
        &self.raw_queries
    }

    /// The number of segments this data source is partitioned over.
    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    /// Access the topk limit.
    pub fn topk(&self) -> Option<usize> {
        self.topk
    }

    /// Whether this data source has an active query.
    pub fn has_query(&self) -> bool {
        !self.raw_queries.is_empty() || self.pre_built_query.is_some()
    }

    /// Create a copy with the topk limit set.
    pub fn with_topk(&self, topk: usize) -> Self {
        self.clone_with(|s| s.topk = Some(topk))
    }

    /// Access the pre-built tantivy query from fast field filters.
    pub fn pre_built_query(&self) -> Option<&Arc<dyn tantivy::query::Query>> {
        self.pre_built_query.as_ref()
    }

    /// Access the source logical `Expr`s that produced `pre_built_query`.
    /// Used by the codec for serialization.
    pub fn fast_field_filter_exprs(&self) -> &[Expr] {
        &self.fast_field_filter_exprs
    }

    /// SingleTableDataSource does not carry pre-set aggregations.
    /// The `AggPushdown` optimizer derives them from the AggregateExec.
    pub fn aggregations(&self) -> Option<&Arc<tantivy::aggregation::agg_req::Aggregations>> {
        None
    }

    /// Aggregate per-partition statistics into overall table statistics.
    ///
    /// - `num_rows` = sum of all partition row counts
    /// - column `min_value` = minimum across all partition minimums
    /// - column `max_value` = maximum across all partition maximums
    fn aggregate_statistics(&self) -> Result<Statistics> {
        use datafusion::common::stats::Precision;
        use datafusion::common::ColumnStatistics;

        // Collect only the partitions that have stats.
        let known: Vec<&PartitionStat> = self
            .partition_stats
            .iter()
            .filter_map(|s| s.as_ref())
            .collect();

        if known.is_empty() {
            return Ok(Statistics::new_unknown(&self.schema.projected));
        }

        let total_rows: usize = known.iter().map(|s| s.num_rows).sum();
        let any_deletes = known.iter().any(|s| s.has_deletes);
        let num_rows = if any_deletes {
            Precision::Inexact(total_rows)
        } else {
            Precision::Exact(total_rows)
        };

        let column_statistics: Vec<ColumnStatistics> = self
            .schema
            .projected
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                let mut overall_min: Precision<ScalarValue> = Precision::Absent;
                let mut overall_max: Precision<ScalarValue> = Precision::Absent;

                for stat in &known {
                    if let Some((_, min_val, max_val)) =
                        stat.column_stats.iter().find(|(n, _, _)| n == name)
                    {
                        if let Some(min_v) = min_val {
                            let p = Precision::Inexact(min_v.clone());
                            overall_min = match overall_min {
                                Precision::Absent => p,
                                prev => prev.min(&p),
                            };
                        }
                        if let Some(max_v) = max_val {
                            let p = Precision::Inexact(max_v.clone());
                            overall_max = match overall_max {
                                Precision::Absent => p,
                                prev => prev.max(&p),
                            };
                        }
                    }
                }

                ColumnStatistics {
                    null_count: Precision::Absent,
                    max_value: overall_max,
                    min_value: overall_min,
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                    byte_size: Precision::Absent,
                }
            })
            .collect();

        Ok(Statistics {
            num_rows,
            total_byte_size: Precision::Absent,
            column_statistics,
        })
    }

    /// Access the projection indices.
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.schema.projection.as_ref()
    }

    /// Whether _score is needed.
    pub fn needs_score(&self) -> bool {
        self.schema.needs_score
    }

    /// Whether _document is needed.
    pub fn needs_document(&self) -> bool {
        self.schema.needs_document
    }
}

impl DataSource for SingleTableDataSource {
    fn open(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let metrics_guard = MetricsGuard(BaselineMetrics::new(&self.metrics, partition));
        let batch_size = context.session_config().batch_size();
        let opener = self.opener.clone();
        let segment_idx = partition;
        let raw_queries = self.raw_queries.clone();
        let ff_projected_schema = self.schema.ff_projected.clone();
        let unified_schema = self.schema.unified.clone();
        let projected_schema = self.schema.projected.clone();
        let projection = self.schema.projection.clone();
        let needs_score = self.schema.needs_score;
        let needs_document = self.schema.needs_document;
        let score_column_idx = self.schema.score_idx;
        let document_column_idx = self.schema.document_idx;
        let topk = self.topk;
        let pre_built_query = self
            .pre_built_query
            .as_ref()
            .map(|q| Arc::from(q.box_clone()));
        let warmup_done = self.warmup_done.clone();
        let needs_warmup = self.opener.needs_warmup();

        let schema = self.schema.projected.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(2);

        let handle = tokio::spawn(async move {
            // Async setup: open index + warmup
            let index = match opener.open().await {
                Ok(i) => i,
                Err(e) => { let _ = tx.send(Err(e)).await; return; }
            };

            // Skip warmup for local/mmap openers — data already accessible.
            if needs_warmup {
            // Warmup runs once across all partitions (best-effort). The first
            // partition to reach this point initialises the OnceCell; others
            // wait briefly or get the cached result.
            //
            // Design note: individual warmup errors are logged but swallowed
            // inside the closure, so the OnceCell always transitions to the
            // initialised state. This is intentional — warmup is advisory and
            // sync reads will still succeed (just slower). Retrying on every
            // partition would be wasteful and unlikely to help with transient
            // errors since the same index/segments are involved.
            {
                let index_ref = index.clone();
                let ff_schema_ref = ff_projected_schema.clone();
                let rq_ref = raw_queries.clone();
                warmup_done.get_or_init(|| async move {
                    let ff_names: Vec<&str> = ff_schema_ref
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect();
                    let _ = crate::warmup::warmup_fast_fields_by_name(&index_ref, &ff_names).await;

                    let queried_fields: Vec<tantivy::schema::Field> = rq_ref
                        .iter()
                        .filter_map(|(field_name, _)| index_ref.schema().get_field(field_name).ok())
                        .collect();
                    if !queried_fields.is_empty() {
                        let _ = crate::warmup::warmup_inverted_index(&index_ref, &queried_fields).await;
                    }
                }).await;
            }

            // Warm up the document store separately when needed.
            if needs_document {
                crate::warmup::warmup_document_store(&index).await.ok();
            }
            } // end if needs_warmup

            // Blocking batch generation — send batches as they're produced.
            let tx_blocking = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                let query =
                    build_combined_query(&index, pre_built_query.as_ref(), &raw_queries)?;
                let cfg = ScanConfig {
                    index,
                    segment_idx,
                    batch_size,
                    ff_projected_schema,
                    unified_schema,
                    projected_schema,
                    projection,
                    score_column_idx,
                    document_column_idx,
                    needs_score,
                    needs_document,
                    topk,
                    query,
                };
                generate_single_table_batch_streaming(
                    &cfg,
                    |batch| tx_blocking.blocking_send(Ok(batch)).is_ok(),
                )
            }).await;

            match result {
                Ok(Err(e)) => { let _ = tx.send(Err(e)).await; }
                Err(e) => { let _ = tx.send(Err(DataFusionError::Internal(format!("spawn_blocking: {e}")))).await; }
                Ok(Ok(())) => {} // tx drops naturally, closing the channel
            }
        });
        let guard = AbortOnDrop(handle);

        // Convert receiver to stream with metrics tracking.
        // The guard is owned by the stream — when the stream is dropped, the
        // spawned task is aborted, preventing leaked background work.
        let stream = futures::stream::unfold((rx, guard), |(mut rx, guard)| async move {
            rx.recv().await.map(|batch| (batch, (rx, guard)))
        });
        let tracked = stream.map(move |result| {
            if let Ok(ref batch) = result {
                metrics_guard.0.record_output(batch.num_rows());
            }
            result
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, tracked)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SingleTableDataSource(segments={}, query={}, score={}, document={}, topk={:?})",
            self.num_segments,
            self.has_query(),
            self.schema.needs_score,
            self.schema.needs_document,
            self.topk,
        )
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.num_segments)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.schema.projected.clone())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        use datafusion::common::stats::Precision;
        use datafusion::common::ColumnStatistics;

        let partition = match partition {
            Some(p) => p,
            None => {
                // Aggregate across all partitions.
                return self.aggregate_statistics();
            }
        };

        let stat = match self.partition_stats.get(partition) {
            Some(Some(s)) => s,
            _ => return Ok(Statistics::new_unknown(&self.schema.projected)),
        };

        let num_rows = if stat.has_deletes {
            Precision::Inexact(stat.num_rows)
        } else {
            Precision::Exact(stat.num_rows)
        };

        let column_statistics: Vec<ColumnStatistics> = self
            .schema
            .projected
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if let Some((_, min_val, max_val)) =
                    stat.column_stats.iter().find(|(n, _, _)| n == name)
                {
                    ColumnStatistics {
                        null_count: Precision::Absent,
                        // Inexact because deleted docs may have held the
                        // actual min/max — the surviving range could be
                        // narrower than what columnar metadata reports.
                        max_value: max_val
                            .clone()
                            .map_or(Precision::Absent, Precision::Inexact),
                        min_value: min_val
                            .clone()
                            .map_or(Precision::Absent, Precision::Inexact),
                        sum_value: Precision::Absent,
                        distinct_count: Precision::Absent,
                        byte_size: Precision::Absent,
                    }
                } else {
                    ColumnStatistics::new_unknown()
                }
            })
            .collect();

        Ok(Statistics {
            num_rows,
            total_byte_size: Precision::Absent,
            column_statistics,
        })
    }

    fn with_fetch(&self, fetch: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let topk = fetch?;
        if !self.schema.needs_score {
            return None; // Only for scored queries (Block-WAND)
        }
        Some(Arc::new(self.clone_with(|ds| {
            ds.topk = Some(topk);
        })))
    }

    fn fetch(&self) -> Option<usize> {
        // Only report fetch guarantee when single partition — per-segment TopK
        // cannot guarantee a global row limit across multiple partitions.
        if self.num_segments <= 1 {
            self.topk
        } else {
            None
        }
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
        // Don't claim to handle physical filters — let DataFusion keep
        // its FilterExec as a safety net. Tantivy-convertible predicates
        // are already pushed at the logical level via supports_filters_pushdown.
        let results: Vec<PushedDown> = filters.iter().map(|_| PushedDown::No).collect();
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(results))
    }
}

// ---------------------------------------------------------------------------
// Core batch generation
// ---------------------------------------------------------------------------

/// Holds immutable context for assembling one `RecordBatch` from a chunk of
/// doc_ids + optional scores. Created once per segment and reused for every
/// chunk, avoiding 14-parameter function signatures.
struct ChunkBuilder<'a> {
    segment_reader: &'a tantivy::SegmentReader,
    ff_projected_schema: &'a SchemaRef,
    unified_schema: &'a SchemaRef,
    projected_schema: &'a SchemaRef,
    projected_indices: Vec<usize>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    segment_ord: u32,
    tantivy_schema: tantivy::schema::Schema,
    store_reader: Option<tantivy::store::StoreReader>,
    dict_cache: DictCache,
}

impl<'a> ChunkBuilder<'a> {
    /// Assemble a single `RecordBatch` from a chunk of doc_ids and optional
    /// scores. Reads fast fields, builds score/document arrays, and projects
    /// to the output schema.
    fn build(
        &self,
        chunk_ids: &[u32],
        chunk_scores: Option<&[f32]>,
    ) -> Result<RecordBatch> {
        // Read fast fields for this chunk.
        let ff_batch = read_segment_fast_fields_to_batch(
            self.segment_reader,
            self.ff_projected_schema,
            Some(chunk_ids),
            None,
            None,
            self.segment_ord,
            Some(&self.dict_cache),
        )?;

        let chunk_rows = ff_batch.num_rows();

        // Build score array.
        let score_array: Option<Arc<dyn arrow::array::Array>> = if self.needs_score {
            match chunk_scores {
                Some(sc) => Some(Arc::new(Float32Array::from(sc.to_vec()))),
                None => Some(arrow::array::new_null_array(&DataType::Float32, chunk_rows)),
            }
        } else {
            None
        };

        // Build document array.
        let doc_array: Option<Arc<dyn arrow::array::Array>> = if self.needs_document {
            let store = self
                .store_reader
                .as_ref()
                .expect("store_reader required when needs_document");
            let mut doc_builder =
                StringBuilder::with_capacity(chunk_ids.len(), chunk_ids.len() * 256);
            for &doc_id in chunk_ids {
                let doc: tantivy::TantivyDocument = store.get(doc_id).map_err(|e| {
                    DataFusionError::Internal(format!("read doc {doc_id}: {e}"))
                })?;
                doc_builder.append_value(&doc.to_json(&self.tantivy_schema));
            }
            Some(Arc::new(doc_builder.finish()) as Arc<dyn arrow::array::Array>)
        } else {
            None
        };

        // Assemble output columns.
        let mut output_columns: Vec<Arc<dyn arrow::array::Array>> =
            Vec::with_capacity(self.projected_indices.len());

        for &unified_idx in &self.projected_indices {
            if unified_idx == self.score_column_idx {
                output_columns.push(
                    score_array.clone().unwrap_or_else(|| {
                        arrow::array::new_null_array(&DataType::Float32, chunk_rows)
                    }),
                );
            } else if unified_idx == self.document_column_idx {
                output_columns.push(
                    doc_array
                        .clone()
                        .expect("_document requested but not built"),
                );
            } else {
                let col_name = self.unified_schema.field(unified_idx).name();
                let ff_col_idx = ff_batch.schema().index_of(col_name).map_err(|_| {
                    DataFusionError::Internal(format!(
                        "fast field column '{col_name}' not found in ff_batch"
                    ))
                })?;
                output_columns.push(ff_batch.column(ff_col_idx).clone());
            }
        }

        RecordBatch::try_new(self.projected_schema.clone(), output_columns)
            .map_err(|e| DataFusionError::Internal(format!("build output batch: {e}")))
    }
}

/// Generate batches for a single segment, streaming each batch through the
/// `emit` callback as it is produced. Returns `Ok(())` when all batches have
/// been emitted. If `emit` returns `false` (receiver dropped), production
/// stops early.
///
/// Configuration for a single-segment batch generation pass.
/// Constructed in `open()` and moved into `spawn_blocking`.
struct ScanConfig {
    index: Index,
    segment_idx: usize,
    batch_size: usize,
    ff_projected_schema: SchemaRef,
    unified_schema: SchemaRef,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    topk: Option<usize>,
    query: Option<Arc<dyn tantivy::query::Query>>,
}

/// Thin orchestrator: query execution is delegated to
/// [`collect_matching_docs`] and batch assembly to [`ChunkBuilder::build`].
fn generate_single_table_batch_streaming(
    cfg: &ScanConfig,
    mut emit: impl FnMut(RecordBatch) -> bool,
) -> Result<()> {
    let index = &cfg.index;
    let segment_idx = cfg.segment_idx;
    let batch_size = cfg.batch_size;
    let needs_score = cfg.needs_score;
    let reader = index
        .reader()
        .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
    let searcher = reader.searcher();
    let segment_readers = searcher.segment_readers();
    if segment_idx >= segment_readers.len() {
        return Ok(());
    }
    let segment_reader = &segment_readers[segment_idx];

    // Collect matching docs via shared query execution logic.
    let (doc_ids, scores) = collect_matching_docs(
        segment_reader,
        &searcher,
        cfg.query.as_ref(),
        &index.schema(),
        needs_score,
        cfg.topk,
    )?;

    if doc_ids.is_empty() {
        return Ok(());
    }

    // Build the chunk builder once for this segment.
    let needs_document = cfg.needs_document;
    let store_reader = if needs_document {
        Some(
            segment_reader
                .get_store_reader(100)
                .map_err(|e| DataFusionError::Internal(format!("open store reader: {e}")))?,
        )
    } else {
        None
    };

    let projected_indices: Vec<usize> = match &cfg.projection {
        Some(indices) => indices.clone(),
        None => (0..cfg.unified_schema.fields().len()).collect(),
    };

    let dict_cache = DictCache::build(segment_reader, &cfg.ff_projected_schema)?;

    let builder = ChunkBuilder {
        segment_reader,
        ff_projected_schema: &cfg.ff_projected_schema,
        unified_schema: &cfg.unified_schema,
        projected_schema: &cfg.projected_schema,
        projected_indices,
        score_column_idx: cfg.score_column_idx,
        document_column_idx: cfg.document_column_idx,
        needs_score,
        needs_document,
        segment_ord: segment_idx as u32,
        tantivy_schema: index.schema(),
        store_reader,
        dict_cache,
    };

    // Process doc_ids in batch_size chunks.
    let total = doc_ids.len();
    let mut offset = 0;

    while offset < total {
        let end = (offset + batch_size).min(total);
        let chunk_ids = &doc_ids[offset..end];
        let chunk_scores = scores.as_ref().map(|s| &s[offset..end]);

        let batch = builder.build(chunk_ids, chunk_scores)?;

        if batch.num_rows() > 0 && !emit(batch) {
            return Ok(()); // receiver dropped, stop producing
        }

        offset = end;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Logical Expr → tantivy query conversion for fast field filters
// ---------------------------------------------------------------------------

/// Try to convert a logical `Expr` (column op literal) to a tantivy query.
///
/// Handles simple comparisons where the column is a tantivy FAST field and
/// the operator is one of `=`, `>`, `>=`, `<`, `<=`.
fn logical_expr_to_tantivy_query(
    expr: &Expr,
    tantivy_schema: &TantivySchema,
) -> Option<Box<dyn tantivy::query::Query>> {
    let Expr::BinaryExpr(binary) = expr else {
        return None;
    };

    let (col_name, scalar, col_on_left) = match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(col), Expr::Literal(sv, _)) => (col.name.clone(), sv.clone(), true),
        (Expr::Literal(sv, _), Expr::Column(col)) => (col.name.clone(), sv.clone(), false),
        _ => return None,
    };

    let field = tantivy_schema.get_field(&col_name).ok()?;
    let field_entry = tantivy_schema.get_field_entry(field);
    if !field_entry.is_fast() {
        return None;
    }

    let op = if col_on_left {
        binary.op
    } else {
        logical_flip_operator(&binary.op)?
    };

    let term = logical_scalar_to_term(field, field_entry.field_type(), &scalar)?;

    match op {
        Operator::Eq => {
            // For indexed non-text fields, use TermQuery (posting list lookup, O(matches)).
            // For text fields or non-indexed fields, use RangeQuery (fast field scan).
            // TermQuery on tokenized text fields would search for individual tokens,
            // not the original value — causing silent false negatives.
            if field_entry.is_indexed() && !matches!(field_entry.field_type(), FieldType::Str(_)) {
                Some(Box::new(tantivy::query::TermQuery::new(
                    term,
                    IndexRecordOption::Basic,
                )))
            } else {
                Some(Box::new(RangeQuery::new(
                    Bound::Included(term.clone()),
                    Bound::Included(term),
                )))
            }
        }
        Operator::Gt => Some(Box::new(RangeQuery::new(
            Bound::Excluded(term),
            Bound::Unbounded,
        ))),
        Operator::GtEq => Some(Box::new(RangeQuery::new(
            Bound::Included(term),
            Bound::Unbounded,
        ))),
        Operator::Lt => Some(Box::new(RangeQuery::new(
            Bound::Unbounded,
            Bound::Excluded(term),
        ))),
        Operator::LtEq => Some(Box::new(RangeQuery::new(
            Bound::Unbounded,
            Bound::Included(term),
        ))),
        Operator::NotEq => {
            // NotEq as union of (< term) OR (> term)
            let lt = Box::new(RangeQuery::new(
                Bound::Unbounded,
                Bound::Excluded(term.clone()),
            ));
            let gt = Box::new(RangeQuery::new(
                Bound::Excluded(term),
                Bound::Unbounded,
            ));
            Some(Box::new(tantivy::query::BooleanQuery::union(vec![lt, gt])))
        }
        _ => None,
    }
}

/// Flip a comparison operator when the column is on the right side.
fn logical_flip_operator(op: &Operator) -> Option<Operator> {
    match op {
        Operator::Eq => Some(Operator::Eq),
        Operator::NotEq => Some(Operator::NotEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        _ => None,
    }
}

/// Convert a DataFusion `ScalarValue` to a tantivy `Term` for the given field.
fn logical_scalar_to_term(
    field: tantivy::schema::Field,
    field_type: &FieldType,
    scalar: &ScalarValue,
) -> Option<Term> {
    match field_type {
        FieldType::I64(_) => {
            let v = match scalar {
                ScalarValue::Int64(Some(v)) => *v,
                ScalarValue::Int32(Some(v)) => i64::from(*v),
                ScalarValue::Int16(Some(v)) => i64::from(*v),
                ScalarValue::Int8(Some(v)) => i64::from(*v),
                ScalarValue::UInt64(Some(v)) => i64::try_from(*v).ok()?,
                ScalarValue::UInt32(Some(v)) => i64::from(*v),
                ScalarValue::UInt16(Some(v)) => i64::from(*v),
                ScalarValue::UInt8(Some(v)) => i64::from(*v),
                ScalarValue::Float64(Some(v)) => {
                    if v.fract() != 0.0 || *v > i64::MAX as f64 || *v < i64::MIN as f64 {
                        return None;
                    }
                    *v as i64
                }
                _ => return None,
            };
            return Some(Term::from_field_i64(field, v));
        }
        FieldType::U64(_) => {
            let v = match scalar {
                ScalarValue::UInt64(Some(v)) => *v,
                ScalarValue::UInt32(Some(v)) => u64::from(*v),
                ScalarValue::UInt16(Some(v)) => u64::from(*v),
                ScalarValue::UInt8(Some(v)) => u64::from(*v),
                ScalarValue::Int64(Some(v)) => u64::try_from(*v).ok()?,
                ScalarValue::Int32(Some(v)) => u64::try_from(*v).ok()?,
                ScalarValue::Int16(Some(v)) => u64::try_from(*v).ok()?,
                ScalarValue::Int8(Some(v)) => u64::try_from(*v).ok()?,
                ScalarValue::Float64(Some(v)) => {
                    if v.fract() != 0.0 || *v < 0.0 || *v > u64::MAX as f64 {
                        return None;
                    }
                    *v as u64
                }
                _ => return None,
            };
            return Some(Term::from_field_u64(field, v));
        }
        FieldType::F64(_) => {
            let v = match scalar {
                ScalarValue::Float64(Some(v)) => *v,
                ScalarValue::Float32(Some(v)) => *v as f64,
                ScalarValue::Int64(Some(v)) => *v as f64,
                ScalarValue::Int32(Some(v)) => *v as f64,
                ScalarValue::Int16(Some(v)) => *v as f64,
                ScalarValue::Int8(Some(v)) => *v as f64,
                ScalarValue::UInt64(Some(v)) => *v as f64,
                ScalarValue::UInt32(Some(v)) => *v as f64,
                ScalarValue::UInt16(Some(v)) => *v as u64 as f64,
                ScalarValue::UInt8(Some(v)) => *v as f64,
                _ => return None,
            };
            return Some(Term::from_field_f64(field, v));
        }
        _ => {}
    }
    match (field_type, scalar) {
        (FieldType::Bool(_), ScalarValue::Boolean(Some(v))) => {
            Some(Term::from_field_bool(field, *v))
        }
        (FieldType::Str(_), ScalarValue::Utf8(Some(s))) => {
            Some(Term::from_field_text(field, s))
        }
        // Date — tantivy DateTime stores nanoseconds internally.
        (FieldType::Date(_), ScalarValue::TimestampMicrosecond(Some(v), _)) => {
            Some(Term::from_field_date(field, DateTime::from_timestamp_micros(*v)))
        }
        (FieldType::Date(_), ScalarValue::TimestampSecond(Some(v), _)) => {
            Some(Term::from_field_date(field, DateTime::from_timestamp_secs(*v)))
        }
        (FieldType::Date(_), ScalarValue::TimestampMillisecond(Some(v), _)) => {
            Some(Term::from_field_date(field, DateTime::from_timestamp_millis(*v)))
        }
        (FieldType::Date(_), ScalarValue::TimestampNanosecond(Some(v), _)) => {
            Some(Term::from_field_date(field, DateTime::from_timestamp_nanos(*v)))
        }
        // IpAddr — mapped to Utf8 in schema_mapping, tantivy stores as Ipv6Addr.
        (FieldType::IpAddr(_), ScalarValue::Utf8(Some(s))) => {
            let ip: std::net::IpAddr = s.parse().ok()?;
            let ipv6 = match ip {
                std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped(),
                std::net::IpAddr::V6(v6) => v6,
            };
            Some(Term::from_field_ip_addr(field, ipv6))
        }
        // Bytes — mapped to Binary in schema_mapping.
        (FieldType::Bytes(_), ScalarValue::Binary(Some(b))) => {
            Some(Term::from_field_bytes(field, b))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Serializable representation of fast field filters for the codec
// ---------------------------------------------------------------------------

/// A simple, serializable representation of a `column op literal` filter
/// expression. Used by the codec to serialize fast field filters that were
/// converted to tantivy queries so workers can re-derive `pre_built_query`.
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct FastFieldFilter {
    field: String,
    op: String,
    value: String,
    value_type: String,
}

/// Serialize a slice of logical `Expr`s (each `column op literal`) to JSON.
pub(crate) fn serialize_fast_field_filters(exprs: &[Expr]) -> Result<String> {
    let filters: Vec<FastFieldFilter> = exprs
        .iter()
        .filter_map(|expr| {
            let Expr::BinaryExpr(binary) = expr else {
                return None;
            };
            let (col_name, scalar, col_on_left) =
                match (binary.left.as_ref(), binary.right.as_ref()) {
                    (Expr::Column(col), Expr::Literal(sv, _)) => {
                        (col.name.clone(), sv.clone(), true)
                    }
                    (Expr::Literal(sv, _), Expr::Column(col)) => {
                        (col.name.clone(), sv.clone(), false)
                    }
                    _ => return None,
                };
            let op = if col_on_left {
                binary.op
            } else {
                logical_flip_operator(&binary.op)?
            };
            let op_str = match op {
                Operator::Eq => "eq",
                Operator::NotEq => "neq",
                Operator::Gt => "gt",
                Operator::GtEq => "gte",
                Operator::Lt => "lt",
                Operator::LtEq => "lte",
                _ => return None,
            };
            let (value, value_type) = scalar_to_json_pair(&scalar)?;
            Some(FastFieldFilter {
                field: col_name,
                op: op_str.to_string(),
                value,
                value_type,
            })
        })
        .collect();
    serde_json::to_string(&filters).map_err(|e| {
        DataFusionError::Internal(format!("serialize fast field filters: {e}"))
    })
}

/// Deserialize fast field filter JSON and reconstruct tantivy queries.
///
/// Returns the reconstructed tantivy queries; the caller combines them with
/// `BooleanQuery::intersection` as usual.
pub(crate) fn deserialize_fast_field_filters(
    json: &str,
    tantivy_schema: &TantivySchema,
) -> Result<Vec<Box<dyn tantivy::query::Query>>> {
    if json.is_empty() {
        return Ok(Vec::new());
    }
    let filters: Vec<FastFieldFilter> = serde_json::from_str(json).map_err(|e| {
        DataFusionError::Internal(format!("deserialize fast field filters: {e}"))
    })?;
    let mut queries = Vec::with_capacity(filters.len());
    for f in &filters {
        let scalar = json_pair_to_scalar(&f.value, &f.value_type)?;
        let op = match f.op.as_str() {
            "eq" => Operator::Eq,
            "neq" => Operator::NotEq,
            "gt" => Operator::Gt,
            "gte" => Operator::GtEq,
            "lt" => Operator::Lt,
            "lte" => Operator::LtEq,
            other => {
                return Err(DataFusionError::Internal(format!(
                    "unknown fast field filter op: {other}"
                )))
            }
        };
        let expr = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(Expr::Column(datafusion::common::Column::new_unqualified(
                &f.field,
            ))),
            op,
            right: Box::new(Expr::Literal(scalar, None)),
        });
        if let Some(q) = logical_expr_to_tantivy_query(&expr, tantivy_schema) {
            queries.push(q);
        }
    }
    Ok(queries)
}

/// Encode a `ScalarValue` as a `(value_string, type_tag)` pair for JSON.
fn scalar_to_json_pair(scalar: &ScalarValue) -> Option<(String, String)> {
    match scalar {
        ScalarValue::Int8(Some(v)) => Some((v.to_string(), "i8".into())),
        ScalarValue::Int16(Some(v)) => Some((v.to_string(), "i16".into())),
        ScalarValue::Int32(Some(v)) => Some((v.to_string(), "i32".into())),
        ScalarValue::Int64(Some(v)) => Some((v.to_string(), "i64".into())),
        ScalarValue::UInt8(Some(v)) => Some((v.to_string(), "u8".into())),
        ScalarValue::UInt16(Some(v)) => Some((v.to_string(), "u16".into())),
        ScalarValue::UInt32(Some(v)) => Some((v.to_string(), "u32".into())),
        ScalarValue::UInt64(Some(v)) => Some((v.to_string(), "u64".into())),
        ScalarValue::Float32(Some(v)) => Some((v.to_string(), "f32".into())),
        ScalarValue::Float64(Some(v)) => Some((v.to_string(), "f64".into())),
        ScalarValue::Boolean(Some(v)) => Some((v.to_string(), "bool".into())),
        ScalarValue::Utf8(Some(v)) => Some((v.clone(), "utf8".into())),
        ScalarValue::Binary(Some(v)) => {
            use base64::Engine;
            Some((
                base64::engine::general_purpose::STANDARD.encode(v),
                "binary".into(),
            ))
        }
        ScalarValue::TimestampSecond(Some(v), _) => Some((v.to_string(), "ts_s".into())),
        ScalarValue::TimestampMillisecond(Some(v), _) => Some((v.to_string(), "ts_ms".into())),
        ScalarValue::TimestampMicrosecond(Some(v), _) => Some((v.to_string(), "ts_us".into())),
        ScalarValue::TimestampNanosecond(Some(v), _) => Some((v.to_string(), "ts_ns".into())),
        _ => None,
    }
}

/// Decode a `(value_string, type_tag)` pair back to a `ScalarValue`.
fn json_pair_to_scalar(value: &str, value_type: &str) -> Result<ScalarValue> {
    let parse_err = |e: std::num::ParseIntError| {
        DataFusionError::Internal(format!("parse {value_type} '{value}': {e}"))
    };
    let parse_float_err = |e: std::num::ParseFloatError| {
        DataFusionError::Internal(format!("parse {value_type} '{value}': {e}"))
    };
    match value_type {
        "i8" => Ok(ScalarValue::Int8(Some(value.parse().map_err(parse_err)?))),
        "i16" => Ok(ScalarValue::Int16(Some(value.parse().map_err(parse_err)?))),
        "i32" => Ok(ScalarValue::Int32(Some(value.parse().map_err(parse_err)?))),
        "i64" => Ok(ScalarValue::Int64(Some(value.parse().map_err(parse_err)?))),
        "u8" => Ok(ScalarValue::UInt8(Some(value.parse().map_err(parse_err)?))),
        "u16" => Ok(ScalarValue::UInt16(Some(value.parse().map_err(parse_err)?))),
        "u32" => Ok(ScalarValue::UInt32(Some(value.parse().map_err(parse_err)?))),
        "u64" => Ok(ScalarValue::UInt64(Some(value.parse().map_err(parse_err)?))),
        "f32" => Ok(ScalarValue::Float32(Some(
            value.parse().map_err(parse_float_err)?,
        ))),
        "f64" => Ok(ScalarValue::Float64(Some(
            value.parse().map_err(parse_float_err)?,
        ))),
        "bool" => Ok(ScalarValue::Boolean(Some(value == "true"))),
        "utf8" => Ok(ScalarValue::Utf8(Some(value.to_string()))),
        "binary" => {
            use base64::Engine;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(value)
                .map_err(|e| {
                    DataFusionError::Internal(format!("decode base64 binary: {e}"))
                })?;
            Ok(ScalarValue::Binary(Some(bytes)))
        }
        "ts_s" => Ok(ScalarValue::TimestampSecond(
            Some(value.parse().map_err(parse_err)?),
            None,
        )),
        "ts_ms" => Ok(ScalarValue::TimestampMillisecond(
            Some(value.parse().map_err(parse_err)?),
            None,
        )),
        "ts_us" => Ok(ScalarValue::TimestampMicrosecond(
            Some(value.parse().map_err(parse_err)?),
            None,
        )),
        "ts_ns" => Ok(ScalarValue::TimestampNanosecond(
            Some(value.parse().map_err(parse_err)?),
            None,
        )),
        other => Err(DataFusionError::Internal(format!(
            "unknown scalar type tag: {other}"
        ))),
    }
}
