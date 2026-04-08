use std::any::Any;
use std::fmt;
use std::ops::Bound;
use std::sync::Arc;

use arrow::array::{AsArray, Float32Array, RecordBatch, StringBuilder};
use arrow::compute::filter_record_batch;
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
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::projection::ProjectionExprs;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, Partitioning, SendableRecordBatchStream};
use futures::stream::{self, StreamExt};
use tantivy::aggregation::agg_req::Aggregations;
use tantivy::collector::TopNComputer;
use tantivy::query::EnableScoring;
use tantivy::query::RangeQuery;
use tantivy::schema::{FieldType, Schema as TantivySchema, Term};
use tantivy::{DateTime, DocId, Document, Index, Score};

use crate::fast_field_reader::read_segment_fast_fields_to_batch;
use crate::full_text_udf::extract_full_text_call;
use crate::index_opener::{DirectIndexOpener, IndexOpener};
use crate::schema_mapping::{tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index};
use crate::util::{build_combined_query, segment_hash_partitioning};

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
                    if let Ok(col) = fast_fields.u64(name) {
                        (
                            Some(ScalarValue::UInt64(Some(col.min_value()))),
                            Some(ScalarValue::UInt64(Some(col.max_value()))),
                        )
                    } else {
                        (None, None)
                    }
                }
                DataType::Int64 => {
                    if let Ok(col) = fast_fields.i64(name) {
                        (
                            Some(ScalarValue::Int64(Some(col.min_value()))),
                            Some(ScalarValue::Int64(Some(col.max_value()))),
                        )
                    } else {
                        (None, None)
                    }
                }
                DataType::Float64 => {
                    if let Ok(col) = fast_fields.f64(name) {
                        (
                            Some(ScalarValue::Float64(Some(col.min_value()))),
                            Some(ScalarValue::Float64(Some(col.max_value()))),
                        )
                    } else {
                        (None, None)
                    }
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    if let Ok(col) = fast_fields.date(name) {
                        (
                            Some(ScalarValue::TimestampMicrosecond(
                                Some(col.min_value().into_timestamp_micros()),
                                None,
                            )),
                            Some(ScalarValue::TimestampMicrosecond(
                                Some(col.max_value().into_timestamp_micros()),
                                None,
                            )),
                        )
                    } else {
                        (None, None)
                    }
                }
                DataType::Boolean => {
                    if let Ok(col) = fast_fields.bool(name) {
                        (
                            Some(ScalarValue::Boolean(Some(col.min_value()))),
                            Some(ScalarValue::Boolean(Some(col.max_value()))),
                        )
                    } else {
                        (None, None)
                    }
                }
                _ => (None, None),
            };

            column_stats.push((name.to_string(), min_val, max_val));
        }

        stats.push(Some(PartitionStat {
            num_rows: alive,
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
    aggregations: Option<Arc<Aggregations>>,
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

    fn from_opener_with_ff_schema(
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
            aggregations: None,
        }
    }

    /// Stash tantivy aggregations for the AggPushdown optimizer rule.
    pub fn set_aggregations(&mut self, aggs: Arc<Aggregations>) {
        self.aggregations = Some(aggs);
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
                    TableProviderFilterPushDown::Exact
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
            unified_schema: self.unified_schema.clone(),
            fast_field_schema: self.fast_field_schema.clone(),
            projected_schema,
            projection: projection.cloned(),
            score_column_idx: self.score_column_idx,
            document_column_idx: self.document_column_idx,
            needs_score,
            needs_document,
            ff_projection: ff_indices,
            ff_projected_schema,
            raw_queries,
            pre_built_query,
            topk: None,
            num_segments,
            pushed_filters: Vec::new(),
            aggregations: self.aggregations.clone(),
            agg_mode: None,
            partition_stats,
        };

        Ok(Arc::new(DataSourceExec::new(Arc::new(data_source))))
    }
}

// ---------------------------------------------------------------------------
// DataSource implementation
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SingleTableDataSource {
    opener: Arc<dyn IndexOpener>,
    unified_schema: SchemaRef,
    fast_field_schema: SchemaRef,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    /// Indices into fast_field_schema that we need to read.
    ff_projection: Vec<usize>,
    /// The projected fast field schema (built from ff_projection).
    ff_projected_schema: SchemaRef,
    raw_queries: Vec<(String, String)>,
    /// Pre-built tantivy queries from fast field filters converted at scan time.
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    pub(crate) topk: Option<usize>,
    num_segments: usize,
    pushed_filters: Vec<Arc<dyn PhysicalExpr>>,
    aggregations: Option<Arc<Aggregations>>,
    /// When set, the data source runs in aggregation mode — executes tantivy's
    /// native aggregation instead of streaming rows. The output schema changes
    /// to match the aggregation results.
    agg_mode: Option<(Arc<Aggregations>, SchemaRef)>,
    /// Per-partition (segment) statistics for partition pruning.
    /// Indexed by partition number. `None` means stats are unavailable for
    /// that partition (e.g. remote opener without metadata).
    partition_stats: Vec<Option<PartitionStat>>,
}

impl SingleTableDataSource {
    fn clone_with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut new = SingleTableDataSource {
            opener: self.opener.clone(),
            unified_schema: self.unified_schema.clone(),
            fast_field_schema: self.fast_field_schema.clone(),
            projected_schema: self.projected_schema.clone(),
            projection: self.projection.clone(),
            score_column_idx: self.score_column_idx,
            document_column_idx: self.document_column_idx,
            needs_score: self.needs_score,
            needs_document: self.needs_document,
            ff_projection: self.ff_projection.clone(),
            ff_projected_schema: self.ff_projected_schema.clone(),
            raw_queries: self.raw_queries.clone(),
            pre_built_query: self.pre_built_query.clone(),
            topk: self.topk,
            num_segments: self.num_segments,
            pushed_filters: self.pushed_filters.clone(),
            aggregations: self.aggregations.clone(),
            agg_mode: self.agg_mode.clone(),
            partition_stats: self.partition_stats.clone(),
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

    /// Access the pushed-down filters.
    pub fn pushed_filters(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.pushed_filters
    }

    /// Return a copy with the given pushed-down filters.
    pub fn with_pushed_filters(&self, filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        self.clone_with(|s| s.pushed_filters = filters)
    }

    /// Access the stashed tantivy aggregations (for future AggPushdown support).
    #[allow(dead_code)]
    pub(crate) fn aggregations(&self) -> Option<&Arc<Aggregations>> {
        self.aggregations.as_ref()
    }

    /// Set aggregation mode. The data source will run tantivy's native
    /// aggregation instead of streaming rows.
    pub(crate) fn with_agg_mode(&self, aggs: Arc<Aggregations>, output_schema: SchemaRef) -> Self {
        self.clone_with(|s| {
            s.agg_mode = Some((aggs, output_schema.clone()));
            s.projected_schema = output_schema;
        })
    }

    #[allow(dead_code)]
    pub(crate) fn agg_mode(&self) -> Option<&(Arc<Aggregations>, SchemaRef)> {
        self.agg_mode.as_ref()
    }

    /// Aggregate per-partition statistics into overall table statistics.
    ///
    /// - `num_rows` = sum of all partition row counts
    /// - column `min_value` = minimum across all partition minimums
    /// - column `max_value` = maximum across all partition maximums
    fn aggregate_statistics(&self) -> Result<Statistics> {
        use datafusion::common::stats::Precision;
        use datafusion::common::ColumnStatistics;

        if self.agg_mode.is_some() {
            return Ok(Statistics::new_unknown(&self.projected_schema));
        }

        // Collect only the partitions that have stats.
        let known: Vec<&PartitionStat> = self
            .partition_stats
            .iter()
            .filter_map(|s| s.as_ref())
            .collect();

        if known.is_empty() {
            return Ok(Statistics::new_unknown(&self.projected_schema));
        }

        let total_rows: usize = known.iter().map(|s| s.num_rows).sum();
        let num_rows = Precision::Exact(total_rows);

        let column_statistics: Vec<ColumnStatistics> = self
            .projected_schema
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
                            let p = Precision::Exact(min_v.clone());
                            overall_min = match overall_min {
                                Precision::Absent => p,
                                prev => prev.min(&p),
                            };
                        }
                        if let Some(max_v) = max_val {
                            let p = Precision::Exact(max_v.clone());
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
        self.projection.as_ref()
    }

    /// Whether _score is needed.
    pub fn needs_score(&self) -> bool {
        self.needs_score
    }

    /// Whether _document is needed.
    pub fn needs_document(&self) -> bool {
        self.needs_document
    }
}

impl DataSource for SingleTableDataSource {
    fn open(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Aggregation mode: run tantivy native aggregation
        if let Some((aggs, agg_schema)) = &self.agg_mode {
            let opener = self.opener.clone();
            let raw_queries = self.raw_queries.clone();
            let pre_built_query = self.pre_built_query.as_ref().map(|q| Arc::from(q.box_clone()));
            let aggs = aggs.clone();
            let schema = agg_schema.clone();

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

                // Build combined query from FTS + fast field filters
                let query = build_combined_query(&index, pre_built_query.as_ref(), &raw_queries)?;

                tokio::task::spawn_blocking(move || {
                    crate::unified::agg_exec::execute_tantivy_agg(&index, &aggs, query.as_ref(), &schema)
                })
                .await
                .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
            });

            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                agg_schema.clone(),
                stream,
            )));
        }

        // Normal row-streaming mode
        let batch_size = context.session_config().batch_size();
        let opener = self.opener.clone();
        let segment_idx = partition;
        let raw_queries = self.raw_queries.clone();
        let ff_projected_schema = self.ff_projected_schema.clone();
        let unified_schema = self.unified_schema.clone();
        let projected_schema = self.projected_schema.clone();
        let projection = self.projection.clone();
        let needs_score = self.needs_score;
        let needs_document = self.needs_document;
        let score_column_idx = self.score_column_idx;
        let document_column_idx = self.document_column_idx;
        let topk = self.topk;
        let pushed_filters = self.pushed_filters.clone();
        let pre_built_query = self
            .pre_built_query
            .as_ref()
            .map(|q| Arc::from(q.box_clone()));

        let schema = self.projected_schema.clone();
        let stream = stream::once(async move {
            let index = opener.open().await?;

            // Warm up fast fields.
            let ff_names: Vec<&str> = ff_projected_schema
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect();
            crate::warmup::warmup_fast_fields_by_name(&index, &ff_names).await?;

            // Warm up inverted index for queried text fields.
            let queried_fields: Vec<tantivy::schema::Field> = raw_queries
                .iter()
                .filter_map(|(field_name, _)| index.schema().get_field(field_name).ok())
                .collect();
            if !queried_fields.is_empty() {
                crate::warmup::warmup_inverted_index(&index, &queried_fields).await?;
            }

            // Build query from raw_queries + pre-built fast field query.
            let query =
                build_combined_query(&index, pre_built_query.as_ref(), &raw_queries)?;

            // Move sync work to blocking thread.
            tokio::task::spawn_blocking(move || {
                generate_single_table_batch(
                    &index,
                    segment_idx,
                    query.as_ref(),
                    &ff_projected_schema,
                    &unified_schema,
                    &projected_schema,
                    projection.as_deref(),
                    score_column_idx,
                    document_column_idx,
                    needs_score,
                    needs_document,
                    topk,
                    &pushed_filters,
                    batch_size,
                )
            })
            .await
            .map_err(|e| {
                DataFusionError::Internal(format!("spawn_blocking join error: {e}"))
            })?
        })
        .flat_map(|result: Result<Vec<RecordBatch>>| match result {
            Ok(batches) => stream::iter(batches.into_iter().map(Ok)).left_stream(),
            Err(e) => stream::once(async move { Err(e) }).right_stream(),
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
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
            self.needs_score,
            self.needs_document,
            self.topk,
        )?;
        if self.agg_mode.is_some() {
            write!(f, ", agg_mode=true")?;
        }
        Ok(())
    }

    fn output_partitioning(&self) -> Partitioning {
        if self.agg_mode.is_some() {
            Partitioning::UnknownPartitioning(1)
        } else {
            segment_hash_partitioning(&self.projected_schema, self.num_segments)
        }
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.projected_schema.clone())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        use datafusion::common::stats::Precision;
        use datafusion::common::ColumnStatistics;

        // In aggregation mode the output schema is different and per-row
        // statistics don't apply.
        if self.agg_mode.is_some() {
            return Ok(Statistics::new_unknown(&self.projected_schema));
        }

        let partition = match partition {
            Some(p) => p,
            None => {
                // Aggregate across all partitions.
                return self.aggregate_statistics();
            }
        };

        let stat = match self.partition_stats.get(partition) {
            Some(Some(s)) => s,
            _ => return Ok(Statistics::new_unknown(&self.projected_schema)),
        };

        let num_rows = Precision::Exact(stat.num_rows);

        let column_statistics: Vec<ColumnStatistics> = self
            .projected_schema
            .fields()
            .iter()
            .map(|field| {
                let name = field.name();
                if let Some((_, min_val, max_val)) =
                    stat.column_stats.iter().find(|(n, _, _)| n == name)
                {
                    ColumnStatistics {
                        null_count: Precision::Absent,
                        max_value: max_val
                            .clone()
                            .map_or(Precision::Absent, Precision::Exact),
                        min_value: min_val
                            .clone()
                            .map_or(Precision::Absent, Precision::Exact),
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

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        None
    }

    fn fetch(&self) -> Option<usize> {
        None
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        ExecutionPlanMetricsSet::new()
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
        let results: Vec<PushedDown> = filters.iter().map(|_| PushedDown::Yes).collect();
        let mut new_filters = self.pushed_filters.clone();
        new_filters.extend(filters);
        let updated = self.clone_with(|s| s.pushed_filters = new_filters);
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(results)
                .with_updated_node(Arc::new(updated) as Arc<dyn DataSource>),
        )
    }
}

// ---------------------------------------------------------------------------
// Core batch generation
// ---------------------------------------------------------------------------

/// Generate batches for a single segment, handling FTS queries, fast fields,
/// scoring, document retrieval, and filter application in one pass.
#[allow(clippy::too_many_arguments)]
fn generate_single_table_batch(
    index: &Index,
    segment_idx: usize,
    query: Option<&Arc<dyn tantivy::query::Query>>,
    ff_projected_schema: &SchemaRef,
    unified_schema: &SchemaRef,
    projected_schema: &SchemaRef,
    projection: Option<&[usize]>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    topk: Option<usize>,
    pushed_filters: &[Arc<dyn PhysicalExpr>],
    batch_size: usize,
) -> Result<Vec<RecordBatch>> {
    let reader = index
        .reader()
        .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
    let searcher = reader.searcher();
    let segment_readers = searcher.segment_readers();
    if segment_idx >= segment_readers.len() {
        return Ok(vec![RecordBatch::new_empty(projected_schema.clone())]);
    }
    let segment_reader = &segment_readers[segment_idx];

    // Step 1: Run query (if any) to collect doc_ids and optional scores.
    let (doc_ids, scores): (Vec<u32>, Option<Vec<f32>>) = match query {
        Some(query) => {
            if needs_score {
                // BM25 scoring enabled.
                let weight = query
                    .weight(EnableScoring::enabled_from_searcher(&searcher))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;

                if let Some(k) = topk {
                    // TopK with Block-WAND pruning.
                    let mut top_n: TopNComputer<Score, DocId, _> = TopNComputer::new(k);

                    let alive_bitset = segment_reader.alive_bitset();
                    if let Some(alive_bitset) = alive_bitset {
                        let mut threshold = Score::MIN;
                        top_n.threshold = Some(threshold);
                        weight
                            .for_each_pruning(
                                Score::MIN,
                                segment_reader,
                                &mut |doc, score| {
                                    if alive_bitset.is_deleted(doc) {
                                        return threshold;
                                    }
                                    top_n.push(score, doc);
                                    threshold = top_n.threshold.unwrap_or(Score::MIN);
                                    threshold
                                },
                            )
                            .map_err(|e| {
                                DataFusionError::Internal(format!("topk query execution: {e}"))
                            })?;
                    } else {
                        weight
                            .for_each_pruning(
                                Score::MIN,
                                segment_reader,
                                &mut |doc, score| {
                                    top_n.push(score, doc);
                                    top_n.threshold.unwrap_or(Score::MIN)
                                },
                            )
                            .map_err(|e| {
                                DataFusionError::Internal(format!("topk query execution: {e}"))
                            })?;
                    }

                    let results = top_n.into_sorted_vec();
                    let mut ids = Vec::with_capacity(results.len());
                    let mut sc = Vec::with_capacity(results.len());
                    for item in results {
                        ids.push(item.doc);
                        sc.push(item.sort_key);
                    }
                    (ids, Some(sc))
                } else {
                    // Full scoring without topK.
                    let mut ids = Vec::new();
                    let mut sc = Vec::new();
                    weight
                        .for_each(segment_reader, &mut |doc, score| {
                            ids.push(doc);
                            sc.push(score);
                        })
                        .map_err(|e| {
                            DataFusionError::Internal(format!("query execution: {e}"))
                        })?;

                    // Filter deleted docs.
                    if let Some(alive_bitset) = segment_reader.alive_bitset() {
                        let mut filtered_ids = Vec::new();
                        let mut filtered_sc = Vec::new();
                        for (doc, score) in ids.into_iter().zip(sc) {
                            if alive_bitset.is_alive(doc) {
                                filtered_ids.push(doc);
                                filtered_sc.push(score);
                            }
                        }
                        (filtered_ids, Some(filtered_sc))
                    } else {
                        (ids, Some(sc))
                    }
                }
            } else {
                // No scoring needed — boolean filter only.
                let tantivy_schema = index.schema();
                let weight = query
                    .weight(EnableScoring::disabled_from_schema(&tantivy_schema))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
                let mut matching_docs = Vec::new();
                weight
                    .for_each_no_score(segment_reader, &mut |docs| {
                        matching_docs.extend_from_slice(docs);
                    })
                    .map_err(|e| {
                        DataFusionError::Internal(format!("query execution: {e}"))
                    })?;
                // Filter deleted docs.
                if let Some(alive_bitset) = segment_reader.alive_bitset() {
                    matching_docs.retain(|&doc| alive_bitset.is_alive(doc));
                }
                (matching_docs, None)
            }
        }
        None => {
            // No query — iterate all alive docs.
            let max_doc = segment_reader.max_doc();
            let alive_bitset = segment_reader.alive_bitset();
            let ids: Vec<u32> = (0..max_doc)
                .filter(|&doc_id| alive_bitset.map_or(true, |bitset| bitset.is_alive(doc_id)))
                .collect();
            (ids, None)
        }
    };

    if doc_ids.is_empty() {
        return Ok(vec![RecordBatch::new_empty(projected_schema.clone())]);
    }

    // Step 2: Read fast fields for the collected doc_ids.
    let ff_batch = read_segment_fast_fields_to_batch(
        segment_reader,
        ff_projected_schema,
        Some(&doc_ids),
        None,
        None,
        segment_idx as u32,
    )?;

    let num_rows = ff_batch.num_rows();

    // Step 3: Build score array.
    let score_array: Option<Arc<dyn arrow::array::Array>> = if needs_score {
        match scores {
            Some(score_vec) => Some(Arc::new(Float32Array::from(score_vec))),
            None => Some(arrow::array::new_null_array(&DataType::Float32, num_rows)),
        }
    } else {
        None
    };

    // Step 4: Build document array if needed.
    let doc_array: Option<Arc<dyn arrow::array::Array>> = if needs_document {
        let tantivy_schema = index.schema();
        let store_reader = segment_reader
            .get_store_reader(100)
            .map_err(|e| DataFusionError::Internal(format!("open store reader: {e}")))?;

        let mut doc_builder =
            StringBuilder::with_capacity(doc_ids.len(), doc_ids.len() * 256);
        for &doc_id in &doc_ids {
            let doc: tantivy::TantivyDocument = store_reader.get(doc_id).map_err(|e| {
                DataFusionError::Internal(format!("read doc {doc_id}: {e}"))
            })?;
            let json = doc.to_json(&tantivy_schema);
            doc_builder.append_value(&json);
        }
        Some(Arc::new(doc_builder.finish()) as Arc<dyn arrow::array::Array>)
    } else {
        None
    };

    // Step 5: Assemble the output batch by picking columns from the unified schema
    // according to the projection.
    let projected_indices: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..unified_schema.fields().len()).collect(),
    };

    let mut output_columns: Vec<Arc<dyn arrow::array::Array>> =
        Vec::with_capacity(projected_indices.len());

    for &unified_idx in &projected_indices {
        if unified_idx == score_column_idx {
            output_columns.push(
                score_array
                    .clone()
                    .unwrap_or_else(|| arrow::array::new_null_array(&DataType::Float32, num_rows)),
            );
        } else if unified_idx == document_column_idx {
            output_columns.push(
                doc_array
                    .clone()
                    .expect("_document requested but not built"),
            );
        } else {
            let col_name = unified_schema.field(unified_idx).name();
            let ff_col_idx = ff_batch.schema().index_of(col_name).map_err(|_| {
                DataFusionError::Internal(format!(
                    "fast field column '{col_name}' not found in ff_batch"
                ))
            })?;
            output_columns.push(ff_batch.column(ff_col_idx).clone());
        }
    }

    let mut output_batch = RecordBatch::try_new(projected_schema.clone(), output_columns)
        .map_err(|e| DataFusionError::Internal(format!("build output batch: {e}")))?;

    // Step 6: Apply pushed-down filters on the assembled output batch.
    // Filters reference column indices in the projected schema, so they must
    // be evaluated after the full batch is built.
    for filter in pushed_filters {
        if output_batch.num_rows() == 0 {
            break;
        }
        let result = filter.evaluate(&output_batch)?;
        let mask = match result {
            datafusion::physical_plan::ColumnarValue::Scalar(
                datafusion::common::ScalarValue::Boolean(Some(true)),
            ) => continue,
            datafusion::physical_plan::ColumnarValue::Scalar(
                datafusion::common::ScalarValue::Boolean(Some(false)),
            ) => {
                return Ok(vec![RecordBatch::new_empty(projected_schema.clone())]);
            }
            datafusion::physical_plan::ColumnarValue::Array(arr) => arr.as_boolean().clone(),
            other => {
                let arr = other.into_array(output_batch.num_rows())?;
                arr.as_boolean().clone()
            }
        };
        output_batch = filter_record_batch(&output_batch, &mask)?;
    }

    if output_batch.num_rows() == 0 {
        return Ok(vec![RecordBatch::new_empty(projected_schema.clone())]);
    }

    // Step 7: Chunk into batch_size pieces.
    let total_rows = output_batch.num_rows();
    if total_rows <= batch_size {
        Ok(vec![output_batch])
    } else {
        let mut batches = Vec::new();
        let mut offset = 0;
        while offset < total_rows {
            let len = (total_rows - offset).min(batch_size);
            batches.push(output_batch.slice(offset, len));
            offset += len;
        }
        Ok(batches)
    }
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
        Operator::Eq => Some(Box::new(RangeQuery::new(
            Bound::Included(term.clone()),
            Bound::Included(term),
        ))),
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
        _ => None,
    }
}

/// Flip a comparison operator when the column is on the right side.
fn logical_flip_operator(op: &Operator) -> Option<Operator> {
    match op {
        Operator::Eq => Some(Operator::Eq),
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
    match (field_type, scalar) {
        (FieldType::U64(_), ScalarValue::UInt64(Some(v))) => {
            Some(Term::from_field_u64(field, *v))
        }
        (FieldType::I64(_), ScalarValue::Int64(Some(v))) => {
            Some(Term::from_field_i64(field, *v))
        }
        (FieldType::F64(_), ScalarValue::Float64(Some(v))) => {
            Some(Term::from_field_f64(field, *v))
        }
        (FieldType::Bool(_), ScalarValue::Boolean(Some(v))) => {
            Some(Term::from_field_bool(field, *v))
        }
        (FieldType::Str(_), ScalarValue::Utf8(Some(s))) => {
            Some(Term::from_field_text(field, s))
        }
        // Numeric type coercions that DataFusion may apply.
        (FieldType::F64(_), ScalarValue::Int64(Some(v))) => {
            Some(Term::from_field_f64(field, *v as f64))
        }
        (FieldType::F64(_), ScalarValue::Float32(Some(v))) => {
            Some(Term::from_field_f64(field, *v as f64))
        }
        (FieldType::I64(_), ScalarValue::Int32(Some(v))) => {
            Some(Term::from_field_i64(field, *v as i64))
        }
        (FieldType::U64(_), ScalarValue::Int64(Some(v))) if *v >= 0 => {
            Some(Term::from_field_u64(field, *v as u64))
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
