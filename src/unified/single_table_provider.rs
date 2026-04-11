use std::any::Any;
use std::fmt;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch, StringBuilder};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::config::ConfigOptions;
use datafusion::common::ScalarValue;
use datafusion::common::{Result, Statistics};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_physical_expr::expressions::Column as PhysicalColumn;
use datafusion_physical_expr::{
    EquivalenceProperties, LexOrdering, PhysicalExpr, PhysicalSortExpr,
};
use datafusion_physical_plan::filter_pushdown::{FilterPushdownPropagation, PushedDown};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion_physical_plan::projection::ProjectionExprs;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayFormatType, Partitioning, SendableRecordBatchStream};
use futures::stream::StreamExt;
use tantivy::query::RangeQuery;
use tantivy::schema::{FieldType, IndexRecordOption, Schema as TantivySchema, Term};
use tantivy::{DateTime, Document, Index};

use crate::fast_field_reader::{read_segment_fast_fields_to_batch, DictCache};
use crate::full_text_udf::extract_full_text_call;
use crate::index_opener::{DirectIndexOpener, IndexOpener};
use crate::schema_mapping::{
    tantivy_schema_to_arrow_from_index, tantivy_schema_to_arrow_with_multi_valued,
};
use crate::type_coercion::{
    apply_fast_field_projection, infer_canonical_fast_field_schema, plan_fast_field_projection,
    FastFieldProjectionPlan,
};
use crate::util::{
    build_combined_query, collect_topk_docs, for_each_matching_doc_chunks, MatchingDocChunksConfig,
};

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
    opener: &DirectIndexOpener,
    ff_schema: &SchemaRef,
) -> Result<Vec<Option<PartitionStat>>> {
    let reader = opener.reader()?;
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
                DataType::UInt64 => match fast_fields.u64(name) {
                    Ok(col) if col.values.num_vals() > 0 => (
                        Some(ScalarValue::UInt64(Some(col.min_value()))),
                        Some(ScalarValue::UInt64(Some(col.max_value()))),
                    ),
                    _ => (None, None),
                },
                DataType::Int64 => match fast_fields.i64(name) {
                    Ok(col) if col.values.num_vals() > 0 => (
                        Some(ScalarValue::Int64(Some(col.min_value()))),
                        Some(ScalarValue::Int64(Some(col.max_value()))),
                    ),
                    _ => (None, None),
                },
                DataType::Float64 => match fast_fields.f64(name) {
                    Ok(col) if col.values.num_vals() > 0 => (
                        Some(ScalarValue::Float64(Some(col.min_value()))),
                        Some(ScalarValue::Float64(Some(col.max_value()))),
                    ),
                    _ => (None, None),
                },
                DataType::Timestamp(TimeUnit::Microsecond, _) => match fast_fields.date(name) {
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
                },
                DataType::Boolean => match fast_fields.bool(name) {
                    Ok(col) if col.values.num_vals() > 0 => (
                        Some(ScalarValue::Boolean(Some(col.min_value()))),
                        Some(ScalarValue::Boolean(Some(col.max_value()))),
                    ),
                    _ => (None, None),
                },
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

#[derive(Debug, Clone)]
struct SplitDescriptor {
    opener: Arc<dyn IndexOpener>,
    fast_field_schema: SchemaRef,
    num_segments: usize,
    partition_stats: Vec<Option<PartitionStat>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SplitExecutionPlan {
    pub(crate) opener: Arc<dyn IndexOpener>,
    pub(crate) fast_field_projection: FastFieldProjectionPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartitionSpec {
    pub(crate) split_idx: usize,
    pub(crate) segment_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterPushdownSupport {
    Query,
    MissingField,
    Unsupported,
}

fn fast_field_schema_for_opener(opener: &Arc<dyn IndexOpener>) -> SchemaRef {
    opener
        .as_any()
        .downcast_ref::<DirectIndexOpener>()
        .map(|direct| tantivy_schema_to_arrow_from_index(direct.index()))
        .unwrap_or_else(|| {
            tantivy_schema_to_arrow_with_multi_valued(
                &opener.schema(),
                &opener.multi_valued_fields(),
            )
        })
}

fn build_split_descriptor(opener: Arc<dyn IndexOpener>) -> Result<SplitDescriptor> {
    let fast_field_schema = fast_field_schema_for_opener(&opener);
    let num_segments = opener.segment_sizes().len().max(1);
    let mut partition_stats =
        if let Some(direct) = opener.as_any().downcast_ref::<DirectIndexOpener>() {
            compute_partition_stats(direct, &fast_field_schema)?
        } else {
            vec![None; num_segments]
        };
    if partition_stats.is_empty() {
        partition_stats.resize(num_segments, None);
    }

    Ok(SplitDescriptor {
        opener,
        fast_field_schema,
        num_segments,
        partition_stats,
    })
}

fn build_unified_schema(fast_field_schema: &SchemaRef) -> (SchemaRef, usize, usize) {
    let mut unified_fields: Vec<Arc<Field>> = fast_field_schema.fields().to_vec();
    let score_column_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_score", DataType::Float32, true)));
    let document_column_idx = unified_fields.len();
    unified_fields.push(Arc::new(Field::new("_document", DataType::Utf8, true)));
    (
        Arc::new(Schema::new(unified_fields)),
        score_column_idx,
        document_column_idx,
    )
}

fn normalize_canonical_fast_field_schema(schema: &SchemaRef) -> SchemaRef {
    let mut fields = Vec::new();

    if schema
        .fields()
        .iter()
        .all(|field| field.name() != "_doc_id")
    {
        fields.push(Field::new("_doc_id", DataType::UInt32, false));
    }
    if schema
        .fields()
        .iter()
        .all(|field| field.name() != "_segment_ord")
    {
        fields.push(Field::new("_segment_ord", DataType::UInt32, false));
    }

    fields.extend(schema.fields().iter().map(|field| field.as_ref().clone()));
    Arc::new(Schema::new(fields))
}

fn analyze_fast_field_filter_support(
    expr: &Expr,
    tantivy_schema: &TantivySchema,
) -> FilterPushdownSupport {
    let Expr::BinaryExpr(binary) = expr else {
        return FilterPushdownSupport::Unsupported;
    };

    let column_name = match (binary.left.as_ref(), binary.right.as_ref()) {
        (Expr::Column(col), Expr::Literal(_, _)) => Some(col.name.as_str()),
        (Expr::Literal(_, _), Expr::Column(col)) => Some(col.name.as_str()),
        _ => None,
    };
    let Some(column_name) = column_name else {
        return FilterPushdownSupport::Unsupported;
    };

    if tantivy_schema.get_field(column_name).is_err() {
        return FilterPushdownSupport::MissingField;
    }

    if logical_expr_to_tantivy_query(expr, tantivy_schema).is_some() {
        FilterPushdownSupport::Query
    } else {
        FilterPushdownSupport::Unsupported
    }
}

fn filter_is_pushdown_safe(expr: &Expr, splits: &[SplitDescriptor]) -> bool {
    let mut any_query = false;

    for split in splits {
        match analyze_fast_field_filter_support(expr, &split.opener.schema()) {
            FilterPushdownSupport::Query => any_query = true,
            FilterPushdownSupport::MissingField => {}
            FilterPushdownSupport::Unsupported => return false,
        }
    }

    any_query
}

pub(crate) fn build_split_fast_field_query(
    exprs: &[Expr],
    tantivy_schema: &TantivySchema,
) -> Option<Arc<dyn tantivy::query::Query>> {
    let mut queries: Vec<Box<dyn tantivy::query::Query>> = Vec::new();

    for expr in exprs {
        match analyze_fast_field_filter_support(expr, tantivy_schema) {
            FilterPushdownSupport::Query => {
                if let Some(query) = logical_expr_to_tantivy_query(expr, tantivy_schema) {
                    queries.push(query);
                }
            }
            FilterPushdownSupport::MissingField => {
                return Some(Arc::new(tantivy::query::EmptyQuery));
            }
            FilterPushdownSupport::Unsupported => return None,
        }
    }

    match queries.len() {
        0 => None,
        1 => queries.into_iter().next().map(Arc::from),
        _ => Some(Arc::new(tantivy::query::BooleanQuery::intersection(
            queries,
        ))),
    }
}

fn translate_partition_stat(
    stat: Option<&PartitionStat>,
    projection: &FastFieldProjectionPlan,
) -> Option<PartitionStat> {
    let stat = stat?;
    let mut column_stats = Vec::new();

    for column in &projection.columns {
        let Some(source_name) = &column.source_name else {
            continue;
        };
        if !matches!(
            column.coercion,
            crate::type_coercion::FastFieldCoercion::Exact
        ) {
            continue;
        }
        if let Some((_, min_val, max_val)) = stat
            .column_stats
            .iter()
            .find(|(name, _, _)| name == source_name)
        {
            column_stats.push((
                column.output_field.name().to_string(),
                min_val.clone(),
                max_val.clone(),
            ));
        }
    }

    Some(PartitionStat {
        num_rows: stat.num_rows,
        has_deletes: stat.has_deletes,
        column_stats,
    })
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
    splits: Vec<SplitDescriptor>,
    unified_schema: SchemaRef,
    fast_field_schema: SchemaRef,
    score_column_idx: usize,
    document_column_idx: usize,
}

impl SingleTableProvider {
    /// Create a provider from an already-opened tantivy index.
    #[must_use]
    pub fn new(index: Index) -> Self {
        let ff_schema = tantivy_schema_to_arrow_from_index(&index);
        Self::from_splits_with_fast_field_schema(
            vec![Arc::new(DirectIndexOpener::new(index))],
            ff_schema,
        )
        .expect("self-derived canonical schema should always be executable")
    }

    /// Create a provider from an [`IndexOpener`] for deferred index opening.
    ///
    /// When the opener is a local [`DirectIndexOpener`], this still inspects
    /// segment cardinality so multi-valued fast fields become `List<T>`.
    /// Generic remote openers fall back to schema-only mapping; workers rely
    /// on serialized `multi_valued_fields` metadata to recover `List<T>`.
    #[must_use]
    pub fn from_opener(opener: Arc<dyn IndexOpener>) -> Self {
        let ff_schema = fast_field_schema_for_opener(&opener);
        Self::from_splits_with_fast_field_schema(vec![opener], ff_schema)
            .expect("self-derived canonical schema should always be executable")
    }

    /// Create a provider spanning multiple split openers.
    ///
    /// The canonical fast field schema is inferred by strict union on field
    /// names, with one promotion rule: if any split exposes a field as
    /// `List<T>` and another exposes the same field as scalar `T`, the
    /// canonical schema uses `List<T>`.
    pub fn from_splits(split_openers: Vec<Arc<dyn IndexOpener>>) -> Result<Self> {
        if split_openers.is_empty() {
            return Err(DataFusionError::Plan(
                "SingleTableProvider requires at least one split opener".into(),
            ));
        }

        let split_schemas: Vec<SchemaRef> = split_openers
            .iter()
            .map(fast_field_schema_for_opener)
            .collect();
        let canonical_ff_schema = infer_canonical_fast_field_schema(&split_schemas)?;
        Self::from_splits_with_fast_field_schema(split_openers, canonical_ff_schema)
    }

    /// Create a provider spanning multiple split openers with an explicit
    /// canonical fast field schema.
    pub fn from_splits_with_fast_field_schema(
        split_openers: Vec<Arc<dyn IndexOpener>>,
        fast_field_schema: SchemaRef,
    ) -> Result<Self> {
        if split_openers.is_empty() {
            return Err(DataFusionError::Plan(
                "SingleTableProvider requires at least one split opener".into(),
            ));
        }

        let fast_field_schema = normalize_canonical_fast_field_schema(&fast_field_schema);

        let splits: Vec<SplitDescriptor> = split_openers
            .into_iter()
            .map(build_split_descriptor)
            .collect::<Result<_>>()?;

        for split in &splits {
            plan_fast_field_projection(&split.fast_field_schema, &fast_field_schema)?;
        }

        let (unified_schema, score_column_idx, document_column_idx) =
            build_unified_schema(&fast_field_schema);

        Ok(Self {
            splits,
            unified_schema,
            fast_field_schema,
            score_column_idx,
            document_column_idx,
        })
    }
}

impl fmt::Debug for SingleTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SingleTableProvider")
            .field("splits", &self.splits.len())
            .field("unified_schema", &self.unified_schema)
            .field("fast_field_schema", &self.fast_field_schema)
            .field("score_column_idx", &self.score_column_idx)
            .field("document_column_idx", &self.document_column_idx)
            .finish_non_exhaustive()
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
        Ok(filters
            .iter()
            .map(|f| {
                if extract_full_text_call(f).is_some() {
                    TableProviderFilterPushDown::Exact
                } else if filter_is_pushdown_safe(f, &self.splits) {
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
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. Extract full_text() calls and fast field filters from pushed-down filters.
        let mut raw_queries: Vec<(String, String)> = Vec::new();
        let mut fast_field_filter_exprs: Vec<Expr> = Vec::new();
        for filter in filters {
            if let Some((field_name, query_string)) = extract_full_text_call(filter) {
                let field_exists = self
                    .splits
                    .iter()
                    .any(|split| split.opener.schema().get_field(&field_name).is_ok());
                if !field_exists {
                    return Err(DataFusionError::Plan(format!(
                        "full_text: field '{field_name}' not found in any split"
                    )));
                }
                raw_queries.push((field_name, query_string));
            } else if filter_is_pushdown_safe(filter, &self.splits) {
                fast_field_filter_exprs.push(filter.clone());
            }
        }
        let cached_pre_built_query = if self.splits.len() == 1 {
            build_split_fast_field_query(&fast_field_filter_exprs, &self.splits[0].opener.schema())
        } else {
            None
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

        ff_indices.sort();
        ff_indices.dedup();
        // _doc_id is always needed: as a fallback when no fast fields are
        // projected, and for async document fetch when needs_document is true.
        let doc_id_idx = self.fast_field_schema.index_of("_doc_id").map_err(|_| {
            DataFusionError::Internal(
                "fast field schema missing required _doc_id column".into(),
            )
        })?;
        if ff_indices.is_empty() || (needs_document && !ff_indices.contains(&doc_id_idx)) {
            ff_indices.push(doc_id_idx);
            ff_indices.sort();
            ff_indices.dedup();
        }

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

        let split_plans: Vec<SplitExecutionPlan> = self
            .splits
            .iter()
            .map(|split| {
                Ok(SplitExecutionPlan {
                    opener: Arc::clone(&split.opener),
                    fast_field_projection: plan_fast_field_projection(
                        &split.fast_field_schema,
                        &ff_projected_schema,
                    )?,
                })
            })
            .collect::<Result<_>>()?;

        let mut partition_map = Vec::new();
        let mut partition_stats = Vec::new();
        for (split_idx, split) in self.splits.iter().enumerate() {
            for segment_idx in 0..split.num_segments {
                partition_map.push(PartitionSpec {
                    split_idx,
                    segment_idx,
                });
                partition_stats.push(translate_partition_stat(
                    split
                        .partition_stats
                        .get(segment_idx)
                        .and_then(|stat| stat.as_ref()),
                    &split_plans[split_idx].fast_field_projection,
                ));
            }
        }

        let data_source = SingleTableDataSource {
            splits: split_plans,
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
            pre_built_query: cached_pre_built_query,
            fast_field_filter_exprs,
            topk: None,
            row_limit: limit,
            partition_map,
            partition_stats,
            warmup_done: self
                .splits
                .iter()
                .map(|_| Arc::new(tokio::sync::OnceCell::new()))
                .collect(),
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
    splits: Vec<SplitExecutionPlan>,
    schema: ScanSchema,
    raw_queries: Vec<(String, String)>,
    /// Cached fast field query for the single-split case.
    pre_built_query: Option<Arc<dyn tantivy::query::Query>>,
    /// Source logical `Expr`s that were successfully converted to tantivy
    /// queries. Stored for serialization and for split-specific query
    /// reconstruction at execution time.
    fast_field_filter_exprs: Vec<Expr>,
    pub(crate) topk: Option<usize>,
    row_limit: Option<usize>,
    partition_map: Vec<PartitionSpec>,
    /// Per-partition (segment) statistics for partition pruning.
    /// Indexed by partition number. `None` means stats are unavailable for
    /// that partition (e.g. remote opener without metadata).
    partition_stats: Vec<Option<PartitionStat>>,
    /// Ensures warmup runs at most once per split.
    warmup_done: Vec<Arc<tokio::sync::OnceCell<()>>>,
    /// Shared metrics set for all partitions.
    metrics: ExecutionPlanMetricsSet,
}

impl SingleTableDataSource {
    /// Construct a `SingleTableDataSource` directly from deserialized codec
    /// fields, bypassing `TableProvider::scan` and `SessionContext`. Used by
    /// `TantivyCodec::try_decode` to reconstruct a `DataSourceExec` on workers.
    pub(crate) fn new_from_codec(
        splits: Vec<SplitExecutionPlan>,
        schema: ScanSchema,
        raw_queries: Vec<(String, String)>,
        fast_field_filter_exprs: Vec<Expr>,
        topk: Option<usize>,
        row_limit: Option<usize>,
        partition_map: Vec<PartitionSpec>,
    ) -> Self {
        let warmup_done = splits
            .iter()
            .map(|_| Arc::new(tokio::sync::OnceCell::new()))
            .collect();
        Self {
            pre_built_query: if splits.len() == 1 {
                build_split_fast_field_query(&fast_field_filter_exprs, &splits[0].opener.schema())
            } else {
                None
            },
            splits,
            schema,
            raw_queries,
            fast_field_filter_exprs,
            topk,
            row_limit,
            partition_stats: vec![None; partition_map.len()],
            partition_map,
            warmup_done,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    fn clone_with(&self, f: impl FnOnce(&mut Self)) -> Self {
        let mut new = SingleTableDataSource {
            splits: self.splits.clone(),
            schema: self.schema.clone(),
            raw_queries: self.raw_queries.clone(),
            pre_built_query: self.pre_built_query.clone(),
            fast_field_filter_exprs: self.fast_field_filter_exprs.clone(),
            topk: self.topk,
            row_limit: self.row_limit,
            partition_map: self.partition_map.clone(),
            partition_stats: self.partition_stats.clone(),
            warmup_done: self.warmup_done.clone(),
            metrics: self.metrics.clone(),
        };
        f(&mut new);
        new
    }

    /// Access the split openers.
    pub fn split_openers(&self) -> Vec<Arc<dyn IndexOpener>> {
        self.splits
            .iter()
            .map(|split| Arc::clone(&split.opener))
            .collect()
    }

    /// Access the sole opener when the data source spans a single split.
    pub fn single_split_opener(&self) -> Option<&Arc<dyn IndexOpener>> {
        match self.splits.as_slice() {
            [split] => Some(&split.opener),
            _ => None,
        }
    }

    /// Access the raw full-text queries.
    pub fn raw_queries(&self) -> &[(String, String)] {
        &self.raw_queries
    }

    /// The number of partitions this data source is partitioned over.
    pub fn num_segments(&self) -> usize {
        self.partition_map.len()
    }

    /// Access the topk limit.
    pub fn topk(&self) -> Option<usize> {
        self.topk
    }

    /// Access the per-partition scan row limit derived from planner hints.
    pub fn row_limit(&self) -> Option<usize> {
        self.row_limit
    }

    /// Whether this data source has an active query.
    pub fn has_query(&self) -> bool {
        !self.raw_queries.is_empty()
            || self.pre_built_query.is_some()
            || !self.fast_field_filter_exprs.is_empty()
    }

    /// Create a copy with the topk limit set.
    #[must_use]
    pub fn with_topk(&self, topk: usize) -> Self {
        self.clone_with(|s| s.topk = Some(topk))
    }

    /// Access the pre-built tantivy query from fast field filters.
    pub fn pre_built_query(&self) -> Option<&Arc<dyn tantivy::query::Query>> {
        self.pre_built_query.as_ref()
    }

    /// Access the canonical fast field schema for this scan.
    pub fn canonical_fast_field_schema(&self) -> SchemaRef {
        let fields: Vec<Field> = self.schema.unified.fields()[..self.schema.score_idx]
            .iter()
            .map(|field| field.as_ref().clone())
            .collect();
        Arc::new(Schema::new(fields))
    }

    /// Access the source logical `Expr`s that produced `pre_built_query`.
    /// Used by the codec for serialization.
    pub fn fast_field_filter_exprs(&self) -> &[Expr] {
        &self.fast_field_filter_exprs
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
    pub fn projection(&self) -> Option<&[usize]> {
        self.schema.projection.as_deref()
    }

    /// Whether _score is needed.
    pub fn needs_score(&self) -> bool {
        self.schema.needs_score
    }

    /// Whether _document is needed.
    pub fn needs_document(&self) -> bool {
        self.schema.needs_document
    }

    fn output_orderings(&self) -> Vec<LexOrdering> {
        let schema = Arc::clone(&self.schema.projected);

        if self.topk.is_some() && self.schema.needs_score {
            if let Ok(score_idx) = schema.index_of("_score") {
                if let Some(ordering) = LexOrdering::new([PhysicalSortExpr::new(
                    Arc::new(PhysicalColumn::new("_score", score_idx)),
                    SortOptions {
                        descending: true,
                        nulls_first: false,
                    },
                )]) {
                    return vec![ordering];
                }
            }
            return Vec::new();
        }

        if let Ok(doc_id_idx) = schema.index_of("_doc_id") {
            if let Some(ordering) = LexOrdering::new([PhysicalSortExpr::new(
                Arc::new(PhysicalColumn::new("_doc_id", doc_id_idx)),
                SortOptions::default(),
            )]) {
                return vec![ordering];
            }
        }

        Vec::new()
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
        let partition_spec = *self.partition_map.get(partition).ok_or_else(|| {
            DataFusionError::Internal(format!("invalid partition index {partition}"))
        })?;
        let split = self
            .splits
            .get(partition_spec.split_idx)
            .ok_or_else(|| DataFusionError::Internal("invalid split index".to_string()))?
            .clone();
        let opener = Arc::clone(&split.opener);
        let segment_idx = partition_spec.segment_idx;
        let raw_queries = self.raw_queries.clone();
        let ff_projected_schema = self.schema.ff_projected.clone();
        let source_ff_schema = split.fast_field_projection.source_schema.clone();
        let fast_field_projection = split.fast_field_projection.clone();
        let unified_schema = self.schema.unified.clone();
        let projected_schema = self.schema.projected.clone();
        let projection = self.schema.projection.clone();
        let needs_score = self.schema.needs_score;
        let needs_document = self.schema.needs_document;
        let score_column_idx = self.schema.score_idx;
        let document_column_idx = self.schema.document_idx;
        let topk = self.topk;
        let pre_built_query = if self.splits.len() == 1 {
            self.pre_built_query
                .as_ref()
                .map(|q| Arc::from(q.box_clone()))
        } else {
            None
        };
        let fast_field_filter_exprs = self.fast_field_filter_exprs.clone();
        let warmup_done = Arc::clone(
            self.warmup_done
                .get(partition_spec.split_idx)
                .ok_or_else(|| DataFusionError::Internal("invalid warmup split index".into()))?,
        );
        let needs_warmup = opener.needs_warmup();
        let row_limit = self.row_limit;
        let cancelled = Arc::new(AtomicBool::new(false));

        let schema = self.schema.projected.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(2);

        let cancelled_task = Arc::clone(&cancelled);
        let handle = tokio::spawn(async move {
            // Async setup: open index + warmup
            let index = match opener.open().await {
                Ok(i) => i,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            // Skip warmup for local/mmap openers — data already accessible.
            if needs_warmup {
                let index_ref = index.clone();
                let ff_schema_ref = ff_projected_schema.clone();
                let rq_ref = raw_queries.clone();
                let ff_filter_exprs_ref = fast_field_filter_exprs.clone();
                let needs_document_ref = needs_document;
                if let Err(err) = warmup_done
                    .get_or_try_init(|| async move {
                        let mut ff_names: std::collections::BTreeSet<String> = ff_schema_ref
                            .fields()
                            .iter()
                            .filter_map(|field| {
                                index_ref
                                    .schema()
                                    .get_field(field.name())
                                    .ok()
                                    .map(|_| field.name().to_string())
                            })
                            .collect();
                        ff_names.extend(crate::warmup::fast_field_filter_field_names(
                            &index_ref.schema(),
                            &ff_filter_exprs_ref,
                        )?);
                        if !ff_names.is_empty() {
                            let ff_names: Vec<String> = ff_names.into_iter().collect();
                            let ff_name_refs: Vec<&str> =
                                ff_names.iter().map(String::as_str).collect();
                            crate::warmup::warmup_fast_fields_by_name(&index_ref, &ff_name_refs)
                                .await?;
                        }

                        let queried_fields: Vec<tantivy::schema::Field> = rq_ref
                            .iter()
                            .filter_map(|(field_name, _)| {
                                index_ref.schema().get_field(field_name).ok()
                            })
                            .collect();
                        if !queried_fields.is_empty() {
                            crate::warmup::warmup_inverted_index(&index_ref, &queried_fields)
                                .await?;
                        }

                        // Document store warmup is NOT needed — document retrieval
                        // uses Searcher::doc_async() which reads blocks on demand
                        // via the async I/O path.

                        Ok::<(), DataFusionError>(())
                    })
                    .await
                {
                    let _ = tx.send(Err(err)).await;
                    return;
                }
            } // end if needs_warmup

            // Blocking batch generation — batches exclude _document. When
            // needs_document, the async loop below adds it via doc_async.
            let (raw_tx, mut raw_rx) = tokio::sync::mpsc::channel::<Result<RecordBatch>>(2);
            let index_for_docs = index.clone();
            let output_schema_for_docs = projected_schema.clone();
            let cancelled_blocking = Arc::clone(&cancelled_task);

            let blocking_handle = tokio::task::spawn_blocking(move || {
                let split_fast_field_query = match pre_built_query {
                    Some(query) => Some(query),
                    None => build_split_fast_field_query(&fast_field_filter_exprs, &index.schema()),
                };
                let query =
                    build_combined_query(&index, split_fast_field_query.as_ref(), &raw_queries)?;
                let cfg = ScanConfig {
                    index,
                    segment_idx,
                    batch_size,
                    source_ff_schema,
                    fast_field_projection,
                    unified_schema,
                    projected_schema,
                    projection,
                    score_column_idx,
                    document_column_idx,
                    needs_score,
                    needs_document,
                    topk,
                    row_limit,
                    query,
                    cancelled: cancelled_blocking,
                };
                generate_single_table_batch_streaming(&cfg, |batch| {
                    raw_tx.blocking_send(Ok(batch)).is_ok()
                })
            });

            // Forward batches, adding _document column async when needed.
            let tantivy_schema = index_for_docs.schema();
            while let Some(result) = raw_rx.recv().await {
                let to_send = match result {
                    Ok(batch) if needs_document => {
                        fill_document_column_async(
                            batch,
                            &index_for_docs,
                            segment_idx,
                            &output_schema_for_docs,
                            &tantivy_schema,
                        )
                        .await
                    }
                    other => other,
                };
                if tx.send(to_send).await.is_err() {
                    break;
                }
            }

            // Propagate errors from the blocking task.
            match blocking_handle.await {
                Ok(Err(e)) => {
                    let _ = tx.send(Err(e)).await;
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(DataFusionError::Internal(format!(
                            "spawn_blocking: {e}"
                        ))))
                        .await;
                }
                Ok(Ok(())) => {}
            }
        });
        let guard = AbortOnDrop { handle, cancelled };

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
            "SingleTableDataSource(partitions={}, query={}, score={}, document={}, topk={:?})",
            self.partition_map.len(),
            self.has_query(),
            self.schema.needs_score,
            self.schema.needs_document,
            self.topk,
        )
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.partition_map.len())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        let orderings = self.output_orderings();
        if orderings.is_empty() {
            EquivalenceProperties::new(Arc::clone(&self.schema.projected))
        } else {
            EquivalenceProperties::new_with_orderings(Arc::clone(&self.schema.projected), orderings)
        }
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
        if self.partition_map.len() <= 1 {
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
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            results,
        ))
    }
}

// ---------------------------------------------------------------------------
// Async document column fill
// ---------------------------------------------------------------------------

/// Add the `_document` column to a batch that was produced without it.
///
/// Reads `_doc_id` from the batch, fetches each document via
/// `Searcher::doc_async` (per-block async I/O), and returns a new batch
/// with `_document` inserted at the correct position per the output schema.
async fn fill_document_column_async(
    batch: RecordBatch,
    index: &Index,
    segment_idx: usize,
    output_schema: &SchemaRef,
    tantivy_schema: &tantivy::schema::Schema,
) -> Result<RecordBatch> {
    let intermediate_schema = batch.schema();

    // Find _doc_id to get document addresses.
    let doc_id_idx = intermediate_schema.index_of("_doc_id").map_err(|_| {
        DataFusionError::Internal(
            "_doc_id column required for document fetch but not in batch".into(),
        )
    })?;
    let doc_ids = batch
        .column(doc_id_idx)
        .as_any()
        .downcast_ref::<arrow::array::UInt32Array>()
        .ok_or_else(|| DataFusionError::Internal("_doc_id column is not UInt32".into()))?;

    let reader = index
        .reader()
        .map_err(|e| DataFusionError::Internal(format!("open reader for doc fetch: {e}")))?;
    let searcher = reader.searcher();

    // Fetch documents async — each call reads only the specific store block needed.
    let mut doc_builder = StringBuilder::with_capacity(doc_ids.len(), doc_ids.len() * 256);
    for idx in 0..doc_ids.len() {
        let doc_id = doc_ids.value(idx);
        let doc_addr = tantivy::DocAddress::new(segment_idx as u32, doc_id);
        let doc: tantivy::TantivyDocument = searcher
            .doc_async(doc_addr)
            .await
            .map_err(|e| DataFusionError::Internal(format!("async doc fetch {doc_id}: {e}")))?;
        doc_builder.append_value(doc.to_json(tantivy_schema));
    }
    let doc_array: Arc<dyn arrow::array::Array> = Arc::new(doc_builder.finish());

    // Build output columns in the order defined by output_schema,
    // pulling from the intermediate batch or inserting _document.
    let mut output_columns: Vec<Arc<dyn arrow::array::Array>> =
        Vec::with_capacity(output_schema.fields().len());

    for field in output_schema.fields() {
        if field.name() == "_document" {
            output_columns.push(Arc::clone(&doc_array));
        } else {
            let col_idx = intermediate_schema.index_of(field.name()).map_err(|_| {
                DataFusionError::Internal(format!(
                    "column '{}' not found in intermediate batch",
                    field.name()
                ))
            })?;
            output_columns.push(batch.column(col_idx).clone());
        }
    }

    RecordBatch::try_new(Arc::clone(output_schema), output_columns)
        .map_err(|e| DataFusionError::Internal(format!("build batch with docs: {e}")))
}

// ---------------------------------------------------------------------------
// Core batch generation
// ---------------------------------------------------------------------------

/// Holds immutable context for assembling one `RecordBatch` from a chunk of
/// doc_ids + optional scores. Created once per segment and reused for every
/// chunk, avoiding 14-parameter function signatures.
struct ChunkBuilder<'a> {
    segment_reader: &'a tantivy::SegmentReader,
    source_ff_schema: &'a SchemaRef,
    fast_field_projection: &'a FastFieldProjectionPlan,
    unified_schema: &'a SchemaRef,
    projected_schema: &'a SchemaRef,
    projected_indices: Vec<usize>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    segment_ord: u32,
    dict_cache: DictCache,
}

impl<'a> ChunkBuilder<'a> {
    /// Assemble a `RecordBatch` from a chunk of doc_ids and optional scores.
    ///
    /// Produces fast fields and scores only — `_document` is excluded here and
    /// added asynchronously by `fill_document_column_async` after this batch
    /// exits `spawn_blocking`. The returned schema is `intermediate_schema`
    /// (projected schema minus `_document`).
    fn build(&self, chunk_ids: &[u32], chunk_scores: Option<&[f32]>) -> Result<RecordBatch> {
        let source_ff_batch = read_segment_fast_fields_to_batch(
            self.segment_reader,
            self.source_ff_schema,
            Some(chunk_ids),
            None,
            None,
            self.segment_ord,
            Some(&self.dict_cache),
        )?;
        let ff_batch = apply_fast_field_projection(&source_ff_batch, self.fast_field_projection)?;
        let chunk_rows = ff_batch.num_rows();

        let score_array: Option<Arc<dyn arrow::array::Array>> = if self.needs_score {
            match chunk_scores {
                Some(sc) => Some(Arc::new(Float32Array::from_iter_values(sc.iter().copied()))),
                None => Some(arrow::array::new_null_array(&DataType::Float32, chunk_rows)),
            }
        } else {
            None
        };

        let mut output_columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        let mut output_fields: Vec<arrow::datatypes::Field> = Vec::new();

        // When _document is needed, always include _doc_id in the intermediate
        // batch so fill_document_column_async can find document addresses.
        let doc_id_already_projected = self
            .projected_indices
            .iter()
            .any(|&idx| self.unified_schema.field(idx).name() == "_doc_id");
        if self.needs_document && !doc_id_already_projected {
            let doc_id_name = "_doc_id";
            if let Ok(ff_idx) = ff_batch.schema().index_of(doc_id_name) {
                output_columns.push(ff_batch.column(ff_idx).clone());
                output_fields.push(
                    arrow::datatypes::Field::new(doc_id_name, DataType::UInt32, false),
                );
            }
        }

        for &unified_idx in &self.projected_indices {
            if unified_idx == self.document_column_idx {
                // Skip _document — it's added async after spawn_blocking.
                continue;
            } else if unified_idx == self.score_column_idx {
                output_columns.push(score_array.clone().unwrap_or_else(|| {
                    arrow::array::new_null_array(&DataType::Float32, chunk_rows)
                }));
                output_fields.push(self.unified_schema.field(unified_idx).clone());
            } else {
                let col_name = self.unified_schema.field(unified_idx).name();
                let ff_col_idx = ff_batch.schema().index_of(col_name).map_err(|_| {
                    DataFusionError::Internal(format!(
                        "fast field column '{col_name}' not found in ff_batch"
                    ))
                })?;
                output_columns.push(ff_batch.column(ff_col_idx).clone());
                output_fields.push(self.unified_schema.field(unified_idx).clone());
            }
        }

        let intermediate_schema = Arc::new(Schema::new(output_fields));
        RecordBatch::try_new(intermediate_schema, output_columns)
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
    source_ff_schema: SchemaRef,
    fast_field_projection: FastFieldProjectionPlan,
    unified_schema: SchemaRef,
    projected_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    score_column_idx: usize,
    document_column_idx: usize,
    needs_score: bool,
    needs_document: bool,
    topk: Option<usize>,
    row_limit: Option<usize>,
    query: Option<Arc<dyn tantivy::query::Query>>,
    cancelled: Arc<AtomicBool>,
}

/// Thin orchestrator: query execution is delegated to the streaming helpers in
/// [`crate::util`] and batch assembly to [`ChunkBuilder::build`].
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

    // Build the chunk builder once for this segment.
    // Document store is NOT opened here — document retrieval is async and
    // happens outside spawn_blocking via fill_document_column_async.
    let needs_document = cfg.needs_document;

    let projected_indices: Vec<usize> = match &cfg.projection {
        Some(indices) => indices.clone(),
        None => (0..cfg.unified_schema.fields().len()).collect(),
    };

    let dict_cache = DictCache::build(segment_reader, &cfg.source_ff_schema)?;

    let builder = ChunkBuilder {
        segment_reader,
        source_ff_schema: &cfg.source_ff_schema,
        fast_field_projection: &cfg.fast_field_projection,
        unified_schema: &cfg.unified_schema,
        projected_schema: &cfg.projected_schema,
        projected_indices,
        score_column_idx: cfg.score_column_idx,
        document_column_idx: cfg.document_column_idx,
        needs_score,
        needs_document,
        segment_ord: segment_idx as u32,
        dict_cache,
    };
    let mut remaining = cfg.row_limit.unwrap_or(usize::MAX);

    if let Some(topk) = cfg.topk {
        let effective_topk = remaining.min(topk);
        if effective_topk == 0 {
            return Ok(());
        }
        let (doc_ids, scores) = collect_topk_docs(
            segment_reader,
            &searcher,
            cfg.query.as_ref(),
            effective_topk,
        )?;
        if doc_ids.is_empty() {
            return Ok(());
        }

        let total = doc_ids.len();
        let mut offset = 0;
        while offset < total && remaining > 0 {
            if cfg.cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }
            let end = (offset + batch_size).min(total).min(offset + remaining);
            let batch = builder.build(&doc_ids[offset..end], Some(&scores[offset..end]))?;
            if batch.num_rows() > 0 && !emit(batch) {
                return Ok(());
            }
            remaining -= end - offset;
            offset = end;
        }
        return Ok(());
    }

    for_each_matching_doc_chunks(
        MatchingDocChunksConfig {
            segment_reader,
            searcher: &searcher,
            query: cfg.query.as_ref(),
            index_schema: &index.schema(),
            needs_score,
            batch_size,
            cancelled: cfg.cancelled.as_ref(),
        },
        |chunk_ids, chunk_scores| {
            if remaining == 0 || cfg.cancelled.load(Ordering::Relaxed) {
                return Ok(false);
            }

            let take = remaining.min(chunk_ids.len());
            let batch = builder.build(
                &chunk_ids[..take],
                chunk_scores.map(|scores| &scores[..take]),
            )?;
            remaining -= take;
            if batch.num_rows() > 0 && !emit(batch) {
                return Ok(false);
            }
            Ok(remaining > 0)
        },
    )
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
            let gt = Box::new(RangeQuery::new(Bound::Excluded(term), Bound::Unbounded));
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
        (FieldType::Str(_), ScalarValue::Utf8(Some(s))) => Some(Term::from_field_text(field, s)),
        // Date — tantivy DateTime stores nanoseconds internally.
        (FieldType::Date(_), ScalarValue::TimestampMicrosecond(Some(v), _)) => Some(
            Term::from_field_date(field, DateTime::from_timestamp_micros(*v)),
        ),
        (FieldType::Date(_), ScalarValue::TimestampSecond(Some(v), _)) => Some(
            Term::from_field_date(field, DateTime::from_timestamp_secs(*v)),
        ),
        (FieldType::Date(_), ScalarValue::TimestampMillisecond(Some(v), _)) => Some(
            Term::from_field_date(field, DateTime::from_timestamp_millis(*v)),
        ),
        (FieldType::Date(_), ScalarValue::TimestampNanosecond(Some(v), _)) => Some(
            Term::from_field_date(field, DateTime::from_timestamp_nanos(*v)),
        ),
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
    serde_json::to_string(&filters)
        .map_err(|e| DataFusionError::Internal(format!("serialize fast field filters: {e}")))
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
    let filters: Vec<FastFieldFilter> = serde_json::from_str(json)
        .map_err(|e| DataFusionError::Internal(format!("deserialize fast field filters: {e}")))?;
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
        let query = logical_expr_to_tantivy_query(&expr, tantivy_schema).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "failed to reconstruct fast field filter '{}' {} during codec decode",
                f.field, f.op
            ))
        })?;
        queries.push(query);
    }
    Ok(queries)
}

/// Deserialize fast field filter JSON back into logical `Expr`s.
pub(crate) fn deserialize_fast_field_filter_exprs(json: &str) -> Result<Vec<Expr>> {
    if json.is_empty() {
        return Ok(Vec::new());
    }

    let filters: Vec<FastFieldFilter> = serde_json::from_str(json)
        .map_err(|e| DataFusionError::Internal(format!("deserialize fast field filters: {e}")))?;

    filters
        .into_iter()
        .map(|filter| {
            let scalar = json_pair_to_scalar(&filter.value, &filter.value_type)?;
            let op = match filter.op.as_str() {
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

            Ok(Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(Expr::Column(datafusion::common::Column::new_unqualified(
                    filter.field,
                ))),
                op,
                right: Box::new(Expr::Literal(scalar, None)),
            }))
        })
        .collect()
}

/// Encode a `ScalarValue` as a `(value_string, type_tag)` pair for JSON.
fn scalar_to_json_pair(scalar: &ScalarValue) -> Option<(String, String)> {
    let timestamp_tag = |prefix: &str, tz: &Option<Arc<str>>| match tz {
        Some(tz) => format!("{prefix}:{tz}"),
        None => prefix.to_string(),
    };

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
        ScalarValue::TimestampSecond(Some(v), tz) => {
            Some((v.to_string(), timestamp_tag("ts_s", tz)))
        }
        ScalarValue::TimestampMillisecond(Some(v), tz) => {
            Some((v.to_string(), timestamp_tag("ts_ms", tz)))
        }
        ScalarValue::TimestampMicrosecond(Some(v), tz) => {
            Some((v.to_string(), timestamp_tag("ts_us", tz)))
        }
        ScalarValue::TimestampNanosecond(Some(v), tz) => {
            Some((v.to_string(), timestamp_tag("ts_ns", tz)))
        }
        _ => None,
    }
}

/// Decode a `(value_string, type_tag)` pair back to a `ScalarValue`.
fn json_pair_to_scalar(value: &str, value_type: &str) -> Result<ScalarValue> {
    let parse_timestamp_tag = |prefix: &str| -> Option<Option<Arc<str>>> {
        if value_type == prefix {
            Some(None)
        } else {
            value_type
                .strip_prefix(&format!("{prefix}:"))
                .map(|tz| Some(Arc::<str>::from(tz)))
        }
    };
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
                .map_err(|e| DataFusionError::Internal(format!("decode base64 binary: {e}")))?;
            Ok(ScalarValue::Binary(Some(bytes)))
        }
        _ => {
            if let Some(tz) = parse_timestamp_tag("ts_s") {
                return Ok(ScalarValue::TimestampSecond(
                    Some(value.parse().map_err(parse_err)?),
                    tz,
                ));
            }
            if let Some(tz) = parse_timestamp_tag("ts_ms") {
                return Ok(ScalarValue::TimestampMillisecond(
                    Some(value.parse().map_err(parse_err)?),
                    tz,
                ));
            }
            if let Some(tz) = parse_timestamp_tag("ts_us") {
                return Ok(ScalarValue::TimestampMicrosecond(
                    Some(value.parse().map_err(parse_err)?),
                    tz,
                ));
            }
            if let Some(tz) = parse_timestamp_tag("ts_ns") {
                return Ok(ScalarValue::TimestampNanosecond(
                    Some(value.parse().map_err(parse_err)?),
                    tz,
                ));
            }

            Err(DataFusionError::Internal(format!(
                "unknown scalar type tag: {value_type}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{json_pair_to_scalar, scalar_to_json_pair};
    use datafusion::common::ScalarValue;
    use std::sync::Arc;

    #[test]
    fn test_timestamp_scalar_json_roundtrip_preserves_timezone() {
        let scalar =
            ScalarValue::TimestampMicrosecond(Some(1_234_567), Some(Arc::<str>::from("UTC")));
        let (value, tag) = scalar_to_json_pair(&scalar).unwrap();
        let decoded = json_pair_to_scalar(&value, &tag).unwrap();

        assert_eq!(decoded, scalar);
    }
}
