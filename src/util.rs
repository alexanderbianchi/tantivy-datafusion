//! Shared utility functions used by both the unified (single-table) and
//! decomposed (three-table join) provider approaches.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::query::{BooleanQuery, EnableScoring, QueryParser};
use tantivy::{DocId, DocSet, Index, Score, Searcher, SegmentReader, TERMINATED};

/// Combine pre-parsed queries with raw `(field_name, query_string)` pairs
/// into a single tantivy query via `BooleanQuery::intersection`.
///
/// Raw queries are parsed using `QueryParser::for_index`, which requires an
/// opened `Index`. Pre-parsed queries (e.g., from fast-field filter conversion)
/// are included as-is.
pub(crate) fn build_combined_query(
    index: &Index,
    pre_parsed: Option<&Arc<dyn tantivy::query::Query>>,
    raw_queries: &[(String, String)],
) -> Result<Option<Arc<dyn tantivy::query::Query>>> {
    let mut queries: Vec<Box<dyn tantivy::query::Query>> = Vec::new();

    if let Some(q) = pre_parsed {
        queries.push(q.box_clone());
    }

    let tantivy_schema = index.schema();
    for (field_name, query_string) in raw_queries {
        let field = match tantivy_schema.get_field(field_name) {
            Ok(field) => field,
            Err(_) => return Ok(Some(Arc::new(tantivy::query::EmptyQuery))),
        };
        let parser = QueryParser::for_index(index, vec![field]);
        let parsed = parser.parse_query(query_string).map_err(|e| {
            DataFusionError::Plan(format!("full_text: failed to parse '{query_string}': {e}"))
        })?;
        queries.push(parsed);
    }

    match queries.len() {
        0 => Ok(None),
        1 => Ok(queries.into_iter().next().map(Arc::from)),
        _ => Ok(Some(Arc::new(BooleanQuery::intersection(queries)))),
    }
}

fn flush_doc_buffer<F>(doc_buffer: &mut Vec<DocId>, on_chunk: &mut F) -> Result<bool>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    if doc_buffer.is_empty() {
        return Ok(true);
    }
    let keep_going = on_chunk(doc_buffer.as_slice(), None)?;
    doc_buffer.clear();
    Ok(keep_going)
}

fn flush_scored_buffer<F>(
    doc_buffer: &mut Vec<DocId>,
    score_buffer: &mut Vec<Score>,
    on_chunk: &mut F,
) -> Result<bool>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    if doc_buffer.is_empty() {
        return Ok(true);
    }
    let keep_going = on_chunk(doc_buffer.as_slice(), Some(score_buffer.as_slice()))?;
    doc_buffer.clear();
    score_buffer.clear();
    Ok(keep_going)
}

pub(crate) struct MatchingDocChunksConfig<'a> {
    pub(crate) segment_reader: &'a SegmentReader,
    pub(crate) searcher: &'a Searcher,
    pub(crate) query: Option<&'a Arc<dyn tantivy::query::Query>>,
    pub(crate) index_schema: &'a tantivy::schema::Schema,
    pub(crate) needs_score: bool,
    pub(crate) batch_size: usize,
    pub(crate) cancelled: &'a AtomicBool,
}

fn execute_scored_query_chunks<F>(
    segment_reader: &SegmentReader,
    searcher: &Searcher,
    query: &Arc<dyn tantivy::query::Query>,
    batch_size: usize,
    cancelled: &AtomicBool,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    let weight = query
        .weight(EnableScoring::enabled_from_searcher(searcher))
        .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
    let alive_bitset = segment_reader.alive_bitset();
    let mut scorer = weight
        .scorer(segment_reader, 1.0)
        .map_err(|e| DataFusionError::Internal(format!("create scorer: {e}")))?;
    let mut ids = Vec::with_capacity(batch_size);
    let mut scores = Vec::with_capacity(batch_size);
    let mut doc = scorer.doc();

    while doc != TERMINATED {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        if alive_bitset.is_none_or(|alive| alive.is_alive(doc)) {
            ids.push(doc);
            scores.push(scorer.score());

            if ids.len() == batch_size && !flush_scored_buffer(&mut ids, &mut scores, on_chunk)? {
                return Ok(());
            }
        }
        doc = scorer.advance();
    }

    flush_scored_buffer(&mut ids, &mut scores, on_chunk)?;
    Ok(())
}

fn execute_unscored_query_chunks<F>(
    segment_reader: &SegmentReader,
    index_schema: &tantivy::schema::Schema,
    query: &Arc<dyn tantivy::query::Query>,
    batch_size: usize,
    cancelled: &AtomicBool,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    let weight = query
        .weight(EnableScoring::disabled_from_schema(index_schema))
        .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
    let alive_bitset = segment_reader.alive_bitset();
    let mut docset = weight
        .scorer(segment_reader, 1.0)
        .map_err(|e| DataFusionError::Internal(format!("create scorer: {e}")))?;
    let mut raw_docs = [0u32; tantivy::COLLECT_BLOCK_BUFFER_LEN];
    let mut doc_buffer = Vec::with_capacity(batch_size);

    loop {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        let num_docs = docset.fill_buffer(&mut raw_docs);
        if num_docs == 0 {
            break;
        }
        for &doc in &raw_docs[..num_docs] {
            if alive_bitset.is_some_and(|alive| !alive.is_alive(doc)) {
                continue;
            }
            doc_buffer.push(doc);

            if doc_buffer.len() == batch_size && !flush_doc_buffer(&mut doc_buffer, on_chunk)? {
                return Ok(());
            }
        }
        if num_docs < raw_docs.len() {
            break;
        }
    }

    flush_doc_buffer(&mut doc_buffer, on_chunk)?;
    Ok(())
}

fn execute_full_scan_chunks<F>(
    segment_reader: &SegmentReader,
    batch_size: usize,
    cancelled: &AtomicBool,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    let max_doc = segment_reader.max_doc();
    let alive_bitset = segment_reader.alive_bitset();
    let mut doc_buffer = Vec::with_capacity(batch_size);

    for doc_id in 0..max_doc {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        if alive_bitset.is_none_or(|bitset| bitset.is_alive(doc_id)) {
            doc_buffer.push(doc_id);
        }
        if doc_buffer.len() == batch_size && !flush_doc_buffer(&mut doc_buffer, on_chunk)? {
            return Ok(());
        }
    }

    flush_doc_buffer(&mut doc_buffer, on_chunk)?;
    Ok(())
}

/// Execute a tantivy query on a single segment and stream matching doc ids in
/// `batch_size` chunks through `on_chunk`.
///
/// Four execution paths:
/// - TopK + scoring: handled separately by [`collect_topk_docs`]
/// - Full scoring: `for_each` with alive_bitset filtering in callback
/// - No scoring: `for_each_no_score` with alive_bitset filtering in callback
/// - No query: iterate all alive docs in the segment
pub(crate) fn for_each_matching_doc_chunks<F>(
    cfg: MatchingDocChunksConfig<'_>,
    mut on_chunk: F,
) -> Result<()>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    let MatchingDocChunksConfig {
        segment_reader,
        searcher,
        query,
        index_schema,
        needs_score,
        batch_size,
        cancelled,
    } = cfg;

    match (query, needs_score) {
        (Some(query), true) => execute_scored_query_chunks(
            segment_reader,
            searcher,
            query,
            batch_size,
            cancelled,
            &mut on_chunk,
        ),
        (Some(query), false) => execute_unscored_query_chunks(
            segment_reader,
            index_schema,
            query,
            batch_size,
            cancelled,
            &mut on_chunk,
        ),
        (None, _) => execute_full_scan_chunks(segment_reader, batch_size, cancelled, &mut on_chunk),
    }
}
