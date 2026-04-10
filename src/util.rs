//! Shared utility functions used by both the unified (single-table) and
//! decomposed (three-table join) provider approaches.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::collector::TopNComputer;
use tantivy::query::{BooleanQuery, EnableScoring, QueryParser};
use tantivy::{DocId, Index, Score, Searcher, SegmentReader};

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
        1 => Ok(Some(Arc::from(queries.into_iter().next().unwrap()))),
        _ => Ok(Some(Arc::new(BooleanQuery::intersection(queries)))),
    }
}

/// Execute a top-k tantivy query on a single segment and collect matching
/// doc ids with BM25 scores.
///
/// This path is already bounded by `k` via Block-WAND pruning and remains
/// separate from the streaming scan path.
pub(crate) fn collect_topk_docs(
    segment_reader: &SegmentReader,
    searcher: &Searcher,
    query: Option<&Arc<dyn tantivy::query::Query>>,
    topk: usize,
) -> Result<(Vec<DocId>, Vec<Score>)> {
    let query = query.ok_or_else(|| {
        DataFusionError::Internal("topk collection requires an active query".into())
    })?;

    let weight = query
        .weight(EnableScoring::enabled_from_searcher(searcher))
        .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
    let mut top_n: TopNComputer<Score, DocId, _> = TopNComputer::new(topk);
    let alive_bitset = segment_reader.alive_bitset();

    if let Some(alive_bitset) = alive_bitset {
        let mut threshold = Score::MIN;
        top_n.threshold = Some(threshold);
        weight
            .for_each_pruning(Score::MIN, segment_reader, &mut |doc, score| {
                if alive_bitset.is_deleted(doc) {
                    return threshold;
                }
                top_n.push(score, doc);
                threshold = top_n.threshold.unwrap_or(Score::MIN);
                threshold
            })
            .map_err(|e| DataFusionError::Internal(format!("topk query execution: {e}")))?;
    } else {
        weight
            .for_each_pruning(Score::MIN, segment_reader, &mut |doc, score| {
                top_n.push(score, doc);
                top_n.threshold.unwrap_or(Score::MIN)
            })
            .map_err(|e| DataFusionError::Internal(format!("topk query execution: {e}")))?;
    }

    let results = top_n.into_sorted_vec();
    let mut ids = Vec::with_capacity(results.len());
    let mut scores = Vec::with_capacity(results.len());
    for item in results {
        ids.push(item.doc);
        scores.push(item.sort_key);
    }

    Ok((ids, scores))
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

/// Execute a tantivy query on a single segment and stream matching doc ids in
/// `batch_size` chunks through `on_chunk`.
///
/// Four execution paths:
/// - TopK + scoring: handled separately by [`collect_topk_docs`]
/// - Full scoring: `for_each` with alive_bitset filtering in callback
/// - No scoring: `for_each_no_score` with alive_bitset filtering in callback
/// - No query: iterate all alive docs in the segment
pub(crate) fn for_each_matching_doc_chunks<F>(
    segment_reader: &SegmentReader,
    searcher: &Searcher,
    query: Option<&Arc<dyn tantivy::query::Query>>,
    index_schema: &tantivy::schema::Schema,
    needs_score: bool,
    batch_size: usize,
    mut on_chunk: F,
) -> Result<()>
where
    F: FnMut(&[DocId], Option<&[Score]>) -> Result<bool>,
{
    match query {
        Some(query) => {
            if needs_score {
                let weight = query
                    .weight(EnableScoring::enabled_from_searcher(searcher))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
                let alive_bitset = segment_reader.alive_bitset();
                let mut ids = Vec::with_capacity(batch_size);
                let mut scores = Vec::with_capacity(batch_size);
                let mut callback_error: Option<DataFusionError> = None;
                let mut stopped = false;

                weight
                    .for_each(segment_reader, &mut |doc, score| {
                        if stopped {
                            return;
                        }
                        if alive_bitset.is_some_and(|alive| !alive.is_alive(doc)) {
                            return;
                        }

                        ids.push(doc);
                        scores.push(score);

                        if ids.len() == batch_size {
                            match flush_scored_buffer(&mut ids, &mut scores, &mut on_chunk) {
                                Ok(true) => {}
                                Ok(false) => stopped = true,
                                Err(err) => {
                                    callback_error = Some(err);
                                    stopped = true;
                                }
                            }
                        }
                    })
                    .map_err(|e| DataFusionError::Internal(format!("query execution: {e}")))?;

                if let Some(err) = callback_error {
                    return Err(err);
                }
                if stopped {
                    return Ok(());
                }

                flush_scored_buffer(&mut ids, &mut scores, &mut on_chunk)?;
                Ok(())
            } else {
                let weight = query
                    .weight(EnableScoring::disabled_from_schema(index_schema))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
                let alive_bitset = segment_reader.alive_bitset();
                let mut doc_buffer = Vec::with_capacity(batch_size);
                let mut callback_error: Option<DataFusionError> = None;
                let mut stopped = false;

                weight
                    .for_each_no_score(segment_reader, &mut |docs| {
                        if stopped {
                            return;
                        }

                        for &doc in docs {
                            if alive_bitset.is_some_and(|alive| !alive.is_alive(doc)) {
                                continue;
                            }
                            doc_buffer.push(doc);

                            if doc_buffer.len() == batch_size {
                                match flush_doc_buffer(&mut doc_buffer, &mut on_chunk) {
                                    Ok(true) => {}
                                    Ok(false) => {
                                        stopped = true;
                                        break;
                                    }
                                    Err(err) => {
                                        callback_error = Some(err);
                                        stopped = true;
                                        break;
                                    }
                                }
                            }
                        }
                    })
                    .map_err(|e| DataFusionError::Internal(format!("query execution: {e}")))?;

                if let Some(err) = callback_error {
                    return Err(err);
                }
                if stopped {
                    return Ok(());
                }

                flush_doc_buffer(&mut doc_buffer, &mut on_chunk)?;
                Ok(())
            }
        }
        None => {
            let max_doc = segment_reader.max_doc();
            let alive_bitset = segment_reader.alive_bitset();
            let mut doc_buffer = Vec::with_capacity(batch_size);

            for doc_id in 0..max_doc {
                if alive_bitset.is_none_or(|bitset| bitset.is_alive(doc_id)) {
                    doc_buffer.push(doc_id);
                }
                if doc_buffer.len() == batch_size
                    && !flush_doc_buffer(&mut doc_buffer, &mut on_chunk)?
                {
                    return Ok(());
                }
            }

            flush_doc_buffer(&mut doc_buffer, &mut on_chunk)?;
            Ok(())
        }
    }
}
