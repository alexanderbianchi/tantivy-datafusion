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
        let field = tantivy_schema.get_field(field_name).map_err(|e| {
            DataFusionError::Plan(format!("full_text: field '{field_name}' not found: {e}"))
        })?;
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

/// Execute a tantivy query on a single segment and collect matching doc_ids
/// with optional BM25 scores.
///
/// Four execution paths:
/// - TopK + scoring: Block-WAND via `for_each_pruning` (bounded by K)
/// - Full scoring: `for_each` with alive_bitset filtering in callback
/// - No scoring: `for_each_no_score` with alive_bitset filtering in callback
/// - No query: iterate all alive docs in the segment
pub(crate) fn collect_matching_docs(
    segment_reader: &SegmentReader,
    searcher: &Searcher,
    query: Option<&Arc<dyn tantivy::query::Query>>,
    index_schema: &tantivy::schema::Schema,
    needs_score: bool,
    topk: Option<usize>,
) -> Result<(Vec<u32>, Option<Vec<f32>>)> {
    match query {
        Some(query) => {
            if needs_score {
                // BM25 scoring enabled.
                let weight = query
                    .weight(EnableScoring::enabled_from_searcher(searcher))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;

                if let Some(k) = topk {
                    // TopK with Block-WAND pruning.
                    let mut top_n: TopNComputer<Score, DocId, _> = TopNComputer::new(k);

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
                            .map_err(|e| {
                                DataFusionError::Internal(format!("topk query execution: {e}"))
                            })?;
                    } else {
                        weight
                            .for_each_pruning(Score::MIN, segment_reader, &mut |doc, score| {
                                top_n.push(score, doc);
                                top_n.threshold.unwrap_or(Score::MIN)
                            })
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
                    Ok((ids, Some(sc)))
                } else {
                    // Full scoring without topK.
                    // Filter deleted docs inside the callback to avoid
                    // collecting and then discarding them in a second pass.
                    let alive_bitset = segment_reader.alive_bitset();
                    let mut ids = Vec::new();
                    let mut sc = Vec::new();
                    if let Some(alive) = alive_bitset {
                        weight
                            .for_each(segment_reader, &mut |doc, score| {
                                if alive.is_alive(doc) {
                                    ids.push(doc);
                                    sc.push(score);
                                }
                            })
                            .map_err(|e| {
                                DataFusionError::Internal(format!("query execution: {e}"))
                            })?;
                    } else {
                        weight
                            .for_each(segment_reader, &mut |doc, score| {
                                ids.push(doc);
                                sc.push(score);
                            })
                            .map_err(|e| {
                                DataFusionError::Internal(format!("query execution: {e}"))
                            })?;
                    }
                    Ok((ids, Some(sc)))
                }
            } else {
                // No scoring needed -- boolean filter only.
                let weight = query
                    .weight(EnableScoring::disabled_from_schema(index_schema))
                    .map_err(|e| DataFusionError::Internal(format!("create weight: {e}")))?;
                // Filter deleted docs inside the callback to avoid a
                // separate retain pass over the collected doc ids.
                let alive_bitset = segment_reader.alive_bitset();
                let mut matching_docs = Vec::new();
                if let Some(alive) = alive_bitset {
                    weight
                        .for_each_no_score(segment_reader, &mut |docs| {
                            matching_docs.extend(docs.iter().filter(|&&d| alive.is_alive(d)));
                        })
                        .map_err(|e| DataFusionError::Internal(format!("query execution: {e}")))?;
                } else {
                    weight
                        .for_each_no_score(segment_reader, &mut |docs| {
                            matching_docs.extend_from_slice(docs);
                        })
                        .map_err(|e| DataFusionError::Internal(format!("query execution: {e}")))?;
                }
                Ok((matching_docs, None))
            }
        }
        None => {
            // No query -- iterate all alive docs.
            let max_doc = segment_reader.max_doc();
            let alive_bitset = segment_reader.alive_bitset();
            let ids: Vec<u32> = (0..max_doc)
                .filter(|&doc_id| alive_bitset.is_none_or(|bitset| bitset.is_alive(doc_id)))
                .collect();
            Ok((ids, None))
        }
    }
}
