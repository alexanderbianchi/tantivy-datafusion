//! Shared utility functions used by both the unified (single-table) and
//! decomposed (three-table join) provider approaches.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_plan::Partitioning;
use tantivy::query::{BooleanQuery, QueryParser};
use tantivy::Index;

/// Declare hash partitioning on `(_doc_id, _segment_ord)` so DataFusion
/// recognises co-partitioned sources.
///
/// Falls back to `UnknownPartitioning` if the schema lacks the join-key columns.
pub(crate) fn segment_hash_partitioning(
    projected_schema: &SchemaRef,
    num_segments: usize,
) -> Partitioning {
    if let (Ok(doc_id_idx), Ok(seg_ord_idx)) = (
        projected_schema.index_of("_doc_id"),
        projected_schema.index_of("_segment_ord"),
    ) {
        Partitioning::Hash(
            vec![
                Arc::new(Column::new("_doc_id", doc_id_idx)),
                Arc::new(Column::new("_segment_ord", seg_ord_idx)),
            ],
            num_segments,
        )
    } else {
        Partitioning::UnknownPartitioning(num_segments)
    }
}

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
            DataFusionError::Plan(format!(
                "full_text: field '{field_name}' not found: {e}"
            ))
        })?;
        let parser = QueryParser::for_index(index, vec![field]);
        let parsed = parser.parse_query(query_string).map_err(|e| {
            DataFusionError::Plan(format!(
                "full_text: failed to parse '{query_string}': {e}"
            ))
        })?;
        queries.push(parsed);
    }

    match queries.len() {
        0 => Ok(None),
        1 => Ok(Some(Arc::from(queries.into_iter().next().unwrap()))),
        _ => Ok(Some(Arc::new(BooleanQuery::intersection(queries)))),
    }
}
