//! **Decomposed three-table approach** (original design).
//!
//! Three separate `DataSource`s (fast fields, inverted index, stored documents)
//! joined via `HashJoinExec` on `(_doc_id, _segment_ord)`. Requires four custom
//! optimizer rules to push operations across join boundaries.

pub mod agg_translator;
pub mod catalog;
pub mod document_provider;
pub mod filter_pushdown;
pub mod inverted_index_provider;
pub mod table_provider;
pub mod topk_pushdown;
pub mod unified_provider;
