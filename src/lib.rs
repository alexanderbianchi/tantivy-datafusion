// ---------------------------------------------------------------------------
// Shared infrastructure (used by both approaches)
// ---------------------------------------------------------------------------
pub(crate) mod agg_exec;
pub mod codec;
pub mod fast_field_reader;
pub mod full_text_udf;
pub mod index_opener;
pub(crate) mod plan_traversal;
pub mod schema_mapping;
pub mod task_estimator;
pub(crate) mod util;
pub mod warmup;

// ---------------------------------------------------------------------------
// Unified single-table approach (recommended for BYOC)
//
// A single DataSource that handles FTS queries, fast field reading, scoring,
// document retrieval, and aggregations internally — no joins needed.
// ---------------------------------------------------------------------------
pub mod single_table_provider;
pub mod agg_pushdown;
pub mod ordinal_group_by;

// ---------------------------------------------------------------------------
// Decomposed three-table approach (original design)
//
// Three separate DataSources (fast fields, inverted index, stored documents)
// joined via HashJoinExec on (_doc_id, _segment_ord). Requires four custom
// optimizer rules to push operations across join boundaries.
// ---------------------------------------------------------------------------
pub mod catalog;
pub mod document_provider;
pub mod filter_pushdown;
pub mod inverted_index_provider;
pub mod table_provider;
pub mod topk_pushdown;
pub mod unified_provider;
pub mod agg_translator;

// ---------------------------------------------------------------------------
// Public re-exports
// ---------------------------------------------------------------------------

// Shared
pub use codec::{OpenerFactory, OpenerFactoryExt, TantivyCodec};
pub use full_text_udf::{extract_full_text_call, full_text_udf};
pub use index_opener::{DirectIndexOpener, IndexOpener, OpenerMetadata};
pub use schema_mapping::{tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index, tantivy_schema_to_arrow_with_multi_valued};
pub use task_estimator::estimate_task_count;

// Unified (single-table)
pub use agg_pushdown::AggPushdown;
pub use ordinal_group_by::OrdinalGroupByOptimization;
pub use single_table_provider::SingleTableProvider;

// Decomposed (three-table)
pub use catalog::{TantivyCatalog, TantivySchema};
pub use document_provider::{DocumentDataSource, TantivyDocumentProvider};
pub use filter_pushdown::FastFieldFilterPushdown;
pub use inverted_index_provider::{InvertedIndexDataSource, TantivyInvertedIndexProvider};
pub use table_provider::{FastFieldDataSource, TantivyTableProvider};
pub use topk_pushdown::TopKPushdown;
pub use unified_provider::UnifiedTantivyTableProvider;
pub use agg_translator::{create_session_with_pushdown, execute_aggregations, translate_aggregations};
