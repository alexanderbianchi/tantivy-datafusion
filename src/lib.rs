// ---------------------------------------------------------------------------
// Shared infrastructure
// ---------------------------------------------------------------------------
pub mod codec;
pub mod fast_field_reader;
pub mod full_text_udf;
pub mod index_opener;
pub mod schema_mapping;
pub(crate) mod util;
pub mod warmup;

// ---------------------------------------------------------------------------
// Unified single-table approach (start reviewing here)
// ---------------------------------------------------------------------------
pub mod unified;

// ---------------------------------------------------------------------------
// Decomposed three-table approach (original design)
// ---------------------------------------------------------------------------
pub mod decomposed;

// ---------------------------------------------------------------------------
// Re-exports: shared
// ---------------------------------------------------------------------------
pub use codec::{OpenerFactory, OpenerFactoryExt, TantivyCodec};
pub use full_text_udf::{extract_full_text_call, full_text_udf};
pub use index_opener::{DirectIndexOpener, IndexOpener, OpenerMetadata};
pub use schema_mapping::{tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index, tantivy_schema_to_arrow_with_multi_valued};

// Re-exports: unified
pub use unified::agg_pushdown::AggPushdown;
pub use unified::ordinal_group_by::OrdinalGroupByOptimization;
pub use unified::single_table_provider::SingleTableProvider;
pub use unified::task_estimator::estimate_task_count;

// Re-exports: decomposed
pub use decomposed::agg_translator::{create_session_with_pushdown, execute_aggregations, translate_aggregations};
pub use decomposed::catalog::{TantivyCatalog, TantivySchema};
pub use decomposed::document_provider::{DocumentDataSource, TantivyDocumentProvider};
pub use decomposed::filter_pushdown::FastFieldFilterPushdown;
pub use decomposed::inverted_index_provider::{InvertedIndexDataSource, TantivyInvertedIndexProvider};
pub use decomposed::table_provider::{FastFieldDataSource, TantivyTableProvider};
pub use decomposed::topk_pushdown::TopKPushdown;
pub use decomposed::unified_provider::UnifiedTantivyTableProvider;
