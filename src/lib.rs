// ---------------------------------------------------------------------------
// Shared infrastructure
// ---------------------------------------------------------------------------
pub mod codec;
pub mod fast_field_reader;
pub mod full_text_udf;
pub(crate) mod index_opener;
pub mod schema_mapping;
pub mod split_runtime;
pub mod sync_exec;
pub(crate) mod type_coercion;
pub(crate) mod util;
pub mod warmup;

// ---------------------------------------------------------------------------
// Unified single-table approach
// ---------------------------------------------------------------------------
pub mod unified;

// ---------------------------------------------------------------------------
// Nested approximate aggregations (ES-compatible)
// ---------------------------------------------------------------------------
pub mod nested_agg;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------
pub use codec::TantivyCodec;
pub use full_text_udf::{extract_full_text_call, full_text_udf};
pub use schema_mapping::{
    tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index,
    tantivy_schema_to_arrow_with_multi_valued,
};
pub use split_runtime::{
    PreparedSplit, SplitDescriptor, SplitRuntimeFactory, SplitRuntimeFactoryExt,
};
pub use sync_exec::{SyncExecutionPool, SyncExecutionPoolExt, SyncExecutionPoolRef};
pub use unified::agg_pushdown::AggPushdown;
pub use unified::single_table_provider::SingleTableProvider;
