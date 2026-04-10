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
// Unified single-table approach
// ---------------------------------------------------------------------------
pub mod unified;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------
pub use codec::{OpenerFactory, OpenerFactoryExt, TantivyCodec};
pub use full_text_udf::{extract_full_text_call, full_text_udf};
pub use index_opener::{DirectIndexOpener, IndexOpener, OpenerMetadata};
pub use schema_mapping::{
    tantivy_schema_to_arrow, tantivy_schema_to_arrow_from_index,
    tantivy_schema_to_arrow_with_multi_valued,
};
pub use unified::agg_pushdown::AggPushdown;
pub use unified::single_table_provider::SingleTableProvider;
