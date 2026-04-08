use std::any::Any;
use std::fmt;

use async_trait::async_trait;
use datafusion::common::Result;
use tantivy::Index;

use crate::schema_mapping::field_cardinality;

/// Defers tantivy `Index` opening from planning time to execution time.
///
/// During planning, only [`schema`](IndexOpener::schema) and
/// [`segment_sizes`](IndexOpener::segment_sizes) are called — these must
/// be cheap and synchronous. The actual [`open`](IndexOpener::open) is
/// called at stream poll time (inside `DataSource::open`), allowing
/// distributed executors to open splits from local storage/cache.
#[async_trait]
pub trait IndexOpener: Send + Sync + fmt::Debug {
    /// Open (or return cached) the tantivy Index.
    /// Called at execution time (stream poll), not planning time.
    async fn open(&self) -> Result<Index>;

    /// Tantivy schema — available without opening the index.
    /// Used during planning for Arrow schema derivation and query parsing.
    fn schema(&self) -> tantivy::schema::Schema;

    /// Number of segments and max_doc per segment.
    /// Used during planning to determine partition count and chunking.
    /// Returns empty vec if unknown (single-partition fallback).
    fn segment_sizes(&self) -> Vec<u32>;

    /// A serializable identifier for this opener.
    ///
    /// Used by [`TantivyCodec`](crate::codec::TantivyCodec) to reconstruct
    /// the opener on remote workers. The caller's opener factory receives
    /// this identifier along with the schema and segment sizes.
    ///
    /// For local (non-distributed) usage this can return an empty string.
    fn identifier(&self) -> &str;

    /// Footer byte range in the split bundle.
    /// Used by storage-backed openers. Returns (0, 0) if not applicable.
    fn footer_range(&self) -> (u64, u64) {
        (0, 0)
    }

    /// Names of fields with `Cardinality::Multivalued` in any segment.
    /// Used for correct Arrow schema construction on remote workers.
    /// Returns empty vec if not known (falls back to scalar types).
    fn multi_valued_fields(&self) -> Vec<String> {
        vec![]
    }

    /// Downcast to a concrete type.
    fn as_any(&self) -> &dyn Any;
}

/// Metadata extracted from an [`IndexOpener`] for codec serialization.
///
/// Passed to the opener factory on the remote worker to reconstruct
/// an [`IndexOpener`] that can open the index from local storage/cache.
#[derive(Debug, Clone)]
pub struct OpenerMetadata {
    pub identifier: String,
    pub tantivy_schema: tantivy::schema::Schema,
    pub segment_sizes: Vec<u32>,
    /// Start offset of the split footer in the bundle file.
    /// Used by storage-backed openers to open the split.
    pub footer_start: u64,
    /// End offset of the split footer.
    pub footer_end: u64,
    /// Names of fields that are multi-valued in at least one segment.
    /// Used to build correct `List<T>` Arrow schemas on remote workers.
    pub multi_valued_fields: Vec<String>,
}

/// An [`IndexOpener`] that wraps an already-opened `Index`.
///
/// This is the default for local (non-distributed) usage. `open()` returns
/// the existing index immediately (just an `Arc` clone internally).
#[derive(Debug, Clone)]
pub struct DirectIndexOpener {
    index: Index,
}

impl DirectIndexOpener {
    pub fn new(index: Index) -> Self {
        Self { index }
    }

    /// Returns a reference to the wrapped tantivy Index.
    ///
    /// This is cheap (no I/O) and is used at planning time to read
    /// per-segment statistics for partition pruning.
    pub fn index(&self) -> &Index {
        &self.index
    }
}

#[async_trait]
impl IndexOpener for DirectIndexOpener {
    async fn open(&self) -> Result<Index> {
        Ok(self.index.clone())
    }

    fn schema(&self) -> tantivy::schema::Schema {
        self.index.schema()
    }

    fn segment_sizes(&self) -> Vec<u32> {
        match self.index.reader() {
            Ok(reader) => {
                let searcher = reader.searcher();
                searcher
                    .segment_readers()
                    .iter()
                    .map(|r| r.max_doc())
                    .collect()
            }
            Err(_) => vec![],
        }
    }

    fn identifier(&self) -> &str {
        ""
    }

    fn multi_valued_fields(&self) -> Vec<String> {
        use tantivy::columnar::Cardinality;
        let schema = self.index.schema();
        let reader = match self.index.reader() {
            Ok(r) => r,
            Err(_) => return vec![],
        };
        let searcher = reader.searcher();
        let segment_readers = searcher.segment_readers();
        if segment_readers.is_empty() {
            return vec![];
        }

        schema
            .fields()
            .filter_map(|(_field, field_entry)| {
                if !field_entry.is_fast() {
                    return None;
                }
                let name = field_entry.name();
                let is_multi = segment_readers.iter().any(|seg| {
                    field_cardinality(seg, name, field_entry.field_type())
                        == Some(Cardinality::Multivalued)
                });
                if is_multi {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
