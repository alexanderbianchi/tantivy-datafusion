use std::any::Any;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::Index;
use tokio::sync::OnceCell;

use crate::schema_mapping::field_cardinality;
use crate::split_runtime::{PreparedSplit, SplitDescriptor};

/// Local adapter that can still derive schema metadata cheaply and prepare a
/// reusable split runtime when the executor needs to scan a split.
///
/// During planning, only [`schema`](IndexOpener::schema) and
/// [`multi_valued_fields`](IndexOpener::multi_valued_fields) are called. The
/// actual split preparation happens at stream poll time.
#[async_trait]
pub trait IndexOpener: Send + Sync + fmt::Debug {
    /// Prepare a reusable split runtime.
    async fn prepare(&self) -> Result<Arc<PreparedSplit>>;

    /// Open (or return cached) the tantivy `Index`.
    ///
    /// Kept as a convenience wrapper for older local code paths.
    async fn open(&self) -> Result<Index> {
        Ok(self.prepare().await?.index().clone())
    }

    /// Tantivy schema — available without opening the split at planning time.
    fn schema(&self) -> tantivy::schema::Schema;

    /// Whether this opener requires async warmup before synchronous reads.
    ///
    /// Returns `false` for local/mmap openers (data already accessible).
    /// Returns `true` for storage-backed openers (S3/GCS) that need
    /// file slices pre-loaded into cache before sync access.
    ///
    /// When `false`, warmup is skipped entirely — saving IndexReader
    /// construction and async I/O overhead per query.
    fn needs_warmup(&self) -> bool {
        true // conservative default — assume warmup needed
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

pub type OpenerMetadata = SplitDescriptor;

/// An [`IndexOpener`] that wraps an already-opened `Index`.
///
/// This is the default for local (non-distributed) usage. `open()` returns
/// the existing index immediately (just an `Arc` clone internally).
#[derive(Debug, Clone)]
pub struct DirectIndexOpener {
    index: Index,
    prepared: Arc<OnceCell<Arc<PreparedSplit>>>,
}

impl DirectIndexOpener {
    pub fn new(index: Index) -> Self {
        Self {
            index,
            prepared: Arc::new(OnceCell::new()),
        }
    }

    /// Returns a reference to the wrapped tantivy Index.
    ///
    /// This is cheap (no I/O) and is used at planning time to read
    /// per-segment statistics for partition pruning.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Return a long-lived reader for this index snapshot.
    pub fn reader(&self) -> Result<tantivy::IndexReader> {
        self.index
            .reader()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))
    }
}

#[async_trait]
impl IndexOpener for DirectIndexOpener {
    async fn prepare(&self) -> Result<Arc<PreparedSplit>> {
        self.prepared
            .get_or_try_init(|| async {
                let prepared = PreparedSplit::new(self.index.clone(), Arc::new(()))?;
                Ok::<Arc<PreparedSplit>, DataFusionError>(Arc::new(prepared))
            })
            .await
            .map(Arc::clone)
    }

    fn schema(&self) -> tantivy::schema::Schema {
        self.index.schema()
    }

    fn multi_valued_fields(&self) -> Vec<String> {
        use tantivy::columnar::Cardinality;
        let schema = self.index.schema();
        let reader = match self.reader() {
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

    fn needs_warmup(&self) -> bool {
        false // mmap — data already accessible, no async pre-loading needed
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
