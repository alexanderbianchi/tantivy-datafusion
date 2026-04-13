use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::prelude::SessionConfig;
use tantivy::{Index, IndexReader, ReloadPolicy, Searcher};

/// Serializable split metadata that can cross the planner/worker boundary.
///
/// `payload` is intentionally opaque to `tantivy-datafusion`. Integrations such
/// as Quickwit can encode whatever worker-local reconstruction data they need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitDescriptor {
    pub split_id: String,
    pub payload: Vec<u8>,
    pub tantivy_schema: tantivy::schema::Schema,
    pub multi_valued_fields: Vec<String>,
}

impl SplitDescriptor {
    pub fn new(
        split_id: impl Into<String>,
        payload: Vec<u8>,
        tantivy_schema: tantivy::schema::Schema,
        multi_valued_fields: Vec<String>,
    ) -> Self {
        Self {
            split_id: split_id.into(),
            payload,
            tantivy_schema,
            multi_valued_fields,
        }
    }
}

/// Worker-local prepared split runtime.
///
/// This is the runtime object the executor should actually reuse:
/// one opened index, one reader, one searcher, plus a keepalive handle for
/// any backing cache/directory state required by the integration.
pub struct PreparedSplit {
    index: Index,
    reader: IndexReader,
    searcher: Searcher,
    keepalive: Arc<dyn Any + Send + Sync>,
}

impl std::fmt::Debug for PreparedSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedSplit").finish_non_exhaustive()
    }
}

impl PreparedSplit {
    pub fn new(index: Index, keepalive: Arc<dyn Any + Send + Sync>) -> Result<Self> {
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| DataFusionError::Internal(format!("open reader: {e}")))?;
        Ok(Self::new_with_reader(index, reader, keepalive))
    }

    pub fn new_with_reader(
        index: Index,
        reader: IndexReader,
        keepalive: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        let searcher = reader.searcher();
        Self {
            index,
            reader,
            searcher,
            keepalive,
        }
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    pub fn reader(&self) -> &IndexReader {
        &self.reader
    }

    pub fn searcher(&self) -> &Searcher {
        &self.searcher
    }

    pub fn keepalive(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.keepalive
    }
}

#[async_trait]
pub trait SplitRuntimeFactory: Send + Sync {
    async fn prepare_split(&self, descriptor: &SplitDescriptor) -> Result<Arc<PreparedSplit>>;
}

pub type SplitRuntimeFactoryRef = Arc<dyn SplitRuntimeFactory>;

struct SplitRuntimeFactoryExtension(SplitRuntimeFactoryRef);

pub trait SplitRuntimeFactoryExt {
    fn set_split_runtime_factory(&mut self, factory: SplitRuntimeFactoryRef);
    fn get_split_runtime_factory(&self) -> Option<SplitRuntimeFactoryRef>;
}

impl SplitRuntimeFactoryExt for SessionConfig {
    fn set_split_runtime_factory(&mut self, factory: SplitRuntimeFactoryRef) {
        self.set_extension(Arc::new(SplitRuntimeFactoryExtension(factory)));
    }

    fn get_split_runtime_factory(&self) -> Option<SplitRuntimeFactoryRef> {
        self.get_extension::<SplitRuntimeFactoryExtension>()
            .map(|ext| Arc::clone(&ext.0))
    }
}
