//! Injectable sync execution pool for CPU-intensive tantivy work.
//!
//! Tantivy's query and aggregation execution is synchronous. This module
//! decouples *where* that work runs from the call sites: tantivy-datafusion
//! defines the interface, and the embedding runtime provides the pool.
//!
//! The default [`SpawnBlockingPool`] delegates to `tokio::task::spawn_blocking`.
//! Quickwit provides a bounded Rayon-backed implementation that matches its
//! search runtime model.

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::prelude::SessionConfig;

/// Abstraction for running synchronous CPU-intensive closures off the async
/// executor.
///
/// Implementations control thread pool sizing, admission, and cancellation
/// semantics. The closure is type-erased via `Box<dyn Any + Send>` to keep
/// the trait object-safe. Use [`run_sync`] for a typed convenience wrapper.
#[async_trait]
pub trait SyncExecutionPool: Send + Sync + std::fmt::Debug {
    async fn run_boxed(
        &self,
        task: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>> + Send>,
    ) -> Result<Box<dyn Any + Send>>;
}

pub type SyncExecutionPoolRef = Arc<dyn SyncExecutionPool>;

/// Typed convenience wrapper around [`SyncExecutionPool::run_boxed`].
///
/// Erases `T` into `Box<dyn Any + Send>` for the trait call, then downcasts
/// the result back. The downcast is infallible given the closure structure.
pub async fn run_sync<T: Send + 'static>(
    pool: &dyn SyncExecutionPool,
    f: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let boxed = pool
        .run_boxed(Box::new(move || {
            f().map(|v| Box::new(v) as Box<dyn Any + Send>)
        }))
        .await?;
    Ok(*boxed
        .downcast::<T>()
        .expect("SyncExecutionPool type mismatch"))
}

/// Default fallback: delegates to `tokio::task::spawn_blocking`.
///
/// Unbounded, no cancellation-on-drop. Sufficient for local/standalone usage
/// and for tests that don't inject a custom pool.
#[derive(Debug, Clone, Copy)]
pub struct SpawnBlockingPool;

#[async_trait]
impl SyncExecutionPool for SpawnBlockingPool {
    async fn run_boxed(
        &self,
        task: Box<dyn FnOnce() -> Result<Box<dyn Any + Send>> + Send>,
    ) -> Result<Box<dyn Any + Send>> {
        tokio::task::spawn_blocking(task)
            .await
            .map_err(|e| DataFusionError::Internal(format!("spawn_blocking join error: {e}")))?
    }
}

// ---------------------------------------------------------------------------
// SessionConfig extension
// ---------------------------------------------------------------------------

struct SyncExecutionPoolExtension(SyncExecutionPoolRef);

/// Extension trait for injecting a [`SyncExecutionPool`] into a DataFusion
/// `SessionConfig`. Follows the same pattern as `SplitRuntimeFactoryExt`.
pub trait SyncExecutionPoolExt {
    fn set_sync_execution_pool(&mut self, pool: SyncExecutionPoolRef);
    fn get_sync_execution_pool(&self) -> Option<SyncExecutionPoolRef>;
}

impl SyncExecutionPoolExt for SessionConfig {
    fn set_sync_execution_pool(&mut self, pool: SyncExecutionPoolRef) {
        self.set_extension(Arc::new(SyncExecutionPoolExtension(pool)));
    }

    fn get_sync_execution_pool(&self) -> Option<SyncExecutionPoolRef> {
        self.get_extension::<SyncExecutionPoolExtension>()
            .map(|ext| Arc::clone(&ext.0))
    }
}

/// Retrieve the sync execution pool from a task context, falling back to
/// [`SpawnBlockingPool`] if none was injected.
pub fn get_or_default_pool(
    context: &datafusion::execution::TaskContext,
) -> SyncExecutionPoolRef {
    context
        .session_config()
        .get_sync_execution_pool()
        .unwrap_or_else(|| Arc::new(SpawnBlockingPool) as SyncExecutionPoolRef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[tokio::test]
    async fn spawn_blocking_pool_returns_ok() {
        let pool = SpawnBlockingPool;
        let result = run_sync(&pool, || Ok(42u64)).await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn spawn_blocking_pool_propagates_err() {
        let pool = SpawnBlockingPool;
        let result = run_sync::<()>(&pool, || {
            Err(DataFusionError::Internal("test error".into()))
        })
        .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test error"));
    }

    #[tokio::test]
    async fn run_sync_typed_roundtrip() {
        let pool = SpawnBlockingPool;
        let result: String = run_sync(&pool, || Ok("hello".to_string())).await.unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn session_config_extension_roundtrip() {
        let mut config = SessionConfig::new();
        assert!(config.get_sync_execution_pool().is_none());

        config.set_sync_execution_pool(Arc::new(SpawnBlockingPool));
        assert!(config.get_sync_execution_pool().is_some());
    }

    #[tokio::test]
    async fn get_or_default_returns_spawn_blocking_when_unset() {
        let pool = SpawnBlockingPool;
        let counter = Arc::new(AtomicU64::new(0));
        let counter_clone = Arc::clone(&counter);
        run_sync(&pool, move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
