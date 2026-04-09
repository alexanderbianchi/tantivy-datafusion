//! Pre-fetches tantivy index data for storage-backed directories.
//!
//! Tantivy's query execution does synchronous I/O. When the index
//! lives on object storage (S3/GCS), sync reads aren't supported.
//! This module pre-loads the needed data into the directory's cache
//! so that tantivy's sync reads hit memory instead of storage.

use std::collections::HashSet;

use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::schema::{Field, FieldType};
use tantivy::{Index, ReloadPolicy};

/// Warm up the inverted index data needed for full-text queries.
///
/// Pre-loads term dictionaries and posting lists for the given fields.
pub async fn warmup_inverted_index(index: &Index, query_fields: &[Field]) -> Result<()> {
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("open reader for warmup: {e}")))?;
    let searcher = reader.searcher();

    let fields: HashSet<Field> = query_fields.iter().copied().collect();

    for segment_reader in searcher.segment_readers() {
        for &field in &fields {
            let inv_index = segment_reader.inverted_index(field).map_err(|e| {
                DataFusionError::Internal(format!("get inverted index for warmup: {e}"))
            })?;

            inv_index.terms().warm_up_dictionary().await.map_err(|e| {
                DataFusionError::Internal(format!("warm term dict: {e}"))
            })?;

            (*inv_index).warm_postings_full(false).await.map_err(|e| {
                DataFusionError::Internal(format!("warm postings: {e}"))
            })?;
        }
    }

    Ok(())
}

/// Warm up fast fields for specific field names only.
///
/// Only pre-loads the fields that will actually be read.
pub async fn warmup_fast_fields_by_name(index: &Index, field_names: &[&str]) -> Result<()> {
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("open reader for warmup: {e}")))?;
    let searcher = reader.searcher();

    for segment_reader in searcher.segment_readers() {
        let ff_reader = segment_reader.fast_fields();
        for &name in field_names {
            let handles = match ff_reader.list_dynamic_column_handles(name).await {
                Ok(h) => h,
                Err(_) => continue, // field not present or not fast
            };
            for handle in handles {
                if let Err(e) = handle.file_slice().read_bytes_async().await {
                    eprintln!("warmup: failed to pre-load fast field '{name}': {e}");
                }
            }
        }
    }

    Ok(())
}

/// Warm up fast fields (columnar data) for all fields in the schema.
///
/// Pre-loads the fast field file slices so tantivy can read them
/// synchronously.
pub async fn warmup_fast_fields(index: &Index) -> Result<()> {
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("open reader for warmup: {e}")))?;
    let searcher = reader.searcher();
    let schema = index.schema();

    for segment_reader in searcher.segment_readers() {
        let ff_reader = segment_reader.fast_fields();
        for (_field, entry) in schema.fields() {
            // Only warm fields that are configured as fast fields.
            if !entry.is_fast() {
                continue;
            }
            // Warm by listing column handles and reading file slices.
            let handles = match ff_reader
                .list_dynamic_column_handles(entry.name())
                .await
            {
                Ok(h) => h,
                Err(_) => continue, // field not present in this segment
            };
            for handle in handles {
                if let Err(e) = handle.file_slice().read_bytes_async().await {
                    eprintln!(
                        "warmup: failed to pre-load fast field '{}': {e}",
                        entry.name()
                    );
                }
            }
        }
    }

    Ok(())
}

/// Warm up the document store for all segments.
///
/// Pre-loads the store file slices so tantivy's `StoreReader` can read
/// documents synchronously inside `spawn_blocking`.
///
/// Tantivy's store uses a block-based LRU cache and does not expose async
/// file-slice warmup like fast fields do. The pragmatic approach here is to
/// open each segment's `StoreReader` (which reads the footer/index into cache)
/// and read a small number of documents to warm the first blocks. Subsequent
/// reads happen inside `spawn_blocking` and will hit the cache for these
/// initial blocks.
pub async fn warmup_document_store(index: &Index) -> Result<()> {
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("open reader for doc warmup: {e}")))?;
    let searcher = reader.searcher();

    for segment_reader in searcher.segment_readers() {
        // Opening the store reader reads the footer/index into cache.
        let store_reader = segment_reader
            .get_store_reader(100)
            .map_err(|e| {
                DataFusionError::Internal(format!("open store reader for warmup: {e}"))
            })?;
        // Reading doc 0 triggers loading the first block, which warms the
        // file handle. We cannot easily enumerate all blocks, but this
        // ensures the store is accessible.
        if segment_reader.max_doc() > 0 {
            let _ = store_reader.get::<tantivy::TantivyDocument>(0);
        }
    }

    Ok(())
}

/// Warm up everything needed for a typical query: fast fields +
/// all text field inverted indexes.
///
/// Call this after opening an index on a storage-backed directory.
pub async fn warmup_all(index: &Index) -> Result<()> {
    let ff_future = warmup_fast_fields(index);
    let inv_future = warmup_all_text_fields(index);
    let (ff_result, inv_result) = tokio::join!(ff_future, inv_future);
    ff_result?;
    inv_result?;
    Ok(())
}

/// Warm up all indexed text fields in the schema.
pub async fn warmup_all_text_fields(index: &Index) -> Result<()> {
    let schema = index.schema();
    let text_fields: Vec<Field> = schema
        .fields()
        .filter_map(|(field, entry)| {
            if entry.is_indexed() {
                if let FieldType::Str(_) = entry.field_type() {
                    return Some(field);
                }
            }
            None
        })
        .collect();

    if text_fields.is_empty() {
        return Ok(());
    }

    warmup_inverted_index(index, &text_fields).await
}
