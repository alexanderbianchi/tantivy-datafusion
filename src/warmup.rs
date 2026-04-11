//! Pre-fetches tantivy index data for storage-backed directories.
//!
//! Tantivy's query execution does synchronous I/O. When the index
//! lives on object storage (S3/GCS), sync reads aren't supported.
//! This module pre-loads the needed data into the directory's cache
//! so that tantivy's sync reads hit memory instead of storage.

use std::collections::{BTreeSet, HashSet};

use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use datafusion::logical_expr::Expr;
use tantivy::schema::{Field, FieldType};
use tantivy::directory::Directory;
use tantivy::{Index, IndexReader, ReloadPolicy};

async fn open_reader_for_warmup(index: &Index) -> Result<IndexReader> {
    let index = index.clone();
    tokio::task::spawn_blocking(move || {
        index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| DataFusionError::Internal(format!("open reader for warmup: {e}")))
    })
    .await
    .map_err(|e| DataFusionError::Internal(format!("warmup reader spawn: {e}")))?
}

/// Collect fast-field names referenced by pushed-down filter expressions.
///
/// Only columns that exist on the given tantivy schema are returned.
pub(crate) fn fast_field_filter_field_names(
    tantivy_schema: &tantivy::schema::Schema,
    filter_exprs: &[Expr],
) -> Result<Vec<String>> {
    let mut field_names = BTreeSet::new();

    for expr in filter_exprs {
        expr.apply(|node| {
            if let Expr::Column(column) = node {
                if tantivy_schema.get_field(&column.name).is_ok() {
                    field_names.insert(column.name.clone());
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }

    Ok(field_names.into_iter().collect())
}

/// Warm up the inverted index data needed for full-text queries.
///
/// Pre-loads term dictionaries and posting lists for the given fields.
pub async fn warmup_inverted_index(index: &Index, query_fields: &[Field]) -> Result<()> {
    let reader = open_reader_for_warmup(index).await?;
    let searcher = reader.searcher();

    let fields: HashSet<Field> = query_fields.iter().copied().collect();

    for segment_reader in searcher.segment_readers() {
        for &field in &fields {
            let inv_index = segment_reader.inverted_index(field).map_err(|e| {
                DataFusionError::Internal(format!("get inverted index for warmup: {e}"))
            })?;

            inv_index
                .terms()
                .warm_up_dictionary()
                .await
                .map_err(|e| DataFusionError::Internal(format!("warm term dict: {e}")))?;

            (*inv_index)
                .warm_postings_full(false)
                .await
                .map_err(|e| DataFusionError::Internal(format!("warm postings: {e}")))?;
        }
    }

    Ok(())
}

/// Warm up fast fields for specific field names only.
///
/// Only pre-loads the fields that will actually be read.
pub async fn warmup_fast_fields_by_name(index: &Index, field_names: &[&str]) -> Result<()> {
    let reader = open_reader_for_warmup(index).await?;
    let searcher = reader.searcher();

    for segment_reader in searcher.segment_readers() {
        let ff_reader = segment_reader.fast_fields();
        for &name in field_names {
            let handles = match ff_reader.list_dynamic_column_handles(name).await {
                Ok(h) => h,
                Err(_) => continue, // field not present or not fast
            };
            for handle in handles {
                handle.file_slice().read_bytes_async().await.map_err(|e| {
                    DataFusionError::Internal(format!(
                        "warmup: failed to pre-load fast field '{name}': {e}"
                    ))
                })?;
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
    let reader = open_reader_for_warmup(index).await?;
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
            let handles = match ff_reader.list_dynamic_column_handles(entry.name()).await {
                Ok(h) => h,
                Err(_) => continue, // field not present in this segment
            };
            for handle in handles {
                handle.file_slice().read_bytes_async().await.map_err(|e| {
                    DataFusionError::Internal(format!(
                        "warmup: failed to pre-load fast field '{}': {e}",
                        entry.name()
                    ))
                })?;
            }
        }
    }

    Ok(())
}

/// Warm up the document store for all segments.
///
/// Pre-loads the `.store` file data into the directory cache via async reads
/// so that tantivy's synchronous `StoreReader` hits cache instead of the
/// underlying storage.
pub async fn warmup_document_store(index: &Index) -> Result<()> {
    let reader = open_reader_for_warmup(index).await?;
    let searcher = reader.searcher();

    for segment_reader in searcher.segment_readers() {
        // Build the .store file path: "{segment_uuid}.store"
        let uuid = segment_reader.segment_id().uuid_string();
        let store_path = std::path::PathBuf::from(format!("{uuid}.store"));

        // Open the file slice and pre-load via async read into the cache.
        let file_slice = index
            .directory()
            .open_read(&store_path)
            .map_err(|err| {
                DataFusionError::Internal(format!(
                    "warmup doc store open {}: {err}",
                    store_path.display()
                ))
            })?;

        file_slice
            .read_bytes_async()
            .await
            .map_err(|err| {
                DataFusionError::Internal(format!(
                    "warmup doc store read {}: {err}",
                    store_path.display()
                ))
            })?;
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

#[cfg(test)]
mod tests {
    use datafusion::prelude::{col, lit};
    use tantivy::schema::{SchemaBuilder, FAST};

    use super::fast_field_filter_field_names;

    #[test]
    fn test_fast_field_filter_field_names_extracts_existing_columns() {
        let mut builder = SchemaBuilder::new();
        builder.add_u64_field("price", FAST);
        builder.add_bool_field("active", FAST);
        let schema = builder.build();

        let filters = vec![
            col("price").gt(lit(3u64)).and(col("active").eq(lit(true))),
            col("missing").eq(lit(1u64)),
        ];

        let names = fast_field_filter_field_names(&schema, &filters).unwrap();
        assert_eq!(names, vec!["active".to_string(), "price".to_string()]);
    }
}
