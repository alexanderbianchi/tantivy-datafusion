use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, DictionaryArray, Float64Builder, Int32Builder,
    Int64Builder, ListBuilder, StringArray, StringBuilder, TimestampMicrosecondBuilder,
    UInt32Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Int32Type, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::error::DataFusionError;
use tantivy::index::SegmentReader;

/// Pre-built Arrow dictionary values for string fast fields.
///
/// Built once per segment, shared (via `Arc`) across all chunks.  The
/// dictionary is streamed in ordinal order so tantivy ordinals map directly
/// to Arrow dictionary indices.
pub struct DictCache {
    entries: HashMap<String, Arc<StringArray>>,
}

impl DictCache {
    /// Build the cache for every `Dictionary`-typed field in `projected_schema`.
    pub fn build(segment_reader: &SegmentReader, projected_schema: &SchemaRef) -> Result<Self> {
        let fast_fields = segment_reader.fast_fields();
        let mut entries = HashMap::new();

        for field in projected_schema.fields() {
            if !matches!(field.data_type(), DataType::Dictionary(_, _)) {
                continue;
            }
            let name = field.name();
            let str_col = match fast_fields.str(name) {
                Ok(Some(col)) => col,
                _ => continue, // missing field — null padding handles it
            };

            let num_terms = str_col.num_terms();
            let mut builder = StringBuilder::with_capacity(num_terms, num_terms * 16);
            let mut streamer = str_col
                .dictionary()
                .stream()
                .map_err(|e| DataFusionError::Internal(format!("stream dict '{name}': {e}")))?;
            while streamer.advance() {
                let s = std::str::from_utf8(streamer.key())
                    .map_err(|e| DataFusionError::Internal(format!("dict utf8 '{name}': {e}")))?;
                builder.append_value(s);
            }
            entries.insert(name.to_string(), Arc::new(builder.finish()));
        }

        Ok(Self { entries })
    }

    /// Get the pre-built dictionary values for a field.
    pub fn get(&self, name: &str) -> Option<&Arc<StringArray>> {
        self.entries.get(name)
    }
}

/// Reads fast fields from a single segment and produces an Arrow RecordBatch.
///
/// If `doc_ids` is `Some`, only those document IDs are read (already filtered by
/// a tantivy query). If `None`, all alive (non-deleted) documents are read.
/// When `doc_id_range` is `Some`, only docs in `[start, end)` are read (for
/// chunked partitions). When `limit` is `Some(n)`, at most `n` documents are
/// returned. Fields are read according to the projected Arrow schema.
pub fn read_segment_fast_fields_to_batch(
    segment_reader: &SegmentReader,
    projected_schema: &SchemaRef,
    doc_ids: Option<&[u32]>,
    doc_id_range: Option<std::ops::Range<u32>>,
    limit: Option<usize>,
    segment_ord: u32,
    dict_cache: Option<&DictCache>,
) -> Result<RecordBatch> {
    let fast_fields = segment_reader.fast_fields();

    let docs_storage: Option<Vec<u32>> = match doc_ids {
        Some(_) => None,
        None => {
            let alive_bitset = segment_reader.alive_bitset();
            let (start, end) = match doc_id_range {
                Some(range) => (range.start, range.end),
                None => (0, segment_reader.max_doc()),
            };
            let iter = (start..end)
                .filter(|&doc_id| alive_bitset.map(|bs| bs.is_alive(doc_id)).unwrap_or(true));
            Some(match limit {
                Some(lim) => iter.take(lim).collect(),
                None => iter.collect(),
            })
        }
    };
    let docs: &[u32] = match &docs_storage {
        Some(docs) => docs.as_slice(),
        None => doc_ids.unwrap_or_default(),
    };

    let num_docs = docs.len();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(projected_schema.fields().len());

    for field in projected_schema.fields() {
        let name = field.name();

        // Handle synthetic internal columns
        if name == "_doc_id" {
            let array: ArrayRef = Arc::new(UInt32Array::from(docs.to_vec()));
            columns.push(array);
            continue;
        }
        if name == "_segment_ord" {
            let array: ArrayRef = Arc::new(UInt32Array::from(vec![segment_ord; num_docs]));
            columns.push(array);
            continue;
        }

        let array: ArrayRef = match field.data_type() {
            DataType::UInt64 => match fast_fields.u64(name) {
                Ok(col) => {
                    let mut builder = UInt64Builder::with_capacity(num_docs);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                }
                Err(_) => arrow::array::new_null_array(field.data_type(), num_docs),
            },
            DataType::Int64 => match fast_fields.i64(name) {
                Ok(col) => {
                    let mut builder = Int64Builder::with_capacity(num_docs);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                }
                Err(_) => arrow::array::new_null_array(field.data_type(), num_docs),
            },
            DataType::Float64 => match fast_fields.f64(name) {
                Ok(col) => {
                    let mut builder = Float64Builder::with_capacity(num_docs);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                }
                Err(_) => arrow::array::new_null_array(field.data_type(), num_docs),
            },
            DataType::Boolean => match fast_fields.bool(name) {
                Ok(col) => {
                    let mut builder = BooleanBuilder::with_capacity(num_docs);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(v) => builder.append_value(v),
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                }
                Err(_) => arrow::array::new_null_array(field.data_type(), num_docs),
            },
            DataType::Timestamp(TimeUnit::Microsecond, None) => match fast_fields.date(name) {
                Ok(col) => {
                    let mut builder = TimestampMicrosecondBuilder::with_capacity(num_docs);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(dt) => builder.append_value(dt.into_timestamp_micros()),
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                }
                Err(_) => arrow::array::new_null_array(field.data_type(), num_docs),
            },
            DataType::Dictionary(_, _) => {
                // Str fast field → DictionaryArray<Int32, Utf8>
                let str_col = match fast_fields.str(name) {
                    Ok(Some(col)) => col,
                    _ => {
                        // Field not present in this segment (schema evolution) — null array
                        columns.push(arrow::array::new_null_array(field.data_type(), num_docs));
                        continue;
                    }
                };

                // Fast path: pre-built segment-wide dictionary shared across chunks.
                // Ordinals from the streamer are 0-based sequential so they map
                // directly to Arrow dictionary indices.
                if let Some(values) = dict_cache.and_then(|c| c.get(name)) {
                    let ords_col = str_col.ords();
                    let mut ord_buf: Vec<Option<u64>> = vec![None; num_docs];
                    ords_col.first_vals(docs, &mut ord_buf);

                    let mut keys_builder = Int32Builder::with_capacity(num_docs);
                    for ord in &ord_buf {
                        match ord {
                            Some(o) => keys_builder.append_value(*o as i32),
                            None => keys_builder.append_null(),
                        }
                    }

                    let dict_values: ArrayRef = Arc::clone(values) as ArrayRef;
                    Arc::new(
                        DictionaryArray::<Int32Type>::try_new(keys_builder.finish(), dict_values)
                            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
                    )
                } else {
                    // Fallback: per-chunk compact dictionary (no cache provided).
                    build_compact_dict_array(&str_col, docs, num_docs, name)?
                }
            }
            DataType::Utf8 => {
                // IpAddr formatted as string (not dictionary-encoded)
                if let Ok(col) = fast_fields.ip_addr(name) {
                    let mut builder = StringBuilder::with_capacity(num_docs, num_docs * 40);
                    for &doc_id in docs {
                        match col.first(doc_id) {
                            Some(ip) => {
                                if let Some(v4) = ip.to_ipv4_mapped() {
                                    builder.append_value(v4.to_string());
                                } else {
                                    builder.append_value(ip.to_string());
                                }
                            }
                            None => builder.append_null(),
                        }
                    }
                    Arc::new(builder.finish())
                } else {
                    // Field not present in this segment (schema evolution) — null array
                    arrow::array::new_null_array(field.data_type(), num_docs)
                }
            }
            DataType::Binary => {
                let bytes_col = match fast_fields.bytes(name) {
                    Ok(Some(col)) => col,
                    _ => {
                        // Field not present in this segment (schema evolution) — null array
                        columns.push(arrow::array::new_null_array(field.data_type(), num_docs));
                        continue;
                    }
                };
                let mut builder = BinaryBuilder::with_capacity(num_docs, num_docs * 64);
                let mut buf = Vec::new();
                for &doc_id in docs {
                    let mut ord_iter = bytes_col.term_ords(doc_id);
                    if let Some(ord) = ord_iter.next() {
                        buf.clear();
                        bytes_col.ord_to_bytes(ord, &mut buf).map_err(|e| {
                            DataFusionError::Internal(format!("ord_to_bytes '{name}': {e}"))
                        })?;
                        builder.append_value(&buf);
                    } else {
                        builder.append_null();
                    }
                }
                Arc::new(builder.finish())
            }
            // --- List<T> branches for multi-valued fields ---
            dt @ DataType::List(inner) => {
                build_list_array(inner, dt, name, fast_fields, docs, num_docs)?
            }
            other => {
                return Err(DataFusionError::Internal(format!(
                    "Unsupported Arrow data type for fast field '{name}': {other:?}"
                )));
            }
        };
        columns.push(array);
    }

    RecordBatch::try_new(projected_schema.clone(), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

/// Build a `ListArray` for a multi-valued fast field, dispatching on the inner type.
///
/// `list_data_type` is the full `DataType::List(...)` used to construct null arrays
/// when the fast field is missing from a segment (schema evolution).
fn build_list_array(
    inner_field: &Arc<Field>,
    list_data_type: &DataType,
    name: &str,
    fast_fields: &tantivy::fastfield::FastFieldReaders,
    docs: &[u32],
    num_docs: usize,
) -> Result<ArrayRef> {
    match inner_field.data_type() {
        DataType::UInt64 => match fast_fields.u64(name) {
            Ok(col) => {
                let mut builder = ListBuilder::new(UInt64Builder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        builder.values().append_value(val);
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            }
            Err(_) => Ok(arrow::array::new_null_array(list_data_type, num_docs)),
        },
        DataType::Int64 => match fast_fields.i64(name) {
            Ok(col) => {
                let mut builder = ListBuilder::new(Int64Builder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        builder.values().append_value(val);
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            }
            Err(_) => Ok(arrow::array::new_null_array(list_data_type, num_docs)),
        },
        DataType::Float64 => match fast_fields.f64(name) {
            Ok(col) => {
                let mut builder = ListBuilder::new(Float64Builder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        builder.values().append_value(val);
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            }
            Err(_) => Ok(arrow::array::new_null_array(list_data_type, num_docs)),
        },
        DataType::Boolean => match fast_fields.bool(name) {
            Ok(col) => {
                let mut builder = ListBuilder::new(BooleanBuilder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        builder.values().append_value(val);
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            }
            Err(_) => Ok(arrow::array::new_null_array(list_data_type, num_docs)),
        },
        DataType::Timestamp(TimeUnit::Microsecond, None) => match fast_fields.date(name) {
            Ok(col) => {
                let mut builder = ListBuilder::new(TimestampMicrosecondBuilder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        builder.values().append_value(val.into_timestamp_micros());
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            }
            Err(_) => Ok(arrow::array::new_null_array(list_data_type, num_docs)),
        },
        DataType::Utf8 => {
            // Could be a Str fast field or IpAddr formatted as string
            if let Ok(Some(str_col)) = fast_fields.str(name) {
                let mut builder = ListBuilder::new(StringBuilder::new());
                let mut buf = String::new();
                for &doc_id in docs {
                    for ord in str_col.term_ords(doc_id) {
                        buf.clear();
                        str_col.ord_to_str(ord, &mut buf).map_err(|e| {
                            DataFusionError::Internal(format!("ord_to_str '{name}': {e}"))
                        })?;
                        builder.values().append_value(&buf);
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            } else if let Ok(col) = fast_fields.ip_addr(name) {
                let mut builder = ListBuilder::new(StringBuilder::new());
                for &doc_id in docs {
                    for val in col.values_for_doc(doc_id) {
                        if let Some(v4) = val.to_ipv4_mapped() {
                            builder.values().append_value(v4.to_string());
                        } else {
                            builder.values().append_value(val.to_string());
                        }
                    }
                    builder.append(true);
                }
                Ok(Arc::new(builder.finish()))
            } else {
                // Field not present in this segment (schema evolution) — null array
                Ok(arrow::array::new_null_array(list_data_type, num_docs))
            }
        }
        DataType::Binary => {
            let bytes_col = match fast_fields.bytes(name) {
                Ok(Some(col)) => col,
                _ => {
                    // Field not present in this segment (schema evolution) — null array
                    return Ok(arrow::array::new_null_array(list_data_type, num_docs));
                }
            };
            let mut builder = ListBuilder::new(BinaryBuilder::new());
            let mut buf = Vec::new();
            for &doc_id in docs {
                for ord in bytes_col.term_ords(doc_id) {
                    buf.clear();
                    bytes_col.ord_to_bytes(ord, &mut buf).map_err(|e| {
                        DataFusionError::Internal(format!("ord_to_bytes '{name}': {e}"))
                    })?;
                    builder.values().append_value(&buf);
                }
                builder.append(true);
            }
            Ok(Arc::new(builder.finish()))
        }
        other => Err(DataFusionError::Internal(format!(
            "Unsupported inner type for List fast field '{name}': {other:?}"
        ))),
    }
}

/// Build a compact `DictionaryArray` from only the ordinals referenced in
/// `docs`.  This is the original per-chunk path, used when no `DictCache` is
/// available.
fn build_compact_dict_array(
    str_col: &tantivy::columnar::StrColumn,
    docs: &[u32],
    num_docs: usize,
    name: &str,
) -> Result<ArrayRef> {
    let mut raw_ords: Vec<Option<u64>> = Vec::with_capacity(num_docs);
    let mut seen_ords: Vec<u64> = Vec::new();
    for &doc_id in docs {
        match str_col.term_ords(doc_id).next() {
            Some(ord) => {
                raw_ords.push(Some(ord));
                if let Err(pos) = seen_ords.binary_search(&ord) {
                    seen_ords.insert(pos, ord);
                }
            }
            None => raw_ords.push(None),
        }
    }

    let mut dict_builder = StringBuilder::with_capacity(seen_ords.len(), seen_ords.len() * 16);
    let mut buf = String::new();
    for &ord in &seen_ords {
        buf.clear();
        str_col
            .ord_to_str(ord, &mut buf)
            .map_err(|e| DataFusionError::Internal(format!("dict build '{name}': {e}")))?;
        dict_builder.append_value(&buf);
    }
    let dict_values: ArrayRef = Arc::new(dict_builder.finish());

    let mut keys_builder = Int32Builder::with_capacity(num_docs);
    for raw in &raw_ords {
        match raw {
            Some(ord) => {
                let compact_idx = seen_ords.binary_search(ord).unwrap();
                keys_builder.append_value(compact_idx as i32);
            }
            None => keys_builder.append_null(),
        }
    }

    Ok(Arc::new(
        DictionaryArray::<Int32Type>::try_new(keys_builder.finish(), dict_values)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
    ))
}
