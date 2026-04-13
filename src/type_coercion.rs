use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanArray, BooleanBuilder, Float64Array,
    Float64Builder, Int64Array, Int64Builder, ListBuilder, RecordBatch, StringArray, StringBuilder,
    TimestampMicrosecondArray, TimestampMicrosecondBuilder, UInt64Array, UInt64Builder,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::common::Result;
use datafusion::error::DataFusionError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FastFieldCoercion {
    Exact,
    Cast(DataType),
    ScalarToList {
        item_type: DataType,
        cast_scalar_to: Option<DataType>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FastFieldColumnPlan {
    pub(crate) output_field: Arc<Field>,
    pub(crate) source_name: Option<String>,
    pub(crate) coercion: FastFieldCoercion,
}

#[derive(Debug, Clone)]
pub(crate) struct FastFieldProjectionPlan {
    pub(crate) output_schema: SchemaRef,
    pub(crate) columns: Vec<FastFieldColumnPlan>,
}

pub(crate) fn infer_canonical_fast_field_schema(split_schemas: &[SchemaRef]) -> Result<SchemaRef> {
    if split_schemas.is_empty() {
        return Err(DataFusionError::Plan(
            "at least one split schema is required".to_string(),
        ));
    }

    let mut fields = vec![
        Field::new("_doc_id", DataType::UInt32, false),
        Field::new("_segment_ord", DataType::UInt32, false),
    ];

    for schema in split_schemas {
        for field in schema.fields() {
            let name = field.name();
            if name == "_doc_id" || name == "_segment_ord" {
                continue;
            }

            let Some(existing_idx) = fields.iter().position(|candidate| candidate.name() == name)
            else {
                fields.push(field.as_ref().clone());
                continue;
            };

            let merged = merge_field_types(fields[existing_idx].data_type(), field.data_type())
                .ok_or_else(|| {
                    DataFusionError::Plan(format!(
                        "conflicting fast field types for '{name}': {:?} vs {:?}; \
                         provide an explicit canonical schema",
                        fields[existing_idx].data_type(),
                        field.data_type()
                    ))
                })?;

            fields[existing_idx] = Field::new(name, merged, true);
        }
    }

    Ok(Arc::new(Schema::new(fields)))
}

pub(crate) fn plan_fast_field_projection(
    split_schema: &SchemaRef,
    canonical_schema: &SchemaRef,
) -> Result<FastFieldProjectionPlan> {
    let mut source_fields = Vec::new();
    let mut source_names = Vec::new();
    let mut column_plans = Vec::with_capacity(canonical_schema.fields().len());

    for output_field in canonical_schema.fields() {
        let output_name = output_field.name();
        let maybe_source = split_schema
            .fields()
            .iter()
            .find(|candidate| candidate.name() == output_name)
            .cloned();

        let (source_name, coercion) = match maybe_source {
            Some(source_field) => {
                let coercion =
                    plan_fast_field_coercion(source_field.data_type(), output_field.data_type())
                        .map_err(|e| {
                            DataFusionError::Plan(format!(
                                "cannot coerce split field '{output_name}' from {:?} to {:?}: {e}",
                                source_field.data_type(),
                                output_field.data_type()
                            ))
                        })?;

                if !source_names.iter().any(|name| name == source_field.name()) {
                    source_names.push(source_field.name().to_string());
                    source_fields.push(source_field.as_ref().clone());
                }

                (Some(source_field.name().to_string()), coercion)
            }
            None => (None, FastFieldCoercion::Exact),
        };

        column_plans.push(FastFieldColumnPlan {
            output_field: Arc::clone(output_field),
            source_name,
            coercion,
        });
    }

    if source_fields.is_empty() {
        if let Some(doc_id_field) = split_schema
            .fields()
            .iter()
            .find(|field| field.name() == "_doc_id")
        {
            source_fields.push(doc_id_field.as_ref().clone());
            source_names.push("_doc_id".to_string());
        }
    }

    Ok(FastFieldProjectionPlan {
        output_schema: Arc::clone(canonical_schema),
        columns: column_plans,
    })
}

pub(crate) fn apply_fast_field_projection(
    source_batch: &RecordBatch,
    projection_plan: &FastFieldProjectionPlan,
) -> Result<RecordBatch> {
    let num_rows = source_batch.num_rows();
    let mut columns = Vec::with_capacity(projection_plan.columns.len());

    for column_plan in &projection_plan.columns {
        let array = match &column_plan.source_name {
            Some(source_name) => {
                let source_idx = source_batch.schema().index_of(source_name).map_err(|_| {
                    DataFusionError::Internal(format!(
                        "source fast field '{source_name}' missing from batch"
                    ))
                })?;
                let source_array = source_batch.column(source_idx);
                coerce_array(source_array, &column_plan.coercion)?
            }
            None => arrow::array::new_null_array(column_plan.output_field.data_type(), num_rows),
        };
        columns.push(array);
    }

    RecordBatch::try_new(Arc::clone(&projection_plan.output_schema), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

fn coerce_array(array: &ArrayRef, coercion: &FastFieldCoercion) -> Result<ArrayRef> {
    match coercion {
        FastFieldCoercion::Exact => Ok(Arc::clone(array)),
        FastFieldCoercion::Cast(target_type) => {
            cast(array, target_type).map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
        }
        FastFieldCoercion::ScalarToList {
            item_type,
            cast_scalar_to,
        } => {
            let scalar = match cast_scalar_to {
                Some(target_type) => cast(array, target_type)
                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
                None => Arc::clone(array),
            };
            wrap_scalar_array_in_list(&scalar, item_type)
        }
    }
}

fn plan_fast_field_coercion(source: &DataType, target: &DataType) -> Result<FastFieldCoercion> {
    if source == target {
        return Ok(FastFieldCoercion::Exact);
    }

    if let DataType::List(inner) = target {
        if matches!(source, DataType::List(_)) {
            return Err(DataFusionError::Plan(
                "list coercions require an exact list type match".to_string(),
            ));
        }

        let item_type = inner.data_type().clone();
        if source == &item_type {
            return Ok(FastFieldCoercion::ScalarToList {
                item_type,
                cast_scalar_to: None,
            });
        }
        if scalar_cast_supported(source, &item_type) {
            return Ok(FastFieldCoercion::ScalarToList {
                item_type: item_type.clone(),
                cast_scalar_to: Some(item_type),
            });
        }

        return Err(DataFusionError::Plan(format!(
            "unsupported scalar-to-list coercion from {source:?} to {target:?}"
        )));
    }

    if scalar_cast_supported(source, target) {
        return Ok(FastFieldCoercion::Cast(target.clone()));
    }

    Err(DataFusionError::Plan(format!(
        "unsupported fast field coercion from {source:?} to {target:?}"
    )))
}

fn scalar_cast_supported(source: &DataType, target: &DataType) -> bool {
    if source == target {
        return true;
    }

    match (source, target) {
        (DataType::Dictionary(_, value_type), target)
            if value_type.as_ref() == &DataType::Utf8 && is_utf8_like(target) =>
        {
            true
        }
        (source, target)
            if is_utf8_like(target) && matches!(source, DataType::Utf8 | DataType::Utf8View) =>
        {
            true
        }
        (source, target) if is_utf8_like(target) && is_numeric(source) => true,
        (DataType::Boolean, target) if is_utf8_like(target) => true,
        (DataType::Timestamp(_, _), target) if is_utf8_like(target) => true,
        (source, target) if is_numeric(source) && is_numeric(target) => true,
        (DataType::Timestamp(_, _), DataType::Timestamp(_, _)) => true,
        _ => false,
    }
}

fn is_utf8_like(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::Utf8View)
}

fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn merge_field_types(left: &DataType, right: &DataType) -> Option<DataType> {
    if left == right {
        return Some(left.clone());
    }

    let left_promoted = promotable_scalar_type(left)?;
    let right_promoted = promotable_scalar_type(right)?;
    if left_promoted == right_promoted {
        return Some(DataType::List(Arc::new(Field::new(
            "item",
            left_promoted,
            true,
        ))));
    }

    None
}

fn promotable_scalar_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::UInt64
        | DataType::Int64
        | DataType::Float64
        | DataType::Boolean
        | DataType::Utf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::Timestamp(TimeUnit::Microsecond, None) => Some(data_type.clone()),
        DataType::Dictionary(_, value_type) if value_type.as_ref() == &DataType::Utf8 => {
            Some(DataType::Utf8)
        }
        DataType::List(inner) => Some(inner.data_type().clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_cast_supported_accepts_numeric_to_utf8_view() {
        assert!(scalar_cast_supported(
            &DataType::Float64,
            &DataType::Utf8View
        ));
        assert!(scalar_cast_supported(
            &DataType::UInt64,
            &DataType::Utf8View
        ));
    }

    #[test]
    fn scalar_cast_supported_accepts_utf8_dictionary_to_utf8_view() {
        let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert!(scalar_cast_supported(&dict_type, &DataType::Utf8View));
    }
}

fn wrap_scalar_array_in_list(array: &ArrayRef, item_type: &DataType) -> Result<ArrayRef> {
    match item_type {
        DataType::UInt64 => wrap_typed_array::<UInt64Array, UInt64Builder, u64>(
            array,
            item_type,
            |typed: &UInt64Array, row| typed.value(row),
        ),
        DataType::Int64 => wrap_typed_array::<Int64Array, Int64Builder, i64>(
            array,
            item_type,
            |typed: &Int64Array, row| typed.value(row),
        ),
        DataType::Float64 => wrap_typed_array::<Float64Array, Float64Builder, f64>(
            array,
            item_type,
            |typed: &Float64Array, row| typed.value(row),
        ),
        DataType::Boolean => wrap_typed_array::<BooleanArray, BooleanBuilder, bool>(
            array,
            item_type,
            |typed: &BooleanArray, row| typed.value(row),
        ),
        DataType::Utf8 => wrap_string_array_in_list(array, item_type),
        DataType::Binary => wrap_binary_array_in_list(array, item_type),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            wrap_typed_array::<TimestampMicrosecondArray, TimestampMicrosecondBuilder, i64>(
                array,
                item_type,
                |typed: &TimestampMicrosecondArray, row| typed.value(row),
            )
        }
        other => Err(DataFusionError::Plan(format!(
            "unsupported scalar-to-list inner type: {other:?}"
        ))),
    }
}

trait ValueAppender<T> {
    fn append_value(&mut self, value: T);
}

impl ValueAppender<u64> for UInt64Builder {
    fn append_value(&mut self, value: u64) {
        UInt64Builder::append_value(self, value);
    }
}

impl ValueAppender<i64> for Int64Builder {
    fn append_value(&mut self, value: i64) {
        Int64Builder::append_value(self, value);
    }
}

impl ValueAppender<f64> for Float64Builder {
    fn append_value(&mut self, value: f64) {
        Float64Builder::append_value(self, value);
    }
}

impl ValueAppender<bool> for BooleanBuilder {
    fn append_value(&mut self, value: bool) {
        BooleanBuilder::append_value(self, value);
    }
}

impl ValueAppender<i64> for TimestampMicrosecondBuilder {
    fn append_value(&mut self, value: i64) {
        TimestampMicrosecondBuilder::append_value(self, value);
    }
}

fn wrap_string_array_in_list(array: &ArrayRef, item_type: &DataType) -> Result<ArrayRef> {
    let typed = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "expected array {:?}, got {:?}",
                item_type,
                array.data_type()
            ))
        })?;

    let mut builder = ListBuilder::new(StringBuilder::default());
    for row in 0..typed.len() {
        if typed.is_null(row) {
            builder.append(false);
            continue;
        }
        builder.values().append_value(typed.value(row));
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

fn wrap_binary_array_in_list(array: &ArrayRef, item_type: &DataType) -> Result<ArrayRef> {
    let typed = array
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| {
            DataFusionError::Internal(format!(
                "expected array {:?}, got {:?}",
                item_type,
                array.data_type()
            ))
        })?;

    let mut builder = ListBuilder::new(BinaryBuilder::default());
    for row in 0..typed.len() {
        if typed.is_null(row) {
            builder.append(false);
            continue;
        }
        builder.values().append_value(typed.value(row));
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

fn wrap_typed_array<ArrayType, ValueBuilder, Value>(
    array: &ArrayRef,
    item_type: &DataType,
    value_at: impl Fn(&ArrayType, usize) -> Value,
) -> Result<ArrayRef>
where
    ArrayType: Array + 'static,
    ValueBuilder: arrow::array::ArrayBuilder + Default + ValueAppender<Value>,
    Value: Copy,
{
    let typed = array.as_any().downcast_ref::<ArrayType>().ok_or_else(|| {
        DataFusionError::Internal(format!(
            "expected array {:?}, got {:?}",
            item_type,
            array.data_type()
        ))
    })?;

    let mut builder = ListBuilder::new(ValueBuilder::default());
    for row in 0..typed.len() {
        if typed.is_null(row) {
            builder.append(false);
            continue;
        }
        builder.values().append_value(value_at(typed, row));
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}
