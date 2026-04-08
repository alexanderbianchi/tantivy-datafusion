//! **Unified single-table approach** (recommended for BYOC).
//!
//! A single `DataSource` that handles FTS queries, fast field reading, scoring,
//! document retrieval, and aggregations internally — no joins needed.
//!
//! Start reviewing here: [`SingleTableProvider`] is the entry point.

pub(crate) mod agg_exec;
pub mod agg_pushdown;
pub mod ordinal_group_by;
pub(crate) mod plan_traversal;
pub mod single_table_provider;
pub mod task_estimator;
