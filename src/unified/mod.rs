//! **Unified single-table approach** (recommended for BYOC).
//!
//! A single `DataSource` that handles FTS queries, fast field reading, scoring,
//! document retrieval, and aggregations internally — no joins needed.
//!
//! Start reviewing here: [`SingleTableProvider`] is the entry point.
//!
//! ## Rule Ordering
//!
//! Register [`agg_pushdown::AggPushdown`] before distributed physical
//! optimizer rules. Tantivy aggregation pushdown must see the local
//! `AggregateExec` subtree before network boundaries are inserted.

pub mod agg_data_source;
pub(crate) mod agg_exec;
pub mod agg_pushdown;
pub(crate) mod plan_traversal;
pub mod single_table_provider;
