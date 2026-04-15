//! Nested approximate top-k aggregations for ES-style queries.
//!
//! This module provides a nested aggregation execution path that is separate
//! from the existing single-level `AggPushdown` rule. It supports:
//!
//! - multi-level bucket nesting (terms → terms → metrics)
//! - explicit overscan via configurable `fanout` per level
//! - native tantivy pushdown when all inputs are simple fields
//! - a generic DataFusion fallback for expression-based levels
//!
//! ## Entry point
//!
//! Use [`plan_builder::build_nested_approx_plan`] to construct the physical
//! plan from a [`spec::NestedApproxAggSpec`].

pub mod exec;
pub mod node_table;
pub mod plan_builder;
pub mod spec;
