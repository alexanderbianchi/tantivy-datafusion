# tantivy-datafusion Development Guidelines

These rules apply to all code in this crate. They encode the quality bar
from dd-datafusion, InfluxDB IOx, and SpiceAI — production DataFusion
deployments that learned these lessons the hard way.

## Code Structure

- **Functions under 100 lines.** If it's longer, decompose it.
- **Indent under 3 levels.** Use early return, not deep nesting.
- **No `#[allow(clippy::too_many_arguments)]`.** If a function takes more than
  6 parameters, extract a context struct. A 14-parameter function is a struct
  with methods, not a function.
- **No method aliasing by suffixes.** Don't create `process()` + `process_with_score()`.
  Refactor for flexibility instead.
- **Prefer iterator patterns over manual loops.**
- **No dead code.** Don't add `#[allow(dead_code)]`. If it's unused, delete it.

## Memory and Performance

- **Allocations must be O(record batch), NOT O(row).** This is the single most
  important performance rule. Never allocate per-doc inside a segment scan loop.
- **Never materialize an entire segment into memory before batching.** Stream in
  `batch_size` chunks. If you see `collect()` on all doc_ids/scores before the
  batching loop, that's wrong — the batching loop should drive the collection.
- **Prefer `Arc::clone(&x)` over `x.clone()` for clarity on Arc types.**
- **Use `_else` variants for combinators that allocate:** `ok_or_else`, `unwrap_or_else`,
  `map_or_else` — not `ok_or`, `unwrap_or`, `map_or`.
- **No `.to_vec()` on large data.** Check if zero-copy conversion exists first.

## Async and Tokio

- **No blocking I/O on the async executor.** Tantivy segment reads, warmup, and
  fast field access are synchronous. Wrap them in `tokio::task::spawn_blocking()`.
- **Use MPSC channels (capacity 1-2) to bridge blocking → async.** The pattern:
  spawn_blocking produces batches, sends via channel, async stream receives.
- **Warmup is I/O.** Even if mmap makes it fast for local files, storage-backed
  directories do real network I/O. Design for the general case.

## DataFusion Integration

### Represent Complexity in the Plan

The default question for every design decision:
**"Could this logic be a plan node instead of imperative code?"**

- Schema reconciliation → adapter exec wrapping the source
- Aggregation pushdown → physical optimizer rule rewriting AggregateExec
- Filter pushdown → physical optimizer rule pushing predicates into scans
- Type coercion → adapter exec between source and downstream consumers

If logic is invisible to the optimizer (buried in a helper function), it should
probably be a plan node so the optimizer can reason about it.

### Trait Implementations

- **`supports_filters_pushdown`**: Return `Exact` only if the provider guarantees
  no false positives. Return `Inexact` if it's best-effort pruning (DataFusion
  re-applies the filter automatically). Return `Unsupported` for predicates you
  can't use at all. `Inexact` is almost always the right choice for tantivy fast
  field ranges.
- **`statistics()`**: Return what you know. `num_rows` from segment metadata is
  free. The optimizer uses this for join ordering and memory estimation.
- **`eq_properties()`**: Declare output ordering when you have it. Tantivy segments
  emit docs in doc_id order — declaring `_doc_id ASC` lets the optimizer skip
  redundant sorts and use `SortPreservingMergeExec`.

### Serialization (Distributed Execution)

- **Every custom ExecutionPlan must be codec-serializable.** If it's not in the
  codec, distributed execution silently fails. Test round-trips.
- **Every field that affects execution must survive serialization.** `topk`, `pushed_filters`,
  `pre_built_query`, `agg_mode` — if it's `None` after deserialization when it
  was `Some` before, that's a correctness bug.
- **Test codec round-trips for every provider type.** A mock `OpenerFactory` with
  a RAM-based index suffices.

### Session and Catalog

- **Contributions (optimizer rules, UDFs, codecs) should be merged once at startup,**
  not reconstructed per-query.
- **Object stores and warmup caches belong on `IndexOpener` or `RuntimeEnv`,** not
  per-query state. They persist across queries.

## Tantivy-Specific Rules

- **Handle `alive_bitset` in every code path.** Tantivy's `for_each`, `for_each_no_score`,
  and `for_each_pruning` do NOT filter deleted documents internally. You must apply
  the bitset yourself.
- **`TermQuery` on tokenized fields searches the inverted index,** even for fast fields.
  Use `RangeQuery(Included(term), Included(term))` for exact equality on fast fields.
  Use `TermQuery` only when you specifically want inverted index lookup on a non-tokenized field.
- **BM25 scores are per-segment** with `EnableScoring::enabled_from_searcher` (index-wide IDF).
  Scores are comparable across segments within the same index but NOT across indexes.
- **Multi-valued fast fields produce `List<T>`,** not scalar `T`. Check `field_entry.is_fast()`
  AND the cardinality — `schema_mapping` must detect multi-valued fields from segment metadata.
- **Warmup the right fields.** Only warm fast fields that are projected and inverted index
  fields that are queried. Don't warm everything.

## Testing

- **Every new code path needs at least one test.** No exceptions.
- **Test codec round-trips** for any new or modified ExecutionPlan type.
- **Test edge cases:** empty segments, segments with all docs deleted, zero-match queries,
  segments with different schemas (missing columns).
  kernel has different behavior for each.
- **Test the optimization rules** by verifying the expected plan shape, not just query results.

## What NOT to Do

- Don't add TODO comments tracking future work. If it needs to be done, do it or
  describe the target state in a doc.
- Don't say "acceptable for now" in code comments. Write the code you want to ship.
- Don't suppress clippy warnings. Fix the underlying issue.
- Don't create two parallel implementations of the same concern (e.g., two aggregation
  exec types). Unify them.
- Don't hardcode magic numbers (e.g., `size: 65535` for terms aggregation). Use constants
  or make them configurable.
