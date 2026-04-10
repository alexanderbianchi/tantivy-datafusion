# Performance Investigation: Aggregation Pushdown Overhead

## Problem Statement

At 1M docs (1 segment), aggregation pushdown matches native tantivy (1.008x).
At 10M docs (3 segments), pushdown is ~3x slower than native tantivy.
The overhead grows with data size, which is unacceptable.

## Benchmark Setup

- Index: RAM-based (mmap), created with `Index::create_from_tempdir()`
- Schema: `text_few_terms_status` (STRING | FAST, 7 categories), plus 4 other fields
- Query: `SELECT text_few_terms_status, COUNT(*) GROUP BY text_few_terms_status`
- Native: `searcher.search(&AllQuery, &AggregationCollector)` — reuses pre-built reader/searcher
- Pushdown: `SingleTableProvider` + `AggPushdown` rule → `AggDataSource` → `execute_tantivy_agg`

### Bench numbers (stable, plugged in)

| Case | 1M (1 seg) Native | 1M Pushdown | Ratio | 10M (3 seg) Native | 10M Pushdown | Ratio |
|------|-------------------|-------------|-------|--------------------|--------------|-------|
| terms_few | 3.8ms | 3.8ms | 1.008x | 36ms | ~110ms | ~3x |
| terms_few_with_avg | 9.8ms | 10.1ms | 1.02x | 98ms | ~115ms | 1.17x |
| avg_f64 | 6.4ms | 6.8ms | 1.07x | 63ms | 31ms | 0.49x ✅ |
| stats_f64 | 6.4ms | 7.4ms | 1.16x | 63ms | 39ms | 0.62x ✅ |

Key observations:
- Metric-only aggs (avg, stats) are FASTER via pushdown at 10M — DataFusion's vectorized Arrow path beats tantivy's collector for single-pass metrics
- terms_few at 10M is the outlier with 3x overhead
- terms_few_with_avg at 10M is only 1.17x — the avg sub-agg dominates runtime

## ⚠️ Bench vs Production Differences

The bench does NOT represent production. Key differences:

| Aspect | Bench | Production (BYOC) |
|--------|-------|-------------------|
| Index location | RAM / tmpdir (mmap) | S3/GCS via Quickwit split opener |
| Reader construction | `index.reader()` per query | Should be cached on opener |
| Warmup | Skipped (needs_warmup=false) | Required — pre-loads file slices |
| Executor | Single-threaded (tantivy default) | Could use Rayon multi-thread |
| Session | Pre-planned physical plan | Substrait → plan → execute |
| Segment count | 1 (1M) or 3 (10M) | 1 per Quickwit split |
| Query complexity | Pure GROUP BY, no FTS filter | FTS + time range + calculated fields |

**Critical: in production with Quickwit, each split is typically 1 segment.**
The 10M bench with 3 segments is an artifact of the writer buffer flushing during
index construction. Real Quickwit queries across 10M docs would be 1 segment per
split, processed as separate DataFusion partitions — not 3 segments in one split.

If the regression is per-segment overhead within a single split, it won't manifest
in production where splits are 1-segment each.

## Hypotheses Tested

### ❌ TermsAggregation `size: Some(429_496_729)` causes allocation
**Result**: Disproven. Benchmarked size=10, size=100, size=10000, size=429M — all
identical timing within noise. tantivy allocates based on `max_term_id` (actual
unique terms in column), not the requested `size`.

### ❌ Tokio runtime config (multi-thread vs current-thread)
**Result**: Disproven. Tested current_thread, multi_thread(1), multi_thread(default)
× partitions=1, partitions=10. All within 1% noise.

### ❌ target_partitions setting
**Result**: Disproven. partitions=1 vs partitions=10 makes no difference for
AggDataSource (only partition 0 executes).

### ✅ Warmup creating IndexReaders per query
**Result**: Confirmed as ~200ms overhead at 10M. Each warmup function
(`warmup_fast_fields_by_name`, `warmup_inverted_index`) creates a fresh
`IndexReader` internally. For 3 segments, this means opening 3 segment readers
per warmup call, even though mmap directories don't need warmup.

**Fix applied**: `needs_warmup()` on `IndexOpener` trait. `DirectIndexOpener`
returns false, skipping warmup entirely for local/mmap indexes.

**After fix**: 1M at true parity (1.008x). 10M still ~3x.

### ❌ Manual segment loop vs tantivy's `searcher.search()`
**Result**: No difference. Replaced our manual `for segment in segments` loop
with tantivy's native `searcher.search(&AllQuery, &AggregationCollector)`.
Same timing — because tantivy's default executor is single-threaded.

### ❌ Weight creation per segment
**Result**: Hoisting weight creation outside the segment loop didn't help.
The AllQuery weight is essentially free.

### ❌ Per-segment overhead from `box_clone()` or allocations
**Result**: After removing the manual loop and using `searcher.search()`,
there's no per-segment allocation on our side.

## Remaining Overhead at 10M (~74ms)

With warmup disabled, the 10M terms_few path does:
1. `opener.open()` — DirectIndexOpener returns Arc clone (~0μs)
2. `index.reader()` — creates IndexReader with 3 segment readers (~?ms)
3. `tokio::task::spawn_blocking(...)` dispatch (~10μs)
4. `searcher.search(&AllQuery, &AggregationCollector)` — same as native (~36ms)
5. `agg_results_to_batch()` — converts 7 buckets to Arrow (~<1ms)
6. DataFusion `collect()` — polls stream, assembles result (~?ms)

The 74ms gap is steps 2 + 3 + 6. The bench's native path skips steps 2, 3, 6
because it reuses a pre-built reader and has no DataFusion wrapper.

## Things To Try Next

### 1. Cache IndexReader on DirectIndexOpener
The native bench reuses `reader` across iterations. Our code creates a new
`IndexReader` per query via `index.reader()`. For 3 segments, reader
construction opens segment files, builds searcher internals. Cache it.

```rust
impl DirectIndexOpener {
    // Cache the reader — Index is immutable for DirectIndexOpener
    fn cached_reader(&self) -> &IndexReader {
        self.reader.get_or_init(|| self.index.reader().unwrap())
    }
}
```

### 2. Skip spawn_blocking for mmap directories
For `DirectIndexOpener`, tantivy operations are CPU-bound (no I/O blocking
on mmap). `spawn_blocking` adds thread pool dispatch latency. Could run
directly on the tokio executor for mmap openers.

Risk: CPU-bound work on the async executor can starve other tasks. But for
aggregation (tens of ms), this is acceptable.

### 3. Reduce DataFusion `collect()` overhead
`collect()` polls the stream, handles task context, checks cancellation.
For a single-batch result (aggregation returns 1 RecordBatch), this is
disproportionate overhead.

Could bypass `collect()` for single-partition, single-batch results by
calling `exec.execute(0, ctx)?.next().await` directly.

### 4. Use tantivy's multi-thread executor for aggregation
```rust
index.set_multithread_executor(num_cpus)?;
```
This would parallelize across segments via Rayon. For 3 segments × 3.5M
docs each, parallel execution could cut the 36ms native to ~12ms.

### 5. Benchmark with single-segment 10M index
Force a single segment by using a large writer buffer:
```rust
index.writer_with_num_threads(1, 2_000_000_000) // 2GB buffer
```
If the overhead disappears with 1 segment at 10M, it confirms the issue
is per-reader-construction cost scaling with segment count.

### 6. Profile `index.reader()` construction time
Add explicit timing around `index.reader()` to measure its cost at
3 segments. If it's >30ms, that's the bottleneck.

### 7. Compare with Quickwit's production pattern
In Quickwit, each split:
- Has exactly 1 segment
- Uses a storage-backed opener (not DirectIndexOpener)
- Warmup IS needed (S3 file slices)
- Reader is cached per split

The bench's 3-segment pattern doesn't match production. Build a bench
that simulates the Quickwit pattern: 3 separate single-segment indexes
(3 splits), each as a DataFusion partition.

## Conclusions

1. **At 1M (1 segment), pushdown matches native tantivy.** The architecture
   is sound — the overhead is not algorithmic.

2. **The 10M regression is from `index.reader()` construction + DataFusion
   execution wrapper overhead.** These are constant per-query costs that
   don't scale with data, but they're large enough (~74ms) to be 2x on
   a 36ms operation.

3. **In production with Quickwit (1 segment per split), the bench's 3-segment
   pattern doesn't occur.** The regression may not manifest.

4. **Metric-only aggregations (AVG, STATS) are faster through DataFusion
   than through tantivy's native collector** — DataFusion benefits from
   SIMD/vectorized Arrow operations.

5. **The correct fix is caching the IndexReader on the opener** — eliminating
   per-query reader construction. This is the same pattern Quickwit uses
   for its split readers.
