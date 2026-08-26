# SNP age interval store: implementation and benchmark notes

This document describes the compact interval-store implementation, the August
11, 2026 Gate 2/3 benchmarks, and initial SLURM sizing for SINGER tree
sequences with approximately 25 million sites per draw. The interval store is
an **all-SNP store**. TE and synonymous position lists are downstream
selections and never determine which SNPs are retained.

## What is stored

For each SNP and each usable mutation observation, the store records the
posterior age interval `[node_time, parent_time]` plus its draw ID. Ragged
`offsets` map each SNP row to its interval records. Separate per-SNP counts
record presence, usable draws, usable intervals, root skips, and missing draws.
A packed two-bit status matrix preserves draw-level absent/present/root/usable
state.

The store therefore retains the complete piecewise-uniform posterior implied
by all usable intervals. It does not materialize a dense SNP-by-age-grid
matrix. Mean ages, CDFs, cell masses, and boundary probabilities are computed
from the interval endpoints when requested.

The production builder forms the union of sites across every input draw. A SNP
does not need to appear in a TE or synonymous list to be included.

## Implemented scalability changes

### Selective TSZ catalog access

The catalog pass opens `.tsz` as Zarr and reads only the shared coordinate
dictionary, encoded site positions, sequence length, and top-level metadata.
It decodes `sites/position` through the coordinate dictionary. The full tree
sequence is loaded only once per draw during interval extraction. Ordinary
`.trees` files retain a full-load fallback.

On `run.combined.100.tsz`, selective catalog access took 4.50 seconds in the
post-implementation smoke test. The earlier instrumented Gate 2 comparison
was 1.72 seconds and 771 MB peak RSS for selective site access versus 38.58
seconds and 11.22 GB peak RSS for a full TSZ load.

### Fractional edge coordinates

The real ARG contains fractional edge breakpoints even though VCF site
positions are integral. This is valid: recombination-tree boundaries may fall
between sites. Parent lookup therefore uses a structured `(child, left
float64)` key and exact guard checks; it must not cast edge coordinates to
integers.

### Bucketed all-SNP assembly

Each usable interval is appended to a row-range bucket on node-local scratch.
After all draws are extracted, one bucket at a time is stably sorted by SNP
row and copied into the final contiguous arrays. Per-row counts are accumulated
with `np.bincount(...).astype(np.uint32)`; present-draw counts use aligned plain
addition where site rows are unique.

The final merge is substantially I/O-bound on Quobyte. More CPUs do not speed
up that phase.

### Parallel scalar correctness audit

`tools/benchmark_interval_store_gate2.py` accepts `--audit-workers`. On Linux, forked
workers share the loaded tree sequence and parent arrays read-only. Mutation
selection is performed once, so serial and parallel runs audit the same
deterministic strata. Reports include the selected-ID digest, exact mismatch
count, first mismatches, worker timings, and requested/used worker counts.

This parallelism applies to the validation audit, not to the production
builder. Safe parallel extraction would require per-draw bucket/count/status
shards followed by a deterministic parent merge. The current builder must not
be parallelized by allowing workers to mutate its shared arrays or bucket
handles.

### Candidate access strategies

The interval reader supports four explicit strategies:

- `gather`: read only requested rows;
- `coalesced`: combine nearby requested rows into contiguous slabs;
- `scan`: sequentially scan the complete endpoint arrays and evaluate only
  requested rows;
- `cache`: sequentially build a candidate-only cache in requested scratch,
  then query that cache.

Candidate caches are validated against the source store, published atomically,
and cleaned up on success or error unless explicitly retained. They accelerate
downstream repeated access only; they are not substitutes for the all-SNP
store.

## Gate 2 correctness and precision

The real-draw benchmark used `run.combined.100.tsz`:

- 25,030,335 site rows;
- 25,983,474 mutations;
- 24,902,283 usable intervals;
- 1,081,191 root-skipped mutations;
- fractional-edge structured parent lookup;
- 10,000 scalar `ts.at(position).parent(node)` checks, with zero mismatches;
- 100 balanced buckets, from 228,052 to 258,711 intervals per bucket;
- no float32 interval collapses in a 10,000-interval precision sample;
- maximum individual CDF error `2.69e-7`;
- maximum aggregate CDF error `2.33e-10`.

The serial 10,000-mutation audit took 5,792.99 seconds (96.55 minutes). Four
forked workers checked the same 10,000 mutations in 2,109.95 seconds (35.17
minutes), a 2.75x wall-time speedup. Their summed worker CPU time was 140.33
minutes, so the speedup is intentionally described as sublinear. The complete
four-CPU Gate 2 job took 40m29s and peaked at 19.9 GB RSS.

## Gate 3 all-SNP fixture and candidate access

A three-draw fixture was built from draws 100--102 using float32 endpoints and
the corrected chromosome-offset table:

- 27,245,216 union SNP rows;
- 74,715,451 usable interval records;
- 1.6 GiB reported filesystem use;
- 24m26s wall time on one CPU;
- 34.8 GB peak RSS;
- node-local bucket scratch;
- complete deep validation before atomic publication.

The real synonymous list contained 485,671 requested coordinates. In this
partial-draw fixture, 399,673 resolved, 396,678 were eligible, 85,998 were not
in the three-draw union, and 2,995 were resolved but ineligible. These rows
were used only for the scattered-access benchmark.

With 396,678 candidates and 21 boundaries, all strategies were numerically
identical. Warm best times were:

| Strategy | Best repeat | Estimated endpoint/offset bytes read |
| --- | ---: | ---: |
| scan | 19.03 s | 890.4 MB |
| gather | 20.12 s | 16.9 MB |
| cache | 20.16 s | 16.9 MB from cache |
| coalesced | 20.49 s | 60.7 MB |

The cold first gather took 26.05 seconds. The candidate cache occupied 16.9 MB
and took 0.64 seconds to build after earlier strategies had warmed source
pages. The full Gate 3 job took 4m23s and peaked at 1.45 GB RSS.

The close timings show that per-row posterior evaluation dominates this
three-draw fixture. Use `gather` for a single boundary pass because it is the
simplest and reads the fewest bytes. Use `cache` when the same candidate set is
read repeatedly. Do not claim a universal cache or scan advantage from this
warm-cache, three-draw result. The CLI keeps `gather` as its explicit default;
`auto` continues to refuse to invent an unmeasured threshold.

## Initial 75-draw resource profile

These are conservative starting requests for approximately 25--30 million
union SNPs and 75 input draws of the measured size. They are projections from
one- and three-draw measurements, not results from a 75-draw production job.

### Recommended default build

- CPUs: **1**
- memory: **48 GB**
- time limit: **16 hours** (12 hours may work, but leaves less I/O margin)
- final float32 store: approximately **17.1 GiB**
- node-local packed bucket scratch: approximately **22.6 GiB**
- recommended free node-local scratch: **at least 32 GiB**
- Quobyte free space during atomic construction: allow approximately twice
  the final-store size plus margin if an older store is retained

The final-size projection uses 1,867,886,275 interval records, nine bytes per
float32 record across `below`, `above`, and `draw_id`, approximately 0.91 GiB
of fixed catalog/count arrays, and approximately 0.48 GiB of packed status.
The 16-hour request includes conservative allowance for the measured
Quobyte-bound final merge and flush.

### Validation audit

- CPUs: **4**
- memory: **48 GB**
- time limit: **1 hour**

This profile is for the 10,000-mutation real-draw audit. It is separate from
the 75-draw store construction.

### Parallel build status

A multi-worker production build is not yet implemented. A safe future design
would use independent per-draw shards and merge them in draw-ID order. Based
on the measured approximately 20 GB peak footprint of one loaded real draw,
rough planning envelopes are 64--80 GB for two extraction workers and
128--160 GB for four. These are design estimates, not approved defaults, and
parallel extraction would not reduce the Quobyte-bound final merge time.

## Reproducibility artifacts

- `results/interval-gate2-run-combined-100.json`: serial Gate 2 report;
- `results/interval-gate2-run-combined-100-w4.json`: four-worker Gate 2 report;
- `results/interval-gate3-3draw-store/`: validated all-SNP fixture;
- `results/interval-gate3-input-summary.json`: candidate resolution counts;
- `results/interval-gate3-candidate-access.json`: Gate 3 strategy report.

No 75-draw production job was launched during these benchmarks.
