# Compact SNP Posterior Interval Store: Implementation Plan

## Goal

Replace the dense SNP-by-age-bin CDF store with a compact interval store that preserves the mutation-age posterior for every SNP in every posterior ARG draw.

The new store will record each usable mutation interval directly as its mutation-node age (`below`) and parent-node age (`above`). It will not choose an age-bin width or construct a global age grid during the build. Consequently, users can reconstruct CDFs at any later bin width without rebuilding the store.

For the current data, the dense design requires terabytes because it materializes every SNP across every age bin. The interval design is expected to require approximately 35 GB with lossless `float64` endpoints, or approximately 18 GB with optional `float32` endpoints.

The store must cover all approximately 25 million SNPs. Restricting construction to the current TE and synonymous lists is therefore outside the scope of this design.

## Decisions from the plan review

This revision incorporates the findings in `INTERVAL_STORE_PLAN_REVIEW.md`. The following decisions record where the review is accepted and where this plan intentionally differs.

### Accepted recommendations

1. **Correct chromosome offsets before implementation.** The naive two-column chromosome-length table omits the one-base gaps used when the chromosomes were concatenated. This can silently attach ages to the wrong variants.
2. **Remove per-site and per-mutation Python object loops.** Extraction must operate on NumPy table columns and blockwise compiled operations. Iterating through approximately 1.95 billion mutation records in Python is not viable.
3. **Use one extraction pass followed by a bucket merge.** A count pass followed by a second extraction pass would needlessly decompress and traverse every ARG twice.
4. **Use bucketed assembly rather than per-SNP random cursor writes.** Bucket files permit sequential scratch writes, bounded in-memory sorts, and contiguous final arrays.
5. **Include `missing_draw_count.npy`.** It is required to preserve and validate the invariant `present_draw_count + missing_draw_count == n_draws`.
6. **Compute candidate CDFs only at required boundaries.** The synonymous matcher needs approximately 19 boundary values rather than a full dense CDF for every candidate.
7. **Use tolerant numerical equivalence tests.** Floating-point accumulation order can prevent bit-for-bit equality with the old dense implementation.
8. **Use the composite-key parent lookup as the first implementation candidate.** Its preconditions and guard checks are explicit and testable, while its runtime and peak memory remain benchmark questions.
9. **Use equal row-range buckets with an explicit memory bound.** Mutation density is sufficiently uniform in the measured draw for equal row ranges, but the builder must detect pathological bucket imbalance.
10. **Use a stable row sort during assembly.** This preserves draw order and mutation-table order within a `(row, draw)` group, making builds reproducible.
11. **Specify unresolved-position behavior.** TE and synonymous input positions need explicit error-or-drop policies and separate reporting for unresolved and ineligible rows.
12. **Force composite-key operands to `int64`.** NumPy 2 weak-scalar promotion otherwise permits silent `int32` wrap or imprecise `float64` keys.
13. **Require strict positive interval width.** A valid tskit covering edge has `parent_time > child_time`; equality is corruption, not a production point mass.
14. **Specify two-bit status packing and bucket-offset placement.** Both compact representations need explicit round-trip and boundary invariants.
15. **Retire unsafe dense launchers.** A launcher that still uses the naive chromosome offsets must not remain submittable during migration.
16. **Audit parent lookup on a real draw before Gate 3.** Fixture equivalence is necessary but cannot reproduce production edge count, tree count, or guard behavior.
17. **Use measured count-reduction methods.** Unique site rows use direct indexed addition; repeated mutation rows use `bincount` with an explicit `uint32` conversion; ordered usable rows use adjacent deduplication.
18. **Benchmark candidate interval access.** Scattered gathers, coalesced gathers, full sequential scans, and a compact candidate cache have materially different Quobyte I/O profiles.

### Qualified acceptance: vectorized parent lookup

The review is correct that the parent of a mutation node at position `x` is given by a covering edge satisfying:

```text
edge.child == mutation.node
edge.left <= x < edge.right
```

No covering edge means that the mutation node is a root at that position.

However, a global `np.searchsorted(edge_left, mutation_position)` is not correct after sorting by `(child, left)`, because `left` is ordered only within each child-node group.

The first implementation candidate will encode `(child, left)` as one sortable integer key. All coordinate-integrality checks occur before conversion:

```python
S = int(ts.sequence_length) + 1
int64_max = np.iinfo(np.int64).max
if ts.num_nodes * S - 1 > int64_max:
    raise OverflowError("composite edge key exceeds int64")

edge_child = np.asarray(edges.child, dtype=np.int64)
edge_left = np.asarray(edges.left)
edge_right = np.asarray(edges.right)
site_position = np.asarray(sites.position)

if (
    np.any(~np.isfinite(edge_left))
    or np.any(edge_left != np.floor(edge_left))
    or np.any(~np.isfinite(edge_right))
    or np.any(edge_right != np.floor(edge_right))
    or np.any(~np.isfinite(site_position))
    or np.any(site_position != np.floor(site_position))
):
    raise ValueError("composite edge lookup requires integral coordinates")

edge_left = edge_left.astype(np.int64)
edge_right = edge_right.astype(np.int64)
mutation_node = np.asarray(mutations.node, dtype=np.int64)
mutation_position = site_position.astype(np.int64)[mutations.site]

edge_key = edge_child * S + edge_left
order = np.argsort(edge_key, kind="stable")
edge_key = edge_key[order]

query_key = mutation_node * S + mutation_position
assert edge_key.dtype == np.int64
assert query_key.dtype == np.int64

edge_child_sorted = edge_child[order]
edge_right_sorted = edge_right[order]
edge_parent_sorted = np.asarray(edges.parent, dtype=np.int64)[order]

index = np.searchsorted(edge_key, query_key, side="right") - 1
safe = np.maximum(index, 0)
covered = (
    (index >= 0)
    & (edge_child_sorted[safe] == mutation_node)
    & (mutation_position < edge_right_sorted[safe])
)
```

The child-equality guard rejects a hit from the preceding child group. The right-edge guard rejects a mutation position in a gap between two edges belonging to the same child. Both guards are required.

This method requires integral site positions and integral edge endpoints. It also requires the composite key to fit in signed 64-bit integers. Integrality must be checked before casting, and both key arrays must be asserted as `np.int64` before searching. If either precondition fails, use a vectorized batched binary search with per-child lower and upper edge bounds; do not fall back to Python per-mutation or per-child searching.

Reordered edge columns must be materialized once, as shown above, rather than repeatedly evaluating expressions such as `edge_child[order][safe]`. Gate 2 must measure the key, permutation, reordered columns, and sorting workspace together. An alternative implementation may use `original_index = order[safe]` to avoid retaining some reordered columns if benchmarks show lower peak memory without unacceptable repeated gathers.

The composite-key and fallback algorithms must be tested against `tree.parent()` before either is used on production data.

The review's estimate of approximately 15 seconds to sort 74 million edges is not adopted as a planning assumption. Sorting time, temporary memory, and total extraction time must be measured on a compute node. The design accepts vectorized extraction; it does not assume an unverified runtime.

### Disagreement: endpoint precision

The review recommends `float32` endpoints because its precision is small relative to the current 1,000-generation bin width. This plan retains `float64` as the default because:

- the store is intentionally independent of bin width;
- “complete posterior” should preserve the tree-sequence ages by default;
- old, narrow intervals can lose meaningful width under `float32`; and
- the estimated `float64` store remains manageable at approximately 35 GB.

Endpoints are stored as `float32`, which is the only supported format. The original requirement here was that `float64` be the default and `float32` an explicit, measured deviation; that measurement has since been made and float32 adopted outright. Its worst-case resolution is 4 generations, at the oldest age in the 75-draw production store (36,744,633 generations), against a 1,000-generation analysis bin width, and it is sub-generation across the range the analysis occupies. `float64` would double the endpoint arrays from 13.9 GiB to 27.8 GiB on that store to buy precision far finer than the ages themselves are resolved. Readers take the dtype from store metadata rather than a build flag, so any float64 store written by an earlier version still loads.

### Disagreement: retaining `draw_id.npy`

`draw_id.npy` remains part of the required schema. It is needed to:

- preserve which posterior draw generated each interval;
- distinguish multiple mutations at one SNP within one draw;
- support equal-per-draw weighting as an alternative to the current equal-per-interval semantics; and
- audit or diagnose draw-specific anomalies without rebuilding the store.

Its approximately 1.95 GB cost is justified by the requirement to retain the complete posterior provenance.

### Disagreement: retaining packed per-draw status

Aggregate count arrays cannot identify which particular draws were missing a site or contained only root-skipped mutations. A packed per-draw status array retains this information at an estimated cost of approximately 470 MB.

The status array therefore remains required unless the scientific requirement is explicitly narrowed from “complete posterior for every SNP” to only the marginal interval mixture. Its unique consumers are draw-specific missing-data diagnostics and validation of absent versus present-but-unusable observations. Posterior-draw reweighting is not listed as a justification because usable draws are already identified by `draw_id.npy`.

### Qualified disagreement: CDF point and cell semantics

The updated review is correct that point evaluation and legacy cell integration are distinct operations. It is not correct that subtracting an ordinary right-continuous CDF at cell edges automatically preserves ties-upward for zero-width intervals. For a point mass exactly at a cell's right edge, `P(X <= right) - P(X <= left)` assigns the point to the cell on the left, contrary to the legacy half-open convention.

The API must distinguish:

- right-continuous point CDF: `P(X <= t)`;
- strict or left-limit point CDF: `P(X < t)`; and
- half-open cell mass `[left, right)`, computed as `P(X < right) - P(X < left)`.

This explicit convention reproduces the legacy ties-upward rule for point masses at cell boundaries.

### Qualified acceptance: missing-position reporting

The missing-position policy is a real specification requirement. However, the interval-store builder cannot report TE or synonymous resolution rates unless those external lists are passed to it, and those lists are not inputs to the all-SNP store build. Resolution reporting therefore belongs in a separate validation command and in the downstream TE and synonymous commands.

Rates must be measured against the union catalog across all posterior draws, not inferred from one draw. Unresolved positions and resolved-but-ineligible positions must be reported separately.

### Qualification: cause of the previous failed run

The dense representation, remote accumulator placement, high memory use, and Python-level extraction are all scalability problems. The previous job was manually cancelled, so this plan does not attribute its failure to one bottleneck alone. The new design addresses all four rather than relying on a single causal explanation.

## 0. Correct and validate chromosome coordinates

This is a prerequisite for both the old and new formats.

The ten chromosomes were concatenated with one unused base between consecutive chromosomes. The two-column source table accumulates chromosome lengths without those gaps and is therefore wrong for chromosomes 2–10.

The corrected table currently exists at:

```text
/quobyte/jrigrp/jri/projects/normalizeTEs/chrom_offsets.combined.txt
```

Before interval-store implementation:

1. Validate the three-column table structurally.
2. Confirm `offset + length <= next_offset` for adjacent chromosomes.
3. Confirm the final chromosome does not extend beyond `sequence_length`.
4. Re-measure representative-draw resolution rates for the TE and synonymous position files to validate the offset correction.
5. Test chromosome-end, gap-base, and next-chromosome-start round trips.
6. Record the validated table and its provenance in store metadata.

Final external-list resolution rates must later be measured against the completed union catalog across all draws; a single representative draw is sufficient to validate the offset shift but not to determine the final missing-position rate.

No workflow should use the naive `chrom_offsets.txt` for these combined ARGs.

### Unsafe legacy launchers

`run_combined_age_store.sbatch` in this repository and the external `sampling.sh` currently reference the naive offset table. They must not be submitted. Because the dense all-SNP build is also infeasible, the preferred action is to retire or disable these launchers rather than merely point them at the corrected offsets.

Deleting the repository launcher or modifying the external script requires explicit user authorization and is not part of implementing the interval-store code. Until that cleanup is authorized, documentation and handoff notes must identify both scripts as unsafe. Dense stores built with the naive offsets are invalid for real-data coordinate lookup and cannot serve as production equivalence references.

## 1. Define the interval-store format

Create a new format rather than changing the existing dense-store schema in place.

```text
age_interval_store/
├── metadata.json
├── positions.npy
├── offsets.npy
├── below.npy
├── above.npy
├── draw_id.npy
├── status.npy
├── present_draw_count.npy
├── missing_draw_count.npy
├── usable_draw_count.npy
├── usable_interval_count.npy
└── skipped_root_count.npy
```

### Core arrays

- `positions.npy`, `float64`, shape `(n_snps,)`
  - Sorted global SNP coordinates.
- `offsets.npy`, `uint64`, shape `(n_snps + 1,)`
  - SNP `i` owns records `offsets[i]:offsets[i + 1]`.
  - `offsets[0] == 0` and `offsets[-1] == n_intervals`.
- `below.npy` and `above.npy`, `float64` by default, shape `(n_intervals,)`
  - Original mutation-node and covering-edge-parent ages.
- `draw_id.npy`, `uint8` for at most 255 draws and `uint16` otherwise
  - Posterior draw associated with each interval.
- `status.npy`, packed two-bit values, logical shape `(n_draws, n_snps)`
  - `0`: absent from the draw.
  - `1`: present but without a usable interval.
  - `2`: present with one or more usable intervals.
  - `3`: reserved for future schema use.
  - The physical layout is draw-major so construction writes one contiguous packed row per draw. The API may transpose selected logical views for callers.
  - Four logical statuses are packed into each `uint8`: `s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)`.
  - The physical byte shape is `(n_draws, ceil(n_snps / 4))`; unused high slots in the final byte must be zero.
- Count arrays, `uint32`, shape `(n_snps,)`
  - Permit fast eligibility filtering and consistency validation.

`metadata.json` must include the schema version, endpoint dtype, input draw paths, draw identifiers, chromosome table, coordinate convention, extraction policies, array shapes, and creation command.

Because NumPy has no two-bit equivalent of `packbits`, implement small arithmetic pack and unpack helpers. Test all four values, partial final bytes, selected-row decoding, and a full pack/unpack round trip. Cross-check decoded statuses against aggregate count arrays during deep validation.

## 2. Preserve the current statistical semantics

For an interval with lower age \(L\) and upper age \(U\), the conditional mutation-age CDF at age \(t\) is:

$$
F(t)=
\begin{cases}
0, & t < L \\
(t-L)/(U-L), & L \le t < U \\
1, & t \ge U
\end{cases}
$$

The generic reconstruction helper retains defined behavior for \(L=U\) so it can reproduce legacy synthetic fixtures. A production interval extracted from a valid tree sequence must satisfy \(U>L\), because tskit requires an edge parent to be strictly older than its child. Equality in a built interval store is therefore a corruption signal.

The current dense builder normalizes the intervals within a draw and then multiplies by the number of intervals in that draw. Each interval therefore contributes unit mass to the final pooled distribution. Reconstructing a SNP CDF by averaging its stored intervals preserves this equal-per-interval behavior.

The reconstruction code must also preserve:

- zero-width point-mass behavior;
- nearest-bin assignment with ties upward when reproducing the legacy grid;
- root-skipping and missing-draw policies;
- the minimum usable-draw eligibility threshold; and
- terminal CDF normalization.

Because draw IDs are stored, a future caller may request equal-per-draw weighting without changing the underlying store.

### Point-CDF and legacy-grid conventions

The read API must expose the side convention explicitly:

```python
store.cdf_at(rows, points, side="right")  # P(X <= t)
store.cdf_at(rows, points, side="left")   # P(X < t)
```

Boundary CDFs and Wasserstein calculations use the convention documented by their caller, normally the right-continuous CDF. To reproduce the legacy binned distribution with cells `[left, right)`, compute:

```python
cell_mass = cdf_at(right_edges, side="left") - cdf_at(left_edges, side="left")
```

For production intervals the two side conventions differ only at endpoints of zero probability. For fixture-only zero-width point masses, the explicit left-limit evaluation ensures that a point exactly on a right edge belongs to the following cell, matching the existing ties-upward behavior. The convention remains documented and tested even though valid ARG extraction cannot produce a zero-width interval.

## 3. Build the global SNP catalog

Do not use a Python `set` containing tens of millions of float objects.

Instead:

1. Read each draw's sorted numeric site-position array.
2. Validate finite, integral, strictly increasing positions.
3. Merge differing catalogs using sorted numeric union operations.
4. Write the final catalog to `positions.npy`.

If all draws have identical site catalogs, reuse the first catalog. Every draw still needs to be inspected to establish that identity, so this is a memory simplification rather than an assumed runtime shortcut.

The catalog pass should avoid loading unrelated tree-sequence tables when the TSZ/Zarr representation permits reading only the site-position array. Selective access to `sites/position` is an explicit Gate 2 benchmark item because it could eliminate most of the catalog pass's decompression cost. If selective access is unsupported or unreliable, use ordinary `tszip.load` and record its measured cost.

This phase also validates sequence length and the corrected chromosome table. It does not inspect maximum node age or create age bins.

## 4. Vectorized interval extraction

After the global catalog exists, read each TSZ once for interval extraction.

For one draw:

1. Map site-table rows to global SNP rows with vectorized `searchsorted`.
2. Map every mutation to its site's coordinate and global SNP row.
3. Validate that edge `left` and `right` coordinates are finite integers.
4. Check the signed-64-bit composite-key overflow bound.
5. Find each mutation node's covering parent edge with the tested composite-key lookup, or its vectorized grouped-search fallback.
6. Obtain `below` from the mutation-node time.
7. Obtain `above` from the covering edge's parent-node time.
8. Treat mutations without a covering child edge as root-skipped.
9. Update count arrays with vectorized reductions.
10. Update one contiguous draw-major packed-status row.
11. Partition usable interval records into row-range buckets on node-local scratch.

The extraction implementation must not call `ts.trees()`, construct a Python `Site` object for every site, or construct a Python `Mutation` object for every mutation.

### Count reductions

Measured on all 25,983,474 mutation rows from one real draw, `np.add.at` took 2.61 seconds, `np.bincount` took 0.14 seconds, and `np.unique` took 13.11 seconds. Use the measured methods below rather than a generic reduction.

Site rows are unique within a draw. Validate strict row uniqueness and update present counts directly:

```python
if np.any(site_rows[1:] <= site_rows[:-1]):
    raise ValueError("site rows must be unique and increasing within a draw")
present_draw_count[site_rows] += np.uint32(1)
```

Root-skipped and usable-interval rows can repeat because a site can carry multiple mutations. Reduce each one with `bincount`, processing one temporary at a time:

```python
def add_row_counts(target, rows, n_snps):
    counts64 = np.bincount(rows, minlength=n_snps)
    if counts64.max(initial=0) > np.iinfo(np.uint32).max:
        raise OverflowError("per-row count exceeds uint32")
    target += counts64.astype(np.uint32)

add_row_counts(skipped_root_count, root_mutation_rows, n_snps)
add_row_counts(usable_interval_count, usable_mutation_rows, n_snps)
```

The explicit conversion is required: NumPy does not permit in-place addition of the `int64` result from `bincount` into a `uint32` target. Release each `counts64` temporary before creating the next one so peak memory does not include multiple 25-million-element count arrays.

The mutation table is expected to be ordered by site, making `usable_mutation_rows` nondecreasing. Validate that property and deduplicate adjacent rows without sorting. Guard the empty case explicitly:

```python
if usable_mutation_rows.size:
    starts = np.r_[True, usable_mutation_rows[1:] != usable_mutation_rows[:-1]]
    usable_draw_count[usable_mutation_rows[starts]] += 1
```

If row ordering is not guaranteed, use `np.bincount(..., minlength=n_snps) > 0` as the fallback. Do not use `np.unique` or `np.add.at` for production-scale count reductions unless new measurements overturn the observed difference.

After all draws:

```python
missing_draw_count = n_draws - present_draw_count
```

Large temporary arrays must be processed in blocks where needed; “vectorized” must not mean allocating several unbounded copies of every 74-million-edge column.

## 5. Bucketed scratch and contiguous assembly

### Extraction-time buckets

Partition records by global SNP row range. Each scratch record contains:

```text
row:      uint32
below:    float64 or float32
above:    float64 or float32
draw_id:  uint8 or uint16
```

Append each draw's records to the appropriate bucket in row order. Use a bounded number of buckets, such as 25–100, so file-descriptor usage remains small and each bucket can later be processed within memory limits.

Use equal global-row ranges for the initial implementation. The measured draw has 1.038 mutations per site and does not suggest enough density skew to justify a preliminary counting pass solely for balanced buckets. The builder must nevertheless record bucket counts and stop with a clear resource diagnostic if a pathological region makes a bucket exceed its configured memory bound.

Choose the bucket count so that:

```text
largest_bucket_intervals × packed_record_size × sort_workspace_factor <= memory_budget
```

Use a conservative sort-workspace factor of approximately three until measured otherwise. With 100 balanced buckets, 1.95 billion intervals, and 21-byte `float64` records, the estimate is approximately 1.2 GB for a bucket plus sort workspace.

With approximately 1.95 billion intervals, packed scratch records require approximately:

- 41 GB with `float64` endpoints and `uint8` draw IDs; or
- 25 GB with `float32` endpoints and `uint8` draw IDs.

Structured-array padding may increase these figures, so the implementation must use an explicitly packed on-disk record dtype or separate bucket columns.

### Final assembly

After extraction:

1. Compute `offsets` from `usable_interval_count`.
2. Allocate the final endpoint and draw-ID arrays.
3. Read one bucket at a time.
4. Stable-sort records by `row` only.
5. Verify the bucket's per-row record counts against `offsets`.
6. Write records contiguously into their final row ranges.
7. Delete the bucket only after its final output has been flushed and verified.

For a bucket covering rows `[r0, r1)`, its complete destination is exactly `offsets[r0]:offsets[r1]`. Stable-sorted records must fill that slice without gaps or overflow, and per-row run lengths must equal `np.diff(offsets[r0:r1 + 1])`. This bucket-to-offset invariant is checked before the bucket is removed.

Draws are extracted and appended in draw-table order, while mutations within a draw arrive in mutation-table order. A stable row sort therefore preserves `draw_id` order and mutation-table order within each `(row, draw)` group. This provides deterministic, byte-reproducible record ordering without an additional draw-ID sort key.

This phase does not reopen any TSZ file.

Direct per-SNP cursor writes are not part of the production design. Although the previous dense failure cannot be attributed solely to write order, cursor writes would repeatedly touch widely separated output regions across draws and are poorly suited to Quobyte.

## 6. Atomic publication and cleanup

Build into a unique temporary output directory beside the requested final store. Keep bucket files on node-local scratch.

On success:

1. Flush and close all arrays.
2. Write complete metadata inside the temporary store.
3. Run structural validation against the arrays and metadata together.
4. Atomically rename the temporary output directory to the requested final path.

On a Python exception, remove only temporary paths created by that invocation. Record scratch and output paths in logs so interrupted SLURM jobs can be cleaned up safely. Signal handling should attempt cleanup but must never remove a pre-existing path.

## 7. Read-only interval-store API

Create an `SNPAgeIntervalDataset` class with methods such as:

```python
store.intervals(rows)
store.mean_ages(rows)
store.cdf_at(rows, points, side="right", weighting="interval")
store.cell_masses(rows, edges, weighting="interval")
store.boundary_cdfs(rows, boundaries, side="right", weighting="interval")
store.resolve_native_positions(chromosomes, positions)
```

`cdf_at()` is the single point-evaluation primitive. `cell_masses()` derives half-open bin masses with left-limit CDF values at cell edges. The API must not overload one `cdf()` method with both point evaluation and cell integration.

### Boundary CDF evaluation

For synonymous matching, evaluate each candidate only at the approximately 19 target-quantile boundaries:

1. Read intervals for a block of SNP rows.
2. Evaluate the interval CDF formula at the requested boundaries.
3. Reduce interval contributions by SNP.
4. Divide by the appropriate interval or draw weight.
5. Return a candidate-by-boundary matrix.

For the current 485,671 synonymous candidates and 20 strata, the cached candidate-weight matrix uses approximately 39 MB as `float32`. Only its source changes; the existing sampler's weight-matrix strategy remains appropriate.

### Candidate interval access strategy

“Read intervals for a block of SNP rows” does not by itself specify the I/O pattern. The synonymous candidates occupy approximately 1.9% of store rows and are scattered genome-wide. With `float64` endpoints, one average SNP contributes roughly 624 bytes to each endpoint array. A naive gather can therefore issue approximately 486,000 small reads per array on Quobyte. Conversely, a full contiguous scan reads approximately 31.2 GB of endpoints while discarding about 98% of rows.

Gate 3 must compare four strategies using the real synonymous candidate list:

1. **Direct scattered gathering:** read each requested row's interval slice.
2. **Coalesced gathering:** sort candidate rows and merge nearby rows into larger slabs using a configurable maximum gap.
3. **Full sequential scan:** stream endpoint and offset arrays in row order and retain only candidate rows.
4. **Candidate cache:** perform one sequential scan, write a compact candidate-only interval store on node-local scratch, and use that cache for boundary calculations and repeated downstream access.

The candidate cache is the leading hypothesis because one sequential read can produce a reusable endpoint subset of roughly 600 MB, but it is not selected without measurement. Benchmark wall time, bytes read, read-operation count, system time, cache construction cost, and repeated-access time. The chosen strategy and its parameters must be recorded in downstream metadata.

The API should accept an access strategy such as `auto`, `gather`, `coalesced`, `scan`, or `cache`. `auto` must use benchmark-derived thresholds rather than an undocumented heuristic.

## 8. Adapt the downstream workflow

### TE target construction

For the selected TE SNPs:

1. Resolve their store rows using the corrected chromosome table.
2. Retrieve their interval records.
3. Materialize their CDFs on the requested analysis grid.
4. Average the SNP CDFs to construct the target.
5. Bootstrap the TE SNPs as before.

The TE subset is small enough for temporary dense CDFs.

### Missing and ineligible TE positions

`te_age_target.py` should accept:

```text
--missing-position-policy {error,drop}
```

The default is `error`. Under `drop`, the command must write the requested, resolved, unresolved, eligible, and ineligible counts and the exact dropped coordinates. The effective TE set size `X` becomes the number of resolved eligible TE positions; this reduced value must be recorded and used for matched-set size. The command must fail if no eligible TE positions remain.

Unresolved positions and positions that resolve to a store row but fail the usable-draw threshold are distinct categories and must never be combined silently.

### Synonymous candidate matching

For synonymous candidates:

1. Evaluate CDF values only at target-quantile boundaries.
2. Convert boundary differences into stratum weights.
3. Cache the candidate-by-stratum weight matrix.
4. Sample candidate sets.
5. Materialize a dense age grid only for aggregate proposed or accepted set CDFs.
6. Calculate Wasserstein distances on that aggregate grid.

`sample_age_matched_syn.py` should expose the same `--missing-position-policy {error,drop}` choice. The policy default requires an explicit scientific decision before implementation: `error` maximizes strictness, whereas `drop` is often appropriate for a large candidate pool. Regardless of the default, the command must record requested, resolved, unresolved, eligible, and ineligible counts and write the excluded coordinates. Dropping synonymous candidates changes the available pool but not the TE-derived matched-set size `X`.

An invariant of the new design is that a full dense age grid is never materialized per candidate or for all SNPs.

### Resolution reporting scope

The all-SNP interval builder does not consume TE or synonymous lists and therefore does not report their resolution rates. Add a lightweight validation command, or a shared dry-run mode in the downstream commands, that resolves external lists against the completed union catalog before target construction or matching.

## 9. Validation strategy

### Structural validation

Check that:

- positions are finite, integral, and strictly increasing;
- offsets are monotone, start at zero, and terminate at `n_intervals`;
- endpoint and draw arrays have matching lengths;
- endpoint ages are finite, nonnegative, and satisfy `above > below`;
- draw IDs lie within the recorded draw table;
- record counts implied by offsets match `usable_interval_count`;
- `present_draw_count + missing_draw_count == n_draws`;
- `usable_draw_count <= present_draw_count`;
- packed statuses use only values 0–2, leave unused final-byte slots zero, and agree with aggregate counts; and
- eligibility equals `valid & (usable_draw_count >= required_usable_draws)`.

### Extraction correctness

On small fixtures, compare vectorized parent lookup with `tree.parent(mutation.node)` for every mutation. Include:

- a mutation on a node whose first edge begins after the mutation position;
- a mutation in a gap between two edges belonging to the same child;
- a mutation on a node that is a root across the entire sequence;
- mutations at left-inclusive and right-exclusive edge boundaries;
- rejection or fallback for non-integral edge endpoints; and
- rejection or fallback when the composite key would overflow.

### Real-draw parent audit

Fixture tests cannot reproduce the approximately 18.9 million marginal trees and 74.3 million edges in a production draw. Before Gate 3, perform a reproducible audit on one real TSZ:

1. Use a recorded random seed.
2. Sample 10,000–100,000 mutations without replacement.
3. Stratify the sample between mutations classified as covered and root-skipped so both guard outcomes are exercised; include high node IDs and coordinates near edge boundaries where available.
4. For each sampled mutation, compute `tree = ts.at(position)` and `expected_parent = tree.parent(node)`.
5. Require exact agreement between the scalar result and the vectorized coverage classification.
6. For covered mutations, require exact parent-node equality and exact `below`/`above` equality with the source node times.
7. For predicted root-skipped mutations, require `expected_parent == tskit.NULL`.

This audit is a hard gate. Aggregate counts alone cannot detect a systematic guard error because misclassified intervals can leave all internal count identities consistent.

### Equivalence with the dense implementation

For small fixtures, compare:

- global and native positions;
- present, missing, usable, and root-skipped counts;
- interval records;
- reconstructed CDFs at several bin widths;
- mean ages;
- TE target CDFs and quantile boundaries;
- synonymous stratum weights; and
- Wasserstein distances.

Do not require universal bit-for-bit equality after `uint16` quantization. Compare pre-quantization values with explicit tolerances and initially require quantized values to differ by no more than a small empirically justified bound. Determine that bound from tests rather than assuming it is always one unit, especially when `float32` endpoints are selected.

### Required edge cases

- sites missing from some draws;
- present sites with no usable interval;
- sites below the minimum usable-draw threshold;
- mutations above roots;
- a draw with zero usable mutations, exercising the empty adjacent-dedup path;
- rejection of zero-width intervals during store validation;
- fixture-only zero-width point masses in the generic legacy-grid helper;
- multiple mutations at one site in one draw;
- differing site catalogs between draws;
- chromosome ends, gap bases, and chromosome starts;
- unresolved external TE and synonymous positions under both `error` and `drop` policies;
- positions that resolve but are ineligible, reported separately from unresolved positions;
- a drop-policy TE run in which the effective matched-set size decreases;
- interrupted builds and atomic cleanup; and
- more than 255 draws, requiring `uint16` draw IDs.

## 10. Storage estimates

Assume approximately 1.95 billion usable interval records.

| Component | Default `float64` | Optional `float32` |
|---|---:|---:|
| `below.npy` | 15.6 GB | 7.8 GB |
| `above.npy` | 15.6 GB | 7.8 GB |
| `draw_id.npy`, `uint8` | 1.95 GB | 1.95 GB |
| Packed statuses | 0.47 GB | 0.47 GB |
| Positions and offsets | 0.4 GB | 0.4 GB |
| Count arrays and metadata | ~0.5 GB | ~0.5 GB |
| **Final store** | **~34.5 GB** | **~18.9 GB** |

Expected node-local bucket scratch is approximately 41 GB for `float64` or 25 GB for `float32`, plus sorting workspace for one bucket. These figures must be validated with an explicitly packed record layout and a representative draw.

## 11. Benchmark gates

Do not proceed directly from unit tests to a full 75-draw build.

### Gate 1: synthetic correctness

- Validate grouped parent lookup against tskit tree traversal.
- Assert `edge_key.dtype == query_key.dtype == np.int64` before every composite-key search.
- Test composite-key construction with node IDs and positions whose exact keys exceed `2**53` and whose `int32` products would wrap.
- Round-trip two-bit status packing for complete and partial final bytes.
- Validate all schema invariants and reconstruction behavior.

### Gate 2: one real TSZ

Measure:

- selective `sites/position` access from the TSZ/Zarr store versus full `tszip.load` for catalog construction;
- TSZ load time and peak RSS;
- stable composite-key edge-sort time, temporary storage, and peak memory;
- memory for the key, permutation, reordered child/right/parent columns, and sorting workspace as separate measurements;
- mutation-parent lookup time;
- fallback grouped-search time on a representative subset;
- bucket write throughput and actual bytes per record;
- bucket-size distribution and maximum sort workspace;
- root-skipped and usable interval counts; and
- `float32` versus `float64` reconstruction error.

After the performance measurements, run the stratified real-draw parent audit described in §9. Gate 3 must not begin unless every sampled mutation agrees with scalar tskit traversal.

### Gate 3: restricted multi-draw build

Use several posterior draws and a restricted chromosome or row range to test:

- cross-draw status and counts;
- bucket merging;
- final row contiguity;
- downstream TE target and synonymous matching;
- direct, coalesced, sequential-scan, and candidate-cache access using the realistic scattered synonymous list;
- access-strategy wall time, bytes read, read-operation count, system time, and repeat-read performance;
- candidate-cache size, construction time, and node-local scratch behavior; and
- restart and cleanup behavior.

Because a restricted chromosome may not reproduce genome-wide scattering, the access-strategy portion of Gate 3 may use the completed interval arrays from a representative multi-draw subset across the full coordinate range. Preserve realistic candidate density and spacing even when the number of posterior draws is reduced.

### Gate 4: projected full-run review

Project full runtime, RAM, scratch, and final storage from measured results. Prepare a SLURM command and review it with the user. Do not submit any job without explicit authorization.

## 12. Delivery sequence

1. Validate the corrected chromosome-offset table and resolution rates.
2. With explicit authorization, retire or disable launchers that still use the naive offsets.
3. Specify the interval-store schema and metadata contract.
4. Prototype and test exact grouped mutation-parent lookup.
5. Implement the global SNP catalog builder.
6. Implement vectorized extraction and bucket writing.
7. Implement bucket merge and atomic final publication.
8. Implement the read-only interval-store API.
9. Implement and validate direct, coalesced, sequential-scan, and candidate-cache access strategies.
10. Implement external-position resolution reporting and finalize the synonymous default missing-position policy.
11. Adapt TE target construction.
12. Adapt synonymous candidate matching.
13. Complete equivalence and edge-case tests.
14. Run Gate 2, including the required real-draw parent audit.
15. Run Gate 3 and select the candidate-access strategy from measurements.
16. Complete the projected full-run review.
17. Document measured resource requirements and prepare, but do not submit, a full-run command.

The existing dense builder remains available during migration. No old format or output should be deleted until the interval store passes correctness and downstream equivalence tests.
