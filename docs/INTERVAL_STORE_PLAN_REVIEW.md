# Review: `INTERVAL_STORE_IMPLEMENTATION_PLAN.md`

**Status:** reviewing revision 4. Earlier rounds are retained only as a
disposition record (§C). Actionable content is §B.

Reviewed against `normalize_tes/build_snp_age_store.py`, `normalize_tes/snp_age_dataset.py`,
`normalize_tes/snp_age_distribution.py`, and `slurm/run_combined_age_store.sbatch`, with measurements
from `run.combined.100.tsz` in
`/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined` and from
the project `normalizeTE` environment (numpy 2.5.2, tskit 1.0.3).

**Verdict on revision 4.** Every blocking defect from round 3 is fixed, and the
composite-key snippet in §50–95 is now correct as written — the integrality
checks precede the casts, both operands are `int64`, the dtype assertions are in
place, and the reordered columns are materialized once. The plan is ready to
implement.

Four items remain, none blocking. §B1 is a test-coverage gap worth closing before
Gate 3; §B2 and §B3 are measured performance and typing details; §B4 is an
unaddressed I/O design question in the downstream read path that should be
settled by measurement rather than guessed at.

---

## A. Corrections to earlier rounds of this review

Recorded so the disposition table below is readable. All four stand.

- **A1** (round 1): the `np.searchsorted(left, ...)` sketch searched an array
  sorted only within each child group. Wrong; superseded.
- **A2** (round 1): the "~15 s to lexsort 74 M edges" figure was an unfounded
  estimate presented as a planning input.
- **A3** (round 2): differencing a *right-continuous* CDF at cell edges does not
  reproduce ties-upward. Plan §137 is correct; the left-limit form in §143 and
  §273 is the fix.
- **A4/A5** (round 2): resolution reporting belongs in the downstream commands,
  not the all-SNP builder, and the chr1 rates validate the offset shift only —
  not the final missing-position rate.

Two claims drafted for this round were checked and withdrawn before publication:

- `np.cumsum` on a `uint32` count array does **not** risk overflow — NumPy
  promotes the accumulator to `uint64` automatically. Verified: `np.cumsum` of
  `uint32` returns dtype `uint64` and produces correct values past 2³². No change
  needed at §379.
- `np.add.at` is **not** slower than `np.unique`. Measured below: `np.unique` is
  the slowest of the three by a wide margin, so the plan's warning at §333 is
  correct as written.

---

## B. Open issues

### B1. Extraction correctness is validated only on synthetic fixtures

Plan §499 scopes the `tree.parent()` cross-check to "small fixtures", and its six
listed cases are well chosen. But hand-built fixtures cannot reproduce what these
ARGs actually look like: 18.9 M marginal trees, 74.3 M edges, 10.9 M nodes, and a
node-time range spanning seven orders of magnitude. A systematic error in the
guard logic — for instance one that only manifests for children with many
disjoint edge intervals, or near the 1 bp inter-chromosome gaps — would pass every
fixture and still corrupt the store.

Add a real-data cross-check to Gate 2: sample 10⁴–10⁵ mutations uniformly from one
draw, resolve each with `ts.at(position).parent(node)`, and require exact
agreement with the vectorized result for both the parent id and the `covered`
flag. Sampling rather than exhaustive checking keeps this to minutes.

This matters more than usual because §B-adjacent invariants are weak: nothing in
§484–495 constrains how mutations split between `skipped_root_count` and
`usable_interval_count`. If the guards misfire, the counts stay internally
consistent and the store validates cleanly. Gate 2 already measures both counts
(§580) — state what they should be compared against, and treat a root-skip
fraction far from the sampled ground truth as a gate failure.

### B2. `np.add.at` is the wrong primitive for the count reductions

Plan §320–322 names `np.add.at` three times. It is the unbuffered path and is
substantially slower than the alternatives. Measured in this environment on
25,983,474 mutation rows against a 25,030,335-element counter:

| Primitive | Time | Relative |
|---|---:|---:|
| `np.add.at` | 2.61 s | 1× |
| `np.bincount` | 0.14 s | **18× faster** |
| `np.unique` | 13.11 s | 5× slower than `add.at` |

Extrapolated across three arrays and 75 draws: **~10 min with `np.add.at`, ~0.5
min with `np.bincount`**. Not fatal, but free to avoid.

Two refinements:

- `present_draw_count` is indexed by *site* rows, which tskit guarantees to be
  unique within a draw. Plain fancy-index `present_draw_count[site_rows] += 1` is
  correct there and faster still — `np.add.at` is only needed when indices repeat.
- `skipped_root_count` and `usable_interval_count` are indexed by *mutation* rows,
  which do repeat, so use `np.bincount(rows, minlength=n_snps)`.

The `np.unique` measurement confirms §333's warning was right, and by a larger
margin than the plan claims.

### B3. `np.bincount` returns `int64`; the count arrays are `uint32`

Following from §B2: `count_array += np.bincount(...)` raises rather than
silently truncating. Verified:

```
uint32 += int64  ->  UFuncTypeError: Cannot cast ufunc 'add' output from
                     dtype('int64') to dtype('uint32') with casting rule 'same_kind'
```

An explicit `.astype(np.uint32)` is required on the bincount result. Worth
stating alongside §320–322 so the fix for §B2 does not immediately fail. This is
a loud failure rather than a silent one, so it is a documentation point, not a
correctness risk.

### B4. The downstream read pattern is unspecified, and the obvious choice is likely the slow one

Plan §427 says "Read intervals for a block of SNP rows" without saying whether a
block is a **contiguous row range** (sequential I/O, read and discard) or a
**gathered set of scattered rows** (random I/O). For this workload the difference
is large, and the natural implementation is the wrong one.

The synonymous candidate set is 485,671 rows out of 25,030,335 — **1.9% density**,
scattered genome-wide. TE is 185,232 rows, 0.74%.

- **Gathered:** each row needs `below[offsets[r]:offsets[r+1]]`, roughly 78
  intervals × 8 B ≈ 624 B. That is ~486 K independent small reads from a 15.6 GB
  memory map on Quobyte. At 1 ms per read that is 8 minutes; at 10 ms, well over
  an hour.
- **Contiguous scan:** streaming all of `below` and `above` sequentially is
  31.2 GB. At a few hundred MB/s that is on the order of a minute or two, and the
  1.9% of records actually wanted are filtered in memory.

Sequential scanning is plausibly an order of magnitude faster despite reading ~50×
more data, because Quobyte penalizes small random reads far more than throughput.
The crossover depends on the filesystem, so this should be measured rather than
assumed — but the plan should at least name it as a decision.

Concretely: state in §427 that block reads are contiguous row ranges with
in-memory filtering, and add "boundary-CDF evaluation over a scattered 1–2%
candidate set: gathered versus contiguous-scan throughput" to Gate 3 (§585), which
is where a realistic candidate set first exists.

### B5. Minor

- `metadata.json` (§230) should record `n_intervals` explicitly rather than
  leaving it implied by `offsets[-1]`, so structural validation has an
  independent value to check the offsets array against.
- §356 should state that each `(draw, bucket)` partition is written as one
  contiguous block rather than record-at-a-time. At 100 buckets this is ~260 K
  records ≈ 5.5 MB per flush, which is a healthy write size; doing it per record
  across 100 open descriptors would not be.

---

## C. Disposition of earlier findings

| Finding | Round | Status in revision 4 |
|---|---|---|
| Chromosome offsets wrong | 1 | Accepted — §0, §19 |
| Remove per-site/per-mutation Python loops | 1 | Accepted — §20, §313 |
| One TSZ pass plus bucket merge | 1 | Accepted — §21 |
| Bucketed assembly, not cursor writes | 1 | Accepted — §22, §393 |
| `missing_draw_count.npy` absent | 1 | Accepted — §23, §202 |
| `boundary_cdfs()` for candidate weights | 1 | Accepted — §423–433 |
| "Exactly" too strong for equivalence | 1 | Accepted — §521 |
| `float32` endpoints | 1 | Declined, reasonably — §107–116 |
| Drop `draw_id.npy` | 1 | Declined, reasonably — §118–127 |
| Drop `status.npy` | 1 | Declined; reweighting justification removed — §133 |
| Position restriction | 1 | Withdrawn — all ~25 M SNPs required |
| Parent-lookup sketch | 1 | Rejected, correctly — §A1 |
| Edge-sort runtime estimate | 1 | Rejected, correctly — §A2 |
| `cdf()` conflates point and cell semantics | 2 | Accepted; my mechanism corrected — §A3, plan §135–145 |
| Bucket sizing ordering dependency | 2 | Accepted — §358–366 |
| Stable sort and tiebreak semantics | 2 | Accepted — §382, §389 |
| Missing-position policy | 2 | Accepted; scope relocated — §A4, plan §449–478 |
| `np.unique` cost | 2 | Accepted — §333, and confirmed by measurement in §B2 |
| Selective `sites/position` access | 2 | Accepted — §291, §572 |
| **`int32` composite-key overflow** | 3 | **Fixed** — §56, §73, asserts at §81–82 |
| **`float64` key inexact above 2⁵³** | 3 | **Fixed** — §74, integrality checked before cast at §61–69 |
| Reordered columns materialized twice | 3 | Fixed — §84–86, §101 |
| Zero-width intervals impossible in production | 3 | Accepted — §247, §276, §489, §529–530 |
| Empty-array guard in adjacent dedup | 3 | Fixed — §328 |
| Unsafe legacy launchers | 3 | Accepted — §182–186, §600 |
| Status packing and bucket-offset invariants | 3 | Accepted — §225–226, §232, §387, §494 |

---

## D. Measured reference data

### D1. One posterior draw (`run.combined.100.tsz`)

| Quantity | Value |
|---|---:|
| `num_sites` | 25,030,335 |
| `num_mutations` | 25,983,474 (1.038 per site) |
| `num_trees` | 18,901,585 |
| `num_nodes` | 10,860,599 |
| `num_edges` | 74,260,870 |
| `num_samples` | 26 |
| `sequence_length` | 2,131,846,815 |
| max edge-parent node time | 19,464,592.3 generations |
| tszip load time | ~36 s |
| top-level metadata keys | `kept_intervals`, `mu_position`, `mu_rate` — **no `chrom_offsets`** |

Implied interval count across 75 draws: ≈ **1.95 × 10⁹**, before subtracting
root-skipped mutations.

### D2. Count-reduction primitives

25,983,474 rows into a 25,030,335-element counter, numpy 2.5.2:

| Primitive | Time | Per full build (3 arrays × 75 draws) |
|---|---:|---:|
| `np.add.at` | 2.61 s | ~10 min |
| `np.bincount` | 0.14 s | ~0.5 min |
| `np.unique` | 13.11 s | ~49 min |

### D3. Integer-arithmetic hazards (the round-3 defects)

Both now fixed in the plan; retained because the Gate 1 tests at §563–564 must
reproduce them.

```
np.array([10_000_000], dtype=np.int32) * 2_131_846_816
  ->  dtype int32, value -1445654528          # silent wrap, no exception

child 10,860,598, position 2,131,698,489
  exact int64 key : 23153133397854457
  via float64     : 23153133397854456          # off by one
  2**53           :  9007199254740992
```

With `num_nodes = 10,860,599` and `S ≈ 2.13e9`, keys exceed 2⁵³ for any child id
above roughly 4.2 M — over half the nodes in these ARGs.

### D4. Chromosome offset evidence

Scanning offset deltas in −25…+25 against the ARG's site positions, using
`combined.all.chr.low.snp.pos.txt`:

| chrom | n queried | naive offset | best delta | hits at best | frac | hits at delta 0 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 75,482 | 0 | 0 | 63,359 | 0.839 | 63,359 |
| 2 | 60,770 | 308,452,471 | +1 | 49,962 | 0.822 | 4,214 |
| 3 | 54,363 | 552,127,662 | +2 | 41,307 | 0.760 | 2,362 |
| 4 | 45,196 | 790,145,429 | +3 | 34,833 | 0.771 | 2,425 |
| 5 | 56,164 | 1,040,475,889 | +4 | 46,004 | 0.819 | 2,220 |
| 6 | 39,558 | 1,266,829,338 | +5 | 31,872 | 0.806 | 1,299 |
| 7 | 38,877 | 1,448,186,572 | +6 | 31,746 | 0.817 | 2,116 |
| 8 | 41,897 | 1,633,995,488 | +7 | 34,561 | 0.825 | 2,087 |
| 9 | 34,968 | 1,816,406,690 | +8 | 27,136 | 0.776 | 1,057 |
| 10 | 38,396 | 1,979,411,434 | +9 | 27,884 | 0.726 | 2,207 |

Every consecutive gap is exactly 1, and `sequence_length` (2,131,846,815) is
exactly 10 more than the sum of chromosome lengths (2,131,846,805). Across the
full TE + synonymous input the naive table resolves 113,118 of 670,813 positions.

The `delta 0` column is why this fails silently: at one site per ~85 bp a shifted
coordinate frequently lands on a *different real variant*, giving 1,000–4,000
spurious matches per chromosome rather than a `KeyError`.

The corrected table is `chrom_offsets.combined.txt`. It satisfies
`offset + length <= next offset`, satisfies `offset + length <= sequence_length`
for the last chromosome (2,131,846,814 ≤ 2,131,846,815), and round-trips through
`native_to_global()` / `rows_to_native()` — chr1 base 308,452,471 → global
308,452,471 → back; chr2 base 1 → global 308,452,473 → back; global 308,452,472
is the unused gap base.

Per §A5 these rates come from one draw and validate the shift only.

---

## E. Environment notes

- **numpy 2.5.2, tskit 1.0.3.** NumPy 2 weak-scalar promotion is what made the
  round-3 `int32` defect a silent wrap rather than an exception.
- **`$SLURM_TMPDIR` is not set on Farm.** The cancelled dense run passed
  `--scratch-dir "$SLURM_TMPDIR"`, which argparse received as `""`; `Path("")` is
  `Path(".")`, so the accumulator went to the Quobyte working directory instead of
  node-local disk. Farm uses `job_container/tmpfs` with `TmpFS = /local/scratch`,
  and `$TMPDIR` resolves to a per-job directory on a 2.9 TB local disk — ample for
  the 41 GB (`float64`) or 25 GB (`float32`) of bucket files.
- **The ARGs carry no `chrom_offsets` metadata**, so `--chrom-offsets` is
  mandatory and `_warn_metadata_conflict` never fires. The offsets file is the
  sole source of truth, which is why the error in §D4 went undetected.
- Head-node Python needs `OPENBLAS_NUM_THREADS=1`; OpenBLAS otherwise tries to
  spawn 128 threads and fails to import numpy at all.
- Any dense store built before the offset fix is invalid on real data and must not
  serve as a production equivalence reference — fixtures only.
