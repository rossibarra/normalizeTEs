# normalizeTE — code review, round 3: the remaining issue

**Reviewed at:** `4a57f9d`. **Test suite:** 37 passed.

Rounds 1 and 2 ([CODE_REVIEW.md](CODE_REVIEW.md),
[CODE_REVIEW_ROUND2.md](CODE_REVIEW_ROUND2.md)) are fully resolved, as is the round-1
simplification list except for one item. **`O-j` — the sampler's blockwise weight machinery —
is the only substantive issue left**, and it is the one worth measuring on Quobyte.

Two things assessed in this pass and found *not* to be problems, recorded so they are not
re-litigated:

- The builder's `pdf_accumulator` writes are **sequential**, not random — row indices increase
  strictly within a draw (verified). Its cost is `n_draws` streaming passes, so the only
  question is whether `n_snps × n_age_bins × 4` bytes fits in the build node's RAM. No
  Quobyte benchmark needed. (Round 2 §3 carries the numbers and the correction.)
- No functional defects. Nothing below changes results; this is purely I/O cost.

---

## O-j — The sampler re-reads candidate weights from the store on every proposal

### The defect

`_BlockCache` is constructed **inside** `draw_stratified_set` —
[normalize_tes/sample_age_matched_syn.py:191](../normalize_tes/sample_age_matched_syn.py#L191):

```python
cache = _BlockCache(store, index, cache_blocks)
```

`draw_stratified_set` is called once per proposal from the loop at
[normalize_tes/sample_age_matched_syn.py:248](../normalize_tes/sample_age_matched_syn.py#L248). The cache is therefore built
and discarded for every proposal: **nothing is reused across proposals, and each one re-reads
the store from cold.**

Compounding it, the default `cache_blocks=8`
([line 171](../normalize_tes/sample_age_matched_syn.py#L171), [line 237](../normalize_tes/sample_age_matched_syn.py#L237),
`_BlockCache.__init__` at [line 136](../normalize_tes/sample_age_matched_syn.py#L136)) is smaller than the block
count in most realistic configurations, so LRU also thrashes *within* a single proposal.

Block choice is proportional to per-interval mass, which is spread roughly evenly across the
genome, so selection is effectively uniform and the cache has almost no locality to exploit.

### Projected read volume

Each miss reads a slab of `block_snps × ~20 boundary columns × 2 bytes` = **10 MiB**. With
`block_snps = 250,000` and a 2M-candidate pool:

| store rows | blocks | P=200 | P=2,000 | P=20,000 |
|---|---|---|---|---|
| 2M (dense) | ~8 | 15 GiB | 149 GiB | 1,490 GiB |
| 20M (sparse candidates) | ~80 | 149 GiB | 1,490 GiB | 14,901 GiB |

These are **lower bounds** for the 80-block row: they count each block once per proposal and
ignore intra-proposal LRU thrashing.

The governing variable is `P`, the proposal count — which is rejection-sampling output and
cannot be predicted from the inputs. `--max-proposals` defaults to 100,000.

### Why the original justification does not hold

The module docstring and README justify this machinery as avoiding an
`N-candidates × N-strata` weight matrix in memory. Priced as I/O rather than RAM, the trade is
heavily negative:

| candidates | `candidates × 20 × float32` | store I/O during sampling |
|---|---|---|
| 250,000 | 19 MiB | read once |
| 2,000,000 | 153 MiB | read once |
| 20,000,000 | 1,526 MiB | read once |

At the documented pool size the matrix costs **153 MiB, read once**, and eliminates all
further store reads. The current design avoids that allocation at a cost of hundreds of GiB to
tens of TiB of repeated reads.

### Two fixes

1. **Hoist the cache** out of `draw_stratified_set` into `generate_matches` and size it to hold
   all blocks. Converts per-proposal re-reads into a one-time cost and captures most of the
   benefit without touching the sampling algorithm. Low risk and no change to sampling
   semantics — the cache is a pure read-through of immutable store data. Worth doing regardless
   of whether fix 2 happens.
2. **Materialise the weight matrix** and delete `BlockWeightIndex`, `_BlockCache`, the
   hierarchical block-then-SNP draw, and the exhausted-block retry branch. Larger change, and
   it also removes the subtlest code in the repo. Best done with the existing sampler tests as
   a behavioural reference.

### What to measure on Quobyte

One realistic TE subset against the real candidate pool. Capture:

- `proposals` and `accepted_sets` from `metadata.json` → the acceptance rate and `P`.
- Rejection reasons from `diagnostics.csv` — confirm they are `wasserstein_threshold` rather
  than `sampling:` failures.
- Block count: `max(ceil(n_candidates / block_snps), ceil(coordinate_span / block_snps))`.

`P` and the block count turn the table above into a wall-clock figure:

- `P` near `accepted_sets` (~100–200) → tolerable; apply fix 1 and move on.
- `P` in the thousands or more → this is the bottleneck; fix 2 before production runs.

Worth capturing either way, since `P` is the single parameter the design's cost is most
sensitive to.

---

## Minor, separate: the accumulator cannot be put on node-local scratch

Not the issue above, but open and cheap. The README recommends node-local scratch for the
builder's accumulator:

> *"Each posterior draw sweeps this accumulator in genomic order, so node-local scratch is
> preferable to Quobyte when available."*

The location is hardcoded to the output's parent —
[normalize_tes/build_snp_age_store.py:151](../normalize_tes/build_snp_age_store.py#L151):

```python
temp = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
```

So if the store is written to Quobyte, the accumulator is on Quobyte. There is no
`--scratch-dir`, and neither `$TMPDIR` nor `$SLURM_TMPDIR` is consulted — the documented
recommendation cannot currently be followed. (The `$SLURM_TMPDIR` guidance at README:258
concerns the *sampler* reading store arrays, not this file.)

Fix: add `--scratch-dir`, defaulting to `output_dir.parent` to preserve current behaviour, and
use it for the accumulator only. The final arrays must stay where `os.replace` can atomically
move them into place, so the whole temp directory should not follow the scratch path.

---

## Appendix — reproducing the measurements

**Cache lifetime.** `grep -n "_BlockCache(" normalize_tes/sample_age_matched_syn.py` — the single
construction site is inside `draw_stratified_set`, which the proposal loop calls once per
proposal.

**Read volume.** Monkeypatch `SNPAgeDataset.read_boundary_cdfs` to accumulate
`len(age_indices) * (stop - start) * 2` bytes and a call count, then run the sampler. On a
400-SNP / 8-block toy store with `cache_blocks=8` — the best case, cache ≥ block count — the
observed cost was still 0.72 calls per drawn SNP across 10 proposals. That residual is the
per-proposal cold start.

**Slab size.** `block_snps × n_boundary_columns × 2` bytes, since `cdf_by_age` is `uint16` and
`read_boundary_cdfs` reads one contiguous run per boundary column across the block's
coordinate span.
