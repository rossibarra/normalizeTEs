# Code Review Round 7 — bootstrap-target matching, and Φ-SFS regression check

**Scope:** `bootstrap_target_matcher.py`, `tests/test_bootstrap_target_matcher.py`,
`BOOTSTRAP_TARGET_MATCHING_PLAN.md`, README §6 and §7, and a regression-only
re-check of `phi_sfs.py` against `CODE_REVIEW_ROUND6.md`.
**Reviewers:** Claude (Opus 5) and Codex (gpt-5.6), two independent passes then a
reconciliation round. Every item is agreed by both unless marked.
**Status:** two CRITICAL blockers. The new matcher's output **cannot be consumed
by `phi_sfs.py` at all**, which is the one thing the design says it must do.

---

## 0. What was verified as correct

- **The optimizer's core state machine is sound.** Both reviewers independently
  concluded this. Swaps are improvement-only against one fixed per-replicate
  target; epoch-end certification cannot lose a better mid-epoch state; the
  final `certified_best` is recomputed from `best_rows` and its W1 must agree
  (`bootstrap_target_matcher.py:259-266`); restart selection is the certified
  argmin (`:815-817`).
- **The bootstrap is a correct iid multinomial site bootstrap.**
  `rng.multinomial(n_sites, uniform)` then a count-weighted mean of the
  per-TE-site CDFs is exactly the bootstrap mean CDF.
- **Grid alignment is correct**, which is easy to get wrong here.
  `analysis_points` returns `ages + width/2`, so CDFs are evaluated at shifted
  points while `wasserstein_1` weights by `np.diff(age_bins)`. Since the shift
  is a constant, `diff(points) == diff(ages)` and the two are consistent.
- **Resume validates what it reloads.** `_load_replicate_bundle` checks counts,
  target, and distance, and every reloaded restart still goes through
  `validate_restart_result` with the expected seed identity (`:792-805`).
- **Triangle-inequality checks are real** and would catch an inconsistent
  distance triple (`:505-511`).
- `phi_sfs.py` has **no code regression** from round 6. All 148 tests pass.

---

## 1. CRITICAL — Φ-SFS rejects every bootstrap-target bundle (digest contract)

`bootstrap_target_matcher.py:632,700`; `phi_sfs.py:480-501`

The two modules compute `target_digest` over different inputs:

| | formula |
|---|---|
| `sample_age_matched_controls.py:447` (what Φ-SFS recomputes) | `_sha256_arrays(rows, cdf, age_bins, [threshold])` |
| `bootstrap_target_matcher.py:632` | `_sha256_arrays(rows, cdf, age_bins)` — no threshold |

**Reproduced.** Building a bundle from the repo's own fixtures and feeding it to
`phi_sfs._validate_provenance`:

```
ValueError: matched-control bundle was built for a different target:
it records target_digest 5228fb7c..., but <target> hashes to 6dd0bbe0...
```

This is not a theoretical mismatch. `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §7
explicitly requires the pairing — *"Preserve that pairing through Phi-SFS
analysis"* — and the matcher's own metadata records
`"phi_sfs_selection_blind": True`. The bundle exists to be consumed by Φ-SFS,
and cannot be.

**Fix:** compute the established four-array digest, reading `threshold` from the
target's `wasserstein_threshold_generations` exactly as
`sample_age_matched_controls._load_target` does — ideally by importing that
loader, as `phi_sfs.py` already does, so the three cannot drift. If the
optimizer genuinely needs a threshold-independent identity, keep it under a
separate key. Add an end-to-end matcher→Φ-SFS test.

---

## 2. CRITICAL — bundles omit the replicate identifiers Φ-SFS requires

`bootstrap_target_matcher.py:536-555`; `phi_sfs.py:426-448`

`_load_coordinates` requires `chain_index.npy` and `sample_index.npy`. The
matcher writes neither. Confirmed from the published listing: it writes
`row_indices`, `positions`, `chromosome_codes`, `chromosome_labels`, and the
`B_r/E_r/O_r/R_r` arrays, but no per-replicate identifier arrays at all.

So even after fixing item 1, Φ-SFS fails on the next line.

**Fix, and the design point matters:** do **not** fake `chain_index`/
`sample_index`. Bootstrap replicates are not chain samples — they are
independent, with no within-chain autocorrelation, which is exactly the
property the chain arrays exist to expose. Publish `replicate_id.npy` and
teach `phi_sfs.py` to branch on the bundle's `schema_version`
(`swap-age-matched-controls-v1` vs `bootstrap-target-matches-v1`), carrying
whichever identifiers that schema defines into `replicates.csv`. Keep the
alignment with `B_r/E_r/O_r/R_r` so downstream work can join Φ against matching
error, which plan §7 requires.

---

## 3. HIGH — resume identity does not lock the implementation

`bootstrap_target_matcher.py:705-724`

`identity.json` records config, digests, seed, and `ALGORITHM_VERSION`, but no
release version, Git commit, or NumPy version. A code change mid-run that does
not manually bump `ALGORITHM_VERSION` lets `--resume` combine replicate bundles
optimized by two different implementations, after which the final metadata
attributes all 100 replicates to the current checkout. That is a corrupt
scientific result carrying clean-looking provenance — and these are long HPC
jobs where resume is the normal path, not the exception.

**Fix:** put `software_provenance()` and `np.__version__` into `identity`, so
the existing `existing_identity != identity` comparison rejects the mix. Reject
resume from a dirty checkout in production. Add resume tests for altered code
identity and altered config.

---

## 4. HIGH — a work/output path arrangement can delete the published result

`bootstrap_target_matcher.py:677-678,701-704,837-838`

`run()` publishes to `--output`, then unconditionally does
`shutil.rmtree(work_dir)` unless `--keep-work`. If `--output` is equal to or
nested under `--work-dir`, the publish succeeds and is then deleted — and
`main` still prints success and returns 0.

**This is a regression, not an oversight.** The previous matcher already guards
it at `sample_age_matched_controls.py:456-457`:

```python
if work.resolve() == args.output.resolve():
    raise ValueError("--work-dir and --output must be different paths")
```

**Fix:** restore that guard, and strengthen it to reject either nesting
relationship after resolving both paths.

---

## 5. HIGH — every proposal is scored on the full 22,856-point exact grid

`bootstrap_target_matcher.py:173-177,192-203,229-234`;
`swap_control_sampler.py:105-120,230-245`

The store records `maximum_above = 22,854,736` generations. With
`te_age_target`'s `bin_width = 1,000`, the exact analysis grid is **22,856
points**.

`swap_control_sampler` already solves this with a two-tier design:
`search_bin_width` defaults to 20,000 — a **1,144-point** coarse grid — with
adaptive refinement, and exact-grid certification only for saved sets.
`bootstrap_target_matcher` has no coarse tier; grepping it for
`search_bin_width`, `search_grid`, or `coarse` returns nothing. Every proposal
CDF and every W1 runs at full 22,856-point width.

Projected row-point evaluations at `X = 35,512` (the largest real TE position
file in this repo), at defaults of 100 replicates × 3 restarts × 50 epochs:

| grid | row-point evaluations |
|---|---|
| exact, 22,856 pt | 24.4 × 10¹² |
| coarse, 1,144 pt | 1.2 × 10¹² |

Measured `cdf_at` throughput is 342M row-points/s — but that is on the tiny
in-memory fixture store, so treat it as an **optimistic upper bound**. Even so
it implies roughly 20 hours single-threaded on the exact grid versus about 1
hour on the coarse grid, before the W1 work layered on top at the same width.

**Fix:** reinstate the coarse-search / exact-certify tier that
`swap_control_sampler` already implements, including its adaptive refinement.
This is reuse of an existing, tested pattern, not a redesign. **Benchmark on a
real production store before launch** — no such store exists in this repo, so
the constant above is unvalidated.

**Sub-point — I was wrong, Codex was right.** I initially claimed per-epoch
`aggregate_cdf` certification was roughly half the epoch cost and worth
reducing, citing a microbenchmark showing `row_cdfs` 139.4 ms versus
`aggregate_cdf` 128.5 ms. That benchmark was invalid: it ran against the
7-SNP fixture store with row indices repeated 2,048 times, so it measured
output materialization rather than store work.

Re-measured on a 4,000-SNP store with 400 distinct rows and a 20,938-point
exact grid:

| operation | per epoch |
|---|---|
| proposals, coarse 1,048-pt grid | 9.43 ms |
| proposals, exact 20,938-pt grid | 27.83 ms |
| certification, `aggregate_cdf` exact grid | **1.00 ms** |

`aggregate_cdf_at` really does dispatch to a specialized
`O(intervals + grid)` path, so certification is about a tenth of the epoch,
not half. Reducing its cadence is not worth trading away the guarantee that
every recorded `best_distance` is exact. **Resolution: no cadence knob.**
Certification stays unconditional every epoch, and the only change here is the
coarse proposal tier.

---

## 6. MEDIUM — bootstrap targets are accumulated in float32

`bootstrap_target_matcher.py:109-120,693-699`

`te_cdf_rows` is built as float32 (`:695`), so `bootstrap_cdf` performs the
long count-weighted sum in float32 (`:119-120`) and only casts to float64
afterward. The optimizer's own selected-row cache is float64, so this is a
mixed-precision comparison.

**Agreed severity, with the reasoning stated explicitly:** the error lands on
`T^(r)`, which is *fixed* for the whole optimization, so it adds no inter-epoch
noise and does not interact with the convergence criterion. At `X = 35,512` the
accumulated relative error is ~1e-5, a systematic offset on `B_r` of order one
generation against `B_r ≈ 1,900`. Real, but modest — and free to fix.

**Fix:** accumulate in float64 in blocks, without promoting the whole TE matrix
(which would double 2.9 GiB to 5.8 GiB at this `X`). Test against a float64
reference at production dimensions.

---

## 7. MEDIUM — every bundle records the repository as its source store

`bootstrap_target_matcher.py:628`

```python
"source_store": str(Path(getattr(store, "path", "")).resolve()),
```

`SNPAgeIntervalDataset` exposes `store_dir`, not `path`. **Confirmed by
execution:** `hasattr(store, "path")` is `False`, so the fallback `""` resolves
to the current working directory and the metadata records
`/Users/jeffreyross-ibarra/Projects/normalizeTE` as the source store for every
run.

The store's `content_sha256` is recorded correctly and remains the real
identity, which is why this is MEDIUM rather than higher — but the durable
provenance field is simply wrong.

**Fix:** use `store.store_dir`, or pass the resolved `--store` path through to
`_write_outputs`. Assert it in the CLI test.

---

## 8. MEDIUM — the published bundle falls short of its documented contract

`BOOTSTRAP_TARGET_MATCHING_PLAN.md:334-355`; `bootstrap_target_matcher.py:536-618`

Promised but not published: bootstrap seeds and percentiles, initial restart
rows, per-epoch proposal counts, per-restart `R_r` and QC status, per-restart
runtime, and `seed_sets_digest` — which is computed for the work identity at
`:711` but never carried into the final metadata. Several of these survive only
in the work bundles, which `--keep-work` deletes by default.

**Fix:** publish the promised fields, or mark each explicitly deferred and
narrow the "complete restart traces" claim in both the plan and README §6.

---

## 9. Documentation

The maintainer asked specifically that the docs explain both the new Φ-SFS
approach and the new matching approach. Φ-SFS documentation is in good shape
after round 6. The matching documentation has real gaps.

### 9a. MEDIUM — primary versus experimental status is never stated

`README.md:7-14, 323-399, 401-419`

The top-level "Recommended workflow" still describes only the three-stage
hard-q50 path and does not mention bootstrap-target matching or Φ-SFS at all. A
reader following it never learns §6 or §7 exist. §6 then introduces the new
matcher without saying whether it is now the production path. §7's command
points at `matches/all_te`, the hard-q50 bundle, without comment.

The result is that the docs lead a reader straight into item 1: §6 tells them
bootstrap matching is how uncertainty is propagated, §6 closes by discussing
"Φ-SFS uncertainty", and §7 is the Φ-SFS step — so they will point `--matches`
at the bootstrap bundle and hit a digest error.

**Fix:** in "Recommended workflow", state which workflow is primary at this
release and which is the sensitivity analysis, and list all stages including
Φ-SFS. In §7, show the command for each bundle type explicitly.

### 9b. MEDIUM — the estimand caveat lives only in the plan

`README.md:380-388`; `BOOTSTRAP_TARGET_MATCHING_PLAN.md:375-406`

The plan is careful and correct: bootstrap-target Φ-SFS holds the observed TE
SFS fixed, "is not a bootstrap confidence distribution for the TE SFS and is
not by itself a p-value," and one must not compute the TE SFS from resampled TE
rows. **None of that reaches the README**, which is what people actually read.

**Fix:** lift those three sentences into §6 or §7.

### 9c. MEDIUM — the most important scientific risk is buried

`README.md:389-395`

The deepest risk in this design is that optimizing SNP membership to hit a
precise age CDF may select on properties correlated with age — including
derived allele frequency, which is precisely what Φ-SFS measures. The docs do
flag it, but as one clause in a list: *"Also assess whether W1-repair utility
is associated with SNP frequency."*

Given that Φ-SFS is the downstream statistic and this failure mode would bias
it directly, this deserves its own short paragraph rather than a clause.

### 9d. MEDIUM — README also tells readers to inspect `chain_index`

`README.md:512-514` and the workflow intro direct readers to retain and inspect
`chain_index`, which bootstrap bundles do not have and should not have (item 2).

### 9e. MEDIUM — the new stage has no production wiring

There is no `.sbatch` wrapper, `run_age_match_manifest.py` does not know about
it, and README §8 (Farm/Quobyte) omits it entirely — unlike every other stage.
The plan admits distributed execution is deferred; the README does not.

**Fix:** either add the array-task wrapper and manifest integration, or label
the §6 command explicitly local/pilot-only and keep it outside the supported
production path.

### 9f. LOW — stale cross-reference

`phi_sfs.py:6` points at "README section 6"; Φ-SFS is now section 7. Doc drift
from the renumbering.

---

## 10. LOW — dead imports

`bootstrap_target_matcher.py:10,18,30` — `math`, `Any`, and
`replacement_fraction` are each referenced exactly once, on their own import
line.

---

## 11. Tests

`tests/test_bootstrap_target_matcher.py` is 148 lines for an 875-line module
and covers the bootstrap, seed derivation, one optimizer trace, and one CLI
run. Not covered, roughly in priority order:

1. **matcher → Φ-SFS end to end** — would have caught both CRITICALs.
2. **Resume**: complete-bundle reuse, config mismatch rejection, identity
   mismatch rejection. Resume is the normal production path and is untested.
3. `--work-dir`/`--output` collision (item 4).
4. Provenance rejection: store schema, content, and catalog mismatches against
   both target and seed bundles.
5. QC ratio and triangle arrays on a case with a nonzero `B_r` spread.
6. `source_store` metadata correctness (item 7).
7. `bootstrap_cdf` against a float64 reference at realistic width (item 6).

---

## 12. Suggested order

| # | Item | Why |
|---|---|---|
| 1 | Digest contract (§1) | Blocker — the pipeline does not connect |
| 2 | Replicate identifiers (§2) | Blocker — same, and a schema decision |
| 3 | End-to-end matcher→Φ test (§11.1) | Locks 1 and 2 permanently |
| 4 | Work/output guard (§4) | One line; can destroy a finished run |
| 5 | Coarse search tier (§5) | ~20×, plus ~2× from certification cadence |
| 6 | Resume identity (§3) | Silent corruption on the normal production path |
| 7 | README primary/sensitivity + estimand (§9a–9d) | Docs currently lead into §1 |
| 8 | float32, source_store, dead imports (§6, §7, §10) | Cheap correctness/hygiene |
| 9 | Output contract, production wiring (§8, §9e) | Launch readiness |

Items 1–4 are the launch blockers. Item 5 is the difference between a run that
takes about an hour and one that takes about a day, per TE category — and needs
a real-store benchmark before anyone commits to a schedule.
