# normalizeTE — code review, round 4: the swap-chain sampler and its documentation

**Reviewed at:** `6683365` (`v0.1.0-2-g6683365`), **plus uncommitted edits to
`sample_age_matched_controls.py`** present at review time (mtime 08:01). That
file was being modified during the review; §6 records what changed underneath it
and what is still unfinished there. All other files are at `6683365`.
**Test suite:** 100 passed, 1 failed —
`tests/test_benchmark_interval_store_gate2.py::test_scalar_audit_one_and_two_workers_select_same_mutations`,
pre-existing and macOS-only (it needs Linux `fork` to share the loaded tree
sequence read-only).

Scope: everything added since `a176ddc` — `swap_control_sampler.py`,
`sample_age_matched_controls.py`, `run_age_match_manifest.py`,
`release_provenance.py`, the two SLURM launchers, the `q95 -> q50` change in
`te_age_target.py`, and the rewritten `README.md`,
`SWAP_SAMPLER_HPC_HOWTO.md`, and `AGE_MATCHED_CONTROL_SAMPLER_PLAN.md`.

Reviewed independently by Claude and by codex; where the two disagreed, the
disagreement is recorded and resolved on measured evidence (§3.2). Numeric
claims below were verified by running code, not inferred from reading.

Documentation is the focus, per the review request, so it comes first.

---

## 1. Documentation

The prose quality is high. `SWAP_SAMPLER_HPC_HOWTO.md` is genuinely good
operational writing, and its §10 reports the two-draw validation numbers
honestly, including the sandbox limitation that forced serial chains. §8's
warning that "accepted-chain distances concentrate near the acceptance cutoff"
is the correct observation and is not something most write-ups would catch.

What follows is accuracy against the code, plus one missing idea.

### D1 — `--all-eligible` is documented as functional but does nothing

Declared at [sample_age_matched_controls.py:247](sample_age_matched_controls.py#L247):

```python
candidates.add_argument("--all-eligible", action="store_true")
```

`args.all_eligible` is never read anywhere in `main()`. The candidate universe
branches only on `args.candidate_rows is not None`
([sample_age_matched_controls.py:293](sample_age_matched_controls.py#L293),
[:337](sample_age_matched_controls.py#L337),
[:442](sample_age_matched_controls.py#L442)). Behavior is identical with the
flag, without it, and with neither member of the mutually exclusive group. Still
true after the in-flight edits of §6.

It is presented as meaningful in three places:

- [README.md:217](README.md#L217) and [README.md:226](README.md#L226)
  ("`--all-eligible` uses every eligible non-target SNP as the candidate
  universe");
- [SWAP_SAMPLER_HPC_HOWTO.md:168](SWAP_SAMPLER_HPC_HOWTO.md#L168) and
  [:179](SWAP_SAMPLER_HPC_HOWTO.md#L179); and
- [run_age_match_manifest.py:155](run_age_match_manifest.py#L155), which passes
  it on every production run.

**Fix:** either make the group `required=True` so "all eligible" is an explicit
recorded choice rather than a silent default, or delete the flag and update all
three documents.

### D2 — "The construction state is discarded" is false

[README.md:28](README.md#L28):

> `sample_age_matched_controls.py` first finds a set inside that threshold, then
> runs a constrained random swap walk. The construction state is discarded.

Burn-in ends at **50%** replacement
([swap_control_sampler.py:359](swap_control_sampler.py#L359),
`config.burnin_replacement_fraction = 0.50`), so up to half of the greedy
construction set persists into *every* saved set in that chain.

`SWAP_SAMPLER_HPC_HOWTO.md:156` states this correctly ("The greedy first-passage
set is not one of the 100 outputs"). Bring the README into line.

### D3 — The README never says the 100 sets are not independent

The plan §5.5 is explicit ("The 25% default is a starting point, not proof of
independence") and
[SWAP_SAMPLER_HPC_HOWTO.md:269](SWAP_SAMPLER_HPC_HOWTO.md#L269) makes it a
production gate. The README, which is what people actually read, presents "100
matched control sets" with no qualification at all.

Consecutive saved sets share **~75% of their members by construction**
([swap_control_sampler.py:361](swap_control_sampler.py#L361)). Treating a
membership-driven statistic as AR(1) with lag-one correlation 0.75 gives an
effective sample size per chain of roughly

```text
25 * (1 - 0.75) / (1 + 0.75) ~= 3.6
```

or on the order of 14 independent equivalents across four chains, not 100. The
heuristic is crude and some statistics will mix better or worse, but the order
of magnitude is the point.

If these sets become a null distribution downstream, this is the single most
consequential fact about the output, and it belongs in the README next to the
number 100.

### D4 — Nothing documents what the 100 sets are *for*

There is no section anywhere on consuming the output: what statistic to compute
per set, how to assemble a null distribution from correlated replicates, how to
use `reuse_row_indices.npy` / `reuse_counts.npy` (written by
[sample_age_matched_controls.py:223-224](sample_age_matched_controls.py#L223) and
listed in both documents, but never explained), or how to interpret
`chain_index` / `sample_index`.

For the stated project goal — control sets matching the age profile of dozens
of TE datasets — the documented workflow stops one step short of the science.
This is the largest documentation gap.

### D5 — The threshold *is* the matching quality, and no document says so

The walk accepts any proposal that stays inside the threshold
([swap_control_sampler.py:337](swap_control_sampler.py#L337)):

```python
if trial_distance <= threshold:
```

There is no preference for smaller distances. Since the feasible set's mass
concentrates near its boundary, saved sets land just under the cutoff. The
repository's own validation
([SWAP_SAMPLER_HPC_HOWTO.md:273](SWAP_SAMPLER_HPC_HOWTO.md#L273)) demonstrates
this precisely:

| run | threshold | saved W1 range | saved median | median as % of threshold |
| --- | ---: | --- | ---: | ---: |
| q95 | 3,793.04 | 3,639.31 – 3,792.66 | 3,763.44 | 99.2% |
| q50 | 1,905.10 | 1,790.75 – 1,904.43 | 1,883.12 | 98.8% |

Construction reinforces it: it stops at **first passage** below the threshold
([swap_control_sampler.py:289](swap_control_sampler.py#L289)), so the
best-matched set the greedy phase encounters is never retained.

Consequently "how well are the controls matched?" answers to "exactly as well as
we required." That reframes `--acceptance-quantile` from a safety margin into
the primary scientific knob of the whole pipeline, and it should be stated
plainly in the README rather than left implicit in a HOWTO bullet.

### D6 — Plan versus code drift

`AGE_MATCHED_CONTROL_SAMPLER_PLAN.md` describes three things the code does not do.

| Plan | Reality |
| --- | --- |
| §5.3 requires two explicit modes, `exact-chain` and `screened-chain` | No `--mode` option exists. The walk is exact-chain: every proposal is evaluated on `exact_points` ([swap_control_sampler.py:329](swap_control_sampler.py#L329)). Construction uses the coarse 20,000-generation grid, but construction is not part of the sampling claim. |
| §5.4 makes burn-in require a "minimum accepted-swap count" as a secondary guard | Not implemented. `advance()` ([swap_control_sampler.py:318](swap_control_sampler.py#L318)) checks replacement fraction only. |
| §6.4 specifies checkpoints carrying provenance, counters, coarse and exact CDFs, entry rows, previous saved rows, and completed sets, with "resume only when all provenance and parameter fields match" | [`_atomic_checkpoint`](swap_control_sampler.py#L157) writes four fields — `selected`, `phase`, `rng_state`, `payload` — and **nothing ever reads them back**. Resume is chain-granular, via `chain-results/*.npz`. |

`SWAP_SAMPLER_HPC_HOWTO.md:228` is honest about the last one ("An interrupted
chain currently restarts from its deterministic seed"). The plan is not. Note
also that `--keep-checkpoints` overstates restart granularity; those files are
diagnostic only.

The §5.4 gap matters beyond bookkeeping: a minimum accepted-swap count is
exactly the guard that would blunt the stopping-rule problem in §3.1.

### D7 — Smaller inconsistencies

- **Output listings disagree.** [README.md:242-251](README.md#L242) omits
  `target_cdf.npy` and `age_bins.npy`, which
  [sample_age_matched_controls.py:211-212](sample_age_matched_controls.py#L211)
  writes. `SWAP_SAMPLER_HPC_HOWTO.md:194` lists them.
- **Test commands disagree.** [README.md:79](README.md#L79) is
  `python -m pytest -q tests test_snp_age_distribution.py`;
  [SWAP_SAMPLER_HPC_HOWTO.md:23](SWAP_SAMPLER_HPC_HOWTO.md#L23) is
  `python -m pytest -q tests`. Neither warns that one test fails by design on
  macOS, so a new user's first action produces an unexplained failure.
- **Nine `.md` files, two linked.** Only `CHANGELOG.md` and
  `SWAP_SAMPLER_HPC_HOWTO.md` are referenced from the README.
  `CODE_REVIEW*.md`, `INTERVAL_STORE_PLAN_REVIEW.md`,
  `INTERVAL_STORE_IMPLEMENTATION_PLAN.md`, and
  `GLOBAL_QUANTILE_SAMPLER_IMPLEMENTATION_PLAN.md` are historical. A short
  "Document map" section, or `docs/history/`, would help.
- **Version pinning guidance will rot.** [README.md:90-92](README.md#L90) and
  `SWAP_SAMPLER_HPC_HOWTO.md:26-29` tell users that `v0.1.0` is the q95 baseline
  and that the q50 default must be pinned by commit hash. Correct today; stale
  the moment another tag lands. Tagging `v0.2.0` retires the whole paragraph.

### D8 — A documented footgun that is not enforced

[README.md:293](README.md#L293) and
[SWAP_SAMPLER_HPC_HOWTO.md:93](SWAP_SAMPLER_HPC_HOWTO.md#L93) both instruct the
operator to keep the `#SBATCH --array` range and `AGE_MATCH_TASK_COUNT` in sync.
Nothing checks it.

`_task_values` ([run_age_match_manifest.py:80](run_age_match_manifest.py#L80))
validates `0 <= task_id < task_count`, so an array *larger* than the count fails
loudly. An array *smaller* than the count fails silently: the missing task IDs
are never launched and their manifest rows are simply never processed. With
`--array=0-4` and `AGE_MATCH_TASK_COUNT=10`, half the TE datasets are skipped
and every job reports success.

SLURM exports `SLURM_ARRAY_TASK_COUNT`. One line in each launcher closes it:

```bash
[[ "${SLURM_ARRAY_TASK_COUNT}" == "${AGE_MATCH_TASK_COUNT}" ]] || {
    echo "array size ${SLURM_ARRAY_TASK_COUNT} != AGE_MATCH_TASK_COUNT ${AGE_MATCH_TASK_COUNT}" >&2
    exit 3
}
```

---

## 2. Bugs

### B1 — `_propose_unselected` hangs when every candidate is selected

[swap_control_sampler.py:172-179](swap_control_sampler.py#L172):

```python
while True:
    row = int(candidates[int(rng.integers(candidates.size))])
    if row not in selected_set:
        return row, duplicates
    duplicates += 1
```

`run_chain` guards only `candidates.size < n`
([swap_control_sampler.py:210](swap_control_sampler.py#L210)), so
`candidates.size == n` passes. Every candidate is then selected, and the loop
never terminates.

**Verified:** with a 5-candidate universe and all 5 selected, the call did not
return within 2 seconds.

A hang inside a 24-hour SLURM allocation is worse than an error: the job burns
its full time limit and reports a timeout rather than a diagnosis. Require
`candidates.size > n` at minimum. In practice, require real headroom — with
`candidates.size` only slightly above `n`, burn-in cannot reach 50% replacement
and the chain instead burns the entire `--max-chain-proposals` budget before
failing.

### B2 — `replacement_fraction` runs on every walk proposal

This is the largest practical defect.

[swap_control_sampler.py:318](swap_control_sampler.py#L318):

```python
while replacement_fraction(selected, reference) < required:
```

`replacement_fraction` ([swap_control_sampler.py:70](swap_control_sampler.py#L70))
calls `np.intersect1d`, which sorts — `O(n log n)` — and it sits in the loop
condition, so it executes once per proposal.

**Measured** (`np.intersect1d` on realistic row arrays):

| n | per call | at the default `--max-chain-proposals 10000000` |
| ---: | ---: | ---: |
| 4,061 | 165.5 µs | 27.6 min per chain |
| 35,000 | 1,505.6 µs | **250.9 min (4.2 h) per chain** |

That is time spent in this single call, before any CDF evaluation or Wasserstein
computation. `SWAP_SAMPLER_HPC_HOWTO.md:259-261` lists a ~35,000-SNP target as a
production gate; this alone may be the difference between that gate passing and
hitting the 24-hour wall.

**Fix is `O(1)` per accepted swap.** Maintain a running count of members shared
with the reference set, held as a `set` alongside `selected_set`: on an accepted
swap, decrement if the removed row is in the reference, increment if the added
row is. The replacement fraction is then `1 - shared / n` with no recomputation.
The same value is also recomputed twice in the progress branch
([swap_control_sampler.py:348-349](swap_control_sampler.py#L348)), which the
incremental counter removes for free.

### B3 — The accept test and the certification use different code paths

`advance()` maintains `current_exact` as the mean of per-row `cdf_at` results
([swap_control_sampler.py:302-306](swap_control_sampler.py#L302), updated
incrementally at [:333](swap_control_sampler.py#L333)). Certification recomputes
with `aggregate_cdf` → `_aggregate_uniform_interval_cdf`
([swap_control_sampler.py:365](swap_control_sampler.py#L365)) — a different
algorithm, the difference-array kernel.

**Measured** on a 4,000-row fixture at the 1,000-generation grid:

```text
max |CDF_A - CDF_B|  = 1.52e-10
W1 via A = 1601.469302   W1 via B = 1601.469301
difference           = 2.98e-07 generations
```

This is far below the ~0.4-generation margin observed in the two-draw validation
run, so it is **not urgent**. It is recorded because the failure mode is
expensive when it does occur: since the walk deliberately sits on the boundary
(§D5), a state accepted at `threshold - 1e-7` can fail
`certified_distance > threshold` at
[swap_control_sampler.py:367](swap_control_sampler.py#L367) and raise
`SwapSamplingError`, destroying an entire chain after hours of work.

Two cheap mitigations, either sufficient: accept against a threshold reduced by
a small documented margin, or assign `current_exact = certified` at each save so
the two paths re-synchronize at every checkpoint.

### B4 — Resume trusts its chain artifacts

[sample_age_matched_controls.py:366-370](sample_age_matched_controls.py#L366)
loads completed chains, and the validation at
[:426-432](sample_age_matched_controls.py#L426) recomputes each distance from the
**stored CDF**, not from `row_indices`:

```python
recalculated = np.asarray([
    wasserstein_1(cdf, target_cdf, age_bins) for cdf in result.cdfs
])
```

Shapes, stored-W1-versus-stored-CDF consistency, and within-set uniqueness are
checked. Not checked: that each CDF actually corresponds to its rows, that the
rows are eligible, that they exclude the target, or that they lie in the declared
candidate universe. A corrupt or mismatched `chain-*.npz` holding a
self-consistent CDF/W1 pair publishes as complete.

Recompute at least one CDF per chain from its rows, and re-apply the
eligibility and target-exclusion checks to the loaded rows.

The in-flight edits add `_load_chain_bundle`
([sample_age_matched_controls.py:129](sample_age_matched_controls.py#L129)),
which validates a bundle schema version and an embedded `run_identity` — the
right idea, but it is **not yet wired in**; see §7.

### B5 — `_completed()` accepts zero-byte outputs

[run_age_match_manifest.py:54-77](run_age_match_manifest.py#L54) checks the
metadata flag and filename existence only — no load, no shape check, no identity
check, no distance recomputation.
[tests/test_run_age_match_manifest.py:33](tests/test_run_age_match_manifest.py#L33)
creates the required files with `touch()` and asserts that they count as
complete, so the current behavior is pinned by test.

A truncated Quobyte write is therefore skipped as success on restart. Either load
and shape-check the arrays, or write a separate validation stamp only after
`_write_results` returns.

---

## 3. Statistics

### 3.1 The walk is defensible; the save rule is not

The post-construction walk is a genuine constrained random walk, not greedy
descent — it accepts uphill moves as long as the state stays feasible
([swap_control_sampler.py:337](swap_control_sampler.py#L337)). The proposal
(uniform slot, uniform unselected candidate) is symmetric, so indicator
acceptance satisfies detailed balance with a **uniform distribution over the
connected feasible component**. That is a well-defined and reasonable null, and
it is better than it first appears.

The saved states do not inherit that guarantee:

- burn-in ends at the first time 50% of the construction set has been replaced;
- thinning ends at the first time 25% of the previous saved set has been
  replaced;
- both are **path-dependent stopping times**, and the state at a stopping time
  is not distributed according to the stationary distribution;
- stopping at first crossing also pins consecutive saved sets at *exactly* the
  minimum permitted separation, maximizing their correlation;
- connectivity of the single-swap feasible graph is unproven, and the tighter
  q50 threshold shrinks the feasible set further; and
- no convergence, between-chain, or autocorrelation diagnostics are computed.

**Fix:** thin by a fixed number of accepted swaps, calibrated from measured
autocorrelation, and demote replacement fraction to a reported diagnostic rather
than the stopping rule. The plan already anticipated this in §5.4's "minimum
accepted-swap count"; it simply was not built (§D6).

### 3.2 On `q95 -> q50`: a recorded disagreement

Codex argued that the bootstrap median is an incoherent acceptance boundary,
since approximately half of genuine resamples of the target fail it by
construction, and recommended reverting to 0.95.

**We do not agree, and the repository's own measurements are why.** That
argument treats the threshold as a hypothesis test of "is this control set a
resample of the target." It is not; it is a tolerance. Given §D5 — the walk
lands on the boundary regardless — a tighter threshold buys directly better
matched controls, and `SWAP_SAMPLER_HPC_HOWTO.md:282-288` reports that it costs
essentially nothing:

| | thinning acceptance | serial runtime |
| --- | ---: | ---: |
| q95 | 32.3% | 148 s |
| q50 | 31.4% | 147 s |

On this evidence **0.50 is the better default** and should stay.

The real problem is one neither quantile solves: *neither number is
scientifically motivated*. Both are quantiles of the target's own bootstrap
noise, which says nothing about how well controls could or should match, and
whichever is chosen becomes the answer. Recommended:

1. keep 0.50 as the default;
2. document that the threshold is the matching specification, not a safety
   margin (§D5);
3. add a pre-specified absolute tolerance option — "controls must match within
   X generations" — with the bootstrap quantile as fallback; and
4. re-measure acceptance rate and runtime at q50 on the 75-draw store before
   trusting the two-draw result. Two draws give `m_i = 2` and far narrower
   posteriors than 75; the feasible set will behave differently.

### 3.3 Confirmed sound

Recorded so they are not re-litigated:

- Chain seeds are derived by SHA-256 from the global seed, target digest, chain
  index, and algorithm version
  ([swap_control_sampler.py:62](swap_control_sampler.py#L62)) — stable, and free
  of Python's randomized `hash()`.
- Manifest task assignment is `index % task_count`
  ([run_age_match_manifest.py:122](run_age_match_manifest.py#L122)) and seeds come
  from an explicit per-row `seed` column, so reordering the manifest changes task
  assignment but not any target's RNG.
- Workers share only a read-only interval store; chain state and RNGs are
  process-local.
- Incremental CDF bookkeeping
  ([swap_control_sampler.py:80](swap_control_sampler.py#L80)) is algebraically
  correct, and `old_cdfs[j]` stays valid through a construction epoch because
  `slots` is a permutation, so each slot is touched exactly once.
- Accumulated float64 drift in `current_exact` is negligible (~1e-17 over
  millions of increments) — the concern is the cross-path discrepancy in B3, not
  accumulation.
- Final publication is atomic
  ([sample_age_matched_controls.py:459-462](sample_age_matched_controls.py#L459)),
  and both SLURM launchers stage the
  store once per array task and fail hard through `set -euo pipefail` and
  `check=True`.
- The `uint64` offset promotion hazard flagged in earlier rounds does not appear
  in these new paths; row indices are cast to `int64` explicitly.

---

## 4. Test coverage

The new tests establish that the code runs and that bookkeeping holds. They
cannot detect any of §3.1. `tests/test_swap_control_sampler.py` uses a threshold
of 1,000 on a tiny fixture and asserts shapes, uniqueness, target exclusion, and
minimum replacement.

Tests that would catch a real defect:

- enumerate every feasible set in a tiny universe; compare long-run
  **fixed-step** visitation frequencies against the expected uniform
  distribution;
- separately test the current **first-crossing** save rule against that same
  known distribution — the gap between the two is exactly §3.1;
- measure membership and summary-statistic autocorrelation across saved states,
  and report effective sample size;
- start chains in distinct feasible regions and test convergence and overlap;
- assert that an uphill-but-feasible proposal is actually accepted (this is the
  property that separates the walk from an optimizer, and nothing currently
  pins it);
- compare incremental CDF and W1 against full recomputation over thousands of
  randomized accepted swaps;
- resume from a chain artifact whose `row_indices` and stored CDF disagree, and
  require rejection (B4);
- reject a zero-length or truncated "complete" manifest output (B5) — the
  existing test currently asserts the opposite; and
- exercise `candidates.size == n` and assert a prompt error rather than a hang
  (B1).

---

## 5. Housekeeping

- `crap1` and `crap2` are untracked in the repository root.
- `os.sys.argv` at
  [sample_age_matched_controls.py:435](sample_age_matched_controls.py#L435)
  should be `sys.argv`.
- [run_age_match_manifest.py:157](run_age_match_manifest.py#L157) hardcodes
  `--sets 100 --chains 4 --sets-per-chain 25`, so the manifest workflow cannot
  vary them even though the per-target CLI can. `--acceptance-quantile` is parsed
  and validated in `sample` mode
  ([run_age_match_manifest.py:118](run_age_match_manifest.py#L118)) but unused
  there; the threshold comes from target metadata.
- `gate1_report.py`, `gate1_smoothing_bias.sbatch`, and `design_probes/` are
  orphaned. They were built for the quota-fitting design that
  `AGE_MATCHED_CONTROL_SAMPLER_PLAN.md` has since replaced, and
  `gate1_report.py` cites "plan section 2.2", which no longer denotes what it
  says. Delete them, or move them under `docs/history/`.

---

## 6. In-flight edits to `sample_age_matched_controls.py`

That file was modified during this review (uncommitted, mtime 08:01). The
changes move in the right direction on two of the findings above, but three are
unfinished and one creates a **new** documentation mismatch. Recorded here so
the round is auditable against a moving file.

### 6.1 New: the CLI default is now 10 chains x 10 sets; every document says 4 x 25

[sample_age_matched_controls.py:255-256](sample_age_matched_controls.py#L255):

```python
parser.add_argument("--chains", type=int, default=10)
parser.add_argument("--sets-per-chain", type=int, default=10)
```

`--sets` still defaults to 100, so the invariant holds. But `4 x 25` is baked
into every document — README.md lines 14, 221, 233, 237; SWAP_SAMPLER_HPC_HOWTO.md
lines 4-5, 73, 151, 172, 262; CHANGELOG.md lines 17-18; and
AGE_MATCHED_CONTROL_SAMPLER_PLAN.md lines 11, 146, 208, 243, 257, 287, 563 — and
it is hardcoded at
[run_age_match_manifest.py:157](run_age_match_manifest.py#L157). Production
therefore still runs 4 x 25; only the interactive default changed. Seventeen
document locations now contradict the CLI.

On the statistics this is a **good** change and directly addresses §D3/§3.1:
ten independently seeded chains contributing ten states each has a far better
effective sample size than four chains of twenty-five, because the correlation
is *within* chains. If it is intended, it needs to propagate to the manifest
runner, the sbatch CPU request (`--cpus-per-task=4` cannot feed ten chains; note
`--workers` must be `<= chains`), and all six documents. If it is a scratch
value, revert it before it ships.

### 6.2 New and undocumented: `--work-dir`

[sample_age_matched_controls.py:249-253](sample_age_matched_controls.py#L249)
adds an explicit work directory whose help says "use node-local scratch on HPC",
with `_publish_directory`
([:157](sample_age_matched_controls.py#L157)) copying the finished directory to
the destination filesystem and `os.replace`-ing it there. Publication stays
atomic on the destination, and the `complete is True` check before the rename is
a good guard.

Two gaps:

- It appears in no document, and neither `sample_age_matches.sbatch` nor
  `run_age_match_manifest.py` passes it — so the feature that makes HPC staging
  correct is currently unreachable from the production path.
- **It conflicts with `--resume`.** Put the work directory on `$TMPDIR` and the
  node-local scratch disappears when the allocation ends, so the completed-chain
  artifacts that `--resume` exists to reuse are gone. The failure message
  ([:464](sample_age_matched_controls.py#L464)) still says "Incomplete work
  retained at {work}", pointing at a path that no longer exists. Since
  `sample_age_matches.sbatch` always passes `--resume`, this needs to be
  documented explicitly: either keep the work directory on Quobyte and accept
  the I/O, or accept that a killed job restarts every chain.

### 6.3 Unfinished: the resume-hardening is written but not connected

Three new pieces exist and none is reachable:

- `_save_chain_result` gained a `run_identity` keyword
  ([:89](sample_age_matched_controls.py#L89)) that `main()` never passes
  ([:377](sample_age_matched_controls.py#L377),
  [:393](sample_age_matched_controls.py#L393),
  [:402](sample_age_matched_controls.py#L402)), so no bundle is ever written
  with `bundle_schema_version` or `run_identity`.
- `_load_chain_bundle` ([:129](sample_age_matched_controls.py#L129)) therefore
  raises `invalid or incomplete chain bundle` for every artifact the program
  produces — and `main()` calls `_load_chain_result`
  ([:367](sample_age_matched_controls.py#L367)) anyway, so it is dead code.
- `_atomic_copy_file` ([:141](sample_age_matched_controls.py#L141)) has no
  caller.

**B4 therefore still stands as written.** Completing this is the fix: pass
`run_identity` on save, load through `_load_chain_bundle`, and compare the
embedded identity to the current run before accepting a chain. Recomputing at
least one CDF from `row_indices` is still needed on top of that, since identity
matching does not detect a corrupt array within an otherwise correct bundle.

### 6.4 Unchanged

`--all-eligible` is still unread (§D1). W1 on resume is still recomputed from
stored CDFs rather than rows (§B4). `os.sys.argv` is still at
[:435](sample_age_matched_controls.py#L435). Nothing in `swap_control_sampler.py`
changed, so §B1, §B2, §B3, and §3.1 are unaffected.

---

## 7. Suggested order

1. **B2** — before the ~35,000-SNP production gate; it may be the difference
   between passing and timing out.
2. **B1** — a hang is the worst failure mode in a batch allocation.
3. **D1, D2, D3** — small documentation corrections, and D3 is the one a reader
   most needs.
4. **D8, B5, B4** — silent-success paths, in that order of exposure.
5. **§3.1** — fixed-step thinning plus autocorrelation diagnostics. This is the
   substantive statistical work and should be scheduled, not rushed.
6. **D4, D5** — the two documentation additions that make the output usable and
   its quality interpretable.

Interleave §6 as it lands: §6.3 completes B4, and §6.1 must be resolved one way
or the other before the next release, because right now the CLI default and
every document disagree about the shape of the output.
