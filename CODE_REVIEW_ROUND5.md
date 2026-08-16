# normalizeTE — code review, round 5: distributed fixed-sweep matching

**Reviewed at:** `09b9400` (`v0.2.0-1-g09b9400`), clean tree. The substantive
changes are in `b3be519` ("Implement distributed fixed-sweep age matching");
`09b9400` is a README wording change only.
**Test suite:** 108 passed, 1 skipped. The macOS `fork` audit is now skipped
rather than failed, so a fresh checkout no longer greets a new user with a red
test.

Scope: the Round 4 disposition, the new distributed architecture
(`distributed_age_match.py`, `gather_age_matches.sbatch`, the `sample-chain` and
`gather` manifest modes), `--acceptance-distance`, and the rewritten `README.md`.

Reviewed independently by Claude and by codex. **This round includes an
end-to-end execution of the full 10 x 10 workflow**, which produced both the
mixing measurements in §2 and the only new blocking defect (§3) — a runtime
failure that no amount of code reading would have surfaced.

---

## 1. Round 4 disposition

Verified in code, not taken on trust.

| Round 4 finding | Status | Evidence |
| --- | --- | --- |
| §3.1 path-dependent stopping rule | **Fixed** | `advance()` stops solely on `required_accepted = max(1, ceil(sweeps * n))` ([swap_control_sampler.py:323](swap_control_sampler.py#L323)). Replacement fraction is now reported as a diagnostic only. Algorithm version bumped to `swap-age-controls-v2-fixed-sweeps`. |
| B1 `_propose_unselected` hang | **Fixed** | `candidates.size <= n` now raises ([swap_control_sampler.py:210](swap_control_sampler.py#L210)). |
| B2 `replacement_fraction` in the walk loop | **Fixed** | Replaced by `shared_with_reference`, initialized once per phase ([swap_control_sampler.py:324](swap_control_sampler.py#L324)) and updated `O(1)` on each accepted swap ([:350](swap_control_sampler.py#L350)). Because `new` is drawn from outside `selected_set`, `old == new` is impossible, so the decrement/increment pair cannot double-count; re-entry of a previously removed row is handled correctly. |
| B3 accept-test vs certification | **Fixed** | Both mitigations landed: a conservative `walk_threshold` margin ([swap_control_sampler.py:316](swap_control_sampler.py#L316)) and `current_exact = certified` resynchronization at each save ([:379](swap_control_sampler.py#L379)). |
| B4 resume trusts chain artifacts | **Partial** | `_validate_chain_result` now checks bundle identity, row range, eligibility, target exclusion, declared candidate universe, within-set uniqueness, stored-W1-versus-stored-CDF, and the threshold. It recomputes a CDF from `row_indices` for **only the first saved set** — see §4.1. The chain seed is still trusted — see §4.5. |
| B5 `_completed()` accepts zero-byte outputs | **Fixed** | Arrays are now loaded and shape-checked, metadata `sets`/`set_size` are cross-checked against actual shapes, and `diagnostics.csv` must be nonempty ([run_age_match_manifest.py:71](run_age_match_manifest.py#L71) onward). |
| D1 `--all-eligible` is a no-op | **Fixed** | The mutually exclusive group is now `required=True`, so selecting the all-eligible universe is an explicit, recorded choice. |
| D8 unenforced SLURM array size | **Fixed** | Both launchers compare `SLURM_ARRAY_TASK_COUNT` against their declared count, *and* the Python runner independently requires exactly `targets x chains` tasks ([run_age_match_manifest.py:165](run_age_match_manifest.py#L165)). Defense in depth. |
| D2–D7 documentation | **Fixed** | See §5. |

Two design points worth recording as sound. Chain bundles are published to
`<manifest output>.chains/chain-NNN.npz` on the durable filesystem
([run_age_match_manifest.py:257](run_age_match_manifest.py#L257)) before the
job-local scratch disappears, so the "work lives on `$TMPDIR` but `--resume`
needs it" tension raised in Round 4 §6.2 is genuinely resolved: pre-publication
work is lost and deterministically restarted, published bundles survive. And
`--acceptance-distance` is correctly plumbed, validated finite and positive
([te_age_target.py:286](te_age_target.py#L286)), documented as overriding the
quantile, and recorded in metadata as `acceptance_threshold_source`.

---

## 2. Measured behavior of the fixed-sweep design

An 800-SNP young-biased target was built on a 40,000-row synthetic two-draw
store and the complete `--chains 10 --sets-per-chain 10` workflow was run to
publication.

```text
adjacent replacement per chain : mean 0.605   min 0.599   max 0.609
W1 lag-1 autocorrelation       : mean -0.171  min -0.663  max +0.486
AR(1) ESS heuristic total      : 43.3 of 100 sets
threshold 212.57 | saved W1 min 194.45  max 212.54  median 209.00
unique controls 21,881 of a 40,000 pool | maximum reuse 23 | set size 800
```

Three things follow.

**The sweep mechanism does what it claims.** One accepted-swap sweep produces
60.5% adjacent replacement against a theoretical `1 - e^-1 = 63.2%` under
well-mixed removal. The small shortfall is the expected effect of rows
re-entering the set.

**Effective sample size roughly tripled.** Round 4 estimated ~14 independent
equivalents under the old 25%-replacement rule. The code's own AR(1) heuristic
now reports 43.3. That is a real improvement and, importantly, the pipeline
computes and publishes the number itself rather than leaving the reader to
derive it.

**Boundary concentration is confirmed a third time.** The saved median sits at
98.3% of the threshold, matching the two-draw validation runs
(99.2% and 98.8%, `SWAP_SAMPLER_HPC_HOWTO.md` §10). This is now correctly
documented (§5), so observation and documentation agree.

---

## 3. New blocking defect: the coarse search grid can strand construction

**This is the finding of this round, and it was only reachable by running the
code.**

Greedy construction optimizes W1 on a coarse grid whose width is
`--search-bin-width`, default 20,000 generations
([swap_control_sampler.py](swap_control_sampler.py) `search_grid`), while
acceptance is certified on the exact 1,000-generation analysis grid. Construction
accepts a swap only when it strictly improves the **coarse** distance, so once
the coarse optimum is reached no further swap is ever accepted — even though the
exact distance may remain far above the threshold.

Observed on the fixture above, threshold 212.57:

```text
chain=0 construction epoch=45 accepted=0 exact_w1=1042.5
chain=0 construction epoch=46 accepted=1 exact_w1=1039.4
chain=0 construction epoch=47 accepted=0 exact_w1=1039.4
chain=0 construction epoch=48 accepted=0 exact_w1=1039.4
chain=0 construction epoch=49 accepted=0 exact_w1=1039.4
chain=0 construction epoch=50 accepted=0 exact_w1=1039.4
SwapSamplingError: chain 0 did not reach threshold after 50 epochs
```

The run wasted all 50 epochs on a plateau. The identical command with
`--search-bin-width 1000` succeeded in 26 seconds and published all 100 sets.

This is not a simple ratio rule: the two-draw production validation used a
threshold of 1,905 against the same 20,000-generation grid and worked. It is
reachable whenever a target's threshold is fine relative to the search grid —
small categories, a tight `--acceptance-distance`, or simply the 75-draw store
behaving differently from two draws. The failure is total for the affected
chain, and under the distributed workflow one stranded chain blocks that
target's gather entirely.

Recommended, in order of value:

1. **Adapt instead of repeating.** When an epoch accepts zero swaps and the
   exact distance is still above the threshold, halve `search_bin_width` and
   continue rather than running 49 more identical epochs. Small change, large
   robustness gain, and it removes the tuning burden from the operator.
2. **Improve the diagnosis.** The existing error reports best W1 and threshold,
   which is good; add that a plateau with zero accepted swaps indicates the
   coarse search grid is too wide relative to the threshold.
3. **Document the interaction** in the README and the HOWTO, and mention it in
   the production gates alongside the small-category gate.

---

## 4. Still open

### 4.1 `_validate_chain_result` recomputes only the first saved set

[sample_age_matched_controls.py:344](sample_age_matched_controls.py#L344)
verifies every stored distance against every stored CDF, but
[:352](sample_age_matched_controls.py#L352) recomputes the CDF from
`row_indices` for `rows[0]` alone:

```python
recomputed = aggregate_cdf(store, rows[0], analysis_points(age_bins))
```

Ten of the 100 published sets are therefore verified against their rows; the
other ninety are trusted. Sets 1–9 of a bundle can carry altered row indices
alongside their original CDFs and distances and still pass, provided the
substituted rows remain eligible, non-target, and unique — all of which the
other checks confirm independently of the CDF.

The corresponding test corrupts `payload["row_indices"][0, 0]`
([tests/test_swap_control_sampler.py:245](tests/test_swap_control_sampler.py#L245)),
which is precisely the one index that is checked. It would not catch corruption
at `[1, 0]`.

`README.md:278-280` states that publication fails if any set "has a stored CDF
inconsistent with its rows". That is an overclaim as written.

**Fix:** recompute every saved set's aggregate CDF in bounded blocks. The cost
is 10 `aggregate_cdf` calls per chain at `O(75n + B)` each — negligible against
chain runtime, and gather is a separate short job.

### 4.2 Store identity can be null

`identity` ([distributed_age_match.py:116](distributed_age_match.py#L116))
carries `source_store_schema` and `source_catalog_sha256`, but the catalog
digest is optional
([distributed_age_match.py:86](distributed_age_match.py#L86)) and the equality
check is skipped when the target does not declare one
([:87](distributed_age_match.py#L87)). Two different stores sharing a schema and
lacking catalog digests produce identical identities, so bundles built against
the wrong store can pass gather — and with §4.1, only one set per bundle would
be recomputed against that store.

Require a non-null catalog digest for distributed operation, or add a mandatory
store-content digest. A resolved path is provenance, not identity.

### 4.3 `--acceptance-distance` still pays for the full bootstrap

`build_target()` generates every bootstrap replicate
([te_age_target.py:270](te_age_target.py#L270)) before the threshold is chosen,
and only then does the absolute distance override it
([:289](te_age_target.py#L289)). The advertised alternative to the
quantile-derived boundary therefore still runs the default 10,000-replicate
bootstrap in full.

Either skip the bootstrap when an absolute distance is supplied, or state in the
README that bootstrap distances are deliberately retained for context — the
current text ("Bootstrap distances are still produced for context") implies this
is intentional, so this may be a documentation-only item. Worth confirming which
it is, because on a 185,000-TE target the bootstrap is not cheap.

### 4.4 Bundle publication is overwrite-on-race

`_atomic_copy_file` stages through
`.{name}.publish.{os.getpid()}`
([sample_age_matched_controls.py:151](sample_age_matched_controls.py#L151)).
PIDs are not unique across compute nodes, so two tasks on different nodes can
compute the same staging path, and the final `os.replace` silently overwrites an
existing bundle.

Normal array mapping is collision-free — `divmod(task_id, chains)` assigns one
target/chain pair per task — so this requires an overlapping requeue, retry, or
manual resubmission of the same task. Deterministic seeds mean a legitimate
duplicate should be equivalent, which limits the blast radius. Still worth a
globally unique staging name (job id plus task id plus a random token) and a
refusal to replace an already-published bundle that has not been validated as
identical.

### 4.5 The chain seed is trusted, not re-derived

`_load_chain_result` reads `seed` straight from the bundle
([sample_age_matched_controls.py:127](sample_age_matched_controls.py#L127)) and
it is republished as authoritative provenance in `chain_seeds`
([:280](sample_age_matched_controls.py#L280)). It is never compared against
`derive_chain_seed(global_seed, target_digest, chain_index, algorithm_version)`,
which is deterministic and therefore free to recompute. One line closes the gap.

### 4.6 Test coverage for the distributed safety properties

The new tests prove the happy path publishes and gathers, and that first-set
row/CDF corruption is caught. Missing, roughly in order of how likely each is to
matter:

- corruption in the second or a later saved set (§4.1);
- a missing bundle producing a partial gather;
- bundles with mixed target, threshold, seed, config, store, or software
  identity;
- a bundle whose stored chain seed disagrees with the derived seed (§4.5);
- distributed `--resume` against both a valid and an invalid existing bundle;
- exact accepted-swap counts for non-integer sweeps at `n > 2` — the current
  fixed-sweep test asserts only that a tiny fixture requires one accepted swap;
- the `shared_with_reference` counter when a reference row leaves and later
  re-enters;
- a construction plateau (§3), asserting a prompt, well-labeled failure.

---

## 5. Documentation

The README now holds up well, and two of the fixes are better than what Round 4
asked for.

Addressed: the independence caveat is stated **up front** at lines 35-39 rather
than buried in a gate ("The 100 saved sets are correlated Monte Carlo states,
not 100 independent data replicates"); the new "Using the 100 matched sets"
section (289-303) supplies the downstream recipe that was entirely absent and
explains `reuse_row_indices.npy` / `reuse_counts.npy`; "the construction state
itself is not saved" replaces the false "discarded"; the threshold-as-
specification point is stated crisply at 202-205 and matches the measurements in
§2; the output listing, macOS test note, `0.2.0` version guidance, and a
Document map are all in place. `SWAP_SAMPLER_HPC_HOWTO.md` and the plan were
updated in the same commit.

Two accuracy items remain:

1. **Lines 278-280 overstate CDF validation** — see §4.1. Either weaken the
   sentence or, preferably, strengthen the code to match it.
2. **Nothing documents the `--search-bin-width` interaction** of §3, which is
   the defect most likely to stop a production run.

One smaller note: `schema_version` in the published metadata is still
`"swap-age-matched-controls-v1"` although the algorithm changed materially
(fixed sweeps, 10 x 10, distributed gather). The array *schema* genuinely did not
change and `config.algorithm_version` records the algorithm, so this is
defensible — but a consumer filtering on `schema_version` alone cannot tell v1
outputs from v2 outputs.

---

## 6. Suggested order

1. **§3** — the coarse-grid plateau. It is the only defect here that stops a
   production run outright, and the adaptive-refinement fix is small.
2. **§4.1** — recompute all saved-set CDFs, and add the `[1, 0]` corruption
   test. This also makes the README's existing claim true.
3. **§4.2 and §4.5** — both are small, and both are about not trusting a bundle
   that says the right things about itself.
4. **§5** — the two documentation corrections, once §3 and §4.1 have settled.
5. **§4.4, §4.3, §4.6** — worthwhile, not urgent.

The trajectory across rounds 4 and 5 is good: every Round 4 finding was either
fixed or partially fixed, the statistical core moved from a path-dependent
stopping rule to fixed sweeps with published mixing diagnostics, and the
documentation now states its own limitations accurately. What remains is
mostly validation depth rather than design.
