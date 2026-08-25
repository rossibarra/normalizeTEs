# Code Review Round 8 — production wiring, bootstrap resume, and Φ-SFS polarity

**Scope:** the current working tree, including uncommitted matcher/Φ-SFS changes,
tracked launch scripts and documentation, and visible untracked workflow artifacts.
The review emphasized correctness, reproducibility, stale code, avoidable work, and
whether the documented production workflow can actually be executed.

**Bottom line:** the Round 7 matcher→Φ-SFS interface is repaired and its focused
regression suite passes, but the repository is not production-ready. The documented
primary command and every tracked bootstrap-target SLURM launcher use arguments that
the current CLI has removed. More importantly, the new ancestral-polarity input is
not bound to the target/store provenance, and its merge/publish path can silently
combine incompatible data or overwrite a valid table.

## What was verified

- `python -m pytest -q tests/test_bootstrap_target_matcher.py
  tests/test_phi_sfs.py tests/test_te_age_target.py
  tests/test_swap_control_sampler.py` passes: **74 passed in 31.85 s**.
- Bootstrap-target bundles now publish `replicate_id.npy`, use the established
  four-array target digest, and are accepted by `phi_sfs.py`.
- The matcher now accumulates bootstrap CDFs in float64, records the actual interval
  store path, rejects dangerous output/work nesting, and certifies reported W1
  distances on the exact grid.
- `python -m pytest -q` does **not** run the suite: collection fails because
  `results/snapshot-seedtest/test_snp_age_distribution.py` collides with the real
  root test module. The README's explicit `python -m pytest ...` command avoids that
  particular import mode, but the unrestricted discovery tree remains fragile.

---

## 1. CRITICAL — the recommended workflow and all tracked bootstrap launchers are unexecutable

`bootstrap_target_matcher.py:1045-1092`; `README.md:348-426`; ten tracked
`run_*.sbatch` files

The current matcher accepts `--restarts` and always uses linear W1 with a geometric
coarse grid and stratified initialization. It no longer accepts:

- `--seed-sets`
- `--closest-restarts` / `--diverse-restarts`
- `--distance` / `--log-age-offset`
- `--search-grid-spacing`
- `--selection-tolerance`

Nevertheless, the README's recommended stage-4 command uses the first three removed
options (`README.md:384-399`), says stage 3 is still required to supply
`--seed-sets` (`:20-26,365-368`), and advertises `--search-grid-spacing log` as a
production flag (`:28-31,356-363`). All ten tracked SLURM files that invoke
`bootstrap_target_matcher.py` also pass removed arguments:

`run_deep_bootstrap.sbatch`, `run_disjoint.sbatch`,
`run_logage_disjoint.sbatch`, `run_loggrid_disjoint.sbatch`,
`run_option_d.sbatch`, `run_scale_probe.sbatch`, `run_size_sweep.sbatch`,
`run_small_probe.sbatch`, `run_t5_bootstrap.sbatch`, and
`run_t5r6_bootstrap.sbatch`.

`run_logage_disjoint.sbatch` is doubly broken because it also passes the removed
`--distance` and `--log-age-offset` arguments to `te_age_target.py` (`:36-48`).
These jobs fail in argument parsing before scientific work starts.

The documentation also contradicts itself about release status. README :20-21 and
:350 call bootstrap matching the reported/recommended result, while README :126-127
says v0.3.1 is not cleared for production and `BOOTSTRAP_HPC_VALIDATION.md:1462-1464`
says that banner must remain. `CHANGELOG.md:53-56` claims the README labels the
method experimental and pilot-only, which it does not.

**Fix:** decide the supported CLI and status first. Update one production launcher,
make it the canonical documented command, and move/delete the obsolete experimental
launchers rather than maintaining ten dead variants. Add a test that extracts or
constructs every supported command and runs `parse_args` against the current CLI.

---

## 2. HIGH — Φ-SFS does not authenticate or validate its ancestral-polarity table

`phi_sfs.py:645-665`; `build_ancestral_states.py:206-220`;
`tests/test_phi_sfs.py:250-263`

`build_ancestral_states.py` records the source store schema, content SHA-256, and row
count. Φ-SFS checks only:

```python
if store_meta.get("schema_version") != "ancestral-state-counts-v1": ...
```

It then trusts `store_meta["store"]`, opens that store's positions, and consumes the
two table arrays. It never compares `store_content_sha256` with the already-validated
target/match store identity, never checks `store_rows`, never checks the declared
`bases` order, and never validates the count-array shapes or integer dtypes. The test
fixture omits all of this provenance and is intentionally accepted.

A table from another store with overlapping row coordinates can therefore produce a
complete, plausible Φ-SFS output with the wrong control-site polarization. This is a
silent scientific-integrity failure, not merely weak metadata.

The stored path is also written as `str(args.store)` rather than a resolved path
(`build_ancestral_states.py:209`), then interpreted relative to Φ-SFS's current
working directory. A table built with a relative store path is not portable even
with its own output directory.

**Fix:** bind the table to `target_meta`/`match_meta` using a required non-null store
content digest for interval-store runs; validate schema, bases exactly `A,C,G,T`, row
count, array shapes `(n_rows, 4)` and `(n_rows,)`, unsigned/integer dtypes, and
`counts.sum(axis=1) <= present_draw_count`. Record a resolved source-store path, but
use the digest—not the path—as identity. Add wrong-store, reordered-column,
truncated-array, and relative-path tests.

---

## 3. HIGH — ancestral-table merge and publication can silently corrupt results

`build_ancestral_states.py:140-149,167-221`

Merge mode sums every directory named on `--merge` without reading any part's
metadata. It does not reject:

- parts built from different stores or chromosome modes;
- duplicate part paths;
- overlapping `--draws` slices, or missing slices;
- a previously merged table supplied as though it were one part;
- mismatched software/schema/base ordering.

Because accumulation uses `uint16`, sufficiently large or duplicated merges also
wrap modulo 65,536 without an overflow check. Production currently uses only 75
draws, so overflow is not the immediate risk; unnoticed overlap or mixed stores is.

Publication is unsafe independently of merge correctness. `_save()` creates the
output with `exist_ok=True`, replaces `ancestral_counts.npy`, then
`present_draw_count.npy`, then overwrites `metadata.json`. Re-running against an
existing output destroys it without confirmation, and interruption between those
steps leaves a hybrid directory containing files from two runs while retaining a
valid-looking schema document.

**Fix:** give each part a durable identity containing store digest, chromosome,
explicit draw identities/digests, and parameters. At gather, require compatible
identities and an exact, duplicate-free expected draw set; accumulate in `uint64`,
range-check, then narrow if desired. Publish a new staging directory with
`os.replace`, refuse an existing output, and mark `complete: true` only in the final
metadata. Add unit and end-to-end tests—there are currently no tests for
`build_ancestral_states.py`.

---

## 4. HIGH — the resume identity still does not pin a dirty implementation

`bootstrap_target_matcher.py:872-895`; `release_provenance.py:30-43`;
`CHANGELOG.md:47-48`

Round 7 requested implementation-locked resume, and the matcher now puts
`software_provenance()` in `identity.json`. That is sufficient only for a clean
checkout. For a dirty checkout, provenance records the HEAD commit and a boolean/
description ending in `-dirty`; it does not hash the diff or source files.

Consequently, edit A and edit B made on the same dirty HEAD produce the same software
identity. A long job can resume after arbitrary uncommitted matcher changes and mix
replicate bundles from two implementations, exactly the failure the new identity
claims to prevent. This working tree is currently dirty, so the gap is operational,
not hypothetical. The regression test changes the recorded commit, but never tests
two different dirty source states (`tests/test_bootstrap_target_matcher.py:150-167`).

**Fix:** for production, reject resume and preferably initial execution when
`git_dirty` is true. If dirty development resume must be supported, include a SHA-256
of the tracked diff plus relevant untracked source files (or a manifest of hashes for
all executable modules) in the run identity. Test that changing one source byte on a
dirty HEAD invalidates resume.

---

## 5. HIGH — disjoint controls do not make the effective replicate count 100

`bootstrap_target_matcher.py:914-928,1010-1017,1085-1089`;
`README.md:55-61,356-363`

The code and docs repeatedly claim that removing all previously selected controls
makes the effective replicate count 100 "by construction." Disjoint membership only
removes direct SNP reuse. Later sets are sampled from a candidate universe determined
by earlier selected sets, so the sets are coupled by sequential depletion (sampling
without replacement induces dependence). They also all depend on the same observed
TE sample and store. Independence and effective sample size do not follow from zero
overlap.

The surrounding prose is internally inconsistent: README :59-61 says stage-4
replicates are not independent because they share control SNPs, although the
recommended `--disjoint-replicates` run has maximum reuse one. `phi_sfs.py:48-54`
also calls bootstrap-target replicates independent when the correct schema property
is only that they have no chain index.

This overstatement can directly understate uncertainty in downstream Φ-SFS summaries.

**Fix:** replace every independence/ESS claim with the narrower guarantee: published
sets contain no shared control rows. Describe the sequential-depletion dependence and
estimate ESS/correlation using the actual downstream statistic. Keep `replicate_id`—
it is the right identifier—but describe it as "no chain structure," not independence.

---

## 6. MEDIUM — candidate-row publication overwrites inputs/results and is not atomic as a pair

`build_candidate_rows.py:122-174`

Unlike the target, matcher, and Φ-SFS writers, this CLI never refuses an existing
`--output` or `--report`. The NPY is atomically replaced and the JSON is then written
in place. A failed JSON write leaves a new candidate array beside an old/missing
report, and a re-run silently destroys the prior candidate universe.

Worse, `--report` may equal `--output`: the code writes the NPY and immediately
replaces its contents with JSON, while still printing that both were written. The
same unrestricted paths can alias an input position list.

**Fix:** reject existing outputs and all input/output/report aliasing after resolving
paths. Stage both artifacts in a temporary directory (or as two sibling temporaries),
then publish only after both are complete. Add overwrite and path-collision tests.

---

## 7. MEDIUM — generated snapshots and backup/experiment files break discovery and obscure the supported code

The visible tree contains `results/snapshot-seedtest/`, a 688 KiB copy of project
source including a duplicate `test_snp_age_distribution.py`. Plain `pytest -q` walks
into it and fails collection with an import-file mismatch. There are also untracked
`README.md.bak2`, `SWAP_SAMPLER_HPC_HOWTO.md.bak2`, the untracked
`run_init_comparison.sbatch`, and numerous ignored `*.pre-*.bak` source copies in the
repository directory.

The snapshot exists to support an obsolete seeded-vs-stratified comparison and is
referenced only by `run_init_comparison.sbatch`, which itself calls the removed
matcher CLI. It is prior-work residue, not a current runtime dependency.

**Fix:** archive experimental evidence outside the source/test discovery tree or
store only immutable commit IDs and result summaries. Remove the obsolete local
copies after confirming no recovery need, ignore `.bak2` and generated snapshots,
and add a pytest configuration with explicit `testpaths = tests` plus the intentional
root unittest path handled separately. CI should use one canonical command.

---

## 8. LOW — dead INFO-polarity code and stale release prose remain

`phi_sfs.py:9-20,319-326,473-480`; `CHANGELOG.md:47-56`

Φ-SFS no longer reads ancestral state from INFO, but `_parse_info()` is unused, the
VCF parser still binds an unused `info` variable, and the module docstring still says
lowercase INFO ancestral alleles are rejected. `PolarityResolver` stores
`present_draw_count` as `_present` but never reads it; either use it for the table
contract checks in finding 2 or stop carrying it in the resolver.

The ancestral builder calls `present_draw_count` the number of draws in which a site
appeared (`build_ancestral_states.py:8-14`), but increments it only for resolved sites
whose ancestral state is a single uppercase A/C/G/T (`:108-123`). It is therefore a
usable-ancestral-call count, not a raw presence count. That distinction should be in
the array name/metadata because the documentation makes a missingness assumption from
the stronger, currently false definition.

The changelog also says the matcher publishes `seed_sets_digest` even though seed
sets were removed, and says the docs label the method experimental when they label it
recommended. Changelogs should describe released behavior, not an intermediate
implementation that no longer exists.

**Fix:** delete `_parse_info` and the unused binding, rewrite the input assumptions
for table-based polarity, and reconcile v0.3.1 notes with the final supported CLI.

---

## Test gaps and recommended repair order

1. Repair the canonical CLI/docs/launcher contract (finding 1); otherwise no
   production run can start.
2. Authenticate and validate ancestral tables in Φ-SFS (finding 2).
3. Make ancestral part merge and publication provenance-safe and atomic (finding 3).
4. Close dirty-checkout resume mixing (finding 4).
5. Correct the disjoint/ESS claims before interpreting Φ-SFS uncertainty (finding 5).
6. Harden candidate publication and clean generated source snapshots (findings 6-7).
7. Remove stale code/prose and add tests for both new builder paths (finding 8).

Besides the absent ancestral-builder tests, current matcher tests do not cover a
dirty-to-different-dirty resume, invalid `--init-oversample`, malformed target
boundary/quotas arrays, or disjoint-resume dependence/provenance beyond a tiny two-site
fixture. Candidate-builder tests are absent entirely. The focused 74-test pass is
useful regression evidence, but it does not exercise the unsafe boundaries above.
