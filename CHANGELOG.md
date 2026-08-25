# Changelog

## v0.4.0 — 2026-08-25

Validation of bootstrap-target matching on the 75-draw production store, and the
CLI simplification that followed. **Breaking**: several options were removed and
one became required, so v0.3.1 commands do not run unchanged.

### Matching is now validated and is the reported stage

- Every acceptance gate in `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §10 is closed or
  superseded on the 75-draw store. `BOOTSTRAP_HPC_VALIDATION.md` carries the
  evidence; `BOOTSTRAP_DISCARDED_APPROACHES.md` records what was tried and
  rejected, so a rejected idea is not re-proposed without its numbers.
- `--disjoint-replicates` gives every replicate its own controls: 406,700 unique
  of 406,700 slots, maximum reuse 1, against 260,182 for the hard-q50 sampler.
- The swap screen is now geometric. A uniform 20,000-generation screen put
  50.06% of the target age distribution in its first cell, so the optimizer
  could not see the young end; fixing it took the relative age error at the 10%
  CDF quantile from 21.9% to −0.0%, with no cost at the old end, and cut runtime
  from 3.27 h to 2.18 h.
- Initialization is a stratified draw from the target's own equal-mass age
  strata, so the hard-q50 sampler is no longer part of the pipeline. Measured
  cost against a seeded start: +2.8 generations on `E_r`, 0.2% of the acceptance
  threshold, with identical QC and concordance.
- The absolute QC cap scales with the target's acceptance threshold instead of a
  fixed 500 generations, which rejected 15 of 20 replicates at 600 sites purely
  on target size.

### Φ-SFS polarity comes from the ARG, not the VCF

- `--ancestral-table` is **required**; `--ancestral-mode` and `--ancestral-info`
  are removed. This dataset's input VCF is unpolarized — 31% of sites are
  REF/ALT-swapped against the ARG's own calls — so the old REF default would
  have mis-polarized about a third of every spectrum silently.
- TE sites are polarized biologically; control SNPs by a posterior-weighted
  mixture `p·h(k,n) + (1−p)·h(n−k,n)`, linear in `p` and therefore unbiased at
  any draw count. Majority rule is deliberately not used.
- `build_ancestral_states.py` builds the table from the posterior draws, with a
  validated merge and atomic publication.
- Φ-SFS binds the table to the store by content digest. **Breaking**: bundles
  recording a null store digest are now rejected, because the digest is the
  table's only identity and accepting null accepts any table.

### Removed options

`--seed-sets`, `--closest-restarts`, `--diverse-restarts` (collapsed into
`--restarts`), `--search-grid-spacing`, `--selection-tolerance`, `--distance`,
`--log-age-offset`, `--ancestral-mode`, `--ancestral-info`.

### Operational

- Three launchers cover the scheduler stages: `run_bootstrap_matching.sbatch`,
  `run_ancestral_table.sbatch`, `run_phi_sfs.sbatch`. Ten experimental launchers
  were retired.
- The resume identity hashes the loaded project modules, so two different sets of
  uncommitted edits on one commit no longer share an identity.

### Known and accepted limitations

- The optimizer prefers well-dated SNPs, and ARG dating confidence is the same
  axis as polarity confidence. Both arms end up matched (controls mean `p`
  0.9786, TE target 0.9785), so it does not bias the comparison, but controls are
  drawn from the better-resolved part of the ARG.
- ARG polarity confidence is overconfident: where all 75 draws agree, it is right
  about 91% of the time against TE ground truth. `p` is used uncalibrated.
- The 100 replicates share no control rows, which is not statistical
  independence. Their spread measures how far Φ moves under age-CDF uncertainty
  conditional on the observed TE sites, and is not a confidence interval.


## v0.3.1 — 2026-08-16

- Adds `bootstrap_target_matcher.py`, which assigns every control replicate a
  reproducible bootstrap TE age CDF and minimizes exact-grid Wasserstein-1
  distance to that target rather than sampling against one fixed q50 boundary.
- Uses prespecified stratified restarts, improvement-only SNP swaps, relative
  material-improvement convergence, and certified best-state output.
- Records bootstrap counts and targets, complete restart traces, selected and
  per-restart rows/CDFs, optimizer QC, all three paired W1 distances, matching
  error ratios, triangle checks, coordinate arrays, and SNP-reuse diagnostics.
- Adds provenance-locked, atomically written per-replicate work bundles so an
  interrupted optimization can resume without redrawing bootstrap targets or
  changing restart identities.
- Documents that matching-error thresholds diagnose optimizer convergence,
  while scientific validation requires SNP-to-observed distances to reproduce
  bootstrap-target-to-observed distances across the center and tails.
- Makes bootstrap-target matching the recommended matching stage and retains
  hard-q50 matching for reproducing earlier analyses. `--disjoint-replicates` is
  the production setting: it guarantees that the published sets share no control
  rows, which is not the same as statistical independence and does not give an
  effective replicate count of 100.
- Records the remaining gates, all downstream of matching: the
  polarity-confidence extension of the W1-repair/SFS diagnostic, an effective
  replicate count estimated from the Phi-SFS scores themselves, and the
  chromosome 1-9 input VCF. The derived-frequency arm of that diagnostic is
  complete.

Round 7 review (`CODE_REVIEW_ROUND7.md`) fixes, folded in before release:

- Makes bootstrap-target bundles readable by `phi_sfs.py`. The matcher hashed
  `target_digest` over three arrays while Φ-SFS recomputes the established
  four-array digest including the acceptance threshold, so every bundle was
  rejected; and the bundle published none of the per-replicate identifier
  arrays Φ-SFS loads. Both are covered by an end-to-end regression test.
- Publishes `replicate_id.npy` rather than fabricating `chain_index` and
  `sample_index`. Bootstrap replicates have no chain structure, so those
  columns would assert a within-chain correlation that does not exist.
  `phi_sfs.py` now selects identifier arrays from the bundle's
  `schema_version` and carries them into every output.
- Screens swaps on a coarse age grid with exact-grid certification of every
  recorded distance, the same two-tier device `swap_control_sampler.py` uses.
  The exact grid spans about 22,900 points for the production store; measured
  3.2× faster on a synthetic production-scale grid with an identical
  exact-grid result.
- Accumulates bootstrap target CDFs in float64 over float32-stored per-site
  rows. A float32 accumulation over tens of thousands of TE sites displaced
  the bootstrap target and every distance derived from it.
- Records the interval store directory as `source_store`. The previous
  `getattr(store, "path", "")` fallback resolved to the repository directory
  for every run, because interval stores expose `store_dir`.
- Pins the checkout and NumPy version in the resume identity, so `--resume`
  cannot silently combine replicate bundles produced by two implementations.
- Rejects `--output` equal to or nested inside `--work-dir`, which previously
  published a result and then deleted it while reporting success.
- Publishes bootstrap and restart seeds, per-restart distances, ratios, QC,
  runtimes, and per-epoch proposal counts.
- Documents which matching workflow is primary, shows the Φ-SFS command for both
  bundle types, and brings the estimand caveats and the W1-repair-versus-SFS
  bias risk from the plan into the README.

Post-release CLI simplification, folded into the same version:

- Removes `--seed-sets`, `--closest-restarts` and `--diverse-restarts` from
  `bootstrap_target_matcher.py`. Restarts are now stratified draws from the
  target's own equal-mass age strata and `--restarts` sets their count, so the
  §5 hard-q50 sampler is no longer a prerequisite for §6. Measured on the
  same 100 bootstrap targets for the 4,067-site in-gene set, the median paired
  proportional increase in `E_r` is 5.62%. The two marginal medians are 52.10
  against 50.09 generations (a 4.03% ratio of medians), on a 1,480-generation
  acceptance threshold, with identical QC (100/100) and concordance
  (`cor(B_r, O_r)` 0.99984 against 0.99986). The paired statistic is the stated
  cost estimand; the marginal medians are retained to make the distinction
  explicit.
- Removes `--search-grid-spacing`: the coarse swap screen is always a geometric
  sub-sample of the exact grid. A uniform 20,000-generation screen put 50.06% of
  the in-gene target's age mass in its first cell, giving +21.9% relative age
  error at the 10% CDF quantile against -0.2% for the geometric screen.
- Removes `--distance` and `--log-age-offset` from `bootstrap_target_matcher.py`
  and `te_age_target.py`; W1 is always linear.
- Makes `--ancestral-table` required in `phi_sfs.py` and removes
  `--ancestral-mode` and `--ancestral-info`. Polarity is inferred by SINGER and
  read off the ARG, so it cannot come from the VCF: 31% of chromosome-10 sites
  are REF/ALT swapped relative to the ARG's calls.
- Replaces ten experimental SLURM launchers with one parameterised production
  launcher, `run_bootstrap_matching.sbatch`.

## v0.3.0 — 2026-08-16

- Adds `phi_sfs.py`, a downstream step that compares the unfolded SFS of a
  target TE set with each of its 100 age-matched SNP control sets. Allele
  counts come from one polarized biallelic VCF; eligible sites are those with
  at least 20 callable inbred individuals, each projected to 20 by exact
  hypergeometric expectation over bins 0 through 20, with bins 1 through 19
  retained and never renormalized per site.
- Defines Φ-SFS as the total variation distance between the two projected,
  normalized spectra, computes three equivalent forms, and reports their
  disagreement as `identity_max_abs_error`.
- Requires the matched-control bundle to have been built from the supplied
  target: `target_digest` is recomputed from the target directory with the
  matcher's own loader and hash helper and must match. Store hashes alone
  cannot establish this, so a control bundle from another TE category was
  previously accepted silently.
- Validates row-index and coordinate arrays for shape alignment, integer
  dtype, non-negativity, and within-set duplicate controls, and rejects a
  matched bundle that is not marked complete.
- Reports `retained_fraction` and `endpoint_fraction` per set and for the
  target, because final normalization hides differences in eligible-site count
  and total retained polymorphic mass.
- Records `release_provenance.software_provenance()`, the creation command,
  NumPy version, and creation time in the result metadata, matching every
  other durable output in the pipeline.
- Scans the VCF about five times faster on a representative 200-sample file:
  CHROM and POS are parsed with a bounded split before the coordinate test,
  genotypes are memoized, sites are projected once per distinct `(k, n)` pair
  and gathered per set by weighted matrix product, and the VCF digest is
  accumulated during the single scan instead of a second full read. Published
  arrays are unchanged to within 1e-13.
- Documents the input assumptions the step does not re-derive: records are
  assumed biallelic, the FILTER column is ignored, and ancestral alleles are
  compared case-sensitively so a lowercase low-confidence call is rejected
  rather than silently folded.
- Accepts `.bgz` and `.bgzf` input, reports scan progress, and adds `--quiet`.

## v0.2.1 — 2026-08-15

- Adaptively halves a plateaued coarse construction grid down to exact target
  resolution, then fails clearly after three exact-grid plateau epochs instead
  of exhausting the full construction budget.
- Adds a full SHA-256 interval-store content identity and requires matching
  store and target identities for distributed chain and gather jobs.
- Recomputes every saved-set CDF from its row indices and re-derives every
  deterministic chain seed before resume, publication, or gather.
- Publishes chain bundles through globally unique staging names and an atomic
  no-overwrite claim so overlapping retries cannot replace completed bundles.
- Adds adversarial tests for later-set corruption, missing bundles, invalid
  seeds, store mismatches, non-integer sweep counts, overlap bookkeeping,
  construction refinement, and exact-grid plateau errors.
- Clarifies that the manifest `target` field is a durable output directory and
  that an absolute acceptance distance deliberately retains bootstrapping for
  context.
- Repeats the complete 10-by-10 in-gene pilot under version 0.2.1: all bundles
  and all 100 row-derived CDFs validate, with matched W1 1,755.83--1,904.84 at
  the 1,905.10 threshold.

## v0.2.0 — 2026-08-15

- Changes the production layout from four 25-state chains in one job to ten
  independent 10-state chains, each runnable as its own SLURM task.
- Keeps interval-store copies, checkpoints, CDF caches, and result assembly on
  node-local `$TMPDIR`; atomically publishes validated completed-chain bundles
  and final result directories to Quobyte before scratch disappears.
- Replaces path-dependent membership-crossing save rules with one fixed
  accepted-swap sweep for burn-in and one sweep between saves. Membership
  replacement remains a reported diagnostic.
- Adds durable chain identity, row eligibility/target-exclusion checks,
  row-to-CDF recomputation, truncated-output rejection, and explicit
  `--all-eligible` versus `--candidate-rows` selection.
- Adds an absolute Wasserstein tolerance option while retaining the bootstrap
  median as the default matching specification.
- Adds chain-level membership overlap, W1 autocorrelation, and an explicitly
  heuristic overlap-based effective-sample-size summary.
- Changes the target-builder, manifest runner, SLURM wrapper, and legacy
  threshold fallback from the bootstrap 95th percentile to the bootstrap
  median (`0.50`).
- Keeps the acceptance quantile configurable for explicitly labeled
  sensitivity analyses.
- Repeats the 4,061-SNP in-gene validation with 10,000 bootstraps and 100 fresh
  matched sets under the new default: all ten bundles validated, matched W1 was
  1,761.08--1,904.94 at the 1,905.10 threshold, and adjacent membership
  replacement averaged about 61%.

## v0.1.0 — 2026-08-15

First tagged age-matched control-sampler release.

- Generates 100 matched SNP sets per TE dataset with four independent chains
  and 25 saved states per chain.
- Uses exact full-grid Wasserstein certification, 50% construction-state
  replacement during burn-in, and 25% replacement between saved states.
- Runs many TE datasets through restartable SLURM arrays, stages the immutable
  interval store from Quobyte once per array shard, and uses four CPU workers
  per target.
- Publishes atomic result directories with diagnostics, reuse summaries,
  release version, Git commit, exact tag, and dirty-checkout status.
- Validated locally on the two available ARG draws for 100 control sets of the
  4,061-eligible-SNP in-gene target at both the bootstrap 95th-percentile and
  median constraints.
