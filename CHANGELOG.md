# Changelog

## v0.3.1 — 2026-08-16

- Adds `bootstrap_target_matcher.py`, which assigns every control replicate a
  reproducible bootstrap TE age CDF and minimizes exact-grid Wasserstein-1
  distance to that target rather than sampling against one fixed q50 boundary.
- Uses prespecified closest and diversity restarts, improvement-only SNP swaps,
  relative material-improvement convergence, and certified best-state output.
- Records bootstrap counts and targets, complete restart traces, selected and
  per-restart rows/CDFs, optimizer QC, all three paired W1 distances, matching
  error ratios, triangle checks, coordinate arrays, and SNP-reuse diagnostics.
- Adds provenance-locked, atomically written per-replicate work bundles so an
  interrupted optimization can resume without redrawing bootstrap targets or
  changing restart identities.
- Documents that matching-error thresholds diagnose optimizer convergence,
  while scientific validation requires SNP-to-observed distances to reproduce
  bootstrap-target-to-observed distances across the center and tails.
- Retains hard-q50 matching as a sensitivity analysis and records the remaining
  production gates: linkage-aware bootstrap selection, full-posterior
  calibration, W1-repair versus SFS bias checks, effective replicate counts,
  and distributed HPC execution.

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
