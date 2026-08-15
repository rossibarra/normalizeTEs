# Changelog

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
