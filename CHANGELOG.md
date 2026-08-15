# Changelog

## After v0.1.0

- Changes the target-builder, manifest runner, SLURM wrapper, and legacy
  threshold fallback from the bootstrap 95th percentile to the bootstrap
  median (`0.50`).
- Keeps the acceptance quantile configurable for explicitly labeled
  sensitivity analyses.
- Repeats the 4,061-SNP in-gene validation with 10,000 bootstraps and 100 fresh
  matched sets under the new default.

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
