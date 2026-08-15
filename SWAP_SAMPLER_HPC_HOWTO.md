# Swap-chain age matching on Farm/Quobyte

This runbook produces 100 posterior-age-matched SNP sets for every TE dataset.
The production layout is ten independently seeded chains with ten saved states
per chain. Each chain is a separate one-CPU SLURM array task; no node holds all
ten chain caches.

Quobyte holds one canonical interval store, target directories, completed chain
bundles, and final results. Every active store copy, checkpoint, CDF cache, and
result-assembly directory lives under the job's `$TMPDIR` on `/local/scratch`.
Do not use `$SLURM_TMPDIR`; Farm does not set it.

## 1. Preconditions

From the repository root:

```bash
git fetch --tags
git checkout COMMIT_HASH
conda env create -f environment.yml       # once
conda activate normalizeTE
python -m pytest -q tests test_snp_age_distribution.py
```

One interval-store audit test requires Linux `fork` and fails deliberately on
macOS when asked to use two workers. Run production validation on a Linux
compute node.

Pin an exact commit or immutable release tag and require a clean checkout. The
canonical interval store must already be complete and validated; never modify
it while target or chain jobs are running.

Every TE file is whitespace-delimited chromosome plus 1-based VCF position:

```text
1 57396
1 92776
2 105421
```

## 2. Create the manifest

Create one tab-delimited manifest on Quobyte with absolute paths:

```text
label	positions	target	output	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young	/quobyte/project/matches/young	1003
```

Requirements:

- labels are unique and contain only letters, digits, dots, dashes, and
  underscores;
- every target and final output path is unique;
- seed is an integer and unique seeds are recommended across targets;
- position, target, and output paths are durable Quobyte paths; and
- a complete existing output is skipped, while an incomplete or incompatible
  output stops the task.

Let `T` be the number of non-header manifest rows. With the defaults:

```text
target tasks = T
chain tasks  = 10 * T
gather tasks = T
```

Flat chain task `k` maps to manifest row `k // 10` and chain `k % 10`.

## 3. Resource model

### Target jobs

Target construction stages the interval store and creates a temporary float32
TE-by-age CDF matrix under `$TMPDIR`. For about 185,000 TEs at the measured age
range, allow roughly 16--18 GiB beyond the staged store. The supplied launcher
starts at one CPU and 96 GiB.

### Chain jobs

One chain keeps a selected-row float64 CDF cache of approximately
`8 * n * B` bytes for target size `n` and exact grid length `B`. At `n=35,000`
and `B=22,000`, this is about 5.7 GiB. Each chain task requests one CPU and 32
GiB; replace that provisional request with measured Linux `MaxRSS`.

Ten chain tasks may run on different nodes. Each stages the same immutable
interval store into its own `$TMPDIR/interval_store`. This duplicates temporary
copies across nodes but never creates multiple permanent stores.

### Gather jobs

Gathering stages the store once more so it can independently recompute one CDF
from the saved row indices in every chain. It assembles the final directory
under `$TMPDIR`, copies the complete result to a temporary sibling on Quobyte,
checks `metadata.json` for `complete: true`, and exposes it with a same-filesystem
atomic rename.

Node-local scratch must hold the complete staged store plus at least 20%. Every
launcher checks free space before `rsync`.

## 4. Build target distributions

Set the `#SBATCH --array` range in `build_age_targets.sbatch` to exactly `T`
elements. For ten targets use `0-9`.

```bash
mkdir -p /quobyte/project/normalizeTE/logs

export PROJECT=/quobyte/project/normalizeTE
export STORE=/quobyte/project/data/snp_interval_store
export MANIFEST=/quobyte/project/manifests/te_manifest.tsv
export T=3  # number of non-header manifest rows in this example
export AGE_MATCH_TASK_COUNT=$T

target_job=$(sbatch --parsable \
  --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  build_age_targets.sbatch)
```

Every target uses 10,000 bootstrap replicates and the bootstrap median by
default. The threshold is the operative matching-quality specification: the
feasible walk does not prefer a smaller W1 once a proposal remains inside it,
so saved distances normally concentrate near the boundary.

For a labeled quantile sensitivity analysis:

```bash
export ACCEPTANCE_QUANTILE=0.95
```

For a pre-specified absolute Wasserstein tolerance in generations:

```bash
export ACCEPTANCE_DISTANCE=1500
```

An absolute distance overrides the bootstrap quantile as the threshold, but the
bootstrap distribution is still saved. Never reuse a target output path across
tolerances.

Target resolution defaults to `error`. Set `MISSING_POSITION_POLICY=drop` only
when every exclusion in target `metadata.json` will be reviewed.

The launcher verifies `SLURM_ARRAY_TASK_COUNT == AGE_MATCH_TASK_COUNT`; a short
array cannot silently omit manifest rows.

## 5. Run ten independent chains per target

Wait for every target to complete. Set `sample_age_matches.sbatch` to exactly
`10*T` array elements (`0-29` for three targets), then submit:

```bash
export AGE_MATCH_CHAINS=10
export AGE_MATCH_SETS_PER_CHAIN=10
export AGE_MATCH_CHAIN_TASK_COUNT=$((AGE_MATCH_CHAINS * T))

chain_job=$(sbatch --parsable --dependency="afterok:${target_job}" \
  --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_CHAINS,AGE_MATCH_SETS_PER_CHAIN,AGE_MATCH_CHAIN_TASK_COUNT \
  sample_age_matches.sbatch)
```

Each task:

1. verifies the declared array size;
2. stages the canonical store to `$TMPDIR/interval_store`;
3. constructs and runs exactly one deterministic chain under `$TMPDIR`;
4. performs one fixed accepted-swap sweep per target member before the first
   save and between later saves;
5. validates row eligibility, target exclusion, uniqueness, stored distances,
   and a row-derived CDF;
6. writes one complete local compressed bundle; and
7. atomically copies and reloads that bundle at
   `OUTPUT.chains/chain-NNN.npz` before deleting scratch work.

A killed chain has no completed durable bundle and restarts from its deterministic
seed. A rerun with an existing bundle validates its schema, parameters, target
digest, candidate digest, store catalog, software provenance, rows, and CDF
before skipping it. Local checkpoints are diagnostic only and are not expected
to survive job termination.

The walk uses fixed accepted-swap counts rather than stopping the first time a
membership-replacement threshold is crossed. Replacement fractions are
reported diagnostics. This removes a path-dependent save rule and avoids an
`O(n log n)` set intersection on every proposal.

## 6. Gather and publish the 100 sets

Set `gather_age_matches.sbatch` to exactly `T` array elements and submit it with
an `afterok` dependency on the chain array:

```bash
export AGE_MATCH_GATHER_TASK_COUNT=$T

gather_job=$(sbatch --parsable --dependency="afterok:${chain_job}" \
  --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_CHAINS,AGE_MATCH_SETS_PER_CHAIN,AGE_MATCH_GATHER_TASK_COUNT \
  gather_age_matches.sbatch)
```

The gather task refuses to publish if any of the ten bundles is absent,
truncated, from another run, or inconsistent with its row indices. It writes
all active assembly files under `$TMPDIR`; only the ten completed chain bundles,
a short-lived publication copy, and the final result reside on Quobyte.

Chain bundles remain after publication so the final output can be regenerated
without rerunning chains. Remove `OUTPUT.chains/` only after final validation
and any required archival period.

## 7. Local single-target command

For debugging without distributed SLURM tasks:

```bash
python sample_age_matched_controls.py \
  --store "$TMPDIR/interval_store" \
  --target /quobyte/project/targets/in_gene \
  --all-eligible \
  --output /quobyte/project/matches/in_gene \
  --work-dir "$TMPDIR/match-in-gene" \
  --sets 100 \
  --chains 10 \
  --sets-per-chain 10 \
  --workers 1 \
  --seed 1002 \
  --burnin-accepted-sweeps 1 \
  --sample-accepted-sweeps 1
```

Increase `--workers` only when the node has enough memory for one selected-row
CDF cache per active worker. Use `--candidate-rows candidates.npy` instead of
`--all-eligible` to restrict the declared candidate universe.

With local `--work-dir`, `--resume` can reuse completed chains only while that
same scratch directory still exists. The production distributed workflow uses
durable completed bundles instead.

## 8. Final outputs

Each final output directory contains:

- `row_indices.npy`, shape `(100, n)`;
- `positions.npy`, `chromosome_codes.npy`, and `chromosome_labels.npy`;
- `cdfs.npy` and `wasserstein.npy`;
- `target_cdf.npy` and `age_bins.npy`;
- `chain_index.npy` and `sample_index.npy`;
- `diagnostics.csv` with construction, fixed-sweep, and save records;
- `reuse_row_indices.npy` and `reuse_counts.npy`; and
- `metadata.json` with complete provenance, seeds, chain histories, membership
  overlap, W1 autocorrelation, and an overlap-based ESS heuristic.

Final metadata must contain:

```json
{"schema_version": "swap-age-matched-controls-v1", "complete": true}
```

The schema describes the directory layout; the embedded sampler config records
the version-2 fixed-sweep algorithm. Before downstream analysis, require the
expected Git commit and reject `git_dirty: true`.

## 9. Using the matched sets

Compute the downstream statistic once for every row of `row_indices.npy`.
Preserve `chain_index.npy` and `sample_index.npy` when constructing the matched
null distribution. The 100 states are not guaranteed to be independent;
correlation is within each ten-state chain.

For each scientific statistic:

1. plot values against within-chain sample order;
2. estimate within-chain autocorrelation and effective sample size;
3. compare chain means and ranges;
4. inspect `reuse_counts.npy` for control-SNP concentration; and
5. report Monte Carlo uncertainty using the effective, not nominal, replicate
   count.

The generic membership and W1 diagnostics cannot prove that another statistic
has mixed. If effective sample size is inadequate, increase
`--sample-accepted-sweeps` or the number of independent chains and regenerate
the target's matched sets without changing its acceptance threshold.

## 10. Production gates

Before launching every category on the 75-draw store:

1. run approximately 500-, 4,000-, and 35,000-SNP targets;
2. confirm all ten chain bundles and all 100 exact distances pass;
3. inspect membership replacement, W1 autocorrelation, downstream-statistic
   autocorrelation, overlap, reuse, and chromosome composition;
4. compare chain-level summaries for evidence of disconnected feasible regions;
5. record chain and gather `MaxRSS`, elapsed time, staged-store size, acceptance
   rate, and proposals per accepted sweep;
6. re-measure q50 feasibility on the complete 75-draw store; and
7. adjust memory, time, and array concurrency from those measurements.

### Two-draw fixed-sweep pilot

The 4,061-SNP in-gene target was rerun locally with the version-2 defaults:
ten independently seeded chains, ten saved states per chain, one accepted-swap
sweep for burn-in, and one sweep between saves. All ten durable bundles passed
reload and row-derived CDF validation, and gather published all 100 sets.

- q50 threshold: 1,905.10 generations;
- matched-set W1: 1,761.08--1,904.94, median 1,881.25;
- mean adjacent membership replacement: 0.6089--0.6115 by chain;
- W1 lag-one correlation: -0.519--0.081 by chain;
- membership-overlap AR(1) ESS heuristic: 43.9 total;
- 260,354 unique controls across the 100 sets, with maximum reuse 13.

Each correlation estimate has only nine adjacent pairs and the ESS is an
explicitly crude membership heuristic. This pilot shows that the distributed
fixed-sweep implementation works on the available two-draw store; it does not
establish mixing for a scientific downstream statistic or for the 75-draw
store. Treat the full-store gates above as required before production claims.
