# Swap-chain age matching on Farm/Quobyte

This guide runs many independent TE datasets and produces 100 posterior-age
matched SNP sets for each dataset. The defaults are four independent chains,
25 saved sets per chain, 50% membership replacement during burn-in, and 25%
replacement between saved states.

The interval store and final targets/results live on Quobyte. Each SLURM array
task copies the interval store once to its job-local `$TMPDIR` under
`/local/scratch`, processes several manifest rows, and writes only durable
targets/results back to Quobyte. Do not use `$SLURM_TMPDIR`; it is not set on
Farm.

## 1. Preconditions

From the repository root:

```bash
git fetch --tags
git checkout COMMIT_HASH
conda env create -f environment.yml       # once
conda activate normalizeTE
python -m pytest -q tests
```

Pin an exact commit or immutable tag for production rather than a moving
`main`. `v0.1.0` is the q95 baseline; the q50 default is a follow-up commit and
therefore must be pinned by its commit hash. Do not run with a dirty checkout
and treat it as a tagged release.

The canonical interval store must already be complete and validated. Never
modify it while matching jobs are running.

Every TE position file is whitespace-delimited with chromosome and 1-based
position columns:

```text
1 57396
1 92776
2 105421
```

## 2. Create the manifest

Create one tab-delimited manifest on Quobyte. Paths should be absolute so array
tasks do not depend on their launch directory:

```text
label	positions	target	output	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young	/quobyte/project/matches/young	1003
```

Requirements:

- labels are unique, begin with a letter or digit, and otherwise contain only
  letters, digits, dots, dashes, or underscores;
- every target and output path is unique;
- targets and outputs must be unique; complete existing outputs are skipped,
  while ambiguous or incomplete existing outputs stop the task;
- seed is an integer; and
- position, target, and output paths point to Quobyte, not node-local scratch.

The runner assigns row `i` to array task `i % AGE_MATCH_TASK_COUNT`. Each task
therefore stages the interval store once and then handles its rows sequentially.

## 3. Choose array width and resources

There are two levels of parallelism:

1. SLURM array tasks process independent manifest shards in parallel.
2. A sampling task uses four worker processes so its four chains run in
   parallel.

Start conservatively with 5–10 array tasks. Each sampling task requests four
CPUs and 96 GiB RAM. Increase array width only after measuring Quobyte staging
load and scheduler behavior. More than four CPUs per sampling task does not
help the default four-chain workflow.

For a target of size `n` and exact grid length `B`, each chain's selected-row
float64 CDF cache is approximately `8*n*B` bytes. At `n=35,000` and `B=22,000`,
this is about 5.7 GiB per chain, or 23 GiB across four workers, plus interval
store pages, transient blocks, Python processes, and output arrays. The 96 GiB
starting request is intentionally conservative and must be replaced by measured
production RSS.

Node-local scratch must hold the complete staged interval store plus at least
20%. The supplied scripts abort before copying when this check fails.

## 4. Build target distributions

Edit the `#SBATCH --array` range in `build_age_targets.sbatch` so it contains
exactly `AGE_MATCH_TASK_COUNT` elements. For ten tasks use `0-9`.

Submit from a login node:

```bash
mkdir -p /quobyte/project/normalizeTE/logs

export PROJECT=/quobyte/project/normalizeTE
export STORE=/quobyte/project/data/snp_interval_store
export MANIFEST=/quobyte/project/manifests/te_manifest.tsv
export AGE_MATCH_TASK_COUNT=10

sbatch --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  build_age_targets.sbatch
```

Each manifest row runs `te_age_target.py` with 10,000 bootstrap replicates.
Temporary target CDF matrices go under `$TMPDIR`; complete target directories
publish atomically to their Quobyte paths.

Target resolution defaults to `error` if any requested position is absent or
ineligible. To retain only eligible positions and record every exclusion, add
`MISSING_POSITION_POLICY=drop` to the exported submission variables. Do not use
`drop` without reviewing `position_resolution` and `excluded_positions` in the
target metadata.

The acceptance threshold defaults to the bootstrap median
(`ACCEPTANCE_QUANTILE=0.50`). Set a different quantile only for an explicitly
labeled sensitivity analysis, and never reuse a target path across quantiles.

Check the array before sampling:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
find /quobyte/project/targets -name metadata.json -print | wc -l
```

Do not submit matching jobs for missing or incomplete targets.

## 5. Generate 100 matched sets per target

After all target jobs succeed:

```bash
export PROJECT=/quobyte/project/normalizeTE
export STORE=/quobyte/project/data/snp_interval_store
export MANIFEST=/quobyte/project/manifests/te_manifest.tsv
export AGE_MATCH_TASK_COUNT=10

sbatch --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  sample_age_matches.sbatch
```

Every target runs:

```text
100 sets = 4 independent chains * 25 saved sets
burn-in replacement = 50%
between-sample replacement = 25%
```

The greedy first-passage set is not one of the 100 outputs. Every saved set is
scored on the exact 1,000-generation grid and final publication fails if any
set exceeds the target's bootstrap threshold or contains duplicate controls.

## 6. Single-target command

For debugging or an isolated category:

```bash
python sample_age_matched_controls.py \
  --store "$TMPDIR/interval_store" \
  --target /quobyte/project/targets/in_gene \
  --all-eligible \
  --output /quobyte/project/matches/in_gene \
  --sets 100 \
  --chains 4 \
  --sets-per-chain 25 \
  --workers 4 \
  --seed 1002 \
  --burnin-replacement-fraction 0.50 \
  --sample-replacement-fraction 0.25
```

Use `--candidate-rows candidates.npy` instead of `--all-eligible` to restrict
the declared candidate universe. Target rows are always removed.

## 7. Outputs

Each output directory contains:

- `row_indices.npy`: canonical store rows, shape `(100, n)`;
- `positions.npy`: 1-based native positions, shape `(100, n)`;
- `chromosome_codes.npy` and `chromosome_labels.npy`;
- `cdfs.npy`: exact aggregate CDF for every set;
- `wasserstein.npy`: 100 exact distances;
- `chain_index.npy` and `sample_index.npy`;
- `diagnostics.csv`: construction, burn-in, thinning, and save diagnostics;
- `reuse_row_indices.npy` and `reuse_counts.npy`;
- `target_cdf.npy` and `age_bins.npy`; and
- `metadata.json`: complete provenance, seeds, parameters, timings, and chain
  histories.

Both target and matched-control metadata contain a `software` object:

```json
{
  "name": "normalizeTE",
  "version": "0.1.0",
  "git_commit": "40-character commit hash",
  "git_describe": "v0.1.0-1-gCOMMIT",
  "git_tag": null,
  "git_dirty": false
}
```

Before downstream analysis, require the expected `software.version` and
`software.git_commit`, and reject `git_dirty: true`. An exported source tree
without `.git` retains the release version but records the Git fields as null;
prefer running from a tagged clone so the commit is preserved.

A successfully published `metadata.json` contains:

```json
{"schema_version": "swap-age-matched-controls-v1", "complete": true}
```

Incomplete runs remain at a sibling path named `.OUTPUT_NAME.work`. Inspect
that directory and its chain checkpoints before retrying. The supplied array
runner always enables `--resume`: it validates the saved run identity, reuses
every completed chain, and restarts only interrupted chains. A fresh job with
no work directory starts normally even with `--resume`; already complete
manifest rows from the same release are skipped, while unversioned or
different-version outputs stop the task. An interrupted chain currently
restarts from its deterministic seed; completed chains do not. Store identity
uses its schema and
catalog digest, so a new node-local staging path does not invalidate a retry.
Do not delete failed work until its logs have been reviewed.

## 8. Operational guidance

- Keep BLAS/OpenMP thread counts at one inside each worker. Parallelism comes
  from the four independent chain processes.
- Stage the store once per array task. Do not launch one array task per target
  unless Quobyte can sustain every task copying the store simultaneously.
- Write final targets and match directories directly to Quobyte; they publish
  via same-filesystem atomic rename.
- Never point final output paths into `$TMPDIR`.
- Use different manifest seeds for different targets. Chain seeds are derived
  reproducibly from the target digest, global seed, chain number, and algorithm
  version.
- If a category exhausts its proposal budget, inspect its best construction
  distance and chain acceptance rate. Do not silently relax its threshold.
- Expect accepted-chain distances to concentrate near the acceptance cutoff:
  most feasible sets may lie near the boundary. This is not a failed match,
  but it makes the saved W1 distribution and chain diagnostics essential
  outputs to inspect.
- Keep the `.out`/`.err` logs until all 100-set completeness and diversity
  checks pass.

## 9. Production gates

Before launching every TE category on the 75-draw store:

1. run one approximately 500-SNP target;
2. run one approximately 4,000-SNP target;
3. run one approximately 35,000-SNP target;
4. confirm all four chains finish and all 100 distances pass;
5. inspect membership replacement, overlap, control reuse, and chromosome
   composition;
6. record `MaxRSS`, elapsed time, staged-store size, and proposals per saved
   set; and
7. adjust SLURM memory/time and array width from those measurements.

The 25% between-sample replacement is the initial default. Compare 10%, 25%,
and 50% on representative categories before claiming the 100 sets are
effectively independent.

## 10. Local two-draw validation

The implementation was exercised on the two available ARG draws with the
4,072-position in-gene TE file. Eleven positions were explicitly recorded as
ineligible, leaving 4,061 targets. With 10,000 bootstrap replicates, the W1
range was 315.36–6,868.01 generations, the median was 1,905.05, and the
conservative 95th-percentile cutoff was 3,793.04. All 100 generated sets passed;
their W1 range was 3,639.31–3,792.66 and their median was 3,763.44.

A second 100-set run used the same bootstrap replicates and the 50th-percentile
cutoff of 1,905.10. Its saved W1 range was 1,790.75–1,904.43 and its median was
1,883.12. The tighter chain had a 31.4% thinning-proposal acceptance rate versus
32.3% at the 95th-percentile cutoff, with essentially identical serial runtime
(147 versus 148 seconds). Thus a median constraint remained computationally
practical in this two-draw test and prevented drift toward the broader 95%
boundary.

The local sandbox could not create process semaphores and therefore ran the
four chains serially in 148 seconds; individual chains took 35–37 seconds.
Linux HPC jobs should run them concurrently, but production timing and memory
must be measured on the complete 75-draw store. The follow-up default is 0.50;
use 0.95 only as an explicitly labeled sensitivity analysis.
