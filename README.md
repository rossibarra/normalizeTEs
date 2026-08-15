# normalizeTE

normalizeTE generates SNP control sets whose posterior age distributions match
an observed TE SNP dataset. The production workflow is designed for many TE
datasets, posterior ARG draws stored as `.tsz` files, SLURM, and Quobyte.

## Recommended workflow

The supported production path has three stages:

1. Build one compact all-SNP interval store from all posterior ARG draws.
2. Build one target distribution and bootstrap threshold for each TE dataset.
3. Generate 100 matched control sets per target with four independent swap
   chains, saving 25 sets from each chain.

The interval store is built once and reused for every TE dataset. Do not build
a dense CDF store first; the dense builder is a legacy alternative, not a
prerequisite. See [Alternative and legacy workflows](#alternative-and-legacy-workflows)
for the reasons it still exists.

Each usable mutation contributes a uniform age distribution between the age of
its mutation node (`below`) and its parent node (`above`). For a TE dataset of
size \(X\), `te_age_target.py` averages the \(X\) posterior CDFs and bootstraps
the TE SNPs with replacement. The median bootstrap Wasserstein distance is the
default maximum mismatch allowed for a control set.

`sample_age_matched_controls.py` first finds a set inside that threshold, then
runs a constrained random swap walk. The construction state is discarded.
Each chain replaces at least 50% of that state during burn-in, and successive
saved sets differ by at least 25% of their members. Every saved set is
recomputed on the exact 1,000-generation grid and must remain inside the target
threshold.

## Input data

Assume the starting files are organized as follows:

```text
project-data/
├── posterior/
│   ├── draw_001.tsz
│   ├── draw_002.tsz
│   └── ...
├── te/
│   ├── all_te.pos.txt
│   ├── in_gene.pos.txt
│   └── ...
└── chrom_offsets.txt          # only needed when ARG metadata is absent
```

Each TE position file must contain exactly two whitespace-separated columns:
chromosome and 1-based VCF position. Blank lines and comments beginning with
`#` are allowed.

```text
chr4 100
chr4 27591
chr7 802
```

Chromosome labels must match the ARG's embedded `chrom_offsets` metadata or the
optional offsets file supplied when the interval store is built. Do not
pre-convert positions to cumulative or zero-based coordinates. normalizeTE
maps a VCF position to `chromosome_offset + POS` internally.

Both tszip-compressed `.tsz` files and ordinary tskit tree-sequence files are
accepted.

## 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate normalizeTE
```

The creation command is needed only once. Run the project tests with:

```bash
python -m pytest -q tests test_snp_age_distribution.py
```

For reproducible production runs, use an immutable tag or exact commit rather
than a moving branch:

```bash
git fetch --tags
git checkout COMMIT_HASH
```

`v0.1.0` is the tagged q95 baseline. The bootstrap-median (q50) default is a
follow-up commit on `main`, so pin that exact commit until it receives its own
release tag. Release changes are summarized in [CHANGELOG.md](CHANGELOG.md).

## 2. Build the compact all-SNP interval store

Run this step once for the complete collection of posterior ARG draws. TE lists
do not filter the store; they are resolved later so the same store can serve
every analysis.

```bash
python build_snp_interval_store.py \
  project-data/posterior/*.tsz \
  --interval-store age_interval_store \
  --chrom-offsets project-data/chrom_offsets.txt \
  --interval-dtype float32 \
  --min-usable-fraction 0.1 \
  --num-buckets 100 \
  --bucket-memory-gb 2 \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}"
```

Omit `--chrom-offsets` when every ARG carries compatible chromosome metadata.
The output directory must not already exist. By default, a SNP is eligible
when at least 10% of posterior draws provide a usable mutation-node-to-parent
interval. Use `--missing error` or `--root error` when missing sites or
mutations above roots should stop construction instead of being recorded and
skipped.

The recommended production format uses `float32` interval endpoints. Use
`float64` only when the additional endpoint precision justifies approximately
twice the endpoint storage. Run `python build_snp_interval_store.py --help` for
all options.

### Farm/Quobyte store build

For approximately 25--30 million SNPs and 75 combined SINGER draws, a
conservative starting request is one CPU, 48 GB RAM, 16 hours, and at least 32
GiB free in node-local `$TMPDIR`. For example, save and submit the following as
`build_interval_store.sbatch`:

```bash
#!/bin/bash -l
#SBATCH --account=jrigrp
#SBATCH --partition=low
#SBATCH --job-name=snp-interval-store
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=16:00:00

set -euo pipefail
module load conda
conda activate normalizeTE

python build_snp_interval_store.py project-data/posterior/*.tsz \
  --interval-store /quobyte/project/data/snp_interval_store \
  --chrom-offsets project-data/chrom_offsets.txt \
  --interval-dtype float32 \
  --min-usable-fraction 0.1 \
  --num-buckets 100 \
  --bucket-memory-gb 2 \
  --scratch-dir "${TMPDIR:?SLURM did not set TMPDIR}"
```

On Farm, use `$TMPDIR`, not `$SLURM_TMPDIR`; Farm provides each job with
`$TMPDIR` under `/local/scratch`. The measured projection is approximately
17.1 GiB for a final 75-draw float32 store and 22.6 GiB of packed bucket
scratch. Keep additional Quobyte headroom for atomic publication and any older
store retained at the destination.

The production builder is single-worker. Its final merge is I/O-bound, so
requesting more CPUs does not speed that phase. Keep the completed store
immutable while target and matching jobs are running.

## 3. Prepare the TE datasets

Use one position file per biological category. If a source file already
contains exactly the TEs to analyze, no subsampling step is needed. Every
included position must resolve to an eligible row in the interval store.

By default, unresolved or ineligible positions stop target construction. The
optional `--missing-position-policy drop` records exclusions and continues;
only use it when downstream review of `position_resolution` and
`excluded_positions` in `metadata.json` is part of the analysis.

## 4. Build one TE target

```bash
python te_age_target.py \
  --store age_interval_store \
  --te-positions project-data/te/in_gene.pos.txt \
  --output targets/in_gene \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}" \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --seed 1002
```

The command averages the TE posterior CDFs and compares 10,000 bootstrap
resamples with the observed target. The default acceptance threshold is the
bootstrap median (`--acceptance-quantile 0.50`). Use another quantile only for
an explicitly labeled sensitivity analysis, and never reuse one target output
path across quantiles.

For an interval store, target construction creates a temporary float32
TE-by-age CDF matrix under `--scratch-dir`. At roughly 185,000 TEs and the
measured maximum age, allow approximately 16--18 GiB of additional scratch.
The output directory publishes atomically and must not already exist.

Important target outputs include:

- `te_chromosomes.npy`, `te_positions.npy`, and `te_row_indices.npy`;
- `target_cdf.npy` and `age_bins.npy`;
- `bootstrap_wasserstein.npy`; and
- `metadata.json`, including the threshold, parameters, position resolution,
  seed, store identity, and software provenance.

## 5. Generate 100 matched control sets

For one target:

```bash
python sample_age_matched_controls.py \
  --store age_interval_store \
  --target targets/in_gene \
  --all-eligible \
  --output matches/in_gene \
  --sets 100 \
  --chains 4 \
  --sets-per-chain 25 \
  --workers 4 \
  --seed 1002
```

`--all-eligible` uses every eligible non-target SNP as the candidate universe.
To restrict controls, resolve the desired universe to canonical store rows and
pass the resulting one-dimensional NumPy array with `--candidate-rows`.
Target rows are always excluded.

The production defaults are:

- 100 saved sets = 4 independent chains × 25 sets;
- 50% membership replacement during burn-in; and
- 25% membership replacement between successive saved states.

The CLI defaults to one worker, so specify `--workers 4` to run all four chains
concurrently when four CPUs are available. More than four CPUs does not help
the default four-chain configuration. BLAS and OpenMP thread counts should
remain one inside each worker.

Each output directory contains:

- `row_indices.npy`, shape `(100, X)`;
- `positions.npy`, `chromosome_codes.npy`, and `chromosome_labels.npy`;
- `cdfs.npy` and `wasserstein.npy` for all 100 sets;
- `chain_index.npy` and `sample_index.npy`;
- `diagnostics.csv` with construction, burn-in, thinning, and save records;
- `reuse_row_indices.npy` and `reuse_counts.npy`; and
- `metadata.json` with seeds, settings, timings, chain histories, store
  identity, and software provenance.

Final publication fails if any set contains duplicate controls or exceeds the
target's exact Wasserstein threshold. An interrupted run remains at
`.OUTPUT_NAME.work`; rerun with `--resume` to reuse completed chains.

## 6. Run many TE datasets on Farm/Quobyte

For many categories, create one tab-delimited manifest on Quobyte:

```text
label	positions	target	output	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young	/quobyte/project/matches/young	1003
```

Then use the supplied `build_age_targets.sbatch` and
`sample_age_matches.sbatch` array launchers. Each array task stages the
immutable interval store from Quobyte to its job-local `$TMPDIR` once, processes
several manifest rows, and writes only durable targets and results back to
Quobyte.

```bash
mkdir -p /quobyte/project/normalizeTE/logs

export PROJECT=/quobyte/project/normalizeTE
export STORE=/quobyte/project/data/snp_interval_store
export MANIFEST=/quobyte/project/manifests/te_manifest.tsv
export AGE_MATCH_TASK_COUNT=10

sbatch --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  build_age_targets.sbatch
```

After every target task succeeds:

```bash
sbatch --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  sample_age_matches.sbatch
```

The `#SBATCH --array` range in each launcher must contain exactly
`AGE_MATCH_TASK_COUNT` elements; `0-9` corresponds to 10 tasks. Start with
5--10 array tasks and increase only after measuring Quobyte staging load and
scheduler behavior. Each matching task requests four CPUs because its four
chains run in separate processes.

See [SWAP_SAMPLER_HPC_HOWTO.md](SWAP_SAMPLER_HPC_HOWTO.md) for the complete
manifest rules, memory model, submission settings, restart behavior, output
validation, and production gates.

## Chromosome offsets

By default, the interval builder reads a `chrom_offsets` table from every ARG's
top-level metadata and fails if it is absent or inconsistent. Pass
`--chrom-offsets FILE` to supply the table externally.

The file is whitespace-separated. Blank lines are ignored, and `#` starts a
comment. Two layouts are accepted.

Two columns specify chromosome and length. Offsets are accumulated in file
order, so that order must match the chromosomes concatenated into the ARG:

```text
# chrom  length
chr1     308452471
chr2     243675191
chr3     238017767
```

This is the first two columns of a reference `.fai` file. Do not pass the raw
`.fai`: its third column is a byte offset into the FASTA, not a genome offset.

Three columns specify chromosome, cumulative offset, and length explicitly:

```text
# chrom  offset      length
chr1     0           308452471
chr2     308452471   243675191
chr3     552127662   238017767
```

The first chromosome normally has offset zero. Rows must be ordered by strictly
increasing, non-overlapping offsets; chromosome names must be unique; and no
chromosome may extend beyond the tree sequences' `sequence_length`. When a
supplied file disagrees with embedded metadata, the file wins and the builder
prints a warning. The resolved table and its source are recorded in the store's
`metadata.json`.

## Software provenance

Target and match metadata record the normalizeTE release version, Git commit,
nearest Git description, exact tag when present, and whether tracked files were
dirty. For production, require the expected version and commit and reject runs
with `git_dirty: true`.

An exported source tree without `.git` retains the release version but records
Git fields as null. Prefer a tagged clone or exact committed checkout so the
full provenance is preserved.

## Accessory script: `snp_age_distribution.py`

This utility estimates age distributions for a small selected set of SNPs
without building the reusable interval store. It accepts `.trees` and `.tsz`
files plus shell-style globs. Positions are exact numeric coordinates stored in
the tree sequence; unlike the production workflow, it does not accept
chromosome-position pairs or translate native coordinates with
`chrom_offsets`.

```bash
python snp_age_distribution.py posterior/*.tsz \
  --position 100 \
  --position 27591 \
  --bin-width 1000 \
  > snp_ages.csv
```

For a longer list, provide one numeric position per line with
`--positions-file`. Use `--intervals` to write node-to-parent bounds rather than
binned distributions. This script is intended for inspection and small
queries; use the compact interval store for reusable genome-scale data.

## Alternative and legacy workflows

These workflows remain available for reproducibility and specialized use, but
they are not part of the recommended production path above.

### Dense CDF store

`build_snp_age_store.py` is an alternative to
`build_snp_interval_store.py`—never run both as sequential steps. It converts
every SNP posterior to a quantized CDF on a fixed age grid:

```bash
python build_snp_age_store.py \
  project-data/posterior/*.tsz \
  --numpy-store age_store \
  --bin-width 1000 \
  --block-snps 100000 \
  --min-usable-fraction 0.1 \
  --scratch-dir "$TMPDIR"
```

Use this format only to reproduce older dense-store analyses or with tooling
that explicitly requires the dense schema. Its temporary float32 accumulator
uses approximately `4 * number_of_SNPs * number_of_age_bins` bytes—about 15
GiB for 20 million SNPs and 200 bins, or 75 GiB for 1,000 bins. Fine grids
therefore consume substantially more disk and scratch, and changing bin width
requires rebuilding the store. `--omit-transpose` reduces final disk use at the
cost of slower candidate scans.

### Stratified rejection sampler

`sample_age_matched_syn.py` is the older proposal-and-rejection sampler. It
divides a target into equal-mass age strata, materializes candidate weights,
draws complete proposed sets, and accepts proposals below the Wasserstein
threshold. It can reproduce earlier analyses, but it is less suitable for many
large TE datasets because it builds target-specific candidate weights and may
spend many proposals in rejection sampling.

The production workflow instead uses `sample_age_matched_controls.py`, which
works directly from the canonical interval store and enforces burn-in and
between-sample replacement for the saved sets.
