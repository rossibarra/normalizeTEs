# normalizeTE

## Age-matched synonymous controls

The matching workflow uses the complete age distributions of a set
of TE SNPs to select equally sized sets of synonymous SNPs with comparable age
uncertainty. It is intended to scale to roughly 500--10,000 TE SNPs and a pool
of millions of synonymous candidates on HPC storage.

### NumPy age-distribution store

All variant distributions from the input ARG posterior are first converted
to cumulative distribution functions (CDFs) on a common 1,000-generation age
grid. Positions, age bins, and quantized CDFs are stored in large NumPy
arrays rather than CSV files. Separate SNP-major and age-major layouts can be
used for fast retrieval of selected SNPs and fast blockwise scans of the full
synonymous candidate pool, respectively. The arrays are written once and
read in large contiguous blocks from Quobyte.

### TE target and bootstrap threshold

For an input set of \(X\) TE positions, the workflow retrieves and averages
their individual CDFs to obtain the target TE age distribution. It then
bootstraps the \(X\) TE SNPs with replacement and calculates the one-dimensional
Wasserstein distance between each bootstrap CDF and the observed target CDF.
The median bootstrap distance defines the default maximum acceptable mismatch
for a control set. At least 1,000 bootstrap replicates are recommended so this
threshold is reasonably stable. The quantile remains configurable for explicit
sensitivity analyses.

### Stratified synonymous sampling

The target TE distribution is divided at its 5%, 10%, ..., 95% quantiles,
forming 20 intervals that each contain approximately 5% of its probability.
For each interval, every synonymous candidate receives a weight equal to the
probability mass of its own age distribution within that interval. Candidate
weights are calculated blockwise so a full 20-million-SNP by 20-interval
matrix is never held in memory.

Each proposed control set normally contains approximately \(X/20\) synonymous
SNPs from each 5% stratum, with adjustments when \(X\) is not divisible by 20.
When several quantiles land on the same discrete age boundary, those strata
are merged and their quotas combined. Sampling is without replacement within a
set and is driven by the interval-specific weights. Because individual SNP
distributions can span several intervals, the actual combined CDF of every
proposed set is calculated after sampling.

### Acceptance and repeated samples

The combined synonymous CDF is compared with the target TE CDF using
Wasserstein distance. A proposed set is accepted when its distance is no larger
than the TE bootstrap threshold. Sampling continues until 100 accepted sets
have been obtained or a configurable attempt limit is reached. Selecting by a
bootstrap-derived threshold, rather than simply retaining the closest sets,
avoids requiring synonymous controls to match more closely than ordinary
sampling variation among the TE SNPs.

Each run records the input TE positions, accepted synonymous positions,
random seeds, target and matched CDFs, Wasserstein distances, bootstrap
percentiles, interval counts, rejection counts, and overlap among repeated
control sets. This makes the matching process reproducible and exposes age
regions where the synonymous candidate pool cannot adequately reproduce the TE
distribution.

## How to generate age-matched synonymous sets

Assume the starting files are organized as follows:

```text
project-data/
├── posterior/
│   ├── draw_001.tsz
│   ├── draw_002.tsz
│   └── ...
├── te_positions.txt
└── syn_positions.txt
```

Each position file must contain exactly two whitespace-separated columns:
chromosome and 1-based VCF position. Cumulative coordinates are not accepted:

```text
chr4 100
chr4 27591
chr7 802
```

The chromosome labels must exactly match those in the ARG's embedded
`chrom_offsets` metadata, or those in the optional chromosome offsets file
described below. No reference `.fai` is required, and the ARG does not
need to have been produced by ARGtest.
The ARGs used by this workflow store one-based positions internally, matching
VCF coordinates. The commands convert `POS` to `chromosome_offset + POS`;
users should never pre-convert position lists to cumulative or zero-based
coordinates.

### 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate normalizeTE
```

The creation command is needed only once. To run this repository's test suite
without collecting the separately vendored SINGER workflow tests, use:

```bash
python -m pytest -q tests test_snp_age_distribution.py
```

### 2. Build the reusable age store

Run this once for a collection of posterior tree sequences:

```bash
python build_snp_age_store.py \
  project-data/posterior/*.tsz \
  --numpy-store age_store \
  --bin-width 1000 \
  --block-snps 100000 \
  --min-usable-fraction 0.1 \
  --scratch-dir "$TMPDIR"
```

The builder finds all variant positions across the tree sequences, estimates
their age distributions, converts them to quantized CDFs, and writes the
position and CDF NumPy arrays under `age_store/`. Both tszip-compressed `.tsz`
and ordinary tskit files are accepted. The output directory must not already
exist. By default a SNP is eligible only when at least 10% of posterior draws
provide a usable mutation-node-to-parent interval. Adjust this with
`--min-usable-fraction`. Use `--missing error` or `--root error` if missing
posterior sites or mutations above roots should stop the build instead of being
recorded and skipped.

`--omit-transpose` reduces final disk use at the cost of slower candidate
scans. `--chrom-offsets` supplies the chromosome offset table from a file when
the tree sequences do not carry usable `chrom_offsets` metadata; see
[Optional chromosome offsets file](#optional-chromosome-offsets-file). Run
`python build_snp_age_store.py --help` for the complete CLI.

`valid` is derived from whether at least one usable age interval exists.
`eligible` additionally means that the SNP meets the requested posterior-draw
coverage threshold. TE and synonymous input positions must resolve to eligible
rows. During construction, the builder creates a temporary disk-backed
floating-point accumulator; allow scratch space in addition to the final
quantized arrays. Its size is approximately
`4 * number_of_SNPs * number_of_age_bins` bytes: at 20 million SNPs this is
about 15 GiB for 200 bins or 75 GiB for 1,000 bins. Each posterior draw sweeps
this accumulator in genomic order. Use `--scratch-dir` to place it on
node-local storage such as `$TMPDIR`; final store files are still
assembled beside `--numpy-store` for atomic publication.

On Farm, use `$TMPDIR`, not `$SLURM_TMPDIR`. The latter is not set. Jobs receive
a per-job `$TMPDIR` on `/local/scratch`.

### Compact all-SNP interval store (recommended)

The compact builder retains the complete interval posterior for every SNP in
the union of the input draws without materializing a dense SNP-by-age matrix.
TE and synonymous lists are downstream selections and do not filter the
store. For approximately 25--30 million SNPs and 75 TSZ draws like the
measured combined SINGER files, the conservative default request is one CPU,
48 GB RAM, 16 hours, and at least 32 GiB free in node-local `$TMPDIR`. Submit
the following from the repository root as, for example,
`sbatch build_interval_store.sbatch`:

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
  --interval-store age_interval_store \
  --chrom-offsets project-data/chrom_offsets.txt \
  --interval-dtype float32 \
  --min-usable-fraction 0.1 \
  --num-buckets 100 \
  --bucket-memory-gb 2 \
  --scratch-dir "${TMPDIR:?SLURM did not set TMPDIR}"
```

The measured projection is approximately 17.1 GiB for the final 75-draw
float32 store and 22.6 GiB of packed bucket scratch. Atomic construction also
creates the new final arrays beside `--interval-store`, so retain additional
Quobyte headroom if an older store remains present.

The production builder is currently single-worker. Its final merge is
Quobyte-I/O-bound, and merely allocating more CPUs does not make that phase
faster. Multiple CPUs are implemented for the independent scalar correctness
audit. A four-CPU audit of 10,000 mutations took 40m29s end to end and used
19.9 GB peak RSS:

```bash
#!/bin/bash -l
#SBATCH --account=jrigrp
#SBATCH --partition=low
#SBATCH --job-name=interval-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
module load conda
conda activate normalizeTE

python benchmark_interval_store_gate2.py project-data/posterior/draw.tsz \
  --output "results/interval-gate2-${SLURM_JOB_ID}.json" \
  --audit-size 10000 \
  --audit-workers "$SLURM_CPUS_PER_TASK" \
  --scratch-dir "${TMPDIR:?SLURM did not set TMPDIR}"
```

Interval-store TE target construction uses a temporary float32 TE-by-age CDF
matrix in node-local scratch instead of retaining a float64 matrix in RAM. At
about 185,000 TEs, a 1,000-generation bin width, and the measured maximum age,
allow roughly 16--18 GiB of additional `$TMPDIR` space. Bootstrap matrix
multiplication can use four CPUs. Regular-grid CDF rows are built with
slope/intercept difference accumulators in O(interval records + output cells),
not interval-records times age-bins:

```bash
#!/bin/bash -l
#SBATCH --account=jrigrp
#SBATCH --partition=low
#SBATCH --job-name=te-age-target
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

set -euo pipefail
module load conda
conda activate normalizeTE
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python te_age_target.py \
  --store age_interval_store \
  --te-positions te_positions.txt \
  --output te_age_target \
  --scratch-dir "${TMPDIR:?SLURM did not set TMPDIR}"
```

For an interactive allocation, request the same resources with `srun` and run
the corresponding script body after the shell starts. For example, the store
build allocation is:

```bash
srun --account=jrigrp --partition=low \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --mem=48G --time=16:00:00 --pty bash -l
```

Inside that allocated shell, load `normalizeTE` and run the builder command
above. `$TMPDIR` is created for the job on `/local/scratch`; do not substitute
`$SLURM_TMPDIR`.

Synonymous matching defaults to `--candidate-access gather`. For repeated or
explicitly cached access, select `--candidate-access cache` and provide
`--candidate-cache-dir "$TMPDIR"`; the temporary cache is removed after the
candidate weights are materialized. The Gate 3 cache for 396,678 candidates
was 16.9 MB. See `INTERVAL_STORE_BENCHMARKS.md` for measured details and the
limits of these projections.

### Optional chromosome offsets file

By default the builder reads the chromosome offset table from each tree
sequence's top-level `chrom_offsets` metadata and fails if it is absent. Pass
`--chrom-offsets FILE` to supply that table externally instead, which makes
tree sequences without the metadata — for example plain msprime or SINGER
output — usable in the native-coordinate workflow:

```bash
python build_snp_age_store.py \
  project-data/posterior/*.tsz \
  --numpy-store age_store \
  --chrom-offsets project-data/chrom_offsets.txt
```

The file is whitespace-separated (spaces or tabs). Blank lines are ignored, and
`#` starts a comment that runs to the end of the line. Every non-blank row
names one chromosome, and all rows must use the same number of columns. Two
layouts are accepted.

**Two columns — chromosome and length.** Offsets are accumulated in file order,
starting at 0, so the row order must match the order in which the chromosomes
were concatenated into the ARG:

```text
# chrom  length
chr1     308452471
chr2     243675191
chr3     238017767
```

This is the first two columns of a reference `.fai` index, so it can be
produced with `cut -f1,2 reference.fa.fai`. Do not pass a raw `.fai` file: its
third column is a byte offset into the FASTA file, not a genome offset.

**Three columns — chromosome, offset, and length.** Use this when the ARG's
chromosomes are not laid out back to back, or to state the layout explicitly:

```text
# chrom  offset      length
chr1     0           308452471
chr2     308452471   243675191
chr3     552127662   238017767
```

`offset` is the cumulative coordinate of the base immediately before the
chromosome's first base, matching the `chrom_offsets` metadata convention: a
1-based VCF position `POS` on that chromosome maps to store coordinate
`offset + POS`. The first chromosome therefore normally has offset 0.

The table must satisfy the same rules as embedded metadata: chromosome names
are unique, offsets and lengths are integers with `offset >= 0` and
`length > 0`, rows are sorted by strictly increasing offset, and no chromosome
extends past the tree sequences' `sequence_length`. A supplied table must also
be non-overlapping (`offset + length <= next offset`), which catches transposed
or stale lengths that would otherwise mislabel native coordinates.

The chromosome labels in this file are the labels your TE and synonymous
position files must use. When a tree sequence carries `chrom_offsets` metadata
that disagrees with the file, the file wins and the builder writes a warning to
standard error. The resolved table is copied into the store's `metadata.json`
as `chromosomes`, with `chromosomes_source` recording either `arg_metadata` or
the offsets-file path, so later steps need no coordinate arguments.

### 3. Choose the TE subset

Create a file containing the \(X\) TE SNPs to match. For example, this selects a
reproducible random subset of 5,000 positions from a larger TE list:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

X = 5_000
rng = np.random.default_rng(12345)
rows = []
for raw in Path("project-data/te_positions.txt").read_text().splitlines():
    fields = raw.split("#", 1)[0].split()
    if fields:
        if len(fields) != 2:
            raise ValueError("expected chromosome and 1-based VCF position columns")
        rows.append(fields)
positions = np.asarray(rows, dtype=str)
if X > positions.shape[0]:
    raise ValueError(f"requested {X} TE SNPs, but only {positions.shape[0]} are available")
selected = positions[rng.choice(positions.shape[0], size=X, replace=False)]
np.savetxt("te_subset.txt", selected, fmt="%s")
PY
```

Skip this step when `te_positions.txt` already contains exactly the desired
subset.

The older `simulate_neutral_trees.py` example creates a plain msprime tree
sequence without ARGtest `chrom_offsets` metadata. Its output can still be used
in this native-coordinate production workflow by building the store with
`--chrom-offsets`, or by adding the metadata to the tree sequence first.

### 4. Construct the TE target and matching threshold

```bash
python te_age_target.py \
  --store age_store \
  --te-positions te_subset.txt \
  --output targets/te_subset \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --seed 12345
```

This command averages the \(X\) TE CDFs, bootstraps the TE SNPs, and stores the
median Wasserstein acceptance threshold. It also divides the target
distribution at 5% probability increments. Repeated boundaries on the discrete
age grid are merged, and integer sampling quotas are adjusted to total exactly
\(X\).

Each bootstrap resample is compared with the observed TE target.
`--bootstrap-batch-size` controls temporary bootstrap memory.

Important outputs under `targets/te_subset/` include:

- `te_chromosomes.npy`: chromosome labels for the selected TE SNPs
- `te_positions.npy`: corresponding 1-based VCF positions
- `target_cdf.npy`: summed and normalized TE target CDF
- `bootstrap_wasserstein.npy`: bootstrap distances in generations
- `interval_boundary_indices.npy`: sampling-interval boundaries
- `interval_quotas.npy`: number of synonymous SNPs requested per interval
- `metadata.json`: threshold, parameters, seed, and provenance

### 5. Generate 100 matched synonymous sets

```bash
python sample_age_matched_syn.py \
  --store age_store \
  --target targets/te_subset \
  --syn-positions project-data/syn_positions.txt \
  --output matches/te_subset \
  --accepted-sets 100 \
  --max-proposals 100000 \
  --block-snps 250000 \
  --seed 67890
```

The sampler reads synonymous-candidate weights once in large blocks and
materializes a reusable float32 candidate-by-age-stratum matrix. It proposes
sets without replacement without rereading those weights from the store, then
evaluates each proposal using its complete combined age CDF. A set is retained
only when its Wasserstein distance from the
TE target is at or below the bootstrap threshold. Synonymous SNPs are unique
within one set but may be reused across different accepted sets.
For pre-resolved workflows, `--syn-indices` accepts a NumPy array of store row
indices and `--syn-mask` accepts a Boolean store-length NumPy mask instead of a
native-coordinate position file.

The principal outputs are:

- `syn_chromosomes.npy`: chromosome labels with shape `(100, X)`
- `syn_positions.npy`: 1-based VCF positions with shape `(100, X)`
- `syn_row_indices.npy`: corresponding rows in the age store
- `syn_cdf.npy`: combined CDF for every accepted set
- `wasserstein.npy`: distance of every accepted set from the TE target
- `interval_assignment.npy`: sampling interval assigned to every selected SNP
- `diagnostics.csv`: accepted and rejected proposal diagnostics
- `metadata.json`: run parameters, seed, threshold, and proposal counts

`metadata.json` also records the weight-matrix shape, dtype, and byte size, plus
the proposal count, rejection count, and acceptance rate. With 20 age strata,
the matrix uses about 80 bytes per synonymous candidate (about 1.5 GiB for 20
million candidates).

If fewer than 100 proposals pass before `--max-proposals`, the command stops
with a diagnostic error rather than silently relaxing the threshold. Increase
the proposal limit only after inspecting the rejection rate and confirming that
the synonymous pool has usable probability mass across the target age range.

### HPC and Quobyte notes

- Keep the NumPy store immutable after construction so jobs can read it safely
  in parallel.
- Submit independent TE subsets as separate SLURM jobs.
- Use large `--block-snps` values that fit comfortably in node memory; benchmark
  values between 250,000 and 1,000,000 on the production system.
- If node-local scratch is available, stage frequently accessed arrays or
  boundary blocks under `$TMPDIR` and copy only final outputs to Quobyte.
- Keep the store and result arrays as a small number of large files rather than
  creating per-SNP files.

## Swap-chain matched controls (recommended workflow)

The recommended sampler for the full eligible control pool is now
`sample_age_matched_controls.py`. It uses stochastic one-for-one swaps against
the canonical interval store and does not require the large global alias index.
For every TE target it generates 100 sets as four independent chains with 25
saved sets each. The greedy construction state is discarded; chains replace at
least 50% of that state during burn-in and at least 25% of members between saved
sets. Every saved set is certified with the exact full-grid Wasserstein test.
Target and match `metadata.json` files record the normalizeTE release version,
Git commit, nearest Git description, exact tag when present, and whether tracked
files were dirty when the command ran.

For one target:

```bash
python sample_age_matched_controls.py \
  --store interval_store \
  --target targets/in_gene \
  --all-eligible \
  --output matches/in_gene \
  --sets 100 --chains 4 --sets-per-chain 25 --workers 4 \
  --seed 1002
```

For many TE datasets on Farm, use `run_age_match_manifest.py` with
`build_age_targets.sbatch` and `sample_age_matches.sbatch`. Array tasks stage
the immutable interval store from Quobyte to Farm's per-job `$TMPDIR`, run four
chain workers per target, and write atomically published results back to
Quobyte. See [`SWAP_SAMPLER_HPC_HOWTO.md`](SWAP_SAMPLER_HPC_HOWTO.md) for the
manifest format, submission commands, resource model, outputs, and validation
gates.

For reproducible production runs, pin either an immutable tag or an exact
commit rather than relying on a moving branch. `v0.1.0` is the tagged q95
baseline; the q50 default is the documented follow-up commit on `main`:

```bash
git fetch --tags
git checkout COMMIT_HASH
```

Release changes are summarized in [`CHANGELOG.md`](CHANGELOG.md).

## Documentation for accessory scripts

### `snp_age_distribution.py`

This smaller utility estimates age distributions for a selected set of SNPs
without building the reusable NumPy store. It accepts ordinary `.trees` files,
tszip-compressed `.tsz` files, and shell-style input globs. Positions are exact
numeric coordinates as stored in the tree sequence; unlike the main workflow,
this accessory command does not accept chromosome-position pairs or translate
native coordinates using `chrom_offsets`.

For every requested position, each retained mutation contributes a uniform age
distribution between the mutation node and its parent in the marginal tree.
The command combines those intervals, integrates them into age bins, normalizes
the distribution, and writes CSV to standard output:

```bash
python snp_age_distribution.py posterior/*.tsz \
  --position 100 \
  --position 27591 \
  --bin-width 1000 \
  > snp_ages.csv
```

For a longer list, provide one numeric position per line:

```bash
python snp_age_distribution.py posterior/*.tsz \
  --positions-file positions.txt \
  > snp_ages.csv
```

Use `--intervals` to write the underlying node-to-parent bounds instead of the
binned distributions. By default, positions absent from a posterior draw and
mutations above a root are recorded or skipped; `--missing error` and
`--root error` make either condition fatal. This command is intended for small
queries and inspection. Use `build_snp_age_store.py` for genome-scale reusable
data.
