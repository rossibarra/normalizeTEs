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
The upper 95th-percentile bootstrap distance defines the maximum acceptable
mismatch for a synonymous control set. At least 1,000 bootstrap replicates are
recommended so this tail threshold is reasonably stable.

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

The chromosome labels must exactly match the `chrom_offsets` table embedded by
ARGtest in the merged tree-sequence metadata. No reference `.fai` is required.
ARGtest stores zero-based tskit coordinates internally. The commands convert a
one-based VCF position `POS` to `chromosome_offset + POS - 1`; users should
never pre-convert position lists to cumulative or zero-based coordinates.

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
  --min-usable-fraction 0.1
```

The builder finds all variant positions across the tree sequences, estimates
their age distributions, converts them to quantized CDFs, and writes the
position and CDF NumPy arrays under `age_store/`. Both tszip-compressed `.tsz`
and ordinary tskit files are accepted. The output directory must not already
exist. By default a SNP is eligible only when at least 10% of posterior draws
provide a usable mutation-node-to-parent interval. Use `--min-usable-draws` to
set an absolute count instead, and use `--missing error` or `--root error` if
missing posterior sites or mutations above roots should stop the build instead
of being recorded and skipped.

Use `--mutation-weighting draw` to give each usable posterior draw equal total
weight rather than weighting every mutation interval equally. Optional
`--omit-transpose` reduces final disk use at the cost of slower candidate
scans, while `--checksums` records SHA-256 input checksums at additional I/O
cost. Run `python build_snp_age_store.py --help` for the complete CLI.

`valid` means that at least one usable age interval exists for a SNP.
`eligible` additionally means that the SNP meets the requested posterior-draw
coverage threshold. TE and synonymous input positions must resolve to eligible
rows. During construction, the builder creates a temporary disk-backed
floating-point accumulator alongside the output directory; allow scratch space
in addition to the final quantized arrays.

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
sequence without ARGtest `chrom_offsets` metadata, so its output is not a valid
input to this native-coordinate production workflow unless that metadata is
added first.

### 4. Construct the TE target and matching threshold

```bash
python te_age_target.py \
  --store age_store \
  --te-positions te_subset.txt \
  --output targets/te_subset \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.95 \
  --seed 12345
```

This command averages the \(X\) TE CDFs, bootstraps the TE SNPs, and stores the
95th-percentile Wasserstein acceptance threshold. It also divides the target
distribution at 5% probability increments. Repeated boundaries on the discrete
age grid are merged, and integer sampling quotas are adjusted to total exactly
\(X\).

The default bootstrap compares each resample with the observed TE target.
`--bootstrap-reference two-sample` instead compares two independent TE
resamples, and `--bootstrap-batch-size` controls temporary bootstrap memory.

Important outputs under `targets/te_subset/` include:

- `te_chromosomes.npy`: chromosome labels for the selected TE SNPs
- `te_positions.npy`: corresponding 1-based VCF positions
- `target_cdf.npy`: summed and normalized TE target CDF
- `bootstrap_wasserstein.npy`: bootstrap distances in generations
- `interval_boundary_indices.npy`: sampling-interval boundaries
- `interval_quotas.npy`: number of synonymous SNPs requested per interval
- `metadata.json`: threshold, parameters, seeds, and provenance

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

The sampler calculates synonymous-candidate weights in large blocks, proposes
sets without replacement, and evaluates each proposal using its complete
combined age CDF. A set is retained only when its Wasserstein distance from the
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
- `metadata.json`: run parameters, seeds, threshold, and proposal counts

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
  boundary blocks under `$SLURM_TMPDIR` and copy only final outputs to Quobyte.
- Keep the store and result arrays as a small number of large files rather than
  creating per-SNP files.
