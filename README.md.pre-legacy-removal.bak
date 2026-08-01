# normalizeTE

## SNP age distributions

`snp_age_distribution.py` estimates an age distribution for each requested SNP
from one or more tree sequences. For every matching mutation, it uses the age
of the mutation's node as the lower bound and the age of that node's parent in
the marginal tree as the upper bound. It treats this interval as a uniform
distribution, combines intervals across tree-sequence replicates, normalizes
the resulting mixture to probability one, and discretizes it into generation
bins. The default bins are centered on ages rounded to the nearest 1,000
generations.

The command requires one or more `.trees` files and exact SNP positions. Supply
positions by repeating `--position`:

```bash
conda activate normalizeTE

python snp_age_distribution.py \
  results/neutral_100kb.trees \
  --position 1250 \
  --position 48200
```

Alternatively, provide a plain-text file containing one SNP position per line:

```bash
python snp_age_distribution.py \
  results/neutral_100kb.trees \
  --positions-file snp_positions.txt
```

Multiple posterior or replicate tree sequences can be analyzed together:

```bash
python snp_age_distribution.py \
  "results/posterior/*.trees" \
  --positions-file snp_positions.txt \
  > snp_age_distributions.csv
```

Positions must exactly match site positions stored in the tree sequences. The
default CSV output contains these columns:

- `position`: SNP position in base pairs
- `age_bin`: center of the generation-age bin
- `probability`: normalized probability mass in that bin
- `interval_count`: number of mutation-age intervals contributing to the SNP
- `missing_replicates`: number of input tree sequences without that position

Use `--bin-width` to change the default 1,000-generation bins. Use
`--intervals` to output the underlying lower and upper node ages instead of the
discretized distributions. Missing SNPs and mutations above root nodes are
skipped by default; `--missing error` and `--root error` make either condition
an error.

## Age-matched synonymous controls

The planned matching workflow will use the complete age distributions of a set
of TE SNPs to select equally sized sets of synonymous SNPs with comparable age
uncertainty. It is intended to scale to roughly 500--10,000 TE SNPs and a pool
of millions of synonymous candidates on HPC storage.

### NumPy age-distribution store

All variant distributions from the input ARG posterior will first be converted
to cumulative distribution functions (CDFs) on a common 1,000-generation age
grid. Positions, age bins, and quantized CDFs will be stored in large NumPy
arrays rather than CSV files. Separate SNP-major and age-major layouts can be
used for fast retrieval of selected SNPs and fast blockwise scans of the full
synonymous candidate pool, respectively. The arrays will be written once and
read in large contiguous blocks from Quobyte.

### TE target and bootstrap threshold

For an input set of \(X\) TE positions, the workflow will retrieve and average
their individual CDFs to obtain the target TE age distribution. It will then
bootstrap the \(X\) TE SNPs with replacement and calculate the one-dimensional
Wasserstein distance between each bootstrap CDF and the observed target CDF.
The upper 95th-percentile bootstrap distance will define the maximum acceptable
mismatch for a synonymous control set. At least 1,000 bootstrap replicates are
recommended so this tail threshold is reasonably stable.

### Stratified synonymous sampling

The target TE distribution will be divided at its 5%, 10%, ..., 95% quantiles,
forming 20 intervals that each contain approximately 5% of its probability.
For each interval, every synonymous candidate receives a weight equal to the
probability mass of its own age distribution within that interval. Candidate
weights will be calculated blockwise so a full 20-million-SNP by 20-interval
matrix is never held in memory.

Each proposed control set will contain \(X/20\) synonymous SNPs sampled from
each interval, with adjustments when \(X\) is not divisible by 20. Sampling is
without replacement within a set and is driven by the interval-specific
weights. Because individual SNP distributions can span several intervals, the
actual combined CDF of every proposed set will be calculated after sampling.

### Acceptance and repeated samples

The combined synonymous CDF will be compared with the target TE CDF using
Wasserstein distance. A proposed set is accepted when its distance is no larger
than the TE bootstrap threshold. Sampling continues until 100 accepted sets
have been obtained or a configurable attempt limit is reached. Selecting by a
bootstrap-derived threshold, rather than simply retaining the closest sets,
avoids requiring synonymous controls to match more closely than ordinary
sampling variation among the TE SNPs.

Each run will record the input TE positions, accepted synonymous positions,
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
│   ├── draw_001.trees
│   ├── draw_002.trees
│   └── ...
├── te_positions.txt
└── syn_positions.txt
```

Each position file must contain one exact base-pair position per line. Positions
must use the same coordinate system as the sites in the tree sequences.

### 1. Activate the environment

```bash
conda activate normalizeTE
```

### 2. Build the reusable age store

Run this once for a collection of posterior tree sequences:

```bash
python build_snp_age_store.py \
  project-data/posterior/*.trees \
  --numpy-store age_store \
  --bin-width 1000 \
  --block-snps 100000
```

The builder finds all variant positions across the tree sequences, estimates
their age distributions, converts them to quantized CDFs, and writes the
position and CDF NumPy arrays under `age_store/`. The output directory must not
already exist. Use `--missing error` or `--root error` if missing posterior
sites or mutations above roots should stop the build instead of being recorded
and skipped.

### 3. Choose the TE subset

Create a file containing the \(X\) TE SNPs to match. For example, this selects a
reproducible random subset of 5,000 positions from a larger TE list:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

X = 5_000
rng = np.random.default_rng(12345)
positions = np.loadtxt("project-data/te_positions.txt", ndmin=1)
if X > positions.size:
    raise ValueError(f"requested {X} TE SNPs, but only {positions.size} are available")
selected = rng.choice(positions, size=X, replace=False)
np.savetxt("te_subset.txt", selected, fmt="%.15g")
PY
```

Skip this step when `te_positions.txt` already contains exactly the desired
subset.

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

Important outputs under `targets/te_subset/` include:

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

The principal outputs are:

- `syn_positions.npy`: array of shape `(100, X)` containing matched positions
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
