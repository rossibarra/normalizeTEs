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
