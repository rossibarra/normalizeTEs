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
