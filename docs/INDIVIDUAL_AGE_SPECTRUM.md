# Per-individual derived-allele age spectra

This is a side analysis. It shares the interval store with the TE/control
matching workflow and changes nothing in it: no production module, launcher, or
artifact of that workflow is touched, and nothing here is an input to Phi-SFS.

## The question

For one individual, across every variant it is genotyped at, how old are the
derived alleles it carries? SNPs and TEs are not separated — every biallelic
record that resolves to a store row is treated identically, and polarity for all
of them comes from the ARG rather than, as in Phi-SFS, from TE biology. The answer is a single distribution per individual, pooled
over sites.

## What one site contributes

Within one posterior draw, a biallelic variant with a single mutation has exactly one
age interval — the branch the mutation sits on — and exactly one derived allele,
the one that branch's descendants carry. So a draw either does or does not put a
derived allele in a given individual:

| genotype | draws that contribute | mass |
|---|---|---|
| homozygous ALT | those calling ALT derived | `P(ALT derived)` |
| homozygous REF | those calling REF derived | `P(REF derived)` |
| heterozygous | every usable draw, using whichever allele that draw calls derived | 1 |

Each contributing draw enters with weight `1 / (usable draws at that row)`, so a
site's total mass is the posterior probability that the individual carries a
derived allele there. A site whose carried allele is derived in every draw weighs
twice one that is derived in half of them. A heterozygous site weighs 1 because
some allele is derived in every draw.

The individual's spectrum is the sum of these per-site mixtures. It is published
**unnormalized**, so its total is the expected number of segregating sites at
which the individual carries a derived allele and individuals stay comparable in
scale; `spectrum.tsv` also carries the normalized probability per bin.

## Why a per-draw polarity table is needed

The *weights* in that table could be read straight off `build_ancestral_states`,
which records how many draws called each base ancestral. The *ages* could not.

Selecting the draws in which a chosen allele is derived selects a subset of the
row's age intervals, and that subset is not an unbiased sample of the row's ages:
a draw in which the derived allele is the common one puts the mutation on a
deeper branch than a draw in which it is the rare one. Age and polarity are
correlated within a row across draws, so they have to be read from the same draw.

`normalize_tes.build_draw_polarity` therefore writes a
`(store rows) × (posterior draws)` table of ancestral base indices aligned to the
store's row order and `draw_id` numbering, so an interval's `draw_id` indexes the
matching polarity column directly. `0=A 1=C 2=G 3=T`, and `255` for a draw that
gave the row no usable single-character A/C/G/T state — which covers both a draw
lacking the site and a draw annotating it with something that cannot polarize a
biallelic variant. The two are not distinguished on purpose: neither can orient a
site, and treating them differently would be conditioning on missingness.

`--ancestral-table` is offered as an explicitly approximate alternative. It
applies the correct per-site weight to the row's *marginal* age distribution.
Use it for a quick look, not for a result.

## Draws that are excluded

A draw is usable at a row only when it contributes exactly one age interval and
names one of the two observed alleles ancestral:

- a draw with several mutations at the site has neither a single mutation age nor
  a single derived allele;
- a draw naming a third base cannot orient the site.

Both are dropped from the row's denominator rather than resolved by a rule, so
the weight is conditioned on the draws that can actually answer the question.
This is the same conditioning Phi-SFS applies to its polarity weight, and it is
stricter than the store's own `eligible` mask, which cannot see per-draw mutation
counts on a store built before they were recorded. `--min-usable-draws` then
drops rows with too few usable draws; the default, 8, matches the production
store's `minimum_usable_draws`.

## Run it

Both commands refuse to overwrite an existing output, and the spectrum checks its
output path *before* the VCF scan rather than after it.

```bash
STORE=/path/all_te_snp_age_interval_store
POSTERIOR_DIR=/path/posterior
POLARITY=results/draw-polarity-75draw
SPECTRUM=results/individual-ages

# 1. one reusable per-draw polarity table, shared by every later run
python -m normalize_tes.build_draw_polarity \
  --store "$STORE" --output "$POLARITY" "$POSTERIOR_DIR"/*.tsz

# 2. the per-individual spectra
python -m normalize_tes.individual_age_spectrum \
  --store "$STORE" \
  --draw-polarity "$POLARITY" \
  --vcf /path/chr1.vcf.gz \
  --output "$SPECTRUM" \
  --min-usable-draws 8
```

Useful flags:

| flag | purpose |
|---|---|
| `--samples` / `--samples-file` | restrict to named individuals; per-sample cost is linear |
| `--include-positions` / `--exclude-positions` | restrict or subtract a `chrom position` site list, e.g. all TE positions |
| `--allele-weighting dosage` | count each carried copy rather than each site once |
| `--bin-steps` | piecewise `WIDTH:LIMIT` bin widths in generations; `--bin-scale log`/`linear` with `--bin-min`/`--bin-max`/`--n-bins` are the alternatives |
| `--multiallelic skip` | tolerate multiallelic records instead of failing on them |
| `--merge` | sum per-chromosome parts; exact, because the mixture is a sum over sites |

Bins default to piecewise-constant widths in generations — 100 across the first
10,000, then 1,000, 5,000, 10,000, and 100,000 — because resolution should follow
where the question is rather than a single functional form. The first bin starts
at 0 and the last is unbounded, so no mass is discarded: mutations on terminal
branches genuinely have age intervals starting at 0. The operator guide,
[../derived_distribution_readme.md](../derived_distribution_readme.md), carries
the segment table.

### As SLURM array jobs

```bash
sbatch --array=0-14 --export=ALL,STORE="$STORE",TREES="$POSTERIOR_DIR/*.tsz",\
OUTPUT=results/draw-polarity-parts,PER_TASK=5 slurm/run_draw_polarity.sbatch

sbatch --export=ALL,STORE="$STORE",MERGE=1,\
PARTS="results/draw-polarity-parts/part-*",OUTPUT="$POLARITY",EXPECT_DRAWS=75 \
  slurm/run_draw_polarity.sbatch

sbatch --array=0-9 --export=ALL,STORE="$STORE",POLARITY="$POLARITY",\
VCFS="/path/chr*.vcf",OUTPUT=results/individual-ages-parts \
  slurm/run_individual_age_spectrum.sbatch

sbatch --export=ALL,STORE="$STORE",MERGE=1,\
PARTS="results/individual-ages-parts/part-*",OUTPUT="$SPECTRUM" \
  slurm/run_individual_age_spectrum.sbatch
```

## Output

| file | contents |
|---|---|
| `spectrum.tsv` | `sample, bin_index, age_low, age_high, mass, density, probability` |
| `summary.tsv` | `sample, sites_used, total_weight, mean_age`, and bin-interpolated `q05…q95` |
| `mass.npy` | `(samples × bins)` unnormalized mass |
| `bin_edges.npy` | `bins + 1` edges, first `0`, last `inf` |
| `total_weight.npy`, `mean_numerator.npy`, `sites_used.npy` | the sums a `--merge` adds |
| `samples.txt`, `metadata.json` | sample order, provenance, and per-VCF SHA-256 |

`sites_used` counts the sites where the individual was callable and the row had
enough usable draws. It is the denominator the mixture was built over, not a
count of sites that contributed mass: a homozygote at a site where its allele is
never the derived one is counted there and contributes nothing.

`mean_age` is computed from the interval midpoints themselves, not from the bins,
so it does not depend on the binning. The quantiles do: they interpolate inside a
bin assuming uniform density, and a quantile falling in the unbounded last bin is
reported at that bin's lower edge and is a lower bound.

## Resource notes

Measured on the 75-draw production store (31,240,944 rows, 1.87 billion
intervals, 18.2 GB on Quobyte), one core, chromosome 10, 26 samples:

| stage | measured |
|---|---|
| `open_snp_age_store` | 53 s |
| first `store.intervals` block, 5,000 rows, cold | 224 s |
| next `store.intervals` block, 20,000 rows, warm | 1.4 s |
| 5,000 VCF records end to end (marginal source) | 10 min 39 s wall, 2.7 s CPU, 1.6 GB peak RSS |

The 0% CPU figure is the point: this analysis is bound by Quobyte latency on the
store's `below`/`above` arrays, not by arithmetic. The first touch of a region is
expensive and the rest of that region is nearly free, so a run's cost is
dominated by a one-off cold start rather than by the number of sites. Whole-VCF
timings have not been measured; extrapolating the warm rate is a guess from two
points, not a projection.

Peak memory is set by `--chunk-records` (default 20,000) times the number of
bins and samples, plus the store's validation reads. 32 GB was ample.
