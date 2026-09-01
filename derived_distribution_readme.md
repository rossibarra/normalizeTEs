# Derived-allele age distributions per individual

A side pipeline of normalizeTE. Given VCFs and the posterior ARG draws, it
builds, for each individual, the combined distribution of the ages of the
derived alleles that individual carries.

It does not separate SNPs from TEs. Every biallelic record in the VCF that
resolves to a store row is treated the same way, and the result is one
distribution per individual over everything in the file.

It shares the interval store with the TE/control matching workflow and changes
nothing in it. No production module, launcher, or artifact of that workflow is
touched, and nothing here feeds Phi-SFS. The method rationale and the recorded
measurements live in
[docs/INDIVIDUAL_AGE_SPECTRUM.md](docs/INDIVIDUAL_AGE_SPECTRUM.md); this file is
the operator guide.

## What a site contributes

Within one posterior draw, a biallelic variant with a single mutation has exactly one
age interval — the branch the mutation sits on — and exactly one derived allele,
the one that branch's descendants carry. So each draw either does or does not put
a derived allele in a given individual:

| genotype | draws that contribute | mass at the site |
|---|---|---|
| homozygous ALT | those calling ALT derived | `P(ALT derived)` |
| homozygous REF | those calling REF derived | `P(REF derived)` |
| heterozygous | every usable draw, at whichever allele that draw calls derived | 1 |

Each contributing draw enters with weight `1 / (usable draws at that row)`, so a
site's total mass is the posterior probability that the individual carries a
derived allele there. A site whose carried allele is derived in every draw weighs
twice one derived in half of them.

A heterozygous site weighs 1, the same as a homozygous site whose allele is
derived in 100% of draws — the two differ in where the mass sits, not how much
there is. Pass `--allele-weighting dosage` to count carried copies instead, which
makes that homozygote weigh 2 and leaves the heterozygote at 1.

An individual's spectrum is the sum of these per-site mixtures, published
**unnormalized**: its total is the expected number of segregating sites at which
the individual carries a derived allele, so individuals stay comparable in scale.
`spectrum.tsv` also carries the normalized probability per bin.

Ages are in ARG generations throughout.

## Every variant is treated the same

There is no variant-class logic anywhere in this pipeline. It does not know what
a TE is, and it does not read the TE position lists the matching workflow uses.
Every VCF record that is biallelic, has A/C/G/T for both REF and ALT, and
resolves to a store row contributes, whatever kind of variant it represents. The
production store is built from a combined SNP+TE dataset, so TE presence/absence
records carry age intervals and ancestral calls just as SNPs do, and they enter
the mixture on the same terms.

One consequence is worth stating, because it differs from Phi-SFS. Phi-SFS
polarizes TE sites by biology — an insertion is the derived state, so its weight
is exactly 1 — and uses the ARG only for control SNPs. This pipeline uses the
ARG's per-draw ancestral call for *everything*, TE records included. That is the
right choice for a uniform per-individual age distribution, and it means a TE's
polarity here carries the ARG's uncertainty rather than the biological
convention. If you want the two classes handled differently, split the VCF with
`--include-positions` and `--exclude-positions` and run the pipeline twice.

Records the pipeline skips, all counted in `metadata.json`:

| skipped | count field | why |
|---|---|---|
| multiallelic records | `multiallelic_skipped` | an error by default; pass `--multiallelic skip` |
| non-A/C/G/T REF or ALT | `non_acgt_skipped` | cannot be matched against an ancestral base call |
| positions absent from the store | `unresolved` | no posterior age exists for them |
| rows below the draw floor | `rows_below_min_usable_draws` | too few draws both date and orient them |

## Draws that are excluded

A draw is usable at a row only when it contributes exactly one age interval and
names one of the two observed alleles ancestral. A draw with several mutations at
the site has neither a single mutation age nor a single derived allele; a draw
naming a third base cannot orient the site. Both are dropped from the row's
denominator rather than resolved by a rule. `--min-usable-draws` then drops rows
with too few of them; the default, 8, matches the production store's
`minimum_usable_draws`.

## Before you start

Use the project environment and confirm the suite passes:

```bash
conda activate normalizeTE
python -m pytest -q tests
```

### What you need from the main workflow

This pipeline consumes the interval store and nothing else the matching workflow
produces, so only one step of [README.md](README.md) has to have been run:

| main workflow step | needed here |
|---|---|
| [1. Build the interval store](README.md#1-build-the-interval-store) | **yes** — the only prerequisite |
| 2. Candidate control universe | no |
| 3. Preliminary TE target | no |
| 4. TE polarity mask | no |
| 5. Final target and matched controls | no |
| [6. Ancestral-state table](README.md#6-build-the-ancestral-state-table) | only for the approximate `--ancestral-table` mode below |
| 7. Phi-SFS | no |

If you have no store yet, build one as the main README describes — it needs the
posterior ARG draws and a chromosome-offset file, and it is the expensive step:

```bash
python -m normalize_tes.build_snp_interval_store \
  "$POSTERIOR_DIR"/*.tsz \
  --interval-store "$STORE" \
  --chrom-offsets "$CHROM_OFFSETS" \
  --min-usable-fraction 0.1 \
  --num-buckets 100 \
  --bucket-memory-gb 2 \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}"
```

The store is shared, not consumed: one store serves this pipeline and every TE
category the matching workflow runs, so if the store already exists, start at
step 1 below.

### Inputs

- a completed interval store recording both `content_sha256` and `inputs`. The
  current builder always writes both; a store built by an older version may lack
  the digest, in which case it cannot be matched against any polarity or
  ancestral table and is refused rather than published against;
- the store's complete set of posterior ARG draws as tszip archives — the same
  files that built it. `build_draw_polarity` authenticates every draw you pass
  against the store's recorded `inputs`, so a different set of tree files of the
  right cardinality is rejected instead of silently accepted;
- one or more biallelic VCFs whose CHROM labels match the store's chromosomes;
- optionally, a `chrom position` list, if you want to analyse only part of the
  VCF — see below;
- an output location on durable storage. Every published path must be new — both
  commands refuse to overwrite, and the spectrum checks its output path *before*
  the VCF scan rather than after it.

Set the paths once:

```bash
STORE=/path/all_te_snp_age_interval_store
POSTERIOR_DIR=/path/posterior
CHROM_OFFSETS=/path/chrom_offsets.txt
POLARITY=results/draw-polarity-75draw
SPECTRUM=results/individual-ages
```

## 1. Build the per-draw polarity table

`build_ancestral_states` records how *many* draws called each base ancestral.
That marginal is enough for Phi-SFS, which never has to know which draws voted
which way. Pairing a mutation's age with its polarity does.

```bash
python -m normalize_tes.build_draw_polarity \
  --store "$STORE" \
  --output "$POLARITY" \
  "$POSTERIOR_DIR"/*.tsz
```

| flag | purpose |
|---|---|
| `--store` | interval store whose rows and draws the table is aligned to |
| `--output` | new table directory |
| `trees` | the store's complete posterior draw set |
| `--draws` | `START:STOP` slice of the sorted tree list, for one array task |
| `--chromosome` | chromosome label when each ARG covers a single chromosome |
| `--merge` | part directories to gather into `--output` |
| `--expect-draws` | total draws a merge must contain |

The table is `(store rows) × (posterior draws)` of `uint8`: `0=A 1=C 2=G 3=T`,
and `255` for a draw that gave the row no usable single-character A/C/G/T state.
Columns are ordered by the store's own `draw_id`, so an interval's `draw_id`
indexes the matching polarity column directly, whatever order the tree files were
listed in. For the 75-draw production store that is about 2.2 GB.

`255` covers both a draw lacking the site and a draw annotating it with something
that cannot polarize a biallelic variant. The two are deliberately not
distinguished: neither can orient a site, and treating them differently would be
conditioning on missingness.

Build this once. It is reusable across every individual, VCF, and later run.

## 2. Build the spectra

```bash
python -m normalize_tes.individual_age_spectrum \
  --store "$STORE" \
  --draw-polarity "$POLARITY" \
  --vcf /path/chr1.vcf \
  --output "$SPECTRUM" \
  --min-usable-draws 8
```

| flag | purpose |
|---|---|
| `--store` | interval store supplying posterior variant ages |
| `--draw-polarity` | per-draw polarity table from step 1 |
| `--vcf` | one or more biallelic VCFs holding the genotypes |
| `--output` | new result directory |
| `--samples` / `--samples-file` | individuals to analyse; default every VCF sample |
| `--include-positions` / `--exclude-positions` | `chrom position` file to restrict to, or to subtract; the only way to split the VCF into groups |
| `--min-usable-draws` | minimum draws that both date and orient a site (default 8) |
| `--allele-weighting` | `site` (default) or `dosage` |
| `--bin-scale` | `steps` (default), `log`, or `linear` |
| `--bin-steps` | `WIDTH:LIMIT` segments in generations, used by the default `steps` scale |
| `--bin-min` / `--bin-max` / `--n-bins` | inner binning for `log` and `linear` only |
| `--multiallelic` | `error` (default) or `skip` |
| `--unknown-chromosome` | `error` (default) or `skip`, for CHROM labels absent from the store |
| `--chunk-records` | VCF records per accumulation block (default 20,000) |
| `--merge` | part directories to sum into `--output` |
| `--quiet` | suppress progress |

### Binning

Bins are in generations and default to piecewise-constant resolution, fine where
recent mutations land and coarse across the deep tail:

| segment | width | bins |
|---|---|---|
| 0 – 10,000 | 100 | 100 |
| 10,000 – 100,000 | 1,000 | 90 |
| 100,000 – 500,000 | 5,000 | 80 |
| 500,000 – 2,000,000 | 10,000 | 150 |
| 2,000,000 – 10,000,000 | 100,000 | 80 |
| 10,000,000 – ∞ | unbounded | 1 |

That is 501 bins, written as `--bin-steps 100:10000 1000:100000 5000:500000
10000:2000000 100000:10000000`. Override it with your own segments, each running
to its limit from where the previous one stopped:

```bash
--bin-steps 100:10000 5000:200000 10000:1000000
```

A limit that is not a whole number of its own bins is rejected rather than
absorbed into a short final bin, which would leave one silently irregular column
in the middle of a regular series. `--bin-scale log` and `--bin-scale linear`
remain available and use `--bin-min`, `--bin-max`, and `--n-bins` instead.

The first bin always starts at 0 and the last is unbounded, so no mass is
discarded: mutations on terminal branches genuinely have age intervals starting
at 0, and the oldest intervals in the production store reach 3.7e7 generations.

Every VCF passed in one run must declare the same samples in the same order; use
`--samples` to pin an explicit order across files that differ.

### The approximate alternative

`--ancestral-table` accepts the marginal table from `build_ancestral_states`
instead. It applies the correct per-site weight to the row's *marginal* age
distribution, so it needs no new 2.2 GB artifact — and it is wrong in a specific
way. The draws in which a chosen allele is derived are not an unbiased sample of
that row's ages: a draw where the derived allele is the common one puts the
mutation on a deeper branch than a draw where it is the rare one. Use it for a
quick look, not for a result. The weights it produces are right; the ages are not.

## Farm/Quobyte launchers

```bash
# per-draw polarity, as a 15-task array over 75 draws, then gathered
sbatch --array=0-14 --export=ALL,STORE="$STORE",TREES="$POSTERIOR_DIR/*.tsz",\
OUTPUT=results/draw-polarity-parts,PER_TASK=5 slurm/run_draw_polarity.sbatch

sbatch --export=ALL,STORE="$STORE",MERGE=1,\
PARTS="results/draw-polarity-parts/part-*",OUTPUT="$POLARITY",EXPECT_DRAWS=75 \
  slurm/run_draw_polarity.sbatch

# one spectrum task per chromosome VCF, then summed
sbatch --array=0-9 --export=ALL,STORE="$STORE",POLARITY="$POLARITY",\
VCFS="/path/chr*.vcf",OUTPUT=results/individual-ages-parts \
  slurm/run_individual_age_spectrum.sbatch

sbatch --export=ALL,STORE="$STORE",MERGE=1,\
PARTS="results/individual-ages-parts/part-*",OUTPUT="$SPECTRUM" \
  slurm/run_individual_age_spectrum.sbatch
```

The spectrum launcher also honours `SAMPLES_FILE`, `INCLUDE`, `EXCLUDE`,
`MIN_DRAWS`, `WEIGHTING`, `BIN_SCALE`, `BIN_STEPS` (quote the whole segment
list), and `BIN_MIN`/`BIN_MAX`/`N_BINS` for the log and linear scales.

Merging is exact, not an approximation: the mixture is a sum over sites, so
summing per-chromosome parts gives the same numbers as one whole-genome run. A
merge refuses parts that disagree on store, polarity table, weighting,
`--min-usable-draws`, bin edges, or sample order, and refuses to count the same
VCF twice.

## Output

| file | contents |
|---|---|
| `spectrum.tsv` | `sample, bin_index, age_low, age_high, mass, density, probability` |
| `summary.tsv` | `sample, sites_used, total_weight, mean_age, q05_age … q95_age` |
| `mass.npy` | `(samples × bins)` unnormalized mass |
| `bin_edges.npy` | `bins + 1` edges, first `0`, last `inf` |
| `total_weight.npy`, `mean_numerator.npy`, `sites_used.npy` | the sums a `--merge` adds |
| `samples.txt` | sample order for every array |
| `metadata.json` | store and polarity identity, every setting, per-VCF SHA-256, and the record counts below |

`sites_used` counts sites where the individual was callable and the row had
enough usable draws. It is the denominator the mixture was built over, not a
count of sites that contributed mass: a homozygote at a site where its allele is
never derived is counted there and contributes nothing.

`mean_age` is computed from the interval midpoints themselves, not from the bins,
so it does not depend on the binning. The quantiles do: they interpolate inside a
bin assuming uniform density, and a quantile falling in the unbounded last bin is
reported at that bin's lower edge and is a lower bound.

## Verify a run

Check `metadata.json` before using the numbers:

- `store_content_sha256` matches the store you meant to use, and
  `polarity_source` is `per-draw` rather than `marginal-approximate`;
- `counts.unresolved` is a small share of `counts.records`. A large one usually
  means a CHROM-label or coordinate-convention mismatch, not missing data;
- `counts.rows_below_min_usable_draws` is what `--min-usable-draws` removed;
- `counts.multiallelic_skipped` and `counts.non_acgt_skipped` are what the VCF
  filters dropped;
- `vcfs[].sha256` names the exact bytes scanned.

Both commands publish through a staging directory and rename, so an interrupted
run leaves no half-written result to mistake for a finished one.

## Measured cost

On the 75-draw production store (31,240,944 rows, 1.87 billion intervals, 18.2 GB
on Quobyte), one core, chromosome 10, 26 samples:

| stage | measured |
|---|---|
| `open_snp_age_store` | 53 s |
| first `store.intervals` block, 5,000 rows, cold | 224 s |
| next `store.intervals` block, 20,000 rows, warm | 1.4 s |
| 5,000 VCF records end to end | 10 min 39 s wall, 2.7 s CPU, 1.6 GB peak RSS |

That end-to-end row was measured before the binning change, on the 162-bin log
scale and through `--ancestral-table` rather than a per-draw table. The timings
are unaffected — binning is not where the time goes — but the memory figure is,
and it has not been re-measured. See below.

The 0%-CPU figure is the point: this is bound by Quobyte latency on the store's
`below`/`above` arrays, not by arithmetic. The first touch of a region is
expensive and the rest of that region is nearly free, so cost is dominated by a
one-off cold start rather than by the number of sites. Whole-VCF timings have not
been measured; extrapolating the warm rate from two points would be a guess, not
a projection.

Per-sample cost is linear in the number of samples, so `--samples` is worth using
on a sample-rich VCF.

Peak memory scales with `--chunk-records` times the bin count: the accumulator
holds a few `(chunk_records x bins)` float64 blocks while integrating each block
of rows. The 501-bin default is about three times the 162-bin scale the run above
used, so expect a peak of roughly 3-5 GB at the default `--chunk-records 20000`
rather than the 1.6 GB in the table. That is an arithmetic expectation, not a
measurement — it has not been run at 501 bins. It is well inside the 32 GB the
launcher requests either way; halve `--chunk-records` if you push the binning
much finer than the default.

The per-draw polarity build has been exercised on synthetic ARGs and not yet on
the 75 production draws; budget it like `build_ancestral_states`, which it
mirrors — decompression dominates at roughly a minute per draw.
