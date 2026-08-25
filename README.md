# normalizeTE v0.4.0

normalizeTE generates SNP control sets whose posterior age distributions match
an observed TE variant dataset. The production workflow is designed for many TE
datasets, posterior ARG draws stored as `.tsz` files, SLURM, and Quobyte.

## Recommended workflow

The supported production path has four stages:

1. Build one compact all-SNP interval store from all posterior ARG draws (§2).
2. Build one target distribution and bootstrap threshold for each TE dataset
   (§4).
3. Match 100 control sets against per-replicate bootstrap TE targets (§6).
4. Calculate Φ-SFS against the matched sets (§7), using the per-site ancestral
   table built from the same ARG draws.

`run_bootstrap_matching.sbatch` is the canonical launcher for stages 2 and 3: it
stages the store to node-local scratch, builds the target if it is missing, and
runs the matcher with resume. It is the only supported bootstrap-matching
launcher in the repository.

**Which matching workflow to use.** Bootstrap-target matching (§6) has replaced
the hard-q50 swap sampler (§5) as the reported result. Hard-q50 defines its
tolerance from bootstrap uncertainty but does not propagate it, so its saved
sets occupy a narrow shell just inside q50: on the 75-draw in-gene target its
100 sets span 1,367.98–1,480.47 generations against a 1,480.48 threshold, a
median 1.27% inside it with a standard deviation of 1.4% of the threshold.
`BOOTSTRAP_HPC_VALIDATION.md` carries the validation.

**§5 is not a prerequisite for §6.** An earlier design seeded the optimizer from
a hard-q50 bundle through a `--seed-sets` flag. That flag no longer exists: each
restart is now a stratified draw from the target's own equal-mass age strata,
which the target bundle already ships. §5 is retained for reproducing earlier
analyses and is documented in `SWAP_SAMPLER_HPC_HOWTO.md`; it is not run in
production.

Φ-SFS reads either bundle. It selects the per-replicate identifier arrays from
the bundle's `schema_version`, so no flag distinguishes them.

The interval store is built once and reused for every TE dataset. Do not build
a dense CDF store first; the dense builder is a legacy alternative, not a
prerequisite. See [Alternative and legacy workflows](#alternative-and-legacy-workflows)
for the reasons it still exists.

Each usable mutation contributes a uniform age distribution between the age of
its mutation node (`below`) and its parent node (`above`). For a TE dataset of
size \(X\), `te_age_target.py` averages the \(X\) posterior CDFs and bootstraps
the TE variants with replacement. The median bootstrap Wasserstein distance is
the default maximum mismatch allowed for a control set.

`bootstrap_target_matcher.py` (§6) gives each of its 100 replicates its own
bootstrap TE target and minimizes the exact-grid Wasserstein-1 distance to it,
so the replicates span that uncertainty instead of sitting against one fixed
boundary. §6 explains what is being matched and why.

The retained §5 sampler, `sample_age_matched_controls.py`, instead finds a set
inside the single threshold and then runs a constrained random swap walk. The
construction state itself is not saved. Each chain performs one fixed
accepted-swap sweep per set member before its first save and another fixed sweep
between saves. Membership replacement is reported as a mixing diagnostic rather
than used as a path-dependent stopping rule. Every saved set is recomputed on
the exact 1,000-generation grid and must remain inside the target threshold.

### What the 100 replicates are, and what they are not

`--disjoint-replicates` (the production setting, §6) optimizes each replicate
against the candidate universe minus every row already published. Stating the
consequence precisely matters, because an earlier version of this document
claimed more than the design delivers.

**Guaranteed, and verified on the published in-gene bundle.** The 100 sets share
no control rows: 406,700 of 406,700 slots are distinct rows, maximum reuse 1.
That removes direct SNP reuse between replicates entirely.

**Measured.** Sampling without replacement across replicates couples them
through sequential depletion — later replicates draw from a pool the earlier
ones have already thinned. The depletion is small and its effect is not
detectable here: the 100 sets consume 406,700 of 23,026,051 candidates (1.77% of
the pool), and `E_r` shows no degradation with replicate index (ordinary
least-squares slope −0.05 generations per replicate; first-25 median 53.18
against last-25 median 50.71, on a 1,480-generation acceptance threshold).

**Not claimed.** Statistical independence, and therefore not an effective
replicate count of 100. Every replicate is still built against the same observed
TE sample and the same interval store, which is inherent to bootstrapping rather
than a defect of this stage. Estimate an effective replicate count from the
downstream statistic itself before quoting a Monte Carlo standard error.

The 100 sets saved by the §5 sampler are a different object: correlated Monte
Carlo states, with correlation confined within each ten-set chain. Retain
`chain_index.npy` and `sample_index.npy` for those, and measure autocorrelation
of the actual downstream statistic before interpreting the empirical null.

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

One multiprocessing audit requires Linux `fork` and is skipped on macOS;
production validation should run the complete suite on a Linux compute node.

For reproducible production runs, use an immutable tag or exact commit rather
than a moving branch:

```bash
git fetch --tags
git checkout COMMIT_HASH
```

`v0.1.0` is the tagged q95 baseline. Version `0.2.0` introduced the bootstrap
median (q50), fixed accepted-swap sweeps, and the 10-chain distributed
workflow. Version `0.2.1` adds adaptive construction and stronger distributed
integrity checks. Version `0.3.0` adds the Φ-SFS analysis step (§7). Version
`0.3.1` adds bootstrap-target matching (§6) and makes it the recommended
matching stage: every acceptance gate in `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §10
is closed or superseded on the 75-draw production store, and
`BOOTSTRAP_HPC_VALIDATION.md` records the evidence. Release changes are
summarized in [CHANGELOG.md](CHANGELOG.md).

The **matching** stages (§4 and §6) are cleared for production. The **Φ-SFS**
stage (§7) is not: the derived-frequency arm of C2 is complete, but its
polarity-confidence extension, an effective replicate count estimated from the
Φ-SFS scores, and the input VCF for chromosomes 1–9 remain open.

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

Version 0.2.1 records a SHA-256 identity over every interval-store array plus
the metadata needed to interpret it. Distributed matching requires this digest
in both the interval store and target directory and rejects a mismatch. Rebuild
stores and targets made by earlier versions before using the distributed
workflow.

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
bootstrap median (`--acceptance-quantile 0.50`). The walk has no preference for
smaller distances once it is feasible, so this threshold is the matching-quality
specification, not merely a safety margin; saved distances normally concentrate
near it. Use another quantile only for an explicitly labeled sensitivity
analysis, and never reuse one target output path across tolerances.

For a pre-specified scientific tolerance, replace the quantile-derived boundary
with an absolute Wasserstein limit in generations:

```bash
python te_age_target.py ... --acceptance-distance 1500
```

Bootstrap distances are deliberately still produced for context, while
metadata records that the absolute distance supplied the operative threshold.
Thus `--acceptance-distance` changes the threshold but does not skip the
bootstrap computation.

For an interval store, target construction creates a temporary float32
TE-by-age CDF matrix under `--scratch-dir`. At roughly 185,000 TEs and the
measured maximum age, allow approximately 27 GB of additional scratch on the
75-draw production store, whose 36,746-point grid is 1.6x wider than earlier
two-draw measurements implied.
The output directory publishes atomically and must not already exist.

Important target outputs include:

- `te_chromosomes.npy`, `te_positions.npy`, and `te_row_indices.npy`;
- `target_cdf.npy` and `age_bins.npy`;
- `bootstrap_wasserstein.npy`; and
- `metadata.json`, including the threshold, parameters, position resolution,
  seed, store identity, and software provenance.

## 5. Generate 100 matched control sets (retained, not the production path)

> This is the hard-q50 swap sampler. It has been **replaced by §6** as the
> reported result and is no longer a prerequisite for it — §6 initializes itself
> from the target's own age strata. Run §5 only to reproduce an earlier analysis
> or as a labelled sensitivity analysis. `SWAP_SAMPLER_HPC_HOWTO.md` is its
> runbook.

For one target:

```bash
python sample_age_matched_controls.py \
  --store age_interval_store \
  --target targets/in_gene \
  --all-eligible \
  --output matches/in_gene \
  --work-dir "${TMPDIR:?TMPDIR is not set}/match-in-gene" \
  --sets 100 \
  --chains 10 \
  --sets-per-chain 10 \
  --workers 1 \
  --seed 1002
```

`--all-eligible` uses every eligible non-target SNP as the candidate universe.
To restrict controls, resolve the desired universe to canonical store rows and
pass the resulting one-dimensional NumPy array with `--candidate-rows`.
Target rows are always excluded.

The defaults are:

- 100 saved sets = 10 independent chains × 10 sets;
- one accepted-swap sweep per set member during burn-in; and
- one accepted-swap sweep per member between saved states.

Construction starts on the coarse `--search-bin-width 20000` grid. If an epoch
accepts no improving swaps while the exact distance is still too large, the
sampler halves that width and recomputes construction state, stopping at the
exact target-grid width. Three consecutive zero-acceptance epochs at exact
resolution produce an explicit plateau error instead of wasting the remaining
construction budget. Refinements are recorded in chain diagnostics; the
matching threshold is never relaxed.

The local CLI defaults to one worker to bound memory. Increase `--workers` only
when the node can hold one exact selected-row CDF cache per active worker. The
production SLURM workflow instead runs each chain as a separate one-CPU array
task that the scheduler can place independently across nodes.

Each output directory contains:

- `row_indices.npy`, shape `(100, X)`;
- `positions.npy`, `chromosome_codes.npy`, and `chromosome_labels.npy`;
- `cdfs.npy` and `wasserstein.npy` for all 100 sets;
- `target_cdf.npy` and `age_bins.npy` used for exact certification;
- `chain_index.npy` and `sample_index.npy`;
- `diagnostics.csv` with construction, burn-in, thinning, and save records;
- `reuse_row_indices.npy` and `reuse_counts.npy`; and
- `metadata.json` with seeds, settings, chain histories, membership overlap,
  Wasserstein autocorrelation, an overlap-based ESS heuristic, store identity,
  and software provenance.

Final publication fails if any set contains duplicate controls, violates the
declared candidate universe, has a stored CDF inconsistent with its rows, or
exceeds the target's exact Wasserstein threshold. With `--work-dir`, temporary
checkpoints and result assembly stay on local scratch; the completed directory
is copied to a temporary sibling of `--output` and exposed by an atomic rename.

For the local all-chain command, `--resume` reuses completed chains only while
the same work directory still exists. Production restarts use durable completed
chain bundles as described below; an interrupted individual chain restarts from
its deterministic seed.

### Using the 100 matched sets

Compute the same downstream statistic once for every row of `row_indices.npy`
and compare the observed TE statistic with that empirical matched-control
distribution. Preserve `chain_index.npy` and `sample_index.npy` when joining
results. Plot the statistic by chain and estimate its within-chain
autocorrelation; correlation does not bias a simple Monte Carlo average, but it
reduces precision and must not be ignored in standard errors or effective
replicate counts.

`reuse_row_indices.npy` and `reuse_counts.npy` report how often each control SNP
appears across all sets. Use them to detect a small subset of controls
dominating the null distribution. The `chain_diversity` metadata provides
generic age-distance and membership diagnostics, but the decisive mixing check
must use the actual scientific statistic being tested.

## 6. Optimize controls against bootstrap TE targets

> **This is the recommended matching stage**, replacing §5 rather than running
> alongside it. Every acceptance gate in `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §10
> is closed or superseded on the 75-draw production store; see
> `BOOTSTRAP_HPC_VALIDATION.md` for the evidence and
> `BOOTSTRAP_DISCARDED_APPROACHES.md` for what was tried and rejected.
>
> One non-default flag carries the result and should be treated as the
> production default: `--disjoint-replicates`, which optimizes each replicate
> against the candidate universe minus every row already published. Read
> "What the 100 replicates are, and what they are not" above for exactly what
> that does and does not buy.
>
> Several earlier options are gone from the CLI, their behaviour now
> unconditional. The coarse swap screen is always a geometric sub-sample of the
> exact grid (there is no `--search-grid-spacing`), and restarts are always stratified
> draws from the target's own age strata (there is no `--seed-sets`, and no
> `--closest-restarts`/`--diverse-restarts`; `--restarts` sets the count).
> Against a uniform screen and non-disjoint replicates on the same target, the
> current defaults take unique controls from 195,836 to 406,700 — 1.56x the §5
> sampler rather than 0.75x — maximum reuse from 30 to 1, QC from 96/100 to
> 100/100, and the relative age error at the 10% CDF quantile from +21.9% to
> −0.2%.

`bootstrap_target_matcher.py` propagates uncertainty in the TE age CDF by
assigning every control set its own bootstrap TE target. It then performs
improvement-only SNP swaps and saves the best certified state. It does not
perform the constrained random walk used by the §5 workflow.

### What this stage actually matches

The flags below say how; this says what.

**Aggregate posterior age CDFs, not per-variant ages.** The target is the mean
of the TE sites' posterior age CDFs, and a control set is scored by the mean of
*its* sites' CDFs. Nothing requires any individual control SNP to resemble any
individual TE. On the 4,067-site in-gene target the aggregate CDF reaches 10% at
1,614 generations, while the median *site's own* CDF reaches 10% only at 6,845
— because 73.8% of sites put a little mass below 2,000 generations, and a little
mass from three quarters of the sites is most of the young tail. Reading the
aggregate 10% crossing as "10% of these TEs are younger than 1,614 generations"
is wrong, and it is the easiest mistake to make with this output.

**The acceptance threshold is a percentile of distances, not of ages.**
`te_age_target.py` resamples the TE sites 10,000 times with replacement, and for
each resample measures the Wasserstein-1 distance between the resampled age CDF
and the observed one. The threshold is the median of those 10,000 *distances* —
1,480.48 generations for the in-gene target. It answers "how far from the
observed age distribution does the TE sample's own sampling noise typically put
you?", and a control set is acceptable when it is no further away than that. It
is not the median TE age, nor an age quantile of any kind.

**Every replicate gets its own bootstrapped target.** Replicate `r` is optimized
against bootstrap target `T^(r)`, not against the observed target `T`. That is
what carries age-CDF uncertainty into the null: the spread of control sets is
allowed to be as wide as the TE sample's own uncertainty, instead of being
squeezed against one fixed boundary. The §5 hard-q50 sampler uses the same
bootstrap only to *define* a tolerance and then discards it, which is why its
100 sets sit in a narrow shell just inside that tolerance.

**The swap screen is geometric because a uniform one is blind at the young
end.** Scoring every proposed swap on the exact 36,746-point analysis grid
dominates the run, so proposals are screened on a coarse grid first and every
recorded distance is then recertified exactly. The old coarse grid was uniform
at `--search-bin-width` 20,000 generations, and on the in-gene target that put
50.06% of the age mass inside its single first cell: the optimizer could not see
young-end structure at all and rejected young-improving swaps before the exact
grid ever evaluated them. The coarse grid is now a geometric sub-sample of the
exact grid, so the young end keeps full exact resolution. Measured effect on the
relative age error at the 10% CDF quantile: +21.9% under the uniform screen,
−0.2% under the geometric one, with no loss at the old end. Every coarse point
is an exact-grid point, so the screen is a sub-sample of the exact objective
rather than a different discretization, and a coarse misjudgement can cost
search efficiency but never the correctness of the published state.

### Running it

The production launcher is `run_bootstrap_matching.sbatch` (§8). The equivalent
direct command is:

```bash
python bootstrap_target_matcher.py \
  --store age_interval_store \
  --target targets/in_gene \
  --candidate-rows candidate_rows.npy \
  --output bootstrap_matches/in_gene \
  --work-dir results/work-in-gene \
  --resume \
  --replicates 100 \
  --restarts 3 \
  --disjoint-replicates \
  --seed 1002
```

Keep `--work-dir` on durable storage, not on `$TMPDIR`: completed replicate
bundles are what `--resume` reuses after a preemption or a time limit, and
node-local scratch does not survive the job. The store, by contrast, must be
staged to node-local scratch — see §8.

Use `--all-eligible` instead of `--candidate-rows` only when the intended
control universe is every eligible non-target SNP. Candidate and store
identities are validated against the target.

For replicate `r`, the output records:

\[
B_r=D(T^{(r)},T),\qquad
E_r=D(S_r,T^{(r)}),\qquad
O_r=D(S_r,T),\qquad
R_r=E_r/B_r.
\]

The optimizer QC requires `R_r < 0.5` and an absolute cap on `E_r`. The cap
scales with the target's own acceptance threshold —
`--qc-max-absolute-fraction` times that threshold, default 0.34, which
reproduces the historical 500-generation cap at the in-gene target — so it does
not have to be re-tuned per target size. Override it with `--qc-max-ratio` or a
fixed `--qc-max-absolute` only under a prespecified calibration. These are
convergence diagnostics, not independent evidence of biological validity. The
scientific propagation check is whether the distribution of `O_r` reproduces
`B_r` across its center and tails.

Defaults are `--restarts 3` independent stratified restarts per replicate, 10
minimum and 50 maximum exact proposal epochs, and five materially stagnant
epochs for convergence. Each restart draws its starting set by filling the
target's 20 equal-mass age strata from the candidate pool, so the start is
already shaped like the target and no external seed library is needed. Material
improvement is scaled to `B_r`. Every restart retains its complete best-W1 trace
and certified best rows/CDF. The published state is the minimum-W1 result across
the prespecified restarts; selection never uses Φ-SFS.

Completed replicate bundles are saved under the work directory. After an
interruption, repeat the identical command with `--resume`. Provenance or
parameter differences are rejected. Successful publication is atomic and
removes the work directory unless `--keep-work` is supplied.

Canonical outputs include:

- bootstrap counts and target CDFs;
- selected and per-restart SNP rows/CDFs;
- `B_r`, `E_r`, `O_r`, `R_r`, QC, and triangle-inequality arrays;
- full restart traces;
- `replicates.csv` and `restarts.csv`;
- chromosome/position and SNP-reuse arrays; and
- metadata containing input identities, configuration, seeds, and warnings.

### What the bootstrap-target Φ-SFS distribution does and does not mean

Read this before interpreting any number from this stage.

- It **holds the observed TE SFS fixed** and varies only which SNPs are matched.
  It therefore measures how the matched-control comparison responds to
  uncertainty in the TE *age* CDF, and nothing else.
- It is **not a bootstrap confidence distribution** for the TE SFS, and **not a
  p-value**. Do not compute the TE SFS from the resampled TE rows: that would
  additionally propagate finite-TE-set SFS uncertainty and answer a different
  question. A joint age-and-SFS bootstrap would be a separately named analysis.
- 100 replicates give **weak tail resolution**, and disjoint membership does
  not make them 100 independent observations. `reuse_counts.npy` shows whether
  membership is shared — under `--disjoint-replicates` it is not — but shared
  membership is only one of the couplings. Estimate an effective replicate count
  from the Φ-SFS scores themselves rather than treating their spread as 100
  independent draws.

**The main risk to watch.** The optimizer chooses SNP membership to hit a
precise age CDF. If a SNP's usefulness for repairing W1 is correlated with its
derived allele frequency, the matching step will itself bias the SFS — which is
exactly the quantity Φ-SFS measures, so the bias would be invisible in the
matching diagnostics and appear as signal. Test the association between
W1-repair utility and derived-frequency contribution before trusting any
Φ-SFS difference from this stage. `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §8 sets
out the diagnostic.

The implemented bootstrap is an iid multinomial TE-site bootstrap. Do not give
the 100 replicates an inferential interpretation until spatial dependence among
TE age contributions has been assessed. If iid exchangeability is unsupported,
a prespecified genomic-block bootstrap must replace it.

Gate 10 originally required the hard-q50 sampler to be run alongside as a
sensitivity analysis. That gate was **superseded**: publishing two null
distributions requires justifying which one is reported, so a single prespecified
method was adopted instead. §5 remains available for reproducing earlier
analyses. See `BOOTSTRAP_TARGET_MATCHING_PLAN.md` for the statistical design and
the RNA in-gene pilot, and `BOOTSTRAP_HPC_VALIDATION.md` for where each gate
stands.

## 7. Calculate Φ-SFS

`phi_sfs.py` compares the unfolded SFS of the target TE set with every
published matched SNP set. Allele *counts* come from one biallelic VCF rather
than from the subset of posterior ARGs in which a site is represented: posterior
ARG presence affects age matching, but it does not change how many copies of an
allele were observed.

**Which allele is derived, however, does come from the ARG.** The SINGER input
VCF for this dataset is not polarized — comparing it against the ARG's own
inferred ancestral states, 31% of chromosome-10 sites are exactly REF/ALT
swapped — so treating REF as ancestral would silently mis-polarize about a third
of every spectrum. The unfolded spectrum therefore does depend on the ARG, even
though the counts do not. `phi_sfs.py` consequently **requires**
`--ancestral-table`, a directory built by `build_ancestral_states.py` from the
same posterior draws as the interval store. There is no `--ancestral-mode` and
no `--ancestral-info`; reading polarity out of the VCF is no longer possible.

The prespecified polarity treatment, with its evidence, is in
`BOOTSTRAP_HPC_VALIDATION.md`: TE sites are polarized biologically (a TE
insertion is the derived state), and control SNPs by a posterior-weighted linear
mixture over the ARG's per-site ancestral calls, used uncalibrated. Only the
control arm consults the table.

Build the table once per store, from the same draws:

```bash
python build_ancestral_states.py \
  --store age_interval_store \
  --output ancestral_states \
  project-data/posterior/*.tsz
```

For a job array, give each task a slice with `--draws START:STOP` and a distinct
`--output`, then sum the parts with `--merge part_000 part_001 ...`, no tree
arguments, and `--expect-draws N` for the number of posterior draws in the full
store. The output directory must not already exist; a merge validates that the
parts share a non-null store identity, contribute a disjoint draw set, and
contain exactly `N` distinct draws.

Then run Φ-SFS against a matched bundle. The §6 bootstrap bundle is the reported
one:

```bash
python phi_sfs.py \
  --target targets/in_gene \
  --matches bootstrap_matches/in_gene \
  --vcf variants.vcf.gz \
  --ancestral-table ancestral_states \
  --output phi_sfs/in_gene_bootstrap
```

A §5 hard-q50 bundle is passed the same way; nothing else changes:

```bash
python phi_sfs.py \
  --target targets/in_gene \
  --matches matches/in_gene \
  --vcf variants.vcf.gz \
  --ancestral-table ancestral_states \
  --output phi_sfs/in_gene
```

Both bundles must have been built from the target given to `--target`. That is
enforced, not assumed: `phi_sfs.py` recomputes the target's `target_digest` and
requires the bundle to record the same value.

The per-replicate identifier columns follow the bundle's `schema_version` and
are carried into every output:

| bundle | `schema_version` | identifiers |
|---|---|---|
| §5 swap sampler | `swap-age-matched-controls-v1` | `chain_index`, `sample_index` |
| §6 bootstrap matcher | `bootstrap-target-matches-v1` | `replicate_id` |

Bootstrap replicates deliberately have no chain or sample columns: they have no
chain structure, so inventing those columns would imply a within-chain
correlation that does not exist. "No chain structure" is not "independent" —
see "What the 100 replicates are, and what they are not" above.

Plain, `.gz`, and `.bgz`/`.bgzf` VCFs are all accepted. The scan reports
progress periodically; pass `--quiet` to suppress it.

Before reading the VCF, `phi_sfs.py` recomputes the target's `target_digest`
and requires it to equal the one recorded in `matches/metadata.json`. This is
what proves the matched sets were sampled for *this* target: the store hashes
alone cannot, because every target built from one SNP store shares them. A
matched bundle from a different TE category is rejected rather than silently
compared.

### Site assumptions

These are assumptions about the input, not things the script derives. Each one
is also recorded in the output `metadata.json`.

- **Biallelic.** Records are assumed biallelic, which is what the upstream
  preprocessing produces. A comma in ALT is treated as an error rather than
  split into separate alleles here.
- **FILTER is ignored.** The declared input is the already-filtered
  preprocessing VCF, so every record at a requested coordinate is used
  regardless of its FILTER value.
- **Polarity comes from the ancestral table, not the VCF.** REF is not assumed
  ancestral, and no INFO field is consulted. TE target sites are polarized
  biologically; control sites take the posterior-weighted mixture from
  `--ancestral-table`.
- **One allele per individual.** Each callable inbred individual contributes
  one observed allele. Haploid calls and homozygous diploid calls are accepted.
  A missing diploid allele makes that individual missing at the site.
  Heterozygous calls fail by default; pass `--heterozygous missing` to exclude
  heterozygous individuals from that site's callable count.

Every requested site must be present in the VCF; the run fails listing the
missing coordinates rather than analyzing a subset.

For a site with `k` derived alleles among `n` callable individuals, sites with
`n < 20` are dropped and eligible sites are projected probabilistically to 20:

\$[
h_j(k,n)=
\frac{\binom{k}{j}\binom{n-k}{20-j}}{\binom{n}{20}},
\qquad j=0,\ldots,20.
$]\

Only unfolded bins 1 through 19 enter the comparison. Individual site
projections are **not** renormalized after removing endpoint bins, so a site
contributes `1 - h_0 - h_20` rather than one: a site whose derived count is
likely to project to 0 or 20 supplies proportionally less polymorphic mass,
which is the intended weighting. Site contributions are first summed within the
TE target and within each matched SNP set; the two completed spectra are then
normalized independently. For target spectrum `t` and matched-set spectrum
`s_r`, the score is

\$[
\Phi_{\mathrm{SFS},r}
=\sum_{j=1}^{19}\max(t_j-s_{rj},0)
=\sum_{j=1}^{19}\max(s_{rj}-t_j,0)
=\frac{1}{2}\sum_{j=1}^{19}|t_j-s_{rj}|
=1-\sum_{j=1}^{19}\min(t_j,s_{rj}).
]$\

All four forms are equal because both spectra sum to one. **Φ-SFS is therefore
the total variation distance between the two projected, normalized spectra** —
it is not a bespoke quantity, and the standard properties of that distance
apply. In particular `0 ≤ Φ-SFS ≤ 1`, and the statistic is *symmetric* in its
two arguments even though the stored bin-level residuals are oriented as TE
minus SNP. Zero means the two spectra coincide; one means they share no mass in
any bin. The last form makes the "non-overlap" reading literal: `1 - Φ` is
exactly the mass the two spectra hold in common.

The script computes the first three forms independently and checks that they
agree, reporting the discrepancy as `identity_max_abs_error`.

**Interpretation caveat.** Normalizing each spectrum discards its absolute
scale, so two sets with very different eligible-site counts, missingness, or
endpoint mass can produce identical spectra. Those differences are invisible in
Φ and must be inspected separately: `replicates.csv` reports `input_sites`,
`eligible_sites`, `dropped_n_lt_20`, and both `retained_fraction` and
`endpoint_fraction` for every set, with the matching `target_*` values in
`metadata.json`. A target and a control set whose retained fractions differ
substantially are not really comparable, however small Φ is.

The output directory contains canonical NumPy arrays for raw and normalized
spectra, TE-minus-SNP residuals, positive residual contributions, the Φ-SFS
scores, and aligned chain/sample indices. `replicates.csv` contains filtering,
endpoint-mass, overlap, score, and identity-check diagnostics. `bins.csv`
contains the raw and normalized spectra and residual contribution for every
replicate and bin. `metadata.json` records the VCF hash, the site assumptions
above, the recomputed `target_digest`, and the same software and Git provenance
as the target and matched-control steps. The output directory must not already
exist and is published atomically.

The 100 scores are matched-control replicates, not 100 independent biological
replicates. For a §5 bundle, inspect them by `chain_index`, because the ten
states saved from each chain are correlated. A §6 bundle has no chain structure
and, under `--disjoint-replicates`, no shared control SNPs either; what remains
is sequential depletion of the candidate pool plus the shared observed TE sample
and store. Retain the SNP reuse diagnostics, and estimate an effective replicate
count from the Φ-SFS scores themselves.

### Polarity: what is assumed, what is measured, and what is accepted

Polarity does not come from the VCF. This dataset's input VCF is unpolarized —
compared against the ARGs' own inferred ancestral states, **31% of sites are
exactly REF/ALT-swapped** — so treating REF as ancestral would mis-polarize
about a third of every spectrum, silently. Two sources are used instead.

**TE sites are polarized biologically.** A TE insertion is the derived state.
The genotyping convention encodes presence as ALT — every one of the 12,614
chromosome-10 TE records is `A`/`G`, with no other combination — so ALT is
derived at every TE site. The known exception is a TE that reached fixation and
was later removed by a deletion, which makes the deletion derived; that is rare
here, with only 3.1% of TE sites above insertion frequency 0.9.

**Control SNPs are polarized by the ARG, as a probability rather than a call.**
A site contributes `p·h(k,n) + (1−p)·h(n−k,n)`, where `p` is the posterior
proportion of draws calling ALT derived, over the draws that gave the site a
usable ancestral call. The mixture is linear in `p` and therefore unbiased at any
draw count. Majority rule is deliberately **not** used: thresholding `p` is
biased in a way that depends on how many draws a site appears in, and TE and
control sites differ systematically in that count.

#### Two measured biases, both accepted

**The ARG's polarity confidence is overconfident.** TE sites are a labelled test
set, since biology fixes their answer. Where all 75 draws agree the ARG is right
only about 91% of the time, so `p` measures posterior consistency rather than
accuracy — the draws share data and model, and are wrong together. `p` is used
uncalibrated. The consequence to carry when reading results: control spectra are
somewhat sharper than the ARG's measured accuracy warrants.

**The optimizer prefers well-dated SNPs, and dating confidence tracks polarity
confidence.** A narrow age posterior gives a sharper per-site CDF, which is more
useful for shaping an aggregate CDF precisely, so the optimizer selects such
sites: median across-draw age SD is 0.62 of the TE target's, against 0.83 for the
older sampler. That is the same underlying ARG certainty as polarity —
`corr(age-posterior SD, polarity confidence) = −0.25`, running from mean `p`
0.992 in the best-dated fifth of candidates to 0.880 in the worst.

The reason this is acceptable rather than disqualifying is that **it does not
create a mismatch between the two arms**:

| set | mean polarity confidence `p` |
|---|---:|
| candidate pool | 0.9367 |
| §5 sampler controls | 0.9752 |
| §6 optimizer controls | **0.9786** |
| TE target | **0.9785** |

Selection shifts controls away from the pool, but the sampler shifts them almost
as far, and both land on the TE target to four decimal places — because the
target is young and therefore well-resolved too. So control spectra are not
systematically more or less folded than the TE spectrum. What remains is that
controls are drawn from the better-resolved part of the ARG, which correlates
with genomic context; no downstream consequence has been demonstrated, and
nothing in the age-matching diagnostics would reveal one.

Report the distribution of `p` for the target and for each control set alongside
any Φ-SFS result, so a reader can see how much of the spectrum rests on
contested calls.

## 8. Run many TE datasets on Farm/Quobyte

The three stages that need a scheduler each have a launcher. All take their
inputs from environment variables, submit with `sbatch`, and are the commands the
rest of this document refers to.

| stage | launcher | notes |
|---|---|---|
| control matching | `run_bootstrap_matching.sbatch` | stages the store to node-local scratch; durable `--work-dir` with `--resume` |
| ancestral table | `run_ancestral_table.sbatch` | one job over all draws, or a SLURM array plus a gather with `MERGE=1` |
| Φ-SFS | `run_phi_sfs.sbatch` | needs a VCF covering every requested site, genome-wide |

```bash
# 1. controls
sbatch --export=ALL,STORE=/path/interval_store,TARGET=results/targets/in_gene,\
OUTPUT=results/bootstrap_matches/in_gene,CANDIDATE_ROWS=results/candidate-rows.npy \
  run_bootstrap_matching.sbatch

# 2. ancestral polarity, as an array of 15 tasks of 5 draws, then gathered
sbatch --array=0-14 --export=ALL,STORE=/path/interval_store,\
TREES="/path/run.combined.*.tsz",OUTPUT=results/ancestral-parts,PER_TASK=5 \
  run_ancestral_table.sbatch
sbatch --export=ALL,STORE=/path/interval_store,MERGE=1,\
PARTS="results/ancestral-parts/part-*",OUTPUT=results/ancestral-75draw,EXPECT_DRAWS=75 \
  run_ancestral_table.sbatch

# 3. Phi-SFS
sbatch --export=ALL,TARGET=results/targets/in_gene,\
MATCHES=results/bootstrap_matches/in_gene,VCF=/path/all.chr.vcf.gz,\
ANCESTRAL=results/ancestral-75draw,OUTPUT=results/phi_sfs/in_gene \
  run_phi_sfs.sbatch
```

### The production launcher

`run_bootstrap_matching.sbatch` runs §4 and §6 for one target. It is the only
supported bootstrap-matching launcher; the eleven one-off experiment scripts
from the validation campaign have been removed, along with the removed CLI
options they passed. Every parameter is an environment variable, so one file
serves every category:

```bash
mkdir -p /quobyte/project/normalizeTE/logs

sbatch --export=ALL,\
PROJECT=/quobyte/project/normalizeTE,\
STORE=/quobyte/project/data/snp_interval_store,\
TARGET=/quobyte/project/targets/in_gene,\
TE_POSITIONS=/quobyte/project/te/in_gene.pos.txt,\
OUTPUT=/quobyte/project/bootstrap_matches/in_gene,\
CANDIDATE_ROWS=/quobyte/project/candidate-rows.npy,\
WORK_DIR=/quobyte/project/work/in_gene,\
REPLICATES=100,SEED=1002 \
  run_bootstrap_matching.sbatch
```

Three properties of that launcher are not optional and are the reasons it
exists:

- **It is submitted with `sbatch`, never `srun`.** An `srun` started from an
  interactive shell dies when that shell does, which is how the first production
  attempt was lost after 48 minutes.
- **It rsyncs the interval store to node-local `$TMPDIR` first**, after checking
  that the scratch filesystem holds the store size plus 20%. Against the store
  on Quobyte the job is I/O-bound and projects to 4–21 h of pure store reads for
  one target; staged, the same run measured 97% CPU with 21 major page faults.
  The production store is 18.2 GB.
- **`WORK_DIR` is on durable storage and `--resume` is always passed.** Each
  completed replicate is written there as its own provenance-locked bundle, so a
  preemption or a time limit costs the replicate in flight and nothing else.
  Resubmit the identical command; a provenance or parameter difference is
  rejected rather than silently mixed.

Resource requests come from measurement, not from a formula. At the launcher's
defaults (6 CPUs, 96 GB, 12 h) a 4,067-site target with 100 replicates × 3
restarts took 2 h 12 m wall clock and 37.7 GiB peak RSS. Target construction is
the other memory peak: 16.0 GB for a 35,512-site target. Scale `--time` roughly
linearly in target size and replicate count.

`CANDIDATE_ROWS` is the array written by `build_candidate_rows.py`; set it to
`all` to use every eligible non-target row instead. Omit `TE_POSITIONS` when
`TARGET` already exists, and set `REPLICATES`, `RESTARTS`, `ACCEPTANCE_QUANTILE`
or `MISSING_POSITION_POLICY` to depart from the defaults.

### The §5 sampler's manifest workflow (retained)

The rest of this section drives the §5 hard-q50 sampler, which is no longer the
production path. Use it only to reproduce an earlier analysis.

For many categories, create one tab-delimited manifest on Quobyte:

```text
label	positions	target	output	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young	/quobyte/project/matches/young	1003
```

`positions` is the input chromosome/VCF-position text file. `target` is a
durable **output directory**, not a single file: `build_age_targets.sbatch`
creates it with the target CDF, exact age grid, bootstrap distances, resolved
TE rows, threshold, store identity, and metadata. `output` is the separate
durable directory where gather publishes the 100 matched control sets. Give
every manifest row unique `target` and `output` directory paths; they must not
already contain incomplete results.

Target construction uses `build_age_targets.sbatch`. Matching uses two stages:
`sample_age_matches.sbatch` runs one independently seeded chain per array task,
and `gather_age_matches.sbatch` validates ten durable chain bundles before
publishing one final 100-set directory. Every task stages the immutable interval
store to its own `$TMPDIR`; active checkpoints and CDF caches never live on
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

For `T` manifest rows, matching requires `10*T` chain tasks and `T` gather
tasks. Edit the launchers' `#SBATCH --array` ranges accordingly, then submit the
gather array with an `afterok` dependency:

```bash
export T=3  # number of non-header rows in this example manifest
export AGE_MATCH_CHAINS=10
export AGE_MATCH_SETS_PER_CHAIN=10
export AGE_MATCH_CHAIN_TASK_COUNT=$((10 * T))
export AGE_MATCH_GATHER_TASK_COUNT=$T

chain_job=$(sbatch --parsable \
  --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_CHAINS,AGE_MATCH_SETS_PER_CHAIN,AGE_MATCH_CHAIN_TASK_COUNT \
  sample_age_matches.sbatch)

sbatch --dependency="afterok:${chain_job}" \
  --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_CHAINS,AGE_MATCH_SETS_PER_CHAIN,AGE_MATCH_GATHER_TASK_COUNT \
  gather_age_matches.sbatch
```

Each launcher verifies `SLURM_ARRAY_TASK_COUNT` against its declared task count,
so a shortened array cannot silently omit targets or chains. A chain task writes
only one atomically published `OUTPUT.chains/chain-NNN.npz` bundle to Quobyte
before its scratch directory disappears. Rerunning the array validates and
skips existing complete bundles. The gather job refuses to publish until all ten
bundles match the target, candidate universe, full store-content identity,
derived chain seeds, software provenance, and every exact row-derived CDF
check. Concurrent duplicate chain tasks cannot overwrite an existing bundle.

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

## Document map

- `run_bootstrap_matching.sbatch` is the canonical production launcher (§8).
- [BOOTSTRAP_HPC_VALIDATION.md](BOOTSTRAP_HPC_VALIDATION.md) is the validation
  record for the production matching route: measurements, gate status, and the
  prespecified polarity decisions.
- [BOOTSTRAP_TARGET_MATCHING_PLAN.md](BOOTSTRAP_TARGET_MATCHING_PLAN.md) is the
  statistical design behind §6.
- [BOOTSTRAP_DISCARDED_APPROACHES.md](BOOTSTRAP_DISCARDED_APPROACHES.md) records
  what was tried and rejected, with the evidence.
- [SWAP_SAMPLER_HPC_HOWTO.md](SWAP_SAMPLER_HPC_HOWTO.md) is the Farm/Quobyte
  runbook for the retained §5 swap sampler, not for the production path.
- [AGE_MATCHED_CONTROL_SAMPLER_PLAN.md](AGE_MATCHED_CONTROL_SAMPLER_PLAN.md)
  records the §5 sampler design and validation gates.
- [CHANGELOG.md](CHANGELOG.md) records release-level behavior changes.
- `INTERVAL_STORE_*`, `GLOBAL_QUANTILE_*`, and `CODE_REVIEW*` documents are
  design history and review records, not operator instructions.

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
works directly from the canonical interval store and uses fixed accepted-swap
sweeps so saving is not triggered by a path-dependent replacement crossing.
