# normalizeTE v0.4.0

normalizeTE builds SNP control sets whose posterior age distributions match an
observed TE variant dataset, then compares the unfolded site frequency spectrum
of the TEs against those controls. It works from posterior ARG draws, one TE
position file per biological category, and one biallelic VCF. The products are a
reusable all-SNP age-interval store, a per-category TE age target, 100 matched
control sets, and a Φ-SFS score for each of them.

The current release is 0.4.0; `CHANGELOG.md` records behavior changes between
releases. Every validation gate is closed, for the matching stages and for
Φ-SFS alike; `BOOTSTRAP_HPC_VALIDATION.md` §5 records the evidence.

## Input data

```text
project-data/
├── posterior/
│   ├── draw_001.tsz
│   └── ...
├── snp/
│   └── all_snp.pos.txt        # the filtered SNP list controls are drawn from
├── te/
│   ├── all_te.pos.txt         # the TE superset, excluded from the control pool
│   ├── in_gene.pos.txt        # one file per analysis category
│   └── ...
└── chrom_offsets.txt          # only needed when ARG metadata is absent
```

What a tool accepts depends on how it opens the draws.
`build_snp_interval_store.py` loads through `tszip.load()`, which takes both
tszip-compressed `.tsz` archives and ordinary tskit tree-sequence files.
`build_ancestral_states.py` and `build_te_polarity_mask.py` call
`tszip.decompress()` and therefore require `.tsz`. The standalone helper
`snp_age_distribution.py` uses `tszip.load()` and takes either.

Every position file holds exactly two whitespace-separated columns, chromosome
and 1-based VCF position. Blank lines and `#` comments are allowed. Do not
pre-convert to cumulative or zero-based coordinates; normalizeTE maps a VCF
position to `chromosome_offset + POS` internally.

```text
chr4 100
chr4 27591
chr7 802
```

Chromosome labels must match the ARG's embedded `chrom_offsets` metadata, or the
optional offsets file passed to step 1. That file is whitespace-separated with
`#` comments, and takes either two columns (chromosome, length; offsets
accumulate in file order, which must match the concatenation order in the ARG)
or three columns (chromosome, cumulative offset, length). Offsets must be
strictly increasing and non-overlapping, names unique, and no chromosome may
extend past the tree sequences' `sequence_length`. A supplied file overrides
embedded metadata and the builder prints a warning. Two columns is the first two
columns of a reference `.fai`; do not pass the raw `.fai`, whose third column is
a byte offset into the FASTA.

## Setup

```bash
conda env create -f environment.yml
conda activate normalizeTE
python -m pytest -q tests test_snp_age_distribution.py
```

The environment is created once. One multiprocessing audit requires Linux
`fork` and is skipped on macOS, so run the suite on a Linux compute node before
a production run. For a reproducible run, check out an immutable tag or commit
rather than a moving branch:

```bash
git fetch --tags
git checkout COMMIT_HASH
```

## Pipeline

Each step lists the flags used in its command. `--help` on any script gives the
full set, including options not shown here.

### 1. Build the all-SNP interval store

Each usable mutation gives a SNP a uniform age distribution between its mutation
node and that node's parent. This step collects those intervals for every SNP in
every posterior draw into one compact store. Run it once; TE lists do not filter
it, so the same store serves every analysis.

```bash
python build_snp_interval_store.py \
  project-data/posterior/*.tsz \
  --interval-store age_interval_store \
  --chrom-offsets project-data/chrom_offsets.txt \
  --min-usable-fraction 0.1 \
  --num-buckets 100 \
  --bucket-memory-gb 2 \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}"
```

| flag | meaning |
|---|---|
| `trees` (positional) | the posterior draw files, one ARG per draw |
| `--interval-store` | output directory for the store |
| `--chrom-offsets` | per-chromosome global coordinate offset; store positions are offset + VCF POS |
| `--min-usable-fraction` | a SNP is eligible only if this fraction of draws gave it a usable interval |
| `--num-buckets` | how many row ranges the records are split into on disk. Buckets are sorted one at a time, so more buckets means lower peak memory and more scratch files |
| `--bucket-memory-gb` | memory ceiling for sorting one bucket. The build aborts naming the offending bucket rather than being OOM-killed; raise `--num-buckets` in response, not this |
| `--scratch-dir` | where bucket files are written. Use node-local scratch |

Omit `--chrom-offsets` when every ARG carries compatible chromosome metadata. By
default a SNP is eligible when at least 10% of draws give it a usable interval,
and missing sites or mutations above roots are recorded and skipped; use
`--missing error` or `--root error` to stop instead.

Produces `age_interval_store/`, whose `metadata.json` records a `content_sha256`
over every array. Later stages check that digest and reject a mismatch, so keep
the store immutable once built.

### 2. Build the candidate control universe

The store is built from a combined SNP+TE dataset, so every TE variant is an
eligible store row and would otherwise be available as a control. This step
restricts the pool to the filtered SNP list and removes every TE variant.

```bash
python build_candidate_rows.py \
  --store age_interval_store \
  --include-positions project-data/snp/all_snp.pos.txt \
  --exclude-positions project-data/te/all_te.pos.txt \
  --output candidate_rows.npy \
  --min-resolved-fraction 0.70
```

| flag | meaning |
|---|---|
| `--store` | the store whose rows define the universe |
| `--include-positions` | restrict the pool to these positions before excluding |
| `--exclude-positions` | positions to remove; pass every TE variant here |
| `--output` | destination `.npy` for the candidate rows |
| `--min-resolved-fraction` | minimum share of listed positions that must resolve to store rows |

Construction stops when fewer than `--min-resolved-fraction` of the listed
positions resolve to store rows (default 0.95); lower it only with evidence that
the shortfall is genuine absence from the ARGs.

Produces `candidate_rows.npy`, a one-dimensional array of canonical store rows,
plus a JSON report beside it. Row indices are store-specific, so rebuild this
whenever the store is rebuilt.

### 3. Build a TE age target

Averages the posterior age CDFs of one TE category into a target CDF, and
bootstraps the TE variants with replacement to set the acceptance threshold: the
median bootstrap Wasserstein-1 distance is the largest mismatch a control set
may have.

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

| flag | meaning |
|---|---|
| `--store` | store to read posterior ages from |
| `--te-positions` | chromosome and 1-based VCF position, one TE per line |
| `--output` | destination directory for the target bundle |
| `--scratch-dir` | node-local parent for the temporary TE-by-age CDF matrix |
| `--bootstrap-replicates` | resamples of the TE set used to calibrate the threshold |
| `--acceptance-quantile` | quantile of the bootstrap distance distribution taken as the threshold |
| `--seed` | seed for the bootstrap resampling |

`--acceptance-quantile 0.50` is the default and the matching-quality
specification; use another quantile only for a labeled sensitivity analysis, and
never reuse one output path across tolerances. `--acceptance-distance` sets an
absolute Wasserstein limit in generations instead, without skipping the
bootstrap. Unresolved or ineligible positions stop construction; the optional
`--missing-position-policy drop` records the exclusions and continues.
`--scratch-dir` holds a temporary TE-by-age CDF matrix whose size grows with the
number of TEs and the width of the age grid.

Produces `targets/in_gene/`, holding the target CDF and age grid, the resolved
TE rows, the bootstrap distances, and the threshold.

When the polarity mask is used, this build is **preliminary**. It exists to
supply the site list the mask is built against, and the target the rest of the
pipeline consumes is the rebuild in step 4. Give it its own directory —
`targets/in_gene_prelim` below — and keep it.

### 4. Build the TE polarity mask and rebuild the target

A TE insertion is the derived state. A posterior draw that called the insertion
allele ancestral placed the mutation on a different branch of the ARG, so the
age it recorded for that site is the age of a different event. This step records
which draws polarized each TE in agreement with biology, and the target is then
rebuilt from each site's agreeing draws alone.

The order looks circular and is worth reading twice. The mask builder takes its
site list from a target's `te_row_indices.npy`, so a **preliminary** target must
be built first (step 3); the **mask** is built against that preliminary target;
then the **final** target is a fresh build of the same position file with the
mask applied. The two targets are different directories. The preliminary one is
not disposable: the mask's rows are positionally bound to it, and
`te_age_target.py` rejects a mask whose rows do not match the resolved TE rows
in the same order.

```bash
# step 3 already built targets/in_gene_prelim from this position file

python build_te_polarity_mask.py \
  --store age_interval_store \
  --target targets/in_gene_prelim \
  --output te_polarity_mask \
  --absence-allele A \
  project-data/posterior/*.tsz

python te_age_target.py \
  --store age_interval_store \
  --te-positions project-data/te/in_gene.pos.txt \
  --output targets/in_gene \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}" \
  --te-polarity-mask te_polarity_mask \
  --max-flipped-fraction 0.5 \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --seed 1002
```

`build_te_polarity_mask.py`:

| flag | meaning |
|---|---|
| `--store` | interval store the target rows index into; supplies each tree file's `draw_id` |
| `--target` | preliminary target whose `te_row_indices.npy` defines the sites |
| `trees` (positional) | every `.tsz` the store was built from; a draw not passed is uncovered |
| `--absence-allele` | allele encoding TE absence, which biology makes ancestral (default `A`) |
| `--output` | destination directory for the mask |

`te_age_target.py`, on the rebuild, in addition to the step 3 flags:

| flag | meaning |
|---|---|
| `--te-polarity-mask` | mask directory; each TE's age CDF is then built from its agreeing draws only |
| `--max-flipped-fraction` | discard any TE whose flipped fraction, among draws with data for it, exceeds this; requires `--te-polarity-mask` |

Pass every tree the store was built from. Mask columns are indexed by the
store's own `draw_id` rather than by argument order, and `te_age_target.py`
refuses a mask that does not cover all of the store's draws, because an
uncovered draw is indistinguishable there from a flipped one and would be
dropped from every site. The mask is bound to the store by content digest as
well.

`--max-flipped-fraction 0.5` is the recommended production value. It discards
only TEs whose ARG draws mostly contradict the biological polarization, and
every surviving TE keeps at least half of its draws. A site where no draw agrees
retains all of its draws rather than ending up with no age at all, and the count
is reported rather than hidden. Stricter thresholds shift the target younger,
because flipped fraction correlates with TE age; the threshold sweep and the
production measurements are in `BOOTSTRAP_HPC_VALIDATION.md`.

Produces `te_polarity_mask/`, holding the per-site-per-draw agreement and
presence arrays, the TE rows they are aligned to, and a per-draw report; and a
rebuilt `targets/in_gene/`, whose `metadata.json` records under `te_polarity`
the mask used, the `max_flipped_fraction` applied, how many draw-site ages were
dropped, and how many TEs were discarded and kept.

### 5. Match control sets to bootstrap TE targets

Gives each of 100 replicates its own bootstrap TE target and minimizes the
exact-grid Wasserstein-1 distance to it by improvement-only SNP swaps, so the
published sets span the TE sample's own age uncertainty instead of sitting
against one fixed boundary. `--disjoint-replicates` is the production setting:
each replicate is optimized against the candidate universe minus every row
already published, so no control SNP appears in two sets.

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

| flag | meaning |
|---|---|
| `--store` | store supplying candidate SNP ages |
| `--target` | target bundle from step 3, or its step 4 rebuild: age CDF, threshold and strata |
| `--candidate-rows` | control universe from step 2, TE variants removed |
| `--output` | destination directory for the published sets |
| `--work-dir` | where completed replicate bundles are staged for `--resume` |
| `--resume` | continue an interrupted run; parameters and inputs must match |
| `--replicates` | bootstrap replicates to match, one published set each |
| `--restarts` | independent stratified restarts per replicate; the best W1 is published |
| `--disjoint-replicates` | no control SNP may appear in two published sets |
| `--seed` | seed for resampling, initialization and proposals |

Keep `--work-dir` on durable storage: completed replicate bundles are what
`--resume` reuses after a preemption or time limit, and node-local scratch does
not survive the job. After an interruption, repeat the identical command; a
provenance or parameter difference is rejected rather than mixed in. Use
`--all-eligible` in place of `--candidate-rows` only when the intended control
universe is every eligible non-target SNP.

Produces `bootstrap_matches/in_gene/`, holding the selected rows and CDFs per
replicate, the per-replicate distances and QC arrays, the full restart traces,
`replicates.csv`, `restarts.csv`, and SNP-reuse arrays.

### 6. Build the ancestral-state table

Allele counts come from the VCF, but which allele is derived does not: this
dataset's VCF is unpolarized, so polarity is read from the ARGs. This step
accumulates, for every store row, how many draws called each of A/C/G/T
ancestral. Build it once per store, from the same draws.

```bash
python build_ancestral_states.py \
  --store age_interval_store \
  --output ancestral_states \
  project-data/posterior/*.tsz
```

| flag | meaning |
|---|---|
| `--store` | store whose rows the table is aligned to |
| `--output` | destination directory for the table |
| `trees` (positional) | the posterior draws to accumulate |

For a job array, give each task a slice with `--draws START:STOP` and its own
`--output`, then gather with `--merge part_000 part_001 ...`, no tree arguments,
and `--expect-draws N` for the number of draws in the full store. A merge
validates that the parts share a non-null store identity, contribute a disjoint
draw set, and total exactly `N` draws.

Produces `ancestral_states/`, a per-row ancestral base count plus a present-draw
count.

### 7. Calculate Φ-SFS

Compares the unfolded, projected SFS of the target TE set with that of every
published control set. Φ-SFS is the total variation distance between the two
normalized spectra, so it runs from 0 (identical) to 1 (no shared mass). TE
sites are polarized biologically; control SNPs take a posterior-weighted
polarity mixture from the table built in step 6, which is why
`--ancestral-table` is required.

```bash
python phi_sfs.py \
  --target targets/in_gene \
  --matches bootstrap_matches/in_gene \
  --vcf variants.vcf.gz \
  --ancestral-table ancestral_states \
  --output phi_sfs/in_gene
```

| flag | meaning |
|---|---|
| `--target` | TE target directory from step 3, or its step 4 rebuild |
| `--matches` | matched control sets from step 5 |
| `--vcf` | biallelic genotype VCF covering every requested site |
| `--ancestral-table` | polarity table from step 6, used for control SNPs only |
| `--output` | destination directory for spectra and scores |

Plain, `.gz`, and `.bgz`/`.bgzf` VCFs are accepted; `--quiet` suppresses scan
progress, and `--heterozygous missing` excludes heterozygous individuals from a
site's callable count instead of failing. The VCF must cover every requested
site genome-wide — the TE target plus every control SNP in every set — or the
run aborts listing the missing coordinates. `phi_sfs.py` recomputes the target's
`target_digest` and requires the matched bundle to record the same value, so a
bundle built for another TE category is rejected rather than silently compared.

The stage assumes the following about its input rather than deriving any of it,
so check them before a production run. Each is also recorded in the output
`metadata.json`.

- **Records are biallelic and TE ALT encodes presence.** A comma in ALT is an
  error, not a multiallelic record to split. At a TE site the insertion is the
  derived state and the genotyping convention encodes presence as ALT, so
  P(ALT is derived) is exactly 1 there; control SNPs take the posterior-weighted
  mixture from `--ancestral-table` instead. REF is never assumed ancestral and
  no INFO field is consulted.
- **FILTER is ignored.** The declared input is the already-filtered
  preprocessing VCF, so every record at a requested coordinate is used whatever
  its FILTER value says.
- **At least 20 callable individuals per site.** A site with fewer than 20 is
  ineligible and dropped, counted as `dropped_n_lt_20`; every eligible site is
  projected to exactly 20 by hypergeometric expectation. Bins 1 through 19 are
  kept and are deliberately not renormalized per site, so the excluded
  endpoint mass `h_0 + h_20` reduces that site's contribution.
- **One allele per individual.** These are inbred lines: each callable
  individual contributes a single observed allele. Haploid and homozygous
  diploid calls are accepted, and any missing allele makes the whole individual
  uncallable at that site. A heterozygous call is an error by default;
  `--heterozygous missing` treats those individuals as missing instead, which
  lowers the site's callable count and can push it below 20.
- **Every requested site must be present.** The TE target plus every control
  SNP in every set, genome-wide; a missing coordinate aborts the run rather
  than analyzing a subset.
- **Normalization discards absolute scale.** Two sets with very different
  eligible-site counts, missingness, or endpoint mass can give identical
  spectra. Read `retained_fraction` and `endpoint_fraction` in `replicates.csv`,
  against the `target_retained_fraction` and `target_endpoint_fraction` in
  `metadata.json`, before comparing sets: a target and a control set whose
  retained fractions differ substantially are not comparable however small Φ
  is.

Produces `phi_sfs/in_gene/`, holding raw and normalized spectra, TE-minus-SNP
residuals, the Φ-SFS scores, `replicates.csv`, `bins.csv`, and metadata.

## Running on Farm/Quobyte

Four launchers cover the stages that need a scheduler. Every parameter is an
environment variable, so one file serves every category.

| stage | launcher | notes |
|---|---|---|
| steps 3, 4 and 5 | `run_bootstrap_matching.sbatch` | builds the target if it is absent — with the polarity mask when `TE_POLARITY_MASK` is set — then matches |
| step 4 | `run_te_polarity_mask.sbatch` | takes its tree list from the store's own `metadata["inputs"]`, so every draw is covered |
| step 6 | `run_ancestral_table.sbatch` | one job over all draws, or an array plus a gather with `MERGE=1` |
| step 7 | `run_phi_sfs.sbatch` | needs a VCF covering every requested site, genome-wide |

```bash
# step 3: the preliminary target the mask is built against. No launcher
# builds a target on its own, so submit te_age_target.py as an ordinary job.

# step 4: the polarity mask, against that preliminary target
sbatch --export=ALL,STORE=/path/interval_store,\
TARGET=results/targets/in_gene_prelim,OUTPUT=results/te_polarity_mask \
  run_te_polarity_mask.sbatch

# steps 4-5: rebuild the target with the mask, then match controls
sbatch --export=ALL,STORE=/path/interval_store,TARGET=results/targets/in_gene,\
TE_POSITIONS=/path/te/in_gene.pos.txt,OUTPUT=results/bootstrap_matches/in_gene,\
CANDIDATE_ROWS=results/candidate-rows.npy,WORK_DIR=results/work/in_gene,\
TE_POLARITY_MASK=results/te_polarity_mask,MAX_FLIPPED_FRACTION=0.5,\
REPLICATES=100,SEED=1002 \
  run_bootstrap_matching.sbatch

# step 6: ancestral polarity, as 15 array tasks of 5 draws, then gathered
sbatch --array=0-14 --export=ALL,STORE=/path/interval_store,\
TREES="/path/run.combined.*.tsz",OUTPUT=results/ancestral-parts,PER_TASK=5 \
  run_ancestral_table.sbatch
sbatch --export=ALL,STORE=/path/interval_store,MERGE=1,\
PARTS="results/ancestral-parts/part-*",OUTPUT=results/ancestral-75draw,EXPECT_DRAWS=75 \
  run_ancestral_table.sbatch

# step 7: Phi-SFS
sbatch --export=ALL,TARGET=results/targets/in_gene,\
MATCHES=results/bootstrap_matches/in_gene,VCF=/path/all.chr.vcf.gz,\
ANCESTRAL=results/ancestral-75draw,OUTPUT=results/phi_sfs/in_gene \
  run_phi_sfs.sbatch
```

Three properties of `run_bootstrap_matching.sbatch` are not optional. It is
submitted with `sbatch`, never `srun`, because an `srun` started from an
interactive shell dies with that shell. It rsyncs the interval store to
node-local `$TMPDIR` first, after checking that the scratch filesystem holds the
store size plus 20%, because the job is I/O-bound against Quobyte and CPU-bound
against staged scratch. And `WORK_DIR` is on durable storage with `--resume`
always passed, so a preemption costs the replicate in flight and nothing else.
Set `CANDIDATE_ROWS=all` to use every eligible non-target row, omit
`TE_POSITIONS` when `TARGET` already exists, and set `REPLICATES`, `RESTARTS`,
`ACCEPTANCE_QUANTILE`, or `MISSING_POSITION_POLICY` to depart from the defaults.
`TE_POLARITY_MASK` and `MAX_FLIPPED_FRACTION` apply to target construction, so
`TARGET` must name a new directory: pointing them at an existing unmasked target
is rejected rather than matched against silently, and a `TARGET` that already
exists is checked to have been built with the mask and threshold being
requested. `run_te_polarity_mask.sbatch` needs only `STORE`, `TARGET` and
`OUTPUT`, with `ABSENCE_ALLELE` defaulting to `A`.

Measured runtimes and memory are in `BOOTSTRAP_HPC_VALIDATION.md`. Two figures
to read carefully. The 4,067-site, 100-replicate matching run measured there —
2 h 12 m wall clock, 37.7 GiB peak RSS — was measured **without** a polarity
mask; masked target construction filters intervals per site before building the
per-site CDFs and has not been measured separately, so treat the published
numbers as a lower bound for a masked run rather than an estimate of one. The
mask build itself was measured: 4,067 sites by 75 draws took 1 h 21 m at the
launcher's 2 CPUs and 96 GB, and the cost is dominated by decompressing each
`.tsz` once rather than by the number of sites.

Interval-store builds have no launcher in this repository; submit
`build_snp_interval_store.py` as an ordinary single-CPU `sbatch` job with
`--scratch-dir "$TMPDIR"`. The preliminary step 3 target has no launcher of its
own either — `run_bootstrap_matching.sbatch` builds a target only as a prelude
to matching — so submit `te_age_target.py` the same way. On Farm use `$TMPDIR`,
not `$SLURM_TMPDIR`.

## Outputs

| artifact | from | contents |
|---|---|---|
| `age_interval_store/` | step 1 | per-SNP age intervals across all draws, with `content_sha256` |
| `candidate_rows.npy` | step 2 | the TE-excluded control universe as canonical store rows |
| `targets/CATEGORY/` | steps 3, 4 | target CDF, age grid, resolved TE rows, bootstrap distances, threshold; the masked rebuild also records `te_polarity` |
| `te_polarity_mask/` | step 4 | `agrees_with_biology.npy` and `draw_present.npy`, both `(n_te_sites, n_draws)` and indexed by the store's `draw_id`; the `te_row_indices.npy` they align to; and metadata with the per-draw report, the covered draw ids, and the store digest |
| `bootstrap_matches/CATEGORY/` | step 5 | 100 control sets with their CDFs, distances, QC arrays, restart traces, `replicates.csv` |
| `ancestral_states/` | step 6 | per-row ancestral base counts and present-draw counts |
| `phi_sfs/CATEGORY/` | step 7 | spectra, residuals, Φ-SFS scores, `replicates.csv`, `bins.csv` |

Every output directory is published atomically and must not already exist.
Replicates are identified by `replicate_id` throughout; they have no chain
structure. Each directory's `metadata.json` records the release version, Git
commit and description, whether tracked files were dirty, all seeds, the full
configuration, and the identities of its inputs. For production, require the
expected version and commit and reject runs with `git_dirty: true`. An exported
tree without `.git` keeps the release version but records the Git fields as
null.

The 100 control sets are matched-control replicates, not 100 independent
biological replicates. `BOOTSTRAP_HPC_VALIDATION.md` states precisely what
`--disjoint-replicates` does and does not buy, and what the spread of Φ across
replicates means.

## Further reading

- [BOOTSTRAP_HPC_VALIDATION.md](BOOTSTRAP_HPC_VALIDATION.md) — the validation
  record for the production route: measurements, resource figures, gate status,
  and the prespecified polarity decisions.
- [BOOTSTRAP_TARGET_MATCHING_PLAN.md](BOOTSTRAP_TARGET_MATCHING_PLAN.md) — the
  statistical design behind step 5.
- [PHI_SFS_IMPLEMENTATION_PLAN.md](PHI_SFS_IMPLEMENTATION_PLAN.md) — the
  definition of the projection and of Φ-SFS itself.
- [BOOTSTRAP_DISCARDED_APPROACHES.md](BOOTSTRAP_DISCARDED_APPROACHES.md) — what
  was tried and rejected, with the evidence, including the abandoned hard-q50
  swap sampler and the legacy dense-store and rejection-sampler workflows.
- [CHANGELOG.md](CHANGELOG.md) — release-level behavior changes.
- `snp_age_distribution.py` estimates age distributions for a handful of SNPs
  directly from tree sequences, without building a store; it takes native
  numeric coordinates rather than chromosome-position pairs. Run it with
  `--help`.
- `INTERVAL_STORE_*`, `GLOBAL_QUANTILE_*`, `AGE_MATCHED_CONTROL_SAMPLER_PLAN.md`,
  `SWAP_SAMPLER_HPC_HOWTO.md`, and `CODE_REVIEW*` are design history and review
  records, not operator instructions.
