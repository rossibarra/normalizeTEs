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

Both tszip-compressed `.tsz` files and ordinary tskit tree-sequence files are
accepted.

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

### 4. Match control sets to bootstrap TE targets

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
| `--target` | target bundle from step 3: age CDF, threshold and strata |
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

### 5. Build the ancestral-state table

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

### 6. Calculate Φ-SFS

Compares the unfolded, projected SFS of the target TE set with that of every
published control set. Φ-SFS is the total variation distance between the two
normalized spectra, so it runs from 0 (identical) to 1 (no shared mass). TE
sites are polarized biologically; control SNPs take a posterior-weighted
polarity mixture from the table built in step 5, which is why
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
| `--target` | TE target directory from step 3 |
| `--matches` | matched control sets from step 4 |
| `--vcf` | biallelic genotype VCF covering every requested site |
| `--ancestral-table` | polarity table from step 5, used for control SNPs only |
| `--output` | destination directory for spectra and scores |

Plain, `.gz`, and `.bgz`/`.bgzf` VCFs are accepted; `--quiet` suppresses scan
progress, and `--heterozygous missing` excludes heterozygous individuals from a
site's callable count instead of failing. The VCF must cover every requested
site genome-wide — the TE target plus every control SNP in every set — or the
run aborts listing the missing coordinates. `phi_sfs.py` recomputes the target's
`target_digest` and requires the matched bundle to record the same value, so a
bundle built for another TE category is rejected rather than silently compared.
The site assumptions this stage makes are recorded in its output
`metadata.json` and set out in `BOOTSTRAP_HPC_VALIDATION.md`.

Produces `phi_sfs/in_gene/`, holding raw and normalized spectra, TE-minus-SNP
residuals, the Φ-SFS scores, `replicates.csv`, `bins.csv`, and metadata.

## Running on Farm/Quobyte

Three launchers cover the stages that need a scheduler. Every parameter is an
environment variable, so one file serves every category.

| stage | launcher | notes |
|---|---|---|
| steps 3 and 4 | `run_bootstrap_matching.sbatch` | builds the target if it is absent, then matches |
| step 5 | `run_ancestral_table.sbatch` | one job over all draws, or an array plus a gather with `MERGE=1` |
| step 6 | `run_phi_sfs.sbatch` | needs a VCF covering every requested site, genome-wide |

```bash
# steps 3-4: build the target if needed, then match controls
sbatch --export=ALL,STORE=/path/interval_store,TARGET=results/targets/in_gene,\
TE_POSITIONS=/path/te/in_gene.pos.txt,OUTPUT=results/bootstrap_matches/in_gene,\
CANDIDATE_ROWS=results/candidate-rows.npy,WORK_DIR=results/work/in_gene,\
REPLICATES=100,SEED=1002 \
  run_bootstrap_matching.sbatch

# step 5: ancestral polarity, as 15 array tasks of 5 draws, then gathered
sbatch --array=0-14 --export=ALL,STORE=/path/interval_store,\
TREES="/path/run.combined.*.tsz",OUTPUT=results/ancestral-parts,PER_TASK=5 \
  run_ancestral_table.sbatch
sbatch --export=ALL,STORE=/path/interval_store,MERGE=1,\
PARTS="results/ancestral-parts/part-*",OUTPUT=results/ancestral-75draw,EXPECT_DRAWS=75 \
  run_ancestral_table.sbatch

# step 6: Phi-SFS
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
Measured runtimes and memory are in `BOOTSTRAP_HPC_VALIDATION.md`.

Interval-store builds have no launcher in this repository; submit
`build_snp_interval_store.py` as an ordinary single-CPU `sbatch` job with
`--scratch-dir "$TMPDIR"`. On Farm use `$TMPDIR`, not `$SLURM_TMPDIR`.

## Outputs

| artifact | from | contents |
|---|---|---|
| `age_interval_store/` | step 1 | per-SNP age intervals across all draws, with `content_sha256` |
| `candidate_rows.npy` | step 2 | the TE-excluded control universe as canonical store rows |
| `targets/CATEGORY/` | step 3 | target CDF, age grid, resolved TE rows, bootstrap distances, threshold |
| `bootstrap_matches/CATEGORY/` | step 4 | 100 control sets with their CDFs, distances, QC arrays, restart traces, `replicates.csv` |
| `ancestral_states/` | step 5 | per-row ancestral base counts and present-draw counts |
| `phi_sfs/CATEGORY/` | step 6 | spectra, residuals, Φ-SFS scores, `replicates.csv`, `bins.csv` |

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
  statistical design behind step 4.
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
