# normalizeTE v0.5.2

normalizeTE builds SNP control sets matched to the posterior ages of a TE category,
then compares the unfolded site-frequency spectra of the TEs and controls. The
production workflow uses posterior ARG draws, SNP and TE position lists, and a
genome-wide biallelic VCF.

This README is the operator guide. Method definitions and the evidence behind the
production settings are linked under [Methods and validation](#methods-and-validation).

## Repository layout

- `normalize_tes/` contains the production package and supported command modules.
- `tools/` contains development benchmarks, diagnostics, probes, and simulations.
- `slurm/` contains scheduler launchers and their shared conda bootstrap helper.
- `tests/` contains the complete test suite.
- `docs/` contains methods, validation records, plans, reviews, and the changelog.

Run production commands from the repository root with
`python -m normalize_tes.COMMAND`; examples below use that form throughout.

## Before you start

Create the environment and run the tests on a Linux compute node:

```bash
conda env create -f environment.yml
conda activate normalizeTE
python -m pytest -q tests
```

Use an immutable release tag or commit for production:

```bash
git fetch --tags
git checkout COMMIT_HASH
```

The workflow expects:

- posterior ARG draws; the ancestry and TE-polarity stages require tszip archives;
- SNP and TE position files with two whitespace-separated columns: chromosome and
  1-based VCF position;
- chromosome labels matching the ARG metadata or a compatible chromosome-offset
  file supplied when the store is built;
- a filtered, genome-wide, biallelic VCF in which TE ALT encodes insertion/presence;
- node-local `$TMPDIR` for interval-store and target scratch data;
- durable storage for published outputs and matcher resume state.

Blank lines and `#` comments are allowed in position files. Run each script with
`--help` for its complete input contract and advanced options.

## Configure one category

Set these paths once and use them throughout the commands below:

```bash
POSTERIOR_DIR=/path/project-data/posterior
CHROM_OFFSETS=/path/project-data/chrom_offsets.txt
SNP_POSITIONS=/path/project-data/snp/all_snp.pos.txt
ALL_TE_POSITIONS=/path/project-data/te/all_te.pos.txt
TE_POSITIONS=/path/project-data/te/in_gene.pos.txt
VCF=/path/variants.vcf.gz

STORE=results/age_interval_store
CANDIDATES=results/candidate_rows.npy
PRELIM_TARGET=results/targets/in_gene_prelim
POLARITY_MASK=results/te_polarity_masks/in_gene
TARGET=results/targets/in_gene
MATCHES=results/bootstrap_matches/in_gene
WORK_DIR=results/work/in_gene
ANCESTRAL=results/ancestral_states
PHI=results/phi_sfs/in_gene

mkdir -p results results/targets results/te_polarity_masks \
  results/bootstrap_matches results/work results/phi_sfs
```

Every published output path must be new. The tools refuse to overwrite existing
artifacts.

## Run the pipeline

The commands below show the production path for one TE category. Run heavy commands
inside a scheduled compute allocation, not on a login/head node.

### 1. Build the interval store

Build one reusable posterior-age store for all categories:

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

| flag | purpose |
|---|---|
| `trees` | posterior ARG draws, one tree sequence per draw |
| `--interval-store` | new store directory to publish |
| `--chrom-offsets` | chromosome offsets when compatible metadata is not embedded in every ARG |
| `--min-usable-fraction` | minimum fraction of draws with a usable age interval for an eligible row |
| `--num-buckets` | number of temporary row partitions; increase this to reduce per-bucket memory |
| `--bucket-memory-gb` | per-bucket sort-memory ceiling |
| `--scratch-dir` | temporary bucket location; use node-local scratch |

Omit `--chrom-offsets` only when every draw contains compatible chromosome metadata.
The completed store records a content digest used by downstream identity checks.

### 2. Build the candidate control universe

Restrict controls to the filtered SNP list and exclude all known TE positions:

```bash
python -m normalize_tes.build_candidate_rows \
  --store "$STORE" \
  --include-positions "$SNP_POSITIONS" \
  --exclude-positions "$ALL_TE_POSITIONS" \
  --output "$CANDIDATES" \
  --min-resolved-fraction 0.70
```

| flag | purpose |
|---|---|
| `--store` | store whose canonical rows are being selected |
| `--include-positions` | filtered SNP positions allowed in the control universe |
| `--exclude-positions` | all TE positions to remove from that universe |
| `--output` | new candidate-row `.npy`; a provenance report is written beside it |
| `--min-resolved-fraction` | minimum fraction of requested positions that must resolve to store rows |

Candidate rows are store-specific. Rebuild this artifact whenever the store changes.
The justification for the production resolution threshold belongs in the validation
record, not in this how-to.

### 3. Build the preliminary TE target

The preliminary target supplies the ordered TE rows needed to build the polarity
mask. It is not the target used for matching.

```bash
python -m normalize_tes.te_age_target \
  --store "$STORE" \
  --te-positions "$TE_POSITIONS" \
  --output "$PRELIM_TARGET" \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}" \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --seed 1002
```

| flag | purpose |
|---|---|
| `--store` | interval store supplying TE ages |
| `--te-positions` | TE category to resolve and summarize |
| `--output` | new preliminary target directory |
| `--scratch-dir` | node-local location for the temporary TE-by-age CDF matrix |
| `--bootstrap-replicates` | TE resamples used to calibrate the matching threshold |
| `--acceptance-quantile` | bootstrap-distance quantile used as that threshold |
| `--seed` | bootstrap random seed |

Keep this directory: the polarity mask records the target it was built against.

Sizing note: an unmasked target streams its TE-by-age CDF through `--scratch-dir`,
so scratch is the constraint. A masked target (step 5) does not — it builds the
whole CDF block in memory — so there `--mem` is the constraint, and the run prints
its projected peak before building. Measured resource figures are in
[BOOTSTRAP_HPC_VALIDATION.md](docs/BOOTSTRAP_HPC_VALIDATION.md).

### 4. Build the TE polarity mask

Record which posterior draws polarize each TE in agreement with TE presence being
derived:

```bash
python -m normalize_tes.build_te_polarity_mask \
  --store "$STORE" \
  --target "$PRELIM_TARGET" \
  --output "$POLARITY_MASK" \
  --absence-allele A \
  "$POSTERIOR_DIR"/*.tsz
```

| flag | purpose |
|---|---|
| `--store` | store that defines row and draw IDs |
| `--target` | preliminary target supplying the ordered TE rows |
| `--output` | new category-specific mask directory |
| `--absence-allele` | allele encoding TE absence; default `A` |
| `trees` | every source draw recorded by the store |

Pass the complete draw set. Partial masks are rejected by target construction.

### 5. Build the final target and match controls

Build a new target from agreeing draws, discard TEs above the production flipped-draw
threshold, and construct the matched control sets:

```bash
python -m normalize_tes.te_age_target \
  --store "$STORE" \
  --te-positions "$TE_POSITIONS" \
  --output "$TARGET" \
  --scratch-dir "${TMPDIR:?TMPDIR is not set}" \
  --te-polarity-mask "$POLARITY_MASK" \
  --max-flipped-fraction 0.5 \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --seed 1002

python -m normalize_tes.bootstrap_target_matcher \
  --store "$STORE" \
  --target "$TARGET" \
  --candidate-rows "$CANDIDATES" \
  --output "$MATCHES" \
  --work-dir "$WORK_DIR" \
  --resume \
  --replicates 100 \
  --restarts 3 \
  --disjoint-replicates \
  --seed 1002
```

Final-target additions:

| flag | purpose |
|---|---|
| `--te-polarity-mask` | use only agreeing posterior draws for each TE age CDF |
| `--max-flipped-fraction` | discard a TE when its flipped fraction exceeds this value |

Matcher flags:

| flag | purpose |
|---|---|
| `--store` | store supplying candidate SNP ages |
| `--target` | final masked target |
| `--candidate-rows` | TE-excluded control universe and its provenance sidecar |
| `--output` | new matched-control bundle |
| `--work-dir` | durable per-replicate state used by `--resume` |
| `--resume` | continue an interrupted compatible run |
| `--replicates` | number of matched control sets |
| `--restarts` | optimization restarts per control set |
| `--disjoint-replicates` | prevent reuse of a control SNP between published sets |
| `--seed` | matching random seed |

The preliminary and final targets must use different directories. Keep `WORK_DIR` on
durable storage and repeat the identical command after preemption.

### 6. Build the ancestral-state table

Build one store-aligned ancestral-state table for control-SNP polarization:

```bash
python -m normalize_tes.build_ancestral_states \
  --store "$STORE" \
  --output "$ANCESTRAL" \
  "$POSTERIOR_DIR"/*.tsz
```

| flag | purpose |
|---|---|
| `--store` | store whose rows and source draws the table must match |
| `--output` | new ancestral-state table directory |
| `trees` | the store's complete posterior draw set |

For an array build, use `--draws START:STOP` for each part and merge the parts with
`--merge ... --expect-draws N`. The production launcher example below shows this
pattern.

### 7. Calculate Phi-SFS

Calculate the unfolded SFS comparison for the TE target and every matched set:

```bash
python -m normalize_tes.phi_sfs \
  --target "$TARGET" \
  --matches "$MATCHES" \
  --vcf "$VCF" \
  --ancestral-table "$ANCESTRAL" \
  --output "$PHI"
```

| flag | purpose |
|---|---|
| `--target` | final TE target |
| `--matches` | matched-control bundle from step 5 |
| `--vcf` | filtered genome-wide biallelic VCF covering all requested sites |
| `--ancestral-table` | store-aligned posterior ancestral-state table |
| `--output` | new Phi-SFS result directory |

The default rejects heterozygous calls. Use `--heterozygous missing` only when the
analysis should treat heterozygous individuals as uncallable. Sites with fewer than
20 callable individuals are excluded from the spectrum.

## Farm/Quobyte launchers

Submit launchers with `sbatch` from the repository checkout. The launchers activate
the conda environment themselves. `$TMPDIR` is node-local scratch; matcher
`WORK_DIR` must remain on Quobyte or other durable storage.

Build the polarity mask after the preliminary target exists:

```bash
sbatch --export=ALL,STORE="$STORE",TARGET="$PRELIM_TARGET",OUTPUT="$POLARITY_MASK" \
  slurm/run_te_polarity_mask.sbatch
```

Build the final masked target and match controls:

```bash
sbatch --export=ALL,STORE="$STORE",TARGET="$TARGET",TE_POSITIONS="$TE_POSITIONS",\
OUTPUT="$MATCHES",CANDIDATE_ROWS="$CANDIDATES",WORK_DIR="$WORK_DIR",\
TE_POLARITY_MASK="$POLARITY_MASK",MAX_FLIPPED_FRACTION=0.5,\
REPLICATES=100,RESTARTS=3,SEED=1002,SCRATCH_HEADROOM_GB=32 \
  slurm/run_bootstrap_matching.sbatch
```

The launcher builds `TARGET` only when it is absent. If it already exists, the
launcher verifies that its recorded mask and flipped-fraction threshold match the
request before matching.

Build the ancestral table as an array and merge it after every array task succeeds:

```bash
sbatch --array=0-14 --export=ALL,STORE="$STORE",\
TREES="$POSTERIOR_DIR/*.tsz",OUTPUT=results/ancestral-parts,PER_TASK=5 \
  slurm/run_ancestral_table.sbatch

sbatch --export=ALL,STORE="$STORE",MERGE=1,\
PARTS="results/ancestral-parts/part-*",OUTPUT="$ANCESTRAL",EXPECT_DRAWS=75 \
  slurm/run_ancestral_table.sbatch
```

Calculate Phi-SFS:

```bash
sbatch --export=ALL,TARGET="$TARGET",MATCHES="$MATCHES",VCF="$VCF",\
ANCESTRAL="$ANCESTRAL",OUTPUT="$PHI" \
  slurm/run_phi_sfs.sbatch
```

Scheduler allocations, measured resource use, scratch sizing, and parameter evidence
are recorded in [BOOTSTRAP_HPC_VALIDATION.md](docs/BOOTSTRAP_HPC_VALIDATION.md).

### Submit many TE categories

The store, candidate universe, and ancestral table are shared across categories.
Give every category its own preliminary target, polarity mask, final target, matched
bundle, durable work directory, and seed. First create a tab-separated manifest:

```text
label	positions	prelim_target	polarity_mask	target	matches	work_dir	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te_prelim	/quobyte/project/polarity_masks/all_te	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	/quobyte/project/work/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene_prelim	/quobyte/project/polarity_masks/in_gene	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	/quobyte/project/work/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young_prelim	/quobyte/project/polarity_masks/young	/quobyte/project/targets/young	/quobyte/project/matches/young	/quobyte/project/work/young	1003
```

The manifest rules are:

- the first line is the header shown above;
- fields are separated by literal tabs and paths must not contain tabs, newlines, or
  commas;
- every `prelim_target` must already have been built from that row's `positions`;
- every other category-specific path must be unique; mask, target, and match paths
  must not exist on a first submission;
- `work_dir` is durable and may be reused only to resume the identical matching run;
- seeds should be fixed before submission and remain unchanged on resubmission.

Set the shared inputs, then submit one mask job and one dependent target/matching job
per manifest row:

```bash
PROJECT=/quobyte/project/normalizeTE
STORE=/quobyte/project/data/age_interval_store
CANDIDATES=/quobyte/project/data/candidate_rows.npy
MANIFEST=/quobyte/project/manifests/te_categories.tsv

while IFS=$'\t' read -r label positions prelim mask target matches work seed; do
  [[ "$label" == label ]] && continue
  [[ -n "$label" ]] || continue

  mask_job=$(sbatch --parsable \
    --job-name="mask-${label}" \
    --export=ALL,PROJECT="$PROJECT",STORE="$STORE",TARGET="$prelim",OUTPUT="$mask" \
    slurm/run_te_polarity_mask.sbatch)
  mask_job=${mask_job%%;*}

  match_job=$(sbatch --parsable \
    --job-name="match-${label}" \
    --dependency="afterok:${mask_job}" \
    --export=ALL,PROJECT="$PROJECT",STORE="$STORE",TARGET="$target",\
TE_POSITIONS="$positions",OUTPUT="$matches",CANDIDATE_ROWS="$CANDIDATES",\
WORK_DIR="$work",TE_POLARITY_MASK="$mask",MAX_FLIPPED_FRACTION=0.5,\
REPLICATES=100,RESTARTS=3,SEED="$seed",SCRATCH_HEADROOM_GB=32 \
    slurm/run_bootstrap_matching.sbatch)
  match_job=${match_job%%;*}

  printf '%s\tmask=%s\tmatch=%s\n' "$label" "$mask_job" "$match_job"
done < "$MANIFEST"
```

The categories run concurrently, while each matching job waits for its own mask. Save
the printed job IDs. Check the mask jobs before trusting the dependent runs, and use
`squeue`, `sacct`, and the scheduler logs to confirm that every manifest row completed.

This loop intentionally does not rebuild preliminary targets: no current production
launcher performs a target-only run. Build those targets first using step 3 in
scheduled compute allocations. It also does not silently skip existing masks or
outputs; for a partial rerun, submit only the missing categories or resubmit an
interrupted matcher with its original target, output, work directory, and seed.

## Verify a production run

Before accepting the results:

1. Confirm every artifact records the expected release version, Git commit, and
   non-null input identities in `metadata.json`.
2. Confirm the candidate-row report meets the requested resolution threshold and is
   bound to `STORE`.
3. Confirm the final target records `POLARITY_MASK`, the intended
   `max_flipped_fraction`, and plausible kept/discarded counts.
4. Confirm all matcher QC checks pass and inspect `replicates.csv`, `restarts.csv`,
   and SNP-reuse diagnostics.
5. Confirm Phi-SFS used the intended heterozygote policy and inspect target/control
   retained and endpoint fractions before interpreting Phi.

The exact acceptance criteria and the tests supporting them are in
[BOOTSTRAP_HPC_VALIDATION.md](docs/BOOTSTRAP_HPC_VALIDATION.md).

## Outputs

| artifact | purpose |
|---|---|
| `age_interval_store/` | reusable posterior age intervals and store identity |
| `candidate_rows.npy` plus `.json` | TE-excluded, store-bound control universe |
| `targets/CATEGORY_prelim/` | ordered TE rows used to construct the polarity mask |
| `te_polarity_masks/CATEGORY/` | per-TE, per-draw polarity agreement mask |
| `targets/CATEGORY/` | final masked TE age target and acceptance threshold |
| `bootstrap_matches/CATEGORY/` | matched control sets, restart traces, and QC |
| `ancestral_states/` | posterior ancestral-base counts for store rows |
| `phi_sfs/CATEGORY/` | spectra, residuals, Phi-SFS scores, and QC summaries |

Outputs are published atomically and are never overwritten. Matched-control sets are
not independent biological replicates; see the validation report for the correct
interpretation of their spread.

## Methods and validation

- [BOOTSTRAP_HPC_VALIDATION.md](docs/BOOTSTRAP_HPC_VALIDATION.md) — production settings,
  validation tests, measured resources, decision evidence, and acceptance criteria.
- [BOOTSTRAP_TARGET_MATCHING_PLAN.md](docs/BOOTSTRAP_TARGET_MATCHING_PLAN.md) — bootstrap
  target and matching design.
- [PHI_SFS_IMPLEMENTATION_PLAN.md](docs/PHI_SFS_IMPLEMENTATION_PLAN.md) — SFS projection,
  polarization mixture, and Phi-SFS definition.
- [BOOTSTRAP_DISCARDED_APPROACHES.md](docs/BOOTSTRAP_DISCARDED_APPROACHES.md) — evaluated
  approaches that are not part of the production route.
- [CHANGELOG.md](docs/CHANGELOG.md) — release-level behavior changes.
- [CODE_REVIEW_ROUND9.md](docs/CODE_REVIEW_ROUND9.md) — latest implementation review.

Historical `INTERVAL_STORE_*`, `GLOBAL_QUANTILE_*`, sampler plans, and older code
reviews document development history; they are not operator instructions.
