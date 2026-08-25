# Discarded approaches and superseded findings

Companion to `BOOTSTRAP_HPC_VALIDATION.md`, which carries the current strategy.
This file holds what was tried and rejected, and the readings that later
measurements overturned. It exists so the main report stays short, and so a
rejected idea is not re-proposed without its evidence.

Each entry records what was tried, what it measured, and why it is not in the
recommended route.

Several options named below (`--closest-restarts`, `--diverse-restarts`,
`--selection-tolerance`, `--search-grid-spacing`, `--seed-sets`, `--distance`,
`--log-age-offset`) have since been removed from the CLI. They are named here as
the record of what was measured, not as commands to run.

---

## Rejected interventions

### More search budget

`--patience 25 --max-epochs 300` against the shipped `--patience 5
--max-epochs 50`, same 20 replicates and bootstrap targets.

| | 15 epochs | patience 25 | change |
|---|---:|---:|---:|
| epochs used | 15 (fixed) | median 56, max 111 | 3.7x |
| `E_r` median | 320.74 | 297.35 | −7.3% |
| **`O_r` median** | **2054.88** | **2051.50** | **−0.2%** |
| QC pass | 19/20 | 19/20 | none |

Nearly four times the search effort bought 7% on the optimizer's internal error
and 0.2% on the quantity the science depends on, because `|O_r − B_r|` runs at
roughly 41% of `E_r` rather than tracking it. **Rejected: the shipped budget is
correct.**

### More restarts

`--closest-restarts 4 --diverse-restarts 2` against 2+1.

| | 3 restarts | 6 restarts | change |
|---|---:|---:|---:|
| `E_r` median | 288.15 | 276.16 | −4.2% |
| `O_r` median | 1800.57 | 1819.67 | **+1.1%** |
| unique controls | 195,836 | 187,585 | **−4.2%** |
| controls used >=10x | 4,967 | 5,491 | +10.6% |

2.5x the compute moved the scientific distance the wrong way and reduced
diversity, because minimum-W1 selection over more restarts converges harder onto
the same well-determined SNPs. **Rejected: keep 3 restarts.**

### Annealing or multi-swap moves

`probe_pair_swaps.py` confirmed the published states are not single-swap local
optima -- a systematic search finds improving swaps immediately, because each
epoch samples about 4,000 of roughly 9x10^10 possible swaps. But pairs beat the
best single move by only 0.28-1.55 coarse generations on four replicates
spanning the range, so the two-move escape those methods exploit is not the
binding constraint. **Rejected without building: the mechanism is not the
bottleneck.**

### Diversity-aware restart selection

`--selection-tolerance` lets a replicate publish any restart within a fraction
of its own best W1, taking the one whose rows are least reused by earlier
replicates. It works:

| selection rule | unique | vs hard-q50 | max reuse | `E_r` med | QC |
|---|---:|---:|---:|---:|---:|
| minimum W1 | 195,836 | 0.753 | 30 | 288.1 | 96/100 |
| tolerance 5% | 212,296 | 0.816 | 20 | 289.2 | 96/100 |
| tolerance 15% | 236,747 | 0.910 | 14 | 301.6 | 94/100 |

At 5% the diversity gain is nearly free; at 15% it recovers 91% of hard-q50's
diversity for a 4.7% matching cost. **Superseded by `--disjoint-replicates`**,
which reaches 406,700 unique controls -- 1.56x hard-q50 -- for a 2% cost. The
flag is retained and tested but unused, and is mutually exclusive with disjoint
mode.

### The log-age distance metric

`--distance log-age` with `age_grid_weights` and `wasserstein_1_log_age`,
weighting by `d(log t)` so young bins carry up to 25,470x the weight of old
ones. Built to fix a 21.9% relative age error at the 10% CDF quantile.

It does fix it -- but so does a paired control that changes only the *search
grid* and keeps linear W1:

| quantile | linear metric<br>linear screen | linear metric<br>log screen | log metric<br>log screen |
|---:|---:|---:|---:|
| 10% | 21.9% | **−0.0%** | −0.1% |
| 25% | 9.2% | 0.0% | 0.0% |
| 50% | −3.7% | −0.0% | −0.0% |
| 90% | −0.0% | −0.0% | −0.0% |

and the control's lower-tail concordance is better (O/B at 5%: 0.994 against
0.966). **Rejected: the metric change is unnecessary.** Keeping linear W1 leaves
every published threshold valid and "age-matched" unredefined. The log-age code
is retained behind a non-default flag; `wasserstein_1` is unchanged and
linear-mode weights are bit-identical to `np.diff(ages)`.

### Running both samplers side by side

Publishing the hard-q50 and bootstrap-target bundles together, hard-q50 as the
reference and bootstrap-target as an uncertainty-propagating sensitivity
analysis. **Rejected on methodological grounds**: two null distributions require
justifying which one is reported, which is a forking-paths problem. A single
prespecified method is more defensible even when it is the less flattering one.

### A folded Phi-SFS sensitivity check

Proposed to bound ARG polarization bias, since folding (`k -> min(k, n-k)`) is
polarization-invariant. **Rejected**: folding merges bin `j` with bin `20-j` and
therefore cancels antisymmetric differences exactly -- an excess of rare derived
alleles with a deficit of common ones, which is the purifying-selection
signature and the most likely real result. A null folded result could not
distinguish "no artifact" from "folding removed the signal". Replaced by
polarity perturbation, which keeps all 19 bins.

---

## Abandoned pipeline stages

Whole stages that were once part of the documented workflow and are not run any
more. They are recorded here rather than in the README, which is a how-to for
the supported route.

### The hard-q50 swap sampler (former README §5) — abandoned

`sample_age_matched_controls.py`. **Abandoned**: the bootstrap-target matcher
(now README step 4) replaced it as the reported result, and it is no longer a
prerequisite for anything. It is not part of the pipeline and should not be run
for a new analysis; the code and `SWAP_SAMPLER_HPC_HOWTO.md` are retained only
for reproducing earlier analyses.

**Why it was replaced.** Hard-q50 defines its tolerance from bootstrap
uncertainty but does not propagate it, so its saved sets occupy a narrow shell
just inside q50. On the 75-draw in-gene target its 100 sets span
1,367.98–1,480.47 generations against a 1,480.48 threshold, a median 1.27%
inside it with a standard deviation of 1.4% of the threshold. The bootstrap
matcher instead gives every replicate its own bootstrap target, so the
replicates span that uncertainty instead of sitting against one fixed boundary.
`BOOTSTRAP_HPC_VALIDATION.md` T2 carries the measurement.

**A second reason it stopped being a prerequisite.** An earlier design seeded the
optimizer from a hard-q50 bundle through `--seed-sets`. That flag is gone: each
restart is now a stratified draw from the target's own equal-mass age strata,
which the target bundle already ships.

**Gate 10.** The gate originally required hard-q50 to be run alongside as a
sensitivity analysis. It was **superseded** — see "Running both samplers side by
side" above — because publishing two null distributions requires justifying
which one is reported.

**What the sampler did.** It found a set inside the single threshold and then ran
a constrained random swap walk; the construction state itself was not saved.
Each chain performed one fixed accepted-swap sweep per set member before its
first save and another fixed sweep between saves, so saving was not triggered by
a path-dependent replacement crossing. Membership replacement was reported as a
mixing diagnostic rather than used as a stopping rule. Every saved set was
recomputed on the exact 1,000-generation grid and had to remain inside the
target threshold. Construction started on the coarse `--search-bin-width 20000`
grid; if an epoch accepted no improving swaps while the exact distance was still
too large, the sampler halved that width and recomputed construction state,
stopping at the exact target-grid width, and three consecutive zero-acceptance
epochs at exact resolution raised an explicit plateau error. The matching
threshold was never relaxed.

The command, as it was documented:

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

Defaults were 100 saved sets as 10 independent chains × 10 sets, one
accepted-swap sweep per set member during burn-in, and one per member between
saved states. `--all-eligible` used every eligible non-target SNP as the
candidate universe; `--candidate-rows` restricted it. The local CLI defaulted to
one worker to bound memory, and the SLURM workflow ran each chain as a separate
one-CPU array task.

Each output directory held `row_indices.npy` of shape `(100, X)`; `positions`,
`chromosome_codes`, and `chromosome_labels`; `cdfs.npy` and `wasserstein.npy`;
the `target_cdf` and `age_bins` used for exact certification; `chain_index.npy`
and `sample_index.npy`; `diagnostics.csv`; `reuse_row_indices.npy` and
`reuse_counts.npy`; and a `metadata.json` with seeds, settings, chain histories,
membership overlap, Wasserstein autocorrelation, an overlap-based ESS heuristic,
store identity, and provenance. Publication failed if any set contained
duplicate controls, violated the declared candidate universe, had a stored CDF
inconsistent with its rows, or exceeded the exact Wasserstein threshold.

**Its 100 sets are correlated Monte Carlo states**, with correlation confined
within each ten-set chain — a different object from the bootstrap matcher's
replicates. Retain `chain_index.npy` and `sample_index.npy` when joining
results, plot the downstream statistic by chain, and measure its within-chain
autocorrelation before interpreting the empirical null. Correlation does not
bias a simple Monte Carlo average, but it reduces precision and must not be
ignored in standard errors or effective replicate counts.

**Φ-SFS still reads these bundles**, with no flag to distinguish them: the
per-replicate identifier arrays are selected from the bundle's
`schema_version`, and are carried into every output.

| bundle | `schema_version` | identifiers |
|---|---|---|
| hard-q50 swap sampler | `swap-age-matched-controls-v1` | `chain_index`, `sample_index` |
| bootstrap-target matcher | `bootstrap-target-matches-v1` | `replicate_id` |

**Its SLURM manifest workflow**, also abandoned, drove many categories from one
tab-delimited manifest on Quobyte:

```text
label	positions	target	output	seed
all_te	/quobyte/project/te/all.pos.txt	/quobyte/project/targets/all_te	/quobyte/project/matches/all_te	1001
in_gene	/quobyte/project/te/in_gene.pos.txt	/quobyte/project/targets/in_gene	/quobyte/project/matches/in_gene	1002
young	/quobyte/project/te/young.pos.txt	/quobyte/project/targets/young	/quobyte/project/matches/young	1003
```

`positions` is the input chromosome/VCF-position file; `target` and `output` are
durable output *directories*, unique per row, that must not already contain
incomplete results. `build_age_targets.sbatch` built the targets;
`sample_age_matches.sbatch` ran one independently seeded chain per array task
and `gather_age_matches.sbatch` validated ten durable chain bundles before
publishing one 100-set directory. For `T` manifest rows that is `10*T` chain
tasks and `T` gather tasks, with the gather array submitted under an `afterok`
dependency:

```bash
export PROJECT=/quobyte/project/normalizeTE
export STORE=/quobyte/project/data/snp_interval_store
export MANIFEST=/quobyte/project/manifests/te_manifest.tsv
export AGE_MATCH_TASK_COUNT=10

sbatch --export=ALL,PROJECT,STORE,MANIFEST,AGE_MATCH_TASK_COUNT \
  build_age_targets.sbatch

export T=3
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

Each launcher verified `SLURM_ARRAY_TASK_COUNT` against its declared task count,
staged the store to its own `$TMPDIR`, and wrote one atomically published
`OUTPUT.chains/chain-NNN.npz` bundle per chain task. Reruns validated and skipped
complete bundles, and the gather refused to publish until all ten matched the
target, candidate universe, store identity, derived chain seeds, provenance, and
every exact row-derived CDF check. `SWAP_SAMPLER_HPC_HOWTO.md` holds the full
manifest rules, memory model, and restart behavior, and
`AGE_MATCHED_CONTROL_SAMPLER_PLAN.md` the sampler design and its gates.

### Legacy alternatives kept for reproducibility

Neither is part of the pipeline; both predate the interval-store route and are
retained only to reproduce older analyses.

**Dense CDF store.** `build_snp_age_store.py` is an alternative to
`build_snp_interval_store.py` — never run both as sequential steps. It quantizes
every SNP posterior to a CDF on a fixed age grid:

```bash
python build_snp_age_store.py \
  project-data/posterior/*.tsz \
  --numpy-store age_store \
  --bin-width 1000 \
  --block-snps 100000 \
  --min-usable-fraction 0.1 \
  --scratch-dir "$TMPDIR"
```

Its temporary float32 accumulator uses about
`4 * number_of_SNPs * number_of_age_bins` bytes — roughly 15 GiB for 20 million
SNPs and 200 bins, or 75 GiB for 1,000 bins — so fine grids cost substantially
more disk and scratch, and changing the bin width requires a rebuild.
`--omit-transpose` reduces final disk use at the cost of slower candidate scans.

**Stratified rejection sampler.** `sample_age_matched_syn.py` divides a target
into equal-mass age strata, materializes candidate weights, draws complete
proposed sets, and accepts those below the Wasserstein threshold. It is less
suitable for many large TE datasets because it builds target-specific candidate
weights and can spend many proposals in rejection; §1.3 of
`BOOTSTRAP_HPC_VALIDATION.md` records the O(pool) inner loop behind its 19-hour
job.

---

## Superseded readings

Conclusions that later measurements overturned. Recorded because each was stated
with more confidence than the evidence supported, and the pattern is worth
seeing.

| reading | why it was wrong |
|---|---|
| "294 of 300 restarts stopped at plateau, so more search cannot lower the floor" | The plateau is a *sampling* plateau. Each epoch samples ~4,000 of ~9x10^10 possible swaps, so five quiet epochs mean the draw missed, not that nothing exists. Improving swaps are found immediately by systematic search. |
| "The floor is structural; the lower-tail distortion is permanent" | It was a screening-grid artifact. A log-spaced screen removes it entirely. |
| "The lower-tail distortion follows from W1 being denominated in generations" | Also wrong, and the same fix disproves it: a paired control changing only the screen recovers the young end with linear W1 intact. |
| "Small targets will be harder -- `R_r` near 0.4 at 600 sites" | `R_r` at 600 sites is 0.139, *better* than 0.165 at 4,067. The estimate came from two targets that differed in both size and TE composition. Holding composition fixed by subsampling reversed it: `B_r` scales as ~n^-0.46 while `E_r` scales as ~n^-0.34, so the ratio improves as targets shrink. |
| "The in-gene target at 4,067 sites is close to the worst case, not typical" | `SWAP_SAMPLER_HPC_HOWTO.md` §10 already anticipates a ~500-site target, and most categories are under 5,000 -- so 4,067 is near the top of the real range, and is the *worst* case for gate 5 while being the *best* measured case for gate 7. |
| "TE sites appear in only ~58% of draws" | That came from the per-draw tskit VCF exports, which do not describe the ARGs. Measured from the store, TE targets have a median present-draw count of 75 of 75. |
| "ARG polarization bias could masquerade as signal in Phi-SFS" | It attenuates rather than manufactures: perturbing polarity at the measured error rate reduces Phi by 15%, monotonically across three rates. A positive result is robust to it; a null result is not. |
| "No Claude Code setting stamps messages with a time" | `showMessageTimestamps` exists. One grep would have settled it. |

Two process notes from the same period, since they cost more time than any of
the above: a `sed` pattern intended for one line matched three and silently
repointed two job inputs at directories that did not exist, and a successful
`sbatch` submission was reported as a running job twice when the job had already
failed. Verify that an edit changed only what was intended, and distinguish
submitted from running from produced-the-expected-output.
