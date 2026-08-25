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
