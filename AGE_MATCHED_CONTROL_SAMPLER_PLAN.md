# Age-matched control sampling for many TE sets: implementation plan v2

This supersedes `GLOBAL_QUANTILE_SAMPLER_IMPLEMENTATION_PLAN.md`. It keeps that
plan's engineering discipline — provenance, atomic publication, deterministic
seeds, explicit failure — and replaces its statistical core and its cost model.

Goal, unchanged: hold posterior ages for all ~23 million SNPs in the canonical
interval store, and build control SNP sets whose age profile matches each of
dozens of TE categories.

## 0. Why v1 is replaced

Three findings. The first is the important one.

### 0.1 The quota rule does not target the quantity that is scored

v1 §4.4 sets the quota for stratum `j` to the target's probability mass in
stratum `j`, then draws a control with probability proportional to
`w_ij = P_i(age in bin j)`. The existing `te_age_target.equal_mass_boundaries`
plus `largest_remainder_quotas` does the same thing over 20 target-specific
strata, so this is a property of the current pipeline, not something v1
introduced.

Acceptance, however, scores the mean of the selected SNPs' **full** posteriors.
Writing `F_i` for candidate `i`'s posterior CDF and `pi_i` for its final
inclusion probability under the complete sequential algorithm, the exact
statement is

```text
E[aggregate CDF of the matched set] = (1 / n) * sum_i pi_i * F_i
```

and the useful approximation is

```text
E[aggregate CDF of the matched set] ~= sum_j p_j * G_j
G_j = (sum_i w_ij F_i) / (sum_i w_ij)
```

The mixture form is exact only for independent within-stratum draws with no
global uniqueness constraint. Today's sampler uses `rng.choice(replace=False)`
within a stratum and removes selected SNPs from later strata, so inclusion
probabilities are not exactly `q_j * w_ij / sum_k w_kj`, and later strata depend
on which strata ran first. Randomizing stratum order averages that dependence;
it does not remove it. The mixture is the low-depletion limit, and `n / N` here
is tiny, but it must be **diagnosed rather than assumed** — see §2.3.

`G_j` is the mean posterior of the control SNPs holding mass in bin `j`. With 75
SINGER draws a SNP's posterior is far wider than one global-quantile bin, so
`G_j` is not concentrated in bin `j`; it is broad. The map `T -> sum_j p_j G_j`
is therefore a **smoothing operator**, and it is the identity only when the
target equals the whole control pool. For any TE category whose age profile
actually differs from the genome-wide SNP profile — that is, every interesting
category — the matched set is systematically shrunk toward the pool and never
reaches the target.

Equivalently: the target's bin masses are already posterior-smoothed. Using them
as quotas and then drawing SNPs whose posteriors smooth again applies the
smoothing kernel one time too many.

Measured on a 20,000-SNP / 75-draw / 200-bin synthetic stand-in, young-biased
overlapping target, target SNPs excluded from the control pool
(`design_probes/deconvolution_check.py`):

| | W1 to target (generations) | acceptance |
| --- | ---: | ---: |
| unmatched control pool | 12,363 | — |
| bootstrap 95% threshold | 81 | — |
| **quota = target bin mass (v1 and current code)** | **692** | **0%** |
| **quota fitted to `{G_j}`** | **30** | **100%** |
| oracle: best achievable by any quota vector | 3.5 | — |

The realized median (692) sits essentially on the predicted expectation (661),
confirming this is systematic bias, not proposal noise. No amount of alias-table
engineering removes it.

It bites hardest where categories are large. At `n = 200` the bootstrap
threshold is loose (279) and both quota rules pass; at `n = 2,000` the threshold
tightens to 94 and only the fitted rule passes. The bias does not shrink with
`n`; the threshold shrinks as `n^-1/2`.

**Caveat, stated plainly:** this is a synthetic lognormal stand-in, not the real
store. The magnitude is model-dependent. The mechanism is not — it is an
identity that holds whenever posteriors are wider than strata, which the real
75-draw store certainly satisfies. Gate 1 confirms the magnitude on real data
before anything else is built.

### 0.2 The 175 GiB index buys nothing that a 3 GiB index does not

v1 §5–7 commits to a persistent `(K=1000, N=23.5e6)` float32 alias-probability
array plus a uint32 alias-index array (175 GiB), an 87.5 GiB temporary weight
matrix, 350+ GiB node-local scratch, 384–512 GB RAM, 24–48 hours, and a compiled
Numba/C++ kernel.

Sparsity does not rescue the exact approach: measured on the same stand-in, a
SNP's posterior has positive weight in **52% of bins** (mean 103 of 200), so CSR
saves only ~1.9x.

But the exact weights are not needed. v1 decision 8 already concedes that the
alias layer is an acceleration layer and that acceptance is exact. So the
proposal layer may be approximate. Drawing `R` ages per SNP from its own
posterior and bucketing them by age boundary makes a SNP's draw probability
proportional to its Monte Carlo count in that bin — an unbiased estimate of
`R * w_ij` — at `N * R * 4` bytes:

| R | entries | storage |
| ---: | ---: | ---: |
| 16 | 376 M | 1.4 GiB |
| 32 | 752 M | 2.8 GiB |
| 64 | 1.5 B | 5.6 GiB |

Measured accepted-set quality is indistinguishable from exact `w_ij`
(`design_probes/mc_bucket_check.py`, `design_probes/combined_check.py`). At `n = 2,000`
with fitted quotas both reach the threshold; at `n = 200` both accept at 100%.

Because bucketing `N * R` values against any boundary vector is a `searchsorted`
plus a counting sort, **fixed global boundaries become a default rather than a
constraint**; v1 decision 4 can be relaxed. Do not assume this is free: 752
million values is memory-traffic bound and the bucket index has to be stored, so
Gate 2 must measure it before target-specific rebucketing is promised per
category.

### 0.3 The plan optimizes the cheapest phase

`_aggregate_uniform_interval_cdf` already scores a proposal exactly in
`O(75n + B)` on the uniform analysis grid. For `n = 10,000` that is 750,000
interval records, about 6.4 MiB of canonical input. Proposal generation is
`O(n)` random lookups. Neither justifies a 23.5-billion-cell product. Throughput
is governed by acceptance rate — which §0.1 shows was going to be near zero.

## 1. Confirmed decisions

Carried over from v1 unchanged:

1. The control pool is the full ~23.5 million eligible SNPs, not a
   synonymous-only subset.
2. The canonical `snp-age-interval-v1` store remains the source of truth and is
   never replaced or deleted. Every derived product is reproducible from it and
   records its digests.
3. Posterior weighting stays equal-per-usable-interval, matching
   `cdf_at(..., weighting="interval")`. See §7.1 — this needs one explicit
   confirmation.
4. Final scoring uses the exact aggregate CDF reconstructed from the interval
   store, on the same analysis grid, with the same `wasserstein_1`. (The
   *threshold* applied to that distance changes — see §2.5.)
5. Sampling is without replacement within a matched set; controls may be reused
   across independently generated sets.
6. Heavy work runs on SLURM compute nodes; transient files go to node-local
   scratch; results publish atomically.

Changed:

7. **Quotas are fitted to the bin response functions `{Ghat_j}`, not copied
   from the target's bin masses.** (§2.2)
8. **The proposal index is a Monte Carlo posterior-draw matrix, not an alias
   table.** (§3.1)
9. **Boundaries are global by default but not fixed by construction.** (§3.2)
10. **Acceptance is split into design adequacy and realized-set
    variability, and no longer rests on the target-only bootstrap alone.**
    (§2.5)

## 2. The estimator

### 2.1 Estimand

State it explicitly, because §0.1 turns on it, and state it in the language that
survives review:

> A control sample **calibrated so that its mean posterior age distribution
> matches the TE category's mean posterior age distribution.**

Not "drawing the same proportion of controls from each TE age stratum." The
fitted quotas of §2.2 are design weights that invert a blurred sampling
operator; they are not estimates of the fraction of TEs in latent age bin `j`,
and papers and metadata must not describe them as such.

Under this estimand, target-mass quotas are an internal construction that
demonstrably fails to deliver the stated goal, and calibration is the correct
estimator — not cheating. The cost of the change is interpretive, and §2.5
carries the diagnostics that keep it honest.

### 2.2 Bin response functions and quota fitting

For boundaries `b_0 < ... < b_K` over the eligible control pool, the response
function is defined from the **frozen-draw bucket multiplicities that sampling
actually uses**, not from the ideal weights:

```text
Ghat_j = (sum_i m_ij F_i) / (sum_i m_ij)
```

where `m_ij` is SNP `i`'s multiplicity in bucket `j` and `F_i` is its exact
interval-store posterior. Using `m_ij` rather than `w_ij` is not a compromise:
it makes fitting and sampling consistent, so much of the sketch error in §3.1 is
absorbed into `Ghat_j` instead of appearing as unexplained bias.

`Ghat_j` is **not** computed as `(M^T @ CDF) / colmass`; that contraction is
`N*K*B` and impossible at production scale. Estimate it by subsampling `S`
bucket **entries** (default 5,000, so multiplicity is carried automatically),
gathering their intervals, and calling the existing exact
`_aggregate_uniform_interval_cdf`. Cost is `O(S * 75 + B)` per bin — minutes,
once per index. Record `S`, the seed, and a per-bin standard error.

Quotas for a target `T` are then chosen **lexicographically**, not by a single
penalized objective:

1. Solve `d* = min over x in simplex of W1(x @ Ghat, T)`. This is convex; on a
   fixed grid one-dimensional W1 is a weighted L1 norm of CDF differences, so it
   is an LP, or a projected subgradient solve in well under a second.
2. Among `x` with `W1(x @ Ghat, T) <= d* + epsilon`, minimize a **design**
   penalty:

```text
lambda_1 * || x - x_targetmass ||_1        keep near the natural mixture
+ lambda_2 * sum_j (second_difference(x)_j)^2   forbid oscillatory adjacent bins
+ lambda_3 * sum_j x_j^2 / distinct_j           discourage concentration in weak bins
```

3. Apportion to integers with the project's existing
   `largest_remainder_quotas`, then **rescore the rounded vector** — generic
   largest-remainder rounding is not guaranteed to preserve the fitted CDF, and
   the rounding step must not be allowed to silently spend the tolerance.

`epsilon` is set from measured resolution limits — sketch uncertainty (§3.1),
integer rounding, proposal-to-proposal variation, and exclusion-correction error
— and regularization may consume only a small declared fraction of the final
equivalence tolerance. Never choose `lambda` because the quota curve looks
smooth.

`{Ghat_j}` are broad and strongly collinear, so `x` is **not identifiable**:
very different quota vectors give essentially the same fitted CDF. That is
tolerable because `x` is never interpreted, but it is why the design penalty
exists — quotas still control candidate concentration, exclusion sensitivity,
depletion, covariate composition, and reproducibility. Stability gates therefore
apply to the **fitted response**, not to the quota vector (§8, Gate 2).

Record the unregularized fit, the chosen penalty weights, the target-mass fit,
and `d*` for every category.

### 2.3 Validating the linear-response approximation

Because §0.1's mixture form is an approximation, every category must report:

- within-bin sampling fraction and unique-SNP effective sample size;
- entry-level and SNP-level rejection/collision rates, separately — entry-level
  collision badly understates SNP-level concentration;
- predicted W1 from `sum_j (q_j / n) Ghat_j` versus the empirical mean W1 of
  repeated exact proposals.

A large gap between predicted and empirical W1 means the low-depletion
approximation has failed locally — typically a tail bin with low effective
capacity, or a few SNPs dominating several fitted bins — and the category must
be flagged rather than accepted.

### 2.4 Proposal draw

Sample bins in randomized order. To fill bin `j`'s quota, draw uniformly from
bucket `j`, rejecting entries that are already selected in the current set or
excluded for this category. Multiplicity within the bucket encodes `w_ij`.

Name the design precisely, as v1 §4.5 did not: this is **successive
probability-proportional-to-size sampling without replacement, conditional on a
randomized stratum order**. It is not equivalent to Gumbel-top-k or to
noncentral-hypergeometric designs, which give different subset probabilities.
Test it against a small float64 reference implementation rather than asserting
equivalence.

Uniform-within-bucket sampling is correct only if the following are handled, and
each needs an explicit test:

- a SNP appearing several times in one bucket must not be selected twice;
- uniqueness applies globally across all buckets, not per bucket;
- rejection yields the correct sequential renormalization only while entries for
  already-selected *or excluded* SNPs are redrawn rather than skipped;
- entry-level collision rate understates SNP-level concentration — record
  unique-SNP ESS and maximum multiplicity, not just redraw counts;
- a fitted quota can exceed a bucket's unique *eligible* capacity even when its
  entry count is large. Validate against distinct eligible candidates after the
  exclusion mask, not against `total_bucket_mass`.

### 2.5 Acceptance

The target-only bootstrap threshold answers "how far might another realization
of the target differ from the observed target?" That is the target's own
sampling variability, which an engineered control sample has no obligation to
reproduce, and it tightens as `n^-1/2` while sampler approximation bias does
not. §0.1 is exactly that mismatch surfacing. Split acceptance in two.

**Criterion 1 — design adequacy** (evaluated once per category, before
sampling). Is the fitted expected response `sum_j (q_j / n) Ghat_j` close enough
to the observed target? Judge against a **prespecified scientific equivalence
margin** — an age mismatch large enough to materially confound the downstream
comparison — and additionally require a declared relative improvement over the
unmatched control pool. In the §0.1 measurement that pool distance was 12,363
generations, so "how much of the gap did matching close" is a meaningful and
size-stable statistic in a way the bootstrap threshold is not.

**Criterion 2 — realized-set variability** (evaluated per proposal). Simulate
from the fitted design and characterize the distribution of exact W1 values,
which absorbs integer quotas, finite control variation, depletion, and exact
interval-store scoring. A proposal is accepted if it is consistent with that
simulated distribution.

Scoring itself is unchanged: reconstruct the proposal's exact aggregate CDF via
the same `aggregate_cdf_at` uniform-grid path, on the same analysis grid, with
the same equal-per-interval weighting, the same edge convention, and the same
`wasserstein_1`. The frozen draws affect only proposal construction; they never
enter scoring.

**Selection-bias warning.** Accepting the best 100 sets out of a very large
proposal pool conditions the controls toward unusually favorable random
realizations, and the induced selection is currently unrecorded. This applies to
the pipeline as it stands today. Prefer a calibrated design whose ordinary
proposals pass at a high prespecified rate, and always record the proposal count
that produced the accepted sets.

**If inferential uncertainty in the target must be propagated,** use a
two-source bootstrap: resample target SNPs, refit the calibration, sample
controls under the refitted design, and evaluate the target-control discrepancy.
That estimates the uncertainty of the complete procedure rather than of the
target alone.

**Report on every proposal, accepted or not** — these are what the aggregate CDF
cannot see, and they are the guard against a mixture that reproduces a curve
without resembling the target biologically:

- the aggregate-CDF W1 and its predicted counterpart from §2.3;
- the W1 between the matched set's and the TE set's distributions of per-SNP
  posterior point estimates (`store.mean_ages`) — a mixture balancing very young
  against very old SNPs can match the aggregate curve while containing no SNP of
  typical target age;
- the posterior-width distribution — controls with broad, uncertain posteriors
  can reproduce the mean CDF of TEs with sharp ones;
- unique-SNP effective sample size and maximum inclusion propensity;
- fitted quota profile versus target-mass profile;
- balance on whatever genomic covariates §5.5 settles on;
- stability of all of the above across equivalent fits.

### 2.6 Where this goes next

Binning is a means, not the ideal formulation. The clean statement of the
problem is direct per-SNP calibration: find inclusion probabilities `pi_i` with

```text
sum_i pi_i = n,   0 <= pi_i <= 1,   sum_i pi_i F_i ~= n * T
```

and then convert them into a fixed-size sample by a balanced sampling design.
That eliminates the artificial stratum layer and optimizes the scored quantity
directly.

It is not practical at production scale as stated: 23.5 million variables
against ~23,000 grid constraints implies a dense response matrix of roughly
5.4e11 entries, over 2 TB even in float32, with highly redundant constraints
that may be infeasible exactly once `pi_i <= 1` and fixed size are imposed.

A **reduced-feature** version is tractable and is the right long-term direction:
represent each `F_i` by tens of spline coefficients, CDF landmarks, or
multiscale interval masses, calibrate those features by streaming matrix-vector
products with entropy balancing or bounded exponential tilting, and keep exact
full-grid scoring as validation. Do not build it now. Fitted bin responses are
simpler, auditable, and likely adequate provided §2.3's linear-response
approximation is empirically validated.

## 3. Data products

### 3.1 Posterior draw matrix — `snp-posterior-draws-v1`

```text
posterior_draws/
├── metadata.json
├── candidate_rows.npy      int64  (N,)   strictly increasing canonical rows
└── sampled_ages.npy        float32 (N, R) R ages per SNP from its own posterior
```

Built in one streaming pass over the interval store: for each row block, read
its ragged intervals, draw `R` interval indices uniformly in `[0, m_i)` and `R`
uniforms, and set `age = L + u * (U - L)`. Pure NumPy, vectorized per block. No
compiled kernel. `R = 64` default, ~5.6 GiB at `N = 23.5e6`.

Metadata records the source-store digests, `R`, the draw seed, block size, per-
phase timings, and array digests. Same staging/`complete: true`/atomic-rename
discipline as v1 §7 Phase F.

**Frozen-draw correlation.** One matrix reused for every set of every category
makes its Monte Carlo partition error **common to all categories and all
replicates**; 100 accepted sets do not average it away. Two mitigations, both
required:

1. *Decorrelate replicates.* Store `R = 64` and bucket each accepted-set
   replicate from a distinct, seed-derived 16-column subset.
2. *Measure the common component.* Build the panel as `2R` and split it into two
   independent halves. Construct responses and fit quotas independently on each,
   cross-evaluate each fitted solution against the **other** half's response
   matrix, and run exact interval-store scoring on pilot proposals from both.
   Two panels suffice to estimate the common sketch component; independent
   sketches are not needed per accepted set.

Production is gated on **response stability** across the two halves — the fitted
expected CDF, realized exact W1, effective sample size, and covariate balance —
not on quota-vector stability, which is expected to be unstable because §2.2's
fit is non-identifiable. If the halves disagree materially, increase `R`.

### 3.2 Boundaries and responses — `age-bin-response-v1`

```text
bin_response/
├── metadata.json
├── boundary_probabilities.npy   float64 (K+1,)
├── boundary_ages.npy            float64 (K+1,)  strictly increasing
├── bin_response.npy             float64 (K, B)  Ghat_j on the analysis grid
├── bin_response_stderr.npy      float64 (K, B)
├── bucket_offsets.npy           int64   (K+1,)
├── bucket_entries.npy           uint32  (N*R,)  SNP row per bucket entry
├── entry_multiplicity_max.npy   int64   (K,)    max m_ij within the bucket
├── distinct_candidates.npy      int64   (K,)
└── total_bucket_mass.npy        int64   (K,)
```

`bin_response.npy` holds `Ghat_j` as defined in §2.2 — built from the frozen
bucket multiplicities `m_ij`, with `F_i` taken exactly from the interval store.
It is meaningful only in combination with the specific draw matrix it was built
from, whose digest it records and against which it refuses to open.

`K = 200` default, not 1,000. The fit in §2.2 needs enough basis functions to
span `T`, not fine resolution: `K = 200` already reached an oracle W1 of 3.5
against an 81-generation threshold. `bin_response.npy` is 184 MB at `K = 1000`,
`B = 23,000`, so `K` is a tuning knob, not a cost driver. Sweep it at Gate 2.

Global equal-mass boundaries of the aggregate control CDF are the default.
Target-specific boundaries remain available because rebucketing is cheap, and
the two should be compared at Gate 2 — but see §0.2 on not over-promising the
rebucketing cost before it is measured.

Global aggregate CDF construction keeps v1 §7 Phase B: a streaming difference-
array kernel over the ragged interval arrays, accumulating in float64,
validating monotonicity, finiteness, and terminal value. Use `np.bincount`
rather than `np.add.at`; the latter is the slow `ufunc.at` path and appears in
`_aggregate_uniform_interval_cdf` today.

Boundary compression, half-open conventions, and full support at both ends are
unchanged from v1 §4.2 and must be tested explicitly.

### 3.3 What is deleted from v1

`weights_by_bin.npy`, `alias_probability.npy`, `alias_index.npy`,
`effective_sample_size.npy`, the compiled weight and alias kernels, the durable
per-bin alias build manifest, the active-bin staging layer, and the 350 GiB
scratch / 512 GB RAM envelope. `ESS_j` is replaced by `distinct_candidates.npy`,
which is the direct quantity and is exact.

## 4. Pipeline

### 4.1 `build_posterior_draws.py`

Resolve the candidate universe once (v1 §7 Phase A is kept verbatim, including
`--all-eligible` to avoid parsing a 271 MB text file per build), then stream the
draw matrix. Publish atomically.

### 4.2 `build_bin_response.py`

Global aggregate CDF, boundaries, bucketing, bucket subsampling, `G_j`
estimation, diagnostics. Publish atomically. Records the digest of the draw
matrix it was built from and refuses to open against a different one.

### 4.3 `build_te_targets.py` — batched target construction

v1 §9 batched sampling but not target construction, which
`te_age_target.build_target` currently does one category at a time with its own
scratch CDF matrix and bootstrap. Dozens of categories need one manifest, shared
position resolution, one store open, and per-category seeds derived
deterministically from a master seed and a stable category label — never from
manifest row order or worker count. Overlapping categories must be declared, not
discovered.

### 4.4 `sample_age_matched_controls.py`

Per category: verify provenance digests match; apply the exclusion mask (§5.1);
fit quotas (§2.2); draw proposals (§2.4); score and accept (§2.5); publish
atomically with full diagnostics. `sample_age_matched_syn.py` is retained as a
thin compatibility wrapper until migration completes.

Progress and checkpoint reporting is carried over from v1 §8.3 unchanged; it was
the right list. Add: fitted-vs-target-mass quota comparison, chosen `lambda`,
lost mass and lost distinct candidates from exclusion, and both acceptance
distances.

### 4.5 `sample_many_age_matched_controls.py`

Manifest-driven batch runner, as v1 §9, but the staging problem largely
evaporates: the draw matrix is ~5.6 GiB and the response matrix ~184 MB, so an
entire node-local copy is trivial. Partition categories into node-sized batches;
open the interval store and both derived products once per allocation.

## 5. Scientific gaps v1 did not cover

### 5.1 Control exclusion — required, currently absent

The all-SNP pool contains the target TE SNPs themselves. They must be excluded
from their own controls. Beyond that, the following are **PI decisions**, and
the plan implements whichever is chosen rather than guessing:

- exclusion of SNPs within a configurable bp window of the target TE insertion
  or annotated TE body;
- exclusion of SNPs in LD with target variants;
- exclusion of all SNPs belonging to the same TE category, or to any TE.

Implement as a per-category boolean mask applied at bucket-draw time by
rejection.

**Exclusion changes the response and the fit must follow.** Rejection sampling
correctly implements the excluded *sampling* distribution, but fitting against
the unadjusted `bin_response` then predicts the wrong curve. The exclusion-
adjusted response is

```text
Ghat_j^(-E) = (sum_{i not in E} m_ij F_i) / (sum_{i not in E} m_ij)
```

which is obtained **exactly** (conditional on the frozen sketch) by subtracting
delta terms from the stored numerator and denominator:

```text
D_j = sum_{i in E} m_ij           H_j = sum_{i in E} m_ij F_i
```

For the TE SNPs themselves and modest bp windows this is cheap. For large
window- or LD-based exclusions, `H_j` touches every excluded SNP's full CDF and
gets expensive; then either recompute affected high-quota bins by a
category-specific streaming pass, or demonstrate the delta is negligible against
a declared bound.

Note that lost bucket mass alone is **not** a sufficient diagnostic: a small
removed mass matters if the removed SNPs have a distinct response. Report `D_j`,
the norm of the induced change in `Ghat_j`, and the lost distinct-candidate
count per bin.

### 5.2 Cross-category reuse — a design decision, not an implementation detail

The plan must state whether controls may be shared between replicates of one
category, between TE families, and how nested or overlapping categories are
handled. Recommended default: **allow reuse across categories, forbid duplicates
within a set, and record reuse rates.** Enforcing global disjointness across
dozens of categories would make extreme-age categories infeasible and would make
results depend on category processing order. Flagged for PI confirmation.

### 5.3 The acceptance threshold is not comparable across category sizes

§2.5 replaces the single bootstrap criterion; this section records why, and what
still has to be settled with the PI regardless of which criterion is used.

The bootstrap threshold measures the *target's own* sampling variability, which
a control set has no obligation to reproduce. It shrinks as `n^-1/2`, so large
categories get very tight absolute thresholds while categories of tens of SNPs
get permissive and unstable ones. In the §0.1 measurements it ranged from 279
generations at `n = 200` to 81 at `n = 2,000` — a 3.4x swing from set size alone.

Required before production:

- a declared minimum category size, with bootstrap uncertainty reported;
- calibration by simulation at representative `n`, not assumed transferable;
- a predeclared infeasibility policy — v1 fails with a diagnostic, which is
  correct, but there must be a scientific fallback (report best achievable W1
  and its ratio to the threshold, and let the PI decide) rather than an empty
  result directory;
- acceptance diagnostics broken out by category size and profile shape.

### 5.4 W1 on a linear age scale is dominated by the old tail

The analysis grid runs to roughly 22.85 million generations at 1,000-generation
resolution. A W1 in generations therefore weights the ancient tail enormously
relative to the recent history where TE dynamics are usually interesting. A
log-age or rank/quantile-scale W1 may be the more appropriate distance. This
changes thresholds and possibly conclusions. **PI decision**; the code should
take the distance scale as an explicit parameter and record it.

### 5.5 Age-only matching

Age correlates with allele frequency, local SNP density, recombination rate and
LD, mappability, genic/intergenic context, and distance to genes. Matching on
age alone can trade age confounding for genomic-context confounding. The plan
should state the downstream estimand and which covariates are matched,
stratified, restricted, or adjusted afterward. The v2 index is a much better
foundation for this than v1 was: the draw matrix is `(N, R)` and additional
matching axes extend the bucket key rather than multiplying a 175 GiB table.

## 6. Testing with `run.combined.98.tsz` and `run.combined.99.tsz`

The fixtures referenced in `INTERVAL_STORE_BENCHMARKS.md` used draws 100–102,
which are not present. Rebuild the test fixture from the two draws that are.

- **`results/interval-2draw-store/`** — a two-draw all-SNP interval store from
  draws 98 and 99 via the existing `build_snp_interval_store.py` with float32
  endpoints and `chrom_offsets.combined.txt`. From the measured three-draw build
  (27.2 M rows, 74.7 M intervals, 24m26s, 34.8 GB peak RSS), expect roughly
  25–27 M rows, ~50 M intervals, ~1.1 GiB, and under 30 GB peak. **This exceeds
  comfortable headroom on a 36 GiB laptop — build it on a compute node.**
- Two draws give `m_i = 2`, so posteriors are far narrower than production's 75.
  The §0.1 smoothing effect will therefore be *weaker* here than in production.
  A two-draw fixture that already shows the bias is strong confirmation; one that
  does not is not a refutation, and Gate 1 must say so explicitly.
- Derive synthetic TE categories from the fixture by posterior-mean quantile —
  young-biased, old-biased, broad, and bimodal — at `n` = 100, 1,000, and 10,000,
  so that all four profile shapes and all three size regimes are covered without
  waiting for real TE annotations.
- Real TE position lists, when available, are resolved through
  `snp_position_resolution.py` exactly as the synonymous list is today.

Unit and property tests are carried over from v1 §10.1 and §10.2 with the alias
items removed and these added:

- bucket multiplicity converges to `w_ij` as `R` grows;
- `G_j` subsample estimate converges to the full-bucket aggregate;
- fitted quotas reproduce `T` better than target-mass quotas whenever `T != G`,
  and identically when `T == G`;
- the regularization ladder returns the target-mass solution as `lambda -> inf`;
- exclusion masks change `{G_j}` by less than the recorded bound;
- replicate decorrelation: two replicates using disjoint draw-column subsets
  have lower correlation in their selected sets than two using the same subset;
- successive-PPS draws match a small float64 reference implementation.

## 7. Questions that must be answered before Gate 2

1. **Interval versus draw weighting.** Equal-per-usable-interval weighting makes
   `draw_id` irrelevant: a draw contributing several usable mutation
   observations receives proportionally more posterior weight. Confirm this is
   scientifically intended rather than inherited. Equal-per-draw weighting
   changes `w_ij`, `G_j`, and every CDF.
2. **Exclusion policy** (§5.1).
3. **Cross-category reuse policy** (§5.2).
4. **Distance scale** (§5.4).
5. **Minimum category size and infeasibility policy** (§5.3).

## 8. Gates

Three gates replace v1's five. Nothing here needs a 24–48 hour build.

### Gate 1 — confirm the bias on real data (blocking, cheap)

Implemented: `gate1_smoothing_bias.sbatch` builds the two-draw store from
`run.combined.98.tsz` and `run.combined.99.tsz` and then runs `gate1_report.py`,
which measures realized W1 and acceptance for target-mass quotas versus fitted
quotas across the four synthetic profile shapes and three sizes, and writes
`results/gate1-smoothing-bias.json`.

The report emits an explicit verdict string — "section 2.2 is REQUIRED", "is NOT
required", "inconclusive", or "the acceptance criterion itself needs the §2.5
rework" — and never reports an infeasible or failed design as clearing the
threshold. Failed draws count against the acceptance rate rather than being
dropped, and a fitted design that would demand more distinct SNPs than a bin
holds is reported as infeasible rather than attempted.

This is the decision gate. If target-mass quotas already clear thresholds on
real data at production draw counts, §2.2 is unnecessary complexity and should
be dropped. If they do not — which the synthetic evidence predicts — nothing
else in the plan should be built until the fitted rule is in place.

### Gate 2 — parameter sweep and correctness

Sweep `R` in {16, 32, 64}, `K` in {100, 200, 500, 1000}, global versus
target-specific boundaries, and the §2.2 design-penalty weights. Correctness
fixtures against brute-force `interval_cdf`. Benchmark the rebucketing cost that
§0.2 assumes is cheap.

`R` is gated on **response stability, not quota stability** (§3.1), using: the
cross-sketch W1 between fitted expected responses; the change in `d*` between
`R = 32` and `R = 64` and 128; per-bin bucket mass and unique-SNP ESS; proposal
collision and rejection rates; the exact proposal W1 distribution; and
exclusion-adjusted versions of all of these. Quota-vector instability is
expected and is not a failure.

Also measure §2.3's predicted-versus-empirical W1 gap, which is the check on the
low-depletion approximation.

Deliver measured defaults replacing every default asserted in this document.

### Gate 3 — production pilot

Full draw matrix and response product built on a compute node. Five
representative real TE categories at small accepted-set counts, then a full
100-set category. Confirm progress reporting, checkpoint recovery, acceptance
rate, exact scoring, and end-to-end runtime before launching the manifest.

## 9. Resource envelope

Planning figures, to be replaced by Gate 2 and 3 measurements.

| Product | v1 | v2 |
| --- | ---: | ---: |
| canonical interval store | 17 GiB | 17 GiB (unchanged) |
| temporary weight matrix | 87.5 GiB | none |
| persistent proposal index | 175 GiB | 5.6 GiB |
| response matrix | none | 0.18 GiB |
| peak node-local scratch | 350 GiB | ~40 GiB |
| RAM | 384–512 GB | ~48 GB |
| build wall time | 24–48 h | hours |
| compiled kernel | required | none |

The draw-matrix build is one streaming pass over the same 17 GiB the store build
already writes, so the 16-hour store-build measurement is the right anchor for
its extrapolation, not a new class of job.

## 10. Definition of done

1. The canonical interval store is unchanged and fully usable.
2. Both derived products build reproducibly from declared seeds and refuse to
   open against mismatched source digests.
3. Gate 1 has answered whether fitted quotas are required, on real data, and the
   answer is recorded.
4. Matched sets contain no duplicate SNPs and honor the category exclusion mask.
5. Acceptance uses the unchanged exact full-CDF Wasserstein calculation plus the
   §2.5 acceptance structure, including the point-estimate criterion.
6. Representative categories across all three size regimes and all four profile
   shapes show no systematic age bias, with the fitted-versus-target-mass
   comparison reported for each.
7. All five §7 questions are answered and recorded in metadata.
8. A full 100-set category completes within the measured envelope, and the
   documented batch workflow processes the manifest without rescanning the
   control universe per category.
9. Interrupted jobs leave useful progress and resume safely.
