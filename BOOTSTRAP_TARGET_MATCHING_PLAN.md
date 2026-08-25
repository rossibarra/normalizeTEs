# Bootstrap-Target Age-Matching Design

## 1. Purpose

Replace the current fixed-target, hard-q50 control sampler with a workflow that
propagates uncertainty in the observed TE age CDF into the matched SNP sets.

The current sampler matches every SNP set to one observed TE target CDF and
requires its Wasserstein distance to fall below the median TE-bootstrap
distance. This uses bootstrap uncertainty to define a tolerance, but it does
not propagate that uncertainty: saved sets occupy a narrow shell immediately
inside q50.

The proposed workflow instead assigns every matched SNP set its own bootstrap
TE target CDF and minimizes its distance to that target. The desired relation
for replicate `r` is

$$
D(S_r,T^{(r)}) \ll D(T^{(r)},T),
$$

where:

- `T` is the observed TE age CDF;
- `T^(r)` is bootstrap TE target `r`;
- `S_r` is the optimized SNP-set age CDF; and
- `D` is the same exact-grid Wasserstein-1 distance used by the existing
  matching workflow.

When the matching error on the left is small, the triangle inequality implies
that the SNP set's displacement from the observed target closely tracks the
bootstrap perturbation:

$$
D(S_r,T) \approx D(T^{(r)},T).
$$

This design propagates uncertainty by changing the target CDF, rather than by
permitting progressively worse matches to one fixed target.

## 2. Statistical estimand

Let the eligible TE set contain `X` sites, with posterior age CDFs
`C_1(a),...,C_X(a)` evaluated on the exact analysis grid. The observed target
is

$$
T(a)=\frac{1}{X}\sum_{i=1}^{X}C_i(a).
$$

For bootstrap replicate `r`, draw multinomial counts

$$
(N_1^{(r)},\ldots,N_X^{(r)})
\sim \operatorname{Multinomial}
\left(X;\frac1X,\ldots,\frac1X\right),
$$

and construct

$$
T^{(r)}(a)
=\frac1X\sum_{i=1}^{X}N_i^{(r)}C_i(a).
$$

Each bootstrap target remains a complete CDF, not merely a scalar distance or
percentile. It therefore carries the direction and age-bin structure of that
bootstrap perturbation.

For a SNP set `A` of size `X`, let

$$
S_A(a)=\frac1X\sum_{i\in A}C_i(a).
$$

The optimizer seeks a set

$$
A_r^* \approx \arg\min_{A\subset\mathcal C,\ |A|=X}
D(S_A,T^{(r)}),
$$

where `C` is the declared eligible candidate-SNP universe after excluding the
target TE rows.

The discrete optimization is not expected to find a provable global optimum.
The output must therefore be described as the best state found under a
declared initialization, proposal schedule, seed, and compute budget.

## 3. Pilot evidence

The design was tested on the RNA in-gene target using the two available
posterior ARGs:

- 4,072 requested TE sites;
- 4,061 eligible TE sites;
- 376,813 requested candidate SNPs;
- 373,647 eligible candidate SNPs;
- 10,000 TE bootstrap distances; and
- a 1,000-generation exact analysis grid.

Twenty bootstrap targets were optimized for 15 exact improvement epochs. Each
optimization started from the closest of the 100 existing hard-q50 matched
sets. A proposed SNP swap was accepted only when it reduced exact-grid W1 to
that replicate's bootstrap target. The best state was retained.

| Quantity | Median W1 | Mean W1 | Range |
|---|---:|---:|---:|
| Bootstrap TE to observed TE | 1,276 | 1,484 | 536-5,038 |
| Initial SNP set to bootstrap TE | 2,346 | 2,555 | 1,774-6,170 |
| Optimized SNP set to bootstrap TE | 216 | 230 | 95-475 |
| Optimized SNP set to observed TE | 1,426 | 1,578 | 670-4,971 |

The optimized SNP-to-observed distances tracked their corresponding bootstrap
distances closely:

$$
\operatorname{cor}
\left(D(T^{(r)},T),D(S_r,T)\right)=0.989.
$$

Additional pilot diagnostics were:

- median absolute difference between the two distances: 92 generations;
- median relative absolute difference: 6.2%;
- median matching-error ratio: 0.131;
- mean matching-error ratio: 0.187;
- matching-error-ratio range: 0.035-0.502;
- 19 of 20 replicates passed a strict ratio `< 0.5` after 15 epochs;
- mean reduction from initial to final matching error: 90.5%;
- median membership replacement from the initializing set: 32.6%; and
- median pairwise SNP sharing among the 20 optimized sets: 3.2%.

The single ratio failure had a small bootstrap displacement of 536 generations
and a matching error of 269 generations (`R=0.502`). It was still improving at
the fixed 15-epoch stopping point. This illustrates why both relative and
absolute error must be reported.

These results are a two-posterior-ARG pilot, not a production calibration.
Production documentation must include the full best-distance trace for every
pilot replicate, not only its final ratio, because the traces demonstrate that
the fixed 15-epoch pilot budget was arbitrary and do not establish convergence.

## 4. Why threshold sampling is not retained

The existing hard-q50 sampler intentionally performs a constrained random walk
after construction. Because the number of possible SNP sets grows rapidly with
distance, saved sets accumulate immediately inside the outer threshold. In the
RNA in-gene pilot, the 100 hard-q50 controls occupied bootstrap percentiles
48.5%-50.0%, with an exact-distance standard deviation of only 7.5 generations.

Applying a survival-probability weight

$$
w(D)=1-\widehat F_{\mathrm{boot}}(D)
$$

did not solve this problem. Direct acceptance became sticky in the upper tail;
an uncapped Metropolis ratio became an unconstrained walk beyond the empirical
bootstrap maximum; and a q99-capped version accumulated at q99. The q99 cap
increased median W1 from about 1,875 to 4,889 while improving adjacent-set
membership replacement by only about one percentage point.

The bootstrap-target design therefore saves the optimized best state. It must
not follow optimization with a constrained random walk that deliberately moves
away from the assigned bootstrap target.

## 5. Production workflow

### 5.1 Build bootstrap targets

1. Resolve the TE positions once against the immutable age store.
2. Preserve the exact eligible TE rows and their complete posterior age CDFs.
3. Generate exactly one bootstrap target CDF for every requested matched set.
4. Assign stable replicate IDs and deterministically derived seeds.
5. Save the multinomial counts, or an equivalent reproducible representation,
   rather than saving only bootstrap distances.
6. Retain every prespecified bootstrap replicate, including upper-tail targets.
   Difficult targets must not be silently dropped or redrawn.

The initial production design should use 100 prespecified bootstrap targets,
one per matched SNP set. The iid multinomial bootstrap assumes exchangeable,
independent TE-site contributions, but nearby TEs inferred from a shared ARG
can have correlated age CDFs. Bootstrap design is therefore a production
precondition, not an optional refinement. Before inferential use, quantify
spatial autocorrelation in TE age-CDF summaries and select a documented
genomic-block bootstrap when iid exchangeability is not supported. Showing
that optimized controls reproduce the selected bootstrap distribution cannot
validate the bootstrap itself.

### 5.2 Initialize each optimization

Initialization affects a finite-budget discrete optimizer. Use a declared seed
library rather than one arbitrary starting state.

Recommended first implementation:

1. retain a library of diverse, valid sets from fixed-target construction;
2. calculate each seed set's exact W1 to the bootstrap target;
3. select several promising and membership-diverse starts; and
4. run independent optimization restarts from those starts.

At least one restart should use a seed not selected solely for minimum W1, so
that the procedure can diagnose initialization-induced local optima. The number
of minimum-W1 and diagnostic diversity starts, their compute budgets, and the
selection rule must be fixed before examining downstream Phi-SFS results.

### 5.3 Exact improvement-only optimization

For each restart:

1. cache exact-grid CDFs for the selected SNP rows;
2. propose a selected slot and an unselected candidate SNP;
3. update the aggregate CDF incrementally;
4. calculate exact-grid W1 to the assigned bootstrap target;
5. accept only strict improvements, subject to a floating-point tolerance;
6. periodically recompute the aggregate CDF from selected rows to prevent
   incremental numerical drift; and
7. retain the best certified state ever visited.

The first implementation may use randomized greedy swaps, as in the pilot.
Proposal symmetry is not required because this stage is optimization rather
than MCMC sampling. The output must not be described as a posterior or uniform
sample of matched SNP sets.

The candidate universe, target exclusions, no-duplicate rule, age-CDF
evaluation, and exact W1 implementation must remain identical across all
replicates.

### 5.4 Convergence and stopping

Do not stop merely because a state crosses a tolerance. Stop according to an
optimization convergence rule. Record the complete best-distance trace.

The initial rule should require all of:

1. a minimum number of exact proposal epochs;
2. no improvement exceeding a declared material threshold for `K` consecutive
   epochs, with the threshold scaled relative to `B_r` rather than expressed
   only as one absolute number of generations;
3. a maximum proposal/epoch budget; and
4. exact certification of the best state from its row indices.

An epoch should have a declared proposal count, such as one proposal per set
member, but convergence must be based on achieved improvement rather than the
number of accepted swaps. Production values for the material threshold, `K`,
and maximum budget should be calibrated with traces from more than the two-ARG
pilot.

### 5.5 Replicate QC

For replicate `r`, define

$$
B_r=D(T^{(r)},T),
\qquad
E_r=D(S_r,T^{(r)}),
\qquad
O_r=D(S_r,T),
$$

and the relative matching-error ratio

$$
R_r=\frac{E_r}{B_r}.
$$

The pilot supports an initial provisional criterion

$$
R_r<0.5,
$$

combined with an absolute error criterion such as

$$
E_r\le500\text{ generations}.
$$

These values are provisional and must be recalibrated with the full posterior
ARG collection. For bootstrap targets with very small `B_r`, the ratio is
unstable; the report must show both `E_r` and `R_r`, and the final QC rule may
need a small-distance exception stated in advance.

`E_r` and `R_r` are optimizer diagnostics: they test whether the finite search
found a sufficiently close representation of its assigned target. Because the
optimizer directly minimizes `E_r`, passing these thresholds is not independent
evidence of biological match quality and becomes nearly automatic for a
sufficiently converged search. Scientific validation must instead emphasize
the prespecified distributional concordance of `O_r` with `B_r`, including
center, tails, signed differences, and replicate-level association.

QC failure triggers additional deterministic optimization restarts or a larger
compute budget for the same bootstrap target. It must not trigger replacement
with a new bootstrap draw. If the prespecified maximum effort is exhausted,
retain the failed replicate and mark it failed; do not silently omit it.

The triangle inequality must be checked numerically:

$$
|B_r-E_r|\le O_r\le B_r+E_r.
$$

All three distances must use the same fixed age grid, endpoint convention,
normalization, and W1 implementation. The inequality is not a valid software
check if any of those differ among the three calculations.

### 5.6 Select among successful restarts

Choosing only the minimum-W1 restart may reduce membership diversity and make
outputs sensitive to a small subset of candidates. The first implementation
should save all restart diagnostics. Two defensible selection rules should be
compared before production:

1. choose the best-W1 state; or
2. among states passing QC, select deterministically from a near-optimal band
   while favoring membership diversity.

Any diversity-aware rule must be specified without using Phi-SFS or another
downstream result, which would bias the scientific statistic.

Minimum-W1 selection can nevertheless affect the SFS indirectly if SNPs that
repair particular age-CDF bins have unusual allele frequencies. Before
production, test the association between a candidate's W1-repair utility and
its derived-frequency contribution. Compare the SFS of selected best-W1 states
with the SFS of other QC-passing near-optimal restarts. This diagnostic and its
decision rule must be specified before using Phi-SFS results; merely avoiding
direct selection on Phi-SFS is not sufficient.

## 6. Output contract

### 6.1 Bootstrap-target artifacts

Save:

- replicate ID and seed;
- multinomial bootstrap counts or sampled TE row indices;
- bootstrap target CDF;
- `B_r`, its empirical bootstrap percentile, and the observed target CDF
  identity; and
- target/store provenance.

### 6.2 Optimization artifacts

For every restart, save:

- bootstrap replicate ID;
- initialization source and seed;
- initial SNP rows and initial W1;
- proposal and accepted-swap counts by epoch;
- best-distance trace;
- best SNP row indices and certified CDF;
- final `E_r`, `O_r`, `R_r`, and QC status;
- termination reason and runtime; and
- exact algorithm/configuration version.

### 6.3 Published matched-set bundle

Preserve the existing aligned array conventions where possible:

- `row_indices.npy`, shaped replicates by target-set size;
- `cdfs.npy` for the selected best states;
- `bootstrap_target_cdfs.npy`;
- `bootstrap_to_observed_w1.npy` (`B_r`);
- `match_to_bootstrap_w1.npy` (`E_r`);
- `match_to_observed_w1.npy` (`O_r`);
- `matching_error_ratio.npy` (`R_r`);
- replicate IDs, restart identities, and QC status arrays;
- position/chromosome arrays; and
- complete metadata and reuse diagnostics.

Publication must be atomic and must reject incomplete or provenance-mismatched
restart bundles.

## 7. Relationship to Phi-SFS

Replicate `r` defines a pair consisting of bootstrap age target `T^(r)` and
matched SNP set `S_r`. Preserve that pairing through Phi-SFS analysis.

The initial analysis should propagate uncertainty in age matching only:

- use `T^(r)` to choose the matched SNP set;
- calculate the SNP SFS from `S_r`; and
- compare it with the full observed TE SFS using the existing Phi-SFS
definition.

The prespecified Phi-SFS estimand is the distribution, over bootstrap-age
replicates, of the non-overlap between each optimized SNP-set SFS and the one
full observed TE SFS. It measures how the matched-control comparison changes
when TE age-CDF uncertainty is propagated while the observed TE SFS is held
fixed. It is not a bootstrap confidence distribution for the TE SFS and is not
by itself a p-value.

Do not automatically calculate the TE SFS from the resampled TE rows. Doing so
would additionally propagate finite-TE-set SFS uncertainty and would answer a
broader question. A joint age-and-SFS bootstrap can be added later as a
separately named sensitivity analysis.

Report Phi-SFS together with `B_r`, `E_r`, `O_r`, and `R_r`. Check whether
Phi-SFS is associated with residual matching error or bootstrap distance. A
strong association would mean that age-match quality remains a relevant
downstream nuisance variable.

The 100 optimized sets are bootstrap-linked control replicates, not posterior
draws from a probabilistic model over SNP sets. Their inferential interpretation
depends on the validity of the TE bootstrap and on optimization QC.

## 8. Diversity and bias diagnostics

For the published sets, report:

- unique control SNPs across all replicates;
- maximum and distribution of SNP reuse counts;
- pairwise and within-seed membership overlap;
- replacement from each initializing set;
- number of distinct initial seeds selected;
- chromosome and other required matching-covariate balance;
- W1 and QC metrics by initialization source; and
- sensitivity to independent restart seeds.

Estimate an effective replicate count for Phi-SFS from the observed
cross-replicate dependence induced by SNP reuse and shared initial seeds. Reuse
counts alone are descriptive and do not establish that 100 optimized controls
provide 100 independent pieces of information. The precise effective-N method
must be declared with the production analysis, with a sensitivity analysis if
the dependence model is approximate.

Optimization can repeatedly select SNPs that are unusually effective at
repairing particular CDF bins. High reuse is not automatically invalid, but it
reduces effective diversity and can couple Phi-SFS values across replicates.

## 9. Validation tests

### 9.1 Bootstrap correctness

- Multinomial counts sum to `X` for every replicate.
- Reconstructed bootstrap CDFs match saved CDFs exactly within tolerance.
- Repeating with the same seed is deterministic.
- The saved `B_r` values reproduce the target bootstrap-distance distribution.
- No bootstrap replicate is redrawn because it is difficult to optimize.

### 9.2 Optimizer correctness

- Every accepted swap strictly improves exact W1 within tolerance.
- Best-so-far W1 is monotonically nonincreasing.
- Incremental CDF updates agree with complete recomputation.
- Selected rows remain unique, eligible, in range, and outside the TE target.
- Saved CDFs and all three distances recompute from row indices.
- Different initialization seeds can reach distinct valid local optima.
- Fixed inputs and seeds reproduce identical outputs.

### 9.3 QC and propagation

- `R_r`, absolute error, and triangle-inequality checks are correct on hand
  fixtures.
- All three distances use identical age grids, endpoint conventions,
  normalization, and W1 code paths.
- QC failures retain their original bootstrap target.
- Published output contains exactly one status record per prespecified
  bootstrap replicate.
- On representative data, compare distributions of `B_r` and `O_r`, their
  correlation, signed and absolute differences, and tail behavior.
- Verify that upper-tail bootstrap targets are neither omitted nor
  systematically assigned worse relative matching errors.
- Treat `E_r` and `R_r` as convergence diagnostics and test scientific
  validity through the prespecified concordance of `O_r` with `B_r`.

### 9.4 Regression comparison

Compare against the current hard-q50 workflow on:

- distribution of SNP-to-observed W1;
- membership diversity and reuse;
- runtime and memory;
- failure/restart rate;
- Phi-SFS distribution and effective replicate count; and
- association between Phi-SFS and age-match diagnostics.
- association between candidate W1-repair utility and derived-frequency/SFS
  contribution, plus comparison of best and near-optimal QC-passing restarts.

## 10. Production acceptance gates

Before replacing the existing sampler:

1. rerun the pilot with the full posterior ARG collection;
2. test spatial dependence among TE age-CDF contributions, select and document
   an iid or genomic-block bootstrap accordingly, and generate at least 100
   prespecified bootstrap targets;
3. calibrate convergence from best-distance traces rather than fixing 15
   epochs by convention;
4. demonstrate a high prespecified QC pass rate without redrawing targets;
5. show that `O_r` reproduces `B_r` across the center and tails of the
   bootstrap distribution;
6. quantify sensitivity to initialization and restart selection;
7. show acceptable SNP reuse, effective diversity, and a declared effective
   replicate count for Phi-SFS;
8. validate all outputs by recomputation from canonical row indices;
9. define the handling of any unresolved QC failures before looking at
   Phi-SFS; and
10. retain the current hard-q50 workflow as a reproducible sensitivity
    analysis during the transition.

## 11. Recommended implementation sequence

1. Extend target construction to materialize reproducible bootstrap target
   identities and CDFs, not only their distances.
2. Implement an exact improvement-only optimizer that saves best-so-far states
   and full convergence traces.
3. Add multiple deterministic restarts and initialization provenance.
4. Implement absolute and relative QC without redraw-on-failure.
5. Gather restart bundles atomically into one replicate-aligned output.
6. Add diversity, reuse, propagation, and tail diagnostics.
7. Validate on the RNA in-gene target with the full posterior collection.
8. Run Phi-SFS while preserving bootstrap-replicate pairing.
9. Compare scientific conclusions with the existing hard-q50 sensitivity
   analysis before making the new workflow the production default.

## 12. Implementation status

The first local, resumable implementation is provided by
`bootstrap_target_matcher.py`. It implements:

- deterministic iid multinomial bootstrap counts and complete target CDFs,
  accumulated in float64 over float32-stored per-site rows;
- coarse-grid improvement-only swap screening with exact-grid certification of
  every recorded distance, matching `swap_control_sampler`'s two-tier device
  (`--search-bin-width`, default 20,000 generations);
- two closest plus one diagnostic diversity restart by default;
- relative material-improvement convergence with minimum/maximum epoch and
  patience controls;
- complete restart traces and best-state certification;
- relative and absolute optimizer QC without bootstrap redraw;
- all three W1 distances, ratios, and triangle-inequality validation;
- provenance-locked per-replicate resume bundles;
- minimum-W1 selection across prespecified restarts;
- atomic publication of replicate, restart, position, and reuse artifacts;
- compatibility with the existing hard-q50 bundle as the seed library and
  sensitivity workflow; and
- a published bundle that `phi_sfs.py` can read (section 7): it records the
  project-wide four-array `target_digest`, including the acceptance threshold,
  and publishes `replicate_id.npy` as its per-replicate identifier.

### Identifiers are replicate-scoped, deliberately

This stage does **not** publish `chain_index.npy` or `sample_index.npy`. Those
arrays exist so that a consumer can account for correlation among the ten
states saved from one swap chain. Bootstrap replicates have no such structure,
so emitting those columns would assert a within-chain correlation that does not
exist. `phi_sfs.py` selects identifier arrays from the bundle's
`schema_version` instead. The dependence that does matter here is shared
control SNPs, reported through `reuse_row_indices.npy` and `reuse_counts.npy`.

The current CLI is a single-node implementation. Its default command and
output contract are documented in `README.md`. Distributed array-task
launchers and durable cross-node gather are not yet implemented, so this stage
is labelled experimental and pilot-only in the README and is not part of the
supported production path.

### Deferred output fields

`restart_initial_rows` is not published. It is recoverable from
`restart_seed_indices.npy` together with the seed bundle identified by
`seed_sets` and `seed_sets_digest` in the metadata, so it is derivable rather
than lost. Everything else listed in section 6 is published, including
bootstrap and restart seeds, per-restart distances, ratios, QC, runtimes, and
per-epoch proposal counts.

The following remain production gates rather than implemented claims:

1. test spatial dependence and decide between iid and genomic-block bootstrap;
2. implement a block-bootstrap mode if iid exchangeability is unsupported;
3. calibrate convergence and QC using the full posterior ARG collection;
4. quantify W1-repair-utility association with derived-frequency/SFS
   contribution;
5. compare best-W1 with near-optimal diversity-aware restart selection;
6. estimate effective Phi-SFS replicate count from reuse dependence;
7. implement distributed restart execution and atomic gather for HPC; and
8. demonstrate center-and-tail concordance of `O_r` with `B_r` on production
   targets before replacing hard q50.
