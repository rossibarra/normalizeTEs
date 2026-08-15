# Age-matched control sampler: implemented fixed-sweep design

## 1. Decision and scope

The production sampler uses the canonical all-SNP interval store directly and
constructs matched control sets with stochastic one-for-one swaps. It replaces
the global-quantile index, quota-fitting, and stratified rejection designs.

For each TE dataset, the production output is:

```text
100 sets = 10 independently seeded chains × 10 saved states
```

Confirmed behavior:

- every set contains the same number of SNPs as the eligible target;
- sampling is without replacement within a set;
- controls may recur across sets and chains;
- target rows are excluded;
- `--all-eligible` or `--candidate-rows` must be chosen explicitly;
- construction finds the first exact-feasible state but does not save it;
- burn-in and thinning use fixed accepted-swap counts, not membership-crossing
  stopping times;
- one accepted-swap sweep means `ceil(target_size × sweep_count)` accepted
  swaps;
- the default is one sweep before the first save and one sweep between saves;
- membership replacement is measured and reported, not used to stop the walk;
- construction adaptively refines a plateaued coarse grid down to the exact
  target-grid width without changing the threshold;
- every saved set is certified on the exact 1,000-generation target grid; and
- no dense candidate index or target-specific candidate-weight matrix is
  required.

The output is a correlated Monte Carlo ensemble. Ten independent chains reduce
within-trajectory dependence, but the software must not describe the 100 states
as independent observations.

## 2. Statistical specification

### 2.1 SNP and set posterior CDFs

For SNP `i`, each usable interval `[L_ir, U_ir]` contributes a uniform
distribution and usable intervals have equal weight:

```text
F_i(t) = (1 / m_i) * sum_r F_uniform(t; L_ir, U_ir)
```

For target rows `A` and a control set `S`, both of size `n`:

```text
T(t)   = (1 / n) * sum_i_in_A F_i(t)
C_S(t) = (1 / n) * sum_i_in_S F_i(t)
```

### 2.2 Matching distance and tolerance

Saved sets use the existing discrete one-dimensional Wasserstein calculation:

```text
W1(C_S, T) = sum_b |C_S(b) - T(b)| * grid_width_b
```

The default tolerance is the median of 10,000 target-SNP bootstrap distances.
The quantile remains configurable for labeled sensitivity analyses. A
pre-specified `--acceptance-distance` in generations overrides the quantile
threshold while retaining the bootstrap distribution for context.

The tolerance is the matching-quality specification. The feasible walk accepts
uphill and downhill proposals without preferring smaller W1, provided the trial
remains inside a tiny numerical safety margin below the threshold. Saved states
therefore normally concentrate near the requested boundary.

### 2.3 Incremental swap identity

Replacing selected row `old` with unselected row `new` updates the aggregate:

```text
C_new = C_current + (F_new - F_old) / n
```

The chain keeps selected-row CDFs in float64. At every save, the aggregate is
recomputed through the interval-store aggregation kernel, certified, and used
to resynchronize incremental state.

## 3. Candidate universe

For every target:

1. resolve chromosome and 1-based positions against the interval-store catalog;
2. apply the declared missing/ineligible policy;
3. choose all eligible rows or an explicit candidate-row array;
4. reject duplicate and ineligible candidates;
5. remove target rows;
6. require strictly more candidates than the target size, ensuring at least one
   unselected proposal exists; and
7. require and record the target, candidate, full store-content, store-catalog,
   and software digests.

The default missing-target policy is `error`. A `drop` run records every
excluded coordinate and reason.

## 4. Chain algorithm

### 4.1 Deterministic independent seeds

Each chain seed is derived by SHA-256 from:

```text
global seed + target digest + chain index + algorithm version
```

Each chain begins from its own uniform sample of `n` distinct candidate rows.

### 4.2 Greedy construction

Construction finds a feasible starting component:

1. randomly permute selected slots;
2. draw candidate replacement rows;
3. reject already selected rows;
4. evaluate the incremental aggregate on the coarse search grid;
5. accept only strict coarse-grid improvements;
6. certify on the exact grid after each epoch and more often near the
   threshold;
7. when a zero-acceptance epoch remains above threshold, halve the search-grid
   width and recompute construction state, down to exact resolution;
8. fail after three consecutive zero-acceptance epochs at exact resolution;
   and
9. stop at the first exact-feasible state.

Construction failure reports the trajectory and seed and never relaxes the
tolerance automatically.

### 4.3 Exact feasible-region walk

The implemented walk has one mode: every proposal is evaluated on the exact
grid. A proposal selects one occupied slot uniformly and one unselected
candidate uniformly. It is accepted whenever its exact W1 is within the walk
tolerance, whether it improves or worsens the current distance.

This proposal plus indicator acceptance defines a symmetric constrained walk
within a connected feasible component. Connectivity is not assumed; ten
independent starts and between-chain diagnostics are required.

### 4.4 Fixed-sweep burn-in and thinning

First-crossing membership rules are not used because states observed at a
path-dependent stopping time need not have the stationary distribution and
crossings pin replacement at the minimum allowed value.

The default burn-in performs exactly one accepted-swap sweep after construction.
Each later save follows exactly one additional accepted-swap sweep from the
previous save. Proposals rejected by the W1 boundary do not count toward a
sweep. The global proposal budget still applies.

The walk maintains reference-set overlap in `O(1)` per accepted swap. It reports
actual replacement from construction and the preceding saved state, but it does
not repeatedly sort/intersect `n` rows inside the proposal loop.

One sweep is an initial scientifically explicit default, not proof of
independence. Full-store pilots must measure autocorrelation of the actual
downstream statistic and may increase sweep counts.

## 5. Distributed execution and durable state

Production uses three SLURM stages:

1. one target task per TE dataset;
2. one chain task per `(target, chain index)` pair; and
3. one gather task per target, dependent on all chain tasks.

Each chain task stages the one canonical interval store to `$TMPDIR`, writes all
active checkpoints and caches under `$TMPDIR`, and atomically publishes one
compressed bundle to `OUTPUT.chains/chain-NNN.npz`. The bundle includes:

- chain index and deterministic seed;
- ten row-index sets;
- ten exact CDFs and W1 distances;
- construction and walk diagnostics; and
- a canonical run identity containing store, target, candidate, software,
  seed, chain-count, and algorithm configuration fields.

After publication, the chain reloads the durable bundle and validates its
identity, re-derived seed, eligibility, target exclusion, uniqueness,
distances, and all ten CDFs recomputed from row indices. Only then is scratch
work removed. Publication uses a unique temporary name and an atomic
no-overwrite claim, so an overlapping retry cannot replace a completed bundle.

Interrupted-chain checkpoints are diagnostic and node-local; an interrupted
chain restarts deterministically. Completed durable chain bundles are the unit
of cross-job resume.

The gather task validates all expected bundles, assembles the final directory
under `$TMPDIR`, copies it to a temporary sibling on the destination filesystem,
checks `complete: true`, and atomically renames it to the final output. No
long-lived `.OUTPUT.work` directory is written on Quobyte.

## 6. Output and diagnostics

The final output includes row and native-position matrices, exact CDFs and W1
distances, chain/sample indices, diagnostics, target CDF/grid copies, reuse
counts, and metadata.

Metadata records per-chain:

- construction trajectory and first-feasible W1;
- fixed-sweep proposal and accepted-swap counts;
- actual adjacent membership replacement;
- W1 lag-one autocorrelation; and
- a clearly labeled membership-overlap AR(1) ESS heuristic.

The heuristic does not substitute for autocorrelation of the scientific
downstream statistic.

## 7. Correctness requirements

The implementation must:

- fail promptly when the candidate universe has no unselected row;
- avoid full set intersection on each proposal;
- accept uphill-but-feasible moves;
- resynchronize incremental and independently aggregated CDFs at saves;
- reject a bundle whose stored CDF is inconsistent with its row indices;
- reject a bundle whose seed differs from deterministic derivation;
- reject distributed work without a matching full store-content digest;
- reject truncated or zero-byte outputs even when metadata says complete;
- verify SLURM's actual array size against the declared task count; and
- publish chain files without overwrite and final directories atomically on
  their destination filesystems.

## 8. Validation plan

### Unit and property tests

- incremental CDF updates equal complete recomputation;
- deterministic seeds are stable and chain-specific;
- exactly the configured accepted-swap count occurs per phase;
- membership diagnostics equal direct set intersections;
- candidate count equal to target size fails rather than hanging;
- coarse construction plateaus refine to exact resolution or fail clearly;
- later-set stored-row/CDF mismatches, wrong seeds, mixed identities, missing
  bundles, and truncated outputs are rejected;
- non-integer sweeps round up to the declared accepted-swap count;
- overlap diagnostics equal direct intersections after saved transitions;
- an uphill feasible proposal is accepted; and
- scratch-to-durable publication leaves no partial visible output.

### Statistical tests

- enumerate tiny feasible components and compare long-run visitation with the
  expected constrained walk behavior;
- compare multiple independent starts for component overlap;
- report membership, W1, and downstream-statistic autocorrelation and ESS; and
- measure sensitivity to 0.5, 1, and 2 accepted-swap sweeps.

### Production gates

Before full rollout on the 75-draw store:

1. run targets near 500, 4,000, and 35,000 SNPs;
2. confirm ten complete bundles and 100 exact-certified outputs per target;
3. measure per-chain RSS, time, acceptance, and staged-store I/O;
4. confirm q50 and any absolute tolerance on the full posterior;
5. inspect membership, reuse, chromosome composition, and between-chain
   agreement;
6. inspect adaptive-refinement histories and any exact-grid plateau; and
7. establish an adequate sweep count using the actual downstream statistic.

The version-2.1 `10 × 10` workflow has passed a complete local pilot for the
4,061-SNP in-gene target on the available two-draw store, including all-set CDF,
derived-seed, and full store-content validation. A deterministic synthetic test
also confirms adaptive recovery from a coarse-grid plateau. These tests do not
validate mixing for the 75-draw store or for a scientific downstream statistic.

## 9. Definition of done

The sampler is ready for production when:

1. the interval store is the only required large reusable index;
2. target and candidate universes are explicit and provenance-checked;
3. ten independent chains each publish ten validated states;
4. construction states are not included among outputs;
5. fixed accepted-swap sweeps control burn-in and thinning;
6. all sets contain exactly `n` distinct eligible non-target controls;
7. all 100 sets pass exact W1 certification;
8. bundle and final-output publication survive retry without silent truncation;
9. diversity, overlap, reuse, and chain diagnostics are retained;
10. representative full-store targets pass resource and scientific gates; and
11. documentation does not call the 100 states independent without measured
    support from the downstream statistic.
