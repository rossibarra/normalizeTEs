# Age-matched control sampler: swap-chain implementation plan

## 1. Decision and scope

This plan supersedes `GLOBAL_QUANTILE_SAMPLER_IMPLEMENTATION_PLAN.md` and the
earlier quota-fitting/index proposal. The production sampler will use the
canonical SNP interval store directly and construct matched control sets by
stochastic one-for-one swaps.

The first release must generate **100 control sets per TE dataset**, using
**four independent chains with 25 saved sets per chain**.

Confirmed defaults:

- controls per set equal the number of eligible target TEs;
- sampling is without replacement within a set;
- controls may recur across different saved sets and chains;
- target TEs are excluded from the control pool;
- greedy construction stops when the exact Wasserstein criterion is first met;
- burn-in continues until at least 50% of the first feasible set has been
  replaced;
- consecutive saved sets within a chain must differ by at least 25% of their
  members;
- every saved set is certified on the exact 1,000-generation analysis grid;
- the canonical `snp-age-interval-v1` store remains the source of truth; and
- no global alias table, dense candidate-by-bin matrix, or Monte Carlo bucket
  index is part of version 1.

This is an age-matching algorithm. It must report other genomic composition
diagnostics, but it must not silently introduce chromosome, gene, frequency,
distance, LD, or annotation constraints. Any such constraint is a separately
declared scientific design choice.

## 2. Why this replaces the indexed proposals

The original global-quantile design required approximately 175 GiB of alias
arrays and tens of billions of table cells. A later design reduced storage but
still required a global proposal index and quota calibration. Both approaches
optimize proposal generation before establishing that an index is necessary.

Direct uniform rejection is not viable for atypical targets. On the two-draw
real-data pilot, an in-gene target with 4,061 eligible TEs had an estimated 95%
bootstrap threshold of 3,573 generations, while an ordinary random control set
had Wasserstein distance 650,651 generations. For the 35,373-TE target, none of
1,000 independent uniform proposals passed; the best distance was 599,519
against a threshold of 2,564.

One-for-one stochastic improvement, however, converged quickly:

| Target | Eligible TEs | First accepted epoch | Pilot optimization time |
| --- | ---: | ---: | ---: |
| all structural TEs | 35,373 | 4 | about 9 s |
| in-gene structural TEs | 4,061 | 5 | about 1.2 s |
| `crap1` pilot | 496 | 2 | 0.07 s |
| `crap2` pilot | 494 | 1 | 0.03 s |

These measurements used two real SINGER draws and a coarse 20,000-generation
search grid with exact 1,000-generation scoring after each epoch. They prove
local feasibility, not 75-draw production performance. Production gates below
must confirm the result using the complete interval store.

## 3. Statistical objects

### 3.1 SNP posterior CDF

For SNP `i`, each usable interval `[L_ir, U_ir]` is uniform and usable intervals
receive equal weight, matching the existing interval-store behavior:

```text
F_i(t) = (1 / m_i) * sum_r F_uniform(t; L_ir, U_ir).
```

For a target containing `n` eligible TE SNPs:

```text
T(t) = (1 / n) * sum_i_in_target F_i(t).
```

For a control set `S` of the same size:

```text
C_S(t) = (1 / n) * sum_i_in_S F_i(t).
```

### 3.2 Distance and threshold

The exact acceptance statistic remains the existing one-dimensional
Wasserstein calculation on the 1,000-generation target grid:

```text
W1(C_S, T) = sum_b |C_S(b) - T(b)| * grid_width_b.
```

The initial threshold remains the conservative observed 95th percentile of
target-SNP bootstrap distances, computed separately for each TE dataset at its
actual eligible size. Bootstrap replicates resample target SNP rows with
replacement and retain each row's complete posterior CDF.

Version 1 must use the same CDF convention, grid, weighting, and empirical
quantile method for target construction, bootstrap distances, and saved-set
certification. Metadata records the bootstrap replicate count, seed, quantile,
and full distance summary.

The bootstrap threshold is a matching tolerance, not evidence that saved
control sets are independent or uniformly sampled. Those are separate
diagnostic questions.

### 3.3 Incremental swap identity

If selected row `old` is replaced by unselected row `new`, the aggregate CDF
updates exactly as

```text
C_new = C_current + (F_new - F_old) / n.
```

This identity is used on both the coarse search grid and exact analysis grid.
It avoids reconstructing a set aggregate from all `n` rows for each proposal.

The Wasserstein distance still costs `O(B)` per evaluated swap for `B` grid
points. Version 1 may batch CDF construction for proposed rows, but it must not
materialize full-grid CDFs for the entire 23.5-million-row pool.

## 4. Candidate universe

For each TE dataset:

1. open and validate the canonical interval store;
2. resolve target chromosome/position pairs exactly;
3. apply the declared missing/ineligible target policy;
4. form the control universe from the declared eligible store rows or an
   explicit eligible control-row list;
5. exclude every resolved target row;
6. reject duplicate target or candidate rows;
7. verify `candidate_count >= target_size`; and
8. record target-resolution and candidate-universe digests.

The default missing-target policy for production should be `error`. A `drop`
mode is useful for audits and pilots, but every excluded coordinate and reason
must be written to the result metadata.

## 5. Two-phase algorithm

### 5.1 Independent chains and deterministic seeds

Run four chains. Derive each chain seed deterministically from:

```text
global seed + target digest + chain index + algorithm version
```

Use a stable documented derivation rather than Python's process-randomized
`hash()`. Store the derived seeds.

Each chain starts from an independently drawn uniform set of `n` distinct
candidate rows. Do not seed later chains from an earlier accepted set.

### 5.2 Phase A: greedy construction

The construction phase finds a feasible starting state; it is not counted
among the 100 saved sets.

For each epoch:

1. randomly permute all selected slots;
2. draw a batch of candidate replacement rows;
3. skip any replacement already present in the current set;
4. evaluate `C_trial = C + (F_new - F_old) / n` on the coarse grid;
5. accept the swap only if coarse-grid Wasserstein distance strictly improves;
6. update membership, selected rows, aggregate CDF, and diagnostics; and
7. compute the exact full-grid distance at the end of the epoch.

Near the threshold, exact scoring should occur more frequently so construction
stops close to the first exact threshold crossing rather than overshooting to
an unnecessarily perfect match. A safe initial rule is:

- exact score after every epoch while exact distance exceeds twice the
  threshold;
- below twice the threshold, exact score after every configurable block of
  accepted swaps; and
- stop immediately after the first exact-certified state.

Coarse distance may guide construction but never certifies a set. The exact
first-passage set becomes the entry state for burn-in and is not written as one
of the 100 requested samples.

If no feasible state is found within the maximum epochs or proposal budget,
fail with the best exact distance, distance trajectory, acceptance trajectory,
and seed. Do not relax the threshold automatically.

### 5.3 Phase B: feasible-region walk

After construction, use stochastic one-for-one swaps to move within the
matched region. The default proposal chooses:

- one selected slot uniformly; and
- one row uniformly from the control universe, rejecting rows already selected.

For a mathematically exact constrained chain, accept a proposal if and only if
its **exact** distance is at or below the threshold. Because the proposal is
symmetric over equal-sized subsets, this indicator acceptance rule satisfies
detailed balance with a uniform distribution over the connected feasible
states.

Version 1 must not claim that saved sets are independent or uniformly sampled
unless empirical mixing and connectivity diagnostics support that claim.
Single-swap feasible states could mix slowly or form disconnected components.
The four independent chains and membership diagnostics are therefore required.

Two execution modes should be implemented explicitly:

1. `exact-chain`: every proposed swap is evaluated on the exact grid; this is
   the reference method and the only mode eligible for a uniform-feasible-set
   interpretation.
2. `screened-chain`: a coarse calculation rejects clearly invalid proposals,
   but every provisionally acceptable proposal is evaluated exactly before its
   state changes. Screening must have no false rejection if a uniform target is
   claimed; otherwise the output is labeled an optimization-derived matched
   ensemble rather than a uniform constrained sample.

The first implementation should favor correctness and measurement over an
unproven coarse-only chain.

### 5.4 Burn-in

Let `S_entry` be the exact first-passage state and `S_current` the chain state.
Burn-in ends only when all of the following hold:

```text
1 - |S_current intersect S_entry| / n >= 0.50
exact_W1(S_current, T) <= threshold
minimum accepted-swap count is satisfied
```

The 50% membership replacement is authoritative. Accepted-swap count is a
secondary guard because a slot can change multiple times and an earlier member
can re-enter the set.

As a rough planning approximation, 50% replacement requires approximately
`-n * log(0.5) = 0.693n` accepted swaps under well-mixed removal. The code must
measure actual membership replacement instead of assuming the approximation.

### 5.5 Saving 25 sets per chain

After burn-in, save the current state only after exact certification. For every
later saved set in the same chain, continue until:

```text
1 - |S_current intersect S_previous_saved| / n >= 0.25
exact_W1(S_current, T) <= threshold.
```

Thus consecutive saved states share at most 75% of their members. The default
25% replacement corresponds to approximately `-n * log(0.75) = 0.288n`
accepted swaps, but actual set intersection controls saving.

Save 25 states from each of four chains for exactly 100 output sets. Output
ordering is deterministic: chain index first, then within-chain sample index.

The 25% default is a starting point, not proof of independence. Gate 3 compares
10%, 25%, and 50% replacement using autocorrelation, effective sample size,
membership overlap, and the downstream statistic. Change the production
default only with documented evidence.

## 6. Efficient implementation

### 6.1 New modules and CLI

Add a focused implementation rather than extending the legacy stratified
sampler indefinitely:

```text
swap_control_sampler.py
sample_age_matched_controls.py
tests/test_swap_control_sampler.py
```

The CLI should accept at least:

```text
--store
--target-positions / --target
--candidate-rows / --all-eligible
--output
--sets 100
--chains 4
--sets-per-chain 25
--seed
--bootstrap-replicates
--acceptance-quantile 0.95
--analysis-bin-width 1000
--search-bin-width 20000
--burnin-replacement-fraction 0.50
--sample-replacement-fraction 0.25
--max-construction-epochs
--max-chain-proposals
--candidate-batch-rows
--progress-every
--checkpoint-every
--scratch-dir
```

Validate that `sets == chains * sets_per_chain` when all three are supplied.

### 6.2 CDF access and memory

Keep the interval store memory-mapped and read-only. Do not precompute a full
candidate-by-grid matrix.

Maintain for each active chain:

- selected canonical row indices;
- an efficient membership structure;
- current aggregate CDF on search and exact grids;
- the construction entry set and previous saved set;
- current exact and coarse distances; and
- RNG state and counters.

During greedy epochs, evaluate old and proposed row CDFs in blocks. A block may
hold float32 row CDFs on the search grid, while aggregate and Wasserstein sums
remain float64. Select block sizes from an explicit memory formula.

For the exact chain, benchmark these alternatives:

1. compute `F_old` and `F_new` from their small ragged interval records for each
   proposal;
2. cache exact CDFs only for currently selected rows plus a bounded LRU of
   recently proposed rows; and
3. batch exact candidate CDF calculation while preserving sequential state
   transitions.

Do not add a persistent all-candidate CDF cache without a separate storage and
precision gate.

### 6.3 Membership operations

The selected set is small relative to the candidate universe. Use:

- a row array indexed by selected slot;
- a hash set or equivalent for `O(1)` selected-membership tests; and
- optional row-to-slot mapping when needed for cache updates.

Every accepted swap updates these structures atomically. Tests must assert
their agreement after randomized sequences.

### 6.4 Checkpoint and resume

Checkpoint each chain independently and atomically. A checkpoint contains:

- schema and algorithm version;
- normalizeTE release version, Git commit, exact tag, description, and dirty
  checkout status;
- target/store/candidate digests;
- chain and derived seed;
- RNG bit-generator state;
- phase, epoch, proposal, and accepted-swap counters;
- selected rows;
- coarse and exact aggregate CDFs;
- construction entry and previous saved rows when applicable;
- saved sets completed by that chain; and
- current diagnostic summaries.

Resume only when all provenance and parameter fields match. Refuse stale or
partial checkpoints. Termination signals should request a checkpoint and clean
exit where safe.

## 7. Outputs and diagnostics

Publish one result directory atomically:

```text
matched_controls/
├── metadata.json
├── row_indices.npy          # int64, (100, n)
├── chromosomes.npy          # string, (100, n), or compact encoded equivalent
├── positions.npy            # native positions, (100, n)
├── wasserstein.npy          # float64, (100,)
├── chain_index.npy          # integer, (100,)
├── sample_index.npy         # integer, (100,)
├── diagnostics.csv
├── target_cdf.npy
├── age_bins.npy
└── checkpoints/             # absent or marked complete after final publication
```

Required per-chain and per-sample diagnostics:

- construction epochs, proposals, accepted swaps, and exact distance path;
- burn-in proposals, accepted swaps, acceptance rate, and achieved replacement;
- proposals and accepted swaps between saved samples;
- exact Wasserstein distance for every saved set;
- overlap with previous sample, chain entry, and other chain outputs;
- duplicate proposal count;
- candidate-CDF time, coarse-score time, and exact-score time;
- target and control counts by chromosome;
- per-control reuse across all 100 sets;
- within-chain autocorrelation of Wasserstein distance and relevant downstream
  statistics; and
- warnings when diversity or mixing gates are not met.

All 100 sets must contain exactly `n` unique rows and must pass the exact
threshold. Any failure prevents final publication.

## 8. Testing

### 8.1 Unit tests

- Incremental CDF update equals full aggregate reconstruction.
- Incremental Wasserstein score equals `wasserstein_1`.
- Accepted swaps update row arrays and membership structures consistently.
- Duplicate and target-row proposals are rejected without changing state.
- Greedy swaps never increase the chosen search-grid objective.
- Exact certification rejects coarse false positives.
- Burn-in uses actual set intersection, not accepted-swap count.
- Saving requires at least 25% replacement from the previous saved set.
- Four chains produce exactly 25 ordered outputs each.
- Every saved set is unique internally and below the exact threshold.
- Seeds reproduce rows, distances, checkpoints, and resume behavior.
- Provenance mismatch rejects checkpoint resume.
- Atomic output refuses overwrite and cleans incomplete staging.

### 8.2 Property tests

On randomized tiny interval stores:

- swap updates remain exact over long randomized sequences;
- a swap followed by its reverse restores the original aggregate;
- exact-chain proposal probabilities are symmetric;
- constrained-chain transitions never leave the feasible region;
- replacement fractions agree with direct set intersection; and
- control reuse across sets is allowed while within-set duplication is not.

### 8.3 Statistical tests

Use small systems where all equal-sized subsets can be enumerated:

- compare exact-chain state frequencies with the uniform distribution over the
  connected feasible component;
- identify deliberately disconnected feasible examples and verify diagnostics
  do not claim global uniformity;
- compare multiple independent chains;
- measure autocorrelation and effective sample size under different replacement
  fractions; and
- show that first-passage construction states are not treated as saved samples.

## 9. Benchmark and validation gates

### Gate 1: deterministic correctness

Pass all unit, property, and enumeration tests. Require exact agreement within
declared floating-point tolerances and bitwise reproducibility for fixed seeds
on the supported platform.

### Gate 2: two-draw real-data regression

Turn the four completed pilots into a reproducible benchmark command. Confirm:

- target resolution counts;
- bootstrap thresholds with fixed seeds;
- greedy convergence;
- exact/coarse distance agreement at checkpoints;
- bounded peak RSS;
- four-chain checkpoint/resume; and
- 100 exact-certified outputs for at least one small and one large target.

The exploratory timing numbers are reference observations, not hard regression
limits until the benchmark harness controls caches and hardware.

### Gate 3: diversity and thinning

For representative small, medium, and large targets, compare 10%, 25%, and 50%
between-sample replacement. Measure:

- membership overlap distributions;
- per-SNP inclusion/reuse distribution;
- Wasserstein autocorrelation and ESS;
- downstream-statistic autocorrelation and ESS;
- agreement across four independently initialized chains; and
- proposals and runtime per saved set.

Retain 50% burn-in and select the smallest between-sample replacement fraction
that satisfies the documented diversity target. The initial production default
is 25%.

### Gate 4: full 75-draw store pilot

Before broad production, run the complete workflow on the full interval store
for at least:

- one target near 500 SNPs;
- one target near 4,000 SNPs;
- one target near 35,000 SNPs;
- a strongly young target;
- a strongly old or long-tailed target; and
- a narrow or multimodal target.

Record construction time, chain acceptance rate, time to required membership
replacement, exact scoring time, RSS, interval-store I/O, checkpoint size, and
total time for 100 sets.

No-go conditions include failure to construct any chain, inability to reach
burn-in replacement within budget, severe disagreement among chains, exact
false certification, or resource use outside the cluster envelope.

### Gate 5: scientific composition audit

For target and saved controls, report chromosome composition and any available
predeclared covariates. Decide explicitly whether observed imbalance is
acceptable or requires a new constrained design. Do not retrofit constraints
silently after seeing downstream results.

### Gate 6: production rollout

Run a modest category batch first. Confirm atomic output, resumption, logging,
100-set completeness, exact thresholds, and aggregate filesystem load before
launching all TE datasets.

## 10. Implementation order

### Phase 1: pure swap primitives

- Incremental aggregate and Wasserstein updates.
- Membership and swap-state invariants.
- Replacement-fraction calculations.
- Deterministic seed derivation.
- Tiny enumeration and property tests.

### Phase 2: greedy constructor

- Blockwise search-grid candidate evaluation.
- Periodic exact certification and first-passage stopping.
- Progress, budgets, and failure diagnostics.
- Two-draw regression benchmark.

### Phase 3: exact constrained chain

- Symmetric proposals and exact feasibility acceptance.
- 50% burn-in replacement.
- 25% between-sample replacement.
- Four chains and 25 outputs per chain.
- Diversity and mixing diagnostics.

### Phase 4: checkpointed CLI and outputs

- Target/candidate resolution and provenance.
- Atomic checkpoint/resume.
- Atomic final publication.
- Complete metadata and diagnostics.

### Phase 5: production performance

- Candidate CDF batching and bounded caches.
- Full 75-draw Gate 4 benchmarks.
- SLURM resource template and batch-category runner only after per-category
  behavior is measured.

## 11. Definition of done

Version 1 is ready for production only when:

1. the interval store remains unchanged and is the only required large index;
2. target and candidate resolution is exact and provenance-checked;
3. four independently seeded chains each produce 25 saved sets;
4. first-passage construction states are excluded from the requested outputs;
5. burn-in replaces at least 50% of the entry set;
6. consecutive saved states replace at least 25% of members;
7. every saved set contains exactly `n` distinct non-target controls;
8. all 100 sets pass exact full-grid Wasserstein certification;
9. checkpoint/resume is exactly reproducible;
10. diversity, overlap, reuse, and chain diagnostics are retained;
11. representative full-store targets pass the performance and scientific
    gates; and
12. documentation does not claim independence or uniform global sampling beyond
    what the measured chain diagnostics support.
