# Global-quantile control sampler implementation plan

## 1. Purpose and confirmed decisions

This plan replaces per-target scans and per-proposal full-pool probability
operations with a reusable sampling index for many SNP test sets.

The following decisions are inputs to the design:

1. The intended control pool is approximately 23.5 million SNPs. It is not a
   smaller synonymous-only subset.
2. Ages and posterior uncertainty must remain available for every control SNP
   because many different TE and non-TE SNP sets will be tested later.
3. The compact interval store remains the canonical representation of those
   ages. A derived sampling index must be reproducible from it and must not
   replace it.
4. Sampling will use one fixed global set of age boundaries rather than
   target-specific quantile boundaries.
5. The initial requested quantile step is 0.001, giving at most 1,000 global
   intervals before repeated boundaries are compressed.
6. Each target uses only fixed intervals with nonzero integer quotas. It does
   not invent new category-specific boundaries.
7. Every proposed matched set is still checked against the target's complete
   age CDF with the target-specific Wasserstein threshold.
8. Global alias tables are an acceleration layer. Exact target construction,
   arbitrary test-set summaries, and final proposal scoring continue to use
   the canonical interval data.

The method change is confined to the proposal distribution. The final
acceptance criterion remains the full-CDF Wasserstein test, but the new method
must still be validated empirically against the existing target-specific
stratification procedure.

## 2. Goals

- Pay the cost of scanning all control SNP posteriors once, not once per test
  category.
- Make proposal generation scale primarily with the requested matched-set
  size rather than `candidate_count * number_of_strata`.
- Retain the exact interval posterior for all SNPs.
- Support arbitrary future SNP test sets without rebuilding the global index.
- Preserve sampling without replacement within each matched set.
- Preserve reuse of controls across independently generated matched sets.
- Provide progress, timing, checkpoint, provenance, and integrity information.
- Keep the persistent derived index near 188 GB for 23.5 million candidates
  and 1,000 bins, excluding the existing interval store.
- Keep all heavy work on SLURM compute nodes and use node-local scratch for
  transient matrices.

## 3. Non-goals

- Do not materialize the original fine, regular age grid for every SNP.
- Do not replace or delete the interval store after building the index.
- Do not silently remove low-mass target tails.
- Do not silently relax a Wasserstein threshold to obtain enough matches.
- Do not assume that 0.001 global mass implies 23,500 distinct usable SNPs in
  every bin. It implies approximately 23,500 candidate-equivalents of summed
  posterior mass before boundary discretization.
- Do not use the current Python per-row `_batch_cdf` path to evaluate 1,001
  boundary points for 23.5 million rows.

## 4. Statistical specification

### 4.1 Candidate posterior

For eligible candidate SNP `i`, let `m_i` be its number of usable posterior
intervals. For interval `r`, with endpoints `L_ir < U_ir`, its conditional age
distribution is uniform on `[L_ir, U_ir]`. The candidate CDF is

```text
F_i(t) = (1 / m_i) * sum_r F_uniform(t; L_ir, U_ir).
```

This retains the interval store's current equal-per-usable-interval weighting.
Any future draw-weighting alternative requires a new index schema and must not
be introduced implicitly.

### 4.2 Global control distribution

For `N` eligible control candidates, define

```text
G(t) = (1 / N) * sum_i F_i(t).
```

The requested probabilities are

```text
q_j = j / 1000,  j = 0, ..., 1000.
```

Physical boundaries are inverse values of `G` evaluated on a declared base
age grid. Version 1 should use the same 1,000-generation resolution as the
current target analysis unless a benchmark justifies another value. Record
both requested probabilities and realized global masses.

Repeated physical boundaries must be compressed deterministically. The output
therefore contains at most 1,000 intervals. When boundaries are compressed,
their requested probability shares are combined rather than discarded.

The first boundary must cover the youngest supported age and the terminal bin
must cover the oldest interval endpoint. Boundary and side conventions must be
tested explicitly. Production interval stores contain positive-width
intervals, but endpoint ties still require deterministic half-open semantics.

### 4.3 Fixed-bin candidate weights

For physical boundaries `b_0, ..., b_K`, define candidate weight

```text
w_ij = P_i(b_j <= age < b_(j+1)).
```

The final bin includes the oldest supported endpoint according to the declared
terminal convention. For every eligible candidate:

```text
w_ij >= 0
sum_j w_ij = 1       within numeric tolerance.
```

For a 0.001 global quantile grid, the expected summed mass is approximately

```text
sum_i w_ij ~= 0.001 * N ~= 23,500
```

for an uncompressed interior bin. This is candidate-equivalent mass, not a
guarantee about the count or effective sample size of positive candidates.

### 4.4 Target-specific quotas

For target CDF `T`, evaluate target masses on the fixed boundaries:

```text
p_j = T(b_(j+1)^-) - T(b_j^-).
```

Use the project's deterministic largest-remainder allocation to convert
`p_j` into integer quotas that sum exactly to the target SNP count. An interval
is active for that target if and only if its integer quota is positive.

Do not apply an arbitrary probability cutoff. Small masses naturally receive
zero quota when the target set is too small to allocate a SNP to them.

Before sampling, verify for each active interval that:

- summed control weight is positive;
- positive candidate count is at least the quota; and
- effective sample size is recorded:

```text
ESS_j = (sum_i w_ij)^2 / sum_i(w_ij^2).
```

An insufficient interval must produce a diagnostic error. A future
neighbor-merging policy may be added, but it must be explicit, deterministic,
recorded in metadata, and separately validated. Version 1 should fail rather
than silently merge.

### 4.5 Alias sampling and uniqueness

Build a Walker/Vose alias table for each fixed interval. To draw from interval
`j`:

1. draw a uniform candidate-table column;
2. use the stored alias probability to choose the column or its alias;
3. map the result to the canonical interval-store row;
4. reject and redraw if that candidate is already selected anywhere in the
   current matched set.

Rejecting candidates already selected in the set gives the same sequential
renormalization over the remaining candidates as weighted sampling without
replacement. Randomize interval processing order as the existing sampler does.

Track duplicate redraws by interval. A high redraw rate indicates concentrated
weights or cross-interval overlap and must be visible in diagnostics.

### 4.6 Final acceptance

After all quotas are filled:

1. reconstruct the proposed set's complete aggregate CDF from the interval
   store on the target analysis grid;
2. calculate the same one-dimensional Wasserstein distance used now; and
3. accept only when the distance is at or below the target's bootstrap-derived
   threshold.

The target-specific bootstrap threshold is unchanged. Record all accepted and
rejected proposal distances and rejection reasons.

## 5. Data products and schemas

### 5.1 Canonical interval store

The existing `snp-age-interval-v1` store remains the source of truth. The
derived index records its schema identifier, resolved path, catalog digest,
candidate-row digest, endpoint dtype, draw count, interval count, and relevant
construction parameters.

### 5.2 Global quantile index

Use a new directory schema, provisionally
`global-age-quantile-alias-v1`:

```text
global_quantile_index/
├── metadata.json
├── candidate_rows.npy
├── boundary_probabilities.npy
├── boundary_ages.npy
├── realized_interval_mass.npy
├── positive_candidate_count.npy
├── effective_sample_size.npy
├── total_candidate_weight.npy
├── alias_probability.npy
└── alias_index.npy
```

Required dtypes and shapes:

- `candidate_rows.npy`: `int64`, `(N,)`, strictly increasing canonical rows;
- `boundary_probabilities.npy`: `float64`, `(K + 1,)`;
- `boundary_ages.npy`: `float64`, `(K + 1,)`, strictly increasing after
  compression;
- interval diagnostics: `float64` or suitable integer, `(K,)`;
- `alias_probability.npy`: `float32`, `(K, N)`;
- `alias_index.npy`: `uint32`, `(K, N)` while `N < 2**32`.

Bin-major alias layout makes one interval's complete table contiguous. This is
the access pattern used during sampling and makes active-bin staging possible.

The metadata must include:

- schema and algorithm versions;
- complete command and creation time;
- source-store provenance and hashes;
- candidate input provenance and resolution summary;
- requested quantile step and base age-grid width;
- requested and realized number of bins;
- boundary/endpoint conventions;
- weighting semantics;
- dtypes, shapes, and byte counts;
- per-phase timings and peak RSS;
- build host/job identifiers and CPU count;
- random-independent alias construction details;
- validation tolerances and validation summary; and
- digests for all small arrays plus chunk or whole-file digests for large
  arrays.

### 5.3 Temporary weight matrix

Construction requires a temporary bin-major matrix:

```text
weights_by_bin.npy: float32, shape (K, N)
```

At `N = 23.5 million` and `K = 1,000`, this is about 94 GB decimal
(87.5 GiB). Keep it on node-local scratch. It is a build artifact, not a
required persistent product, and should be deleted only after the complete
alias index has passed validation and been published.

### 5.4 Optional future test-set cache

Do not build this in version 1. If exact interval-store aggregation is too slow
for thousands of exploratory test sets, consider a row-major quantized CDF or
float32 bin-mass cache. This would be a separate acceleration product with its
own precision study. It must not be called the canonical ages.

## 6. Storage and resource envelope

For 23.5 million candidates and 1,000 bins:

| Product | Approximate size |
| --- | ---: |
| Existing canonical interval store | 17 GiB |
| Temporary float32 weights | 87.5 GiB |
| Final float32 alias probabilities | 87.5 GiB |
| Final uint32 alias indices | 87.5 GiB |
| Candidate rows and small metadata | < 1 GiB |

Persistent core after removing temporary weights is approximately 193 GiB
including the canonical interval store. Peak node-local construction space is
approximately 280 GiB before safety margin. Require at least 350 GiB free
node-local scratch and verify it before starting.

A nominal 300 GB quota has little room for simultaneously retaining all
weights, aliases, and the interval store. Keep temporary weights on node-local
scratch and publish only the final aliases. Use explicit byte checks rather
than relying on nominal disk labels.

Initial full-build resource request, subject to benchmark gates:

- 32 CPUs;
- 384--512 GB RAM;
- at least 350 GiB node-local scratch;
- 24--48 hours;
- one node, because the first implementation uses shared memory and local
  scratch.

These are planning limits, not measured requirements. The benchmark phase must
replace them with evidence-based requests.

## 7. Construction pipeline

Implement a new CLI, provisionally `build_global_quantile_index.py`. Do not
expand `sample_age_matched_syn.py` into a combined builder/sampler.

### Phase A: validate and resolve the candidate universe

1. Open and shallow-validate the canonical interval store.
2. Resolve the 23.5-million-position control list once, or accept pre-resolved
   indices.
3. Apply the declared missing-position policy.
4. Apply store eligibility exactly once.
5. Sort rows and reject duplicates.
6. Save `candidate_rows.npy` and resolution diagnostics.
7. Hash the candidate rows and source catalog for stale-index detection.

If the intended pool is exactly every eligible store row, support an explicit
`--all-eligible` mode so a 271 MB text file need not be parsed on every build.
The mode must still record the eligibility rule and resolved row digest.

### Phase B: construct the global aggregate CDF

Do not call `cdf_at()` for every candidate. Implement a streaming aggregate
kernel over the underlying ragged interval arrays.

For candidate `i`, each interval has mass `1 / (N * m_i)`. On a uniform base
age grid, adapt the existing difference-array idea from
`_aggregate_uniform_interval_cdf`:

- one difference array tracks active linear slopes;
- one tracks intercepts; and
- one tracks completed interval mass.

Process interval records in large contiguous blocks. Use `np.bincount` or a
compiled reduction rather than billions of scalar `np.add.at` calls when
benchmarks show a benefit. Accumulate in float64. The output is one global CDF
on the declared base grid.

Validate monotonicity, finiteness, start mass, and terminal value near one.

### Phase C: choose and compress global boundaries

1. Invert the global aggregate CDF at probabilities `0, 0.001, ..., 1`.
2. Use an explicitly documented search side.
3. Map probabilities to physical base-grid boundaries.
4. Force complete support at both ends.
5. Compress repeated physical boundaries.
6. Combine requested probability shares for compressed bins.
7. Re-evaluate realized global mass at the final boundaries.

Write the small boundary and mass arrays immediately. Print the requested and
realized bin count, minimum/maximum realized mass, and number of compressed
boundaries.

### Phase D: compute candidate-by-bin weights

The naive `interval_count * bin_count` calculation is forbidden. Use a
blockwise interval-overlap kernel with complexity approximately
`O(I log K + N*K)`, where `I` is usable interval records and `K` is bins.

For each SNP-row block:

1. load its contiguous ragged intervals;
2. map each interval's lower and upper endpoints to start/end bins with
   `searchsorted`;
3. add partial mass to the first and last overlapping bins;
4. represent fully covered interior bins with per-row range-difference
   coefficients;
5. cumulatively sum those coefficients across bins and multiply by physical
   bin widths;
6. add edge contributions;
7. clamp only roundoff-scale negatives, treating larger negatives as errors;
8. verify per-row sums near one; and
9. transpose the completed block into the bin-major scratch memmap.

For interval `[L, U]` belonging to a SNP with `m` intervals, its density
coefficient is

```text
c = 1 / (m * (U - L)).
```

An interior bin contributes `c * bin_width`. Edge bins contribute their exact
overlap width times `c`. This handles nonuniform physical widths created by
global quantiles without evaluating every interval at every boundary.

Choose the SNP block size from a memory formula and expose it as a CLI option.
A block of 10,000--25,000 rows is a safer initial range than 250,000 when
holding 1,000-bin float64 work arrays.

Implement the hot kernel in compiled code. Numba with `njit` is acceptable if
its version is pinned in `environment.yml` and compilation/runtime behavior is
tested on the cluster. A C++/OpenMP extension is an alternative. A Python loop
over 23.5 million SNPs is not acceptable.

### Phase E: build alias tables

Alias construction is `O(N*K)` and comprises about 23.5 billion table cells.
It must not use a Python loop per cell.

1. Treat bins as independent tasks.
2. For each bin, read its contiguous float32 weight row.
3. Sum in float64 and validate against recorded total weight.
4. Build Walker/Vose probability and alias arrays in compiled code.
5. Parallelize across bins, limiting workers so aggregate memory and I/O remain
   within measured bounds.
6. Write each completed bin into preallocated bin-major output arrays.
7. Mark the bin complete in a durable progress manifest only after validating
   its output.

The `uint32` alias array stores candidate-table columns, not canonical store
rows. Sampling maps through `candidate_rows.npy`.

For every completed bin validate:

- probabilities are finite and in `[0, 1]` within tolerance;
- alias indices are in `[0, N)`;
- a zero-total bin is rejected before construction; and
- empirical samples on selected bins agree with normalized source weights.

### Phase F: publish atomically

1. Build and validate in a uniquely named staging directory.
2. Flush all memmaps and close descriptors.
3. Write metadata last with `complete: false` during staging.
4. Run structural validation and selected empirical alias checks.
5. Set `complete: true`, flush, and atomically rename on the final filesystem.
6. Never open an index lacking `complete: true` for production sampling.

Because node-local scratch and Quobyte are different filesystems, the final
copy cannot itself be a cross-filesystem atomic rename. Copy into a hidden
Quobyte staging directory, validate sizes/digests there, then rename that
directory locally on Quobyte.

## 8. Runtime sampler changes

### 8.1 New reader

Add a `GlobalQuantileAliasIndex` class that:

- validates metadata, source-store provenance, and array shapes;
- memory-maps alias arrays read-only;
- exposes fixed physical boundaries and diagnostics;
- maps sampled table columns to canonical store rows; and
- can stage selected active-bin slabs to node-local scratch.

### 8.2 Generic control sampler

Add `sample_age_matched_controls.py` and retain
`sample_age_matched_syn.py` as a compatibility entry point until migration is
complete. The generic CLI should accept:

```text
--store
--global-index
--target
--output
--accepted-sets
--max-proposals
--seed
--progress-every
--scoring-store / --stage-scoring-store
```

It must not rebuild candidate weights. It should:

1. verify the target and index refer to the same source catalog;
2. compute target masses at fixed boundaries;
3. allocate quotas;
4. identify active bins;
5. validate active-bin supply and ESS;
6. stage active alias rows if requested;
7. generate unique proposals through alias draws;
8. score complete CDFs from interval records;
9. apply the target's Wasserstein threshold; and
10. atomically publish results.

### 8.3 Progress and diagnostics

Emit flushed progress to stdout and a checkpoint diagnostics file:

- setup and provenance validation time;
- active bin count and quota range;
- bytes staged and staging time;
- proposal count;
- accepted count;
- rolling acceptance rate;
- duplicate redraw count and rate;
- sampling versus scoring time;
- latest and quantiles of Wasserstein distances; and
- estimated time to requested accepted-set count.

Write progress at a configurable interval and on termination signals where
safe. Completed accepted sets should be checkpointable so a time-limit signal
does not discard a long run. Final publication remains atomic.

### 8.4 Exact test-set summaries

Keep a small, separate command for arbitrary SNP sets, for example
`summarize_snp_set_ages.py`. It resolves a set, reconstructs its aggregate CDF
from the canonical interval store, and writes summary products and provenance.
It does not require the alias index unless the user also requests matching.

## 9. Many-category execution strategy

Avoid launching hundreds of jobs that each stage or scan the same 200 GB
index independently.

Add a manifest-driven batch runner:

```text
sample_many_age_matched_controls.py --manifest categories.tsv ...
```

The manifest contains target path, output path, seed, accepted-set count, and
optional category label. The runner:

1. opens the interval store and global index once;
2. determines the union of active bins for its assigned category batch;
3. stages those alias rows once to node-local storage when space permits;
4. processes categories sequentially or with a small measured worker count;
5. reuses read-only memory maps across workers; and
6. writes one result directory and diagnostic log per category.

Partition a large category manifest into a modest number of node-sized batches
rather than one SLURM job per category. Benchmark Quobyte load before selecting
the concurrent node count.

The scoring store's endpoint/offset arrays are approximately the whole control
pool. Stage the required read-only arrays once per allocation if node-local
space permits. Do not copy packed status arrays if the scorer does not consume
them; document the minimal valid scoring-stage schema instead of creating an
unvalidated partial store ad hoc.

## 10. Testing plan

### 10.1 Unit tests

- Global aggregate CDF against brute-force `interval_cdf` on tiny fixtures.
- Quantile inversion and deterministic repeated-boundary compression.
- Fixed-bin weights against brute-force CDF differences.
- Intervals contained in one bin, spanning two bins, and spanning many bins.
- Endpoints exactly on boundaries and terminal endpoint behavior.
- Multiple intervals per SNP and per-row normalization.
- Missing/ineligible rows and duplicate candidate requests.
- Alias construction against known two-, three-, and sparse-weight vectors.
- Alias probability bounds and index bounds.
- Empirical alias frequencies with deterministic statistical tolerances.
- Duplicate rejection and global within-set uniqueness.
- Largest-remainder quotas with fewer target SNPs than bins.
- Active-bin selection based only on positive quotas.
- Zero-control-mass and insufficient-positive-candidate errors.
- Reproducibility for identical seeds.
- Metadata/source digest mismatch rejection.
- Interrupted staging directories are never opened as complete indexes.

### 10.2 Property and equivalence tests

For random small interval stores:

- all weights are finite and nonnegative;
- every eligible row sums to one;
- summed candidate bin weights reproduce the aggregate CDF bin masses;
- alias samples converge to normalized weights;
- no candidate repeats within a matched set; and
- stored row mappings round-trip to native coordinates.

### 10.3 Scientific validation

Compare the old target-specific sampler with the global-boundary sampler on a
set of representative categories:

- small, medium, and large target SNP counts;
- young, middle-aged, old, broad, and multimodal target distributions;
- categories concentrated in few global bins and categories spanning most
  bins.

Compare:

- Wasserstein-distance distributions for proposals and accepted sets;
- acceptance rates;
- mean accepted CDF and pointwise deviations from the target;
- per-bin quota realization;
- genomic/chromosomal composition diagnostics where relevant;
- candidate reuse across independent sets; and
- runtime and memory.

The methods need not generate identical random sets. They must demonstrate
equivalent or improved accepted-set matching without systematic age bias.

## 11. Benchmark gates

Do not launch the full 1,000-bin build until all gates pass.

### Gate 1: correctness fixture

Use the existing three-draw interval fixture and a few thousand candidates.
Compare every global CDF, boundary, weight, and alias distribution with a
brute-force reference. Require no semantic discrepancies and document numeric
tolerances.

### Gate 2: orthogonal scaling benchmarks

Run two complementary benchmarks:

1. approximately 1% of candidate rows across all 75 draws, capturing
   interval-record processing cost; and
2. all candidate rows across three draws, capturing `N*K` output and alias
   construction cost.

Both use 1,000 requested bins. Record phase timings, CPU utilization, memory,
scratch bytes, read/write throughput, and extrapolated full-build time.

### Gate 3: full-size dry resource validation

Before computation, calculate exact planned shapes and bytes from `N` and the
realized `K`. Verify:

- node-local free space exceeds required scratch plus 20% margin;
- final filesystem free space exceeds final index plus 20% margin;
- alias index dtype can address `N`;
- requested memory covers work arrays plus 25% margin; and
- time request exceeds Gate 2 extrapolation plus 50% margin.

Abort before allocating large files if any check fails.

### Gate 4: full index build

Build once, validate structurally, and run empirical alias checks for every bin
at a modest sample size plus deeper checks for selected tail and low-ESS bins.
Retain the build report.

### Gate 5: production pilot

Run at least five representative real categories with a small accepted-set
count, then the full 100 sets for one category. Confirm progress reporting,
checkpoint recovery, acceptance rate, exact final scoring, and end-to-end
runtime before launching all categories.

## 12. Performance expectations and failure modes

The present `_batch_cdf` implementation broadcasts interval values against all
query points and loops over SNP rows. At production dimensions it can require
weeks and can create unacceptable temporaries. It must not be used as the
global-index builder.

The optimized builder still processes roughly 1.8 billion posterior interval
records and writes tens of billions of table cells. A several-hour to one-day
one-time build is plausible on a suitably sized compute node, but no runtime
claim should be made until Gate 2 measurements exist.

Primary risks and mitigations:

| Risk | Mitigation |
| --- | --- |
| 1,000 bins create excessive build time | Compiled overlap and alias kernels; Gate 2 extrapolation |
| Physical quantile boundaries repeat | Deterministic compression with realized-mass reporting |
| Candidate mass is concentrated despite global mass | Store positive counts and ESS; monitor redraws |
| Small targets activate very few bins | Expected largest-remainder behavior; retain exact final scoring |
| Broad targets activate nearly all bins | Stage index once per category batch, not once per category |
| Alias files overload remote random I/O | Bin-major layout and node-local staging |
| Final scoring remains slow | Stage minimal scoring arrays; profile separately |
| Build dies before publication | Staging schema, durable completed-bin manifest, no partial opens |
| Source store changes | Catalog and candidate-row digests; reject stale index |
| Float32 alias probabilities introduce bias | Empirical tests against float64 normalized weights |
| Fixed strata alter proposal behavior | Real-category scientific equivalence study |

## 13. Implementation phases and deliverables

### Phase 0: baseline instrumentation

- Add phase timers and progress output to the current sampler.
- Measure where the existing 23.5-million-control run spends time.
- Deliver a baseline benchmark report.

### Phase 1: statistical and schema primitives

- Add global aggregate CDF and boundary helpers.
- Define metadata/schema validation.
- Add fixed-boundary target mass and quota helpers.
- Deliver unit-tested small-array code.

### Phase 2: compiled weight kernel

- Implement interval-overlap range updates.
- Add blockwise scratch writer and resource estimator.
- Pass brute-force correctness tests.
- Deliver Gate 1 and initial Gate 2 measurements.

### Phase 3: alias builder and reader

- Implement compiled Walker/Vose construction.
- Add bin-parallel execution and progress manifest.
- Add structural and empirical validation.
- Deliver a complete three-draw index fixture.

### Phase 4: global sampler

- Implement target quotas over fixed bins.
- Implement alias draws with global uniqueness.
- Retain exact full-CDF scoring and Wasserstein acceptance.
- Add checkpoints and diagnostics.
- Deliver old-versus-new fixture equivalence tests.

### Phase 5: HPC staging and batch execution

- Add node-local active-bin/scoring-store staging.
- Add manifest-driven many-category runner.
- Add SLURM launch template with resource preflight checks.
- Deliver measured concurrency guidance.

### Phase 6: production validation and rollout

- Complete Gates 2--5.
- Freeze schema version and provenance requirements.
- Document index construction, category preparation, sampling, resumption, and
  arbitrary test-set summaries.
- Retire misleading synonymous-only terminology while preserving a temporary
  compatibility wrapper.

## 14. Definition of done

The implementation is ready for broad production use only when:

1. the canonical interval store remains unchanged and fully usable;
2. the global index builds reproducibly from the declared 23.5-million control
   universe;
3. every candidate weight row satisfies normalization tolerances;
4. every alias bin passes structural and empirical checks;
5. source/index provenance mismatches fail closed;
6. target quotas use only fixed global bins and sum exactly to target size;
7. matched sets contain no duplicate SNPs;
8. final acceptance uses the unchanged complete-CDF Wasserstein calculation;
9. representative categories show no systematic matching bias relative to the
   existing method;
10. a full 100-set category completes within the measured production envelope;
11. interrupted jobs leave useful progress and can resume safely where
    supported; and
12. the documented batch workflow can process many categories without each
    job rescanning or restaging the entire control universe.
