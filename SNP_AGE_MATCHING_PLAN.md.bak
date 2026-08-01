# Scalable SNP age-matching implementation plan

## Goal and statistical contract

Build a three-stage workflow that:

1. converts one or more posterior ARG/tree-sequence draws into a reusable, binary NumPy store containing an age-uncertainty CDF for every variant position;
2. accepts a set of `X` TE positions, calculates their aggregate age CDF, and bootstraps those TE SNPs to define an empirical Wasserstein acceptance threshold; and
3. generates 100 sets of `X` synonymous SNPs whose aggregate age CDFs fall within that threshold.

All distributions use one shared age grid. Each SNP's PDF sums to one and its CDF ends at one. The aggregate distribution for a set of SNPs is the arithmetic mean of their individual distributions/CDFs. A synonymous SNP must be unique within a matched set, but may appear in different replicate sets.

The default matching scheme divides the aggregate TE CDF into 20 equal-probability intervals (5% of TE mass per interval) and proposes `X/20` synonymous SNPs per interval, weighted by each synonymous SNP's probability mass in that interval. The final decision is based on the full CDF and Wasserstein distance, not merely interval membership.

## Proposed files and commands

### Code

- `snp_age_distribution.py`: retain the existing CSV/query behavior, but add all-variant extraction and NumPy-store output.
- `snp_age_store.py`: store schema, quantization, validation, position lookup, and block readers.
- `te_age_target.py`: resolve TE positions, construct the target CDF, and bootstrap the Wasserstein threshold.
- `sample_age_matched_syn.py`: build query-specific interval weights, propose synonymous sets, score them, and retain 100 accepted sets.
- `tests/test_snp_age_store.py`, `tests/test_te_age_target.py`, and `tests/test_sample_age_matched_syn.py`.

### CLI sketches

```bash
python snp_age_distribution.py posterior/*.trees \
  --all-variants \
  --numpy-store age_store \
  --bin-width 1000 \
  --block-snps 100000

python te_age_target.py \
  --store age_store \
  --te-positions te_positions.txt \
  --output target/run_001 \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.95 \
  --seed 12345

python sample_age_matched_syn.py \
  --store age_store \
  --target target/run_001 \
  --syn-positions syn_positions.txt \
  --output matches/run_001 \
  --accepted-sets 100 \
  --proposal-batch 1000 \
  --seed 67890
```

`--syn-positions` may be omitted if the store has a precomputed `is_syn` mask. The first implementation should keep biological classification external to ARG extraction and support both modes.

## Stage 1: extract all per-variant distributions

### Extend `snp_age_distribution.py`

Add an `--all-variants` mode that enumerates the sorted union of site positions across all input tree sequences. Existing position-filtered CSV behavior remains backward compatible. For each position and posterior draw:

1. Find the site and marginal tree at that position.
2. For every retained mutation, use the mutation node time as `below` and its parent-node time as `above`.
3. Integrate a uniform distribution on `[below, above]` into the shared nearest-1,000-generation cells, using the current exact-overlap calculation.
4. Treat a zero-width interval as a point mass in the nearest bin, with half-bin ties rounded upward, matching current behavior.
5. Combine contributions across posterior draws and normalize once per position.

The default compatibility policy should remain `--mutation-weighting interval`: every retained mutation interval contributes equal mass, exactly as the current script does. Add a future-safe `--mutation-weighting draw` option under which every posterior draw contributes equal total mass and multiple retained mutations at a site divide that draw's mass. Record the chosen policy in metadata. This distinction matters for recurrent/back mutations and must never be implicit.

Root mutations have no finite upper age. Support the existing `--root skip|error` behavior and record skipped-root counts. Do not invent an upper bound by default. A site with no usable interval after applying policies is retained with a false validity flag and an all-zero CDF; downstream matching rejects it unless explicitly told to drop it.

For a position absent from a posterior draw, support `--missing skip|error`. Under `skip`, normalize over the usable contributions only and store the number of present, missing, usable, and skipped-root draws/intervals. This makes posterior coverage auditable. Duplicate site positions within one tree sequence should be rejected because position lookup would otherwise be ambiguous. Multiple mutations at one site are handled by the explicit weighting policy above.

### NumPy store schema

Use a directory containing a small number of large `.npy` files plus JSON metadata:

```text
age_store/
  metadata.json
  age_bins.npy
  positions.npy
  cdf_by_snp.npy
  cdf_by_age.npy
  valid.npy
  present_draw_count.npy
  usable_interval_count.npy
  skipped_root_count.npy
  missing_draw_count.npy
```

Precise representation:

| Array | Shape | dtype | Meaning |
|---|---:|---|---|
| `age_bins` | `(B,)` | `uint64` | sorted bin centers in generations |
| `positions` | `(N,)` | `float64` | sorted, unique tskit site positions |
| `cdf_by_snp` | `(N, B)` | `uint16` | row-contiguous quantized SNP CDFs |
| `cdf_by_age` | `(B, N)` | `uint16` | transposed copy for all-candidate boundary scans |
| `valid` | `(N,)` | `bool` | at least one usable age interval |
| count arrays | `(N,)` | `uint32` | extraction and missingness diagnostics |

Quantize with `q = round(65535 * CDF)` after enforcing monotonicity in floating point. Set every valid row's last element exactly to 65535. Decode as `q.astype(float32) / 65535`. The maximum per-cell quantization error is about `7.63e-6`; metadata must include the scale and scheme. Use `uint64` age bins to avoid assumptions about maximum ancestral age. Keep positions as `float64`, because tskit coordinates need not be integers; reject non-finite positions and document exact coordinate matching.

The two CDF orientations deliberately trade disk space for I/O efficiency: `cdf_by_age` supports contiguous scans of all synonymous candidates at 21 TE interval boundaries, while `cdf_by_snp` supports contiguous retrieval and scoring of selected SNPs. Provide `--omit-transpose` for constrained storage, with a documented performance penalty. Do not use compressed archives because they defeat partial reads.

`metadata.json` records schema version, creation command, input file names and checksums, number of posterior draws, sequence length, bin width and edge convention, number of bins/SNPs, dtypes/shapes, quantization scale, interval weighting, root/missing policies, and extraction totals. Write it only after all arrays validate successfully.

### Construction without excessive RAM

Implement these interfaces in `snp_age_store.py`:

```python
def discover_positions(tree_files: Sequence[Path]) -> np.ndarray: ...
def determine_age_grid(tree_files: Sequence[Path], bin_width: float) -> np.ndarray: ...
def build_store(tree_files: Sequence[Path], output_dir: Path, *,
                bin_width: float, block_snps: int,
                missing: str, root: str,
                mutation_weighting: str) -> None: ...
def validate_store(store_dir: Path, *, deep: bool = False) -> StoreReport: ...
def resolve_positions(sorted_store_positions: np.ndarray,
                      query_positions: np.ndarray) -> np.ndarray: ...
```

Use `numpy.lib.format.open_memmap` to allocate final arrays. Discover the union of positions and maximum usable upper age first. Process sorted positions in configurable blocks, accumulating only a block-sized floating PDF/CDF buffer. Quantize and flush each block into `cdf_by_snp`. Build `cdf_by_age` in tiled transposition passes so neither full matrix is resident in memory. Write into `output_dir.tmp.<job-id>` and atomically rename it only after validation, preventing interrupted jobs from appearing complete.

Position resolution uses `np.searchsorted` on sorted `positions`, followed by exact equality verification. It must report missing and duplicate query positions rather than silently dropping them. Optionally add a separate integer-position fast path only after verifying all stored coordinates are integral.

Validation checks shapes/dtypes against metadata, strictly increasing unique positions and age bins, monotone CDF rows, final CDF value 65535 for valid rows, zero rows for invalid entries, nonnegative/consistent counts, and equality of sampled tiles (or all tiles in `--deep` mode) between the two orientations. Include SHA-256 checksums as an optional expensive finalization step rather than on every open.

## Stage 2: TE target and bootstrap threshold

Implement in `te_age_target.py`:

```python
def load_position_list(path: Path) -> np.ndarray: ...
def aggregate_cdf(cdf_rows: np.ndarray) -> np.ndarray: ...
def wasserstein_1(cdf_a: np.ndarray, cdf_b: np.ndarray,
                  bin_centers: np.ndarray) -> float: ...
def bootstrap_wasserstein(cdf_rows: np.ndarray, n_replicates: int,
                          rng: np.random.Generator,
                          batch_size: int) -> np.ndarray: ...
def equal_mass_boundaries(target_cdf: np.ndarray,
                          probabilities: np.ndarray) -> BoundarySet: ...
```

Resolve all `X` TE positions exactly, reject duplicates by default, and reject missing or invalid store rows with a complete diagnostic list. Decode selected rows to `float32` or accumulate to `float64`; compute the aggregate TE CDF as their mean. For nonuniform age grids, calculate discrete one-dimensional Wasserstein distance by integrating absolute CDF differences over adjacent bin coordinates:

```text
W1 = sum(abs(CDF_A[:-1] - CDF_B[:-1]) * diff(age_bins))
```

The result is in generations. For a uniform 1,000-generation grid this reduces to the familiar sum times 1,000.

Bootstrap SNP rows, not bins: for each replicate, draw `X` TE row indices with replacement, average their CDFs, and measure W1 to the observed aggregate TE CDF. Vectorize in batches to cap memory. The default acceptance threshold is the empirical 95th percentile of these distances. This is a **95th-percentile tolerance threshold**, not a confidence interval for the mean W1: a synonymous set passes when its distance to the observed TE target is no greater than the distance seen in 95% of TE bootstrap resamples.

Use 10,000 bootstrap replicates by default; allow 5,000 for exploratory work. One hundred replicates are too few to estimate a stable 95th percentile. Report the percentile estimate plus a Monte Carlo uncertainty interval obtained by resampling the bootstrap distances or by order-statistic bounds. Optionally support `--bootstrap-reference two-sample`, comparing two independent TE resamples, but default to `observed` because synonymous samples are scored against the observed target. Record the method and seed.

Derive the 0%, 5%, ..., 100% age boundaries by inverse lookup in the aggregate TE CDF. Rounded/repeated boundaries are expected on a discrete grid. Preserve all 20 requested strata in metadata, but mark zero-width strata; sampling code must merge them with the nearest nonzero stratum or allocate their quota by largest-remainder apportionment according to attainable interval mass. The exact policy is an unresolved decision to settle with tests before production use.

Write a target directory containing:

```text
target/run_001/
  te_positions.npy
  te_row_indices.npy
  target_cdf.npy
  bootstrap_wasserstein.npy
  interval_boundaries.npy
  interval_quotas.npy
  metadata.json
```

Quotas must sum exactly to `X`. If `X` is not divisible by 20, use largest-remainder allocation of the ideal `0.05 * X` quotas with seeded random tie-breaking, and record the result.

## Stage 3: generate 100 age-matched synonymous sets

### Query-specific weighted sampling

For TE interval `j = [a_j, a_{j+1})` (last interval closed on the right), define synonymous SNP `i`'s weight as its CDF difference across the boundaries. Boundary lookup must respect the stored grid and interval convention. Quantized subtraction can be done in `uint32`; convert to floating point only for sampling.

Do not construct or save a `20,000,000 x 20` weight matrix. Instead, use hierarchical block sampling:

1. Scan `cdf_by_age` at only the distinct boundary rows, in large contiguous candidate blocks.
2. Apply the synonymous-candidate index/mask and validity filter.
3. Accumulate only a `(20, number_of_blocks)` table of total interval weights and candidate counts.
4. For each draw, select a block proportional to its current total weight, load the two boundary slices for that block, and select a SNP proportional to its local weight.
5. After selecting a SNP, subtract its weights from all affected block totals for that set so it cannot be selected again in another interval. Reset the small block-total table for the next proposed set.

This implements successive probability-proportional-to-size sampling without replacement while keeping memory proportional to blocks, boundaries, and `X`, rather than all candidates times intervals. Cache recently used boundary blocks within a node. Benchmark block sizes such as 250,000-1,000,000 SNPs on Quobyte; make the choice configurable.

Implement:

```python
def build_interval_block_index(store: AgeStore, syn_indices: CandidateIndex,
                               boundaries: BoundarySet,
                               block_snps: int) -> BlockWeightIndex: ...
def draw_stratified_set(index: BlockWeightIndex, quotas: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray: ...
def score_set(store: AgeStore, row_indices: np.ndarray,
              target_cdf: np.ndarray) -> MatchDiagnostics: ...
def generate_matches(..., accepted_sets: int = 100,
                     proposal_batch: int = 1000) -> MatchResult: ...
```

Draw interval order randomly per proposal to avoid systematically disadvantaging old or young strata. If an interval has zero total eligible mass or cannot fill its quota after enforcing uniqueness, reject the proposal with a reason; never fall back to zero-weight candidates silently. Score every complete proposal against the full target CDF. Accept it if `W1 <= threshold`, continuing in batches until 100 are accepted or `--max-proposals` is reached.

Do not automatically retain the closest 100 from a fixed batch: that overmatches the target. If more than 100 proposals pass in the final batch, choose randomly among passing proposals using the recorded RNG. Report acceptance rate. Add optional guardrails for maximum absolute CDF difference and interval leakage, but keep Wasserstein as the primary criterion.

Use `numpy.random.SeedSequence` to derive independent, recorded streams for bootstrap, proposal generation, interval ordering, and final tie selection. Results must be reproducible across reruns with the same store schema/version, candidate set, parameters, NumPy version, and seeds.

### Outputs

```text
matches/run_001/
  syn_positions.npy       # (100, X), same coordinate dtype as store
  syn_row_indices.npy     # (100, X), uint32 or uint64 as required by N
  syn_cdf.npy             # (100, B), float32
  wasserstein.npy         # (100,), float64 generations
  interval_assignment.npy # (100, X), uint8
  diagnostics.csv
  metadata.json
```

`diagnostics.csv` contains proposal/accepted IDs, seed lineage, W1, threshold, bootstrap percentile, maximum absolute CDF difference, mean/median age summaries, interval quotas, realized interval mass/leakage, duplicate-rejection count, and proposal rejection reason. Metadata references the source store and target metadata/checksums.

## Quobyte and SLURM execution design

- Keep the store to a handful of large files; never create one file per SNP or per distribution.
- Read large contiguous slices. Avoid element-wise memory-map access over Quobyte.
- Treat store arrays as immutable after construction, allowing concurrent read-only jobs.
- Where `$SLURM_TMPDIR` has sufficient space, stage the boundary slices, block index, target rows, and selected-row working set there; copying the entire store is optional and should be benchmark-driven.
- For store construction, parallelize extraction by disjoint SNP blocks into per-task temporary arrays, then run one deterministic merge/transposition/final validation job. Never let workers write overlapping regions.
- For matching, one SLURM task per TE set is the starting design. Parallelize independent TE sets or proposal batches rather than issuing many tiny reads against one store.
- Record wall time, bytes read, peak RSS, block size, node/storage location, and acceptance rate so `.npy`, block size, and staging choices can be tuned empirically.
- On network storage, `np.load(..., mmap_mode="r")` is useful only when access remains blockwise; explicit `np.asarray(memmap[slice])` reads make I/O boundaries clearer.

## Testing and validation plan

### Unit tests

- Exact integration of uniform intervals into bins, normalization, and zero-width point masses.
- Multiple mutations with both interval-weighted and draw-weighted semantics.
- Missing sites, root mutations, all-invalid sites, duplicate site coordinates, and noninteger positions.
- Quantization error bounds, CDF monotonicity, terminal value, and transpose equality.
- Exact/missing/duplicate position lookup.
- W1 for identical, adjacent-bin, distant-bin, and nonuniform-grid distributions.
- Seeded bootstrap reproducibility, batching invariance within numerical tolerance, and percentile calculation.
- Equal-mass boundaries with flat CDF regions, repeated boundaries, and `X` not divisible by 20.
- Weighted sampling frequencies on a tiny known pool, uniqueness within sets, cross-set reuse, exact quotas, zero-weight failures, and randomized interval order.
- Acceptance/rejection exactly at the threshold and deterministic seed lineage.

### Integration and performance tests

- Compare the new store's decoded PDFs against existing CSV output for `results/neutral_100kb.trees` at selected sites.
- Build a small multi-draw fixture containing missing sites, recurrent mutations, root mutations, and point intervals.
- End-to-end test: construct store, define TE/syn candidates, bootstrap target, obtain 100 accepted sets, and verify every output shape and invariant.
- Confirm aggregate accepted CDFs satisfy the stored W1 threshold when independently recomputed.
- Synthetic scale benchmark using enough rows to extrapolate storage, sequential boundary-scan time, selected-row scoring time, proposal throughput, and Quobyte I/O.
- Run a calibration study: split TE SNPs into pseudo-target and pseudo-candidate pools and verify that the nominal 95th-percentile threshold yields a plausible pass rate. This tests whether the bootstrap reference matches the actual selection procedure.

## Milestones

1. **Schema and semantics:** finalize mutation/draw weighting, root/missing policies, discrete-bin convention, bootstrap reference, and repeated-boundary handling.
2. **Store implementation:** add all-variant extraction, blockwise `.npy` construction, position lookup, metadata, validation, and compatibility tests.
3. **Target implementation:** add aggregate CDF, W1, bootstrap threshold, equal-mass boundaries, CLI, and deterministic outputs.
4. **Sampler prototype:** implement hierarchical block weights and exact within-set uniqueness on the 100 kb dataset.
5. **Calibration:** quantify match quality, leakage, bootstrap coverage, acceptance rate, and whether 20 intervals are sufficient.
6. **HPC optimization:** benchmark block sizes, array orientations, Quobyte reads, node-local staging, and SLURM layouts at increasing synthetic scale.
7. **Production hardening:** resume-safe outputs, checksums, provenance, failure diagnostics, and an end-to-end workflow wrapper.

## Unresolved decisions to settle before production

- Whether recurrent mutations should be weighted per interval (current behavior) or normalized per posterior draw.
- Whether the posterior files have exactly aligned sites; if not, whether missing draws are ignorable or indicate a data error.
- Whether root mutations should remain excluded or use a scientifically justified finite upper prior.
- Whether the acceptance null should be observed-target versus bootstrap (recommended) or bootstrap versus bootstrap.
- How to merge/reallocate quotas for repeated 5% boundaries on a discrete CDF.
- Whether TE and synonymous lists can overlap and, if so, whether all TE positions must be excluded from the synonymous candidate pool.
- Whether synonymous candidates may be reused across the 100 accepted sets (recommended: yes) and whether overlap diagnostics or a cross-set cap is needed.
- Maximum proposal count and behavior when fewer than 100 samples pass; default should fail with diagnostics rather than loosen the threshold silently.
- Whether both CDF orientations fit the production storage budget; benchmark before omitting either.
- Required coordinate namespace for multi-contig data. If positions are not globally unique, replace `positions` with sorted `(contig_id, position)` keys and add a contig table before implementation.
