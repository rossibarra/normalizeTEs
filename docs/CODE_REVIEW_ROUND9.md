# Code Review — Round 9

Date: 2026-08-25  
Reviewed revision: `2a94782` (`bootstrap-target-hpc-validation`)

## Summary

The established pipeline code is in good condition: the complete test suite passes,
the Python sources compile, the scheduler scripts pass shell syntax checks, and the
tracked working tree was clean at the time of review. This sweep deliberately did
not revisit the scientific algorithms.

The pipeline is not yet ready for a full production run. The new TE-polarity path
has two production-blocking integration bugs and no focused tests. A third high
priority provenance gap means an ancestral-state table can claim a store identity
without proving that it was computed from that store's posterior draws. Findings
4–6 should also be addressed before treating cross-artifact provenance and resource
preflights as reliable.

## Findings

### 1. High — polarity filtering can create an all-NaN TE CDF

`te_age_target.masked_row_cdfs()` decides whether to fall back based on whether the
polarity mask contains any agreeing draw for the TE. It does not check whether any
of those agreeing draws supplied a usable age interval in the interval store.

If all usable intervals came from flipped draws, `selector` is empty even though
`keep[i].any()` is true. The resulting filtered interval row has length zero.
`snp_interval_dataset._batch_cdf()` leaves empty rows entirely NaN, and target
construction does not reject non-finite CDFs before bootstrap and publication.

Relevant code:

- `normalize_tes/te_age_target.py:216–251`, especially the selector at line 237
- `normalize_tes/snp_interval_dataset.py:803–826`, especially the empty-row behavior at lines
  811–815
- `normalize_tes/te_age_target.py:427–457`, where the CDFs flow into bootstrap and aggregation

Recommended correction: base the fallback on whether the interval-level selector
contains an element, then explicitly require every constructed CDF row, the target
CDF, the bootstrap distances, and the threshold to be finite. Whether an empty
filtered row should fall back, be dropped, or stop the run is a scientific-policy
choice; silently publishing NaNs is the implementation bug.

### 2. High — the bootstrap launcher can silently ignore a requested polarity mask

`slurm/run_bootstrap_matching.sbatch` constructs `--te-polarity-mask` and
`--max-flipped-fraction` arguments, but passes them only inside the block that builds
`TARGET` when that directory does not exist.

The mask builder itself requires an existing preliminary target to supply
`te_row_indices.npy`. Consequently the actual workflow requires distinct artifacts:

```text
preliminary unmasked target -> polarity mask -> final masked target -> matching
```

If an operator supplies the preliminary target as `TARGET` together with
`TE_POLARITY_MASK`, the launcher sees the target directory, skips target construction,
silently ignores the mask, and performs matching against the unmasked target.

Relevant code:

- `slurm/run_bootstrap_matching.sbatch:104–136`
- `normalize_tes/build_te_polarity_mask.py:121–126`, where an existing target is required

Recommended correction: when a polarity variable is set and `TARGET` already
exists, verify that its metadata records the exact requested mask and threshold or
stop with an actionable error. Document and enforce separate preliminary and final
target paths.

### 3. High — the ancestral table does not prove it used the store's source draws

`normalize_tes/build_ancestral_states.py` accepts arbitrary tree files and stamps the selected
store's `content_sha256` into the output metadata. It does not compare the supplied
draws with the store's recorded `metadata["inputs"]`.

The merge path verifies that parts claim the same store, use disjoint resolved paths,
and total the expected number of draws. Those checks do not establish that the paths
are the store's source draws. For example, the expected number of distinct but wrong
draws could be accumulated and published with the selected store's digest.

Relevant code:

- `normalize_tes/build_ancestral_states.py:105–142`, which accumulates the supplied files
- `normalize_tes/build_ancestral_states.py:228–317`, which validates merge parts and counts
- `normalize_tes/build_ancestral_states.py:348–363`, which stamps the store digest

The analogous input-to-draw-ID validation in
`build_te_polarity_mask.store_draw_columns()` (`normalize_tes/build_te_polarity_mask.py:33–66`)
provides a suitable implementation pattern. Direct construction should require a
subset of the store's recorded inputs, and a final merge should require the exact
expected set, not only the expected cardinality.

### 4. Medium — candidate-row store provenance is written but ignored

`normalize_tes/build_candidate_rows.py` writes a JSON sidecar containing the store content digest,
catalog digest, row count, and candidate-array digest. The production matcher loads
only the `.npy`, checks that its numeric rows are currently eligible and in range,
and hashes the raw array. It never reads or authenticates the sidecar.

A candidate array from another store with a compatible row count can therefore be
accepted even though the same row numbers identify different SNPs.

Relevant code:

- `normalize_tes/build_candidate_rows.py:183–227`
- `normalize_tes/bootstrap_target_matcher.py:415–421`

Recommended correction: make the report part of the candidate artifact contract and
validate its store content digest, catalog digest, row count, and array digest before
using the rows.

### 5. Medium — polarity-mask provenance validation is incomplete

The mask builder loads `te_row_indices.npy` from its target and coerces it to
`int64`, but does not authenticate the target metadata against the supplied store or
validate the original row-array dtype, dimensionality, bounds, or uniqueness.
It then records the supplied store's digest in the mask metadata.

The target consumer rejects different digests only when both the expected and
recorded values are truthy. A missing or null mask digest therefore bypasses the
identity check.

Relevant code:

- `normalize_tes/build_te_polarity_mask.py:136–175`
- `normalize_tes/te_age_target.py:291–307`

Recommended correction: authenticate the source target against the store before
using its row indices, validate the row array, and require an exact non-null content
digest when consuming a polarity mask.

### 6. Medium — scratch and memory behavior is misleading

The bootstrap launcher checks for only the staged store size plus 20% of free
scratch. Target construction may simultaneously require a large TE-by-age CDF, so
the preflight can succeed and the job can still exhaust scratch.

For a masked interval target, `_analysis_cdfs()` takes the general path and builds
the complete CDF as an in-memory `float64` array before converting it to `float32`.
Nevertheless, target metadata describes the working storage as a temporary
scratch-backed NPY. This materially understates RAM requirements for large TE
categories and makes the scratch-space calculation inside that path irrelevant.

Relevant code:

- `slurm/run_bootstrap_matching.sbatch:90–101`
- `normalize_tes/te_age_target.py:175–209`, especially lines 188–192
- `normalize_tes/te_age_target.py:655–667`, which records inaccurate storage and algorithm metadata

Recommended correction: either implement a blockwise scratch-backed masked writer,
or report and preflight its actual in-memory behavior. The launcher should reserve
space for both the staged store and the target CDF, with a safety margin.

### 7. Low to medium — remaining operational edge cases

#### Unmatched ancestry glob is not detected early

`slurm/run_ancestral_table.sbatch` expands `tree_files=( $TREES )` with Bash's default
`nullglob` setting. An unmatched glob remains as one literal string, so the following
non-empty-array check passes. The Python process eventually fails on the nonexistent
path, but the launcher's intended early diagnostic does not work.

Relevant code: `slurm/run_ancestral_table.sbatch:60–64`.

#### Duplicate resolved interval-store inputs are accepted

`build_snp_interval_store.build_interval_store()` resolves every tree path but does
not reject duplicate resolved paths. Relative, absolute, or symlink aliases of one
file can therefore cause one posterior draw to be counted twice and assigned two
draw IDs.

Relevant code: `normalize_tes/build_snp_interval_store.py:435–461`.

#### Existing outputs are detected after expensive work

Some long-running commands do most of their computation before checking whether the
destination already exists. Target construction checks in `write_target()` only
after CDF construction and bootstrap; Phi-SFS checks after scanning the VCF and
computing the spectra. An obvious output collision can consequently waste hours.

Relevant code:

- `normalize_tes/te_age_target.py:461–465`
- `normalize_tes/phi_sfs.py:799–803`
- `normalize_tes/build_ancestral_states.py:145–158`

These commands should perform a resolved output-path and collision preflight before
loading or computing expensive inputs, while retaining the publication-time check
to protect against races.

## Important omissions from the streamlined README

1. The TE-polarity-mask stage is absent. The README should document
   `normalize_tes/build_te_polarity_mask.py`, `slurm/run_te_polarity_mask.sbatch`,
   `--te-polarity-mask`, `--max-flipped-fraction`, and the required preliminary
   target -> mask -> final target workflow.
2. The Farm section says there are three launchers, but there are now four. The
   polarity-mask launcher should be added to the table and example commands.
3. The outputs table omits the polarity-mask artifact.
4. A production choice or recommended value for `MAX_FLIPPED_FRACTION` is not
   provided.
5. The resource guidance should distinguish unmasked scratch-backed target building
   from the current in-memory masked path, and scratch sizing must include the target
   CDF rather than only store size plus 20%.
6. The statement that ordinary tskit tree-sequence files are accepted is too broad.
   The interval-store builder uses a loader that supports ordinary and tszip files,
   while the ancestry and polarity-mask builders call `tszip.decompress()` and thus
   require tszip archives.
7. Important Phi-SFS input assumptions should appear directly in the README rather
   than only through a validation-document reference: TE ALT encodes insertion or
   presence, FILTER is ignored, the minimum callable-sample rule, genotype handling,
   and the need to review retained and endpoint fractions.
8. The v0.4 changelog should record the TE-age polarity-mask behavior and its
   operational/compatibility implications.

## Validation performed

- Complete test suite: **171 passed**, with four multiprocessing `fork()`
  deprecation warnings.
- Forced Python compilation: passed.
- `bash -n` over the scheduler launchers and conda bootstrap: passed.
- `git diff --check`: passed.
- Tracked working tree at review time: clean at `2a94782`.
- No focused tests reference `normalize_tes/build_te_polarity_mask.py`, `masked_row_cdfs()`, or the
  TE-polarity target path. Passing the established suite therefore does not exercise
  findings 1, 2, or 5.
- The repository also contained untracked generated result artifacts and an
  untracked `tools/compare_polarity_rebuild.py`; these were not modified during review.

## Readiness recommendation

Do not begin the full production pipeline until findings 1–3 are corrected and the
polarity workflow has focused regression and launcher tests. Findings 4–6 should
also be closed before relying on the documented cross-artifact authentication and
resource preflight guarantees. Finding 7 is operational hardening and can follow,
although the duplicate-draw check is inexpensive and should be added before the next
interval-store rebuild.
