# normalizeTE code review

Review date: 2026-08-03. Reviewed at commit `54b3045` plus uncommitted working-tree
changes (`README.md`, `build_snp_age_store.py`, `sample_age_matched_syn.py`,
`snp_age_dataset.py`, `te_age_target.py`, `environment.yml`, three test files).

Scope: the top-level Python modules and their tests. `singer-snakemake/` is a vendored
external repository and was not reviewed.

Test suite status at time of review: **33 passed**.

Line references point at the working-tree versions listed above.

---

## 1. Real bugs

### B5 — Every tree sequence is reloaded once per SNP block *(performance, severe)*

The loop nesting in `build_store` is `for block: for path: _load(path)`
([build_snp_age_store.py:188-201](build_snp_age_store.py#L188-L201)), so each posterior
draw is decompressed and parsed once per SNP block rather than once per run.

Verified by instrumenting `_load` with a counter — 4 draws, 100 SNPs, `block_snps=10`
(10 blocks):

```
total _load calls: 52   (per file: [13, 13, 13, 13])
expected if loaded once per file: 4
```

The 13 per file = 10 block iterations + `discover_positions` + `determine_age_grid` +
`loaded_headers`.

At the documented target scale — 20M SNPs, `block_snps=100000`, 100 posterior draws —
that is ~20,000 full tszip decompressions instead of ~100.

**Fix:** invert the loops so each tree sequence is loaded once and all SNP blocks are
filled from it, or hold the parsed tree sequences across blocks.

### B6 — `ts.at(position)` per site instead of one `ts.trees()` pass *(performance)*

Used in two places: `determine_age_grid`
([build_snp_age_store.py:92](build_snp_age_store.py#L92)) and the main extraction loop
([build_snp_age_store.py:218](build_snp_age_store.py#L218)). Each call constructs a fresh
`Tree` and seeks to the position, so cost grows with the number of trees as well as the
number of sites.

Measured (msprime, 50 haploid samples, `recombination_rate=1e-8`):

| sites  | trees  | `ts.at()` per site | one `ts.trees()` pass | ratio |
|--------|--------|--------------------|-----------------------|-------|
| 3,692  | 3,152  | 0.06s              | 0.01s                 | 3.8x  |
| 14,696 | 12,364 | 0.54s              | 0.06s                 | 9.2x  |
| 58,559 | 12,364 | 2.13s              | 0.21s                 | 10.0x |

The ratio grows with size — this is not a constant-factor penalty.

**Fix, extraction loop:** iterate `for tree in ts.trees(): for site in tree.sites():` once
and scatter results into the block, instead of seeking per site.

**Fix, `determine_age_grid`:** it only needs an upper bound on mutation-parent node time.
`ts.tables.nodes.time.max()` is a valid upper bound and takes ~0.1 ms. A few trailing
zero-mass age bins are harmless. Verified equivalent-or-greater on the fixtures above
(154830.1 exact vs 155888.1 bound).

### B3 — `min_usable_draws` is unreachable through the Python API

`build_store`'s signature defaults `min_usable_fraction=0.1`
([build_snp_age_store.py:135](build_snp_age_store.py#L135)), and
[line 149](build_snp_age_store.py#L149) rejects both parameters being set. So any caller
passing `min_usable_draws` alone hits the mutual-exclusion error.

Verified:

```python
build_store([path], out, bin_width=10, min_usable_draws=1)
# ValueError: specify only one of min_usable_draws and min_usable_fraction
```

The CLI only works because of the compensating expression at
[lines 327-329](build_snp_age_store.py#L327-L329), which is itself a symptom.

**Fix:** default `min_usable_fraction=None` and resolve the default inside the function,
or collapse to a single knob (see O-d).

### B2 — The coordinate interface is mislabeled "1-based VCF", and is lossy by construction

**Upstream convention, now confirmed.** ARGtest performs no ±1 conversion anywhere. It
exports with `ts.write_vcf()` and no `position_transform`, so tskit's default applies:

```
VCF POS = round(site.position)      # no +1
```

Verified on tskit 1.0.3 — `write_vcf`'s `position_transform=None` resolves to `np.round`,
and a ts with sites at `[5.0, 10.5, 11.5, 40.0]` exports as POS `[5, 10, 12, 40]`
(half-to-even: `10.5 -> 10`, `11.5 -> 12`).

**Consequence 1 — the arithmetic is right, the name is wrong.** Because ARGtest's POS
column *is* the (rounded) 0-based tskit coordinate,
`native_to_global`'s `offset + position` ([snp_age_dataset.py:114](snp_age_dataset.py#L114))
is correct. But the code and README call this a "1-based VCF position" throughout —
`load_native_position_list`'s docstring and error text
([snp_age_dataset.py:165-184](snp_age_dataset.py#L165-L184)), `native_to_global`'s and
`rows_to_native`'s docstrings, the `"is not a 1-based VCF coordinate"` error, and the
README's *"exactly two whitespace-separated columns: chromosome and 1-based VCF position"*.

It is not 1-based. Anyone who takes that label at face value and prepares a TE list from an
externally produced (genuinely 1-based) VCF, or from a 0-based BED, will be off by one.
The failure mode is either `KeyError: positions not found` (loud, fine) or — when the
neighbouring integer happens to also be a site in the store — **silent selection of the
wrong SNP**. That is the real hazard, and it is created entirely by the misleading label.

**Consequence 2 — the (chrom, POS) interface cannot represent fractional site positions.**
The store matches positions by exact float64 equality
(`"position_matching": "exact float64 equality"`), but POS has been through `np.round`.
Verified: a store whose only sites are `[10.5, 40.0]` exports as POS `[10, 40]`, and

```
resolve_native_positions(["chr1"], [10])  ->  KeyError: positions not found: 10
```

The site is unreachable. Worse, two sites rounding to the same integer produce a duplicate
POS, and a list containing it resolves to whichever site sits on the exact integer — or to
nothing — with no indication that a second site was meant.

This is a regression in interface robustness: the previous raw-global-coordinate interface
(cf. `README.md.native-coordinates.bak`) had no lossy round-trip. Switching to
(chrom, POS) pairs introduced one.

**Consequence 3 — `rows_to_native`'s lower bound is off by one.** Chromosome *c* occupies
tskit coordinates `[offset, offset+length)`, so local positions are `[0, length)`. But
[snp_age_dataset.py:136](snp_age_dataset.py#L136) requires `1 <= value <= length`.
Verified on a two-chromosome store with sites at globals `[0, 50, 99, 100, 150]`
(chr1 offset 0 len 100, chr2 offset 100 len 100):

```
global   0.0 -> REJECTED: store position 0 is not a 1-based VCF coordinate
global  50.0 -> chr1 50
global  99.0 -> chr1 99
global 100.0 -> REJECTED: store position 100 is not a 1-based VCF coordinate
global 150.0 -> chr2 50
```

The chromosome assignment itself is correct — `searchsorted(offsets, x, side="right") - 1`
puts `global == offset` into the right chromosome, so `value == length` is unreachable for
any chromosome that has a successor. The single real defect is that **local position 0 —
the first base of every chromosome — is rejected.** `load_native_position_list`'s
`position < 1` check ([snp_age_dataset.py:183](snp_age_dataset.py#L183)) has the same
problem from the other direction.

(ARGtest itself cannot emit POS 0, because it does not pass `allow_position_zero` and
`write_vcf` raises on a site at position 0.0. So this is unreachable via ARGtest's own VCF
export, but reachable for any store row selected as a synonymous candidate via
`--syn-mask` or `--syn-indices`.)

**Recommended fix — one assertion that resolves this, B1, and the bounds question
together.** Site positions are integers in practice (msprime with `discrete_genome=True`,
and SINGER ARGs built from VCF). Turn that belief into an enforced invariant in
`build_store`:

```python
if np.any(positions != np.floor(positions)):
    raise ValueError("site positions must be integral for (chrom, POS) addressing")
```

With that in place the `np.round` round-trip is exact and total, the duplicate-POS and
unreachable-site cases become impossible, B1 cannot occur, and the bounds need no
special-casing beyond correcting them to `0 <= value < length`.

Separately: **rename the interface**. It is a chromosome-local tskit coordinate, not a
1-based VCF position — even though it is what appears in ARGtest's POS column. Say so in
the README and in the three docstrings, and note explicitly that lists prepared from
external 1-based VCFs need a `-1`.

### B7 — `quantization_scale` metadata is read and then ignored

The chain:

1. **Written** — [build_snp_age_store.py:274](build_snp_age_store.py#L274) writes
   `"quantization_scale": QUANTIZATION_SCALE` (always `65535`) into the store's
   `metadata.json`.
2. **Read** — [snp_age_dataset.py:31](snp_age_dataset.py#L31) loads it into
   `self.quantization_scale`.
3. **Ignored** — `read_cdfs` ([line 146](snp_age_dataset.py#L146)) and
   `read_boundary_cdfs` ([line 161](snp_age_dataset.py#L161)) both divide by the module
   constant `QUANTIZATION_SCALE`, not by `self.quantization_scale`.

The only consumer of the attribute is `_decode_cdfs`
([sample_age_matched_syn.py:61](sample_age_matched_syn.py#L61)), guarded by
`np.issubdtype(values.dtype, np.integer)`. Since `read_cdfs` already returns float32, that
branch is unreachable for a real `SNPAgeDataset` — it only fires for the `FakeStore` in
the tests, which sets `quantization_scale` as a plain attribute.

So the metadata field is effectively write-only. Nothing is wrong today because exactly one
scale exists; it is a trap for whoever changes it.

**Fix:** either delete the attribute and the metadata key and treat `65535` as a format
constant, or actually use `self.quantization_scale` in both readers.

### B4 — Fallback that substitutes ages where edge indices are expected

[sample_age_matched_syn.py:424-427](sample_age_matched_syn.py#L424-L427) falls back from
`interval_boundary_indices.npy` to `interval_boundaries.npy` when the first is absent, and
passes the result straight into `build_interval_block_index` as `boundary_indices`.

Those two files hold different quantities. For a 4-bin grid:

```
interval_boundary_indices.npy  (edge indices)         [0, 1, 2, 4]
interval_boundaries.npy        (ages, generations)    [0.0, 0.0, 1000.0, 3000.0]
```

`te_age_target.write_target` always writes both, so the fallback is currently dead. If it
ever fires it produces either a bounds error or wrong strata.

**Fix:** delete the fallback and fail with a clear message if the file is missing.

### B8 — Full deep re-validation on every build

`build_store` calls `validate_store(temp, deep=True)`
([build_snp_age_store.py:291](build_snp_age_store.py#L291)) before publishing. Deep mode
re-reads every CDF row and, when the transpose exists, compares it block by block against
`cdf_by_snp` ([snp_age_dataset.py:286-290](snp_age_dataset.py#L286-L290)). That roughly
doubles build I/O to re-derive what was just written in the same function.

**Fix:** default to the shallow check after a build, and expose deep validation as a
separate opt-in command for verifying a store that was moved or is suspected corrupt.

### B9 — `interval_assignment` is `uint8`

[sample_age_matched_syn.py:260](sample_age_matched_syn.py#L260) casts stratum assignments
to `uint8`. Correct for the default 20 strata; silently wraps if a caller passes custom
`probabilities` yielding more than 255 intervals.

**Fix:** `uint16`, or validate the interval count.

### B10 — Mislabeled diagnostic

[sample_age_matched_syn.py:344](sample_age_matched_syn.py#L344) reports
`len(rows_out) / max_proposals` as the "last acceptance rate". It is the overall rate.

### B1 — Non-representable store positions crash at the end of a run *(accepted risk, not fixing)*

`rows_to_native` rejects store positions that are non-integral or outside
`[1, length]`. Such positions pass every earlier gate — store construction, the `eligible`
flag, candidate loading, and all of sampling — and only fail in `generate_matches` at the
final coordinate-translation step ([sample_age_matched_syn.py:329](sample_age_matched_syn.py#L329)).

Verified end-to-end with a store containing a site at 100.5:

```
CRASHED at the final output step:
ValueError: store position 100.5 is not a 1-based VCF coordinate
```

All requested sets had already been accepted; the completed work was discarded.

**Decision (2026-08-03): not fixing as its own item.** The guard fires on store *contents*
rather than user input — i.e. on whatever `discover_positions` found in the ARG — but
ARGtest's ARGs come from msprime (`discrete_genome=True`) or SINGER-from-VCF, so site
positions are integers and this cannot arise.

**Superseded by B2.** The recommended one-line integrality assertion in `build_store` makes
this structurally impossible rather than merely unlikely, and is worth adding on that basis
alone. Note also that the *non-integer* half of this guard is the same defect as B2's
consequence 2 — a fractional site position is unaddressable through (chrom, POS) whether or
not it is ever selected. The `local position 0` half is B2's consequence 3.

---

## 2. Documentation

The step-by-step how-to in the README is accurate — the commands, flags, and listed output
files match the code. Gaps below.

- **D1 — No install step.** The README opens at `conda activate normalizeTE`. Nothing
  states `conda env create -f environment.yml`.
- **D2 — `environment.yml` is wrong in both directions.** `matplotlib` is declared and
  used nowhere in the project. `numpy` is used in every module but present only
  transitively via msprime/tskit. `pytest` is absent despite 33 tests. No documented way
  to run the tests.
- **D3 — `simulate_neutral_trees.py` is a dead end and undocumented.** It emits plain
  msprime tree sequences with no `chrom_offsets` metadata, so `build_snp_age_store.py`
  rejects its output with *"lacks top-level chrom_offsets metadata"*. This is the first
  thing a new user would try.
- **D4 — `snp_age_distribution.py` is undocumented and cannot read `.tsz`.** It uses
  `tskit.load` ([snp_age_distribution.py:67](snp_age_distribution.py#L67)) while the
  README directs users to `.tsz` inputs throughout.
- **D5 — The coordinate convention is documented wrongly** (see B2). The README says
  *"chromosome and 1-based VCF position"*; it is actually a chromosome-local **0-based
  tskit coordinate**, which is what ARGtest's `write_vcf` happens to print in the POS
  column. Needs an explicit statement: global = `offset + POS`; POS is the 0-based tskit
  coordinate within the chromosome; integer positions only; chromosome labels must match
  `chrom_offsets` exactly; **lists prepared from an external, genuinely 1-based VCF need a
  `-1`**. This is the highest-value documentation fix in the report — the current label can
  silently select the neighbouring SNP.
- **D6 — Undocumented flags:** `--omit-transpose`, `--checksums`,
  `--mutation-weighting`, `--bootstrap-reference`, `--bootstrap-batch-size`,
  `--syn-indices`, `--syn-mask`.
- **D7 — README step 3's snippet** uses `np.loadtxt(..., dtype=str, ndmin=2)`, which
  breaks on the `#` comments that `load_native_position_list` supports.
- **D8 — The top half of the README is still in future tense** — "will be stored", "will
  be written", "It then bootstrap the X TE SNPs". It reads as the design plan rather than
  documentation of shipped code.
- **D9 — Repo hygiene.** Nine `.bak` files are tracked in git: `README.md.bak`,
  `README.md.20260801.bak`, `README.md.howto.bak`, `README.md.pre-legacy-removal.bak`,
  `SNP_AGE_MATCHING_PLAN.md.bak`, `build_snp_age_store.py.bak`,
  `sample_age_matched_syn.py.bak`, `te_age_target.py.bak`, plus
  `SNP_AGE_MATCHING_PLAN.md` itself which is now partly stale against the code. Two more
  are untracked (`README.md.native-coordinates.bak`, `environment.yml.bak`). Git already
  provides this history. Worth revising the `.bak` rule in
  [AGENTS.md](AGENTS.md#L3) that produced them. `results/` also contains committed
  outputs (`.trees`, `.csv`).
- **D10 — `eligible` vs `valid` is not explained to users.** `--min-usable-fraction`
  silently makes SNPs ineligible; the distinction deserves a sentence in the README.

---

## 3. Over-engineering / streamlining

### O-a — Six duck-typing branches that exist only for the test doubles

- `_decode_cdfs` ([sample_age_matched_syn.py:58](sample_age_matched_syn.py#L58))
- `hasattr(store, "read_cdfs")` ([line 67](sample_age_matched_syn.py#L67))
- `hasattr(store, "read_boundary_cdfs")` ([line 90](sample_age_matched_syn.py#L90))
- `hasattr(store, "rows_to_native")` ([line 328](sample_age_matched_syn.py#L328))
- `getattr(store, "eligible", store.valid)` ([line 396](sample_age_matched_syn.py#L396))
- `getattr(store, "eligible", getattr(store, "valid", None))`
  ([te_age_target.py:236](te_age_target.py#L236))

Every one is a fallback for a store shape the real `SNPAgeDataset` never has. They exist
to accommodate `FakeStore` / `SpyStore` in the tests.

**Fix:** build a tiny real store in the tests (the `_write_ts` helper in
`tests/test_build_snp_age_store.py` already does most of it), or declare one `Protocol`,
and call the methods directly.

### O-b — Fallback chains with exactly one possible value

The threshold lookup at
[sample_age_matched_syn.py:431-434](sample_age_matched_syn.py#L431-L434) tries three
metadata keys; `te_age_target` only ever writes
`wasserstein_threshold_generations`. Plus B4.

### O-c — Three of eleven store arrays are pure functions of the others

- `usable_draw_fraction` == `usable_draw_count / n_posterior_draws`
- `eligible` == `valid & (usable_draw_count >= minimum_usable_draws)`
- `valid` == `usable_interval_count > 0`

`validate_store` then spends [lines 261-267](snp_age_dataset.py#L261-L267) recomputing
them to check they agree with themselves.

**Fix:** persist `usable_draw_count` and `usable_interval_count`; expose the rest as
computed properties on `SNPAgeDataset`.

### O-d — Two knobs for one threshold

`min_usable_draws` and `min_usable_fraction`, plus a mutual-exclusion check that is broken
(B3) and a three-line CLI expression working around it. Keep the fraction; let callers
compute a count if they need one.

### O-e — `quantile_order_statistic_interval`

Fifteen lines ([te_age_target.py:116-130](te_age_target.py#L116-L130)) producing
`threshold_monte_carlo_95_interval_generations`, which is written to metadata and used in
no decision anywhere in the codebase.

### O-f — `--bootstrap-reference two-sample`

An extra branch and an extra RNG draw in `bootstrap_wasserstein`
([te_age_target.py:94-98](te_age_target.py#L94-L98)). The documented workflow only ever
uses `observed`, and the option is not mentioned in the README.

### O-g — `_chromosomes` is 33 lines of validation

[build_snp_age_store.py:31-63](build_snp_age_store.py#L31-L63) checks integer-ness via
float round-trip, ordering by increasing offset, and extends-beyond-`sequence_length`. All
of these guard against a malformed upstream file that would be obvious immediately.
Presence, shape, unique names, and non-overlap is sufficient.

### O-h — Branches that cannot execute

- `if site is None: continue` ([build_snp_age_store.py:216](build_snp_age_store.py#L216))
  — `ts.site()` raises on a bad id; it never returns `None`.
- `_site_map`'s duplicate-position check
  ([build_snp_age_store.py:72](build_snp_age_store.py#L72)) — tskit rejects duplicate site
  positions when the tree sequence is constructed, so this is unreachable. The non-finite
  check is likewise unreachable.
- `_expand`'s `matches or [pattern]`
  ([build_snp_age_store.py:302](build_snp_age_store.py#L302)) — turns a mistyped glob into
  `"all tree-sequence inputs must exist"` with no filename. Better to report the pattern
  that matched nothing.
- `_weighted_choice`'s `min(..., weights.size - 1)` clamp
  ([sample_age_matched_syn.py:186](sample_age_matched_syn.py#L186)).
- `write_result`'s `if tmp.exists(): shutil.rmtree(tmp)`
  ([sample_age_matched_syn.py:355](sample_age_matched_syn.py#L355)) — the path is
  pid-qualified.

### O-i — `_site_map` decodes full `Site` objects to collect positions

[build_snp_age_store.py:66-82](build_snp_age_store.py#L66-L82) iterates `ts.sites()`,
which decodes ancestral state, metadata, and the mutation list for every site, then uses
only the dict keys. `ts.tables.sites.position` is the same data directly — and the main
extraction loop already uses exactly that
([line 203](build_snp_age_store.py#L203)).

### O-j — The blockwise weight machinery *(largest complexity win)*

`BlockWeightIndex`, `_BlockCache`, hierarchical block-then-SNP PPS sampling, cross-stratum
total decrementing, and the exhausted-block retry branch total roughly 150 lines
([sample_age_matched_syn.py:32-258](sample_age_matched_syn.py#L32-L258)) and are the
subtlest code in the repository.

The stated justification (module docstring, and README "Stratified synonymous sampling")
is avoiding an N-candidates by N-strata weight matrix. That matrix is:

| candidate pool | 20 strata, float32 |
|----------------|--------------------|
| 250,000        | 20 MB              |
| 2,000,000      | 160 MB             |
| 20,000,000     | 1.6 GB             |

Even the 20M worst case is unremarkable for an HPC node, and the README's own realistic
pool is well below it. Materializing the matrix reduces each stratum draw to a single
`rng.choice(p=...)` with straightforward without-replacement bookkeeping.

If the blockwise path is kept, note that `_BlockCache` is the real cost driver: it retains
8 blocks by default ([line 159](sample_age_matched_syn.py#L159)), so once the candidate
pool spans many blocks, most `cache.get` calls re-read and recompute boundary weights from
disk — once per drawn SNP, per proposal.

### O-k — `wasserstein_1` implemented twice

Validated version at [te_age_target.py:43](te_age_target.py#L43); unvalidated duplicate at
[sample_age_matched_syn.py:263](sample_age_matched_syn.py#L263). Import one.

### O-l — `load_position_list` is a one-line alias

[te_age_target.py:32-34](te_age_target.py#L32-L34) now just calls
`load_native_position_list`. Either drop it or keep it only if the indirection is load-bearing
for the tests.

### O-m — Unused imports

`asdict` in [te_age_target.py:11](te_age_target.py#L11); `Path` in
`tests/test_te_age_target.py:1`. (pyflakes reports no others.)

---

## 4. Suggested order of work

1. **B2 / D5** — correct the "1-based VCF" labeling and add the integrality assertion.
   Cheap, and it is the only item here that can silently produce a *wrong answer* rather
   than an error. Fixing it also subsumes B1 and settles the `rows_to_native` bounds.
2. **B5 + B6** — together these likely dominate build wall-clock at the documented scale.
3. **B3** — small, and the current CLI workaround is load-bearing.
4. **D9** — delete the nine tracked `.bak` files; **D1, D2** — fix `environment.yml` and
   add install plus test-running instructions.
5. **B4, B7, B8, B9, B10** — small, independent correctness and clarity fixes.
6. **O-a, O-b, O-c, O-h, O-i, O-k, O-m** — mechanical cleanups, low risk.
7. **O-j** — the biggest simplification, but do it last, after the correctness items and
   with the existing sampler tests as a reference for behaviour.

### Verification note

Every empirical claim in this report was executed against the working tree in the
`normalizeTE` conda environment (tskit 1.0.3): the `_load` call counts (B5), the `ts.at()`
scaling table (B6), the `min_usable_draws` failure (B3), the `write_vcf` position transform
and the three coordinate results (B2), and the end-of-run crash (B1). The candidate-matrix
sizes in O-j are arithmetic, not measurements. The unexecuted judgement calls are: whether
the O-j rewrite preserves the sampler's statistical behaviour, and whether any real ARGtest
ARG contains a fractional site position — the B2 assertion is recommended precisely because
it converts that open question into an enforced invariant.
