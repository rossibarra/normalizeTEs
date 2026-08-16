# Code Review Round 6 — Φ-SFS (`phi_sfs.py`)

**Scope:** `phi_sfs.py`, `tests/test_phi_sfs.py`, `PHI_SFS_IMPLEMENTATION_PLAN.md`,
README §6.
**Reviewers:** Claude (Opus 5) and Codex (gpt-5.6), two independent passes
followed by a reconciliation round. Every item below is agreed by both unless
explicitly marked.
**Status:** all findings resolved. Item 2 was rejected by the maintainer as
described below; everything else was implemented and is covered by tests. See
`PHI_SFS_IMPLEMENTATION_PLAN.md` §12 for what remains deliberately deferred.

---

## 0. What was verified as correct

This section is deliberate: the statistical core is sound, and none of the
findings below say otherwise.

- `hypergeometric_projection` was checked against `scipy.stats.hypergeom.pmf`
  for `n ∈ {20, 21, 25, 50, 200, 5000}` × `k ∈ {0, 1, 2, n/3, n/2, n-1, n}`.
  **Zero mismatches at `atol=1e-12`.** The log-gamma formulation is also stable
  at `n = 2×10⁶` (sums to 1.0 exactly in double precision).
- Support bounds `lower = max(0, m-(n-k))`, `upper = min(m, k)` are correct;
  bins outside the support are correctly left at zero.
- Accumulation is order-invariant (verified numerically).
- The no-site-level-renormalization rule is implemented correctly: `raw +=
  values[1:20]` keeps `1 - h₀ - h₂₀` per site, exactly as the plan specifies.
- Φ is confirmed to equal the **total variation distance** between the two
  normalized spectra, and to equal `1 - Σⱼ min(tⱼ, s_rⱼ)`.
- Polarization arithmetic is right: `k = alt_count` when REF is ancestral,
  `k = n - alt_count` when ALT is ancestral.
- `positions.npy` (from `store.rows_to_native`) and `te_positions.npy` (from
  `te_vcf_positions`) are both **native VCF positions**, not global offsets, so
  the coordinate join against the VCF is correct.
- Output publication is atomic in the same style as the rest of the repo.
- All 6 existing tests pass.

---

## 1. CRITICAL — the matches bundle is never verified to belong to the target

`phi_sfs.py:221-227`

`_validate_provenance` compares only `source_store_content_sha256` and
`source_catalog_sha256`. Both are **store** identities. Every target built from
the same SNP store shares them. So a matched-control bundle generated for
`in_gene` can be run against the `all_te` target and will pass validation
silently, producing a scientifically invalid Φ with fully "valid-looking"
provenance in `metadata.json`.

This is the one finding that can publish wrong numbers with no error.

**Fix — and it costs nothing.** The matcher already writes the right key.
`sample_age_matched_controls.py:447` computes:

```python
target_digest = _sha256_arrays(target_rows, target_cdf, age_bins,
                               np.asarray([threshold], dtype=np.float64))
```

All four inputs live in the target directory: `te_row_indices.npy`,
`target_cdf.npy`, `age_bins.npy`, and
`metadata["wasserstein_threshold_generations"]`. So `phi_sfs.py` can recompute
the digest from the target directory alone — **no interval store, no new
dependency** — and require it to equal `matches/metadata.json["target_digest"]`.

Reuse `_sha256_arrays` rather than reimplementing it, so the two stay in lock
step. Also check `matches/metadata.json` records completeness.

Corollary: once `target_digest` is checked, the existing store-hash comparison
is nearly redundant — the matcher already validated store identity against the
target at match time (`sample_age_matched_controls.py:420-425`). Keep it, but
`target_digest` is the check that was actually missing.

---

## 2. ~~HIGH~~ REJECTED — "biallelic" is enforced only as "no comma in ALT"

> **Maintainer decision: not implemented.** Multiallelic records are already
> dropped upstream, so the biallelic property is a property of the input rather
> than something this step should re-derive. The assumption is instead
> documented in README §6 ("Site assumptions") and recorded as `biallelic_policy`
> in the output `metadata.json`. The pre-existing comma-in-ALT check is
> retained. The original finding is kept below for the record.

`phi_sfs.py:154-156`

`if "," in alt` is the entire multiallelic guard. It accepts `ALT="."`
(monomorphic), empty ALT, symbolic alleles (`<DEL>`, `<NON_REF>`), and breakend
syntax (`G]chr2:456]`), all of which are then treated as an ordinary binary SNP
and polarized as though `alt` were a real base. At a requested coordinate this
silently analyzes the wrong mutation class.

**Fix:** reject `alt` in `(".", "")`, and reject any ALT containing `<`, `[`, or
`]`. Raise with the coordinate, consistent with the other site errors.

**Deliberately not doing:** requiring single-base REF/ALT. Neither reviewer
confirmed that the upstream store is SNP-only, and a hard indel rejection could
kill a legitimate run. If the store *is* SNP-only, tighten this later and say so
in the plan.

---

## 3. MEDIUM — bundle row indices are ignored

`phi_sfs.py:188-205`

`_load_coordinates` reads the coordinate arrays but never loads
`te_row_indices.npy` or `row_indices.npy`, so a stale or independently corrupted
`positions.npy` / `chromosome_codes.npy` changes which SNPs are analyzed while
provenance still looks clean. The plan's integration contract (§3.1, §4)
assumes these arrays are used.

**Fix (right-sized):** load both, require their shapes to equal the
corresponding coordinate arrays, require non-negative integers, and reject
within-set duplicate rows (the matcher already guarantees this at
`sample_age_matched_controls.py:204-206`, so a violation means corruption).

**Deliberately not doing:** re-resolving coordinates against the interval store.
That would pull a heavy dependency into a step that is otherwise store-free, for
a failure mode the shape/duplicate checks already cover. *(Codex initially
proposed full re-resolution; converged on the cheap form.)*

---

## 4. Efficiency

These are the items that matter for a production VCF. Benchmarks below are
measured on this machine with a 500-sample VCF line.

### 4a. MEDIUM — every VCF line is fully split before the coordinate test

`phi_sfs.py:141-151`

`fields = raw.rstrip("\n").split("\t")` allocates every sample field for every
record, then line 150 discards ~all of them.

| approach | per line | 30M-line VCF |
|---|---|---|
| full `split("\t")` | 7.8 µs | ~230 s |
| bounded `split("\t", 2)` | 0.15 µs | ~5 s |

**~50× on this step, and it scales with sample count.**

**Fix:** `chrom, _, rest = raw.split("\t", 2)`, take POS off the front of
`rest`, test membership, and only then split the remainder. Keep the existing
`len(fields) < 10` validation on the requested-record path.

### 4b. MEDIUM — per-sample regex genotype parsing

`phi_sfs.py:110-124`

`re.split(r"[/|]", gt)` plus `int()` per sample per requested site. Genotype
strings are drawn from a tiny alphabet (`0/0`, `1/1`, `0|0`, `1|1`, `0`, `1`,
`./.`, `.`), so this is pure repeated work.

| approach | 1M genotypes |
|---|---|
| current `re.split` | 0.40 s |
| dict memo on the raw GT string | 0.04 s |

**~10×.** At 1M requested sites × 500 samples that is ~200 s → ~20 s.

**Fix:** module-level cache keyed on `(heterozygous_policy, gt)` → allele or
`None`. Cache only *successful* parses; keep the error paths uncached so the
`chrom:pos` in the message stays accurate.

### 4c. MEDIUM — spectrum accumulation is a Python loop, 100× over

`phi_sfs.py:255-289`

`eligible()` rebuilds a list of 21-element arrays per replicate, and
`normalized_spectrum` adds them one at a time **with a shape check per site**.
For 100 replicates × N sites that is 100N array additions and 100N validations.
Line 289 then re-loops the same projections to recompute endpoint mass.

The plan already prescribes the fix (§4, last paragraph): *"Resolve and project
unique SNP store rows once, then gather their projection vectors."*

**Fix:** after reading counts, build once
- `P`: `(n_unique, 19)` matrix of retained projections,
- `endpoints`: `(n_unique,)` vector of `h₀ + h₂₀`,
- a `SiteCount → row` index map.

Then per replicate: `P[idx].sum(axis=0)` and `endpoints[idx].sum()`. This
removes the inner Python loop, the repeated shape validation, and the duplicate
endpoint pass in one change.

### 4d. MEDIUM — the VCF is read twice

`phi_sfs.py:208-213, 366`

`_sha256(args.vcf)` re-reads the entire file after the scan, doubling I/O on a
multi-GB compressed VCF.

**Fix:** hash the raw bytes during the single scan pass (wrap the file object
and update the digest as blocks are read), or accept a validated sidecar digest.

---

## 5. Correctness and provenance gaps

### 5a. MEDIUM — missing provenance disables validation instead of failing

`phi_sfs.py:224-226` compares hashes only when *both* sides are non-`None`. A
hand-built or truncated bundle with no store identity at all passes silently.

**Fix:** require the fields for the schemas you support; fail clearly when
absent.

### 5b. MEDIUM — output metadata breaks the repo's provenance convention

`phi_sfs.py:355-378`

`te_age_target.py:399`, `sample_age_matched_controls.py:450`, and
`distributed_age_match.py:126` all record `release_provenance.software_provenance()`;
the matcher also records `creation_command` and `numpy_version`. `phi_sfs.py`
imports none of it.

The result: **the only output in the chain with no git provenance is the one
carrying the final statistic.**

**Fix:** import `software_provenance` and record it plus `creation_command`,
`numpy_version`, and a creation timestamp, matching the matcher's metadata
block.

### 5c. MEDIUM — retained/endpoint mass reported as absolute, not fraction

`phi_sfs.py:289-305, 354-375`

Plan §7 requires *"the retained fractions for the target and every matched set
must be reported"*, because final normalization hides exactly these
differences. The CSV reports absolute masses, which are not comparable across
replicates with different eligible-site counts.

`retained_mass + endpoint_mass == eligible_sites` holds exactly (verified), so
this is a two-line addition: `retained_fraction` and `endpoint_fraction`.

### 5d. LOW — compression detected by suffix only

`phi_sfs.py:96-97` — `.bgz` / `.bgzf` (BGZF, gzip-compatible) are opened as
plain text and fail confusingly. Recognize those suffixes or sniff the gzip
magic bytes.

### 5e. LOW — ancestral INFO matching is case-sensitive, policy undocumented

`phi_sfs.py:157-169`. Under the 1000G convention a lowercase `AA` value encodes
a *low-confidence* call, so this currently hard-fails on such files.

**Resolution:** do **not** silently `.upper()` — that would quietly promote
low-confidence polarizations into the analysis. Document the policy explicitly
and either reject lowercase with a clear message or add an opt-in flag to
accept and normalize. *(Claude initially proposed unconditional `.upper()`;
Codex's objection is the scientifically safer call and was adopted.)*

### 5f. LOW — no progress output on a multi-hour scan

`phi_sfs.py:137-185` scans the whole VCF silently. Print records scanned
periodically, as the other long-running scripts here do.

### 5g. LOW — Φ identity diagnostics computed twice

`phi_sfs.py:89-90` computes reverse-positive and half-L1 inside `phi_sfs()`,
then `phi_sfs.py:282-283` recomputes both for the CSV. Return them from the
single calculation.

**Resolved disagreement:** Claude proposed deleting the identity assertion
(`:91-92`) and the `isinstance` guard (`:36-37`) as unreachable — the identity
is algebraic given validated inputs, not contingent. Codex argued to keep both
as cheap tripwires protecting direct callers and future refactors. **Agreed:
keep them, remove only the duplication.** Cost is 100 evaluations per run, i.e.
nothing.

---

## 6. Documentation

### 6a. LOW — README should name the statistic

`README.md:359-380`

The math is correct but under-explained. Add:

- **Φ-SFS is the total variation distance** between the two normalized
  projected spectra. Naming it lets a reader connect it to a large existing
  literature instead of treating it as a bespoke quantity.
- The third equivalent form, `Φ = 1 - Σⱼ min(tⱼ, s_rⱼ)`, which makes the
  "non-overlap" wording in the plan literal.
- That `Φ ∈ [0, 1]` and is **symmetric**, even though the stored bin residuals
  are oriented TE-minus-SNP.
- That normalization **discards** differences in eligible-site count and total
  retained polymorphic mass — so the retained fractions from §5c must be
  inspected separately. This is the interpretive trap most likely to bite.

Also add the `= Σⱼ max(s_rⱼ - tⱼ, 0)` form, which the plan has (§2.4) but the
README drops.

### 6b. LOW — module docstrings are too terse for the trust level

`phi_sfs.py:34-35, 63-64, 77-78`. The plan carries the full derivation; the
functions carry one line each. Someone reading `hypergeometric_projection` in
isolation cannot see the `n ≥ 20` rule, the bins-0/20 exclusion, or the
no-renormalization decision. Lift the key statements from the plan into the
docstrings.

### 6c. LOW — the plan overstates what is implemented

`PHI_SFS_IMPLEMENTATION_PLAN.md:145-146, 210-212, 269-324`. The plan asserts
target-identity validation, unique-row gathering, retained fractions, rejection
diagnostics, and command/dependency/time metadata — none currently present.
Either implement them (items 1, 3, 4c, 5b, 5c do most of it) or mark the
remainder explicitly deferred. As written the plan gives a reader false
confidence.

---

## 7. Tests

`tests/test_phi_sfs.py` — LOW, but the gap is wide relative to plan §8.

Current tests assert shapes and ranges; `test_end_to_end` never checks a single
Φ value numerically, and `test_info_ancestral_and_heterozygous_policy` only
asserts that heterozygosity raises — it never verifies the reversed
polarization it sets up.

Worth adding, roughly in priority order:

1. A hand-calculated end-to-end Φ, so a silent numerical regression is caught.
2. Reversed polarization (`AA=G` with a known `k`) checked numerically.
3. `n = 19 / 20 / 21` boundary cases around the eligibility filter.
4. Mismatched `target_digest` is rejected (covers item 1).
5. Corrupted row/coordinate shape alignment is rejected (covers item 3).
6. Invalid ALT forms: `.`, `<DEL>`, breakend (covers item 2).
7. Zero retained mass in a replicate fails rather than emitting a score.
8. Monte Carlo cross-check of the projection against random downsampling
   (plan §8.1) — cheap, and the strongest single guard on the core math.

---

## 8. Explicitly rejected

Both reviewers agreed to **drop** these rather than leave them as open items.

- **Concurrent-writer hardening / fsync of the output directory.**
  (Codex's original MEDIUM.) The `mkdtemp` + `os.replace` pattern here is
  identical to `te_age_target.py:299-330` and the matcher. Hardening this one
  script alone makes the codebase inconsistent, for a race the manifest design
  already forbids by requiring unique output directories.

- **Enforcing `FILTER == PASS` by default.**
  (Codex's original MEDIUM.) The declared input is the already-filtered SINGER
  preprocessing VCF. Skipping non-PASS records would reclassify them as
  "requested sites absent from the VCF" and kill the run with a misleading
  error. **Document that FILTER is ignored**; add an opt-in `--require-pass`
  only if a real need appears, and have it report filtered sites distinctly.

- **Requiring single-base REF/ALT.** See item 2.

---

## 9. Suggested order of work

| # | Item | Why now |
|---|---|---|
| 1 | Target-digest validation (§1) | Blocker — silent wrong answers |
| 2 | ALT validation (§2) | Blocker — silent wrong site class |
| 3 | Row-index checks (§3) | Cheap, closes the alignment gap |
| 4 | Bounded split + GT memo (§4a, §4b) | ~50× and ~10× on the hot loop |
| 5 | Unique-projection matrix (§4c) | Removes the 100× Python loop |
| 6 | Provenance + retained fractions (§5b, §5c) | Needed to interpret results |
| 7 | Single-pass hash (§4d) | Halves I/O |
| 8 | README + docstrings (§6) | Before anyone else reads the output |
| 9 | Tests (§7) | Locks all of the above in |

Items 1–3 are the launch blockers. Items 4–5 are the difference between a scan
that takes minutes and one that takes hours. Items 6–9 are quality, not
correctness, but item 6 is what makes the numbers interpretable.

---

## 10. Verification of the fixes

After implementation, Codex re-reviewed the rewritten `phi_sfs.py` and
`tests/test_phi_sfs.py` against this list. It confirmed every agreed item as
fixed and found no new numerical or digest bug — specifically clearing the
`_HashingStream → BufferedReader → GzipFile → TextIOWrapper` chain, the bounded
split field offsets, the genotype cache under both heterozygous policies, and
the `bincount`-weighted accumulation against the previous per-site summation.

Three further defects it did find were fixed in turn:

1. Row indices and positions were cast with `.astype(np.int64)` *before*
   validation, so a float array storing `2.9` silently truncated to `2` and
   resolved to the wrong site. Loading now rejects a non-integer dtype
   outright (`_load_integers`). Verified that every upstream producer already
   writes int64 or uint16, so this rejects corruption without blocking a real
   bundle.
2. Progress counted physical lines, including headers, while calling them
   "records".
3. Two tests could not have failed: the retained/endpoint fractions were only
   exercised on a fixture where eligible and input counts were equal, so
   dividing by the wrong denominator would have gone unnoticed, and no test
   covered non-integer or negative row indices.

### Independent checks run

- `hypergeometric_projection` against `scipy.stats.hypergeom.pmf` over
  `n ∈ {20, 21, 25, 50, 200, 5000}` × 7 values of `k` — zero mismatches at
  `atol=1e-12`; stable and normalized at `n = 2×10⁶`.
- Old versus new implementation on a synthetic 86 MB, 200,000-record,
  200-sample VCF with 20 matched sets: **5.3× faster**, and every published
  array identical to within `1e-13` (differences are float-association noise
  at the `1e-16` level).
- Accumulation alone at production shape, 100 sets × 10,000 sites: **8×**.
- Full test suite: 144 passed, 1 skipped. `pyflakes` clean.

### Follow-up: how strict the store-hash check should be

Codex's verification pass noted that `source_catalog_sha256: null` in both
bundles satisfies the equality test vacuously. Investigating that turned up a
larger problem with the first implementation of item 5a, which required
`source_store_content_sha256` to be non-null.

The two hashes are not independent. `catalog_sha256`
(`build_snp_interval_store.py:644`) is `sha256` of the positions array alone;
`content_sha256` (`snp_interval_dataset.py:41`) hashes every declared array —
positions among them — plus the semantic metadata keys. Content strictly
subsumes catalog, so requiring catalog to be non-null adds no discriminating
power.

More importantly, the dense store (`build_snp_age_store.py`,
`snp_age_dataset.py`) records **neither** digest, so `te_age_target.py` writes
both as `null` for a dense-store target. There were therefore only two
reachable cases: an interval store, where both hashes exist and the leniency
never engaged; and a dense store, where the non-null requirement rejected the
bundle outright. That made `phi_sfs.py` stricter than every other step —
`sample_age_matched_controls.py:420-427` compares these hashes only when a
value exists — so it would have been the one step to reject a bundle the rest
of the pipeline produced and accepted.

**Resolved:** the non-null requirement was removed. Both keys must still be
*present* in both bundles and must agree, which is what item 5a was actually
aimed at — a hand-built or truncated bundle carrying no store identity at all.
`target_digest` remains unconditional, so the target binding from item 1 is
unaffected. Two tests cover the dense-store shape: null hashes are accepted,
and a mismatched target is still rejected when they are null.
