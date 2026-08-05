# normalizeTE — code review, round 2

Follow-up to the round-1 review in [CODE_REVIEW.md](CODE_REVIEW.md), assessing the fixes
made in response to it.

> **All findings in this document were resolved in `063daa6` and `fe7152c`, verified
> 2026-08-04.** `R1 §2.2` (`side="left"`), `R1 §2.3` (multi-chromosome test), `N1a`
> (vectorised `nodes.time[edges.parent].max()`), `N3`, `N4`, and `D4` are all fixed; 41 tests
> pass. Resolution detail is at the end of this document. The `O-*` items in §5 remain open by
> design.

**Reviewed:** commits `0b4efee`, `a8f4fda`, `444d6d3`, plus uncommitted working-tree changes
to `README.md`, `build_snp_age_store.py`, and `tests/test_build_snp_age_store.py`.

**Test suite:** 39 passed (33 at round 1). Several of the new tests are regression tests for
round-1 findings.

**Method:** every claim below was executed against the working tree in the `normalizeTE`
conda environment (tskit 1.0.3, msprime). Reproduction snippets are in the appendix.

---

## Verdict at a glance

| | |
|---|---|
| Round-1 findings fixed and verified | 21 |
| Round-1 findings I now **retract** as incorrect | 1 (`B2`'s coordinate reading) |
| Bugs still live | **1** — a one-word `searchsorted` defect, [snp_age_dataset.py:134](snp_age_dataset.py#L134) |
| New issues from the rewrite | 1 open trade-off (`N1a`), 2 minor (`N3`, `N4`) |
| Deliberately deferred | the `O-*` simplifications, `D4` |

The builder rewrite is a real improvement. The one live bug crashes a completed run at its
final step on entirely valid data, and no test can currently catch it because every fixture
in the suite declares a single chromosome.

---

## 1. Fixes I agree with

| # | Round-1 finding | Fix | Assessment |
|---|---|---|---|
| B5 | Tree sequences reloaded once per SNP block | `inspect_inputs` merges position discovery, age grid, and chromosome validation into one pass; extraction loads each draw once | Verified 2 loads per draw. `test_builder_loads_each_draw_twice_independent_of_blocks` pins it *independent of `block_snps`* — the right invariant, better than asserting a count. |
| B6 | `ts.at(position)` per site | Single `ts.trees()` / `tree.sites()` walk | Removed the superlinear pattern. See `N1a` for the age-grid half, which has since been revisited. |
| B3 | `min_usable_draws` unreachable via Python API | `min_usable_fraction` now defaults to `None` | Verified working; `test_minimum_usable_draws_python_api` guards it. |
| B4 | `interval_boundaries.npy` fallback fed ages as edge indices | Now raises `FileNotFoundError` | Correct — the fallback could only ever be wrong. |
| B7 | `quantization_scale` metadata read then ignored | Both readers use `self.quantization_scale`; `validate_store` validates the value *and* uses it for the terminal check | Went further than I proposed. Test decodes at `scale=1000`. |
| B8 | Full deep re-validation on every build | `validate_store(temp, deep=False)` | |
| B9 | `interval_assignment` was `uint8` | `uint16` | |
| B10 | "last acceptance rate" mislabeled | "overall acceptance rate" | |
| B2 | *(coordinate reading retracted — see §2)* | **Integrality assertion added** to `inspect_inputs` and `discover_positions` | This was the actionable half of B2 and it closes the whole fractional-position class, including all of `B1`. |
| N2 | Scalar `np.searchsorted` per site per draw | Hoisted to one vectorised `np.searchsorted(positions, ts.tables.sites.position)` per draw, indexed by `site.id`; the validity check is vectorised too | Exactly the fix. Measured 1.22 µs → 0.071 µs amortised (**17x**), ~0.7 h saved at 20M sites × 100 draws. |
| O-h | Unreachable branches | `_site_map` and `if site is None` deleted | |
| O-i | `_site_map` decoded full `Site` objects | Uses `ts.tables.sites.position` | |
| O-k | `wasserstein_1` duplicated | Imported from `te_age_target`; no import cycle | |
| O-m | Unused `asdict` | Removed | |
| D1 | No install step | `conda env create -f environment.yml` documented | |
| D2 | `environment.yml` wrong both ways | `numpy` + `pytest` added, unused `matplotlib` removed | |
| D3 | `simulate_neutral_trees.py` a dead end | Caveat documented | |
| D6 | Undocumented flags | All now documented, with a `--help` pointer | |
| D8 | README in future tense | Corrected | |
| D9 | 9 `.bak` files tracked | Deleted, along with the stale plan doc; `*.bak` gitignored | |
| D10 | `valid` vs `eligible` unexplained | Explained, plus an unprompted scratch-space note | |

Tests were **strengthened, not weakened**. The fractional fixture positions (`10.5`, `7.25`)
became integers only because the new integrality assertion rejects them — a legitimate
consequence of the fix, not a loosened expectation.

---

## 2. R1 — Retraction, and the one bug that survives

### 2.1 Retraction

Round 1 argued that the repo's `1 <= POS <= length` window and its "one-based" rationale
were both wrong. **That was my error.** I inferred the convention from ARGtest's `write_vcf`
call — no `position_transform`, so tskit's default `np.round` applies — and concluded its
ARGs carried 0-based coordinates. The maintainer has since confirmed that **ARGtest's ARGs
and its VCFs are both 1-based.**

Under that convention chromosome *c* owns global positions `(offset, offset + length]`, and
everything the repo does is right:

- `global = offset + POS` — correct.
- `1 <= value <= length` — correct.
- `native[i] = value`, no `±1` — correct.
- The README's *"The ARGs used by this workflow store one-based positions internally"* and
  the matching code comment — correct.
- Global position 0 belongs to no chromosome, so `rows_to_native` refusing it is correct
  behaviour, not a crash risk. The `B1` class is fully closed by the integrality assertion.
- `test_native_coordinates_cover_first_and_last_base` asserting `POS 100 → global 100.0` on
  a length-100 chromosome — correct.

`444d6d3` was the right change; `a8f4fda`'s `offset + POS - 1` was the wrong one.

### 2.2 The surviving bug: wrong `searchsorted` side in the chromosome lookup

[snp_age_dataset.py:134](snp_age_dataset.py#L134), in `rows_to_native`:

```python
choices = np.searchsorted(offsets, globals_, side="right") - 1
```

Because chromosome *c* owns `(offset, offset + length]`, a global position equal to the
**next** chromosome's offset still belongs to *c*. `side="right"` assigns it to *c+1*, and
the (correct) `1 <= value <= length` check then rejects it as `value == 0`.

Verified on a properly 1-based two-chromosome store — chr1 owning globals `[1, 100]`,
chr2 owning `[101, 200]`, `sequence_length = 201`:

```
global    1.0 -> chr1   1   OK
global   50.0 -> chr1  50   OK
global  100.0 -> CRASH: store position 100 is not a 1-based VCF coordinate   (should be chr1 100)
global  101.0 -> chr2   1   OK
global  150.0 -> chr2  50   OK
global  200.0 -> chr2 100   OK
```

The lookup itself, isolated:

```
g = 100.0    searchsorted(offsets, g, 'right') - 1 = 1   WRONG
             searchsorted(offsets, g, 'left')  - 1 = 0   OK
```

**Scope.** The last base of every chromosome except the final one — 9 positions in a
10-chromosome ARG. The final chromosome is unaffected because no higher offset exists.

**Impact.** `native_to_global` looks up by name and is unaffected, so the round trip is
asymmetric: `native_to_global("chr1", 100)` returns `100.0`, but `rows_to_native(100.0)`
raises. `rows_to_native` is called once, at the very end of `generate_matches`
([sample_age_matched_syn.py:323](sample_age_matched_syn.py#L323)), after all sampling has
succeeded. So a single such SNP landing in any accepted synonymous set discards the entire
completed run — the `B1` failure mode, on genuinely valid data.

**Fix.** One word: `side="right"` → `side="left"`. Verified — the round trip becomes a total
identity across the whole coordinate space, and global 0 is still correctly refused by the
existing `choices < 0` guard:

```
global 100.0 -> chr1 100   OK
full round-trip identity: True
global   0.0 -> store position precedes the first chromosome offset
```

### 2.3 Why no test caught it — the more useful finding

Every fixture in the suite declares exactly one chromosome. The only two `chrom_offsets`
tables are `tests/test_build_snp_age_store.py:17` and the one in
`tests/test_snp_age_dataset.py`, both single-entry:

```
$ grep -rn "chrom_offsets" tests/
tests/test_build_snp_age_store.py:17:  "chrom_offsets": [{"chrom": "chr1", "offset": 0, "length": 100}]
```

With one chromosome there is no higher offset, so the last base resolves correctly and this
bug cannot manifest. **The entire multi-chromosome path is unexercised** — which matters more
than the one-word fix, since that path is the whole point of the native-coordinate
interface.

Worth adding: a two-chromosome fixture that exercises `rows_to_native` at each chromosome's
**first and last** base and asserts round-trip identity against `native_to_global`.

---

## 3. N1 — The accumulator, and the age-grid trade-off

### 3.1 Documented (good)

`build_store` allocates a temporary `(n_snps, n_age_bins)` float32 disk memmap and updates
one row per `(site, draw)` pair. `--block-snps` does not bound it.

> **Correction (2026-08-05).** Round 2 described these as *random* row writes. That was
> wrong. `tree.sites()` yields sites in increasing position order and rows follow the sorted
> position union, so each draw sweeps the accumulator **forward, sequentially** — verified:
> row indices are strictly increasing within a draw. The README's "sweeps this accumulator in
> genomic order" is the accurate description. The cost is therefore `n_draws` sequential
> passes, not random I/O, which is a far better access pattern than I credited. The README now states the size formula and the magnitudes, and recommends
node-local scratch:

| n_snps | n_age_bins | accumulator (float32) |
|---|---|---|
| 20M | 200 | ~15 GiB |
| 20M | 1000 | ~75 GiB |

That is the right disclosure. Two notes remain.

### 3.2 N1a — The age-grid bound was reverted, at the cost of a second full tree traversal

Round 1 suggested `nodes.time.max()` for the age grid (O(1), and a valid upper bound). That
was adopted, then reverted in the working tree because an unrelated ancient node inflates
`n_age_bins` and therefore the accumulator. `inspect_inputs` now performs a **full
`ts.trees()` × sites × mutations walk** to find the exact maximum mutation-parent time, with
`test_age_grid_ignores_ancient_nodes_not_bounding_mutations` guarding the case.

**The concern is legitimate** — an isolated node in no marginal tree can inflate the bound
arbitrarily. But the builder now pays two full tree traversals per draw (one in
`inspect_inputs`, one in extraction), and there is a vectorised option with the same
protection:

```python
maximum = float(nodes.time[edges.parent].max())
```

Any mutation's tree-parent must appear as an edge parent, so this is a valid upper bound, and
it excludes isolated nodes by construction. Measured three ways:

| Bound | Their isolated-node fixture | Dense real ARG (58,559 sites / 12,364 trees) | Cost on that ARG |
|---|---|---|---|
| `nodes.time.max()` | 1,000,000 — inflated | — | ~0.1 ms |
| `nodes.time[edges.parent].max()` | **100** | **182,507** | **0.1 ms** |
| Exact mutation-parent walk (current) | 20 | 182,507 | 231 ms |

On a realistic ARG the vectorised bound is **identical** to the exact one at **1714x** lower
cost, while still collapsing the isolated-node case from 1,000,000 to 100.

**Recommendation:** use `nodes.time[edges.parent].max()` and drop the second traversal. Note
that `test_age_grid_ignores_ancient_nodes_not_bounding_mutations` currently asserts the
*exact* value (`== 20`), which over-constrains the implementation; it would be better phrased
as "the grid is not inflated by the ancient node" (e.g. asserting the last bin is far below
1,000,000).

**Strictly better, if the tight bound matters:** allocate the accumulator with the cheap
vectorised bound, then truncate the age grid to the last bin carrying any mass before
quantizing. The persisted store — `cdf_by_snp` and `cdf_by_age`, both `n × b` — then has an
exactly tight `b`, and only the transient accumulator is slightly oversized. As it stands,
trailing all-zero-mass bins are stored for every SNP in both layouts.

---

## 4. Remaining minor items

**N3 — `missing_draw_count` decrements a uint32 memmap.**
[build_snp_age_store.py:233](build_snp_age_store.py#L233). Safe today: tskit forbids
duplicate site positions, so a row is decremented at most once per draw. A second decrement
would wrap to 4294967295 rather than going negative. `validate_store`'s
`present + missing == draws` check would catch it, but the invariant the arithmetic depends
on deserves a comment.

**N4 — Leftover CLI scaffolding.**
[build_snp_age_store.py:336-338](build_snp_age_store.py#L336-L338) still computes
`min_usable_fraction=(0.1 if both are None else ...)`. Now that `build_store` resolves the
`0.1` default itself, the CLI can pass both arguments straight through. Harmless residue of
the `B3` fix.

**D4 — The one documentation item still outstanding.** `snp_age_distribution.py` uses
`tskit.load` (was [snp_age_distribution.py:67](snp_age_distribution.py#L67)), so it cannot read
the `.tsz` inputs the README now standardises on, and it remains undocumented as a tool.

---

## 5. Deferred by design

Not addressed, consistent with round 1 ordering them last:

`O-a` six duck-typing store branches · `O-b` three-key threshold chain · `O-c` redundant
derived arrays (`eligible`, `usable_draw_fraction`) · `O-d` two coverage knobs · `O-e` unused
`quantile_order_statistic_interval` · `O-f` `two-sample` bootstrap reference · `O-g`
`_chromosomes` validation size · `O-h` remnants (`_expand`'s `matches or [pattern]`,
`_weighted_choice`'s clamp, `write_result`'s `tmp.exists()`) · `O-j` the blockwise weight
machinery · `O-l` `load_position_list` alias.

`O-j` remains the largest available simplification and is still best done last.

---

## 6. Priority

1. **R1 §2.2** — `side="right"` → `side="left"`, plus the two-chromosome fixture from §2.3.
   The only live bug; it destroys completed runs on valid multi-chromosome data.
2. **N1a §3.2** — swap the new full traversal for `nodes.time[edges.parent].max()`;
   optionally truncate the grid after accumulation.
3. **N3, N4, D4** — small and independent.
4. **`O-*`** — quality cleanups whenever convenient.

---

## Appendix — reproducing the checks

**R1 (§2.2), the coordinate bug.** Build a store from a 1-based two-chromosome tree
sequence: `chrom_offsets = [{chr1, 0, 100}, {chr2, 100, 100}]`, `sequence_length = 201`,
sites at globals `1, 50, 100, 101, 150, 200`. Call `rows_to_native` on every row; global
`100.0` raises. Re-run with `side="left"` substituted in the `choices = ...` line and all six
rows resolve, with `native_to_global(*rows_to_native(rows))` equal to `store.positions[rows]`.

**§2.3, the test gap.** `grep -rn "chrom_offsets" tests/`.

**N1a (§3.2), the three age-grid bounds.** Take the fixture from
`test_age_grid_ignores_ancient_nodes_not_bounding_mutations` and compare
`nodes.time.max()`, `nodes.time[edges.parent].max()`, and a full mutation-parent walk. Then
repeat on `msprime.sim_ancestry(samples=25, ploidy=2, population_size=10_000,
sequence_length=8_000_000, recombination_rate=1e-8, random_seed=1)` with
`sim_mutations(rate=4e-8, random_seed=2)` and time both.

**N2 (already fixed), the searchsorted saving.** Time `np.searchsorted` scalar-per-call in a
Python loop against one vectorised call over the same 20,000 queries into a 20M-element
sorted array.

---

## 7. Resolution — verified 2026-08-04 (`063daa6`, `fe7152c`)

All six round-2 findings fixed. 41 tests pass (39 at the time of writing).

| # | Fix | Verified |
|---|---|---|
| `R1 §2.2` | `side="right"` → `side="left"`, with a comment stating why a position equal to the next offset is the preceding chromosome's final base | On a 1-based two-chromosome store (chr1 `[1,100]`, chr2 `[101,200]`), all six boundary positions resolve correctly and `native_to_global(*rows_to_native(rows))` is an exact identity. A TE list using `chr1 100` / `chr2 100` now runs end to end; results round-trip to the same store rows. |
| `R1 §2.3` | `test_multichromosome_boundaries_round_trip` added — two chromosomes, each one's first *and* last base, plus round-trip assertion | Closes the gap that let the bug through. |
| `N1a` | Full traversal replaced with `nodes.time[edges.parent].max()`, matching the recommendation, with the reasoning in a comment | Isolated node at t=1e6 yields a grid ending at 100, not 1,000,000. The over-constrained `== 20` assertion was relaxed to `< 1_000_000` as suggested. |
| `N3` | Comment added recording the tskit unique-position invariant the decrement relies on | |
| `N4` | CLI passes `min_usable_fraction` straight through; `build_store` resolves the default via `effective_usable_fraction` | Bonus, unprompted: `metadata["minimum_usable_fraction"]` now records the *effective* value (0.1) instead of `None`, so default-built stores carry accurate provenance. Test asserts it. |
| `D4` | `snp_age_distribution.py` switched to `tszip.load`, help text updated, `.tsz` test added, and a full README section documenting the accessory script and its differing coordinate convention | Confirmed reading both `.trees` and `.tsz`. |

### Two small notes, neither blocking

- `test_multichromosome_boundaries_round_trip` builds its subject with
  `SNPAgeDataset.__new__(SNPAgeDataset)` and hand-sets three attributes, bypassing
  `__init__`. It does exercise the real `rows_to_native`, so it is a valid test, but it will
  silently stop covering the method if that method ever reads another attribute. This is the
  `O-a` pattern (§5) in the tests rather than the library; a small real two-chromosome store
  fixture would be sturdier.
- The vectorised age bound is looser than the exact one on tiny fixtures (100 vs 20 on the
  isolated-node test), which is expected and harmless — on a dense real ARG the two agree
  exactly. Truncating the grid to the last bin carrying mass after accumulation remains
  available if a tight `b` in the persisted store is ever wanted.
