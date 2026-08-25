# Bootstrap-target matching: recommended route and open work

`BOOTSTRAP_TARGET_MATCHING_PLAN.md` §10 lists ten acceptance gates that had to
be met before bootstrap-target matching could replace the §5 hard-q50 sampler.
This document records where that stands, on the 75-draw production store.

Rejected approaches and readings that later measurements overturned live in
`BOOTSTRAP_DISCARDED_APPROACHES.md`, so they are not re-proposed without their
evidence and do not clutter this one.

## Recommended route

Run §6 bootstrap-target matching **alone** -- not alongside §5. Submit
`run_bootstrap_matching.sbatch`, which is the canonical launcher; the equivalent
direct command is:

```bash
python bootstrap_target_matcher.py \
  --store "$TMPDIR/interval_store" \
  --target targets/in_gene \
  --candidate-rows candidate_rows.npy \
  --output bootstrap_matches/in_gene \
  --work-dir results/work-in-gene --resume \
  --replicates 100 --restarts 3 \
  --min-epochs 10 --max-epochs 50 --patience 5 \
  --disjoint-replicates \
  --seed 1002
```

Two behaviours that were flags during the validation campaign are now
unconditional, so the options that selected them have been removed. The coarse
swap screen is always a geometric sub-sample of the exact grid (there is no
`--search-grid-spacing`), and each restart is always a stratified draw from the
target's own equal-mass age strata (there is no `--seed-sets`, so §5 is not a
prerequisite; `--restarts` sets the count and there is no
`--closest-restarts`/`--diverse-restarts` split). `--disjoint-replicates`
remains a flag and is the production setting.

**What `--disjoint-replicates` delivers, stated precisely.** Guaranteed and
verified: the 100 published sets share no control rows -- 406,700 of 406,700
slots are distinct, maximum reuse 1. Measured: the sets consume 1.77% of the
23,026,051-row candidate pool, and `E_r` shows no degradation with replicate
index (slope -0.05 generations per replicate; first-25 median 53.18 against
last-25 median 50.71), so the sequential-depletion dependence is not detectable
here. **Not claimed:** statistical independence, and therefore not an effective
replicate count of 100. Replicates still share the observed TE sample and the
interval store, which is inherent to bootstrapping rather than a defect, and an
effective replicate count for the downstream statistic must still be estimated
from that statistic.

Measured against a uniform swap screen with non-disjoint replicates, on the
4,067-site in-gene target:

| | shipped defaults | recommended | §5 hard-q50 |
|---|---:|---:|---:|
| unique controls | 195,836 | **406,700** | 260,182 |
| maximum reuse | 30 | **1** | 13 |
| QC pass | 96/100 | **100/100** | — |
| `cor(B_r, O_r)` | 0.9972 | **0.9999** | — |
| lower-tail `O/B` (5%) | 1.288 | **0.994** | — |
| relative age error at 10% quantile | 21.9% | **−0.0%** | — |
| runtime | 2.65 h | 2.18 h | 1.05 h |

Every objection that stood against this stage is now closed. It has better
membership diversity than the sampler it replaces, not worse.

**Why not run both.** Publishing two null distributions requires justifying
which one is reported, which is a forking-paths problem. A single prespecified
method is more defensible even when it is the less flattering one.

---

## Definitions

The four symbols used throughout this document and in
`BOOTSTRAP_TARGET_MATCHING_PLAN.md`. All distances are Wasserstein-1 between age
CDFs, in generations.

Three objects, for replicate `r`:

| symbol | what it is |
|---|---|
| `T` | the **observed** TE age CDF, from the actual TE sites (4,067 for the in-gene target) |
| `T⁽ʳ⁾` | replicate `r`'s **bootstrap** TE target — the same sites resampled with replacement |
| `S_r` | the **matched control set** for replicate `r` — the same number of SNPs, chosen to match `T⁽ʳ⁾` |

Three distances and a ratio:

| symbol | definition | meaning | T5 median |
|---|---|---|---:|
| `B_r` | `D(T⁽ʳ⁾, T)` | how far this bootstrap draw moved the target — **the uncertainty being propagated** | 1688.9 |
| `E_r` | `D(S_r, T⁽ʳ⁾)` | how closely the optimizer hit its **assigned** target — a convergence measure | 288.1 |
| `O_r` | `D(S_r, T)` | how far the published set sits from the **real** observed distribution — **the scientific quantity** | 1800.6 |
| `R_r` | `E_r / B_r` | optimizer error as a fraction of the displacement being reproduced; QC gate is `< 0.5` | 0.181 |

Three things about the scales that are easy to get wrong:

- **A perfect matcher gives `E/B = 0` but `O/B = 1`**, not zero. If `S_r` matched
  `T⁽ʳ⁾` exactly then `O_r = D(T⁽ʳ⁾, T) = B_r`. The quantity comparable to `R_r`
  is therefore `|O/B − 1|`, not `O/B`.
- **`R_r` bounds the realized error**: the triangle inequality gives
  `|O/B − 1| ≤ E/B`, and on T5 the realized error runs at a median of 41% of that
  bound. `R_r` is conservative by construction.
- **`R_r` is convergence QC, not evidence of propagation.** The scientific check
  is whether the *distribution* of `O_r` reproduces the distribution of `B_r`
  across replicates — that is gate 5, checked at the ensemble level, because
  propagation is a claim about the ensemble rather than any single replicate.


### A note on ages quoted from CDF arrays

`age_bins.npy` holds grid **labels**, and the CDF is evaluated half a bin to
their right: `cdf_evaluation` in the target metadata records
`P(X < right_cell_edge)`, and `analysis_points()` evaluates at `label + bin_width/2`.
So interpolating a quantile against `age_bins` returns a label, and the physical
age is that label plus half a bin — 500 generations at the 1,000-generation
production grid.

Every age in this document is quoted as a **physical age**, with the half bin
added. Relative errors are computed on physical ages too, which is why the young
end reads 21.9% rather than the 31.5% an uncorrected label-space calculation
gives: the difference is unchanged, the denominator is half a bin larger.

---

## 1. Inputs

### 1.1 Present in this directory

| Input | Detail |
|---|---|
| `run.combined.98.tsz` | 24,805,868 sites, 26 samples, 2,131,846,815 bp concatenated |
| `run.combined.99.tsz` | 24,929,209 sites |
| `run.combined.100.tsz` | 25,030,335 sites |
| `run.combined.99.age.max6mis.nomispo.rna.struc.te.ingene.pos.txt` | 4,072 TE positions — the RNA in-gene pilot target from plan §3 |
| `chrom_offsets.combined.txt` | 10 chromosomes, 1 bp concatenation gaps |
| `results/interval-gate3-3draw-store/` | 3-draw store, 27,245,216 SNPs, 74,715,451 intervals, `maximum_above` 22,854,736 |

The TE file is exactly the pilot target (4,072 requested / 4,061 eligible in
plan §3), so the ladder below reproduces and then extends the published pilot
rather than starting a new one.

### 1.2 Present elsewhere

In `/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined/`:

- the full posterior: 75 `.tsz` draws, so gate 1 is expensive, not blocked;
- `run.combined.99.age.max6mis.nomispo.rna.struc.te.sorted.pos.txt`, 35,512
  positions — the mid-size target for T9;
- `b73_nam_max6mis_sv_genotyping_format_nodup.final.pos.txt`, 185,232 positions
  — the large target for T9, and the case `SWAP_SAMPLER_HPC_HOWTO.md` §3 sizes
  target memory against;
- `for_te_chr10/chr.10.combined.snp.te.sorted.vcf.gz`, the SINGER *input* VCF
  for chromosome 10 (§1.4);
- **`all_te_snp_age_interval_store/`, the complete 75-draw production store.**

The production store is readable and usable as-is, which removes the largest
build from the ladder:

| field | value |
|---|---|
| `schema_version` | `snp-age-interval-v1` |
| `n_posterior_draws` | 75 |
| `n_snps` | 31,240,944 |
| `n_intervals` | 1,867,881,588 |
| `maximum_above` | 36,744,633.16 |
| `content_sha256` | present (`8e53c668…`) |
| on disk | 18.2 GB |

It records `content_sha256`, so targets built from it carry a real store-content
identity and the checks in §1.5 have force.

Its chromosome offsets are identical to this repository's
`chrom_offsets.combined.txt` in all ten entries, including the 1 bp
concatenation gaps (chr1 length 308,452,471, chr2 offset 308,452,472). The two
conventions agree, so position lists resolve the same way against either.

**`maximum_above` is 36,744,633, not the 22,854,737 of the 3-draw store.** At
`--bin-width 1000` the exact analysis grid is therefore ~36,745 points rather
than ~22,855 — 1.6× the README's "about 22,900". Every memory and time figure
derived from the grid scales with it, so the README's 16-18 GiB target-job
allowance for ~185,000 TEs becomes roughly 27 GB
(185,232 × 36,745 × 4 B). Re-derive resource requests from this number, not
from the published one.

### 1.3 The control universe is all SNPs, with all TE variants excluded

Decided: the control pool is **all SNPs, minus every TE variant**. The
`*.low.snp.pos.txt` files are not used. This removes the input ambiguity that
`SAMPLER_SCALING_NOTES.md` §"First scientific/input check" flags.

It also costs nothing. Both the §5 sampler and the §6 matcher propose with
unweighted `rng.choice(candidates, size=n, replace=False)`
(`swap_control_sampler.py:262`, `bootstrap_target_matcher.py:236`), which numpy
implements by Floyd's algorithm — measured at 0.3 ms for `n=4,061` and 1.2 ms
for `n=35,512` against a 27.2M-row pool, i.e. *faster* than against a 374k pool
and independent of pool size. The per-epoch cost is `n` store row reads whatever
the universe contains.

This is worth stating plainly because it is the opposite of the old sampler's
behaviour. `sample_age_matched_syn.py:186-197` copies a full probability vector
over the candidate pool and calls **weighted** `rng.choice` per stratum per
proposal — an O(pool) inner loop, which is diagnosis item 2 of the scaling
notes and the mechanism behind the 19-hour job. That mechanism does not exist
in the current code path. The scaling notes' warning against the 23.36M pool
applies to `sample_age_matched_syn.py` only, and should not be carried over to
these tests.

**TE variants are excluded from the control pool.** `--all-eligible` alone would
not do this: it removes only the target rows, and the store is built from a
combined SNP+TE dataset — the VCF contigs are literally
`chr.N.combined.snp.te.sorted` — so every *other* TE variant would remain
eligible as a control. Matching TEs against controls that are themselves TEs
weakens the contrast the analysis rests on.

`build_candidate_rows.py` produces the excluded universe as a `--candidate-rows`
array. The TE superset already exists and needs no new file:
`b73_nam_max6mis_sv_genotyping_format_nodup.final.pos.txt`, 185,232 positions.
Verified nesting, on chromosome-position pairs:

```
in_gene (4,072)  ⊂  rna.struc.te (35,512)  ⊂  b73_nam ... final (185,232)
```

with zero elements outside at each step, so one exclusion list covers every
target in the programme. Pass the identical array to T2 and T3: the seed
library and the matcher must share one universe, or the matcher would be
initialized from rows it is not allowed to propose.

**The resolution rate is expected to be ~71%, and that is benign.** Measured
against the 3-draw store:

| list | resolved | rate |
|---|---:|---:|
| in_gene | 4,046 / 4,072 | 99.36% |
| rna.struc.te | 35,262 / 35,512 | 99.30% |
| b73_nam ... final | 131,454 / 185,232 | 70.97% |

The two nested subsets resolve at ~99.3% through the same store and code path,
which is what rules out a chromosome-offset mismatch: a bad offset convention
would break them too. The 29% shortfall is genuine absence from the ARGs,
consistent with the naming — the target lists are `max6mis.nomispo`-filtered
and the b73_nam list is the pre-filter genotyping set. An absent position is
harmless, because a position that is not a store row cannot be selected as a
control either.

`build_candidate_rows.py` therefore stops by default below 95% rather than
warning, since an unresolved exclusion and a mis-resolved one are
indistinguishable from the count alone, and a mis-resolved TE stays in the pool
while appearing excluded. For this dataset pass `--min-resolved-fraction 0.70`,
and re-establish the evidence above before doing so on any new store.

**Restrict the universe to the filtered SNP list, and keep the TE exclusion as a
check.** Starting from every eligible store row admits ~3.8M rows that are in
neither the SNP list nor the TE list — sites that failed `max6mis` or appear
only in some draws. The targets are all `max6mis.nomispo`-filtered, so controls
drawn from a laxer set would differ from the target in missingness, and README
§7 warns explicitly that a target and a control set whose retained fractions
differ substantially are not comparable however small Φ is.

`run.combined.99.age.max6mis.snp.pos.txt` (23,359,072 positions) is that
filtered universe, and it turns out to solve the TE problem on its own:

```bash
B=/quobyte/jrigrp/beil/te_evo/singer_analysis/argtest/results/combined
python build_candidate_rows.py \
  --store results/store-3draw \
  --include-positions $B/run.combined.99.age.max6mis.snp.pos.txt \
  --exclude-positions $B/b73_nam_max6mis_sv_genotyping_format_nodup.final.pos.txt \
  --output results/candidate-rows.npy \
  --min-resolved-fraction 0.70
```

Measured on the 3-draw store:

| step | rows |
|---|---:|
| store rows | 27,245,216 |
| eligible | 26,394,075 |
| SNP list resolved / eligible | 22,954,653 (98.27%) / 22,420,982 |
| TE rows resolved / eligible | 131,454 (70.97%) / 130,819 |
| **removed by the TE exclusion** | **0** |
| candidates | 22,420,982 |

The TE exclusion removes **nothing** from the SNP list: the two are disjoint, as
the file naming implies.

**Both lists are required, and they fail differently.** This is the reason to
prefer the include-list, and it is not that the TE list is dispensable:

- *Exclude-only* (all eligible rows minus TEs) makes the pool everything except
  the TEs you listed. Its correctness rests entirely on that list being
  **complete**. A TE missing from it remains a control. Completeness cannot be
  verified from position lists: the nesting above proves the three analysis
  categories sit inside the b73_nam list, but nothing proves the store holds no
  TE outside it.
- *Include-list* makes the pool only what is on the SNP list. A TE absent from
  the b73_nam list is still excluded, because it is not on the SNP list either.
  The TE list no longer has to be complete for the pool to be sound.

The eligible rows decompose as 22,420,982 on the SNP list, 130,819 known TEs,
and **3,842,274 in neither list** — rows of unverified type that could contain
TEs. Exclude-only admits all 3.84M of them and yields 26,263,256 candidates;
the include-list drops them.

With the include-list, the only remaining path for a TE into the pool is one
that is *on the SNP list itself*, mislabeled at source. `--exclude-positions` is
exactly the test for that, so keep it permanently: `removed by the TE exclusion
== 0` is a cheap invariant that a future SNP list built without the TE filter
would violate, and the report JSON records the count. A complete TE list is
still preferable to an incomplete one, and deriving both tracks from the
original VCF (§1.4) removes the need to trust either list.

### 1.4 Missing, and required before the ladder can complete

1. **The original polarized VCF, for all ten chromosomes.** Required for T7/T8
   and for plan §9.4's Φ-SFS regression comparison. T1-T6 do not depend on it.

   `results/vcf/` does not satisfy this: those are `##source=tskit` exports, one
   per posterior draw per chromosome. README §7 requires allele counts from one
   polarized VCF *rather than* from the ARGs, so an export would make the SFS a
   function of the same ARGs that produced the ages.

   The right file is the SINGER **input** VCF, of which
   `for_te_chr10/chr.10.combined.snp.te.sorted.vcf.gz` is the chromosome-10
   instance — `##source=argprep.maf_to_sites`, contig `10`, 26 samples,
   3,628,984 records. Verified on it:

   | check | result |
   |---|---|
   | records with a non-`.` ID | 12,614 |
   | chr10 TE-list positions | 12,614 |
   | ID'd positions that are TE-list positions | 12,614 / 12,614 |
   | TE-list positions carrying no ID | 0 |
   | SNP-list positions present | 1,286,453 / 1,286,453 |
   | SNP-list positions carrying an ID | 0 |
   | multiallelic records at SNP-list sites | 0 |

   **The ID column is the definitive TE marker**: ID'd records and TE-list
   positions are in exact bijection, and no SNP-list position carries one. This
   closes the completeness question of §1.3 at the source rather than by trusting
   a position list, and confirms the two tracks are disjoint in the VCF itself.
   Every SNP-list position is present and biallelic, so `phi_sfs.py` will not
   fail on them. 282,135 multiallelic records exist elsewhere in the file and are
   harmless: `phi_sfs.py:365` skips any record that is not requested before the
   biallelic check.

   What is still needed is **chromosomes 1-9 in this same form**, since
   `phi_sfs.py` takes a single `--vcf` and requires every requested site to be
   present — the TE target plus every control SNP across all 100 sets, 260,258
   unique controls in the §5 two-draw pilot.

   Two properties to confirm with the data's author before T7:

   - **It is not polarized, and `--ancestral-mode ref` must not be used on it.**
     The ancestral allele is inferred by SINGER and comes from the ARG. §1.6 is
     the controlling constraint for T7 and T8.
   - **`REF` is one of the 26 samples**, not a pseudo-sample: it carries the ALT
     allele at TE sites such as `10:56392`, so it is a real genome. Decide
     whether it contributes an allele to the SFS; the choice shifts every
     spectrum.
2. **A §5 hard-q50 matched bundle for this target.** As of the CLI
   simplification this is no longer a prerequisite: `--seed-sets` is gone and
   each restart is a stratified draw from the target's own age strata. It was a
   hard dependency when this ladder was written, which is what T2 exists for.

### 1.6 The ancestral allele comes from the ARG, and this breaks Φ-SFS as built

> Written before C5 was implemented. `phi_sfs.py` now requires
> `--ancestral-table` and the VCF-polarity options are gone; the measurements
> below are why. See C5 for the current state.

The ancestral state is not a property of the input VCF. It is inferred by
SINGER and read off the ARG. At the time, `phi_sfs.py` offered only two ways to
obtain it — `--ancestral-mode ref` (REF is ancestral) and `--ancestral-mode
info` (an INFO field) — and **both read polarity out of the VCF file**. Neither
describes this data. This was a blocking code gap, recorded as C5.

Measured on chromosome 10, comparing the SINGER input VCF against the
`##source=tskit` per-draw exports in `results/vcf/`, which carry REF equal to
the ARG's inferred ancestral state:

| comparison | result |
|---|---:|
| draw 98 vs input: same REF/ALT | 949,811 (68.97%) |
| draw 98 vs input: **exactly swapped** | **427,313 (31.03%)** |
| draw 98 vs input: neither | 0 |
| draw 99 vs input: swapped | 426,807 (30.93%) |
| draw 100 vs input: swapped | 431,721 (30.99%) |

Every disagreement is an exact REF/ALT swap and never a different allele, which
confirms the exports are re-polarized copies of the same biallelic sites.
**Running `--ancestral-mode ref` against the input VCF would mis-polarize about
31% of sites**, silently, and every unfolded spectrum and every Φ-SFS value
would be wrong with no diagnostic to catch it.

Three consequences follow, and they are not merely operational.

**Polarity carries posterior uncertainty.** Draws 98 and 99 disagree about
REF/ALT at 22,918 of 1,296,634 shared chr10 sites (1.77%). The ancestral call is
an inference with a posterior distribution, so the unfolded SFS does too. The
current design propagates uncertainty in the TE *age* CDF and holds the observed
TE SFS fixed (README §6). Under ARG-derived polarity the SFS is not a fixed
observation.

**README §7's stated rationale is now wrong and must be corrected.** It says
allele counts come from a VCF "rather than from the subset of posterior ARGs in
which a site is represented", because "posterior ARG presence affects age
matching, but it does not change a site's observed frequency." Polarity flips
the derived count, so with ARG-derived ancestral states the unfolded frequency
*does* depend on the ARG. The design paragraph asserts an independence the data
does not have.

**It compounds the C2 risk.** Age and polarity are both inferred from the same
trees. The matching selects controls on ARG ages, and the statistic then
measures an ARG-polarized spectrum, so the two are not independent measurements.
C2 must be extended to ask whether W1-repair utility is associated with
*polarity confidence*, not only with derived frequency.

**No single draw's export can serve as the VCF.** Coverage of the 1,299,067
requestable chr10 sites (SNP list plus TE list):

| draw | records | SNP-list sites missing | TE-list sites missing |
|---|---:|---:|---:|
| 98 | 1,377,124 | 73,450 (5.71%) | 5,308 (42.08%) |
| 99 | 1,380,088 | 0 (0.00%) | 5,247 (41.60%) |
| 100 | 1,393,275 | 59,875 (4.65%) | 5,220 (41.38%) |

Draw 99 covers the SNP list exactly, which is not a coincidence — the list is
`run.combined.99...`, derived from that draw. But **every draw is missing
~42% of TE positions**, and `phi_sfs.py:563-566` aborts when any requested site
is absent. Across all three draws 0.38% of requestable sites are missing
everywhere and 9.15% are missing from at least one, so a union is not a
per-draw object either.

The workable shape is therefore: genotypes from the input VCF, which has every
site, plus a separately supplied per-site ancestral allele derived from the
ARGs, with a prespecified rule for combining the 75 draws. That rule is a
scientific decision, not an implementation detail — a designated draw, a
majority vote, or per-draw propagation are materially different analyses, and
the 1.77% pairwise disagreement is the scale of what is being decided.

### 1.5 The existing 3-draw store must be rebuilt

`results/interval-gate3-3draw-store/metadata.json` has `catalog_sha256` but no
`content_sha256`, and its `inputs` record draws 100/101/102 from beil's
directory — not the three `.tsz` files here.

This matters more than it looks. The identity checks in
`bootstrap_target_matcher.py:394-407`, `sample_age_matched_controls.py:418-426`,
and `distributed_age_match.py:90` are all of the form *"fail if the recorded
value is not None and differs."* A target built from a store with no
`content_sha256` records `None`, so the check is **silently skipped** rather
than failed. Runs against that store are not wrong, but they carry no
store-content identity guarantee at all — which is precisely what a production
acceptance test is supposed to establish.

Rebuild before anything below (P1).

---

## 2. Prerequisites

### P0 — clean, pinned checkout and full test suite on Linux

```bash
git status --porcelain          # must be empty
python -m pytest -q tests test_snp_age_distribution.py
```

Run on a compute node, not the head node: one interval-store audit test
requires Linux `fork`. This covers plan §9.1 (bootstrap correctness) and §9.2
(optimizer correctness). A failure here stops the ladder.

### P1 — optional: rebuild the 3-draw store for fast rehearsal

Optional now that the production store is available (P2). Its only remaining
value is quick iteration on T4's cost model before committing 75-draw compute.
Skip it and run the ladder directly on the production store if that is not
needed.

```bash
HPC_CPUS=4 HPC_MEM=64G HPC_TIME=08:00:00 ~/.claude/bin/hpc_run '
python build_snp_interval_store.py \
  run.combined.98.tsz run.combined.99.tsz run.combined.100.tsz \
  --interval-store results/store-3draw \
  --chrom-offsets chrom_offsets.combined.txt \
  --scratch-dir "$TMPDIR/store-build" \
  --interval-dtype float32'
```

Confirm `metadata.json` contains `content_sha256` before proceeding.

### P2 — validate and stage the existing 75-draw store

**No build is required.** `all_te_snp_age_interval_store/` (§1.2) is complete,
carries `content_sha256`, and uses the same chromosome offsets as this
repository. Instead:

1. confirm its 75 `inputs` are the intended draws and that no job is writing to
   it;
2. treat it as immutable — never modify it while target or matching jobs run;
   and
3. stage it to `$TMPDIR/interval_store` per `SWAP_SAMPLER_HPC_HOWTO.md` §3.
   Node-local scratch must hold 18.2 GB plus 20%.

Rebuild only if step 1 fails.

### P3 — build the TE-excluded candidate universe

Run `build_candidate_rows.py` as in §1.3, once per store, and pass the result to
both T2 and T3.

Row indices are store-specific, so this is repeated for the 3-draw and 75-draw
stores and the two arrays are never interchanged.
`results/interval-gate3-syn-rows.npy` belongs to the old store and must not be
reused at all.

---

## 3. The test ladder

Each test names the plan §10 gate it serves. Tests T1–T4 use the 3-draw store
and are cheap rehearsals; T5 onward require the 75-draw store.

> **This ladder is a completed record, not a runbook.** The commands below are
> reproduced as they were run, against the CLI of the time. Several options they
> pass (`--seed-sets`, `--closest-restarts`, `--diverse-restarts`,
> `--search-grid-spacing`, `--distance`, `--log-age-offset`) no longer exist:
> the behaviours they selected are now unconditional. Do not copy these
> commands. The supported command is under "Recommended route" above, and
> `run_bootstrap_matching.sbatch` is the launcher.

### T1 — target construction on the rebuilt 3-draw store

*Gate: prerequisite for all others.*

```bash
HPC_MEM=32G HPC_TIME=02:00:00 ~/.claude/bin/hpc_run '
python te_age_target.py \
  --store results/store-3draw \
  --te-positions run.combined.99.age.max6mis.nomispo.rna.struc.te.ingene.pos.txt \
  --output results/targets/in_gene_3draw \
  --bootstrap-replicates 10000 \
  --acceptance-quantile 0.50 \
  --bin-width 1000 \
  --seed 1002'
```

Memory: the TE×age float32 CDF matrix is 4,061 × ~22,855 × 4 B ≈ 371 MB. Small
for this target; do not carry that number to a 35,000-site target, where it is
~3.2 GB, or to the ~185,000-site case, where `SWAP_SAMPLER_HPC_HOWTO.md` §3
allows 16–18 GiB.

**Pass:** 4,061 eligible TE sites; `metadata.json` records
`source_store_content_sha256`; the exact grid has ~22,855 points.

### T2 — hard-q50 seed library (§5 workflow)

*Gate: 10 (hard-q50 retained as sensitivity analysis), and a hard prerequisite
for §6.*

Run the ten-chain workflow of `SWAP_SAMPLER_HPC_HOWTO.md` §5–§6 against the T1
target to produce `results/matches/in_gene_3draw`. This is not optional
scaffolding: the §6 matcher consumes it as its initialization library, and gate
10 requires the hard-q50 result to exist alongside the §6 result anyway.

**Pass:** all ten chain bundles validate and gather publishes 100 sets with
`complete: true`.

### T3 — pilot reproduction

*Gate: baseline for gates 3, 5, 6.*

Transcript of what was run, against the CLI of the time. Three of its options no
longer exist, so this will not parse today; it is here as the record.

```text
HPC_MEM=32G HPC_TIME=12:00:00 ~/.claude/bin/hpc_run '
python bootstrap_target_matcher.py \
  --store results/store-3draw \
  --target results/targets/in_gene_3draw \
  --seed-sets results/matches/in_gene_3draw \
  --candidate-rows results/candidate-rows.npy \
  --output results/bootstrap_matches/in_gene_3draw_pilot \
  --work-dir "$TMPDIR/bootstrap-pilot" \
  --replicates 20 \
  --closest-restarts 1 \
  --diverse-restarts 0 \
  --min-epochs 15 --max-epochs 15 \
  --seed 1002'
```

These flags deliberately reproduce the published pilot configuration (20
replicates, single closest restart, fixed 15 epochs) so the numbers are
comparable to plan §3. The current equivalent is `--restarts 1` with stratified
initialization.

**Pass:** the four W1 quantities land near the plan §3 table (medians ~1,276 /
~2,346 / ~216 / ~1,426), `cor(B_r, O_r)` ≈ 0.99, and 19–20 of 20 replicates
satisfy `R_r < 0.5`. The pilot used two ARGs and this uses three, so expect
displacement — a *qualitative* mismatch (e.g. correlation well below 0.9, or a
majority of QC failures) means something changed in the pipeline and must be
explained before spending 75-draw compute.

Note that the published pilot drew from 373,647 candidates and this draws from
~26.3M TE-excluded rows, so the optimizer has a far richer pool and the matching
error `E_r` should if anything improve. A *worse* `E_r` than the pilot would be
the surprising result and would need explaining.

Run T2 with the same `--candidate-rows` array.

### T4 — cost model and epoch scaling

*Gate: 3 (convergence calibration), and the go/no-go for T5's resource request.*

This is the test the ladder exists for. The optimizer's dominant cost is store
row reads: **one `row_cdfs` read per proposal, `n` proposals per epoch, per
restart, per replicate** (`bootstrap_target_matcher.py:234-244`). At the
defaults that is

```
replicates × restarts × epochs × n
  = 100 × 3 × (10…50) × 4,061
  = 12.2M … 60.9M row reads
```

Each read on the production store touches 75 draws' intervals and is evaluated
on a coarse grid derived from `maximum_above` 36,744,633 — 1.6× the 3-draw
store's span (§1.2), so scale the projection below accordingly.

for the *smallest* target in the programme — and, per §1.3, that count is
independent of how large the candidate universe is, so an all-SNP pool does not
inflate it. `results/interval-gate3-candidate-access.json`
measured ~396,678 rows gathered in ~20 s on the 3-draw store (~20k rows/s). If
read time scales with intervals per row, the 75-draw store gives roughly 800
rows/s, projecting **4–21 hours of pure store I/O for one target on one core**.
`bootstrap_target_matcher.py` has no `--workers` and its replicate loop is
strictly sequential, so there is no in-process escape from that number.

Measure, on both stores:

- wall time per replicate and per restart-epoch;
- store row reads per second, cold and warm, node-local versus Quobyte;
- peak `MaxRSS`;
- epochs actually used before the patience rule fires, and the full best-W1
  trace per restart;
- whether the coarse/exact split is paying for itself (accepted swaps per
  epoch, and exact-certification time as a fraction of epoch time).

**Pass:** a defensible epoch budget replaces the pilot's arbitrary 15, derived
from where the traces flatten — this is gate 3 — *and* a projected 75-draw
runtime that fits a SLURM allocation. If it does not fit, C4 below (distributed
execution) stops being a nice-to-have and becomes a blocker for T5.

### T5 — full-posterior rerun

*Gate: 1.*

Repeat T1, T2, T3 against the 75-draw store, now with production settings
(`--replicates 100`, default restarts, the T4-calibrated epoch budget).

**Pass:** the run completes, and every published distance recomputes from
`row_indices.npy` (gate 8 — the matcher already certifies on the exact grid;
this is an independent check by a separate script, not a re-read of its own
output).

### T6 — propagation, QC, diversity

*Gates: 4, 5, 7 (partial), 9.*

From the T5 bundle, without touching Φ-SFS:

- QC pass rate against the prespecified `R_r < 0.5`, `E_r ≤ 500` (gate 4). No
  bootstrap redraws; confirm from `replicates.csv` that every prespecified
  replicate has exactly one status record.
- Distributions of `B_r` and `O_r`: correlation, signed and absolute
  differences, and behaviour in **both tails** — gate 5 is explicitly
  center-and-tail concordance, and the pilot's one failure was a *small*-`B_r`
  replicate, so the lower tail is not a formality.
- Sensitivity to initialization: W1 and QC by restart source, number of
  distinct seeds selected, and best versus near-optimal restart (gate 6).
- `reuse_counts.npy`: unique controls, maximum reuse, pairwise overlap,
  replacement from each initializing set, chromosome balance (gate 7,
  descriptive half).
- Declare, in advance, how unresolved QC failures are handled (gate 9). This is
  a written decision, not a computation, and it must be recorded before T7.

### T7 — Φ-SFS with a declared effective replicate count

*Gates: 7 (inferential half).*

Run `phi_sfs.py` on the T5 bundle and, separately, on the T2/T5 hard-q50 bundle.

**C5 is now closed in code**: `phi_sfs.py` requires `--ancestral-table` and can
no longer read polarity from the VCF, so the 31%-mis-polarization route is
unreachable. T7 still requires chromosomes 1-9 of the input VCF (§1.4), the
polarity-confidence extension of C2, and the effective-N analysis below. The
derived-frequency arm of C2 has been completed.

**Pass:** an effective replicate count derived from the observed cross-replicate
dependence by a **prespecified** method, with a sensitivity analysis if the
dependence model is approximate. Reuse counts alone do not discharge this gate;
plan §8 says so directly.

### T8 — regression comparison against hard-q50

*Gate: 10, and plan §9.4.*

Compare the two bundles on: SNP-to-observed W1 distribution, membership
diversity and reuse, runtime and memory, failure/restart rate, Φ-SFS
distribution and effective replicate count, and the association between Φ-SFS
and age-match diagnostics.

**Pass:** the scientific conclusions are compared and any difference is
explained before §6 is proposed as the default. A difference in Φ-SFS is *not*
by itself evidence that §6 is better — see C2.

### T9 — target-size scaling

*Gate: production readiness beyond the pilot category.*

The in-gene target is 4,061 sites. `SAMPLER_SCALING_NOTES.md` records a
production category of 35,512. `SWAP_SAMPLER_HPC_HOWTO.md` §10 requires ~500-,
4,000-, and 35,000-SNP targets for the §5 sampler; the same ladder applies here,
because every cost in T4 is linear in `n` and the store cost is linear in draws.

**Pass:** measured time and memory at all three sizes, and a resource table that
extrapolates to the full category list.

---

## 4. Gates that cannot be tested until code is written

These are not scheduling problems. No amount of HPC time closes them.

### C1 — spatial dependence and block bootstrap (gate 2)

Plan §10.2 and status item 1–2. The implemented bootstrap is iid multinomial
over TE sites. Nothing in the repository tests spatial dependence among TE
age-CDF contributions, and there is no block-bootstrap mode. Both must be
written. Until then the 100 replicates carry no inferential interpretation, as §5b
records.

**Required:** a dependence test (TE sites are physically clustered along
chromosomes; the ARG makes nearby sites' age posteriors correlated by
construction, so the null of exchangeability is not the likely outcome), a
documented iid-versus-block decision, and a block mode if iid fails.

### C2 — W1-repair utility versus derived frequency: **frequency arm measured**

The risk formerly flagged in README §6 as "the main risk to watch": the optimizer picks SNP
membership to hit an age CDF, so if repair utility correlates with derived
allele frequency, the matcher biases the SFS — the exact quantity Φ-SFS
measures, invisible in every matching diagnostic.

`probe_repair_utility_frequency.py` measures it on chromosome 10, where
genotypes are available. Two findings.

**Published controls are much rarer than the pool, but that is age matching, not
optimization.** Mean derived frequency of selected controls against the
23.0M-row candidate pool's 0.2992:

| bundle | chr10 controls | mean derived freq | shift |
|---|---:|---:|---:|
| §5 hard-q50 | 15,193 | 0.1665 | −0.1327 |
| §6 min-W1 | 11,384 | 0.1619 | −0.1373 |
| §6 disjoint | 23,741 | 0.1523 | −0.1468 |

§5 is a constrained random walk with no optimization pressure and shows almost
the whole shift, so it is a consequence of matching a young-skewed target: young
variants are rare. Optimization adds a further −0.014, about 10% of the total.

**The utility association is age-mediated.** Scoring 6,000 candidates by the
exact-grid W1 change from swapping each into a published set:

```
UNCONDITIONAL  corr(freq, utility) = +0.2478
               corr(freq, age)     = +0.2919   <- the confound
               corr(age,  utility) = +0.9873

weighted mean within-stratum corr(freq, utility) = +0.0297
```

Utility is almost entirely determined by age, which is what an age-matching
objective should do. Frequency predicts utility only because it predicts age;
conditioning on age collapses the association from +0.248 to +0.030, and the
within-stratum correlations have no consistent sign (−0.314 in the youngest
stratum, +0.11 to +0.17 in the older ones).

**Conclusion: the optimizer selects on age, not on frequency.** This closes the
derived-frequency arm, but not the polarity-confidence extension in §1.6.

Two caveats. This is chromosome 10, 6,000 candidates, one bootstrap target, six
age strata; the youngest stratum spans 20–64,720 generations, so residual
within-stratum age variation could mask a weak effect. The claim is not "no
association" but "none large enough to matter beside a 0.987 age correlation".
And it tests frequency specifically. The separate finding that the optimizer
prefers well-dated SNPs (posterior width 0.62 of the TE target against 0.83 for
§5) is another selection effect, unaddressed by this diagnostic.

### C3 — effective-N method (gate 7)

A method must be declared, not just reuse counts reported. See T7.

### C5 — an ancestral-allele source outside the VCF (new)

**Closed in code.** `build_ancestral_states.py` builds the table and
`phi_sfs.py` now *requires* `--ancestral-table`; the VCF-derived
`--ancestral-mode`/`--ancestral-info` options have been removed, so the
mis-polarization route described below is no longer reachable. The paragraphs
that follow are the original analysis, kept for its measurements.

As written, this blocked T7 and T8 outright: `phi_sfs.py` could only take
polarity from REF or an INFO field, and this data has it in neither (§1.6).

**Extracting the ancestral states is cheap.** The interval-store scan reads
`sites.position`, `mutations.site`, `mutations.node`, `nodes.time`, and the edge
table, but never `sites.ancestral_state` — so we do not already have it, though
only because it was not saved. Measured on `run.combined.99.tsz`:

| step | time |
|---|---:|
| `tszip.decompress` | 50.7 s |
| read site table + ancestral states | 0.0 s |
| unpack to per-site characters | 0.1 s |
| **per draw** | **50.8 s** |

All 24,929,209 sites carry a single-character ancestral state over `{A,C,G,T}`.
The entire cost is decompression, which the store build already paid; the
ancestral states themselves are a packed array that comes free once the tables
are in memory. 75 draws is ~1.06 h serial and minutes as a one-CPU job array.
If the store is ever rebuilt, capture ancestral states in that same pass at
zero marginal cost.

Required artifact: a per-store-row `uint8[4]` count of `A/C/G/T` ancestral calls
across the draws in which each site appears, plus a per-row present-draw count.
At 31,240,944 rows that is 125 MB, and it covers the union of draws rather than
any single one.

#### Risk: ARG polarization leans on allele frequency

Accepted as a limitation rather than something to fix. Recorded so it is not
mistaken for signal later.

TE sites give a partial truth set, but the insertion-is-derived assumption has a
known exception: a TE that fixes and is later removed by a deletion makes the
*deletion* derived. That predicts a frequency signature, and it is present --
the ARG's disagreement rate against insertion-is-derived rises monotonically
with insertion frequency:

| insertion frequency | disagree rate |
|---|---:|
| 0.0-0.1 | 3.7% |
| 0.3-0.5 | 18.7% |
| 0.7-0.9 | 59.4% |
| 0.9-1.0 | 75.2% |

Median insertion frequency is 0.080 where the ARG agrees and 0.583 where it
disagrees. Fixed-then-deleted is genuinely rare -- only 3.1% of TE sites sit
above frequency 0.9, accounting for 17% of disagreements -- so the C5 error rate
of 13.6% is inflated by real biology and falls to roughly 11% once
high-frequency sites are excluded.

The gradient is confounded, though: fixed-then-deleted biology and an ARG that
simply calls the common allele ancestral predict the same pattern. Separating
them on 1,710,961 ordinary chr10 SNPs, where a pure frequency rule would put
ancestral on the major allele almost always:

| minor-allele frequency | ancestral = major allele |
|---|---:|
| 0.00-0.05 | 77.5% |
| 0.10-0.20 | 79.0% |
| 0.30-0.40 | 64.4% |
| 0.40-0.50 | 54.7% |
| **overall** | **74.3%** |

So the ARG carries real signal -- 25.7% of sites get a minor-allele-ancestral
call that frequency-parsimony would never produce -- but the lean is
unmistakable, and near frequency 0.5 the call is close to a coin flip.

**Why this cannot be engineered away.** There is no outgroup in this dataset;
polarity comes from tree topology, and rare-allele-is-derived is a correct
prior most of the time. Restricting to confidently polarized sites would be
worse than doing nothing, because confidence is itself frequency-dependent and
the restriction would bias the frequency spectrum directly -- the quantity being
measured.

**What to do instead: perturb the polarity, not the spectrum.** Folding
(`k -> min(k, n-k)`) is polarization-invariant, but it merges bin `j` with bin
`20-j` and therefore cancels *antisymmetric* differences exactly. An excess of
rare derived alleles paired with a deficit of common ones -- the purifying-
selection signature, and the most likely real result -- folds to nothing. A null
folded result would not distinguish "no polarization artifact" from "folding
removed the signal", so it only supports a conclusion in one direction.

Use a polarity-perturbation sensitivity analysis instead. It keeps all 19 bins:

1. Fit the polarization error rate as a function of minor-allele frequency from
   the table above (77.5% at minor frequency below 0.05 falling to 54.7% near
   0.5, with the TE truth set bounding the absolute rate).
2. For each of many draws, flip each site's polarity independently with its
   fitted frequency-dependent probability, and recompute Φ for the target and
   every control set.
3. Report the spread of Φ under that perturbation beside the observed value.

This answers the question that actually matters -- *how far could polarization
error move the answer* -- rather than whether a lower-powered statistic agrees.
It uses the measured error model rather than assuming one, it retains the
antisymmetric contrast that folding destroys, and because the same model is
applied to arms with different frequency spectra it reproduces the differential
effect rather than cancelling it.

**Measured, on chr10.** `probe_polarity_sensitivity.py` runs this against the
real projection and scoring code, polarizing by the ARG's own majority ancestral
call and re-scoring under flips at the fitted frequency-dependent rate:

| mean flip probability | Φ median | 95% interval | shift from observed |
|---:|---:|---|---:|
| 0.110 (matches the TE truth-set rate) | 0.1232 | [0.1165, 0.1309] | **−15.0%** |
| 0.186 | 0.1086 | [0.1003, 0.1163] | −25.1% |
| 0.310 (frequency-shortfall upper bound) | 0.0844 | [0.0754, 0.0939] | −41.7% |

Observed Φ is 0.1449 with polarity as called.

Two things follow, and the first reverses how this risk was framed above.

**Polarization error attenuates Φ; it does not manufacture it.** Flipping pushes
both spectra toward folded and shrinks their difference, monotonically across
all three rates. So the error costs power rather than creating false signal, and
**the observed Φ is a floor rather than a ceiling** — with perfect polarity the
TE-versus-control difference would be larger. A positive result is therefore
robust to this bias; a null result is not, and must not be read as evidence of
no difference.

**The effect is first-order, not a rounding detail.** At the realistic rate it
is a 15% attenuation. C5's ancestral-table reader should carry the perturbation
machinery from the start rather than have it added later, and every reported Φ
should be accompanied by its attenuation interval.

Caveats that travel with these numbers: chromosome 10 only, and the control arm
is a frequency-blind random SNP sample rather than an age-matched set, so 0.1449
is not the scientific Φ. Read the table as a sensitivity magnitude. The
independent-flip model is also an approximation — real polarization error is
systematic in frequency rather than independent across sites — which is a
further reason to treat the attenuation as indicative rather than exact.

Report the folded statistic too, but only as one-directional evidence: a *large*
folded Φ in the same direction is positive evidence the result survives
polarization-invariant measurement. A small one is uninformative.

The bias does **not** cancel between arms. TEs and their matched controls have
different frequency spectra by construction, and the polarization lean is
frequency-dependent, so it acts unequally on the two. That is why the check is
worth running rather than assumed away.

#### Use the polarity-weighted mixture, not majority rule

Let `p` be the posterior proportion of draws calling one allele ancestral. Do
not threshold it. A site whose alleles are `{A,T}` with `k` copies of `T` among
`n` callable samples has derived count `k` under one polarization and `n-k`
under the other, so its projected contribution is the mixture

\[
p\,h(k,n) + (1-p)\,h(n-k,n).
\]

This is the posterior mean of the site's contribution, and it is strictly
better founded than majority rule, which treats `p=0.51` and `p=1.0`
identically.

Two properties make it cheap and safe:

- **It is one line.** `h(n-k, n) == reverse(h(k, n))` exactly — verified for
  every `(k,n)` at `n = 20, 23, 26`. Since `phi_sfs.py` already computes
  `h(k,n)` once per distinct `(k,n)` pair, the mixture is
  `p * h + (1 - p) * h[::-1]`. No new projection machinery.
- **It never reweights sites against each other.** `h_0` and `h_20` swap under
  the reversal, so the retained mass `1 - h_0 - h_20` that
  `PHI_SFS_IMPLEMENTATION_PLAN.md` §2 defines as a site's contribution is
  invariant under polarity. The weighting redistributes
  mass *within* a site only, which keeps the `retained_fraction` and
  `endpoint_fraction` diagnostics comparable across sets.

#### What must still be prespecified

1. **Which estimand — compute both, because neither costs anything.**
   Φ-SFS is total variation, a *nonlinear* function of the spectra, so
   `Φ(E[SFS]) ≠ E[Φ(SFS)]`. They answer different questions and there is no
   reason to choose.

   `E[Φ(SFS)]` is not more expensive. `phi_sfs.py:416` already computes
   `derived = alt_count if ancestral == ref else callable_count - alt_count`, so
   `k` and `n` come from the observed genotypes and polarity only selects the
   orientation. The VCF scan is therefore **polarity-independent and runs once**,
   and it is the dominant cost — tens of millions of records parsed in Python.
   Everything downstream is arithmetic on a per-distinct-`(k,n)` projection
   cache that already exists and, by the reversal identity, already serves both
   orientations. Re-summing 101 spectra over ~4,061 sites for 75 draws is a few
   seconds and ~1.3 MB.

   **Conditioning `p` on presence dissolves the coverage problem for
   `Φ(E[SFS])`.** Estimate `p` over the draws in which the site actually
   appears. Then every requested site has observed `k` and `n` from the VCF and
   a well-defined `p` from its own present-draws — no intersection, no fallback,
   no site dropped. The per-draw coverage gaps in §1.6 constrain only the
   per-draw `E[Φ(SFS)]` route, which needs a coherent full-site polarity
   assignment; they do not touch this one.

   For reference, restricting to the all-draw intersection is not viable
   anyway. Sites present in *all* of the first `d` chr10 draws:

   | draws | SNP-list sites in all | TE-list sites in all |
   |---:|---:|---:|
   | 1 | 1,226,578 (95.3%) | 7,394 (58.6%) |
   | 3 | 1,141,467 (88.7%) | 6,977 (55.3%) |
   | 6 | 1,058,487 (82.3%) | 6,594 (52.3%) |
   | 12 | 975,550 (75.8%) | 6,234 (49.4%) |

   Still falling at 12, so at 75 draws it would discard well over a third of
   control sites and half the TEs, non-randomly with respect to age.

   **This is also why the mixture must be linear and majority rule must not be
   used.** Present-draw counts differ systematically between target and
   controls — TEs appear in ~58% of draws, SNPs in ~95% — so a site's `p` is
   estimated from about 44 draws for a TE and 71 for a control. The linear
   mixture is unbiased at any draw count, because it is linear in `p`. Majority
   rule thresholds `p` and is therefore biased in a way that depends on the
   draw count. The effective ancestral weight actually applied to a site, given
   its true `p`:

   | true `p` | linear mixture | majority, m=44 (TE) | majority, m=71 (SNP) | TE−SNP gap |
   |---:|---:|---:|---:|---:|
   | 0.50 | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
   | 0.55 | 0.5500 | 0.7458 | 0.8017 | −0.0559 |
   | 0.60 | 0.6000 | 0.9087 | 0.9562 | −0.0476 |
   | 0.70 | 0.7000 | 0.9971 | 0.9998 | −0.0027 |
   | 0.80 | 0.8000 | 1.0000 | 1.0000 | 0.0000 |

   Majority rule would apply materially different polarity weights to TEs and to
   their matched controls at the *same* true `p`, purely because TEs appear in
   fewer draws. That is a differential bias in the exact quantity Φ-SFS
   measures, and it would be invisible in every matching diagnostic — the same
   failure mode as C2. The linear mixture has no such gap at any `p` or `m`.

   Make `Φ(E[SFS])` with presence-conditioned `p` the primary estimand. The
   per-draw `E[Φ(SFS)]` spread remains available as a secondary uncertainty
   measure, but it requires the absent-site rule and therefore an extra
   assumption; state it if reported.

   Still record the present-draw count `m` per site and report its distribution
   for the target against the controls. The mixture is unbiased regardless, but
   a reader should see that TE polarity rests on fewer draws than control
   polarity.
2. **How `p` is estimated.** Over the draws in which the site is present, per
   above. SINGER draws are autocorrelated MCMC samples, so `p` is less precise
   than its granularity suggests — this does not bias the linear mixture but
   does affect any interval placed on it. One assumption to state rather than
   hide: presence-conditioning is unbiased only if a site's being represented in
   a draw is independent of its polarity in that draw.
3. **An ancestral-table reader** that validates that both alleles are REF or
   ALT at every requested site. Implemented as the required `--ancestral-table`
   option.
4. **A correction to README §7's independence claim.** Made: §7 now says allele
   *counts* come from the VCF while polarity comes from the ARG.

Report, alongside every Φ-SFS result, the distribution of `p` over requested
sites and the share of spectrum mass contributed by sites near `p = 0.5`. Those
sites contribute a nearly symmetric — effectively folded — projection, which is
honest but carries little directional information, and a result resting largely
on them means something different from one resting on confidently polarized
sites.

Note that this makes the ARG dependence smooth and explicit rather than removing
it. The coupling in §1.6 between age and polarity, both inferred from the same
trees, is unaffected.

### C4 — distributed execution and gather (status item 7)

`bootstrap_target_matcher.py` is single-node, single-process, sequential over
replicates. There is no `--workers`, no array-task launcher, and no gather —
unlike §5, which has `sample_age_matches.sbatch`, `gather_age_matches.sbatch`,
and `run_age_match_manifest.py`.

The good news: replicate seeding is already
`derive_seed(args.seed, target_digest, replicate)` (line 855), so replicate `r`
is reproducible independent of which process computes it. Splitting replicates
across array tasks is therefore statistically sound and needs only a replicate-range
CLI flag plus an atomic gather modelled on `gather_age_matches.sbatch`. Do **not**
improvise this by pointing several concurrent invocations at one shared
`--work-dir`: they would race on the same per-replicate bundle paths and each
would restart from replicate 0.

Whether C4 is a blocker or a convenience is decided by T4.

---

## 5. Where the gates stand

| gate | status |
|---|---|
| 1. full-posterior rerun | **closed** — 100 replicates x 3 restarts on the 75-draw store |
| 2. spatial dependence / bootstrap choice | **closed** — iid retained; effect ~2-3% against a 1.1% Monte Carlo noise floor |
| 3. convergence calibration | **closed** — from restart traces; production converges at a median of 20 epochs, 294/300 on plateau |
| 4. QC pass rate without redraw | **closed** — 100/100 with the recommended flags; the absolute cap now scales with the target threshold |
| 5. `O_r` reproduces `B_r`, centre and tails | **closed** — lower-tail `O/B` 0.994, `cor` 0.9999, relative age error <=0.1% at every quantile |
| 6. initialization and restart sensitivity | **closed** — the diverse restart wins 32/100; selection moves the published result by ~1% of `B_r` |
| 7. reuse, diversity, effective replicate count | **partly closed** — disjoint replicates give 406,700 unique controls with maximum reuse 1, and no `E_r` drift across replicate index; an effective replicate count for Phi-SFS must still be estimated from the Phi-SFS scores (C3) |
| 8. recomputation from canonical row indices | **closed** — bit-exact, 0.000e+00 on all three distances |
| 9. handling of unresolved QC failures | **moot at 100/100**; if failures recur, they concentrate at small `B_r` and must be retained rather than dropped |
| 10. hard-q50 retained as sensitivity analysis | **superseded** — a single prespecified method was chosen instead (see Recommended route) |

### Phi-SFS polarity: prespecified decisions

Recorded before any Phi-SFS number is produced, so the choices are not made
after seeing results.

1. **TE sites are polarized biologically.** A TE insertion is the derived state.
   All 12,614 chr10 TE records are `A`/`G` with no other combination, so
   insertion is ALT and `ancestral == REF` holds at every TE site, so the TE arm
   needs no table lookup at all. The known exception, a TE that
   fixed and was later deleted, is rare: only 3.1% of TE sites sit above
   insertion frequency 0.9.
2. **Control SNPs use the polarity-weighted mixture.** A site with observed
   derived count `k` among `n` callable samples contributes
   `p*h(k,n) + (1-p)*h(n-k,n)`, where `p` is the posterior proportion from the
   ancestral table. This is the posterior mean of the site's contribution, it is
   linear in `p` and therefore unbiased at any draw count, and it automatically
   downweights uncertain sites toward a folded contribution. Majority rule is
   **not** used: thresholding `p` is biased in a way that depends on the number
   of draws a site appears in, and TE and control sites differ systematically in
   that count.
3. **`p` is conditioned on presence** -- estimated over the draws in which the
   site actually appears. Every requested site then has a well-defined weight,
   with no intersection or fallback rule needed.
4. **`p` is used as reported, uncalibrated.** Against TE ground truth the ARG is
   only ~90.8% correct where all 75 draws agree, so raw `p` is overconfident by
   roughly 9-15 points across the bins where most sites sit. Applying a
   calibration curve was considered and **deliberately not adopted**. The
   consequence to note when reading results: control spectra are somewhat
   sharper than the ARG's measured accuracy warrants, which makes Phi slightly
   larger than a calibrated analysis would give.
5. **No perturbation analysis.** The mixture already carries polarity
   uncertainty into the spectrum, so resampling polarity on top of it would
   largely double-count. The 15% attenuation measured earlier applies to a
   hard-polarized analysis, not to this one.

Report the distribution of `p` for the target and for each published control set
alongside every Phi-SFS result, so a reader can see how much of the spectrum
rests on contested calls. TE sites average `p` = 0.978 and controls 0.937.

### Accepted limitation: the optimizer prefers well-dated SNPs

Recorded as a known and accepted property, not an open question.

Matching operates on posterior-*mean* CDFs, so a site the ARG dates to within a
few thousand generations and one it dates to within a few hundred thousand are
interchangeable to the objective. They are not equally *useful*, though: a
narrow posterior gives a sharper per-site CDF, which is more effective for
shaping an aggregate CDF precisely. A precision-seeking optimizer therefore
prefers them. Median across-draw age SD of the selected controls, relative to
the TE target:

| set | ratio to TE target |
|---|---:|
| §5 sampler controls | 0.83 |
| §6 optimizer controls | **0.62** |

Two consequences, both measured.

**It is the mechanism behind the diversity cost of minimum-W1 selection.** The
"useful" subpopulation is smaller than the eligible one, which is why strict
minimum-W1 selection reached only 195,836 unique controls. Disjoint replicates
resolve that by construction, so the diversity consequence no longer applies.

**Dating confidence and polarity confidence are the same underlying axis**, so
selecting on one does select on the other. Measured across 5,000 candidates,
`corr(age-posterior SD, polarity confidence) = -0.254` (Spearman -0.298):

| age-posterior SD (gen) | n | mean `p` | `p >= 0.99` |
|---|---:|---:|---:|
| 7 - 36,462 | 1,000 | 0.9924 | 94.4% |
| 84,096 - 148,258 | 1,000 | 0.9395 | 67.3% |
| 277,106 - 3,017,890 | 1,000 | 0.8803 | 48.0% |

**But it does not create a polarity mismatch between the arms.** Selection on ARG
dating confidence does shift polarity confidence upward relative to the pool,
but the sampler shifts it almost as much, and — the part that matters — the two
arms end up matched:

| set | mean polarity confidence `p` | `p >= 0.99` | `p < 0.9` |
|---|---:|---:|---:|
| candidate pool | 0.9367 | 68.6% | 20.9% |
| §5 sampler controls | 0.9752 | 84.9% | 8.3% |
| **§6 optimizer controls** | **0.9786** | 86.9% | 7.2% |
| **TE target** | **0.9785** | 85.8% | 7.3% |

Controls and target agree to four decimal places, so control spectra are not
systematically more or less folded than the TE spectrum under the polarity
mixture. The optimizer adds only +0.0035 beyond the sampler, about 8% of the
shift from the pool — the same proportion it adds on derived frequency.

The residual selection effect is accepted: controls are drawn from the
better-resolved part of the ARG, which correlates with genomic context. Nothing
in the age-matching diagnostics would reveal it, and no downstream consequence
has been demonstrated. It is documented so a reader can weigh it rather than
discover it.

### Remaining work, in priority order

Nothing above blocks the matching stage. What remains is downstream.

1. **C5 — the ancestral-allele reader: implemented and validated.** The table is
   built across all 75 draws, `--ancestral-table` is wired in and authenticated
   against the store by content digest, and the two-arm polarity design is in
   place: TE sites polarized biologically, control SNPs by a posterior-weighted
   mixture. Exercised end to end on chromosome 10. Nothing methodological is
   outstanding here; the production run needs only the data in item 2.
2. **Genotypes for chromosomes 1-9 — staging, not validation.** The method is
   validated: polarity is resolved per site, the projection is per site, and
   nothing in either path is chromosome-aware, so chromosome 10 exercises every
   code path. What the remaining chromosomes supply is *data*. The matched sets
   span the genome — 382,959 of the 406,700 control sites (94.2%) and 3,803 of
   the 4,067 TE sites sit outside chromosome 10 — and `phi_sfs.py` aborts if any
   requested site is missing from the VCF. Restricting to chromosome 10 instead
   would mean rebuilding the target and rematching against chromosome-10 sites
   only, which is a different and roughly sixteen-fold smaller study.
3. **Say what the Phi-SFS spread means — no effective count is needed.** The
   effective-replicate-count question arose when sets shared controls heavily
   (maximum reuse 30), which is dependence *on top of* the bootstrap structure
   and would have made the spread too narrow. Disjoint replicates removed it,
   and the residual sequential-depletion dependence is not detectable: 1.77% of
   the pool consumed, a 0.9% finite-population factor, and no trend in `E_r`
   across replicate index. What remains is the bootstrap's own design -- every
   replicate resamples the same observed TE sites -- which is not a defect to
   correct for. The requirement is therefore a sentence, not a statistic: the
   spread of Phi across replicates measures **how far Phi moves under age-CDF
   uncertainty, conditional on the TE sites actually observed**. It is not a
   confidence interval on a population parameter and must not be reported as
   one.
4. **Target-size scaling beyond 35,466 sites.** Measured at 600, 1,500, 4,067
   and 35,466; the recommended route holds across all four. A 185,232-site
   target projects to ~125 h single-node and would need distributed execution,
   but few categories reach that size.

## 5a. Results so far

Measured on the 75-draw production store unless stated.

### P0 — test suite

160 passed, 0 failed, on a Linux compute node (27.6 s).

### P3 — candidate universe

| | rows |
|---|---:|
| store rows | 31,240,944 |
| eligible | 29,089,747 |
| SNP list resolved | 23,359,072 / 23,359,072 (100.00%) |
| TE rows resolved / eligible | 136,714 (73.81%) / 134,833 |
| removed by the TE exclusion | **0** |
| candidates | 23,026,051 |

The SNP list resolves completely against the 75-draw union, against 98.27% on
the 3-draw store. The disjointness invariant of §1.3 holds here too.

### T1 — target construction

`results/targets/in_gene_75draw`, 10,000 bootstrap replicates, q50, 1,000-generation
grid. **W1 acceptance threshold 1480.48 generations**, against 1905.10 for the
same target on the 2-draw store in `SWAP_SAMPLER_HPC_HOWTO.md` §10. Tighter, as
expected from 75 draws.

### T2 — hard-q50 seed library, complete

`results/matches/in_gene_75draw`, ten chains × ten sets, five workers, store
staged to node-local scratch.

| quantity | value |
|---|---:|
| wall clock | 1 h 02 m 38 s |
| user time | 17,548 s (469% CPU) |
| peak RSS | 18.0 GB |
| sets published | 100 × 4,067 sites, `complete: true` |
| W1 min / median / max | 1367.98 / 1461.63 / 1480.47 |
| W1 standard deviation | 20.44 |
| all sets inside threshold 1480.48 | yes |
| unique controls | 260,182 |
| maximum reuse | 13 |
| membership replacement per sweep | 0.399-0.466 |

Two observations.

**It reproduces the two-draw pilot's diversity almost exactly.**
`SWAP_SAMPLER_HPC_HOWTO.md` §10 reports 260,258 unique controls and maximum
reuse 14 on the two-draw store; this gives 260,182 and 13 on the 75-draw store.

**It confirms the §4 argument for replacing this sampler.** All 100 sets sit in
a 112-generation band immediately inside the threshold, standard deviation 20.44
— 1.4% of the threshold itself. The walk has no preference for a smaller W1 once
inside, so the saved sets occupy a narrow shell rather than spanning the
bootstrap uncertainty. That is exactly the behaviour bootstrap-target matching
exists to fix, now measured on production data rather than argued from the
two-ARG pilot.

### T5 — production run, 100 replicates: **complete**

`results/bootstrap_matches/in_gene_75draw`, 100 replicates x 3 restarts,
calibrated convergence, 75-draw store. Wall clock 2 h 40 m, peak RSS 34.9 GB,
`complete: true`. Gate 1 is met.

| quantity | median | mean | range |
|---|---:|---:|---:|
| `B_r` bootstrap TE -> observed | 1688.9 | 1720.8 | 312.5-4120.4 |
| `E_r` matched -> bootstrap TE | 288.1 | 286.9 | **222.9-336.5** |
| `O_r` matched -> observed TE | 1800.6 | 1842.6 | 515.8-4260.5 |
| `R_r` matching-error ratio | 0.181 | 0.218 | 0.062-1.011 |

**Convergence (gate 3, confirmed at scale).** Epochs median 20, mean 22.6, range
11-50. 294 of 300 restarts stopped on `material_improvement_plateau`; only 6 hit
the 50-epoch ceiling. The budget calibrated from T3's traces predicted "high
teens to twenties" and the production median is 20, so the convergence rule is
sound and the pilot's fixed 15 is retired.

**Gate 4 - QC: 96/100 pass.** All four failures are `R_r >= 0.5`; none breach the
absolute limit, and `E_r` never exceeds 336.5 against a cap of 500.

**Gate 6 - restart sensitivity: satisfied, and the diverse restart earns its
place.** Closest restarts have median best-W1 305.0 and are selected 68 times;
the diagnostic diverse restart has median 311.2 and still wins **32 of 100**
replicates. Restarts reach materially different local optima - within-replicate
spread across the three is 13.6% median, 29.2% at the 90th percentile, 148.7% at
worst - yet the choice among them moves the published result by only 18.4
generations at the median, 1.09% of `B_r`. The answer is robust to restart
selection even though individual restarts are not.

#### One structural finding explains both remaining defects

`E_r` is confined to 222.9-336.5 generations across all 100 replicates. That is
an **irreducible matching-error floor**: 294 of 300 restarts stopped at plateau
rather than at the epoch ceiling, so more epochs will not lower it. It is a
property of swapping whole sites against a 36,746-point grid, not of the search
budget.

Both defects below follow from that floor.

**Gate 5 - `O_r` reproduces `B_r` in the centre and upper tail, but not the
lower tail.** Correlation is 0.9970 and the median relative difference 7.33%,
but the ratio is not uniform:

| quantile | `B_r` | `O_r` | `O/B` |
|---:|---:|---:|---:|
| 1% | 413.4 | 519.2 | **1.256** |
| 5% | 593.2 | 765.7 | **1.291** |
| 10% | 709.4 | 852.0 | 1.201 |
| 25% | 1119.5 | 1267.4 | 1.132 |
| 50% | 1688.9 | 1800.6 | 1.066 |
| 75% | 2285.7 | 2343.1 | 1.025 |
| 95% | 2998.4 | 3108.7 | 1.037 |
| 99% | 3358.0 | 3501.4 | 1.043 |

Where a bootstrap target lands close to the observed target, a ~290-generation
floor is a large fraction of the distance being reproduced, and `O_r` overstates
`B_r` by 20-29%. **The method cannot faithfully propagate bootstrap targets
nearer than roughly 300 generations to the observed distribution.** Gate 5 is
met in the centre and upper tail and failed in the lower tail; this must be
stated wherever the Phi-SFS distribution is interpreted, because the affected
replicates are exactly those most similar to the observed TE distribution.

**Gate 9 - the QC failures are structural, and must not simply be dropped.**
The four failing replicates have `B_r` = 312.5, 414.4, 438.9, 594.1 - all among
the six smallest of 100 (median 426.7 against 1697.7 for passes). They fail
because `R_r = E_r/B_r` divides a floored numerator by a small denominator.
Discarding them would systematically delete the replicates whose bootstrap
targets sit closest to the observed target, truncating precisely the lower tail
that is already distorted. Prespecify retention with the distortion reported,
not exclusion.

**Gate 7 - diversity is materially worse than hard-q50.** Against T2 on the same
target:

| | §5 hard-q50 | §6 bootstrap-target |
|---|---:|---:|
| unique controls | 260,182 | **195,836** |
| maximum reuse | 13 | **30** |
| mean reuse | 1.56 | 2.08 |
| controls used >= 10 times | - | 4,967 |

Optimising toward a precise CDF concentrates selection on SNPs that are
unusually effective at repairing particular bins - the mechanism plan §8 warns
about, now measured. 25% fewer unique controls and 2.3x the maximum reuse are
more membership sharing than the sampler this stage is meant to replace, which
weighs against it in the gate-10 comparison. Note that reuse bounds membership
sharing, not an effective replicate count: that has to be estimated from the
downstream statistic. `--disjoint-replicates`, adopted after this measurement,
removes the membership sharing entirely.

### T3 — pilot reproduction on 75 draws

Same configuration as the published pilot (20 replicates, one closest restart,
fixed 15 epochs). Wall clock 10 m 59 s, 100% CPU, peak RSS 19.2 GB.

| quantity | plan §3 pilot (2 ARGs) | T3 (75 ARGs) |
|---|---:|---:|
| `B_r` median | 1,276 | 1,928.8 |
| initial SNP set → bootstrap TE, median | 2,346 | 2,346.0 |
| `E_r` median | 216 | 320.7 |
| `O_r` median | 1,426 | 2,054.9 |
| `cor(B_r, O_r)` | 0.989 | **0.9972** |
| median &#124;`O_r`−`B_r`&#124; | 92 | 131.1 |
| median relative difference | 6.2% | **6.320%** |
| `R_r` median / mean | 0.131 / 0.187 | 0.165 / **0.187** |
| `R_r` range | 0.035–0.502 | 0.062–0.555 |
| QC pass | 19/20 | **19/20** |
| triangle inequality holds | — | yes |

Absolute distances are larger on 75 draws, as expected from a different
posterior, but every *relative* diagnostic reproduces: the median relative
difference matches to 0.12 percentage points, the mean matching-error ratio is
identical at 0.187, and the QC pass rate is the same 19/20. This is the first
evidence for gates 4 and 5 on production data.

### Gate 3 — convergence calibration

From the 20 restart traces, mean improvement per epoch as a fraction of `B_r`:

| epoch | mean best W1 | gain / `B_r` | replicates still gaining >0.1% `B_r` |
|---:|---:|---:|---:|
| 2 | 464.7 | 0.0369 | 20/20 |
| 6 | 358.8 | 0.0118 | 20/20 |
| 8 | 337.2 | 0.0041 | 17/20 |
| 10 | 327.9 | 0.0025 | 11/20 |
| 12 | 322.6 | 0.0010 | 9/20 |
| 15 | 318.5 | 0.0006 | 4/20 |

Epochs to reach within 5% of the final value: median 8, 90th percentile 11, max
14. Within 1%: median 12, max 15.

**The pilot's fixed 15 epochs was arbitrary but not badly chosen.** It captured
most of the available improvement — the last five epochs contribute 0.605% of
`B_r` on average — while still truncating 4 of 20 replicates that were making
material gains. The shipped defaults (`--min-epochs 10 --max-epochs 50
--patience 5` against a 0.1%·`B_r` material threshold) sit correctly relative to
this decay and should stop most replicates in the high teens to twenties. Gate 3
is calibrated from traces rather than convention, as required.

### T4 — single-chain cost model

One chain, one set, store staged to node-local scratch:

| quantity | value |
|---|---:|
| store staging (18.2 GB, rsync) | 88.5 s |
| wall clock | 4 m 07.6 s |
| user time | 234.9 s |
| CPU utilisation | 97% |
| peak RSS | 14.6 GB |
| major page faults | 21 |

Construction converged in 5 epochs (1991, 1349, 991, 782, 785 accepted swaps;
exact W1 176,667 → 936.5) with **no grid refinements**, and the saved set scored
1400.8 against the 1480.48 threshold.

Two things follow. The run is **CPU-bound, not I/O-bound**, once the store is
staged — 97% CPU and 21 major faults — so the §3 projection of 4-21 hours of
store I/O per target does not apply to a staged store, and staging is
mandatory rather than optional. And peak RSS is 14.6 GB rather than the ~1.2 GB
the selected-row CDF cache alone predicts (`8 × 4,061 × 36,745`), because
resident pages of the mmapped 18.2 GB store count toward RSS. Size chain jobs
from the measured figure, not from the cache formula in
`SWAP_SAMPLER_HPC_HOWTO.md` §3.

Extrapolating to ten saved sets per chain: construction cost ~5,898 accepted
swaps against ~4,061 per subsequent sweep, so a ten-set chain is roughly 4.7×
a one-set chain, about 19 minutes. Ten chains is ~3.2 h serial and well under
an hour at five-way concurrency. **C4 (distributed execution) is therefore a
convenience for this target size, not a blocker** — revisit at the 35,512- and
185,232-site targets in T9, where both `n` and the CDF cache grow.

### C5 — ancestral states extracted, and a new differential risk

`build_ancestral_states.py` completed over all 75 draws. Every draw resolved
100% of its sites against the store catalog, and **every one of the
31,240,944 store rows has at least one ancestral call** (max present-draw count
75). 6,698,584 rows (21.4%) fall below 0.9 agreement across draws.

**Correction to an earlier claim in §1.6.** The "TE sites appear in only ~58% of
draws" figure came from the per-draw tskit VCF exports in `results/vcf/`, and it
does not describe the ARGs. Measured from the store, TE target sites have a
**median present-draw count of 75** (mean 72.7; only 0.84% appear in fewer than
38 draws). Whatever thins those export files, it is not ARG representation. The
coverage argument against a per-draw `E[Φ(SFS)]` is therefore much weaker than
stated, and presence-conditioning remains correct but is doing less work than
described.

**The finding that matters is an asymmetry in polarity confidence.** Letting `p`
be the majority-allele posterior proportion:

| | TE target (n=4,067) | control candidates (200k sample) |
|---|---:|---:|
| median present draws | 75 | 75 |
| mean present draws | 72.7 | 68.5 |
| mean `p` | **0.9785** | **0.9367** |
| `p = 1.0` (unanimous) | 85.81% | 68.57% |
| `p >= 0.95` | 89.77% | 73.98% |
| `p >= 0.9` | 92.72% | 79.09% |
| `p < 0.6` (near-folded) | 0.91% | 4.04% |

This is the opposite direction from the concern recorded above: controls are
*less* confidently polarized than TEs, not more.

It creates a systematic, non-biological contributor to Φ-SFS. Under the linear
mixture a site with `p < 1` contributes a partly folded projection, and a site
at `p ≈ 0.5` contributes an almost symmetric one. Because control sites carry
systematically lower `p`, **control spectra are systematically more folded than
the TE spectrum even when the underlying biology is identical**, and Φ-SFS is
the total variation between them. The mixture is unbiased for each site
separately, but the *difference* between two spectra with different polarity
confidence is not zero.

**This asymmetry is real knowledge, not an artifact to remove.** TE polarity is
known biologically — a TE insertion is the derived state in essentially every
case — so the TE arm should be polarized from biology at `p = 1`, not from the
ARG. SNP polarity cannot be established that way and must come from the ARG with
its uncertainty. The two arms genuinely differ in what is known about them, and
the analysis should carry that difference rather than erase it.

Two consequences follow.

**Polarity comes from two sources, and `phi_sfs.py` must implement both.** TE
target sites take a fixed biological polarization; control SNP sites take the
mixture from the ancestral table.

The TE convention is fixed and confirmed: **all 12,614 chr10 TE records are
`A`/`G`**, with no other combination, so REF=`A` is absence and ALT=`G` is the
insertion. Because insertion is the derived state, `ancestral == REF` holds at
every TE site, so TE polarity is a fixed convention rather than a lookup. The TE
arm therefore needs no new code — only the control arm consults the ancestral
table.

#### The TE sites are a free calibration set, and they show `p` is overconfident

Because TE polarity is known biologically, every TE site is a labelled test of
the ARG's own polarity inference. On the 8,019 chr10 TE sites that resolve to
store rows:

| `p` bin | n | ARG correct | nominal `p` | gap |
|---|---:|---:|---:|---:|
| 0.50-0.60 | 218 | 52.75% | 55.03% | −2.28% |
| 0.60-0.70 | 254 | 66.93% | 64.90% | +2.03% |
| 0.70-0.80 | 321 | 68.85% | 74.89% | −6.05% |
| 0.80-0.90 | 434 | 75.12% | 84.93% | −9.81% |
| 0.90-0.95 | 334 | 81.74% | 92.71% | −10.97% |
| 0.95-0.99 | 409 | 82.40% | 97.33% | −14.94% |
| 0.99-1.00 | 6,049 | 90.76% | 100.00% | −9.24% |

Overall the ARG is correct at 86.44% of TE sites while nominal `p` averages
95.41%.

**Posterior consistency is not accuracy.** Where all 75 draws agree — which is
where most sites live, 6,049 of 8,019 — the ARG is right only 90.76% of the
time. The draws share data and model, so they are consistently wrong together,
and `p` cannot see it. Calibration is good at low `p` and degrades as `p` rises,
which is the worst possible shape: it is most overconfident exactly where the
mixture treats a site as settled.

Consequences for the design:

1. **Using raw `p` in the mixture understates polarity uncertainty**, by roughly
   9-15 points across the confident bins where nearly all sites sit. The control
   spectra will be sharper than the evidence supports.
2. **Calibrate `p` before using it.** Map nominal `p` to empirical accuracy with
   this curve and use the calibrated probability as the mixture weight. The
   calibration set costs nothing and grows once chromosomes 1-9 arrive.
3. **The null floor rises accordingly**, so the floor calibration below must use
   calibrated rather than nominal probabilities or it will be too low.

Two caveats before this is applied. TE sites may not be representative of SNPs —
they sit disproportionately in repetitive, low-recombination contexts where ARG
inference is harder, so the curve may be pessimistic for controls. And a fraction
of the disagreements may be genuine biology rather than ARG error, since
insertion is the derived state *almost* always; that fraction inflates the
apparent error rate. Both argue for treating the curve as an upper bound on
control-arm error until it can be checked against an independent polarization.

**Uncertainty inflates Φ-SFS whichever estimand is chosen, so it needs a floor
rather than a correction.** If the control and TE spectra were biologically
identical, imperfect SNP polarity would still make the estimated control
spectrum differ from the TE spectrum, giving Φ > 0. Note also that total
variation is convex in its first argument, so by Jensen
`Φ(E[SFS]) ≤ E[Φ(SFS)]`: the mixture route reports the smaller value, and the
per-draw route the larger, but neither is zero under the null.

Required before T7 is read:

1. report the `p` distribution for the target and for every published control
   set, not just their means;
2. **calibrate a null floor** — compute Φ-SFS under the null that the control and
   TE spectra are biologically identical, with the observed SNP polarity
   uncertainty applied, and read every observed Φ against it. This is the same
   device that made gate 2 interpretable: without the permuted arm there, a 2-3%
   effect would have been read as real; and
3. run Φ-SFS restricted to confidently polarized control sites (`p >= 0.95`) as
   a prespecified sensitivity analysis, which lowers the floor and shows how much
   of the result depends on contested calls.

Matching polarity confidence between the arms is *not* an option: it would mean
degrading known TE polarity to match SNP ignorance, discarding real information
to make an artifact cancel.

### C5 — extraction cost and method

`build_ancestral_states.py` accumulates per-site A/C/G/T ancestral counts and a
present-draw count, conditioned on presence, with `--draws` slicing for array
tasks and `--merge` for gathering.

Coordinate convention, worth recording because it cost a wrong first run: the
store catalog holds `tables.sites.position` **verbatim**
(`build_snp_interval_store.py:82,169`). No one-based shift belongs in a
store-row lookup; the metadata string "one-based within chromosome;
global=offset+POS" describes how *VCF position lists* are converted, not what
`positions.npy` contains. With the shift applied, draw 99 resolved 16.7% of its
sites; without it, 24,929,209 of 24,929,209 (100%).

## 5b. Measurements relocated from the README

The README was reduced to a how-to, and the quantitative material it carried was
moved here verbatim. Nothing below is new evidence; several items restate or sit
beside numbers already recorded above, and where the two disagree both are kept
and the disagreement is flagged rather than resolved.

### Interval-store build resources

For approximately 25–30 million SNPs and 75 combined SINGER draws, a
conservative starting request is one CPU, 48 GB RAM, 16 hours, and at least 32
GiB free in node-local `$TMPDIR`. The measured projection is approximately 17.1
GiB for a final 75-draw float32 store and 22.6 GiB of packed bucket scratch.
Keep additional Quobyte headroom for atomic publication and any older store
retained at the destination. The production builder is single-worker and its
final merge is I/O-bound, so more CPUs do not speed that phase.

> **Discrepancy to resolve.** §1.2 records the completed 75-draw production
> store as **18.2 GB on disk**, while the README's projection above is **17.1
> GiB** (≈18.4 GB). One is a pre-build projection and the other a measurement of
> the built store, so they are not the same quantity, but they are close enough
> to be confused for each other. Both are kept as written.

### Target-construction scratch

For an interval store, target construction creates a temporary float32
TE-by-age CDF matrix under `--scratch-dir`. At roughly 185,000 TEs and the
measured maximum age, allow approximately 27 GB of additional scratch on the
75-draw production store, whose 36,746-point grid is 1.6x wider than earlier
two-draw measurements implied.

> **Note.** §1.2 quotes the same grid as "~36,745 points" and the T5 section as
> a "36,746-point grid"; the README used 36,746. Both figures already appear in
> this document and are left as written.

### What the matching stage actually matches

**Aggregate posterior age CDFs, not per-variant ages.** The target is the mean
of the TE sites' posterior age CDFs, and a control set is scored by the mean of
*its* sites' CDFs. Nothing requires any individual control SNP to resemble any
individual TE. On the 4,067-site in-gene target the aggregate CDF reaches 10% at
1,614 generations, while the median *site's own* CDF reaches 10% only at 6,845
— because 73.8% of sites put a little mass below 2,000 generations, and a little
mass from three quarters of the sites is most of the young tail. Reading the
aggregate 10% crossing as "10% of these TEs are younger than 1,614 generations"
is wrong, and it is the easiest mistake to make with this output.

**The acceptance threshold is a percentile of distances, not of ages.**
`te_age_target.py` resamples the TE sites 10,000 times with replacement, and for
each resample measures the Wasserstein-1 distance between the resampled age CDF
and the observed one. The threshold is the median of those 10,000 *distances* —
1,480.48 generations for the in-gene target. It answers "how far from the
observed age distribution does the TE sample's own sampling noise typically put
you?", and a control set is acceptable when it is no further away than that. It
is not the median TE age, nor an age quantile of any kind.

### Why the swap screen is geometric

Scoring every proposed swap on the exact 36,746-point analysis grid dominates
the run, so proposals are screened on a coarse grid first and every recorded
distance is then recertified exactly. The old coarse grid was uniform at
`--search-bin-width` 20,000 generations, and on the in-gene target that put
50.06% of the age mass inside its single first cell: the optimizer could not see
young-end structure at all and rejected young-improving swaps before the exact
grid ever evaluated them. The coarse grid is now a geometric sub-sample of the
exact grid, so the young end keeps full exact resolution. Measured effect on the
relative age error at the 10% CDF quantile: +21.9% under the uniform screen,
−0.2% under the geometric one, with no loss at the old end. Every coarse point
is an exact-grid point, so the screen is a sub-sample of the exact objective
rather than a different discretization, and a coarse misjudgement can cost
search efficiency but never the correctness of the published state.

> **Discrepancy to resolve.** The README recorded the geometric screen's
> relative age error at the 10% quantile as **−0.2%**. The "Recommended route"
> table above records **−0.0%**, `BOOTSTRAP_DISCARDED_APPROACHES.md`'s log-age
> table records **−0.0%** for the linear metric with a log screen and −0.1% for
> the log metric, and `CHANGELOG.md` v0.4.0 records **−0.0%**. The +21.9%
> uniform-screen figure is identical everywhere. Both values are kept; one of
> them needs correcting at source.

### Production-launcher resource measurements

Resource requests come from measurement, not from a formula. At
`run_bootstrap_matching.sbatch`'s defaults (6 CPUs, 96 GB, 12 h) a 4,067-site
target with 100 replicates × 3 restarts took 2 h 12 m wall clock and 37.7 GiB
peak RSS. Target construction is the other memory peak: 16.0 GB for a
35,512-site target. Scale `--time` roughly linearly in target size and replicate
count. Against the store on Quobyte the job is I/O-bound and projects to 4–21 h
of pure store reads for one target; staged to node-local scratch, the same run
measured 97% CPU with 21 major page faults. The production store is 18.2 GB.

> **Note.** T5 above records 34.9 GB peak RSS and 2 h 40 m wall clock for its
> 100-replicate run. That run used the uniform screen and non-disjoint
> replicates, so it is a different configuration and not directly comparable to
> the 37.7 GiB / 2 h 12 m figure; both are kept.

### What the bootstrap-target Φ-SFS distribution does and does not mean

- It **holds the observed TE SFS fixed** and varies only which SNPs are matched.
  It therefore measures how the matched-control comparison responds to
  uncertainty in the TE *age* CDF, and nothing else.
- It is **not a bootstrap confidence distribution** for the TE SFS, and **not a
  p-value**. Do not compute the TE SFS from the resampled TE rows: that would
  additionally propagate finite-TE-set SFS uncertainty and answer a different
  question. A joint age-and-SFS bootstrap would be a separately named analysis.
- 100 replicates give **weak tail resolution**, and disjoint membership does not
  make them 100 independent observations. `reuse_counts.npy` shows whether
  membership is shared — under `--disjoint-replicates` it is not — but shared
  membership is only one of the couplings.
- The implemented bootstrap is an iid multinomial TE-site bootstrap. Do not give
  the 100 replicates an inferential interpretation until spatial dependence
  among TE age contributions has been assessed (C1). If iid exchangeability is
  unsupported, a prespecified genomic-block bootstrap must replace it.

### Φ-SFS site assumptions

These are assumptions about the input, not things `phi_sfs.py` derives. Each is
recorded in the output `metadata.json`.

- **Biallelic.** Records are assumed biallelic, which is what the upstream
  preprocessing produces. A comma in ALT is treated as an error rather than
  split into separate alleles.
- **FILTER is ignored.** The declared input is the already-filtered
  preprocessing VCF, so every record at a requested coordinate is used
  regardless of its FILTER value.
- **Polarity comes from the ancestral table, not the VCF.** REF is not assumed
  ancestral and no INFO field is consulted. TE target sites are polarized
  biologically; control sites take the posterior-weighted mixture from
  `--ancestral-table`.
- **One allele per individual.** Each callable inbred individual contributes one
  observed allele. Haploid and homozygous diploid calls are accepted; a missing
  diploid allele makes that individual missing at the site. Heterozygous calls
  fail by default, and `--heterozygous missing` excludes those individuals from
  the site's callable count instead.
- Every requested site must be present in the VCF; the run fails listing the
  missing coordinates rather than analyzing a subset.
- **Normalization discards absolute scale.** Two sets with very different
  eligible-site counts, missingness, or endpoint mass can give identical
  spectra, and those differences are invisible in Φ. `replicates.csv` reports
  `input_sites`, `eligible_sites`, `dropped_n_lt_20`, `retained_fraction`, and
  `endpoint_fraction` for every set, with the matching `target_*` values in
  `metadata.json`. A target and a control set whose retained fractions differ
  substantially are not comparable however small Φ is.

`PHI_SFS_IMPLEMENTATION_PLAN.md` §2 carries the hypergeometric projection and
the four equivalent forms of Φ-SFS, including the identity that makes it the
total variation distance between the two projected, normalized spectra.
`phi_sfs.py` computes three of those forms independently and reports the
discrepancy as `identity_max_abs_error`.

## 6. Recording

Every test writes a durable record under `results/` containing the git commit,
`git_dirty`, the store `content_sha256`, all seeds, the full command, SLURM job
id, elapsed time, and `MaxRSS`. Plan §3 requires that pilot documentation
publish the **complete best-distance trace for every replicate**, not just final
ratios, because the traces are the evidence that the epoch budget was calibrated
rather than assumed. Apply that to T3, T4, and T5.

Update `BOOTSTRAP_TARGET_MATCHING_PLAN.md` §12 and the README's matching step
as gates close.

The original rule here — that the README's "not yet cleared for production"
banner and the §5-as-reference recommendation could not be relaxed while any
gate stayed open — has been discharged for the **matching** stages: every gate
in §5 above is closed or superseded, and the README now documents
bootstrap-target matching as the only matching step, with the hard-q50 sampler
moved to `BOOTSTRAP_DISCARDED_APPROACHES.md` as abandoned. It still binds the **Phi-SFS** stage, which remains uncleared while
C2's polarity-confidence extension, the effective-N analysis, and the chromosome
1-9 VCF are outstanding, and README says so.
