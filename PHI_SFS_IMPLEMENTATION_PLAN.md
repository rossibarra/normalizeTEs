# Φ-SFS Implementation Plan

## 1. Objective

Add a deterministic downstream analysis that compares the unfolded site
frequency spectrum (SFS) of a target TE set with the SFS of each of its 100
age-matched SNP control sets.

The analysis will:

1. discard sites observed in fewer than 20 inbred individuals;
2. project every eligible site probabilistically to 20 individuals with the
   hypergeometric distribution;
3. retain unfolded projected-frequency bins 1 through 19;
4. sum the unnormalized site-level projections within each set;
5. normalize the completed TE and SNP spectra independently; and
6. calculate one Φ-SFS score for each matched SNP set.

This is a downstream step. It must not change the target-building or matched-SNP
sampling algorithms.

## 2. Statistical definition

### 2.1 Site-level projection

Let a site have an observed derived-allele count `k` among `n` callable inbred
individuals. Sites with `n < 20` are excluded. For an eligible site, let `J` be
the derived-allele count after projection to `m = 20` individuals. Its expected
contribution to projected bin `j` is

$$
h_j(k,n)
= \Pr(J=j \mid k,n,m=20)
= \frac{\binom{k}{j}\binom{n-k}{20-j}}{\binom{n}{20}},
\qquad j=0,\ldots,20.
$$

The implementation must evaluate this as a stable hypergeometric probability
mass function, not by explicitly calculating large binomial coefficients and
not by performing a random downsampling draw.

The vector over bins 0 through 20 must sum to one within floating-point
tolerance. Projection mass in bins 0 and 20 is calculated for diagnostics but
is excluded from the SFS comparison.

An individual site's retained mass over bins 1 through 19 is generally less
than one. It must **not** be renormalized. Thus a site contributes

$$
\sum_{j=1}^{19} h_j(k,n)
= 1-h_0(k,n)-h_{20}(k,n)
$$

to the polymorphic SFS.

### 2.2 Target TE spectrum

For target TE sites indexed by `i`, first sum their expected site-level
contributions:

$$
T_j = \sum_{i \in \mathrm{TE}} h_j(k_i,n_i),
\qquad j=1,\ldots,19.
$$

Normalize only after all TE sites have been accumulated:

$$
t_j = \frac{T_j}{\sum_{\ell=1}^{19}T_\ell}.
$$

The target TE spectrum is computed once per target dataset.

### 2.3 Matched SNP spectra

For matched SNP set `r`, sum the site-level projections:

$$
S_{rj} = \sum_{i \in r} h_j(k_i,n_i),
\qquad j=1,\ldots,19,
$$

then normalize the completed spectrum:

$$
s_{rj} = \frac{S_{rj}}{\sum_{\ell=1}^{19}S_{r\ell}}.
$$

This is repeated independently for all 100 matched sets.

### 2.4 Φ-SFS

For matched set `r`, define the signed bin residual as

$$
d_{rj}=t_j-s_{rj}.
$$

The Φ-SFS statistic is the positive TE-minus-SNP residual mass:

$$
\boxed{
\Phi_{\mathrm{SFS},r}
= \sum_{j=1}^{19}\max(t_j-s_{rj},0)
}.
$$

Both final spectra sum to one, so the following identities must hold:

$$
\begin{aligned}
\Phi_{\mathrm{SFS},r}
&= \sum_{j=1}^{19} \max(s_{rj} - t_j,\, 0) \\
&= \frac{1}{2} \sum_{j=1}^{19} \lvert t_j - s_{rj} \rvert \\
&= 1 - \sum_{j=1}^{19} \min(t_j,\, s_{rj})
\end{aligned}
$$

The last form is the identity that makes Φ-SFS the total variation distance
between the two projected, normalized spectra.

Consequently, `0 <= Φ-SFS <= 1`. The scalar is symmetric even though the
stored bin-level residuals are oriented as TE minus SNP.

The intended interpretation is:

> Φ-SFS is the fraction of normalized TE-SFS mass that does not overlap the
> normalized SFS of a matched SNP set across projected unfolded frequency bins
> 1 through 19.

This is a descriptive SFS non-overlap statistic, not a causal mixture estimate.

## 3. Required inputs and integration contract

### 3.1 Existing matched-control products

The analysis should consume the existing published matched-control directory:

- `row_indices.npy`: matched SNP store rows, shaped as sets by sites;
- `positions.npy`: aligned matched SNP positions;
- `chromosome_codes.npy` and `chromosome_labels.npy`;
- `chain_index.npy`: chain identity for each matched set;
- `sample_index.npy`: sample identity within each chain;
- `metadata.json`: store identity, target identity, seeds, and provenance; and
- reuse diagnostics, where available.

The target directory supplies the corresponding resolved target information,
including `te_row_indices.npy`, `te_chromosomes.npy`, `te_positions.npy`, and
its metadata.

The analysis must validate that target and matched-control metadata identify
the same target and compatible source store before calculating an SFS.

Comparing store hashes is necessary but **not sufficient** for this: every
target built from one SNP store shares them, so store identity alone cannot
distinguish a matched bundle sampled for this target from one sampled for a
different TE category. The binding check is `target_digest`, which the matcher
computes over the target's row indices, mean CDF, age grid, and acceptance
threshold. The analysis recomputes it from the target directory — using the
matcher's own loader and hash helper, so the two cannot drift apart — and
requires equality. It also requires the matched bundle to be marked complete.

The store hashes must be present in both bundles and must agree, which rejects
a hand-built or truncated bundle carrying no store identity at all. They are
not required to be non-null. `content_sha256` already subsumes
`catalog_sha256`, and the dense store records neither, so requiring a non-null
digest would make this the only step that rejects a dense-store bundle the
matcher itself accepts.

Row indices are validated but not re-resolved. `te_row_indices.npy` and
`row_indices.npy` must align in shape with their coordinate arrays, must be
non-negative, and must contain no duplicate control within a set. Re-resolving
coordinates against the interval store would make this otherwise store-free
step depend on the store for a failure mode these checks already cover.

### 3.2 Frequency-data source

The SNP-age store identifies sites and their age distributions but does not
contain the observed derived count `k` or callable count `n`. The authoritative
frequency source for both TE and matched SNP sites is the polarized, biallelic
VCF used for the analysis. A site's frequency is calculated once from the VCF;
it does not depend on how many posterior ARGs contain that SNP. Posterior ARG
representation affects the mutation-age distribution and matching stage only.

The source adapter must return, for each canonical site:

- chromosome and position;
- a stable site identifier when available;
- observed callable inbred-individual count `n`;
- observed derived-allele count `k`;
- the VCF REF and ALT alleles and their polarization status; and
- enough provenance to demonstrate that TE and SNP counts came from the same
  samples and filtering rules.

The implementation uses chromosome and one-based VCF position to resolve the
project's biallelic records and fails on duplicate records. It supports either
REF-as-ancestral polarized VCFs or a configurable ancestral INFO field. The
ancestral allele must equal REF or ALT. When REF is ancestral, `k` is the ALT
count; when ALT is ancestral, `k` is the REF count.

Each inbred individual contributes one allele: haploid and homozygous diploid
calls are accepted. A partially or wholly missing genotype is not callable.
Heterozygous calls either fail or are treated as missing under an explicit CLI
policy. Sites with fewer than 20 callable individuals are then dropped.

The following validation policies apply:

1. how canonical age-store rows resolve to frequency-source records;
2. how chromosome offsets are converted back to native coordinates;
3. how duplicate positions are distinguished;
4. multiallelic sites are rejected;
5. how recurrent or multiple mutations at one ARG site are treated;
6. missing genotypes do not contribute to `n`; and
7. ancestral-state or derived-state conflicts are rejected.

Resolution must fail clearly on ambiguity. It must not select the first record
at a duplicated position silently.

## 4. Proposed software structure

Implement Φ-SFS as a new downstream module and command-line program rather
than adding it to `sample_age_matched_controls.py`.

The internal design should separate five responsibilities:

1. **Input validation and site resolution**: validate the target and matched
   bundles, resolve all required sites against the frequency source, and obtain
   `k` and `n`.
2. **Projection**: calculate a deterministic 21-bin hypergeometric probability
   vector for every unique `(k, n)` pair. Cache repeated pairs.
3. **Spectrum accumulation**: sum unnormalized projections for the target and
   for each matched set, retaining endpoint and filtering diagnostics.
4. **Normalization and Φ-SFS**: normalize bins 1 through 19, calculate signed
   residuals, positive residual contributions, overlap, and identity checks.
5. **Output and provenance**: write atomic, machine-readable result artifacts
   with complete input identities and analysis parameters.

Resolve and project unique sites once, then gather their projection vectors per
matched set. This avoids repeating genotype lookup and hypergeometric
calculations when controls are reused across matched sets.

The implementation caches one level deeper than "unique row": it keys the
projection cache on the distinct `(k, n)` pair rather than on the site, because
`h_j(k,n)` depends only on those two numbers. Many sites share a pair, so the
number of hypergeometric evaluations is bounded by the number of distinct
`(k, n)` combinations rather than by the number of sites — on representative
data this is hundreds of evaluations for millions of site slots.

Accumulation then gathers by that cache row: each set counts how many of its
eligible sites map to each distinct pair and takes one matrix product against
the `(n_distinct, 19)` projection matrix, rather than adding one length-19
vector per site. Measured at 100 sets by 10,000 sites, this is about 8 times
faster than per-site accumulation and numerically identical to within 1e-13.

## 5. Processing workflow

### Phase A: validate the analysis bundle

1. Open and validate the target and matched-control directories.
2. Confirm that the matched sets have the expected shape and aligned chain and
   sample arrays.
3. Confirm target identity, store schema, store-content identity, and target
   digest compatibility.
4. Verify that every referenced row is in range and that aligned chromosome and
   position arrays agree with the canonical store.
5. Record the requested projection size as 20 inbred individuals.

### Phase B: resolve frequencies

1. Form the union of target rows and unique matched SNP rows.
2. Resolve each row against the authoritative frequency source.
3. Validate `k` and `n` as integers satisfying `0 <= k <= n`.
4. Exclude sites with `n < 20` and record the reason.
5. Apply the fixed multiallelic, duplicate, mutation, missingness, and
   polarization policies.
6. Produce an auditable resolution summary before spectrum calculation.

### Phase C: project sites

1. Compute `h_j(k,n)` for all bins 0 through 20.
2. Verify that each complete probability vector sums to one.
3. Retain bins 1 through 19 without site-level renormalization.
4. Record the excluded bin-0 and bin-20 probability mass.
5. Cache results by `(k, n)` and associate them with resolved site rows.

### Phase D: calculate spectra and Φ-SFS

1. Sum the target TE site vectors to obtain `T_j`.
2. Normalize `T_j` over bins 1 through 19 to obtain `t_j`.
3. For each matched row, sum its SNP site vectors to obtain `S_rj`.
4. Normalize `S_rj` over bins 1 through 19 to obtain `s_rj`.
5. Calculate `d_rj = t_j - s_rj` and `max(d_rj, 0)`.
6. Sum positive residuals to obtain `Φ_SFS_r`.
7. Independently calculate reverse-positive residual mass and half the L1
   distance as computational checks.
8. Preserve `chain_index` and `sample_index` with every result.

### Phase E: summarize and diagnose

Summarize the 100 Φ-SFS values with the mean, median, standard deviation,
range, and selected quantiles. These describe variation among the matched sets;
they are not automatically a confidence interval based on 100 independent
replicates.

Inspect Φ-SFS by chain and sample order. The current design produces ten sets
from each of ten swap chains, so within-chain autocorrelation and repeated use
of control SNPs must be considered when interpreting dispersion or effective
replicate count.

## 6. Output specification

Write a new result directory atomically. At minimum, include the following.

### 6.1 Replicate-level table

One row per matched SNP set. `replicates.csv` carries, in order:

- `replicate`, the matched-set index;
- `chain_index` and `sample_index`;
- `input_sites` and `eligible_sites`;
- `dropped_n_lt_20`;
- `retained_mass`, the raw retained polymorphic projection mass;
- `endpoint_mass`, the excluded bin-0 plus bin-20 mass;
- `retained_fraction` and `endpoint_fraction`, each divided by
  `eligible_sites`, because the absolute masses are not comparable across sets
  with different eligible-site counts;
- `phi_sfs`;
- `overlap`, `1 - Φ_SFS`;
- `reverse_positive` and `half_l1`; and
- `identity_max_abs_error`, the largest disagreement among the three forms.

There is no per-set count of sites "rejected for each other reason": every
rejection cause other than `n < 20` — multiallelic ALT, unresolvable ancestral
state, heterozygosity under the `error` policy, an absent record — is a hard
failure of the whole run rather than a silent per-site exclusion, so such a
count would be identically zero. The target label is not duplicated into the
table; the target identity lives once in `metadata.json`.

### 6.2 Bin-level table

One row per matched set and bin, containing:

- target and matched-set identifiers;
- chain and sample identifiers;
- unfolded derived-count bin, 1 through 19;
- raw TE projected mass `T_j`;
- normalized TE mass `t_j`;
- raw matched-SNP projected mass `S_rj`;
- normalized matched-SNP mass `s_rj`;
- signed residual `t_j - s_rj`; and
- positive TE residual contribution.

### 6.3 Target diagnostics

Record the target's input, eligible, and rejected site counts; raw retained
mass; endpoint masses; normalized spectrum; and site-resolution diagnostics.

### 6.4 Metadata

Record:

- schema and algorithm versions;
- exact command and parameters;
- projection size and retained bins;
- target and matched-control identities;
- frequency-source path and content identity;
- ancestral-state and site-resolution policies;
- dependency versions;
- creation time; and
- output completeness status.

## 7. Edge cases and failure behavior

The analysis must fail rather than emit a misleading score when:

- the target has no eligible sites;
- the target's retained mass across bins 1 through 19 is zero;
- a matched set has no eligible sites;
- a matched set's retained mass is zero;
- `k` or `n` is invalid;
- site resolution is ambiguous;
- target and matched bundles are incompatible; or
- polarization cannot be established under the selected policy.

Dropping `n < 20` sites is expected behavior, not a failure, but the retained
fractions for the target and every matched set must be reported. Large
differences in missingness or endpoint mass should be prominently flagged
because final normalization can conceal them.

## 8. Validation and test plan

### 8.1 Projection unit tests

- Probabilities across bins 0 through 20 sum to one.
- For `n = 20`, the projection is a point mass at `j = k`.
- A site with `n < 20` is excluded.
- Boundary values such as `k = 0` and `k = n` follow the selected eligibility
  policy and produce the expected endpoint mass.
- Invalid values (`k < 0`, `n < 0`, `k > n`, or nonintegers) fail clearly.
- Numerically challenging large-`n` cases remain finite and normalized.
- Expected projection agrees with a large Monte Carlo downsampling check on a
  small synthetic fixture.

### 8.2 Spectrum tests

- Individual site projections are not renormalized after removing endpoints.
- Accumulation is invariant to input site order.
- Final TE and SNP spectra each sum to one over bins 1 through 19.
- Unfolded high-frequency bins remain separate from low-frequency bins.
- Reused SNP rows produce identical cached projection vectors.

### 8.3 Φ-SFS tests

- Identical normalized spectra give `Φ-SFS = 0`.
- Disjoint spectra give `Φ-SFS = 1`.
- A hand-calculated partial-overlap example gives the expected value.
- The positive TE residual sum equals the positive SNP residual sum.
- Both equal half the L1 distance within tolerance.
- Every score is finite and lies in `[0, 1]` within tolerance.

### 8.4 Integration tests

- A small target and multiple matched sets produce correctly aligned
  replicate- and bin-level outputs.
- Chain and sample identities survive unchanged.
- Duplicate-position and multiallelic fixtures exercise the declared policies.
- Mixed callable sample sizes exercise the `n >= 20` filter.
- Rerunning identical inputs produces byte-equivalent numerical arrays or
  equivalent values under a declared metadata timestamp exception.
- Incompatible target, store, or matched-set identities are rejected.
- Interrupted output creation cannot expose a directory marked complete.

## 9. Interpretation and reporting

Report the full distribution of 100 Φ-SFS values and retain bin-level signed
residuals. A scalar Φ-SFS shows total non-overlap but not whether the difference
comes from rare, intermediate-frequency, or high-frequency derived variants.

Recommended diagnostics are:

1. all 100 Φ-SFS values, labeled or grouped by chain;
2. Φ-SFS against within-chain sample order;
3. target TE SFS against the mean or median matched-SNP SFS;
4. signed residual curves across bins 1 through 19; and
5. a matched-set by frequency-bin residual heatmap.

ARG-derived polarization supports an unfolded analysis, but polarization error
can transfer mass between rare and high-frequency bins. If an ancestral-state
confidence measure is available, a threshold-based sensitivity analysis is a
useful later extension; it is not required for the first implementation.

## 10. Acceptance criteria

The implementation is ready for production when:

1. the authoritative `k`/`n` source and all resolution policies are documented;
2. target and matched sites are resolved without silent ambiguity;
3. projection is deterministic and passes the probability tests;
4. sites are filtered at `n >= 20` and projected to 20 inbred individuals;
5. bins 1 through 19 are retained without site-level renormalization;
6. final TE and per-set SNP spectra are normalized independently;
7. exactly one validated Φ-SFS score is produced per matched set;
8. all three equivalent Φ-SFS calculations agree within tolerance;
9. chain, sample, filtering, endpoint-mass, and provenance diagnostics are
   preserved; and
10. unit and integration tests pass on both synthetic and representative
    project data.

## 11. Recommended implementation order

1. Validate the polarized VCF source and resolution policies on representative
   target and matched sites.
2. Implement and test deterministic site-level projection.
3. Implement target and matched-set accumulation and normalization.
4. Implement Φ-SFS and its mathematical consistency checks.
5. Implement the downstream CLI and atomic result schema.
6. Add end-to-end fixtures covering missingness, polarization, endpoint mass,
   duplicate positions, and matched-set metadata.
7. Run a small real-data pilot and inspect filtering, endpoint mass, chain
   behavior, SNP reuse, and residual-by-bin diagnostics before full production.

## 12. Implementation status

Recorded after the round 6 review (`CODE_REVIEW_ROUND6.md`) so that this plan
does not imply guarantees the code does not provide.

### Implemented

Sections 2 through 4, 5 phases A through D, 6.1 through 6.4, and 7 are
implemented in `phi_sfs.py`. This includes `target_digest` binding of the
matched bundle to the target, row-index alignment and duplicate checks, the
distinct-`(k, n)` projection cache and gathered accumulation, retained and
endpoint fractions, single-pass VCF digesting, and the standard
`release_provenance` software and Git provenance carried by every other durable
output in this pipeline.

Section 8's projection, spectrum, and Φ-SFS test groups are covered in
`tests/test_phi_sfs.py`, including a random-subsampling cross-check of the
closed-form projection and hand-calculated end-to-end scores.

### Deferred

These were specified above but are **not** implemented. They are optional
analysis conveniences, not correctness requirements, and each is derivable from
the published arrays.

1. **Phase E summary statistics** (section 5). The mean, median, standard
   deviation, range, and quantiles of the 100 Φ-SFS values are not written to
   `metadata.json`. Compute them from `phi_sfs.npy`, which is published with
   aligned `chain_index.npy` and `sample_index.npy`.
2. **Phase E chain diagnostics** (section 5) and the plots in section 9. The
   inputs are published; the plotting is left to downstream analysis.
3. **Automatic flagging of large retained-fraction differences** (section 7).
   The fractions are now reported per set and for the target, but nothing
   compares them and warns. Inspect `retained_fraction` in `replicates.csv`
   against `target_retained_fraction` in `metadata.json`.
4. **Ancestral-state confidence sensitivity analysis** (section 9). Explicitly
   scoped as a later extension in the original plan. Lowercase ancestral
   annotations are currently rejected rather than treated as a confidence
   tier.

### Deliberately rejected

1. **Enforcing `FILTER = PASS`.** The declared input is the already-filtered
   preprocessing VCF. Skipping non-PASS records would reclassify them as
   "requested sites absent from the VCF" and fail the run with a misleading
   error. FILTER is ignored, and the policy is recorded in the output metadata.
2. **Rejecting non-single-base REF/ALT.** Multiallelic records are excluded
   upstream, so the biallelic property is an input assumption, documented in
   the README and the output metadata rather than re-derived here.
3. **Concurrent-writer hardening of the output publication.** The
   `mkdtemp` plus `os.replace` pattern matches `te_age_target.write_target` and
   the matcher; the manifest design already requires unique output directories.
