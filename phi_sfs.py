#!/usr/bin/env python3
"""Calculate Phi-SFS for a TE target and its matched SNP control sets.

Phi-SFS is the total variation distance between the projected, normalized,
unfolded site frequency spectrum of a TE target set and that of one age-matched
SNP control set. README section 6 and PHI_SFS_IMPLEMENTATION_PLAN.md section 2
carry the full derivation.

Input assumptions, all of which are recorded in the output metadata:

* The VCF is biallelic. Multiallelic records are produced upstream only as
  separate biallelic records, so a comma in ALT is treated as an error rather
  than split here.
* The VCF FILTER column is ignored. The declared input is the already
  filtered, already polarized preprocessing VCF, so every record at a
  requested coordinate is used.
* Ancestral alleles are matched case-sensitively. A lowercase INFO value
  conventionally marks a low-confidence call and is rejected rather than
  silently folded to upper case.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

from release_provenance import software_provenance
from sample_age_matched_controls import _load_target, _sha256_arrays


SCHEMA_VERSION = "phi-sfs-v1"
PROJECTION_SIZE = 20
RETAINED_BINS = np.arange(1, PROJECTION_SIZE, dtype=np.int64)
COMPRESSED_SUFFIXES = (".gz", ".bgz", ".bgzf")
PROGRESS_RECORDS = 5_000_000

_UNSET = object()
_GENOTYPE_ERRORS = {
    "invalid": "invalid GT {gt!r} at {chrom}:{position}",
    "non-biallelic": "non-biallelic GT {gt!r} at {chrom}:{position}",
    "heterozygous": "heterozygous GT {gt!r} at inbred site {chrom}:{position}",
}


@dataclass(frozen=True)
class SiteCount:
    derived: int
    callable: int


class PhiResult(NamedTuple):
    """Phi-SFS with the bin-level residuals and the two identity checks."""

    value: float
    residual: np.ndarray
    positive: np.ndarray
    reverse_positive: float
    half_l1: float


def hypergeometric_projection(k: int, n: int, m: int = PROJECTION_SIZE) -> np.ndarray:
    """Return the expected derived-count distribution after projection to m.

    A site observed with `k` derived alleles among `n` callable inbred
    individuals contributes probability mass to projected bin `j` equal to

        h_j(k, n) = C(k, j) * C(n - k, m - j) / C(n, m),   j = 0, ..., m,

    the hypergeometric probability of drawing `j` derived alleles in a sample
    of `m` drawn without replacement. This is the exact expectation over all
    subsamples, not a random downsampling draw, so the result is deterministic.

    The returned vector covers bins 0 through m inclusive and sums to one.
    Sites with `n < m` cannot be projected and are rejected here; callers drop
    them before reaching this function. Probabilities are evaluated in log
    space via `lgamma` so that large `n` stays finite: the identity holds to
    full double precision at n = 2e6.
    """
    if not isinstance(k, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise TypeError("k and n must be integers")
    k, n = int(k), int(n)
    if n < m:
        raise ValueError(f"cannot project n={n} observations to m={m}")
    if k < 0 or k > n:
        raise ValueError(f"derived count k={k} must satisfy 0 <= k <= n={n}")
    result = np.zeros(m + 1, dtype=np.float64)
    lower = max(0, m - (n - k))
    upper = min(m, k)
    denominator = math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)
    for j in range(lower, upper + 1):
        log_probability = (
            math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(k - j + 1)
            + math.lgamma(n - k + 1)
            - math.lgamma(m - j + 1)
            - math.lgamma(n - k - (m - j) + 1)
            - denominator
        )
        result[j] = math.exp(log_probability)
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError(f"invalid hypergeometric projection for k={k}, n={n}, m={m}")
    result /= total
    return result


def project_sites(
    counts: dict[tuple[str, int], SiteCount],
) -> tuple[dict[tuple[str, int], int], np.ndarray, np.ndarray]:
    """Project every eligible site, evaluating each distinct (k, n) pair once.

    Sites with fewer than `PROJECTION_SIZE` callable individuals are ineligible
    and are absent from the returned coordinate map; that is the `n >= 20`
    filter, and it is expected behaviour rather than an error.

    Each returned row holds bins 1 through 19 of one site's projection and is
    deliberately **not** renormalized after the endpoint bins are removed, so
    it sums to `1 - h_0 - h_20` rather than to one. A site whose derived count
    is very likely to project to 0 or 20 therefore contributes proportionally
    less polymorphic mass, which is the intended weighting. The companion
    endpoint array holds the excluded `h_0 + h_20` mass for diagnostics.

    Returns the coordinate-to-row map, an (n_distinct, 19) projection matrix,
    and the aligned (n_distinct,) endpoint masses.
    """
    rows: dict[tuple[str, int], int] = {}
    distinct: dict[SiteCount, int] = {}
    retained: list[np.ndarray] = []
    endpoints: list[float] = []
    for coordinate, item in counts.items():
        if item.callable < PROJECTION_SIZE:
            continue
        row = distinct.get(item)
        if row is None:
            vector = hypergeometric_projection(item.derived, item.callable)
            row = len(retained)
            distinct[item] = row
            retained.append(vector[1:PROJECTION_SIZE])
            endpoints.append(float(vector[0] + vector[PROJECTION_SIZE]))
        rows[coordinate] = row
    projections = (
        np.asarray(retained, dtype=np.float64) if retained
        else np.zeros((0, PROJECTION_SIZE - 1), dtype=np.float64)
    )
    return rows, projections, np.asarray(endpoints, dtype=np.float64)


def accumulate_spectrum(
    coordinates: Sequence[tuple[str, int]],
    rows: dict[tuple[str, int], int],
    projections: np.ndarray,
    endpoints: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Sum the projections of the eligible members of one site set.

    Sites are gathered by distinct (k, n) row and weighted by how often that
    row occurs, so a set of N sites costs one length-19 matrix product rather
    than N array additions. Returns the unnormalized bins 1 through 19, the
    excluded endpoint mass, and the number of eligible sites.
    """
    indices = np.fromiter(
        (rows[coordinate] for coordinate in coordinates if coordinate in rows),
        dtype=np.int64,
    )
    weights = np.bincount(indices, minlength=projections.shape[0]).astype(np.float64)
    return weights @ projections, float(weights @ endpoints), int(indices.size)


def normalized_spectrum(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one accumulated spectrum over bins 1 through 19.

    Normalization happens once per set, after every site has contributed, so a
    site's weight in the final spectrum is proportional to its retained
    projection mass. Normalizing site by site instead would give every eligible
    site equal weight and would discard the endpoint-mass information.

    Normalization also discards the set's absolute scale, so differences in
    eligible-site count and total retained mass become invisible in the final
    spectrum. Callers must report the retained fractions separately.
    """
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (PROJECTION_SIZE - 1,):
        raise ValueError("spectrum must contain bins 1 through 19")
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("spectrum has zero retained mass in bins 1 through 19")
    return values, values / total


def phi_sfs(te: np.ndarray, snp: np.ndarray) -> PhiResult:
    """Return Phi-SFS and its bin-level residuals for two normalized spectra.

    Phi-SFS is the total variation distance between the two spectra. All four
    of these forms are equal, because both spectra sum to one:

        Phi = sum_j max(t_j - s_j, 0)      positive TE-minus-SNP residual mass
            = sum_j max(s_j - t_j, 0)      positive SNP-minus-TE residual mass
            = (1/2) sum_j |t_j - s_j|      half the L1 distance
            = 1 - sum_j min(t_j, s_j)      one minus the overlapping mass

    Phi therefore lies in [0, 1] and is symmetric in its two arguments, even
    though the returned bin residuals are oriented as TE minus SNP. Zero means
    the two spectra coincide; one means they share no mass in any bin.

    The two alternative forms are returned alongside the score and are checked
    against it here. The identity is algebraic rather than contingent, so this
    is a cheap tripwire against a malformed input reaching this function, not a
    test of the statistic.
    """
    te = np.asarray(te, dtype=np.float64)
    snp = np.asarray(snp, dtype=np.float64)
    expected = (PROJECTION_SIZE - 1,)
    if te.shape != expected or snp.shape != expected:
        raise ValueError("normalized spectra must contain bins 1 through 19")
    if not np.isclose(te.sum(), 1.0) or not np.isclose(snp.sum(), 1.0):
        raise ValueError("both spectra must be normalized")
    residual = te - snp
    positive = np.maximum(residual, 0.0)
    value = float(positive.sum())
    reverse = float(np.maximum(-residual, 0.0).sum())
    half_l1 = float(np.abs(residual).sum() / 2.0)
    if not (np.isclose(value, reverse) and np.isclose(value, half_l1)):
        raise RuntimeError("Phi-SFS consistency identity failed")
    return PhiResult(value, residual, positive, reverse, half_l1)


class _HashingStream(io.RawIOBase):
    """Raw byte stream that digests everything read through it."""

    def __init__(self, handle):
        self._handle = handle
        self.digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        read = self._handle.readinto(buffer)
        if read:
            self.digest.update(memoryview(buffer)[:read])
        return read

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            super().close()


def _open_vcf(path: Path):
    """Open a VCF as text over a hashing stream, so one pass yields both.

    Returns the text handle, the hashing stream, and the buffered byte stream,
    so that the caller can drain any bytes the text layer did not consume
    before reading the digest.
    """
    hashing = _HashingStream(path.open("rb"))
    buffered = io.BufferedReader(hashing, buffer_size=1 << 20)
    compressed = path.suffix.lower() in COMPRESSED_SUFFIXES
    stream = gzip.GzipFile(fileobj=buffered) if compressed else buffered
    return io.TextIOWrapper(stream, encoding="utf-8"), hashing, buffered


def _parse_info(info: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    if info == ".":
        return result
    for item in info.split(";"):
        key, separator, value = item.partition("=")
        result[key] = value if separator else None
    return result


def _decode_genotype(gt: str, heterozygous: str) -> int | None | str:
    """Return one individual's allele, None when not callable, or an error tag.

    Each inbred individual contributes a single observed allele, so haploid and
    homozygous diploid calls are accepted and any missing allele makes the
    whole individual uncallable. Results are cached by the caller because
    genotype strings are drawn from a very small alphabet.
    """
    alleles = gt.replace("|", "/").split("/")
    if not alleles or any(allele == "." for allele in alleles):
        return None
    values = []
    for allele in alleles:
        if not allele.isdigit():
            return "invalid"
        value = int(allele)
        if value not in (0, 1):
            return "non-biallelic"
        values.append(value)
    if len(set(values)) > 1:
        return None if heterozygous == "missing" else "heterozygous"
    return values[0]


def read_site_counts(
    vcf: Path,
    coordinates: set[tuple[str, int]],
    *,
    ancestral_mode: str,
    ancestral_info: str,
    heterozygous: str,
    progress: bool = True,
) -> tuple[dict[tuple[str, int], SiteCount], str]:
    """Read callable and polarized derived counts for the requested coordinates.

    Returns the per-coordinate counts and the SHA-256 of the VCF bytes, which
    is accumulated during this single pass rather than in a second full read.

    Only CHROM and POS are parsed for records that are not requested, because
    splitting every sample column of every record dominates the scan on a
    sample-rich VCF.
    """
    found: dict[tuple[str, int], SiteCount] = {}
    genotypes: dict[str, int | None | str] = {}
    handle, hashing, buffered = _open_vcf(vcf)
    next_report = PROGRESS_RECORDS
    records = 0
    try:
        for line_number, raw in enumerate(handle, 1):
            if raw.startswith("#"):
                continue
            records += 1
            if progress and records >= next_report:
                print(f"  scanned {records:,} VCF records", flush=True)
                next_report += PROGRESS_RECORDS
            try:
                chrom, position_text, rest = raw.split("\t", 2)
            except ValueError:
                raise ValueError(f"{vcf}:{line_number}: malformed VCF record") from None
            try:
                position = int(position_text)
            except ValueError as error:
                raise ValueError(f"{vcf}:{line_number}: invalid POS") from error
            coordinate = (chrom, position)
            if coordinate not in coordinates:
                continue
            if coordinate in found:
                raise ValueError(f"duplicate VCF record at {chrom}:{position}")
            fields = rest.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"{vcf}:{line_number}: expected VCF samples and FORMAT")
            ref, alt, info, formats = fields[1], fields[2], fields[5], fields[6]
            if "," in alt:
                raise ValueError(f"multiallelic VCF record at {chrom}:{position}")
            if ancestral_mode == "ref":
                ancestral = ref
            else:
                value = _parse_info(info).get(ancestral_info)
                if value is None or "," in value:
                    raise ValueError(
                        f"missing or ambiguous INFO/{ancestral_info} at {chrom}:{position}"
                    )
                ancestral = value
            if ancestral not in (ref, alt):
                raise ValueError(
                    f"ancestral allele {ancestral!r} is neither REF {ref!r} nor ALT "
                    f"{alt!r} at {chrom}:{position}; ancestral alleles are compared "
                    "case-sensitively, and a lowercase value conventionally marks a "
                    "low-confidence call that this analysis does not accept"
                )
            format_fields = formats.split(":")
            if "GT" not in format_fields:
                raise ValueError(f"VCF record lacks GT at {chrom}:{position}")
            gt_index = format_fields.index("GT")
            alt_count = 0
            callable_count = 0
            for sample in fields[7:]:
                if gt_index:
                    parts = sample.split(":")
                    gt = parts[gt_index] if gt_index < len(parts) else "."
                else:
                    end = sample.find(":")
                    gt = sample if end < 0 else sample[:end]
                allele = genotypes.get(gt, _UNSET)
                if allele is _UNSET:
                    allele = _decode_genotype(gt, heterozygous)
                    genotypes[gt] = allele
                if allele is None:
                    continue
                if allele.__class__ is str:
                    raise ValueError(_GENOTYPE_ERRORS[allele].format(
                        gt=gt, chrom=chrom, position=position,
                    ))
                alt_count += allele
                callable_count += 1
            derived = alt_count if ancestral == ref else callable_count - alt_count
            found[coordinate] = SiteCount(derived=derived, callable=callable_count)
        while buffered.read(1 << 20):
            pass
    finally:
        handle.close()
    if progress:
        print(f"  scanned {records:,} VCF records", flush=True)
    return found, hashing.digest.hexdigest()


def _load_integers(path: Path, label: str) -> np.ndarray:
    """Load an integer array, refusing to coerce a non-integer dtype.

    Casting first would silently truncate: a position or row index stored as
    2.9 would become 2 and resolve to the wrong site rather than failing.
    """
    values = np.load(path, allow_pickle=False)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} must be an integer array, not {values.dtype}")
    return values.astype(np.int64)


def _load_coordinates(target: Path, matches: Path):
    """Load and cross-validate the target and matched-control site arrays."""
    te_chromosomes = np.load(target / "te_chromosomes.npy", allow_pickle=False).astype(str)
    labels = np.load(matches / "chromosome_labels.npy", allow_pickle=False).astype(str)
    te_positions = _load_integers(target / "te_positions.npy", "target positions")
    te_rows = _load_integers(target / "te_row_indices.npy", "target row indices")
    positions = _load_integers(matches / "positions.npy", "matched positions")
    codes = _load_integers(matches / "chromosome_codes.npy", "matched chromosome codes")
    rows = _load_integers(matches / "row_indices.npy", "matched row indices")
    chains = _load_integers(matches / "chain_index.npy", "chain indices")
    samples = _load_integers(matches / "sample_index.npy", "sample indices")
    if te_chromosomes.shape != te_positions.shape or te_chromosomes.ndim != 1:
        raise ValueError("target chromosome and position arrays are not aligned 1-D arrays")
    if te_rows.shape != te_positions.shape:
        raise ValueError("target row indices do not align with target positions")
    if positions.shape != codes.shape or positions.ndim != 2:
        raise ValueError("matched chromosome codes and positions are not aligned 2-D arrays")
    if rows.shape != positions.shape:
        raise ValueError("matched row indices do not align with matched positions")
    if np.any(te_rows < 0) or np.any(rows < 0):
        raise ValueError("row indices must be non-negative")
    if chains.shape != (positions.shape[0],) or samples.shape != (positions.shape[0],):
        raise ValueError("chain/sample arrays do not align with matched sets")
    if np.any(codes < 0) or np.any(codes >= labels.size):
        raise ValueError("matched chromosome code is out of range")
    ordered = np.sort(rows, axis=1)
    if ordered.shape[1] > 1 and np.any(np.diff(ordered, axis=1) == 0):
        raise ValueError("a matched control set contains duplicate control rows")
    match_chromosomes = labels[codes]
    return te_chromosomes, te_positions, match_chromosomes, positions, chains, samples


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_provenance(target: Path, matches: Path) -> tuple[dict, dict, str]:
    """Require that the matched bundle was built from this exact target.

    The store hashes alone cannot establish this, because every target built
    from one SNP store shares them. The matcher records `target_digest`, a hash
    over the target's row indices, mean CDF, age grid, and acceptance
    threshold, so recomputing it from the target directory is what actually
    binds the two bundles together. The digest is computed with the matcher's
    own loader and hash helper so the two cannot drift apart.

    The store hashes must be *present* in both bundles and must agree, which
    rejects a hand-built or truncated bundle that carries no store identity at
    all. They are not required to be non-null: the dense store records neither
    digest, and every other step in this pipeline compares them only when a
    value exists, so demanding a non-null digest here would make this the one
    step that rejects a bundle the rest of the pipeline produced and accepted.
    """
    target_rows, target_cdf, age_bins, threshold, target_meta = _load_target(target)
    match_meta = _json(matches / "metadata.json")
    if match_meta.get("complete") is not True:
        raise ValueError(f"matched-control bundle is not marked complete: {matches}")
    for key in ("source_store_content_sha256", "source_catalog_sha256"):
        if key not in target_meta or key not in match_meta:
            raise ValueError(
                f"target and matched-control metadata must both record {key}"
            )
        if target_meta[key] != match_meta[key]:
            raise ValueError(f"target and matched-control {key} values differ")
    expected = match_meta.get("target_digest")
    if not expected:
        raise ValueError(f"matched-control metadata records no target_digest: {matches}")
    digest = _sha256_arrays(
        target_rows, target_cdf, age_bins, np.asarray([threshold], dtype=np.float64)
    )
    if digest != expected:
        raise ValueError(
            "matched-control bundle was built for a different target: "
            f"it records target_digest {expected}, but {target} hashes to {digest}"
        )
    return target_meta, match_meta, digest


def calculate(args: argparse.Namespace) -> None:
    target_meta, match_meta, target_digest = _validate_provenance(args.target, args.matches)
    te_chrom, te_pos, snp_chrom, snp_pos, chains, samples = _load_coordinates(
        args.target, args.matches
    )
    te_coordinates = list(zip(te_chrom.tolist(), te_pos.tolist()))
    snp_coordinates = [
        list(zip(snp_chrom[row].tolist(), snp_pos[row].tolist()))
        for row in range(snp_pos.shape[0])
    ]
    if not snp_coordinates:
        raise ValueError("matched-control bundle contains no sets")
    requested = set(te_coordinates)
    for row in snp_coordinates:
        requested.update(row)
    print(
        f"Scanning {args.vcf} for {len(requested):,} requested sites "
        f"across {len(snp_coordinates)} matched sets",
        flush=True,
    )
    counts, vcf_sha256 = read_site_counts(
        args.vcf,
        requested,
        ancestral_mode=args.ancestral_mode,
        ancestral_info=args.ancestral_info,
        heterozygous=args.heterozygous,
        progress=not args.quiet,
    )
    missing = sorted(requested.difference(counts))
    if missing:
        preview = ", ".join(f"{chrom}:{pos}" for chrom, pos in missing[:10])
        raise ValueError(f"{len(missing)} requested sites are absent from the VCF: {preview}")

    site_rows, projections, endpoints = project_sites(counts)
    print(
        f"Projected {projections.shape[0]:,} distinct (k, n) pairs "
        f"covering {len(site_rows):,} eligible sites",
        flush=True,
    )

    te_counts, te_endpoint, te_eligible = accumulate_spectrum(
        te_coordinates, site_rows, projections, endpoints
    )
    te_raw, te_normalized = normalized_spectrum(te_counts)

    replicate_raw = np.empty((len(snp_coordinates), PROJECTION_SIZE - 1), dtype=np.float64)
    replicate_normalized = np.empty_like(replicate_raw)
    residuals = np.empty_like(replicate_raw)
    positive = np.empty_like(replicate_raw)
    phi = np.empty(len(snp_coordinates), dtype=np.float64)
    rows: list[dict[str, object]] = []

    for replicate, coordinates in enumerate(snp_coordinates):
        counts_vector, endpoint, eligible = accumulate_spectrum(
            coordinates, site_rows, projections, endpoints
        )
        raw, normalized = normalized_spectrum(counts_vector)
        result = phi_sfs(te_normalized, normalized)
        replicate_raw[replicate] = raw
        replicate_normalized[replicate] = normalized
        residuals[replicate] = result.residual
        positive[replicate] = result.positive
        phi[replicate] = result.value
        retained_mass = float(raw.sum())
        rows.append({
            "replicate": replicate,
            "chain_index": int(chains[replicate]),
            "sample_index": int(samples[replicate]),
            "input_sites": len(coordinates),
            "eligible_sites": eligible,
            "dropped_n_lt_20": len(coordinates) - eligible,
            "retained_mass": retained_mass,
            "endpoint_mass": endpoint,
            "retained_fraction": retained_mass / eligible,
            "endpoint_fraction": endpoint / eligible,
            "phi_sfs": result.value,
            "overlap": 1.0 - result.value,
            "reverse_positive": result.reverse_positive,
            "half_l1": result.half_l1,
            "identity_max_abs_error": max(
                abs(result.value - result.reverse_positive),
                abs(result.value - result.half_l1),
            ),
        })

    output = args.output
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        arrays = {
            "bins.npy": RETAINED_BINS,
            "te_raw_sfs.npy": te_raw,
            "te_normalized_sfs.npy": te_normalized,
            "snp_raw_sfs.npy": replicate_raw,
            "snp_normalized_sfs.npy": replicate_normalized,
            "residual_te_minus_snp.npy": residuals,
            "positive_te_residual.npy": positive,
            "phi_sfs.npy": phi,
            "chain_index.npy": chains,
            "sample_index.npy": samples,
        }
        for name, values in arrays.items():
            np.save(staging / name, values, allow_pickle=False)
        with (staging / "replicates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (staging / "bins.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "replicate", "chain_index", "sample_index", "derived_count_bin",
                "te_raw", "te_normalized", "snp_raw", "snp_normalized",
                "te_minus_snp", "positive_te_residual",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for replicate in range(len(rows)):
                for offset, bin_index in enumerate(RETAINED_BINS):
                    writer.writerow({
                        "replicate": replicate,
                        "chain_index": int(chains[replicate]),
                        "sample_index": int(samples[replicate]),
                        "derived_count_bin": int(bin_index),
                        "te_raw": float(te_raw[offset]),
                        "te_normalized": float(te_normalized[offset]),
                        "snp_raw": float(replicate_raw[replicate, offset]),
                        "snp_normalized": float(replicate_normalized[replicate, offset]),
                        "te_minus_snp": float(residuals[replicate, offset]),
                        "positive_te_residual": float(positive[replicate, offset]),
                    })
        te_retained_mass = float(te_raw.sum())
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "software": software_provenance(),
            "creation_command": " ".join(sys.argv),
            "creation_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "numpy_version": np.__version__,
            "projection_size": PROJECTION_SIZE,
            "retained_bins": [1, 19],
            "site_projection_renormalized": False,
            "final_spectra_normalized": True,
            "phi_definition": "sum(max(te_normalized_sfs - snp_normalized_sfs, 0))",
            "phi_equivalent_forms": [
                "sum(max(snp_normalized_sfs - te_normalized_sfs, 0))",
                "0.5 * sum(abs(te_normalized_sfs - snp_normalized_sfs))",
                "1 - sum(minimum(te_normalized_sfs, snp_normalized_sfs))",
            ],
            "phi_interpretation": (
                "total variation distance between the projected normalized spectra; "
                "symmetric and bounded in [0, 1]"
            ),
            "target": str(args.target.resolve()),
            "matches": str(args.matches.resolve()),
            "target_digest": target_digest,
            "vcf": str(args.vcf.resolve()),
            "vcf_sha256": vcf_sha256,
            "ancestral_mode": args.ancestral_mode,
            "ancestral_info": args.ancestral_info if args.ancestral_mode == "info" else None,
            "ancestral_case_policy": (
                "case-sensitive; a lowercase ancestral allele is rejected rather than folded"
            ),
            "heterozygous_policy": args.heterozygous,
            "biallelic_policy": (
                "records are assumed biallelic; a comma in ALT is rejected"
            ),
            "filter_policy": (
                "the VCF FILTER column is ignored; every record at a requested "
                "coordinate is used"
            ),
            "replicates": len(rows),
            "distinct_projections": int(projections.shape[0]),
            "target_input_sites": len(te_coordinates),
            "target_eligible_sites": te_eligible,
            "target_dropped_n_lt_20": len(te_coordinates) - te_eligible,
            "target_retained_mass": te_retained_mass,
            "target_endpoint_mass": te_endpoint,
            "target_retained_fraction": te_retained_mass / te_eligible,
            "target_endpoint_fraction": te_endpoint / te_eligible,
            "target_source_store_content_sha256": target_meta.get("source_store_content_sha256"),
            "matches_source_store_content_sha256": match_meta.get("source_store_content_sha256"),
        }
        with (staging / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--vcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ancestral-mode", choices=("ref", "info"), default="ref",
        help="take REF as ancestral (default) or read an INFO annotation",
    )
    parser.add_argument("--ancestral-info", default="AA")
    parser.add_argument("--heterozygous", choices=("error", "missing"), default="error")
    parser.add_argument(
        "--quiet", action="store_true",
        help="suppress periodic VCF scan progress",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calculate(args)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
