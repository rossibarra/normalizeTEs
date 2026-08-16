#!/usr/bin/env python3
"""Calculate Phi-SFS for a TE target and its matched SNP control sets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "phi-sfs-v1"
PROJECTION_SIZE = 20
RETAINED_BINS = np.arange(1, PROJECTION_SIZE, dtype=np.int64)


@dataclass(frozen=True)
class SiteCount:
    derived: int
    callable: int


def hypergeometric_projection(k: int, n: int, m: int = PROJECTION_SIZE) -> np.ndarray:
    """Return the expected derived-count distribution after projection to m."""
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


def normalized_spectrum(projections: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate site projections and normalize the final bins 1..19."""
    raw = np.zeros(PROJECTION_SIZE - 1, dtype=np.float64)
    for projection in projections:
        values = np.asarray(projection, dtype=np.float64)
        if values.shape != (PROJECTION_SIZE + 1,):
            raise ValueError("site projection must contain bins 0 through 20")
        raw += values[1:PROJECTION_SIZE]
    retained = float(raw.sum())
    if not math.isfinite(retained) or retained <= 0:
        raise ValueError("spectrum has zero retained mass in bins 1 through 19")
    return raw, raw / retained


def phi_sfs(te: np.ndarray, snp: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Phi-SFS, signed TE-minus-SNP residuals, and positive residuals."""
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
    return value, residual, positive


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _parse_info(info: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    if info == ".":
        return result
    for item in info.split(";"):
        key, separator, value = item.partition("=")
        result[key] = value if separator else None
    return result


def _individual_allele(gt: str, heterozygous: str, coordinate: tuple[str, int]) -> int | None:
    alleles = re.split(r"[/|]", gt)
    if not alleles or any(allele == "." for allele in alleles):
        return None
    try:
        values = [int(allele) for allele in alleles]
    except ValueError as error:
        raise ValueError(f"invalid GT {gt!r} at {coordinate[0]}:{coordinate[1]}") from error
    if any(value not in (0, 1) for value in values):
        raise ValueError(f"non-biallelic GT {gt!r} at {coordinate[0]}:{coordinate[1]}")
    if len(set(values)) > 1:
        if heterozygous == "missing":
            return None
        raise ValueError(f"heterozygous GT {gt!r} at inbred site {coordinate[0]}:{coordinate[1]}")
    return values[0]


def read_site_counts(
    vcf: Path,
    coordinates: set[tuple[str, int]],
    *,
    ancestral_mode: str,
    ancestral_info: str,
    heterozygous: str,
) -> dict[tuple[str, int], SiteCount]:
    """Read callable and polarized derived counts for requested VCF coordinates."""
    found: dict[tuple[str, int], SiteCount] = {}
    with _open_text(vcf) as handle:
        for line_number, raw in enumerate(handle, 1):
            if raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"{vcf}:{line_number}: expected VCF samples and FORMAT")
            chrom = fields[0]
            try:
                position = int(fields[1])
            except ValueError as error:
                raise ValueError(f"{vcf}:{line_number}: invalid POS") from error
            coordinate = (chrom, position)
            if coordinate not in coordinates:
                continue
            if coordinate in found:
                raise ValueError(f"duplicate VCF record at {chrom}:{position}")
            ref, alt = fields[3], fields[4]
            if "," in alt:
                raise ValueError(f"multiallelic VCF record at {chrom}:{position}")
            if ancestral_mode == "ref":
                ancestral = ref
            else:
                value = _parse_info(fields[7]).get(ancestral_info)
                if value is None or "," in value:
                    raise ValueError(
                        f"missing or ambiguous INFO/{ancestral_info} at {chrom}:{position}"
                    )
                ancestral = value
            if ancestral not in (ref, alt):
                raise ValueError(
                    f"ancestral allele {ancestral!r} is neither REF nor ALT at {chrom}:{position}"
                )
            formats = fields[8].split(":")
            if "GT" not in formats:
                raise ValueError(f"VCF record lacks GT at {chrom}:{position}")
            gt_index = formats.index("GT")
            called: list[int] = []
            for sample in fields[9:]:
                parts = sample.split(":")
                gt = parts[gt_index] if gt_index < len(parts) else "."
                allele = _individual_allele(gt, heterozygous, coordinate)
                if allele is not None:
                    called.append(allele)
            alt_count = int(sum(called))
            n = len(called)
            k = alt_count if ancestral == ref else n - alt_count
            found[coordinate] = SiteCount(derived=k, callable=n)
    return found


def _load_coordinates(target: Path, matches: Path):
    te_chromosomes = np.load(target / "te_chromosomes.npy", allow_pickle=False).astype(str)
    te_positions = np.load(target / "te_positions.npy", allow_pickle=False).astype(np.int64)
    positions = np.load(matches / "positions.npy", allow_pickle=False).astype(np.int64)
    codes = np.load(matches / "chromosome_codes.npy", allow_pickle=False).astype(np.int64)
    labels = np.load(matches / "chromosome_labels.npy", allow_pickle=False).astype(str)
    chains = np.load(matches / "chain_index.npy", allow_pickle=False).astype(np.int64)
    samples = np.load(matches / "sample_index.npy", allow_pickle=False).astype(np.int64)
    if te_chromosomes.shape != te_positions.shape or te_chromosomes.ndim != 1:
        raise ValueError("target chromosome and position arrays are not aligned 1-D arrays")
    if positions.shape != codes.shape or positions.ndim != 2:
        raise ValueError("matched chromosome codes and positions are not aligned 2-D arrays")
    if chains.shape != (positions.shape[0],) or samples.shape != (positions.shape[0],):
        raise ValueError("chain/sample arrays do not align with matched sets")
    if np.any(codes < 0) or np.any(codes >= labels.size):
        raise ValueError("matched chromosome code is out of range")
    match_chromosomes = labels[codes]
    return te_chromosomes, te_positions, match_chromosomes, positions, chains, samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_provenance(target: Path, matches: Path) -> tuple[dict, dict]:
    target_meta, match_meta = _json(target / "metadata.json"), _json(matches / "metadata.json")
    for key in ("source_store_content_sha256", "source_catalog_sha256"):
        left, right = target_meta.get(key), match_meta.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"target and matched-control {key} values differ")
    return target_meta, match_meta


def calculate(args: argparse.Namespace) -> None:
    target_meta, match_meta = _validate_provenance(args.target, args.matches)
    te_chrom, te_pos, snp_chrom, snp_pos, chains, samples = _load_coordinates(
        args.target, args.matches
    )
    te_coordinates = list(zip(te_chrom.tolist(), te_pos.tolist()))
    snp_coordinates = [
        list(zip(snp_chrom[row].tolist(), snp_pos[row].tolist()))
        for row in range(snp_pos.shape[0])
    ]
    requested = set(te_coordinates)
    for row in snp_coordinates:
        requested.update(row)
    counts = read_site_counts(
        args.vcf,
        requested,
        ancestral_mode=args.ancestral_mode,
        ancestral_info=args.ancestral_info,
        heterozygous=args.heterozygous,
    )
    missing = sorted(requested.difference(counts))
    if missing:
        preview = ", ".join(f"{chrom}:{pos}" for chrom, pos in missing[:10])
        raise ValueError(f"{len(missing)} requested sites are absent from the VCF: {preview}")

    projection_cache: dict[SiteCount, np.ndarray] = {}
    def projection(item: SiteCount) -> np.ndarray:
        if item not in projection_cache:
            projection_cache[item] = hypergeometric_projection(item.derived, item.callable)
        return projection_cache[item]

    def eligible(coordinates: list[tuple[str, int]]):
        site_counts = [counts[coordinate] for coordinate in coordinates]
        retained = [item for item in site_counts if item.callable >= PROJECTION_SIZE]
        projections = [projection(item) for item in retained]
        return site_counts, retained, projections

    te_all, te_retained, te_projections = eligible(te_coordinates)
    te_raw, te_normalized = normalized_spectrum(te_projections)
    if not snp_coordinates:
        raise ValueError("matched-control bundle contains no sets")
    replicate_raw = np.empty((len(snp_coordinates), PROJECTION_SIZE - 1), dtype=np.float64)
    replicate_normalized = np.empty_like(replicate_raw)
    residuals = np.empty_like(replicate_raw)
    positive = np.empty_like(replicate_raw)
    phi = np.empty(len(snp_coordinates), dtype=np.float64)
    rows: list[dict[str, object]] = []

    for replicate, coordinates in enumerate(snp_coordinates):
        all_counts, retained, projections = eligible(coordinates)
        raw, normalized = normalized_spectrum(projections)
        value, residual, positive_residual = phi_sfs(te_normalized, normalized)
        reverse_positive = float(np.maximum(-residual, 0.0).sum())
        half_l1 = float(np.abs(residual).sum() / 2.0)
        replicate_raw[replicate] = raw
        replicate_normalized[replicate] = normalized
        residuals[replicate] = residual
        positive[replicate] = positive_residual
        phi[replicate] = value
        endpoint = float(sum(item[0] + item[-1] for item in projections))
        rows.append({
            "replicate": replicate,
            "chain_index": int(chains[replicate]),
            "sample_index": int(samples[replicate]),
            "input_sites": len(all_counts),
            "eligible_sites": len(retained),
            "dropped_n_lt_20": len(all_counts) - len(retained),
            "retained_mass": float(raw.sum()),
            "endpoint_mass": endpoint,
            "phi_sfs": value,
            "overlap": 1.0 - value,
            "reverse_positive": reverse_positive,
            "half_l1": half_l1,
            "identity_max_abs_error": max(
                abs(value - reverse_positive), abs(value - half_l1)
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
        te_endpoint = float(sum(item[0] + item[-1] for item in te_projections))
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "projection_size": PROJECTION_SIZE,
            "retained_bins": [1, 19],
            "site_projection_renormalized": False,
            "final_spectra_normalized": True,
            "phi_definition": "sum(max(te_normalized_sfs - snp_normalized_sfs, 0))",
            "target": str(args.target.resolve()),
            "matches": str(args.matches.resolve()),
            "vcf": str(args.vcf.resolve()),
            "vcf_sha256": _sha256(args.vcf),
            "ancestral_mode": args.ancestral_mode,
            "ancestral_info": args.ancestral_info if args.ancestral_mode == "info" else None,
            "heterozygous_policy": args.heterozygous,
            "replicates": len(rows),
            "target_input_sites": len(te_all),
            "target_eligible_sites": len(te_retained),
            "target_dropped_n_lt_20": len(te_all) - len(te_retained),
            "target_retained_mass": float(te_raw.sum()),
            "target_endpoint_mass": te_endpoint,
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    calculate(args)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
