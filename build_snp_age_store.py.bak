#!/usr/bin/env python3
"""Build a reusable NumPy SNP age-distribution store from posterior ARG draws."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import tskit

from snp_age_distribution import AgeInterval, discretize_intervals
from snp_age_dataset import QUANTIZATION_SCALE, SCHEMA_VERSION, validate_store


def _load(path: Path) -> tskit.TreeSequence:
    return tskit.load(str(path))


def _site_map(ts: tskit.TreeSequence, source: Path) -> dict[float, tskit.Site]:
    result: dict[float, tskit.Site] = {}
    for site in ts.sites():
        position = float(site.position)
        if not math.isfinite(position):
            raise ValueError(f"non-finite site position in {source}")
        if position in result:
            raise ValueError(f"duplicate site position {position:g} in {source}")
        result[position] = site
    return result


def discover_positions(tree_files: Sequence[Path]) -> np.ndarray:
    positions: set[float] = set()
    for path in tree_files:
        positions.update(_site_map(_load(path), path))
    return np.asarray(sorted(positions), dtype=np.float64)


def determine_age_grid(tree_files: Sequence[Path], bin_width: float) -> np.ndarray:
    if not math.isfinite(bin_width) or bin_width <= 0 or not float(bin_width).is_integer():
        raise ValueError("bin width must be a positive integer number of generations")
    maximum = 0.0
    for path in tree_files:
        ts = _load(path)
        for site in ts.sites():
            tree = ts.at(site.position)
            for mutation in site.mutations:
                parent = tree.parent(mutation.node)
                if parent != tskit.NULL:
                    maximum = max(maximum, float(ts.node(parent).time))
    last = int(math.floor(maximum / bin_width + 0.5))
    return np.arange(last + 1, dtype=np.uint64) * np.uint64(int(bin_width))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantize_cdf(pdf: np.ndarray) -> np.ndarray:
    if not np.any(pdf):
        return np.zeros(pdf.size, dtype=np.uint16)
    pdf = pdf / np.sum(pdf, dtype=np.float64)
    cdf = np.maximum.accumulate(np.cumsum(pdf, dtype=np.float64))
    cdf[-1] = 1.0
    quantized = np.rint(cdf * QUANTIZATION_SCALE).astype(np.uint16)
    quantized = np.maximum.accumulate(quantized)
    quantized[-1] = QUANTIZATION_SCALE
    return quantized


def _interval_pdf(intervals: list[AgeInterval], age_index: dict[float, int], n_bins: int, bin_width: float) -> np.ndarray:
    pdf = np.zeros(n_bins, dtype=np.float64)
    for centre, probability in discretize_intervals(intervals, bin_width).items():
        pdf[age_index[centre]] += probability
    return pdf


def build_store(
    tree_files: Sequence[Path], output_dir: Path, *, bin_width: float = 1000,
    block_snps: int = 100_000, missing: str = "skip", root: str = "skip",
    mutation_weighting: str = "interval", omit_transpose: bool = False,
    checksums: bool = False,
) -> None:
    paths = [Path(path).resolve() for path in tree_files]
    output_dir = Path(output_dir)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all tree-sequence inputs must exist")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if missing not in {"skip", "error"} or root not in {"skip", "error"}:
        raise ValueError("missing and root policies must be 'skip' or 'error'")
    if mutation_weighting not in {"interval", "draw"}:
        raise ValueError("mutation weighting must be 'interval' or 'draw'")
    if block_snps <= 0:
        raise ValueError("block-snps must be positive")
    positions = discover_positions(paths)
    age_bins = determine_age_grid(paths, bin_width)
    if positions.size == 0:
        raise ValueError("input tree sequences contain no sites")
    lengths = {float(_load(path).sequence_length) for path in paths}
    if len(lengths) != 1:
        raise ValueError("all tree sequences must have the same sequence length")
    age_index = {float(value): i for i, value in enumerate(age_bins)}
    temp = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        np.save(temp / "positions.npy", positions)
        np.save(temp / "age_bins.npy", age_bins)
        n, b = positions.size, age_bins.size
        cdf = np.lib.format.open_memmap(temp / "cdf_by_snp.npy", mode="w+", dtype=np.uint16, shape=(n, b))
        valid = np.lib.format.open_memmap(temp / "valid.npy", mode="w+", dtype=np.bool_, shape=(n,))
        counts = {name: np.lib.format.open_memmap(temp / f"{name}.npy", mode="w+", dtype=np.uint32, shape=(n,)) for name in ("present_draw_count", "usable_interval_count", "skipped_root_count", "missing_draw_count")}
        extraction_totals = {name: 0 for name in (
            "present_draw_count", "usable_interval_count",
            "skipped_root_count", "missing_draw_count",
        )}
        for start in range(0, n, block_snps):
            stop = min(start + block_snps, n)
            width = stop - start
            pdf_block = np.zeros((width, b), dtype=np.float64)
            present_block = np.zeros(width, dtype=np.uint32)
            interval_block = np.zeros(width, dtype=np.uint32)
            root_block = np.zeros(width, dtype=np.uint32)
            missing_block = np.zeros(width, dtype=np.uint32)
            query = positions[start:stop]
            # Keep memory bounded by one tree sequence plus one SNP block. This
            # deliberately favors predictable HPC memory use over retaining
            # every posterior ARG simultaneously.
            for path in paths:
                ts = _load(path)
                site_positions = np.asarray(ts.tables.sites.position)
                found_indices = np.searchsorted(site_positions, query)
                found = found_indices < site_positions.size
                if np.any(found):
                    found[found] &= site_positions[found_indices[found]] == query[found]
                if missing == "error" and not np.all(found):
                    position = float(query[np.flatnonzero(~found)[0]])
                    raise ValueError(f"position {position:g} is absent from {path}")
                missing_block += (~found).astype(np.uint32)
                present_block += found.astype(np.uint32)
                for local in np.flatnonzero(found):
                    position = float(query[local])
                    site = ts.site(int(found_indices[local]))
                    if site is None:
                        continue
                    tree = ts.at(position)
                    intervals = []
                    for mutation in site.mutations:
                        parent = tree.parent(mutation.node)
                        if parent == tskit.NULL:
                            root_block[local] += 1
                            if root == "error":
                                raise ValueError(f"mutation {mutation.id} at {position:g} in {path} is above a root node")
                            continue
                        below, above = float(ts.node(mutation.node).time), float(ts.node(parent).time)
                        if above < below:
                            raise ValueError(f"invalid age interval at {position:g} in {path}")
                        intervals.append(AgeInterval(position, below, above, str(path), mutation.id))
                    interval_block[local] += len(intervals)
                    if intervals:
                        draw_pdf = _interval_pdf(intervals, age_index, b, bin_width)
                        # ``discretize_intervals`` returns an equal-interval
                        # mixture. Restore its interval count for global
                        # interval weighting; leave each draw at unit mass for
                        # draw weighting.
                        if mutation_weighting == "interval":
                            draw_pdf *= len(intervals)
                        pdf_block[local] += draw_pdf
            for local, row in enumerate(range(start, stop)):
                cdf[row] = _quantize_cdf(pdf_block[local])
                valid[row] = bool(interval_block[local])
            counts["present_draw_count"][start:stop] = present_block
            counts["usable_interval_count"][start:stop] = interval_block
            counts["skipped_root_count"][start:stop] = root_block
            counts["missing_draw_count"][start:stop] = missing_block
            extraction_totals["present_draw_count"] += int(present_block.sum(dtype=np.uint64))
            extraction_totals["usable_interval_count"] += int(interval_block.sum(dtype=np.uint64))
            extraction_totals["skipped_root_count"] += int(root_block.sum(dtype=np.uint64))
            extraction_totals["missing_draw_count"] += int(missing_block.sum(dtype=np.uint64))
        cdf.flush(); valid.flush()
        for array in counts.values(): array.flush()
        if not omit_transpose:
            by_age = np.lib.format.open_memmap(temp / "cdf_by_age.npy", mode="w+", dtype=np.uint16, shape=(b, n))
            for start in range(0, n, block_snps):
                stop = min(start + block_snps, n)
                by_age[:, start:stop] = cdf[start:stop].T
            by_age.flush()
        del cdf, valid, counts
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "n_snps": int(n), "n_age_bins": int(b), "n_posterior_draws": len(paths),
            "sequence_length": lengths.pop(), "bin_width": int(bin_width),
            "age_bin_convention": "centres; cells [centre-width/2, centre+width/2); nearest ties upward",
            "position_matching": "exact float64 equality", "mutation_weighting": mutation_weighting,
            "missing_policy": missing, "root_policy": root, "quantization_scale": QUANTIZATION_SCALE,
            "quantization_scheme": "round(65535*cdf), monotone, valid terminal=65535",
            "has_cdf_by_age": not omit_transpose,
            "creation_command": " ".join(sys.argv),
            "extraction_totals": extraction_totals,
            "inputs": [{"path": str(path), **({"sha256": _sha256(path)} if checksums else {})} for path in paths],
            "arrays": {
                "positions": {"dtype": "float64", "shape": [int(n)]},
                "age_bins": {"dtype": "uint64", "shape": [int(b)]},
                "cdf_by_snp": {"dtype": "uint16", "shape": [int(n), int(b)]},
                "cdf_by_age": None if omit_transpose else {"dtype": "uint16", "shape": [int(b), int(n)]},
            },
        }
        (temp / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_store(temp, deep=True)
        os.replace(temp, output_dir)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _expand(patterns: Sequence[str]) -> list[Path]:
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        result.extend(Path(value) for value in (matches or [pattern]))
    return list(dict.fromkeys(result))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", nargs="+", help="tree-sequence files or glob patterns")
    parser.add_argument("--numpy-store", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=1000)
    parser.add_argument("--block-snps", type=int, default=100_000)
    parser.add_argument("--missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--root", choices=("skip", "error"), default="skip")
    parser.add_argument("--mutation-weighting", choices=("interval", "draw"), default="interval")
    parser.add_argument("--omit-transpose", action="store_true")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args(argv)
    try:
        build_store(_expand(args.trees), args.numpy_store, bin_width=args.bin_width,
                    block_snps=args.block_snps, missing=args.missing, root=args.root,
                    mutation_weighting=args.mutation_weighting,
                    omit_transpose=args.omit_transpose, checksums=args.checksums)
    except (OSError, ValueError, tskit.FileFormatError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
