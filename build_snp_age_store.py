#!/usr/bin/env python3
"""Build a reusable NumPy SNP age-distribution store from posterior ARG draws."""

from __future__ import annotations

import argparse
import glob
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
import tszip

from snp_age_distribution import AgeInterval, discretize_intervals
from snp_age_dataset import QUANTIZATION_SCALE, SCHEMA_VERSION, validate_store


def _load(path: Path) -> tskit.TreeSequence:
    """Load ordinary tskit files and tszip-compressed ``.tsz`` files."""
    return tszip.load(str(path))


def _warn_metadata_conflict(
    ts: tskit.TreeSequence, path: Path,
    supplied: list[dict[str, int | str]], source: str | Path,
) -> None:
    """Report when an offsets file overrides a usable ARG chrom_offsets table."""
    metadata = ts.metadata
    if not isinstance(metadata, dict) or "chrom_offsets" not in metadata:
        return
    try:
        embedded = _chromosome_table(metadata["chrom_offsets"], path, ts.sequence_length)
    except ValueError:
        embedded = None
    if embedded != supplied:
        print(
            f"warning: {path} carries chrom_offsets metadata that disagrees with "
            f"{source}; the supplied offsets file takes precedence",
            file=sys.stderr,
        )


def _chromosome_table(
    raw: object, source: str | Path, sequence_length: float, *, disjoint: bool = False
) -> list[dict[str, int | str]]:
    """Validate a chromosome offset table from ARG metadata or an offsets file."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} has invalid chrom_offsets metadata")
    result: list[dict[str, int | str]] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{source} has invalid chrom_offsets entry")
        try:
            chrom = str(entry["chrom"])
            offset = int(entry["offset"])
            length = int(entry["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{source} has invalid chrom_offsets entry: {entry!r}") from error
        if chrom in seen or offset < 0 or length <= 0:
            raise ValueError(f"{source} has invalid chromosome entry: {entry!r}")
        if float(offset) != float(entry["offset"]) or float(length) != float(entry["length"]):
            raise ValueError(f"{source} chromosome offsets and lengths must be integers")
        if offset + length > sequence_length:
            raise ValueError(f"{source} chromosome {chrom} extends beyond sequence_length")
        seen.add(chrom)
        result.append({"chrom": chrom, "offset": offset, "length": length})
    if any(result[i]["offset"] >= result[i + 1]["offset"] for i in range(len(result) - 1)):
        raise ValueError(f"{source} chrom_offsets must be ordered by increasing offset")
    # User-supplied tables get the stronger check: a typo that lets one
    # chromosome run into the next would silently mislabel native coordinates.
    if disjoint and any(
        int(result[i]["offset"]) + int(result[i]["length"]) > int(result[i + 1]["offset"])
        for i in range(len(result) - 1)
    ):
        raise ValueError(f"{source} chromosome intervals overlap")
    return result


def _chromosomes(ts: tskit.TreeSequence, source: Path) -> list[dict[str, int | str]]:
    """Return and validate the ARG's chromosome offset table."""
    metadata = ts.metadata
    if not isinstance(metadata, dict) or "chrom_offsets" not in metadata:
        raise ValueError(
            f"{source} lacks top-level chrom_offsets metadata required for "
            "native chromosome coordinates; supply --chrom-offsets instead"
        )
    return _chromosome_table(metadata["chrom_offsets"], source, ts.sequence_length)


def load_chrom_offsets(path: str | Path) -> list[dict[str, int | str]]:
    """Read a chromosome offsets file into an unvalidated chrom_offsets table.

    Accepts two whitespace-separated layouts: ``chrom length``, whose offsets
    are accumulated in file order, and explicit ``chrom offset length``.
    """
    rows: list[tuple[str, list[int]]] = []
    widths: set[int] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) not in (2, 3):
                raise ValueError(
                    f"{path}:{line_number}: expected 'chrom length' or "
                    f"'chrom offset length', found {len(fields)} columns"
                )
            widths.add(len(fields))
            if len(widths) > 1:
                raise ValueError(f"{path}: every row must use the same number of columns")
            try:
                values = [int(field) for field in fields[1:]]
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: offsets and lengths must be integers"
                ) from error
            rows.append((fields[0], values))
    if not rows:
        raise ValueError(f"chromosome offsets file is empty: {path}")
    table: list[dict[str, int | str]] = []
    cumulative = 0
    for chrom, values in rows:
        offset, length = (cumulative, values[0]) if len(values) == 1 else values
        table.append({"chrom": chrom, "offset": offset, "length": length})
        cumulative = offset + length
    return table


def inspect_inputs(
    tree_files: Sequence[Path], bin_width: float,
    chrom_offsets: Sequence[dict[str, int | str]] | None = None,
    chrom_offsets_source: str | Path = "supplied chromosome offsets",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | str]], float]:
    """Inspect every draw once to establish shared positions, grid, and metadata."""
    if not math.isfinite(bin_width) or bin_width <= 0 or not float(bin_width).is_integer():
        raise ValueError("bin width must be a positive integer number of generations")
    positions: set[float] = set()
    maximum = 0.0
    chromosomes = None
    sequence_length = None
    for path in tree_files:
        ts = _load(path)
        if chrom_offsets is None:
            current_chromosomes = _chromosomes(ts, path)
        else:
            # Revalidated per draw so the bound against sequence_length holds
            # for every input, not just the first.
            current_chromosomes = _chromosome_table(
                list(chrom_offsets), chrom_offsets_source, ts.sequence_length, disjoint=True
            )
            if chromosomes is None:
                _warn_metadata_conflict(ts, path, current_chromosomes, chrom_offsets_source)
        if chromosomes is None:
            chromosomes = current_chromosomes
            sequence_length = float(ts.sequence_length)
        elif current_chromosomes != chromosomes:
            raise ValueError("all tree sequences must have identical chrom_offsets metadata")
        elif float(ts.sequence_length) != sequence_length:
            raise ValueError("all tree sequences must have the same sequence length")
        site_positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
        if np.any(~np.isfinite(site_positions)) or np.any(site_positions != np.floor(site_positions)):
            raise ValueError(f"{path} contains a non-integer or non-finite site position")
        positions.update(site_positions.tolist())
        # Every finite mutation-parent node occurs as an edge parent. This
        # vectorized bound excludes isolated ancient nodes without requiring a
        # second full marginal-tree traversal.
        edge_parents = np.asarray(ts.tables.edges.parent, dtype=np.int64)
        if edge_parents.size:
            maximum = max(
                maximum,
                float(np.max(np.asarray(ts.tables.nodes.time)[edge_parents])),
            )
    # Downstream CDF integration requires at least two grid points. This also
    # keeps a valid zero-age-only fixture from producing a degenerate store.
    last = max(1, int(math.floor(maximum / bin_width + 0.5)))
    age_bins = np.arange(last + 1, dtype=np.uint64) * np.uint64(int(bin_width))
    return (
        np.asarray(sorted(positions), dtype=np.float64),
        age_bins,
        chromosomes or [],
        float(sequence_length or 0),
    )


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
    omit_transpose: bool = False, min_usable_fraction: float = 0.1,
    scratch_dir: Path | None = None, chrom_offsets: Path | None = None,
) -> None:
    paths = [Path(path).resolve() for path in tree_files]
    output_dir = Path(output_dir)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all tree-sequence inputs must exist")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if missing not in {"skip", "error"} or root not in {"skip", "error"}:
        raise ValueError("missing and root policies must be 'skip' or 'error'")
    if block_snps <= 0:
        raise ValueError("block-snps must be positive")
    if not 0 <= min_usable_fraction <= 1:
        raise ValueError("min_usable_fraction must lie in [0, 1]")
    supplied_offsets = None
    if chrom_offsets is not None:
        chrom_offsets = Path(chrom_offsets)
        supplied_offsets = load_chrom_offsets(chrom_offsets)
    positions, age_bins, chromosomes, sequence_length = inspect_inputs(
        paths, bin_width, supplied_offsets,
        chrom_offsets if chrom_offsets is not None else "supplied chromosome offsets",
    )
    if positions.size == 0:
        raise ValueError("input tree sequences contain no sites")
    required_usable_draws = int(math.ceil(min_usable_fraction * len(paths)))
    age_index = {float(value): i for i, value in enumerate(age_bins)}
    scratch_parent = output_dir.parent if scratch_dir is None else Path(scratch_dir)
    if not scratch_parent.is_dir():
        raise NotADirectoryError(f"scratch directory does not exist: {scratch_parent}")
    temp: Path | None = None
    scratch: Path | None = None
    try:
        temp = Path(tempfile.mkdtemp(
            prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
        scratch = Path(tempfile.mkdtemp(
            prefix=f"{output_dir.name}.accumulator.", dir=scratch_parent))
        np.save(temp / "positions.npy", positions)
        np.save(temp / "age_bins.npy", age_bins)
        n, b = positions.size, age_bins.size
        cdf = np.lib.format.open_memmap(temp / "cdf_by_snp.npy", mode="w+", dtype=np.uint16, shape=(n, b))
        eligible = np.lib.format.open_memmap(temp / "eligible.npy", mode="w+", dtype=np.bool_, shape=(n,))
        counts = {name: np.lib.format.open_memmap(temp / f"{name}.npy", mode="w+", dtype=np.uint32, shape=(n,)) for name in ("present_draw_count", "usable_draw_count", "usable_interval_count", "skipped_root_count", "missing_draw_count", "multiple_mutation_draw_count")}
        for array in counts.values():
            array[:] = 0
        counts["missing_draw_count"][:] = len(paths)
        # The accumulator is disk-backed: extraction keeps only one posterior
        # ARG resident while visiting each tree and site exactly once.
        pdf_accumulator = np.lib.format.open_memmap(
            scratch / "pdf_accumulator.npy", mode="w+", dtype=np.float32, shape=(n, b)
        )
        pdf_accumulator[:] = 0
        extraction_totals = {name: 0 for name in (
            "present_draw_count", "usable_draw_count", "usable_interval_count",
            "skipped_root_count", "missing_draw_count",
        )}
        for path in paths:
            ts = _load(path)
            site_positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
            site_rows = np.searchsorted(positions, site_positions)
            if np.any(site_rows >= n) or np.any(positions[site_rows] != site_positions):
                raise ValueError(f"{path} contains a site absent from the shared position index")
            seen_rows = np.zeros(n, dtype=np.bool_) if missing == "error" else None
            for tree in ts.trees():
                for site in tree.sites():
                    position = float(site.position)
                    row = int(site_rows[site.id])
                    if seen_rows is not None:
                        seen_rows[row] = True
                    counts["present_draw_count"][row] += 1
                    # tskit guarantees unique site positions, so each row is
                    # decremented at most once per posterior draw.
                    counts["missing_draw_count"][row] -= 1
                    if len(site.mutations) > 1:
                        counts["multiple_mutation_draw_count"][row] += 1
                    intervals = []
                    for mutation in site.mutations:
                        parent = tree.parent(mutation.node)
                        if parent == tskit.NULL:
                            counts["skipped_root_count"][row] += 1
                            if root == "error":
                                raise ValueError(f"mutation {mutation.id} at {position:g} in {path} is above a root node")
                            continue
                        below, above = float(ts.node(mutation.node).time), float(ts.node(parent).time)
                        if above < below:
                            raise ValueError(f"invalid age interval at {position:g} in {path}")
                        intervals.append(AgeInterval(position, below, above, str(path), mutation.id))
                    counts["usable_interval_count"][row] += len(intervals)
                    if intervals:
                        counts["usable_draw_count"][row] += 1
                        draw_pdf = _interval_pdf(intervals, age_index, b, bin_width)
                        # ``discretize_intervals`` returns an equal-interval
                        # mixture; restore interval count before pooling draws.
                        draw_pdf *= len(intervals)
                        pdf_accumulator[row] += draw_pdf.astype(np.float32)
            if seen_rows is not None and not np.all(seen_rows):
                position = float(positions[np.flatnonzero(~seen_rows)[0]])
                raise ValueError(f"position {position:g} is absent from {path}")

        for start in range(0, n, block_snps):
            stop = min(start + block_snps, n)
            for row in range(start, stop):
                cdf[row] = _quantize_cdf(np.asarray(pdf_accumulator[row], dtype=np.float64))
            interval_block = np.asarray(counts["usable_interval_count"][start:stop])
            usable_draw_block = np.asarray(counts["usable_draw_count"][start:stop])
            eligible[start:stop] = (usable_draw_block > 0) & (
                usable_draw_block >= required_usable_draws
            ) & (np.asarray(counts["multiple_mutation_draw_count"][start:stop]) == 0)
        for name, array in counts.items():
            extraction_totals[name] = int(array.sum(dtype=np.uint64))
        del pdf_accumulator
        shutil.rmtree(scratch)
        cdf.flush(); eligible.flush()
        for array in counts.values(): array.flush()
        if not omit_transpose:
            by_age = np.lib.format.open_memmap(temp / "cdf_by_age.npy", mode="w+", dtype=np.uint16, shape=(b, n))
            for start in range(0, n, block_snps):
                stop = min(start + block_snps, n)
                by_age[:, start:stop] = cdf[start:stop].T
            by_age.flush()
        del cdf, eligible, counts
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "n_snps": int(n), "n_age_bins": int(b), "n_posterior_draws": len(paths),
            "sequence_length": sequence_length, "bin_width": int(bin_width),
            "age_bin_convention": "centres; cells [centre-width/2, centre+width/2); nearest ties upward",
            "position_matching": "exact float64 equality",
            "position_coordinate_system": "one-based within chromosome; global=offset+POS",
            "missing_policy": missing, "root_policy": root,
            "quantization_scheme": "round(65535*cdf), monotone, valid terminal=65535",
            "has_cdf_by_age": not omit_transpose,
            "chromosomes": chromosomes,
            "chromosomes_source": (
                "arg_metadata" if chrom_offsets is None else str(chrom_offsets)
            ),
            "minimum_usable_draws": required_usable_draws,
            "minimum_usable_fraction": min_usable_fraction,
            "multiple_mutation_policy": "exclude site if any draw has multiple mutation records",
            "creation_command": " ".join(sys.argv),
            "extraction_totals": extraction_totals,
            "inputs": [{"path": str(path)} for path in paths],
            "arrays": {
                "positions": {"dtype": "float64", "shape": [int(n)]},
                "age_bins": {"dtype": "uint64", "shape": [int(b)]},
                "cdf_by_snp": {"dtype": "uint16", "shape": [int(n), int(b)]},
                "cdf_by_age": None if omit_transpose else {"dtype": "uint16", "shape": [int(b), int(n)]},
                "multiple_mutation_draw_count": {
                    "dtype": "uint32", "shape": [int(n)]
                },
            },
        }
        (temp / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_store(temp)
        os.replace(temp, output_dir)
    except BaseException:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise


def _expand(patterns: Sequence[str]) -> list[Path]:
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"tree-sequence pattern matched no files: {pattern}")
        result.extend(Path(value) for value in matches)
    return list(dict.fromkeys(result))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", nargs="+", help="tree-sequence files or glob patterns")
    parser.add_argument("--numpy-store", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=1000)
    parser.add_argument("--block-snps", type=int, default=100_000)
    parser.add_argument("--missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--root", choices=("skip", "error"), default="skip")
    parser.add_argument("--omit-transpose", action="store_true")
    parser.add_argument("--min-usable-fraction", type=float, default=0.1)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument(
        "--chrom-offsets", type=Path,
        help="chromosome offsets file used instead of the ARG's chrom_offsets metadata",
    )
    args = parser.parse_args(argv)
    try:
        build_store(_expand(args.trees), args.numpy_store, bin_width=args.bin_width,
                    block_snps=args.block_snps, missing=args.missing, root=args.root,
                    omit_transpose=args.omit_transpose,
                    min_usable_fraction=args.min_usable_fraction,
                    scratch_dir=args.scratch_dir, chrom_offsets=args.chrom_offsets)
    except (OSError, ValueError, tskit.FileFormatError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
