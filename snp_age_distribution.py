#!/usr/bin/env python3
"""Estimate SNP-age distributions from replicate tree sequences.

Each mutation at a requested site contributes a unit-mass uniform distribution
between the mutation node's time and its parent node's time in the marginal
tree. The mixture is integrated into bins centred on multiples of ``bin_width``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import tskit


@dataclass(frozen=True)
class AgeInterval:
    """An age constraint contributed by one mutation in one replicate."""

    position: float
    below: float
    above: float
    tree_file: str
    mutation_id: int


def _position_key(position: float) -> float:
    """Canonicalise input positions without imposing integer coordinates."""
    position = float(position)
    if not math.isfinite(position):
        raise ValueError(f"SNP position must be finite, got {position!r}")
    return position


def collect_intervals(
    tree_files: Iterable[str | Path],
    positions: Iterable[float],
    *,
    missing: str = "skip",
    root: str = "skip",
) -> tuple[dict[float, list[AgeInterval]], dict[float, int]]:
    """Collect mutation-node/parent-node age intervals.

    Sites are matched exactly by their floating-point position. Every mutation
    at a matching site contributes independently. ``missing`` controls sites
    absent from a replicate; ``root`` controls mutations whose node is a root.
    Both accept ``"skip"`` or ``"error"``. Zero-width intervals are retained
    and later treated as point masses.
    """
    if missing not in {"skip", "error"} or root not in {"skip", "error"}:
        raise ValueError("missing and root policies must be 'skip' or 'error'")
    requested = tuple(dict.fromkeys(_position_key(p) for p in positions))
    requested_set = set(requested)
    intervals: dict[float, list[AgeInterval]] = {p: [] for p in requested}
    missing_counts = {p: 0 for p in requested}

    for tree_file in tree_files:
        tree_file = Path(tree_file)
        ts = tskit.load(str(tree_file))
        matched: set[float] = set()
        for site in ts.sites():
            position = float(site.position)
            if position not in requested_set:
                continue
            matched.add(position)
            tree = ts.at(position)
            for mutation in site.mutations:
                below = float(ts.node(mutation.node).time)
                parent = tree.parent(mutation.node)
                if parent == tskit.NULL:
                    if root == "error":
                        raise ValueError(
                            f"mutation {mutation.id} at {position:g} in "
                            f"{tree_file} is above a root node"
                        )
                    continue
                above = float(ts.node(parent).time)
                if above < below:
                    raise ValueError(
                        f"invalid age interval [{below}, {above}] for mutation "
                        f"{mutation.id} in {tree_file}"
                    )
                intervals[position].append(
                    AgeInterval(
                        position, below, above, str(tree_file), mutation.id
                    )
                )
        for position in requested_set - matched:
            missing_counts[position] += 1
            if missing == "error":
                raise ValueError(
                    f"position {position:g} is absent from {tree_file}"
                )
    return intervals, missing_counts


def discretize_intervals(
    intervals: Sequence[AgeInterval], bin_width: float = 1000
) -> dict[float, float]:
    """Return a normalized mixture on nearest-``bin_width`` bin centres.

    A nonzero interval has unit total mass and is integrated exactly over bin
    cells ``[centre-width/2, centre+width/2)``. A zero-width interval is a point
    mass assigned to its nearest bin (ties round upward).
    """
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    mass: defaultdict[float, float] = defaultdict(float)
    for interval in intervals:
        low, high = interval.below, interval.above
        if high == low:
            centre = math.floor(low / bin_width + 0.5) * bin_width
            mass[centre] += 1.0
            continue
        first = math.floor(low / bin_width + 0.5)
        last = math.floor(math.nextafter(high, -math.inf) / bin_width + 0.5)
        for index in range(first, last + 1):
            centre = index * bin_width
            left = centre - bin_width / 2
            right = centre + bin_width / 2
            overlap = max(0.0, min(high, right) - max(low, left))
            mass[centre] += overlap / (high - low)
    total = math.fsum(mass.values())
    if total == 0:
        return {}
    return {centre: value / total for centre, value in sorted(mass.items())}


def estimate_distributions(
    tree_files: Iterable[str | Path],
    positions: Iterable[float],
    *,
    bin_width: float = 1000,
    missing: str = "skip",
    root: str = "skip",
) -> tuple[dict[float, dict[float, float]], dict[float, list[AgeInterval]], dict[float, int]]:
    """Collect intervals and estimate a separate distribution per SNP."""
    intervals, missing_counts = collect_intervals(
        tree_files, positions, missing=missing, root=root
    )
    distributions = {
        position: discretize_intervals(values, bin_width)
        for position, values in intervals.items()
    }
    return distributions, intervals, missing_counts


def _read_positions(values: Sequence[str], filename: str | None) -> list[float]:
    positions = [float(value) for value in values]
    if filename is not None:
        with open(filename, encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                text = line.split("#", 1)[0].strip()
                if text:
                    try:
                        positions.append(float(text.split()[0]))
                    except ValueError as error:
                        raise ValueError(
                            f"invalid position on {filename}:{line_number}"
                        ) from error
    if not positions:
        raise ValueError("provide at least one SNP position")
    return positions


def _tree_files(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        files.extend(matches if matches else [pattern])
    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError("provide at least one tree-sequence file")
    missing = [filename for filename in files if not Path(filename).is_file()]
    if missing:
        raise FileNotFoundError(f"tree-sequence file not found: {missing[0]}")
    return files


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", nargs="+", help=".trees files or glob patterns")
    parser.add_argument("--position", "-p", action="append", default=[], help="SNP bp position (repeatable)")
    parser.add_argument("--positions-file", help="one SNP position per line")
    parser.add_argument("--bin-width", type=float, default=1000)
    parser.add_argument("--missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--root", choices=("skip", "error"), default="skip")
    parser.add_argument("--intervals", action="store_true", help="write interval rows instead of distributions")
    args = parser.parse_args(argv)
    try:
        positions = _read_positions(args.position, args.positions_file)
        files = _tree_files(args.trees)
        distributions, intervals, missing_counts = estimate_distributions(
            files,
            positions,
            bin_width=args.bin_width,
            missing=args.missing,
            root=args.root,
        )
    except (OSError, ValueError, tskit.FileFormatError) as error:
        parser.error(str(error))

    writer = csv.writer(sys.stdout, lineterminator="\n")
    if args.intervals:
        writer.writerow(("position", "below_age", "above_age", "tree_file", "mutation_id"))
        for position in intervals:
            for interval in intervals[position]:
                writer.writerow((position, interval.below, interval.above, interval.tree_file, interval.mutation_id))
    else:
        writer.writerow(("position", "age_bin", "probability", "interval_count", "missing_replicates"))
        for position, distribution in distributions.items():
            for age_bin, probability in distribution.items():
                writer.writerow((position, age_bin, probability, len(intervals[position]), missing_counts[position]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
