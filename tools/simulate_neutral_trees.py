#!/usr/bin/env python3
"""Generate neutral msprime tree-sequence replicates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import msprime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/neutral_trees"))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(args.output_dir.glob("neutral_*.trees"))
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing tree-sequence files in "
            f"{args.output_dir}"
        )

    ancestries = msprime.sim_ancestry(
        samples=25,
        ploidy=2,
        population_size=50_000,
        sequence_length=10_000_000,
        recombination_rate=1e-8,
        num_replicates=args.replicates,
        random_seed=args.seed,
    )

    rows = []
    for replicate, ancestry in enumerate(ancestries, start=1):
        mutation_seed = args.seed + replicate
        tree_sequence = msprime.sim_mutations(
            ancestry,
            rate=1e-8,
            random_seed=mutation_seed,
        )
        filename = f"neutral_{replicate:03d}.trees"
        tree_sequence.dump(args.output_dir / filename)
        rows.append(
            {
                "replicate": replicate,
                "file": filename,
                "mutation_seed": mutation_seed,
                "num_samples": tree_sequence.num_samples,
                "num_trees": tree_sequence.num_trees,
                "num_sites": tree_sequence.num_sites,
                "num_mutations": tree_sequence.num_mutations,
            }
        )

    with (args.output_dir / "manifest.csv").open("w", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
