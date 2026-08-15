#!/usr/bin/env python3
"""Run target construction or swap sampling for a shard of a TE manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from release_provenance import PROJECT_VERSION


REQUIRED_COLUMNS = ("label", "positions", "target", "output", "seed")


def _manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no target rows")
    labels = [row["label"] for row in rows]
    if any(not isinstance(label, str) or
           re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) is None
           for label in labels):
        raise ValueError(
            "manifest labels must use only letters, digits, dot, dash, underscore"
        )
    if len(labels) != len(set(labels)):
        raise ValueError("manifest labels must be unique")
    for column in ("positions", "target", "output"):
        values = [row[column] for row in rows]
        if any(not value for value in values):
            raise ValueError(f"manifest {column} paths must be nonempty")
        if column != "positions" and len(values) != len(set(values)):
            raise ValueError(f"manifest {column} paths must be unique")
    for row in rows:
        try:
            int(row["seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"seed for {row['label']} is not an integer") from error
    return rows


def _completed(path: Path, mode: str) -> bool:
    """Recognize durable outputs; reject ambiguous pre-existing paths."""
    if not path.exists():
        return False
    metadata_path = path / "metadata.json"
    if not path.is_dir() or not metadata_path.is_file():
        raise ValueError(f"existing output is incomplete: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    software = metadata.get("software")
    output_version = software.get("version") if isinstance(software, dict) else None
    if output_version != PROJECT_VERSION:
        raise ValueError(
            f"existing output {path} was made by normalizeTE "
            f"{output_version!r}, expected {PROJECT_VERSION!r}"
        )
    try:
        if mode == "sample":
            required = ("row_indices.npy", "wasserstein.npy", "diagnostics.csv")
            complete = metadata.get("complete") is True
            if not complete or any(not (path / name).is_file() for name in required):
                raise ValueError
            rows = np.load(path / "row_indices.npy", mmap_mode="r", allow_pickle=False)
            distances = np.load(
                path / "wasserstein.npy", mmap_mode="r", allow_pickle=False
            )
            if (
                rows.ndim != 2
                or rows.size == 0
                or distances.shape != (rows.shape[0],)
                or int(metadata.get("sets", -1)) != rows.shape[0]
                or int(metadata.get("set_size", -1)) != rows.shape[1]
                or (path / "diagnostics.csv").stat().st_size == 0
            ):
                raise ValueError
        else:
            required = ("te_row_indices.npy", "target_cdf.npy", "age_bins.npy")
            complete = "wasserstein_threshold_generations" in metadata
            if not complete or any(not (path / name).is_file() for name in required):
                raise ValueError
            rows = np.load(
                path / "te_row_indices.npy", mmap_mode="r", allow_pickle=False
            )
            cdf = np.load(path / "target_cdf.npy", mmap_mode="r", allow_pickle=False)
            ages = np.load(path / "age_bins.npy", mmap_mode="r", allow_pickle=False)
            if (
                rows.ndim != 1
                or rows.size == 0
                or cdf.ndim != 1
                or cdf.shape != ages.shape
                or cdf.size < 2
            ):
                raise ValueError
    except (OSError, ValueError, EOFError):
        raise ValueError(f"existing output is incomplete: {path}") from None
    return True


def _task_values(args: argparse.Namespace) -> tuple[int, int]:
    task_id = args.task_id
    task_count = args.task_count
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_count is None:
        task_count = int(os.environ.get("AGE_MATCH_TASK_COUNT", "1"))
    if task_count <= 0 or not 0 <= task_id < task_count:
        raise ValueError("task id must lie in [0, task count)")
    return task_id, task_count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("build-targets", "sample", "sample-chain", "gather")
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--sets-per-chain", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=256)
    parser.add_argument("--acceptance-quantile", type=float, default=0.50)
    parser.add_argument("--acceptance-distance", type=float)
    parser.add_argument(
        "--missing-position-policy", choices=("error", "drop"), default="error"
    )
    parser.add_argument("--cdf-block-rows", type=int, default=256)
    parser.add_argument("--search-bin-width", type=int, default=20_000)
    parser.add_argument("--burnin-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--sample-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--max-construction-epochs", type=int, default=50)
    parser.add_argument("--max-chain-proposals", type=int, default=10_000_000)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "build-targets" and not 0 < args.acceptance_quantile < 1:
        raise ValueError("acceptance quantile must be strictly between zero and one")
    if args.acceptance_distance is not None and args.acceptance_distance <= 0:
        raise ValueError("acceptance distance must be positive")
    rows = _manifest(args.manifest)
    task_id, task_count = _task_values(args)
    if args.mode == "sample-chain":
        expected_tasks = len(rows) * args.chains
        if task_count != expected_tasks:
            raise ValueError(
                f"sample-chain requires {expected_tasks} array tasks "
                f"({len(rows)} targets * {args.chains} chains), got {task_count}"
            )
        row_index, chain_index = divmod(task_id, args.chains)
        assigned = [(rows[row_index], chain_index)]
    elif args.mode == "gather":
        if task_count != len(rows):
            raise ValueError(
                f"gather requires one array task per target ({len(rows)}), "
                f"got {task_count}"
            )
        assigned = [(rows[task_id], None)]
    else:
        assigned = [
            (row, None) for index, row in enumerate(rows)
            if index % task_count == task_id
        ]
    print(
        f"task {task_id}/{task_count}: {len(assigned)} assignments from "
        f"{len(rows)} targets",
        flush=True,
    )
    project = Path(__file__).resolve().parent
    for ordinal, (row, chain_index) in enumerate(assigned, start=1):
        suffix = "" if chain_index is None else f" chain={chain_index}"
        print(
            f"[{ordinal}/{len(assigned)}] {args.mode}: {row['label']}{suffix}",
            flush=True,
        )
        if args.mode == "build-targets":
            destination = Path(row["target"])
            if _completed(destination, "build-targets"):
                print(f"complete; skipping {destination}", flush=True)
                continue
            command = [
                sys.executable, str(project / "te_age_target.py"),
                "--store", str(args.store),
                "--te-positions", row["positions"],
                "--output", row["target"],
                "--bootstrap-replicates", str(args.bootstrap_replicates),
                "--bootstrap-batch-size", str(args.bootstrap_batch_size),
                "--acceptance-quantile", str(args.acceptance_quantile),
                "--cdf-block-rows", str(args.cdf_block_rows),
                "--seed", row["seed"],
                "--missing-position-policy", args.missing_position_policy,
            ]
            if args.scratch_dir is not None:
                target_scratch = args.scratch_dir / f"target-{row['label']}"
                target_scratch.mkdir(parents=True, exist_ok=True)
                command.extend(["--scratch-dir", str(target_scratch)])
            if args.acceptance_distance is not None:
                command.extend([
                    "--acceptance-distance", str(args.acceptance_distance)
                ])
        elif args.mode == "sample":
            destination = Path(row["output"])
            if _completed(destination, "sample"):
                print(f"complete; skipping {destination}", flush=True)
                continue
            command = [
                sys.executable, str(project / "sample_age_matched_controls.py"),
                "--store", str(args.store),
                "--target", row["target"],
                "--all-eligible",
                "--output", row["output"],
                "--sets", str(args.chains * args.sets_per_chain),
                "--chains", str(args.chains),
                "--sets-per-chain", str(args.sets_per_chain),
                "--workers", str(args.workers),
                "--seed", row["seed"],
                "--search-bin-width", str(args.search_bin_width),
                "--burnin-accepted-sweeps", str(args.burnin_accepted_sweeps),
                "--sample-accepted-sweeps", str(args.sample_accepted_sweeps),
                "--max-construction-epochs", str(args.max_construction_epochs),
                "--max-chain-proposals", str(args.max_chain_proposals),
                "--cdf-block-rows", str(args.cdf_block_rows),
            ]
            if args.scratch_dir is not None:
                command.extend([
                    "--work-dir", str(args.scratch_dir / f"match-{row['label']}")
                ])
            if args.keep_checkpoints:
                command.append("--keep-checkpoints")
            if args.resume:
                command.append("--resume")
        elif args.mode == "sample-chain":
            if args.scratch_dir is None:
                raise ValueError("sample-chain requires --scratch-dir")
            assert chain_index is not None
            chain_dir = Path(f"{row['output']}.chains")
            chain_output = chain_dir / f"chain-{chain_index:03d}.npz"
            command = [
                sys.executable, str(project / "distributed_age_match.py"), "chain",
                "--store", str(args.store),
                "--target", row["target"],
                "--all-eligible",
                "--chain-index", str(chain_index),
                "--chain-output", str(chain_output),
                "--work-dir", str(
                    args.scratch_dir / f"chain-{row['label']}-{chain_index:03d}"
                ),
                "--chains", str(args.chains),
                "--sets-per-chain", str(args.sets_per_chain),
                "--seed", row["seed"],
                "--search-bin-width", str(args.search_bin_width),
                "--burnin-accepted-sweeps", str(args.burnin_accepted_sweeps),
                "--sample-accepted-sweeps", str(args.sample_accepted_sweeps),
                "--max-construction-epochs", str(args.max_construction_epochs),
                "--max-chain-proposals", str(args.max_chain_proposals),
                "--cdf-block-rows", str(args.cdf_block_rows),
                "--resume",
            ]
        else:
            if args.scratch_dir is None:
                raise ValueError("gather requires --scratch-dir")
            destination = Path(row["output"])
            if _completed(destination, "sample"):
                print(f"complete; skipping {destination}", flush=True)
                continue
            command = [
                sys.executable, str(project / "distributed_age_match.py"), "gather",
                "--store", str(args.store),
                "--target", row["target"],
                "--all-eligible",
                "--chain-dir", f"{row['output']}.chains",
                "--output", row["output"],
                "--work-dir", str(args.scratch_dir / f"gather-{row['label']}"),
                "--chains", str(args.chains),
                "--sets-per-chain", str(args.sets_per_chain),
                "--seed", row["seed"],
                "--search-bin-width", str(args.search_bin_width),
                "--burnin-accepted-sweeps", str(args.burnin_accepted_sweeps),
                "--sample-accepted-sweeps", str(args.sample_accepted_sweeps),
                "--max-construction-epochs", str(args.max_construction_epochs),
                "--max-chain-proposals", str(args.max_chain_proposals),
                "--cdf-block-rows", str(args.cdf_block_rows),
            ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
