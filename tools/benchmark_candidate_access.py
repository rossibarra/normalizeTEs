#!/usr/bin/env python3
"""Gate 3 benchmark for scattered interval-store candidate access.

This command performs no scheduler submission. Run it on an appropriately
allocated compute node for production stores and point ``--scratch-dir`` at
node-local storage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from normalize_tes.snp_interval_dataset import SNPAgeIntervalDataset


def load_numeric_vector(path: str | Path, *, integer: bool) -> np.ndarray:
    source = Path(path)
    values = np.load(source) if source.suffix == ".npy" else np.loadtxt(source)
    values = np.asarray(values)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError(f"{source} must contain one finite numeric column")
    if integer:
        if np.any(values != np.floor(values)):
            raise ValueError(f"{source} contains non-integer candidate rows")
        return values.astype(np.int64)
    return values.astype(np.float64)


def _coalesced_slabs(rows: np.ndarray, block_rows: int, gap: int) -> list[tuple[int, int]]:
    rows = np.sort(rows)
    slabs: list[tuple[int, int]] = []
    start = 0
    while start < rows.size:
        stop = start + 1
        while (
            stop < rows.size
            and rows[stop] - rows[stop - 1] <= gap + 1
            and rows[stop] - rows[start] < block_rows
        ):
            stop += 1
        slabs.append((int(rows[start]), int(rows[stop - 1]) + 1))
        start = stop
    return slabs


def estimate_io(
    store: SNPAgeIntervalDataset, rows: np.ndarray, strategy: str, *,
    block_rows: int, coalesce_gap: int,
) -> dict[str, int]:
    record_bytes = store._below.dtype.itemsize + store._above.dtype.itemsize + store._draw_id.dtype.itemsize
    if strategy in {"gather", "cache"}:
        intervals = int(np.sum(store.offsets[rows + 1] - store.offsets[rows], dtype=np.uint64))
        operations = int(rows.size * 3)
        offset_entries = int(rows.size * 2)
    elif strategy == "coalesced":
        slabs = _coalesced_slabs(rows, block_rows, coalesce_gap)
        intervals = sum(int(store.offsets[stop] - store.offsets[start]) for start, stop in slabs)
        operations = len(slabs) * 3
        offset_entries = sum(stop - start + 1 for start, stop in slabs)
    elif strategy == "scan":
        intervals = store.n_intervals
        blocks = (store.positions.size + block_rows - 1) // block_rows
        operations = blocks * 3
        offset_entries = int(store.positions.size + blocks)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    record_total = int(intervals * record_bytes)
    offset_total = int(offset_entries * store.offsets.dtype.itemsize)
    return {
        "estimated_bytes_read": record_total + offset_total,
        "estimated_record_bytes_read": record_total,
        "estimated_offset_bytes_read": offset_total,
        "estimated_endpoint_read_operations": int(operations),
        "estimated_intervals_read": int(intervals),
    }


def _timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - start


def run_benchmark(
    store: SNPAgeIntervalDataset,
    rows: np.ndarray,
    boundaries: np.ndarray,
    scratch_dir: str | Path,
    *,
    repeats: int = 2,
    block_rows: int = 100_000,
    coalesce_gap: int = 64,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    keep_cache: bool = False,
) -> dict:
    """Run all four strategies and return a JSON-serializable report."""
    rows = np.asarray(rows)
    boundaries = np.asarray(boundaries, dtype=np.float64)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("candidate rows must be a one-dimensional integer array")
    rows = rows.astype(np.int64, copy=False)
    if rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("candidate rows must be nonempty and unique")
    if np.any(rows < 0) or np.any(rows >= store.positions.size):
        raise IndexError("candidate row out of bounds")
    if boundaries.ndim != 1 or boundaries.size == 0 or np.any(~np.isfinite(boundaries)):
        raise ValueError("boundaries must be a nonempty finite vector")
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="interval-candidate-benchmark-", dir=scratch))
    cache_path = work / "candidate-cache"
    report = {
        "store": str(Path(store.store_dir).resolve()),
        "n_store_rows": int(store.positions.size),
        "n_store_intervals": store.n_intervals,
        "n_candidate_rows": int(rows.size),
        "n_boundaries": int(boundaries.size),
        "block_rows": int(block_rows),
        "coalesce_gap": int(coalesce_gap),
        "repeats": int(repeats),
        "results": {},
    }
    completed = False
    try:
        reference = None
        for strategy in ("gather", "coalesced", "scan"):
            def evaluate(strategy=strategy):
                return store.boundary_cdfs(
                    rows, boundaries, access_strategy=strategy,
                    block_rows=block_rows, coalesce_gap=coalesce_gap,
                )
            values, first = _timed(evaluate)
            repeat_times = []
            for _ in range(repeats):
                repeated, elapsed = _timed(evaluate)
                np.testing.assert_allclose(repeated, values, rtol=rtol, atol=atol, equal_nan=True)
                repeat_times.append(elapsed)
            if reference is None:
                reference = values
                maximum_error = 0.0
            else:
                np.testing.assert_allclose(values, reference, rtol=rtol, atol=atol, equal_nan=True)
                finite = np.isfinite(values) & np.isfinite(reference)
                maximum_error = float(np.max(np.abs(values[finite] - reference[finite]), initial=0.0))
            result = {
                "first_seconds": first,
                "repeat_seconds": repeat_times,
                "best_repeat_seconds": min(repeat_times),
                "max_abs_error_vs_gather": maximum_error,
                "equal_to_gather": True,
            }
            result.update(estimate_io(
                store, rows, strategy, block_rows=block_rows, coalesce_gap=coalesce_gap
            ))
            report["results"][strategy] = result

        cache, build_seconds = _timed(
            lambda: store.build_candidate_cache(rows, cache_path, block_rows=block_rows)
        )
        def evaluate_cache():
            return store.boundary_cdfs(
                rows, boundaries, access_strategy="cache", cache=cache
            )
        values, first = _timed(evaluate_cache)
        np.testing.assert_allclose(values, reference, rtol=rtol, atol=atol, equal_nan=True)
        repeat_times = []
        for _ in range(repeats):
            repeated, elapsed = _timed(evaluate_cache)
            np.testing.assert_allclose(repeated, reference, rtol=rtol, atol=atol, equal_nan=True)
            repeat_times.append(elapsed)
        finite = np.isfinite(values) & np.isfinite(reference)
        cache_bytes = sum(path.stat().st_size for path in cache_path.iterdir() if path.is_file())
        result = {
            "build_seconds": build_seconds,
            "first_seconds": first,
            "repeat_seconds": repeat_times,
            "best_repeat_seconds": min(repeat_times),
            "cache_bytes_on_disk": int(cache_bytes),
            "build_estimated_bytes_read": int(store.n_intervals * (
                store._below.dtype.itemsize + store._above.dtype.itemsize + store._draw_id.dtype.itemsize
            )),
            "max_abs_error_vs_gather": float(np.max(np.abs(values[finite] - reference[finite]), initial=0.0)),
            "equal_to_gather": True,
        }
        result.update(estimate_io(
            store, rows, "cache", block_rows=block_rows, coalesce_gap=coalesce_gap
        ))
        report["results"]["cache"] = result
        if keep_cache:
            report["cache_path"] = str(cache_path)
        completed = True
        return report
    finally:
        if not keep_cache or not completed:
            shutil.rmtree(work, ignore_errors=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("candidate_rows", type=Path, help="one-column .npy or text row indices")
    parser.add_argument("boundaries", type=Path, help="one-column .npy or text age boundaries")
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--block-rows", type=int, default=100_000)
    parser.add_argument("--coalesce-gap", type=int, default=64)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--output", type=Path, help="write JSON report; must not already exist")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    store = SNPAgeIntervalDataset.open(args.store)
    report = run_benchmark(
        store,
        load_numeric_vector(args.candidate_rows, integer=True),
        load_numeric_vector(args.boundaries, integer=False),
        args.scratch_dir,
        repeats=args.repeats,
        block_rows=args.block_rows,
        coalesce_gap=args.coalesce_gap,
        keep_cache=args.keep_cache,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite report: {args.output}")
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
