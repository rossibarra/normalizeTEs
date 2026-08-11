#!/usr/bin/env python3
"""Benchmark Gate 2 interval extraction on one tree-sequence draw.

This command is intentionally single-draw and read-only with respect to its
input.  It writes one JSON report atomically and never submits work itself.
Production TSZ inputs should be run through the repository's HPC wrapper.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np
import tskit
import tszip

from build_snp_interval_store import (
    _integral_int64,
    _lexicographic_parent_lookup,
    _record_dtype,
)
from snp_interval_dataset import interval_cdf


T = TypeVar("T")


def _rss_bytes() -> int:
    """Return current resident bytes on Linux, with a portable fallback."""
    try:
        with Path("/proc/self/status").open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == "darwin" else 1024))


class _RSSSampler:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.start = _rss_bytes()
        self.peak = self.start
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, _rss_bytes())

    def __enter__(self) -> "_RSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak = max(self.peak, _rss_bytes())
        self._stop.set()
        self._thread.join()


def _measure(operation: Callable[[], T]) -> tuple[T, dict[str, int | float]]:
    gc.collect()
    started = time.perf_counter()
    with _RSSSampler() as memory:
        value = operation()
    elapsed = time.perf_counter() - started
    ended = _rss_bytes()
    return value, {
        "seconds": elapsed,
        "rss_start_bytes": memory.start,
        "rss_end_bytes": ended,
        "rss_peak_bytes": memory.peak,
        "rss_peak_increment_bytes": max(0, memory.peak - memory.start),
    }


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _selective_sites(path: Path) -> np.ndarray:
    """Read and decode only the coordinate dictionary and site-position column."""
    from tszip.compression import load_zarr

    with path.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise ValueError("selective access requires a TSZ ZIP store")
    with load_zarr(path) as root:
        coordinates = np.asarray(root["coordinates"][:])
        encoded = np.asarray(root["sites/position"][:], dtype=np.int64)
        return np.asarray(coordinates[encoded], dtype=np.float64)


def _load_tree_sequence(path: Path) -> tskit.TreeSequence:
    return tszip.load(str(path))


def _parent_lookup_phases(
    ts: tskit.TreeSequence,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Run the applicable production parent lookup once and report each phase."""
    tables = ts.tables
    edge_child = np.asarray(tables.edges.child, dtype=np.int64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int64)
    edge_left_raw = np.asarray(tables.edges.left, dtype=np.float64)
    edge_right_raw = np.asarray(tables.edges.right, dtype=np.float64)
    if np.any(~np.isfinite(edge_left_raw)) or np.any(~np.isfinite(edge_right_raw)):
        raise ValueError("edge coordinates must be finite")
    mutation_site = np.asarray(tables.mutations.site, dtype=np.int64)
    mutation_node = np.asarray(tables.mutations.node, dtype=np.int64)
    site_position = _integral_int64(np.asarray(tables.sites.position), "site positions")
    mutation_position = site_position[mutation_site]
    integral_edges = bool(
        np.all(edge_left_raw == np.floor(edge_left_raw))
        and np.all(edge_right_raw == np.floor(edge_right_raw)))
    phases: dict[str, object] = {
        "algorithm": "int64_composite" if integral_edges else "structured_child_float64_left",
        "integral_edge_coordinates": integral_edges,
    }

    if integral_edges:
        edge_left = edge_left_raw.astype(np.int64, copy=False)
        edge_right = edge_right_raw.astype(np.int64, copy=False)
        sequence_length = int(ts.sequence_length)
        if float(sequence_length) != float(ts.sequence_length):
            raise ValueError("composite lookup requires an integral sequence length")
        stride = sequence_length + 1
        if ts.num_nodes and (ts.num_nodes - 1) > np.iinfo(np.int64).max // stride:
            raise OverflowError("composite edge key exceeds int64")

        def construct_keys() -> tuple[np.ndarray, np.ndarray]:
            return (
                edge_child * np.int64(stride) + edge_left,
                mutation_node * np.int64(stride) + mutation_position,
            )
    else:
        edge_left = edge_left_raw
        edge_right = edge_right_raw
        key_dtype = np.dtype([("child", "<i8"), ("left", "<f8")])

        def construct_keys() -> tuple[np.ndarray, np.ndarray]:
            edge_keys = np.empty(edge_child.size, dtype=key_dtype)
            edge_keys["child"] = edge_child
            edge_keys["left"] = edge_left
            query_keys = np.empty(mutation_node.size, dtype=key_dtype)
            query_keys["child"] = mutation_node
            query_keys["left"] = mutation_position
            return edge_keys, query_keys

    (edge_key, query_key), phases["key_construction"] = _measure(construct_keys)
    if integral_edges and (edge_key.dtype != np.int64 or query_key.dtype != np.int64):
        raise AssertionError("composite keys must be int64")
    phases["key_construction"]["retained_array_bytes"] = int(edge_key.nbytes + query_key.nbytes)

    order, phases["stable_edge_sort"] = _measure(lambda: np.argsort(
        edge_key, kind="stable",
        **({"order": ("child", "left")} if not integral_edges else {})))
    phases["stable_edge_sort"]["retained_array_bytes"] = int(order.nbytes)

    def reorder() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return edge_key[order], edge_child[order], edge_right[order], edge_parent[order]

    reordered, phases["edge_reorder"] = _measure(reorder)
    edge_key_sorted, child_sorted, right_sorted, parent_sorted = reordered
    phases["edge_reorder"]["retained_array_bytes"] = int(
        sum(array.nbytes for array in reordered)
    )

    def search() -> np.ndarray:
        index = np.searchsorted(edge_key_sorted, query_key, side="right") - 1
        safe = np.maximum(index, 0)
        covered = (
            (index >= 0)
            & (child_sorted[safe] == mutation_node)
            & (mutation_position.astype(np.float64) < right_sorted[safe])
        )
        result = np.full(mutation_node.size, tskit.NULL, dtype=np.int64)
        result[covered] = parent_sorted[safe[covered]]
        return result

    parents, phases["search_and_guards"] = _measure(search)
    phases["search_and_guards"]["retained_array_bytes"] = int(parents.nbytes)
    phases["array_inventory_bytes"] = {
        "edge_input_columns": int(sum(a.nbytes for a in (edge_child, edge_parent, edge_left, edge_right))),
        "mutation_query_columns": int(sum(a.nbytes for a in (mutation_node, mutation_position))),
        "keys": int(edge_key.nbytes + query_key.nbytes),
        "permutation": int(order.nbytes),
        "reordered_columns_and_key": int(sum(a.nbytes for a in reordered)),
        "parent_result": int(parents.nbytes),
    }
    return parents, mutation_position, mutation_site, phases


def _count_and_bucket_metrics(
    parents: np.ndarray, mutation_rows: np.ndarray, n_sites: int,
    num_buckets: int,
) -> dict[str, object]:
    covered = parents != tskit.NULL

    def reduce_counts() -> tuple[np.ndarray, np.ndarray]:
        usable = np.bincount(mutation_rows[covered], minlength=n_sites).astype(np.uint32)
        skipped = np.bincount(mutation_rows[~covered], minlength=n_sites).astype(np.uint32)
        return usable, skipped

    (usable, skipped), timing = _measure(reduce_counts)
    usable_rows = mutation_rows[covered]
    bucket_ids = np.minimum(
        (usable_rows.astype(np.uint64) * num_buckets) // max(1, n_sites),
        num_buckets - 1,
    ).astype(np.int64)
    bucket_counts = np.bincount(bucket_ids, minlength=num_buckets).astype(np.uint64)
    record_sizes = {"float64": 4 + 8 + 8 + 1, "float32": 4 + 4 + 4 + 1}
    distributions: dict[str, object] = {}
    for dtype, size in record_sizes.items():
        byte_counts = bucket_counts * size
        distributions[dtype] = {
            "packed_record_bytes": size,
            "total_scratch_bytes": int(byte_counts.sum(dtype=np.uint64)),
            "largest_bucket_bytes": int(byte_counts.max(initial=0)),
            "largest_bucket_sort_workspace_bytes_factor_3": int(3 * byte_counts.max(initial=0)),
        }
    return {
        "timing_and_rss": timing,
        "present_site_count": int(n_sites),
        "mutation_count": int(mutation_rows.size),
        "usable_interval_count": int(covered.sum()),
        "root_skipped_count": int((~covered).sum()),
        "sites_with_usable_intervals": int(np.count_nonzero(usable)),
        "sites_with_root_skips": int(np.count_nonzero(skipped)),
        "bucket_count": num_buckets,
        "bucket_interval_counts": [int(value) for value in bucket_counts],
        "bucket_min_intervals": int(bucket_counts.min(initial=0)),
        "bucket_max_intervals": int(bucket_counts.max(initial=0)),
        "bucket_mean_intervals": float(bucket_counts.mean()),
        "bucket_imbalance_max_over_mean": (
            float(bucket_counts.max(initial=0) / bucket_counts.mean())
            if bucket_counts.mean() else 0.0
        ),
        "scratch_estimates": distributions,
    }


def _bucket_write_metrics(
    ts: tskit.TreeSequence, parents: np.ndarray, mutation_rows: np.ndarray,
    scratch_dir: Path | None,
) -> dict[str, object]:
    """Materialize and fsync one draw's actual float64 scratch records."""
    covered = parents != tskit.NULL
    mutation_node = np.asarray(ts.tables.mutations.node, dtype=np.int64)
    node_time = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    dtype = _record_dtype(np.dtype("float64"), np.dtype("uint8"))

    def make_records() -> np.ndarray:
        records = np.empty(int(covered.sum()), dtype=dtype)
        records["row"] = mutation_rows[covered]
        records["below"] = node_time[mutation_node[covered]]
        records["above"] = node_time[parents[covered]]
        records["draw_id"] = 0
        return records

    records, construction = _measure(make_records)
    parent = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) if scratch_dir is None else scratch_dir
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    descriptor, name = tempfile.mkstemp(prefix="gate2-records-", suffix=".bin", dir=parent)
    os.close(descriptor)

    def write_records() -> int:
        with open(name, "wb") as handle:
            records.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        return os.path.getsize(name)

    try:
        written, writing = _measure(write_records)
    finally:
        Path(name).unlink(missing_ok=True)
    return {
        "record_count": int(records.size),
        "record_bytes": int(dtype.itemsize),
        "written_bytes": int(written),
        "construction": construction,
        "writing": writing,
        "write_bytes_per_second": float(written / writing["seconds"]),
        "scratch_directory": str(parent),
    }


def _fallback_subset_metrics(
    ts: tskit.TreeSequence, parents: np.ndarray, mutation_position: np.ndarray,
    *, node_sample_size: int, seed: int,
) -> dict[str, object]:
    """Benchmark structured-key fallback on complete edge histories for sampled nodes."""
    tables = ts.tables
    edge_child = np.asarray(tables.edges.child, dtype=np.int64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int64)
    edge_left = np.asarray(tables.edges.left, dtype=np.float64)
    edge_right = np.asarray(tables.edges.right, dtype=np.float64)
    mutation_node = np.asarray(tables.mutations.node, dtype=np.int64)
    nodes = np.unique(mutation_node)
    rng = np.random.default_rng(seed)
    selected_nodes = (
        nodes if nodes.size <= node_sample_size
        else rng.choice(nodes, size=node_sample_size, replace=False)
    )

    def select() -> tuple[np.ndarray, np.ndarray]:
        return np.isin(edge_child, selected_nodes), np.isin(mutation_node, selected_nodes)

    (edge_mask, mutation_mask), selection = _measure(select)

    def lookup() -> np.ndarray:
        return _lexicographic_parent_lookup(
            edge_child[edge_mask], edge_left[edge_mask], edge_right[edge_mask],
            edge_parent[edge_mask], mutation_node[mutation_mask],
            mutation_position[mutation_mask].astype(np.float64))

    fallback, lookup_measurement = _measure(lookup)
    matches = np.array_equal(fallback, parents[mutation_mask])
    if not matches:
        raise ValueError("structured fallback differs from composite lookup")
    return {
        "sampled_nodes": int(selected_nodes.size),
        "selected_edges": int(edge_mask.sum()),
        "selected_mutations": int(mutation_mask.sum()),
        "selection": selection,
        "lookup": lookup_measurement,
        "matches_composite": matches,
    }


def _precision_metrics(
    ts: tskit.TreeSequence, parents: np.ndarray, *, sample_size: int,
    point_count: int, seed: int,
) -> dict[str, object]:
    usable = np.flatnonzero(parents != tskit.NULL)
    rng = np.random.default_rng(seed)
    selected = (
        usable if usable.size <= sample_size
        else np.sort(rng.choice(usable, size=sample_size, replace=False))
    )
    node_time = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    mutation_node = np.asarray(ts.tables.mutations.node, dtype=np.int64)[selected]
    below64 = node_time[mutation_node]
    above64 = node_time[parents[selected]]
    below32 = below64.astype(np.float32)
    above32 = above64.astype(np.float32)
    collapsed = above32 <= below32
    endpoints = np.concatenate((below64, above64))
    points = (
        np.unique(np.quantile(endpoints, np.linspace(0, 1, point_count)))
        if endpoints.size else np.empty(0, dtype=np.float64)
    )
    if selected.size and points.size:
        cdf64 = interval_cdf(below64, above64, points)
        cdf32 = interval_cdf(below32, above32, points)
        cdf_error = np.abs(cdf64 - cdf32)
        aggregate_error = np.abs(cdf64.mean(axis=0) - cdf32.mean(axis=0))
    else:
        cdf_error = aggregate_error = np.zeros(1, dtype=np.float64)
    lower_error = np.abs(below64 - below32.astype(np.float64))
    upper_error = np.abs(above64 - above32.astype(np.float64))
    midpoint_error = np.abs(
        (below64 + above64) / 2
        - (below32.astype(np.float64) + above32.astype(np.float64)) / 2
    )

    def errors(values: np.ndarray) -> dict[str, float]:
        return {
            "maximum_absolute": float(values.max(initial=0)),
            "mean_absolute": float(values.mean()) if values.size else 0.0,
            "p99_absolute": float(np.quantile(values, 0.99)) if values.size else 0.0,
        }

    return {
        "sample_size": int(selected.size),
        "evaluation_point_count": int(points.size),
        "collapsed_interval_count": int(collapsed.sum()),
        "below_endpoint_error": errors(lower_error),
        "above_endpoint_error": errors(upper_error),
        "midpoint_error": errors(midpoint_error),
        "individual_cdf_error": errors(cdf_error),
        "aggregate_cdf_error": errors(aggregate_error),
    }


def _stratified_audit_indices(
    parents: np.ndarray, size: int, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    usable = np.flatnonzero(parents != tskit.NULL)
    roots = np.flatnonzero(parents == tskit.NULL)
    target = min(size, parents.size)
    root_take = min(roots.size, target // 2)
    usable_take = min(usable.size, target - root_take)
    root_take = min(roots.size, target - usable_take)

    def choose(values: np.ndarray, count: int) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        return values if count == values.size else rng.choice(values, count, replace=False)

    chosen = np.concatenate((choose(usable, usable_take), choose(roots, root_take)))
    rng.shuffle(chosen)
    return chosen, {"predicted_usable": usable_take, "predicted_root": root_take}


def _scalar_parent_audit(
    ts: tskit.TreeSequence, parents: np.ndarray, mutation_position: np.ndarray,
    *, sample_size: int, seed: int,
) -> dict[str, object]:
    selected, strata = _stratified_audit_indices(parents, sample_size, seed)
    mutation_node = np.asarray(ts.tables.mutations.node, dtype=np.int64)

    def audit() -> list[dict[str, int | float]]:
        mismatches: list[dict[str, int | float]] = []
        for mutation_id in selected:
            i = int(mutation_id)
            expected = int(ts.at(float(mutation_position[i])).parent(int(mutation_node[i])))
            actual = int(parents[i])
            if expected != actual and len(mismatches) < 20:
                mismatches.append({
                    "mutation_id": i,
                    "position": int(mutation_position[i]),
                    "node": int(mutation_node[i]),
                    "vector_parent": actual,
                    "scalar_parent": expected,
                })
        return mismatches

    mismatches, measurement = _measure(audit)
    return {
        "seed": seed,
        "requested_sample_size": sample_size,
        "actual_sample_size": int(selected.size),
        "strata": strata,
        "timing_and_rss": measurement,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches,
        "passed": len(mismatches) == 0 and selected.size == min(sample_size, parents.size),
    }


def benchmark_gate2(
    tree_file: Path, *, num_buckets: int = 100, audit_size: int = 10_000,
    precision_sample_size: int = 10_000, precision_points: int = 21,
    fallback_node_sample: int = 10_000, scratch_dir: Path | None = None,
    seed: int = 1729,
) -> dict[str, object]:
    path = Path(tree_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (num_buckets <= 0 or audit_size <= 0 or precision_sample_size <= 0
            or precision_points < 2 or fallback_node_sample <= 0):
        raise ValueError("bucket/sample sizes must be positive and precision_points >= 2")

    selective: dict[str, object]
    try:
        positions, selective_measurement = _measure(lambda: _selective_sites(path))
        selective = {
            "available": True,
            "timing_and_rss": selective_measurement,
            "site_count": int(positions.size),
            "positions_sha256": _array_digest(positions),
        }
        del positions
    except Exception as error:
        selective = {
            "available": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    ts, full_measurement = _measure(lambda: _load_tree_sequence(path))
    full_positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
    full = {
        "timing_and_rss": full_measurement,
        "site_count": int(ts.num_sites),
        "positions_sha256": _array_digest(full_positions),
        "num_mutations": int(ts.num_mutations),
        "num_edges": int(ts.num_edges),
        "num_nodes": int(ts.num_nodes),
        "num_trees": int(ts.num_trees),
    }
    if selective.get("available"):
        selective["matches_full_load"] = (
            selective["site_count"] == full["site_count"]
            and selective["positions_sha256"] == full["positions_sha256"]
        )
        if not selective["matches_full_load"]:
            raise ValueError("selective TSZ site positions disagree with full load")

    parents, mutation_position, mutation_rows, parent_phases = _parent_lookup_phases(ts)
    count_metrics = _count_and_bucket_metrics(
        parents, mutation_rows, ts.num_sites, num_buckets
    )
    bucket_write = _bucket_write_metrics(
        ts, parents, mutation_rows,
        None if scratch_dir is None else Path(scratch_dir))
    fallback = _fallback_subset_metrics(
        ts, parents, mutation_position,
        node_sample_size=fallback_node_sample, seed=seed)
    precision = _precision_metrics(
        ts, parents, sample_size=precision_sample_size,
        point_count=precision_points, seed=seed,
    )
    audit = _scalar_parent_audit(
        ts, parents, mutation_position, sample_size=audit_size, seed=seed,
    )
    if not audit["passed"]:
        raise ValueError(f"scalar parent audit failed: {audit['mismatch_count']} mismatches")
    return {
        "schema_version": 1,
        "gate": 2,
        "input": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        },
        "configuration": {
            "num_buckets": num_buckets,
            "audit_size": audit_size,
            "precision_sample_size": precision_sample_size,
            "precision_points": precision_points,
            "fallback_node_sample": fallback_node_sample,
            "scratch_dir": None if scratch_dir is None else str(scratch_dir),
            "seed": seed,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tskit": tskit.__version__,
            "tszip": tszip.__version__,
            "platform": platform.platform(),
        },
        "selective_sites_access": selective,
        "full_tsz_load": full,
        "composite_parent_lookup": parent_phases,
        "counts_and_buckets": count_metrics,
        "bucket_write_throughput": bucket_write,
        "fallback_grouped_search": fallback,
        "float32_precision": precision,
        "scalar_parent_audit": audit,
    }


def write_json_atomic(output: Path, report: dict[str, object]) -> None:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path, help="one real TSZ (synthetic .trees also supported)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-buckets", type=int, default=100)
    parser.add_argument("--audit-size", type=int, default=10_000)
    parser.add_argument("--precision-sample-size", type=int, default=10_000)
    parser.add_argument("--precision-points", type=int, default=21)
    parser.add_argument("--fallback-node-sample", type=int, default=10_000)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = benchmark_gate2(
        args.tree, num_buckets=args.num_buckets,
        audit_size=args.audit_size,
        precision_sample_size=args.precision_sample_size,
        precision_points=args.precision_points,
        fallback_node_sample=args.fallback_node_sample,
        scratch_dir=args.scratch_dir, seed=args.seed,
    )
    write_json_atomic(args.output, report)
    print(f"Wrote Gate 2 benchmark report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
