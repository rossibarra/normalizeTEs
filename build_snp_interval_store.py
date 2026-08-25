#!/usr/bin/env python3
"""Build a compact per-SNP posterior mutation-age interval store.

The builder reads each ARG once to form a sorted union site catalog and once
to extract intervals.  Extraction uses table columns only; it does not walk
marginal trees.  Usable records are appended to row-range bucket files, then
stable-sorted into the final ragged NumPy arrays.
"""

from __future__ import annotations

import argparse
import hashlib
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

from build_snp_age_store import (
    _chromosome_table,
    load_chrom_offsets,
)
from snp_interval_dataset import compute_interval_store_content_sha256


SCHEMA_VERSION = "snp-age-interval-v1"
STATUS_ABSENT = np.uint8(0)
STATUS_PRESENT_NO_INTERVAL = np.uint8(1)
STATUS_USABLE = np.uint8(2)
UINT32_MAX = np.iinfo(np.uint32).max


def _load(path: Path) -> tskit.TreeSequence:
    """Load ordinary tskit files and tszip-compressed ``.tsz`` files."""
    return tszip.load(str(path))


def _selective_tsz_catalog(
    path: Path,
) -> tuple[np.ndarray, object, float]:
    """Read decoded site positions and top-level metadata from a TSZ archive.

    TSZ stores site positions as indices into a shared coordinate dictionary;
    returning ``sites/position`` directly would therefore be incorrect.
    """
    from tskit.metadata import parse_metadata_schema
    from tszip.compression import load_zarr

    with load_zarr(path) as root:
        coordinates = np.asarray(root["coordinates"][:])
        encoded_positions = np.asarray(root["sites/position"][:], dtype=np.int64)
        if np.any(encoded_positions < 0) or np.any(encoded_positions >= coordinates.size):
            raise ValueError(f"{path} contains an invalid encoded site position")
        positions = np.asarray(coordinates[encoded_positions], dtype=np.float64)
        sequence_length = float(root.attrs["sequence_length"])
        schema_text = np.asarray(
            root["metadata_schema"][:], dtype=np.uint8
        ).tobytes().decode("utf-8")
        metadata_bytes = np.asarray(
            root["metadata"][:], dtype=np.uint8
        ).tobytes()
        metadata = parse_metadata_schema(schema_text).decode_row(metadata_bytes)
    return positions, metadata, sequence_length


def _catalog_header(path: Path) -> tuple[np.ndarray, object, float, str]:
    """Return catalog fields, using selective TSZ access when possible."""
    if path.suffix.lower() == ".tsz":
        positions, metadata, sequence_length = _selective_tsz_catalog(path)
        return positions, metadata, sequence_length, "selective_tsz_zarr"
    ts = _load(path)
    return (
        np.asarray(ts.tables.sites.position, dtype=np.float64),
        ts.metadata,
        float(ts.sequence_length),
        "full_tree_sequence_fallback",
    )


def pack_status_row(values: np.ndarray) -> np.ndarray:
    """Pack four two-bit status values into each byte."""
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 3):
        raise ValueError("statuses must be a one-dimensional array in [0, 3]")
    result = np.zeros((values.size + 3) // 4, dtype=np.uint8)
    for shift in range(4):
        part = values[shift::4]
        result[: part.size] |= part << np.uint8(2 * shift)
    return result


def unpack_status_row(packed: np.ndarray, size: int) -> np.ndarray:
    """Decode a two-bit packed status row."""
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 1 or size < 0 or packed.size != (size + 3) // 4:
        raise ValueError("packed status row has the wrong size")
    result = np.empty(size, dtype=np.uint8)
    for shift in range(4):
        part = result[shift::4]
        part[:] = (packed[: part.size] >> np.uint8(2 * shift)) & np.uint8(3)
    return result


def _integral_int64(values: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values)
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError(f"{label} must be finite integers")
    info = np.iinfo(np.int64)
    if np.any(values < info.min) or np.any(values > info.max):
        raise OverflowError(f"{label} exceed signed 64-bit range")
    return values.astype(np.int64)


def _lexicographic_parent_lookup(
    edge_child: np.ndarray,
    edge_left: np.ndarray,
    edge_right: np.ndarray,
    edge_parent: np.ndarray,
    mutation_node: np.ndarray,
    mutation_position: np.ndarray,
) -> np.ndarray:
    """Vectorized fallback lookup using structured ``(child, left)`` keys."""
    key_dtype = np.dtype([("child", "<i8"), ("left", "<f8")])
    edge_keys = np.empty(edge_child.size, dtype=key_dtype)
    edge_keys["child"] = edge_child
    edge_keys["left"] = edge_left
    order = np.argsort(edge_keys, kind="stable", order=("child", "left"))
    edge_keys = edge_keys[order]
    query_keys = np.empty(mutation_node.size, dtype=key_dtype)
    query_keys["child"] = mutation_node
    query_keys["left"] = mutation_position
    index = np.searchsorted(edge_keys, query_keys, side="right") - 1
    safe = np.maximum(index, 0)
    child = edge_child[order]
    right = edge_right[order]
    parent = edge_parent[order]
    covered = (
        (index >= 0)
        & (child[safe] == mutation_node)
        & (mutation_position < right[safe])
    )
    result = np.full(mutation_node.size, tskit.NULL, dtype=np.int64)
    result[covered] = parent[safe[covered]]
    return result


def lookup_mutation_parents(
    ts: tskit.TreeSequence, *, allow_fractional_edges: bool = True
) -> np.ndarray:
    """Return each mutation node's covering-edge parent, or ``tskit.NULL``.

    Integral coordinates use exact signed-64-bit composite keys. Fractional
    edge endpoints use a fully vectorized structured-key fallback. Site
    coordinates are required to be integral by the store format.
    """
    tables = ts.tables
    mutation_site = np.asarray(tables.mutations.site, dtype=np.int64)
    mutation_node = np.asarray(tables.mutations.node, dtype=np.int64)
    site_position = _integral_int64(
        np.asarray(tables.sites.position), "site positions"
    )
    mutation_position = site_position[mutation_site]
    if mutation_node.size == 0:
        return np.empty(0, dtype=np.int64)

    edge_child = np.asarray(tables.edges.child, dtype=np.int64)
    edge_parent = np.asarray(tables.edges.parent, dtype=np.int64)
    edge_left_raw = np.asarray(tables.edges.left)
    edge_right_raw = np.asarray(tables.edges.right)
    if edge_child.size == 0:
        return np.full(mutation_node.size, tskit.NULL, dtype=np.int64)
    if np.any(~np.isfinite(edge_left_raw)) or np.any(~np.isfinite(edge_right_raw)):
        raise ValueError("edge coordinates must be finite")

    integral_edges = (
        np.all(edge_left_raw == np.floor(edge_left_raw))
        and np.all(edge_right_raw == np.floor(edge_right_raw))
    )
    if not integral_edges:
        if not allow_fractional_edges:
            raise ValueError("composite edge lookup requires integral coordinates")
        return _lexicographic_parent_lookup(
            edge_child,
            edge_left_raw.astype(np.float64, copy=False),
            edge_right_raw.astype(np.float64, copy=False),
            edge_parent,
            mutation_node,
            mutation_position.astype(np.float64),
        )

    edge_left = _integral_int64(edge_left_raw, "edge left coordinates")
    edge_right = _integral_int64(edge_right_raw, "edge right coordinates")
    sequence_length = int(ts.sequence_length)
    if float(sequence_length) != float(ts.sequence_length) or sequence_length < 0:
        raise ValueError("sequence length must be a nonnegative integer")
    stride = sequence_length + 1
    if ts.num_nodes and ts.num_nodes * stride - 1 > np.iinfo(np.int64).max:
        raise OverflowError("composite edge key exceeds int64")

    edge_key = edge_child * np.int64(stride) + edge_left
    query_key = mutation_node * np.int64(stride) + mutation_position
    if edge_key.dtype != np.int64 or query_key.dtype != np.int64:
        raise AssertionError("composite keys must be int64")
    order = np.argsort(edge_key, kind="stable")
    edge_key = edge_key[order]
    edge_child_sorted = edge_child[order]
    edge_right_sorted = edge_right[order]
    edge_parent_sorted = edge_parent[order]
    index = np.searchsorted(edge_key, query_key, side="right") - 1
    safe = np.maximum(index, 0)
    covered = (
        (index >= 0)
        & (edge_child_sorted[safe] == mutation_node)
        & (mutation_position < edge_right_sorted[safe])
    )
    result = np.full(mutation_node.size, tskit.NULL, dtype=np.int64)
    result[covered] = edge_parent_sorted[safe[covered]]
    return result


def audit_mutation_parent_lookup(
    ts: tskit.TreeSequence, *, sample_size: int = 10_000, seed: int = 0
) -> dict[str, int]:
    """Cross-check a stratified mutation sample against scalar tree traversal.

    The sample is split between predicted covered and root-skipped mutations
    whenever both classes exist.  This is intended as the real-draw Gate 2
    audit; it is deliberately separate from production extraction.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    parents = lookup_mutation_parents(ts)
    n_mutations = parents.size
    if n_mutations == 0:
        return {"sampled": 0, "covered": 0, "root_skipped": 0, "seed": int(seed)}

    rng = np.random.default_rng(seed)
    covered_ids = np.flatnonzero(parents != tskit.NULL)
    root_ids = np.flatnonzero(parents == tskit.NULL)
    target = min(int(sample_size), n_mutations)
    covered_n = min(covered_ids.size, target // 2 if root_ids.size else target)
    root_n = min(root_ids.size, target // 2 if covered_ids.size else target)
    remaining = target - covered_n - root_n
    if remaining:
        covered_extra = min(covered_ids.size - covered_n, remaining)
        covered_n += covered_extra
        remaining -= covered_extra
        root_n += min(root_ids.size - root_n, remaining)

    selected_parts = []
    if covered_n:
        selected_parts.append(rng.choice(covered_ids, size=covered_n, replace=False))
    if root_n:
        selected_parts.append(rng.choice(root_ids, size=root_n, replace=False))
    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    selected.sort()

    tables = ts.tables
    mutation_site = np.asarray(tables.mutations.site, dtype=np.int64)
    mutation_node = np.asarray(tables.mutations.node, dtype=np.int64)
    site_position = np.asarray(tables.sites.position, dtype=np.float64)
    node_time = np.asarray(tables.nodes.time, dtype=np.float64)
    for mutation_id in selected:
        node = int(mutation_node[mutation_id])
        position = float(site_position[mutation_site[mutation_id]])
        expected = int(ts.at(position).parent(node))
        observed = int(parents[mutation_id])
        if observed != expected:
            raise AssertionError(
                f"mutation {mutation_id} at {position:g}: vector parent "
                f"{observed} != scalar parent {expected}"
            )
        if observed != tskit.NULL:
            below = float(node_time[node])
            above = float(node_time[observed])
            if not above > below:
                raise AssertionError(
                    f"mutation {mutation_id} has nonpositive audited interval"
                )
    return {
        "sampled": int(selected.size),
        "covered": int(covered_n),
        "root_skipped": int(root_n),
        "seed": int(seed),
    }


def _inspect_inputs(
    paths: Sequence[Path], chrom_offsets: Path | None
) -> tuple[np.ndarray, list[dict[str, int | str]], float, list[str]]:
    supplied = load_chrom_offsets(chrom_offsets) if chrom_offsets is not None else None
    positions = np.empty(0, dtype=np.float64)
    chromosomes: list[dict[str, int | str]] | None = None
    sequence_length: float | None = None
    access_methods: list[str] = []
    for path in paths:
        current_positions, metadata, current_sequence_length, method = _catalog_header(path)
        access_methods.append(method)
        _integral_int64(current_positions, f"{path} site positions")
        if current_positions.size and np.any(current_positions[1:] <= current_positions[:-1]):
            raise ValueError(f"{path} site positions must be strictly increasing")
        positions = (
            current_positions.copy()
            if positions.size == 0
            else np.union1d(positions, current_positions)
        )
        if supplied is None:
            if not isinstance(metadata, dict) or "chrom_offsets" not in metadata:
                raise ValueError(
                    f"{path} lacks top-level chrom_offsets metadata; "
                    "supply --chrom-offsets"
                )
            current_chromosomes = _chromosome_table(
                metadata["chrom_offsets"], path, current_sequence_length, disjoint=True
            )
        else:
            current_chromosomes = _chromosome_table(
                list(supplied), chrom_offsets or "supplied chromosome offsets",
                current_sequence_length, disjoint=True,
            )
            if chromosomes is None and isinstance(metadata, dict) and "chrom_offsets" in metadata:
                try:
                    embedded = _chromosome_table(
                        metadata["chrom_offsets"], path, current_sequence_length
                    )
                except ValueError:
                    embedded = None
                if embedded != current_chromosomes:
                    print(
                        f"warning: {path} carries chrom_offsets metadata that disagrees "
                        f"with {chrom_offsets}; the supplied offsets file takes precedence",
                        file=sys.stderr,
                    )
        if chromosomes is None:
            chromosomes = current_chromosomes
            sequence_length = current_sequence_length
        elif current_chromosomes != chromosomes:
            raise ValueError("all tree sequences must have identical chromosome offsets")
        elif current_sequence_length != sequence_length:
            raise ValueError("all tree sequences must have the same sequence length")
    return positions, chromosomes or [], float(sequence_length or 0), access_methods


def _add_row_counts(target: np.ndarray, rows: np.ndarray, n_snps: int) -> None:
    counts64 = np.bincount(np.asarray(rows, dtype=np.int64), minlength=n_snps)
    if counts64.max(initial=0) > UINT32_MAX:
        raise OverflowError("per-row count exceeds uint32")
    counts = counts64.astype(np.uint32)
    if np.any(target > UINT32_MAX - counts):
        raise OverflowError("cumulative per-row count exceeds uint32")
    target += counts


def _record_dtype(endpoint_dtype: np.dtype, draw_dtype: np.dtype) -> np.dtype:
    return np.dtype([
        ("row", "<u4"),
        ("below", endpoint_dtype.str),
        ("above", endpoint_dtype.str),
        ("draw_id", draw_dtype.str),
    ], align=False)


def _array_metadata(array: np.ndarray) -> dict[str, object]:
    return {"dtype": array.dtype.name, "shape": [int(v) for v in array.shape]}


def _local_validate_store(path: Path) -> None:
    """Validate builder invariants without depending on the reader module."""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected interval-store schema")
    positions = np.load(path / "positions.npy", mmap_mode="r")
    offsets = np.load(path / "offsets.npy", mmap_mode="r")
    below = np.load(path / "below.npy", mmap_mode="r")
    above = np.load(path / "above.npy", mmap_mode="r")
    draw_id = np.load(path / "draw_id.npy", mmap_mode="r")
    n_snps = positions.size
    if n_snps == 0 or np.any(~np.isfinite(positions)) or np.any(positions != np.floor(positions)):
        raise ValueError("invalid positions")
    if np.any(positions[1:] <= positions[:-1]):
        raise ValueError("positions are not strictly increasing")
    if offsets.shape != (n_snps + 1,) or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("invalid offsets")
    if offsets[-1] != below.size or below.shape != above.shape or below.shape != draw_id.shape:
        raise ValueError("interval array lengths disagree")
    if np.any(~np.isfinite(below)) or np.any(~np.isfinite(above)) or np.any(below < 0) or np.any(above <= below):
        raise ValueError("invalid interval endpoints")
    if draw_id.size and int(draw_id.max()) >= metadata["n_posterior_draws"]:
        raise ValueError("draw ID out of range")
    names = ("present_draw_count", "missing_draw_count", "usable_draw_count", "usable_interval_count", "skipped_root_count")
    counts = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in names}
    if any(array.shape != (n_snps,) or array.dtype != np.uint32 for array in counts.values()):
        raise ValueError("invalid count array")
    if np.any(counts["present_draw_count"] + counts["missing_draw_count"] != metadata["n_posterior_draws"]):
        raise ValueError("present and missing counts disagree")
    if np.any(counts["usable_draw_count"] > counts["present_draw_count"]):
        raise ValueError("usable draw count exceeds present count")
    if np.any(np.diff(offsets).astype(np.uint64) != counts["usable_interval_count"]):
        raise ValueError("offsets disagree with usable interval counts")
    status = np.load(path / "status.npy", mmap_mode="r")
    expected_shape = (metadata["n_posterior_draws"], (n_snps + 3) // 4)
    if status.shape != expected_shape or status.dtype != np.uint8:
        raise ValueError("invalid packed status array")
    if n_snps % 4 and status.size:
        used_bits = 2 * (n_snps % 4)
        if np.any(status[:, -1] >> np.uint8(used_bits)):
            raise ValueError("unused packed status bits are nonzero")


def _external_validate_store(path: Path) -> None:
    """Use the reader's validator when available, tolerating its final name."""
    try:
        import snp_interval_dataset as reader
    except ImportError:
        return
    validator = getattr(reader, "validate_store", None)
    if validator is None:
        validator = getattr(reader, "validate_interval_store", None)
    if validator is not None:
        try:
            validator(path, deep=True)
        except TypeError:
            validator(path)


def build_interval_store(
    tree_files: Sequence[Path], output_dir: Path, *,
    scratch_dir: Path | None = None,
    chrom_offsets: Path | None = None,
    min_usable_fraction: float = 0.1,
    missing: str = "skip",
    root: str = "skip",
    num_buckets: int = 64,
    bucket_memory_bytes: int = 4 * 1024**3,
) -> None:
    """Build and atomically publish an interval store."""
    paths = [Path(path).resolve() for path in tree_files]
    output_dir = Path(output_dir)
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("all tree-sequence inputs must exist")
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise NotADirectoryError(f"output parent does not exist: {output_dir.parent}")
    if missing not in {"skip", "error"} or root not in {"skip", "error"}:
        raise ValueError("missing and root policies must be 'skip' or 'error'")
    if not math.isfinite(min_usable_fraction) or not 0 <= min_usable_fraction <= 1:
        raise ValueError("min_usable_fraction must lie in [0, 1]")
    if num_buckets <= 0 or bucket_memory_bytes <= 0:
        raise ValueError("bucket count and memory bound must be positive")
    if len(paths) > UINT32_MAX:
        raise OverflowError("number of draws exceeds uint32 count capacity")
    # float32 is the only endpoint format. Its worst-case resolution is 4
    # generations, at the oldest age observed in a production store (36.7M
    # generations); against the 1,000-generation analysis bin width that is
    # 0.4% of one bin, and it is sub-generation everywhere the analysis
    # actually sits. float64 doubles the endpoint arrays -- 13.9 GiB to 27.8
    # GiB on the 75-draw store -- to buy precision far below the resolution of
    # the ages themselves. Readers take the dtype from store metadata, so
    # stores written as float64 by earlier versions still load.
    endpoint_dtype = np.dtype("float32")
    draw_dtype = np.dtype("uint8" if len(paths) <= 255 else "uint16")
    if len(paths) > np.iinfo(draw_dtype).max + 1:
        raise OverflowError("number of draws exceeds draw ID capacity")

    scratch_parent = output_dir.parent if scratch_dir is None else Path(scratch_dir)
    if not scratch_parent.is_dir():
        raise NotADirectoryError(f"scratch directory does not exist: {scratch_parent}")
    temp: Path | None = None
    scratch: Path | None = None
    handles: list[object] = []
    try:
        positions, chromosomes, sequence_length, catalog_access = _inspect_inputs(
            paths, chrom_offsets
        )
        if positions.size == 0:
            raise ValueError("input tree sequences contain no sites")
        n_snps = positions.size
        if n_snps > np.iinfo(np.uint32).max:
            raise OverflowError("SNP row count exceeds uint32 scratch format")
        temp = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
        scratch = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.buckets.", dir=scratch_parent))
        np.save(temp / "positions.npy", positions)
        counts = {
            name: np.zeros(n_snps, dtype=np.uint32)
            for name in ("present_draw_count", "usable_draw_count", "usable_interval_count", "skipped_root_count")
        }
        status = np.lib.format.open_memmap(
            temp / "status.npy", mode="w+", dtype=np.uint8,
            shape=(len(paths), (n_snps + 3) // 4),
        )
        record_dtype = _record_dtype(endpoint_dtype, draw_dtype)
        bucket_paths = [scratch / f"bucket-{i:04d}.bin" for i in range(num_buckets)]
        handles = [path.open("ab") for path in bucket_paths]
        bucket_counts = np.zeros(num_buckets, dtype=np.uint64)
        maximum_above = 0.0

        for draw_index, path in enumerate(paths):
            ts = _load(path)
            site_positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
            site_rows = np.searchsorted(positions, site_positions).astype(np.int64, copy=False)
            if (
                np.any(site_rows >= n_snps)
                or np.any(positions[site_rows] != site_positions)
                or (site_rows.size and np.any(site_rows[1:] <= site_rows[:-1]))
            ):
                raise ValueError(f"{path} sites do not map uniquely into the union catalog")
            if missing == "error" and site_rows.size != n_snps:
                absent = np.setdiff1d(np.arange(n_snps), site_rows, assume_unique=True)[0]
                raise ValueError(f"position {positions[absent]:g} is absent from {path}")
            if np.any(counts["present_draw_count"][site_rows] == UINT32_MAX):
                raise OverflowError("present draw count exceeds uint32")
            counts["present_draw_count"][site_rows] += np.uint32(1)

            mutation_site = np.asarray(ts.tables.mutations.site, dtype=np.int64)
            mutation_rows = site_rows[mutation_site]
            parent = lookup_mutation_parents(ts)
            covered = parent != tskit.NULL
            root_rows = mutation_rows[~covered]
            usable_rows = mutation_rows[covered]
            _add_row_counts(counts["skipped_root_count"], root_rows, n_snps)
            if root == "error" and root_rows.size:
                raise ValueError(f"a mutation in {path} is above a root node")
            _add_row_counts(counts["usable_interval_count"], usable_rows, n_snps)
            if usable_rows.size:
                if np.any(usable_rows[1:] < usable_rows[:-1]):
                    usable_distinct = np.flatnonzero(
                        np.bincount(usable_rows, minlength=n_snps) > 0
                    )
                else:
                    starts = np.r_[True, usable_rows[1:] != usable_rows[:-1]]
                    usable_distinct = usable_rows[starts]
                if np.any(counts["usable_draw_count"][usable_distinct] == UINT32_MAX):
                    raise OverflowError("usable draw count exceeds uint32")
                counts["usable_draw_count"][usable_distinct] += np.uint32(1)
            else:
                usable_distinct = np.empty(0, dtype=np.int64)

            logical_status = np.zeros(n_snps, dtype=np.uint8)
            logical_status[site_rows] = STATUS_PRESENT_NO_INTERVAL
            logical_status[usable_distinct] = STATUS_USABLE
            status[draw_index] = pack_status_row(logical_status)

            if usable_rows.size:
                node_time = np.asarray(ts.tables.nodes.time, dtype=np.float64)
                mutation_node = np.asarray(ts.tables.mutations.node, dtype=np.int64)[covered]
                below64 = node_time[mutation_node]
                above64 = node_time[parent[covered]]
                if (
                    np.any(~np.isfinite(below64)) or np.any(~np.isfinite(above64))
                    or np.any(below64 < 0) or np.any(above64 <= below64)
                ):
                    raise ValueError(f"{path} contains an invalid mutation-age interval")
                maximum_above = max(maximum_above, float(np.max(above64, initial=0.0)))
                bucket_ids = np.minimum(
                    (usable_rows.astype(np.uint64) * num_buckets) // n_snps,
                    num_buckets - 1,
                ).astype(np.int64)
                for bucket in np.flatnonzero(np.bincount(bucket_ids, minlength=num_buckets)):
                    selected = bucket_ids == bucket
                    records = np.empty(int(np.sum(selected)), dtype=record_dtype)
                    records["row"] = usable_rows[selected]
                    records["below"] = below64[selected]
                    records["above"] = above64[selected]
                    records["draw_id"] = draw_index
                    records.tofile(handles[int(bucket)])
                    bucket_counts[int(bucket)] += records.size

        for handle in handles:
            handle.close()
        handles = []
        status.flush()
        del status
        missing_count = np.asarray(len(paths) - counts["present_draw_count"], dtype=np.uint32)
        counts["missing_draw_count"] = missing_count
        for name, values in counts.items():
            np.save(temp / f"{name}.npy", values)

        offsets = np.empty(n_snps + 1, dtype=np.uint64)
        offsets[0] = 0
        np.cumsum(counts["usable_interval_count"], dtype=np.uint64, out=offsets[1:])
        np.save(temp / "offsets.npy", offsets)
        n_intervals = int(offsets[-1])
        below = np.lib.format.open_memmap(temp / "below.npy", mode="w+", dtype=endpoint_dtype, shape=(n_intervals,))
        above = np.lib.format.open_memmap(temp / "above.npy", mode="w+", dtype=endpoint_dtype, shape=(n_intervals,))
        draw_id = np.lib.format.open_memmap(temp / "draw_id.npy", mode="w+", dtype=draw_dtype, shape=(n_intervals,))

        for bucket, path in enumerate(bucket_paths):
            scratch_bytes = int(bucket_counts[bucket]) * record_dtype.itemsize
            if scratch_bytes * 3 > bucket_memory_bytes:
                raise MemoryError(
                    f"bucket {bucket} needs an estimated {scratch_bytes * 3} bytes "
                    f"including sort workspace, over limit {bucket_memory_bytes}"
                )
            records = np.fromfile(path, dtype=record_dtype)
            if records.size != bucket_counts[bucket]:
                raise ValueError(f"bucket {bucket} record count changed on disk")
            order = np.argsort(records["row"], kind="stable")
            records = records[order]
            # Rows are assigned with floor(row * B / N). Integer ceiling gives
            # the exact inverse range without allocating an N-element array.
            row_start = min((bucket * n_snps + num_buckets - 1) // num_buckets, n_snps)
            row_stop = min(((bucket + 1) * n_snps + num_buckets - 1) // num_buckets, n_snps)
            if records.size and (
                int(records["row"][0]) < row_start or int(records["row"][-1]) >= row_stop
            ):
                raise ValueError(f"bucket {bucket} contains a row outside its range")
            observed = np.bincount(
                records["row"].astype(np.int64) - row_start,
                minlength=row_stop - row_start,
            )
            expected = np.diff(offsets[row_start:row_stop + 1])
            if not np.array_equal(observed.astype(np.uint64), expected):
                raise ValueError(f"bucket {bucket} disagrees with offsets")
            destination = slice(int(offsets[row_start]), int(offsets[row_stop]))
            if destination.stop - destination.start != records.size:
                raise ValueError(f"bucket {bucket} does not fill its destination")
            below[destination] = records["below"]
            above[destination] = records["above"]
            draw_id[destination] = records["draw_id"]
            below.flush(); above.flush(); draw_id.flush()
            path.unlink()
        del below, above, draw_id

        required_usable_draws = int(math.ceil(min_usable_fraction * len(paths)))
        arrays = {
            "positions": {"dtype": "float64", "shape": [int(n_snps)]},
            "offsets": {"dtype": "uint64", "shape": [int(n_snps + 1)]},
            "below": {"dtype": endpoint_dtype.name, "shape": [n_intervals]},
            "above": {"dtype": endpoint_dtype.name, "shape": [n_intervals]},
            "draw_id": {"dtype": draw_dtype.name, "shape": [n_intervals]},
            "status": {"dtype": "uint8", "shape": [len(paths), int((n_snps + 3) // 4)]},
        }
        arrays.update({name: _array_metadata(values) for name, values in counts.items()})
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "n_snps": int(n_snps),
            "n_intervals": n_intervals,
            "n_posterior_draws": len(paths),
            "catalog_sha256": hashlib.sha256(
                memoryview(np.ascontiguousarray(positions)).cast("B")
            ).hexdigest(),
            "sequence_length": sequence_length,
            "maximum_above": maximum_above,
            "maximum_parent_time": maximum_above,
            "endpoint_dtype": endpoint_dtype.name,
            "interval_dtype": endpoint_dtype.name,
            "draw_id_dtype": draw_dtype.name,
            "status_encoding": "draw-major, four two-bit values per uint8; 0=absent, 1=present-no-usable-interval, 2=usable",
            "position_coordinate_system": "one-based within chromosome; global=offset+POS",
            "position_matching": "exact integral float64 equality",
            "catalog_access_methods": catalog_access,
            "chromosomes": chromosomes,
            "chromosomes_source": "arg_metadata" if chrom_offsets is None else str(chrom_offsets),
            "missing_policy": missing,
            "root_policy": root,
            "minimum_usable_fraction": min_usable_fraction,
            "minimum_usable_draws": required_usable_draws,
            "interval_weighting": "equal per usable mutation interval",
            "creation_command": " ".join(sys.argv),
            "inputs": [{"draw_id": i, "path": str(path)} for i, path in enumerate(paths)],
            "extraction_totals": {name: int(values.sum(dtype=np.uint64)) for name, values in counts.items()},
            "bucket_counts": [int(value) for value in bucket_counts],
            "record_dtype": record_dtype.descr,
            "arrays": arrays,
        }
        metadata["content_sha256"] = compute_interval_store_content_sha256(
            temp, metadata
        )
        (temp / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _local_validate_store(temp)
        _external_validate_store(temp)
        os.replace(temp, output_dir)
        temp = None
        shutil.rmtree(scratch)
        scratch = None
    except BaseException:
        for handle in handles:
            handle.close()
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        raise


def _expand(patterns: Sequence[str]) -> list[Path]:
    result: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"tree-sequence pattern matched no files: {pattern}")
        result.extend(Path(match) for match in matches)
    return list(dict.fromkeys(result))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", nargs="+", help="tree-sequence files or glob patterns")
    parser.add_argument("--interval-store", required=True, type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--chrom-offsets", type=Path)
    parser.add_argument("--min-usable-fraction", type=float, default=0.1)
    parser.add_argument("--missing", choices=("skip", "error"), default="skip")
    parser.add_argument("--root", choices=("skip", "error"), default="skip")
    parser.add_argument("--num-buckets", type=int, default=64)
    parser.add_argument("--bucket-memory-gb", type=float, default=4.0)
    args = parser.parse_args(argv)
    try:
        build_interval_store(
            _expand(args.trees), args.interval_store,
            scratch_dir=args.scratch_dir, chrom_offsets=args.chrom_offsets,
            min_usable_fraction=args.min_usable_fraction,
            missing=args.missing, root=args.root,
            num_buckets=args.num_buckets,
            bucket_memory_bytes=int(args.bucket_memory_gb * 1024**3),
        )
    except (OSError, ValueError, tskit.FileFormatError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
