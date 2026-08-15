"""Read-only access to compact SNP posterior interval stores.

The format stores a ragged list of mutation-age intervals for each global SNP
row.  Array files are NumPy ``.npy`` files so the potentially large endpoint
columns can be memory mapped without copying them into RAM.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


INTERVAL_SCHEMA_VERSION = "snp-age-interval-v1"
STATUS_ABSENT = np.uint8(0)
STATUS_PRESENT_UNUSABLE = np.uint8(1)
STATUS_PRESENT_USABLE = np.uint8(2)


_CONTENT_IDENTITY_METADATA_KEYS = (
    "schema_version",
    "n_posterior_draws",
    "chromosomes",
    "interval_weighting",
    "missing_policy",
    "root_policy",
    "minimum_usable_fraction",
    "minimum_usable_draws",
)


def compute_interval_store_content_sha256(
    store_dir: str | Path,
    metadata: dict | None = None,
    *,
    chunk_bytes: int = 8 * 1024**2,
) -> str:
    """Hash every declared array plus metadata that affects interpretation."""
    path = Path(store_dir)
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if metadata is None:
        metadata = _read_metadata(path)
    arrays = metadata.get("arrays")
    if not isinstance(arrays, dict) or not arrays:
        raise ValueError("cannot hash an interval store without declared arrays")
    semantic = {
        key: metadata.get(key) for key in _CONTENT_IDENTITY_METADATA_KEYS
    }
    digest = hashlib.sha256()
    digest.update(b"normalizeTE-interval-store-content-v1\0")
    digest.update(json.dumps(
        semantic, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))
    for name in sorted(arrays):
        encoded = name.encode("utf-8")
        array_path = path / f"{name}.npy"
        size = array_path.stat().st_size
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        with array_path.open("rb") as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class IntervalStoreReport:
    n_snps: int
    n_intervals: int
    n_posterior_draws: int
    endpoint_dtype: str


@dataclass(frozen=True)
class IntervalBatch:
    """A self-contained ragged interval selection.

    Intervals for selected row ``i`` occupy
    ``offsets[i]:offsets[i + 1]``.  Repeated requested rows are retained.
    """

    rows: np.ndarray
    offsets: np.ndarray
    below: np.ndarray
    above: np.ndarray
    draw_id: np.ndarray


class CandidateIntervalCache:
    """Compact scratch cache containing intervals for selected source rows."""

    def __init__(self, path: Path, metadata: dict, arrays: dict[str, np.ndarray]):
        self.path = path
        self.metadata = metadata
        self.source_rows = arrays["source_rows"]
        self.offsets = arrays["offsets"]
        self._below = arrays["below"]
        self._above = arrays["above"]
        self._draw_id = arrays["draw_id"]

    @classmethod
    def open(cls, path: str | Path) -> "CandidateIntervalCache":
        cache = Path(path)
        try:
            metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid candidate interval cache: {cache}") from error
        if metadata.get("schema_version") != "snp-age-candidate-cache-v1":
            raise ValueError("unsupported candidate cache schema")
        names = ("source_rows", "offsets", "below", "above", "draw_id")
        try:
            arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read candidate cache arrays in {cache}") from error
        n_rows = int(metadata["n_rows"])
        n_intervals = int(metadata["n_intervals"])
        endpoint_dtype = np.dtype(metadata["endpoint_dtype"])
        draw_dtype = np.dtype(metadata["draw_id_dtype"])
        expected = {
            "source_rows": (np.dtype("int64"), (n_rows,)),
            "offsets": (np.dtype("uint64"), (n_rows + 1,)),
            "below": (endpoint_dtype, (n_intervals,)),
            "above": (endpoint_dtype, (n_intervals,)),
            "draw_id": (draw_dtype, (n_intervals,)),
        }
        for name, (dtype, shape) in expected.items():
            if arrays[name].dtype != dtype or arrays[name].shape != shape:
                raise ValueError(f"candidate cache {name} has the wrong dtype or shape")
        rows = arrays["source_rows"]
        offsets = arrays["offsets"]
        if np.any(rows[1:] <= rows[:-1]):
            raise ValueError("candidate cache source rows are not strictly increasing")
        if int(offsets[0]) != 0 or int(offsets[-1]) != n_intervals or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("candidate cache offsets are invalid")
        return cls(cache, metadata, arrays)

    def intervals(self, source_rows: np.ndarray) -> IntervalBatch:
        rows = np.asarray(source_rows)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("source rows must be a one-dimensional integer array")
        rows = rows.astype(np.int64, copy=False)
        if np.unique(rows).size != rows.size:
            raise ValueError("source rows contain duplicates")
        local = np.searchsorted(self.source_rows, rows)
        found = local < self.source_rows.size
        valid = np.flatnonzero(found)
        found[valid] = self.source_rows[local[valid]] == rows[valid]
        if not np.all(found):
            raise KeyError(f"source rows not present in candidate cache: {_format_values(rows[~found])}")
        return _copy_ragged_rows(rows, local, self.offsets, self._below, self._above, self._draw_id)

    def cdf_at(
        self, source_rows: np.ndarray, points: np.ndarray, *, side: str = "right",
        weighting: str = "interval",
    ) -> np.ndarray:
        return _batch_cdf(self.intervals(source_rows), points, side=side, weighting=weighting)


def pack_status(status: np.ndarray) -> np.ndarray:
    """Pack logical two-bit statuses along the final axis."""
    values = np.asarray(status)
    if values.ndim < 1:
        raise ValueError("status must have at least one dimension")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("status values must be integers")
    if np.any(values < 0) or np.any(values > 3):
        raise ValueError("status values must lie in 0..3")
    values = values.astype(np.uint8, copy=False)
    packed = np.zeros(values.shape[:-1] + ((values.shape[-1] + 3) // 4,), dtype=np.uint8)
    for slot in range(4):
        part = values[..., slot::4]
        packed[..., :part.shape[-1]] |= part << np.uint8(2 * slot)
    return packed


def unpack_status(packed: np.ndarray, n_snps: int) -> np.ndarray:
    """Unpack the final axis of a two-bit status array."""
    values = np.asarray(packed)
    if values.ndim < 1 or values.dtype != np.uint8:
        raise ValueError("packed status must be a uint8 array with at least one dimension")
    if not isinstance(n_snps, (int, np.integer)) or n_snps < 0:
        raise ValueError("n_snps must be a nonnegative integer")
    if values.shape[-1] != (int(n_snps) + 3) // 4:
        raise ValueError("packed status width does not match n_snps")
    output = np.empty(values.shape[:-1] + (int(n_snps),), dtype=np.uint8)
    for slot in range(4):
        output[..., slot::4] = (
            values[..., :output[..., slot::4].shape[-1]] >> np.uint8(2 * slot)
        ) & np.uint8(3)
    return output


def interval_cdf(
    below: np.ndarray, above: np.ndarray, points: np.ndarray, *, side: Literal["left", "right"] = "right"
) -> np.ndarray:
    """Evaluate individual uniform-interval CDFs, including fixture point masses.

    The result has shape ``broadcast(below, above).shape + points.shape``.
    Zero-width intervals are supported here solely to reproduce legacy
    synthetic fixtures; production stores reject them.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    lower, upper = np.broadcast_arrays(np.asarray(below), np.asarray(above))
    query = np.asarray(points)
    if query.ndim != 1:
        raise ValueError("points must be one-dimensional")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(upper < lower):
        raise ValueError("interval endpoints must be finite and above must be >= below")
    if np.any(~np.isfinite(query)):
        raise ValueError("points must be finite")
    shape = lower.shape + (1,)
    lo = lower.reshape(shape)
    hi = upper.reshape(shape)
    t = query.reshape((1,) * lower.ndim + (query.size,))
    width = hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.clip((t - lo) / width, 0.0, 1.0)
    point = width == 0
    if np.any(point):
        point_value = t >= lo if side == "right" else t > lo
        result = np.where(point, point_value, result)
    return np.asarray(result, dtype=np.float64)


class SNPAgeIntervalDataset:
    """Memory-mapped, read-only view of an interval store."""

    def __init__(self, store_dir: Path, metadata: dict, arrays: dict[str, np.ndarray]):
        self.store_dir = store_dir
        self.metadata = metadata
        self.positions = arrays["positions"]
        self.offsets = arrays["offsets"]
        self._below = arrays["below"]
        self._above = arrays["above"]
        self._draw_id = arrays["draw_id"]
        self._status = arrays["status"]
        self.present_draw_count = arrays["present_draw_count"]
        self.missing_draw_count = arrays["missing_draw_count"]
        self.usable_draw_count = arrays["usable_draw_count"]
        self.usable_interval_count = arrays["usable_interval_count"]
        self.skipped_root_count = arrays["skipped_root_count"]
        self.n_posterior_draws = int(metadata["n_posterior_draws"])
        self.minimum_usable_draws = int(metadata.get("minimum_usable_draws", 1))
        self.chromosomes = tuple(metadata["chromosomes"])
        self._chromosome_by_name = {str(row["chrom"]): row for row in self.chromosomes}

    @classmethod
    def open(cls, store_dir: str | Path, *, deep: bool = False) -> "SNPAgeIntervalDataset":
        path = Path(store_dir)
        validate_interval_store(path, deep=deep)
        metadata = _read_metadata(path)
        names = (
            "positions", "offsets", "below", "above", "draw_id", "status",
            "present_draw_count", "missing_draw_count", "usable_draw_count",
            "usable_interval_count", "skipped_root_count",
        )
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in names}
        return cls(path, metadata, arrays)

    @property
    def valid(self) -> np.ndarray:
        return self.usable_interval_count > 0

    @property
    def eligible(self) -> np.ndarray:
        return self.valid & (self.usable_draw_count >= self.minimum_usable_draws)

    @property
    def n_intervals(self) -> int:
        return int(self.offsets[-1])

    def resolve_positions(self, query_positions: np.ndarray) -> np.ndarray:
        query = np.asarray(query_positions, dtype=np.float64)
        if query.ndim != 1 or np.any(~np.isfinite(query)):
            raise ValueError("query positions must be a one-dimensional finite array")
        unique, counts = np.unique(query, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(f"duplicate query positions: {_format_values(unique[counts > 1])}")
        indices = np.searchsorted(self.positions, query)
        found = indices < self.positions.size
        in_range = np.flatnonzero(found)
        found[in_range] = self.positions[indices[in_range]] == query[in_range]
        if not np.all(found):
            raise KeyError(f"positions not found: {_format_values(query[~found])}")
        return indices.astype(np.int64, copy=False)

    def native_to_global(self, chromosomes: np.ndarray, vcf_positions: np.ndarray) -> np.ndarray:
        chroms = np.asarray(chromosomes, dtype=str)
        positions = np.asarray(vcf_positions)
        if chroms.ndim != 1 or positions.ndim != 1 or chroms.shape != positions.shape:
            raise ValueError("chromosomes and VCF positions must be aligned 1-D arrays")
        numeric = positions.astype(np.float64)
        if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise ValueError("VCF positions must be finite integers")
        native = numeric.astype(np.int64)
        output = np.empty(native.size, dtype=np.float64)
        for i, (chrom, position) in enumerate(zip(chroms, native)):
            entry = self._chromosome_by_name.get(str(chrom))
            if entry is None:
                raise KeyError(f"chromosome not found in store metadata: {chrom}")
            if position < 1 or position > int(entry["length"]):
                raise ValueError(
                    f"VCF position {position} lies outside {chrom} length {entry['length']}"
                )
            output[i] = int(entry["offset"]) + int(position)
        return output

    def resolve_native_positions(self, chromosomes: np.ndarray, vcf_positions: np.ndarray) -> np.ndarray:
        return self.resolve_positions(self.native_to_global(chromosomes, vcf_positions))

    def rows_to_native(self, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = _checked_indices(row_indices, self.positions.size, "row")
        globals_ = np.asarray(self.positions[rows], dtype=np.float64)
        ordered = sorted(self.chromosomes, key=lambda item: int(item["offset"]))
        starts = np.asarray([int(item["offset"]) + 1 for item in ordered], dtype=np.float64)
        choices = np.searchsorted(starts, globals_, side="right") - 1
        if np.any(choices < 0):
            raise ValueError("store position precedes the first chromosome")
        width = max((len(str(item["chrom"])) for item in ordered), default=1)
        names = np.empty(rows.size, dtype=f"U{width}")
        native = np.empty(rows.size, dtype=np.int64)
        for i, choice in enumerate(choices):
            entry = ordered[int(choice)]
            value = globals_[i] - int(entry["offset"])
            if value != math.floor(value) or not 1 <= value <= int(entry["length"]):
                raise ValueError(f"store position {globals_[i]:g} is not a native coordinate")
            names[i] = str(entry["chrom"])
            native[i] = int(value)
        return names, native

    def intervals(self, row_indices: np.ndarray) -> IntervalBatch:
        rows = _checked_indices(row_indices, self.positions.size, "row")
        return _copy_ragged_rows(
            rows, rows, self.offsets, self._below, self._above, self._draw_id
        )

    def mean_ages(self, row_indices: np.ndarray, *, weighting: str = "interval") -> np.ndarray:
        batch = self.intervals(row_indices)
        output = np.full(batch.rows.size, np.nan, dtype=np.float64)
        for i in range(batch.rows.size):
            start, stop = map(int, batch.offsets[i:i + 2])
            if start != stop:
                midpoint = (batch.below[start:stop].astype(np.float64) + batch.above[start:stop]) / 2
                output[i] = _weighted_reduce(midpoint, batch.draw_id[start:stop], weighting)
        return output

    def cdf_at(
        self, row_indices: np.ndarray, points: np.ndarray, *, side: str = "right",
        weighting: str = "interval",
    ) -> np.ndarray:
        query = np.asarray(points)
        if query.ndim != 1 or np.any(~np.isfinite(query)):
            raise ValueError("points must be a one-dimensional finite array")
        if weighting not in {"interval", "draw"}:
            raise ValueError("weighting must be 'interval' or 'draw'")
        return _batch_cdf(self.intervals(row_indices), query, side=side, weighting=weighting)

    def boundary_cdfs(
        self,
        row_indices: np.ndarray,
        boundaries: np.ndarray,
        *,
        access_strategy: str = "gather",
        block_rows: int = 100_000,
        coalesce_gap: int = 64,
        cache: CandidateIntervalCache | str | Path | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Evaluate boundary CDFs using an explicit row-access strategy.

        ``scan`` intentionally reads complete contiguous row blocks, while
        ``coalesced`` fills short gaps between requested rows. ``cache`` reads
        a compact scratch cache built by :meth:`build_candidate_cache`.
        """
        rows = _checked_indices(row_indices, self.positions.size, "row")
        if rows.size == 0:
            return np.empty((0, np.asarray(boundaries).size), dtype=np.float64)
        if np.unique(rows).size != rows.size:
            raise ValueError("row indices contain duplicates")
        if access_strategy not in {"gather", "coalesced", "scan", "cache"}:
            raise ValueError("unknown interval access strategy")
        if block_rows <= 0 or coalesce_gap < 0:
            raise ValueError("block_rows must be positive and coalesce_gap nonnegative")
        if access_strategy == "cache":
            if cache is None:
                raise ValueError("cache strategy requires a candidate cache")
            candidate_cache = cache if isinstance(cache, CandidateIntervalCache) else CandidateIntervalCache.open(cache)
            expected_store = str(Path(self.store_dir).resolve())
            if candidate_cache.metadata.get("source_store") != expected_store:
                raise ValueError("candidate cache belongs to a different interval store")
            if (
                int(candidate_cache.metadata.get("n_source_snps", -1)) != self.positions.size
                or int(candidate_cache.metadata.get("n_source_intervals", -1)) != self.n_intervals
            ):
                raise ValueError("candidate cache source dimensions are stale")
            return candidate_cache.cdf_at(rows, boundaries, **kwargs)
        if access_strategy == "gather":
            return self.cdf_at(rows, boundaries, **kwargs)

        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        sorted_output = np.empty(
            (rows.size, np.asarray(boundaries).size), dtype=np.float64
        )
        if access_strategy == "coalesced":
            start = 0
            while start < sorted_rows.size:
                stop = start + 1
                while (
                    stop < sorted_rows.size
                    and sorted_rows[stop] - sorted_rows[stop - 1] <= coalesce_gap + 1
                    and sorted_rows[stop] - sorted_rows[start] < block_rows
                ):
                    stop += 1
                slab_start = int(sorted_rows[start])
                slab_stop = int(sorted_rows[stop - 1]) + 1
                interval_start = int(self.offsets[slab_start])
                interval_stop = int(self.offsets[slab_stop])
                slab_below = np.array(self._below[interval_start:interval_stop], copy=True)
                slab_above = np.array(self._above[interval_start:interval_stop], copy=True)
                slab_draw = np.array(self._draw_id[interval_start:interval_stop], copy=True)
                local_offsets = np.asarray(
                    self.offsets[slab_start:slab_stop + 1] - interval_start,
                    dtype=np.uint64,
                )
                selected = sorted_rows[start:stop]
                batch = _copy_ragged_rows(
                    selected,
                    selected - slab_start,
                    local_offsets,
                    slab_below,
                    slab_above,
                    slab_draw,
                )
                sorted_output[start:stop] = _batch_cdf(
                    batch, boundaries, side=kwargs.get("side", "right"),
                    weighting=kwargs.get("weighting", "interval"),
                )
                start = stop
        else:
            cursor = 0
            for slab_start in range(0, self.positions.size, block_rows):
                slab_stop = min(slab_start + block_rows, self.positions.size)
                selected_stop = np.searchsorted(sorted_rows, slab_stop, side="left")
                interval_start = int(self.offsets[slab_start])
                interval_stop = int(self.offsets[slab_stop])
                # Force a complete sequential endpoint scan. Candidate CDFs
                # are then evaluated from the in-memory slab only.
                slab_below = np.array(self._below[interval_start:interval_stop], copy=True)
                slab_above = np.array(self._above[interval_start:interval_stop], copy=True)
                slab_draw = np.array(self._draw_id[interval_start:interval_stop], copy=True)
                if selected_stop > cursor:
                    selected = sorted_rows[cursor:selected_stop]
                    local_offsets = np.asarray(
                        self.offsets[slab_start:slab_stop + 1] - interval_start,
                        dtype=np.uint64,
                    )
                    batch = _copy_ragged_rows(
                        selected,
                        selected - slab_start,
                        local_offsets,
                        slab_below,
                        slab_above,
                        slab_draw,
                    )
                    sorted_output[cursor:selected_stop] = _batch_cdf(
                        batch, boundaries, side=kwargs.get("side", "right"),
                        weighting=kwargs.get("weighting", "interval"),
                    )
                cursor = selected_stop
        output = np.empty_like(sorted_output)
        output[order] = sorted_output
        return output

    def build_candidate_cache(
        self, row_indices: np.ndarray, cache_dir: str | Path, *, block_rows: int = 100_000
    ) -> CandidateIntervalCache:
        """Sequentially scan the store and atomically publish a candidate cache.

        ``cache_dir`` must not already exist. Candidate source rows are stored
        sorted; cache reads restore any requested ordering.
        """
        rows = _checked_indices(row_indices, self.positions.size, "row")
        if rows.size == 0 or np.unique(rows).size != rows.size:
            raise ValueError("candidate rows must be nonempty and unique")
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        rows = np.sort(rows)
        destination = Path(cache_dir)
        if destination.exists():
            raise FileExistsError(f"candidate cache already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            lengths = np.asarray(self.offsets[rows + 1] - self.offsets[rows], dtype=np.uint64)
            cache_offsets = np.empty(rows.size + 1, dtype=np.uint64)
            cache_offsets[0] = 0
            np.cumsum(lengths, out=cache_offsets[1:])
            n_intervals = int(cache_offsets[-1])
            np.save(temporary / "source_rows.npy", rows)
            np.save(temporary / "offsets.npy", cache_offsets)
            below = np.lib.format.open_memmap(
                temporary / "below.npy", mode="w+", dtype=self._below.dtype, shape=(n_intervals,)
            )
            above = np.lib.format.open_memmap(
                temporary / "above.npy", mode="w+", dtype=self._above.dtype, shape=(n_intervals,)
            )
            draw_id = np.lib.format.open_memmap(
                temporary / "draw_id.npy", mode="w+", dtype=self._draw_id.dtype, shape=(n_intervals,)
            )
            write_cursor = 0
            for row_start in range(0, self.positions.size, block_rows):
                row_stop = min(row_start + block_rows, self.positions.size)
                interval_start = int(self.offsets[row_start])
                interval_stop = int(self.offsets[row_stop])
                # Copies intentionally force sequential reads even in blocks
                # containing no candidates; this is the strategy being tested.
                source_below = np.array(self._below[interval_start:interval_stop], copy=True)
                source_above = np.array(self._above[interval_start:interval_stop], copy=True)
                source_draw = np.array(self._draw_id[interval_start:interval_stop], copy=True)
                first = np.searchsorted(rows, row_start, side="left")
                last = np.searchsorted(rows, row_stop, side="left")
                if first == last or interval_start == interval_stop:
                    continue
                selected = rows[first:last]
                local_starts = np.asarray(self.offsets[selected] - interval_start, dtype=np.int64)
                local_stops = np.asarray(self.offsets[selected + 1] - interval_start, dtype=np.int64)
                changes = np.zeros(source_below.size + 1, dtype=np.int32)
                np.add.at(changes, local_starts, 1)
                np.add.at(changes, local_stops, -1)
                mask = np.cumsum(changes[:-1]) > 0
                count = int(mask.sum())
                below[write_cursor:write_cursor + count] = source_below[mask]
                above[write_cursor:write_cursor + count] = source_above[mask]
                draw_id[write_cursor:write_cursor + count] = source_draw[mask]
                write_cursor += count
            if write_cursor != n_intervals:
                raise RuntimeError("candidate cache interval count mismatch")
            below.flush(); above.flush(); draw_id.flush()
            del below, above, draw_id
            metadata = {
                "schema_version": "snp-age-candidate-cache-v1",
                "source_store": str(Path(self.store_dir).resolve()),
                "n_source_snps": int(self.positions.size),
                "n_source_intervals": self.n_intervals,
                "n_rows": int(rows.size),
                "n_intervals": n_intervals,
                "endpoint_dtype": self._below.dtype.name,
                "draw_id_dtype": self._draw_id.dtype.name,
                "build_strategy": "full-sequential-scan",
                "block_rows": int(block_rows),
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            CandidateIntervalCache.open(temporary)
            os.replace(temporary, destination)
            return CandidateIntervalCache.open(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def aggregate_cdf_at(
        self,
        row_indices: np.ndarray,
        points: np.ndarray,
        *,
        side: str = "right",
        weighting: str = "interval",
        block_rows: int = 512,
    ) -> np.ndarray:
        """Return the mean SNP CDF without retaining a SNP-by-grid matrix."""
        rows = _checked_indices(row_indices, self.positions.size, "row")
        if rows.size == 0 or block_rows <= 0:
            raise ValueError("rows must be nonempty and block_rows positive")
        query = np.asarray(points, dtype=np.float64)
        if query.ndim != 1 or np.any(~np.isfinite(query)):
            raise ValueError("points must be a one-dimensional finite array")
        widths = np.diff(query)
        if (
            weighting == "interval" and query.size >= 2
            and np.all(widths > 0)
            and np.allclose(widths, widths[0], rtol=0, atol=0)
        ):
            return self._aggregate_uniform_interval_cdf(rows, query)
        total = np.zeros(query.size, dtype=np.float64)
        for start in range(0, rows.size, block_rows):
            block = self.cdf_at(
                rows[start:start + block_rows], query, side=side, weighting=weighting
            )
            if np.any(~np.isfinite(block)):
                raise ValueError("aggregate rows include a SNP without usable intervals")
            total += block.sum(axis=0, dtype=np.float64)
        return total / rows.size

    def write_regular_grid_cdfs(
        self,
        row_indices: np.ndarray,
        points: np.ndarray,
        output_path: str | Path,
        *,
        block_rows: int = 512,
        dtype: str | np.dtype = "float32",
    ) -> np.memmap:
        """Write SNP CDF rows on a uniform grid in O(intervals + output cells).

        Three difference arrays encode the linear and complete portions of
        every positive-width interval. This avoids constructing an
        ``intervals x grid`` temporary for each SNP. The output is an NPY
        memmap and the destination must not already exist.
        """
        rows = _checked_indices(row_indices, self.positions.size, "row")
        query = np.asarray(points, dtype=np.float64)
        output_dtype = np.dtype(dtype)
        if rows.size == 0 or block_rows <= 0:
            raise ValueError("rows must be nonempty and block_rows positive")
        if query.ndim != 1 or query.size < 2 or np.any(~np.isfinite(query)):
            raise ValueError("points must be a finite vector with at least two values")
        widths = np.diff(query)
        if not np.all(widths > 0) or not np.allclose(widths, widths[0], rtol=0, atol=0):
            raise ValueError("points must be a strictly increasing uniform grid")
        if output_dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("CDF output dtype must be float32 or float64")
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(f"CDF output already exists: {destination}")
        output = np.lib.format.open_memmap(
            destination, mode="w+", dtype=output_dtype,
            shape=(rows.size, query.size),
        )
        origin = float(query[0])
        step = float(widths[0])
        columns = np.arange(query.size, dtype=np.float64)
        for start in range(0, rows.size, block_rows):
            stop = min(start + block_rows, rows.size)
            block = self.intervals(rows[start:stop])
            counts = np.diff(block.offsets).astype(np.int64, copy=False)
            if np.any(counts == 0):
                raise ValueError("CDF rows include a SNP without usable intervals")
            owners = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
            lower = np.asarray(block.below, dtype=np.float64)
            upper = np.asarray(block.above, dtype=np.float64)
            interval_width = upper - lower
            if np.any(interval_width <= 0):
                raise ValueError("CDF rows include a nonpositive interval")
            per_interval = 1.0 / counts[owners]
            linear_start = np.clip(
                np.ceil((lower - origin) / step).astype(np.int64), 0, query.size
            )
            full_start = np.clip(
                np.ceil((upper - origin) / step).astype(np.int64), 0, query.size
            )
            shape = (counts.size, query.size + 1)
            slope = np.zeros(shape, dtype=np.float64)
            intercept = np.zeros(shape, dtype=np.float64)
            complete = np.zeros(shape, dtype=np.float64)
            active = linear_start < full_start
            inverse_width = per_interval / interval_width
            slope_values = step * inverse_width
            intercept_values = (origin - lower) * inverse_width
            np.add.at(
                slope, (owners[active], linear_start[active]), slope_values[active]
            )
            np.add.at(
                slope, (owners[active], full_start[active]), -slope_values[active]
            )
            np.add.at(
                intercept,
                (owners[active], linear_start[active]),
                intercept_values[active],
            )
            np.add.at(
                intercept,
                (owners[active], full_start[active]),
                -intercept_values[active],
            )
            begins_full = full_start < query.size
            np.add.at(
                complete,
                (owners[begins_full], full_start[begins_full]),
                per_interval[begins_full],
            )
            np.cumsum(slope, axis=1, out=slope)
            slope[:, :-1] *= columns
            np.cumsum(intercept, axis=1, out=intercept)
            slope += intercept
            del intercept
            np.cumsum(complete, axis=1, out=complete)
            slope += complete
            del complete
            output[start:stop] = np.clip(slope[:, :-1], 0.0, 1.0).astype(output_dtype)
        output.flush()
        return output

    def _aggregate_uniform_interval_cdf(
        self, rows: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """Aggregate positive-width uniform intervals on a regular grid in O(I+B)."""
        batch = self.intervals(rows)
        counts = np.diff(batch.offsets).astype(np.int64, copy=False)
        if np.any(counts == 0):
            raise ValueError("aggregate rows include a SNP without usable intervals")
        per_interval = np.repeat(1.0 / (rows.size * counts), counts)
        lower = np.asarray(batch.below, dtype=np.float64)
        upper = np.asarray(batch.above, dtype=np.float64)
        if np.any(upper <= lower):
            raise ValueError("aggregate rows include a nonpositive interval")

        n_points = points.size
        origin = float(points[0])
        step = float(points[1] - points[0])
        linear_start = np.clip(
            np.ceil((lower - origin) / step).astype(np.int64), 0, n_points)
        full_start = np.clip(
            np.ceil((upper - origin) / step).astype(np.int64), 0, n_points)
        active = linear_start < full_start

        slope_delta = np.zeros(n_points + 1, dtype=np.float64)
        intercept_delta = np.zeros(n_points + 1, dtype=np.float64)
        full_delta = np.zeros(n_points + 1, dtype=np.float64)
        inverse_width = per_interval / (upper - lower)
        slopes = step * inverse_width
        intercepts = (origin - lower) * inverse_width
        np.add.at(slope_delta, linear_start[active], slopes[active])
        np.add.at(slope_delta, full_start[active], -slopes[active])
        np.add.at(intercept_delta, linear_start[active], intercepts[active])
        np.add.at(intercept_delta, full_start[active], -intercepts[active])
        begins_full = full_start < n_points
        np.add.at(full_delta, full_start[begins_full], per_interval[begins_full])
        result = (
            np.cumsum(slope_delta[:-1]) * np.arange(n_points)
            + np.cumsum(intercept_delta[:-1])
            + np.cumsum(full_delta[:-1])
        )
        return np.clip(result, 0.0, 1.0)

    def cell_masses(
        self, row_indices: np.ndarray, edges: np.ndarray, *, weighting: str = "interval"
    ) -> np.ndarray:
        values = np.asarray(edges)
        if values.ndim != 1 or values.size < 2 or np.any(~np.isfinite(values)):
            raise ValueError("edges must be a one-dimensional finite array with at least two values")
        if np.any(values[1:] <= values[:-1]):
            raise ValueError("edges must be strictly increasing")
        left = self.cdf_at(row_indices, values, side="left", weighting=weighting)
        return left[:, 1:] - left[:, :-1]

    def read_status(
        self, *, draws: np.ndarray | None = None, rows: np.ndarray | None = None
    ) -> np.ndarray:
        draw_rows = (
            np.arange(self.n_posterior_draws, dtype=np.int64)
            if draws is None else _checked_indices(draws, self.n_posterior_draws, "draw")
        )
        snp_rows = (
            np.arange(self.positions.size, dtype=np.int64)
            if rows is None else _checked_indices(rows, self.positions.size, "row")
        )
        return _decode_selected_status(np.asarray(self._status[draw_rows]), snp_rows)


def _copy_ragged_rows(
    output_rows: np.ndarray,
    local_rows: np.ndarray,
    offsets: np.ndarray,
    below_source: np.ndarray,
    above_source: np.ndarray,
    draw_source: np.ndarray,
) -> IntervalBatch:
    lengths = np.asarray(offsets[local_rows + 1] - offsets[local_rows], dtype=np.uint64)
    local_offsets = np.empty(local_rows.size + 1, dtype=np.uint64)
    local_offsets[0] = 0
    np.cumsum(lengths, out=local_offsets[1:])
    total = int(local_offsets[-1])
    below = np.empty(total, dtype=below_source.dtype)
    above = np.empty(total, dtype=above_source.dtype)
    draw_id = np.empty(total, dtype=draw_source.dtype)
    cursor = 0
    for row, length in zip(local_rows, lengths):
        count = int(length)
        start = int(offsets[row])
        stop = start + count
        below[cursor:cursor + count] = below_source[start:stop]
        above[cursor:cursor + count] = above_source[start:stop]
        draw_id[cursor:cursor + count] = draw_source[start:stop]
        cursor += count
    return IntervalBatch(output_rows.copy(), local_offsets, below, above, draw_id)


def _batch_cdf(
    batch: IntervalBatch, points: np.ndarray, *, side: str, weighting: str
) -> np.ndarray:
    query = np.asarray(points)
    if query.ndim != 1 or np.any(~np.isfinite(query)):
        raise ValueError("points must be a one-dimensional finite array")
    if weighting not in {"interval", "draw"}:
        raise ValueError("weighting must be 'interval' or 'draw'")
    output = np.full((batch.rows.size, query.size), np.nan, dtype=np.float64)
    for i in range(batch.rows.size):
        start, stop = map(int, batch.offsets[i:i + 2])
        if start == stop:
            continue
        values = interval_cdf(
            batch.below[start:stop], batch.above[start:stop], query, side=side
        )
        if weighting == "interval":
            output[i] = values.mean(axis=0)
        else:
            draws = batch.draw_id[start:stop]
            output[i] = np.mean(
                [values[draws == draw].mean(axis=0) for draw in np.unique(draws)], axis=0
            )
    return output


def _weighted_reduce(values: np.ndarray, draws: np.ndarray, weighting: str) -> float:
    if weighting == "interval":
        return float(np.mean(values))
    if weighting != "draw":
        raise ValueError("weighting must be 'interval' or 'draw'")
    return float(np.mean([values[draws == draw].mean() for draw in np.unique(draws)]))


def _decode_selected_status(packed: np.ndarray, rows: np.ndarray) -> np.ndarray:
    byte = rows // 4
    shift = ((rows % 4) * 2).astype(np.uint8)
    return ((packed[:, byte] >> shift[np.newaxis, :]) & np.uint8(3)).astype(np.uint8)


def _read_metadata(path: Path) -> dict:
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing interval-store metadata in {path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("interval-store metadata must be a JSON object")
    return metadata


def _checked_indices(values: np.ndarray, size: int, label: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{label} indices must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= size):
        raise IndexError(f"{label} index out of bounds")
    return result


def _format_values(values: np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(f"{x:g}" for x in values[:limit])
    return shown + (f", ... ({values.size} total)" if values.size > limit else "")


def _validate_chromosomes(metadata: dict) -> None:
    rows = metadata.get("chromosomes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("metadata chromosomes must be a nonempty list")
    seen: set[str] = set()
    previous_end = -1
    for row in rows:
        try:
            chrom = str(row["chrom"])
            offset = int(row["offset"])
            length = int(row["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid chromosome metadata entry") from error
        if not chrom or chrom in seen or offset < 0 or length <= 0 or offset < previous_end:
            raise ValueError("invalid or overlapping chromosome metadata")
        seen.add(chrom)
        previous_end = offset + length


def validate_interval_store(
    store_dir: str | Path, *, deep: bool = False, row_block_size: int = 100_000
) -> IntervalStoreReport:
    """Validate the interval-store schema and, optionally, record-level invariants."""
    path = Path(store_dir)
    metadata = _read_metadata(path)
    required_keys = {
        "schema_version", "n_snps", "n_intervals", "n_posterior_draws",
        "endpoint_dtype", "arrays",
    }
    missing = required_keys - metadata.keys()
    if missing:
        raise ValueError(f"metadata missing required keys: {', '.join(sorted(missing))}")
    if metadata["schema_version"] != INTERVAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported interval schema version: {metadata['schema_version']!r}")
    try:
        n_snps = int(metadata["n_snps"])
        n_intervals = int(metadata["n_intervals"])
        n_draws = int(metadata["n_posterior_draws"])
        endpoint_dtype = np.dtype(metadata["endpoint_dtype"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid interval-store dimensions or endpoint dtype") from error
    if n_snps < 0 or n_intervals < 0 or n_draws <= 0:
        raise ValueError("invalid interval-store dimensions")
    if endpoint_dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("endpoint_dtype must be float32 or float64")
    if not isinstance(metadata["arrays"], dict):
        raise ValueError("metadata arrays must be an object")
    content_digest = metadata.get("content_sha256")
    if content_digest is not None and (
        not isinstance(content_digest, str)
        or len(content_digest) != 64
        or any(character not in "0123456789abcdef" for character in content_digest)
    ):
        raise ValueError("metadata content_sha256 must be a lowercase SHA-256 digest")
    draw_dtype = np.dtype("uint8" if n_draws <= 255 else "uint16")
    required = {
        "positions": (np.dtype("float64"), (n_snps,)),
        "offsets": (np.dtype("uint64"), (n_snps + 1,)),
        "below": (endpoint_dtype, (n_intervals,)),
        "above": (endpoint_dtype, (n_intervals,)),
        "draw_id": (draw_dtype, (n_intervals,)),
        "status": (np.dtype("uint8"), (n_draws, (n_snps + 3) // 4)),
    }
    for name in ("present_draw_count", "missing_draw_count", "usable_draw_count", "usable_interval_count", "skipped_root_count"):
        required[name] = (np.dtype("uint32"), (n_snps,))
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in required.items():
        try:
            array = np.load(path / f"{name}.npy", mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read {name}.npy") from error
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"{name}.npy has dtype/shape {array.dtype}/{array.shape}, expected {dtype}/{shape}")
        description = metadata["arrays"].get(name)
        if not isinstance(description, dict) or description.get("dtype") != dtype.name:
            raise ValueError(f"metadata has an invalid dtype entry for {name}")
        if description.get("shape") != list(shape):
            raise ValueError(f"metadata has an invalid shape entry for {name}")
        arrays[name] = array
    _validate_chromosomes(metadata)
    positions = arrays["positions"]
    if np.any(~np.isfinite(positions)) or np.any(positions != np.floor(positions)):
        raise ValueError("positions must be finite integers")
    if np.any(positions[1:] <= positions[:-1]):
        raise ValueError("positions must be strictly increasing")
    offsets = arrays["offsets"]
    if int(offsets[0]) != 0 or int(offsets[-1]) != n_intervals or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets are not a valid ragged index")
    count_from_offsets = offsets[1:] - offsets[:-1]
    if np.any(count_from_offsets != arrays["usable_interval_count"]):
        raise ValueError("offset record counts disagree with usable_interval_count")
    present = arrays["present_draw_count"]
    missing_count = arrays["missing_draw_count"]
    usable = arrays["usable_draw_count"]
    if np.any(present.astype(np.uint64) + missing_count != n_draws):
        raise ValueError("present_draw_count + missing_draw_count must equal n_posterior_draws")
    if np.any(usable > present):
        raise ValueError("usable_draw_count exceeds present_draw_count")
    if n_snps % 4:
        used_bits = 2 * (n_snps % 4)
        if np.any(arrays["status"][:, -1] >> np.uint8(used_bits)):
            raise ValueError("unused status slots must be zero")
    if deep:
        if content_digest is not None:
            actual_digest = compute_interval_store_content_sha256(path, metadata)
            if actual_digest != content_digest:
                raise ValueError("interval-store content digest does not match its arrays")
        below, above, draw_id = arrays["below"], arrays["above"], arrays["draw_id"]
        if np.any(~np.isfinite(below)) or np.any(~np.isfinite(above)):
            raise ValueError("interval endpoints must be finite")
        if np.any(below < 0) or np.any(above <= below):
            raise ValueError("production intervals must satisfy 0 <= below < above")
        if np.any(draw_id >= n_draws):
            raise ValueError("draw_id is out of range")
        if row_block_size <= 0:
            raise ValueError("row_block_size must be positive")
        for start in range(0, n_snps, row_block_size):
            stop = min(start + row_block_size, n_snps)
            rows = np.arange(start, stop, dtype=np.int64)
            status = _decode_selected_status(np.asarray(arrays["status"]), rows)
            if np.any(status == 3):
                raise ValueError("status value 3 is reserved")
            if np.any(np.count_nonzero(status, axis=0) != present[start:stop]):
                raise ValueError("packed statuses disagree with present_draw_count")
            if np.any(np.count_nonzero(status == STATUS_PRESENT_USABLE, axis=0) != usable[start:stop]):
                raise ValueError("packed statuses disagree with usable_draw_count")
            first, last = int(offsets[start]), int(offsets[stop])
            seen = np.zeros((n_draws, stop - start), dtype=bool)
            if first != last:
                owner = np.repeat(np.arange(stop - start), count_from_offsets[start:stop].astype(np.int64))
                local_draw = np.asarray(draw_id[first:last], dtype=np.int64)
                seen[local_draw, owner] = True
                if np.any(status[local_draw, owner] != STATUS_PRESENT_USABLE):
                    raise ValueError("an interval belongs to a draw without usable status")
            if np.any(seen != (status == STATUS_PRESENT_USABLE)):
                raise ValueError("usable statuses and interval draw IDs disagree")
    return IntervalStoreReport(n_snps, n_intervals, n_draws, endpoint_dtype.name)


# A concise alias is useful to builders that already import ``validate_store``
# from the dense-store reader.  Keep the format-specific name canonical.
validate_store = validate_interval_store
