"""Read-only access to a NumPy SNP age-distribution store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
QUANTIZATION_SCALE = 65535


@dataclass(frozen=True)
class StoreReport:
    n_snps: int
    n_age_bins: int
    n_valid: int
    has_transpose: bool


class SNPAgeDataset:
    """Memory-mapped, read-only view of a SNP age dataset."""

    def __init__(self, store_dir: Path, metadata: dict, arrays: dict[str, np.ndarray]):
        self.store_dir = store_dir
        self.metadata = metadata
        self.quantization_scale = int(metadata.get("quantization_scale", QUANTIZATION_SCALE))
        self.positions = arrays["positions"]
        self.age_bins = arrays["age_bins"]
        self.valid = arrays["valid"]
        self._cdf_by_snp = arrays["cdf_by_snp"]
        self._cdf_by_age = arrays.get("cdf_by_age")
        self.present_draw_count = arrays["present_draw_count"]
        self.usable_interval_count = arrays["usable_interval_count"]
        self.skipped_root_count = arrays["skipped_root_count"]
        self.missing_draw_count = arrays["missing_draw_count"]

    @classmethod
    def open(cls, store_dir: str | Path, *, validate: bool = True) -> "SNPAgeDataset":
        path = Path(store_dir)
        if validate:
            validate_store(path)
        try:
            metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"not a SNP age store (metadata.json missing): {path}") from error
        names = [
            "positions", "age_bins", "cdf_by_snp", "valid",
            "present_draw_count", "usable_interval_count",
            "skipped_root_count", "missing_draw_count",
        ]
        arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in names}
        transpose = path / "cdf_by_age.npy"
        if transpose.exists():
            arrays["cdf_by_age"] = np.load(transpose, mmap_mode="r")
        return cls(path, metadata, arrays)

    @property
    def shape(self) -> tuple[int, int]:
        return self._cdf_by_snp.shape

    def resolve_positions(self, query_positions: np.ndarray) -> np.ndarray:
        """Resolve exact coordinates, rejecting duplicate or missing queries."""
        query = np.asarray(query_positions, dtype=np.float64)
        if query.ndim != 1:
            raise ValueError("query positions must be one-dimensional")
        if not np.all(np.isfinite(query)):
            raise ValueError("query positions must be finite")
        unique, counts = np.unique(query, return_counts=True)
        duplicated = unique[counts > 1]
        if duplicated.size:
            raise ValueError(f"duplicate query positions: {_format_values(duplicated)}")
        indices = np.searchsorted(self.positions, query)
        found = indices < self.positions.size
        if np.any(found):
            found[found] &= self.positions[indices[found]] == query[found]
        if not np.all(found):
            raise KeyError(f"positions not found: {_format_values(query[~found])}")
        return indices.astype(np.int64, copy=False)

    def read_cdfs(self, row_indices: np.ndarray, *, decode: bool = True) -> np.ndarray:
        indices = _checked_indices(row_indices, self.positions.size, "row")
        values = np.asarray(self._cdf_by_snp[indices])
        if decode:
            return values.astype(np.float32) / np.float32(QUANTIZATION_SCALE)
        return values

    def read_boundary_cdfs(
        self, age_indices: np.ndarray, start: int, stop: int, *, decode: bool = True
    ) -> np.ndarray:
        """Read CDF values as ``(age boundaries, contiguous SNP range)``."""
        ages = _checked_indices(age_indices, self.age_bins.size, "age")
        if not (0 <= start <= stop <= self.positions.size):
            raise IndexError("invalid SNP slice")
        if self._cdf_by_age is not None:
            values = np.asarray(self._cdf_by_age[ages, start:stop])
        else:
            values = np.asarray(self._cdf_by_snp[start:stop, ages]).T
        if decode:
            return values.astype(np.float32) / np.float32(QUANTIZATION_SCALE)
        return values


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


def validate_store(store_dir: str | Path, *, deep: bool = False) -> StoreReport:
    """Validate schema, arrays, CDF invariants, and optional transpose."""
    path = Path(store_dir)
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing store metadata in {path}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {metadata.get('schema_version')!r}")
    required = {
        "age_bins": (np.dtype("uint64"), (metadata["n_age_bins"],)),
        "positions": (np.dtype("float64"), (metadata["n_snps"],)),
        "cdf_by_snp": (np.dtype("uint16"), (metadata["n_snps"], metadata["n_age_bins"])),
        "valid": (np.dtype("bool"), (metadata["n_snps"],)),
        "present_draw_count": (np.dtype("uint32"), (metadata["n_snps"],)),
        "usable_interval_count": (np.dtype("uint32"), (metadata["n_snps"],)),
        "skipped_root_count": (np.dtype("uint32"), (metadata["n_snps"],)),
        "missing_draw_count": (np.dtype("uint32"), (metadata["n_snps"],)),
    }
    arrays = {}
    for name, (dtype, shape) in required.items():
        try:
            array = np.load(path / f"{name}.npy", mmap_mode="r")
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read {name}.npy") from error
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"{name}.npy has dtype/shape {array.dtype}/{array.shape}, expected {dtype}/{shape}")
        arrays[name] = array
    positions, bins = arrays["positions"], arrays["age_bins"]
    if np.any(~np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be finite and strictly increasing")
    if bins.size == 0 or np.any(bins[1:] <= bins[:-1]):
        raise ValueError("age bins must be strictly increasing and nonempty")
    draws = metadata.get("n_posterior_draws")
    if draws is not None and np.any(
        arrays["present_draw_count"].astype(np.uint64)
        + arrays["missing_draw_count"].astype(np.uint64) != draws
    ):
        raise ValueError("present and missing draw counts are inconsistent")
    if np.any(arrays["usable_interval_count"] < arrays["valid"].astype(np.uint32)):
        raise ValueError("valid flags and usable interval counts are inconsistent")
    valid, cdf = arrays["valid"], arrays["cdf_by_snp"]
    block = max(1, min(100_000, cdf.shape[0]))
    ranges = range(0, cdf.shape[0], block) if deep else ([0] if cdf.shape[0] else [])
    for start in ranges:
        rows = np.asarray(cdf[start:min(start + block, cdf.shape[0])])
        flags = np.asarray(valid[start:min(start + block, cdf.shape[0])])
        if np.any(np.diff(rows.astype(np.int32), axis=1) < 0):
            raise ValueError("CDF rows are not monotone")
        if np.any(rows[flags, -1] != QUANTIZATION_SCALE):
            raise ValueError("valid CDF rows must terminate at 65535")
        if np.any(rows[~flags] != 0):
            raise ValueError("invalid CDF rows must be zero")
    transpose_path = path / "cdf_by_age.npy"
    has_transpose = transpose_path.exists()
    if has_transpose:
        transposed = np.load(transpose_path, mmap_mode="r")
        if transposed.dtype != np.uint16 or transposed.shape != cdf.shape[::-1]:
            raise ValueError("cdf_by_age.npy has invalid dtype or shape")
        if deep:
            for start in range(0, cdf.shape[0], block):
                stop = min(start + block, cdf.shape[0])
                if not np.array_equal(cdf[start:stop], transposed[:, start:stop].T):
                    raise ValueError("CDF orientations disagree")
        elif cdf.shape[0] and not np.array_equal(cdf[:block], transposed[:, :block].T):
            raise ValueError("CDF orientations disagree")
    return StoreReport(int(cdf.shape[0]), int(cdf.shape[1]), int(np.count_nonzero(valid)), has_transpose)
