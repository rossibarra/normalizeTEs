"""Read-only access to a NumPy SNP age-distribution store."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 2
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
        self.eligible = arrays["eligible"]
        self._cdf_by_snp = arrays["cdf_by_snp"]
        self._cdf_by_age = arrays.get("cdf_by_age")
        self.present_draw_count = arrays["present_draw_count"]
        self.usable_draw_count = arrays["usable_draw_count"]
        self.usable_draw_fraction = arrays["usable_draw_fraction"]
        self.usable_interval_count = arrays["usable_interval_count"]
        self.skipped_root_count = arrays["skipped_root_count"]
        self.missing_draw_count = arrays["missing_draw_count"]
        self.chromosomes = tuple(metadata["chromosomes"])
        self._chromosome_by_name = {str(row["chrom"]): row for row in self.chromosomes}

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
            "positions", "age_bins", "cdf_by_snp", "valid", "eligible",
            "present_draw_count", "usable_draw_count", "usable_draw_fraction", "usable_interval_count",
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

    def native_to_global(
        self, chromosomes: np.ndarray, vcf_positions: np.ndarray
    ) -> np.ndarray:
        """Convert chromosome plus 1-based VCF coordinates to store coordinates."""
        chroms = np.asarray(chromosomes, dtype=str)
        positions = np.asarray(vcf_positions)
        if chroms.ndim != 1 or positions.ndim != 1 or chroms.shape != positions.shape:
            raise ValueError("chromosomes and VCF positions must be aligned 1-D arrays")
        if not np.issubdtype(positions.dtype, np.integer):
            numeric = positions.astype(np.float64)
            if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
                raise ValueError("VCF positions must be finite integers")
            positions = numeric.astype(np.int64)
        else:
            positions = positions.astype(np.int64, copy=False)
        output = np.empty(positions.size, dtype=np.float64)
        for i, (chrom, position) in enumerate(zip(chroms, positions)):
            entry = self._chromosome_by_name.get(str(chrom))
            if entry is None:
                raise KeyError(f"chromosome not found in ARG metadata: {chrom}")
            if position < 1 or position > int(entry["length"]):
                raise ValueError(
                    f"VCF position {position} lies outside {chrom} length {entry['length']}"
                )
            # These ARGs store native chromosome positions one-based, matching
            # the user-facing VCF coordinate convention.
            output[i] = int(entry["offset"]) + int(position)
        return output

    def resolve_native_positions(
        self, chromosomes: np.ndarray, vcf_positions: np.ndarray
    ) -> np.ndarray:
        return self.resolve_positions(self.native_to_global(chromosomes, vcf_positions))

    def rows_to_native(self, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return chromosome names and 1-based VCF positions for store rows."""
        rows = _checked_indices(row_indices, self.positions.size, "row")
        globals_ = np.asarray(self.positions[rows], dtype=np.float64)
        names = np.empty(rows.size, dtype=f"U{max(len(str(x['chrom'])) for x in self.chromosomes)}")
        native = np.empty(rows.size, dtype=np.int64)
        ordered = sorted(self.chromosomes, key=lambda item: int(item["offset"]))
        offsets = np.asarray([int(item["offset"]) for item in ordered], dtype=np.float64)
        choices = np.searchsorted(offsets, globals_, side="right") - 1
        if np.any(choices < 0):
            raise ValueError("store position precedes the first chromosome offset")
        for i, choice in enumerate(choices):
            entry = ordered[int(choice)]
            value = globals_[i] - int(entry["offset"])
            if value != math.floor(value) or not 1 <= value <= int(entry["length"]):
                raise ValueError(f"store position {globals_[i]:g} is not a 1-based VCF coordinate")
            names[i] = str(entry["chrom"])
            native[i] = int(value)
        return names, native

    def read_cdfs(self, row_indices: np.ndarray, *, decode: bool = True) -> np.ndarray:
        indices = _checked_indices(row_indices, self.positions.size, "row")
        values = np.asarray(self._cdf_by_snp[indices])
        if decode:
            return values.astype(np.float32) / np.float32(self.quantization_scale)
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
            return values.astype(np.float32) / np.float32(self.quantization_scale)
        return values


def load_native_position_list(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read exactly two columns: chromosome and 1-based integer VCF position."""
    names: list[str] = []
    positions: list[int] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            fields = text.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected chromosome and 1-based VCF position"
                )
            try:
                position = int(fields[1])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: VCF position must be an integer") from error
            if position < 1:
                raise ValueError(f"{path}:{line_number}: VCF position must be at least 1")
            names.append(fields[0])
            positions.append(position)
    if not names:
        raise ValueError(f"position list is empty: {path}")
    pairs = list(zip(names, positions))
    if len(set(pairs)) != len(pairs):
        raise ValueError("position list contains duplicate chromosome-position pairs")
    width = max(len(value) for value in names)
    return np.asarray(names, dtype=f"U{width}"), np.asarray(positions, dtype=np.int64)

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
        "eligible": (np.dtype("bool"), (metadata["n_snps"],)),
        "present_draw_count": (np.dtype("uint32"), (metadata["n_snps"],)),
        "usable_draw_count": (np.dtype("uint32"), (metadata["n_snps"],)),
        "usable_draw_fraction": (np.dtype("float32"), (metadata["n_snps"],)),
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
    chromosomes = metadata.get("chromosomes")
    if not isinstance(chromosomes, list) or not chromosomes:
        raise ValueError("metadata must contain a nonempty chromosomes table")
    if np.any(~np.isfinite(positions)) or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must be finite and strictly increasing")
    if bins.size == 0 or np.any(bins[1:] <= bins[:-1]):
        raise ValueError("age bins must be strictly increasing and nonempty")
    draws = metadata.get("n_posterior_draws")
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("metadata n_posterior_draws must be a positive integer")
    if draws is not None and np.any(
        arrays["present_draw_count"].astype(np.uint64)
        + arrays["missing_draw_count"].astype(np.uint64) != draws
    ):
        raise ValueError("present and missing draw counts are inconsistent")
    if np.any(arrays["usable_interval_count"] < arrays["valid"].astype(np.uint32)):
        raise ValueError("valid flags and usable interval counts are inconsistent")
    if np.any(arrays["usable_draw_count"] > arrays["present_draw_count"]):
        raise ValueError("usable draw counts exceed present draw counts")
    expected_fraction = arrays["usable_draw_count"].astype(np.float64) / int(draws)
    if not np.allclose(arrays["usable_draw_fraction"], expected_fraction):
        raise ValueError("usable draw fractions are inconsistent")
    required_draws = int(metadata.get("minimum_usable_draws", 1))
    expected_eligible = arrays["valid"] & (arrays["usable_draw_count"] >= required_draws)
    if not np.array_equal(arrays["eligible"], expected_eligible):
        raise ValueError("eligible flags are inconsistent with the coverage threshold")
    valid, cdf = arrays["valid"], arrays["cdf_by_snp"]
    scale = metadata.get("quantization_scale", QUANTIZATION_SCALE)
    if not isinstance(scale, int) or not 0 < scale <= np.iinfo(np.uint16).max:
        raise ValueError("metadata quantization_scale must fit in uint16")
    block = max(1, min(100_000, cdf.shape[0]))
    ranges = range(0, cdf.shape[0], block) if deep else ([0] if cdf.shape[0] else [])
    for start in ranges:
        rows = np.asarray(cdf[start:min(start + block, cdf.shape[0])])
        flags = np.asarray(valid[start:min(start + block, cdf.shape[0])])
        if np.any(np.diff(rows.astype(np.int32), axis=1) < 0):
            raise ValueError("CDF rows are not monotone")
        if np.any(rows[flags, -1] != scale):
            raise ValueError(f"valid CDF rows must terminate at {scale}")
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
