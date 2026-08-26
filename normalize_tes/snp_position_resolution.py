"""Exact, policy-aware resolution of requested SNP positions.

The dense and interval age stores both expose a sorted ``positions`` array and
an aligned ``eligible`` mask.  This module deliberately resolves against those
arrays directly: unlike the legacy all-or-error resolver, it retains an
aligned result for every request and distinguishes coordinates absent from the
store from coordinates that are present but scientifically ineligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


MissingPositionPolicy = Literal["error", "drop"]


class PositionResolutionError(ValueError):
    """A requested-position policy rejected a completed resolution audit."""

    def __init__(self, message: str, resolution: "PositionResolution"):
        super().__init__(message)
        self.resolution = resolution


@dataclass(frozen=True)
class PositionResolution:
    """Request-aligned position lookup and eligibility result.

    ``row_indices`` has one entry per request and uses ``-1`` only for an
    unresolved coordinate.  ``eligible_mask`` is false for both unresolved
    and resolved-but-ineligible requests.  The category and included
    properties preserve the original request order, which lets downstream
    commands apply exactly the same filtering to native and global arrays.
    """

    global_positions: np.ndarray
    row_indices: np.ndarray
    resolved_mask: np.ndarray
    eligible_mask: np.ndarray
    policy: MissingPositionPolicy
    label: str = "positions"
    chromosomes: np.ndarray | None = None
    native_positions: np.ndarray | None = None

    @property
    def requested_global_positions(self) -> np.ndarray:
        """All requested global coordinates, in their original order."""
        return self.global_positions

    @property
    def requested_chromosomes(self) -> np.ndarray | None:
        """All requested chromosome labels, when native coordinates were supplied."""
        return self.chromosomes

    @property
    def requested_native_positions(self) -> np.ndarray | None:
        """All requested native coordinates, when supplied."""
        return self.native_positions

    @property
    def requested_count(self) -> int:
        return int(self.global_positions.size)

    @property
    def resolved_count(self) -> int:
        return int(np.count_nonzero(self.resolved_mask))

    @property
    def unresolved_count(self) -> int:
        return self.requested_count - self.resolved_count

    @property
    def ineligible_mask(self) -> np.ndarray:
        return self.resolved_mask & ~self.eligible_mask

    @property
    def ineligible_count(self) -> int:
        return int(np.count_nonzero(self.ineligible_mask))

    @property
    def included_mask(self) -> np.ndarray:
        return self.eligible_mask

    @property
    def eligible_count(self) -> int:
        return int(np.count_nonzero(self.eligible_mask))

    @property
    def excluded_mask(self) -> np.ndarray:
        return ~self.eligible_mask

    @property
    def excluded_count(self) -> int:
        return self.requested_count - self.eligible_count

    @property
    def included_request_indices(self) -> np.ndarray:
        return np.flatnonzero(self.included_mask)

    @property
    def unresolved_request_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.resolved_mask)

    @property
    def ineligible_request_indices(self) -> np.ndarray:
        return np.flatnonzero(self.ineligible_mask)

    @property
    def excluded_request_indices(self) -> np.ndarray:
        return np.flatnonzero(self.excluded_mask)

    @property
    def included_rows(self) -> np.ndarray:
        return self.row_indices[self.included_mask]

    @property
    def included_global_positions(self) -> np.ndarray:
        return self.global_positions[self.included_mask]

    @property
    def unresolved_global_positions(self) -> np.ndarray:
        return self.global_positions[~self.resolved_mask]

    @property
    def ineligible_global_positions(self) -> np.ndarray:
        return self.global_positions[self.ineligible_mask]

    @property
    def excluded_global_positions(self) -> np.ndarray:
        return self.global_positions[self.excluded_mask]

    @property
    def included_chromosomes(self) -> np.ndarray | None:
        return None if self.chromosomes is None else self.chromosomes[self.included_mask]

    @property
    def included_native_positions(self) -> np.ndarray | None:
        if self.native_positions is None:
            return None
        return self.native_positions[self.included_mask]

    @property
    def unresolved_chromosomes(self) -> np.ndarray | None:
        if self.chromosomes is None:
            return None
        return self.chromosomes[~self.resolved_mask]

    @property
    def unresolved_native_positions(self) -> np.ndarray | None:
        if self.native_positions is None:
            return None
        return self.native_positions[~self.resolved_mask]

    @property
    def ineligible_chromosomes(self) -> np.ndarray | None:
        if self.chromosomes is None:
            return None
        return self.chromosomes[self.ineligible_mask]

    @property
    def ineligible_native_positions(self) -> np.ndarray | None:
        if self.native_positions is None:
            return None
        return self.native_positions[self.ineligible_mask]

    @property
    def excluded_chromosomes(self) -> np.ndarray | None:
        if self.chromosomes is None:
            return None
        return self.chromosomes[self.excluded_mask]

    @property
    def excluded_native_positions(self) -> np.ndarray | None:
        if self.native_positions is None:
            return None
        return self.native_positions[self.excluded_mask]

    def summary(self) -> dict[str, int | str]:
        """Return the exact category counts for command metadata."""
        return {
            "label": self.label,
            "policy": self.policy,
            "requested_count": self.requested_count,
            "resolved_count": self.resolved_count,
            "unresolved_count": self.unresolved_count,
            "eligible_count": self.eligible_count,
            "ineligible_count": self.ineligible_count,
            "excluded_count": self.excluded_count,
        }

    def excluded_coordinates(self) -> list[dict[str, int | float | str]]:
        """Return every excluded coordinate, in request order, with its cause."""
        records: list[dict[str, int | float | str]] = []
        for request_index in self.excluded_request_indices:
            i = int(request_index)
            value = float(self.global_positions[i])
            global_value: int | float = int(value) if value.is_integer() else value
            record: dict[str, int | float | str] = {
                "request_index": i,
                "global_position": global_value,
                "reason": "unresolved" if not self.resolved_mask[i] else "ineligible",
            }
            if self.chromosomes is not None:
                record["chromosome"] = str(self.chromosomes[i])
            if self.native_positions is not None:
                record["native_position"] = int(self.native_positions[i])
            records.append(record)
        return records


def resolve_requested_positions(
    store: object,
    global_positions: np.ndarray,
    *,
    chromosomes: np.ndarray | None = None,
    native_positions: np.ndarray | None = None,
    policy: MissingPositionPolicy = "error",
    label: str = "positions",
) -> PositionResolution:
    """Resolve exact global coordinates and apply an error-or-drop policy.

    Optional native-coordinate arrays are carried through the result for exact
    audit output; lookup itself always uses the completed store's union
    catalog.  The function fails for an empty request and if filtering would
    leave no eligible position.
    """
    if policy not in ("error", "drop"):
        raise ValueError("missing-position policy must be 'error' or 'drop'")

    query = _global_position_array(global_positions)
    if query.size == 0:
        raise ValueError(f"{label} request is empty")
    chroms, native = _optional_native_arrays(chromosomes, native_positions, query.size)

    catalog = np.asarray(getattr(store, "positions"))
    if catalog.ndim != 1 or catalog.size == 0:
        raise ValueError("store positions must be a nonempty one-dimensional array")
    if not np.issubdtype(catalog.dtype, np.number):
        raise ValueError("store positions must be numeric")
    if np.any(~np.isfinite(catalog)) or np.any(catalog[1:] <= catalog[:-1]):
        raise ValueError("store positions must be finite and strictly increasing")

    eligibility = np.asarray(getattr(store, "eligible"))
    if eligibility.dtype != np.bool_ or eligibility.shape != catalog.shape:
        raise ValueError("store eligible mask must be boolean and match positions")

    insertion = np.searchsorted(catalog, query)
    resolved = insertion < catalog.size
    if np.any(resolved):
        resolved[resolved] &= catalog[insertion[resolved]] == query[resolved]
    rows = np.full(query.size, -1, dtype=np.int64)
    rows[resolved] = insertion[resolved].astype(np.int64, copy=False)
    eligible = np.zeros(query.size, dtype=np.bool_)
    eligible[resolved] = eligibility[rows[resolved]]

    result = PositionResolution(
        global_positions=query,
        row_indices=rows,
        resolved_mask=resolved,
        eligible_mask=eligible,
        policy=policy,
        label=str(label),
        chromosomes=chroms,
        native_positions=native,
    )

    if policy == "error" and result.excluded_count:
        raise PositionResolutionError(_error_message(result), result)
    if result.eligible_count == 0:
        raise PositionResolutionError(
            f"no eligible {label} remain after position resolution", result
        )
    return result


def resolve_native_position_requests(
    store: object,
    chromosomes: np.ndarray,
    native_positions: np.ndarray,
    *,
    policy: MissingPositionPolicy = "error",
    label: str = "positions",
) -> PositionResolution:
    """Convert aligned native coordinates, then resolve them against the store."""
    chroms, native = _optional_native_arrays(
        chromosomes, native_positions, expected_size=None
    )
    assert chroms is not None and native is not None
    if native.size == 0:
        raise ValueError(f"{label} request is empty")
    converter = getattr(store, "native_to_global", None)
    if converter is None:
        raise TypeError("store does not provide native_to_global()")
    global_positions = converter(chroms, native)
    return resolve_requested_positions(
        store,
        global_positions,
        chromosomes=chroms,
        native_positions=native,
        policy=policy,
        label=label,
    )


def _global_position_array(values: np.ndarray) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("global positions must be numeric") from error
    if result.ndim != 1:
        raise ValueError("global positions must be one-dimensional")
    if np.any(~np.isfinite(result)):
        raise ValueError("global positions must be finite")
    if np.any(result != np.floor(result)):
        raise ValueError("global positions must be integral")
    unique, counts = np.unique(result, return_counts=True)
    duplicated = unique[counts > 1]
    if duplicated.size:
        shown = ", ".join(format(float(value), ".15g") for value in duplicated[:10])
        suffix = f", ... ({duplicated.size} total)" if duplicated.size > 10 else ""
        raise ValueError(f"duplicate global positions: {shown}{suffix}")
    return result


def _optional_native_arrays(
    chromosomes: np.ndarray | None,
    native_positions: np.ndarray | None,
    expected_size: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if (chromosomes is None) != (native_positions is None):
        raise ValueError("chromosomes and native positions must be supplied together")
    if chromosomes is None:
        return None, None
    chroms = np.asarray(chromosomes, dtype=str)
    native = np.asarray(native_positions)
    if chroms.ndim != 1 or native.ndim != 1 or chroms.shape != native.shape:
        raise ValueError("chromosomes and native positions must be aligned 1-D arrays")
    if expected_size is not None and chroms.size != expected_size:
        raise ValueError("native and global position arrays must be aligned")
    if not np.issubdtype(native.dtype, np.integer):
        try:
            numeric = native.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("native positions must be integers") from error
        if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise ValueError("native positions must be finite integers")
        native = numeric.astype(np.int64)
    else:
        native = native.astype(np.int64, copy=False)
    if np.any(native < 1):
        raise ValueError("native positions must be at least 1")
    return chroms, native


def _error_message(result: PositionResolution) -> str:
    parts = [
        f"{result.label} resolution excluded {result.excluded_count} of "
        f"{result.requested_count} requests"
    ]
    if result.unresolved_count:
        parts.append(f"unresolved={result.unresolved_count}")
    if result.ineligible_count:
        parts.append(f"resolved-but-ineligible={result.ineligible_count}")
    return "; ".join(parts)
