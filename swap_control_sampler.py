"""Swap-chain construction of SNP control sets matched by posterior-age CDF."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from snp_age_store import is_interval_store, open_snp_age_store
from te_age_target import wasserstein_1


class SwapSamplingError(RuntimeError):
    """Raised when a matched chain cannot finish within its declared budget."""


@dataclass(frozen=True)
class SwapConfig:
    sets_per_chain: int = 25
    search_bin_width: int = 20_000
    burnin_replacement_fraction: float = 0.50
    sample_replacement_fraction: float = 0.25
    max_construction_epochs: int = 50
    max_chain_proposals: int = 10_000_000
    cdf_block_rows: int = 256
    exact_check_accepted: int = 1_000
    progress_every: int = 100_000
    algorithm_version: str = "swap-age-controls-v1"

    def validate(self) -> None:
        if self.sets_per_chain <= 0:
            raise ValueError("sets_per_chain must be positive")
        if self.search_bin_width <= 0 or self.cdf_block_rows <= 0:
            raise ValueError("grid width and CDF block rows must be positive")
        if not 0 < self.burnin_replacement_fraction <= 1:
            raise ValueError("burn-in replacement fraction must be in (0, 1]")
        if not 0 < self.sample_replacement_fraction <= 1:
            raise ValueError("sample replacement fraction must be in (0, 1]")
        if self.max_construction_epochs <= 0 or self.max_chain_proposals <= 0:
            raise ValueError("proposal budgets must be positive")
        if self.exact_check_accepted <= 0 or self.progress_every <= 0:
            raise ValueError("progress intervals must be positive")


@dataclass
class ChainOutput:
    chain_index: int
    seed: int
    row_indices: np.ndarray
    cdfs: np.ndarray
    wasserstein: np.ndarray
    diagnostics: list[dict[str, Any]]
    construction: dict[str, Any]


def derive_chain_seed(global_seed: int, target_digest: str,
                      chain_index: int, algorithm_version: str) -> int:
    """Derive a stable uint64 seed without Python's randomized hash()."""
    payload = f"{global_seed}\0{target_digest}\0{chain_index}\0{algorithm_version}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def replacement_fraction(current: np.ndarray, reference: np.ndarray) -> float:
    """Return the fraction of reference members absent from current."""
    left = np.asarray(current, dtype=np.int64)
    right = np.asarray(reference, dtype=np.int64)
    if left.ndim != 1 or right.shape != left.shape or left.size == 0:
        raise ValueError("sets must be nonempty aligned vectors")
    shared = np.intersect1d(left, right, assume_unique=True).size
    return 1.0 - shared / left.size


def incremental_cdf(current: np.ndarray, old: np.ndarray, new: np.ndarray,
                    set_size: int) -> np.ndarray:
    if set_size <= 0:
        raise ValueError("set_size must be positive")
    return np.asarray(current, dtype=np.float64) + (
        np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)
    ) / set_size


def analysis_points(age_bins: np.ndarray) -> np.ndarray:
    ages = np.asarray(age_bins, dtype=np.float64)
    if ages.ndim != 1 or ages.size < 2 or np.any(np.diff(ages) <= 0):
        raise ValueError("age_bins must be a strictly increasing vector")
    widths = np.diff(ages)
    if not np.allclose(widths, widths[0], rtol=0, atol=0):
        raise ValueError("interval-store scoring requires a uniform age grid")
    return ages + widths[0] / 2


def search_grid(maximum_age: float, bin_width: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(maximum_age) or maximum_age <= 0 or bin_width <= 0:
        raise ValueError("maximum age and bin width must be positive")
    last = max(1, int(np.floor(maximum_age / bin_width + 0.5)))
    ages = np.arange(last + 1, dtype=np.float64) * bin_width
    return ages, ages + bin_width / 2


def row_cdfs(store: object, rows: np.ndarray, points: np.ndarray, *,
             block_rows: int, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    """Evaluate row CDFs in bounded blocks."""
    indices = np.asarray(rows, dtype=np.int64)
    output = np.empty((indices.size, np.asarray(points).size), dtype=dtype)
    for start in range(0, indices.size, block_rows):
        stop = min(start + block_rows, indices.size)
        if is_interval_store(store):
            values = store.cdf_at(
                indices[start:stop], points, side="left", weighting="interval"
            )
        else:
            values = store.read_cdfs(indices[start:stop])
        output[start:stop] = np.asarray(values, dtype=dtype)
    return output


def aggregate_cdf(store: object, rows: np.ndarray, points: np.ndarray) -> np.ndarray:
    indices = np.asarray(rows, dtype=np.int64)
    if is_interval_store(store):
        return np.asarray(store.aggregate_cdf_at(
            indices, points, side="left", weighting="interval"
        ), dtype=np.float64)
    return np.asarray(store.read_cdfs(indices), dtype=np.float64).mean(axis=0)


def eligible_candidates(store: object, target_rows: np.ndarray,
                        candidate_rows: np.ndarray | None = None) -> np.ndarray:
    """Resolve a sorted unique eligible control universe excluding targets."""
    if candidate_rows is None:
        rows = np.flatnonzero(np.asarray(store.eligible))
    else:
        raw = np.asarray(candidate_rows)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("candidate rows must be a one-dimensional integer array")
        rows = raw.astype(np.int64, copy=False)
        if np.any(rows < 0) or np.any(rows >= np.asarray(store.positions).size):
            raise ValueError("candidate rows contain out-of-range values")
        if np.unique(rows).size != rows.size:
            raise ValueError("candidate rows contain duplicates")
        if not np.all(np.asarray(store.eligible)[rows]):
            raise ValueError("candidate rows include ineligible store rows")
        rows = np.sort(rows)
    return np.setdiff1d(rows, np.asarray(target_rows, dtype=np.int64), assume_unique=True)


def _w1(cdf: np.ndarray, target: np.ndarray, age_bins: np.ndarray) -> float:
    return wasserstein_1(np.asarray(cdf), np.asarray(target), np.asarray(age_bins))


def _atomic_checkpoint(path: Path, *, selected: np.ndarray, phase: str,
                       rng: np.random.Generator, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            selected=np.asarray(selected, dtype=np.int64),
            phase=np.asarray(phase),
            rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
            payload=np.asarray(json.dumps(payload)),
        )
    os.replace(temporary, path)


def _propose_unselected(rng: np.random.Generator, candidates: np.ndarray,
                        selected_set: set[int]) -> tuple[int, int]:
    duplicates = 0
    while True:
        row = int(candidates[int(rng.integers(candidates.size))])
        if row not in selected_set:
            return row, duplicates
        duplicates += 1


def run_chain(
    store_path: str | Path,
    target_rows: np.ndarray,
    target_cdf: np.ndarray,
    age_bins: np.ndarray,
    threshold: float,
    *,
    candidate_rows: np.ndarray | None,
    global_seed: int,
    target_digest: str,
    chain_index: int,
    config: SwapConfig,
    checkpoint_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ChainOutput:
    """Construct, burn in, and collect one independent matched-control chain."""
    config.validate()
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    emit = progress or (lambda message: None)
    store = open_snp_age_store(store_path)
    if not is_interval_store(store):
        raise ValueError("swap sampler version 1 requires an interval store")
    target_rows = np.asarray(target_rows, dtype=np.int64)
    n = target_rows.size
    if n == 0 or np.unique(target_rows).size != n:
        raise ValueError("target rows must be nonempty and unique")
    candidates = eligible_candidates(store, target_rows, candidate_rows)
    if candidates.size < n:
        raise SwapSamplingError("candidate universe is smaller than the target set")
    exact_ages = np.asarray(age_bins, dtype=np.float64)
    exact_points = analysis_points(exact_ages)
    target_cdf = np.asarray(target_cdf, dtype=np.float64)
    if target_cdf.shape != exact_ages.shape:
        raise ValueError("target CDF and age grid are incompatible")
    maximum = float(store.metadata["maximum_above"])
    coarse_ages, coarse_points = search_grid(maximum, config.search_bin_width)
    target_search = aggregate_cdf(store, target_rows, coarse_points)
    seed = derive_chain_seed(global_seed, target_digest, chain_index,
                             config.algorithm_version)
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=n, replace=False).astype(np.int64)
    selected_set = set(map(int, selected))
    current_search = aggregate_cdf(store, selected, coarse_points)
    construction_start = time.perf_counter()
    exact_distance = np.inf
    total_proposals = total_accepted = duplicate_proposals = 0
    construction_history: list[dict[str, Any]] = []

    for epoch in range(1, config.max_construction_epochs + 1):
        slots = rng.permutation(n)
        proposed = rng.choice(candidates, size=n, replace=False).astype(np.int64)
        old_rows = selected[slots].copy()
        old_cdfs = row_cdfs(store, old_rows, coarse_points,
                            block_rows=config.cdf_block_rows)
        new_cdfs = row_cdfs(store, proposed, coarse_points,
                            block_rows=config.cdf_block_rows)
        current_distance = _w1(current_search, target_search, coarse_ages)
        near_threshold = exact_distance <= 2 * threshold
        epoch_accepted = 0
        epoch_proposals = 0
        accepted_since_exact = 0
        reached_threshold = False
        for j, slot in enumerate(slots):
            total_proposals += 1
            epoch_proposals += 1
            old = int(selected[slot])
            new = int(proposed[j])
            if new in selected_set:
                duplicate_proposals += 1
                continue
            trial = incremental_cdf(current_search, old_cdfs[j], new_cdfs[j], n)
            trial_distance = _w1(trial, target_search, coarse_ages)
            if trial_distance < current_distance:
                current_search = trial
                current_distance = trial_distance
                selected_set.remove(old)
                selected_set.add(new)
                selected[slot] = new
                epoch_accepted += 1
                total_accepted += 1
                accepted_since_exact += 1
                if (near_threshold and
                        accepted_since_exact >= config.exact_check_accepted):
                    exact = aggregate_cdf(store, selected, exact_points)
                    exact_distance = _w1(exact, target_cdf, exact_ages)
                    accepted_since_exact = 0
                    if exact_distance <= threshold:
                        reached_threshold = True
                        break
        if not reached_threshold:
            exact = aggregate_cdf(store, selected, exact_points)
            exact_distance = _w1(exact, target_cdf, exact_ages)
        record = {
            "epoch": epoch,
            "proposals": epoch_proposals,
            "accepted_swaps": epoch_accepted,
            "coarse_wasserstein": current_distance,
            "exact_wasserstein": exact_distance,
        }
        construction_history.append(record)
        emit(f"chain={chain_index} construction epoch={epoch} "
             f"accepted={epoch_accepted} exact_w1={exact_distance:.6g}")
        if checkpoint_path is not None:
            _atomic_checkpoint(Path(checkpoint_path), selected=selected,
                               phase="construction", rng=rng,
                               payload={"chain_index": chain_index, **record})
        if reached_threshold or exact_distance <= threshold:
            break
    else:
        raise SwapSamplingError(
            f"chain {chain_index} did not reach threshold after "
            f"{config.max_construction_epochs} epochs; best W1={exact_distance:.6g}, "
            f"threshold={threshold:.6g}"
        )

    entry = selected.copy()
    # The constrained walk is certified against the exact grid. Keep selected
    # row CDFs in float64 so proposal acceptance is not changed by a float32
    # cache approximation; production memory sizing must include this matrix.
    exact_cache = row_cdfs(
        store, selected, exact_points,
        block_rows=config.cdf_block_rows, dtype=np.dtype("float64"),
    )
    current_exact = exact_cache.mean(axis=0, dtype=np.float64)
    exact_distance = _w1(current_exact, target_cdf, exact_ages)
    diagnostics: list[dict[str, Any]] = []
    saved_rows: list[np.ndarray] = []
    saved_cdfs: list[np.ndarray] = []
    saved_distances: list[float] = []
    walk_proposals = walk_accepted = walk_duplicates = 0

    def advance(reference: np.ndarray, required: float, phase: str) -> dict[str, Any]:
        nonlocal current_exact, exact_distance, walk_proposals, walk_accepted
        nonlocal walk_duplicates
        phase_proposals = phase_accepted = phase_duplicates = 0
        while replacement_fraction(selected, reference) < required:
            if walk_proposals >= config.max_chain_proposals:
                raise SwapSamplingError(
                    f"chain {chain_index} exceeded {config.max_chain_proposals} "
                    f"walk proposals during {phase}"
                )
            slot = int(rng.integers(n))
            new, duplicates = _propose_unselected(rng, candidates, selected_set)
            phase_duplicates += duplicates
            walk_duplicates += duplicates
            old = int(selected[slot])
            new_cdf = np.asarray(store.cdf_at(
                np.asarray([new], dtype=np.int64), exact_points,
                side="left", weighting="interval"
            )[0], dtype=np.float64)
            trial = incremental_cdf(current_exact, exact_cache[slot], new_cdf, n)
            trial_distance = _w1(trial, target_cdf, exact_ages)
            phase_proposals += 1
            walk_proposals += 1
            if trial_distance <= threshold:
                selected_set.remove(old)
                selected_set.add(new)
                selected[slot] = new
                exact_cache[slot] = new_cdf
                current_exact = trial
                exact_distance = trial_distance
                phase_accepted += 1
                walk_accepted += 1
            if walk_proposals % config.progress_every == 0:
                emit(f"chain={chain_index} phase={phase} proposals={walk_proposals} "
                     f"accepted={walk_accepted} replacement="
                     f"{replacement_fraction(selected, reference):.4f}")
        return {
            "phase": phase,
            "proposals": phase_proposals,
            "accepted_swaps": phase_accepted,
            "duplicate_redraws": phase_duplicates,
            "replacement_fraction": replacement_fraction(selected, reference),
            "wasserstein": exact_distance,
        }

    diagnostics.append(advance(entry, config.burnin_replacement_fraction, "burnin"))
    for sample_index in range(config.sets_per_chain):
        if sample_index:
            diagnostics.append(advance(
                saved_rows[-1], config.sample_replacement_fraction, "thinning"
            ))
        certified = aggregate_cdf(store, selected, exact_points)
        certified_distance = _w1(certified, target_cdf, exact_ages)
        if certified_distance > threshold:
            raise SwapSamplingError(
                f"chain {chain_index} sample {sample_index} failed exact certification"
            )
        saved_rows.append(selected.copy())
        saved_cdfs.append(certified)
        saved_distances.append(certified_distance)
        diagnostics.append({
            "phase": "saved",
            "sample_index": sample_index,
            "wasserstein": certified_distance,
            "accepted_swaps_total": walk_accepted,
            "proposals_total": walk_proposals,
            "replacement_from_entry": replacement_fraction(selected, entry),
            "replacement_from_previous": (
                1.0 if sample_index == 0 else
                replacement_fraction(selected, saved_rows[-2])
            ),
        })
        emit(f"chain={chain_index} saved={sample_index + 1}/"
             f"{config.sets_per_chain} exact_w1={certified_distance:.6g}")
        if checkpoint_path is not None:
            _atomic_checkpoint(
                Path(checkpoint_path), selected=selected, phase="sampling", rng=rng,
                payload={"chain_index": chain_index,
                         "saved": sample_index + 1,
                         "wasserstein": certified_distance},
            )

    construction = {
        "elapsed_seconds": time.perf_counter() - construction_start,
        "proposals": total_proposals,
        "accepted_swaps": total_accepted,
        "duplicate_proposals": duplicate_proposals,
        "entry_wasserstein": construction_history[-1]["exact_wasserstein"],
        "history": construction_history,
        "walk_proposals": walk_proposals,
        "walk_accepted_swaps": walk_accepted,
        "walk_duplicate_redraws": walk_duplicates,
        "config": asdict(config),
    }
    return ChainOutput(
        chain_index=chain_index,
        seed=seed,
        row_indices=np.stack(saved_rows),
        cdfs=np.stack(saved_cdfs),
        wasserstein=np.asarray(saved_distances, dtype=np.float64),
        diagnostics=diagnostics,
        construction=construction,
    )
