#!/usr/bin/env python3
"""Match SNP sets to prespecified bootstrap replicates of a TE age CDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from release_provenance import software_provenance
from sample_age_matched_controls import _load_target, _sha256_arrays
from snp_age_store import is_interval_store, open_snp_age_store, store_schema
from swap_control_sampler import (
    aggregate_cdf,
    analysis_points,
    eligible_candidates,
    incremental_cdf,
    row_cdfs,
    search_grid,
)
from te_age_target import wasserstein_1


SCHEMA_VERSION = "bootstrap-target-matches-v1"
ALGORITHM_VERSION = "bootstrap-target-exact-greedy-v1"


@dataclass(frozen=True)
class OptimizerConfig:
    replicates: int = 100
    closest_restarts: int = 2
    diverse_restarts: int = 1
    min_epochs: int = 10
    max_epochs: int = 50
    patience: int = 5
    material_improvement_ratio: float = 1e-3
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-12
    cdf_block_rows: int = 256
    search_bin_width: int = 20_000
    qc_max_ratio: float = 0.5
    qc_max_absolute: float | None = None
    qc_max_absolute_fraction: float = 0.34
    algorithm_version: str = ALGORITHM_VERSION

    def validate(self) -> None:
        if self.replicates <= 0:
            raise ValueError("replicates must be positive")
        if self.closest_restarts < 0 or self.diverse_restarts < 0:
            raise ValueError("restart counts must be nonnegative")
        if self.closest_restarts + self.diverse_restarts <= 0:
            raise ValueError("at least one restart is required")
        if not 0 < self.min_epochs <= self.max_epochs:
            raise ValueError("epochs must satisfy 0 < min <= max")
        if self.patience <= 0 or self.cdf_block_rows <= 0:
            raise ValueError("patience and CDF block size must be positive")
        if self.search_bin_width <= 0:
            raise ValueError("search width must be positive")
        if self.material_improvement_ratio < 0:
            raise ValueError("material improvement ratio must be nonnegative")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("numerical tolerances must be nonnegative")
        if self.qc_max_absolute is not None and self.qc_max_absolute <= 0:
            raise ValueError("QC thresholds must be positive")
        if self.qc_max_ratio <= 0 or self.qc_max_absolute_fraction <= 0:
            raise ValueError("QC thresholds must be positive")


@dataclass
class RestartResult:
    seed_index: int
    restart_kind: str
    seed: int
    rows: np.ndarray
    cdf: np.ndarray
    initial_distance: float
    best_distance: float
    match_to_observed: float
    epochs: int
    proposals: int
    accepted: int
    termination: str
    trace: list[dict[str, float | int]]
    elapsed_seconds: float = 0.0


def derive_seed(global_seed: int, target_digest: str, replicate: int,
                restart: int | None = None) -> int:
    suffix = "bootstrap" if restart is None else f"restart\0{restart}"
    payload = (
        f"{global_seed}\0{target_digest}\0{replicate}\0{suffix}"
        f"\0{ALGORITHM_VERSION}"
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def bootstrap_counts(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("bootstrap site count must be positive")
    probabilities = np.full(n_sites, 1.0 / n_sites)
    return rng.multinomial(n_sites, probabilities).astype(np.uint32)


def bootstrap_cdf(counts: np.ndarray, cdf_rows: np.ndarray,
                  *, block_rows: int = 4096) -> np.ndarray:
    """Return the count-weighted mean CDF, accumulated in float64.

    `cdf_rows` is stored as float32 to bound memory at production TE-set
    sizes, but the weighted sum runs over one row per TE site and must not
    accumulate at that precision: at 35,000 sites a float32 accumulation
    carries roughly 1e-5 relative error, which displaces the bootstrap target
    and every distance derived from it. Blocks keep the float64 promotion
    bounded rather than materializing a float64 copy of the whole matrix.
    """
    weights = np.asarray(counts)
    rows = np.asarray(cdf_rows)
    if weights.ndim != 1 or rows.ndim != 2 or rows.shape[0] != weights.size:
        raise ValueError("bootstrap counts and TE CDF rows are not aligned")
    if not np.issubdtype(weights.dtype, np.integer) or np.any(weights < 0):
        raise ValueError("bootstrap counts must be nonnegative integers")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    total = int(weights.sum(dtype=np.uint64))
    if total <= 0:
        raise ValueError("bootstrap counts sum to zero")
    accumulated = np.zeros(rows.shape[1], dtype=np.float64)
    for start in range(0, rows.shape[0], block_rows):
        stop = min(start + block_rows, rows.shape[0])
        block = weights[start:stop]
        if not block.any():
            continue
        accumulated += block.astype(np.float64) @ rows[start:stop].astype(np.float64)
    return accumulated / total


def select_seed_indices(
    seed_cdfs: np.ndarray,
    target_cdf: np.ndarray,
    age_bins: np.ndarray,
    *,
    closest: int,
    diverse: int,
    rng: np.random.Generator,
) -> list[tuple[int, str]]:
    if seed_cdfs.ndim != 2 or seed_cdfs.shape[1:] != target_cdf.shape:
        raise ValueError("seed CDFs do not align with target CDF")
    total = closest + diverse
    if total > seed_cdfs.shape[0]:
        raise ValueError("requested restarts exceed available seed sets")
    distances = np.asarray([
        wasserstein_1(cdf, target_cdf, age_bins) for cdf in seed_cdfs
    ])
    order = np.argsort(distances, kind="stable")
    chosen = [(int(index), "closest") for index in order[:closest]]
    remaining = order[closest:]
    if diverse:
        selected = rng.choice(remaining, size=diverse, replace=False)
        chosen.extend((int(index), "diverse") for index in selected)
    return chosen


def optimize_restart(
    store: object,
    candidates: np.ndarray,
    initial_rows: np.ndarray,
    bootstrap_target: np.ndarray,
    observed_target: np.ndarray,
    age_bins: np.ndarray,
    bootstrap_distance: float,
    *,
    coarse_target: np.ndarray,
    coarse_ages: np.ndarray,
    coarse_points: np.ndarray,
    seed_index: int,
    restart_kind: str,
    seed: int,
    config: OptimizerConfig,
    progress: Callable[[str], None] | None = None,
) -> RestartResult:
    """Screen swaps on a coarse grid; certify and report on the exact grid.

    Proposals are scored against the coarse-grid bootstrap target, because the
    exact analysis grid spans `maximum_above / bin_width` points — about 22,900
    for the production store — and scoring every proposal there dominates the
    run. `swap_control_sampler` already uses this two-tier design; the coarse
    width is `config.search_bin_width`, and setting it to the exact grid width
    makes the two grids identical.

    Every distance that is recorded, compared against `best_distance`, or used
    by the convergence rule is an EXACT-grid distance recomputed from the
    selected rows. The coarse grid only decides which swaps to try, so a coarse
    misjudgement costs search efficiency, never correctness of the published
    state.
    """
    emit = progress or (lambda _: None)
    started = time.perf_counter()
    selected = np.asarray(initial_rows, dtype=np.int64).copy()
    n = selected.size
    if n == 0 or np.unique(selected).size != n:
        raise ValueError("initial rows must be nonempty and unique")
    candidate_values = np.asarray(candidates, dtype=np.int64)
    if not np.all(np.isin(selected, candidate_values, assume_unique=False)):
        raise ValueError("initial rows are outside the candidate universe")
    points = analysis_points(age_bins)
    cache = row_cdfs(
        store, selected, coarse_points,
        block_rows=config.cdf_block_rows, dtype=np.dtype("float64"),
    )
    current = cache.mean(axis=0, dtype=np.float64)
    current_distance = wasserstein_1(current, coarse_target, coarse_ages)

    certified = aggregate_cdf(store, selected, points)
    exact_distance = wasserstein_1(certified, bootstrap_target, age_bins)
    initial_distance = exact_distance
    best_distance = exact_distance
    best_rows = selected.copy()
    selected_set = set(map(int, selected))
    rng = np.random.default_rng(seed)
    trace: list[dict[str, float | int]] = []
    proposals = accepted = 0
    stagnant = 0
    termination = "maximum_epochs"
    material = config.material_improvement_ratio * max(bootstrap_distance, 1.0)

    for epoch in range(1, config.max_epochs + 1):
        before = best_distance
        epoch_proposals = epoch_accepted = 0
        slots = rng.permutation(n)
        proposed = rng.choice(candidate_values, size=n, replace=False)
        for start in range(0, n, config.cdf_block_rows):
            stop = min(start + config.cdf_block_rows, n)
            new_cdfs = row_cdfs(
                store, proposed[start:stop], coarse_points,
                block_rows=config.cdf_block_rows, dtype=np.dtype("float64"),
            )
            for local, proposal_index in enumerate(range(start, stop)):
                slot = int(slots[proposal_index])
                new = int(proposed[proposal_index])
                epoch_proposals += 1
                proposals += 1
                if new in selected_set:
                    continue
                old = int(selected[slot])
                trial = incremental_cdf(current, cache[slot], new_cdfs[local], n)
                trial_distance = wasserstein_1(trial, coarse_target, coarse_ages)
                tolerance = max(
                    config.absolute_tolerance,
                    config.relative_tolerance * max(current_distance, 1.0),
                )
                if trial_distance < current_distance - tolerance:
                    selected_set.remove(old)
                    selected_set.add(new)
                    selected[slot] = new
                    cache[slot] = new_cdfs[local]
                    current = trial
                    current_distance = trial_distance
                    epoch_accepted += 1
                    accepted += 1

        # Reset incremental drift from the per-row cache; no store access.
        current = cache.mean(axis=0, dtype=np.float64)
        current_distance = wasserstein_1(current, coarse_target, coarse_ages)

        # Exact certification every epoch: measured at ~1 ms against ~9 ms
        # for the epoch's coarse proposals, because aggregate_cdf_at uses a
        # specialised O(intervals + grid) path rather than materialising a
        # row-by-grid matrix. Certifying less often would trade the guarantee
        # that every recorded best_distance is exact for roughly a tenth of
        # the epoch cost.
        certified = aggregate_cdf(store, selected, points)
        exact_distance = wasserstein_1(certified, bootstrap_target, age_bins)
        if exact_distance < best_distance:
            best_distance = exact_distance
            best_rows = selected.copy()
        improvement = before - best_distance
        trace.append({
            "epoch": epoch,
            "proposals": epoch_proposals,
            "accepted": epoch_accepted,
            "best_distance": best_distance,
            "improvement": improvement,
        })
        emit(
            f"epoch={epoch} accepted={epoch_accepted} "
            f"best_w1={best_distance:.6g}"
        )
        if improvement > material:
            stagnant = 0
        else:
            stagnant += 1
        if epoch >= config.min_epochs and stagnant >= config.patience:
            termination = "material_improvement_plateau"
            break

    certified_best = aggregate_cdf(store, best_rows, points)
    certified_best_distance = wasserstein_1(
        certified_best, bootstrap_target, age_bins
    )
    if not np.isclose(
        certified_best_distance, best_distance, rtol=1e-10, atol=1e-8
    ):
        raise RuntimeError("best-state certification changed its W1 distance")
    return RestartResult(
        seed_index=seed_index,
        restart_kind=restart_kind,
        seed=seed,
        rows=best_rows,
        cdf=certified_best,
        initial_distance=initial_distance,
        best_distance=certified_best_distance,
        match_to_observed=wasserstein_1(
            certified_best, observed_target, age_bins
        ),
        epochs=len(trace),
        proposals=proposals,
        accepted=accepted,
        termination=termination,
        trace=trace,
        elapsed_seconds=time.perf_counter() - started,
    )


def validate_restart_result(
    result: RestartResult,
    *,
    store: object,
    candidates: np.ndarray,
    bootstrap_target: np.ndarray,
    observed_target: np.ndarray,
    age_bins: np.ndarray,
    expected_seed_index: int,
    expected_kind: str,
    expected_seed: int,
) -> None:
    rows = np.asarray(result.rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("restart result rows must be nonempty and unique")
    if not np.all(np.isin(rows, candidates)):
        raise ValueError("restart result contains rows outside the candidate universe")
    if (
        result.seed_index != expected_seed_index
        or result.restart_kind != expected_kind
        or result.seed != expected_seed
    ):
        raise ValueError("restart result seed identity differs")
    certified = aggregate_cdf(store, rows, analysis_points(age_bins))
    if not np.allclose(result.cdf, certified, rtol=1e-10, atol=1e-8):
        raise ValueError("restart result CDF does not match its rows")
    best = wasserstein_1(certified, bootstrap_target, age_bins)
    observed = wasserstein_1(certified, observed_target, age_bins)
    if not np.isclose(result.best_distance, best, rtol=1e-10, atol=1e-8):
        raise ValueError("restart result bootstrap distance does not certify")
    if not np.isclose(result.match_to_observed, observed, rtol=1e-10, atol=1e-8):
        raise ValueError("restart result observed distance does not certify")
    trace_best = np.asarray([float(record["best_distance"]) for record in result.trace])
    if trace_best.size != result.epochs or np.any(np.diff(trace_best) > 1e-8):
        raise ValueError("restart result best-distance trace is invalid")


def target_digest_for(path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Return the project-wide target digest and the target arrays behind it.

    The digest must be byte-identical to the one `sample_age_matched_controls`
    records and `phi_sfs` recomputes, or the published bundle cannot be read
    downstream. It therefore uses that module's own loader and hash helper —
    including the acceptance threshold, which is part of the established
    four-array contract — rather than a local reimplementation.
    """
    rows, cdf, ages, threshold, metadata = _load_target(path)
    digest = _sha256_arrays(
        rows, cdf, ages, np.asarray([threshold], dtype=np.float64)
    )
    return digest, rows, cdf, ages, threshold, metadata


def _load_seed_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = np.load(path / "row_indices.npy", allow_pickle=False).astype(np.int64)
    cdfs = np.load(path / "cdfs.npy", allow_pickle=False).astype(np.float64)
    with (path / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if rows.ndim != 2 or rows.shape[0] == 0 or cdfs.shape[0] != rows.shape[0]:
        raise ValueError("seed rows and CDFs are incompatible")
    return rows, cdfs, metadata


def _validate_inputs(store: object, target_meta: dict, seed_meta: dict) -> None:
    actual_schema = store_schema(store)
    actual_content = getattr(store, "metadata", {}).get("content_sha256")
    actual_catalog = getattr(store, "metadata", {}).get("catalog_sha256")
    for label, metadata in (("target", target_meta), ("seed", seed_meta)):
        expected_schema = metadata.get("source_store_schema")
        expected_content = metadata.get("source_store_content_sha256")
        expected_catalog = metadata.get("source_catalog_sha256")
        if expected_schema is not None and expected_schema != actual_schema:
            raise ValueError(f"{label} and store schemas differ")
        if expected_content is not None and expected_content != actual_content:
            raise ValueError(f"{label} and store content identities differ")
        if expected_catalog is not None and expected_catalog != actual_catalog:
            raise ValueError(f"{label} and store catalogs differ")


def _candidate_rows(args: argparse.Namespace, store: object,
                    target_rows: np.ndarray) -> tuple[np.ndarray, str | None]:
    if args.candidate_rows is None:
        return eligible_candidates(store, target_rows, None), None
    raw = np.load(args.candidate_rows, allow_pickle=False)
    rows = eligible_candidates(store, target_rows, raw)
    return rows, _sha256_arrays(np.asarray(raw))


def _save_replicate_bundle(
    path: Path,
    *,
    counts: np.ndarray,
    target: np.ndarray,
    bootstrap_distance: float,
    results: list[RestartResult],
) -> None:
    metadata = []
    for result in results:
        metadata.append({
            "seed_index": result.seed_index,
            "restart_kind": result.restart_kind,
            "seed": result.seed,
            "initial_distance": result.initial_distance,
            "best_distance": result.best_distance,
            "match_to_observed": result.match_to_observed,
            "epochs": result.epochs,
            "proposals": result.proposals,
            "accepted": result.accepted,
            "termination": result.termination,
            "trace": result.trace,
            "elapsed_seconds": result.elapsed_seconds,
        })
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                counts=np.asarray(counts, dtype=np.uint32),
                target=np.asarray(target, dtype=np.float64),
                bootstrap_distance=np.asarray(bootstrap_distance),
                rows=np.stack([result.rows for result in results]),
                cdfs=np.stack([result.cdf for result in results]),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_replicate_bundle(
    path: Path,
    *,
    expected_counts: np.ndarray,
    expected_target: np.ndarray,
    expected_distance: float,
    expected_restarts: int,
) -> list[RestartResult]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            counts = archive["counts"]
            target = archive["target"]
            distance = float(archive["bootstrap_distance"])
            rows = archive["rows"]
            cdfs = archive["cdfs"]
            metadata = json.loads(str(archive["metadata"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid replicate bundle: {path}") from error
    if not np.array_equal(counts, expected_counts):
        raise ValueError(f"replicate bundle bootstrap counts differ: {path}")
    if not np.array_equal(target, expected_target):
        raise ValueError(f"replicate bundle bootstrap target differs: {path}")
    if not np.isclose(distance, expected_distance, rtol=0, atol=1e-10):
        raise ValueError(f"replicate bundle bootstrap distance differs: {path}")
    if (
        rows.ndim != 2 or cdfs.ndim != 2
        or rows.shape[0] != expected_restarts
        or cdfs.shape[0] != expected_restarts
        or not isinstance(metadata, list) or len(metadata) != expected_restarts
    ):
        raise ValueError(f"replicate bundle restart arrays differ: {path}")
    results = []
    for index, record in enumerate(metadata):
        results.append(RestartResult(
            seed_index=int(record["seed_index"]),
            restart_kind=str(record["restart_kind"]),
            seed=int(record["seed"]),
            rows=np.asarray(rows[index], dtype=np.int64),
            cdf=np.asarray(cdfs[index], dtype=np.float64),
            initial_distance=float(record["initial_distance"]),
            best_distance=float(record["best_distance"]),
            match_to_observed=float(record["match_to_observed"]),
            epochs=int(record["epochs"]),
            proposals=int(record["proposals"]),
            accepted=int(record["accepted"]),
            termination=str(record["termination"]),
            trace=list(record["trace"]),
            elapsed_seconds=float(record["elapsed_seconds"]),
        ))
    return results


def _write_outputs(
    output: Path,
    *,
    store: object,
    target_path: Path,
    seed_path: Path,
    target_rows: np.ndarray,
    observed_target: np.ndarray,
    age_bins: np.ndarray,
    acceptance_threshold: float,
    counts: np.ndarray,
    bootstrap_targets: np.ndarray,
    bootstrap_distances: np.ndarray,
    bootstrap_seeds: np.ndarray,
    all_restarts: list[list[RestartResult]],
    selected_restart: np.ndarray,
    config: OptimizerConfig,
    global_seed: int,
    candidate_digest: str | None,
    seed_sets_digest: str,
    target_digest: str,
    store_dir: Path,
    elapsed: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        replicate_count = len(all_restarts)
        restart_count = config.closest_restarts + config.diverse_restarts
        selected = [all_restarts[r][selected_restart[r]] for r in range(replicate_count)]
        rows = np.stack([result.rows for result in selected])
        cdfs = np.stack([result.cdf for result in selected])
        match_bootstrap = np.asarray([result.best_distance for result in selected])
        match_observed = np.asarray([result.match_to_observed for result in selected])
        ratios = np.divide(
            match_bootstrap, bootstrap_distances,
            out=np.full_like(match_bootstrap, np.inf),
            where=bootstrap_distances > 0,
        )
        # The absolute criterion guards against a large matching error that a
        # large bootstrap displacement would hide from the ratio. Its scale has
        # to be the target's own, because E_r grows as targets shrink -- roughly
        # n^-0.34 measured across 600 to 35,466 sites -- while a fixed cap does
        # not. A 500-generation cap calibrated on the 4,067-site in-gene target
        # (threshold 1480.5) is 0.34 of that threshold, and rejects three
        # quarters of replicates at 600 sites purely from size. Passing
        # --qc-max-absolute restores a fixed cap for a prespecified analysis.
        absolute_cap = (
            config.qc_max_absolute if config.qc_max_absolute is not None
            else config.qc_max_absolute_fraction * acceptance_threshold
        )
        qc = (
            (ratios < config.qc_max_ratio)
            & (match_bootstrap <= absolute_cap)
        )
        triangle_ok = (
            match_observed + 1e-8 >= np.abs(bootstrap_distances - match_bootstrap)
        ) & (
            match_observed <= bootstrap_distances + match_bootstrap + 1e-8
        )
        if not np.all(triangle_ok):
            raise RuntimeError("one or more replicate distances violate triangle inequality")

        restart_rows = np.stack([
            np.stack([result.rows for result in replicate])
            for replicate in all_restarts
        ])
        restart_cdfs = np.stack([
            np.stack([result.cdf for result in replicate])
            for replicate in all_restarts
        ])
        maximum_epochs = max(
            len(result.trace) for replicate in all_restarts for result in replicate
        )
        trace_distance = np.full(
            (replicate_count, restart_count, maximum_epochs), np.nan
        )
        trace_improvement = np.full_like(trace_distance, np.nan)
        trace_accepted = np.full_like(trace_distance, -1, dtype=np.int64)
        trace_proposals = np.full_like(trace_accepted, -1)
        for r, replicate in enumerate(all_restarts):
            for j, result in enumerate(replicate):
                for e, record in enumerate(result.trace):
                    trace_distance[r, j, e] = float(record["best_distance"])
                    trace_improvement[r, j, e] = float(record["improvement"])
                    trace_accepted[r, j, e] = int(record["accepted"])
                    trace_proposals[r, j, e] = int(record["proposals"])

        # Per-restart diagnostics, so a rejected restart can be inspected
        # without retaining the work directory.
        def per_restart(getter, dtype=np.float64):
            return np.asarray(
                [[getter(result) for result in replicate] for replicate in all_restarts],
                dtype=dtype,
            )

        restart_best = per_restart(lambda result: result.best_distance)
        restart_observed = per_restart(lambda result: result.match_to_observed)
        restart_ratio = np.divide(
            restart_best, bootstrap_distances[:, None],
            out=np.full_like(restart_best, np.inf),
            where=bootstrap_distances[:, None] > 0,
        )

        arrays = {
            "replicate_id.npy": np.arange(replicate_count, dtype=np.int64),
            "bootstrap_seeds.npy": np.asarray(bootstrap_seeds, dtype=np.uint64),
            "restart_seeds.npy": per_restart(lambda r: r.seed, np.uint64),
            "restart_seed_indices.npy": per_restart(lambda r: r.seed_index, np.int64),
            "restart_initial_w1.npy": per_restart(lambda r: r.initial_distance),
            "restart_match_to_bootstrap_w1.npy": restart_best,
            "restart_match_to_observed_w1.npy": restart_observed,
            "restart_matching_error_ratio.npy": restart_ratio,
            "restart_qc_pass.npy": (
                (restart_ratio < config.qc_max_ratio)
                & (restart_best <= absolute_cap)
            ),
            "restart_elapsed_seconds.npy": per_restart(lambda r: r.elapsed_seconds),
            "restart_trace_proposals.npy": trace_proposals,
            "row_indices.npy": rows,
            "cdfs.npy": cdfs,
            "target_cdf.npy": observed_target,
            "age_bins.npy": age_bins,
            "bootstrap_counts.npy": counts,
            "bootstrap_target_cdfs.npy": bootstrap_targets,
            "bootstrap_to_observed_w1.npy": bootstrap_distances,
            "match_to_bootstrap_w1.npy": match_bootstrap,
            "match_to_observed_w1.npy": match_observed,
            "matching_error_ratio.npy": ratios,
            "qc_pass.npy": qc,
            "triangle_ok.npy": triangle_ok,
            "selected_restart.npy": selected_restart,
            "restart_best_rows.npy": restart_rows,
            "restart_best_cdfs.npy": restart_cdfs,
            "restart_trace_w1.npy": trace_distance,
            "restart_trace_improvement.npy": trace_improvement,
            "restart_trace_accepted.npy": trace_accepted,
        }
        native_chromosomes, native_positions = store.rows_to_native(rows.ravel())
        labels, codes = np.unique(native_chromosomes, return_inverse=True)
        arrays["positions.npy"] = native_positions.reshape(rows.shape)
        arrays["chromosome_labels.npy"] = labels
        arrays["chromosome_codes.npy"] = codes.astype(np.uint16).reshape(rows.shape)
        unique_rows, reuse = np.unique(rows, return_counts=True)
        arrays["reuse_row_indices.npy"] = unique_rows
        arrays["reuse_counts.npy"] = reuse.astype(np.uint16)
        for name, values in arrays.items():
            np.save(staging / name, np.asarray(values), allow_pickle=False)

        with (staging / "replicates.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "replicate", "selected_restart", "seed_index", "restart_kind",
                "bootstrap_to_observed_w1", "match_to_bootstrap_w1",
                "match_to_observed_w1", "matching_error_ratio", "qc_pass",
                "triangle_ok", "epochs", "proposals", "accepted", "termination",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r, result in enumerate(selected):
                writer.writerow({
                    "replicate": r,
                    "selected_restart": int(selected_restart[r]),
                    "seed_index": result.seed_index,
                    "restart_kind": result.restart_kind,
                    "bootstrap_to_observed_w1": bootstrap_distances[r],
                    "match_to_bootstrap_w1": match_bootstrap[r],
                    "match_to_observed_w1": match_observed[r],
                    "matching_error_ratio": ratios[r],
                    "qc_pass": bool(qc[r]),
                    "triangle_ok": bool(triangle_ok[r]),
                    "epochs": result.epochs,
                    "proposals": result.proposals,
                    "accepted": result.accepted,
                    "termination": result.termination,
                })

        with (staging / "restarts.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "replicate", "restart", "selected", "seed_index", "restart_kind",
                "seed", "initial_w1", "best_w1", "match_to_observed_w1",
                "epochs", "proposals", "accepted", "termination",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r, replicate in enumerate(all_restarts):
                for j, result in enumerate(replicate):
                    writer.writerow({
                        "replicate": r,
                        "restart": j,
                        "selected": j == selected_restart[r],
                        "seed_index": result.seed_index,
                        "restart_kind": result.restart_kind,
                        "seed": result.seed,
                        "initial_w1": result.initial_distance,
                        "best_w1": result.best_distance,
                        "match_to_observed_w1": result.match_to_observed,
                        "epochs": result.epochs,
                        "proposals": result.proposals,
                        "accepted": result.accepted,
                        "termination": result.termination,
                    })

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "software": software_provenance(),
            "algorithm_version": ALGORITHM_VERSION,
            "creation_command": " ".join(sys.argv),
            "target": str(target_path.resolve()),
            "seed_sets": str(seed_path.resolve()),
            "source_store": str(Path(store_dir).resolve()),
            "source_store_schema": store_schema(store),
            "source_catalog_sha256": getattr(store, "metadata", {}).get("catalog_sha256"),
            "source_store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
            "target_digest": target_digest,
            "candidate_rows_digest": candidate_digest,
            "seed_sets_digest": seed_sets_digest,
            "replicate_identifiers": ["replicate_id"],
            "global_seed": global_seed,
            "replicates": replicate_count,
            "set_size": int(target_rows.size),
            "restarts_per_replicate": restart_count,
            "qc_absolute_cap_generations": float(absolute_cap),
            "qc_absolute_cap_source": (
                "fixed --qc-max-absolute" if config.qc_max_absolute is not None
                else f"{config.qc_max_absolute_fraction} x acceptance threshold"
            ),
            "qc_passes": int(qc.sum()),
            "qc_failures": int((~qc).sum()),
            "qc_interpretation": "optimizer convergence diagnostic, not biological validation",
            "bootstrap_kind": "iid multinomial TE-site bootstrap",
            "bootstrap_linkage_warning": (
                "inferential use requires exchangeability support or replacement "
                "with a prespecified genomic-block bootstrap"
            ),
            "selection_rule": "minimum certified W1 across prespecified restarts",
            "phi_sfs_selection_blind": True,
            "config": asdict(config),
            "elapsed_seconds": elapsed,
            "unique_controls_across_sets": int(unique_rows.size),
            "maximum_control_reuse": int(reuse.max()),
        }
        with (staging / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config = OptimizerConfig(
        replicates=args.replicates,
        closest_restarts=args.closest_restarts,
        diverse_restarts=args.diverse_restarts,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        material_improvement_ratio=args.material_improvement_ratio,
        cdf_block_rows=args.cdf_block_rows,
        search_bin_width=args.search_bin_width,
        qc_max_ratio=args.qc_max_ratio,
        qc_max_absolute=args.qc_max_absolute,
        qc_max_absolute_fraction=args.qc_max_absolute_fraction,
    )
    config.validate()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    store = open_snp_age_store(args.store)
    if not is_interval_store(store):
        raise ValueError("bootstrap-target optimization requires an interval store")
    (target_digest, target_rows, observed_target, age_bins,
     acceptance_threshold, target_meta) = target_digest_for(args.target)
    seed_rows, seed_cdfs, seed_meta = _load_seed_bundle(args.seed_sets)
    _validate_inputs(store, target_meta, seed_meta)
    if seed_rows.shape[1] != target_rows.size or seed_cdfs.shape[1:] != observed_target.shape:
        raise ValueError("seed sets do not match target size or age grid")
    candidates, candidate_digest = _candidate_rows(args, store, target_rows)
    if candidates.size <= target_rows.size:
        raise ValueError("candidate universe must exceed target set size")
    if not np.all(np.isin(seed_rows, candidates)):
        raise ValueError("one or more seed rows are outside the candidate universe")
    points = analysis_points(age_bins)
    exact_step = float(age_bins[1] - age_bins[0])
    if config.search_bin_width < exact_step:
        raise ValueError("search_bin_width cannot be finer than the exact target grid")
    coarse_ages, coarse_points = search_grid(
        float(store.metadata["maximum_above"]), config.search_bin_width
    )
    te_cdf_rows = row_cdfs(
        store, target_rows, points,
        block_rows=config.cdf_block_rows, dtype=np.dtype("float32"),
    )
    reconstructed = te_cdf_rows.mean(axis=0, dtype=np.float64)
    if not np.allclose(reconstructed, observed_target, rtol=1e-6, atol=1e-7):
        raise ValueError("target CDF does not match exact TE-row reconstruction")
    te_coarse_rows = row_cdfs(
        store, target_rows, coarse_points,
        block_rows=config.cdf_block_rows, dtype=np.dtype("float32"),
    )
    work_dir = (
        args.work_dir if args.work_dir is not None
        else args.output.with_name(f".{args.output.name}.work")
    )
    resolved_work, resolved_output = work_dir.resolve(), args.output.resolve()
    if resolved_work == resolved_output:
        raise ValueError("--work-dir and --output must be different paths")
    if resolved_output.is_relative_to(resolved_work):
        raise ValueError("--output must not be inside --work-dir; it would be "
                         "published and then deleted with the work directory")
    if resolved_work.is_relative_to(resolved_output):
        raise ValueError("--work-dir must not be inside --output")
    seed_sets_digest = _sha256_arrays(seed_rows, seed_cdfs)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        # A resumed run must not mix bundles produced by two implementations,
        # so the identity pins the checkout and NumPy, not just the declared
        # algorithm version.
        "software": software_provenance(),
        "numpy_version": np.__version__,
        "target_digest": target_digest,
        "source_store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
        "candidate_rows_digest": candidate_digest,
        "seed_sets_digest": seed_sets_digest,
        "global_seed": args.seed,
        "config": asdict(config),
    }
    identity_path = work_dir / "identity.json"
    if work_dir.exists():
        if not args.resume:
            raise FileExistsError(f"work directory already exists: {work_dir}")
        try:
            existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid work identity: {identity_path}") from error
        if existing_identity != identity:
            raise ValueError("existing work directory parameters or provenance differ")
    else:
        work_dir.mkdir(parents=True)
        identity_path.write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    bundle_dir = work_dir / "replicates"
    bundle_dir.mkdir(exist_ok=True)
    counts = np.empty((config.replicates, target_rows.size), dtype=np.uint32)
    bootstrap_targets = np.empty(
        (config.replicates, observed_target.size), dtype=np.float64
    )
    bootstrap_distances = np.empty(config.replicates, dtype=np.float64)
    bootstrap_seeds = np.empty(config.replicates, dtype=np.uint64)
    all_restarts: list[list[RestartResult]] = []
    selected_restart = np.empty(config.replicates, dtype=np.int64)

    for replicate in range(config.replicates):
        bootstrap_seed = derive_seed(args.seed, target_digest, replicate)
        bootstrap_seeds[replicate] = bootstrap_seed
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        counts[replicate] = bootstrap_counts(target_rows.size, bootstrap_rng)
        bootstrap_targets[replicate] = bootstrap_cdf(
            counts[replicate], te_cdf_rows
        )
        coarse_target = bootstrap_cdf(counts[replicate], te_coarse_rows)
        bootstrap_distances[replicate] = wasserstein_1(
            bootstrap_targets[replicate], observed_target, age_bins
        )
        choices = select_seed_indices(
            seed_cdfs, bootstrap_targets[replicate], age_bins,
            closest=config.closest_restarts,
            diverse=config.diverse_restarts,
            rng=bootstrap_rng,
        )
        results: list[RestartResult] = []
        bundle_path = bundle_dir / f"replicate-{replicate:04d}.npz"
        print(
            f"replicate={replicate} bootstrap_w1={bootstrap_distances[replicate]:.6g}",
            flush=True,
        )
        resumed_bundle = bundle_path.exists()
        if resumed_bundle:
            if not args.resume:
                raise FileExistsError(f"replicate bundle already exists: {bundle_path}")
            results = _load_replicate_bundle(
                bundle_path,
                expected_counts=counts[replicate],
                expected_target=bootstrap_targets[replicate],
                expected_distance=bootstrap_distances[replicate],
                expected_restarts=len(choices),
            )
            print(f"replicate={replicate} resumed complete bundle", flush=True)
        else:
            for restart, (seed_index, kind) in enumerate(choices):
                restart_seed = derive_seed(
                    args.seed, target_digest, replicate, restart
                )
                result = optimize_restart(
                    store, candidates, seed_rows[seed_index],
                    bootstrap_targets[replicate], observed_target, age_bins,
                    bootstrap_distances[replicate],
                    coarse_target=coarse_target,
                    coarse_ages=coarse_ages,
                    coarse_points=coarse_points,
                    seed_index=seed_index,
                    restart_kind=kind,
                    seed=restart_seed,
                    config=config,
                    progress=lambda message, r=replicate, j=restart: print(
                        f"replicate={r} restart={j} {message}", flush=True
                    ),
                )
                results.append(result)
        for restart, ((seed_index, kind), result) in enumerate(zip(choices, results)):
            validate_restart_result(
                result,
                store=store,
                candidates=candidates,
                bootstrap_target=bootstrap_targets[replicate],
                observed_target=observed_target,
                age_bins=age_bins,
                expected_seed_index=seed_index,
                expected_kind=kind,
                expected_seed=derive_seed(
                    args.seed, target_digest, replicate, restart
                ),
            )
        if not resumed_bundle:
            _save_replicate_bundle(
                bundle_path,
                counts=counts[replicate],
                target=bootstrap_targets[replicate],
                bootstrap_distance=bootstrap_distances[replicate],
                results=results,
            )
        all_restarts.append(results)
        selected_restart[replicate] = int(np.argmin([
            result.best_distance for result in results
        ]))

    _write_outputs(
        args.output,
        store=store,
        target_path=args.target,
        seed_path=args.seed_sets,
        target_rows=target_rows,
        observed_target=observed_target,
        age_bins=age_bins,
        acceptance_threshold=acceptance_threshold,
        counts=counts,
        bootstrap_targets=bootstrap_targets,
        bootstrap_distances=bootstrap_distances,
        bootstrap_seeds=bootstrap_seeds,
        all_restarts=all_restarts,
        selected_restart=selected_restart,
        config=config,
        global_seed=args.seed,
        candidate_digest=candidate_digest,
        seed_sets_digest=seed_sets_digest,
        target_digest=target_digest,
        store_dir=getattr(store, "store_dir", args.store),
        elapsed=time.perf_counter() - started,
    )
    if not args.keep_work:
        shutil.rmtree(work_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--seed-sets", type=Path, required=True)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-rows", type=Path)
    candidates.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--closest-restarts", type=int, default=2)
    parser.add_argument("--diverse-restarts", type=int, default=1)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--material-improvement-ratio", type=float, default=1e-3)
    parser.add_argument("--cdf-block-rows", type=int, default=256)
    parser.add_argument(
        "--search-bin-width", type=int, default=20_000,
        help="coarse grid width in generations used to screen swaps; every "
             "reported distance is still certified on the exact grid",
    )
    parser.add_argument("--qc-max-ratio", type=float, default=0.5)
    parser.add_argument(
        "--qc-max-absolute", type=float, default=None,
        help="fixed absolute cap on E_r in generations; overrides the "
             "threshold-scaled default and does not adapt to target size",
    )
    parser.add_argument(
        "--qc-max-absolute-fraction", type=float, default=0.34,
        help="absolute E_r cap as a fraction of the target acceptance "
             "threshold (default 0.34, which reproduces the historical "
             "500-generation cap at the in-gene target)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    print(f"Wrote bootstrap-target matches to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
