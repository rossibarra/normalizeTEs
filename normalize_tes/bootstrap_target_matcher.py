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

from .release_provenance import loaded_source_digest, software_provenance
from .sample_age_matched_controls import _load_target, _sha256_arrays
from .snp_age_store import is_interval_store, open_snp_age_store, store_schema
from .swap_control_sampler import (
    aggregate_cdf,
    analysis_points,
    eligible_candidates,
    incremental_cdf,
    row_cdfs,
)
from .te_age_target import (
    masked_row_cdfs,
    wasserstein_1,
)


SCHEMA_VERSION = "bootstrap-target-matches-v1"
ALGORITHM_VERSION = "bootstrap-target-exact-greedy-v1"


@dataclass(frozen=True)
class OptimizerConfig:
    replicates: int = 100
    restarts: int = 3
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
    disjoint_replicates: bool = False
    algorithm_version: str = ALGORITHM_VERSION

    def validate(self) -> None:
        if self.replicates <= 0:
            raise ValueError("replicates must be positive")
        if self.restarts <= 0:
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


def stratified_initial_set(
    store: object, boundary_ages: np.ndarray, quotas: np.ndarray,
    candidates: np.ndarray, rng: np.random.Generator, *,
    oversample: int = 20,
) -> np.ndarray:
    """Draw an initial set matching the target's equal-mass age strata.

    The target bundle ships `interval_boundary_ages` and `interval_quotas`: 20
    equal-mass age strata and how many TE sites fall in each. Filling those
    quotas from the candidate pool gives a starting state already shaped like
    the target, which is what the hard-q50 sampler's construction phase
    produced. Building it here removes the dependency on a hard-q50 seed
    library entirely.

    Candidates are assigned to a stratum by their median age, read off their
    CDF at the 21 boundary ages rather than on the full analysis grid, so this
    costs one narrow read per sampled candidate. A pool of `oversample` times
    the target size is drawn first; any stratum the draw underfills is topped up
    from the unused remainder, so the returned set always has exactly the
    required size even where the pool is thin.
    """
    n_target = int(quotas.sum())
    pool = rng.choice(candidates, size=min(candidates.size, n_target * oversample),
                      replace=False)
    at_boundary = row_cdfs(store, pool, np.asarray(boundary_ages, dtype=np.float64),
                           block_rows=4096, dtype=np.dtype("float32"))
    # median age = first boundary whose CDF reaches 0.5
    reached = at_boundary >= 0.5
    stratum = np.where(reached.any(axis=1), reached.argmax(axis=1) - 1,
                       quotas.size - 1)
    stratum = np.clip(stratum, 0, quotas.size - 1)

    chosen: list[np.ndarray] = []
    used = np.zeros(pool.size, dtype=bool)
    for k, want in enumerate(quotas):
        avail = np.flatnonzero((stratum == k) & ~used)
        take = avail[:int(want)] if avail.size >= want else avail
        used[take] = True
        chosen.append(pool[take])
    have = int(sum(c.size for c in chosen))
    if have < n_target:
        spare = np.flatnonzero(~used)
        chosen.append(pool[spare[:n_target - have]])
    out = np.concatenate(chosen)[:n_target]
    if out.size < n_target:
        raise ValueError(
            f"stratified initialisation drew {out.size:,} of {n_target:,} rows; "
            "raise --init-oversample or widen the candidate pool"
        )
    return np.sort(out)


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
    expected_seed: int,
) -> None:
    rows = np.asarray(result.rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("restart result rows must be nonempty and unique")
    if not np.all(np.isin(rows, candidates)):
        raise ValueError("restart result contains rows outside the candidate universe")
    if result.seed != expected_seed:
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


def _validate_inputs(store: object, target_meta: dict) -> None:
    actual_schema = store_schema(store)
    actual_content = getattr(store, "metadata", {}).get("content_sha256")
    actual_catalog = getattr(store, "metadata", {}).get("catalog_sha256")
    expected_schema = target_meta.get("source_store_schema")
    expected_content = target_meta.get("source_store_content_sha256")
    expected_catalog = target_meta.get("source_catalog_sha256")
    if expected_schema is not None and expected_schema != actual_schema:
        raise ValueError("target and store schemas differ")
    if expected_content is not None and expected_content != actual_content:
        raise ValueError("target and store content identities differ")
    if expected_catalog is not None and expected_catalog != actual_catalog:
        raise ValueError("target and store catalogs differ")


def _authenticate_candidate_rows(path: Path, store: object,
                                 raw: np.ndarray) -> dict:
    """Check the candidate array against its provenance report.

    Row indices are store-specific: the same integers name different SNPs in a
    different store. Checking only that the rows are in range and eligible --
    which is all the array itself supports -- accepts a candidate universe
    built against another store whenever the row counts happen to be
    compatible. `normalize_tes.build_candidate_rows` writes a report recording which store
    it was built from; authenticating against it is what makes the rows mean
    what the matcher assumes.
    """
    report_path = path.with_suffix(path.suffix + ".json")
    if not report_path.exists():
        raise SystemExit(
            f"{path} has no provenance report at {report_path.name}. The report "
            "is part of the candidate artifact: without it the rows cannot be "
            "tied to this store. Rebuild with normalize_tes.build_candidate_rows."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = getattr(store, "metadata", {}) or {}
    for key, actual, label in (
        ("store_content_sha256", metadata.get("content_sha256"), "content"),
        ("store_catalog_sha256", metadata.get("catalog_sha256"), "catalog"),
    ):
        recorded = report.get(key)
        # A missing digest on either side is not a pass. Row indices carry no
        # self-describing identity, so an unauthenticated array is exactly the
        # case this check exists for.
        if not recorded or not actual:
            raise SystemExit(
                f"{report_path.name} or the store is missing the {label} digest, "
                "so the candidate rows cannot be authenticated against this store."
            )
        if recorded != actual:
            raise SystemExit(
                f"{path} was built against a different store ({label} digest "
                f"{recorded[:12]} != {actual[:12]}). Its row indices name "
                "different SNPs here."
            )
    recorded_count = report.get("candidate_rows")
    if recorded_count is not None and int(recorded_count) != int(raw.size):
        raise SystemExit(
            f"{path} holds {raw.size:,} rows but its report records "
            f"{int(recorded_count):,}; the array and its provenance disagree."
        )
    recorded_digest = report.get("candidate_rows_sha256")
    if recorded_digest:
        # The builder's own digest convention, which differs from
        # _sha256_arrays: it hashes str(dtype) where that hashes dtype.str.
        if _candidate_array_digest(raw) != recorded_digest:
            raise SystemExit(
                f"{path} does not match the digest in {report_path.name}; the "
                "array has been modified since it was published."
            )
    return report


def _candidate_array_digest(values: np.ndarray) -> str:
    """Reproduce `build_candidate_rows._sha256_array` exactly."""
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(values)
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _candidate_rows(args: argparse.Namespace, store: object,
                    target_rows: np.ndarray) -> tuple[np.ndarray, str | None]:
    if args.candidate_rows is None:
        return eligible_candidates(store, target_rows, None), None
    raw = np.load(args.candidate_rows, allow_pickle=False)
    _authenticate_candidate_rows(Path(args.candidate_rows), store, np.asarray(raw))
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


SEARCH_LOG_OFFSET = 1_000.0


def log_search_grid(
    age_bins: np.ndarray, points: np.ndarray, n_points: int, offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample the exact grid geometrically in ``age + offset``.

    The linear coarse grid used to screen swaps is uniform at
    ``search_bin_width`` generations -- 20,000 by default -- which puts the
    whole lower quartile of a production TE age distribution inside its first
    cell -- measured at 50.06% of the age mass on the in-gene target, with 35
    of 1,837 cells holding 99%. The screen is therefore blind to structure in
    the lower half of the distribution, and rejects young-improving swaps before
    the exact grid ever evaluates them. Measured cost of that blindness: 31.5%
    relative error at the 10% CDF quantile, falling to 0.0% once the screen is
    log-spaced, with no loss at the old end.

    So the coarse grid is a geometric sub-sample of the exact
    analysis grid instead of a coarser uniform one. Every coarse point is an
    exact-grid point, so the coarse objective is a strict sub-sample of the
    exact objective rather than a different discretization, and the young end
    is retained at full exact resolution (geometric spacing below one bin
    width collapses to consecutive indices). ``n_points`` bounds the request;
    the returned grid is usually smaller after deduplication, and never
    larger, so per-proposal cost stays at or below the linear screen's.
    """
    ages = np.asarray(age_bins, dtype=np.float64)
    exact_points = np.asarray(points, dtype=np.float64)
    if ages.ndim != 1 or ages.size < 2 or exact_points.shape != ages.shape:
        raise ValueError("age bins and analysis points are incompatible")
    if n_points < 2:
        raise ValueError("log search grid needs at least two points")
    requested = np.geomspace(
        ages[0] + offset, ages[-1] + offset, num=int(n_points)
    ) - offset
    indices = np.unique(
        np.clip(np.searchsorted(ages, requested, side="left"), 0, ages.size - 1)
    )
    indices = np.union1d(indices, np.asarray([0, ages.size - 1], dtype=indices.dtype))
    return ages[indices], exact_points[indices]


def _write_outputs(
    output: Path,
    *,
    store: object,
    target_path: Path,
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
        restart_count = config.restarts
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
                "replicate", "selected_restart",
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
                "replicate", "restart", "selected",
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
            "initialisation": (
                "stratified draw from the target's equal-mass age strata"
            ),
            "source_store": str(Path(store_dir).resolve()),
            "source_store_schema": store_schema(store),
            "source_catalog_sha256": getattr(store, "metadata", {}).get("catalog_sha256"),
            "source_store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
            "target_digest": target_digest,
            "candidate_rows_digest": candidate_digest,
            "replicate_identifiers": ["replicate_id"],
            "global_seed": global_seed,
            "replicates": replicate_count,
            "set_size": int(target_rows.size),
            "restarts_per_replicate": restart_count,
            "search_grid_spacing": "log",
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
        restarts=args.restarts,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        material_improvement_ratio=args.material_improvement_ratio,
        cdf_block_rows=args.cdf_block_rows,
        search_bin_width=args.search_bin_width,
        qc_max_ratio=args.qc_max_ratio,
        qc_max_absolute=args.qc_max_absolute,
        qc_max_absolute_fraction=args.qc_max_absolute_fraction,
        disjoint_replicates=args.disjoint_replicates,
    )
    config.validate()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    store = open_snp_age_store(args.store)
    if not is_interval_store(store):
        raise ValueError("bootstrap-target optimization requires an interval store")
    (target_digest, target_rows, observed_target, age_bins,
     acceptance_threshold, target_meta) = target_digest_for(args.target)
    boundary_ages = np.load(
        args.target / "interval_boundary_ages.npy", allow_pickle=False)
    quotas = np.load(args.target / "interval_quotas.npy", allow_pickle=False)
    if int(quotas.sum()) != target_rows.size:
        raise ValueError("target quotas do not sum to the target set size")
    _validate_inputs(store, target_meta)
    candidates, candidate_digest = _candidate_rows(args, store, target_rows)
    if candidates.size <= target_rows.size:
        raise ValueError("candidate universe must exceed target set size")
    points = analysis_points(age_bins)
    exact_step = float(age_bins[1] - age_bins[0])
    if config.search_bin_width < exact_step:
        raise ValueError("search_bin_width cannot be finer than the exact target grid")
    coarse_ages, coarse_points = log_search_grid(
        age_bins, points,
        int(float(store.metadata["maximum_above"]) // config.search_bin_width) + 1,
        SEARCH_LOG_OFFSET,
    )
    print(f"coarse_grid=log points={coarse_points.size}", flush=True)
    # A target built with a polarity mask defines each TE's age CDF over its
    # agreeing draws only. Rebuilding from the store without the mask would not
    # merely fail the reconstruction check below -- it would hand every one of
    # the bootstrap targets the mis-polarized ages the mask exists to remove,
    # while the observed target kept them out. So the mask is required, not
    # optional, whenever the target records one.
    keep_path = args.target / "te_keep_draws.npy"
    declares_mask = (target_meta.get("te_polarity") or None) is not None
    if declares_mask and not keep_path.exists():
        raise ValueError(
            f"{args.target} records a polarity mask in its metadata but has no "
            "te_keep_draws.npy. Its per-TE CDFs cannot be reproduced, and the "
            "bootstrap targets derived from them would silently disagree with "
            "the target. Rebuild the target with the current normalize_tes.te_age_target."
        )
    if keep_path.exists():
        keep_draws = np.load(keep_path, allow_pickle=False)
        if keep_draws.shape[0] != target_rows.size:
            raise ValueError(
                "te_keep_draws.npy does not align with the target's TE rows"
            )
        te_cdf_rows = masked_row_cdfs(
            store, target_rows, points, keep_draws,
        ).astype(np.float32)
    else:
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
    identity = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        # A resumed run must not mix bundles produced by two implementations,
        # so the identity pins the checkout and NumPy, not just the declared
        # algorithm version. On a dirty checkout the commit is not enough --
        # two different sets of uncommitted edits share it -- so the loaded
        # project modules are hashed as well.
        "software": software_provenance(),
        "source_digest": loaded_source_digest(),
        "numpy_version": np.__version__,
        "target_digest": target_digest,
        "source_store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
        "candidate_rows_digest": candidate_digest,
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

    claimed: set[int] = set()
    claimed_arr = np.empty(0, dtype=np.int64)
    for replicate in range(config.replicates):
        # Disjoint mode removes every row an earlier replicate published, so the
        # published sets share no controls. That is zero membership OVERLAP, not
        # zero dependence: each replicate draws from a pool the earlier ones
        # depleted, and all of them bootstrap the same observed TE sample.
        # Stratum depth makes the constraint cheap -- the scarcest age decile
        # holds ~787 sets' worth of candidates against the 100 needed.
        if config.disjoint_replicates and claimed_arr.size:
            replicate_candidates = candidates[~np.isin(candidates, claimed_arr)]
            if replicate_candidates.size < target_rows.size:
                raise ValueError(
                    f"disjoint mode exhausted the candidate pool at replicate "
                    f"{replicate}: {replicate_candidates.size:,} left, "
                    f"{target_rows.size:,} needed"
                )
        else:
            replicate_candidates = candidates
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
                expected_restarts=config.restarts,
            )
            print(f"replicate={replicate} resumed complete bundle", flush=True)
        else:
            for restart in range(config.restarts):
                restart_seed = derive_seed(
                    args.seed, target_digest, replicate, restart
                )
                # Every restart is an independent stratified draw from this
                # replicate's own candidate universe, which in disjoint mode
                # already excludes every row an earlier replicate published, so
                # the starting state is legal by construction.
                initial = stratified_initial_set(
                    store, boundary_ages, quotas, replicate_candidates,
                    np.random.default_rng(restart_seed),
                    oversample=args.init_oversample,
                )
                result = optimize_restart(
                    store, replicate_candidates, initial,
                    bootstrap_targets[replicate], observed_target, age_bins,
                    bootstrap_distances[replicate],
                    coarse_target=coarse_target,
                    coarse_ages=coarse_ages,
                    coarse_points=coarse_points,
                    seed=restart_seed,
                    config=config,
                    progress=lambda message, r=replicate, j=restart: print(
                        f"replicate={r} restart={j} {message}", flush=True
                    ),
                )
                results.append(result)
        for restart, result in enumerate(results):
            validate_restart_result(
                result,
                store=store,
                candidates=replicate_candidates,
                bootstrap_target=bootstrap_targets[replicate],
                observed_target=observed_target,
                age_bins=age_bins,
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
        if config.disjoint_replicates:
            # Claim from the restart that will be published. Selection must
            # therefore be resolved per replicate here, not deferred, so the
            # next replicate sees the rows this one actually took.
            claimed.update(
                int(v) for v in results[selected_restart[replicate]].rows
            )
            claimed_arr = np.fromiter(claimed, dtype=np.int64, count=len(claimed))


    _write_outputs(
        args.output,
        store=store,
        target_path=args.target,
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
        target_digest=target_digest,
        store_dir=getattr(store, "store_dir", args.store),
        elapsed=time.perf_counter() - started,
    )
    if not args.keep_work:
        shutil.rmtree(work_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store supplying candidate SNP ages")
    parser.add_argument("--target", type=Path, required=True,
                        help="TE target directory from normalize_tes.te_age_target; supplies "
                             "the age CDF, the acceptance threshold and the strata")
    parser.add_argument("--init-oversample", type=int, default=20,
                        help="candidates drawn per required site when filling the "
                             "initial age strata. Larger fills the strata better "
                             "and costs one narrow store read per candidate")
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-rows", type=Path,
                            help="control universe from normalize_tes.build_candidate_rows, "
                                 "with all TE variants removed. This is the "
                                 "production setting")
    candidates.add_argument("--all-eligible", action="store_true",
                            help="use every eligible store row instead. This leaves "
                                 "TE variants in the control pool, so controls can "
                                 "be matched against other TEs")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination directory for the published sets")
    parser.add_argument("--work-dir", type=Path,
                        help="where completed replicate bundles are staged so an "
                             "interrupted run can be resumed")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from --work-dir. Input "
                             "identities and parameters must match or it stops")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the work directory after a successful publish")
    parser.add_argument("--replicates", type=int, default=100,
                        help="bootstrap replicates to match, one published control "
                             "set each")
    parser.add_argument(
        "--restarts", type=int, default=3,
        help="independent stratified restarts per replicate; the published set "
             "is the one with the smallest certified W1",
    )
    parser.add_argument("--min-epochs", type=int, default=10,
                        help="exact proposal epochs run before convergence may be "
                             "declared")
    parser.add_argument("--max-epochs", type=int, default=50,
                        help="hard ceiling on epochs per restart")
    parser.add_argument("--patience", type=int, default=5,
                        help="end a restart after this many consecutive epochs "
                             "without material improvement")
    parser.add_argument("--material-improvement-ratio", type=float, default=1e-3,
                        help="an epoch counts as improving only if W1 falls by this "
                             "fraction of B_r, so the bar scales with how far the "
                             "replicate moved the target")
    parser.add_argument("--cdf-block-rows", type=int, default=256,
                        help="rows per block when building CDFs; lower means lower "
                             "peak memory")
    parser.add_argument(
        "--search-bin-width", type=int, default=20_000,
        help="coarse grid width in generations used to screen swaps; every "
             "reported distance is still certified on the exact grid",
    )
    parser.add_argument("--qc-max-ratio", type=float, default=0.5,
                        help="QC gate on R_r = E_r/B_r: optimizer error as a "
                             "fraction of the displacement being reproduced. This "
                             "is convergence QC, not evidence of validity")
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
    parser.add_argument(
        "--disjoint-replicates", action="store_true",
        help="publish 100 mutually disjoint sets: each replicate is optimized "
             "against the candidate universe minus every row already published, "
             "so no control SNP appears in two published sets. This removes "
             "shared membership, not statistical dependence: replicates still "
             "share the observed TE sample and the store, and later sets draw "
             "from a pool the earlier ones depleted",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for bootstrap resampling, initialization and "
                             "proposals")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    print(f"Wrote bootstrap-target matches to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
