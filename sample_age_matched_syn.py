#!/usr/bin/env python3
"""Generate synonymous SNP sets matched to a target SNP age distribution.

Candidate weights are read from the age store once and materialized as a
float32 candidate-by-stratum matrix. Proposals reuse that matrix without
further boundary-CDF reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from snp_age_dataset import load_native_position_list
from snp_age_store import is_interval_store, open_snp_age_store, store_schema
from snp_position_resolution import resolve_native_position_requests
from te_age_target import wasserstein_1


class SamplingError(RuntimeError):
    """Raised when the requested matched sample cannot be constructed."""


@dataclass(frozen=True)
class CandidateWeights:
    candidate_rows: np.ndarray
    boundary_indices: np.ndarray
    values: np.ndarray
    boundary_ages: np.ndarray | None = None

    @property
    def n_intervals(self) -> int:
        return self.boundary_indices.size - 1


@dataclass(frozen=True)
class MatchResult:
    row_indices: np.ndarray
    chromosomes: np.ndarray
    positions: np.ndarray
    cdfs: np.ndarray
    wasserstein: np.ndarray
    interval_assignment: np.ndarray
    proposal_ids: np.ndarray
    attempts: int
    rejection_count: int


def _boundary_weights(store: object, rows: np.ndarray,
                      boundary_indices: np.ndarray,
                      boundary_ages: np.ndarray | None = None,
                      *, access_strategy: str = "gather",
                      block_rows: int = 100_000,
                      coalesce_gap: int = 64) -> np.ndarray:
    """Return interval masses for CDF *edge* indices in ``[0, B]``.

    Edge zero is the implicit CDF value zero; edge ``e > 0`` maps to stored
    CDF column ``e - 1``.  When available, the age-major boundary reader is
    used on one enclosing contiguous store-coordinate slab and then subset to
    the (sorted, possibly gapped) candidate rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    bounds = np.asarray(boundary_indices, dtype=np.int64)
    if is_interval_store(store):
        if boundary_ages is None:
            raise ValueError("interval stores require physical boundary ages")
        ages = np.asarray(boundary_ages, dtype=np.float64)
        if ages.shape != bounds.shape or np.any(np.diff(ages) < 0):
            raise ValueError("boundary ages must align and be nondecreasing")
        edge_cdfs = np.asarray(store.boundary_cdfs(
            rows, ages, side="left", weighting="interval",
            access_strategy=access_strategy, block_rows=block_rows,
            coalesce_gap=coalesce_gap), dtype=np.float64)
        weights = np.diff(edge_cdfs, axis=1)
        np.maximum(weights, 0.0, out=weights)
        return weights
    n_bins = int(np.asarray(store.age_bins).size)
    if np.any(bounds < 0) or np.any(bounds > n_bins):
        raise ValueError(f"boundary edge indices must lie in [0, {n_bins}]")
    positive_edges = np.unique(bounds[bounds > 0])
    edge_cdfs = np.zeros((rows.size, bounds.size), dtype=np.float64)
    if positive_edges.size:
        columns = positive_edges - 1
        slab_start, slab_stop = int(rows[0]), int(rows[-1]) + 1
        raw = np.asarray(store.read_boundary_cdfs(columns, slab_start, slab_stop))
        selected = raw[:, rows - slab_start].T
        lookup = {int(edge): selected[:, j]
                  for j, edge in enumerate(positive_edges)}
        for j, edge in enumerate(bounds):
            if edge:
                edge_cdfs[:, j] = lookup[int(edge)]
    weights = np.diff(edge_cdfs, axis=1)
    # Quantization or float roundoff can produce tiny negative differences.
    np.maximum(weights, 0.0, out=weights)
    return weights


def build_candidate_weights(
    store: object,
    syn_indices: np.ndarray,
    boundary_indices: np.ndarray,
    block_snps: int = 250_000,
    *,
    boundary_ages: np.ndarray | None = None,
    access_strategy: str = "gather",
    coalesce_gap: int = 64,
) -> CandidateWeights:
    """Read candidate interval weights once, using bounded input blocks."""
    rows = np.asarray(syn_indices, dtype=np.int64)
    bounds = np.asarray(boundary_indices, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("syn_indices must be a nonempty one-dimensional array")
    if np.unique(rows).size != rows.size:
        raise ValueError("syn_indices contains duplicates")
    if bounds.ndim != 1 or bounds.size < 2 or np.any(np.diff(bounds) < 0):
        raise ValueError("boundary_indices must be a nondecreasing 1-D array")
    if block_snps <= 0:
        raise ValueError("block_snps must be positive")
    if access_strategy == "auto":
        raise ValueError("auto access requires Gate 3 benchmark thresholds")
    if np.any(rows < 0) or np.any(rows >= np.asarray(store.positions).size):
        raise ValueError("syn_indices contains an out-of-range row index")
    rows = np.sort(rows)

    # Bound both candidate count and the enclosing store-coordinate read.  This
    # matters for sparse masks: 250k candidates might otherwise span nearly the
    # entire 20-million-row store in one age-major slab.
    starts_list: list[int] = []
    start = 0
    while start < rows.size:
        starts_list.append(start)
        coordinate_stop = int(np.searchsorted(
            rows, rows[start] + block_snps, side="left"))
        stop = min(start + block_snps, max(start + 1, coordinate_stop))
        start = stop
    starts = np.asarray(
        [0] if is_interval_store(store) and access_strategy == "scan"
        else starts_list,
        dtype=np.int64,
    )
    weights = np.empty((rows.size, bounds.size - 1), dtype=np.float32)
    for block, start in enumerate(starts):
        stop = (int(starts[block + 1]) if block + 1 < starts.size
                else rows.size)
        weights[start:stop] = _boundary_weights(
            store, rows[start:stop], bounds, boundary_ages,
            access_strategy=access_strategy, block_rows=block_snps,
            coalesce_gap=coalesce_gap).astype(np.float32)
    ages = None if boundary_ages is None else np.asarray(boundary_ages, dtype=np.float64)
    return CandidateWeights(rows.astype(np.int64, copy=False), bounds, weights, ages)


def draw_stratified_set(
    weights: CandidateWeights,
    quotas: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one unique candidate set, returning rows and stratum assignments."""
    quotas = np.asarray(quotas, dtype=np.int64)
    if quotas.shape != (weights.n_intervals,) or np.any(quotas < 0):
        raise ValueError("quotas must be nonnegative with one value per interval")
    if quotas.sum() > weights.candidate_rows.size:
        raise SamplingError("total quota exceeds the number of unique candidates")
    selected = np.zeros(weights.candidate_rows.size, dtype=np.bool_)
    out_rows: list[int] = []
    assignments: list[int] = []
    order = rng.permutation(weights.n_intervals)
    for interval in order:
        quota = int(quotas[interval])
        if quota == 0:
            continue
        probabilities = weights.values[:, interval].copy()
        probabilities[selected] = 0.0
        positive = np.count_nonzero(probabilities > 0)
        if positive == 0:
            raise SamplingError(f"interval {interval} has zero candidate mass")
        if positive < quota:
            raise SamplingError(
                f"interval {interval} quota {quota} exceeds its positive-mass "
                f"candidate count {positive} after earlier selections")
        probabilities /= probabilities.sum(dtype=np.float64)
        chosen = rng.choice(probabilities.size, size=quota, replace=False,
                            p=probabilities)
        selected[chosen] = True
        out_rows.extend(weights.candidate_rows[chosen].tolist())
        assignments.extend([int(interval)] * quota)
    return np.asarray(out_rows, dtype=np.int64), np.asarray(assignments,
                                                             dtype=np.uint16)


def score_set(store: object, row_indices: np.ndarray, target_cdf: np.ndarray,
              age_bins: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    ages = np.asarray(store.age_bins if age_bins is None else age_bins)
    if is_interval_store(store):
        widths = np.diff(ages.astype(np.float64))
        if ages.ndim != 1 or ages.size < 2 or not np.all(widths > 0):
            raise ValueError("target age grid must be strictly increasing")
        if not np.allclose(widths, widths[0], rtol=0, atol=0):
            raise ValueError("interval scoring requires a uniform target age grid")
        width = float(widths[0])
        aggregate = store.aggregate_cdf_at(
            row_indices, ages.astype(np.float64) + width / 2,
            side="left", weighting="interval")
    else:
        aggregate = store.read_cdfs(row_indices).mean(axis=0, dtype=np.float64)
    w1 = wasserstein_1(aggregate, target_cdf, ages)
    return aggregate, w1


def generate_matches(
    store: object,
    weights: CandidateWeights,
    quotas: np.ndarray,
    target_cdf: np.ndarray,
    threshold: float,
    *,
    accepted_sets: int = 100,
    max_proposals: int = 100_000,
    seed: int = 0,
    age_bins: np.ndarray | None = None,
) -> tuple[MatchResult, list[dict[str, Any]]]:
    """Generate proposals until the requested number pass the full-CDF W1 test."""
    if accepted_sets <= 0 or max_proposals <= 0:
        raise ValueError("accepted_sets and max_proposals must be positive")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    rows_out, cdf_out, w1_out, assign_out, ids_out = [], [], [], [], []
    diagnostics: list[dict[str, Any]] = []
    rejection_count = 0
    for proposal_id in range(1, max_proposals + 1):
        try:
            rows, assignment = draw_stratified_set(weights, quotas, rng)
            cdf, w1 = score_set(store, rows, target_cdf, age_bins)
            accepted = w1 <= threshold
            reason = "accepted" if accepted else "wasserstein_threshold"
        except SamplingError as exc:
            rows = assignment = cdf = None
            w1 = np.nan
            accepted, reason = False, f"sampling: {exc}"
        diagnostics.append({"proposal_id": proposal_id,
                            "accepted": accepted,
                            "wasserstein": w1, "threshold": threshold,
                            "reason": reason})
        if accepted:
            rows_out.append(rows); assign_out.append(assignment)
            cdf_out.append(cdf); w1_out.append(w1); ids_out.append(proposal_id)
            if len(rows_out) == accepted_sets:
                stacked_rows = np.stack(rows_out)
                flat_chroms, flat_positions = store.rows_to_native(stacked_rows.ravel())
                chromosomes = flat_chroms.reshape(stacked_rows.shape)
                positions = flat_positions.reshape(stacked_rows.shape)
                return MatchResult(stacked_rows, chromosomes, positions,
                                   np.asarray(cdf_out, dtype=np.float32),
                                   np.asarray(w1_out), np.stack(assign_out),
                                   np.asarray(ids_out), proposal_id,
                                   rejection_count), diagnostics
        else:
            rejection_count += 1
    raise SamplingError(
        f"generated {len(rows_out)} of {accepted_sets} accepted sets after "
        f"{max_proposals} proposals (overall acceptance rate "
        f"{len(rows_out) / max_proposals:.4g})")


def write_result(output_dir: Path, result: MatchResult,
                 diagnostics: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Atomically publish a new result directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    tmp = output_dir.with_name(f"{output_dir.name}.tmp.{os.getpid()}")
    tmp.mkdir(parents=True)
    try:
        np.save(tmp / "syn_positions.npy", result.positions)
        np.save(tmp / "syn_chromosomes.npy", result.chromosomes)
        np.save(tmp / "syn_row_indices.npy", result.row_indices)
        np.save(tmp / "syn_cdf.npy", result.cdfs)
        np.save(tmp / "wasserstein.npy", result.wasserstein)
        np.save(tmp / "interval_assignment.npy", result.interval_assignment)
        with (tmp / "diagnostics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader(); writer.writerows(diagnostics)
        payload = dict(metadata)
        payload.update({"schema_version": 1, "accepted_sets": int(result.row_indices.shape[0]),
                        "set_size": int(result.row_indices.shape[1]),
                        "proposals": result.attempts,
                        "rejections": result.rejection_count,
                        "acceptance_rate": float(
                            result.row_indices.shape[0] / result.attempts)})
        (tmp / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
        os.rename(tmp, output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_candidates(store: object, positions_file: Path | None,
                     indices_file: Path | None, mask_file: Path | None,
                     *, policy: str = "error") -> tuple[np.ndarray, dict[str, Any]]:
    if policy not in {"error", "drop"}:
        raise ValueError("missing-position policy must be 'error' or 'drop'")
    supplied = sum(x is not None for x in (positions_file, indices_file, mask_file))
    if supplied != 1:
        raise ValueError("specify exactly one of --syn-positions, --syn-indices, or --syn-mask")
    if indices_file:
        rows = np.load(indices_file)
    elif mask_file:
        mask = np.load(mask_file)
        if mask.dtype != np.bool_ or mask.shape != np.asarray(store.positions).shape:
            raise ValueError("syn mask must be boolean and match positions shape")
        rows = np.flatnonzero(mask)
    else:
        chromosomes, positions = load_native_position_list(positions_file)
        resolution = resolve_native_position_requests(
            store, chromosomes, positions, policy=policy,
            label="synonymous candidates")
        return resolution.included_rows, {
            "position_resolution": resolution.summary(),
            "excluded_positions": resolution.excluded_coordinates(),
        }
    raw_rows = np.asarray(rows)
    if not np.issubdtype(raw_rows.dtype, np.integer):
        raise ValueError("synonymous row indices must have an integer dtype")
    rows = raw_rows.astype(np.int64, copy=False)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= np.asarray(store.positions).size):
        raise ValueError("synonymous row indices contain invalid values")
    requested_rows = rows.copy()
    eligibility = np.asarray(store.eligible)[rows]
    if not np.all(eligibility):
        if policy == "error":
            raise ValueError(
                f"synonymous candidates include {(~eligibility).sum()} age rows below "
                "the usable-draw threshold")
        rows = rows[eligibility]
    if rows.size == 0:
        raise ValueError("no eligible synonymous candidates remain")
    excluded = []
    if np.any(~eligibility):
        excluded_request_indices = np.flatnonzero(~eligibility)
        excluded_rows = requested_rows[excluded_request_indices]
        chroms, native = store.rows_to_native(excluded_rows)
        excluded = [
            {"request_index": int(i), "global_position": int(store.positions[row]),
             "chromosome": str(chrom), "native_position": int(position),
             "reason": "ineligible"}
            for i, row, chrom, position in zip(
                excluded_request_indices, excluded_rows, chroms, native)
        ]
    return rows, {
        "position_resolution": {
            "label": "synonymous candidates", "policy": policy,
            "requested_count": int(eligibility.size),
            "resolved_count": int(eligibility.size),
            "eligible_count": int(rows.size),
            "excluded_count": int((~eligibility).sum()),
            "unresolved_count": 0,
            "ineligible_count": int((~eligibility).sum())},
        "excluded_positions": excluded,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--syn-positions", type=Path)
    group.add_argument("--syn-indices", type=Path)
    group.add_argument("--syn-mask", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accepted-sets", type=int, default=100)
    parser.add_argument("--max-proposals", type=int, default=100_000)
    parser.add_argument("--block-snps", type=int, default=250_000)
    parser.add_argument("--missing-position-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--candidate-access", choices=("auto", "gather", "coalesced", "scan", "cache"), default="gather")
    parser.add_argument("--coalesce-gap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.candidate_access == "auto":
        raise ValueError(
            "candidate-access auto requires benchmark thresholds from Gate 3; "
            "select gather, coalesced, or scan explicitly"
        )

    store = open_snp_age_store(args.store)
    candidates, resolution_meta = _load_candidates(
        store, args.syn_positions, args.syn_indices, args.syn_mask,
        policy=args.missing_position_policy)
    target_cdf = np.load(args.target / "target_cdf.npy")
    age_path = args.target / "age_bins.npy"
    age_bins = np.load(age_path) if age_path.exists() else np.asarray(store.age_bins)
    boundary_path = args.target / "interval_boundary_indices.npy"
    if not boundary_path.exists():
        raise FileNotFoundError(
            f"target is missing required boundary-index array: {boundary_path}"
        )
    boundaries = np.load(boundary_path)
    boundary_age_path = args.target / "interval_boundary_ages.npy"
    boundary_ages = np.load(boundary_age_path) if boundary_age_path.exists() else None
    quotas = np.load(args.target / "interval_quotas.npy")
    with (args.target / "metadata.json").open() as handle:
        target_meta = json.load(handle)
    if target_cdf.shape != age_bins.shape or age_bins.ndim != 1 or age_bins.size < 2:
        raise ValueError("target CDF and age grid are incompatible")
    if np.any(np.diff(age_bins) <= 0) or not np.isclose(target_cdf[-1], 1.0, atol=1e-5):
        raise ValueError("target age grid must increase and target CDF must end at one")
    expected_schema = target_meta.get("source_store_schema")
    if expected_schema is not None and expected_schema != store_schema(store):
        raise ValueError("target and candidate store schemas do not match")
    if not is_interval_store(store) and not np.array_equal(
        np.asarray(store.age_bins), age_bins
    ):
        raise ValueError("dense candidate store age grid does not match target")
    expected_catalog = target_meta.get("source_catalog_sha256")
    actual_catalog = getattr(store, "metadata", {}).get("catalog_sha256")
    if expected_catalog is not None and expected_catalog != actual_catalog:
        raise ValueError("target and candidate store catalogs do not match")
    if is_interval_store(store) and boundary_ages is None:
        raise ValueError("interval target is missing interval_boundary_ages.npy")
    threshold = float(target_meta["wasserstein_threshold_generations"])
    weights = build_candidate_weights(
        store, candidates, boundaries, args.block_snps,
        boundary_ages=boundary_ages, access_strategy=args.candidate_access,
        coalesce_gap=args.coalesce_gap)
    result, diagnostics = generate_matches(
        store, weights, quotas, target_cdf, threshold,
        accepted_sets=args.accepted_sets, max_proposals=args.max_proposals,
        seed=args.seed, age_bins=age_bins)
    write_result(args.output, result, diagnostics,
                 {"store": str(args.store), "target": str(args.target),
                  "seed": args.seed, "block_snps": args.block_snps,
                  "threshold": threshold,
                  "candidate_count": int(candidates.size),
                  **resolution_meta,
                  "store_schema": store_schema(store),
                  "candidate_access_requested": args.candidate_access,
                  "candidate_access_effective": args.candidate_access,
                  "coalesce_gap": args.coalesce_gap,
                  "weight_matrix_shape": list(weights.values.shape),
                  "weight_matrix_dtype": str(weights.values.dtype),
                  "weight_matrix_bytes": int(weights.values.nbytes),
                  "acceptance_rate": float(
                      result.row_indices.shape[0] / result.attempts)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
