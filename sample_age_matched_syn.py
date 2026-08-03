#!/usr/bin/env python3
"""Generate synonymous SNP sets matched to a target SNP age distribution.

The sampler scans candidate CDFs in blocks and retains only interval totals.  A
proposal is drawn by hierarchical probability-proportional-to-size sampling:
first a block, then a SNP within that block.  Selected SNPs are removed from all
strata for the remainder of that proposal, guaranteeing uniqueness without an
N-candidates by N-strata weight matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from snp_age_dataset import load_native_position_list


class SamplingError(RuntimeError):
    """Raised when the requested matched sample cannot be constructed."""


@dataclass(frozen=True)
class BlockWeightIndex:
    candidate_rows: np.ndarray
    boundary_indices: np.ndarray
    block_starts: np.ndarray
    block_totals: np.ndarray
    block_positive_counts: np.ndarray

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


def _decode_cdfs(values: np.ndarray, store: Any) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        scale = float(getattr(store, "quantization_scale", 65535.0))
        return values.astype(np.float64) / scale
    return values.astype(np.float64, copy=False)


def _read_cdfs(store: Any, rows: np.ndarray) -> np.ndarray:
    if hasattr(store, "read_cdfs"):
        return _decode_cdfs(store.read_cdfs(rows), store)
    return _decode_cdfs(np.asarray(store.cdf_by_snp)[rows], store)


def _boundary_weights(store: Any, rows: np.ndarray,
                      boundary_indices: np.ndarray) -> np.ndarray:
    """Return interval masses for CDF *edge* indices in ``[0, B]``.

    Edge zero is the implicit CDF value zero; edge ``e > 0`` maps to stored
    CDF column ``e - 1``.  When available, the age-major boundary reader is
    used on one enclosing contiguous store-coordinate slab and then subset to
    the (sorted, possibly gapped) candidate rows.
    """
    rows = np.asarray(rows, dtype=np.int64)
    bounds = np.asarray(boundary_indices, dtype=np.int64)
    n_bins = int(np.asarray(store.age_bins).size)
    if np.any(bounds < 0) or np.any(bounds > n_bins):
        raise ValueError(f"boundary edge indices must lie in [0, {n_bins}]")
    positive_edges = np.unique(bounds[bounds > 0])
    edge_cdfs = np.zeros((rows.size, bounds.size), dtype=np.float64)
    if positive_edges.size:
        columns = positive_edges - 1
        if hasattr(store, "read_boundary_cdfs"):
            slab_start, slab_stop = int(rows[0]), int(rows[-1]) + 1
            raw = np.asarray(store.read_boundary_cdfs(
                columns, slab_start, slab_stop))
            expected = (columns.size, slab_stop - slab_start)
            if raw.shape != expected:
                raise ValueError(
                    "read_boundary_cdfs returned shape "
                    f"{raw.shape}, expected {expected}")
            selected = _decode_cdfs(raw[:, rows - slab_start], store).T
        else:
            selected = _read_cdfs(store, rows)[:, columns]
        lookup = {int(edge): selected[:, j]
                  for j, edge in enumerate(positive_edges)}
        for j, edge in enumerate(bounds):
            if edge:
                edge_cdfs[:, j] = lookup[int(edge)]
    weights = np.diff(edge_cdfs, axis=1)
    # Quantization or float roundoff can produce tiny negative differences.
    np.maximum(weights, 0.0, out=weights)
    return weights


def build_interval_block_index(
    store: Any,
    syn_indices: np.ndarray,
    boundary_indices: np.ndarray,
    block_snps: int = 250_000,
) -> BlockWeightIndex:
    """Scan candidate rows once and retain only per-block interval summaries."""
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
    starts = np.asarray(starts_list, dtype=np.int64)
    totals = np.zeros((bounds.size - 1, starts.size), dtype=np.float64)
    counts = np.zeros_like(totals, dtype=np.int64)
    for block, start in enumerate(starts):
        stop = (int(starts[block + 1]) if block + 1 < starts.size
                else rows.size)
        weights = _boundary_weights(store, rows[start:stop], bounds)
        totals[:, block] = weights.sum(axis=0, dtype=np.float64)
        counts[:, block] = np.count_nonzero(weights > 0, axis=0)
    return BlockWeightIndex(rows.astype(np.int64, copy=False), bounds, starts,
                            totals, counts)


class _BlockCache:
    def __init__(self, store: Any, index: BlockWeightIndex, size: int = 8):
        self.store, self.index, self.size = store, index, max(1, size)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, block: int) -> np.ndarray:
        if block in self.cache:
            self.cache.move_to_end(block)
            return self.cache[block]
        start = int(self.index.block_starts[block])
        stop = (int(self.index.block_starts[block + 1])
                if block + 1 < self.index.block_starts.size
                else self.index.candidate_rows.size)
        value = _boundary_weights(
            self.store, self.index.candidate_rows[start:stop],
            self.index.boundary_indices)
        self.cache[block] = value
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return value


def _weighted_choice(weights: np.ndarray, rng: np.random.Generator) -> int:
    total = float(weights.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise SamplingError("no positive candidate mass remains")
    # searchsorted over a cumulative sum avoids normalizing a large vector.
    return min(int(np.searchsorted(np.cumsum(weights, dtype=np.float64),
                                   rng.random() * total, side="right")),
               weights.size - 1)


def draw_stratified_set(
    store: Any,
    index: BlockWeightIndex,
    quotas: np.ndarray,
    rng: np.random.Generator,
    cache_blocks: int = 8,
    interval_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw one unique candidate set, returning rows and stratum assignments."""
    quotas = np.asarray(quotas, dtype=np.int64)
    if quotas.shape != (index.n_intervals,) or np.any(quotas < 0):
        raise ValueError("quotas must be nonnegative with one value per interval")
    if quotas.sum() > index.candidate_rows.size:
        raise SamplingError("total quota exceeds the number of unique candidates")
    for interval, quota in enumerate(quotas):
        if quota and index.block_totals[interval].sum() <= 0:
            raise SamplingError(f"interval {interval} has zero candidate mass")
        if quota > index.block_positive_counts[interval].sum():
            raise SamplingError(
                f"interval {interval} quota {quota} exceeds its positive-mass "
                f"candidate count {index.block_positive_counts[interval].sum()}")

    totals = index.block_totals.copy()
    selected_local: dict[int, set[int]] = {}
    out_rows: list[int] = []
    assignments: list[int] = []
    cache = _BlockCache(store, index, cache_blocks)
    order_rng = rng if interval_rng is None else interval_rng
    order = order_rng.permutation(index.n_intervals)
    for interval in order:
        for _ in range(int(quotas[interval])):
            block = _weighted_choice(totals[interval], rng)
            start = int(index.block_starts[block])
            weights_all = cache.get(block)
            local_weights = weights_all[:, interval].copy()
            block_rows = index.candidate_rows[
                start:(int(index.block_starts[block + 1])
                       if block + 1 < index.block_starts.size
                       else index.candidate_rows.size)]
            used_here = selected_local.get(block)
            if used_here:
                local_weights[np.fromiter(used_here, dtype=np.int64)] = 0.0
            if local_weights.sum() <= 0:
                # Totals may reach this state after cross-stratum removals.
                totals[interval, block] = 0.0
                try:
                    block = _weighted_choice(totals[interval], rng)
                except SamplingError as exc:
                    raise SamplingError(
                        f"interval {interval} cannot fill quota after uniqueness "
                        "filtering") from exc
                start = int(index.block_starts[block])
                weights_all = cache.get(block)
                block_rows = index.candidate_rows[
                    start:(int(index.block_starts[block + 1])
                           if block + 1 < index.block_starts.size
                           else index.candidate_rows.size)]
                local_weights = weights_all[:, interval].copy()
                used_here = selected_local.get(block)
                if used_here:
                    local_weights[np.fromiter(used_here, dtype=np.int64)] = 0.0
            local = _weighted_choice(local_weights, rng)
            row = int(block_rows[local])
            selected_local.setdefault(block, set()).add(local)
            out_rows.append(row)
            assignments.append(int(interval))
            # Remove this SNP's mass from every stratum's block total.
            totals[:, block] -= weights_all[local]
            np.maximum(totals[:, block], 0.0, out=totals[:, block])
    return np.asarray(out_rows, dtype=np.int64), np.asarray(assignments,
                                                             dtype=np.uint8)


def wasserstein_1(cdf_a: np.ndarray, cdf_b: np.ndarray,
                  age_bins: np.ndarray) -> float:
    age_bins = np.asarray(age_bins, dtype=np.float64)
    return float(np.sum(np.abs(np.asarray(cdf_a)[:-1] -
                               np.asarray(cdf_b)[:-1]) * np.diff(age_bins)))


def score_set(store: Any, row_indices: np.ndarray, target_cdf: np.ndarray
              ) -> tuple[np.ndarray, float, float]:
    aggregate = _read_cdfs(store, row_indices).mean(axis=0, dtype=np.float64)
    w1 = wasserstein_1(aggregate, target_cdf, np.asarray(store.age_bins))
    max_cdf = float(np.max(np.abs(aggregate - target_cdf)))
    return aggregate, w1, max_cdf


def generate_matches(
    store: Any,
    index: BlockWeightIndex,
    quotas: np.ndarray,
    target_cdf: np.ndarray,
    threshold: float,
    *,
    accepted_sets: int = 100,
    max_proposals: int = 100_000,
    seed: int | Sequence[int] | np.random.SeedSequence = 0,
    cache_blocks: int = 8,
) -> tuple[MatchResult, list[dict[str, Any]]]:
    """Generate proposals until the requested number pass the full-CDF W1 test."""
    if accepted_sets <= 0 or max_proposals <= 0:
        raise ValueError("accepted_sets and max_proposals must be positive")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    root_seed = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    sampling_seed, ordering_seed = root_seed.spawn(2)
    rows_out, cdf_out, w1_out, assign_out, ids_out = [], [], [], [], []
    diagnostics: list[dict[str, Any]] = []
    rejection_count = 0
    for proposal_id in range(1, max_proposals + 1):
        # A distinct stream makes a proposal reproducible independent of how many
        # random numbers earlier proposals consumed.
        draw_child = sampling_seed.spawn(1)[0]
        order_child = ordering_seed.spawn(1)[0]
        rng = np.random.default_rng(draw_child)
        interval_rng = np.random.default_rng(order_child)
        try:
            rows, assignment = draw_stratified_set(
                store, index, quotas, rng, cache_blocks, interval_rng)
            cdf, w1, max_cdf = score_set(store, rows, target_cdf)
            accepted = w1 <= threshold
            reason = "accepted" if accepted else "wasserstein_threshold"
        except SamplingError as exc:
            rows = assignment = cdf = None
            w1 = max_cdf = np.nan
            accepted, reason = False, f"sampling: {exc}"
        diagnostics.append({"proposal_id": proposal_id,
                            "seed_spawn_key": ".".join(map(str, draw_child.spawn_key)),
                            "interval_seed_spawn_key": ".".join(map(str, order_child.spawn_key)),
                            "accepted": accepted,
                            "wasserstein": w1, "threshold": threshold,
                            "max_abs_cdf": max_cdf, "reason": reason})
        if accepted:
            rows_out.append(rows); assign_out.append(assignment)
            cdf_out.append(cdf); w1_out.append(w1); ids_out.append(proposal_id)
            if len(rows_out) == accepted_sets:
                stacked_rows = np.stack(rows_out)
                if hasattr(store, "rows_to_native"):
                    flat_chroms, flat_positions = store.rows_to_native(stacked_rows.ravel())
                    chromosomes = flat_chroms.reshape(stacked_rows.shape)
                    positions = flat_positions.reshape(stacked_rows.shape)
                else:
                    chromosomes = np.full(stacked_rows.shape, "", dtype="U1")
                    positions = np.asarray(store.positions)[stacked_rows]
                return MatchResult(stacked_rows, chromosomes, positions,
                                   np.asarray(cdf_out, dtype=np.float32),
                                   np.asarray(w1_out), np.stack(assign_out),
                                   np.asarray(ids_out), proposal_id,
                                   rejection_count), diagnostics
        else:
            rejection_count += 1
    raise SamplingError(
        f"generated {len(rows_out)} of {accepted_sets} accepted sets after "
        f"{max_proposals} proposals (last acceptance rate "
        f"{len(rows_out) / max_proposals:.4g})")


def write_result(output_dir: Path, result: MatchResult,
                 diagnostics: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Atomically publish a new result directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    tmp = output_dir.with_name(f"{output_dir.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
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
                        "rejections": result.rejection_count})
        (tmp / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
        os.rename(tmp, output_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _load_candidates(store: Any, positions_file: Path | None,
                     indices_file: Path | None, mask_file: Path | None) -> np.ndarray:
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
        rows = store.resolve_native_positions(chromosomes, positions)
    rows = np.asarray(rows, dtype=np.int64)
    eligibility = np.asarray(getattr(store, "eligible", store.valid))[rows]
    if not np.all(eligibility):
        raise ValueError(
            f"synonymous candidates include {(~eligibility).sum()} age rows below "
            "the usable-draw threshold"
        )
    return rows


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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from snp_age_dataset import SNPAgeDataset
    store = SNPAgeDataset.open(args.store)
    candidates = _load_candidates(store, args.syn_positions, args.syn_indices, args.syn_mask)
    target_cdf = np.load(args.target / "target_cdf.npy")
    boundary_path = args.target / "interval_boundary_indices.npy"
    if not boundary_path.exists():
        boundary_path = args.target / "interval_boundaries.npy"
    boundaries = np.load(boundary_path)
    quotas = np.load(args.target / "interval_quotas.npy")
    with (args.target / "metadata.json").open() as handle:
        target_meta = json.load(handle)
    threshold_value = target_meta.get(
        "wasserstein_threshold_generations",
        target_meta.get("wasserstein_threshold",
                        target_meta.get("acceptance_threshold")))
    if threshold_value is None:
        raise ValueError("target metadata does not contain a Wasserstein threshold")
    threshold = float(threshold_value)
    index = build_interval_block_index(store, candidates, boundaries, args.block_snps)
    result, diagnostics = generate_matches(
        store, index, quotas, target_cdf, threshold,
        accepted_sets=args.accepted_sets, max_proposals=args.max_proposals,
        seed=args.seed)
    write_result(args.output, result, diagnostics,
                 {"store": str(args.store), "target": str(args.target),
                  "seed": args.seed, "block_snps": args.block_snps,
                  "threshold": threshold,
                  "candidate_count": int(candidates.size)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
