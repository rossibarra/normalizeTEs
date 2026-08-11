#!/usr/bin/env python3
"""Construct a TE SNP age target and its bootstrap matching tolerance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from snp_age_dataset import load_native_position_list
from snp_age_store import is_interval_store, open_snp_age_store, store_schema
from snp_position_resolution import resolve_native_position_requests


@dataclass(frozen=True)
class BoundarySet:
    """Compressed equal-mass boundaries on CDF bin edges."""

    indices: np.ndarray
    ages: np.ndarray
    interval_shares: np.ndarray


@dataclass(frozen=True)
class TargetResult:
    te_global_positions: np.ndarray
    te_chromosomes: np.ndarray
    te_positions: np.ndarray
    te_row_indices: np.ndarray
    target_cdf: np.ndarray
    bootstrap_wasserstein: np.ndarray
    boundaries: BoundarySet
    interval_quotas: np.ndarray
    threshold: float
    seed: int | None
    age_bins: np.ndarray | None = None
    boundary_ages: np.ndarray | None = None


def aggregate_cdf(cdf_rows: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean CDF for a nonempty SNP-by-age matrix."""
    return np.asarray(cdf_rows).mean(axis=0, dtype=np.float64)


def wasserstein_1(
    cdf_a: np.ndarray, cdf_b: np.ndarray, bin_centers: np.ndarray
) -> float:
    """Discrete one-dimensional W1 distance, in units of the age grid."""
    a = np.asarray(cdf_a)
    b = np.asarray(cdf_b)
    ages = np.asarray(bin_centers)
    return float(np.sum(np.abs(a[:-1] - b[:-1]) * np.diff(ages), dtype=np.float64))


def bootstrap_wasserstein(
    cdf_rows: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
    batch_size: int = 256,
    *, bin_centers: np.ndarray,
) -> np.ndarray:
    """Bootstrap SNP rows and return W1 distances to the observed target.

    Multinomial row counts are exactly equivalent to drawing row indices with
    replacement, while avoiding an ``replicates x SNPs`` index allocation.
    """
    rows = np.asarray(cdf_rows)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
        raise ValueError("cdf_rows must be a nonempty SNP-by-age matrix")
    if not np.issubdtype(rows.dtype, np.floating):
        rows = rows.astype(np.float32)
    if n_replicates <= 0 or batch_size <= 0:
        raise ValueError("n_replicates and batch_size must be positive")
    ages = np.asarray(bin_centers)
    target = aggregate_cdf(rows)
    widths = np.diff(ages)
    probabilities = np.full(rows.shape[0], 1.0 / rows.shape[0])
    output = np.empty(n_replicates, dtype=np.float64)
    for start in range(0, n_replicates, batch_size):
        stop = min(start + batch_size, n_replicates)
        count = stop - start
        counts = rng.multinomial(rows.shape[0], probabilities, size=count)
        # A scratch-backed interval CDF matrix is float32. Matching the weight
        # dtype avoids NumPy promoting the complete memmap to an in-memory
        # float64 temporary during matrix multiplication.
        weight_dtype = np.float32 if rows.dtype == np.float32 else np.float64
        weights = counts.astype(weight_dtype)
        del counts
        boot = (weights @ rows) / rows.shape[0]
        output[start:stop] = np.sum(
            np.abs(boot[:, :-1] - target[:-1]) * widths,
            axis=1,
        )
    return output


def empirical_threshold(distances: np.ndarray, quantile: float = 0.95) -> float:
    """Return a conservative observed empirical quantile."""
    values = np.asarray(distances)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    return float(np.quantile(values, quantile, method="higher"))


def equal_mass_boundaries(
    target_cdf: np.ndarray,
    *,
    bin_centers: np.ndarray,
) -> BoundarySet:
    """Locate inverse-CDF edges and compress repeated quantile boundaries.

    For ``B`` CDF values, edge indices range from 0 through B. Edge zero has
    cumulative mass zero and edge ``i`` has cumulative mass ``cdf[i - 1]``.
    This convention ensures that interval ``[0, 1)`` contains the youngest
    age bin instead of accidentally omitting it.
    """
    cdf = np.asarray(target_cdf, dtype=np.float64)
    cdf = cdf / cdf[-1]
    probs = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    edge_cdf = np.concatenate(([0.0], cdf))
    requested_indices = np.searchsorted(edge_cdf, probs, side="left")
    requested_indices = np.clip(requested_indices, 0, cdf.size).astype(np.int64)
    requested_indices[0] = 0
    # The terminal edge spans the complete shared grid. This also lets the
    # sampler account for candidate mass older than the target's last nonzero
    # bin, even when the target CDF reaches one early.
    requested_indices[-1] = cdf.size
    indices = np.unique(requested_indices)
    ages = np.asarray(bin_centers)
    # Boundary ages are labels only; sampler-facing boundaries are the exact
    # edge indices. Interior edge i is labelled by the center of bin i-1.
    edge_ages = np.concatenate(([ages[0]], ages))
    shares = np.diff(edge_cdf[indices])
    shares[np.abs(shares) < 1e-15] = 0.0
    shares /= shares.sum()
    return BoundarySet(indices, edge_ages[indices], shares)


def largest_remainder_quotas(
    total: int,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Deterministic largest-remainder integer apportionment."""
    shares = np.asarray(probabilities, dtype=np.float64)
    ideal = total * shares
    quotas = np.floor(ideal).astype(np.int64)
    remaining = total - int(quotas.sum())
    if remaining:
        fractions = ideal - quotas
        ranked = np.argsort(-fractions, kind="stable")
        quotas[ranked[:remaining]] += 1
    return quotas


def _analysis_cdfs(
    store: object, rows: np.ndarray, *, bin_width: int,
    output_path: Path | None = None, block_rows: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy-compatible CDF rows and their age-bin centers."""
    if is_interval_store(store):
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        maximum = float(getattr(store, "metadata")["maximum_above"])
        last = max(1, int(np.floor(maximum / bin_width + 0.5)))
        centers = np.arange(last + 1, dtype=np.uint64) * np.uint64(bin_width)
        right_edges = centers.astype(np.float64) + bin_width / 2
        if output_path is None:
            raise ValueError("interval CDF construction requires a scratch output path")
        if block_rows <= 0:
            raise ValueError("CDF block rows must be positive")
        required_bytes = int(rows.size * right_edges.size * np.dtype("float32").itemsize)
        free_bytes = shutil.disk_usage(output_path.parent).free
        if required_bytes > free_bytes:
            raise OSError(
                f"interval TE CDF scratch needs {required_bytes} bytes but only "
                f"{free_bytes} bytes are free in {output_path.parent}"
            )
        writer = getattr(store, "write_regular_grid_cdfs", None)
        if writer is not None:
            cdfs = writer(
                rows, right_edges, output_path,
                block_rows=block_rows, dtype=np.float32,
            )
        else:
            cdfs = np.lib.format.open_memmap(
                output_path, mode="w+", dtype=np.float32,
                shape=(rows.size, right_edges.size),
            )
            for start in range(0, rows.size, block_rows):
                stop = min(start + block_rows, rows.size)
                cdfs[start:stop] = store.cdf_at(
                    rows[start:stop], right_edges, side="left", weighting="interval"
                ).astype(np.float32)
            cdfs.flush()
        return cdfs, centers
    return (
        np.asarray(store.read_cdfs(rows), dtype=np.float64),
        np.asarray(store.age_bins),
    )


def analysis_grid_edges(age_bins: np.ndarray) -> np.ndarray:
    """Return physical half-open cell edges for a uniform center grid."""
    centers = np.asarray(age_bins, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2 or np.any(np.diff(centers) <= 0):
        raise ValueError("age bins must be a strictly increasing 1-D array")
    widths = np.diff(centers)
    if not np.allclose(widths, widths[0], rtol=0, atol=0):
        raise ValueError("analysis age bins must be uniformly spaced")
    half = widths[0] / 2
    return np.concatenate(([centers[0] - half], centers + half))


def build_target(
    store: object,
    te_positions: np.ndarray,
    te_chromosomes: np.ndarray,
    te_vcf_positions: np.ndarray,
    *,
    n_replicates: int,
    acceptance_quantile: float,
    seed: int | None,
    batch_size: int = 256,
    row_indices: np.ndarray | None = None,
    bin_width: int = 1_000,
    scratch_dir: str | Path | None = None,
    cdf_block_rows: int = 512,
) -> TargetResult:
    """Resolve positions and calculate Stage 2 products with bounded memory."""
    positions = np.asarray(te_positions, dtype=np.float64)
    if np.unique(positions).size != positions.size:
        raise ValueError("TE positions contain duplicates")
    rows = (
        np.asarray(store.resolve_positions(positions), dtype=np.int64)
        if row_indices is None else np.asarray(row_indices, dtype=np.int64)
    )
    if rows.shape != positions.shape:
        raise ValueError("row_indices must align with TE positions")
    if np.any(rows < 0) or np.any(rows >= np.asarray(store.positions).size):
        raise ValueError("row_indices contain an out-of-range value")
    invalid = ~np.asarray(store.eligible[rows], dtype=bool)
    if np.any(invalid):
        raise ValueError(f"invalid TE positions: {_format_values(positions[invalid])}")
    if cdf_block_rows <= 0:
        raise ValueError("cdf_block_rows must be positive")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if is_interval_store(store):
            parent = None if scratch_dir is None else Path(scratch_dir)
            if parent is not None and not parent.is_dir():
                raise NotADirectoryError(parent)
            temporary = tempfile.TemporaryDirectory(prefix="te-target-cdf-", dir=parent)
            cdf_path = Path(temporary.name) / "cdf_by_snp.npy"
        else:
            cdf_path = None
        cdf_rows, ages = _analysis_cdfs(
            store, rows, bin_width=bin_width, output_path=cdf_path,
            block_rows=cdf_block_rows,
        )
        rng = np.random.default_rng(seed)
        distances = bootstrap_wasserstein(
            cdf_rows,
            n_replicates,
            rng,
            batch_size,
            bin_centers=ages,
        )
        target = aggregate_cdf(cdf_rows)
        del cdf_rows
    finally:
        if temporary is not None:
            temporary.cleanup()
    boundary_set = equal_mass_boundaries(target, bin_centers=ages)
    quotas = largest_remainder_quotas(
        positions.size, boundary_set.interval_shares,
    )
    threshold = empirical_threshold(distances, acceptance_quantile)
    return TargetResult(
        positions, np.asarray(te_chromosomes), np.asarray(te_vcf_positions),
        rows, target, distances, boundary_set, quotas, threshold, seed,
        ages, analysis_grid_edges(ages)[boundary_set.indices],
    )


def write_target(output_dir: Path, result: TargetResult, metadata: dict[str, object]) -> None:
    """Atomically create a target directory; never overwrite an existing path."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        boundary_set = result.boundaries
        arrays = {
            "te_global_positions.npy": result.te_global_positions,
            "te_chromosomes.npy": result.te_chromosomes,
            "te_positions.npy": result.te_positions,
            "te_row_indices.npy": result.te_row_indices,
            "target_cdf.npy": result.target_cdf,
            "bootstrap_wasserstein.npy": result.bootstrap_wasserstein,
            "interval_boundaries.npy": boundary_set.ages,
            "interval_boundary_indices.npy": boundary_set.indices,
            "interval_shares.npy": boundary_set.interval_shares,
            "interval_quotas.npy": result.interval_quotas,
        }
        if result.age_bins is not None:
            arrays["age_bins.npy"] = result.age_bins
        if result.boundary_ages is not None:
            arrays["interval_boundary_ages.npy"] = result.boundary_ages
        for name, array in arrays.items():
            np.save(temporary / name, np.asarray(array), allow_pickle=False)
        with (temporary / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _format_values(values: np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(format(float(value), ".15g") for value in values[:limit])
    return shown + (f", ... ({values.size} total)" if values.size > limit else "")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--te-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=256)
    parser.add_argument("--acceptance-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bin-width", type=int, default=1_000)
    parser.add_argument(
        "--scratch-dir", type=Path,
        help="node-local parent for temporary interval CDF storage",
    )
    parser.add_argument("--cdf-block-rows", type=int, default=512)
    parser.add_argument(
        "--missing-position-policy", choices=("error", "drop"), default="error"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    chromosomes, vcf_positions = load_native_position_list(args.te_positions)
    resolution = resolve_native_position_requests(
        store,
        chromosomes,
        vcf_positions,
        policy=args.missing_position_policy,
        label="TE positions",
    )
    positions = resolution.included_global_positions
    included_chromosomes = resolution.included_chromosomes
    included_vcf_positions = resolution.included_native_positions
    assert included_chromosomes is not None and included_vcf_positions is not None
    result = build_target(
        store,
        positions,
        included_chromosomes,
        included_vcf_positions,
        n_replicates=args.bootstrap_replicates,
        acceptance_quantile=args.acceptance_quantile,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
        row_indices=resolution.included_rows,
        bin_width=args.bin_width,
        scratch_dir=args.scratch_dir,
        cdf_block_rows=args.cdf_block_rows,
    )
    boundary_set = result.boundaries
    metadata = {
        "schema_version": 2,
        "source_store": str(args.store.resolve()),
        "source_store_schema": store_schema(store),
        "source_catalog_sha256": getattr(store, "metadata", {}).get("catalog_sha256"),
        "te_position_source": str(args.te_positions.resolve()),
        "n_te_snps": int(positions.size),
        "effective_te_set_size": int(positions.size),
        "position_resolution": resolution.summary(),
        "excluded_positions": resolution.excluded_coordinates(),
        "missing_position_policy": args.missing_position_policy,
        "bin_width": int(np.diff(result.age_bins[:2])[0]),
        "cdf_evaluation": (
            "P(X < right_cell_edge); equal interval weighting"
            if is_interval_store(store) else "stored dense CDF"
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_batch_size": args.bootstrap_batch_size,
        "cdf_block_rows": args.cdf_block_rows,
        "cdf_working_dtype": "float32" if is_interval_store(store) else None,
        "cdf_working_bytes": (
            int(positions.size * result.age_bins.size * np.dtype("float32").itemsize)
            if is_interval_store(store) else None
        ),
        "cdf_working_storage": (
            "temporary scratch-backed NPY, removed after target construction"
            if is_interval_store(store) else "dense store"
        ),
        "cdf_working_algorithm": (
            "regular-grid slope/intercept differences; O(intervals + output cells)"
            if is_interval_store(store) else "stored dense CDF"
        ),
        "acceptance_quantile": args.acceptance_quantile,
        "wasserstein_threshold_generations": result.threshold,
        "boundary_probabilities": np.linspace(0.0, 1.0, 21).tolist(),
        "compressed_boundary_edge_indices": boundary_set.indices.tolist(),
        "compressed_boundary_ages": result.boundary_ages.tolist(),
        "interval_shares": boundary_set.interval_shares.tolist(),
        "seed": args.seed,
        "numpy_version": np.__version__,
    }
    write_target(args.output, result, metadata)
    print(f"Wrote TE target to {args.output}")
    print(f"W1 acceptance threshold: {result.threshold:.6g} generations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
