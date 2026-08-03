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
from statistics import NormalDist
from typing import Sequence

import numpy as np

from snp_age_dataset import load_native_position_list


@dataclass(frozen=True)
class BoundarySet:
    """Requested quantiles and compressed boundaries on CDF bin edges."""

    probabilities: np.ndarray
    requested_indices: np.ndarray
    indices: np.ndarray
    ages: np.ndarray
    interval_shares: np.ndarray


def load_position_list(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read chromosome plus 1-based VCF position pairs."""
    return load_native_position_list(path)


def aggregate_cdf(cdf_rows: np.ndarray) -> np.ndarray:
    """Return the arithmetic mean CDF for a nonempty SNP-by-age matrix."""
    rows = _validate_cdf_rows(cdf_rows)
    return rows.mean(axis=0, dtype=np.float64)


def wasserstein_1(
    cdf_a: np.ndarray, cdf_b: np.ndarray, bin_centers: np.ndarray
) -> float:
    """Discrete one-dimensional W1 distance, in units of the age grid."""
    a = np.asarray(cdf_a, dtype=np.float64)
    b = np.asarray(cdf_b, dtype=np.float64)
    ages = _validate_age_grid(bin_centers)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("CDFs must be one-dimensional arrays with equal shapes")
    if a.size != ages.size:
        raise ValueError("CDF length must equal age-grid length")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("CDFs must contain only finite values")
    return float(np.sum(np.abs(a[:-1] - b[:-1]) * np.diff(ages), dtype=np.float64))


def bootstrap_wasserstein(
    cdf_rows: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
    batch_size: int = 256,
    *,
    bin_centers: np.ndarray | None = None,
    reference: str = "observed",
) -> np.ndarray:
    """Bootstrap SNP rows and return W1 distances to the requested reference.

    Multinomial row counts are exactly equivalent to drawing row indices with
    replacement, while avoiding an ``replicates x SNPs`` index allocation.
    """
    rows = _validate_cdf_rows(cdf_rows)
    if n_replicates <= 0 or batch_size <= 0:
        raise ValueError("n_replicates and batch_size must be positive")
    if reference not in {"observed", "two-sample"}:
        raise ValueError("reference must be 'observed' or 'two-sample'")
    ages = (
        np.arange(rows.shape[1], dtype=np.float64)
        if bin_centers is None
        else _validate_age_grid(bin_centers)
    )
    if ages.size != rows.shape[1]:
        raise ValueError("CDF width must equal age-grid length")
    target = aggregate_cdf(rows)
    widths = np.diff(ages)
    probabilities = np.full(rows.shape[0], 1.0 / rows.shape[0])
    output = np.empty(n_replicates, dtype=np.float64)
    for start in range(0, n_replicates, batch_size):
        stop = min(start + batch_size, n_replicates)
        count = stop - start
        weights = rng.multinomial(rows.shape[0], probabilities, size=count)
        boot = (weights @ rows) / rows.shape[0]
        if reference == "two-sample":
            other_weights = rng.multinomial(rows.shape[0], probabilities, size=count)
            comparison = (other_weights @ rows) / rows.shape[0]
        else:
            comparison = target
        output[start:stop] = np.sum(
            np.abs(boot[:, :-1] - comparison[..., :-1]) * widths,
            axis=1,
        )
    return output


def empirical_threshold(distances: np.ndarray, quantile: float = 0.95) -> float:
    """Return a conservative observed empirical quantile."""
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("distances must be a nonempty finite one-dimensional array")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    return float(np.quantile(values, quantile, method="higher"))


def quantile_order_statistic_interval(
    distances: np.ndarray, quantile: float = 0.95, confidence: float = 0.95
) -> tuple[float, float]:
    """Approximate Monte Carlo interval using binomial order-statistic ranks."""
    values = np.sort(np.asarray(distances, dtype=np.float64))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("distances must be a nonempty finite one-dimensional array")
    if not 0 < quantile < 1 or not 0 < confidence < 1:
        raise ValueError("quantile and confidence must be strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    sigma = np.sqrt(values.size * quantile * (1 - quantile))
    center = values.size * quantile
    lower_rank = max(0, int(np.floor(center - z * sigma)) - 1)
    upper_rank = min(values.size - 1, int(np.ceil(center + z * sigma)) - 1)
    return float(values[lower_rank]), float(values[upper_rank])


def equal_mass_boundaries(
    target_cdf: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    bin_centers: np.ndarray | None = None,
) -> BoundarySet:
    """Locate inverse-CDF edges and compress repeated quantile boundaries.

    For ``B`` CDF values, edge indices range from 0 through B. Edge zero has
    cumulative mass zero and edge ``i`` has cumulative mass ``cdf[i - 1]``.
    This convention ensures that interval ``[0, 1)`` contains the youngest
    age bin instead of accidentally omitting it.
    """
    cdf = np.asarray(target_cdf, dtype=np.float64)
    if cdf.ndim != 1 or cdf.size == 0 or not np.all(np.isfinite(cdf)):
        raise ValueError("target CDF must be a nonempty finite one-dimensional array")
    if np.any(np.diff(cdf) < -1e-10) or cdf[0] < -1e-10 or cdf[-1] <= 0:
        raise ValueError("target CDF must be nondecreasing and have positive terminal mass")
    cdf = cdf / cdf[-1]
    probs = (
        np.linspace(0.0, 1.0, 21, dtype=np.float64)
        if probabilities is None
        else np.asarray(probabilities, dtype=np.float64)
    )
    if (
        probs.ndim != 1
        or probs.size < 2
        or not np.all(np.isfinite(probs))
        or np.any(np.diff(probs) < 0)
        or probs[0] != 0
        or probs[-1] != 1
    ):
        raise ValueError("probabilities must be sorted and span exactly 0 to 1")
    edge_cdf = np.concatenate(([0.0], cdf))
    requested_indices = np.searchsorted(edge_cdf, probs, side="left")
    requested_indices = np.clip(requested_indices, 0, cdf.size).astype(np.int64)
    requested_indices[0] = 0
    # The terminal edge spans the complete shared grid. This also lets the
    # sampler account for candidate mass older than the target's last nonzero
    # bin, even when the target CDF reaches one early.
    requested_indices[-1] = cdf.size
    indices = np.unique(requested_indices)
    ages = (
        np.arange(cdf.size, dtype=np.float64)
        if bin_centers is None
        else _validate_age_grid(bin_centers)
    )
    if ages.size != cdf.size:
        raise ValueError("CDF length must equal age-grid length")
    # Boundary ages are labels only; sampler-facing boundaries are the exact
    # edge indices. Interior edge i is labelled by the center of bin i-1.
    edge_ages = np.concatenate(([ages[0]], ages))
    shares = np.diff(edge_cdf[indices])
    shares[np.abs(shares) < 1e-15] = 0.0
    shares /= shares.sum()
    return BoundarySet(
        probs.copy(), requested_indices, indices, edge_ages[indices], shares
    )


def largest_remainder_quotas(
    total: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integer apportionment with random, reproducible tie-breaking."""
    if total <= 0:
        raise ValueError("total must be positive")
    shares = np.asarray(probabilities, dtype=np.float64)
    if shares.ndim != 1 or shares.size == 0 or np.any(shares < 0):
        raise ValueError("probabilities must be a nonempty nonnegative vector")
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    ideal = total * shares
    quotas = np.floor(ideal).astype(np.int64)
    remaining = total - int(quotas.sum())
    if remaining:
        fractions = ideal - quotas
        tie_order = rng.permutation(shares.size)
        ranked = tie_order[np.argsort(-fractions[tie_order], kind="stable")]
        quotas[ranked[:remaining]] += 1
    return quotas


def build_target(
    store: object,
    te_positions: np.ndarray,
    *,
    n_replicates: int,
    acceptance_quantile: float,
    seed: int | Sequence[int] | None,
    batch_size: int = 256,
    bootstrap_reference: str = "observed",
) -> dict[str, object]:
    """Resolve positions and calculate all in-memory Stage 2 products."""
    positions = np.asarray(te_positions, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("TE positions must be a nonempty one-dimensional array")
    if np.unique(positions).size != positions.size:
        raise ValueError("TE positions contain duplicates")
    row_indices = np.asarray(store.resolve_positions(positions))
    if row_indices.shape != positions.shape:
        raise ValueError("dataset returned an unexpected position-index shape")
    eligibility = getattr(store, "eligible", getattr(store, "valid", None))
    if eligibility is not None:
        invalid = ~np.asarray(eligibility[row_indices], dtype=bool)
        if np.any(invalid):
            raise ValueError(f"invalid TE positions: {_format_values(positions[invalid])}")
    cdf_rows = np.asarray(store.read_cdfs(row_indices), dtype=np.float64)
    ages = np.asarray(store.age_bins)
    seed_sequence = np.random.SeedSequence(seed)
    bootstrap_seed, quota_seed = seed_sequence.spawn(2)
    distances = bootstrap_wasserstein(
        cdf_rows,
        n_replicates,
        np.random.default_rng(bootstrap_seed),
        batch_size,
        bin_centers=ages,
        reference=bootstrap_reference,
    )
    target = aggregate_cdf(cdf_rows)
    boundary_set = equal_mass_boundaries(target, bin_centers=ages)
    quotas = largest_remainder_quotas(
        positions.size,
        boundary_set.interval_shares,
        np.random.default_rng(quota_seed),
    )
    threshold = empirical_threshold(distances, acceptance_quantile)
    uncertainty = quantile_order_statistic_interval(distances, acceptance_quantile)
    return {
        "te_positions": positions,
        "te_row_indices": row_indices,
        "target_cdf": target,
        "bootstrap_wasserstein": distances,
        "boundaries": boundary_set,
        "interval_quotas": quotas,
        "threshold": threshold,
        "threshold_interval": uncertainty,
        "seed_entropy": seed_sequence.entropy,
        "seed_spawn_keys": {
            "bootstrap": list(bootstrap_seed.spawn_key),
            "quota_ties": list(quota_seed.spawn_key),
        },
    }


def write_target(output_dir: Path, result: dict[str, object], metadata: dict[str, object]) -> None:
    """Atomically create a target directory; never overwrite an existing path."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        boundary_set = result["boundaries"]
        assert isinstance(boundary_set, BoundarySet)
        arrays = {
            "te_global_positions.npy": result["te_positions"],
            "te_row_indices.npy": result["te_row_indices"],
            "target_cdf.npy": result["target_cdf"],
            "bootstrap_wasserstein.npy": result["bootstrap_wasserstein"],
            "interval_boundaries.npy": boundary_set.ages,
            "interval_boundary_indices.npy": boundary_set.indices,
            "requested_interval_boundary_indices.npy": boundary_set.requested_indices,
            "interval_shares.npy": boundary_set.interval_shares,
            "interval_quotas.npy": result["interval_quotas"],
        }
        if "te_chromosomes" in result:
            arrays["te_chromosomes.npy"] = result["te_chromosomes"]
            arrays["te_positions.npy"] = result["te_vcf_positions"]
        for name, array in arrays.items():
            np.save(temporary / name, np.asarray(array), allow_pickle=False)
        with (temporary / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_cdf_rows(cdf_rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(cdf_rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] < 2:
        raise ValueError("CDF rows must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(rows)):
        raise ValueError("CDF rows must contain only finite values")
    if np.any(np.diff(rows, axis=1) < -1e-10):
        raise ValueError("each CDF row must be nondecreasing")
    if np.any(rows[:, 0] < -1e-10) or np.any(rows[:, -1] <= 0):
        raise ValueError("CDF rows must be nonnegative and have positive terminal mass")
    return rows / rows[:, -1, None]


def _validate_age_grid(bin_centers: np.ndarray) -> np.ndarray:
    ages = np.asarray(bin_centers, dtype=np.float64)
    if ages.ndim != 1 or ages.size < 2 or not np.all(np.isfinite(ages)):
        raise ValueError("age grid must be a finite one-dimensional array of length >= 2")
    if np.any(np.diff(ages) <= 0):
        raise ValueError("age grid must be strictly increasing")
    return ages


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
    parser.add_argument(
        "--bootstrap-reference", choices=("observed", "two-sample"), default="observed"
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Deferred so the math API and tests do not depend on Stage 1 being present.
    from snp_age_dataset import SNPAgeDataset

    store = SNPAgeDataset.open(args.store)
    chromosomes, vcf_positions = load_position_list(args.te_positions)
    positions = store.native_to_global(chromosomes, vcf_positions)
    result = build_target(
        store,
        positions,
        n_replicates=args.bootstrap_replicates,
        acceptance_quantile=args.acceptance_quantile,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
        bootstrap_reference=args.bootstrap_reference,
    )
    result["te_chromosomes"] = chromosomes
    result["te_vcf_positions"] = vcf_positions
    boundary_set = result["boundaries"]
    assert isinstance(boundary_set, BoundarySet)
    metadata = {
        "schema_version": 1,
        "source_store": str(args.store.resolve()),
        "te_position_source": str(args.te_positions.resolve()),
        "n_te_snps": int(positions.size),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_reference": args.bootstrap_reference,
        "acceptance_quantile": args.acceptance_quantile,
        "wasserstein_threshold_generations": result["threshold"],
        "threshold_monte_carlo_95_interval_generations": list(result["threshold_interval"]),
        "boundary_probabilities": boundary_set.probabilities.tolist(),
        "requested_boundary_edge_indices": boundary_set.requested_indices.tolist(),
        "compressed_boundary_edge_indices": boundary_set.indices.tolist(),
        "interval_shares": boundary_set.interval_shares.tolist(),
        "repeated_requested_boundaries": int(
            boundary_set.requested_indices.size - boundary_set.indices.size
        ),
        "seed_entropy": result["seed_entropy"],
        "seed_spawn_keys": result["seed_spawn_keys"],
        "numpy_version": np.__version__,
    }
    write_target(args.output, result, metadata)
    print(f"Wrote TE target to {args.output}")
    print(f"W1 acceptance threshold: {result['threshold']:.6g} generations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
