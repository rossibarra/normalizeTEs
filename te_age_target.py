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
from release_provenance import software_provenance


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


def empirical_threshold(distances: np.ndarray, quantile: float = 0.50) -> float:
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
    keep_draws: np.ndarray | None = None,
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
        if keep_draws is not None:
            # The fast writer has no per-draw filter, so a masked target takes
            # the general path. Targets are thousands of rows, not millions, so
            # the cost is bounded.
            cdfs = masked_row_cdfs(store, rows, right_edges, keep_draws).astype(np.float32)
        elif writer is not None:
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


def masked_row_cdfs(store: object, rows: np.ndarray, right_edges: np.ndarray,
                    keep: np.ndarray) -> np.ndarray:
    """Per-site CDFs built only from the draws `keep` marks true.

    A draw that mis-polarized a TE placed its mutation on a different branch, so
    the age it recorded belongs to a different event. Dropping those draws is
    the same decision already taken for Phi-SFS, applied to the ages.

    `keep` is `(n_rows, n_draws)`. A row with no agreeing draw keeps all of its
    draws: an empty CDF is not a better estimate than a contaminated one, and
    silently emitting NaN would propagate into the target.
    """
    from snp_interval_dataset import _batch_cdf, IntervalBatch

    batch = store.intervals(np.asarray(rows, dtype=np.int64))
    below, above, draws = batch.below, batch.above, batch.draw_id
    kept_below, kept_above, kept_draw, offsets = [], [], [], [0]
    dropped = 0
    for i in range(np.asarray(rows).size):
        start, stop = int(batch.offsets[i]), int(batch.offsets[i + 1])
        d = np.asarray(draws[start:stop])
        selector = keep[i][d] if keep[i].any() else np.ones(d.size, dtype=bool)
        dropped += int((~selector).sum())
        kept_below.append(np.asarray(below[start:stop])[selector])
        kept_above.append(np.asarray(above[start:stop])[selector])
        kept_draw.append(d[selector])
        offsets.append(offsets[-1] + int(selector.sum()))
    filtered = IntervalBatch(
        rows=np.asarray(rows, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        below=np.concatenate(kept_below) if kept_below else np.empty(0),
        above=np.concatenate(kept_above) if kept_above else np.empty(0),
        draw_id=np.concatenate(kept_draw) if kept_draw else np.empty(0, dtype=np.int64),
    )
    print(f"  dropped {dropped:,} mis-polarized intervals", flush=True)
    return _batch_cdf(filtered, right_edges, side="left", weighting="interval")


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


@dataclass(frozen=True)
class PolaritySelection:
    """Which draws to trust per TE site, and which TEs to keep at all."""

    keep_draws: np.ndarray          # (n_sites, n_draws) bool, aligned to kept sites
    keep_sites: np.ndarray          # (n_sites_in,) bool over the resolved rows
    report: dict


def load_polarity_selection(
    mask_dir: Path, rows: np.ndarray, store: object,
    max_flipped_fraction: float | None,
) -> PolaritySelection:
    """Read a TE polarity mask and turn it into per-site draw and site filters.

    A draw that called the insertion allele ancestral placed the mutation on a
    different branch of the ARG and recorded that branch's age, so its age is an
    estimate of something else. Those draws are dropped from the site's CDF.

    A site where *every* draw disagrees is a different problem: dropping all of
    its draws would leave no age at all, and an absent site is not a better
    estimate than a contaminated one. Such sites keep their draws and are
    counted in the report, so the fallback is visible rather than silent.
    `max_flipped_fraction` is the deliberate way to remove them instead.
    """
    metadata = json.loads((mask_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "te-polarity-mask-v1":
        raise SystemExit(
            f"{mask_dir}: unsupported polarity mask schema "
            f"{metadata.get('schema_version')!r}; rebuild with "
            "build_te_polarity_mask.py"
        )
    if not metadata.get("complete"):
        raise SystemExit(f"{mask_dir}: polarity mask is incomplete")
    expected = getattr(store, "metadata", {}).get("content_sha256")
    recorded = metadata.get("store_content_sha256")
    if expected and recorded and expected != recorded:
        raise SystemExit(
            f"{mask_dir}: polarity mask was built against a different store "
            f"({recorded[:12]} != {expected[:12]}). Its rows index that store, "
            "so applying it here would mask the wrong sites."
        )
    # A draw the mask does not cover is *unknown*, not flipped, but it reaches
    # the arithmetic below as an all-false column and would be dropped from
    # every site. Six covered draws out of 75 would silently discard 69 of them
    # and still report a healthy-looking target, so partial coverage is refused
    # here rather than trusted. Partial masks remain useful for diagnostics.
    store_draws = (getattr(store, "metadata", {}) or {}).get("n_posterior_draws")
    covered = metadata.get("covered_draw_ids")
    if covered is None:
        raise SystemExit(
            f"{mask_dir}: mask predates draw-id column indexing and cannot be "
            "aligned to the store; rebuild it with build_te_polarity_mask.py"
        )
    if store_draws is not None and sorted(covered) != list(range(int(store_draws))):
        missing = sorted(set(range(int(store_draws))) - set(covered))
        raise SystemExit(
            f"{mask_dir}: covers {len(covered)} of the store's {store_draws} "
            f"posterior draws (missing draw ids {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''}). An uncovered draw is "
            "indistinguishable from a flipped one here, so it would be dropped "
            "from every site. Rebuild the mask against all source trees."
        )
    agrees = np.load(mask_dir / "agrees_with_biology.npy", allow_pickle=False)
    present = np.load(mask_dir / "draw_present.npy", allow_pickle=False)
    mask_rows = np.load(mask_dir / "te_row_indices.npy", allow_pickle=False)
    # Order-sensitive: the mask is a positional array, not a lookup keyed by
    # row. A permuted TE list would mask each site with another site's draws
    # and nothing downstream would notice.
    if mask_rows.shape != rows.shape or not np.array_equal(mask_rows, rows):
        raise SystemExit(
            f"{mask_dir}: polarity mask covers {mask_rows.size:,} sites that do "
            f"not match the {rows.size:,} resolved TE rows in the same order. "
            "Rebuild the mask from this run's target."
        )

    if agrees.shape != present.shape:
        raise SystemExit(f"{mask_dir}: agreement and presence arrays disagree in shape")
    if store_draws is not None and agrees.shape[1] != int(store_draws):
        raise SystemExit(
            f"{mask_dir}: mask has {agrees.shape[1]} draw columns but the store "
            f"has {store_draws} draws"
        )
    usable = present.sum(axis=1)
    agreeing = (present & agrees).sum(axis=1)
    flipped = usable - agreeing
    with np.errstate(invalid="ignore", divide="ignore"):
        flipped_fraction = np.where(usable > 0, flipped / np.maximum(usable, 1), 0.0)

    keep_sites = np.ones(rows.size, dtype=bool)
    if max_flipped_fraction is not None:
        # Sites with no draw at all carry no evidence of being flipped, so the
        # threshold cannot speak to them; they are left to the usual coverage
        # handling rather than discarded here.
        keep_sites = (usable == 0) | (flipped_fraction <= max_flipped_fraction)

    keep_draws = present & agrees
    no_agreeing = keep_draws.sum(axis=1) == 0
    keep_draws[no_agreeing] = present[no_agreeing]

    report = {
        "mask": str(mask_dir.resolve()),
        "n_draws": int(present.shape[1]),
        "draw_site_observations": int(usable.sum()),
        "flipped_observations": int(flipped.sum()),
        "flipped_observation_fraction": (
            float(flipped.sum() / usable.sum()) if usable.sum() else 0.0
        ),
        "sites_with_any_flipped_draw": int((flipped > 0).sum()),
        "sites_with_no_agreeing_draw": int(no_agreeing.sum()),
        "max_flipped_fraction": max_flipped_fraction,
        "sites_discarded_by_threshold": int((~keep_sites).sum()),
        "sites_kept": int(keep_sites.sum()),
    }
    return PolaritySelection(keep_draws[keep_sites], keep_sites, report)


def build_target(
    store: object,
    te_positions: np.ndarray,
    te_chromosomes: np.ndarray,
    te_vcf_positions: np.ndarray,
    *,
    n_replicates: int,
    acceptance_quantile: float,
    seed: int | None,
    acceptance_distance: float | None = None,
    batch_size: int = 256,
    row_indices: np.ndarray | None = None,
    bin_width: int = 1_000,
    scratch_dir: str | Path | None = None,
    cdf_block_rows: int = 512,
    keep_draws: np.ndarray | None = None,
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
            block_rows=cdf_block_rows, keep_draws=keep_draws,
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
    if acceptance_distance is not None:
        if not np.isfinite(acceptance_distance) or acceptance_distance <= 0:
            raise ValueError("acceptance_distance must be finite and positive")
        threshold = float(acceptance_distance)
    else:
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
    parser.add_argument("--acceptance-quantile", type=float, default=0.50)
    parser.add_argument(
        "--acceptance-distance", type=float,
        help=("absolute Wasserstein tolerance in generations; overrides the "
              "bootstrap quantile threshold"),
    )
    parser.add_argument(
        "--te-polarity-mask", type=Path,
        help="directory from build_te_polarity_mask.py. Each TE site's age CDF "
             "is then built only from draws that polarized it in agreement with "
             "biology, because a draw that called the insertion ancestral placed "
             "the mutation on a different branch and recorded that branch's age",
    )
    parser.add_argument(
        "--max-flipped-fraction", type=float, default=None,
        help="discard any TE whose flipped fraction, among draws with data for "
             "it, exceeds this. Requires --te-polarity-mask. A TE the ARG mostly "
             "disagrees with is unreliable whether the cause is inference failure "
             "or a genuine fixed-then-deleted insertion",
    )
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
    included_rows = resolution.included_rows

    if args.max_flipped_fraction is not None and args.te_polarity_mask is None:
        raise SystemExit("--max-flipped-fraction requires --te-polarity-mask")
    if args.max_flipped_fraction is not None and not 0.0 <= args.max_flipped_fraction <= 1.0:
        raise SystemExit("--max-flipped-fraction must lie in [0, 1]")

    keep_draws = None
    polarity_report: dict | None = None
    if args.te_polarity_mask is not None:
        selection = load_polarity_selection(
            args.te_polarity_mask, np.asarray(included_rows, dtype=np.int64),
            store, args.max_flipped_fraction,
        )
        polarity_report = selection.report
        keep = selection.keep_sites
        if not keep.any():
            raise SystemExit(
                "--max-flipped-fraction discarded every TE; raise the threshold"
            )
        positions = positions[keep]
        included_chromosomes = included_chromosomes[keep]
        included_vcf_positions = included_vcf_positions[keep]
        included_rows = np.asarray(included_rows)[keep]
        keep_draws = selection.keep_draws
        print(
            f"polarity mask   {polarity_report['flipped_observations']:,} of "
            f"{polarity_report['draw_site_observations']:,} draw-site ages dropped "
            f"({polarity_report['flipped_observation_fraction']:.2%})"
        )
        print(
            f"TEs discarded   {polarity_report['sites_discarded_by_threshold']:,} "
            f"of {keep.size:,}, {polarity_report['sites_kept']:,} kept"
        )
        if polarity_report["sites_with_no_agreeing_draw"]:
            print(
                f"  note: {polarity_report['sites_with_no_agreeing_draw']:,} kept TEs "
                "had no agreeing draw and retain all of theirs"
            )

    result = build_target(
        store,
        positions,
        included_chromosomes,
        included_vcf_positions,
        n_replicates=args.bootstrap_replicates,
        acceptance_quantile=args.acceptance_quantile,
        acceptance_distance=args.acceptance_distance,
        seed=args.seed,
        batch_size=args.bootstrap_batch_size,
        row_indices=included_rows,
        bin_width=args.bin_width,
        scratch_dir=args.scratch_dir,
        cdf_block_rows=args.cdf_block_rows,
        keep_draws=keep_draws,
    )
    boundary_set = result.boundaries
    metadata = {
        "schema_version": 2,
        "software": software_provenance(),
        "source_store": str(args.store.resolve()),
        "source_store_schema": store_schema(store),
        "source_catalog_sha256": getattr(store, "metadata", {}).get("catalog_sha256"),
        "source_store_content_sha256": getattr(
            store, "metadata", {}
        ).get("content_sha256"),
        "te_position_source": str(args.te_positions.resolve()),
        "n_te_snps": int(positions.size),
        "effective_te_set_size": int(positions.size),
        "position_resolution": resolution.summary(),
        "excluded_positions": resolution.excluded_coordinates(),
        "missing_position_policy": args.missing_position_policy,
        "te_polarity": polarity_report,
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
        "acceptance_distance": args.acceptance_distance,
        "acceptance_threshold_source": (
            "absolute_distance"
            if args.acceptance_distance is not None
            else "bootstrap_quantile"
        ),
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
    print(
        f"W1 acceptance threshold: {result.threshold:.6g} generations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
