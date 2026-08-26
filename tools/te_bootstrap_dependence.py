"""Test spatial dependence among TE age-CDF contributions (plan gate 2).

`normalize_tes.te_age_target` bootstraps TE sites with an iid multinomial draw, and the
median of the resulting Wasserstein distances becomes the matching threshold.
That is only defensible if TE age contributions are exchangeable across sites.
They plausibly are not: TEs are clustered along chromosomes, and neighbouring
sites share the tree structure the ages were inferred from, so their age
posteriors are correlated by construction.

This command supplies the evidence gate 2 requires, and the block-bootstrap
mode it may select. It reports two things:

1. **Spatial autocorrelation** of a scalar age summary per TE site, as a
   variogram over genomic-distance bins plus a Moran's I against a
   nearest-neighbour weight. This says whether dependence exists.
2. **The bootstrap W1 distribution under iid and under cluster resampling**, on
   the exact analysis grid. This says whether the dependence *matters*, because
   the q50 of that distribution is the acceptance threshold every downstream
   stage inherits.

If cluster resampling widens the distribution materially, the iid threshold is
too tight and every matched set has been held to a tolerance narrower than the
data support. Read the two together: autocorrelation that does not move the
threshold is a curiosity, and a threshold shift is what changes the analysis.

**The resampling unit is a cluster, not a fixed-length block.** Clusters are
single-linkage groups of TE sites within `--link-bp` of each other on one
chromosome, and they are resampled with replacement to approximately the
original site count. A fixed count of consecutive sites is the wrong unit here:
median within-chromosome TE spacing is about 153 kb while the variogram locates
dependence below 10 kb, so even two consecutive sites usually span thirty times
the correlation range. Blocking by count would mostly group independent sites
and inflate the spread through the mechanical cost of resampling fewer, larger
units rather than through any dependence. Linkage distance lets the measured
correlation range choose the grouping instead of a block length chosen by hand.

The permuted arm reassigns cluster membership at random while preserving the
cluster size distribution exactly, which isolates the mechanical component. Only
the excess of observed over permuted is evidence of dependence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from normalize_tes.release_provenance import software_provenance
from normalize_tes.sample_age_matched_controls import _load_target
from normalize_tes.snp_age_store import open_snp_age_store
from normalize_tes.swap_control_sampler import analysis_points, row_cdfs
from normalize_tes.te_age_target import wasserstein_1


def age_summary(cdf_rows: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return each site's mean age, the integral of its survival function.

    A scalar per site is what the spatial statistics below need. The mean is
    used rather than a quantile because it is a linear functional of the CDF,
    so its spatial structure is the spatial structure of the quantity the
    bootstrap actually averages.
    """
    survival = 1.0 - np.asarray(cdf_rows, dtype=np.float64)
    widths = np.diff(np.asarray(points, dtype=np.float64), prepend=0.0)
    return survival @ widths


def variogram(values: np.ndarray, positions: np.ndarray, chromosomes: np.ndarray,
              edges: np.ndarray) -> list[dict]:
    """Semivariance of `values` by within-chromosome genomic distance bin.

    A flat variogram means no spatial structure. One that rises to a sill
    indicates dependence out to the range where it levels off, which is the
    scale a block bootstrap has to exceed to be useful.
    """
    order = np.lexsort((positions, chromosomes))
    values, positions, chromosomes = values[order], positions[order], chromosomes[order]
    total_variance = float(np.var(values))
    results: list[dict] = []
    for low, high in zip(edges[:-1], edges[1:]):
        numerator = 0.0
        count = 0
        for chrom in np.unique(chromosomes):
            mask = chromosomes == chrom
            p, v = positions[mask], values[mask]
            # Sites are sorted, so the pairs within a distance window are a
            # sliding range rather than a full O(n^2) comparison.
            start = np.searchsorted(p, p + low, side="left")
            stop = np.searchsorted(p, p + high, side="right")
            for index in range(p.size):
                partners = slice(max(start[index], index + 1), stop[index])
                if partners.start >= partners.stop:
                    continue
                differences = v[partners] - v[index]
                numerator += float(np.sum(differences * differences))
                count += differences.size
        results.append({
            "low_bp": float(low),
            "high_bp": float(high),
            "pairs": count,
            "semivariance": numerator / (2 * count) if count else None,
            "ratio_to_variance": (numerator / (2 * count) / total_variance)
            if count and total_variance > 0 else None,
        })
    return results


def morans_i(values: np.ndarray, positions: np.ndarray, chromosomes: np.ndarray,
             neighbours: int) -> float:
    """Moran's I against a k-nearest-within-chromosome adjacency weight.

    Expectation under no autocorrelation is -1/(n-1), effectively zero here.
    """
    order = np.lexsort((positions, chromosomes))
    values = values[order]
    chromosomes_sorted = chromosomes[order]
    deviations = values - values.mean()
    numerator = 0.0
    weight_total = 0.0
    for chrom in np.unique(chromosomes_sorted):
        index = np.flatnonzero(chromosomes_sorted == chrom)
        for shift in range(1, neighbours + 1):
            if index.size <= shift:
                break
            left, right = index[:-shift], index[shift:]
            numerator += float(np.sum(deviations[left] * deviations[right])) * 2.0
            weight_total += 2.0 * left.size
    denominator = float(np.sum(deviations * deviations))
    if denominator == 0 or weight_total == 0:
        return float("nan")
    return (values.size / weight_total) * (numerator / denominator)


def cluster_labels(positions: np.ndarray, chromosomes: np.ndarray,
                   link_bp: float) -> np.ndarray:
    """Single-linkage cluster label per site, for position-ordered input.

    A new cluster starts at a chromosome change or a gap wider than `link_bp`.
    `link_bp <= 0` makes every site its own cluster, which is the iid case.
    """
    if link_bp <= 0:
        return np.arange(positions.size, dtype=np.int64)
    breaks = (chromosomes[1:] != chromosomes[:-1]) | (np.diff(positions) > link_bp)
    return np.r_[0, np.cumsum(breaks)].astype(np.int64)


def cluster_members(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return site indices grouped by cluster, as a flat array plus offsets."""
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels)
    return order, np.r_[0, np.cumsum(counts)]


def cluster_resample(members: np.ndarray, offsets: np.ndarray, n_sites: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw clusters with replacement until at least `n_sites` sites are taken.

    Clusters vary in size, so the resample is truncated to exactly `n_sites` to
    keep every replicate's denominator identical. Without that, a replicate that
    happened to draw large clusters would average more sites than one that drew
    small ones, and the W1 spread would mix resampling variability with a
    varying sample size.
    """
    n_clusters = offsets.size - 1
    sizes = np.diff(offsets)
    drawn = rng.integers(0, n_clusters, size=n_sites)
    lengths = sizes[drawn]
    keep = int(np.searchsorted(np.cumsum(lengths), n_sites, side="left")) + 1
    drawn, lengths = drawn[:keep], lengths[:keep]

    # Ragged-range gather: expand each drawn cluster's member slice without a
    # Python loop. Building this per replicate in Python would dominate the run.
    total = int(lengths.sum())
    starts = offsets[drawn]
    shift = np.repeat(starts - np.cumsum(np.r_[0, lengths[:-1]]), lengths)
    return members[shift + np.arange(total)][:n_sites]


def bootstrap_distances(cdf_rows: np.ndarray, observed: np.ndarray,
                        age_bins: np.ndarray, *, members: np.ndarray,
                        offsets: np.ndarray, replicates: int,
                        rng: np.random.Generator, batch: int = 128) -> np.ndarray:
    """Return exact-grid W1 of `replicates` block-resampled means to `observed`.

    `cdf_rows` must already be float64. Rows are stored as float32 to bound
    memory, but a float32 accumulation over thousands of sites loses precision
    in the tail that every distance is derived from, so the caller promotes once
    rather than per batch.
    """
    if cdf_rows.dtype != np.float64:
        raise ValueError("cdf_rows must be promoted to float64 by the caller")
    n_sites = cdf_rows.shape[0]
    distances = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        counts = np.zeros((stop - start, n_sites), dtype=np.float64)
        for row in range(stop - start):
            picked = cluster_resample(members, offsets, n_sites, rng)
            np.add.at(counts[row], picked, 1.0)
        counts /= n_sites
        means = counts @ cdf_rows
        for row in range(stop - start):
            distances[start + row] = wasserstein_1(means[row], observed, age_bins)
    return distances


def summarize(distances: np.ndarray) -> dict:
    quantiles = [5, 25, 50, 75, 95, 99]
    return {
        "mean": float(distances.mean()),
        "sd": float(distances.std(ddof=1)),
        "quantiles": {
            f"q{q}": float(np.percentile(distances, q)) for q in quantiles
        },
        "max": float(distances.max()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument(
        "--link-bp", type=float, nargs="+",
        default=[0, 1e3, 1e4, 5e4, 1e5],
        help="single-linkage clustering distances; 0 is the iid reference",
    )
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--cdf-block-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    rows, observed, age_bins, threshold, target_meta = _load_target(args.target)
    points = analysis_points(age_bins)

    positions = np.load(args.target / "te_global_positions.npy", allow_pickle=False)
    chromosomes = np.load(args.target / "te_chromosomes.npy", allow_pickle=False)
    print(f"{rows.size:,} TE sites, exact grid {age_bins.size:,} points", flush=True)

    cdf_rows = row_cdfs(store, rows, points,
                        block_rows=args.cdf_block_rows, dtype=np.dtype("float32"))
    print("per-site CDFs built", flush=True)

    values = age_summary(cdf_rows, points)
    cdf_rows = cdf_rows.astype(np.float64)
    edges = np.array([0, 1e4, 1e5, 1e6, 5e6, 2e7], dtype=np.float64)
    spatial = {
        "age_summary": "mean age (integral of survival)",
        "variogram": variogram(values, positions.astype(np.float64),
                               chromosomes, edges),
        "morans_i": morans_i(values, positions.astype(np.float64),
                             chromosomes, args.neighbours),
        "morans_i_neighbours": args.neighbours,
        "morans_i_expectation": -1.0 / (rows.size - 1),
    }
    print(f"Moran's I = {spatial['morans_i']:.4f} "
          f"(expected {spatial['morans_i_expectation']:.2e})", flush=True)

    # Order sites genomically so single-linkage clustering sees true neighbours.
    order = np.lexsort((positions, chromosomes))
    cdf_rows = cdf_rows[order]
    ordered_positions = positions[order].astype(np.float64)
    ordered_chrom = chromosomes[order]
    n_sites = cdf_rows.shape[0]

    rng = np.random.default_rng(args.seed)
    per_link: dict[str, dict] = {}
    for link in args.link_bp:
        labels = cluster_labels(ordered_positions, ordered_chrom, link)
        members, offsets = cluster_members(labels)
        sizes = np.diff(offsets)
        entry: dict = {
            "clusters": int(sizes.size),
            "largest_cluster": int(sizes.max()),
            "singletons": int(np.count_nonzero(sizes == 1)),
            "effective_n_loss": 1.0 - sizes.size / n_sites,
        }
        # The permuted arm keeps the cluster size distribution exactly but
        # reassigns which sites belong together, so it isolates the mechanical
        # cost of resampling fewer, larger units from genuine dependence. Only
        # the excess of observed over permuted is evidence of dependence.
        shuffled = members[rng.permutation(members.size)]
        for label, group_members in (("observed", members), ("permuted", shuffled)):
            distances = bootstrap_distances(
                cdf_rows, observed, age_bins,
                members=group_members, offsets=offsets,
                replicates=args.replicates, rng=rng,
            )
            entry[label] = summarize(distances)
        for statistic in ("q50", "q95"):
            observed_value = entry["observed"]["quantiles"][statistic]
            permuted_value = entry["permuted"]["quantiles"][statistic]
            entry[f"excess_over_permuted_{statistic}"] = (
                observed_value / permuted_value if permuted_value else None
            )
        per_link[str(int(link))] = entry
        print(f"link={int(link):>9,}bp clusters={entry['clusters']:>6,} | "
              f"obs q50={entry['observed']['quantiles']['q50']:8.2f} "
              f"q95={entry['observed']['quantiles']['q95']:8.2f} | "
              f"perm q50={entry['permuted']['quantiles']['q50']:8.2f} "
              f"q95={entry['permuted']['quantiles']['q95']:8.2f} | "
              f"excess q50={entry['excess_over_permuted_q50']:.4f} "
              f"q95={entry['excess_over_permuted_q95']:.4f}", flush=True)

    iid_q50 = per_link[str(int(args.link_bp[0]))]["observed"]["quantiles"]["q50"]
    report = {
        "schema_version": "te-bootstrap-dependence-v1",
        "store": str(args.store),
        "target": str(args.target),
        "n_te_sites": int(rows.size),
        "exact_grid_points": int(age_bins.size),
        "target_threshold_generations": float(threshold),
        "target_source_store_content_sha256": target_meta.get(
            "source_store_content_sha256"),
        "replicates": args.replicates,
        "seed": args.seed,
        "spatial": spatial,
        "bootstrap_by_link_bp": per_link,
        "q50_ratio_to_iid": {
            link: per_link[link]["observed"]["quantiles"]["q50"] / iid_q50
            for link in per_link
        },
        "interpretation": (
            "excess_over_permuted_* is the dependence signal; the raw "
            "cluster-versus-iid ratio confounds dependence with the mechanical "
            "variance of resampling fewer, larger units"
        ),
        "software": software_provenance(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
