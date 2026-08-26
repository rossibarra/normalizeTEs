"""Probe whether the optimizer's matching-error floor is algorithmic.

T5 leaves `E_r` confined to 222.9-336.5 generations, and 294 of 300 restarts
stopped on a material-improvement plateau rather than an epoch cap, so more of
the same search cannot lower it. That plateau is defined over *single*-site
swaps: it means no one swap improves W1 by the material threshold. It says
nothing about whether a simultaneous pair of swaps would.

This distinction decides whether a better search is worth building. If pairs of
individually-worsening swaps combine to improve W1, the floor is a property of
the single-swap neighbourhood and annealing or multi-swap moves have headroom.
If no pair improves either, the floor is deeper than the move set and a
different search would be wasted effort.

The probe samples slots and candidates, confirms the published state really is a
single-swap local optimum over that sample, then searches the pair
neighbourhood. Everything is screened on the coarse grid the optimizer itself
searches on, and any hit is re-certified on the exact grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from normalize_tes.snp_age_store import open_snp_age_store
from normalize_tes.swap_control_sampler import analysis_points, row_cdfs, aggregate_cdf, search_grid
from normalize_tes.te_age_target import wasserstein_1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True)
    p.add_argument("--candidate-rows", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True,
                   help="target bundle, for the TE rows the bootstrap targets weight")
    p.add_argument("--replicates", type=int, nargs="+", required=True)
    p.add_argument("--slots", type=int, default=300)
    p.add_argument("--candidates", type=int, default=400)
    p.add_argument("--anchors", type=int, default=3000,
                   help="least-worsening single moves kept for the pair search")
    p.add_argument("--search-bin-width", type=float, default=20000.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    store = open_snp_age_store(args.store)
    rows_all = np.load(args.matches / "row_indices.npy")
    targets = np.load(args.matches / "bootstrap_target_cdfs.npy", mmap_mode="r")
    ages = np.load(args.matches / "age_bins.npy")
    pool = np.load(args.candidate_rows, mmap_mode="r")
    exact_pts = analysis_points(ages)
    # Reuse the optimizer's own grid builder so the probe searches exactly the
    # neighbourhood the optimizer plateaued on.
    c_ages, c_pts = search_grid(float(store.metadata["maximum_above"]),
                                int(args.search_bin_width))
    rng = np.random.default_rng(args.seed)
    print(f"coarse grid {c_ages.size:,} points, exact {ages.size:,}", flush=True)

    # Rebuild each bootstrap target on the coarse grid from its published
    # multinomial counts, as the matcher did. Interpolating the exact-grid CDF
    # instead would compare against a slightly different target than the one
    # the search actually used.
    te_rows = np.load(args.target / "te_row_indices.npy").astype(np.int64)
    counts = np.load(args.matches / "bootstrap_counts.npy", mmap_mode="r")
    te_coarse = row_cdfs(store, te_rows, c_pts, block_rows=256,
                         dtype=np.dtype("float64"))
    print(f"TE coarse rows built ({te_rows.size:,} sites)", flush=True)

    report = []
    for r in args.replicates:
        sel = rows_all[r].astype(np.int64)
        n = sel.size
        tgt_exact = np.asarray(targets[r], dtype=np.float64)
        w = np.asarray(counts[r], dtype=np.float64)
        tgt_coarse = (w / w.sum()) @ te_coarse

        sel_c = row_cdfs(store, sel, c_pts, block_rows=256, dtype=np.dtype("float64"))
        cur_c = sel_c.mean(axis=0)
        base_coarse = wasserstein_1(cur_c, tgt_coarse, c_ages)
        base_exact = wasserstein_1(aggregate_cdf(store, sel, exact_pts), tgt_exact, ages)

        slots = rng.choice(n, size=min(args.slots, n), replace=False)
        cand = np.asarray(pool)[rng.choice(pool.shape[0], size=args.candidates, replace=False)]
        cand = cand[~np.isin(cand, sel)]
        cand_c = row_cdfs(store, cand, c_pts, block_rows=256, dtype=np.dtype("float64"))

        # delta[s, k] = (cand_k - sel_slot_s) / n, the CDF change of one swap
        delta = (cand_c[None, :, :] - sel_c[slots][:, None, :]) / n
        flat = delta.reshape(-1, c_ages.size)
        resid = cur_c - tgt_coarse
        widths = np.diff(c_ages, prepend=0.0)
        singles = np.abs(resid[None, :] + flat) @ widths
        best_single = float(singles.min())

        # Pair search over the least-worsening singles, excluding pairs that
        # reuse a slot or a candidate (those are not simultaneous swaps).
        order = np.argsort(singles)[: args.anchors]
        slot_of = (order // cand.size).astype(np.int64)
        cand_of = (order % cand.size).astype(np.int64)
        sub = flat[order]
        best_pair, best_ij = np.inf, None
        for a in range(sub.shape[0]):
            ok = (slot_of[a + 1:] != slot_of[a]) & (cand_of[a + 1:] != cand_of[a])
            if not ok.any():
                continue
            vals = np.abs(resid + sub[a] + sub[a + 1:][ok]) @ widths
            m = int(vals.argmin())
            if vals[m] < best_pair:
                best_pair = float(vals[m])
                best_ij = (a, int(np.flatnonzero(ok)[m]) + a + 1)

        entry = {
            "replicate": r,
            "base_coarse_w1": base_coarse,
            "base_exact_w1": base_exact,
            "best_single_coarse_w1": best_single,
            "single_improves": bool(best_single < base_coarse),
            "single_gain": base_coarse - best_single,
            "best_pair_coarse_w1": best_pair,
            "pair_improves": bool(best_pair < base_coarse),
            "pair_gain": base_coarse - best_pair,
            "pair_beats_best_single": bool(best_pair < best_single),
        }
        if best_ij is not None:
            a, b = best_ij
            ia, ib = int(order[a]), int(order[b])
            trial = sel.copy()
            trial[slots[ia // cand.size]] = cand[ia % cand.size]
            trial[slots[ib // cand.size]] = cand[ib % cand.size]
            if np.unique(trial).size == trial.size:
                entry["pair_exact_w1"] = float(
                    wasserstein_1(aggregate_cdf(store, trial, exact_pts), tgt_exact, ages))
                entry["exact_gain"] = base_exact - entry["pair_exact_w1"]
        report.append(entry)
        print(f"rep {r:>3}: base {base_coarse:8.2f} | best single {best_single:8.2f} "
              f"({'improves' if entry['single_improves'] else 'no gain'}) | "
              f"best pair {best_pair:8.2f} "
              f"({'improves' if entry['pair_improves'] else 'no gain'})"
              + (f" | exact {base_exact:.2f} -> {entry['pair_exact_w1']:.2f}"
                 if "pair_exact_w1" in entry else ""), flush=True)

    args.output.write_text(json.dumps(
        {"config": vars(args) | {k: str(v) for k, v in vars(args).items()
                                 if isinstance(v, Path)},
         "results": report}, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
