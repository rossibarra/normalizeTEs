"""Compare a target and its matched sets before and after polarity masking.

Reports the quantities that the polarity fix can move: the target age CDF, the
acceptance threshold, and the per-replicate distances. Old and new QC pass
counts are not directly comparable, because the absolute cap is a fraction of
the acceptance threshold and that threshold itself moves, so the cap is printed
alongside the count rather than the count alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def quantiles(cdf: np.ndarray, bins: np.ndarray, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    # Grid values are bin labels; a cell holds P(age < label + width/2).
    width = float(bins[1] - bins[0])
    centres = bins + width / 2.0
    return [float(np.interp(q, cdf, centres)) for q in qs]


def load_target(path: Path) -> dict:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    cdf = np.load(path / "target_cdf.npy")
    bins = np.load(path / "age_bins.npy")
    return {
        "meta": meta, "cdf": cdf, "bins": bins,
        "n_te": int(meta["n_te_snps"]),
        "threshold": float(meta["wasserstein_threshold_generations"]),
        "quantiles": quantiles(cdf, bins),
        "polarity": meta.get("te_polarity"),
    }


def load_matches(path: Path) -> dict | None:
    if not (path / "metadata.json").exists():
        return None
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    out = {"meta": meta, "qc_passes": meta.get("qc_passes"),
           "max_reuse": meta.get("maximum_control_reuse")}
    for key, name in (("B", "bootstrap_distances"), ("E", "match_to_bootstrap_w1"),
                      ("O", "match_to_observed_w1")):
        f = path / f"{name}.npy"
        if f.exists():
            out[key] = np.load(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-target", type=Path, required=True)
    ap.add_argument("--new-target", type=Path, required=True)
    ap.add_argument("--old-matches", type=Path)
    ap.add_argument("--new-matches", type=Path)
    a = ap.parse_args()

    old, new = load_target(a.old_target), load_target(a.new_target)
    print(f"{'':22s} {'before':>14s} {'after':>14s} {'change':>12s}")
    print(f"{'TE sites':22s} {old['n_te']:14,d} {new['n_te']:14,d} "
          f"{new['n_te']-old['n_te']:+12,d}")
    t0, t1 = old["threshold"], new["threshold"]
    print(f"{'W1 threshold (gen)':22s} {t0:14,.2f} {t1:14,.2f} "
          f"{100*(t1-t0)/t0:+11.2f}%")
    for label, q0, q1 in zip(("age q10", "age q25", "age q50", "age q75", "age q90"),
                             old["quantiles"], new["quantiles"]):
        print(f"{label:22s} {q0:14,.0f} {q1:14,.0f} {100*(q1-q0)/q0:+11.2f}%")

    if new["polarity"]:
        p = new["polarity"]
        print(f"\npolarity mask: {p['flipped_observations']:,} of "
              f"{p['draw_site_observations']:,} draw-site ages dropped "
              f"({p['flipped_observation_fraction']:.2%}); "
              f"{p['sites_discarded_by_threshold']:,} TEs discarded at "
              f"max_flipped_fraction={p['max_flipped_fraction']}")
        if p.get("sites_with_no_agreeing_draw"):
            print(f"  {p['sites_with_no_agreeing_draw']:,} kept TEs had no agreeing "
                  "draw and retain all of theirs")

    if a.old_matches and a.new_matches:
        om, nm = load_matches(a.old_matches), load_matches(a.new_matches)
        if om and nm:
            print(f"\n{'':22s} {'before':>14s} {'after':>14s}")
            for key, label in (("B", "median B_r"), ("E", "median E_r"), ("O", "median O_r")):
                if key in om and key in nm:
                    print(f"{label:22s} {np.median(om[key]):14,.2f} "
                          f"{np.median(nm[key]):14,.2f}")
            if "B" in om and "O" in om and "B" in nm and "O" in nm:
                print(f"{'cor(B_r, O_r)':22s} "
                      f"{np.corrcoef(om['B'], om['O'])[0,1]:14.4f} "
                      f"{np.corrcoef(nm['B'], nm['O'])[0,1]:14.4f}")
            # The cap moves with the threshold, so the pass count alone misleads.
            for tag, m, thr in (("before", om, t0), ("after", nm, t1)):
                frac = m["meta"].get("config", {}).get("qc_max_absolute_fraction")
                cap = m["meta"].get("config", {}).get("qc_max_absolute")
                cap = cap if cap is not None else (frac * thr if frac else None)
                capstr = f"{cap:,.1f}" if cap else "n/a"
                print(f"QC {tag:6s}: {m['qc_passes']}/100 passed, absolute cap "
                      f"{capstr} gen, max control reuse {m['max_reuse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
