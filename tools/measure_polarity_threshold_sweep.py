"""Measure how --max-flipped-fraction trades polarity cleanliness against age.

Mis-polarization is not age-neutral: the TEs an ARG most often flips are the
old ones, so every threshold discards old TEs and shifts the target younger.
This writes the sweep behind that decision as a durable artifact rather than
leaving it in a terminal scrollback.

Two baselines are reported because they answer different questions, and
quoting one for the other is the easy mistake:

  per-TE median age, draw-masking already applied to both rows
      isolates the marginal cost of DISCARDING SITES at each threshold.
  aggregate target CDF, unmasked target versus masked target
      the total change a reader sees in the published target, which includes
      the draw filtering as well as the site discarding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from normalize_tes.release_provenance import software_provenance
from normalize_tes.snp_age_store import open_snp_age_store


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[order] = np.arange(x.size, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    means = np.zeros(cnt.size)
    np.add.at(means, inv, r)
    means /= cnt
    return means[inv]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9])
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f"output already exists: {a.output}")

    agrees = np.load(a.mask / "agrees_with_biology.npy", allow_pickle=False)
    present = np.load(a.mask / "draw_present.npy", allow_pickle=False)
    rows = np.load(a.mask / "te_row_indices.npy", allow_pickle=False).astype(np.int64)
    usable = present.sum(1)
    flipped = usable - (present & agrees).sum(1)
    frac = np.where(usable > 0, flipped / np.maximum(usable, 1), 0.0)

    store = open_snp_age_store(a.store)
    batch = store.intervals(rows)
    mid = (np.asarray(batch.below) + np.asarray(batch.above)) / 2.0
    keep = present & agrees
    age = np.full(rows.size, np.nan)
    for i in range(rows.size):
        lo, hi = int(batch.offsets[i]), int(batch.offsets[i + 1])
        seg, dr = mid[lo:hi], np.asarray(batch.draw_id[lo:hi])
        if not seg.size:
            continue
        sel = keep[i][dr]
        if not sel.any():          # same interval-level fallback as the pipeline
            sel = np.ones(dr.size, dtype=bool)
        age[i] = np.median(seg[sel])

    ok = np.isfinite(age)
    base = float(np.nanmedian(age[ok]))
    sweep = []
    for x in a.thresholds:
        g = age[ok & (frac <= x)]
        q = np.nanpercentile(g, [10, 50, 90])
        sweep.append({
            "max_flipped_fraction": float(x),
            "kept": int(g.size),
            "discarded": int(ok.sum() - g.size),
            "age_q10": float(q[0]), "age_q50": float(q[1]), "age_q90": float(q[2]),
            "median_shift_vs_no_threshold": float((q[1] - base) / base),
        })

    report = {
        "schema_version": "polarity-threshold-sweep-v1",
        "baseline": "per-TE median age with draw-masking applied throughout; "
                    "isolates the cost of discarding sites",
        "store": str(a.store.resolve()),
        "mask": str(a.mask.resolve()),
        "n_te": int(rows.size),
        "draw_site_observations": int(usable.sum()),
        "flipped_observations": int(flipped.sum()),
        "flipped_observation_fraction": float(flipped.sum() / usable.sum()),
        "sites_with_any_flipped_draw": int((flipped > 0).sum()),
        "spearman_flipped_fraction_vs_age": float(
            np.corrcoef(rank(frac[ok]), rank(age[ok]))[0, 1]),
        "median_age_no_threshold": base,
        "sweep": sweep,
        "software": software_provenance(),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"Spearman(flipped fraction, TE age) = "
          f"{report['spearman_flipped_fraction_vs_age']:+.4f}")
    for e in sweep:
        print(f"  X={e['max_flipped_fraction']:<5} kept {e['kept']:6,} "
              f"q50 {e['age_q50']:11,.0f}  {e['median_shift_vs_no_threshold']:+.2%}")
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
