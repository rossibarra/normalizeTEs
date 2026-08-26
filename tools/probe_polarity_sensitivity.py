"""How far can ARG polarization error move Phi-SFS?

The real analysis is blocked: `normalize_tes.phi_sfs` cannot read polarity from the
ancestral table yet (C5), and only chromosome 10 of the input VCF is available.
This is a scoped stand-in that answers the magnitude question the risk section
raises, using the real projection and scoring code on real chr10 data.

It builds a TE spectrum and a size-matched control-SNP spectrum from chr10,
polarized by the ARG's own majority ancestral call, then perturbs that polarity
with the measured frequency-dependent error rate and re-scores. The spread of
Phi under perturbation, next to the observed Phi, says whether polarization
error is a rounding detail or a first-order term.

It is NOT the scientific result: the control set is a frequency-blind random
sample rather than an age-matched one, and chr10 alone. Read it only as a
sensitivity magnitude.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
from normalize_tes.phi_sfs import PROJECTION_SIZE, hypergeometric_projection, phi_sfs

BASES = "ACGT"


def read_chr10(vcf: str):
    """Return per-site TE flag, REF/ALT bases, ALT count and callable count."""
    is_te, ref, alt, kk, nn, pos = [], [], [], [], [], []
    with gzip.open(vcf, "rt") as h:
        for raw in h:
            if raw.startswith("#"):
                continue
            f = raw.rstrip("\n").split("\t")
            if "," in f[4]:
                continue
            gts = [g.split(":")[0] for g in f[9:]]
            k = sum(1 for g in gts if g == "1")
            n = sum(1 for g in gts if g in ("0", "1"))
            if n < PROJECTION_SIZE or k == 0 or k == n:
                continue
            is_te.append(f[2] != "."); ref.append(f[3]); alt.append(f[4])
            kk.append(k); nn.append(n); pos.append(int(f[1]))
    return (np.array(is_te), np.array(ref), np.array(alt),
            np.array(kk, float), np.array(nn, float), np.array(pos))


def spectrum(derived: np.ndarray, callable_n: np.ndarray, cache: dict) -> np.ndarray:
    """Return the normalized spectrum over bins 1..19.

    Per README §7 the per-site projections are summed first and the completed
    spectrum normalized once, so a site whose derived count is likely to project
    to an endpoint contributes proportionally less polymorphic mass. Normalizing
    per site would discard that weighting. `phi_sfs` requires the result to sum
    to one.
    """
    total = np.zeros(PROJECTION_SIZE - 1)
    for k, n in zip(derived.astype(int), callable_n.astype(int)):
        key = (k, n)
        v = cache.get(key)
        if v is None:
            v = hypergeometric_projection(k, n)[1:PROJECTION_SIZE]
            cache[key] = v
        total += v
    mass = total.sum()
    if mass <= 0:
        raise ValueError("spectrum has no polymorphic mass")
    return total / mass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcf", required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--ancestral", type=Path, required=True)
    p.add_argument("--draws", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--flip-scale", type=float, default=1.0,
                   help="multiplier on the fitted flip probability; 1.0 is the "
                        "pessimistic frequency-shortfall model, lower values "
                        "match the rate measured against the TE truth set")
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()

    is_te, ref, alt, k, n, pos = read_chr10(a.vcf)
    meta = json.load(open(a.store / "metadata.json"))
    off = {c["chrom"]: c["offset"] for c in meta["chromosomes"]}["10"]
    cat = np.asarray(np.load(a.store / "positions.npy", mmap_mode="r"))
    g = pos.astype(float) + off
    ins = np.searchsorted(cat, g); ok = ins < cat.size
    ok[ok] &= cat[ins[ok]] == g[ok]
    rows = ins[ok]
    is_te, ref, alt, k, n = is_te[ok], ref[ok], alt[ok], k[ok], n[ok]

    A = np.load(a.ancestral / "ancestral_counts.npy", mmap_mode="r")
    cnt = np.asarray(A[rows], float)
    anc = np.array([BASES[i] for i in cnt.argmax(axis=1)])
    alt_is_derived = anc == ref            # ARG polarity: derived = the non-ancestral allele
    derived = np.where(alt_is_derived, k, n - k)

    minor = np.minimum(k / n, 1 - k / n)
    # Measured error rate: ancestral=major runs 77.5% at minor freq <0.05 down to
    # 54.7% near 0.5. Treat the shortfall from certainty as the flip probability,
    # which is deliberately pessimistic -- it charges every frequency-driven call
    # as potentially wrong.
    err = np.clip((0.225 + 0.55 * minor) * a.flip_scale, 0.0, 0.5)

    rng = np.random.default_rng(a.seed)
    te_m, sn_m = is_te, ~is_te
    idx_sn = rng.choice(np.flatnonzero(sn_m), size=int(te_m.sum()), replace=False)
    sel = np.zeros(is_te.size, bool); sel[np.flatnonzero(te_m)] = True; sel[idx_sn] = True

    cache: dict = {}
    obs = phi_sfs(spectrum(derived[te_m], n[te_m], cache),
                  spectrum(derived[idx_sn], n[idx_sn], cache))
    print(f"chr10 sites used: {int(te_m.sum()):,} TE vs {len(idx_sn):,} control SNPs")
    print(f"observed Phi-SFS (ARG polarity as called): {obs.value:.4f}\n"
          f"mean pessimistic flip probability: {err[sel].mean():.3f}\n")

    phis = np.empty(a.draws)
    for d in range(a.draws):
        flip = rng.random(is_te.size) < err
        der = np.where(flip, n - derived, derived)
        phis[d] = phi_sfs(spectrum(der[te_m], n[te_m], cache),
                          spectrum(der[idx_sn], n[idx_sn], cache)).value
        if d % 50 == 0:
            print(f"  draw {d}: Phi={phis[d]:.4f}", flush=True)
    lo, hi = np.percentile(phis, [2.5, 97.5])
    print(f"\nPhi under perturbed polarity: median {np.median(phis):.4f}  "
          f"95% interval [{lo:.4f}, {hi:.4f}]")
    print(f"shift from observed: {np.median(phis) - obs.value:+.4f} "
          f"({(np.median(phis)-obs.value)/max(obs.value,1e-9):+.1%})")
    a.output.write_text(json.dumps({
        "observed_phi": obs.value, "n_te": int(te_m.sum()), "n_control": len(idx_sn),
        "draws": a.draws, "mean_flip_probability": float(err[sel].mean()),
        "flip_scale": a.flip_scale,
        "perturbed_phi_median": float(np.median(phis)),
        "perturbed_phi_2.5pct": float(lo), "perturbed_phi_97.5pct": float(hi),
        "caveat": "chr10 only; frequency-blind control sample; not the scientific result",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
