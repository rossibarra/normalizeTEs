"""C2: does W1-repair utility correlate with derived allele frequency?

README §6 names this the main risk of the §6 stage. The optimizer chooses SNP
membership to hit a precise age CDF. If a SNP's usefulness for repairing W1 is
correlated with its derived allele frequency, then matching itself biases the
SFS -- which is exactly what Phi-SFS measures, so the bias would be invisible in
every matching diagnostic and would read as signal.

Two measurements, on chromosome 10 where genotypes are available:

1. **Selection bias.** Compare the derived-frequency spectrum of the controls
   the optimizer actually published against the eligible candidate pool it drew
   them from. A shift is the bias, whatever its mechanism.

2. **The association itself.** For a sample of candidates, compute the exact-grid
   W1 change from swapping each into a published set, and correlate that utility
   with derived frequency. This is the README's diagnostic, and it says whether
   any observed shift is caused by the objective rather than by the age matching
   the objective is meant to do.

Both are run for the §5 sampler and the §6 optimizer, because the question is
whether *optimization* introduces a bias that a constrained random walk does not.
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path
import numpy as np
from normalize_tes.snp_age_store import open_snp_age_store
from normalize_tes.swap_control_sampler import analysis_points, row_cdfs, aggregate_cdf
from normalize_tes.te_age_target import wasserstein_1

BASES = "ACGT"


def chr10_frequencies(vcf: str, store, ancestral: Path, offset: float):
    """Return store rows, derived-allele frequency and callable count for chr10 SNPs."""
    pos, alt, n, ref, alt_b = [], [], [], [], []
    with gzip.open(vcf, "rt") as h:
        for raw in h:
            if raw.startswith("#"):
                continue
            f = raw.rstrip("\n").split("\t")
            if f[2] != "." or "," in f[4]:          # SNPs only, biallelic
                continue
            gts = [g.split(":")[0] for g in f[9:]]
            k = sum(1 for g in gts if g == "1")
            m = sum(1 for g in gts if g in ("0", "1"))
            if m < 20 or k == 0 or k == m:
                continue
            pos.append(int(f[1])); alt.append(k); n.append(m)
            ref.append(f[3]); alt_b.append(f[4])
    pos = np.array(pos); alt = np.array(alt, float); n = np.array(n, float)
    ref = np.array(ref); alt_b = np.array(alt_b)

    cat = np.asarray(np.load(Path(store) / "positions.npy", mmap_mode="r")) \
        if isinstance(store, (str, Path)) else np.asarray(store.positions)
    g = pos.astype(float) + offset
    ins = np.searchsorted(cat, g); ok = ins < cat.size
    ok[ok] &= cat[ins[ok]] == g[ok]
    rows = ins[ok]
    alt, n, ref, alt_b = alt[ok], n[ok], ref[ok], alt_b[ok]

    A = np.load(ancestral / "ancestral_counts.npy", mmap_mode="r")
    cnt = np.asarray(A[rows], float)
    anc = np.array([BASES[i] for i in cnt.argmax(axis=1)])
    derived = np.where(anc == ref, alt, n - alt)     # ARG polarity
    return rows, derived / n, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--vcf", required=True)
    p.add_argument("--ancestral", type=Path, required=True)
    p.add_argument("--candidate-rows", type=Path, required=True)
    p.add_argument("--bundles", nargs="+", required=True,
                   help="label=path pairs, e.g. q50=results/matches/in_gene_75draw")
    p.add_argument("--utility-bundle", required=True,
                   help="bundle to compute swap utilities against")
    p.add_argument("--sample", type=int, default=3000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()

    store = open_snp_age_store(a.store)
    offset = {c["chrom"]: c["offset"]
              for c in store.metadata["chromosomes"]}["10"]
    rows, freq, ncall = chr10_frequencies(a.vcf, a.store, a.ancestral, offset)
    print(f"chr10 biallelic SNPs with genotypes and store rows: {rows.size:,}", flush=True)

    cand = np.asarray(np.load(a.candidate_rows, mmap_mode="r"))
    in_pool = np.isin(rows, cand)
    pool_rows, pool_freq = rows[in_pool], freq[in_pool]
    print(f"of which in the control candidate pool: {pool_rows.size:,}\n", flush=True)
    freq_of = dict(zip(pool_rows.tolist(), pool_freq.tolist()))

    report = {"pool_n": int(pool_rows.size), "pool_mean_freq": float(pool_freq.mean()),
              "selection": {}, "utility": {}}
    print("--- 1. selection bias: published controls vs the pool they came from ---")
    print(f"{'bundle':>14}{'chr10 sel':>11}{'mean freq':>11}{'pool':>9}{'shift':>9}{'KS-ish':>9}")
    for spec in a.bundles:
        lab, path = spec.split("=", 1)
        sel = np.load(Path(path) / "row_indices.npy").ravel()
        sf = np.array([freq_of[int(r)] for r in np.unique(sel) if int(r) in freq_of])
        if sf.size < 100:
            print(f"{lab:>14}{sf.size:>11,}  too few chr10 controls"); continue
        # crude distribution distance: max |ECDF difference| over frequency
        grid = np.linspace(0, 1, 201)
        d = np.abs(np.searchsorted(np.sort(sf), grid)/sf.size
                   - np.searchsorted(np.sort(pool_freq), grid)/pool_freq.size).max()
        report["selection"][lab] = {"n": int(sf.size), "mean_freq": float(sf.mean()),
                                    "shift": float(sf.mean()-pool_freq.mean()),
                                    "max_ecdf_gap": float(d)}
        print(f"{lab:>14}{sf.size:>11,}{sf.mean():>11.4f}{pool_freq.mean():>9.4f}"
              f"{sf.mean()-pool_freq.mean():>+9.4f}{d:>9.3f}")

    print("\n--- 2. utility vs frequency: does repairing W1 favour a frequency class? ---")
    ub = Path(a.utility_bundle)
    ages = np.load(ub / "age_bins.npy"); pts = analysis_points(ages)
    tgt = np.load(ub / "bootstrap_target_cdfs.npy")[0]
    setrows = np.load(ub / "row_indices.npy")[0].astype(np.int64)
    base_cdf = aggregate_cdf(store, setrows, pts)
    base = wasserstein_1(base_cdf, tgt, ages)
    nset = setrows.size

    rng = np.random.default_rng(a.seed)
    take = rng.choice(pool_rows.size, size=min(a.sample, pool_rows.size), replace=False)
    cr, cf = pool_rows[take], pool_freq[take]
    cc = row_cdfs(store, cr, pts, block_rows=256, dtype=np.dtype("float32"))
    drop = row_cdfs(store, setrows[:1], pts, block_rows=1, dtype=np.dtype("float32"))[0]
    # utility of candidate c = W1 after replacing one fixed slot with c, minus base.
    util = np.empty(cr.size)
    for i in range(cr.size):
        trial = base_cdf + (cc[i].astype(np.float64) - drop.astype(np.float64)) / nset
        util[i] = wasserstein_1(trial, tgt, ages) - base
    good = np.isfinite(util)
    r = np.corrcoef(cf[good], util[good])[0, 1]
    # rank correlation, robust to the utility distribution's shape
    rk = np.corrcoef(np.argsort(np.argsort(cf[good])),
                     np.argsort(np.argsort(util[good])))[0, 1]
    print(f"candidates scored: {good.sum():,}   base W1 {base:,.1f}")
    print(f"Pearson  corr(derived freq, W1 change) = {r:+.4f}")
    print(f"Spearman corr                          = {rk:+.4f}")
    q = np.quantile(cf[good], [0, .2, .4, .6, .8, 1.0])
    print(f"\n{'derived freq band':>22}{'n':>7}{'mean W1 change':>17}{'% improving':>13}")
    for lo, hi in zip(q[:-1], q[1:]):
        m = good & (cf >= lo) & (cf <= hi)
        print(f"{lo:>10.3f}-{hi:<10.3f}{int(m.sum()):>7,}{util[m].mean():>17.3f}"
              f"{(util[m] < 0).mean():>12.1%}")
    report["utility"] = {"pearson": float(r), "spearman": float(rk),
                         "n_scored": int(good.sum()), "base_w1": float(base)}
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
