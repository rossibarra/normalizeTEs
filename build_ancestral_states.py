"""Accumulate per-site ancestral-allele counts across posterior ARG draws.

The ancestral allele is not a property of the input VCF. SINGER infers it, so it
lives in each ARG's site table and differs between draws. `phi_sfs.py` can read
polarity only from a VCF's REF column or an INFO field, so a separate table is
required; this command builds it.

For every row of an interval store, the output records how many draws called
each of A, C, G, T ancestral, together with the number of draws in which the
site appeared at all. The posterior proportion for an allele is its count
divided by that present-draw count -- conditioned on presence, so every site
has a well-defined proportion regardless of how many draws contain it. That
conditioning is what lets a downstream weighted-average SFS use every requested
site without an intersection or a fallback rule.

Downstream use is a *linear* mixture: a site with observed derived count `k`
among `n` callable samples contributes `p*h(k,n) + (1-p)*h(n-k,n)`, which is
unbiased at any present-draw count. Do not threshold the proportion into a
majority call. Present-draw counts differ systematically between TE targets and
SNP controls, and thresholding a noisier proportion applies a different
effective weight to the two groups at the same underlying value -- a
differential bias in exactly the quantity Phi-SFS measures.

One assumption is worth stating rather than hiding: presence-conditioning is
unbiased only if whether a site is represented in a draw is independent of its
polarity in that draw.

Each draw is processed independently, so `--draws` slices the input list for a
SLURM array and `--merge` combines the per-task outputs. Decompression
dominates the runtime at roughly one minute per draw; reading the ancestral
states themselves is free once the tables are in memory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from release_provenance import software_provenance
from snp_age_store import open_snp_age_store, store_schema

BASES = ("A", "C", "G", "T")
_BASE_INDEX = {base.encode(): index for index, base in enumerate(BASES)}


def _global_positions(ts: object, offsets: dict[str, int], chromosome: str | None,
                      sequence_length: float) -> np.ndarray:
    """Return store-catalog coordinates for a tree sequence's sites.

    `build_snp_interval_store` uses `tables.sites.position` verbatim as the
    catalog coordinate, so no one-based shift belongs here: the ARGs were built
    with sites already at the store's global coordinate. A per-chromosome ARG
    still needs its chromosome offset added, which is why `--chromosome` exists.
    """
    positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
    if chromosome is None:
        if abs(float(ts.sequence_length) - sequence_length) > 1.0:
            raise SystemExit(
                f"tree sequence length {ts.sequence_length:,.0f} does not match the "
                f"store's {sequence_length:,.0f}; pass --chromosome if this ARG "
                "covers a single chromosome"
            )
        return positions
    if chromosome not in offsets:
        raise SystemExit(f"chromosome {chromosome!r} is not in the store metadata")
    return positions + offsets[chromosome]


def _ancestral_indices(ts: object) -> tuple[np.ndarray, np.ndarray]:
    """Return per-site base indices and a mask of sites with a usable state.

    Sites whose ancestral state is not a single A/C/G/T character are excluded
    rather than coerced. They cannot polarize a biallelic SNP.
    """
    sites = ts.tables.sites
    offsets = np.asarray(sites.ancestral_state_offset, dtype=np.int64)
    data = np.frombuffer(bytes(sites.ancestral_state), dtype="S1")
    lengths = np.diff(offsets)
    usable = lengths == 1
    indices = np.full(offsets.size - 1, -1, dtype=np.int8)
    if np.any(usable):
        chars = data[offsets[:-1][usable]]
        mapped = np.full(chars.size, -1, dtype=np.int8)
        for base, index in _BASE_INDEX.items():
            mapped[chars == base] = index
        indices[usable] = mapped
    return indices, indices >= 0


def accumulate(store: object, tree_files: list[Path], *, chromosome: str | None,
               offsets: dict[str, int], sequence_length: float,
               counts: np.ndarray, present: np.ndarray,
               progress: bool = True) -> dict:
    """Add every draw's ancestral calls into `counts` and `present` in place."""
    import tszip

    catalog = np.asarray(store.positions)
    report: list[dict] = []
    for path in tree_files:
        ts = tszip.decompress(str(path))
        positions = _global_positions(ts, offsets, chromosome, sequence_length)
        indices, usable = _ancestral_indices(ts)

        insertion = np.searchsorted(catalog, positions)
        resolved = insertion < catalog.size
        resolved[resolved] &= catalog[insertion[resolved]] == positions[resolved]
        keep = resolved & usable
        rows = insertion[keep]
        bases = indices[keep].astype(np.int64)

        # A draw may carry one site per position, so rows are unique here and a
        # plain fancy-index add is safe; np.add.at would be an order of
        # magnitude slower for no benefit.
        if np.unique(rows).size != rows.size:
            np.add.at(counts, (rows, bases), 1)
            np.add.at(present, rows, 1)
        else:
            counts[rows, bases] += 1
            present[rows] += 1

        entry = {
            "path": str(path),
            "sites": int(ts.num_sites),
            "resolved": int(np.count_nonzero(resolved)),
            "unusable_ancestral": int(np.count_nonzero(resolved & ~usable)),
            "accumulated": int(rows.size),
        }
        report.append(entry)
        if progress:
            print(f"  {path.name}: {entry['accumulated']:,} of {entry['sites']:,} "
                  f"sites accumulated", flush=True)
        del ts
    return {"draws": report}


def _save(output: Path, counts: np.ndarray, present: np.ndarray, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, array in (("ancestral_counts", counts), ("present_draw_count", present)):
        temporary = output / f".{name}.npy.tmp.{os.getpid()}"
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, output / f"{name}.npy")
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("trees", nargs="*", type=Path,
                        help="tree-sequence draws; omit when using --merge")
    parser.add_argument("--draws", type=str,
                        help="slice of the tree list for one array task, START:STOP")
    parser.add_argument("--chromosome", type=str,
                        help="chromosome label when each ARG covers one chromosome")
    parser.add_argument("--merge", type=Path, nargs="+",
                        help="per-task output directories to sum into --output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    n_rows = int(np.asarray(store.positions).size)
    metadata_store = getattr(store, "metadata", {})

    counts = np.zeros((n_rows, 4), dtype=np.uint16)
    present = np.zeros(n_rows, dtype=np.uint16)

    if args.merge:
        detail: list[str] = []
        for part in args.merge:
            counts += np.load(part / "ancestral_counts.npy", allow_pickle=False)
            present += np.load(part / "present_draw_count.npy", allow_pickle=False)
            detail.append(str(part))
        report = {"merged": detail}
    else:
        if not args.trees:
            raise SystemExit("no tree files given and --merge not used")
        trees = sorted(args.trees)
        if args.draws:
            start, _, stop = args.draws.partition(":")
            trees = trees[int(start or 0):int(stop or len(trees))]
        if not trees:
            raise SystemExit("--draws selected no tree files")
        offsets = {c["chrom"]: int(c["offset"])
                   for c in metadata_store.get("chromosomes", [])}
        report = accumulate(
            store, trees, chromosome=args.chromosome, offsets=offsets,
            sequence_length=float(metadata_store.get("sequence_length", 0.0)),
            counts=counts, present=present,
        )

    covered = int(np.count_nonzero(present))
    with np.errstate(invalid="ignore", divide="ignore"):
        top = counts.max(axis=1)
        proportion = np.where(present > 0, top / np.maximum(present, 1), np.nan)
    contested = int(np.count_nonzero((present > 0) & (proportion < 0.9)))

    metadata = {
        "schema_version": "ancestral-state-counts-v1",
        "bases": list(BASES),
        "store": str(args.store),
        "store_schema": store_schema(store),
        "store_content_sha256": metadata_store.get("content_sha256"),
        "store_rows": n_rows,
        "rows_with_any_draw": covered,
        "rows_contested_below_0.9": contested,
        "max_present_draw_count": int(present.max()),
        "conditioning": "proportions are conditioned on presence: divide by present_draw_count",
        "intended_use": "linear mixture p*h(k,n) + (1-p)*h(n-k,n); do not threshold",
        "software": software_provenance(),
        **report,
    }
    _save(args.output, counts, present, metadata)

    print(f"store rows            {n_rows:,}")
    print(f"rows with >=1 draw    {covered:,} ({covered / n_rows:.2%})")
    print(f"max present draws     {int(present.max())}")
    print(f"rows below 0.9 agree  {contested:,}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
