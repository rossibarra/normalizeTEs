"""Record, per TE site, which posterior draws polarized it in agreement with biology.

A TE insertion is the derived state. When a draw calls the insertion allele
ancestral it has placed the mutation on the other side of the local tree, so the
age it records for that site is the age of a different branch. Polarity and age
are one inference, not two, and a draw that gets the polarity wrong has also got
the age wrong.

`normalize_tes.build_ancestral_states` records how many draws chose each base but not which
ones, so it cannot answer "drop the draws that flipped this site". This command
produces that mask: a boolean array of shape `(n_te_sites, n_draws)`, true where
the draw called the absence allele ancestral. `normalize_tes.te_age_target --drop-flipped-draws`
consumes it and builds each site's age CDF from the agreeing draws alone.

Scope is deliberately narrow. This applies only to TE target sites, where biology
fixes the answer. Control SNPs have no such truth, and their polarity uncertainty
is carried by the Phi-SFS mixture instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .release_provenance import software_provenance
from .snp_age_store import open_snp_age_store


def store_draw_columns(store: object, tree_files: list[Path]) -> tuple[int, list[int]]:
    """Map each tree file to the store's own draw id for it.

    The store numbers its posterior draws in `metadata["inputs"]`, and the
    interval records carry that same `draw_id`. Numbering the mask's columns by
    the order the files happen to be listed here would agree with the store only
    by luck, and a mask whose columns are permuted relative to `draw_id` masks
    each site with another draw's polarity while looking entirely healthy. So
    the mapping is taken from the store rather than assumed.
    """
    metadata = getattr(store, "metadata", {}) or {}
    inputs = metadata.get("inputs")
    n_draws = metadata.get("n_posterior_draws")
    if not inputs or n_draws is None:
        raise SystemExit(
            "store metadata has no 'inputs'/'n_posterior_draws' draw mapping, so "
            "tree files cannot be matched to draw ids. Rebuild the store with "
            "normalize_tes.build_snp_interval_store."
        )
    by_path = {str(Path(entry["path"]).resolve()): int(entry["draw_id"])
               for entry in inputs}
    columns: list[int] = []
    for path in tree_files:
        resolved = str(Path(path).resolve())
        if resolved not in by_path:
            raise SystemExit(
                f"{path} is not one of the store's {len(by_path)} source trees, "
                "so it has no draw id. Pass the same tsz files the store was "
                "built from."
            )
        columns.append(by_path[resolved])
    if len(set(columns)) != len(columns):
        raise SystemExit("the same tree file was passed more than once")
    return int(n_draws), columns


def polarity_mask(store: object, te_rows: np.ndarray, tree_files: list[Path],
                  absence_allele: str, progress: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Return `(n_te, n_draws)` agreement mask plus a per-draw report.

    Columns are indexed by the store's `draw_id`, not by argument order, so the
    result lines up with the interval records. Draws whose tree file was not
    passed stay all-false in `seen`: absent, not flipped.
    """
    import tszip

    n_draws, columns = store_draw_columns(store, tree_files)
    catalog = np.asarray(store.positions)
    wanted = catalog[te_rows]
    mask = np.zeros((te_rows.size, n_draws), dtype=bool)
    seen = np.zeros((te_rows.size, n_draws), dtype=bool)
    report: list[dict] = []
    for column, path in zip(columns, tree_files):
        ts = tszip.decompress(str(path))
        positions = np.asarray(ts.tables.sites.position, dtype=np.float64)
        sites = ts.tables.sites
        offsets = np.asarray(sites.ancestral_state_offset, dtype=np.int64)
        data = np.frombuffer(bytes(sites.ancestral_state), dtype="S1")

        insertion = np.searchsorted(positions, wanted)
        present = insertion < positions.size
        present[present] &= positions[insertion[present]] == wanted[present]
        index = insertion[present]
        single = (offsets[index + 1] - offsets[index]) == 1
        usable = np.flatnonzero(present)[single]
        chars = data[offsets[index[single]]]

        seen[usable, column] = True
        mask[usable, column] = chars == absence_allele.encode()
        report.append({
            "draw_id": int(column),
            "path": str(Path(path).resolve()),
            "sites_present": int(present.sum()),
            "usable_ancestral": int(single.sum()),
            "agreeing": int(mask[:, column].sum()),
        })
        if progress:
            print(f"  {Path(path).name}: {int(mask[:, column].sum()):,} of "
                  f"{int(single.sum()):,} usable sites agree", flush=True)
        del ts
    return mask, report, seen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store the target rows index into; supplies "
                             "the draw_id for each tree file")
    parser.add_argument("--target", type=Path, required=True,
                        help="target directory whose te_row_indices.npy defines the sites")
    parser.add_argument("trees", nargs="+", type=Path,
                        help="every tsz the store was built from. A draw that is "
                             "not passed is uncovered, and te_age_target refuses a "
                             "mask that does not cover the whole store")
    parser.add_argument("--absence-allele", default="A",
                        help="allele encoding TE absence, which biology makes "
                             "ancestral; every TE record in this dataset is A/G "
                             "with A as absence (default: A)")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination directory for the mask")
    return parser.parse_args(argv)


def load_target_rows(target: Path, store: object) -> np.ndarray:
    """Read the target's TE rows, having first proved they index this store.

    The mask is stamped with the supplied store's digest and is later applied
    positionally to a target's sites. Both of those make the source target's
    identity part of the mask's meaning, so it is checked here rather than
    assumed: a target built against another store indexes different SNPs, and
    the mask would mislabel every one of them while looking healthy.
    """
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    store_meta = getattr(store, "metadata", {}) or {}
    for key, actual, label in (
        ("source_store_content_sha256", store_meta.get("content_sha256"), "content"),
        ("source_catalog_sha256", store_meta.get("catalog_sha256"), "catalog"),
    ):
        recorded = metadata.get(key)
        if not recorded or not actual:
            raise SystemExit(
                f"{target} or the store is missing the {label} digest, so the "
                "target's rows cannot be authenticated against this store."
            )
        if recorded != actual:
            raise SystemExit(
                f"{target} was built against a different store ({label} digest "
                f"{recorded[:12]} != {actual[:12]}); its row indices name "
                "different SNPs here."
            )

    rows = np.load(target / "te_row_indices.npy", allow_pickle=False)
    if rows.ndim != 1 or rows.size == 0:
        raise SystemExit(f"{target}: te_row_indices.npy must be a non-empty 1-D array")
    if not np.issubdtype(rows.dtype, np.integer):
        raise SystemExit(
            f"{target}: te_row_indices.npy has dtype {rows.dtype}; a float array "
            "would be silently truncated by the int64 cast"
        )
    rows = rows.astype(np.int64, copy=False)
    n_rows = int(np.asarray(store.positions).size)
    if rows.min() < 0 or rows.max() >= n_rows:
        raise SystemExit(
            f"{target}: te_row_indices.npy holds rows outside the store's "
            f"{n_rows:,} rows"
        )
    if np.unique(rows).size != rows.size:
        raise SystemExit(f"{target}: te_row_indices.npy contains duplicate rows")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if args.absence_allele not in ("A", "C", "G", "T"):
        raise SystemExit("--absence-allele must be one of A, C, G, T")
    store = open_snp_age_store(args.store)
    te_rows = load_target_rows(args.target, store)
    trees = sorted(args.trees)
    print(f"{te_rows.size:,} TE sites x {len(trees)} draws", flush=True)

    mask, report, seen = polarity_mask(store, te_rows, trees, args.absence_allele)
    per_site = mask.sum(axis=1)
    usable = seen.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree_fraction = np.where(usable > 0, per_site / np.maximum(usable, 1), np.nan)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(f".{args.output.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        np.save(staging / "agrees_with_biology.npy", mask)
        np.save(staging / "draw_present.npy", seen)
        np.save(staging / "te_row_indices.npy", te_rows)
        (staging / "metadata.json").write_text(json.dumps({
            "schema_version": "te-polarity-mask-v1",
            "store": str(Path(args.store).resolve()),
            "store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
            "target": str(Path(args.target).resolve()),
            "absence_allele": args.absence_allele,
            "n_te_sites": int(te_rows.size),
            "n_draws": int(mask.shape[1]),
            "covered_draw_ids": sorted(int(e["draw_id"]) for e in report),
            "trees_supplied": len(trees),
            "sites_with_no_agreeing_draw": int((per_site == 0).sum()),
            "median_agreeing_fraction": float(np.nanmedian(agree_fraction)),
            "draws": report,
            "software": software_provenance(),
            "complete": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, args.output)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"\nmedian fraction of draws agreeing with biology: "
          f"{np.nanmedian(agree_fraction):.4f}")
    print(f"sites where no draw agrees: {int((per_site == 0).sum()):,} "
          f"(their age CDF would be empty; te_age_target keeps all draws there)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
