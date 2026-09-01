"""Record the ancestral allele each posterior draw calls, draw by draw.

`build_ancestral_states` accumulates, per store row, *how many* draws called
each of A/C/G/T ancestral. That marginal is all Phi-SFS needs, because Phi-SFS
uses polarity as a linear mixture weight and never has to know which particular
draws voted which way.

An analysis that pairs a mutation's age with its polarity does need to know.
Within one posterior draw a biallelic site carries one mutation on one branch:
that branch fixes both the age interval and which allele is derived. Selecting
the draws in which a chosen allele is derived therefore selects a *subset* of
that row's age intervals, and the ages in that subset are not exchangeable with
the ages in the complement -- a draw that puts the mutation on a deep branch is
often the same draw that flips the polarity. Conditioning on the marginal
proportion alone silently averages over both.

This command writes the per-draw call so that pairing is possible: a
`(store rows) x (posterior draws)` table of ancestral base indices, aligned to
the interval store's row order and to its `draw_id` numbering, so that an
interval's `draw_id` indexes the matching column directly.

Encoding: 0=A, 1=C, 2=G, 3=T, and 255 for "this draw gave this row no usable
single-character A/C/G/T ancestral state", which covers both a draw that lacks
the site and a draw that annotates it with something that cannot polarize a
biallelic SNP. The two are not distinguished, deliberately: neither can orient
a site, and a downstream conditioning rule that treated them differently would
be conditioning on missingness.

Cost mirrors `build_ancestral_states`: decompression dominates at roughly one
minute per draw, so `--draws START:STOP` slices the input list for a SLURM
array and `--merge` gathers the parts. A part stores only its own columns, so
an array does not multiply the full table by the number of tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .build_ancestral_states import (
    BASES,
    _ancestral_indices,
    _global_positions,
    store_input_paths,
)
from .release_provenance import software_provenance
from .snp_age_store import open_snp_age_store, store_schema


SCHEMA_VERSION = "ancestral-state-per-draw-v1"
NO_CALL = np.uint8(255)


def accumulate(
    store: object,
    tree_files: list[Path],
    *,
    chromosome: str | None,
    offsets: dict[str, int],
    sequence_length: float,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return the per-draw base table, its store draw ids, and a report.

    Columns come out ordered by the store's own `draw_id`, not by the order the
    tree files were listed, so a part built from an arbitrarily ordered slice
    still merges into the right columns.
    """
    import tszip

    catalog = np.asarray(store.positions)
    n_rows = int(catalog.size)
    known = store_input_paths(store)
    unknown = [str(p) for p in tree_files
               if str(Path(p).resolve()) not in known]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} of {len(tree_files)} supplied tree files are not "
            f"among the store's {len(known)} source draws, so the table would "
            "carry the store's digest without having been computed from it: "
            + ", ".join(unknown[:3]) + ("..." if len(unknown) > 3 else "")
        )
    draw_ids = np.array(
        [known[str(Path(p).resolve())] for p in tree_files], dtype=np.uint16
    )
    order = np.argsort(draw_ids, kind="stable")
    draw_ids = draw_ids[order]
    ordered_files = [tree_files[int(i)] for i in order]

    table = np.full((n_rows, len(ordered_files)), NO_CALL, dtype=np.uint8)
    report: list[dict] = []
    for column, path in enumerate(ordered_files):
        ts = tszip.decompress(str(path))
        positions = _global_positions(ts, offsets, chromosome, sequence_length)
        indices, usable = _ancestral_indices(ts)

        insertion = np.searchsorted(catalog, positions)
        resolved = insertion < catalog.size
        resolved[resolved] &= catalog[insertion[resolved]] == positions[resolved]
        keep = resolved & usable
        rows = insertion[keep]
        # A draw carries at most one site per position, so rows are unique and a
        # plain fancy-index assignment is correct. Assert it rather than assume:
        # a duplicated position would silently keep only the last write.
        if np.unique(rows).size != rows.size:
            raise SystemExit(
                f"{path}: two sites resolve to the same store row; the draw's "
                "site positions are not unique in store coordinates"
            )
        table[rows, column] = indices[keep].astype(np.uint8)

        entry = {
            "path": str(Path(path).resolve()),
            "draw_id": int(draw_ids[column]),
            "sites": int(ts.num_sites),
            "resolved": int(np.count_nonzero(resolved)),
            "unusable_ancestral": int(np.count_nonzero(resolved & ~usable)),
            "recorded": int(rows.size),
        }
        report.append(entry)
        if progress:
            print(f"  {Path(path).name}: draw {entry['draw_id']}, "
                  f"{entry['recorded']:,} of {entry['sites']:,} sites recorded",
                  flush=True)
        del ts
    return table, draw_ids, {"draws": report}


def _save(output: Path, table: np.ndarray, draw_ids: np.ndarray,
          metadata: dict) -> None:
    """Publish atomically through a staging directory, never overwriting."""
    if output.exists():
        raise SystemExit(
            f"output already exists: {output}. Refusing to overwrite a published "
            "per-draw polarity table; remove it explicitly or choose another path."
        )
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for name, array in (("ancestral_base", table), ("draw_ids", draw_ids)):
            with (staging / f"{name}.npy").open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
        (staging / "metadata.json").write_text(
            json.dumps({**metadata, "complete": True}, indent=2, sort_keys=True)
            + "\n", encoding="utf-8",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def open_draw_polarity(path: str | Path, store: object) -> tuple[np.ndarray, dict]:
    """Memory-map a merged table after checking it describes this store.

    Returned rows are indexed by store row and columns by store `draw_id`, so
    callers may index a column with an interval's `draw_id` without a lookup.
    """
    directory = Path(path)
    try:
        metadata = json.loads(
            (directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"invalid or missing per-draw polarity metadata: {directory}"
        ) from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"{directory}: not a per-draw polarity table")
    if not metadata.get("complete"):
        raise SystemExit(f"{directory}: table is incomplete")
    store_metadata = getattr(store, "metadata", {}) or {}
    if metadata.get("store_content_sha256") != store_metadata.get("content_sha256"):
        raise SystemExit(
            f"{directory}: built from a different interval store than the one "
            "given; its polarity columns would not align with these ages"
        )
    n_rows = int(np.asarray(store.positions).size)
    n_draws = int(store_metadata.get("n_posterior_draws", 0))
    table = np.load(directory / "ancestral_base.npy", mmap_mode="r",
                    allow_pickle=False)
    if table.shape != (n_rows, n_draws):
        raise SystemExit(
            f"{directory}: ancestral_base.npy has shape {table.shape}, expected "
            f"{(n_rows, n_draws)}"
        )
    draw_ids = np.load(directory / "draw_ids.npy", allow_pickle=False)
    if not np.array_equal(draw_ids, np.arange(n_draws, dtype=draw_ids.dtype)):
        raise SystemExit(
            f"{directory}: columns are not the store's draws 0..{n_draws - 1} in "
            "order, so an interval's draw_id would index the wrong column"
        )
    return table, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store whose rows and draws the table is aligned to")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination directory for the per-draw table")
    parser.add_argument("trees", nargs="*", type=Path,
                        help="tree-sequence draws; omit when using --merge")
    parser.add_argument("--draws", type=str,
                        help="slice of the sorted tree list for one array task, START:STOP")
    parser.add_argument("--chromosome", type=str,
                        help="chromosome label when each ARG covers one chromosome")
    parser.add_argument("--merge", type=Path, nargs="+",
                        help="per-task part directories to gather into --output")
    parser.add_argument(
        "--expect-draws", type=int,
        help="number of draws the merged table must contain; without it a merge "
             "cannot tell a complete gather from a silently partial one",
    )
    return parser.parse_args(argv)


def _merge(args: argparse.Namespace, store: object, n_rows: int,
           n_draws: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Gather part tables into one full-width table, checking identity.

    The failure this guards against is not a crash but a plausible table: parts
    from two stores, one part listed twice, or a previously merged table passed
    as a part. Each produces a complete-looking output whose columns no longer
    mean what the metadata says.
    """
    store_metadata = getattr(store, "metadata", {}) or {}
    table = np.full((n_rows, n_draws), NO_CALL, dtype=np.uint8)
    filled = np.zeros(n_draws, dtype=bool)
    detail: list[dict] = []
    seen_paths: set[Path] = set()
    seen_draws: set[str] = set()
    for part in args.merge:
        resolved_part = part.resolve()
        if resolved_part in seen_paths:
            raise SystemExit(f"--merge lists {part} more than once")
        seen_paths.add(resolved_part)
        try:
            part_meta = json.loads(
                (part / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"{part}: unreadable part metadata") from error
        if part_meta.get("schema_version") != SCHEMA_VERSION:
            raise SystemExit(f"{part}: not a per-draw polarity table")
        if not part_meta.get("complete"):
            raise SystemExit(f"{part}: table is incomplete")
        if part_meta.get("store_content_sha256") != store_metadata.get("content_sha256"):
            raise SystemExit(
                f"{part}: built from a different interval store than --store")
        if list(part_meta.get("bases", [])) != list(BASES):
            raise SystemExit(f"{part}: base ordering differs")
        if "merged" in part_meta:
            raise SystemExit(
                f"{part}: already a merged table; merging it again cannot add "
                "columns and hides which parts were actually gathered")
        draws = [str(Path(entry["path"]).resolve())
                 for entry in part_meta.get("draws", [])]
        if not draws:
            raise SystemExit(f"{part}: records no contributing draws")
        if len(set(draws)) != len(draws):
            raise SystemExit(f"{part}: records the same draw more than once")
        overlap = seen_draws.intersection(draws)
        if overlap:
            raise SystemExit(
                f"{part}: draws already gathered from an earlier part: "
                + ", ".join(sorted(overlap)[:3])
            )
        seen_draws.update(draws)
        part_table = np.load(part / "ancestral_base.npy", mmap_mode="r",
                             allow_pickle=False)
        part_ids = np.load(part / "draw_ids.npy", allow_pickle=False)
        if part_table.shape != (n_rows, part_ids.size):
            raise SystemExit(
                f"{part}: ancestral_base.npy has shape {part_table.shape}, "
                f"expected {(n_rows, part_ids.size)}")
        if part_table.dtype != np.uint8:
            raise SystemExit(
                f"{part}: ancestral_base.npy has dtype {part_table.dtype}, "
                "expected uint8")
        if part_ids.size != len(draws):
            raise SystemExit(
                f"{part}: {part_ids.size} columns but {len(draws)} draws recorded")
        columns = np.asarray(part_ids, dtype=np.int64)
        if columns.min(initial=0) < 0 or columns.max(initial=-1) >= n_draws:
            raise SystemExit(f"{part}: draw ids fall outside the store's range")
        if np.any(filled[columns]):
            raise SystemExit(f"{part}: some of its draw ids are already gathered")
        table[:, columns] = part_table
        filled[columns] = True
        detail.append({"path": str(resolved_part), "draws": len(draws)})

    known = set(store_input_paths(store))
    if seen_draws != known:
        missing = sorted(known - seen_draws)
        extra = sorted(seen_draws - known)
        pieces = []
        if missing:
            pieces.append(f"{len(missing)} of the store's draws are absent "
                          f"(e.g. {Path(missing[0]).name})")
        if extra:
            pieces.append(f"{len(extra)} gathered draws are not the store's "
                          f"(e.g. {Path(extra[0]).name})")
        raise SystemExit(
            "merged parts do not cover exactly the store's source draws: "
            + "; ".join(pieces))
    if not filled.all():
        raise SystemExit(
            f"{int((~filled).sum())} draw columns were never filled; the gather "
            "is partial")
    if args.expect_draws is not None and len(seen_draws) != args.expect_draws:
        raise SystemExit(
            f"merge produced {len(seen_draws)} distinct draws, expected "
            f"{args.expect_draws}; a part is missing or duplicated")
    return (table, np.arange(n_draws, dtype=np.uint16),
            {"merged": detail, "merged_draws": len(seen_draws),
             "expected_draws": args.expect_draws})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    store_metadata = getattr(store, "metadata", {}) or {}
    n_rows = int(np.asarray(store.positions).size)
    n_draws = int(store_metadata.get("n_posterior_draws", 0))
    if not store_metadata.get("content_sha256"):
        raise SystemExit(
            "the interval store records no content_sha256; a per-draw polarity "
            "table cannot be authenticated against it")
    if n_draws <= 0:
        raise SystemExit("the interval store records no posterior draws")
    if args.merge and args.expect_draws is None:
        raise SystemExit(
            "--merge requires --expect-draws so a partial gather cannot be "
            "published as complete")
    if args.expect_draws is not None and args.expect_draws <= 0:
        raise SystemExit("--expect-draws must be positive")
    if not args.merge and args.expect_draws is not None:
        raise SystemExit("--expect-draws is valid only with --merge")
    if args.merge and args.trees:
        raise SystemExit("tree arguments cannot be combined with --merge")
    if args.merge and args.draws:
        raise SystemExit("--draws cannot be combined with --merge")

    if args.merge:
        table, draw_ids, report = _merge(args, store, n_rows, n_draws)
    else:
        if not args.trees:
            raise SystemExit("no tree files given and --merge not used")
        trees = sorted(args.trees)
        if args.draws:
            start, _, stop = args.draws.partition(":")
            trees = trees[int(start or 0):int(stop or len(trees))]
        if not trees:
            raise SystemExit("--draws selected no tree files")
        resolved = [path.resolve() for path in trees]
        if len(set(resolved)) != len(resolved):
            raise SystemExit(
                "tree arguments select the same draw more than once (possibly "
                "through relative, absolute, or symlink aliases)")
        offsets = {c["chrom"]: int(c["offset"])
                   for c in store_metadata.get("chromosomes", [])}
        table, draw_ids, report = accumulate(
            store, resolved, chromosome=args.chromosome, offsets=offsets,
            sequence_length=float(store_metadata.get("sequence_length", 0.0)),
        )

    called = table != NO_CALL
    per_row = called.sum(axis=1, dtype=np.int64)
    covered = int(np.count_nonzero(per_row))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "bases": list(BASES),
        "no_call_code": int(NO_CALL),
        "encoding": "row-major uint8; column j is the store's draw_id j; "
                    "0=A 1=C 2=G 3=T 255=no usable ancestral call",
        "store": str(Path(args.store).resolve()),
        "store_schema": store_schema(store),
        "store_content_sha256": store_metadata.get("content_sha256"),
        "store_rows": n_rows,
        "store_posterior_draws": n_draws,
        "columns": int(table.shape[1]),
        "rows_with_any_call": covered,
        "calls_recorded": int(per_row.sum()),
        "intended_use": "pair a draw's age interval with that draw's polarity; "
                        "for a marginal proportion use build_ancestral_states",
        "software": software_provenance(),
        **report,
    }
    _save(args.output, table, draw_ids, metadata)

    print(f"store rows            {n_rows:,}")
    print(f"columns written       {table.shape[1]} of {n_draws} draws")
    print(f"rows with >=1 call    {covered:,} ({covered / max(n_rows, 1):.2%})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
