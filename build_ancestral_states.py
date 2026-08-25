"""Accumulate per-site ancestral-allele counts across posterior ARG draws.

The ancestral allele is not a property of the input VCF. SINGER infers it, so it
lives in each ARG's site table and differs between draws. `phi_sfs.py` therefore
takes polarity for control SNPs from the table this command builds, rather than
from any VCF annotation.

For every row of an interval store, the output records how many draws called
each of A, C, G, T ancestral, together with `present_draw_count`: the number of
draws that gave the site a *usable* ancestral call, meaning a single uppercase
A/C/G/T. That is narrower than raw presence -- a draw containing the site but
annotating it with a multi-character or non-ACGT state is not counted -- so do
not read the array as a missingness statistic.

The posterior proportion for an allele is its count divided by that usable-call
count, which gives every site a well-defined proportion regardless of how many
draws contain it, and is what lets a downstream weighted-average SFS use every
requested site without an intersection or a fallback rule.

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
import shutil
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


def store_input_paths(store: object) -> dict[str, int]:
    """Map each of the store's recorded source draws to its draw id.

    The table is stamped with the store's content digest, which asserts that it
    was computed from that store's posterior draws. Nothing else establishes it:
    an arbitrary set of tree files of the right cardinality accumulates happily
    and publishes under the store's identity. Checking the paths against the
    store's own `metadata["inputs"]` is what makes the stamp mean something.
    """
    metadata = getattr(store, "metadata", {}) or {}
    inputs = metadata.get("inputs")
    if not inputs:
        raise SystemExit(
            "store metadata records no 'inputs', so the supplied draws cannot be "
            "authenticated against it. Rebuild the store with "
            "build_snp_interval_store.py."
        )
    return {str(Path(entry["path"]).resolve()): int(entry["draw_id"])
            for entry in inputs}


def accumulate(store: object, tree_files: list[Path], *, chromosome: str | None,
               offsets: dict[str, int], sequence_length: float,
               counts: np.ndarray, present: np.ndarray,
               progress: bool = True) -> dict:
    """Add every draw's ancestral calls into `counts` and `present` in place."""
    import tszip

    catalog = np.asarray(store.positions)
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
            "path": str(path.resolve()),
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
    """Publish the table atomically, refusing to overwrite an existing one.

    Writing the two arrays in place and then the metadata leaves a hybrid
    directory if the run is interrupted between them -- arrays from one run
    beside a schema document that still looks valid. Staging a sibling directory
    and renaming it makes the published table either wholly old or wholly new,
    and `complete` is written only once both arrays are on disk.
    """
    if output.exists():
        raise SystemExit(
            f"output already exists: {output}. Refusing to overwrite a published "
            "ancestral table; remove it explicitly or choose another path."
        )
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for name, array in (("ancestral_counts", counts),
                            ("present_draw_count", present)):
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store whose rows the table is aligned to")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination directory for the ancestral-state table")
    parser.add_argument("trees", nargs="*", type=Path,
                        help="tree-sequence draws; omit when using --merge")
    parser.add_argument("--draws", type=str,
                        help="slice of the tree list for one array task, START:STOP")
    parser.add_argument("--chromosome", type=str,
                        help="chromosome label when each ARG covers one chromosome")
    parser.add_argument("--merge", type=Path, nargs="+",
                        help="per-task output directories to sum into --output")
    parser.add_argument(
        "--expect-draws", type=int,
        help="number of draws the merged table must contain; without it a "
             "merge cannot tell a complete gather from a silently partial one",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    n_rows = int(np.asarray(store.positions).size)
    metadata_store = getattr(store, "metadata", {})
    if not metadata_store.get("content_sha256"):
        raise SystemExit(
            "the interval store records no content_sha256; an ancestral table "
            "cannot be authenticated against targets and matched bundles built "
            "from this store"
        )
    if args.merge and args.expect_draws is None:
        raise SystemExit(
            "--merge requires --expect-draws so a partial gather cannot be "
            "published as complete"
        )
    if args.expect_draws is not None and args.expect_draws <= 0:
        raise SystemExit("--expect-draws must be positive")
    if not args.merge and args.expect_draws is not None:
        raise SystemExit("--expect-draws is valid only with --merge")
    if args.merge and args.trees:
        raise SystemExit("tree arguments cannot be combined with --merge")
    if args.merge and args.draws:
        raise SystemExit("--draws cannot be combined with --merge")

    counts = np.zeros((n_rows, 4), dtype=np.uint16)
    present = np.zeros(n_rows, dtype=np.uint16)

    if args.merge:
        # Summing parts blindly is how a merge silently produces a plausible but
        # wrong table: parts from different stores, a part counted twice, or a
        # previously merged table passed as though it were a part. Each of those
        # yields a complete-looking output with inflated or mixed counts, so the
        # parts have to agree on identity and contribute a disjoint draw set.
        detail: list[dict] = []
        seen_paths: set[Path] = set()
        seen_draws: set[str] = set()
        total = np.zeros((n_rows, 4), dtype=np.uint64)
        total_present = np.zeros(n_rows, dtype=np.uint64)
        for part in args.merge:
            resolved_part = part.resolve()
            if resolved_part in seen_paths:
                raise SystemExit(f"--merge lists {part} more than once")
            seen_paths.add(resolved_part)
            part_meta = json.loads(
                (part / "metadata.json").read_text(encoding="utf-8"))
            if part_meta.get("schema_version") != "ancestral-state-counts-v1":
                raise SystemExit(f"{part}: not an ancestral-state table")
            if not part_meta.get("complete"):
                raise SystemExit(f"{part}: table is incomplete")
            if part_meta.get("store_content_sha256") != metadata_store.get(
                    "content_sha256"):
                raise SystemExit(
                    f"{part}: built from a different interval store than --store")
            if list(part_meta.get("bases", [])) != list(BASES):
                raise SystemExit(f"{part}: base ordering differs")
            if "merged" in part_meta:
                raise SystemExit(
                    f"{part}: already a merged table; merging it again would "
                    "double-count every draw it contains")
            # Identify draws by resolved path rather than the recorded string:
            # two aliases of one file (a symlink, a relative and an absolute
            # spelling) would otherwise look like two draws. Site count is not
            # part of identity: inconsistent metadata must not make the same
            # physical draw appear distinct.
            draws = [
                str(Path(entry["path"]).resolve())
                for entry in part_meta.get("draws", [])
            ]
            if not draws:
                raise SystemExit(f"{part}: records no contributing draws")
            if len(set(draws)) != len(draws):
                raise SystemExit(f"{part}: records the same draw more than once")
            overlap = seen_draws.intersection(draws)
            if overlap:
                raise SystemExit(
                    f"{part}: draws already counted by an earlier part: "
                    + ", ".join(sorted(overlap)[:3])
                )
            seen_draws.update(draws)
            # Accumulate in uint64: uint16 parts wrap at 65,536, which a large or
            # accidentally repeated merge could reach without any error.
            part_counts = np.load(part / "ancestral_counts.npy", allow_pickle=False)
            part_present = np.load(part / "present_draw_count.npy", allow_pickle=False)
            # Check shape and dtype before adding. A (n_rows, 1) counts array or
            # a (1,) presence array broadcasts silently against the accumulator
            # and corrupts every row rather than raising.
            for name, array, shape in (
                ("ancestral_counts", part_counts, (n_rows, 4)),
                ("present_draw_count", part_present, (n_rows,)),
            ):
                if array.shape != shape:
                    raise SystemExit(
                        f"{part}: {name}.npy has shape {array.shape}, expected {shape}")
                if array.dtype.kind != "u":
                    raise SystemExit(
                        f"{part}: {name}.npy has dtype {array.dtype}, expected unsigned")
            if np.any(part_counts.sum(axis=1) > part_present):
                raise SystemExit(
                    f"{part}: some rows record more ancestral calls than draws")
            total += part_counts.astype(np.uint64)
            total_present += part_present.astype(np.uint64)
            detail.append({"path": str(resolved_part), "draws": len(draws)})
        if total.max() > np.iinfo(np.uint16).max or \
                total_present.max() > np.iinfo(np.uint16).max:
            raise SystemExit(
                "merged counts exceed the uint16 output range; the table format "
                "needs widening before this many draws can be combined"
            )
        counts = total.astype(np.uint16)
        present = total_present.astype(np.uint16)
        # Cardinality is not identity: the expected number of distinct but
        # wrong draws passes the count check and publishes under the store's
        # digest. Require the gathered set to be exactly the store's draws.
        known = set(store_input_paths(store))
        if seen_draws != known:
            missing = sorted(known - seen_draws)
            extra = sorted(seen_draws - known)
            detail_parts = []
            if missing:
                detail_parts.append(
                    f"{len(missing)} of the store's draws are absent "
                    + f"(e.g. {Path(missing[0]).name})")
            if extra:
                detail_parts.append(
                    f"{len(extra)} gathered draws are not the store's "
                    + f"(e.g. {Path(extra[0]).name})")
            raise SystemExit(
                "merged parts do not cover exactly the store's source draws: "
                + "; ".join(detail_parts)
            )
        if args.expect_draws is not None and len(seen_draws) != args.expect_draws:
            raise SystemExit(
                f"merge produced {len(seen_draws)} distinct draws, expected "
                f"{args.expect_draws}; a part is missing or duplicated"
            )
        report = {"merged": detail, "merged_draws": len(seen_draws),
                  "expected_draws": args.expect_draws}
    else:
        if not args.trees:
            raise SystemExit("no tree files given and --merge not used")
        trees = sorted(args.trees)
        if args.draws:
            start, _, stop = args.draws.partition(":")
            trees = trees[int(start or 0):int(stop or len(trees))]
        if not trees:
            raise SystemExit("--draws selected no tree files")
        resolved_trees = [path.resolve() for path in trees]
        if len(set(resolved_trees)) != len(resolved_trees):
            raise SystemExit(
                "tree arguments select the same draw more than once (possibly "
                "through relative, absolute, or symlink aliases)"
            )
        trees = resolved_trees
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
        "store": str(Path(args.store).resolve()),
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
