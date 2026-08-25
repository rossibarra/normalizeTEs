"""Build a control-candidate row array that excludes every TE variant.

The interval store is built from a combined SNP+TE dataset, so `--all-eligible`
would leave TE variants other than the matching target in the control universe.
Matching TEs against controls that are themselves TEs weakens the contrast the
analysis rests on. This command resolves one or more TE position lists to store
rows and writes the eligible universe with all of them removed, in the
`--candidate-rows` form that `sample_age_matched_controls.py` and
`bootstrap_target_matcher.py` accept.

Pass the same array to both commands. The seed library and the matcher must
draw from one universe, or the matcher's initialization sets would contain rows
it is not allowed to propose.

Unresolved exclusions are the failure mode worth watching. A TE position that
is genuinely absent from the store is harmless: it cannot be selected as a
control either. A TE position that is present under a *different* global
coordinate is not harmless, because it stays in the candidate pool while
appearing to have been excluded. The two look identical here, so a resolution
rate below `--min-resolved-fraction` stops the command rather than warning. The
usual cause is a chromosome-offset convention mismatch between the store and
the position list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from release_provenance import software_provenance
from snp_age_dataset import load_native_position_list
from snp_age_store import open_snp_age_store, store_schema
from snp_position_resolution import resolve_native_position_requests


def _sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(values)
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _resolve_lists(store: object, paths: list[Path], minimum_fraction: float,
                   kind: str) -> tuple[np.ndarray, list[dict]]:
    """Resolve position lists to eligible store rows, refusing a poor match rate."""
    rows: list[np.ndarray] = []
    reports: list[dict] = []
    for path in paths:
        chromosomes, vcf_positions = load_native_position_list(path)
        resolution = resolve_native_position_requests(
            store, chromosomes, vcf_positions,
            policy="drop", label=f"{kind} positions ({path.name})",
        )
        requested = resolution.requested_count
        resolved = resolution.resolved_count
        fraction = resolved / requested if requested else 0.0
        reports.append({
            "path": str(path),
            "requested": requested,
            "resolved": resolved,
            "unresolved": resolution.unresolved_count,
            "resolved_fraction": fraction,
            "eligible": int(resolution.eligible_count),
            "ineligible": int(np.count_nonzero(resolution.ineligible_mask)),
        })
        if fraction < minimum_fraction:
            raise SystemExit(
                f"{path}: only {resolved:,} of {requested:,} positions "
                f"({fraction:.1%}) resolved to store rows, below the "
                f"{minimum_fraction:.1%} minimum. An unresolved exclusion stays "
                "in the candidate pool, and an unresolved inclusion silently "
                "shrinks it. Check that the store and this list use the same "
                "chromosome-offset convention before overriding."
            )
        # Ineligible and unresolved rows are dropped: neither can be a
        # candidate, so dropping them from either list changes nothing.
        rows.append(resolution.row_indices[resolution.eligible_mask])
    return np.unique(np.concatenate(rows)).astype(np.int64, copy=False), reports


def build(store: object, exclusion_rows: np.ndarray,
          inclusion_rows: np.ndarray | None = None) -> np.ndarray:
    """Return the sorted candidate universe with every exclusion row removed.

    Without `inclusion_rows` the universe is every eligible store row. With it,
    the universe is restricted to those rows first, which is how a control pool
    is held to the same quality filters as the matching target.
    """
    eligible = np.asarray(store.eligible)
    if inclusion_rows is None:
        universe = np.flatnonzero(eligible).astype(np.int64, copy=False)
    else:
        # Intersect with eligibility rather than trusting the caller. The
        # resolver upstream already drops ineligible rows, but `build` is a
        # public entry point and a candidate universe containing an ineligible
        # row is rejected later by `eligible_candidates`, surfacing as a
        # confusing downstream error rather than here.
        universe = np.unique(np.asarray(inclusion_rows, dtype=np.int64))
        universe = universe[eligible[universe]]
    return np.setdiff1d(universe, exclusion_rows, assume_unique=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store defining the row universe")
    parser.add_argument(
        "--exclude-positions", type=Path, required=True, nargs="+",
        help="whitespace-delimited chromosome and 1-based VCF position files",
    )
    parser.add_argument(
        "--include-positions", type=Path, nargs="+",
        help="restrict the universe to these positions before excluding; "
             "omit to start from every eligible store row",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="destination .npy file for the candidate rows")
    parser.add_argument("--report", type=Path,
                        help="optional JSON summary; defaults to OUTPUT.json")
    parser.add_argument("--min-resolved-fraction", type=float, default=0.95,
                        help="stop if fewer than this fraction of listed positions "
                             "resolve to store rows. An unresolved exclusion stays "
                             "in the candidate pool while appearing to have been "
                             "removed, so a low rate is treated as an error")
    return parser.parse_args(argv)


def _publish(
    output: Path, report_path: Path, candidates: np.ndarray, report: dict,
) -> None:
    """Stage both files and expose the NPY last as the artifact commit marker.

    Two independent files cannot be renamed atomically as a pair. Publishing the
    report first and the array last preserves the useful invariant: whenever the
    requested NPY exists after a normal return or caught write failure, its
    provenance report exists too. An exception removes a report installed by
    this attempt before re-raising.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    staged_rows = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    staged_report = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    report_installed = False
    try:
        with staged_rows.open("wb") as handle:
            np.save(handle, candidates, allow_pickle=False)
        staged_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staged_report, report_path)
        report_installed = True
        os.replace(staged_rows, output)
    except BaseException:
        for leftover in (staged_rows, staged_report):
            leftover.unlink(missing_ok=True)
        if report_installed and not output.exists():
            report_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_snp_age_store(args.store)
    exclusion_rows, exclusion_reports = _resolve_lists(
        store, list(args.exclude_positions), args.min_resolved_fraction, "exclusion"
    )
    inclusion_rows: np.ndarray | None = None
    inclusion_reports: list[dict] = []
    if args.include_positions:
        inclusion_rows, inclusion_reports = _resolve_lists(
            store, list(args.include_positions), args.min_resolved_fraction, "inclusion"
        )
    candidates = build(store, exclusion_rows, inclusion_rows)
    if candidates.size == 0:
        raise SystemExit("no eligible candidate rows remain after exclusion")

    total = int(np.asarray(store.positions).size)
    eligible = int(np.count_nonzero(np.asarray(store.eligible)))
    report = {
        "store": str(args.store),
        "store_schema": store_schema(store),
        "store_content_sha256": getattr(store, "metadata", {}).get("content_sha256"),
        "store_catalog_sha256": getattr(store, "metadata", {}).get("catalog_sha256"),
        "store_rows": total,
        "eligible_rows": eligible,
        "universe_rows": int(eligible if inclusion_rows is None else inclusion_rows.size),
        "universe": "all_eligible" if inclusion_rows is None else "include_lists",
        "excluded_rows": int(exclusion_rows.size),
        "candidate_rows": int(candidates.size),
        "candidate_rows_sha256": _sha256_array(candidates),
        "exclusion_lists": exclusion_reports,
        "inclusion_lists": inclusion_reports,
        "min_resolved_fraction": args.min_resolved_fraction,
        "software": software_provenance(),
    }

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".json")
    # Resolve before comparing: two different spellings of one path must not slip
    # through and have the JSON overwrite the array, which the code would then
    # cheerfully report as two successful writes.
    resolved = {
        "output": args.output.resolve(),
        "report": report_path.resolve(),
    }
    for label, path in list(resolved.items()):
        if path.exists():
            raise SystemExit(
                f"{label} already exists: {path}. Refusing to overwrite a "
                "published candidate universe; remove it or choose another path."
            )
    if resolved["output"] == resolved["report"]:
        raise SystemExit("--output and --report must be different paths")
    inputs = {p.resolve() for p in list(args.exclude_positions)
              + list(args.include_positions or [])}
    collisions = inputs.intersection(resolved.values())
    if collisions:
        raise SystemExit(
            "output paths collide with input position lists: "
            + ", ".join(str(p) for p in sorted(collisions))
        )
    _publish(args.output, report_path, candidates, report)

    print(f"store rows      {total:,}")
    print(f"eligible        {eligible:,}")
    print(f"universe        {report['universe_rows']:,} ({report['universe']})")
    print(f"excluded (TE)   {exclusion_rows.size:,}")
    print(f"candidates      {candidates.size:,}")
    for kind, entries in (("include", inclusion_reports), ("exclude", exclusion_reports)):
        for entry in entries:
            print(f"  {kind} {Path(entry['path']).name}: {entry['resolved']:,}/"
                  f"{entry['requested']:,} resolved ({entry['resolved_fraction']:.2%}), "
                  f"{entry['eligible']:,} eligible")
    print(f"wrote {args.output}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
