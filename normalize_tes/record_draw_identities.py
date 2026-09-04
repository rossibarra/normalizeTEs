"""Record content identities for the draws of an already-built interval store.

A store built before `normalize_tes.draw_identity` existed records only a path
per posterior draw, so its consumers can authenticate the draws they are handed
only by where those files sit. Move the draws -- into a subdirectory, onto
another mount -- and every consumer refuses a set of files that is byte for
byte the ones the store was built from.

Rebuilding an 18 GB store to fix a directory rename is absurd, and the fix does
not need it. `inputs` is not among the metadata keys that enter
`content_sha256` (see `snp_interval_dataset._CONTENT_IDENTITY_METADATA_KEYS`),
so recording an identity for each draw, and correcting the path it now lives
at, leaves the store's published identity untouched: every artifact already
stamped with that digest stays valid.

What this command cannot do is *verify* an old store. There is nothing in it to
check the draws against -- that is precisely what was missing -- so the
correspondence has to be asserted by the operator and is checked only as far as
it can be: the same number of draws, matched one-to-one by file name. Read it
as "these are the files that store was built from", in the same register as the
path list it replaces. On a store that already carries identities the command
is a verification instead: the content must match, and only the path moves.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .draw_identity import IDENTITY_FIELDS, draw_identity, identity_key


def _load_metadata(store: Path) -> dict:
    try:
        return json.loads((store / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{store}: unreadable store metadata") from error


def plan_identities(metadata: dict, tree_files: list[Path]) -> list[dict]:
    """Return rewritten `inputs` entries, or exit explaining why it cannot.

    Matching is by file name because that is the only correspondence an
    unauthenticated store offers. It is required to be a bijection: a store
    whose draws cannot be paired one-to-one with the supplied files by name is
    one where the operator's assertion cannot be given a definite meaning, and
    guessing at the pairing is how the wrong draw acquires a draw id.
    """
    inputs = metadata.get("inputs")
    if not inputs:
        raise SystemExit("store metadata records no 'inputs' to update")
    if len(tree_files) != len(inputs):
        raise SystemExit(
            f"the store records {len(inputs)} source draws but {len(tree_files)} "
            "files were supplied; pass exactly the draws it was built from"
        )
    supplied: dict[str, Path] = {}
    for path in tree_files:
        name = Path(path).name
        if name in supplied:
            raise SystemExit(f"two supplied files share the name {name}")
        supplied[name] = Path(path)
    updated: list[dict] = []
    for entry in inputs:
        name = Path(entry["path"]).name
        if name not in supplied:
            raise SystemExit(
                f"the store's draw {entry['draw_id']} is {name}, which is not "
                "among the supplied files"
            )
        path = supplied[name].resolve()
        identity = draw_identity(path)
        recorded = identity_key(entry)
        if recorded is not None and recorded != identity_key(identity):
            differing = [field for field in IDENTITY_FIELDS
                         if entry.get(field) != identity[field]]
            raise SystemExit(
                f"{path} does not have the content recorded for the store's "
                f"draw {entry['draw_id']}: {', '.join(differing)} differ. This "
                "is a different draw, not the same draw in a new location."
            )
        updated.append({**entry, "path": str(path), **identity})
    return updated


def write_metadata(store: Path, metadata: dict, *, backup: bool = True) -> None:
    """Replace the store's metadata atomically, keeping a copy of the old one.

    Written through a temporary file and renamed so an interrupted run cannot
    leave a store with truncated metadata, which no consumer could open.
    """
    target = store / "metadata.json"
    if backup:
        shutil.copy2(target, target.with_suffix(".json.bak"))
    temporary = target.with_name(f".metadata.json.{os.getpid()}")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True,
                        help="interval store directory to update in place")
    parser.add_argument("trees", nargs="+", type=Path,
                        help="the store's source draws at their current paths")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument(
        "--verify-digest", action="store_true",
        help="recompute content_sha256 afterwards to confirm it is unchanged; "
             "this rereads every array in the store",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = Path(args.store)
    metadata = _load_metadata(store)
    published = metadata.get("content_sha256")
    updated = plan_identities(metadata, list(args.trees))
    moved = [entry for entry, old in zip(updated, metadata["inputs"])
             if entry["path"] != str(old["path"])]
    already = sum(1 for entry in metadata["inputs"] if identity_key(entry))
    print(f"draws            {len(updated)}")
    print(f"already identified {already}")
    print(f"paths corrected  {len(moved)}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_metadata(store, {**metadata, "inputs": updated})
    if args.verify_digest:
        from .snp_interval_dataset import compute_interval_store_content_sha256
        recomputed = compute_interval_store_content_sha256(store)
        if recomputed != published:
            raise SystemExit(
                f"content_sha256 changed from {published} to {recomputed}; the "
                "previous metadata is at metadata.json.bak"
            )
        print(f"content_sha256   unchanged ({recomputed})")
    print(f"updated {store / 'metadata.json'} (previous copy at metadata.json.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
