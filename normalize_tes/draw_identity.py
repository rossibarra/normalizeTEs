"""Identify a posterior draw by its content, not by where its file sits.

An ancestral-state or per-draw polarity table is stamped with the interval
store's `content_sha256`, which asserts that it was computed from that store's
posterior draws. Nothing in the table itself establishes that: an arbitrary set
of tree files of the right cardinality accumulates happily and publishes under
the store's identity, so the supplied draws have to be authenticated against
the store's own `metadata["inputs"]` before any accumulation begins.

Matching on the recorded *path* is the wrong test, and it is wrong in both
directions. Moving the draws into a subdirectory fails a set of files that are
byte-identical to the ones the store was built from, while any file with the
right name at the right path passes. Path equality authenticates a location; it
is the content that matters.

The store's own `content_sha256` cannot do this job either. It hashes the
store's arrays plus the metadata keys that affect their interpretation, so it
identifies the store already in hand and says nothing about the tree files just
handed to a consumer. Hashing the draws themselves would identify them, but 75
draws is 77 GB in the pilot: a full extra read of the entire posterior at build
time and again at every consumer.

So identity is derived from what a draw already exposes cheaply:

  * the provenance chain -- every SINGER run and each subsequent tskit or tszip
    operation appends a record carrying its parameters and a timestamp, a few
    kilobytes in total that differ between draws by construction; and
  * the table cardinalities and the sequence length, which are zarr array
    shapes and an attribute, and cost nothing beyond opening the archive.

Reading that takes roughly a third of a second per draw and decompresses
nothing, so the check stays where it belongs: up front, before hours of
accumulation rather than after them.

This is a strong quasi-identity, not a proof. A file whose provenance chain and
table shapes match the recorded draw but whose payload has been altered would
pass where a hash of the file would not. It is chosen because it is cheap
enough to run in every consumer on every run, and a check too expensive to run
is weaker than one that is merely incomplete.

Stores built before this module recorded no identity at all. Those fall back to
the historical path comparison rather than failing, because there is nothing in
an old store to authenticate against; `normalize_tes.record_draw_identities`
upgrades one in place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

IDENTITY_VERSION = "draw-identity-v1"

# Cardinalities and length are structural: they come from array shapes, so they
# are free, and they are payload-derived in a way the provenance chain is not.
_STRUCTURE_FIELDS = (
    "n_sites",
    "n_mutations",
    "n_nodes",
    "n_edges",
    "n_individuals",
    "sequence_length",
)
IDENTITY_FIELDS = (*_STRUCTURE_FIELDS, "provenance_sha256")


def _provenance_digest(records: list[tuple[str, str]]) -> str:
    """Hash a provenance chain through a canonical, length-prefixed encoding.

    The same digest has to come out whether the chain was read from a TSZ
    archive's raw zarr arrays or from a loaded tree sequence's provenance
    table, so the encoding is defined over the decoded strings rather than over
    whichever byte layout the reader happened to see.
    """
    digest = hashlib.sha256()
    digest.update(b"normalizeTE-draw-provenance-v1\0")
    digest.update(len(records).to_bytes(8, "little"))
    for timestamp, record in records:
        for text in (timestamp, record):
            data = text.encode("utf-8")
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
    return digest.hexdigest()


def _decode_ragged(data: object, offsets: object) -> list[str]:
    """Split a tskit ragged column into its decoded strings."""
    raw = np.asarray(data, dtype=np.int8).tobytes()
    bounds = np.asarray(offsets, dtype=np.int64)
    return [raw[bounds[i]:bounds[i + 1]].decode("utf-8")
            for i in range(bounds.size - 1)]


def _identity_from_tsz(path: Path) -> dict:
    """Read identity fields from a TSZ archive without decompressing it."""
    from tszip.compression import load_zarr

    with load_zarr(path) as root:
        records = list(zip(
            _decode_ragged(root["provenances/timestamp"][:],
                           root["provenances/timestamp_offset"][:]),
            _decode_ragged(root["provenances/record"][:],
                           root["provenances/record_offset"][:]),
        ))
        return {
            "n_sites": int(root["sites/position"].shape[0]),
            "n_mutations": int(root["mutations/site"].shape[0]),
            "n_nodes": int(root["nodes/time"].shape[0]),
            "n_edges": int(root["edges/left"].shape[0]),
            "n_individuals": int(root["individuals/flags"].shape[0]),
            "sequence_length": float(root.attrs["sequence_length"]),
            "provenance_sha256": _provenance_digest(records),
        }


def _identity_from_tree_sequence(path: Path) -> dict:
    """Fallback for uncompressed inputs, which have no zarr layout to read."""
    import tszip

    ts = tszip.load(str(path))
    records = [(p.timestamp, p.record) for p in ts.provenances()]
    return {
        "n_sites": int(ts.num_sites),
        "n_mutations": int(ts.num_mutations),
        "n_nodes": int(ts.num_nodes),
        "n_edges": int(ts.num_edges),
        "n_individuals": int(ts.num_individuals),
        "sequence_length": float(ts.sequence_length),
        "provenance_sha256": _provenance_digest(records),
    }


def draw_identity(path: str | Path) -> dict:
    """Return the content identity of one posterior draw."""
    path = Path(path)
    fields = (_identity_from_tsz(path) if path.suffix.lower() == ".tsz"
              else _identity_from_tree_sequence(path))
    return {"identity_version": IDENTITY_VERSION, **fields}


def identity_key(entry: object) -> tuple | None:
    """Return a comparable key for a recorded draw, or None if it has none.

    A partially filled entry yields None rather than a short key: comparing on
    whichever fields happen to be present would quietly weaken the check, and
    "cannot be authenticated by content" is a different state from "does not
    match" that callers need to distinguish.
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("identity_version") != IDENTITY_VERSION:
        return None
    if any(entry.get(field) is None for field in IDENTITY_FIELDS):
        return None
    return tuple(entry[field] for field in IDENTITY_FIELDS)


class DrawIndex:
    """Maps supplied files and recorded draws onto the store's draw ids.

    Two modes, decided by the store rather than by the caller. When every
    recorded input carries a content identity the index matches on that, and a
    relocated draw authenticates exactly as it did before it moved. When the
    store predates identity recording -- or when two of its draws are
    indistinguishable, which would make identity matching ambiguous -- it falls
    back to the historical resolved-path comparison and says so, so that no
    caller silently believes a content check happened when it did not.
    """

    def __init__(self, entries: list[dict]):
        self.entries = [dict(entry) for entry in entries]
        self.ids = [int(entry["draw_id"]) for entry in self.entries]
        self.by_path = {
            str(Path(entry["path"]).resolve()): int(entry["draw_id"])
            for entry in self.entries
        }
        keys = [identity_key(entry) for entry in self.entries]
        self.by_key: dict[tuple, int] = {}
        if all(key is not None for key in keys) and len(set(keys)) == len(keys):
            self.by_key = {key: int(entry["draw_id"])
                           for key, entry in zip(keys, self.entries)}
        self.by_id = {int(entry["draw_id"]): entry for entry in self.entries}

    @classmethod
    def from_store(cls, store: object) -> "DrawIndex":
        metadata = getattr(store, "metadata", {}) or {}
        inputs = metadata.get("inputs")
        if not inputs:
            raise SystemExit(
                "store metadata records no 'inputs', so the supplied draws "
                "cannot be authenticated against it. Rebuild the store with "
                "normalize_tes.build_snp_interval_store."
            )
        return cls(list(inputs))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def authenticated(self) -> bool:
        """True when matching is on content rather than on file location."""
        return bool(self.by_key)

    @property
    def all_ids(self) -> set[int]:
        return set(self.ids)

    def recorded_path(self, draw_id: int) -> str:
        return str(self.by_id[int(draw_id)]["path"])

    def _fallback_note(self) -> str:
        if self.authenticated:
            return ""
        return (
            " This store records no per-draw content identity, so draws are "
            "matched by file path and a relocated draw cannot be recognised; "
            "run normalize_tes.record_draw_identities to upgrade it."
        )

    def id_for_file(self, path: Path) -> tuple[int | None, dict | None, str]:
        """Return `(draw_id, identity, reason)` for one supplied file."""
        resolved = str(Path(path).resolve())
        if not self.authenticated:
            found = self.by_path.get(resolved)
            return found, None, "" if found is not None else "path not recorded"
        try:
            identity = draw_identity(path)
        except FileNotFoundError:
            return None, None, "file does not exist"
        except Exception as error:  # unreadable or not a tree sequence
            return None, None, f"unreadable as a tree sequence ({error})"
        found = self.by_key.get(identity_key(identity))
        if found is None:
            return None, identity, "content matches no recorded draw"
        return found, identity, ""

    def assign_files(self, paths: list[Path], *,
                     noun: str = "tree files") -> tuple[list[int], list[dict]]:
        """Return the store draw id of every supplied file, or exit.

        Duplicates are rejected on the resolved *draw*, not on the path: under
        content matching two different filenames can name one physical draw,
        and counting it twice would double its weight exactly as passing the
        same path twice would.
        """
        assigned: list[int] = []
        identities: list[dict] = []
        unknown: list[str] = []
        for path in paths:
            draw_id, identity, reason = self.id_for_file(path)
            if draw_id is None:
                unknown.append(f"{path} ({reason})" if reason else str(path))
                continue
            assigned.append(draw_id)
            identities.append(identity or {})
        if unknown:
            raise SystemExit(
                f"{len(unknown)} of {len(paths)} supplied {noun} are not among "
                f"the store's {len(self)} source draws, so the table would "
                "carry the store's digest without having been computed from "
                "it: " + ", ".join(unknown[:3])
                + ("..." if len(unknown) > 3 else "")
                + self._fallback_note()
            )
        seen: dict[int, str] = {}
        for draw_id, path in zip(assigned, paths):
            if draw_id in seen:
                raise SystemExit(
                    f"the same posterior draw was supplied more than once "
                    f"(draw {draw_id}): {seen[draw_id]} and {path}"
                )
            seen[draw_id] = str(path)
        return assigned, identities

    def id_for_recorded(self, entry: dict) -> int | None:
        """Return the store draw id for a draw recorded in a part's metadata."""
        key = identity_key(entry)
        if self.authenticated and key is not None:
            return self.by_key.get(key)
        path = entry.get("path")
        if path is None:
            return None
        return self.by_path.get(str(Path(path).resolve()))

    def assign_recorded(self, entries: list[dict], label: str) -> list[int]:
        """Map a part's recorded draws onto store draw ids, or exit."""
        assigned: list[int] = []
        for entry in entries:
            draw_id = self.id_for_recorded(entry)
            if draw_id is None:
                raise SystemExit(
                    f"{label}: records a draw that is not one of the store's "
                    f"{len(self)} source draws: {entry.get('path', entry)}"
                    + self._fallback_note()
                )
            assigned.append(draw_id)
        return assigned
