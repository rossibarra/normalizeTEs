"""Draws must be authenticated by what they contain, not by where they sit.

The path check these replace failed in both directions: a byte-identical draw
that had been moved into a subdirectory was rejected, while any file with the
right name at the right path was accepted. Each test below pins one half of
that.
"""

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import tskit
import tszip

import normalize_tes.build_snp_interval_store as builder
import normalize_tes.record_draw_identities as recorder
from normalize_tes.draw_identity import (
    DrawIndex,
    IDENTITY_VERSION,
    draw_identity,
    identity_key,
)
from normalize_tes.snp_interval_dataset import compute_interval_store_content_sha256


def _write_ts(path, *, positions=(7, 30, 99), provenance="run-a"):
    tables = tskit.TableCollection(sequence_length=100)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = {
        "chrom_offsets": [{"chrom": "chr1", "offset": 0, "length": 100}]
    }
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=20)
    tables.edges.add_row(0, 100, parent=1, child=0)
    for position in positions:
        site = tables.sites.add_row(position, "A")
        tables.mutations.add_row(site=site, node=0, derived_state="T")
    tables.provenances.add_row(json.dumps({"seed": provenance}))
    tables.sort()
    tables.tree_sequence().dump(path)
    return path


def _draw(tmp_path, name, **kwargs):
    """Write one draw as a TSZ archive, the form the pipeline actually uses."""
    ordinary = tmp_path / f"{name}.trees"
    compressed = tmp_path / f"{name}.tsz"
    _write_ts(ordinary, **kwargs)
    tszip.compress(tskit.load(ordinary), compressed)
    ordinary.unlink()
    return compressed


def _store(inputs, n_draws=None):
    return SimpleNamespace(metadata={
        "content_sha256": "store-digest",
        "n_posterior_draws": n_draws if n_draws is not None else len(inputs),
        "inputs": inputs,
    })


def _recorded(path, draw_id=0):
    return {"draw_id": draw_id, "path": str(path), **draw_identity(path)}


# --- the identity itself ---------------------------------------------------

def test_compressed_and_uncompressed_forms_have_one_identity(tmp_path):
    """The digest is defined over decoded provenance, not over a byte layout.

    A TSZ archive is read through zarr and a `.trees` file through tskit, and
    the two readers see entirely different bytes. If they disagreed, the access
    method rather than the draw would decide whether a file authenticated.
    """
    ordinary = _write_ts(tmp_path / "draw.trees")
    compressed = tmp_path / "draw.tsz"
    tszip.compress(tskit.load(ordinary), compressed)
    assert draw_identity(ordinary) == draw_identity(compressed)


def test_distinct_draws_have_distinct_identities(tmp_path):
    first = _draw(tmp_path, "a", provenance="run-a")
    second = _draw(tmp_path, "b", provenance="run-b")
    assert identity_key(draw_identity(first)) != identity_key(draw_identity(second))


def test_identity_is_a_content_property_not_a_file_property(tmp_path):
    original = _draw(tmp_path, "a")
    moved = tmp_path / "elsewhere" / "renamed.tsz"
    moved.parent.mkdir()
    shutil.copy2(original, moved)
    assert draw_identity(original) == draw_identity(moved)


def test_a_partially_recorded_identity_yields_no_key(tmp_path):
    entry = _recorded(_draw(tmp_path, "a"))
    entry.pop("n_mutations")
    assert identity_key(entry) is None
    assert identity_key({**entry, "n_mutations": 3,
                         "identity_version": "something-else"}) is None


# --- matching supplied files against a store -------------------------------

def test_a_moved_draw_still_authenticates(tmp_path):
    """The failure that motivated this: the draws moved, the store did not."""
    original = _draw(tmp_path, "a")
    index = DrawIndex.from_store(_store([_recorded(original)]))
    moved = tmp_path / "tsz_files" / "a.tsz"
    moved.parent.mkdir()
    shutil.move(original, moved)
    assert index.authenticated
    assert index.assign_files([moved])[0] == [0]


def test_a_stranger_at_the_recorded_path_is_refused(tmp_path):
    """The other half: the right name in the right place is not identity."""
    original = _draw(tmp_path, "a", provenance="run-a")
    index = DrawIndex.from_store(_store([_recorded(original)]))
    original.unlink()
    _draw(tmp_path, "b", provenance="run-b").rename(original)
    with pytest.raises(SystemExit, match="content matches no recorded draw"):
        index.assign_files([original])


def test_one_draw_supplied_twice_under_two_names_is_refused(tmp_path):
    original = _draw(tmp_path, "a")
    alias = tmp_path / "copy.tsz"
    shutil.copy2(original, alias)
    index = DrawIndex.from_store(_store([_recorded(original)]))
    with pytest.raises(SystemExit, match="more than once"):
        index.assign_files([original, alias])


def test_an_unreadable_file_is_reported_as_such(tmp_path):
    index = DrawIndex.from_store(_store([_recorded(_draw(tmp_path, "a"))]))
    with pytest.raises(SystemExit, match="does not exist"):
        index.assign_files([tmp_path / "absent.tsz"])


# --- stores that predate identity recording --------------------------------

def test_a_legacy_store_falls_back_to_paths_and_says_so(tmp_path):
    draw = _draw(tmp_path, "a")
    index = DrawIndex.from_store(_store([{"draw_id": 0, "path": str(draw)}]))
    assert not index.authenticated
    assert index.assign_files([draw])[0] == [0]
    with pytest.raises(SystemExit, match="record_draw_identities"):
        index.assign_files([tmp_path / "stranger.tsz"])


def test_indistinguishable_recorded_draws_do_not_authenticate(tmp_path):
    """Identity that cannot separate two draws must not be used to match them.

    Matching on a key shared by two store entries would assign one of them
    arbitrarily, so the index degrades to paths rather than guessing.
    """
    draw = _draw(tmp_path, "a")
    index = DrawIndex.from_store(_store([_recorded(draw, 0), _recorded(draw, 1)]))
    assert not index.authenticated


# --- parts recorded by an array task ---------------------------------------

def test_recorded_draws_map_to_store_ids_after_a_move(tmp_path):
    first, second = _draw(tmp_path, "a"), _draw(tmp_path, "b", provenance="b")
    index = DrawIndex.from_store(_store([_recorded(first, 0), _recorded(second, 1)]))
    relocated = [{**_recorded(second, 1), "path": "/gone/b.tsz"}]
    assert index.assign_recorded(relocated, "part") == [1]


def test_a_recorded_draw_the_store_does_not_know_is_refused(tmp_path):
    index = DrawIndex.from_store(_store([_recorded(_draw(tmp_path, "a"))]))
    stranger = _recorded(_draw(tmp_path, "b", provenance="b"), 0)
    with pytest.raises(SystemExit, match="not one of the store's"):
        index.assign_recorded([stranger], "part")


# --- what the store records ------------------------------------------------

def test_a_built_store_records_an_identity_for_every_draw(tmp_path):
    draws = [_draw(tmp_path, "a", provenance="a"),
             _draw(tmp_path, "b", provenance="b")]
    output = tmp_path / "store"
    builder.build_interval_store(draws, output, min_usable_fraction=0.0)
    metadata = json.loads((output / "metadata.json").read_text())
    for entry, path in zip(metadata["inputs"], draws):
        assert entry["identity_version"] == IDENTITY_VERSION
        assert entry["provenance_sha256"] == draw_identity(path)["provenance_sha256"]
    assert DrawIndex(metadata["inputs"]).authenticated


def test_identity_fields_do_not_enter_the_store_content_digest(tmp_path):
    """The upgrade path depends on this: recording identities is not a rebuild.

    `record_draw_identities` rewrites `inputs` in a published store. That is
    only legitimate because `inputs` is not hashed, so every artifact already
    stamped with the store's digest stays valid.
    """
    store = tmp_path / "store"
    store.mkdir()
    np.save(store / "positions.npy", np.arange(4, dtype=np.float64))
    metadata = {
        "schema_version": "x", "n_posterior_draws": 1, "chromosomes": [],
        "arrays": {"positions": {"dtype": "float64"}},
        "inputs": [{"draw_id": 0, "path": "/old/a.tsz"}],
    }
    before = compute_interval_store_content_sha256(store, metadata)
    identified = {**metadata, "inputs": [
        {"draw_id": 0, "path": "/new/a.tsz", **draw_identity(_draw(tmp_path, "a"))}
    ]}
    assert compute_interval_store_content_sha256(store, identified) == before


# --- upgrading a store that was built before identities --------------------

def test_upgrading_a_legacy_store_rehomes_its_draws(tmp_path):
    draw = _draw(tmp_path, "a")
    moved = tmp_path / "tsz_files" / "a.tsz"
    moved.parent.mkdir()
    shutil.move(draw, moved)
    metadata = {"inputs": [{"draw_id": 0, "path": str(draw)}]}
    updated = recorder.plan_identities(metadata, [moved])
    assert updated[0]["path"] == str(moved)
    assert DrawIndex(updated).authenticated
    assert DrawIndex(updated).assign_files([moved])[0] == [0]


def test_upgrading_refuses_a_file_that_is_a_different_draw(tmp_path):
    """An already-identified store is verified, not overwritten."""
    original = _draw(tmp_path, "a", provenance="a")
    other = _draw(tmp_path, "b", provenance="b")
    metadata = {"inputs": [_recorded(original)]}
    renamed = tmp_path / "moved" / "a.tsz"
    renamed.parent.mkdir()
    shutil.move(other, renamed)
    with pytest.raises(SystemExit, match="different draw"):
        recorder.plan_identities(metadata, [renamed])


def test_upgrading_requires_the_whole_draw_set(tmp_path):
    first, second = _draw(tmp_path, "a"), _draw(tmp_path, "b", provenance="b")
    metadata = {"inputs": [{"draw_id": 0, "path": str(first)},
                           {"draw_id": 1, "path": str(second)}]}
    with pytest.raises(SystemExit, match="but 1 files were supplied"):
        recorder.plan_identities(metadata, [first])
    stranger = _draw(tmp_path, "c", provenance="c")
    with pytest.raises(SystemExit, match="not among the supplied files"):
        recorder.plan_identities(metadata, [first, stranger])


def test_upgrading_writes_atomically_and_keeps_a_backup(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    original = {"inputs": [{"draw_id": 0, "path": "/old/a.tsz"}], "keep": "me"}
    (store / "metadata.json").write_text(json.dumps(original))
    recorder.write_metadata(store, {**original, "inputs": [{"draw_id": 0}]})
    assert json.loads((store / "metadata.json").read_text())["keep"] == "me"
    assert json.loads((store / "metadata.json.bak").read_text()) == original
