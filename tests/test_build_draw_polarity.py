"""Per-draw ancestral-state table: accumulation, column alignment, and gathering."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tskit
import tszip

import normalize_tes.build_draw_polarity as builder


def _store(tmp_path, digest="store-digest", n_draws=4, n_rows=3):
    return SimpleNamespace(
        positions=np.array([5.0, 9.0, 20.0], dtype=np.float64)[:n_rows],
        metadata={
            "content_sha256": digest,
            "n_posterior_draws": n_draws,
            "chromosomes": [{"chrom": "chr1", "offset": 0, "length": 100}],
            "sequence_length": 100.0,
            "inputs": [{"path": str(tmp_path / f"draw{d}.tsz"), "draw_id": d}
                       for d in range(n_draws)],
        },
    )


def _tree_sequence(states):
    """A minimal tree sequence carrying only the site states under test."""
    tables = tskit.TableCollection(sequence_length=100.0)
    for position, state in states:
        tables.sites.add_row(position=position, ancestral_state=state)
    tables.build_index()
    return tables.tree_sequence()


def _fake_tszip(monkeypatch, mapping):
    monkeypatch.setattr(
        tszip, "decompress",
        lambda path: _tree_sequence(mapping[str(path)]), raising=True)


def test_accumulate_orders_columns_by_store_draw_id(tmp_path, monkeypatch):
    store = _store(tmp_path)
    # Hand the builder the draws out of order; columns must still come back
    # ordered by the store's draw_id so an interval's draw_id indexes them.
    files = [tmp_path / "draw2.tsz", tmp_path / "draw0.tsz"]
    _fake_tszip(monkeypatch, {
        str(tmp_path / "draw2.tsz"): [(5.0, "G"), (9.0, "T")],
        str(tmp_path / "draw0.tsz"): [(5.0, "A"), (20.0, "C")],
    })
    table, draw_ids, report = builder.accumulate(
        store, files, chromosome=None, offsets={"chr1": 0},
        sequence_length=100.0, progress=False)
    assert draw_ids.tolist() == [0, 2]
    np.testing.assert_array_equal(table, np.array([
        [0, 2],            # row 0: A in draw 0, G in draw 2
        [255, 3],          # row 1: absent from draw 0, T in draw 2
        [1, 255],          # row 2: C in draw 0, absent from draw 2
    ], dtype=np.uint8))
    assert [entry["draw_id"] for entry in report["draws"]] == [0, 2]


def test_unusable_ancestral_state_is_a_no_call(tmp_path, monkeypatch):
    store = _store(tmp_path)
    files = [tmp_path / "draw0.tsz"]
    _fake_tszip(monkeypatch, {
        str(tmp_path / "draw0.tsz"): [(5.0, "AT"), (9.0, "N"), (20.0, "A")],
    })
    table, _, report = builder.accumulate(
        store, files, chromosome=None, offsets={"chr1": 0},
        sequence_length=100.0, progress=False)
    assert table[:, 0].tolist() == [255, 255, 0]
    assert report["draws"][0]["unusable_ancestral"] == 2


def test_tree_files_outside_the_store_are_refused(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _fake_tszip(monkeypatch, {})
    with pytest.raises(SystemExit, match="not\n?.*among the store"):
        builder.accumulate(store, [tmp_path / "stranger.tsz"], chromosome=None,
                           offsets={"chr1": 0}, sequence_length=100.0,
                           progress=False)


def _part(path, draw_ids, digest="store-digest", n_rows=3, complete=True,
          extra=None):
    path.mkdir(parents=True)
    np.save(path / "ancestral_base.npy",
            np.zeros((n_rows, len(draw_ids)), dtype=np.uint8))
    np.save(path / "draw_ids.npy", np.asarray(draw_ids, dtype=np.uint16))
    metadata = {
        "schema_version": builder.SCHEMA_VERSION,
        "complete": complete,
        "bases": ["A", "C", "G", "T"],
        "store_content_sha256": digest,
        "draws": [{"path": str(path.parent.parent / f"draw{d}.tsz"),
                   "draw_id": int(d)} for d in draw_ids],
    }
    metadata.update(extra or {})
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _merge(tmp_path, monkeypatch, parts, expect=4, output="merged"):
    store = _store(tmp_path)
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: store)
    return builder.main([
        "--store", str(tmp_path / "store"), "--output", str(tmp_path / output),
        "--merge", *[str(part) for part in parts], "--expect-draws", str(expect),
    ])


def test_merge_gathers_disjoint_parts_into_store_draw_order(tmp_path, monkeypatch):
    parts = [_part(tmp_path / "parts" / "b", [2, 3]),
             _part(tmp_path / "parts" / "a", [0, 1])]
    assert _merge(tmp_path, monkeypatch, parts) == 0
    merged = tmp_path / "merged"
    assert np.load(merged / "draw_ids.npy").tolist() == [0, 1, 2, 3]
    assert np.load(merged / "ancestral_base.npy").shape == (3, 4)
    metadata = json.loads((merged / "metadata.json").read_text())
    assert metadata["merged_draws"] == 4
    assert metadata["complete"] is True


def test_merge_refuses_a_partial_gather(tmp_path, monkeypatch):
    parts = [_part(tmp_path / "parts" / "a", [0, 1])]
    with pytest.raises(SystemExit, match="do not cover exactly"):
        _merge(tmp_path, monkeypatch, parts, expect=2)


def test_merge_refuses_parts_from_another_store(tmp_path, monkeypatch):
    parts = [_part(tmp_path / "parts" / "a", [0, 1]),
             _part(tmp_path / "parts" / "b", [2, 3], digest="other")]
    with pytest.raises(SystemExit, match="different interval store"):
        _merge(tmp_path, monkeypatch, parts)


def test_merge_refuses_a_repeated_part(tmp_path, monkeypatch):
    part = _part(tmp_path / "parts" / "a", [0, 1])
    with pytest.raises(SystemExit, match="more than once"):
        _merge(tmp_path, monkeypatch, [part, part])


def test_merge_refuses_an_already_merged_table(tmp_path, monkeypatch):
    parts = [_part(tmp_path / "parts" / "a", [0, 1]),
             _part(tmp_path / "parts" / "b", [2, 3], extra={"merged": []})]
    with pytest.raises(SystemExit, match="already a merged table"):
        _merge(tmp_path, monkeypatch, parts)


def test_merge_requires_expect_draws(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: store)
    part = _part(tmp_path / "parts" / "a", [0, 1])
    with pytest.raises(SystemExit, match="requires --expect-draws"):
        builder.main(["--store", str(tmp_path / "store"),
                      "--output", str(tmp_path / "out"), "--merge", str(part)])


def test_open_checks_store_identity_and_column_order(tmp_path, monkeypatch):
    store = _store(tmp_path)
    parts = [_part(tmp_path / "parts" / "a", [0, 1]),
             _part(tmp_path / "parts" / "b", [2, 3])]
    assert _merge(tmp_path, monkeypatch, parts) == 0
    table, metadata = builder.open_draw_polarity(tmp_path / "merged", store)
    assert table.shape == (3, 4)
    assert metadata["store_content_sha256"] == "store-digest"

    other = _store(tmp_path, digest="different")
    with pytest.raises(SystemExit, match="different interval store"):
        builder.open_draw_polarity(tmp_path / "merged", other)

    np.save(tmp_path / "merged" / "draw_ids.npy",
            np.array([1, 0, 2, 3], dtype=np.uint16))
    with pytest.raises(SystemExit, match="not the store's draws"):
        builder.open_draw_polarity(tmp_path / "merged", store)
