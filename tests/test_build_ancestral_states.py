import json
from types import SimpleNamespace

import numpy as np
import pytest

import build_ancestral_states as builder


def _store(digest="store-digest"):
    return SimpleNamespace(
        positions=np.arange(3, dtype=np.float64),
        metadata={
            "content_sha256": digest,
            "chromosomes": [],
            "sequence_length": 0,
        },
    )


def _part(path, draws):
    path.mkdir()
    np.save(path / "ancestral_counts.npy", np.zeros((3, 4), dtype=np.uint16))
    np.save(path / "present_draw_count.npy", np.zeros(3, dtype=np.uint16))
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": "ancestral-state-counts-v1",
        "complete": True,
        "store_content_sha256": "store-digest",
        "bases": ["A", "C", "G", "T"],
        "draws": draws,
    }))


def test_merge_requires_expected_draw_count(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: _store())
    part = tmp_path / "part"
    _part(part, [{"path": str(tmp_path / "a.tsz"), "sites": 4}])
    with pytest.raises(SystemExit, match="requires --expect-draws"):
        builder.main([
            "--store", str(tmp_path / "store"), "--output", str(tmp_path / "out"),
            "--merge", str(part),
        ])


def test_merge_rejects_duplicate_draw_inside_one_part(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: _store())
    draw = {"path": str(tmp_path / "a.tsz"), "sites": 4}
    part = tmp_path / "part"
    _part(part, [draw, draw])
    with pytest.raises(SystemExit, match="same draw more than once"):
        builder.main([
            "--store", str(tmp_path / "store"), "--output", str(tmp_path / "out"),
            "--merge", str(part), "--expect-draws", "1",
        ])


def test_merge_detects_same_path_even_if_site_metadata_differs(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: _store())
    draw_path = str(tmp_path / "a.tsz")
    first = tmp_path / "first"
    second = tmp_path / "second"
    _part(first, [{"path": draw_path, "sites": 4}])
    _part(second, [{"path": draw_path, "sites": 999}])
    with pytest.raises(SystemExit, match="already counted"):
        builder.main([
            "--store", str(tmp_path / "store"), "--output", str(tmp_path / "out"),
            "--merge", str(first), str(second), "--expect-draws", "2",
        ])


def test_merge_rejects_tree_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: _store())
    part = tmp_path / "part"
    _part(part, [{"path": str(tmp_path / "a.tsz"), "sites": 4}])
    with pytest.raises(SystemExit, match="cannot be combined"):
        builder.main([
            "--store", str(tmp_path / "store"), "--output", str(tmp_path / "out"),
            str(tmp_path / "b.tsz"), "--merge", str(part), "--expect-draws", "1",
        ])


def test_builder_rejects_store_without_content_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "open_snp_age_store", lambda _: _store(None))
    with pytest.raises(SystemExit, match="no content_sha256"):
        builder.main([
            "--store", str(tmp_path / "store"), "--output", str(tmp_path / "out"),
            str(tmp_path / "draw.tsz"),
        ])
