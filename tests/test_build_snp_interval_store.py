import json
from pathlib import Path

import numpy as np
import pytest
import tskit

import build_snp_interval_store as builder
from build_snp_interval_store import (
    audit_mutation_parent_lookup,
    build_interval_store,
    lookup_mutation_parents,
    pack_status_row,
    unpack_status_row,
)
from snp_interval_dataset import SNPAgeIntervalDataset, validate_interval_store


def _write_ts(path: Path, sites: dict[float, list[int]], *, second_interval=False):
    """Write a small ARG; node 3 is a root and node 0 can change parent."""
    tables = tskit.TableCollection(sequence_length=100)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = {
        "chrom_offsets": [{"chrom": "chr1", "offset": 0, "length": 100}]
    }
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 0
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 1
    tables.nodes.add_row(time=20)  # 2
    tables.nodes.add_row(time=100)  # 3
    tables.nodes.add_row(time=30)  # 4
    if second_interval:
        tables.edges.add_row(0, 50, parent=2, child=0)
        tables.edges.add_row(50, 100, parent=4, child=0)
        tables.edges.add_row(0, 100, parent=3, child=2)
        tables.edges.add_row(0, 100, parent=3, child=4)
    else:
        tables.edges.add_row(0, 100, parent=2, child=0)
        tables.edges.add_row(0, 100, parent=3, child=2)
    tables.edges.add_row(0, 100, parent=3, child=1)
    for position, nodes in sites.items():
        site = tables.sites.add_row(position, "0")
        for node in nodes:
            tables.mutations.add_row(site=site, node=node, derived_state="1")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    tables.tree_sequence().dump(path)


def test_two_bit_status_round_trip_partial_byte():
    logical = np.array([0, 1, 2, 3, 2, 0], dtype=np.uint8)
    packed = pack_status_row(logical)
    assert packed.shape == (2,)
    np.testing.assert_array_equal(unpack_status_row(packed, logical.size), logical)
    assert packed[-1] >> 4 == 0


def test_vectorized_parent_lookup_matches_tree_parent_and_boundaries(tmp_path):
    path = tmp_path / "draw.trees"
    _write_ts(path, {10: [0], 49: [0], 50: [0], 99: [0], 70: [3]}, second_interval=True)
    ts = tskit.load(path)
    actual = lookup_mutation_parents(ts)
    expected = np.array([
        ts.at(ts.site(mutation.site).position).parent(mutation.node)
        for mutation in ts.mutations()
    ])
    np.testing.assert_array_equal(actual, expected)
    # Position 50 is right-exclusive for parent 2 and left-inclusive for 4.
    mutation = next(m for m in ts.mutations() if ts.site(m.site).position == 50)
    assert actual[mutation.id] == 4
    root = next(m for m in ts.mutations() if m.node == 3)
    assert actual[root.id] == tskit.NULL
    audit = audit_mutation_parent_lookup(ts, sample_size=5, seed=17)
    assert audit == {"sampled": 5, "covered": 4, "root_skipped": 1, "seed": 17}


def test_parent_audit_rejects_nonpositive_sample_and_handles_no_mutations(tmp_path):
    path = tmp_path / "empty.trees"
    _write_ts(path, {})
    ts = tskit.load(path)
    with pytest.raises(ValueError, match="positive"):
        audit_mutation_parent_lookup(ts, sample_size=0)
    assert audit_mutation_parent_lookup(ts) == {
        "sampled": 0, "covered": 0, "root_skipped": 0, "seed": 0,
    }


def test_fractional_edge_fallback_matches_tree_parent(tmp_path):
    path = tmp_path / "fractional.trees"
    _write_ts(path, {10: [0], 50: [0]}, second_interval=True)
    tables = tskit.load(path).dump_tables()
    # Move the breakpoint without changing integral mutation positions.
    edge = tables.edges[0]
    tables.edges[0] = edge.replace(right=49.5)
    edge = tables.edges[1]
    tables.edges[1] = edge.replace(left=49.5)
    tables.sort()
    ts = tables.tree_sequence()
    actual = lookup_mutation_parents(ts)
    expected = np.array([
        ts.at(ts.site(m.site).position).parent(m.node) for m in ts.mutations()
    ])
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="integral"):
        lookup_mutation_parents(ts, allow_fractional_edges=False)


def test_build_union_ragged_records_counts_status_and_metadata(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10: [0, 2], 20: [3]}, second_interval=True)
    _write_ts(second, {10: [0], 30: [2]}, second_interval=True)
    output = tmp_path / "store"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build_interval_store(
        [first, second], output, scratch_dir=scratch,
        interval_dtype="float32", num_buckets=5,
    )
    report = validate_interval_store(output, deep=True)
    assert (report.n_snps, report.n_intervals, report.n_posterior_draws) == (3, 4, 2)
    store = SNPAgeIntervalDataset.open(output, deep=True)
    assert store.positions.tolist() == [10, 20, 30]
    assert store.present_draw_count.tolist() == [2, 1, 1]
    assert store.missing_draw_count.tolist() == [0, 1, 1]
    assert store.usable_draw_count.tolist() == [2, 0, 1]
    assert store.usable_interval_count.tolist() == [3, 0, 1]
    assert store.skipped_root_count.tolist() == [0, 1, 0]
    assert store.read_status(rows=np.array([0, 1, 2])).tolist() == [[2, 1, 0], [2, 0, 2]]
    batch = store.intervals(np.array([0, 2]))
    assert batch.offsets.tolist() == [0, 3, 4]
    assert batch.draw_id.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(batch.below, [20, 0, 0, 20])
    np.testing.assert_allclose(batch.above, [100, 20, 20, 100])
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["catalog_sha256"]) == 64
    assert metadata["endpoint_dtype"] == "float32"
    assert metadata["maximum_above"] == 100
    assert metadata["minimum_usable_draws"] == 1
    assert list(scratch.iterdir()) == []


def test_missing_and_root_error_are_atomic(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10: [0], 20: [3]})
    _write_ts(second, {10: [0]})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="absent"):
        build_interval_store([first, second], output, scratch_dir=scratch, missing="error")
    assert not output.exists()
    assert list(scratch.iterdir()) == []
    with pytest.raises(ValueError, match="root"):
        build_interval_store([first], tmp_path / "root", scratch_dir=scratch, root="error")
    assert not (tmp_path / "root").exists()
    assert list(scratch.iterdir()) == []


def test_builder_loads_each_draw_twice(tmp_path, monkeypatch):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.trees"
        _write_ts(path, {10 + index: [0]})
        paths.append(path)
    calls = []
    original = builder._load

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(builder, "_load", counted)
    build_interval_store(paths, tmp_path / "store", num_buckets=2)
    assert calls == [paths[0].resolve(), paths[1].resolve()] * 2


def test_rejects_fractional_sites_and_existing_output(tmp_path):
    fractional = tmp_path / "fractional.trees"
    _write_ts(fractional, {10.5: [0]})
    with pytest.raises(ValueError, match="finite integers"):
        build_interval_store([fractional], tmp_path / "bad")
    valid = tmp_path / "valid.trees"
    _write_ts(valid, {10: [0]})
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_interval_store([valid], output)
