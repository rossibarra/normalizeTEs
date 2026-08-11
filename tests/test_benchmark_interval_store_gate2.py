import json

import pytest
import tskit
import tszip

import benchmark_interval_store_gate2 as gate2


def _write_fixture(path):
    tables = tskit.TableCollection(sequence_length=100)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 0
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 1
    tables.nodes.add_row(time=20)  # 2
    tables.nodes.add_row(time=100)  # 3, root
    tables.nodes.add_row(time=30)  # 4
    tables.edges.add_row(0, 50, parent=2, child=0)
    tables.edges.add_row(50, 100, parent=4, child=0)
    tables.edges.add_row(0, 100, parent=3, child=2)
    tables.edges.add_row(0, 100, parent=3, child=4)
    tables.edges.add_row(0, 100, parent=3, child=1)
    for position, nodes in {10: [0, 2], 50: [0], 70: [3], 90: [1]}.items():
        site = tables.sites.add_row(position, "0")
        for node in nodes:
            tables.mutations.add_row(site=site, node=node, derived_state="1")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    tables.tree_sequence().dump(path)


def test_gate2_tsz_report_covers_all_measurements_without_relookup(tmp_path, monkeypatch):
    ordinary = tmp_path / "draw.trees"
    compressed = tmp_path / "draw.tsz"
    _write_fixture(ordinary)
    tszip.compress(tskit.load(ordinary), compressed)
    original = gate2._parent_lookup_phases
    calls = 0

    def counted(ts):
        nonlocal calls
        calls += 1
        return original(ts)

    monkeypatch.setattr(gate2, "_parent_lookup_phases", counted)
    report = gate2.benchmark_gate2(
        compressed, num_buckets=3, audit_size=5,
        precision_sample_size=4, precision_points=5, seed=7,
    )
    assert calls == 1
    assert report["selective_sites_access"]["available"]
    assert report["selective_sites_access"]["matches_full_load"]
    assert report["full_tsz_load"]["site_count"] == 4
    phases = report["composite_parent_lookup"]
    assert set(("key_construction", "stable_edge_sort", "edge_reorder", "search_and_guards")) <= phases.keys()
    counts = report["counts_and_buckets"]
    assert counts["mutation_count"] == 5
    assert counts["usable_interval_count"] == 4
    assert counts["root_skipped_count"] == 1
    assert sum(counts["bucket_interval_counts"]) == 4
    assert report["float32_precision"]["sample_size"] == 4
    audit = report["scalar_parent_audit"]
    assert audit["passed"] and audit["actual_sample_size"] == 5
    assert audit["strata"] == {"predicted_usable": 4, "predicted_root": 1}


def test_ordinary_tree_records_selective_access_unavailable(tmp_path):
    ordinary = tmp_path / "draw.trees"
    _write_fixture(ordinary)
    report = gate2.benchmark_gate2(
        ordinary, num_buckets=2, audit_size=2,
        precision_sample_size=2, precision_points=3,
    )
    assert not report["selective_sites_access"]["available"]
    assert report["scalar_parent_audit"]["passed"]


def test_fractional_edges_use_structured_lookup(tmp_path):
    ordinary = tmp_path / "fractional.trees"
    tables = tskit.TableCollection(sequence_length=100)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=10)
    tables.edges.add_row(0.5, 100, parent=1, child=0)
    site = tables.sites.add_row(10, "0")
    tables.mutations.add_row(site=site, node=0, derived_state="1")
    tables.sort()
    tables.tree_sequence().dump(ordinary)
    report = gate2.benchmark_gate2(
        ordinary, num_buckets=2, audit_size=1,
        precision_sample_size=1, precision_points=3,
        fallback_node_sample=1)
    lookup = report["composite_parent_lookup"]
    assert lookup["algorithm"] == "structured_child_float64_left"
    assert not lookup["integral_edge_coordinates"]
    assert report["scalar_parent_audit"]["passed"]


def test_atomic_json_refuses_overwrite_and_leaves_valid_document(tmp_path):
    output = tmp_path / "gate2.json"
    gate2.write_json_atomic(output, {"gate": 2, "passed": True})
    assert json.loads(output.read_text()) == {"gate": 2, "passed": True}
    with pytest.raises(FileExistsError):
        gate2.write_json_atomic(output, {"gate": 2})
    assert not list(tmp_path.glob(".gate2.json.tmp.*"))
