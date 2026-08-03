import json

import numpy as np
import pytest
import tskit
import tszip

from build_snp_age_store import build_store, determine_age_grid, discover_positions
from snp_age_dataset import SNPAgeDataset, validate_store


def _write_ts(path, sites):
    """sites maps position to mutation node IDs; node 3 is the root."""
    tables = tskit.TableCollection(sequence_length=100)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = {
        "chrom_offsets": [{"chrom": "chr1", "offset": 0, "length": 100}]
    }
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 0
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 1
    tables.nodes.add_row(time=20)                             # 2
    tables.nodes.add_row(time=100)                            # 3
    tables.edges.add_row(0, 100, parent=2, child=0)
    tables.edges.add_row(0, 100, parent=3, child=2)
    tables.edges.add_row(0, 100, parent=3, child=1)
    for position, nodes in sites.items():
        site = tables.sites.add_row(position=position, ancestral_state="0")
        for node in nodes:
            tables.mutations.add_row(site=site, node=node, derived_state="1")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    tables.tree_sequence().dump(path)


def test_build_union_counts_quantization_and_transpose(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10.0: [0], 20.0: [3]})
    _write_ts(second, {10.0: [0], 30.0: [2]})
    output = tmp_path / "store"
    build_store([first, second], output, bin_width=10, block_snps=1)
    dataset = SNPAgeDataset.open(output)
    assert dataset.positions.tolist() == [10.0, 20.0, 30.0]
    assert dataset.valid.tolist() == [True, False, True]
    assert dataset.present_draw_count.tolist() == [2, 1, 1]
    assert dataset.usable_draw_count.tolist() == [2, 0, 1]
    assert dataset.eligible.tolist() == [True, False, True]
    assert dataset.missing_draw_count.tolist() == [0, 1, 1]
    assert dataset.skipped_root_count.tolist() == [0, 1, 0]
    raw = dataset.read_cdfs(np.array([0, 1, 2]), decode=False)
    assert raw[0, -1] == 65535 and raw[2, -1] == 65535
    assert np.all(raw[1] == 0)
    assert validate_store(output, deep=True).has_transpose
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["mutation_weighting"] == "interval"
    with pytest.raises(FileExistsError):
        build_store([first], output)


def test_draw_weighting_differs_from_interval_weighting(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10.0: [0, 2]})
    _write_ts(second, {10.0: [0]})
    interval, draw = tmp_path / "interval", tmp_path / "draw"
    build_store([first, second], interval, bin_width=10, mutation_weighting="interval")
    build_store([first, second], draw, bin_width=10, mutation_weighting="draw")
    a = SNPAgeDataset.open(interval).read_cdfs(np.array([0]))[0]
    b = SNPAgeDataset.open(draw).read_cdfs(np.array([0]))[0]
    assert not np.array_equal(a, b)
    assert b[2] > a[2]  # draw weighting gives more weight to the short young interval


def test_missing_and_root_error_policies(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10.0: [0], 20.0: [3]})
    _write_ts(second, {10.0: [0]})
    with pytest.raises(ValueError, match="absent"):
        build_store([first, second], tmp_path / "missing", bin_width=10, missing="error")
    with pytest.raises(ValueError, match="root"):
        build_store([first], tmp_path / "root", bin_width=10, root="error")


def test_omit_transpose_and_discovery(tmp_path):
    tree = tmp_path / "a.trees"
    _write_ts(tree, {7.0: [0]})
    assert discover_positions([tree]).tolist() == [7.0]
    output = tmp_path / "store"
    build_store([tree], output, bin_width=10, omit_transpose=True)
    report = validate_store(output, deep=True)
    assert not report.has_transpose
    assert not (output / "cdf_by_age.npy").exists()


def test_age_grid_has_at_least_two_points_for_young_intervals(tmp_path):
    tree = tmp_path / "young.trees"
    _write_ts(tree, {7.0: [0]})
    np.testing.assert_array_equal(determine_age_grid([tree], 1_000), [0, 1_000])


def test_tsz_input_and_minimum_usable_fraction(tmp_path):
    ordinary = tmp_path / "draw.trees"
    compressed = tmp_path / "draw.tsz"
    _write_ts(ordinary, {10.0: [0]})
    tszip.compress(tskit.load(ordinary), compressed)
    output = tmp_path / "store"
    build_store([compressed], output, min_usable_fraction=0.1)
    dataset = SNPAgeDataset.open(output)
    assert dataset.eligible.tolist() == [True]
    assert dataset.usable_draw_fraction.tolist() == [1.0]


def test_minimum_usable_fraction_filters_sparse_posterior_coverage(tmp_path):
    draws = []
    for index in range(10):
        tree = tmp_path / f"draw_{index}.trees"
        _write_ts(tree, {10.0: [0]} if index == 0 else {20.0: [0]})
        draws.append(tree)
    ten_percent = tmp_path / "ten_percent"
    twenty_percent = tmp_path / "twenty_percent"
    build_store(draws, ten_percent, min_usable_fraction=0.1)
    build_store(draws, twenty_percent, min_usable_fraction=0.2)
    first = SNPAgeDataset.open(ten_percent)
    second = SNPAgeDataset.open(twenty_percent)
    row = int(first.resolve_positions(np.array([10.0]))[0])
    assert first.usable_draw_count[row] == 1
    assert first.usable_draw_fraction[row] == pytest.approx(0.1)
    assert first.eligible[row]
    assert not second.eligible[row]


def test_minimum_usable_draws_python_api(tmp_path):
    tree = tmp_path / "draw.trees"
    _write_ts(tree, {10.0: [0]})
    output = tmp_path / "store"
    build_store([tree], output, min_usable_draws=1)
    assert SNPAgeDataset.open(output).eligible.tolist() == [True]


def test_builder_loads_each_draw_twice_independent_of_blocks(tmp_path, monkeypatch):
    import build_snp_age_store as builder

    draws = []
    for index in range(2):
        path = tmp_path / f"draw_{index}.trees"
        _write_ts(path, {10.0: [0], 20.0: [2]})
        draws.append(path)
    original = builder._load
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(builder, "_load", counted)
    builder.build_store(draws, tmp_path / "store", bin_width=10, block_snps=1)
    assert len(calls) == 2 * len(draws)


def test_builder_rejects_fractional_arg_positions_early(tmp_path):
    tree = tmp_path / "fractional.trees"
    _write_ts(tree, {10.5: [0]})
    with pytest.raises(ValueError, match="non-integer"):
        build_store([tree], tmp_path / "store")
