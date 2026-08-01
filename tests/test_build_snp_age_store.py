import json

import numpy as np
import pytest
import tskit

from build_snp_age_store import build_store, discover_positions
from snp_age_dataset import SNPAgeDataset, validate_store


def _write_ts(path, sites):
    """sites maps position to mutation node IDs; node 3 is the root."""
    tables = tskit.TableCollection(sequence_length=100)
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
    _write_ts(first, {10.5: [0], 20.0: [3]})
    _write_ts(second, {10.5: [0], 30.0: [2]})
    output = tmp_path / "store"
    build_store([first, second], output, bin_width=10, block_snps=1)
    dataset = SNPAgeDataset.open(output)
    assert dataset.positions.tolist() == [10.5, 20.0, 30.0]
    assert dataset.valid.tolist() == [True, False, True]
    assert dataset.present_draw_count.tolist() == [2, 1, 1]
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
    _write_ts(tree, {7.25: [0]})
    assert discover_positions([tree]).tolist() == [7.25]
    output = tmp_path / "store"
    build_store([tree], output, bin_width=10, omit_transpose=True)
    report = validate_store(output, deep=True)
    assert not report.has_transpose
    assert not (output / "cdf_by_age.npy").exists()
