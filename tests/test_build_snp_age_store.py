import json
from pathlib import Path

import numpy as np
import pytest
import tskit
import tszip

from normalize_tes.build_snp_age_store import build_store, inspect_inputs, load_chrom_offsets
from normalize_tes.snp_age_dataset import SNPAgeDataset, validate_store


def _write_ts(path, sites, *, sequence_length=100, chrom_offsets=None):
    """sites maps position to mutation node IDs; node 3 is the root."""
    tables = tskit.TableCollection(sequence_length=sequence_length)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    if chrom_offsets is None:
        chrom_offsets = [{"chrom": "chr1", "offset": 0, "length": sequence_length}]
    tables.metadata = {} if chrom_offsets == "omit" else {"chrom_offsets": chrom_offsets}
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 0
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)  # 1
    tables.nodes.add_row(time=20)                             # 2
    tables.nodes.add_row(time=100)                            # 3
    tables.edges.add_row(0, sequence_length, parent=2, child=0)
    tables.edges.add_row(0, sequence_length, parent=3, child=2)
    tables.edges.add_row(0, sequence_length, parent=3, child=1)
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
    assert metadata["minimum_usable_fraction"] == pytest.approx(0.1)
    with pytest.raises(FileExistsError):
        build_store([first], output)


def test_missing_and_root_error_policies(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10.0: [0], 20.0: [3]})
    _write_ts(second, {10.0: [0]})
    with pytest.raises(ValueError, match="absent"):
        build_store([first, second], tmp_path / "missing", bin_width=10, missing="error")
    with pytest.raises(ValueError, match="root"):
        build_store([first], tmp_path / "root", bin_width=10, root="error")


def test_multiple_mutations_in_any_draw_exclude_the_site(tmp_path):
    first, second = tmp_path / "a.trees", tmp_path / "b.trees"
    _write_ts(first, {10.0: [0, 2], 20.0: [0]})
    _write_ts(second, {10.0: [0], 20.0: [0]})
    output = tmp_path / "store"
    build_store([first, second], output, bin_width=10)
    dataset = SNPAgeDataset.open(output)
    assert dataset.multiple_mutation_draw_count.tolist() == [1, 0]
    assert dataset.eligible.tolist() == [False, True]


def test_omit_transpose_and_discovery(tmp_path):
    tree = tmp_path / "a.trees"
    _write_ts(tree, {7.0: [0]})
    assert inspect_inputs([tree], 10)[0].tolist() == [7.0]
    output = tmp_path / "store"
    build_store([tree], output, bin_width=10, omit_transpose=True)
    report = validate_store(output, deep=True)
    assert not report.has_transpose
    assert not (output / "cdf_by_age.npy").exists()


def test_age_grid_has_at_least_two_points_for_young_intervals(tmp_path):
    tree = tmp_path / "young.trees"
    _write_ts(tree, {7.0: [0]})
    np.testing.assert_array_equal(inspect_inputs([tree], 1_000)[1], [0, 1_000])


def test_age_grid_ignores_ancient_nodes_not_bounding_mutations(tmp_path):
    tree = tmp_path / "unused_ancient_node.trees"
    _write_ts(tree, {7.0: [0]})
    tables = tskit.load(tree).dump_tables()
    tables.nodes.add_row(time=1_000_000)
    tables.tree_sequence().dump(tree)
    assert inspect_inputs([tree], 10)[1][-1] < 1_000_000


def test_tsz_input_and_minimum_usable_fraction(tmp_path):
    ordinary = tmp_path / "draw.trees"
    compressed = tmp_path / "draw.tsz"
    _write_ts(ordinary, {10.0: [0]})
    tszip.compress(tskit.load(ordinary), compressed)
    output = tmp_path / "store"
    build_store([compressed], output, min_usable_fraction=0.1)
    dataset = SNPAgeDataset.open(output)
    assert dataset.eligible.tolist() == [True]


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
    assert first.usable_draw_count[row] / 10 == pytest.approx(0.1)
    assert first.eligible[row]
    assert not second.eligible[row]


def test_builder_loads_each_draw_twice_independent_of_blocks(tmp_path, monkeypatch):
    import normalize_tes.build_snp_age_store as builder

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


def test_chrom_offsets_file_supplies_missing_arg_metadata(tmp_path):
    tree = tmp_path / "no_metadata.trees"
    _write_ts(tree, {10.0: [0], 120.0: [0]}, sequence_length=200, chrom_offsets="omit")
    with pytest.raises(ValueError, match="chrom-offsets"):
        build_store([tree], tmp_path / "unsupplied", bin_width=10)
    offsets = tmp_path / "offsets.txt"
    offsets.write_text("# chrom\tlength\nchr1\t100\nchr2\t100\n", encoding="utf-8")
    output = tmp_path / "store"
    build_store([tree], output, bin_width=10, chrom_offsets=offsets)
    dataset = SNPAgeDataset.open(output)
    assert dataset.chromosomes == (
        {"chrom": "chr1", "offset": 0, "length": 100},
        {"chrom": "chr2", "offset": 100, "length": 100},
    )
    names, native = dataset.rows_to_native(np.array([0, 1]))
    assert names.tolist() == ["chr1", "chr2"]
    assert native.tolist() == [10, 20]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["chromosomes_source"] == str(offsets)


def test_chrom_offsets_file_accepts_explicit_three_column_layout(tmp_path):
    tree = tmp_path / "draw.trees"
    _write_ts(tree, {10.0: [0]}, sequence_length=200, chrom_offsets="omit")
    offsets = tmp_path / "offsets.txt"
    offsets.write_text("chr1 0 90\nchr2 120 80\n", encoding="utf-8")
    build_store([tree], tmp_path / "store", bin_width=10, chrom_offsets=offsets)
    dataset = SNPAgeDataset.open(tmp_path / "store")
    assert [row["offset"] for row in dataset.chromosomes] == [0, 120]
    assert [row["length"] for row in dataset.chromosomes] == [90, 80]


def test_chrom_offsets_file_overrides_arg_metadata_with_warning(tmp_path, capsys):
    tree = tmp_path / "draw.trees"
    _write_ts(tree, {10.0: [0]}, sequence_length=200)
    offsets = tmp_path / "offsets.txt"
    offsets.write_text("chrA 0 200\n", encoding="utf-8")
    build_store([tree], tmp_path / "store", bin_width=10, chrom_offsets=offsets)
    assert "takes precedence" in capsys.readouterr().err
    dataset = SNPAgeDataset.open(tmp_path / "store")
    assert dataset.rows_to_native(np.array([0]))[0].tolist() == ["chrA"]
    metadata = json.loads((tmp_path / "store" / "metadata.json").read_text())
    assert metadata["chromosomes_source"] == str(offsets)


def test_chrom_offsets_file_rejects_malformed_input(tmp_path):
    tree = tmp_path / "draw.trees"
    _write_ts(tree, {10.0: [0]}, sequence_length=200, chrom_offsets="omit")
    offsets = tmp_path / "offsets.txt"

    offsets.write_text("chr1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1 columns"):
        load_chrom_offsets(offsets)

    offsets.write_text("chr1 0 100\nchr2 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="same number of columns"):
        load_chrom_offsets(offsets)

    offsets.write_text("chr1 zero 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be integers"):
        load_chrom_offsets(offsets)

    offsets.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_chrom_offsets(offsets)

    offsets.write_text("chr1 0 150\nchr2 100 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        build_store([tree], tmp_path / "overlap", bin_width=10, chrom_offsets=offsets)

    offsets.write_text("chr1 0 100\nchr2 100 150\n", encoding="utf-8")
    with pytest.raises(ValueError, match="beyond sequence_length"):
        build_store([tree], tmp_path / "long", bin_width=10, chrom_offsets=offsets)

    offsets.write_text("chr1 0 100\nchr1 100 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid chromosome entry"):
        build_store([tree], tmp_path / "duplicate", bin_width=10, chrom_offsets=offsets)


def test_accumulator_uses_requested_scratch_directory(tmp_path, monkeypatch):
    import normalize_tes.build_snp_age_store as builder

    tree = tmp_path / "draw.trees"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    _write_ts(tree, {10.0: [0]})
    original = builder.np.lib.format.open_memmap
    accumulator_paths = []

    def record_accumulator(filename, *args, **kwargs):
        if Path(filename).name == "pdf_accumulator.npy":
            accumulator_paths.append(Path(filename))
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(builder.np.lib.format, "open_memmap", record_accumulator)
    build_store([tree], tmp_path / "store", scratch_dir=scratch)
    assert len(accumulator_paths) == 1
    assert scratch in accumulator_paths[0].parents
    assert list(scratch.iterdir()) == []
