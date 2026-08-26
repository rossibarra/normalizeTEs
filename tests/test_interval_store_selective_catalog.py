import numpy as np
import tskit
import tszip

import normalize_tes.build_snp_interval_store as builder
from normalize_tes.snp_interval_dataset import SNPAgeIntervalDataset


def _write_tree(path):
    tables = tskit.TableCollection(sequence_length=100)
    tables.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.metadata = {
        "chrom_offsets": [{"chrom": "chr1", "offset": 0, "length": 100}]
    }
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=20)
    tables.edges.add_row(0, 100, parent=1, child=0)
    for position in (7, 30, 99):
        site = tables.sites.add_row(position, "0")
        tables.mutations.add_row(site=site, node=0, derived_state="1")
    tables.sort()
    tables.tree_sequence().dump(path)


def test_selective_tsz_catalog_decodes_coordinates_and_metadata(tmp_path):
    ordinary = tmp_path / "draw.trees"
    compressed = tmp_path / "draw.tsz"
    _write_tree(ordinary)
    source = tskit.load(ordinary)
    tszip.compress(source, compressed)
    positions, metadata, sequence_length = builder._selective_tsz_catalog(compressed)
    np.testing.assert_array_equal(positions, source.tables.sites.position)
    assert metadata == source.metadata
    assert sequence_length == source.sequence_length
    header = builder._catalog_header(compressed)
    assert header[3] == "selective_tsz_zarr"


def test_ordinary_tree_catalog_uses_full_load_fallback(tmp_path):
    ordinary = tmp_path / "draw.trees"
    _write_tree(ordinary)
    positions, metadata, sequence_length, method = builder._catalog_header(ordinary)
    assert positions.tolist() == [7, 30, 99]
    assert metadata["chrom_offsets"][0]["chrom"] == "chr1"
    assert sequence_length == 100
    assert method == "full_tree_sequence_fallback"


def test_tsz_build_performs_only_one_full_tree_sequence_load(tmp_path, monkeypatch):
    ordinary = tmp_path / "draw.trees"
    compressed = tmp_path / "draw.tsz"
    _write_tree(ordinary)
    tszip.compress(tskit.load(ordinary), compressed)
    calls = []
    original_load = builder._load

    def counted(path):
        calls.append(path)
        return original_load(path)

    monkeypatch.setattr(builder, "_load", counted)
    output = tmp_path / "store"
    builder.build_interval_store([compressed], output, num_buckets=2)
    assert calls == [compressed.resolve()]
    store = SNPAgeIntervalDataset.open(output, deep=True)
    assert store.positions.tolist() == [7, 30, 99]
    assert store.metadata["catalog_access_methods"] == ["selective_tsz_zarr"]

