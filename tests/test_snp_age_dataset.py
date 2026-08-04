import json

import numpy as np
import pytest

from snp_age_dataset import SNPAgeDataset, validate_store


def _store(path, transpose=True):
    positions = np.array([1.0, 9.0], dtype=np.float64)
    bins = np.array([0, 1000, 2000], dtype=np.uint64)
    cdf = np.array([[100, 1000, 65535], [0, 0, 0]], dtype=np.uint16)
    arrays = {
        "positions": positions, "age_bins": bins, "cdf_by_snp": cdf,
        "valid": np.array([True, False]),
        "eligible": np.array([True, False]),
        "present_draw_count": np.array([2, 1], dtype=np.uint32),
        "usable_draw_count": np.array([2, 0], dtype=np.uint32),
        "usable_draw_fraction": np.array([1.0, 0.0], dtype=np.float32),
        "usable_interval_count": np.array([2, 0], dtype=np.uint32),
        "skipped_root_count": np.array([0, 1], dtype=np.uint32),
        "missing_draw_count": np.array([0, 1], dtype=np.uint32),
    }
    path.mkdir()
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    if transpose:
        np.save(path / "cdf_by_age.npy", cdf.T)
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": 2, "n_snps": 2, "n_age_bins": 3,
        "n_posterior_draws": 2, "minimum_usable_draws": 1,
        "chromosomes": [{"chrom": "chr1", "offset": 0, "length": 100}],
    }))


def test_open_resolve_and_reads(tmp_path):
    path = tmp_path / "store"
    _store(path)
    dataset = SNPAgeDataset.open(path)
    assert dataset.resolve_positions(np.array([9.0, 1.0])).tolist() == [1, 0]
    assert dataset.resolve_native_positions(np.array(["chr1"]), np.array([9])).tolist() == [1]
    names, native = dataset.rows_to_native(np.array([0, 1]))
    assert names.tolist() == ["chr1", "chr1"]
    assert native.tolist() == [1, 9]
    assert dataset.read_cdfs(np.array([0]), decode=False).tolist() == [[100, 1000, 65535]]
    assert dataset.read_boundary_cdfs(np.array([0, 2]), 0, 2, decode=False).tolist() == [[100, 0], [65535, 0]]
    with pytest.raises(KeyError, match="not found"):
        dataset.resolve_positions(np.array([2.0]))
    with pytest.raises(ValueError, match="duplicate"):
        dataset.resolve_positions(np.array([1.0, 1.0]))


def test_boundary_fallback_without_transpose(tmp_path):
    path = tmp_path / "store"
    _store(path, transpose=False)
    dataset = SNPAgeDataset.open(path)
    assert dataset.read_boundary_cdfs(np.array([1]), 0, 2, decode=False).tolist() == [[1000, 0]]


def test_native_coordinates_cover_first_and_last_base(tmp_path):
    path = tmp_path / "store"
    _store(path)
    dataset = SNPAgeDataset.open(path)
    assert dataset.native_to_global(
        np.array(["chr1", "chr1"]), np.array([1, 100])
    ).tolist() == [1.0, 100.0]
    names, positions = dataset.rows_to_native(np.array([0, 1]))
    assert names.tolist() == ["chr1", "chr1"]
    assert positions.tolist() == [1, 9]


def test_multichromosome_boundaries_round_trip():
    dataset = SNPAgeDataset.__new__(SNPAgeDataset)
    dataset.positions = np.array([1.0, 100.0, 101.0, 200.0])
    dataset.chromosomes = (
        {"chrom": "chr1", "offset": 0, "length": 100},
        {"chrom": "chr2", "offset": 100, "length": 100},
    )
    dataset._chromosome_by_name = {
        entry["chrom"]: entry for entry in dataset.chromosomes
    }
    rows = np.arange(4, dtype=np.int64)
    names, positions = dataset.rows_to_native(rows)
    assert names.tolist() == ["chr1", "chr1", "chr2", "chr2"]
    assert positions.tolist() == [1, 100, 1, 100]
    np.testing.assert_array_equal(
        dataset.native_to_global(names, positions), dataset.positions
    )


def test_validation_detects_nonmonotone_cdf(tmp_path):
    path = tmp_path / "store"
    _store(path)
    np.save(path / "cdf_by_snp.npy", np.array([[100, 50, 65535], [0, 0, 0]], dtype=np.uint16))
    with pytest.raises(ValueError, match="not monotone"):
        validate_store(path, deep=True)


def test_metadata_quantization_scale_is_used_for_decoding(tmp_path):
    path = tmp_path / "store"
    _store(path)
    cdf = np.array([[100, 500, 1000], [0, 0, 0]], dtype=np.uint16)
    np.save(path / "cdf_by_snp.npy", cdf)
    np.save(path / "cdf_by_age.npy", cdf.T)
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["quantization_scale"] = 1000
    (path / "metadata.json").write_text(json.dumps(metadata))
    decoded = SNPAgeDataset.open(path).read_cdfs(np.array([0]))
    np.testing.assert_allclose(decoded, [[0.1, 0.5, 1.0]])
