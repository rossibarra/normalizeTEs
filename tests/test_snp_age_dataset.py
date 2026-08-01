import json

import numpy as np
import pytest

from snp_age_dataset import SNPAgeDataset, validate_store


def _store(path, transpose=True):
    positions = np.array([1.5, 9.0], dtype=np.float64)
    bins = np.array([0, 1000, 2000], dtype=np.uint64)
    cdf = np.array([[100, 1000, 65535], [0, 0, 0]], dtype=np.uint16)
    arrays = {
        "positions": positions, "age_bins": bins, "cdf_by_snp": cdf,
        "valid": np.array([True, False]),
        "present_draw_count": np.array([2, 1], dtype=np.uint32),
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
        "schema_version": 1, "n_snps": 2, "n_age_bins": 3,
    }))


def test_open_resolve_and_reads(tmp_path):
    path = tmp_path / "store"
    _store(path)
    dataset = SNPAgeDataset.open(path)
    assert dataset.resolve_positions(np.array([9.0, 1.5])).tolist() == [1, 0]
    assert dataset.read_cdfs(np.array([0]), decode=False).tolist() == [[100, 1000, 65535]]
    assert dataset.read_boundary_cdfs(np.array([0, 2]), 0, 2, decode=False).tolist() == [[100, 0], [65535, 0]]
    with pytest.raises(KeyError, match="not found"):
        dataset.resolve_positions(np.array([2.0]))
    with pytest.raises(ValueError, match="duplicate"):
        dataset.resolve_positions(np.array([1.5, 1.5]))


def test_boundary_fallback_without_transpose(tmp_path):
    path = tmp_path / "store"
    _store(path, transpose=False)
    dataset = SNPAgeDataset.open(path)
    assert dataset.read_boundary_cdfs(np.array([1]), 0, 2, decode=False).tolist() == [[1000, 0]]


def test_validation_detects_nonmonotone_cdf(tmp_path):
    path = tmp_path / "store"
    _store(path)
    np.save(path / "cdf_by_snp.npy", np.array([[100, 50, 65535], [0, 0, 0]], dtype=np.uint16))
    with pytest.raises(ValueError, match="not monotone"):
        validate_store(path, deep=True)
