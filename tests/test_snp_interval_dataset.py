import json

import numpy as np
import pytest

from snp_interval_dataset import (
    INTERVAL_SCHEMA_VERSION,
    SNPAgeIntervalDataset,
    interval_cdf,
    pack_status,
    unpack_status,
    validate_interval_store,
)


def _store(path, *, endpoint_dtype="float64"):
    # Rows own 2, 0, and 3 intervals respectively.  Row 2 has two intervals
    # in draw 0, making interval and equal-draw weighting observably distinct.
    arrays = {
        "positions": np.array([1.0, 10.0, 102.0], dtype=np.float64),
        "offsets": np.array([0, 2, 2, 5], dtype=np.uint64),
        "below": np.array([0, 10, 0, 0, 20], dtype=endpoint_dtype),
        "above": np.array([10, 20, 10, 20, 30], dtype=endpoint_dtype),
        "draw_id": np.array([0, 1, 0, 0, 1], dtype=np.uint8),
        "present_draw_count": np.array([2, 1, 2], dtype=np.uint32),
        "missing_draw_count": np.array([0, 1, 0], dtype=np.uint32),
        "usable_draw_count": np.array([2, 0, 2], dtype=np.uint32),
        "usable_interval_count": np.array([2, 0, 3], dtype=np.uint32),
        "skipped_root_count": np.array([0, 1, 0], dtype=np.uint32),
    }
    status = np.array([[2, 0, 2], [2, 1, 2]], dtype=np.uint8)
    arrays["status"] = pack_status(status)
    path.mkdir()
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    array_metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in arrays.items()
    }
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": INTERVAL_SCHEMA_VERSION,
        "n_snps": 3,
        "n_intervals": 5,
        "n_posterior_draws": 2,
        "maximum_above": 30.0,
        "endpoint_dtype": endpoint_dtype,
        "minimum_usable_draws": 2,
        "arrays": array_metadata,
        "chromosomes": [
            {"chrom": "chr1", "offset": 0, "length": 100},
            {"chrom": "chr2", "offset": 101, "length": 100},
        ],
    }), encoding="utf-8")
    return path


def test_two_bit_status_round_trip_and_partial_byte():
    status = np.array([[0, 1, 2, 3, 2], [3, 2, 1, 0, 1]], dtype=np.uint8)
    packed = pack_status(status)
    assert packed.shape == (2, 2)
    assert np.all((packed[:, -1] >> 2) == 0)
    np.testing.assert_array_equal(unpack_status(packed, 5), status)
    with pytest.raises(ValueError, match="0..3"):
        pack_status(np.array([4]))


@pytest.mark.parametrize("endpoint_dtype", ["float32", "float64"])
def test_open_intervals_means_and_schema(tmp_path, endpoint_dtype):
    path = _store(tmp_path / "store", endpoint_dtype=endpoint_dtype)
    report = validate_interval_store(path, deep=True, row_block_size=2)
    assert report.endpoint_dtype == endpoint_dtype
    store = SNPAgeIntervalDataset.open(path)
    batch = store.intervals(np.array([2, 0, 2]))
    assert batch.offsets.tolist() == [0, 3, 5, 8]
    assert batch.draw_id.tolist() == [0, 0, 1, 0, 1, 0, 0, 1]
    np.testing.assert_allclose(store.mean_ages(np.array([0, 1])), [10, np.nan], equal_nan=True)
    # Row 2 interval midpoints are 5, 10, 25. Equal interval mean = 40/3;
    # equal draw mean = mean(mean(5, 10), 25) = 16.25.
    np.testing.assert_allclose(store.mean_ages(np.array([2])), [40 / 3])
    np.testing.assert_allclose(store.mean_ages(np.array([2]), weighting="draw"), [16.25])
    assert store.valid.tolist() == [True, False, True]
    assert store.eligible.tolist() == [True, False, True]


def test_cdf_side_semantics_cell_masses_and_weighting(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    np.testing.assert_allclose(
        store.cdf_at(np.array([0]), np.array([0, 5, 10, 15, 20])),
        [[0, 0.25, 0.5, 0.75, 1]],
    )
    assert np.isnan(store.cdf_at(np.array([1]), np.array([10]))[0, 0])
    np.testing.assert_allclose(
        store.cell_masses(np.array([0]), np.array([0, 10, 20])), [[0.5, 0.5]]
    )
    # At t=10, row 2 values are [1, 0.5, 0].
    np.testing.assert_allclose(store.cdf_at(np.array([2]), np.array([10])), [[0.5]])
    np.testing.assert_allclose(
        store.boundary_cdfs(np.array([2]), np.array([10]), weighting="draw"), [[0.375]]
    )
    point = interval_cdf(np.array([10.0]), np.array([10.0]), np.array([9, 10, 11]), side="right")
    strict = interval_cdf(np.array([10.0]), np.array([10.0]), np.array([9, 10, 11]), side="left")
    np.testing.assert_array_equal(point, [[0, 1, 1]])
    np.testing.assert_array_equal(strict, [[0, 0, 1]])


@pytest.mark.parametrize("strategy", ["gather", "coalesced", "scan"])
def test_boundary_access_strategies_preserve_scattered_order(tmp_path, strategy):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    rows = np.array([2, 0])
    expected = store.cdf_at(rows, np.array([5.0, 15.0]), side="left")
    actual = store.boundary_cdfs(
        rows, np.array([5.0, 15.0]), side="left",
        access_strategy=strategy, block_rows=2, coalesce_gap=1)
    np.testing.assert_allclose(actual, expected)


def test_aggregate_cdf_is_blockwise_mean_and_cache_requires_path(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    rows = np.array([2, 0])
    points = np.array([5.0, 15.0])
    expected = store.cdf_at(rows, points).mean(axis=0)
    np.testing.assert_allclose(
        store.aggregate_cdf_at(rows, points, block_rows=1), expected)
    with pytest.raises(ValueError, match="requires a candidate cache"):
        store.boundary_cdfs(rows, points, access_strategy="cache")


def test_candidate_cache_matches_gather_and_restores_requested_order(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    candidates = np.array([2, 0])
    cache = store.build_candidate_cache(candidates, tmp_path / "scratch-cache", block_rows=1)
    assert cache.source_rows.tolist() == [0, 2]
    assert cache.metadata["build_strategy"] == "full-sequential-scan"
    points = np.array([5.0, 15.0, 25.0])
    expected = store.boundary_cdfs(candidates, points, access_strategy="gather")
    actual = store.boundary_cdfs(
        candidates, points, access_strategy="cache", cache=cache
    )
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(FileExistsError):
        store.build_candidate_cache(candidates, tmp_path / "scratch-cache")
    with pytest.raises(KeyError, match="not present"):
        cache.cdf_at(np.array([1]), points)


def test_regular_grid_aggregate_matches_rowwise_cdf(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    rows = np.array([2, 0])
    points = np.arange(-5.0, 40.0, 2.5)
    expected = store.cdf_at(rows, points, side="left").mean(axis=0)
    actual = store.aggregate_cdf_at(rows, points, side="left")
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_regular_grid_cdf_writer_matches_rowwise_cdf(tmp_path, dtype):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    rows = np.array([2, 0])
    points = np.arange(-5.0, 40.0, 2.5)
    expected = store.cdf_at(rows, points, side="left")
    output = store.write_regular_grid_cdfs(
        rows, points, tmp_path / f"cdfs-{dtype}.npy",
        block_rows=1, dtype=dtype,
    )
    np.testing.assert_allclose(output, expected, atol=2e-7)
    with pytest.raises(FileExistsError):
        store.write_regular_grid_cdfs(rows, points, output.filename)


def test_native_coordinates_preserve_one_base_gap(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    assert store.native_to_global(np.array(["chr1", "chr2"]), np.array([100, 1])).tolist() == [100, 102]
    names, native = store.rows_to_native(np.array([0, 2]))
    assert names.tolist() == ["chr1", "chr2"]
    assert native.tolist() == [1, 1]
    assert store.resolve_native_positions(np.array(["chr2"]), np.array([1])).tolist() == [2]
    with pytest.raises(KeyError, match="not found"):
        store.resolve_native_positions(np.array(["chr1"]), np.array([100]))


def test_read_selected_status(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    assert store.read_status(draws=np.array([1]), rows=np.array([2, 1, 0])).tolist() == [[2, 1, 2]]


def test_validation_rejects_corrupt_arrays(tmp_path):
    path = _store(tmp_path / "store")
    np.save(path / "above.npy", np.array([0, 20, 10, 20, 30], dtype=np.float64))
    # Shallow validation deliberately avoids scanning multi-billion-record endpoints.
    validate_interval_store(path)
    with pytest.raises(ValueError, match="below < above"):
        validate_interval_store(path, deep=True)


def test_validation_rejects_nonzero_unused_status_slots(tmp_path):
    path = _store(tmp_path / "store")
    status = np.load(path / "status.npy")
    status[:, -1] |= np.uint8(1 << 6)
    np.save(path / "status.npy", status)
    with pytest.raises(ValueError, match="unused status"):
        validate_interval_store(path)
