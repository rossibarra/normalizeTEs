import numpy as np

from benchmark_candidate_access import estimate_io, load_numeric_vector, run_benchmark
from snp_interval_dataset import SNPAgeIntervalDataset
from test_snp_interval_dataset import _store


def test_gate3_harness_compares_all_strategies(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    report = run_benchmark(
        store,
        np.array([2, 0]),
        np.array([5.0, 15.0, 25.0]),
        tmp_path / "scratch",
        repeats=1,
        block_rows=1,
        coalesce_gap=0,
    )
    assert set(report["results"]) == {"gather", "coalesced", "scan", "cache"}
    assert all(item["equal_to_gather"] for item in report["results"].values())
    assert report["results"]["scan"]["estimated_intervals_read"] == store.n_intervals
    assert report["results"]["cache"]["build_seconds"] >= 0
    assert list((tmp_path / "scratch").iterdir()) == []


def test_io_estimates_and_vector_loader(tmp_path):
    store = SNPAgeIntervalDataset.open(_store(tmp_path / "store"))
    rows = np.array([2, 0])
    gathered = estimate_io(store, rows, "gather", block_rows=2, coalesce_gap=0)
    scanned = estimate_io(store, rows, "scan", block_rows=2, coalesce_gap=0)
    assert gathered["estimated_intervals_read"] == 5
    assert scanned["estimated_intervals_read"] == 5
    path = tmp_path / "rows.npy"
    np.save(path, rows)
    np.testing.assert_array_equal(load_numeric_vector(path, integer=True), rows)
