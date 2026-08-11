import json
from pathlib import Path

import numpy as np
import pytest

from sample_age_matched_syn import (
    SamplingError,
    build_candidate_weights,
    draw_stratified_set,
    generate_matches,
    _load_candidates,
    score_set,
    wasserstein_1,
    write_result,
)
from snp_interval_dataset import INTERVAL_SCHEMA_VERSION, interval_cdf


class FakeStore:
    def __init__(self, cdfs):
        self.cdf_by_snp = np.asarray(cdfs, dtype=np.float64)
        self.positions = np.arange(len(cdfs), dtype=np.float64) + 100
        self.age_bins = np.arange(self.cdf_by_snp.shape[1], dtype=np.float64) * 1000
        self.eligible = np.ones(len(cdfs), dtype=bool)

    def read_cdfs(self, rows):
        return self.cdf_by_snp[rows]

    def read_boundary_cdfs(self, age_indices, start, stop):
        return self.cdf_by_snp[start:stop, :][:, age_indices].T

    def rows_to_native(self, rows):
        rows = np.asarray(rows)
        return np.full(rows.size, "chr1"), self.positions[rows].astype(np.int64)


def make_store():
    # Four young and four old candidates; all distributions end at one.
    return FakeStore([
        [.9, 1, 1], [.8, 1, 1], [.7, 1, 1], [.6, 1, 1],
        [.1, 1, 1], [.2, 1, 1], [.3, 1, 1], [.4, 1, 1],
    ])


def test_wasserstein_uses_age_spacing():
    assert wasserstein_1(np.array([0, 1, 1]), np.array([0, 0, 1]),
                         np.array([0, 1000, 3000])) == 2000


def test_block_index_and_draw_are_unique_with_exact_quotas():
    store = make_store()
    weights = build_candidate_weights(store, np.arange(8), np.array([0, 1, 3]),
                                      block_snps=3)
    rows, assignments = draw_stratified_set(
        weights, np.array([2, 2]), np.random.default_rng(4))
    assert rows.size == np.unique(rows).size == 4
    assert np.bincount(assignments, minlength=2).tolist() == [2, 2]
    assert weights.values.shape == (8, 2)
    assert weights.values.dtype == np.float32


def test_reproducible_generation_and_cross_set_reuse():
    store = make_store()
    weights = build_candidate_weights(store, np.arange(8), np.array([0, 1, 3]), 4)
    kwargs = dict(store=store, weights=weights, quotas=np.array([1, 1]),
                  target_cdf=np.array([.5, 1, 1]), threshold=1000,
                  accepted_sets=5, max_proposals=20, seed=123)
    first, _ = generate_matches(**kwargs)
    second, _ = generate_matches(**kwargs)
    np.testing.assert_array_equal(first.row_indices, second.row_indices)
    assert all(np.unique(row).size == 2 for row in first.row_indices)
    # Eight candidates cannot fill ten output cells without reuse across sets.
    assert np.unique(first.row_indices).size < first.row_indices.size


def test_threshold_rejection_hits_max_proposals():
    store = make_store()
    weights = build_candidate_weights(store, np.arange(8), np.array([0, 1, 3]), 4)
    with pytest.raises(SamplingError, match="0 of 1 accepted"):
        generate_matches(store, weights, np.array([1, 1]),
                         np.array([.505, 1, 1]), threshold=0,
                         accepted_sets=1, max_proposals=3, seed=9)


def test_zero_mass_and_capacity_failures_are_clear():
    store = FakeStore([[1, 1, 1], [1, 1, 1]])
    weights = build_candidate_weights(store, np.arange(2), np.array([0, 1, 3]), 2)
    with pytest.raises(SamplingError, match="interval 1 has zero candidate mass"):
        draw_stratified_set(weights, np.array([0, 1]),
                            np.random.default_rng(1))
    capacity_store = FakeStore([[1, 1, 1], [1, 1, 1], [0, 0, 1]])
    capacity_weights = build_candidate_weights(
        capacity_store, np.arange(3), np.array([0, 1, 3]), 2)
    with pytest.raises(SamplingError, match="exceeds its positive-mass"):
        draw_stratified_set(capacity_weights, np.array([3, 0]),
                            np.random.default_rng(1))


def test_atomic_output(tmp_path: Path):
    store = make_store()
    weights = build_candidate_weights(store, np.arange(8), np.array([0, 1, 3]), 4)
    result, diagnostics = generate_matches(
        store, weights, np.array([1, 1]), np.array([.5, 1, 1]), 1000,
        accepted_sets=2, max_proposals=5, seed=3)
    output = tmp_path / "matches"
    write_result(output, result, diagnostics, {"seed": 3})
    assert np.load(output / "syn_positions.npy").shape == (2, 2)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["proposals"] == result.attempts
    assert metadata["rejections"] == result.rejection_count
    assert metadata["acceptance_rate"] == pytest.approx(2 / result.attempts)
    with pytest.raises(FileExistsError):
        write_result(output, result, diagnostics, {})


def test_boundary_reader_covers_first_bin_without_full_cdf_reads():
    class SpyStore(FakeStore):
        def __init__(self, cdfs):
            super().__init__(cdfs)
            self.full_reads = 0
            self.boundary_reads = []

        def read_cdfs(self, rows):
            self.full_reads += 1
            return super().read_cdfs(rows)

        def read_boundary_cdfs(self, age_indices, start, stop):
            self.boundary_reads.append((tuple(age_indices), start, stop))
            return super().read_boundary_cdfs(age_indices, start, stop)

    store = SpyStore([[.75, 1, 1], [.25, 1, 1], [.5, 1, 1], [.1, 1, 1]])
    # Deliberately unsorted and gapped candidates; construction sorts them and
    # each block uses an enclosing contiguous row slab.
    weights = build_candidate_weights(
        store, np.array([3, 0, 2]), np.array([0, 1, 3]), block_snps=2)
    assert weights.candidate_rows.tolist() == [0, 2, 3]
    # First interval is [edge 0, edge 1], so its total must include CDF[:, 0].
    assert weights.values[:, 0].sum() == pytest.approx(.75 + .5 + .1)
    reads_after_build = len(store.boundary_reads)
    draw_stratified_set(weights, np.array([1, 1]),
                        np.random.default_rng(2))
    assert store.full_reads == 0
    assert store.boundary_reads
    assert len(store.boundary_reads) == reads_after_build


class FakeIntervalStore:
    metadata = {"schema_version": INTERVAL_SCHEMA_VERSION}
    positions = np.arange(3, dtype=float)
    eligible = np.ones(3, dtype=bool)
    intervals = [(0.0, 1_000.0), (1_000.0, 2_000.0), (2_000.0, 3_000.0)]

    def boundary_cdfs(self, rows, points, **kwargs):
        return np.vstack([
            interval_cdf(np.array([self.intervals[r][0]]),
                         np.array([self.intervals[r][1]]), points,
                         side=kwargs["side"])[0]
            for r in rows
        ])

    def aggregate_cdf_at(self, rows, points, **kwargs):
        return self.boundary_cdfs(rows, points, **kwargs).mean(axis=0)

    def rows_to_native(self, rows):
        rows = np.asarray(rows)
        return np.full(rows.size, "chr1"), rows + 1


def test_interval_weights_use_physical_edges_and_scoring_uses_target_grid():
    store = FakeIntervalStore()
    bounds = np.array([0, 1, 3])
    ages = np.array([-500.0, 500.0, 2_500.0])
    weights = build_candidate_weights(
        store, np.array([2, 0, 1]), bounds, block_snps=2,
        boundary_ages=ages, access_strategy="gather")
    np.testing.assert_allclose(weights.values[:, 0], [.5, 0, 0])
    np.testing.assert_allclose(weights.values[:, 1], [.5, 1, .5])
    grid = np.array([0, 1_000, 2_000, 3_000], dtype=float)
    target = np.array([.25, .75, 1, 1])
    aggregate, distance = score_set(store, np.array([0, 1]), target, grid)
    np.testing.assert_allclose(aggregate, target)
    assert distance == pytest.approx(0)


def test_index_candidates_drop_ineligible_and_report_exact_coordinate(tmp_path):
    store = make_store()
    store.eligible[1] = False
    path = tmp_path / "indices.npy"
    np.save(path, np.array([2, 1, 3], dtype=np.int64))
    rows, metadata = _load_candidates(store, None, path, None, policy="drop")
    np.testing.assert_array_equal(rows, [2, 3])
    assert metadata["position_resolution"]["ineligible_count"] == 1
    assert metadata["excluded_positions"] == [{
        "request_index": 1, "global_position": 101, "chromosome": "chr1",
        "native_position": 101, "reason": "ineligible"}]
