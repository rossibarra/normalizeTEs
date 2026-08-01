from pathlib import Path

import numpy as np
import pytest

from sample_age_matched_syn import (
    SamplingError,
    build_interval_block_index,
    draw_stratified_set,
    generate_matches,
    wasserstein_1,
    write_result,
)


class FakeStore:
    def __init__(self, cdfs):
        self.cdf_by_snp = np.asarray(cdfs, dtype=np.float64)
        self.positions = np.arange(len(cdfs), dtype=np.float64) + 100
        self.age_bins = np.arange(self.cdf_by_snp.shape[1], dtype=np.float64) * 1000
        self.valid = np.ones(len(cdfs), dtype=bool)

    def read_cdfs(self, rows):
        return self.cdf_by_snp[rows]


def make_store():
    # Four young and four old candidates; all distributions end at one.
    return FakeStore([
        [0, .9, 1], [0, .8, 1], [0, .7, 1], [0, .6, 1],
        [0, .1, 1], [0, .2, 1], [0, .3, 1], [0, .4, 1],
    ])


def test_wasserstein_uses_age_spacing():
    assert wasserstein_1(np.array([0, 1, 1]), np.array([0, 0, 1]),
                         np.array([0, 1000, 3000])) == 2000


def test_block_index_and_draw_are_unique_with_exact_quotas():
    store = make_store()
    index = build_interval_block_index(store, np.arange(8), np.array([0, 1, 2]),
                                       block_snps=3)
    rows, assignments = draw_stratified_set(
        store, index, np.array([2, 2]), np.random.default_rng(4))
    assert rows.size == np.unique(rows).size == 4
    assert np.bincount(assignments, minlength=2).tolist() == [2, 2]
    assert index.block_totals.shape == (2, 3)


def test_reproducible_generation_and_cross_set_reuse():
    store = make_store()
    index = build_interval_block_index(store, np.arange(8), np.array([0, 1, 2]), 4)
    kwargs = dict(store=store, index=index, quotas=np.array([1, 1]),
                  target_cdf=np.array([0, .5, 1]), threshold=1000,
                  accepted_sets=5, max_proposals=20, seed=123)
    first, _ = generate_matches(**kwargs)
    second, _ = generate_matches(**kwargs)
    np.testing.assert_array_equal(first.row_indices, second.row_indices)
    assert all(np.unique(row).size == 2 for row in first.row_indices)
    # Eight candidates cannot fill ten output cells without reuse across sets.
    assert np.unique(first.row_indices).size < first.row_indices.size


def test_threshold_rejection_hits_max_proposals():
    store = make_store()
    index = build_interval_block_index(store, np.arange(8), np.array([0, 1, 2]), 4)
    with pytest.raises(SamplingError, match="0 of 1 accepted"):
        generate_matches(store, index, np.array([1, 1]),
                         np.array([0, .505, 1]), threshold=0,
                         accepted_sets=1, max_proposals=3, seed=9)


def test_zero_mass_and_capacity_failures_are_clear():
    store = FakeStore([[0, 1, 1], [0, 1, 1]])
    index = build_interval_block_index(store, np.arange(2), np.array([0, 1, 2]), 2)
    with pytest.raises(SamplingError, match="interval 1 has zero candidate mass"):
        draw_stratified_set(store, index, np.array([0, 1]),
                            np.random.default_rng(1))
    capacity_store = FakeStore([[0, 1, 1], [0, 1, 1], [0, 0, 1]])
    capacity_index = build_interval_block_index(
        capacity_store, np.arange(3), np.array([0, 1, 2]), 2)
    with pytest.raises(SamplingError, match="exceeds its positive-mass"):
        draw_stratified_set(capacity_store, capacity_index, np.array([3, 0]),
                            np.random.default_rng(1))


def test_quantized_input_and_atomic_output(tmp_path: Path):
    store = make_store()
    store.cdf_by_snp = np.rint(store.cdf_by_snp * 65535).astype(np.uint16)
    store.quantization_scale = 65535
    index = build_interval_block_index(store, np.arange(8), np.array([0, 1, 2]), 4)
    result, diagnostics = generate_matches(
        store, index, np.array([1, 1]), np.array([0, .5, 1]), 1000,
        accepted_sets=2, max_proposals=5, seed=3)
    output = tmp_path / "matches"
    write_result(output, result, diagnostics, {"seed": 3})
    assert np.load(output / "syn_positions.npy").shape == (2, 2)
    assert (output / "metadata.json").is_file()
    with pytest.raises(FileExistsError):
        write_result(output, result, diagnostics, {})
