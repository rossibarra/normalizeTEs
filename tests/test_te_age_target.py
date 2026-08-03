from pathlib import Path

import numpy as np
import pytest

from te_age_target import (
    aggregate_cdf,
    bootstrap_wasserstein,
    build_target,
    empirical_threshold,
    equal_mass_boundaries,
    largest_remainder_quotas,
    load_position_list,
    quantile_order_statistic_interval,
    wasserstein_1,
    write_target,
)


def test_wasserstein_identical_adjacent_distant_and_nonuniform():
    ages = np.array([0, 1_000, 2_000, 5_000])
    young = np.array([1.0, 1.0, 1.0, 1.0])
    adjacent = np.array([0.0, 1.0, 1.0, 1.0])
    distant = np.array([0.0, 0.0, 0.0, 1.0])
    assert wasserstein_1(young, young, ages) == 0
    assert wasserstein_1(young, adjacent, ages) == 1_000
    assert wasserstein_1(young, distant, ages) == 5_000


def test_aggregate_normalizes_terminal_values():
    rows = np.array([[0, 1, 2], [0, 2, 2]], dtype=float)
    np.testing.assert_allclose(aggregate_cdf(rows), [0, 0.75, 1])


def test_bootstrap_is_reproducible_and_uses_generation_widths():
    rows = np.array([[0, 1, 1], [0, 0, 1]], dtype=float)
    kwargs = dict(n_replicates=25, batch_size=7, bin_centers=np.array([0, 1_000, 3_000]))
    first = bootstrap_wasserstein(rows, rng=np.random.default_rng(42), **kwargs)
    second = bootstrap_wasserstein(rows, rng=np.random.default_rng(42), **kwargs)
    np.testing.assert_array_equal(first, second)
    one_batch = bootstrap_wasserstein(
        rows,
        rng=np.random.default_rng(42),
        **{**kwargs, "batch_size": 25},
    )
    np.testing.assert_array_equal(first, one_batch)
    assert set(first).issubset({0.0, 500.0, 1_000.0})


def test_two_sample_bootstrap_supported():
    rows = np.array([[0, 1, 1], [0, 0, 1]], dtype=float)
    values = bootstrap_wasserstein(
        rows,
        10,
        np.random.default_rng(2),
        4,
        bin_centers=np.array([0, 1_000, 2_000]),
        reference="two-sample",
    )
    assert values.shape == (10,)
    assert np.all(values >= 0)


def test_boundaries_compress_repeats_and_span_cdf_edges():
    cdf = np.array([0.0, 0.6, 0.6, 1.0])
    probabilities = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    result = equal_mass_boundaries(
        cdf, probabilities, bin_centers=np.array([0, 1_000, 2_000, 3_000])
    )
    np.testing.assert_array_equal(result.requested_indices, [0, 2, 2, 2, 4, 4])
    np.testing.assert_array_equal(result.indices, [0, 2, 4])
    np.testing.assert_array_equal(result.ages, [0, 1_000, 3_000])
    np.testing.assert_allclose(result.interval_shares, [0.6, 0.4])


def test_first_bin_mass_is_covered_exactly_once_with_repeated_quantiles():
    # More than 5% in the first bin makes many requested quantiles share edge 1.
    cdf = np.array([0.30, 0.50, 0.95, 1.0])
    result = equal_mass_boundaries(
        cdf,
        bin_centers=np.array([0, 1_000, 2_000, 3_000]),
    )
    assert result.requested_indices[0] == 0
    assert result.requested_indices[-1] == len(cdf)
    assert result.indices[0] == 0
    assert result.indices[-1] == len(cdf)
    assert np.count_nonzero(result.requested_indices == 1) > 1
    np.testing.assert_allclose(result.interval_shares.sum(), 1.0)

    # Each compressed interval is a disjoint slice of the original PDF.
    pdf = np.diff(np.concatenate(([0.0], cdf)))
    covered = np.zeros(pdf.size, dtype=int)
    recovered = []
    for start, stop in zip(result.indices[:-1], result.indices[1:]):
        covered[start:stop] += 1
        recovered.append(pdf[start:stop].sum())
    np.testing.assert_array_equal(covered, np.ones(pdf.size, dtype=int))
    np.testing.assert_allclose(recovered, result.interval_shares)


def test_largest_remainder_exact_total_and_seeded_ties():
    shares = np.full(20, 0.05)
    a = largest_remainder_quotas(503, shares, np.random.default_rng(9))
    b = largest_remainder_quotas(503, shares, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert a.sum() == 503
    assert set(a) == {25, 26}


def test_threshold_and_order_statistic_interval():
    values = np.arange(100, dtype=float)
    assert empirical_threshold(values, 0.95) == 95
    lower, upper = quantile_order_statistic_interval(values, 0.95)
    assert lower <= 95 <= upper


def test_position_parser_rejects_duplicates(tmp_path):
    path = tmp_path / "positions.txt"
    path.write_text("chr1 10\nchr1 20 # comment\nchr1 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_position_list(path)


class FakeStore:
    age_bins = np.array([0, 1_000, 2_000, 3_000], dtype=np.uint64)
    positions = np.array([10.0, 20.0, 30.0])
    valid = np.array([True, True, False])
    cdfs = np.array(
        [[0, 1, 1, 1], [0, 0.5, 1, 1], [0, 0, 0, 0]], dtype=float
    )

    def resolve_positions(self, positions):
        indices = np.searchsorted(self.positions, positions)
        if np.any(indices == self.positions.size) or np.any(self.positions[indices] != positions):
            raise ValueError("missing position")
        return indices

    def read_cdfs(self, indices):
        return self.cdfs[indices]


def test_build_target_with_fake_store_is_reproducible():
    kwargs = dict(
        n_replicates=50,
        acceptance_quantile=0.95,
        seed=123,
        batch_size=10,
    )
    a = build_target(FakeStore(), np.array([10.0, 20.0]), **kwargs)
    b = build_target(FakeStore(), np.array([10.0, 20.0]), **kwargs)
    np.testing.assert_array_equal(a["bootstrap_wasserstein"], b["bootstrap_wasserstein"])
    np.testing.assert_array_equal(a["interval_quotas"], b["interval_quotas"])
    assert a["interval_quotas"].sum() == 2
    assert len(a["interval_quotas"]) == len(a["boundaries"].indices) - 1


def test_build_target_rejects_invalid_rows():
    with pytest.raises(ValueError, match="invalid TE positions: 30"):
        build_target(
            FakeStore(),
            np.array([30.0]),
            n_replicates=10,
            acceptance_quantile=0.95,
            seed=1,
        )


def test_atomic_write_refuses_overwrite(tmp_path):
    result = build_target(
        FakeStore(),
        np.array([10.0, 20.0]),
        n_replicates=10,
        acceptance_quantile=0.95,
        seed=1,
    )
    output = tmp_path / "target"
    write_target(output, result, {"schema_version": 1})
    assert (output / "target_cdf.npy").exists()
    with pytest.raises(FileExistsError):
        write_target(output, result, {"schema_version": 1})
