import numpy as np
import pytest

from te_age_target import (
    aggregate_cdf,
    bootstrap_wasserstein,
    build_target,
    empirical_threshold,
    equal_mass_boundaries,
    largest_remainder_quotas,
    wasserstein_1,
    write_target,
)
from snp_interval_dataset import INTERVAL_SCHEMA_VERSION, interval_cdf
from snp_age_dataset import load_native_position_list


def test_wasserstein_identical_adjacent_distant_and_nonuniform():
    ages = np.array([0, 1_000, 2_000, 5_000])
    young = np.array([1.0, 1.0, 1.0, 1.0])
    adjacent = np.array([0.0, 1.0, 1.0, 1.0])
    distant = np.array([0.0, 0.0, 0.0, 1.0])
    assert wasserstein_1(young, young, ages) == 0
    assert wasserstein_1(young, adjacent, ages) == 1_000
    assert wasserstein_1(young, distant, ages) == 5_000


def test_aggregate_averages_cdfs():
    rows = np.array([[0, 0.5, 1], [0, 1, 1]], dtype=float)
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


def test_boundaries_compress_repeats_and_span_cdf_edges():
    cdf = np.array([0.0, 0.6, 0.6, 1.0])
    result = equal_mass_boundaries(
        cdf, bin_centers=np.array([0, 1_000, 2_000, 3_000])
    )
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
    assert result.indices[0] == 0
    assert result.indices[-1] == len(cdf)
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


def test_largest_remainder_exact_total_and_deterministic_ties():
    shares = np.full(20, 0.05)
    a = largest_remainder_quotas(503, shares)
    b = largest_remainder_quotas(503, shares)
    np.testing.assert_array_equal(a, b)
    assert a.sum() == 503
    assert set(a) == {25, 26}


def test_empirical_threshold():
    values = np.arange(100, dtype=float)
    assert empirical_threshold(values, 0.95) == 95


def test_position_parser_rejects_duplicates(tmp_path):
    path = tmp_path / "positions.txt"
    path.write_text("chr1 10\nchr1 20 # comment\nchr1 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_native_position_list(path)


class FakeStore:
    age_bins = np.array([0, 1_000, 2_000, 3_000], dtype=np.uint64)
    positions = np.array([10.0, 20.0, 30.0])
    eligible = np.array([True, True, False])
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


def test_build_target_is_reproducible():
    kwargs = dict(
        n_replicates=50,
        acceptance_quantile=0.95,
        seed=123,
        batch_size=10,
    )
    native = (np.array(["chr1", "chr1"]), np.array([10, 20]))
    a = build_target(FakeStore(), np.array([10.0, 20.0]), *native, **kwargs)
    b = build_target(FakeStore(), np.array([10.0, 20.0]), *native, **kwargs)
    np.testing.assert_array_equal(a.bootstrap_wasserstein, b.bootstrap_wasserstein)
    np.testing.assert_array_equal(a.interval_quotas, b.interval_quotas)
    assert a.interval_quotas.sum() == 2
    assert len(a.interval_quotas) == len(a.boundaries.indices) - 1


def test_build_target_rejects_invalid_rows():
    with pytest.raises(ValueError, match="invalid TE positions: 30"):
        build_target(
            FakeStore(),
            np.array([30.0]),
            np.array(["chr1"]),
            np.array([30]),
            n_replicates=10,
            acceptance_quantile=0.95,
            seed=1,
        )


def test_atomic_write_refuses_overwrite(tmp_path):
    result = build_target(
        FakeStore(),
        np.array([10.0, 20.0]),
        np.array(["chr1", "chr1"]),
        np.array([10, 20]),
        n_replicates=10,
        acceptance_quantile=0.95,
        seed=1,
    )
    output = tmp_path / "target"
    write_target(output, result, {"schema_version": 1})
    assert (output / "target_cdf.npy").exists()
    assert (output / "age_bins.npy").exists()
    assert (output / "interval_boundary_ages.npy").exists()
    with pytest.raises(FileExistsError):
        write_target(output, result, {"schema_version": 1})


class FakeIntervalStore:
    metadata = {"schema_version": INTERVAL_SCHEMA_VERSION, "maximum_above": 2_500.0}
    positions = np.array([10.0, 20.0])
    eligible = np.array([True, True])
    below = np.array([[0.0], [1_500.0]])
    above = np.array([[1_000.0], [2_500.0]])

    def __init__(self):
        self.cdf_calls = []

    def cdf_at(self, rows, points, **kwargs):
        self.cdf_calls.append(np.asarray(rows).copy())
        return np.vstack([
            interval_cdf(self.below[row], self.above[row], points, side=kwargs["side"]).mean(axis=0)
            for row in rows
        ])


def test_interval_target_uses_right_cell_edges_and_persists_physical_boundaries(tmp_path):
    store = FakeIntervalStore()
    result = build_target(
        store, np.array([10.0, 20.0]),
        np.array(["chr1", "chr1"]), np.array([10, 20]),
        row_indices=np.array([0, 1]), n_replicates=10,
        acceptance_quantile=.95, seed=2, bin_width=1_000,
        scratch_dir=tmp_path, cdf_block_rows=1)
    np.testing.assert_array_equal(result.age_bins, [0, 1_000, 2_000, 3_000])
    # The first row has half its uniform interval mass below the 500 edge.
    assert result.target_cdf[0] == pytest.approx(.25)
    edges = np.array([-500, 500, 1_500, 2_500, 3_500], dtype=float)
    np.testing.assert_array_equal(result.boundary_ages, edges[result.boundaries.indices])
    assert [call.tolist() for call in store.cdf_calls] == [[0], [1]]
    assert list(tmp_path.iterdir()) == []


def test_interval_scratch_matrix_matches_in_memory_reference(tmp_path):
    store = FakeIntervalStore()
    rows = np.array([0, 1])
    points = np.array([500.0, 1_500.0, 2_500.0, 3_500.0])
    expected_rows = store.cdf_at(rows, points, side="left", weighting="interval")
    expected = bootstrap_wasserstein(
        expected_rows, 25, np.random.default_rng(9), 7,
        bin_centers=np.array([0, 1_000, 2_000, 3_000]),
    )
    result = build_target(
        store, np.array([10.0, 20.0]), np.array(["chr1", "chr1"]),
        np.array([10, 20]), row_indices=rows, n_replicates=25,
        acceptance_quantile=.95, seed=9, batch_size=7, bin_width=1_000,
        scratch_dir=tmp_path, cdf_block_rows=1,
    )
    np.testing.assert_allclose(result.bootstrap_wasserstein, expected, atol=1e-4)
    assert list(tmp_path.iterdir()) == []
