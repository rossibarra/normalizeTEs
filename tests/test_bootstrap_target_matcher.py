import csv
import json

import numpy as np
import pytest

from bootstrap_target_matcher import (
    OptimizerConfig,
    bootstrap_cdf,
    bootstrap_counts,
    derive_seed,
    main,
    optimize_restart,
    select_seed_indices,
    validate_restart_result,
)
from snp_age_store import open_snp_age_store
from swap_control_sampler import analysis_points, eligible_candidates
from test_swap_control_sampler import _interval_store, _target


def _seed_bundle(path, store_path, target_path):
    store = open_snp_age_store(store_path)
    ages = np.load(target_path / "age_bins.npy")
    points = analysis_points(ages)
    rows = np.array([[2, 3], [4, 5], [3, 6]], dtype=np.int64)
    cdfs = np.stack([
        store.aggregate_cdf_at(row, points, side="left", weighting="interval")
        for row in rows
    ])
    path.mkdir()
    np.save(path / "row_indices.npy", rows)
    np.save(path / "cdfs.npy", cdfs)
    (path / "metadata.json").write_text(json.dumps({
        "source_store_schema": store.metadata["schema_version"],
        "source_catalog_sha256": "fixture-catalog",
        "source_store_content_sha256": "a" * 64,
    }))
    return path


def test_bootstrap_counts_and_cdf_are_reproducible():
    rows = np.array([[0.0, 0.5, 1.0], [0.2, 0.8, 1.0]])
    first = bootstrap_counts(2, np.random.default_rng(7))
    second = bootstrap_counts(2, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first.sum() == 2
    np.testing.assert_allclose(
        bootstrap_cdf(np.array([1, 1]), rows), rows.mean(axis=0)
    )
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_cdf(np.array([1]), rows)


def test_seed_derivation_and_selection_are_stable():
    cdfs = np.array([
        [0.0, 0.2, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.8, 1.0],
    ])
    target = cdfs[1]
    ages = np.array([0.0, 1.0, 2.0])
    selected = select_seed_indices(
        cdfs, target, ages, closest=1, diverse=1,
        rng=np.random.default_rng(11),
    )
    assert selected[0] == (1, "closest")
    assert selected[1][1] == "diverse"
    assert derive_seed(3, "x", 2) == derive_seed(3, "x", 2)
    assert derive_seed(3, "x", 2, 0) != derive_seed(3, "x", 2, 1)


def test_exact_optimizer_trace_is_monotone_and_certified(tmp_path):
    store_path = _interval_store(tmp_path / "store")
    target_path = _target(tmp_path / "target", store_path)
    store = open_snp_age_store(store_path)
    target_rows = np.load(target_path / "te_row_indices.npy")
    target = np.load(target_path / "target_cdf.npy")
    ages = np.load(target_path / "age_bins.npy")
    candidates = eligible_candidates(store, target_rows)
    config = OptimizerConfig(
        replicates=1, closest_restarts=1, diverse_restarts=0,
        min_epochs=2, max_epochs=4, patience=2,
        material_improvement_ratio=0, cdf_block_rows=2,
        qc_max_ratio=1, qc_max_absolute=100,
    )
    result = optimize_restart(
        store, candidates, np.array([4, 5]), target, target, ages, 10,
        seed_index=0, restart_kind="closest", seed=19, config=config,
    )
    best = np.array([record["best_distance"] for record in result.trace])
    assert np.all(np.diff(best) <= 1e-12)
    certified = store.aggregate_cdf_at(
        result.rows, analysis_points(ages), side="left", weighting="interval"
    )
    np.testing.assert_allclose(result.cdf, certified)
    assert result.best_distance <= result.initial_distance
    assert np.unique(result.rows).size == result.rows.size
    validate_restart_result(
        result,
        store=store,
        candidates=candidates,
        bootstrap_target=target,
        observed_target=target,
        age_bins=ages,
        expected_seed_index=0,
        expected_kind="closest",
        expected_seed=19,
    )


def test_cli_writes_aligned_atomic_bundle(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    output = tmp_path / "bootstrap_matches"
    assert main([
        "--store", str(store),
        "--target", str(target),
        "--seed-sets", str(seeds),
        "--all-eligible",
        "--output", str(output),
        "--replicates", "2",
        "--closest-restarts", "1",
        "--diverse-restarts", "1",
        "--min-epochs", "2",
        "--max-epochs", "3",
        "--patience", "1",
        "--material-improvement-ratio", "0",
        "--cdf-block-rows", "2",
        "--qc-max-ratio", "10",
        "--qc-max-absolute", "1000",
        "--seed", "23",
    ]) == 0
    rows = np.load(output / "row_indices.npy")
    assert rows.shape == (2, 2)
    assert np.load(output / "bootstrap_counts.npy").shape == (2, 2)
    assert np.load(output / "bootstrap_target_cdfs.npy").shape == (2, 4)
    assert np.load(output / "restart_best_rows.npy").shape == (2, 2, 2)
    assert np.all(np.load(output / "triangle_ok.npy"))
    with (output / "replicates.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["complete"] is True
    assert metadata["replicates"] == 2
    assert metadata["restarts_per_replicate"] == 2
    assert metadata["qc_interpretation"].startswith("optimizer convergence")
    assert not any(path.name.startswith(f".{output.name}.tmp") for path in tmp_path.iterdir())
