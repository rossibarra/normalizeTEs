import csv
import json

import numpy as np
import pytest

from bootstrap_target_matcher import (
    OptimizerConfig,
    bootstrap_cdf,
    bootstrap_counts,
    derive_seed,
    log_search_grid,
    main,
    optimize_restart,
    select_seed_indices,
    validate_restart_result,
)
from te_age_target import wasserstein_1, wasserstein_1_log_age
from snp_age_store import open_snp_age_store
from swap_control_sampler import analysis_points, eligible_candidates, search_grid
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
        search_bin_width=int(ages[1] - ages[0]),
        qc_max_ratio=1, qc_max_absolute=100,
    )
    coarse_ages, coarse_points = search_grid(
        float(store.metadata["maximum_above"]), config.search_bin_width
    )
    coarse_target = store.aggregate_cdf_at(
        target_rows, coarse_points, side="left", weighting="interval"
    )
    result = optimize_restart(
        store, candidates, np.array([4, 5]), target, target, ages, 10,
        coarse_target=coarse_target,
        coarse_ages=coarse_ages,
        coarse_points=coarse_points,
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


def _run_matcher(store, target, seeds, output, *extra):
    return main([
        "--store", str(store), "--target", str(target), "--seed-sets", str(seeds),
        "--all-eligible", "--output", str(output),
        "--replicates", "2", "--closest-restarts", "1", "--diverse-restarts", "1",
        "--min-epochs", "2", "--max-epochs", "3", "--patience", "1",
        "--material-improvement-ratio", "0", "--cdf-block-rows", "2",
        "--search-bin-width", "10",
        "--qc-max-ratio", "10", "--qc-max-absolute", "1000", "--seed", "23",
        *extra,
    ])


def test_bootstrap_bundle_is_consumable_by_phi_sfs(tmp_path):
    """The whole point of the bundle is to be read by the Phi-SFS step.

    Rounds 7 found the two ways this silently failed: a target_digest computed
    over different arrays than phi_sfs recomputes, and missing per-replicate
    identifier arrays. Both were invisible to every other test.
    """
    import phi_sfs
    from snp_age_store import open_snp_age_store

    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    output = tmp_path / "bootstrap_matches"
    assert _run_matcher(store, target, seeds, output) == 0

    # The digest must certify against the target directory ...
    target_meta, match_meta, digest = phi_sfs._validate_provenance(target, output)
    assert match_meta["target_digest"] == digest
    # ... and the identifiers must load under this bundle's own schema.
    assert match_meta["schema_version"] == "bootstrap-target-matches-v1"
    opened = open_snp_age_store(store)
    chromosomes, positions = opened.rows_to_native(
        np.load(target / "te_row_indices.npy")
    )
    np.save(target / "te_chromosomes.npy", chromosomes)
    np.save(target / "te_positions.npy", positions)
    *_, identifiers = phi_sfs._load_coordinates(
        target, output, match_meta["schema_version"]
    )
    assert list(identifiers) == ["replicate_id"]
    assert identifiers["replicate_id"].tolist() == [0, 1]
    # Chain/sample identifiers must NOT be invented for independent replicates.
    assert not (output / "chain_index.npy").exists()
    assert not (output / "sample_index.npy").exists()


def test_output_inside_work_dir_is_rejected(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="must not be inside"):
        _run_matcher(store, target, seeds, work / "nested", "--work-dir", str(work))
    with pytest.raises(ValueError, match="different paths"):
        _run_matcher(store, target, seeds, work, "--work-dir", str(work))


def test_resume_rejects_a_changed_implementation(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    work = tmp_path / "work"
    assert _run_matcher(
        store, target, seeds, tmp_path / "out",
        "--work-dir", str(work), "--keep-work",
    ) == 0
    identity = json.loads((work / "identity.json").read_text())
    assert identity["software"]["name"] == "normalizeTE"
    assert identity["numpy_version"]
    identity["software"]["git_commit"] = "0" * 40
    (work / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="parameters or provenance differ"):
        _run_matcher(
            store, target, seeds, tmp_path / "out2",
            "--work-dir", str(work), "--resume",
        )


def test_resume_reuses_completed_replicate_bundles(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    work = tmp_path / "work"
    assert _run_matcher(
        store, target, seeds, tmp_path / "first",
        "--work-dir", str(work), "--keep-work",
    ) == 0
    first = np.load(tmp_path / "first" / "row_indices.npy")
    assert _run_matcher(
        store, target, seeds, tmp_path / "second",
        "--work-dir", str(work), "--resume",
    ) == 0
    np.testing.assert_array_equal(
        first, np.load(tmp_path / "second" / "row_indices.npy")
    )


def test_source_store_provenance_is_the_store_not_the_repository(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    seeds = _seed_bundle(tmp_path / "seeds", store, target)
    output = tmp_path / "out"
    assert _run_matcher(store, target, seeds, output) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source_store"] == str(store.resolve())
    assert metadata["seed_sets_digest"]


def test_bootstrap_cdf_accumulates_in_float64(tmp_path):
    """float32 accumulation over many TE rows displaces the bootstrap target."""
    rng = np.random.default_rng(0)
    n_sites, grid = 40_000, 64
    rows32 = rng.random((n_sites, grid)).astype(np.float32)
    rows32.sort(axis=1)
    counts = bootstrap_counts(n_sites, np.random.default_rng(5))
    reference = (
        counts.astype(np.float64) @ rows32.astype(np.float64) / counts.sum()
    )
    np.testing.assert_allclose(
        bootstrap_cdf(counts, rows32), reference, rtol=0, atol=1e-12
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


def _retarget(path, metric, offset=None):
    """Stamp a distance metric onto a fixture target's metadata."""
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["distance_metric"] = metric
    if offset is not None:
        metadata["log_age_offset"] = offset
    (path / "metadata.json").write_text(json.dumps(metadata))
    return path


def test_log_search_grid_resolves_the_young_end_without_costing_more_points():
    """The coarse screen has to see the region the log-age metric prices.

    A uniform 20,000-generation screen puts the whole lower quartile of a
    production TE age distribution inside its first cell, so under log-age
    weighting the optimizer would be blind to exactly the ages the metric is
    meant to price. The geometric screen keeps the young end at full exact
    resolution and is no more expensive than the uniform one.
    """
    ages = np.arange(36_746, dtype=np.float64) * 1_000.0
    points = ages + 500.0
    linear_points = int(36_745_000 // 20_000) + 1
    coarse_ages, coarse_points = log_search_grid(
        ages, points, linear_points, 1_000.0
    )
    assert coarse_ages.size <= linear_points
    assert np.all(np.diff(coarse_ages) > 0)
    assert np.all(np.isin(coarse_ages, ages))
    np.testing.assert_allclose(coarse_points, coarse_ages + 500.0)
    assert coarse_ages[0] == ages[0] and coarse_ages[-1] == ages[-1]
    # Full exact resolution across the lower three quartiles of a production
    # in-gene TE age distribution (q75 is about 80,000 generations).
    fine = np.flatnonzero(np.diff(coarse_ages) > 1_000)
    assert coarse_ages[fine[0]] > 100_000
    with pytest.raises(ValueError, match="at least two"):
        log_search_grid(ages, points, 1, 1_000.0)


def test_matcher_refuses_a_metric_that_disagrees_with_its_target(tmp_path):
    """The acceptance threshold is metric-specific, so a mismatch is not a
    rescalable inconvenience -- it silently compares log-age nats to
    generations. Both directions must fail loudly."""
    store_path = _interval_store(tmp_path / "store")
    linear_target = _target(tmp_path / "linear", store_path)
    log_target = _retarget(
        _target(tmp_path / "log", store_path), "log-age", 10.0
    )
    seeds = _seed_bundle(tmp_path / "seeds", store_path, linear_target)
    with pytest.raises(ValueError, match="non-linear distance metric"):
        _run_matcher(store_path, log_target, seeds, tmp_path / "a")
    with pytest.raises(ValueError, match="requires a target built"):
        _run_matcher(
            store_path, linear_target, seeds, tmp_path / "b",
            "--distance", "log-age", "--log-age-offset", "10",
        )
    with pytest.raises(ValueError, match="offsets differ"):
        _run_matcher(
            store_path, log_target, seeds, tmp_path / "c",
            "--distance", "log-age", "--log-age-offset", "99",
        )


def test_log_age_bundle_records_log_age_distances(tmp_path):
    store_path = _interval_store(tmp_path / "store")
    target_path = _retarget(
        _target(tmp_path / "target", store_path), "log-age", 10.0
    )
    seeds = _seed_bundle(tmp_path / "seeds", store_path, target_path)
    output = tmp_path / "matches"
    assert _run_matcher(
        store_path, target_path, seeds, output,
        "--distance", "log-age", "--log-age-offset", "10",
        "--search-grid-spacing", "log",
    ) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["distance_metric"] == "log-age"
    assert metadata["distance_units"] == "log-age nats"
    assert metadata["log_age_offset"] == 10.0
    assert metadata["search_grid_spacing"] == "log"
    store = open_snp_age_store(store_path)
    ages = np.load(output / "age_bins.npy")
    points = analysis_points(ages)
    rows = np.load(output / "row_indices.npy")
    observed = np.load(output / "target_cdf.npy")
    bootstrap = np.load(output / "bootstrap_target_cdfs.npy")
    recorded_e = np.load(output / "match_to_bootstrap_w1.npy")
    recorded_o = np.load(output / "match_to_observed_w1.npy")
    recorded_b = np.load(output / "bootstrap_to_observed_w1.npy")
    for r in range(rows.shape[0]):
        cdf = store.aggregate_cdf_at(
            rows[r], points, side="left", weighting="interval"
        )
        assert np.isclose(
            recorded_e[r], wasserstein_1_log_age(cdf, bootstrap[r], ages, offset=10.0)
        )
        assert np.isclose(
            recorded_o[r], wasserstein_1_log_age(cdf, observed, ages, offset=10.0)
        )
        assert np.isclose(
            recorded_b[r],
            wasserstein_1_log_age(bootstrap[r], observed, ages, offset=10.0),
        )
        # Not merely a rescaling coincidence: the linear distance differs.
        assert not np.isclose(
            recorded_e[r], wasserstein_1(cdf, bootstrap[r], ages)
        )
