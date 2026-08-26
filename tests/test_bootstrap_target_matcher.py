import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from normalize_tes.bootstrap_target_matcher import (
    OptimizerConfig,
    bootstrap_cdf,
    bootstrap_counts,
    derive_seed,
    log_search_grid,
    main,
    optimize_restart,
    validate_restart_result,
)
from normalize_tes.snp_age_store import open_snp_age_store
from normalize_tes.swap_control_sampler import analysis_points, eligible_candidates, search_grid
from test_swap_control_sampler import _interval_store, _target


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


def test_seed_derivation_is_stable():
    assert derive_seed(3, "x", 2) == derive_seed(3, "x", 2)
    assert derive_seed(3, "x", 2, 0) != derive_seed(3, "x", 2, 1)
    # The bootstrap seed and every restart seed of the same replicate differ,
    # so restarts are independent stratified draws rather than repeats.
    assert derive_seed(3, "x", 2) != derive_seed(3, "x", 2, 0)
    assert derive_seed(3, "x", 2, 0) != derive_seed(3, "y", 2, 0)


def test_exact_optimizer_trace_is_monotone_and_certified(tmp_path):
    store_path = _interval_store(tmp_path / "store")
    target_path = _target(tmp_path / "target", store_path)
    store = open_snp_age_store(store_path)
    target_rows = np.load(target_path / "te_row_indices.npy")
    target = np.load(target_path / "target_cdf.npy")
    ages = np.load(target_path / "age_bins.npy")
    candidates = eligible_candidates(store, target_rows)
    config = OptimizerConfig(
        replicates=1, restarts=1,
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
        seed=19, config=config,
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
        expected_seed=19,
    )


def _run_matcher(store, target, output, *extra):
    return main([
        "--store", str(store), "--target", str(target),
        "--all-eligible", "--output", str(output),
        "--replicates", "2", "--restarts", "2",
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
    from normalize_tes import phi_sfs
    from normalize_tes.snp_age_store import open_snp_age_store

    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    output = tmp_path / "bootstrap_matches"
    assert _run_matcher(store, target, output) == 0

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
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="must not be inside"):
        _run_matcher(store, target, work / "nested", "--work-dir", str(work))
    with pytest.raises(ValueError, match="different paths"):
        _run_matcher(store, target, work, "--work-dir", str(work))


def test_resume_rejects_a_changed_implementation(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    work = tmp_path / "work"
    assert _run_matcher(
        store, target, tmp_path / "out",
        "--work-dir", str(work), "--keep-work",
    ) == 0
    identity = json.loads((work / "identity.json").read_text())
    assert identity["software"]["name"] == "normalizeTE"
    assert identity["numpy_version"]
    identity["software"]["git_commit"] = "0" * 40
    (work / "identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="parameters or provenance differ"):
        _run_matcher(
            store, target, tmp_path / "out2",
            "--work-dir", str(work), "--resume",
        )


def test_resume_reuses_completed_replicate_bundles(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    work = tmp_path / "work"
    assert _run_matcher(
        store, target, tmp_path / "first",
        "--work-dir", str(work), "--keep-work",
    ) == 0
    first = np.load(tmp_path / "first" / "row_indices.npy")
    assert _run_matcher(
        store, target, tmp_path / "second",
        "--work-dir", str(work), "--resume",
    ) == 0
    np.testing.assert_array_equal(
        first, np.load(tmp_path / "second" / "row_indices.npy")
    )


def test_source_store_provenance_is_the_store_not_the_repository(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    output = tmp_path / "out"
    assert _run_matcher(store, target, output) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source_store"] == str(store.resolve())
    # The seed-library path is gone, so the bundle must not claim one.
    assert metadata["initialisation"].startswith("stratified draw")
    assert "seed_sets" not in metadata and "seed_sets_digest" not in metadata


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
    output = tmp_path / "bootstrap_matches"
    assert main([
        "--store", str(store),
        "--target", str(target),
        "--all-eligible",
        "--output", str(output),
        "--replicates", "2",
        "--restarts", "2",
        "--min-epochs", "2",
        "--max-epochs", "3",
        "--patience", "1",
        "--material-improvement-ratio", "0",
        "--cdf-block-rows", "2",
        # The fixture store spans 27 generations, so the coarse log screen
        # needs a search width it can put at least two points inside.
        "--search-bin-width", "10",
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


def test_resume_rejects_a_different_dirty_source_state(tmp_path, monkeypatch):
    """Two dirty edits on one commit must not share a resume identity.

    `software_provenance` records the HEAD commit and a dirty flag, which are
    identical for any two sets of uncommitted edits. Without hashing the loaded
    modules a long job could resume across an implementation change and mix
    replicate bundles from two versions of the code.
    """
    from normalize_tes import release_provenance

    first = release_provenance.loaded_source_digest()
    assert first["sha256"] and first["modules"]

    # Simulate an edit to one loaded module by pointing the digest at a copy of
    # the repo whose matcher source differs by a single byte.
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    root = Path(release_provenance.__file__).resolve().parent.parent
    for name in first["modules"]:
        destination = shadow / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / name).read_bytes())
    target = shadow / "normalize_tes/bootstrap_target_matcher.py"
    target.write_bytes(target.read_bytes() + b"\n# one byte of drift\n")

    real_modules = {}
    for name in first["modules"]:
        stem = Path(name).stem
        module_name = "normalize_tes" if stem == "__init__" else f"normalize_tes.{stem}"
        if module_name in sys.modules:
            real_modules[name] = sys.modules[module_name]
    for name, module in real_modules.items():
        monkeypatch.setattr(module, "__file__", str(shadow / name), raising=False)
    second = release_provenance.loaded_source_digest(shadow)

    assert second["modules"] == first["modules"]
    assert second["sha256"] != first["sha256"], (
        "a one-byte change to a loaded module must change the resume identity"
    )
