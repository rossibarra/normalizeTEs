import json

import numpy as np
import pytest

from normalize_tes.distributed_age_match import main as distributed_main
from normalize_tes.sample_age_matched_controls import _atomic_copy_file, main
from normalize_tes.snp_interval_dataset import INTERVAL_SCHEMA_VERSION, pack_status
from normalize_tes.swap_control_sampler import (
    SwapSamplingError,
    SwapConfig,
    derive_chain_seed,
    feasible_walk_accepts,
    incremental_cdf,
    replacement_fraction,
    run_chain,
)


def _interval_store(path):
    count = 7
    below = np.array([0, 2, 5, 8, 11, 14, 17], dtype=np.float64)
    above = below + 10
    arrays = {
        "positions": np.arange(1, count + 1, dtype=np.float64),
        "offsets": np.arange(count + 1, dtype=np.uint64),
        "below": below,
        "above": above,
        "draw_id": np.zeros(count, dtype=np.uint8),
        "status": pack_status(np.full((1, count), 2, dtype=np.uint8)),
        "present_draw_count": np.ones(count, dtype=np.uint32),
        "missing_draw_count": np.zeros(count, dtype=np.uint32),
        "usable_draw_count": np.ones(count, dtype=np.uint32),
        "usable_interval_count": np.ones(count, dtype=np.uint32),
        "skipped_root_count": np.zeros(count, dtype=np.uint32),
    }
    path.mkdir()
    for name, values in arrays.items():
        np.save(path / f"{name}.npy", values)
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": INTERVAL_SCHEMA_VERSION,
        "n_snps": count,
        "n_intervals": count,
        "n_posterior_draws": 1,
        "maximum_above": float(above.max()),
        "endpoint_dtype": "float64",
        "minimum_usable_draws": 1,
        "arrays": {
            name: {"dtype": value.dtype.name, "shape": list(value.shape)}
            for name, value in arrays.items()
        },
        "chromosomes": [{"chrom": "1", "offset": 0, "length": 100}],
        "catalog_sha256": "fixture-catalog",
        "content_sha256": "a" * 64,
    }), encoding="utf-8")
    return path


def _target(path, store_path, *, rows=None, threshold=1_000.0):
    from normalize_tes.snp_interval_dataset import SNPAgeIntervalDataset
    from normalize_tes.te_age_target import (
        analysis_grid_edges,
        equal_mass_boundaries,
        largest_remainder_quotas,
    )

    store = SNPAgeIntervalDataset.open(store_path)
    rows = np.asarray(
        [0, 1] if rows is None else rows, dtype=np.int64
    )
    ages = np.array([0, 10, 20, 30], dtype=np.float64)
    cdf = store.aggregate_cdf_at(rows, ages + 5, side="left", weighting="interval")
    # The equal-mass strata a real te_age_target directory ships; the
    # bootstrap-target matcher initialises every restart from them.
    boundaries = equal_mass_boundaries(cdf, bin_centers=ages)
    quotas = largest_remainder_quotas(rows.size, boundaries.interval_shares)
    path.mkdir()
    np.save(path / "te_row_indices.npy", rows)
    np.save(path / "age_bins.npy", ages)
    np.save(path / "target_cdf.npy", cdf)
    np.save(path / "bootstrap_wasserstein.npy", np.array([threshold]))
    np.save(path / "interval_quotas.npy", quotas)
    np.save(
        path / "interval_boundary_ages.npy",
        analysis_grid_edges(ages)[boundaries.indices],
    )
    (path / "metadata.json").write_text(json.dumps({
        "source_store_schema": INTERVAL_SCHEMA_VERSION,
        "source_catalog_sha256": "fixture-catalog",
        "source_store_content_sha256": "a" * 64,
        "wasserstein_threshold_generations": threshold,
    }), encoding="utf-8")
    return path


def test_incremental_update_replacement_and_seed_are_exact():
    current = np.array([0.2, 0.8, 1.0])
    old = np.array([0.0, 0.5, 1.0])
    new = np.array([0.4, 1.0, 1.0])
    np.testing.assert_allclose(
        incremental_cdf(current, old, new, 2), [0.4, 1.05, 1.0]
    )
    assert replacement_fraction(np.array([1, 2, 3, 4]),
                                np.array([1, 2, 5, 6])) == pytest.approx(0.5)
    assert derive_chain_seed(7, "abc", 0, "v1") == derive_chain_seed(7, "abc", 0, "v1")
    assert derive_chain_seed(7, "abc", 0, "v1") != derive_chain_seed(7, "abc", 1, "v1")
    assert feasible_walk_accepts(5.0, 6.0, 7.0)
    assert not feasible_walk_accepts(5.0, 8.0, 7.0)


def test_run_chain_burns_in_thins_and_certifies(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(
        sets_per_chain=3,
        search_bin_width=10,
        burnin_accepted_sweeps=0.5,
        sample_accepted_sweeps=0.5,
        max_construction_epochs=3,
        max_chain_proposals=1_000,
        cdf_block_rows=2,
        progress_every=100,
    )
    result = run_chain(
        store, rows, cdf, ages, 1_000.0,
        candidate_rows=None, global_seed=9, target_digest="fixture",
        chain_index=0, config=config,
    )
    assert result.row_indices.shape == (3, 2)
    assert result.cdfs.shape == (3, 4)
    assert result.cdfs.dtype == np.float64
    assert np.all(result.wasserstein <= 1_000)
    assert all(np.unique(row).size == row.size for row in result.row_indices)
    assert not np.any(np.isin(result.row_indices, rows))
    thinning = [row for row in result.diagnostics if row["phase"] == "thinning"]
    assert all(row["required_accepted_swaps"] == 1 for row in thinning)


def test_noninteger_sweeps_round_up_and_overlap_counter_matches_sets(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store, rows=[0, 1, 2])
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(
        sets_per_chain=3,
        search_bin_width=10,
        burnin_accepted_sweeps=0.5,
        sample_accepted_sweeps=0.5,
        max_construction_epochs=3,
        max_chain_proposals=10_000,
        cdf_block_rows=2,
        progress_every=10_000,
    )
    result = run_chain(
        store, rows, cdf, ages, 1_000.0,
        candidate_rows=None, global_seed=17, target_digest="noninteger",
        chain_index=0, config=config,
    )
    phases = [
        record for record in result.diagnostics
        if record["phase"] in {"burnin", "thinning"}
    ]
    assert all(record["required_accepted_swaps"] == 2 for record in phases)
    thinning = [
        record for record in result.diagnostics if record["phase"] == "thinning"
    ]
    for record, previous, current in zip(
        thinning, result.row_indices[:-1], result.row_indices[1:], strict=True
    ):
        assert record["replacement_fraction"] == pytest.approx(
            replacement_fraction(current, previous)
        )


def test_construction_refines_coarse_plateau_to_reach_threshold(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store, threshold=7.1)
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(
        sets_per_chain=1,
        search_bin_width=100,
        burnin_accepted_sweeps=0.5,
        sample_accepted_sweeps=0.5,
        max_construction_epochs=20,
        max_chain_proposals=10_000,
        cdf_block_rows=2,
        progress_every=10_000,
    )
    result = run_chain(
        store, rows, cdf, ages, 7.1,
        candidate_rows=None, global_seed=0, target_digest="plateau",
        chain_index=0, config=config,
    )
    refinements = result.construction["search_refinements"]
    assert refinements
    assert refinements[0] == {
        "epoch": 1, "from_width": 100, "to_width": 50,
    }
    assert result.construction["entry_wasserstein"] <= 7.1


def test_construction_reports_plateau_at_exact_grid(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store, threshold=6.0)
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(
        sets_per_chain=1,
        search_bin_width=100,
        max_construction_epochs=20,
        max_exact_plateau_epochs=3,
        max_chain_proposals=10_000,
        cdf_block_rows=2,
        progress_every=10_000,
    )
    with pytest.raises(
        SwapSamplingError, match="plateaued for 3 epochs at the exact"
    ):
        run_chain(
            store, rows, cdf, ages, 6.0,
            candidate_rows=None, global_seed=12, target_digest="plateau",
            chain_index=0, config=config,
        )


def test_run_chain_rejects_candidate_universe_with_no_unselected_row(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(sets_per_chain=1, search_bin_width=10)
    with pytest.raises(SwapSamplingError, match="more rows than the target set"):
        run_chain(
            store, rows, cdf, ages, 1_000.0,
            candidate_rows=np.array([2, 3]), global_seed=9,
            target_digest="fixture", chain_index=0, config=config,
        )


def test_cli_writes_four_exact_sets_atomically(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    output = tmp_path / "controls"
    assert main([
        "--store", str(store), "--target", str(target),
        "--all-eligible", "--output", str(output),
        "--sets", "4", "--chains", "2", "--sets-per-chain", "2",
        "--workers", "2", "--seed", "11", "--search-bin-width", "10",
        "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]) == 0
    rows = np.load(output / "row_indices.npy")
    assert rows.shape == (4, 2)
    assert np.load(output / "wasserstein.npy").shape == (4,)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is True
    assert metadata["sets"] == 4
    assert metadata["software"]["name"] == "normalizeTE"
    assert metadata["software"]["version"] == "0.5.2"
    assert metadata["algorithm_version"] == (
        "swap-age-controls-v2.1-adaptive-construction"
    )
    assert metadata["maximum_wasserstein"] == pytest.approx(
        float(np.load(output / "wasserstein.npy").max())
    )
    assert metadata["software"]["git_commit"]
    assert not (output / "checkpoints").exists()
    with pytest.raises(FileExistsError):
        main([
            "--store", str(store), "--target", str(target),
            "--all-eligible", "--output", str(output),
        ])


def test_cli_resume_flag_can_start_fresh_work(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    output = tmp_path / "controls"
    assert main([
        "--store", str(store), "--target", str(target),
        "--all-eligible", "--output", str(output), "--resume",
        "--sets", "1", "--chains", "1", "--sets-per-chain", "1",
        "--workers", "1", "--seed", "13", "--search-bin-width", "10",
        "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]) == 0
    assert (output / "metadata.json").is_file()
    assert not output.with_name(f".{output.name}.work").exists()


def test_distributed_chains_publish_from_scratch_and_gather(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    chain_dir = tmp_path / "durable-chains"
    common = [
        "--store", str(store), "--target", str(target), "--all-eligible",
        "--chains", "2", "--sets-per-chain", "2", "--seed", "19",
        "--search-bin-width", "10", "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]
    for chain in range(2):
        work = tmp_path / f"scratch-chain-{chain}"
        assert distributed_main([
            "chain", *common, "--chain-index", str(chain),
            "--chain-output", str(chain_dir / f"chain-{chain:03d}.npz"),
            "--work-dir", str(work),
        ]) == 0
        assert not work.exists()
    assert distributed_main([
        "chain", *common, "--chain-index", "0",
        "--chain-output", str(chain_dir / "chain-000.npz"),
        "--work-dir", str(tmp_path / "resume-work"), "--resume",
    ]) == 0

    output = tmp_path / "durable-result"
    gather_work = tmp_path / "scratch-gather"
    assert distributed_main([
        "gather", *common, "--chain-dir", str(chain_dir),
        "--output", str(output), "--work-dir", str(gather_work),
    ]) == 0
    assert not gather_work.exists()
    assert np.load(output / "row_indices.npy").shape == (4, 2)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["distributed_chains"] is True
    assert metadata["distributed_chain_tasks"] == 2
    assert metadata["workers"] == 1
    assert metadata["chains"] == 2


def test_distributed_gather_rejects_row_cdf_mismatch(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    chain_dir = tmp_path / "chains"
    chain_path = chain_dir / "chain-000.npz"
    common = [
        "--store", str(store), "--target", str(target), "--all-eligible",
        "--chains", "1", "--sets-per-chain", "2", "--seed", "23",
        "--search-bin-width", "10", "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]
    assert distributed_main([
        "chain", *common, "--chain-index", "0",
        "--chain-output", str(chain_path),
        "--work-dir", str(tmp_path / "chain-work"),
    ]) == 0
    with np.load(chain_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["row_indices"] = payload["row_indices"].copy()
    used = set(map(int, payload["row_indices"][1]))
    payload["row_indices"][1, 0] = next(
        row for row in range(2, 7) if row not in used
    )
    with chain_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="sample 1 stored CDF does not match"):
        distributed_main([
            "chain", *common, "--chain-index", "0",
            "--chain-output", str(chain_path),
            "--work-dir", str(tmp_path / "resume-work"), "--resume",
        ])
    with pytest.raises(ValueError, match="sample 1 stored CDF does not match"):
        distributed_main([
            "gather", *common, "--chain-dir", str(chain_dir),
            "--output", str(tmp_path / "result"),
            "--work-dir", str(tmp_path / "gather-work"),
        ])


def test_distributed_gather_requires_every_chain_bundle(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    chain_dir = tmp_path / "chains"
    common = [
        "--store", str(store), "--target", str(target), "--all-eligible",
        "--chains", "2", "--sets-per-chain", "1", "--seed", "29",
        "--search-bin-width", "10", "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]
    assert distributed_main([
        "chain", *common, "--chain-index", "0",
        "--chain-output", str(chain_dir / "chain-000.npz"),
        "--work-dir", str(tmp_path / "chain-work"),
    ]) == 0
    with pytest.raises(FileNotFoundError, match="chain-001"):
        distributed_main([
            "gather", *common, "--chain-dir", str(chain_dir),
            "--output", str(tmp_path / "result"),
            "--work-dir", str(tmp_path / "gather-work"),
        ])


def test_distributed_resume_rejects_wrong_derived_seed(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    chain_path = tmp_path / "chains" / "chain-000.npz"
    common = [
        "--store", str(store), "--target", str(target), "--all-eligible",
        "--chains", "1", "--sets-per-chain", "1", "--seed", "31",
        "--search-bin-width", "10", "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]
    assert distributed_main([
        "chain", *common, "--chain-index", "0",
        "--chain-output", str(chain_path),
        "--work-dir", str(tmp_path / "chain-work"),
    ]) == 0
    with np.load(chain_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["seed"] = np.asarray(int(payload["seed"]) + 1, dtype=np.uint64)
    with chain_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="does not match derived seed"):
        distributed_main([
            "chain", *common, "--chain-index", "0",
            "--chain-output", str(chain_path),
            "--work-dir", str(tmp_path / "resume-work"), "--resume",
        ])


def test_distributed_requires_matching_store_content_identity(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_store_content_sha256"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="store contents do not match"):
        distributed_main([
            "chain", "--store", str(store), "--target", str(target),
            "--all-eligible", "--chains", "1", "--sets-per-chain", "1",
            "--chain-index", "0", "--chain-output", str(tmp_path / "chain.npz"),
            "--work-dir", str(tmp_path / "work"),
        ])
    metadata.pop("source_store_content_sha256")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="requires target and store content digests"):
        distributed_main([
            "chain", "--store", str(store), "--target", str(target),
            "--all-eligible", "--chains", "1", "--sets-per-chain", "1",
            "--chain-index", "0", "--chain-output", str(tmp_path / "chain.npz"),
            "--work-dir", str(tmp_path / "work"),
        ])


def test_distributed_gather_rejects_mixed_bundle_identity(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    chain_dir = tmp_path / "chains"
    chain_path = chain_dir / "chain-000.npz"
    common = [
        "--store", str(store), "--target", str(target), "--all-eligible",
        "--chains", "1", "--sets-per-chain", "1", "--seed", "37",
        "--search-bin-width", "10", "--burnin-accepted-sweeps", "0.5",
        "--sample-accepted-sweeps", "0.5",
        "--max-construction-epochs", "3", "--max-chain-proposals", "1000",
        "--cdf-block-rows", "2", "--progress-every", "100",
    ]
    assert distributed_main([
        "chain", *common, "--chain-index", "0",
        "--chain-output", str(chain_path),
        "--work-dir", str(tmp_path / "chain-work"),
    ]) == 0
    with np.load(chain_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    identity = json.loads(str(payload["run_identity"]))
    identity["global_seed"] += 1
    payload["run_identity"] = np.asarray(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ))
    with chain_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="parameters or provenance differ"):
        distributed_main([
            "gather", *common, "--chain-dir", str(chain_dir),
            "--output", str(tmp_path / "result"),
            "--work-dir", str(tmp_path / "gather-work"),
        ])


def test_atomic_chain_publication_refuses_overwrite(tmp_path):
    source = tmp_path / "source.npz"
    destination = tmp_path / "durable" / "chain-000.npz"
    source.write_bytes(b"new-complete-bundle")
    destination.parent.mkdir()
    destination.write_bytes(b"existing-complete-bundle")
    with pytest.raises(FileExistsError):
        _atomic_copy_file(source, destination)
    assert destination.read_bytes() == b"existing-complete-bundle"
    assert not list(destination.parent.glob(".*.publish.*"))
