import json

import numpy as np
import pytest

from distributed_age_match import main as distributed_main
from sample_age_matched_controls import main
from snp_interval_dataset import INTERVAL_SCHEMA_VERSION, pack_status
from swap_control_sampler import (
    SwapConfig,
    derive_chain_seed,
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
    }), encoding="utf-8")
    return path


def _target(path, store_path):
    from snp_interval_dataset import SNPAgeIntervalDataset

    store = SNPAgeIntervalDataset.open(store_path)
    rows = np.array([0, 1], dtype=np.int64)
    ages = np.array([0, 10, 20, 30], dtype=np.float64)
    cdf = store.aggregate_cdf_at(rows, ages + 5, side="left", weighting="interval")
    path.mkdir()
    np.save(path / "te_row_indices.npy", rows)
    np.save(path / "age_bins.npy", ages)
    np.save(path / "target_cdf.npy", cdf)
    np.save(path / "bootstrap_wasserstein.npy", np.array([1_000.0]))
    (path / "metadata.json").write_text(json.dumps({
        "source_store_schema": INTERVAL_SCHEMA_VERSION,
        "source_catalog_sha256": "fixture-catalog",
        "wasserstein_threshold_generations": 1_000.0,
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


def test_run_chain_rejects_candidate_universe_with_no_unselected_row(tmp_path):
    store = _interval_store(tmp_path / "store")
    target = _target(tmp_path / "target", store)
    rows = np.load(target / "te_row_indices.npy")
    cdf = np.load(target / "target_cdf.npy")
    ages = np.load(target / "age_bins.npy")
    config = SwapConfig(sets_per_chain=1, search_bin_width=10)
    with pytest.raises(Exception, match="more rows than the target set"):
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
    assert metadata["software"]["version"] == "0.2.0"
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
        "--chains", "1", "--sets-per-chain", "1", "--seed", "23",
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
    used = set(map(int, payload["row_indices"][0]))
    payload["row_indices"][0, 0] = next(row for row in range(2, 7) if row not in used)
    with chain_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    with pytest.raises(ValueError, match="stored CDF does not match"):
        distributed_main([
            "gather", *common, "--chain-dir", str(chain_dir),
            "--output", str(tmp_path / "result"),
            "--work-dir", str(tmp_path / "gather-work"),
        ])
