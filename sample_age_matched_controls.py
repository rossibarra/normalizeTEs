#!/usr/bin/env python3
"""Generate 100 posterior-age-matched SNP control sets with swap chains."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from release_provenance import software_provenance
from snp_age_store import open_snp_age_store, store_schema
from swap_control_sampler import (
    ChainOutput,
    SwapConfig,
    aggregate_cdf,
    analysis_points,
    run_chain,
)
from te_age_target import wasserstein_1


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _load_target(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rows = np.load(path / "te_row_indices.npy", allow_pickle=False)
    cdf = np.load(path / "target_cdf.npy", allow_pickle=False)
    ages = np.load(path / "age_bins.npy", allow_pickle=False)
    threshold = metadata.get("wasserstein_threshold_generations")
    if threshold is None:
        distances = np.load(path / "bootstrap_wasserstein.npy", allow_pickle=False)
        threshold = float(np.quantile(distances, 0.50, method="higher"))
    rows = np.asarray(rows, dtype=np.int64)
    cdf = np.asarray(cdf, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("target row indices must be nonempty and unique")
    if cdf.shape != ages.shape or ages.ndim != 1 or ages.size < 2:
        raise ValueError("target CDF and age grid are incompatible")
    if np.any(np.diff(ages) <= 0) or not np.isclose(cdf[-1], 1.0, atol=1e-5):
        raise ValueError("target age grid must increase and CDF must end at one")
    return rows, cdf, ages, float(threshold), metadata


def _worker(arguments: tuple[Any, ...]) -> ChainOutput:
    (store_path, target_rows, target_cdf, age_bins, threshold,
     candidate_path, global_seed, target_digest, chain_index,
     config, checkpoint_path) = arguments
    candidate_rows = (
        None if candidate_path is None
        else np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    )
    return run_chain(
        store_path, target_rows, target_cdf, age_bins, threshold,
        candidate_rows=candidate_rows,
        global_seed=global_seed,
        target_digest=target_digest,
        chain_index=chain_index,
        config=config,
        checkpoint_path=checkpoint_path,
        progress=lambda message: print(message, flush=True),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _save_chain_result(
    path: Path,
    result: ChainOutput,
    *,
    run_identity: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        payload: dict[str, np.ndarray] = {
            "chain_index": np.asarray(result.chain_index),
            "seed": np.asarray(result.seed, dtype=np.uint64),
            "row_indices": result.row_indices,
            "cdfs": result.cdfs,
            "wasserstein": result.wasserstein,
            "diagnostics": np.asarray(json.dumps(result.diagnostics)),
            "construction": np.asarray(json.dumps(result.construction)),
        }
        if run_identity is not None:
            payload.update({
                "bundle_schema_version": np.asarray(
                    "swap-age-matched-chain-v1"
                ),
                "run_identity": np.asarray(json.dumps(
                    run_identity, sort_keys=True, separators=(",", ":")
                )),
            })
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _load_chain_result(path: Path) -> ChainOutput:
    with np.load(path, allow_pickle=False) as archive:
        return ChainOutput(
            chain_index=int(archive["chain_index"]),
            seed=int(archive["seed"]),
            row_indices=np.asarray(archive["row_indices"], dtype=np.int64),
            cdfs=np.asarray(archive["cdfs"], dtype=np.float64),
            wasserstein=np.asarray(archive["wasserstein"], dtype=np.float64),
            diagnostics=json.loads(str(archive["diagnostics"])),
            construction=json.loads(str(archive["construction"])),
        )


def _load_chain_bundle(path: Path) -> tuple[ChainOutput, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        if (
            "bundle_schema_version" not in archive
            or str(archive["bundle_schema_version"]) != "swap-age-matched-chain-v1"
            or "run_identity" not in archive
        ):
            raise ValueError(f"invalid or incomplete chain bundle: {path}")
        identity = json.loads(str(archive["run_identity"]))
    return _load_chain_result(path), identity


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy one completed file and expose it with a same-filesystem rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.publish.{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"publication staging path already exists: {temporary}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_directory(source: Path, destination: Path) -> None:
    """Copy a complete scratch directory and atomically publish it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.publish.{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"publication staging path already exists: {temporary}")
    try:
        shutil.copytree(source, temporary)
        metadata = json.loads(
            (temporary / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("complete") is not True:
            raise RuntimeError("scratch result is not marked complete")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    shutil.rmtree(source)


def _write_results(staging: Path, outputs: list[ChainOutput], store: object,
                   target_cdf: np.ndarray, age_bins: np.ndarray,
                   threshold: float, metadata: dict[str, Any]) -> None:
    ordered = sorted(outputs, key=lambda result: result.chain_index)
    rows = np.concatenate([result.row_indices for result in ordered], axis=0)
    cdfs = np.concatenate([result.cdfs for result in ordered], axis=0)
    all_distances = np.concatenate([result.wasserstein for result in ordered])
    chain_index = np.concatenate([
        np.full(result.row_indices.shape[0], result.chain_index, dtype=np.uint16)
        for result in ordered
    ])
    sample_index = np.concatenate([
        np.arange(result.row_indices.shape[0], dtype=np.uint16)
        for result in ordered
    ])
    if np.any(all_distances > threshold):
        raise RuntimeError("one or more saved sets exceed the exact threshold")
    for row in rows:
        if np.unique(row).size != row.size:
            raise RuntimeError("a saved set contains duplicate controls")
    flat_chromosomes, flat_positions = store.rows_to_native(rows.ravel())
    labels, codes = np.unique(flat_chromosomes, return_inverse=True)
    shape = rows.shape
    np.save(staging / "row_indices.npy", rows, allow_pickle=False)
    np.save(staging / "positions.npy", flat_positions.reshape(shape), allow_pickle=False)
    np.save(staging / "chromosome_codes.npy",
            codes.astype(np.uint16).reshape(shape), allow_pickle=False)
    np.save(staging / "chromosome_labels.npy", labels, allow_pickle=False)
    np.save(staging / "cdfs.npy", cdfs, allow_pickle=False)
    np.save(staging / "wasserstein.npy", all_distances, allow_pickle=False)
    np.save(staging / "chain_index.npy", chain_index, allow_pickle=False)
    np.save(staging / "sample_index.npy", sample_index, allow_pickle=False)
    np.save(staging / "target_cdf.npy", target_cdf, allow_pickle=False)
    np.save(staging / "age_bins.npy", age_bins, allow_pickle=False)
    diagnostic_rows: list[dict[str, Any]] = []
    for result in ordered:
        for record in result.diagnostics:
            diagnostic_rows.append({"chain_index": result.chain_index, **record})
    fields = sorted({key for record in diagnostic_rows for key in record})
    with (staging / "diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    unique_rows, counts = np.unique(rows, return_counts=True)
    np.save(staging / "reuse_row_indices.npy", unique_rows, allow_pickle=False)
    np.save(staging / "reuse_counts.npy", counts.astype(np.uint16), allow_pickle=False)
    chain_diversity: list[dict[str, Any]] = []
    for result in ordered:
        adjacent_replacement = []
        for previous, current in zip(result.row_indices[:-1], result.row_indices[1:]):
            shared = np.intersect1d(previous, current, assume_unique=True).size
            adjacent_replacement.append(1.0 - shared / previous.size)
        chain_distances = np.asarray(result.wasserstein, dtype=np.float64)
        if (
            chain_distances.size >= 3
            and np.std(chain_distances[:-1])
            and np.std(chain_distances[1:])
        ):
            w1_lag1 = float(np.corrcoef(
                chain_distances[:-1], chain_distances[1:]
            )[0, 1])
        else:
            w1_lag1 = None
        mean_replacement = (
            float(np.mean(adjacent_replacement)) if adjacent_replacement else None
        )
        membership_overlap = (
            None if mean_replacement is None else 1.0 - mean_replacement
        )
        membership_ess = (
            None
            if membership_overlap is None
            else result.row_indices.shape[0]
            * (1.0 - membership_overlap) / (1.0 + membership_overlap)
        )
        chain_diversity.append({
            "chain_index": result.chain_index,
            "mean_adjacent_replacement_fraction": mean_replacement,
            "minimum_adjacent_replacement_fraction": (
                float(np.min(adjacent_replacement))
                if adjacent_replacement else None
            ),
            "wasserstein_lag1_autocorrelation": w1_lag1,
            "membership_overlap_ar1_ess_heuristic": membership_ess,
        })

    payload = dict(metadata)
    payload.update({
        "schema_version": "swap-age-matched-controls-v1",
        "complete": True,
        "sets": int(rows.shape[0]),
        "set_size": int(rows.shape[1]),
        "threshold": threshold,
        "maximum_wasserstein": float(all_distances.max()),
        "chain_seeds": [result.seed for result in ordered],
        "chain_construction": [result.construction for result in ordered],
        "unique_controls_across_sets": int(unique_rows.size),
        "maximum_control_reuse": int(counts.max()),
        "chain_diversity": chain_diversity,
        "membership_overlap_ar1_ess_heuristic_total": float(sum(
            record["membership_overlap_ar1_ess_heuristic"] or 0.0
            for record in chain_diversity
        )),
    })
    _atomic_json(staging / "metadata.json", payload)


def _validate_chain_result(
    result: ChainOutput,
    *,
    chain_index: int,
    sets_per_chain: int,
    store: object,
    target_rows: np.ndarray,
    target_cdf: np.ndarray,
    age_bins: np.ndarray,
    threshold: float,
    candidate_rows: np.ndarray | None,
) -> None:
    if result.chain_index != chain_index:
        raise ValueError(
            f"chain bundle index {result.chain_index} does not match {chain_index}"
        )
    expected_rows = (sets_per_chain, target_rows.size)
    expected_cdfs = (sets_per_chain, age_bins.size)
    if result.row_indices.shape != expected_rows:
        raise ValueError(
            f"chain {chain_index} rows have shape {result.row_indices.shape}, "
            f"expected {expected_rows}"
        )
    if result.cdfs.shape != expected_cdfs:
        raise ValueError(
            f"chain {chain_index} CDFs have shape {result.cdfs.shape}, "
            f"expected {expected_cdfs}"
        )
    if result.wasserstein.shape != (sets_per_chain,):
        raise ValueError(f"chain {chain_index} has an invalid distance vector")

    rows = np.asarray(result.row_indices, dtype=np.int64)
    flat = rows.ravel()
    n_store_rows = np.asarray(store.positions).size
    if np.any(flat < 0) or np.any(flat >= n_store_rows):
        raise ValueError(f"chain {chain_index} contains out-of-range rows")
    if not np.all(np.asarray(store.eligible[flat], dtype=bool)):
        raise ValueError(f"chain {chain_index} contains ineligible rows")
    if np.any(np.isin(flat, target_rows)):
        raise ValueError(f"chain {chain_index} contains target rows")
    if candidate_rows is not None:
        allowed = np.asarray(candidate_rows, dtype=np.int64)
        locations = np.searchsorted(allowed, flat)
        found = locations < allowed.size
        found[found] &= allowed[locations[found]] == flat[found]
        if not np.all(found):
            raise ValueError(f"chain {chain_index} contains undeclared candidates")
    for selected in rows:
        if np.unique(selected).size != selected.size:
            raise ValueError(f"chain {chain_index} contains duplicate controls")

    recalculated = np.asarray([
        wasserstein_1(cdf, target_cdf, age_bins) for cdf in result.cdfs
    ])
    if not np.allclose(result.wasserstein, recalculated, rtol=1e-10, atol=1e-8):
        raise ValueError(f"chain {chain_index} distances do not match its CDFs")
    if np.any(recalculated > threshold):
        raise ValueError(f"chain {chain_index} exceeds the target threshold")

    recomputed = aggregate_cdf(store, rows[0], analysis_points(age_bins))
    if not np.allclose(result.cdfs[0], recomputed, rtol=1e-9, atol=1e-10):
        raise ValueError(
            f"chain {chain_index} stored CDF does not match its row indices"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-rows", type=Path)
    candidates.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--work-dir", type=Path,
        help=("exact directory for temporary checkpoints and result assembly; "
              "use node-local scratch on HPC"),
    )
    parser.add_argument("--sets", type=int, default=100)
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--sets-per-chain", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-bin-width", type=int, default=20_000)
    parser.add_argument("--burnin-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--sample-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--max-construction-epochs", type=int, default=50)
    parser.add_argument("--max-chain-proposals", type=int, default=10_000_000)
    parser.add_argument("--cdf-block-rows", type=int, default=256)
    parser.add_argument("--exact-check-accepted", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help=("reuse completed chains in --work-dir (or .OUTPUT.work) when "
              "present; otherwise start a fresh run"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sets != args.chains * args.sets_per_chain:
        raise ValueError("--sets must equal --chains times --sets-per-chain")
    if args.workers <= 0 or args.workers > args.chains:
        raise ValueError("workers must be in [1, chains]")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    target_rows, target_cdf, age_bins, threshold, target_meta = _load_target(args.target)
    store = open_snp_age_store(args.store)
    expected_schema = target_meta.get("source_store_schema")
    if expected_schema is not None and expected_schema != store_schema(store):
        raise ValueError("target and interval store schemas do not match")
    expected_catalog = target_meta.get("source_catalog_sha256")
    actual_catalog = getattr(store, "metadata", {}).get("catalog_sha256")
    if expected_catalog is not None and expected_catalog != actual_catalog:
        raise ValueError("target and interval store catalogs do not match")
    candidate_values = None
    if args.candidate_rows is not None:
        raw = np.load(args.candidate_rows, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("candidate row file must contain a 1-D integer array")
        candidate_values = np.sort(np.asarray(raw, dtype=np.int64))
    config = SwapConfig(
        sets_per_chain=args.sets_per_chain,
        search_bin_width=args.search_bin_width,
        burnin_accepted_sweeps=args.burnin_accepted_sweeps,
        sample_accepted_sweeps=args.sample_accepted_sweeps,
        max_construction_epochs=args.max_construction_epochs,
        max_chain_proposals=args.max_chain_proposals,
        cdf_block_rows=args.cdf_block_rows,
        exact_check_accepted=args.exact_check_accepted,
        progress_every=args.progress_every,
    )
    config.validate()
    target_digest = _sha256_arrays(
        target_rows, target_cdf, age_bins, np.asarray([threshold], dtype=np.float64)
    )
    software = software_provenance()
    work = (
        args.work_dir
        if args.work_dir is not None
        else args.output.with_name(f".{args.output.name}.work")
    )
    if work.resolve() == args.output.resolve():
        raise ValueError("--work-dir and --output must be different paths")
    work_preexisting = work.exists()
    if work_preexisting and not args.resume:
        raise FileExistsError(
            f"work directory already exists: {work}; inspect it or pass --resume"
        )
    work.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = work / "checkpoints"
    checkpoints.mkdir(exist_ok=args.resume)
    chain_results = work / "chain-results"
    chain_results.mkdir(exist_ok=args.resume)
    started = time.perf_counter()
    common = (
        str(args.store), target_rows, target_cdf, age_bins, threshold,
        None if args.candidate_rows is None else str(args.candidate_rows),
        args.seed, target_digest,
    )
    candidate_digest = None
    if args.candidate_rows is not None:
        candidate_digest = _sha256_arrays(
            np.load(args.candidate_rows, mmap_mode="r", allow_pickle=False)
        )
    run_identity = {
        "source_store_schema": store_schema(store),
        "source_catalog_sha256": actual_catalog,
        "target_digest": target_digest,
        "candidate_rows_digest": candidate_digest,
        "software": software,
        "global_seed": args.seed,
        "chains": args.chains,
        "sets_per_chain": args.sets_per_chain,
        "config": asdict(config),
    }
    identity_path = work / "run-identity.json"
    if args.resume and work_preexisting:
        if not identity_path.exists():
            raise ValueError(f"resume work is missing {identity_path}")
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != run_identity:
            raise ValueError("resume parameters or provenance do not match work directory")
    else:
        _atomic_json(identity_path, run_identity)
    jobs = [
        (*common, chain, config, str(checkpoints / f"chain-{chain}.npz"))
        for chain in range(args.chains)
        if not (chain_results / f"chain-{chain}.npz").exists()
    ]
    outputs: list[ChainOutput] = []
    for chain in range(args.chains):
        path = chain_results / f"chain-{chain}.npz"
        if path.exists():
            result, identity = _load_chain_bundle(path)
            if identity != run_identity:
                raise ValueError(
                    f"completed chain parameters or provenance differ: {path}"
                )
            outputs.append(result)
    if outputs:
        print(f"Resuming with {len(outputs)} completed chains", flush=True)
    try:
        if args.workers == 1:
            for job in jobs:
                result = _worker(job)
                _save_chain_result(
                    chain_results / f"chain-{result.chain_index}.npz", result,
                    run_identity=run_identity,
                )
                outputs.append(result)
        else:
            try:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.workers
                )
            except (NotImplementedError, PermissionError) as error:
                print(
                    f"Process workers unavailable ({error}); running chains serially",
                    flush=True,
                )
                for job in jobs:
                    result = _worker(job)
                    _save_chain_result(
                        chain_results / f"chain-{result.chain_index}.npz", result,
                        run_identity=run_identity,
                    )
                    outputs.append(result)
            else:
                with executor:
                    futures = [executor.submit(_worker, job) for job in jobs]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        _save_chain_result(
                            chain_results / f"chain-{result.chain_index}.npz", result,
                            run_identity=run_identity,
                        )
                        outputs.append(result)
        expected_chains = list(range(args.chains))
        if sorted(result.chain_index for result in outputs) != expected_chains:
            raise RuntimeError("completed chain indices are missing or duplicated")
        for result in outputs:
            _validate_chain_result(
                result,
                chain_index=result.chain_index,
                sets_per_chain=args.sets_per_chain,
                store=store,
                target_rows=target_rows,
                target_cdf=target_cdf,
                age_bins=age_bins,
                threshold=threshold,
                candidate_rows=candidate_values,
            )
        metadata = {
            "software": software,
            "creation_command": " ".join(sys.argv),
            "source_store": str(args.store.resolve()),
            "source_store_schema": store_schema(store),
            "source_catalog_sha256": actual_catalog,
            "target": str(args.target.resolve()),
            "target_digest": target_digest,
            "target_metadata": target_meta,
            "candidate_rows": (
                "all eligible rows excluding target" if args.candidate_rows is None
                else str(args.candidate_rows.resolve())
            ),
            "global_seed": args.seed,
            "chains": args.chains,
            "sets_per_chain": args.sets_per_chain,
            "workers": args.workers,
            "config": asdict(config),
            "elapsed_seconds": time.perf_counter() - started,
            "numpy_version": np.__version__,
        }
        _write_results(work, outputs, store, target_cdf, age_bins, threshold, metadata)
        if not args.keep_checkpoints:
            shutil.rmtree(checkpoints)
            shutil.rmtree(chain_results)
            identity_path.unlink()
        if work.parent.resolve() == args.output.parent.resolve():
            os.replace(work, args.output)
        else:
            _publish_directory(work, args.output)
    except BaseException:
        print(f"Incomplete work retained at {work}", flush=True)
        raise
    print(f"Wrote {args.sets} matched control sets to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
