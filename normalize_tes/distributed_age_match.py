#!/usr/bin/env python3
"""Run one scratch-local swap chain or gather durable chain bundles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .release_provenance import software_provenance
from .sample_age_matched_controls import (
    _atomic_copy_file,
    _load_chain_bundle,
    _load_target,
    _publish_directory,
    _save_chain_result,
    _sha256_arrays,
    _validate_chain_result,
    _worker,
    _write_results,
)
from .snp_age_store import open_snp_age_store, store_schema
from .swap_control_sampler import ChainOutput, SwapConfig, derive_chain_seed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate-rows", type=Path)
    candidates.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--sets-per-chain", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-bin-width", type=int, default=20_000)
    parser.add_argument("--burnin-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--sample-accepted-sweeps", type=float, default=1.0)
    parser.add_argument("--max-construction-epochs", type=int, default=50)
    parser.add_argument("--max-exact-plateau-epochs", type=int, default=3)
    parser.add_argument("--max-chain-proposals", type=int, default=10_000_000)
    parser.add_argument("--cdf-block-rows", type=int, default=256)
    parser.add_argument("--exact-check-accepted", type=int, default=1_000)
    parser.add_argument("--progress-every", type=int, default=100_000)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)

    chain = commands.add_parser("chain", help="run and publish one chain bundle")
    _add_common_arguments(chain)
    chain.add_argument("--chain-index", required=True, type=int)
    chain.add_argument("--chain-output", required=True, type=Path)
    chain.add_argument("--work-dir", required=True, type=Path)
    chain.add_argument("--resume", action="store_true")

    gather = commands.add_parser(
        "gather", help="validate all chain bundles and publish one result directory"
    )
    _add_common_arguments(gather)
    gather.add_argument("--chain-dir", required=True, type=Path)
    gather.add_argument("--output", required=True, type=Path)
    gather.add_argument("--work-dir", required=True, type=Path)
    return parser.parse_args(argv)


def _context(args: argparse.Namespace) -> dict[str, Any]:
    if args.chains <= 0 or args.sets_per_chain <= 0:
        raise ValueError("--chains and --sets-per-chain must be positive")
    target_rows, target_cdf, age_bins, threshold, target_meta = _load_target(
        args.target
    )
    store = open_snp_age_store(args.store)
    expected_schema = target_meta.get("source_store_schema")
    actual_schema = store_schema(store)
    if expected_schema is not None and expected_schema != actual_schema:
        raise ValueError("target and interval store schemas do not match")
    expected_catalog = target_meta.get("source_catalog_sha256")
    actual_catalog = getattr(store, "metadata", {}).get("catalog_sha256")
    if expected_catalog is not None and expected_catalog != actual_catalog:
        raise ValueError("target and interval store catalogs do not match")
    expected_content = target_meta.get("source_store_content_sha256")
    actual_content = getattr(store, "metadata", {}).get("content_sha256")
    if not expected_content or not actual_content:
        raise ValueError(
            "distributed matching requires target and store content digests; "
            "rebuild the interval store and target with normalizeTE 0.2.1 or later"
        )
    if expected_content != actual_content:
        raise ValueError("target and interval store contents do not match")

    candidate_digest = None
    candidate_values = None
    if args.candidate_rows is not None:
        raw = np.load(args.candidate_rows, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("candidate row file must contain a 1-D integer array")
        candidate_digest = _sha256_arrays(raw)
        candidate_values = np.sort(np.asarray(raw, dtype=np.int64))

    config = SwapConfig(
        sets_per_chain=args.sets_per_chain,
        search_bin_width=args.search_bin_width,
        burnin_accepted_sweeps=args.burnin_accepted_sweeps,
        sample_accepted_sweeps=args.sample_accepted_sweeps,
        max_construction_epochs=args.max_construction_epochs,
        max_exact_plateau_epochs=args.max_exact_plateau_epochs,
        max_chain_proposals=args.max_chain_proposals,
        cdf_block_rows=args.cdf_block_rows,
        exact_check_accepted=args.exact_check_accepted,
        progress_every=args.progress_every,
    )
    config.validate()
    target_digest = _sha256_arrays(
        target_rows, target_cdf, age_bins,
        np.asarray([threshold], dtype=np.float64),
    )
    software = software_provenance()
    identity = {
        "source_store_schema": actual_schema,
        "source_catalog_sha256": actual_catalog,
        "source_store_content_sha256": actual_content,
        "target_digest": target_digest,
        "candidate_rows_digest": candidate_digest,
        "software": software,
        "global_seed": args.seed,
        "chains": args.chains,
        "sets_per_chain": args.sets_per_chain,
        "config": asdict(config),
    }
    return {
        "target_rows": target_rows,
        "target_cdf": target_cdf,
        "age_bins": age_bins,
        "threshold": threshold,
        "target_meta": target_meta,
        "store": store,
        "catalog": actual_catalog,
        "store_content": actual_content,
        "candidate_digest": candidate_digest,
        "candidate_values": candidate_values,
        "config": config,
        "target_digest": target_digest,
        "software": software,
        "identity": identity,
    }


def _run_chain(args: argparse.Namespace) -> int:
    if not 0 <= args.chain_index < args.chains:
        raise ValueError("--chain-index must lie in [0, chains)")
    context = _context(args)
    expected_seed = derive_chain_seed(
        args.seed,
        context["target_digest"],
        args.chain_index,
        context["config"].algorithm_version,
    )
    if args.chain_output.exists():
        if not args.resume:
            raise FileExistsError(f"chain output already exists: {args.chain_output}")
        result, identity = _load_chain_bundle(args.chain_output)
        if identity != context["identity"]:
            raise ValueError("existing chain bundle parameters or provenance differ")
        _validate_chain_result(
            result,
            chain_index=args.chain_index,
            sets_per_chain=args.sets_per_chain,
            store=context["store"],
            target_rows=context["target_rows"],
            target_cdf=context["target_cdf"],
            age_bins=context["age_bins"],
            threshold=context["threshold"],
            candidate_rows=context["candidate_values"],
            expected_seed=expected_seed,
        )
        print(f"Complete chain bundle already exists: {args.chain_output}", flush=True)
        return 0

    if args.work_dir.exists() and not args.resume:
        raise FileExistsError(f"scratch work directory already exists: {args.work_dir}")
    args.work_dir.mkdir(parents=True, exist_ok=args.resume)
    checkpoint = args.work_dir / f"chain-{args.chain_index}.checkpoint.npz"
    local_bundle = args.work_dir / f"chain-{args.chain_index:03d}.npz"
    job = (
        str(args.store),
        context["target_rows"],
        context["target_cdf"],
        context["age_bins"],
        context["threshold"],
        None if args.candidate_rows is None else str(args.candidate_rows),
        args.seed,
        context["target_digest"],
        args.chain_index,
        context["config"],
        str(checkpoint),
    )
    try:
        result = _worker(job)
        _validate_chain_result(
            result,
            chain_index=args.chain_index,
            sets_per_chain=args.sets_per_chain,
            store=context["store"],
            target_rows=context["target_rows"],
            target_cdf=context["target_cdf"],
            age_bins=context["age_bins"],
            threshold=context["threshold"],
            candidate_rows=context["candidate_values"],
            expected_seed=expected_seed,
        )
        _save_chain_result(
            local_bundle, result, run_identity=context["identity"]
        )
        _atomic_copy_file(local_bundle, args.chain_output)
        copied, copied_identity = _load_chain_bundle(args.chain_output)
        if copied_identity != context["identity"]:
            raise RuntimeError("published chain identity does not match local result")
        _validate_chain_result(
            copied,
            chain_index=args.chain_index,
            sets_per_chain=args.sets_per_chain,
            store=context["store"],
            target_rows=context["target_rows"],
            target_cdf=context["target_cdf"],
            age_bins=context["age_bins"],
            threshold=context["threshold"],
            candidate_rows=context["candidate_values"],
            expected_seed=expected_seed,
        )
    except BaseException:
        print(f"Incomplete scratch work retained at {args.work_dir}", flush=True)
        raise
    shutil.rmtree(args.work_dir)
    print(f"Published chain {args.chain_index} to {args.chain_output}", flush=True)
    return 0


def _gather(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    context = _context(args)
    outputs: list[ChainOutput] = []
    bundle_paths: list[Path] = []
    for chain_index in range(args.chains):
        path = args.chain_dir / f"chain-{chain_index:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed chain bundle: {path}")
        result, identity = _load_chain_bundle(path)
        if identity != context["identity"]:
            raise ValueError(f"chain bundle parameters or provenance differ: {path}")
        _validate_chain_result(
            result,
            chain_index=chain_index,
            sets_per_chain=args.sets_per_chain,
            store=context["store"],
            target_rows=context["target_rows"],
            target_cdf=context["target_cdf"],
            age_bins=context["age_bins"],
            threshold=context["threshold"],
            candidate_rows=context["candidate_values"],
            expected_seed=derive_chain_seed(
                args.seed,
                context["target_digest"],
                chain_index,
                context["config"].algorithm_version,
            ),
        )
        outputs.append(result)
        bundle_paths.append(path.resolve())

    if args.work_dir.exists():
        raise FileExistsError(f"scratch work directory already exists: {args.work_dir}")
    args.work_dir.mkdir(parents=True)
    metadata = {
        "software": context["software"],
        "creation_command": " ".join(sys.argv),
        "source_store": str(args.store.resolve()),
        "source_store_schema": store_schema(context["store"]),
        "source_catalog_sha256": context["catalog"],
        "source_store_content_sha256": context["store_content"],
        "target": str(args.target.resolve()),
        "target_digest": context["target_digest"],
        "target_metadata": context["target_meta"],
        "candidate_rows": (
            "all eligible rows excluding target"
            if args.candidate_rows is None
            else str(args.candidate_rows.resolve())
        ),
        "global_seed": args.seed,
        "chains": args.chains,
        "sets_per_chain": args.sets_per_chain,
        "workers": 1,
        "distributed_chain_tasks": args.chains,
        "distributed_chains": True,
        "chain_bundles": [str(path) for path in bundle_paths],
        "algorithm_version": context["config"].algorithm_version,
        "config": asdict(context["config"]),
        "elapsed_seconds": time.perf_counter() - started,
        "numpy_version": np.__version__,
    }
    try:
        _write_results(
            args.work_dir,
            outputs,
            context["store"],
            context["target_cdf"],
            context["age_bins"],
            context["threshold"],
            metadata,
        )
        _publish_directory(args.work_dir, args.output)
    except BaseException:
        print(f"Incomplete scratch work retained at {args.work_dir}", flush=True)
        raise
    print(
        f"Published {args.chains * args.sets_per_chain} matched sets to "
        f"{args.output}",
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "chain":
        return _run_chain(args)
    return _gather(args)


if __name__ == "__main__":
    raise SystemExit(main())
