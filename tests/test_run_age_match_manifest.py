import json

import numpy as np
import pytest

from release_provenance import PROJECT_VERSION
from run_age_match_manifest import _completed, _manifest, main, parse_args


def _two_target_manifest(path):
    path.write_text(
        "label\tpositions\ttarget\toutput\tseed\n"
        "first\tfirst.pos\ttarget-first\toutput-first\t11\n"
        "second\tsecond.pos\ttarget-second\toutput-second\t12\n",
        encoding="utf-8",
    )
    return path


def test_manifest_runner_defaults_to_bootstrap_median():
    args = parse_args([
        "build-targets", "--manifest", "targets.tsv", "--store", "store"
    ])
    assert args.acceptance_quantile == 0.50
    assert args.chains == 10
    assert args.sets_per_chain == 10


def test_manifest_validates_labels_paths_and_seeds(tmp_path):
    manifest = tmp_path / "targets.tsv"
    manifest.write_text(
        "label\tpositions\ttarget\toutput\tseed\n"
        "in_gene\tin.pos\ttargets/in\tmatches/in\t42\n",
        encoding="utf-8",
    )
    assert _manifest(manifest)[0]["label"] == "in_gene"
    manifest.write_text(
        "label\tpositions\ttarget\toutput\tseed\n"
        "../escape\tin.pos\ttargets/in\tmatches/in\t42\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="labels"):
        _manifest(manifest)


def test_completed_recognizes_targets_and_matches(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    np.save(target / "te_row_indices.npy", np.array([1, 2], dtype=np.int64))
    np.save(target / "target_cdf.npy", np.array([0.5, 1.0]))
    np.save(target / "age_bins.npy", np.array([0.0, 1.0]))
    (target / "metadata.json").write_text(
        json.dumps({
            "wasserstein_threshold_generations": 10,
            "software": {"version": PROJECT_VERSION},
        }),
        encoding="utf-8",
    )
    assert _completed(target, "build-targets")

    match = tmp_path / "match"
    match.mkdir()
    np.save(match / "row_indices.npy", np.array([[3, 4]], dtype=np.int64))
    np.save(match / "wasserstein.npy", np.array([1.0]))
    (match / "diagnostics.csv").write_text("phase\nsaved\n", encoding="utf-8")
    (match / "metadata.json").write_text(
        json.dumps({
            "complete": True,
            "sets": 1,
            "set_size": 2,
            "software": {"version": PROJECT_VERSION},
        }),
        encoding="utf-8",
    )
    assert _completed(match, "sample")
    (match / "diagnostics.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        _completed(match, "sample")


def test_completed_rejects_truncated_arrays(tmp_path):
    match = tmp_path / "match"
    match.mkdir()
    (match / "row_indices.npy").touch()
    np.save(match / "wasserstein.npy", np.array([1.0]))
    (match / "diagnostics.csv").write_text("phase\nsaved\n", encoding="utf-8")
    (match / "metadata.json").write_text(json.dumps({
        "complete": True,
        "sets": 1,
        "set_size": 2,
        "software": {"version": PROJECT_VERSION},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _completed(match, "sample")


def test_sample_chain_requires_exact_target_times_chain_task_count(tmp_path):
    manifest = _two_target_manifest(tmp_path / "targets.tsv")
    with pytest.raises(ValueError, match="requires 20 array tasks"):
        main([
            "sample-chain", "--manifest", str(manifest), "--store", "store",
            "--task-id", "0", "--task-count", "19",
            "--scratch-dir", str(tmp_path / "scratch"),
        ])


def test_sample_chain_flat_task_index_maps_to_target_and_chain(
    tmp_path, monkeypatch,
):
    manifest = _two_target_manifest(tmp_path / "targets.tsv")
    commands = []
    monkeypatch.setattr(
        "run_age_match_manifest.subprocess.run",
        lambda command, check: commands.append(command),
    )
    assert main([
        "sample-chain", "--manifest", str(manifest), "--store", "store",
        "--task-id", "13", "--task-count", "20",
        "--scratch-dir", str(tmp_path / "scratch"),
    ]) == 0
    command = commands[0]
    assert command[command.index("--chain-index") + 1] == "3"
    assert command[command.index("--target") + 1] == "target-second"
    assert command[command.index("--chain-output") + 1].endswith(
        "output-second.chains/chain-003.npz"
    )


def test_gather_requires_one_task_per_target(tmp_path):
    manifest = _two_target_manifest(tmp_path / "targets.tsv")
    with pytest.raises(ValueError, match="one array task per target"):
        main([
            "gather", "--manifest", str(manifest), "--store", "store",
            "--task-id", "0", "--task-count", "3",
            "--scratch-dir", str(tmp_path / "scratch"),
        ])
