import json

import pytest

from release_provenance import PROJECT_VERSION
from run_age_match_manifest import _completed, _manifest, parse_args


def test_manifest_runner_defaults_to_bootstrap_median():
    args = parse_args([
        "build-targets", "--manifest", "targets.tsv", "--store", "store"
    ])
    assert args.acceptance_quantile == 0.50


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
    for name in ("te_row_indices.npy", "target_cdf.npy", "age_bins.npy"):
        (target / name).touch()
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
    for name in ("row_indices.npy", "wasserstein.npy", "diagnostics.csv"):
        (match / name).touch()
    (match / "metadata.json").write_text(
        json.dumps({
            "complete": True,
            "software": {"version": PROJECT_VERSION},
        }),
        encoding="utf-8",
    )
    assert _completed(match, "sample")
    (match / "diagnostics.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        _completed(match, "sample")
