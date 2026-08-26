import json

import numpy as np

from sample_age_matched_syn import main as sample_main
from te_age_target import main as target_main
from test_snp_interval_dataset import _store


def test_interval_target_then_cached_matching_cleans_scratch(tmp_path):
    store = _store(tmp_path / "store")
    te_positions = tmp_path / "te_positions.txt"
    syn_positions = tmp_path / "syn_positions.txt"
    coordinates = "chr1 1\nchr2 1\n"
    te_positions.write_text(coordinates, encoding="utf-8")
    syn_positions.write_text(coordinates, encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = tmp_path / "target"

    assert target_main([
        "--store", str(store),
        "--te-positions", str(te_positions),
        "--output", str(target),
        "--bootstrap-replicates", "10",
        "--bootstrap-batch-size", "3",
        "--bin-width", "10",
        "--cdf-block-rows", "1",
        "--scratch-dir", str(scratch),
        "--seed", "7",
    ]) == 0
    assert list(scratch.iterdir()) == []
    target_metadata = json.loads(
        (target / "metadata.json").read_text(encoding="utf-8")
    )
    assert target_metadata["software"]["version"] == "0.5.1"

    matches = tmp_path / "matches"
    assert sample_main([
        "--store", str(store),
        "--target", str(target),
        "--syn-positions", str(syn_positions),
        "--output", str(matches),
        "--accepted-sets", "1",
        "--max-proposals", "10",
        "--block-snps", "1",
        "--candidate-access", "cache",
        "--candidate-cache-dir", str(scratch),
        "--seed", "8",
    ]) == 0
    assert list(scratch.iterdir()) == []
    metadata = json.loads((matches / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["candidate_access_effective"] == "cache"
    assert metadata["candidate_cache_bytes"] > 0
    assert metadata["candidate_cache_build_seconds"] >= 0
    assert np.load(matches / "syn_positions.npy").shape == (1, 2)
