import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from phi_sfs import (
    SiteCount,
    accumulate_spectrum,
    hypergeometric_projection,
    main,
    normalized_spectrum,
    phi_sfs,
    project_sites,
)
from sample_age_matched_controls import _sha256_arrays


# ---------------------------------------------------------------- projection


def test_projection_probability_and_point_mass():
    projected = hypergeometric_projection(7, 20)
    assert projected.sum() == pytest.approx(1)
    assert projected[7] == pytest.approx(1)
    assert np.count_nonzero(projected > 1e-12) == 1


def test_projection_known_case_and_validation():
    projected = hypergeometric_projection(1, 21)
    assert projected[0] == pytest.approx(1 / 21)
    assert projected[1] == pytest.approx(20 / 21)
    with pytest.raises(ValueError, match="cannot project"):
        hypergeometric_projection(3, 19)
    with pytest.raises(ValueError, match="0 <= k"):
        hypergeometric_projection(22, 21)


def test_projection_matches_random_subsampling():
    """Cross-check the closed form against the definition it stands for."""
    k, n, draws = 7, 30, 40_000
    rng = np.random.default_rng(0)
    alleles = np.zeros(n, dtype=np.int64)
    alleles[:k] = 1
    chosen = rng.random((draws, n)).argsort(axis=1)[:, :20]
    empirical = np.bincount(alleles[chosen].sum(axis=1), minlength=21) / draws
    assert np.allclose(empirical, hypergeometric_projection(k, n), atol=0.005)


def test_projection_stable_at_large_n():
    projected = hypergeometric_projection(1, 2_000_000)
    assert projected.sum() == pytest.approx(1)
    assert projected[0] == pytest.approx(1 - 1e-5, abs=1e-9)


# --------------------------------------------------------- site projection


def test_project_sites_filters_below_twenty_and_caches_pairs():
    counts = {
        ("chr1", 1): SiteCount(derived=1, callable=19),
        ("chr1", 2): SiteCount(derived=1, callable=20),
        ("chr1", 3): SiteCount(derived=1, callable=21),
        ("chr1", 4): SiteCount(derived=1, callable=21),
    }
    rows, projections, endpoints = project_sites(counts)
    assert ("chr1", 1) not in rows
    assert projections.shape == (2, 19)
    assert rows[("chr1", 3)] == rows[("chr1", 4)]
    assert projections[rows[("chr1", 2)]][0] == pytest.approx(1)
    assert endpoints[rows[("chr1", 2)]] == pytest.approx(0)
    assert projections[rows[("chr1", 3)]][0] == pytest.approx(20 / 21)
    assert endpoints[rows[("chr1", 3)]] == pytest.approx(1 / 21)


def test_sites_are_not_renormalized_after_endpoint_removal():
    counts = {
        ("chr1", 1): SiteCount(derived=1, callable=21),
        ("chr1", 2): SiteCount(derived=10, callable=20),
    }
    rows, projections, endpoints = project_sites(counts)
    coordinates = [("chr1", 1), ("chr1", 2)]
    raw_counts, endpoint, eligible = accumulate_spectrum(
        coordinates, rows, projections, endpoints
    )
    assert eligible == 2
    assert raw_counts.sum() == pytest.approx(20 / 21 + 1)
    assert endpoint == pytest.approx(1 / 21)
    assert raw_counts.sum() + endpoint == pytest.approx(eligible)
    raw, normalized = normalized_spectrum(raw_counts)
    assert normalized.sum() == pytest.approx(1)


def test_accumulation_is_order_invariant_and_counts_repeats():
    counts = {
        ("chr1", 1): SiteCount(derived=3, callable=25),
        ("chr1", 2): SiteCount(derived=9, callable=25),
    }
    rows, projections, endpoints = project_sites(counts)
    forward = accumulate_spectrum([("chr1", 1), ("chr1", 2)], rows, projections, endpoints)
    reverse = accumulate_spectrum([("chr1", 2), ("chr1", 1)], rows, projections, endpoints)
    assert np.allclose(forward[0], reverse[0])
    repeated = accumulate_spectrum(
        [("chr1", 1), ("chr1", 1)], rows, projections, endpoints
    )
    assert repeated[2] == 2
    assert np.allclose(
        repeated[0],
        2 * accumulate_spectrum([("chr1", 1)], rows, projections, endpoints)[0],
    )


def test_zero_retained_mass_fails():
    counts = {("chr1", 1): SiteCount(derived=1, callable=10)}
    rows, projections, endpoints = project_sites(counts)
    raw_counts, _, eligible = accumulate_spectrum(
        [("chr1", 1)], rows, projections, endpoints
    )
    assert eligible == 0
    with pytest.raises(ValueError, match="zero retained mass"):
        normalized_spectrum(raw_counts)


# ------------------------------------------------------------------ the score


def test_phi_identities():
    te = np.zeros(19)
    snp = np.zeros(19)
    te[0], snp[1] = 1, 1
    result = phi_sfs(te, snp)
    assert result.value == pytest.approx(1)
    assert result.positive.sum() == pytest.approx(1)
    assert result.reverse_positive == pytest.approx(result.value)
    assert result.half_l1 == pytest.approx(result.value)
    assert phi_sfs(te, te).value == pytest.approx(0)


def test_phi_is_total_variation_distance_and_symmetric():
    rng = np.random.default_rng(1)
    for _ in range(5):
        te = rng.random(19)
        te /= te.sum()
        snp = rng.random(19)
        snp /= snp.sum()
        value = phi_sfs(te, snp).value
        assert value == pytest.approx(1 - np.minimum(te, snp).sum())
        assert value == pytest.approx(phi_sfs(snp, te).value)
        assert 0 <= value <= 1


def test_phi_rejects_unnormalized_input():
    with pytest.raises(ValueError, match="normalized"):
        phi_sfs(np.full(19, 0.1), np.full(19, 1 / 19))


# -------------------------------------------------------------- the fixtures


def _write_bundle(root: Path, *, positions=None, row_indices=None, target_digest=None):
    """Write a target and matched-control bundle that pass provenance checks."""
    target = root / "target"
    matches = root / "matches"
    target.mkdir()
    matches.mkdir()

    te_rows = np.array([0, 1], dtype=np.int64)
    cdf = np.array([0.5, 1.0], dtype=np.float64)
    ages = np.array([100.0, 200.0], dtype=np.float64)
    threshold = 12.5
    np.save(target / "te_chromosomes.npy", np.array(["chr1", "chr1"]), allow_pickle=False)
    np.save(target / "te_positions.npy", np.array([10, 20]), allow_pickle=False)
    np.save(target / "te_row_indices.npy", te_rows, allow_pickle=False)
    np.save(target / "target_cdf.npy", cdf, allow_pickle=False)
    np.save(target / "age_bins.npy", ages, allow_pickle=False)

    if positions is None:
        positions = np.array([[30, 40], [40, 50]])
    if row_indices is None:
        row_indices = np.array([[2, 3], [3, 4]], dtype=np.int64)
    np.save(matches / "positions.npy", np.asarray(positions), allow_pickle=False)
    np.save(matches / "row_indices.npy", np.asarray(row_indices), allow_pickle=False)
    np.save(matches / "chromosome_codes.npy",
            np.zeros(np.shape(positions), dtype=np.uint16), allow_pickle=False)
    np.save(matches / "chromosome_labels.npy", np.array(["chr1"]), allow_pickle=False)
    np.save(matches / "chain_index.npy", np.array([0, 1]), allow_pickle=False)
    np.save(matches / "sample_index.npy", np.array([3, 4]), allow_pickle=False)

    digest = _sha256_arrays(
        te_rows, cdf, ages, np.asarray([threshold], dtype=np.float64)
    )
    (target / "metadata.json").write_text(json.dumps({
        "source_store_content_sha256": "store",
        "source_catalog_sha256": "catalog",
        "wasserstein_threshold_generations": threshold,
    }))
    (matches / "metadata.json").write_text(json.dumps({
        "schema_version": "swap-age-matched-controls-v1",
        "source_store_content_sha256": "store",
        "source_catalog_sha256": "catalog",
        "complete": True,
        "target_digest": target_digest if target_digest is not None else digest,
    }))
    return target, matches


def _record(position: int, derived: int, *, callable_count: int = 20, info: str = "."):
    """One biallelic haploid record with an exact derived count."""
    calls = (
        ["1"] * derived
        + ["0"] * (callable_count - derived)
        + ["."] * (20 - callable_count)
    )
    return f"chr1\t{position}\t.\tA\tG\t.\tPASS\t{info}\tGT\t" + "\t".join(calls)


def _vcf_text(info: str = "."):
    # Site 50 has only ten callable individuals, so it fails the n >= 20 filter.
    records = [
        _record(10, 4, info=info),
        _record(20, 8, info=info),
        _record(30, 4, info=info),
        _record(40, 12, info=info),
        _record(50, 5, callable_count=10, info=info),
    ]
    header = (
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"s{index}" for index in range(20))
        + "\n"
    )
    return header + "\n".join(records) + "\n"


def _run(target, matches, vcf, output, *extra):
    return main([
        "--target", str(target), "--matches", str(matches),
        "--vcf", str(vcf), "--output", str(output), *extra,
    ])


# ------------------------------------------------------------- end to end


def test_end_to_end_matches_hand_calculation(tmp_path):
    """Every site has n = 20, so each projects to a point mass at its own k.

    The TE set is k = 4 and k = 8, so t is 0.5 at bins 4 and 8. Replicate 0 is
    k = 4 and k = 12, so it shares only the bin-4 mass and Phi is 0.5.
    Replicate 1 keeps only k = 12 once site 50 is dropped, so it shares nothing
    with the target and Phi is 1.
    """
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    output = tmp_path / "phi"
    vcf.write_text(_vcf_text())
    assert _run(target, matches, vcf, output) == 0

    bins = np.load(output / "bins.npy")
    assert bins.tolist() == list(range(1, 20))

    te = np.load(output / "te_normalized_sfs.npy")
    assert te[3] == pytest.approx(0.5)   # bin 4
    assert te[7] == pytest.approx(0.5)   # bin 8
    assert te.sum() == pytest.approx(1)

    snp = np.load(output / "snp_normalized_sfs.npy")
    assert snp[0][3] == pytest.approx(0.5)
    assert snp[0][11] == pytest.approx(0.5)   # bin 12
    assert snp[1][11] == pytest.approx(1.0)

    phi = np.load(output / "phi_sfs.npy")
    assert phi.tolist() == pytest.approx([0.5, 1.0])

    residual = np.load(output / "residual_te_minus_snp.npy")
    assert residual[0][7] == pytest.approx(0.5)
    assert residual[0][11] == pytest.approx(-0.5)

    assert np.load(output / "chain_index.npy").tolist() == [0, 1]
    assert len((output / "bins.csv").read_text().splitlines()) == 1 + 2 * 19


def test_end_to_end_metadata_and_diagnostics(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    output = tmp_path / "phi"
    vcf.write_text(_vcf_text())
    assert _run(target, matches, vcf, output) == 0

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["complete"] is True
    assert metadata["replicates"] == 2
    assert metadata["target_eligible_sites"] == 2
    assert metadata["target_dropped_n_lt_20"] == 0
    # Every eligible site has n = 20, so no mass reaches bins 0 or 20.
    assert metadata["target_retained_fraction"] == pytest.approx(1)
    assert metadata["target_endpoint_fraction"] == pytest.approx(0)
    # Four eligible sites but only three distinct (k, n) pairs: sites 10 and 30
    # are both k = 4 among n = 20, so they share one cached projection.
    assert metadata["distinct_projections"] == 3
    assert metadata["software"]["name"] == "normalizeTE"
    assert metadata["creation_command"]
    assert metadata["creation_time_utc"]
    assert metadata["numpy_version"]
    assert len(metadata["vcf_sha256"]) == 64
    assert metadata["target_digest"] == json.loads(
        (matches / "metadata.json").read_text()
    )["target_digest"]

    replicates = (output / "replicates.csv").read_text().splitlines()
    header = replicates[0].split(",")
    assert "retained_fraction" in header and "endpoint_fraction" in header
    second = dict(zip(header, replicates[2].split(",")))
    assert int(second["input_sites"]) == 2
    assert int(second["eligible_sites"]) == 1
    assert int(second["dropped_n_lt_20"]) == 1
    # Replicate 1 keeps one of its two sites, so the fractions must divide by
    # the eligible count and not the input count: dividing by input_sites
    # would give 0.5 here.
    assert float(second["retained_mass"]) == pytest.approx(1)
    assert float(second["retained_fraction"]) == pytest.approx(1)


def test_vcf_sha256_matches_a_direct_digest(tmp_path):
    import hashlib

    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    assert _run(target, matches, vcf, tmp_path / "phi") == 0
    metadata = json.loads((tmp_path / "phi" / "metadata.json").read_text())
    assert metadata["vcf_sha256"] == hashlib.sha256(vcf.read_bytes()).hexdigest()


def test_compressed_input_is_read_and_hashed(tmp_path):
    import hashlib

    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf.bgz"
    vcf.write_bytes(gzip.compress(_vcf_text().encode()))
    assert _run(target, matches, vcf, tmp_path / "phi") == 0
    metadata = json.loads((tmp_path / "phi" / "metadata.json").read_text())
    assert metadata["vcf_sha256"] == hashlib.sha256(vcf.read_bytes()).hexdigest()
    assert np.load(tmp_path / "phi" / "phi_sfs.npy").tolist() == pytest.approx([0.5, 1.0])


def test_alt_ancestral_reverses_polarization(tmp_path):
    """With AA=G the ALT allele is ancestral, so k becomes n - alt_count."""
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text(info="AA=G"))
    assert _run(target, matches, vcf, tmp_path / "phi",
                "--ancestral-mode", "info") == 0
    te = np.load(tmp_path / "phi" / "te_normalized_sfs.npy")
    assert te[15] == pytest.approx(0.5)   # bin 16 = 20 - 4
    assert te[11] == pytest.approx(0.5)   # bin 12 = 20 - 8
    assert te[3] == pytest.approx(0)


def test_heterozygous_calls_fail_by_default(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text().replace("\t1\t", "\t0/1\t", 1))
    with pytest.raises(ValueError, match="heterozygous"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_heterozygous_missing_policy_drops_the_individual(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    # Site 10 loses one derived individual: k = 3 among n = 19, so it is dropped.
    vcf.write_text(_vcf_text().replace("\t1\t", "\t0/1\t", 1))
    assert _run(target, matches, vcf, tmp_path / "phi",
                "--heterozygous", "missing") == 0
    metadata = json.loads((tmp_path / "phi" / "metadata.json").read_text())
    assert metadata["target_input_sites"] == 2
    assert metadata["target_eligible_sites"] == 1
    assert metadata["target_dropped_n_lt_20"] == 1


def test_lowercase_ancestral_allele_is_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text(info="AA=g"))
    with pytest.raises(ValueError, match="neither REF"):
        _run(target, matches, vcf, tmp_path / "phi", "--ancestral-mode", "info")


def test_missing_site_is_reported(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text().replace(_record(30, 4) + "\n", ""))
    with pytest.raises(ValueError, match="absent from the VCF"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_existing_output_is_never_overwritten(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    output = tmp_path / "phi"
    output.mkdir()
    with pytest.raises(FileExistsError):
        _run(target, matches, vcf, output)


# ------------------------------------------------------------- bundle checks


def test_matches_built_for_another_target_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path, target_digest="0" * 64)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="built for a different target"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_incomplete_matched_bundle_is_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    metadata = json.loads((matches / "metadata.json").read_text())
    metadata["complete"] = False
    (matches / "metadata.json").write_text(json.dumps(metadata))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="not marked complete"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_absent_store_provenance_is_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    metadata = json.loads((matches / "metadata.json").read_text())
    del metadata["source_catalog_sha256"]
    (matches / "metadata.json").write_text(json.dumps(metadata))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="must both record"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_null_store_provenance_is_accepted(tmp_path):
    """A dense store records neither digest, and no other step rejects that.

    The keys must still be present in both bundles; only their being null is
    tolerated. `target_digest` is unconditional, so the bundles stay bound.
    """
    target, matches = _write_bundle(tmp_path)
    for path in (target / "metadata.json", matches / "metadata.json"):
        metadata = json.loads(path.read_text())
        metadata["source_store_content_sha256"] = None
        metadata["source_catalog_sha256"] = None
        path.write_text(json.dumps(metadata))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    assert _run(target, matches, vcf, tmp_path / "phi") == 0
    assert np.load(tmp_path / "phi" / "phi_sfs.npy").tolist() == pytest.approx([0.5, 1.0])


def test_null_store_provenance_still_requires_a_matching_target(tmp_path):
    target, matches = _write_bundle(tmp_path, target_digest="0" * 64)
    for path in (target / "metadata.json", matches / "metadata.json"):
        metadata = json.loads(path.read_text())
        metadata["source_store_content_sha256"] = None
        metadata["source_catalog_sha256"] = None
        path.write_text(json.dumps(metadata))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="built for a different target"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_mismatched_store_provenance_is_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    metadata = json.loads((matches / "metadata.json").read_text())
    metadata["source_store_content_sha256"] = "other"
    (matches / "metadata.json").write_text(json.dumps(metadata))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="values differ"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_endpoint_fraction_is_reported_when_n_exceeds_twenty(tmp_path):
    """With 21 callable individuals, real mass falls in bins 0 and 20."""
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    header = (
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"s{index}" for index in range(21))
        + "\n"
    )
    records = []
    for position, derived in ((10, 1), (20, 8), (30, 4), (40, 12), (50, 20)):
        calls = ["1"] * derived + ["0"] * (21 - derived)
        records.append(
            f"chr1\t{position}\t.\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(calls)
        )
    vcf.write_text(header + "\n".join(records) + "\n")
    assert _run(target, matches, vcf, tmp_path / "phi") == 0

    metadata = json.loads((tmp_path / "phi" / "metadata.json").read_text())
    assert metadata["target_eligible_sites"] == 2
    assert metadata["target_endpoint_fraction"] > 0
    # Site 10 is k = 1 of n = 21, so it loses exactly 1/21 to bin 0; site 20
    # loses nothing. The two fractions must together account for every site.
    assert metadata["target_endpoint_fraction"] == pytest.approx((1 / 21) / 2)
    assert (
        metadata["target_retained_fraction"] + metadata["target_endpoint_fraction"]
        == pytest.approx(1)
    )


def test_non_integer_row_indices_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    np.save(matches / "row_indices.npy", np.array([[2.9, 3.0], [3.0, 4.0]]),
            allow_pickle=False)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="must be an integer array"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_negative_row_indices_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path, row_indices=np.array([[-1, 3], [3, 4]]))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="non-negative"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_non_integer_positions_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    np.save(matches / "positions.npy", np.array([[30.0, 40.0], [40.0, 50.0]]),
            allow_pickle=False)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="must be an integer array"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_misaligned_row_indices_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path)
    np.save(matches / "row_indices.npy", np.array([[2, 3, 9], [3, 4, 9]]),
            allow_pickle=False)
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="do not align"):
        _run(target, matches, vcf, tmp_path / "phi")


def test_duplicate_controls_within_a_set_are_rejected(tmp_path):
    target, matches = _write_bundle(tmp_path, row_indices=np.array([[2, 2], [3, 4]]))
    vcf = tmp_path / "sites.vcf"
    vcf.write_text(_vcf_text())
    with pytest.raises(ValueError, match="duplicate control rows"):
        _run(target, matches, vcf, tmp_path / "phi")
