"""Per-individual derived-allele age spectra, checked against hand computation."""

import json

import numpy as np
import pytest

from normalize_tes import individual_age_spectrum as ias
from normalize_tes.build_draw_polarity import SCHEMA_VERSION as POLARITY_SCHEMA
from normalize_tes.snp_interval_dataset import (
    INTERVAL_SCHEMA_VERSION,
    compute_interval_store_content_sha256,
    pack_status,
)


N_DRAWS = 4
# Row 0 pairs young ages with ALT-derived draws and old ages with REF-derived
# draws, which is the correlation the per-draw table exists to preserve.
INTERVALS = [
    # (row, draw, below, above)
    (0, 0, 100.0, 200.0),
    (0, 1, 100.0, 200.0),
    (0, 2, 1000.0, 2000.0),
    (0, 3, 1000.0, 2000.0),
    (1, 0, 10.0, 20.0),
    (1, 1, 10.0, 20.0),
    (1, 2, 10.0, 20.0),
    (1, 3, 10.0, 20.0),
    (2, 0, 40.0, 50.0),   # two mutations in draw 0 make it unusable at row 2
    (2, 0, 60.0, 70.0),
    (2, 1, 50.0, 60.0),
    (2, 2, 50.0, 60.0),
    (2, 3, 50.0, 60.0),
]
# 0=A, 1=C. Every record below is REF=A, ALT=C, so "A ancestral" means ALT derived.
ANCESTRAL = np.array([
    [0, 0, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
], dtype=np.uint8)


def _store(path):
    """Build the fixture store, or reuse it when a test asks for it twice."""
    if path.exists():
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return path, metadata["content_sha256"]
    positions = np.array([5.0, 9.0, 20.0], dtype=np.float64)
    counts = np.zeros(3, dtype=np.uint32)
    for row, *_ in INTERVALS:
        counts[row] += 1
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.uint64)
    arrays = {
        "positions": positions,
        "offsets": offsets,
        "below": np.array([i[2] for i in INTERVALS], dtype=np.float64),
        "above": np.array([i[3] for i in INTERVALS], dtype=np.float64),
        "draw_id": np.array([i[1] for i in INTERVALS], dtype=np.uint8),
        "present_draw_count": np.full(3, N_DRAWS, dtype=np.uint32),
        "missing_draw_count": np.zeros(3, dtype=np.uint32),
        "usable_draw_count": np.full(3, N_DRAWS, dtype=np.uint32),
        "usable_interval_count": counts,
        "skipped_root_count": np.zeros(3, dtype=np.uint32),
        "status": pack_status(np.full((N_DRAWS, 3), 2, dtype=np.uint8)),
    }
    path.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    metadata = {
        "schema_version": INTERVAL_SCHEMA_VERSION,
        "n_snps": 3,
        "n_intervals": int(offsets[-1]),
        "n_posterior_draws": N_DRAWS,
        "endpoint_dtype": "float64",
        "minimum_usable_draws": 1,
        "sequence_length": 100.0,
        "arrays": {name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                   for name, value in arrays.items()},
        "chromosomes": [{"chrom": "chr1", "offset": 0, "length": 100}],
        "inputs": [{"path": str(path / f"draw{d}.tsz"), "draw_id": d}
                   for d in range(N_DRAWS)],
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    digest = compute_interval_store_content_sha256(path, metadata)
    metadata["content_sha256"] = digest
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path, digest


def _polarity(path, digest, table=ANCESTRAL):
    if path.exists():
        return path
    path.mkdir(parents=True)
    np.save(path / "ancestral_base.npy", table)
    np.save(path / "draw_ids.npy", np.arange(N_DRAWS, dtype=np.uint16))
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": POLARITY_SCHEMA,
        "complete": True,
        "bases": ["A", "C", "G", "T"],
        "store_content_sha256": digest,
        "draws": [{"path": f"draw{d}.tsz", "draw_id": d} for d in range(N_DRAWS)],
    }), encoding="utf-8")
    return path


def _ancestral_table(path, digest):
    counts = np.zeros((3, 4), dtype=np.uint16)
    for row in range(3):
        for base in ANCESTRAL[row]:
            counts[row, base] += 1
    path.mkdir(parents=True)
    np.save(path / "ancestral_counts.npy", counts)
    np.save(path / "present_draw_count.npy",
            np.full(3, N_DRAWS, dtype=np.uint16))
    (path / "metadata.json").write_text(json.dumps({
        "schema_version": "ancestral-state-counts-v1",
        "complete": True,
        "bases": ["A", "C", "G", "T"],
        "store_content_sha256": digest,
    }), encoding="utf-8")
    return path


def _vcf(path, genotypes=("1/1", "0/1", "0/0")):
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\thomalt\thet\thomref",
    ]
    for position in (5, 9, 20):
        lines.append("\t".join(
            ["chr1", str(position), ".", "A", "C", ".", "PASS", ".", "GT",
             *genotypes]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(tmp_path, *extra, output="out", vcf=None):
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest)
    vcf = vcf or _vcf(tmp_path / "in.vcf")
    result = tmp_path / output
    assert ias.main([
        "--store", str(store), "--draw-polarity", str(polarity),
        "--vcf", str(vcf), "--output", str(result),
        "--min-usable-draws", "1", "--quiet", *extra,
    ]) == 0
    return result


def _summary(result):
    rows = [line.split("\t")
            for line in (result / "summary.tsv").read_text().splitlines()]
    header = rows[0]
    return {row[0]: dict(zip(header, row)) for row in rows[1:]}


def test_weight_and_mean_age_follow_per_draw_polarity(tmp_path):
    result = _run(tmp_path)
    summary = _summary(result)

    # Row 0: ALT derived in 2 of 4 draws, REF derived in the other 2.
    # Row 1: ALT derived in all 4.  Row 2: draw 0 is multiply mutated, so the
    # denominator is 3, with ALT derived in 2 of them.
    expected_weight = {
        "homalt": 0.5 + 1.0 + 2 / 3,
        "het": 1.0 + 1.0 + 1.0,
        "homref": 0.5 + 0.0 + 1 / 3,
    }
    expected_numerator = {
        "homalt": 0.5 * 150 + 1.0 * 15 + (2 / 3) * 55,
        "het": (0.5 * 150 + 0.5 * 1500) + 1.0 * 15 + 1.0 * 55,
        "homref": 0.5 * 1500 + 0.0 + (1 / 3) * 55,
    }
    for sample, weight in expected_weight.items():
        assert float(summary[sample]["total_weight"]) == pytest.approx(weight)
        assert float(summary[sample]["mean_age"]) == pytest.approx(
            expected_numerator[sample] / weight)
        assert int(summary[sample]["sites_used"]) == 3

    mass = np.load(result / "mass.npy")
    samples = (result / "samples.txt").read_text().split()
    for index, sample in enumerate(samples):
        assert mass[index].sum() == pytest.approx(expected_weight[sample])


def test_marginal_source_loses_the_age_polarity_pairing(tmp_path):
    """The approximate source keeps the weight and averages the wrong ages."""
    # Restrict to the two rows where every draw is usable, so the two sources
    # share a denominator and only the age pairing can differ. Row 2's
    # multiply-mutated draw is dropped here but still counted by the marginal
    # table, which would make the weights differ for a second, separate reason.
    sites = tmp_path / "sites.txt"
    sites.write_text("chr1 5\nchr1 9\n", encoding="utf-8")
    exact = _summary(_run(tmp_path, "--include-positions", str(sites),
                          output="exact"))
    store, digest = _store(tmp_path / "store")
    table = _ancestral_table(tmp_path / "ancestral", digest)
    vcf = _vcf(tmp_path / "in.vcf")
    result = tmp_path / "marginal"
    assert ias.main([
        "--store", str(store), "--ancestral-table", str(table),
        "--vcf", str(vcf), "--output", str(result),
        "--include-positions", str(sites),
        "--min-usable-draws", "1", "--quiet",
    ]) == 0
    marginal = _summary(result)

    # Same total weight: the marginal proportion is the right weight.
    assert float(marginal["homalt"]["total_weight"]) == pytest.approx(
        float(exact["homalt"]["total_weight"]))
    # Different ages: row 0's ALT-derived draws are the young ones, and the
    # marginal source averages them with the old REF-derived draws.
    assert float(exact["homalt"]["mean_age"]) != pytest.approx(
        float(marginal["homalt"]["mean_age"]))


def test_dosage_weighting_doubles_homozygotes_only(tmp_path):
    site = _summary(_run(tmp_path, output="site"))
    dosage = _summary(_run(tmp_path, "--allele-weighting", "dosage",
                           output="dosage"))
    assert float(dosage["homalt"]["total_weight"]) == pytest.approx(
        2 * float(site["homalt"]["total_weight"]))
    assert float(dosage["homref"]["total_weight"]) == pytest.approx(
        2 * float(site["homref"]["total_weight"]))
    assert float(dosage["het"]["total_weight"]) == pytest.approx(
        float(site["het"]["total_weight"]))
    # Doubling both halves of a site cannot move its mean age.
    assert float(dosage["homalt"]["mean_age"]) == pytest.approx(
        float(site["homalt"]["mean_age"]))


def test_missing_genotype_makes_the_individual_uncallable(tmp_path):
    vcf = _vcf(tmp_path / "missing.vcf", genotypes=("1/1", "./1", "0/0"))
    summary = _summary(_run(tmp_path, vcf=vcf))
    assert int(summary["het"]["sites_used"]) == 0
    assert float(summary["het"]["total_weight"]) == 0.0
    assert int(summary["homalt"]["sites_used"]) == 3


def test_min_usable_draws_drops_thin_rows(tmp_path):
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest)
    vcf = _vcf(tmp_path / "in.vcf")
    result = tmp_path / "out"
    # Row 2 has only three usable draws, so a floor of four removes it.
    assert ias.main([
        "--store", str(store), "--draw-polarity", str(polarity),
        "--vcf", str(vcf), "--output", str(result),
        "--min-usable-draws", "4", "--quiet",
    ]) == 0
    summary = _summary(result)
    assert int(summary["homalt"]["sites_used"]) == 2
    assert float(summary["homalt"]["total_weight"]) == pytest.approx(1.5)


def test_third_base_ancestral_call_is_not_evidence_either_way(tmp_path):
    table = ANCESTRAL.copy()
    table[1, :2] = 2  # G ancestral: cannot orient an A/C site
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest, table=table)
    vcf = _vcf(tmp_path / "in.vcf")
    result = tmp_path / "out"
    assert ias.main([
        "--store", str(store), "--draw-polarity", str(polarity),
        "--vcf", str(vcf), "--output", str(result),
        "--min-usable-draws", "1", "--quiet",
    ]) == 0
    # Row 1 keeps weight 1 for the homozygote: the two orienting draws both
    # call ALT derived, so conditioning on them leaves the weight unchanged.
    summary = _summary(result)
    assert float(summary["homalt"]["total_weight"]) == pytest.approx(
        0.5 + 1.0 + 2 / 3)


def test_polarity_table_from_another_store_is_refused(tmp_path):
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", "0" * 64)
    vcf = _vcf(tmp_path / "in.vcf")
    with pytest.raises(SystemExit, match="different interval store"):
        ias.main([
            "--store", str(store), "--draw-polarity", str(polarity),
            "--vcf", str(vcf), "--output", str(tmp_path / "out"), "--quiet",
        ])


def test_existing_output_is_never_overwritten(tmp_path):
    result = _run(tmp_path)
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest)
    with pytest.raises(SystemExit, match="already exists"):
        ias.main([
            "--store", str(store), "--draw-polarity", str(polarity),
            "--vcf", str(tmp_path / "in.vcf"), "--output", str(result),
            "--min-usable-draws", "1", "--quiet",
        ])


def test_merge_sums_parts_and_refuses_a_repeated_vcf(tmp_path):
    first = _run(tmp_path, output="part-a")
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest)
    second_vcf = _vcf(tmp_path / "second.vcf")
    second = tmp_path / "part-b"
    assert ias.main([
        "--store", str(store), "--draw-polarity", str(polarity),
        "--vcf", str(second_vcf), "--output", str(second),
        "--min-usable-draws", "1", "--quiet",
    ]) == 0
    merged = tmp_path / "merged"
    assert ias.main(["--store", str(store), "--output", str(merged),
                     "--merge", str(first), str(second), "--quiet"]) == 0
    combined = _summary(merged)
    single = _summary(first)
    assert float(combined["het"]["total_weight"]) == pytest.approx(
        2 * float(single["het"]["total_weight"]))
    assert float(combined["het"]["mean_age"]) == pytest.approx(
        float(single["het"]["mean_age"]))
    with pytest.raises(SystemExit, match="more than once"):
        ias.main(["--store", str(store), "--output", str(tmp_path / "again"),
                  "--merge", str(first), str(first)])


def test_unknown_chromosome_is_an_error_by_default(tmp_path):
    store, digest = _store(tmp_path / "store")
    polarity = _polarity(tmp_path / "polarity", digest)
    vcf = tmp_path / "other.vcf"
    vcf.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\n"
        "scaffold9\t5\t.\tA\tC\t.\tPASS\t.\tGT\t1/1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="absent from the store"):
        ias.main(["--store", str(store), "--draw-polarity", str(polarity),
                  "--vcf", str(vcf), "--output", str(tmp_path / "out"), "--quiet"])


@pytest.mark.parametrize("below,above", [(0.0, 5.0), (3.0, 3.0), (0.5, 900.0),
                                         (120.0, 130.0), (5.0, 1e9)])
def test_bin_mass_matches_direct_integration(below, above):
    binning = ias.AgeBinning.build("log", 1.0, 1000.0, 12)
    got = ias._bin_mass(np.zeros(1, dtype=np.int64), np.array([below]),
                        np.array([above]), np.array([0.75]), binning, 1)[0]
    edges = binning.edges
    if above <= below:
        expected = np.zeros(binning.n_bins)
        index = min(int(np.searchsorted(edges, below, side="right") - 1),
                    binning.n_bins - 1)
        expected[index] = 0.75
    else:
        overlap = (np.minimum(above, edges[1:]) - np.maximum(below, edges[:-1]))
        expected = 0.75 * np.clip(overlap, 0.0, None) / (above - below)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.sum() == pytest.approx(0.75)


# --------------------------------------------------------- piecewise binning


def test_default_steps_give_100_generation_resolution_to_10000():
    binning = ias.AgeBinning.from_steps(ias.DEFAULT_BIN_STEPS)
    edges = binning.edges
    assert edges[0] == 0.0
    assert edges[-1] == np.inf
    # The first 100 bins are 100 generations wide and reach exactly 10,000.
    np.testing.assert_array_equal(edges[:101], np.arange(0, 10001, 100.0))
    widths = np.diff(edges[:-1])
    assert sorted(set(np.round(widths, 6).tolist())) == [
        100.0, 1000.0, 5000.0, 10000.0, 100000.0]
    # Widths never narrow as age increases.
    assert np.all(np.diff(widths) >= -1e-9)


def test_steps_reject_a_limit_that_does_not_divide():
    with pytest.raises(SystemExit, match="does not divide evenly"):
        ias.AgeBinning.from_steps(["100:10050"])


def test_steps_reject_a_non_increasing_limit():
    with pytest.raises(SystemExit, match="limits must increase"):
        ias.AgeBinning.from_steps(["100:10000", "1000:10000"])


@pytest.mark.parametrize("segment", ["100", "abc:10000", "0:10000", "-100:1000"])
def test_steps_reject_malformed_segments(segment):
    with pytest.raises(SystemExit):
        ias.AgeBinning.from_steps([segment])


def test_step_bins_integrate_an_interval_across_a_width_change():
    """An interval straddling the 10,000-generation width change is exact."""
    binning = ias.AgeBinning.from_steps(["100:10000", "1000:100000"])
    below, above = 9850.0, 12500.0
    got = ias._bin_mass(np.zeros(1, dtype=np.int64), np.array([below]),
                        np.array([above]), np.array([1.0]), binning, 1)[0]
    edges = binning.edges
    overlap = np.minimum(above, edges[1:]) - np.maximum(below, edges[:-1])
    expected = np.clip(overlap, 0.0, None) / (above - below)
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.sum() == pytest.approx(1.0)


def test_default_binning_places_a_young_site_in_its_100_generation_bin(tmp_path):
    result = _run(tmp_path)  # row 1's intervals are all [10, 20]
    edges = np.load(result / "bin_edges.npy")
    mass = np.load(result / "mass.npy")
    samples = (result / "samples.txt").read_text().split()
    assert edges[1] == 100.0 and edges[2] == 200.0
    assert edges.size - 1 == 501
    homalt = mass[samples.index("homalt")]
    # Bin [0, 100) takes row 1 ([10, 20], weight 1) and row 2 ([50, 60], 2/3).
    assert homalt[0] == pytest.approx(1.0 + 2 / 3)
    # Bin [100, 200) takes row 0's ALT-derived draws ([100, 200], weight 1/2),
    # which the 100-generation resolution keeps separate from the younger sites.
    assert homalt[1] == pytest.approx(0.5)
    assert homalt[2] == pytest.approx(0.0)
