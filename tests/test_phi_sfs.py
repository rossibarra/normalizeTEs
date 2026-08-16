import json
from pathlib import Path

import numpy as np
import pytest

from phi_sfs import hypergeometric_projection, main, normalized_spectrum, phi_sfs


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


def test_spectrum_does_not_renormalize_individual_sites():
    first = hypergeometric_projection(1, 21)
    second = hypergeometric_projection(10, 20)
    raw, normalized = normalized_spectrum([first, second])
    assert raw.sum() == pytest.approx(20 / 21 + 1)
    assert normalized.sum() == pytest.approx(1)


def test_phi_identities():
    te = np.zeros(19)
    snp = np.zeros(19)
    te[0], snp[1] = 1, 1
    value, residual, positive = phi_sfs(te, snp)
    assert value == pytest.approx(1)
    assert positive.sum() == pytest.approx(1)
    assert np.maximum(-residual, 0).sum() == pytest.approx(value)
    assert np.abs(residual).sum() / 2 == pytest.approx(value)
    assert phi_sfs(te, te)[0] == pytest.approx(0)


def _write_bundle(root: Path):
    target = root / "target"
    matches = root / "matches"
    target.mkdir()
    matches.mkdir()
    np.save(target / "te_chromosomes.npy", np.array(["chr1", "chr1"]), allow_pickle=False)
    np.save(target / "te_positions.npy", np.array([10, 20]), allow_pickle=False)
    np.save(matches / "positions.npy", np.array([[30, 40], [40, 50]]), allow_pickle=False)
    np.save(matches / "chromosome_codes.npy", np.zeros((2, 2), dtype=np.uint16), allow_pickle=False)
    np.save(matches / "chromosome_labels.npy", np.array(["chr1"]), allow_pickle=False)
    np.save(matches / "chain_index.npy", np.array([0, 1]), allow_pickle=False)
    np.save(matches / "sample_index.npy", np.array([3, 4]), allow_pickle=False)
    metadata = {"source_store_content_sha256": "same", "source_catalog_sha256": "catalog"}
    (target / "metadata.json").write_text(json.dumps(metadata))
    (matches / "metadata.json").write_text(json.dumps(metadata))
    return target, matches


def _write_vcf(path: Path):
    rows = [
        (10, "0", "1"),
        (20, "0", "1"),
        (30, "0", "1"),
        (40, "0/0", "1/1"),
        (50, ".", "1"),
    ]
    text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n"
    for pos, a, b in rows:
        # Repeat each genotype ten times so n is 20, except the missing call at 50.
        samples = [a, b] * 10
        text += f"chr1\t{pos}\t.\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(samples) + "\n"
    path.write_text(text)


def test_end_to_end(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    output = tmp_path / "phi"
    _write_vcf(vcf)
    assert main([
        "--target", str(target), "--matches", str(matches),
        "--vcf", str(vcf), "--output", str(output),
    ]) == 0
    phi = np.load(output / "phi_sfs.npy")
    assert phi.shape == (2,)
    assert np.all((0 <= phi) & (phi <= 1))
    assert np.load(output / "chain_index.npy").tolist() == [0, 1]
    assert len((output / "bins.csv").read_text().splitlines()) == 1 + 2 * 19
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["target_eligible_sites"] == 2
    assert metadata["replicates"] == 2
    assert metadata["complete"] is True


def test_info_ancestral_and_heterozygous_policy(tmp_path):
    target, matches = _write_bundle(tmp_path)
    vcf = tmp_path / "sites.vcf"
    _write_vcf(vcf)
    text = vcf.read_text().replace("\tPASS\t.\tGT", "\tPASS\tAA=G\tGT")
    vcf.write_text(text.replace("0/0", "0/1"))
    with pytest.raises(ValueError, match="heterozygous"):
        main([
            "--target", str(target), "--matches", str(matches), "--vcf", str(vcf),
            "--output", str(tmp_path / "fail"), "--ancestral-mode", "info",
        ])
