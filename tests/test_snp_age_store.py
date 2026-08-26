import json

import pytest

from normalize_tes.snp_age_store import open_snp_age_store


def test_dispatch_rejects_missing_and_unknown_metadata(tmp_path):
    with pytest.raises(ValueError, match="metadata"):
        open_snp_age_store(tmp_path / "missing")
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "metadata.json").write_text(
        json.dumps({"schema_version": "future"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported"):
        open_snp_age_store(unknown)
