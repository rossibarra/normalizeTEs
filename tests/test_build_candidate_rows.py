import json
import os
from pathlib import Path

import numpy as np
import pytest

from build_candidate_rows import _publish


def test_publish_exposes_rows_only_after_report(tmp_path):
    output = tmp_path / "rows.npy"
    report_path = tmp_path / "audit" / "rows.json"
    values = np.array([2, 5, 8], dtype=np.int64)
    report = {"candidate_rows": 3}

    _publish(output, report_path, values, report)

    np.testing.assert_array_equal(np.load(output), values)
    assert json.loads(report_path.read_text()) == report


def test_publish_failure_cannot_leave_rows_without_report(tmp_path, monkeypatch):
    output = tmp_path / "rows.npy"
    report_path = tmp_path / "audit" / "rows.json"
    real_replace = os.replace

    def fail_rows(source, destination):
        if Path(destination) == output:
            raise OSError("simulated final publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_rows)
    with pytest.raises(OSError, match="simulated"):
        _publish(output, report_path, np.array([1]), {"candidate_rows": 1})

    assert not output.exists()
    assert not report_path.exists()
    assert not list(tmp_path.rglob("*.tmp.*"))
