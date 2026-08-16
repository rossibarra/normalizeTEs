"""Release and Git provenance recorded in durable workflow outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


PROJECT_NAME = "normalizeTE"
PROJECT_VERSION = "0.3.0"


def _git(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def software_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Return stable release identity plus best-effort Git checkout details."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    commit = _git(root, "rev-parse", "HEAD")
    describe = _git(root, "describe", "--tags", "--always", "--dirty")
    exact_tag = _git(root, "describe", "--tags", "--exact-match", "HEAD")
    return {
        "name": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "git_commit": commit,
        "git_describe": describe,
        "git_tag": exact_tag,
        "git_dirty": bool(describe and describe.endswith("-dirty")),
    }
