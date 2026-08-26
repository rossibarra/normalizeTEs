"""Release and Git provenance recorded in durable workflow outputs."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_NAME = "normalizeTE"
PROJECT_VERSION = "0.5.1"


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


def loaded_source_digest(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Hash every project module currently loaded, for resume identity.

    `software_provenance` pins a commit and a dirty flag. On a dirty checkout
    that is not enough: two different sets of uncommitted edits on the same HEAD
    produce identical provenance, so a long job could resume across an
    implementation change and mix results from two versions of the code -- the
    exact failure the identity exists to prevent.

    Hashing the loaded modules closes that. It walks `sys.modules` rather than
    the directory so the digest covers what this process actually imported, not
    every file that happens to sit beside it: editing an unrelated script does
    not invalidate a resume, and editing an imported one does.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    names = []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        resolved = Path(path).resolve()
        if resolved.parent != root or resolved.suffix != ".py":
            continue
        digest.update(resolved.name.encode("utf-8"))
        digest.update(hashlib.sha256(resolved.read_bytes()).digest())
        names.append(resolved.name)
    return {"modules": names, "sha256": digest.hexdigest()}


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
