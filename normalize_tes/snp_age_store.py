"""Open dense and interval SNP-age stores through one dispatch point."""

from __future__ import annotations

import json
from pathlib import Path

from .snp_age_dataset import SCHEMA_VERSION as DENSE_SCHEMA_VERSION, SNPAgeDataset
from .snp_interval_dataset import (
    INTERVAL_SCHEMA_VERSION,
    SNPAgeIntervalDataset,
)


def store_schema(store: object) -> object:
    """Return the schema identifier recorded by an opened store."""
    metadata = getattr(store, "metadata", {})
    return metadata.get("schema_version") if isinstance(metadata, dict) else None


def is_interval_store(store: object) -> bool:
    return isinstance(store, SNPAgeIntervalDataset) or store_schema(store) == INTERVAL_SCHEMA_VERSION


def open_snp_age_store(path: str | Path, *, deep: bool = False):
    """Open a supported SNP-age store after inspecting its metadata schema."""
    store_path = Path(path)
    try:
        metadata = json.loads((store_path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing SNP-age store metadata: {store_path}") from error
    schema = metadata.get("schema_version")
    if schema == INTERVAL_SCHEMA_VERSION:
        return SNPAgeIntervalDataset.open(store_path, deep=deep)
    if schema == DENSE_SCHEMA_VERSION:
        if deep:
            from .snp_age_dataset import validate_store

            validate_store(store_path, deep=True)
        return SNPAgeDataset.open(store_path)
    raise ValueError(f"unsupported SNP-age store schema: {schema!r}")
