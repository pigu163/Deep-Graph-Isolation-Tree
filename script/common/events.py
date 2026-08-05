"""Shared canonical event-table loader for link-stream / graph baselines.

Reads the canonical ``data/processed/<dataset>/events.parquet`` (schema in
``script.dataset.linkstream.schema``) plus ``splits.json`` / ``metadata.json``.
This is the single data entry point for all edge-stream and transaction-graph
model adapters so that data stays unchanged across baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from script.common.results import REPO_ROOT
from script.dataset.linkstream import schema as ls_schema
from script.dataset.linkstream.utils import read_json

PROCESSED_ROOT = REPO_ROOT / "data" / "processed"

BASE_COLUMNS = [
    ls_schema.EVENT_ID,
    ls_schema.TIMESTAMP,
    ls_schema.SRC,
    ls_schema.DST,
    ls_schema.LABEL,
    ls_schema.LABEL_AVAILABLE,
]


def load_event_table(dataset: str, extra_columns: tuple[str, ...] = ()) -> dict:
    """Load the canonical event table.

    Returns dict with keys: meta, splits, event_id, timestamp, src, dst,
    label, label_available plus every name in ``extra_columns``.
    """
    proc = PROCESSED_ROOT / dataset
    meta = read_json(proc / ls_schema.METADATA_JSON)
    splits = read_json(proc / ls_schema.SPLITS_JSON)
    df = pl.scan_parquet(proc / ls_schema.EVENTS_PARQUET)
    for col in extra_columns:
        if col not in df.columns:
            raise KeyError(f"{dataset}: column {col!r} not in events.parquet")
    df = df.select([*BASE_COLUMNS, *extra_columns]).collect()
    out = {
        "meta": meta,
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "event_id": df[ls_schema.EVENT_ID].to_numpy().astype(np.int64),
        "timestamp": df[ls_schema.TIMESTAMP].to_numpy().astype(np.float64),
        "src": df[ls_schema.SRC].to_numpy().astype(np.int64),
        "dst": df[ls_schema.DST].to_numpy().astype(np.int64),
        "label": df[ls_schema.LABEL].to_numpy().astype(np.int8),
        "label_available": df[ls_schema.LABEL_AVAILABLE].to_numpy(),
    }
    for col in extra_columns:
        arr = df[col].to_numpy()
        if arr.dtype.kind in ("f", "i", "b"):
            out[col] = arr
        else:
            # categorical strings stay as object arrays (no re-encoding here)
            out[col] = arr
    return out


def event_feature_columns(dataset: str) -> list[str]:
    """Numeric edge-feature columns usable by graph/edge baselines.

    Structural-only datasets return [] (adapters then rely on structure).
    Never includes src/dst/event_id/timestamp/label columns.
    """
    meta = read_json(PROCESSED_ROOT / dataset / ls_schema.METADATA_JSON)
    names = [n for n in meta.get("edge_feature_names", []) if isinstance(n, str)]
    if not names:
        return []
    df = pl.scan_parquet(PROCESSED_ROOT / dataset / ls_schema.EVENTS_PARQUET)
    have = set(df.collect_schema().names())
    return [n for n in names if n in have]


def load_event_features(dataset: str) -> dict[str, np.ndarray]:
    """Load numeric edge-feature columns as float64 arrays (NaN preserved)."""
    cols = event_feature_columns(dataset)
    if not cols:
        return {}
    df = pl.scan_parquet(PROCESSED_ROOT / dataset / ls_schema.EVENTS_PARQUET) \
        .select(cols).collect()
    return {c: df[c].to_numpy().astype(np.float64) for c in cols}
