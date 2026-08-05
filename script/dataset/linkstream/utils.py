"""Small shared helpers: seeding, deterministic node mapping, splits, metadata.

Nothing here depends on a specific dataset; dataset-specific parsing lives in
`preprocessing.py`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from . import schema

# ---------------------------------------------------------------------------
# Randomness
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed all random generators used by the research stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Deterministic node mapping
# ---------------------------------------------------------------------------


def build_node_mapping(src_raw: pl.Series, dst_raw: pl.Series) -> pl.DataFrame:
    """Map raw node ids to compact integers by first appearance order.

    The returned frame has columns `raw_id`, `node_id`.  Nodes are ordered
    by their first appearance *in the stable-sorted event stream*, walking
    each event as (src, dst) in stream order — this is the deterministic
    rule required by the task (section 9).  Raw ids may be any single dtype
    (string or integer), but `src_raw` and `dst_raw` must share the same
    dtype.
    """
    n = len(src_raw)
    order_df = pl.concat(
        [
            pl.DataFrame({"raw_id": src_raw, "_e": np.arange(n, dtype=np.int64), "_r": 0}),
            pl.DataFrame({"raw_id": dst_raw, "_e": np.arange(n, dtype=np.int64), "_r": 1}),
        ]
    ).sort(["_e", "_r"])  # interleave: per event, src then dst
    order_df = order_df.with_row_index("_pos")
    # Keep the first occurrence of each raw id, then sort by its first
    # position -> first-appearance order.  `unique` alone does not guarantee
    # output order, hence the explicit sort.
    first = (
        order_df.unique(subset=schema.RAW_ID, keep="first")
        .sort("_pos")
        .drop(["_pos", "_e", "_r"])
    )
    return first.with_row_index(schema.NODE_ID)


def apply_node_mapping(events: pl.DataFrame, node_df: pl.DataFrame) -> pl.DataFrame:
    """Join raw src/dst columns onto the compact node ids.

    `events` must contain `src_raw` and `dst_raw`; `node_df` must contain
    `raw_id` and `node_id`.  Returns `events` with new int64 `src`/`dst`
    columns.
    """
    node_map = node_df.rename({schema.RAW_ID: "raw_id_src", schema.NODE_ID: "src"})
    events = events.join(node_map, left_on="src_raw", right_on="raw_id_src", how="left")
    node_map = node_df.rename({schema.RAW_ID: "raw_id_dst", schema.NODE_ID: "dst"})
    events = events.join(node_map, left_on="dst_raw", right_on="raw_id_dst", how="left")
    if events[schema.SRC].null_count() > 0 or events[schema.DST].null_count() > 0:
        raise ValueError("node mapping failed: some raw ids were not mapped")
    # polars >= 1.x does not duplicate the right-side join keys, so
    # raw_id_src/raw_id_dst may not exist; drop whatever is present.
    drop_cols = [c for c in ("src_raw", "dst_raw", "raw_id_src", "raw_id_dst")
                 if c in events.columns]
    return events.drop(drop_cols)


def build_nodes_table(
    node_df: pl.DataFrame,
    events: pl.DataFrame,
    node_type: pl.Series | None = None,
) -> pl.DataFrame:
    """Assemble the canonical nodes.parquet frame.

    `node_df`: raw_id + node_id (first-appearance order).
    `events`:  sorted event table with event_id, timestamp, src, dst.
    """
    long = pl.concat(
        [
            events.select(
                nid=pl.col(schema.SRC),
                eid=pl.col(schema.EVENT_ID),
                ts=pl.col(schema.TIMESTAMP),
            ),
            events.select(
                nid=pl.col(schema.DST),
                eid=pl.col(schema.EVENT_ID),
                ts=pl.col(schema.TIMESTAMP),
            ),
        ]
    )
    first = long.sort("eid").group_by("nid").agg(
        pl.col("eid").first().alias(schema.FIRST_EVENT_ID),
        pl.col("ts").first().alias(schema.FIRST_TIMESTAMP),
    )
    nodes = node_df.join(
        first, left_on=schema.NODE_ID, right_on="nid", how="left"
    ).select(
        pl.col(schema.NODE_ID),
        pl.col(schema.RAW_ID),
        pl.col(schema.FIRST_EVENT_ID),
        pl.col(schema.FIRST_TIMESTAMP),
    )
    if node_type is not None:
        nodes = nodes.with_columns(node_type.alias(schema.NODE_TYPE))
    return nodes.sort(schema.NODE_ID)


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------


def make_splits(num_events: int, train_ratio: float, val_ratio: float) -> dict[str, list[int]]:
    """Chronological, contiguous, half-open [start, end) event-id ranges."""
    if not (0.0 <= train_ratio <= 1.0) or not (0.0 <= val_ratio <= 1.0):
        raise ValueError("ratios must be in [0, 1]")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1")
    train_end = int(num_events * train_ratio)
    val_end = int(num_events * (train_ratio + val_ratio))
    return {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, num_events],
    }


# ---------------------------------------------------------------------------
# JSON metadata
# ---------------------------------------------------------------------------


def write_json(path: Path, obj: Any, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (use --overwrite to replace it)")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_metadata(
    out_dir: Path,
    *,
    dataset_name: str,
    num_events: int,
    num_nodes: int,
    num_labeled_events: int,
    num_positive_labels: int,
    timestamp_min: float,
    timestamp_max: float,
    edge_feature_names: Sequence[str],
    raw_files: Sequence[str],
    event_label_definition: str,
    notes: Sequence[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Write the canonical metadata.json for a processed dataset."""
    meta = {
        "dataset_name": dataset_name,
        "num_events": int(num_events),
        "num_nodes": int(num_nodes),
        "num_labeled_events": int(num_labeled_events),
        "num_positive_labels": int(num_positive_labels),
        "timestamp_min": float(timestamp_min),
        "timestamp_max": float(timestamp_max),
        "edge_feature_names": list(edge_feature_names),
        "raw_files": list(raw_files),
        "node_mapping_rule": schema.NODE_MAPPING_RULE,
        "event_label_definition": event_label_definition,
        "notes": list(notes or []),
    }
    write_json(out_dir / schema.METADATA_JSON, meta, overwrite=overwrite)
    return meta


# ---------------------------------------------------------------------------
# Git LFS detection
# ---------------------------------------------------------------------------


def is_lfs_pointer(path: Path) -> bool:
    """Return True if `path` is a Git LFS pointer file instead of real data.

    LFS pointer files are small text files starting with
    `version https://git-lfs.github.com/spec/v1`.  Such files must be
    materialized with `git lfs pull` before preprocessing.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def check_no_lfs_pointer(paths: Sequence[Path]) -> None:
    """Raise a clear error if any raw file is still an LFS pointer."""
    for p in paths:
        if is_lfs_pointer(p):
            raise RuntimeError(
                f"{p} is a Git LFS pointer, not real data. "
                f"Run `git lfs pull` (or fetch the dataset) and re-run preprocessing. "
                f"Pointer content: {p.read_text()[:120]!r}"
            )
