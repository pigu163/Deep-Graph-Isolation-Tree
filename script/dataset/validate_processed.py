#!/usr/bin/env python
"""Validate a processed dataset against the canonical schema and temporal rules.

Checks (task section 17.3):
  - required files exist and metadata fields are complete
  - event ids are contiguous 0..N-1
  - timestamps are non-decreasing
  - src/dst are within [0, num_nodes)
  - labels are in {-1, 0, 1} and label_available matches label != -1
  - split boundaries are contiguous and cover [0, N)
  - temporal-index offsets are valid and segment event ids are ascending
  - sampled history queries never return the current or future event

Usage:
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/validate_processed.py \
        --dataset paysim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from linkstream import schema, utils
from linkstream.temporal_index import TemporalNeighborIndex

DEFAULT_OUTPUT_ROOT = "/home/zhu/Deep Graph Isolation Tree/data/processed"


def _check(ok: bool, msg: str, errors: list[str], verbose: bool = True) -> None:
    if verbose:
        print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not ok:
        errors.append(msg)


def validate_dataset(out_dir: Path, verbose: bool = True) -> tuple[bool, list[str]]:
    errors: list[str] = []

    # ---- required files -----------------------------------------------------
    required = [schema.EVENTS_PARQUET, schema.NODES_PARQUET,
                schema.METADATA_JSON, schema.SPLITS_JSON]
    for f in required:
        _check((out_dir / f).exists(), f"file exists: {f}", errors, verbose)
    index_dir = out_dir / schema.TEMPORAL_INDEX_DIR
    for f in schema.TEMP_INDEX_FILES:
        _check((index_dir / f).exists(), f"index file exists: {f}", errors, verbose)

    metadata = utils.read_json(out_dir / schema.METADATA_JSON)
    for field in ("dataset_name", "num_events", "num_nodes", "num_labeled_events",
                  "timestamp_min", "timestamp_max", "edge_feature_names",
                  "node_mapping_rule"):
        _check(field in metadata, f"metadata field: {field}", errors, verbose)
    num_events = int(metadata["num_events"])
    num_nodes = int(metadata["num_nodes"])

    # ---- event table ---------------------------------------------------------
    events = pl.read_parquet(out_dir / schema.EVENTS_PARQUET)
    for c in schema.REQUIRED_EVENT_COLUMNS:
        _check(c in events.columns, f"event column: {c}", errors, verbose)
    N = events.height
    _check(N == num_events, f"row count matches metadata ({N} == {num_events})",
           errors, verbose)

    eid = events[schema.EVENT_ID].to_numpy()
    _check(np.array_equal(eid, np.arange(N)), "event ids contiguous 0..N-1",
           errors, verbose)

    ts = events[schema.TIMESTAMP].to_numpy()
    _check(np.all(np.diff(ts) >= 0), "timestamps non-decreasing", errors, verbose)
    _check(float(ts.min()) == metadata["timestamp_min"]
           and float(ts.max()) == metadata["timestamp_max"],
           "timestamp min/max match metadata", errors, verbose)

    src = events[schema.SRC].to_numpy()
    dst = events[schema.DST].to_numpy()
    _check(src.min() >= 0 and src.max() < num_nodes,
           "src ids within [0, num_nodes)", errors, verbose)
    _check(dst.min() >= 0 and dst.max() < num_nodes,
           "dst ids within [0, num_nodes)", errors, verbose)

    label = events[schema.LABEL].to_numpy()
    _check(set(np.unique(label)) <= {-1, 0, 1}, "labels in {-1, 0, 1}",
           errors, verbose)
    label_avail = events[schema.LABEL_AVAILABLE].to_numpy()
    _check(np.array_equal(label_avail, label != -1),
           "label_available == (label != -1)", errors, verbose)

    # ---- node table -----------------------------------------------------------
    nodes = pl.read_parquet(out_dir / schema.NODES_PARQUET)
    for c in schema.REQUIRED_NODE_COLUMNS:
        _check(c in nodes.columns, f"node column: {c}", errors, verbose)
    node_ids = nodes[schema.NODE_ID].to_numpy()
    _check(len(node_ids) == num_nodes
           and np.array_equal(node_ids, np.arange(num_nodes)),
           "node ids contiguous 0..num_nodes-1", errors, verbose)

    # ---- splits -----------------------------------------------------------------
    splits = utils.read_json(out_dir / schema.SPLITS_JSON)
    try:
        tr, te = splits["train"], splits["test"]
        ok = (tr[0] == 0 and tr[1] == te[0] and te[1] == N and 0 <= tr[1] <= N)
    except (KeyError, IndexError, TypeError):
        ok = False
    _check(ok, "splits contiguous and cover [0, N)", errors, verbose)

    # ---- temporal index ----------------------------------------------------------
    index = TemporalNeighborIndex.load(index_dir)
    _check(index.num_events == N and index.num_nodes == num_nodes,
           "index dimensions match events/nodes", errors, verbose)
    _check(np.array_equal(index.timestamps, ts),
           "index timestamps equal event timestamps", errors, verbose)
    for name, offsets, ids, expected_len in (
            ("src", index.src_offsets, index.src_event_ids, N),
            ("dst", index.dst_offsets, index.dst_event_ids, N),
            ("node", index.node_offsets, index.node_event_ids, N + int((src != dst).sum()))):
        _check(len(offsets) == num_nodes + 1
               and np.all(np.diff(offsets) >= 0)
               and offsets[0] == 0 and offsets[-1] == expected_len,
               f"{name} offsets valid", errors, verbose)
        sorted_ok = True
        for o, e in zip(offsets[:-1], offsets[1:]):
            seg = ids[o:e]
            if len(seg) > 1 and not np.all(np.diff(seg) > 0):
                sorted_ok = False
                break
        _check(sorted_ok, f"{name} segment event ids ascending", errors, verbose)
        _check(int(ids.min()) >= 0 and int(ids.max()) < N,
               f"{name} event ids in range", errors, verbose)

    # ---- no future leakage (sampled) -----------------------------------------------
    probe = np.unique(np.linspace(0, max(N - 1, 0), min(N, 1000)).astype(int))
    leaked = 0
    for e in probe:
        for node in (int(src[e]), int(dst[e])):
            hist = index.get_past_incident_events(node, int(e))
            if hist.size and int(hist.max()) >= int(e):
                leaked += 1
    _check(leaked == 0, f"no future leakage on {len(probe)} sampled events",
           errors, verbose)

    ok = not errors
    print(f"== {out_dir.name}: {'OK' if ok else f'{len(errors)} FAILURES'} ==")
    return ok, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=schema.PROCESSED_DATASET_NAMES)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quiet", action="store_true", help="only print summary")
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.dataset
    if not out_dir.exists():
        print(f"error: {out_dir} does not exist; run preprocess.py first")
        sys.exit(1)

    ok, _ = validate_dataset(out_dir, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
