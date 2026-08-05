"""Canonical link-stream schema shared by preprocessing, loading and validation.

Every processed dataset is stored as an ordered event table in Parquet with
the columns defined here.  The temporal order is authoritative: `event_id`
is the position after stable sorting by (timestamp, original_row_id).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical event-table columns (data/processed/<dataset>/events.parquet)
# ---------------------------------------------------------------------------

EVENT_ID = "event_id"              # int64, position after stable temporal sort
TIMESTAMP = "timestamp"            # float64 or int64, numeric event time
SRC = "src"                        # int64, compact source node id
DST = "dst"                        # int64, compact destination node id
LABEL = "label"                    # int8, 0 / 1 / -1 (unknown)
LABEL_AVAILABLE = "label_available"  # bool
ORIGINAL_ROW_ID = "original_row_id"  # int64, row id before sorting
EDGE_TYPE = "edge_type"            # optional categorical/string
AMOUNT = "amount"                  # optional float32

# Optional extra raw columns that are dataset-specific (kept verbatim).
EXTRA_COLUMNS = (
    "payment_currency",
    "receiving_currency",
    "is_flagged_fraud",
)

# Feature columns (numeric, float32) get a common prefix so that the runtime
# layer can distinguish them from the required/optional scalar columns.
FEATURE_PREFIX = "f_"

REQUIRED_EVENT_COLUMNS = (
    EVENT_ID,
    TIMESTAMP,
    SRC,
    DST,
    LABEL,
    LABEL_AVAILABLE,
    ORIGINAL_ROW_ID,
)

# ---------------------------------------------------------------------------
# Canonical nodes table (data/processed/<dataset>/nodes.parquet)
# ---------------------------------------------------------------------------

NODE_ID = "node_id"                # int64 compact id
RAW_ID = "raw_id"                  # original identifier (string or int)
NODE_TYPE = "node_type"            # optional entity type
FIRST_EVENT_ID = "first_event_id"  # first observed event in the sorted stream
FIRST_TIMESTAMP = "first_timestamp"

REQUIRED_NODE_COLUMNS = (
    NODE_ID,
    RAW_ID,
    FIRST_EVENT_ID,
    FIRST_TIMESTAMP,
)

# ---------------------------------------------------------------------------
# Directory / file layout
# ---------------------------------------------------------------------------

EVENTS_PARQUET = "events.parquet"
NODES_PARQUET = "nodes.parquet"
METADATA_JSON = "metadata.json"
SPLITS_JSON = "splits.json"
TEMPORAL_INDEX_DIR = "temporal_index"

TEMP_INDEX_FILES = (
    "src_offsets.npy",
    "src_event_ids.npy",
    "dst_offsets.npy",
    "dst_event_ids.npy",
    "node_offsets.npy",
    "node_event_ids.npy",
    "timestamps.npy",
)

PAIR_INDEX_FILES = (
    "pair_keys.npy",
    "pair_offsets.npy",
    "pair_event_ids.npy",
)

# ---------------------------------------------------------------------------
# Supported dataset names
# ---------------------------------------------------------------------------

DATASET_NAMES = ("midas_darpa", "ibm_aml", "paysim", "elliptic_pp")
PROCESSED_DATASET_NAMES = ("midas_darpa", "ibm_hi_small", "ibm_li_small", "paysim", "elliptic_pp")

NODE_MAPPING_RULE = "first appearance after stable temporal sort"


def processed_dir(data_root: str | Path, dataset_name: str) -> Path:
    """Return the processed directory for a dataset."""
    return Path(data_root) / "processed" / dataset_name
