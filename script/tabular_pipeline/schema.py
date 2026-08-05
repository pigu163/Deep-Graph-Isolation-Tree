"""Tabular pipeline: canonical column schema for the five datasets.

Design rules (task doc Sec. 6-7):

* Every row is one independent sample (an event, or a transaction node for
  Elliptic++).
* `numeric`  columns are passed through directly (float32).
* `categorical` columns use a deterministic one-hot encoder (fit on train
  only, unknown category -> all-zero row).
* `freq` columns are high-cardinality identifiers (accounts / src / dst)
  encoded by their **train-set frequency** (fit on train only, unseen -> 0).
  This is an unsupervised, simple count statistic — no complex graph
  statistics are added.
* `drop` columns are excluded from the feature matrix.
* `sample_id` is the canonical row identifier used in `scores.parquet`
  (event_id for event datasets, txId for Elliptic++ transaction nodes).
* `timestamp` is kept as both a numeric feature (where the paper / task list
  it as a candidate feature, e.g. PaySim `step`) and as the timestamp
  column reported in `scores.parquet`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Canonical event-table column names (data/processed/<ds>/events.parquet)
# ---------------------------------------------------------------------------

EVENT_ID = "event_id"
TIMESTAMP = "timestamp"
SRC = "src"
DST = "dst"
LABEL = "label"
LABEL_AVAILABLE = "label_available"
EDGE_TYPE = "edge_type"
AMOUNT = "amount"

# ---------------------------------------------------------------------------
# Elliptic++ transaction files
# ---------------------------------------------------------------------------

ELLIPTIC_FEATURES_CSV = "txs_features.csv"
ELLIPTIC_CLASSES_CSV = "txs_classes.csv"

# The feature table has NO header: col 1 = txId, col 2 = time_step, then the
# numeric features.  (polars 1.43 names no-header columns column_1, column_2,
# ... — not column_0.)  Elliptic++ ships 165 features; the original Elliptic
# layout is 93 local + 72 aggregated.  We default to the `local` subset and
# provide an `all` mode (see datasets.py / README).
ELLIPTIC_TXID_COL = "column_1"
ELLIPTIC_TIME_COL = "column_2"
ELLIPTIC_LOCAL_COUNT = 93  # first 93 features are local (assumption, see README)
ELLIPTIC_ALL_COUNT = 165

PROCESSED_TABULAR_NAMES = (
    "midas_darpa",
    "ibm_hi_small",
    "ibm_li_small",
    "paysim",
    "elliptic_pp",
)


@dataclass
class DatasetSchema:
    """Per-dataset feature mapping."""

    dataset_name: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    freq: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)
    sample_id_column: str = EVENT_ID
    timestamp_column: str = TIMESTAMP
    label_column: str = LABEL
    label_available_column: str = LABEL_AVAILABLE
    notes: list[str] = field(default_factory=list)

    @property
    def used_raw_columns(self) -> list[str]:
        return self.numeric + self.categorical + self.freq


# ---------------------------------------------------------------------------
# Per-dataset schemas
# ---------------------------------------------------------------------------

# PaySim: transaction = sample.  Label = isFraud.  isFlaggedFraud is dropped
# by default (auxiliary only).  High-cardinality accounts (src/dst) are
# frequency-encoded on the train set only.
PAYSIM_SCHEMA = DatasetSchema(
    dataset_name="paysim",
    numeric=[
        TIMESTAMP,
        "f_amount", "f_oldbalanceOrg", "f_newbalanceOrig",
        "f_oldbalanceDest", "f_newbalanceDest",
    ],
    categorical=[EDGE_TYPE],  # type: CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER
    freq=[SRC, DST],
    drop=["is_flagged_fraud"],
    notes=[
        "sample = transaction; label = isFraud.",
        "isFlaggedFraud dropped (auxiliary only, never ground truth).",
        "nameOrig/nameDest are frequency-encoded via compact src/dst node ids "
        "(train-only counts).",
    ],
)

# IBM AML (HI/LI small share the same schema): transaction = sample.
IBM_SCHEMA = DatasetSchema(
    dataset_name="ibm_aml",  # overwritten per regime below
    numeric=[
        TIMESTAMP,
        "f_amount_paid", "f_amount_received",
        "f_same_bank", "f_same_currency",
    ],
    categorical=[EDGE_TYPE, "payment_currency", "receiving_currency"],
    freq=[SRC, DST],
    notes=[
        "sample = transaction; label = Is Laundering.",
        "bank:account node ids are frequency-encoded (train-only counts).",
        "Amount Paid used as the canonical amount column.",
    ],
)

# MIDAS-DARPA: communication event = sample.  Only structure columns exist.
MIDAS_SCHEMA = DatasetSchema(
    dataset_name="midas_darpa",
    numeric=[TIMESTAMP],
    freq=[SRC, DST],
    notes=[
        "sample = communication event; label = ground-truth anomaly (by row).",
        "No rich edge features; src/dst are frequency-encoded.  No complex "
        "graph statistics are added (weak results are acceptable and fair).",
    ],
)

# Elliptic++: transaction NODE = sample (not the money-flow edge).
# Features + labels are rebuilt from txs_features.csv / txs_classes.csv.
# The numeric feature columns (column_2..) are appended dynamically in
# datasets.py once the actual feature set is known.
ELLIPTIC_SCHEMA = DatasetSchema(
    dataset_name="elliptic_pp",
    numeric=["time_step"],       # time_step used both as feature and timestamp
    freq=[],                     # txId is NOT a feature
    categorical=[],
    sample_id_column="txId",
    timestamp_column="time_step",
    notes=[
        "sample = Bitcoin transaction NODE.",
        "Features from txs_features.csv (no header): col_0 = txId, col_1 = "
        "time_step, col_2.. = features.  local = first 93 features, all = 165.",
        "Label from txs_classes.csv: illicit=1, licit=0, unknown=-1.",
        "txId used only as sample id, never as a numeric feature.",
    ],
)


def get_schema(dataset_name: str) -> DatasetSchema:
    """Return a fresh copy of the schema for a dataset.

    A copy is returned so per-dataset mutation (e.g. Elliptic++ dynamic
    numeric columns) never leaks into the shared module-level objects.
    """
    name = dataset_name
    if name.startswith("ibm_"):
        s = copy.deepcopy(IBM_SCHEMA)
        s.dataset_name = name  # record the concrete regime
        return s
    schemas = {
        "midas_darpa": MIDAS_SCHEMA,
        "paysim": PAYSIM_SCHEMA,
        "elliptic_pp": ELLIPTIC_SCHEMA,
    }
    if name not in schemas:
        raise ValueError(
            f"unknown tabular dataset {name!r}; expected one of "
            f"{PROCESSED_TABULAR_NAMES}")
    return copy.deepcopy(schemas[name])
