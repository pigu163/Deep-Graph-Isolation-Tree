"""Dataset-specific CSV parsing and the shared finalization pipeline.

Each parser reads a raw CSV into a common *raw event frame* and then calls
`finalize_events`, which performs stable temporal sorting, deterministic
node mapping, split generation and Parquet / temporal-index output.

Raw event frame contract (columns required by `finalize_events`):

    timestamp        numeric event time (int or float)
    src_raw          raw source identifier (string or int)
    dst_raw          raw destination identifier (same dtype as src_raw)
    label            int (0 / 1 / -1, -1 = unknown)
    label_available  bool
    original_row_id  row id in the raw file before sorting

Optional raw columns that survive into the canonical table:
    edge_type (string), amount (float), plus any `f_`-prefixed numeric
    feature columns and dataset-specific extra columns (see schema.py).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from . import schema, utils
from .temporal_index import TemporalNeighborIndex

# ---------------------------------------------------------------------------
# Shared finalization
# ---------------------------------------------------------------------------


def finalize_events(
    *,
    out_dir: Path,
    dataset_name: str,
    events: pl.DataFrame,
    raw_files: list[str],
    event_label_definition: str,
    notes: list[str] | None = None,
    node_type: pl.Series | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.0,
    compression: str = "zstd",
    build_pair_index: bool = False,
    overwrite: bool = False,
) -> dict:
    """Turn a raw event frame into the canonical processed dataset layout."""
    out_dir = Path(out_dir)
    missing = [c for c in ("timestamp", "src_raw", "dst_raw", "label",
                           "label_available", "original_row_id")
               if c not in events.columns]
    if missing:
        raise ValueError(f"raw event frame missing columns: {missing}")

    # ---- 1. stable temporal sort: (timestamp, original_row_id) ------------
    # The secondary key makes the order of equal-time events fully
    # deterministic regardless of the sort algorithm's stability.
    events = events.sort(["timestamp", "original_row_id"])
    events = events.with_row_index(schema.EVENT_ID)  # 0 .. N-1

    # ---- 2. deterministic node mapping (first appearance) -----------------
    node_df = utils.build_node_mapping(events["src_raw"], events["dst_raw"])
    events = utils.apply_node_mapping(events, node_df)
    num_nodes = len(node_df)

    # ---- 3. canonical typing ----------------------------------------------
    feature_cols = [c for c in events.columns if c.startswith(schema.FEATURE_PREFIX)]
    for c in feature_cols:
        events = events.with_columns(pl.col(c).cast(pl.Float32, strict=False))

    column_order = [schema.EVENT_ID, schema.TIMESTAMP, schema.SRC, schema.DST,
                    schema.LABEL, schema.LABEL_AVAILABLE, schema.ORIGINAL_ROW_ID]
    if schema.EDGE_TYPE in events.columns:
        column_order.append(schema.EDGE_TYPE)
    if schema.AMOUNT in events.columns:
        column_order.append(schema.AMOUNT)
    for c in schema.EXTRA_COLUMNS:
        if c in events.columns:
            column_order.append(c)
    column_order += feature_cols
    events = events.select(
        [pl.col(schema.EVENT_ID).cast(pl.Int64),
         pl.col(schema.TIMESTAMP).cast(pl.Float64),
         pl.col(schema.SRC).cast(pl.Int64),
         pl.col(schema.DST).cast(pl.Int64),
         pl.col(schema.LABEL).cast(pl.Int8),
         pl.col(schema.LABEL_AVAILABLE).cast(pl.Boolean),
         pl.col(schema.ORIGINAL_ROW_ID).cast(pl.Int64)]
        + [pl.col(c) for c in column_order[7:]]
    )

    # ---- 4. write event table ---------------------------------------------
    if out_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists (use --overwrite to replace it)")
    out_dir.mkdir(parents=True, exist_ok=True)
    events.write_parquet(out_dir / schema.EVENTS_PARQUET, compression=compression)

    # ---- 5. node table ----------------------------------------------------
    nodes = utils.build_nodes_table(node_df, events, node_type=node_type)
    nodes.write_parquet(out_dir / schema.NODES_PARQUET, compression=compression)

    # ---- 6. chronological splits ------------------------------------------
    num_events = len(events)
    splits = utils.make_splits(num_events, train_ratio, val_ratio)
    utils.write_json(out_dir / schema.SPLITS_JSON, splits, overwrite=overwrite)

    # ---- 7. temporal index --------------------------------------------------
    index = TemporalNeighborIndex.build(
        src=events[schema.SRC].to_numpy(),
        dst=events[schema.DST].to_numpy(),
        timestamps=events[schema.TIMESTAMP].to_numpy(),
        num_nodes=num_nodes,
        build_pair_index=build_pair_index,
    )
    index.save(out_dir / schema.TEMPORAL_INDEX_DIR)

    # ---- 8. metadata --------------------------------------------------------
    labeled = events.filter(pl.col(schema.LABEL_AVAILABLE))
    meta = utils.write_metadata(
        out_dir,
        dataset_name=dataset_name,
        num_events=num_events,
        num_nodes=num_nodes,
        num_labeled_events=len(labeled),
        num_positive_labels=int(labeled.filter(pl.col(schema.LABEL) == 1).height),
        timestamp_min=float(events[schema.TIMESTAMP].min()),
        timestamp_max=float(events[schema.TIMESTAMP].max()),
        edge_feature_names=feature_cols,
        raw_files=raw_files,
        event_label_definition=event_label_definition,
        notes=notes or [],
        overwrite=overwrite,
    )
    meta["splits"] = splits
    return meta


# ---------------------------------------------------------------------------
# MIDAS-DARPA
# ---------------------------------------------------------------------------


def preprocess_midas_darpa(
    data_root: Path,
    out_root: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    compression: str,
    build_pair_index: bool,
    overwrite: bool,
) -> dict:
    """darpa_processed.csv (src,dst,time) + darpa_ground_truth.csv (0/1 per row)."""
    data_dir = Path(data_root) / "midas_darpa"
    proc_path = data_dir / "darpa_processed.csv"
    gt_path = data_dir / "darpa_ground_truth.csv"
    utils.check_no_lfs_pointer([proc_path, gt_path])

    events = pl.read_csv(
        proc_path, has_header=False,
        new_columns=["src_raw", "dst_raw", "timestamp"],
    )
    labels = pl.read_csv(gt_path, has_header=False, new_columns=["label"])

    if len(events) != len(labels):
        raise ValueError(
            f"DARPA label/event count mismatch: "
            f"{len(labels)} labels vs {len(events)} events")

    events = events.with_columns(
        pl.col("src_raw").cast(pl.Int64),
        pl.col("dst_raw").cast(pl.Int64),
        pl.col("timestamp").cast(pl.Float64),
        labels["label"].cast(pl.Int8).alias("label"),
        pl.lit(True).alias("label_available"),
    )
    events = events.with_row_index("original_row_id")

    return finalize_events(
        out_dir=Path(out_root) / "midas_darpa",
        dataset_name="midas_darpa",
        events=events,
        raw_files=[str(proc_path), str(gt_path)],
        event_label_definition="ground-truth anomaly label aligned by row",
        notes=["No rich edge features; structure-only event stream."],
        train_ratio=train_ratio, val_ratio=val_ratio,
        compression=compression, build_pair_index=build_pair_index,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# PaySim
# ---------------------------------------------------------------------------


def preprocess_paysim(
    data_root: Path,
    out_root: Path,
    *,
    train_ratio: float,
    val_ratio: float,
    compression: str,
    build_pair_index: bool,
    overwrite: bool,
) -> dict:
    """paysim.csv with the standard PaySim columns (step, type, amount, ...)."""
    path = Path(data_root) / "paysim" / "paysim.csv"
    utils.check_no_lfs_pointer([path])

    df = pl.scan_csv(path).collect()
    expected = {"step", "type", "amount", "nameOrig", "oldbalanceOrg",
                "newbalanceOrig", "nameDest", "oldbalanceDest",
                "newbalanceDest", "isFraud", "isFlaggedFraud"}
    actual = set(df.columns)
    if not expected.issubset(actual):
        raise ValueError(f"PaySim schema mismatch; missing columns: {expected - actual}")

    events = df.select(
        pl.col("step").cast(pl.Float64).alias("timestamp"),
        pl.col("nameOrig").cast(pl.String).alias("src_raw"),
        pl.col("nameDest").cast(pl.String).alias("dst_raw"),
        pl.col("isFraud").cast(pl.Int8).alias("label"),
        pl.lit(True).alias("label_available"),
        pl.col("amount").cast(pl.Float64).alias(schema.AMOUNT),
        pl.col("type").cast(pl.String).alias(schema.EDGE_TYPE),
        pl.col("amount").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "amount"),
        pl.col("oldbalanceOrg").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "oldbalanceOrg"),
        pl.col("newbalanceOrig").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "newbalanceOrig"),
        pl.col("oldbalanceDest").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "oldbalanceDest"),
        pl.col("newbalanceDest").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "newbalanceDest"),
        pl.col("isFlaggedFraud").cast(pl.Int8).alias("is_flagged_fraud"),
    )
    events = events.with_row_index("original_row_id")

    return finalize_events(
        out_dir=Path(out_root) / "paysim",
        dataset_name="paysim",
        events=events,
        raw_files=[str(path)],
        event_label_definition="isFraud (1 = fraudulent transaction)",
        notes=[
            "isFlaggedFraud kept only as an auxiliary column, never as label.",
            "Node prefixes (C.../M...) preserved in raw ids.",
        ],
        train_ratio=train_ratio, val_ratio=val_ratio,
        compression=compression, build_pair_index=build_pair_index,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# IBM AML
# ---------------------------------------------------------------------------


IBM_TRANS_COLUMNS = [
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


def preprocess_ibm_aml(
    data_root: Path,
    out_root: Path,
    *,
    size: str = "SMALL",
    regime: str = "HI",
    train_ratio: float,
    val_ratio: float,
    compression: str,
    build_pair_index: bool,
    overwrite: bool,
) -> dict:
    """IBM AML synthetic transactions (HI/LI x SMALL/MEDIUM/LARGE)."""
    size = size.upper()
    regime = regime.upper()
    if regime not in ("HI", "LI"):
        raise ValueError("--regime must be HI or LI")

    trans_path = (Path(data_root) / "ibm_aml" / size / f"{regime}-{size.capitalize()}_Trans.csv")
    if not trans_path.exists():
        raise FileNotFoundError(
            f"{trans_path} not found. Only SMALL is preprocessed by default; "
            f"MEDIUM/LARGE must be requested explicitly.")
    utils.check_no_lfs_pointer([trans_path])

    # `new_columns` renames the duplicate raw "Account" columns to unique
    # names (From Account / To Account).  Bank/Account ids are forced to
    # String so leading zeros survive CSV type inference.
    df = pl.read_csv(
        trans_path,
        new_columns=IBM_TRANS_COLUMNS,
        schema_overrides={
            "From Bank": pl.String,
            "To Bank": pl.String,
            "From Account": pl.String,
            "To Account": pl.String,
        },
    )

    # Parse "2022/09/01 00:20" (and tolerate a seconds component), then
    # convert to unix-epoch seconds (int64 -> float64).
    parsed = pl.col("Timestamp").str.strptime(
        pl.Datetime("ms"), "%Y/%m/%d %H:%M", strict=False)
    parsed = parsed.fill_null(
        pl.col("Timestamp").str.strptime(
            pl.Datetime("ms"), "%Y/%m/%d %H:%M:%S", strict=False))
    df = df.with_columns(parsed.alias("_ts_dt"))
    bad = df.filter(pl.col("_ts_dt").is_null())
    if bad.height > 0:
        raise ValueError(f"IBM timestamp parse failed on rows like "
                         f"{bad.head(3).to_dicts()}")

    events = df.with_columns(
        (pl.col("From Bank").cast(pl.String) + ":"
         + pl.col("From Account").cast(pl.String)).alias("src_raw"),
        (pl.col("To Bank").cast(pl.String) + ":"
         + pl.col("To Account").cast(pl.String)).alias("dst_raw"),
        pl.col("Is Laundering").cast(pl.Int8).alias("label"),
        pl.lit(True).alias("label_available"),
        pl.col("Amount Paid").cast(pl.Float64).alias(schema.AMOUNT),
        pl.col("Payment Format").cast(pl.String).alias(schema.EDGE_TYPE),
        pl.col("Payment Currency").cast(pl.String).alias("payment_currency"),
        pl.col("Receiving Currency").cast(pl.String).alias("receiving_currency"),
        pl.col("Amount Paid").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "amount_paid"),
        pl.col("Amount Received").cast(pl.Float32).alias(schema.FEATURE_PREFIX + "amount_received"),
        (pl.col("From Bank") == pl.col("To Bank")).cast(pl.Int8)
        .alias(schema.FEATURE_PREFIX + "same_bank"),
        (pl.col("Payment Currency") == pl.col("Receiving Currency")).cast(pl.Int8)
        .alias(schema.FEATURE_PREFIX + "same_currency"),
    ).with_columns(
        pl.col("_ts_dt").dt.epoch("s").cast(pl.Float64).alias("timestamp"),
    ).drop("_ts_dt")
    events = events.with_row_index("original_row_id")

    dataset_name = f"ibm_{regime.lower()}_{size.lower()}"
    return finalize_events(
        out_dir=Path(out_root) / dataset_name,
        dataset_name=dataset_name,
        events=events,
        raw_files=[str(trans_path)],
        event_label_definition="Is Laundering (1 = laundering transaction)",
        notes=[
            "Node IDs combine bank ID and account ID.",
            "Categorical columns (currencies, payment format) preserved verbatim.",
            "Amount Paid used as the canonical amount column.",
        ],
        train_ratio=train_ratio, val_ratio=val_ratio,
        compression=compression, build_pair_index=build_pair_index,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Elliptic++
# ---------------------------------------------------------------------------

_ELLIPTIC_CLASS_TO_LABEL = {
    "1": 1, "illicit": 1, "fraud": 1,
    "2": 0, "licit": 0,
}


def _map_elliptic_class(expr, dtype: pl.DataType | None = None) -> pl.Expr:
    """Map Elliptic class symbols to canonical labels (1/0/-1).

    Numeric classes: 1 -> illicit -> 1, 2 -> licit -> 0, everything else
    (e.g. 3 = unknown) -> -1.  String symbols are lowercased first.
    Accepts a ``pl.Expr`` (production path) or a ``pl.Series`` (tests);
    `dtype` defaults to the input's inferred dtype when not given.
    """
    if isinstance(expr, pl.Series):
        s = expr
        expr = pl.col(s.name)
        if dtype is None:
            dtype = s.dtype
    elif dtype is None:
        raise ValueError("dtype is required when mapping an Expr")
    if dtype.is_numeric():
        return pl.when(expr == 1).then(1).when(expr == 2).then(0).otherwise(-1)
    low = expr.cast(pl.String).str.to_lowercase()
    return (pl.when(low.is_in(["1", "illicit", "fraud"])).then(1)
              .when(low.is_in(["2", "licit"])).then(0)
              .otherwise(-1))


def _find_column(names: list[str], candidates: list[str], fallback: int | None) -> str:
    for c in candidates:
        if c in names:
            return c
    if fallback is not None and fallback < len(names):
        return names[fallback]
    raise ValueError(
        f"could not identify a column among {candidates} in {names}")


def preprocess_elliptic_pp(
    data_root: Path,
    out_root: Path,
    *,
    features_mode: str = "none",
    train_ratio: float,
    val_ratio: float,
    compression: str,
    build_pair_index: bool,
    overwrite: bool,
) -> dict:
    """Elliptic++ Bitcoin transaction graph.

    node = transaction; edge = money flow from upstream (src) to downstream
    (dst) transaction.  Event timestamp and label are those of the DESTINATION
    transaction.  Default `features_mode="none"` keeps a structure-only event
    stream (no feature columns).
    """
    data_dir = Path(data_root) / "elliptic_pp"
    classes_path = data_dir / "txs_classes.csv"
    edgelist_path = data_dir / "txs_edgelist.csv"
    features_path = data_dir / "txs_features.csv"
    utils.check_no_lfs_pointer([classes_path, edgelist_path, features_path])

    classes = pl.read_csv(classes_path)
    edgelist = pl.read_csv(edgelist_path)
    # Project the large feature file: only txId + time_step unless `all`.
    # The features table has NO header (txId, time_step, then 165 features).
    feat_cols = pl.scan_csv(features_path, has_header=False).collect_schema().names()
    txid_col = _find_column(feat_cols, ["txId", "tx_id", "id"], fallback=0)
    time_col = _find_column(feat_cols, ["time_step", "time", "timestamp"], fallback=1)
    feature_scan = pl.scan_csv(features_path, has_header=False)
    if features_mode == "none":
        features = feature_scan.select([txid_col, time_col]).collect()
        selected_features: list[str] = []
    elif features_mode == "all":
        features = feature_scan.select(
            [txid_col, time_col] + [c for c in feat_cols if c not in (txid_col, time_col)]
        ).collect()
        selected_features = [c for c in features.columns if c not in (txid_col, time_col)]
    else:
        raise ValueError("--elliptic-features must be 'none' or 'all'")

    # Classes: locate the transaction-id and class columns.
    cls_names = classes.columns
    cls_txid = _find_column(cls_names, ["txId", "tx_id", "id"], fallback=0)
    cls_label = _find_column(cls_names, ["class", "label", "Class"], fallback=1)

    # Edgelist: directed tx1 -> tx2.
    edge_names = edgelist.columns
    if len(edge_names) < 2:
        raise ValueError(f"elliptic edgelist has < 2 columns: {edge_names}")
    if "txId1" in edge_names and "txId2" in edge_names:
        e_src, e_dst = "txId1", "txId2"
    else:
        e_src, e_dst = edge_names[0], edge_names[1]

    edges = edgelist.select(
        pl.col(e_src).cast(pl.String).alias("src_raw"),
        pl.col(e_dst).cast(pl.String).alias("dst_raw"),
    ).with_row_index("original_row_id")

    # Destination transaction metadata (time step + class), joined by the
    # destination transaction id.  Left join keeps the edgelist order.
    if features_mode == "all":
        dst_info = features.select(
            [pl.col(txid_col).cast(pl.String).alias("_dst_tx"),
             pl.col(time_col).cast(pl.Float64).alias("timestamp")]
            + [pl.col(c) for c in selected_features]
        )
    else:
        dst_info = features.select(
            pl.col(txid_col).cast(pl.String).alias("_dst_tx"),
            pl.col(time_col).cast(pl.Float64).alias("timestamp"),
        )
    classes_info = classes.select(
        pl.col(cls_txid).cast(pl.String).alias("_dst_tx"),
        _map_elliptic_class(pl.col(cls_label), classes.schema[cls_label]).alias("label"),
    )
    dst_info = dst_info.join(classes_info, on="_dst_tx", how="left")

    if features_mode == "all":
        for c in selected_features:
            dst_info = dst_info.with_columns(
                pl.col(c).cast(pl.Float32).alias(schema.FEATURE_PREFIX + c))

    events = edges.join(dst_info, left_on="dst_raw", right_on="_dst_tx", how="left")
    if events["timestamp"].null_count() > 0:
        raise ValueError("some edge destinations are missing from txs_features.csv")
    events = events.with_columns(
        pl.col("label").fill_null(-1),
        (pl.col("label") != -1).alias("label_available"),
    )

    return finalize_events(
        out_dir=Path(out_root) / "elliptic_pp",
        dataset_name="elliptic_pp",
        events=events,
        raw_files=[str(classes_path), str(edgelist_path), str(features_path)],
        event_label_definition=(
            "Event labels are derived from destination transaction node labels."),
        notes=[
            "node = Bitcoin transaction; edge = directed money flow (upstream -> downstream).",
            "Event timestamp is the destination transaction's time step.",
            f"features_mode={features_mode}; 'all' includes raw per-node features "
            "whose temporal validity is not guaranteed.",
        ],
        train_ratio=train_ratio, val_ratio=val_ratio,
        compression=compression, build_pair_index=build_pair_index,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREPROCESSORS = {
    "midas_darpa": preprocess_midas_darpa,
    "ibm_aml": preprocess_ibm_aml,
    "paysim": preprocess_paysim,
    "elliptic_pp": preprocess_elliptic_pp,
}
