"""Dataset loading / building for the shared tabular pipeline.

Public API (task doc Sec. 6)::

    @dataclass
    class TabularData:
        X: np.ndarray
        y: np.ndarray
        label_available: np.ndarray
        sample_ids: np.ndarray
        feature_names: list[str]
        timestamps: np.ndarray   # extra convenience column

    load_tabular_dataset(dataset_name, split="all",
                         fit_preprocessor_on="train") -> TabularData

Built outputs land in ``data/processed_tabular/<dataset>/``.

Sample identity:

* midas_darpa / paysim / ibm_* : one row per event, row order == event_id
  order (chronological), ``sample_id = event_id``.
* elliptic_pp : one row per **transaction node** (not money-flow edge),
  rebuilt from txs_features.csv / txs_classes.csv, ordered by
  ``(time_step, txId)``, ``sample_id = txId``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .preprocessing import TabularPreprocessor
from .schema import (
    ELLIPTIC_ALL_COUNT,
    ELLIPTIC_CLASSES_CSV,
    ELLIPTIC_FEATURES_CSV,
    ELLIPTIC_LOCAL_COUNT,
    ELLIPTIC_TIME_COL,
    ELLIPTIC_TXID_COL,
    LABEL,
    LABEL_AVAILABLE,
    DatasetSchema,
    get_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "processed_tabular"

SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
_SPLIT_ORDER = ["train", "val", "test"]


@dataclass
class TabularData:
    X: np.ndarray
    y: np.ndarray
    label_available: np.ndarray
    sample_ids: np.ndarray
    feature_names: list[str]
    timestamps: np.ndarray  # float64, chronological step used in scores.parquet


# ---------------------------------------------------------------------------
# raw frames
# ---------------------------------------------------------------------------


def _read_events(dataset_name: str) -> pl.DataFrame:
    """Read the event table (subset of columns) preserving event_id order."""
    path = DATA_ROOT / "processed" / dataset_name / "events.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the link-stream preprocessing first "
            f"(script/dataset/preprocess.py) for dataset {dataset_name!r}.")
    schema = get_schema(dataset_name)
    wanted = list(dict.fromkeys(
        [LABEL, LABEL_AVAILABLE] + schema.used_raw_columns
        + [schema.sample_id_column, schema.timestamp_column]
        + [c for c in ("src", "dst") if c in schema.freq]
    ))
    frame = pl.scan_parquet(path)
    cols = frame.collect_schema().names()
    missing = [c for c in wanted if c not in cols]
    if missing:
        raise ValueError(
            f"[{dataset_name}] events.parquet missing schema columns: {missing} "
            f"(available: {cols})")
    return frame.select([c for c in wanted if c in cols]).collect()


def _map_elliptic_class(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map Elliptic++ class column to (label, label_available).

    illicit / '1' -> 1 ; licit / '2' -> 0 ; unknown / others -> -1.
    """
    out = np.full(len(values), -1, dtype=np.int8)
    available = np.zeros(len(values), dtype=bool)
    for i, v in enumerate(values):
        s = str(v).strip().lower()
        if s in ("1", "illicit", "fraud"):
            out[i] = 1
            available[i] = True
        elif s in ("2", "licit"):
            out[i] = 0
            available[i] = True
    return out, available


def _elliptic_frame(features_mode: str = "local") -> pl.DataFrame:
    """Rebuild transaction-node samples from txs_features.csv/classes."""
    data_dir = DATA_ROOT / "elliptic_pp"
    feat_path = data_dir / ELLIPTIC_FEATURES_CSV
    cls_path = data_dir / ELLIPTIC_CLASSES_CSV
    if not feat_path.exists() or not cls_path.exists():
        raise FileNotFoundError(
            f"elliptic raw files missing: {feat_path} / {cls_path}")

    scan = pl.scan_csv(feat_path, has_header=False)
    col_names = scan.collect_schema().names()  # column_0 .. column_{N-1}
    txid = ELLIPTIC_TXID_COL
    time_col = ELLIPTIC_TIME_COL
    feature_cols = [c for c in col_names if c not in (txid, time_col)]

    if features_mode == "local":
        feature_cols = feature_cols[:ELLIPTIC_LOCAL_COUNT]
    elif features_mode == "all":
        feature_cols = feature_cols[:ELLIPTIC_ALL_COUNT]
    else:
        raise ValueError("elliptic features mode must be 'local' or 'all'")

    feats = scan.select([txid, time_col] + feature_cols).collect()
    feats = feats.rename({txid: "txId", time_col: "time_step"})
    feats = feats.with_columns(pl.col("txId").cast(pl.String).alias("txId"))

    classes = pl.read_csv(cls_path)
    cls_txid = "txId" if "txId" in classes.columns else classes.columns[0]
    cls_class = "class" if "class" in classes.columns else classes.columns[1]
    cdf = classes.select(
        pl.col(cls_txid).cast(pl.String).alias("txId"),
        pl.col(cls_class).cast(pl.String).alias("class_raw"),
    ).to_numpy()
    labels, available = _map_elliptic_class(cdf[:, 1])
    cls_frame = pl.DataFrame({
        "txId": cdf[:, 0].astype(str),
        LABEL: labels,
        LABEL_AVAILABLE: available,
    })

    frame = feats.join(cls_frame, on="txId", how="left")
    frame = frame.with_columns(
        pl.col(LABEL).fill_null(-1),
        pl.col(LABEL_AVAILABLE).fill_null(False),
        pl.col("time_step").cast(pl.Float64).alias("time_step"),
    )
    # chronological ordering by (time_step, txId)
    frame = frame.sort(
        ["time_step", pl.col("txId").cast(pl.Int64, strict=False)],
        nulls_last=True,
    )
    return frame


def _build_raw_frame(dataset_name: str, elliptic_features: str) -> pl.DataFrame:
    schema = get_schema(dataset_name)
    if dataset_name == "elliptic_pp":
        frame = _elliptic_frame(elliptic_features)
    else:
        frame = _read_events(dataset_name)
        # drop the columns the schema does not want (e.g. is_flagged_fraud)
        drop = [c for c in schema.drop if c in frame.columns]
        if drop:
            frame = frame.drop(drop)
    # canonical identity column: "event_id" (event_id or elliptic txId)
    id_col = "txId" if dataset_name == "elliptic_pp" else schema.sample_id_column
    if id_col != "event_id" and "event_id" not in frame.columns:
        frame = frame.rename({id_col: "event_id"})
    return frame


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def _make_splits(n_rows: int) -> dict[str, list[int]]:
    t = int(round(n_rows * SPLIT_RATIOS["train"]))
    v = int(round(n_rows * SPLIT_RATIOS["val"]))
    return {
        "train": [0, t],
        "val": [t, t + v],
        "test": [t + v, n_rows],
    }


# ---------------------------------------------------------------------------
# build / load
# ---------------------------------------------------------------------------


def build_tabular_dataset(
    dataset_name: str,
    out_dir: str | Path | None = None,
    *,
    max_rows: int | None = None,
    elliptic_features: str = "local",
    overwrite: bool = False,
) -> Path:
    """Build (or refresh) `data/processed_tabular/<dataset>/` outputs."""
    schema = get_schema(dataset_name)
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _build_raw_frame(dataset_name, elliptic_features)
    if max_rows is not None:
        raw = raw.head(max_rows)

    if dataset_name == "elliptic_pp":
        # Dynamic schema: numeric = time_step + every feature column present
        # in the rebuilt transaction frame (local or all subset).  The
        # transaction id was renamed to `event_id` in _build_raw_frame and is
        # NOT a feature (sample id only).
        feature_cols = [
            c for c in raw.columns
            if c not in ("txId", "event_id", "time_step", LABEL, LABEL_AVAILABLE)
        ]
        schema.numeric = ["time_step"] + feature_cols
        schema.timestamp_column = "time_step"

    n = raw.height
    if n == 0:
        raise ValueError(f"[{dataset_name}] empty raw frame")

    # label / id / timestamp arrays (row order is the canonical order)
    y = np.asarray(raw[LABEL].to_numpy(), dtype=np.int8)
    label_available = np.asarray(raw[LABEL_AVAILABLE].to_numpy(), dtype=bool)
    sample_ids = np.asarray(raw["event_id"].to_numpy()).astype(np.int64)
    timestamps = np.asarray(raw["time_step"].to_numpy(), dtype=np.float64) \
        if "time_step" in raw.columns else \
        np.asarray(raw[schema.timestamp_column].to_numpy(), dtype=np.float64)

    # chronological split: row order is already chronological
    splits = _make_splits(n)
    train_end = splits["train"][1]
    pre = TabularPreprocessor(schema).fit(raw.head(train_end))
    X = pre.transform(raw)
    feature_names = pre.feature_names

    # ---- write outputs -----------------------------------------------------
    _write_json(out_dir / "splits.json", splits, overwrite)
    pl.DataFrame(
        {name: X[:, i] for i, name in enumerate(feature_names)}
    ).write_parquet(out_dir / "features.parquet")
    np.save(out_dir / "labels.npy", y)
    np.save(out_dir / "label_available.npy", label_available)
    np.save(out_dir / "sample_ids.npy", sample_ids)
    np.save(out_dir / "timestamps.npy", timestamps)

    schema_info = {
        "dataset_name": dataset_name,
        "n_samples": int(n),
        "feature_names": feature_names,
        "n_features": int(X.shape[1]),
        "dropped_columns": list(schema.drop),
        "column_encodings": {
            "numeric": list(schema.numeric),
            "frequency_encoded": [f"{c}_freq" for c in schema.freq],
            "categorical_one_hot": list(schema.categorical),
        },
        "elliptic_features_mode": elliptic_features
        if dataset_name == "elliptic_pp" else None,
        "notes": list(schema.notes),
    }
    _write_json(out_dir / "feature_schema.json", schema_info, overwrite)
    _write_json(out_dir / "preprocessing.json", pre.state_summary(), overwrite)

    metadata = {
        "dataset_name": dataset_name,
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "splits": splits,
        "n_train": splits["train"][1] - splits["train"][0],
        "n_val": splits["val"][1] - splits["val"][0],
        "n_test": splits["test"][1] - splits["test"][0],
        "n_positive": int((y == 1).sum()),
        "n_unknown_label": int((y == -1).sum()),
        "sample_id_column": schema.sample_id_column,
        "sample_order": "event_id (chronological)" if dataset_name != "elliptic_pp"
        else "(time_step, txId) chronological",
        "elliptic_features_mode": elliptic_features
        if dataset_name == "elliptic_pp" else None,
        "notes": list(schema.notes),
    }
    _write_json(out_dir / "metadata.json", metadata, overwrite)
    return out_dir


def ensure_tabular_dataset(dataset_name: str, **kwargs) -> Path:
    """Build the tabular dataset if its outputs are missing."""
    out_dir = Path(kwargs.pop("out_dir", None)) if "out_dir" in kwargs \
        else DEFAULT_OUTPUT_ROOT / dataset_name
    if not (out_dir / "features.parquet").exists():
        build_tabular_dataset(dataset_name, out_dir, **kwargs)
    return out_dir


def _load_np(out_dir: Path, name: str) -> np.ndarray:
    return np.load(out_dir / name, allow_pickle=True)


def load_tabular_dataset(
    dataset_name: str,
    split: str = "all",
    fit_preprocessor_on: str = "train",
) -> TabularData:
    """Load prebuilt tabular data (builds on first use).

    `split` filters rows by the chronological split ("all" returns every row).
    `fit_preprocessor_on` is accepted for API compatibility; the preprocessor
    is always fitted on the train split at build time.
    """
    if dataset_name not in ("midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"):
        raise ValueError(f"unknown dataset {dataset_name!r}")
    out_dir = ensure_tabular_dataset(dataset_name)

    X = pl.read_parquet(out_dir / "features.parquet").to_numpy()
    y = _load_np(out_dir, "labels.npy")
    label_available = _load_np(out_dir, "label_available.npy")
    sample_ids = _load_np(out_dir, "sample_ids.npy")
    timestamps = _load_np(out_dir, "timestamps.npy")
    with open(out_dir / "feature_schema.json", encoding="utf-8") as f:
        schema_info = json.load(f)
    feature_names = list(schema_info["feature_names"])

    if split in ("train", "val", "test"):
        with open(out_dir / "splits.json", encoding="utf-8") as f:
            splits = json.load(f)
        start, end = splits[split]
        X, y = X[start:end], y[start:end]
        label_available = label_available[start:end]
        sample_ids = sample_ids[start:end]
        timestamps = timestamps[start:end]

    return TabularData(
        X=X,
        y=y,
        label_available=label_available,
        sample_ids=sample_ids,
        feature_names=feature_names,
        timestamps=timestamps,
    )


def load_tabular_splits(dataset_name: str) -> dict[str, list[int]]:
    """Return row-index splits {train/val/test: [start, end)}."""
    out_dir = DEFAULT_OUTPUT_ROOT / dataset_name
    with open(out_dir / "splits.json", encoding="utf-8") as f:
        return json.load(f)


def load_tabular_timestamps(dataset_name: str) -> np.ndarray:
    """Return the chronological timestamps for every sample (row order)."""
    out_dir = DEFAULT_OUTPUT_ROOT / dataset_name
    return _load_np(out_dir, "timestamps.npy")


def _write_json(path: Path, obj: object, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists (pass overwrite=True to rebuild)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
