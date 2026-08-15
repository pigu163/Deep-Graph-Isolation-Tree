"""Tests for the shared tabular pipeline (script/tabular_pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from script.tabular_pipeline import datasets  # noqa: E402
from script.tabular_pipeline.datasets import (  # noqa: E402
    TabularData,
    build_tabular_dataset,
    load_tabular_dataset,
)
from script.tabular_pipeline.preprocessing import TabularPreprocessor  # noqa: E402
from script.tabular_pipeline.schema import (  # noqa: E402
    PROCESSED_TABULAR_NAMES,
    get_schema,
)
from script.tabular_pipeline.temporal_features import event_temporal_features
from script.tabular_pipeline.transforms import (  # noqa: E402
    FrequencyEncoder,
    MeanImputer,
    OrdinalEncoder,
    StandardScaler,
)


def _paysim_like_frame(n: int = 100, seed: int = 0) -> pl.DataFrame:
    rng = np.random.RandomState(seed)
    return pl.DataFrame({
        "event_id": np.arange(n, dtype=np.int64),
        "timestamp": rng.uniform(1, 700, n),
        "amount": rng.exponential(1000, n),
        "f_amount": rng.exponential(1000, n).astype(np.float32),
        "f_oldbalanceOrg": rng.uniform(0, 1e6, n).astype(np.float32),
        "f_newbalanceOrig": rng.uniform(0, 1e6, n).astype(np.float32),
        "f_oldbalanceDest": rng.uniform(0, 1e6, n).astype(np.float32),
        "f_newbalanceDest": rng.uniform(0, 1e6, n).astype(np.float32),
        "edge_type": rng.choice(["CASH_IN", "CASH_OUT", "TRANSFER",
                                 "DEBIT", "PAYMENT"], n),
        "src": rng.randint(0, 200, n).astype(np.int64),
        "dst": rng.randint(0, 200, n).astype(np.int64),
        "label": rng.choice([0, 1], n, p=[0.9, 0.1]).astype(np.int8),
        "label_available": np.ones(n, dtype=bool),
    })


class TestTransforms:
    def test_frequency_encoder(self):
        enc = FrequencyEncoder().fit(np.array([1, 1, 2, 3, 3, 3]))
        out = enc.transform(np.array([1, 2, 3, 99]))
        assert out.tolist() == [2.0, 1.0, 3.0, 0.0]  # unseen -> 0

    def test_frequency_deterministic(self):
        a = FrequencyEncoder().fit(np.array([5, 5, 1, 2]))
        b = FrequencyEncoder().fit(np.array([5, 5, 1, 2]))
        assert (a.transform(np.array([5, 1, 7]))
                == b.transform(np.array([5, 1, 7]))).all()

    def test_ordinal_unknown(self):
        enc = OrdinalEncoder().fit(np.array(["a", "b", "a"]))
        out = enc.transform(np.array(["a", "zzz"]))
        assert out[0] == 0.0 and out[1] == -1.0  # unknown -> -1

    def test_imputer_nan(self):
        imp = MeanImputer().fit(np.array([1.0, 2.0, 3.0]))
        out = imp.transform(np.array([np.nan, 2.0, np.inf]))
        assert out[0] == pytest.approx(2.0)
        assert np.isfinite(out[2])

    def test_scaler(self):
        sc = StandardScaler().fit(np.array([0.0, 2.0, 4.0]))
        out = sc.transform(np.array([2.0]))
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        sc2 = StandardScaler().fit(np.array([5.0, 5.0, 5.0]))  # zero std
        assert sc2.std == 1.0


class TestPreprocessor:
    def test_fit_train_only_and_reuse(self):
        schema = get_schema("paysim")
        frame = _paysim_like_frame(120)
        train = frame.head(60)
        val = frame.slice(60, 60)
        pre = TabularPreprocessor(schema).fit(train)
        X_train = pre.transform(train)
        X_val = pre.transform(val)
        assert X_train.dtype == np.float32
        assert X_train.shape[0] == 60 and X_val.shape[0] == 60
        # transforming the same split twice is identical
        assert (X_val == pre.transform(val)).all()

    def test_unknown_category_all_zero(self):
        schema = get_schema("paysim")
        frame = _paysim_like_frame(200)
        pre = TabularPreprocessor(schema).fit(frame.head(120))
        k = sum(1 for n in pre.feature_names if n.startswith("edge_type=="))
        assert k == 5  # the 5 PaySim transaction types
        val = frame.with_columns(pl.lit("BOGUS_TYPE").alias("edge_type"))
        X = pre.transform(val)
        oh_start = X.shape[1] - k
        assert (X[:, oh_start:] == 0).all()  # unknown -> all-zero one-hot row

    def test_deterministic_two_fits(self):
        schema = get_schema("paysim")
        frame = _paysim_like_frame(80)
        pre1 = TabularPreprocessor(schema).fit(frame.head(40))
        pre2 = TabularPreprocessor(schema).fit(frame.head(40))
        assert pre1.feature_names == pre2.feature_names
        assert (pre1.transform(frame) == pre2.transform(frame)).all()

    def test_feature_names_recorded(self):
        schema = get_schema("paysim")
        frame = _paysim_like_frame(50)
        pre = TabularPreprocessor(schema).fit(frame.head(25))
        names = pre.feature_names
        assert "src_freq" in names and "dst_freq" in names
        assert any("edge_type==" in n for n in names)
        assert len(names) == pre.transform(frame).shape[1]


class TestTemporalFeatures:
    def test_past_only_and_same_timestamp_isolation(self):
        # Rows 0/1 share time=1, so neither can see the other.
        out = event_temporal_features(
            np.array([0, 0, 0, 2]), np.array([1, 1, 2, 1]),
            np.array([1., 1., 2., 3.]))
        assert np.all(out[:2, :5] == 0)
        assert out[2, 0] == 2 and out[2, 2] == 1
        assert out[2, 4] == 0
        assert out[3, 1] == 2 and out[3, 3] == 1
        assert out[3, 6] == pytest.approx(2.)

class TestDatasetBuild:
    @pytest.mark.parametrize("ds", ["midas_darpa"])
    def test_build_outputs_complete(self, tmp_path, ds):
        out = build_tabular_dataset(ds, tmp_path / ds, max_rows=2000)
        for f in ["features.parquet", "labels.npy", "label_available.npy",
                  "sample_ids.npy", "splits.json", "feature_schema.json",
                  "preprocessing.json", "metadata.json"]:
            assert (out / f).exists(), f
        X = pl.read_parquet(out / "features.parquet").to_numpy()
        y = np.load(out / "labels.npy")
        la = np.load(out / "label_available.npy")
        sid = np.load(out / "sample_ids.npy")
        assert X.shape[0] == y.shape[0] == la.shape[0] == sid.shape[0] == 2000
        assert X.dtype == np.float32
        with open(out / "splits.json") as f:
            splits = json.load(f)
        assert set(splits) == {"train", "test"}
        assert splits == {"train": [0, 1400], "test": [1400, 2000]}
        for s, e in splits.values():
            assert 0 <= s <= e <= 2000

    def test_build_deterministic(self, tmp_path):
        a = build_tabular_dataset("midas_darpa", tmp_path / "a", max_rows=1500)
        b = build_tabular_dataset("midas_darpa", tmp_path / "b", max_rows=1500)
        Xa = pl.read_parquet(a / "features.parquet").to_numpy()
        Xb = pl.read_parquet(b / "features.parquet").to_numpy()
        assert np.array_equal(Xa, Xb)

    def test_load_api(self, tmp_path, monkeypatch):
        monkeypatch.setattr(datasets, "DEFAULT_OUTPUT_ROOT", tmp_path)
        build_tabular_dataset("midas_darpa", tmp_path / "midas_darpa",
                              max_rows=2000)
        data = load_tabular_dataset("midas_darpa", split="all")
        assert isinstance(data, TabularData)
        assert data.X.shape[0] == 2000
        assert data.X.dtype == np.float32
        assert len(data.feature_names) == data.X.shape[1]
        tr = load_tabular_dataset("midas_darpa", split="train")
        assert tr.X.shape[0] < data.X.shape[0]


class TestRealSchemas:
    """Read-only schema probes on the real processed datasets (fast)."""

    DATA_ROOT = REPO_ROOT / "data" / "processed"

    def test_all_datasets_have_events(self):
        for ds in PROCESSED_TABULAR_NAMES:
            assert (self.DATA_ROOT / ds / "events.parquet").exists(), ds
            assert (self.DATA_ROOT / ds / "splits.json").exists(), ds

    def test_paysim_columns(self):
        frame = pl.scan_parquet(self.DATA_ROOT / "paysim" / "events.parquet")
        cols = frame.collect_schema().names()
        for c in ["event_id", "timestamp", "src", "dst", "label",
                  "label_available", "edge_type", "amount", "f_amount",
                  "f_oldbalanceOrg", "f_newbalanceOrig", "f_oldbalanceDest",
                  "f_newbalanceDest", "is_flagged_fraud"]:
            assert c in cols, c

    def test_ibm_columns(self):
        frame = pl.scan_parquet(self.DATA_ROOT / "ibm_hi_small" / "events.parquet")
        cols = frame.collect_schema().names()
        for c in ["payment_currency", "receiving_currency", "f_amount_paid",
                  "f_amount_received", "f_same_bank", "f_same_currency"]:
            assert c in cols, c

    def test_midas_columns(self):
        frame = pl.scan_parquet(self.DATA_ROOT / "midas_darpa" / "events.parquet")
        cols = frame.collect_schema().names()
        assert "src" in cols and "dst" in cols and "timestamp" in cols

    def test_elliptic_raw_files(self):
        feat = REPO_ROOT / "data" / "elliptic_pp" / "txs_features.csv"
        cls = REPO_ROOT / "data" / "elliptic_pp" / "txs_classes.csv"
        if feat.exists() and cls.exists():
            n_cols = len(pl.scan_csv(feat, has_header=False)
                         .collect_schema().names())
            assert n_cols >= 167, n_cols  # txId + time_step + 165 features
            header = pl.read_csv(cls, n_rows=1).columns
            assert "txId" in header
