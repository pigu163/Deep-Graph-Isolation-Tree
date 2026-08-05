"""TabularPreprocessor: train-only fitted feature pipeline.

Rule (task doc Sec. 6): scaler / imputer / categorical encoder are fitted on
the **train split only**; val/test transform strictly reuses the fitted
state.  Labels are never used to fit the preprocessor.

Encoding choices (recorded in `feature_schema.json`):

* numeric columns        -> mean imputation + z-score (train fit)
* high-cardinality ids   -> train-set frequency count, then z-score
* categorical columns    -> deterministic one-hot (sorted categories,
                            unknown category -> all-zero row)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .schema import DatasetSchema
from .transforms import FrequencyEncoder, MeanImputer, OrdinalEncoder, StandardScaler


class TabularPreprocessor:
    """Fits on train, transforms any split deterministically."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema
        self.numeric: dict[str, tuple[MeanImputer, StandardScaler]] = {}
        self.freq: dict[str, tuple[FrequencyEncoder, StandardScaler]] = {}
        self.categorical: dict[str, OrdinalEncoder] = {}
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, df: pl.DataFrame) -> "TabularPreprocessor":
        schema = self.schema
        missing = [c for c in schema.used_raw_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{schema.dataset_name}] raw frame missing columns "
                f"used by the schema: {missing}")
        for c in schema.numeric:
            col = np.asarray(df[c].to_numpy(), dtype=np.float64)
            imp = MeanImputer().fit(col)
            scaler = StandardScaler().fit(imp.transform(col))
            self.numeric[c] = (imp, scaler)
        for c in schema.freq:
            col = np.asarray(df[c].to_numpy()).ravel()
            enc = FrequencyEncoder().fit(col)
            counts = enc.transform(col)
            scaler = StandardScaler().fit(counts)
            self.freq[c] = (enc, scaler)
        for c in schema.categorical:
            self.categorical[c] = OrdinalEncoder().fit(
                np.asarray(df[c].to_numpy()).ravel())

        self._feature_names = self._build_feature_names(df)
        return self

    def _build_feature_names(self, df: pl.DataFrame) -> list[str]:
        names: list[str] = []
        names += list(self.schema.numeric)
        names += [f"{c}_freq" for c in self.schema.freq]
        for c in self.schema.categorical:
            enc = self.categorical[c]
            uniq = sorted(enc.mapping, key=lambda v: enc.mapping[v])
            names += [f"{c}=={v}" for v in uniq]
        return names

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names:
            raise RuntimeError("preprocessor not fitted yet")
        return list(self._feature_names)

    # ------------------------------------------------------------- transform
    def transform(self, df: pl.DataFrame) -> np.ndarray:
        schema = self.schema
        cols: list[np.ndarray] = []
        for c in schema.numeric:
            imp, scaler = self.numeric[c]
            col = np.asarray(df[c].to_numpy(), dtype=np.float64)
            cols.append(scaler.transform(imp.transform(col)).reshape(-1, 1))
        for c in schema.freq:
            enc, scaler = self.freq[c]
            col = np.asarray(df[c].to_numpy()).ravel()
            cols.append(scaler.transform(enc.transform(col)).reshape(-1, 1))
        for c in schema.categorical:
            enc = self.categorical[c]
            codes = enc.transform(np.asarray(df[c].to_numpy()).ravel()).astype(int)
            k = enc.n_categories
            onehot = np.zeros((len(codes), k), dtype=np.float64)
            valid = codes >= 0
            onehot[valid, codes[valid]] = 1.0
            cols.append(onehot)
        if not cols:
            raise ValueError(f"[{schema.dataset_name}] no feature columns")
        X = np.hstack(cols)
        return X.astype(np.float32)

    # ---------------------------------------------------------------- state
    def state_summary(self) -> dict:
        """JSON-safe summary recorded in preprocessing.json."""
        schema = self.schema
        return {
            "dataset_name": schema.dataset_name,
            "numeric": {
                c: {"imputer": imp.state(), "scaler": sc.state()}
                for c, (imp, sc) in self.numeric.items()
            },
            "freq": {
                c: {
                    "encoder_summary": {
                        "dtype": str(enc.values.dtype),
                        "n_values": int(len(enc.values)),
                        "top5": enc.values[:5].tolist(),
                    },
                    "scaler": sc.state(),
                }
                for c, (enc, sc) in self.freq.items()
            },
            "categorical": {
                c: enc.state() for c, enc in self.categorical.items()
            },
            "feature_names": self._feature_names,
            "notes": schema.notes,
        }

    @classmethod
    def from_state_summary(cls, schema: DatasetSchema, state: dict) -> "TabularPreprocessor":
        pre = cls(schema)
        for c, s in state.get("numeric", {}).items():
            imp = MeanImputer.from_state(s["imputer"])
            sc = StandardScaler.from_state(s["scaler"])
            pre.numeric[c] = (imp, sc)
        for c, s in state.get("freq", {}).items():
            # full frequency maps are intentionally not persisted (too large);
            # features.parquet already holds the transformed matrix.
            sc = StandardScaler.from_state(s["scaler"])
            pre.freq[c] = (FrequencyEncoder(), sc)
        for c, s in state.get("categorical", {}).items():
            pre.categorical[c] = OrdinalEncoder.from_state(s)
        pre._feature_names = list(state.get("feature_names", []))
        return pre
