"""Thin wrapper around the locally vendored Extended Isolation Forest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_EIF_SOURCE = _ROOT / "source code" / "eif"


def _load_eif():
    if str(_EIF_SOURCE) not in sys.path:
        sys.path.insert(0, str(_EIF_SOURCE))
    try:
        from eif import iForest
    except ImportError as exc:  # pragma: no cover - exercised only on missing build
        raise RuntimeError(
            "EIF extension is unavailable; build source code/eif with "
            "'/home/zhu/.conda/envs/IF/bin/python setup.py build_ext --inplace'."
        ) from exc
    return iForest


class EIFAnomalyDetector:
    """Train-only Extended Isolation Forest; larger scores are more anomalous."""

    def __init__(self, n_estimators: int = 300, max_samples: int = 256,
                 extension_level: int = -1, random_state: int = 42,
                 batch_size: int = 100_000):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state
        self.batch_size = batch_size
        self.model = None
        self.sample_size_: int | None = None
        self.extension_level_: int | None = None

    @staticmethod
    def _array(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64, order="C")
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix (n_samples, n_features)")
        if X.shape[0] < 2:
            raise ValueError("EIF requires at least two fit rows")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    def fit(self, X: np.ndarray) -> "EIFAnomalyDetector":
        X = self._array(X)
        self.sample_size_ = min(self.max_samples, X.shape[0])
        self.extension_level_ = (X.shape[1] - 1 if self.extension_level < 0
                                 else self.extension_level)
        if not 0 <= self.extension_level_ < X.shape[1]:
            raise ValueError("extension_level must be in [0, n_features - 1]")
        self.model = _load_eif()(X, self.n_estimators, self.sample_size_,
                                 ExtensionLevel=self.extension_level_,
                                 seed=self.random_state)
        return self

    def score(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() must be called before score()")
        X = self._array(X)
        b = batch_size or self.batch_size
        out = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], b):
            out[start:start + b] = self.model.compute_paths(X[start:start + b])
        return out
