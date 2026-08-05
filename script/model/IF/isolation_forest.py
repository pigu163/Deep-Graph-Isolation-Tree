"""Isolation Forest baseline.

Paper-fair defaults (task doc Sec. 8.2):

    n_estimators=300, max_samples=256, contamination="auto",
    random_state=seed, n_jobs=-1

Anomaly score direction is unified as **larger = more anomalous**:

    anomaly_score = -model.score_samples(X)

The model fits on the **train split only**; the contamination parameter is
never set from labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

__all__ = ["IFAnomalyDetector"]


class IFAnomalyDetector:
    """Thin wrapper around scikit-learn IsolationForest."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_samples: int = 256,
        contamination: str | float = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
        batch_size: int = 100_000,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> "IFAnomalyDetector":
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix (n_samples, n_features)")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Anomaly scores; larger = more anomalous (batched scoring)."""
        if self.model is None:
            raise RuntimeError("fit() must be called before score()")
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix (n_samples, n_features)")
        b = batch_size or self.batch_size
        out = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], b):
            end = min(start + b, X.shape[0])
            out[start:end] = -self.model.score_samples(X[start:end])
        return out
