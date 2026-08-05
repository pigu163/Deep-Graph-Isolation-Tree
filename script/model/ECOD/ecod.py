"""ECOD — Empirical-CDF-based Outlier Detection (Li et al., TKDE 2022).

Faithful to the public PyOD implementation
(https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py, BSD-2).
ECOD is parameter-free: it models each feature's tail via empirical CDFs and
sums the negative log tail probabilities of the left/right tails weighted by
the feature skew.

Score direction: larger = more anomalous (PyOD ``decision_function``).
Unsupervised: only ``X_train`` is used for fitting (CDFs built on train
sample); labels are never used by the model.
"""

from __future__ import annotations

import numpy as np


class ECODDetector:
    """Thin wrapper around pyod.models.ecod.ECOD with our fit/score API."""

    def __init__(self, contamination: float = 0.1, n_jobs: int = -1,
                 seed: int = 42) -> None:
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray) -> "ECODDetector":
        from pyod.models.ecod import ECOD

        self._model = ECOD(contamination=self.contamination, n_jobs=self.n_jobs)
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        """Outlier scores; larger = more anomalous."""
        if self._model is None:
            raise RuntimeError("fit() must be called before score()")
        if batch_size is None or X.shape[0] <= batch_size:
            return self._model.decision_function(X)
        scores = np.empty(X.shape[0], dtype=np.float64)
        for s in range(0, X.shape[0], batch_size):
            scores[s:s + batch_size] = self._model.decision_function(
                X[s:s + batch_size])
        return scores
