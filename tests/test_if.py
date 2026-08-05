"""Tests for the IF baseline wrapper (script/model/IF)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from script.model.IF.isolation_forest import IFAnomalyDetector  # noqa: E402


def _synthetic(n_normal: int = 300, n_anom: int = 30, seed: int = 7):
    rng = np.random.RandomState(seed)
    normal = rng.normal(0, 1.0, size=(n_normal, 4))
    anom = rng.uniform(10, 15, size=(n_anom, 4))
    X = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    return X, y


class TestIF:
    def test_score_direction(self):
        """score_samples: larger normally; anomaly_score = negative, larger=
        more anomalous."""
        X, y = _synthetic()
        model = IFAnomalyDetector(n_estimators=50, max_samples=64,
                                  random_state=42)
        model.fit(X)
        raw = model.model.score_samples(X)
        scores = model.score(X)
        assert np.allclose(scores, -raw)
        assert scores[y == 1].mean() > scores[y == 0].mean()

    def test_fit_train_only(self):
        """Scoring does not re-fit; state is unchanged between score calls."""
        X, _ = _synthetic()
        model = IFAnomalyDetector(n_estimators=50, max_samples=64,
                                  random_state=42)
        model.fit(X[:200])
        s1 = model.score(X)
        s2 = model.score(X)
        assert np.allclose(s1, s2)
        assert model.model is not None

    def test_anomaly_higher_score(self):
        X, y = _synthetic()
        model = IFAnomalyDetector(n_estimators=100, max_samples=128,
                                  random_state=1)
        model.fit(X)
        s = model.score(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_batched_scoring_matches(self):
        X, _ = _synthetic()
        model = IFAnomalyDetector(n_estimators=50, max_samples=64,
                                  random_state=42, batch_size=32)
        model.fit(X)
        assert np.allclose(model.score(X, batch_size=7), model.score(X))

    def test_paper_defaults(self):
        model = IFAnomalyDetector()
        assert model.n_estimators == 300
        assert model.max_samples == 256
        assert model.contamination == "auto"
        assert model.n_jobs == -1
