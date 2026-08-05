"""统一 metrics 契约测试（任务文档 Sec. 7.3）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from script.common.metrics import compute_metrics  # noqa: E402


def test_unknown_labels_excluded():
    y = np.array([0, 1, -1, 0, 1, -1], dtype=np.int64)
    s = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.99])
    m = compute_metrics(y, s)
    assert m["n_eval"] == 4
    assert m["n_pos"] == 2
    assert m["roc_auc"] is not None
    assert m["pr_auc"] is not None


def test_single_class_auc_null():
    y = np.array([0, 0, 0], dtype=np.int64)
    s = np.array([0.1, 0.2, 0.3])
    m = compute_metrics(y, s)
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None
    assert any("single class" in w for w in m["warnings"])
    # Precision@K etc. still computed
    assert m["precision_at_k"] is not None


def test_all_unknown_n_eval_zero():
    y = np.full(5, -1, dtype=np.int64)
    s = np.zeros(5)
    m = compute_metrics(y, s)
    assert m["n_eval"] == 0
    assert m["roc_auc"] is None


def test_score_direction_larger_more_anomalous():
    # perfect separation: positives must get higher scores
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    s = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y, s)
    assert m["roc_auc"] == pytest.approx(1.0)
    # reversed scores -> worst AUC
    m2 = compute_metrics(y, -s)
    assert m2["roc_auc"] == pytest.approx(0.0)


def test_non_finite_score_raises():
    y = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(y, np.array([0.1, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(y, np.array([0.1, np.inf]))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(np.array([0, 1]), np.array([0.1]))


def test_deterministic_ties_top_k():
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    s = np.array([1.0, 1.0, 1.0, 0.0])  # three-way tie at the top
    m1 = compute_metrics(y, s, k=2)
    m2 = compute_metrics(y, s, k=2)
    assert m1 == m2  # stable argsort -> identical K metrics


def test_k_defaults_to_n_pos():
    y = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    s = np.array([0.5, 0.4, 0.3, 0.9, 0.8])
    m = compute_metrics(y, s)
    # K = n_pos = 2 -> top-2 by score are the two positives
    assert m["recall_at_k"] == pytest.approx(1.0)
    assert m["precision_at_k"] == pytest.approx(1.0)
