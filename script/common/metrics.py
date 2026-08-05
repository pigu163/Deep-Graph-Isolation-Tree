"""统一评估指标。

所有模型的 anomaly_score 约定统一为：**越大越异常**。
unknown label (-1) 的样本不进入任何指标计算。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

__all__ = ["compute_metrics"]


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int | None = None,
) -> dict:
    """Compute the canonical metric set for one evaluated split.

    Args:
        y_true: int labels, 1 = anomaly/positive, 0 = normal, -1 = unknown
            (unknown are excluded before any computation).
        y_score: anomaly scores, larger = more anomalous.
        k: K for Precision@K / Recall@K / F1@K.  Defaults to the number of
            positive labels in the evaluated set.

    Returns a dict with keys:
        roc_auc, pr_auc, precision_at_k, recall_at_k, f1_at_k,
        n_eval, n_pos, warnings (list[str]).
    ROC-AUC / PR-AUC are ``None`` when the split has a single class; the
    corresponding message is recorded in ``warnings``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays")
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_score={len(y_score)}")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score contains non-finite values (nan/inf)")

    known = y_true != -1
    y_true = y_true[known]
    y_score = y_score[known]
    n_eval = int(len(y_true))
    n_pos = int((y_true == 1).sum())
    warnings: list[str] = []

    if n_eval == 0:
        return {
            "roc_auc": None, "pr_auc": None,
            "precision_at_k": None, "recall_at_k": None, "f1_at_k": None,
            "n_eval": 0, "n_pos": 0, "warnings": ["no known labels in split"],
        }

    if k is None:
        k = n_pos
    k = max(1, min(int(k), n_eval))

    # K-based metrics: top-k by anomaly score.
    order = np.argsort(-y_score, kind="stable")
    y_top = y_true[order[:k]]
    precision_k = float(precision_score(y_top, np.ones(k, dtype=np.int64),
                                        zero_division=0.0))
    recall_k = float(recall_score(y_top, np.ones(k, dtype=np.int64),
                                  zero_division=0.0))
    f1_k = float(f1_score(y_top, np.ones(k, dtype=np.int64), zero_division=0.0))

    out = {
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
        "f1_at_k": f1_k,
        "n_eval": n_eval,
        "n_pos": n_pos,
        "warnings": warnings,
    }

    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        msg = f"single class in evaluated split (n_pos={n_pos}, n_eval={n_eval}); " \
              "ROC-AUC/PR-AUC are null"
        warnings.append(msg)
        out["roc_auc"] = None
        out["pr_auc"] = None
        return out

    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out
