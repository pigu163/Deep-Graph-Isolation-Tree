"""数据接口检查（任务文档 Sec. 9.1 / 10.4 / 14）。

三个模型共用同一组接口检查；结果写入每次运行的 ``interface_check.json``。
"""

from __future__ import annotations

import numpy as np

__all__ = ["tabular_interface_check"]


def tabular_interface_check(
    X: np.ndarray,
    y: np.ndarray,
    label_available: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: list[str],
    scores: np.ndarray,
    fitted_on: str = "train",
) -> dict:
    """Return the canonical tabular data-interface check dict.

    Checks: shapes, finite X/scores, feature-name count, sample-id
    uniqueness, score/sample count alignment, score direction, fit split.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    label_available = np.asarray(label_available)
    sample_ids = np.asarray(sample_ids)
    scores = np.asarray(scores)
    return {
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "label_available_shape": list(label_available.shape),
        "X_finite": bool(np.isfinite(X).all()),
        "n_feature_names": int(len(feature_names)),
        "feature_names_match_X": bool(len(feature_names) == X.shape[1]),
        "sample_ids_unique": bool(len(np.unique(sample_ids)) == len(sample_ids)),
        "scores_finite": bool(np.isfinite(scores).all()),
        "score_count_matches_samples": bool(len(scores) == X.shape[0]),
        "score_direction": "larger=more anomalous",
        "fitted_on": fitted_on,
    }
