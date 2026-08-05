"""Deviation-Enhanced Anomaly Scoring (DEAS) — paper Eq. (7)-(8).

For a data object ``o`` and forest ``T``:

    g(o|τ)     = (1 / |p(o|τ)|) Σ_{k ∈ path} |o(j_k) − η_k|      (Eq. 7)
    F_DEAS(o|T) = 2^{−E_{τ∈T}|p(o|τ)| / C(T)} × E_{τ∈T} g(o|τ)  (Eq. 8)

where ``C(T)`` is the iForest normalising factor ``c(n)`` (average path
length of an unsuccessful BST search) evaluated at the subsample size ``n``.
Path length ``|p(o|τ)|`` is the number of branching steps along the traversed
path (paper Algorithm 2), without a leaf-size correction — this matches the
paper formula directly.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["average_path_length", "deas_score", "deas_score_from_means"]


def average_path_length(n: int) -> float:
    """iForest normalising factor c(n)."""
    n = int(n)
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + np.euler_gamma) - 2.0 * (n - 1.0) / n


def deas_score_from_means(
    avg_path: np.ndarray,
    avg_g: np.ndarray,
    n_subsample: int,
) -> np.ndarray:
    """Eq. (8) from per-object *averages* over the forest (memory-friendly)."""
    avg_path = np.asarray(avg_path, dtype=np.float64)
    avg_g = np.asarray(avg_g, dtype=np.float64)
    c_n = average_path_length(n_subsample)
    if c_n > 0.0:
        depth_term = np.power(2.0, -avg_path / c_n)
    else:
        depth_term = np.power(2.0, -avg_path)  # degenerate n=1 case
    return np.asarray(depth_term * avg_g, dtype=np.float64)


def deas_score(
    path_lengths: np.ndarray,
    deviations: np.ndarray,
    n_subsample: int,
) -> np.ndarray:
    """Compute Eq. (8) over the forest.

    Args:
        path_lengths: (n_samples, T) per-tree path lengths.
        deviations:   (n_samples, T) per-tree average deviation g.
        n_subsample:  subsample size used when growing the trees (c(n)).

    Returns anomaly scores (n_samples,), larger = more anomalous.
    """
    path_lengths = np.asarray(path_lengths, dtype=np.float64)
    deviations = np.asarray(deviations, dtype=np.float64)
    if path_lengths.ndim != 2 or deviations.shape != path_lengths.shape:
        raise ValueError(
            f"expected 2-D arrays of equal shape, got "
            f"{path_lengths.shape} and {deviations.shape}")
    avg_path = path_lengths.mean(axis=1)
    avg_g = deviations.mean(axis=1)
    return deas_score_from_means(avg_path, avg_g, n_subsample)
