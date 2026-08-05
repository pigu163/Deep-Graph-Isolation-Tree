"""Deep isolation tree (DIT) construction and traversal.

Implements paper Algorithm 1 (construction) and the per-tree traversal part
of Algorithm 2:

* random axis-parallel split: choose dimension ``j_k`` uniformly among
  ``{1..d}``, pick ``η_k ~ U(min, max)`` of that dimension in the node pool;
* stop when ``|P_k| <= 1`` or the depth limit ``J`` is reached;
* traversal records the path length ``|p(x|τ)|`` and the accumulated
  deviation ``β = Σ_k |x(j_k) − η_k|`` along the path (Eq. 7).

A node whose pool has identical values on every candidate dimension is a
leaf (no split is possible).
"""

from __future__ import annotations

import numpy as np

try:  # numba is present in the project environment; retain a safe fallback.
    from numba import njit, set_num_threads
except ImportError:  # pragma: no cover - exercised only in minimal installs
    njit = None
    set_num_threads = None


def _traverse_numpy(proj, feature, threshold, left, right, is_leaf, max_depth):
    """Fallback used only when numba is unavailable."""
    b = proj.shape[0]
    cur = np.zeros(b, dtype=np.int64)
    path = np.zeros(b, dtype=np.float64)
    beta = np.zeros(b, dtype=np.float64)
    alive = np.ones(b, dtype=bool)
    for _ in range(max_depth):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        c = cur[idx]
        not_leaf = ~is_leaf[c]
        i2 = idx[not_leaf]
        if len(i2) == 0:
            break
        c2 = cur[i2]
        dims = feature[c2]
        vals = proj[i2, dims]
        beta[i2] += np.abs(vals - threshold[c2])
        path[i2] += 1.0
        cur[i2] = np.where(vals <= threshold[c2], left[c2], right[c2])
    return path, np.where(path > 0.0, beta / np.maximum(path, 1e-12), 0.0)


if njit is not None:
    @njit(cache=True)
    def _traverse_numba(proj, feature, threshold, left, right, is_leaf, max_depth):
        b = proj.shape[0]
        path = np.zeros(b, dtype=np.float64)
        g = np.zeros(b, dtype=np.float64)
        for i in range(b):
            node = 0
            beta = 0.0
            depth = 0
            for _ in range(max_depth):
                if is_leaf[node]:
                    break
                value = proj[i, feature[node]]
                diff = value - threshold[node]
                beta += diff if diff >= 0.0 else -diff
                depth += 1
                node = left[node] if value <= threshold[node] else right[node]
            path[i] = depth
            if depth > 0:
                g[i] = beta / depth
        return path, g
else:
    _traverse_numba = None


__all__ = ["DeepIsolationTree"]


class DeepIsolationTree:
    """One isolation tree over a single representation space."""

    def __init__(self, max_depth: int = 8, rng: np.random.RandomState | None = None):
        self.max_depth = int(max_depth)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        # numpy arrays, index 0 = root
        self.feature: np.ndarray | None = None
        self.threshold: np.ndarray | None = None
        self.left: np.ndarray | None = None
        self.right: np.ndarray | None = None
        self.node_size: np.ndarray | None = None
        self.is_leaf: np.ndarray | None = None
        self.n_nodes = 0
        self._fitted = False

    # ------------------------------------------------------------------ build
    def build(self, proj: np.ndarray) -> "DeepIsolationTree":
        """Grow the tree on `proj` (n_samples, d) from representation space."""
        proj = np.asarray(proj, dtype=np.float64)
        if proj.ndim != 2:
            raise ValueError("proj must be (n_samples, d)")
        n, d = proj.shape
        # worst case: full binary tree of depth J -> 2^(J+1)-1 nodes
        capacity = max(2 ** (self.max_depth + 1), 2 * n + 1)
        self.feature = np.full(capacity, -1, dtype=np.int64)
        self.threshold = np.zeros(capacity, dtype=np.float64)
        self.left = np.zeros(capacity, dtype=np.int64)
        self.right = np.zeros(capacity, dtype=np.int64)
        self.node_size = np.zeros(capacity, dtype=np.int64)
        self.is_leaf = np.zeros(capacity, dtype=bool)
        self.n_nodes = 0
        self._build_rec(proj, np.arange(n), depth=0, d=d)
        self.feature = self.feature[:self.n_nodes]
        self.threshold = self.threshold[:self.n_nodes]
        self.left = self.left[:self.n_nodes]
        self.right = self.right[:self.n_nodes]
        self.node_size = self.node_size[:self.n_nodes]
        self.is_leaf = self.is_leaf[:self.n_nodes]
        self._fitted = True
        return self

    def _new_node(self) -> int:
        node = self.n_nodes
        self.n_nodes += 1
        return node

    def _build_rec(self, proj: np.ndarray, indices: np.ndarray, depth: int,
                   d: int) -> int:
        node = self._new_node()
        self.node_size[node] = len(indices)
        if len(indices) <= 1 or depth >= self.max_depth:
            self.is_leaf[node] = True
            self.feature[node] = -1
            return node
        j = int(self.rng.randint(0, d))
        vals = proj[indices, j]
        vmin, vmax = float(vals.min()), float(vals.max())
        if not np.isfinite(vmax - vmin) or vmin == vmax:
            # no usable variation on the chosen dimension -> leaf
            self.is_leaf[node] = True
            self.feature[node] = -1
            return node
        eta = float(self.rng.uniform(vmin, vmax))
        left_mask = vals <= eta
        if left_mask.all() or (~left_mask).all():
            # degeneracy guard (should be impossible given vmin != vmax)
            self.is_leaf[node] = True
            self.feature[node] = -1
            return node
        self.feature[node] = j
        self.threshold[node] = eta
        self.left[node] = self._build_rec(proj, indices[left_mask], depth + 1, d)
        self.right[node] = self._build_rec(proj, indices[~left_mask], depth + 1, d)
        return node

    # --------------------------------------------------------------- traverse
    def traverse(self, proj: np.ndarray, num_workers: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Traverse a batch and return path length and mean split deviation.

        The compiled loop removes repeated Python/advanced-indexing overhead.
        ``num_workers`` is accepted as a runtime resource hint; the current
        per-tree kernel is intentionally single-threaded because profiling
        showed Python thread-pool traversal is slower on this workload.
        """
        if not self._fitted:
            raise RuntimeError("build() must be called before traverse()")
        proj = np.ascontiguousarray(proj, dtype=np.float64)
        if _traverse_numba is not None:
            return _traverse_numba(proj, self.feature, self.threshold,
                                   self.left, self.right, self.is_leaf,
                                   self.max_depth)
        return _traverse_numpy(proj, self.feature, self.threshold,
                               self.left, self.right, self.is_leaf,
                               self.max_depth)

    def config(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "n_nodes": self.n_nodes if self._fitted else None,
        }
