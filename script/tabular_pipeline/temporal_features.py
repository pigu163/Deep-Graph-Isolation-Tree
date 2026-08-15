"""Strict past-only temporal features for transaction-event tabular data."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    from numba.typed import Dict
    from numba import types
except ImportError:  # pragma: no cover
    njit = None

EVENT_TEMPORAL_NAMES = [
    "temporal_src_prev_tx_count", "temporal_dst_prev_tx_count",
    "temporal_src_prev_unique_dst_count", "temporal_dst_prev_unique_src_count",
    "temporal_src_dst_prev_tx_count", "temporal_src_prev_gap",
    "temporal_dst_prev_gap", "temporal_src_dst_prev_gap",
]
ELLIPTIC_TEMPORAL_NAMES = ["temporal_prev_transaction_count"]

if njit is not None:
    @njit(cache=True)
    def _event_features(src, dst, ts, n_nodes):
        n = len(src)
        out = np.empty((n, 8), dtype=np.float64)
        out[:, 5:] = -1.0
        src_count = np.zeros(n_nodes, np.int64)
        dst_count = np.zeros(n_nodes, np.int64)
        src_unique = np.zeros(n_nodes, np.int64)
        dst_unique = np.zeros(n_nodes, np.int64)
        src_last = np.full(n_nodes, np.nan)
        dst_last = np.full(n_nodes, np.nan)
        pair_count = Dict.empty(key_type=types.int64, value_type=types.int64)
        pair_last = Dict.empty(key_type=types.int64, value_type=types.float64)
        i = 0
        base = np.int64(n_nodes)
        while i < n:
            j = i + 1
            while j < n and ts[j] == ts[i]:
                j += 1
            # Score a timestamp group before adding any of its events.
            for k in range(i, j):
                a, b = src[k], dst[k]
                key = np.int64(a) * base + np.int64(b)
                out[k, 0] = src_count[a]
                out[k, 1] = dst_count[b]
                out[k, 2] = src_unique[a]
                out[k, 3] = dst_unique[b]
                out[k, 4] = pair_count[key] if key in pair_count else 0
                if not np.isnan(src_last[a]): out[k, 5] = ts[k] - src_last[a]
                if not np.isnan(dst_last[b]): out[k, 6] = ts[k] - dst_last[b]
                if key in pair_last: out[k, 7] = ts[k] - pair_last[key]
            for k in range(i, j):
                a, b = src[k], dst[k]
                key = np.int64(a) * base + np.int64(b)
                if key not in pair_count:
                    src_unique[a] += 1
                    dst_unique[b] += 1
                    pair_count[key] = 0
                pair_count[key] += 1
                src_count[a] += 1
                dst_count[b] += 1
                src_last[a] = ts[k]
                dst_last[b] = ts[k]
                pair_last[key] = ts[k]
            i = j
        return out
else:
    _event_features = None


def event_temporal_features(src: np.ndarray, dst: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Return 8 causal account/pair features; equal timestamps are isolated."""
    if _event_features is None:
        raise RuntimeError("numba is required for scalable temporal feature extraction")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if not (len(src) == len(dst) == len(ts)):
        raise ValueError("src, dst and timestamps must have equal length")
    if len(src) == 0:
        return np.empty((0, len(EVENT_TEMPORAL_NAMES)), dtype=np.float64)
    if (src < 0).any() or (dst < 0).any():
        raise ValueError("temporal account ids must be non-negative compact ids")
    n_nodes = int(max(src.max(), dst.max())) + 1
    return _event_features(src, dst, ts, n_nodes)


def elliptic_temporal_features(timestamps: np.ndarray) -> np.ndarray:
    """Causal transaction-volume feature for transaction-node-only Elliptic++."""
    ts = np.asarray(timestamps, dtype=np.float64)
    out = np.empty((len(ts), 1), dtype=np.float64)
    seen, i = 0, 0
    while i < len(ts):
        j = i + 1
        while j < len(ts) and ts[j] == ts[i]: j += 1
        out[i:j, 0] = seen
        seen += j - i
        i = j
    return out
