"""CSR-style temporal neighbor index for link streams.

The index stores, for every node, the sorted list of its *past* event ids
separately for out / in / incident relations, plus an optional directed-pair
index.  Queries are pure binary searches over sorted segments
(`numpy.searchsorted`); no full-table scans are involved.

Leakage rule (section 4.2 of the task): a query for event `i` returns only
event ids strictly smaller than `i`, regardless of equal timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import schema


def _build_csr(group_ids: np.ndarray, values: np.ndarray, num_groups: int):
    """Return (offsets, sorted_values) where values are sorted inside each group.

    `values` are event ids 0..N-1; a stable argsort keeps them ascending
    within every group.  The returned array is `values[order]`, i.e. the
    actual event ids in per-group temporal order.
    """
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    counts = np.bincount(sorted_groups, minlength=num_groups)
    offsets = np.zeros(num_groups + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, values[order].astype(np.int64)


class TemporalNeighborIndex:
    """Compressed temporal neighbor index over an ordered link stream.

    Arrays are stored as flat int64/float64 `.npy` files under
    `<processed_dir>/temporal_index/` (see schema.py).
    """

    def __init__(
        self,
        *,
        src_offsets: np.ndarray,
        src_event_ids: np.ndarray,
        dst_offsets: np.ndarray,
        dst_event_ids: np.ndarray,
        node_offsets: np.ndarray,
        node_event_ids: np.ndarray,
        timestamps: np.ndarray,
        pair_keys: np.ndarray | None = None,
        pair_offsets: np.ndarray | None = None,
        pair_event_ids: np.ndarray | None = None,
    ) -> None:
        self.src_offsets = src_offsets
        self.src_event_ids = src_event_ids
        self.dst_offsets = dst_offsets
        self.dst_event_ids = dst_event_ids
        self.node_offsets = node_offsets
        self.node_event_ids = node_event_ids
        self.timestamps = timestamps
        self.pair_keys = pair_keys
        self.pair_offsets = pair_offsets
        self.pair_event_ids = pair_event_ids
        self.num_events = len(timestamps)
        self.num_nodes = len(src_offsets) - 1

    # ------------------------------------------------------------------
    # Construction / persistence
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        src: np.ndarray,
        dst: np.ndarray,
        timestamps: np.ndarray,
        num_nodes: int,
        build_pair_index: bool = False,
    ) -> "TemporalNeighborIndex":
        """Build the index from aligned event arrays (event order 0..N-1).

        `src`/`dst` must already be in stable temporal order (the caller
        sorts the stream first).
        """
        if len(src) != len(dst) or len(src) != len(timestamps):
            raise ValueError("src/dst/timestamps length mismatch")
        num_events = len(src)
        event_ids = np.arange(num_events, dtype=np.int64)

        src_offsets, src_event_ids = _build_csr(src, event_ids, num_nodes)
        dst_offsets, dst_event_ids = _build_csr(dst, event_ids, num_nodes)
        # Interleave (src_i, dst_i) per event so that each group keeps its
        # event ids in ascending order (a naive concatenate([src, dst]) puts
        # every src before every dst and breaks in-group ordering, which the
        # binary-search leakage guard relies on).
        n = num_events
        incident_groups = np.empty(2 * n, dtype=src.dtype)
        incident_vals = np.empty(2 * n, dtype=event_ids.dtype)
        incident_groups[0::2] = src
        incident_groups[1::2] = dst
        incident_vals[0::2] = event_ids
        incident_vals[1::2] = event_ids
        node_offsets, node_event_ids = _build_csr(
            incident_groups, incident_vals, num_nodes)

        pair_keys = pair_offsets = pair_event_ids = None
        if build_pair_index:
            pair_keys, pair_offsets, pair_event_ids = cls._build_pair_index(
                src, dst, event_ids, num_nodes)

        return cls(
            src_offsets=src_offsets,
            src_event_ids=src_event_ids,
            dst_offsets=dst_offsets,
            dst_event_ids=dst_event_ids,
            node_offsets=node_offsets,
            node_event_ids=node_event_ids,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            pair_keys=pair_keys,
            pair_offsets=pair_offsets,
            pair_event_ids=pair_event_ids,
        )

    @staticmethod
    def _build_pair_index(src, dst, event_ids, num_nodes) -> tuple:
        # pair_key = src * num_nodes + dst fits in int64 for any realistic
        # dataset (num_nodes**2 < 2**63).
        pair_key = src.astype(np.int64) * num_nodes + dst.astype(np.int64)
        order = np.lexsort((event_ids, pair_key))
        sorted_keys = pair_key[order]
        uniq_keys, counts = np.unique(sorted_keys, return_counts=True)
        offsets = np.zeros(len(uniq_keys) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        return uniq_keys, offsets, order.astype(np.int64)

    def save(self, index_dir: Path | str) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "src_offsets": self.src_offsets,
            "src_event_ids": self.src_event_ids,
            "dst_offsets": self.dst_offsets,
            "dst_event_ids": self.dst_event_ids,
            "node_offsets": self.node_offsets,
            "node_event_ids": self.node_event_ids,
            "timestamps": self.timestamps,
        }
        for name, arr in arrays.items():
            np.save(index_dir / f"{name}.npy", arr, allow_pickle=False)
        if self.pair_keys is not None:
            np.save(index_dir / "pair_keys.npy", self.pair_keys, allow_pickle=False)
            np.save(index_dir / "pair_offsets.npy", self.pair_offsets, allow_pickle=False)
            np.save(index_dir / "pair_event_ids.npy", self.pair_event_ids, allow_pickle=False)

    @classmethod
    def load(cls, index_dir: Path | str) -> "TemporalNeighborIndex":
        index_dir = Path(index_dir)

        def _load(fname: str) -> np.ndarray:
            return np.load(index_dir / fname, allow_pickle=False)

        # schema.TEMP_INDEX_FILES entries carry a ".npy" suffix; use the bare
        # name as the attribute key.
        arrays = {name[:-4]: _load(name) for name in schema.TEMP_INDEX_FILES}
        pair = {}
        if (index_dir / "pair_keys.npy").exists():
            for name in schema.PAIR_INDEX_FILES:
                pair[name[:-4]] = _load(name)
        return cls(
            src_offsets=arrays["src_offsets"],
            src_event_ids=arrays["src_event_ids"],
            dst_offsets=arrays["dst_offsets"],
            dst_event_ids=arrays["dst_event_ids"],
            node_offsets=arrays["node_offsets"],
            node_event_ids=arrays["node_event_ids"],
            timestamps=arrays["timestamps"],
            pair_keys=pair.get("pair_keys"),
            pair_offsets=pair.get("pair_offsets"),
            pair_event_ids=pair.get("pair_event_ids"),
        )

    # ------------------------------------------------------------------
    # Core segment query (binary search, no full scans)
    # ------------------------------------------------------------------

    def _query_segment(
        self,
        seg_event_ids: np.ndarray,
        seg_ts: np.ndarray,
        event_id: int,
        max_events: int | None,
        time_window: float | None,
    ) -> np.ndarray:
        """Return seg_event_ids[start:end] with ids strictly < event_id.

        `seg_event_ids` ascending, `seg_ts` non-decreasing (guaranteed by the
        stable global sort).  With `time_window`, only events with
        ts >= ts[event_id] - window are returned.  The most recent
        `max_events` events are kept when a cap is given.
        """
        # ids strictly smaller than the current event (safe with ties).
        end = int(np.searchsorted(seg_event_ids, event_id, side="left"))
        start = 0
        if time_window is not None:
            lower_ts = float(self.timestamps[event_id]) - float(time_window)
            start = int(np.searchsorted(seg_ts[:end], lower_ts, side="left"))
        if max_events is not None and (end - start) > max_events:
            start = end - max_events
        return seg_event_ids[start:end]

    # ------------------------------------------------------------------
    # Public queries (all return numpy int64 arrays of event ids)
    # ------------------------------------------------------------------

    def get_past_out_events(
        self,
        node_id: int,
        event_id: int,
        max_events: int | None = None,
        time_window: float | None = None,
    ) -> np.ndarray:
        """Past events where `node_id` was the source."""
        o, e = self.src_offsets[node_id], self.src_offsets[node_id + 1]
        seg = self.src_event_ids[o:e]
        return self._query_segment(seg, self.timestamps[seg], event_id,
                                   max_events, time_window)

    def get_past_in_events(
        self,
        node_id: int,
        event_id: int,
        max_events: int | None = None,
        time_window: float | None = None,
    ) -> np.ndarray:
        """Past events where `node_id` was the destination."""
        o, e = self.dst_offsets[node_id], self.dst_offsets[node_id + 1]
        seg = self.dst_event_ids[o:e]
        return self._query_segment(seg, self.timestamps[seg], event_id,
                                   max_events, time_window)

    def get_past_incident_events(
        self,
        node_id: int,
        event_id: int,
        max_events: int | None = None,
        time_window: float | None = None,
    ) -> np.ndarray:
        """Past events where `node_id` was either endpoint."""
        o, e = self.node_offsets[node_id], self.node_offsets[node_id + 1]
        seg = self.node_event_ids[o:e]
        return self._query_segment(seg, self.timestamps[seg], event_id,
                                   max_events, time_window)

    def get_past_pair_events(
        self,
        src: int,
        dst: int,
        event_id: int,
        max_events: int | None = None,
        time_window: float | None = None,
    ) -> np.ndarray:
        """Past events of the directed pair (src, dst)."""
        if self.pair_keys is None:
            raise RuntimeError("pair index was not built "
                               "(re-run preprocess.py with --build-pair-index)")
        key = src * self.num_nodes + dst
        loc = int(np.searchsorted(self.pair_keys, key, side="left"))
        if loc >= len(self.pair_keys) or self.pair_keys[loc] != key:
            return np.empty(0, dtype=np.int64)
        o, e = self.pair_offsets[loc], self.pair_offsets[loc + 1]
        seg = self.pair_event_ids[o:e]
        return self._query_segment(seg, self.timestamps[seg], event_id,
                                   max_events, time_window)

    def has_pair_index(self) -> bool:
        return self.pair_keys is not None


# ---------------------------------------------------------------------------
# Temporal subgraph sampling (section 12 of the task)
# ---------------------------------------------------------------------------


@dataclass
class TemporalSubgraph:
    """A local, strictly-past temporal subgraph around a query event."""

    node_ids: torch.LongTensor
    event_ids: torch.LongTensor
    src: torch.LongTensor
    dst: torch.LongTensor
    ts: torch.Tensor
    edge_feat: torch.Tensor | None


def sample_temporal_subgraph(
    index: TemporalNeighborIndex,
    event_id: int,
    *,
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    ts_arr: np.ndarray,
    edge_feat: np.ndarray | None = None,
    num_hops: int = 1,
    max_neighbors: int = 20,
    time_window: float | None = None,
    direction: str = "both",
) -> TemporalSubgraph:
    """Sample the past temporal neighborhood of `event_id`.

    BFS expansion: start from {src, dst} of the current event, query only
    past incident/out/in events (strictly before `event_id`), keep at most
    `max_neighbors` most recent events per node per hop, and never include
    the current or future events.  `direction` selects which history is used
    per node: "out", "in" or "both".
    """
    if num_hops < 1:
        raise ValueError("num_hops must be >= 1")
    if direction not in ("out", "in", "both"):
        raise ValueError("direction must be 'out', 'in' or 'both'")
    if event_id < 0 or event_id >= index.num_events:
        raise IndexError(f"event_id {event_id} out of range")

    hist_fn = {
        "out": index.get_past_out_events,
        "in": index.get_past_in_events,
        "both": index.get_past_incident_events,
    }[direction]

    selected: set[int] = set()
    frontier: set[int] = {int(src_arr[event_id]), int(dst_arr[event_id])}
    for _ in range(num_hops):
        next_frontier: set[int] = set()
        for node in frontier:
            hist = hist_fn(node, event_id, max_events=max_neighbors,
                           time_window=time_window)
            selected.update(int(e) for e in hist)
            if selected:
                next_frontier.update(int(src_arr[e]) for e in hist)
                next_frontier.update(int(dst_arr[e]) for e in hist)
        frontier = next_frontier

    event_ids = np.array(sorted(selected), dtype=np.int64)
    # Unique involved nodes: current endpoints + nodes of selected history.
    node_ids = sorted({int(src_arr[event_id]), int(dst_arr[event_id])}
                      | {int(n) for e in event_ids for n in (src_arr[e], dst_arr[e])})

    sub_src = src_arr[event_ids].astype(np.int64)
    sub_dst = dst_arr[event_ids].astype(np.int64)
    sub_ts = ts_arr[event_ids].astype(np.float32)
    sub_feat = None if edge_feat is None else np.asarray(
        edge_feat[event_ids], dtype=np.float32)

    return TemporalSubgraph(
        node_ids=torch.from_numpy(np.asarray(node_ids, dtype=np.int64)),
        event_ids=torch.from_numpy(event_ids),
        src=torch.from_numpy(sub_src),
        dst=torch.from_numpy(sub_dst),
        ts=torch.from_numpy(sub_ts),
        edge_feat=None if sub_feat is None else torch.from_numpy(sub_feat),
    )
