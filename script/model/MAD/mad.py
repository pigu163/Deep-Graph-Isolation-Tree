"""MAD: Multi-Scale Anomaly Detection in Link Streams (paper-guided port).

This module provides the online MAD object (query interface) and the
vectorised batch scorer used by the runner for edge-event anomaly detection.

Data model
----------
* Events are (t, u, v), stored in ``events.parquet`` under the canonical
  link-stream schema, ordered by event_id after a stable temporal sort.
* History is strictly past-only: the window for a query at slice ``t``
  always ends at ``t-1``; the current and future events never enter it.
* Slice policies: native-step / fixed-width / event-count
  (:mod:`script.model.MAD.probability`).

Query modes (task Sec. 9.3)
---------------------------
* ``active-edge``: phi = {(u, v)} for the current event.  This is the
  edge-query special case (M = 1) of the arbitrary-subgraph MAD framework:
  the only random variable is s, so the score of an active event is
  (1 - P)^2 / max(P(1-P), var_floor).  It detects unexpected-active
  relations (densification); for a missing relation no event exists to
  score, which is exactly why ``local-candidate-set`` exists.
* ``local-candidate-set``: phi = {(u,v)} plus a bounded set of relations
  the two endpoints interacted with in the past, so that both
  expected-but-missing and unexpected-active relations are considered.
* ``full-scope``: phi = all relations active in the queried slice plus the
  endpoints' recent historical relations (restricted scope).  A truly
  global relation space E = V x V is never enumerated (infeasible for
  PaySim's ~9M accounts) and no dense matrix is materialised.

Online semantics
----------------
Correct processing order: query history -> estimate P -> score the current
query -> update the state with the current slice.  The batch scorer applies
exactly that order, slice by slice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .multiscale import (
    VAR_FLOOR,
    MultiScaleResult,
    next_power_of_two,
    order_relations,
    score_query,
)
from .probability import (
    SlicePolicy,
    estimate_probabilities,
)
from .stationarity import select_window

__all__ = ["MAD", "EdgeEventRecord", "QUERY_MODES", "run_edge_event_batch"]

QUERY_MODES = ("active-edge", "local-candidate-set", "full-scope")
WINDOW_MODES = ("fixed", "expanding", "stationary")


@dataclass
class EdgeEventRecord:
    """One scored event (row of scores.parquet)."""

    sample_id: int
    timestamp_or_step: float
    label: int
    label_available: bool
    anomaly_score: float
    split: str
    query_mode: str
    selected_scale: int
    selected_relation_count: int
    history_window: int
    explanation: str

    def as_row(self) -> dict:
        return {
            "sample_id": int(self.sample_id),
            "timestamp_or_step": float(self.timestamp_or_step),
            "label": int(self.label),
            "label_available": bool(self.label_available),
            "anomaly_score": float(self.anomaly_score),
            "split": self.split,
            "query_mode": self.query_mode,
            "selected_scale": int(self.selected_scale),
            "selected_relation_count": int(self.selected_relation_count),
            "history_window": int(self.history_window),
            "explanation": self.explanation,
        }


class MAD:
    """Multi-scale anomaly detector over a link stream.

    State is a history of per-slice active relations (``history_slices``).
    The object can be fed slice by slice (streaming) or queried offline.
    """

    def __init__(
        self,
        num_nodes: int,
        *,
        slice_policy: SlicePolicy | None = None,
        window_mode: str = "fixed",
        window_k: int = 100,
        smoothing: float = 0.0,
        var_floor: float = VAR_FLOOR,
        max_stationary_k: int | None = None,
    ) -> None:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be >= 1")
        self.num_nodes = int(num_nodes)
        self.slice_policy = slice_policy or SlicePolicy("native-step")
        if window_mode not in WINDOW_MODES:
            raise ValueError(f"unknown window_mode {window_mode!r}")
        self.window_mode = window_mode
        self.window_k = int(window_k)
        self.smoothing = float(smoothing)
        self.var_floor = float(var_floor)
        self.max_stationary_k = max_stationary_k

        self.first_slice: int | None = None
        self.last_processed_slice: int | None = None
        self.history_slices: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def pair_key(self, src: int, dst: int) -> int:
        return int(src) * self.num_nodes + int(dst)

    def add_slice(self, t: int, active_pair_keys: np.ndarray) -> None:
        """Record the set of active relations of slice ``t``.

        Multiple events of the same relation inside one slice are counted
        once (indicator semantics, matching the reference implementation).
        """
        keys = np.unique(np.asarray(active_pair_keys, dtype=np.int64))
        self.history_slices[t] = keys
        if self.first_slice is None or t < self.first_slice:
            self.first_slice = t
        if self.last_processed_slice is None or t > self.last_processed_slice:
            self.last_processed_slice = t

    # ------------------------------------------------------------------
    # Window helpers
    # ------------------------------------------------------------------

    def _window_for(
        self,
        t: int,
        phi_keys: np.ndarray,
    ) -> tuple[np.ndarray, int, int, float | None]:
        """Return (counts aligned to phi_keys, delta, win_len, fitness)."""
        first = self.first_slice if self.first_slice is not None else t
        if t <= first:
            return np.zeros(len(phi_keys), dtype=np.int64), 0, 0, None
        lo, hi, counts, delta, fitness = select_window(
            phi_keys, self.history_slices, t, first,
            mode=self.window_mode,
            window_k=self.window_k,
            max_stationary_k=self.max_stationary_k,
            var_floor=self.var_floor,
        )
        return counts, delta, hi - lo + 1, fitness

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_relations(
        self,
        t: int,
        relations: list[tuple[int, int]],
        active: np.ndarray | None = None,
    ) -> tuple[MultiScaleResult, int]:
        """Multi-scale anomaly score of an arbitrary relation set at slice t.

        Args:
            t: queried slice.
            relations: list of directed (src, dst) relations (phi).
            active: per-relation binary activity at slice t.  When None,
                    the activity is read from the internally recorded slice.

        Returns (MultiScaleResult, history_window_len).
        """
        m = len(relations)
        if m == 0:
            res = MultiScaleResult(
                score=0.0, n_relations=0, mem_size=0,
                x=np.zeros(0), th_mean=np.zeros(0), th_var=np.zeros(0),
                per_variable_score=np.zeros(0),
                selected_scale=0, selected_relation_count=0,
                explanation="empty relation set",
            )
            return res, 0

        phi_keys = np.array(
            [self.pair_key(s, d) for (s, d) in relations], dtype=np.int64)

        if active is None:
            tset = self.history_slices.get(t)
            obs = np.array(
                [1.0 if (tset is not None and int(k) in tset) else 0.0
                 for k in phi_keys], dtype=np.float64)
        else:
            obs = np.asarray(active, dtype=np.float64)
        if len(obs) != m:
            raise ValueError("active must have one value per relation")

        counts, delta, win_len, _ = self._window_for(t, phi_keys)
        if delta <= 0:
            probs = np.zeros(m, dtype=np.float64)
        else:
            probs = estimate_probabilities(counts, delta, self.smoothing)
        order = order_relations(relations, probs)
        mem = next_power_of_two(m)
        res = score_query(obs, probs, order, mem, self.var_floor)
        return res, win_len

    def score_edge(self, t: int, src: int, dst: int,
                   active: bool | None = None) -> tuple[MultiScaleResult, int]:
        """Score the activity of a single directed relation (M=1 query)."""
        if active is None:
            tset = self.history_slices.get(t)
            is_active = bool(tset is not None and self.pair_key(src, dst) in tset)
        else:
            is_active = bool(active)
        return self.score_relations(t, [(src, dst)],
                                    np.array([1.0 if is_active else 0.0]))

    def score_node(self, t: int, node_id: int, direction: str = "out",
                   max_relations: int = 256) -> tuple[MultiScaleResult, int]:
        """Score the incident relations of one node (node-query special case,
        phi = past relations of the node).  ``direction``: out/in/both."""
        rels: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for s_id, keys in self.history_slices.items():
            if s_id >= t:
                continue
            for k in keys:
                src = k // self.num_nodes
                dst = k - src * self.num_nodes
                if direction in ("out", "both") and src == node_id:
                    r = (src, dst)
                elif direction in ("in", "both") and dst == node_id:
                    r = (src, dst)
                else:
                    continue
                if r not in seen:
                    seen.add(r)
                    rels.append(r)
            if len(rels) > max_relations * 4:
                break
        rels = rels[:max_relations]
        return self.score_relations(t, rels)

    def score_slice(self, t: int,
                    relation_scope: list[tuple[int, int]] | None = None,
                    ) -> tuple[MultiScaleResult, int]:
        """Score the activity of a whole slice.

        ``relation_scope`` restricts phi.  When None, phi = the set of all
        relations active at slice ``t`` (restricted full-scope; the unseen
        relation space is not enumerated).
        """
        if relation_scope is None:
            tset = self.history_slices.get(t)
            rels = []
            if tset is not None:
                for k in tset:
                    src = k // self.num_nodes
                    dst = k - src * self.num_nodes
                    rels.append((src, dst))
        else:
            rels = list(relation_scope)
        return self.score_relations(t, rels)

    def score_subgraph(self, t: int,
                       relations: list[tuple[int, int]]) -> tuple[MultiScaleResult, int]:
        """Score an arbitrary subgraph query (phi = ``relations``)."""
        return self.score_relations(t, list(relations))


# ---------------------------------------------------------------------------
# Batch scorer used by the runner (edge-event anomaly detection)
# ---------------------------------------------------------------------------


def _split_of_event(splits: dict, event_id: int) -> str:
    for name in ("train", "val", "test"):
        lo, hi = splits[name]
        if lo <= event_id < hi:
            return name
    return "none"


def _counts_for_keys(
    phi_keys: np.ndarray,
    window_counts: dict[int, int],
    delta: int,
) -> np.ndarray:
    counts = np.zeros(len(phi_keys), dtype=np.float64)
    if delta <= 0:
        return counts
    for i, k in enumerate(phi_keys):
        counts[i] = window_counts.get(int(k), 0)
    return counts


def _candidate_relations(
    src: int,
    dst: int,
    num_nodes: int,
    temporal_index,
    src_arr: np.ndarray,
    dst_arr: np.ndarray,
    event_id: int,
    max_candidates: int,
    include_self: bool = True,
) -> list[tuple[int, int]]:
    """Bounded candidate relation set for local-candidate-set queries.

    {(u, v)} plus the most recent out-relations of u and in-relations of v
    taken from the temporal index (strictly past events only).
    """
    seen: set[tuple[int, int]] = set()
    rels: list[tuple[int, int]] = []
    if include_self:
        rels.append((int(src), int(dst)))
        seen.add((int(src), int(dst)))

    def _add(r):
        if r not in seen:
            seen.add(r)
            rels.append(r)

    if temporal_index is not None:
        # Out relations of src (most recent events, ascending event ids ->
        # take the last max_candidates).
        out_ev = temporal_index.get_past_out_events(
            int(src), int(event_id), max_events=max_candidates)
        if len(out_ev) > max_candidates:
            out_ev = out_ev[-max_candidates:]
        for e in out_ev:
            _add((int(src), int(dst_arr[e])))
        in_ev = temporal_index.get_past_in_events(
            int(dst), int(event_id), max_events=max_candidates)
        if len(in_ev) > max_candidates:
            in_ev = in_ev[-max_candidates:]
        for e in in_ev:
            _add((int(src_arr[e]), int(dst)))
    return rels[:max_candidates]


def run_edge_event_batch(
    *,
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    timestamps: np.ndarray,
    slice_ids: np.ndarray,
    labels: np.ndarray,
    label_available: np.ndarray,
    splits: dict,
    query_mode: str = "active-edge",
    window_mode: str = "fixed",
    window_k: int = 100,
    smoothing: float = 0.0,
    var_floor: float = VAR_FLOOR,
    max_candidates: int = 256,
    first_scored_slice: int | None = None,
    max_samples: int | None = None,
    temporal_index=None,
) -> tuple[list[EdgeEventRecord], dict, MAD]:
    """Streaming edge-event scoring, slice by slice (strictly past-only).

    Correct order per slice: query history -> estimate P -> score the
    current slice -> update the state with the current slice.

    Returns (records, stats, mad_state).
    """
    if query_mode not in QUERY_MODES:
        raise ValueError(f"unknown query mode {query_mode!r}")
    if window_mode not in WINDOW_MODES:
        raise ValueError(f"unknown window mode {window_mode!r}")

    n = len(src)
    orig_n = n
    keep_idx: np.ndarray | None = None
    if max_samples is not None and max_samples > 0 and max_samples < n:
        keep = np.zeros(n, dtype=bool)
        keep[n - max_samples:] = True
        keep_idx = np.flatnonzero(keep)
        src = src[keep_idx]
        dst = dst[keep_idx]
        timestamps = timestamps[keep_idx]
        slice_ids = slice_ids[keep_idx]
        labels = labels[keep_idx]
        label_available = label_available[keep_idx]
        n = len(keep_idx)

    records: list[EdgeEventRecord] = []
    mad = MAD(
        num_nodes,
        window_mode=window_mode,
        window_k=window_k,
        smoothing=smoothing,
        var_floor=var_floor,
    )

    # Stable sort by slice id preserves event_id order within a slice.
    order = np.argsort(slice_ids, kind="stable")
    unique_slices, first_pos = np.unique(slice_ids[order], return_index=True)
    bounds = np.append(first_pos, n)
    first_slice_t = int(unique_slices[0]) if len(unique_slices) else 0

    # Sliding-window counts (active slices per relation) for fixed/expanding.
    # The window is *calendar*: empty slices count towards delta, exactly
    # like the reference ``delta = max_time - min_time + 1`` (paper eq. 9).
    window_counts: dict[int, int] = {}
    window_deque: deque[tuple[int, np.ndarray]] = deque()

    def _window_delta(t: int) -> int:
        """Calendar length of the history window ending at t-1."""
        if t <= first_slice_t:
            return 0
        if window_mode == "fixed":
            lo = max(first_slice_t, t - window_k)
        else:  # expanding (stationary degrades to expanding in batch mode)
            lo = first_slice_t
        return t - lo

    def _push_slice(t: int, active_keys: np.ndarray) -> None:
        window_deque.append((t, active_keys))
        for k in active_keys:
            window_counts[int(k)] = window_counts.get(int(k), 0) + 1
        if window_mode == "fixed":
            # keep only slices in (t - window_k, t]
            while window_deque and t - window_deque[0][0] >= window_k:
                _old_t, old_keys = window_deque.popleft()
                for k in old_keys:
                    c = window_counts.get(int(k), 1) - 1
                    if c <= 0:
                        window_counts.pop(int(k), None)
                    else:
                        window_counts[int(k)] = c

    def _active_set(active_t: np.ndarray) -> set[int]:
        return {int(k) for k in active_t}

    n_scored = 0
    max_hist_event_id = -1  # max event_id already in the history (past-only)
    max_hist_ts = -1.0
    leakage_checks: list[dict] = []
    leak_target = 50

    def _check_leak(eid: int) -> None:
        # strict past-only: the whole history precedes the current event
        assert max_hist_event_id < eid, \
            f"leak: max_history_event_id={max_hist_event_id} >= current={eid}"
        if len(leakage_checks) < leak_target and n_scored % 97 == 0:
            leakage_checks.append({
                "current_event_id": int(eid),
                "max_history_event_id": int(max_hist_event_id),
                "current_timestamp": float(timestamps[eid]),
                "max_history_timestamp": float(max_hist_ts),
                "history_size": int(delta),
            })

    for s in range(len(unique_slices)):
        t = int(unique_slices[s])
        lo, hi = bounds[s], bounds[s + 1]
        ev = order[lo:hi]
        if len(ev) == 0:
            continue

        pair_keys = (src[ev].astype(np.int64) * num_nodes + dst[ev].astype(np.int64))
        active_t = np.unique(pair_keys)
        delta = _window_delta(t)

        do_score = first_scored_slice is None or t >= first_scored_slice
        if do_score:
            if query_mode == "active-edge":
                denom_p = delta + 2.0 * smoothing
                for j in range(len(ev)):
                    k = int(pair_keys[j])
                    count = window_counts.get(k, 0)
                    p = (count + smoothing) / denom_p if denom_p > 0 else 0.0
                    var = p * (1.0 - p)
                    denom = max(var, var_floor)
                    score = (1.0 - p) ** 2 / denom
                    eid = int(ev[j])
                    _check_leak(eid)
                    rec = EdgeEventRecord(
                        sample_id=eid,
                        timestamp_or_step=float(timestamps[eid]),
                        label=int(labels[eid]),
                        label_available=bool(label_available[eid]),
                        anomaly_score=float(score),
                        split=_split_of_event(splits, eid),
                        query_mode=query_mode,
                        selected_scale=0,
                        selected_relation_count=1,
                        history_window=int(delta),
                        explanation="edge-query M=1: s=(1-P)^2/max(P(1-P),floor)",
                    )
                    records.append(rec)
                    n_scored += 1
            elif query_mode == "local-candidate-set":
                active_set = _active_set(active_t)
                for j in range(len(ev)):
                    eid = int(ev[j])
                    u, v = int(src[eid]), int(dst[eid])
                    _check_leak(eid)
                    rels = _candidate_relations(
                        u, v, num_nodes, temporal_index,
                        src, dst, eid, max_candidates, include_self=True)
                    m = len(rels)
                    phi_keys = np.array(
                        [int(a) * num_nodes + int(b) for (a, b) in rels],
                        dtype=np.int64)
                    obs = np.array(
                        [1.0 if int(k) in active_set else 0.0 for k in phi_keys],
                        dtype=np.float64)
                    counts = _counts_for_keys(phi_keys, window_counts, delta)
                    probs = (estimate_probabilities(counts, delta, smoothing)
                             if delta > 0 else np.zeros(m))
                    order_idx = order_relations(rels, probs)
                    mem = next_power_of_two(m)
                    res = score_query(obs, probs, order_idx, mem, var_floor)
                    rec = EdgeEventRecord(
                        sample_id=eid,
                        timestamp_or_step=float(timestamps[eid]),
                        label=int(labels[eid]),
                        label_available=bool(label_available[eid]),
                        anomaly_score=float(res.score),
                        split=_split_of_event(splits, eid),
                        query_mode=query_mode,
                        selected_scale=res.selected_scale,
                        selected_relation_count=res.selected_relation_count,
                        history_window=int(delta),
                        explanation=res.explanation,
                    )
                    records.append(rec)
                    n_scored += 1
            else:  # full-scope: restricted relation space of the slice
                active_set = _active_set(active_t)
                # phi = every relation active in this slice (observed space)
                rels = []
                for k in active_t:
                    uu = k // num_nodes
                    vv = k - uu * num_nodes
                    rels.append((uu, vv))
                m = len(rels)
                if m == 0:
                    for j in range(len(ev)):
                        eid = int(ev[j])
                        _check_leak(eid)
                        records.append(EdgeEventRecord(
                            sample_id=eid, timestamp_or_step=float(timestamps[eid]),
                            label=int(labels[eid]),
                            label_available=bool(label_available[eid]),
                            anomaly_score=0.0, split=_split_of_event(splits, eid),
                            query_mode=query_mode, selected_scale=0,
                            selected_relation_count=0, history_window=int(delta),
                            explanation="full-scope: empty slice"))
                        n_scored += 1
                    mad.add_slice(t, active_t)
                    _push_slice(t, active_t)
                    continue
                phi_keys = np.array(
                    [int(a) * num_nodes + int(b) for (a, b) in rels],
                    dtype=np.int64)
                obs = np.ones(m, dtype=np.float64)
                counts = _counts_for_keys(phi_keys, window_counts, delta)
                probs = (estimate_probabilities(counts, delta, smoothing)
                         if delta > 0 else np.zeros(m))
                order_idx = order_relations(rels, probs)
                mem = next_power_of_two(m)
                res = score_query(obs, probs, order_idx, mem, var_floor)
                for j in range(len(ev)):
                    eid = int(ev[j])
                    _check_leak(eid)
                    records.append(EdgeEventRecord(
                        sample_id=eid, timestamp_or_step=float(timestamps[eid]),
                        label=int(labels[eid]),
                        label_available=bool(label_available[eid]),
                        anomaly_score=float(res.score),
                        split=_split_of_event(splits, eid),
                        query_mode=query_mode,
                        selected_scale=res.selected_scale,
                        selected_relation_count=res.selected_relation_count,
                        history_window=int(delta),
                        explanation=res.explanation))
                    n_scored += 1

        # Update state with the current slice (after scoring).
        mad.add_slice(t, active_t)
        _push_slice(t, active_t)
        max_hist_event_id = max(max_hist_event_id, int(ev[-1]))
        max_hist_ts = max(max_hist_ts, float(timestamps[ev[-1]]))

    stats = {
        "n_events_loaded": int(orig_n),
        "n_events_processed": int(n),
        "n_scored": n_scored,
        "n_slices": int(len(unique_slices)),
        "query_mode": query_mode,
        "window_mode": window_mode,
        "window_mode_effective": (
            window_mode if window_mode != "stationary"
            else "expanding (stationary degraded in batch mode)"),
        "leakage_check": {
            "passed": all(
                c["max_history_event_id"] < c["current_event_id"]
                for c in leakage_checks),
            "n_checks": len(leakage_checks),
            "samples": leakage_checks,
        },
    }
    return records, stats, mad
