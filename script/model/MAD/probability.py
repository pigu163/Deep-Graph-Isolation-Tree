"""Probability estimation from history (paper Section 3.2) and slice policy.

Discrete-time policies (task Sec. 9.2)
---------------------------------------
The raw stream is a list of (t, u, v) events.  MAD operates on *slices*
(discrete time steps).  Three policies are supported:

* ``native-step``   : slice_id = int(timestamp)  -- used when the timestamp
                      itself is already a discrete step (PaySim, Elliptic++,
                      MIDAS-DARPA).
* ``fixed-width``   : slice_id = floor((timestamp - min_time) / bin_width)
                      -- used for continuous timestamps (IBM AML, Unix
                      seconds), bin width configured per dataset.
* ``event-count``   : slice_id = event_id // bin_size -- fixed number of
                      events per slice.

Probability model (paper eq. 9)
------------------------------
For a stationary window of K slices ending just before the queried slice:

    P(e) = (1/K) * sum_{k=1..K} 1_{L(t-k, phi)}(e)

i.e. the fraction of historical slices in which relation e was ACTIVE
(an indicator, so multiple events of the same relation inside one slice are
counted once -- exactly the reference implementation where the link stream
stores one entry per (u, v, t) triplet).

Numerical handling
------------------
* zero history / unseen relation -> P = 0 (the variance floor keeps the
  standardised score finite; the relation is flagged as unseen).
* probability exactly 0 or 1 -> handled by the variance floor; no 0/0.
* optional additive smoothing P' = (count + alpha) / (K + 2*alpha) is
  configurable (alpha > 0); default 0.0 keeps the paper definition exact.
* short / non-stationary history -> window selection policy
  (fixed / expanding / stationary) is implemented in ``stationarity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "SlicePolicy",
    "compute_slice_ids",
    "estimate_probability",
    "estimate_probabilities",
]

SLICE_POLICIES = ("native-step", "fixed-width", "event-count")


@dataclass
class SlicePolicy:
    """How raw timestamps are mapped onto discrete slices."""

    kind: str = "native-step"
    bin_width: float | None = None   # fixed-width
    bin_size: int | None = None      # event-count
    min_time: float = 0.0            # fixed-width lower bound

    def __post_init__(self) -> None:
        if self.kind not in SLICE_POLICIES:
            raise ValueError(f"unknown slice policy {self.kind!r}; "
                             f"choose from {SLICE_POLICIES}")
        if self.kind == "fixed-width" and (self.bin_width is None
                                           or self.bin_width <= 0):
            raise ValueError("fixed-width policy requires bin_width > 0")
        if self.kind == "event-count" and (self.bin_size is None
                                           or self.bin_size <= 0):
            raise ValueError("event-count policy requires bin_size > 0")

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        if self.bin_width is not None:
            d["bin_width"] = self.bin_width
        if self.bin_size is not None:
            d["bin_size"] = self.bin_size
        if self.min_time != 0.0:
            d["min_time"] = self.min_time
        return d


def compute_slice_ids(
    timestamps: np.ndarray,
    policy: SlicePolicy,
    event_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Map every event to its slice id (non-negative int64)."""
    ts = np.asarray(timestamps)
    if policy.kind == "native-step":
        # Discrete step timestamps; guard against float rounding.
        slices = np.rint(ts).astype(np.int64)
    elif policy.kind == "fixed-width":
        slices = np.floor((ts - policy.min_time) / policy.bin_width).astype(np.int64)
    elif policy.kind == "event-count":
        if event_ids is None:
            event_ids = np.arange(len(ts), dtype=np.int64)
        slices = np.floor_divide(np.asarray(event_ids, dtype=np.int64),
                                 int(policy.bin_size))
    else:  # pragma: no cover - guarded by __post_init__
        raise ValueError(policy.kind)
    if len(slices) and slices.min() < 0:
        raise ValueError(
            f"slice policy {policy.kind!r} produced negative slice ids "
            f"(min={slices.min()}); check min_time/bin_width")
    return slices


def estimate_probability(
    active_slice_count: int,
    window_slices: int,
    smoothing: float = 0.0,
) -> float:
    """P(e) for one relation (paper eq. 9), with optional additive smoothing.

    P = (count + alpha) / (K + 2*alpha), alpha = ``smoothing``.
    With alpha = 0 this is exactly the paper definition.
    """
    if window_slices <= 0:
        raise ValueError("window_slices must be >= 1")
    if active_slice_count < 0 or active_slice_count > window_slices:
        raise ValueError(f"active_slice_count {active_slice_count} outside "
                         f"[0, {window_slices}]")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")
    return (active_slice_count + smoothing) / (window_slices + 2.0 * smoothing)


def estimate_probabilities(
    active_slice_counts: Sequence[int] | np.ndarray,
    window_slices: int,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Vectorised :func:`estimate_probability`."""
    counts = np.asarray(active_slice_counts, dtype=np.float64)
    return (counts + smoothing) / (window_slices + 2.0 * smoothing)
