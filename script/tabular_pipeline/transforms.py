"""Deterministic feature transforms for the tabular pipeline.

All encoders fit on the **train split only**; val/test transform strictly
reuses the fitted state.  Encoding is deterministic (sorting by value, ties
broken by the values themselves) and supports unknown categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


def _as_float(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    return arr.astype(np.float64)


@dataclass
class MeanImputer:
    """Replace NaN / +/-inf with the train-set column mean."""

    fill_value: float = 0.0

    def fit(self, values: np.ndarray) -> "MeanImputer":
        arr = _as_float(values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self.fill_value = 0.0
        else:
            self.fill_value = float(finite.mean())
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = _as_float(values).copy()
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = self.fill_value
        return arr

    def state(self) -> dict:
        return {"fill_value": self.fill_value}

    @classmethod
    def from_state(cls, state: dict) -> "MeanImputer":
        return cls(fill_value=float(state["fill_value"]))


@dataclass
class StandardScaler:
    """z-score using train-set mean/std (ddof=0)."""

    mean: float | None = None
    std: float | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        arr = _as_float(values)
        self.mean = float(arr.mean())
        std = float(arr.std())
        self.std = 1.0 if std < _EPS else std
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = _as_float(values)
        return (arr - self.mean) / self.std

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, state: dict) -> "StandardScaler":
        return cls(mean=float(state["mean"]), std=float(state["std"]))


@dataclass
class OrdinalEncoder:
    """Deterministic ordinal encoding of a categorical column.

    Categories are sorted lexicographically (by raw value, cast to str for
    stable ordering across dtypes).  Unknown categories map to -1.
    """

    mapping: dict = field(default_factory=dict)  # raw value (as str) -> code
    _keys: np.ndarray = field(init=False, repr=False, default=None)

    def fit(self, values: np.ndarray) -> "OrdinalEncoder":
        uniq = sorted({str(v) for v in np.asarray(values).ravel().tolist()})
        self.mapping = {v: i for i, v in enumerate(uniq)}
        self._keys = np.asarray(uniq, dtype=object)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values).ravel()
        s = arr.astype(str).astype(object)
        if self._keys is None:
            self._keys = np.asarray(
                sorted(self.mapping, key=lambda v: self.mapping[v]), dtype=object)
        idx = np.searchsorted(self._keys, s)
        idx = np.clip(idx, 0, len(self._keys) - 1)
        hit = self._keys[idx] == s
        codes = np.where(hit, idx, -1).astype(np.float64)
        return codes

    @property
    def n_categories(self) -> int:
        return len(self.mapping)

    def state(self) -> dict:
        return {"mapping": self.mapping, "n_categories": len(self.mapping)}

    @classmethod
    def from_state(cls, state: dict) -> "OrdinalEncoder":
        enc = cls(mapping=dict(state["mapping"]))
        return enc


@dataclass
class FrequencyEncoder:
    """Encode a high-cardinality identifier by its train-set frequency.

    The mapping is stored as two sorted arrays so state can be large without
    bloating JSON (preprocessing.json keeps only a summary; the arrays live
    alongside it in a .npz checkpoint).
    """

    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def fit(self, values: np.ndarray) -> "FrequencyEncoder":
        arr = np.asarray(values).ravel()
        if arr.dtype.kind in "OUS":
            # string ids: encode via pandas-free factorisation (sorted unique)
            uniq, inv = np.unique(arr.astype(str), return_inverse=True)
            self.values = np.asarray(uniq, dtype=object)
        else:
            uniq, counts = np.unique(arr, return_counts=True)
            self.values = uniq
            self.counts = counts.astype(np.float64)
        if self.values.dtype.kind == "O":
            self.counts = np.bincount(np.unique(arr.astype(str), return_inverse=True)[1]).astype(np.float64)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values).ravel()
        if self.values.dtype.kind == "O":
            uniq, inv = np.unique(arr.astype(str), return_inverse=True)
            pos = {v: i for i, v in enumerate(self.values.tolist())}
            out = np.zeros(len(arr), dtype=np.float64)
            for i, u in enumerate(uniq.tolist()):
                j = pos.get(u)
                if j is not None:
                    out[inv == i] = self.counts[j]
            return out
        # numeric path: searchsorted lookup (unseen -> 0)
        idx = np.searchsorted(self.values, arr)
        idx = np.clip(idx, 0, len(self.values) - 1)
        hit = self.values[idx] == arr
        return np.where(hit, self.counts[idx], 0.0)

    def state(self) -> dict:
        return {
            "dtype": str(self.values.dtype),
            "n_values": int(len(self.values)),
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "FrequencyEncoder":
        enc = cls()
        raw = state["values"]
        if state.get("dtype", "").startswith("object") or isinstance(raw, list) and any(
            isinstance(v, str) for v in raw
        ):
            enc.values = np.asarray(raw, dtype=object)
        else:
            enc.values = np.asarray(raw, dtype=np.int64)
        enc.counts = np.asarray(state["counts"], dtype=np.float64)
        return enc
