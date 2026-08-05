"""Tabular pipeline: shared preprocessing for IF / DIF.

Builds `data/processed_tabular/<dataset>/` outputs and exposes the public
`load_tabular_dataset` API (task doc Sec. 6).
"""

from __future__ import annotations

from .datasets import (
    DEFAULT_OUTPUT_ROOT,
    TabularData,
    build_tabular_dataset,
    load_tabular_dataset,
    load_tabular_splits,
    load_tabular_timestamps,
)

__all__ = [
    "TabularData",
    "build_tabular_dataset",
    "load_tabular_dataset",
    "load_tabular_splits",
    "load_tabular_timestamps",
    "DEFAULT_OUTPUT_ROOT",
]
