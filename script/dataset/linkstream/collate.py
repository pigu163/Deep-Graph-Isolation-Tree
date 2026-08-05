"""Collate helpers and DataLoader factory for link-stream event batches.

Base loading is event-only and keeps the exact incoming order
(`shuffle=False`, `drop_last=False` by default).  The optional
`TemporalContextCollator` wraps the base collate to add past temporal
subgraphs per event without changing the base loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .datasets import LinkStreamDataset
from .temporal_index import TemporalNeighborIndex, sample_temporal_subgraph


@dataclass
class LinkStreamBatch:
    """A mini-batch of ordered temporal link events."""

    event_id: torch.LongTensor
    src: torch.LongTensor
    dst: torch.LongTensor
    ts: torch.Tensor
    label: torch.LongTensor
    label_available: torch.BoolTensor
    edge_feat: torch.Tensor | None


def collate_link_stream(samples: list[dict]) -> LinkStreamBatch:
    """Stack samples into a `LinkStreamBatch`, keeping their exact order."""
    return LinkStreamBatch(
        event_id=torch.tensor([s["event_id"] for s in samples], dtype=torch.long),
        src=torch.tensor([s["src"] for s in samples], dtype=torch.long),
        dst=torch.tensor([s["dst"] for s in samples], dtype=torch.long),
        ts=torch.tensor([s["ts"] for s in samples], dtype=torch.float32),
        label=torch.tensor([s["label"] for s in samples], dtype=torch.long),
        label_available=torch.tensor(
            [s["label_available"] for s in samples], dtype=torch.bool),
        edge_feat=(
            torch.stack([s["edge_feat"] for s in samples])
            if samples and samples[0]["edge_feat"] is not None else None
        ),
    )


def make_link_stream_loader(
    dataset: LinkStreamDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    """DataLoader factory with temporal-research-safe defaults.

    Defaults: no shuffling (the stream order is sacred), no dropping of the
    final incomplete batch, no worker processes.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_link_stream,
    )


class TemporalContextCollator:
    """Collator that augments each event batch with past temporal subgraphs.

    Returns {"events": LinkStreamBatch, "contexts": list[TemporalSubgraph]}.
    Extraction is strictly past (event ids < current id), so there is no
    temporal leakage.  This is intentionally optional: the base loader stays
    event-only.
    """

    def __init__(
        self,
        temporal_index: TemporalNeighborIndex,
        *,
        dataset: LinkStreamDataset | None = None,
        num_hops: int = 1,
        max_neighbors: int = 20,
        time_window: float | None = None,
        direction: str = "both",
    ) -> None:
        if dataset is not None:
            self._src, self._dst, self._ts, self._edge_feat = dataset.get_arrays()
        else:
            self._src = self._dst = self._ts = self._edge_feat = None
        self.temporal_index = temporal_index
        self.num_hops = num_hops
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.direction = direction

    def _require_arrays(self) -> None:
        if self._src is None:
            raise RuntimeError(
                "TemporalContextCollator needs a LinkStreamDataset to read "
                "event arrays; pass dataset=... at construction")

    def __call__(self, samples: list[dict]) -> dict:
        self._require_arrays()
        batch = collate_link_stream(samples)
        contexts = [
            sample_temporal_subgraph(
                self.temporal_index,
                int(eid),
                src_arr=self._src,
                dst_arr=self._dst,
                ts_arr=self._ts,
                edge_feat=self._edge_feat,
                num_hops=self.num_hops,
                max_neighbors=self.max_neighbors,
                time_window=self.time_window,
                direction=self.direction,
            )
            for eid in batch.event_id.tolist()
        ]
        return {"events": batch, "contexts": contexts}
