"""Collate helpers and DataLoader factory for link-stream event batches.

Base loading is event-only and keeps the exact incoming order
(`shuffle=False`, `drop_last=False` by default).  The optional
`TemporalContextCollator` wraps the base collate to add past temporal
subgraphs per event without changing the base loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .datasets import LinkStreamDataset
from .temporal_index import (
    TemporalNeighborBatch,
    TemporalNeighborIndex,
    sample_temporal_neighbors,
    sample_temporal_subgraph,
)


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


@dataclass
class TemporalContextBatch:
    """Ragged temporal subgraphs packed for one seed-event batch."""

    event_ids: torch.LongTensor
    src: torch.LongTensor
    dst: torch.LongTensor
    ts: torch.Tensor
    edge_feat: torch.Tensor | None
    batch_index: torch.LongTensor
    offsets: torch.LongTensor
    sizes: torch.LongTensor
    valid: torch.BoolTensor


def pack_temporal_subgraphs(contexts: list) -> TemporalContextBatch:
    """Flatten variable-length contexts while retaining their seed ownership."""
    sizes = torch.tensor([len(c.event_ids) for c in contexts], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
    return TemporalContextBatch(
        event_ids=torch.cat([c.event_ids for c in contexts]),
        src=torch.cat([c.src for c in contexts]),
        dst=torch.cat([c.dst for c in contexts]),
        ts=torch.cat([c.ts for c in contexts]),
        edge_feat=None if contexts[0].edge_feat is None else torch.cat([c.edge_feat for c in contexts]),
        batch_index=torch.repeat_interleave(torch.arange(len(contexts)), sizes),
        offsets=offsets,
        sizes=sizes,
        valid=torch.tensor([c.is_valid for c in contexts], dtype=torch.bool),
    )


def collate_link_stream(samples: list[dict]) -> LinkStreamBatch:
    """Stack samples into a `LinkStreamBatch`, keeping their exact order."""
    return LinkStreamBatch(
        event_id=torch.tensor([s["event_id"] for s in samples], dtype=torch.long),
        src=torch.tensor([s["src"] for s in samples], dtype=torch.long),
        dst=torch.tensor([s["dst"] for s in samples], dtype=torch.long),
        ts=torch.tensor([s["ts"] for s in samples], dtype=torch.float64),
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
    collate_fn=None,
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
        collate_fn=collate_fn or collate_link_stream,
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
        max_context_events: int | None = 500,
        min_context_events: int = 0,
        max_neighbors: int = 20,
        time_window: float | None = 7 * 24 * 60 * 60,
        direction: str = "both",
        context_workers: int = 0,
    ) -> None:
        if dataset is not None:
            self._src, self._dst, self._ts, self._edge_feat = dataset.get_arrays()
        else:
            self._src = self._dst = self._ts = self._edge_feat = None
        self.temporal_index = temporal_index
        self.num_hops = num_hops
        self.max_context_events = max_context_events
        self.min_context_events = min_context_events
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.direction = direction
        self.context_workers = max(0, int(context_workers))
        self._pool = (ThreadPoolExecutor(max_workers=self.context_workers)
                      if self.context_workers > 1 else None)

    def _require_arrays(self) -> None:
        if self._src is None:
            raise RuntimeError(
                "TemporalContextCollator needs a LinkStreamDataset to read "
                "event arrays; pass dataset=... at construction")

    def sample_event_ids(self, event_ids: np.ndarray | torch.Tensor | list[int]) -> list:
        """Extract independent past contexts for one seed batch.

        Threads only parallelise independent seeds. The temporal CSR index and
        event arrays remain shared read-only rather than being copied per
        DataLoader worker process.
        """
        self._require_arrays()
        ids = [int(eid) for eid in np.asarray(event_ids).reshape(-1)]

        def sample(eid: int):
            return sample_temporal_subgraph(
                self.temporal_index, eid, src_arr=self._src, dst_arr=self._dst,
                ts_arr=self._ts, edge_feat=self._edge_feat, num_hops=self.num_hops,
                max_neighbors=self.max_neighbors, max_context_events=self.max_context_events,
                min_context_events=self.min_context_events, time_window=self.time_window,
                direction=self.direction,
            )

        if self._pool is not None and len(ids) > 1:
            return list(self._pool.map(sample, ids))
        return [sample(eid) for eid in ids]

    def __call__(self, samples: list[dict]) -> dict:
        batch = collate_link_stream(samples)
        contexts = self.sample_event_ids(batch.event_id)
        return {"events": batch, "contexts": contexts,
                "context_batch": pack_temporal_subgraphs(contexts)}


class DenseTemporalNeighborCollator:
    """Attach TGN-style fixed-width past-neighbor references to a batch."""

    def __init__(self, temporal_index: TemporalNeighborIndex, *,
                 n_neighbors: int = 20, direction: str = "both") -> None:
        self.temporal_index = temporal_index
        self.n_neighbors = n_neighbors
        self.direction = direction

    def __call__(self, samples: list[dict]) -> dict:
        batch = collate_link_stream(samples)
        ts = batch.ts.detach().cpu().numpy().astype(np.float64)
        src_context: TemporalNeighborBatch = sample_temporal_neighbors(
            self.temporal_index,
            node_ids=batch.src.detach().cpu().numpy(),
            cut_timestamps=ts,
            n_neighbors=self.n_neighbors,
            direction=self.direction,
        )
        dst_context: TemporalNeighborBatch = sample_temporal_neighbors(
            self.temporal_index,
            node_ids=batch.dst.detach().cpu().numpy(),
            cut_timestamps=ts,
            n_neighbors=self.n_neighbors,
            direction=self.direction,
        )
        return {"events": batch, "src_context": src_context, "dst_context": dst_context}
