"""PyTorch map-style dataset over a processed link-stream event table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from . import schema, utils


class LinkStreamDataset(torch.utils.data.Dataset):
    """Event-level dataset backed by `events.parquet` + `splits.json`.

    Each item is one temporal link event:

        {"event_id", "src", "dst", "ts", "label",
         "label_available", "edge_feat"}

    The split is a contiguous, chronological event-id range
    (`train` / `val` / `test` / `all`).  Arrays are loaded once into
    contiguous NumPy arrays and wrapped with `torch.from_numpy` (no copies).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "all",
        include_edge_features: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.metadata = utils.read_json(self.processed_dir / schema.METADATA_JSON)
        splits = utils.read_json(self.processed_dir / schema.SPLITS_JSON)

        num_events = int(self.metadata["num_events"])
        if split == "all":
            self.start, self.end = 0, num_events
        elif split in splits:
            self.start, self.end = map(int, splits[split])
        else:
            raise KeyError(f"unknown split {split!r} (available: {sorted(splits)})")
        self.split = split

        self.edge_feature_names = (
            list(self.metadata["edge_feature_names"]) if include_edge_features else []
        )

        base_cols = [schema.EVENT_ID, schema.TIMESTAMP, schema.SRC, schema.DST,
                     schema.LABEL, schema.LABEL_AVAILABLE]
        lf = pl.scan_parquet(self.processed_dir / schema.EVENTS_PARQUET)
        # Row-level filter: [start, end) half-open.
        lf = lf.filter(
            pl.col(schema.EVENT_ID).is_between(self.start, self.end, closed="left"))
        lf = lf.select(base_cols + self.edge_feature_names)
        df = lf.collect()

        self.event_id = df[schema.EVENT_ID].to_numpy()
        self.ts = df[schema.TIMESTAMP].to_numpy()
        self.src = df[schema.SRC].to_numpy()
        self.dst = df[schema.DST].to_numpy()
        self.label = df[schema.LABEL].to_numpy()
        self.label_available = df[schema.LABEL_AVAILABLE].to_numpy()
        self.edge_feat = (
            df.select(self.edge_feature_names).to_numpy()
            if self.edge_feature_names else None
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.event_id)

    def __getitem__(self, index: int) -> dict:
        return {
            "event_id": int(self.event_id[index]),
            "src": int(self.src[index]),
            "dst": int(self.dst[index]),
            "ts": float(self.ts[index]),
            "label": int(self.label[index]),
            "label_available": bool(self.label_available[index]),
            "edge_feat": (
                torch.from_numpy(np.ascontiguousarray(self.edge_feat[index]))
                if self.edge_feat is not None else None
            ),
        }

    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return int(self.metadata["num_nodes"])

    @property
    def edge_feature_dim(self) -> int:
        return len(self.edge_feature_names)

    @property
    def dataset_name(self) -> str:
        return str(self.metadata["dataset_name"])

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return the FULL-stream arrays (src, dst, ts, edge_feat).

        Used by the temporal context collator: the temporal index addresses
        global event ids, so the full stream (not the split slice) is needed.
        """
        df = pl.read_parquet(
            self.processed_dir / schema.EVENTS_PARQUET,
            columns=[schema.SRC, schema.DST, schema.TIMESTAMP] + self.edge_feature_names,
        )
        edge_feat = df.select(self.edge_feature_names).to_numpy() if self.edge_feature_names else None
        return (
            df[schema.SRC].to_numpy(),
            df[schema.DST].to_numpy(),
            df[schema.TIMESTAMP].to_numpy(),
            edge_feat,
        )
