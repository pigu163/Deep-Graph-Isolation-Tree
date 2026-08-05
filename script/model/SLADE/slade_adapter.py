"""SLADE adapter — Self-supervised Learning for Anomaly Detection in Edge
Streams (KDD 2024, arXiv:2402.11933), using the official implementation
(https://github.com/jhsk777/SLADE).

The official code is imported as-is (SLADE_TGN model, memory modules,
neighbor finder); only the data plumbing is adapted:

- events.parquet -> SLADE ``Data`` (sources/destinations/timestamps/
  edge_idxs/labels);
- node ids are re-mapped to a contiguous 0..N-1 range inside the adapter
  (the official Memory arrays index nodes directly; our canonical ids are
  contiguous but remapping guarantees safety and matches the official
  assumption);
- chronological splits come from data/processed/*/splits.json (60/20/20)
  instead of the official training-ratio quantile split;
- evaluation uses our shared metric code (unknown labels excluded), scoring
  train/val/test with a past-only neighbor finder built over full data.

Score direction: larger = more anomalous
((2 - drift_sim - recovery_sim)/4, following the official evaluation).

Leakage note: the neighbor finder keeps per-node past interactions sorted by
timestamp and ``find_before`` uses ``np.searchsorted(..., 'left')`` so only
events with timestamp < query timestamp are returned — current and future
events are never used as history.
"""

from __future__ import annotations

import math
import time
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
SLADE_DIR = _REPO / "source code" / "SLADE"
if str(SLADE_DIR) not in sys.path:
    sys.path.insert(0, str(SLADE_DIR))

from utils.data_processing import Data  # noqa: E402
from utils.utils import get_neighbor_finder  # noqa: E402
from model.SLADE_TGN import SLADE_TGN  # noqa: E402


def build_data(event_id, timestamp, src, dst, label, label_available):
    """Contiguous node remap + SLADE Data object."""
    uniq = np.unique(np.concatenate([src, dst]))
    mapping = {int(n): i for i, n in enumerate(uniq)}
    src_map = np.array([mapping[int(n)] for n in src], dtype=np.int64)
    dst_map = np.array([mapping[int(n)] for n in dst], dtype=np.int64)
    return Data(
        sources=src_map,
        destinations=dst_map,
        timestamps=timestamp.astype(np.float64),
        edge_idxs=np.arange(len(src), dtype=np.int64),
        labels=label.astype(np.int64),
    ), int(len(uniq)) - 1


def score_events(model: SLADE_TGN, data: Data, batch_size: int,
                 n_neighbors: int, device: torch.device) -> np.ndarray:
    """Score every event (larger = more anomalous). Mirrors the official
    evaluation loop without its internal roc_auc computation."""
    n = len(data.sources)
    pred = np.zeros(n, dtype=np.float64)
    num_batch = math.ceil(n / batch_size)
    model.eval()
    with torch.no_grad():
        for k in range(num_batch):
            s = k * batch_size
            e = min(n, s + batch_size)
            src_b = torch.from_numpy(data.sources[s:e]).long().to(device)
            dst_b = torch.from_numpy(data.destinations[s:e]).long().to(device)
            ts_b = torch.from_numpy(data.timestamps[s:e]).float().to(device)
            src_nb, _, src_nb_t = model.neighbor_finder.get_temporal_neighbor(
                data.sources[s:e], data.timestamps[s:e], n_neighbors)
            dst_nb, _, dst_nb_t = model.neighbor_finder.get_temporal_neighbor(
                data.destinations[s:e], data.timestamps[s:e], n_neighbors)
            src_nb = torch.from_numpy(src_nb).long().to(device)
            dst_nb = torch.from_numpy(dst_nb).long().to(device)
            src_nb_t = torch.from_numpy(src_nb_t).long().to(device)
            dst_nb_t = torch.from_numpy(dst_nb_t).long().to(device)
            rec, drift, _, _ = model.compute_anomaly_score(
                src_b, dst_b, ts_b, src_nb, dst_nb, src_nb_t, dst_nb_t,
                n_neighbors)
            pred[s:e] = (-drift.reshape(-1).cpu().numpy()
                         - rec.reshape(-1).cpu().numpy() + 2.0) / 4.0
    return pred


def run_slade(events: dict, splits: dict[str, tuple[int, int]],
              *, device: str = "cpu", seed: int = 42,
              batch_size: int = 100, n_neighbors: int = 20, n_heads: int = 2,
              n_epochs: int = 10, lr: float = 3e-6, dropout: float = 0.1,
              message_dim: int = 128, memory_dim: int = 256,
              memory_updater: str = "gru", message_updater: str = "mlp",
              srf: float = 0.1, drf: float = 0.1, lr_decay: float = 0.8,
              weight_decay: float = 1e-4, log=print, return_timing: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    """Train SLADE on the train split and score all events (past-only)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit_started = time.perf_counter()

    tr_s, tr_e = splits["train"]
    full_data, max_idx = build_data(
        events["event_id"], events["timestamp"], events["src"], events["dst"],
        events["label"], events["label_available"])
    # slice out the train portion (splits are event_id ranges)
    tr_src = full_data.sources[tr_s:tr_e]
    tr_dst = full_data.destinations[tr_s:tr_e]
    tr_ts = full_data.timestamps[tr_s:tr_e]
    tr_lab = full_data.labels[tr_s:tr_e]
    train_data = Data(tr_src, tr_dst, tr_ts, np.arange(tr_s, tr_e, dtype=np.int64),
                      tr_lab)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False,
                                           max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False,
                                          max_node_idx=max_idx)

    src_neighbors, _, src_nb_time = train_ngh_finder.get_temporal_neighbor(
        train_data.sources, train_data.timestamps, n_neighbors)
    dst_neighbors, _, dst_nb_time = train_ngh_finder.get_temporal_neighbor(
        train_data.destinations, train_data.timestamps, n_neighbors)

    dev = torch.device(device)
    n_nodes = max_idx + 1
    model = SLADE_TGN(
        neighbor_finder=train_ngh_finder, device=dev, n_nodes=n_nodes,
        n_edges=len(full_data.sources), n_layers=1, n_heads=n_heads,
        dropout=dropout, message_dimension=message_dim,
        memory_dimension=memory_dim, n_neighbors=n_neighbors,
        memory_agg_type="TGAT", negative_memory_type="train",
        message_updater=message_updater, memory_updater=memory_updater,
        src_reg_factor=srf, dst_reg_factor=drf)
    model = model.to(dev)

    src_t = torch.from_numpy(train_data.sources).long().to(dev)
    dst_t = torch.from_numpy(train_data.destinations).long().to(dev)
    ts_t = torch.from_numpy(train_data.timestamps).float().to(dev)
    src_nb_t = torch.from_numpy(src_neighbors).long().to(dev)
    dst_nb_t = torch.from_numpy(dst_neighbors).long().to(dev)
    src_nb_time_t = torch.from_numpy(src_nb_time).long().to(dev)
    dst_nb_time_t = torch.from_numpy(dst_nb_time).long().to(dev)
    seen_nodes = torch.from_numpy(np.unique(
        np.concatenate([train_data.sources, train_data.destinations]))).long().to(dev)

    n_train = len(train_data.sources)
    num_batch = math.ceil(n_train / batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    for epoch in range(n_epochs):
        model.memory.__init_memory__()
        model.set_neighbor_finder(train_ngh_finder)
        losses = []
        for k in range(num_batch):
            optimizer.zero_grad()
            s = k * batch_size
            e = min(n_train, s + batch_size)
            _, _, _, _, loss = model.compute_node_diff_score(
                src_t[s:e], dst_t[s:e], ts_t[s:e], src_nb_t[s:e], dst_nb_t[s:e],
                src_nb_time_t[s:e], dst_nb_time_t[s:e], n_neighbors, seen_nodes)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            model.memory.detach_memory()
        scheduler.step()
        log(f"  epoch {epoch} loss={np.mean(losses):.4f}")

    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    # score all splits with a full past-only neighbor finder
    model.set_neighbor_finder(full_ngh_finder)
    scores = score_events(model, full_data, batch_size, n_neighbors, dev)
    timing = {"fit_seconds": fit_seconds, "score_seconds": time.perf_counter() - score_started}
    return (scores, timing) if return_timing else scores
