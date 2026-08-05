"""TGN adapter — Temporal Graph Networks (Rossi et al. 2020,
arXiv:2006.10637), using the official implementation
(https://github.com/twitter-research/tgn, MIT License).

The official model code is imported as-is; only the data plumbing is
adapted to the canonical event table:

- events.parquet -> official ``Data`` (sources/destinations/timestamps/
  edge_idxs/labels); node ids re-mapped to a contiguous 0..N-1 range;
- node features: the repository uses random node features for non-attributed
  graphs (its ``--randomize_features`` path); we do the same (N(0,1));
- edge features: numeric ``f_*`` columns from events.parquet when present
  (NaN -> 0), otherwise a constant dummy feature;
- two-stage official protocol kept: (1) self-supervised link prediction on
  the train split (TGN memory + graph attention, BCE with negative edges);
  (2) frozen TGN + trained MLP decoder for fraud/edge classification,
  trained on train events with known labels only;
- chronological 60/20/20 splits from splits.json; all splits scored with a
  past-only neighbor finder (searchsorted, ts < query).

Score: P(fraud | edge) from the decoder (larger = more anomalous).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
TGN_DIR = _REPO / "source code" / "tgn"
if str(TGN_DIR) not in sys.path:
    sys.path.insert(0, str(TGN_DIR))

from model.tgn import TGN  # noqa: E402
from utils.utils import MLP, RandEdgeSampler, get_neighbor_finder  # noqa: E402
from utils.data_processing import (  # noqa: E402
    Data,
    compute_time_statistics,
)


def build_data(event_id, timestamp, src, dst, label, label_available):
    uniq = np.unique(np.concatenate([src, dst]))
    mapping = {int(n): i for i, n in enumerate(uniq)}
    src_map = np.array([mapping[int(n)] for n in src], dtype=np.int64)
    dst_map = np.array([mapping[int(n)] for n in dst], dtype=np.int64)
    return (Data(sources=src_map, destinations=dst_map,
                 timestamps=timestamp.astype(np.float64),
                 edge_idxs=np.arange(len(src), dtype=np.int64),
                 labels=label.astype(np.int64)),
            int(len(uniq)) - 1)


def build_edge_features(dataset: str, n_events: int,
                        device_agnostic: bool = True) -> np.ndarray:
    """Edge feature matrix indexed by event position."""
    from script.common.events import event_feature_columns, load_event_features

    cols = event_feature_columns(dataset)
    if not cols:
        feats = np.ones((n_events, 1), dtype=np.float32)
        return feats
    feats = np.stack([load_event_features(dataset)[c][:n_events]
                      for c in cols], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


def score_events(tgn: TGN, decoder: MLP, data: Data, batch_size: int,
                 n_neighbors: int, device: torch.device) -> np.ndarray:
    """P(fraud|edge) for every event (larger = more anomalous)."""
    n = len(data.sources)
    pred = np.zeros(n, dtype=np.float64)
    num_batch = math.ceil(n / batch_size)
    tgn.eval()
    decoder.eval()
    with torch.no_grad():
        for k in range(num_batch):
            s = k * batch_size
            e = min(n, s + batch_size)
            src_b = data.sources[s:e]
            dst_b = data.destinations[s:e]
            ts_b = data.timestamps[s:e]
            idx_b = data.edge_idxs[s:e]
            src_emb, _, _ = tgn.compute_temporal_embeddings(
                src_b, dst_b, dst_b, ts_b, idx_b, n_neighbors)
            pred[s:e] = decoder(src_emb).sigmoid().reshape(-1).cpu().numpy()
    return pred


def run_tgn(events: dict, splits: dict[str, tuple[int, int]], *,
            device: str = "cpu", seed: int = 42,
            batch_size: int = 100, n_neighbors: int = 10, n_heads: int = 2,
            n_layers: int = 1, dropout: float = 0.1,
            n_epoch_ss: int = 10, n_epoch_sup: int = 10,
            lr: float = 3e-4, node_dim: int = 100, message_dim: int = 100,
            memory_dim: int = 172, log=print) -> np.ndarray:
    """Two-stage TGN: self-supervised link prediction then supervised
    fraud-edge decoder. Returns per-event fraud probability."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr_s, tr_e = splits["train"]
    full_data, max_idx = build_data(
        events["event_id"], events["timestamp"], events["src"], events["dst"],
        events["label"], events["label_available"])
    tr = slice(tr_s, tr_e)
    train_data = Data(full_data.sources[tr], full_data.destinations[tr],
                      full_data.timestamps[tr],
                      full_data.edge_idxs[tr], full_data.labels[tr])

    dev = torch.device(device)
    n_nodes = max_idx + 1

    node_features = np.random.randn(n_nodes, node_dim).astype(np.float32)
    edge_features = build_edge_features(events["dataset"], len(full_data.sources))

    train_ngh_finder = get_neighbor_finder(train_data, uniform=False,
                                           max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False,
                                          max_node_idx=max_idx)

    mean_ss, std_ss, mean_sd, std_sd = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)

    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=dev, n_layers=n_layers,
              n_heads=n_heads, dropout=dropout, use_memory=True,
              memory_update_at_start=True,
              message_dimension=message_dim, memory_dimension=memory_dim,
              embedding_module_type="graph_attention",
              message_function="identity", aggregator_type="last",
              memory_updater_type="gru", n_neighbors=n_neighbors,
              mean_time_shift_src=mean_ss, std_time_shift_src=std_ss,
              mean_time_shift_dst=mean_sd, std_time_shift_dst=std_sd)
    tgn = tgn.to(dev)

    # ---- stage 1: self-supervised link prediction -------------------------
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=lr)
    sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    n_train = len(train_data.sources)
    num_batch = math.ceil(n_train / batch_size)
    for epoch in range(n_epoch_ss):
        tgn.memory.__init_memory__()
        tgn.set_neighbor_finder(train_ngh_finder)
        losses = []
        tgn.train()
        for k in range(num_batch):
            s = k * batch_size
            e = min(n_train, s + batch_size)
            src_b = train_data.sources[s:e]
            dst_b = train_data.destinations[s:e]
            idx_b = train_data.edge_idxs[s:e]
            ts_b = train_data.timestamps[s:e]
            size = e - s
            _, neg_b = sampler.sample(size)
            optimizer.zero_grad()
            pos_prob, neg_prob = tgn.compute_edge_probabilities(
                src_b, dst_b, neg_b, ts_b, idx_b, n_neighbors)
            loss = (criterion(pos_prob.squeeze(), torch.ones(size, device=dev))
                    + criterion(neg_prob.squeeze(), torch.zeros(size, device=dev)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tgn.memory.detach_memory()
        log(f"  stage1 epoch {epoch} loss={np.mean(losses):.4f}")

    # ---- stage 2: supervised fraud-edge decoder ---------------------------
    known = (train_data.labels != -1)
    sup_src = train_data.sources[known]
    sup_dst = train_data.destinations[known]
    sup_ts = train_data.timestamps[known]
    sup_idx = train_data.edge_idxs[known]
    sup_lab = train_data.labels[known].astype(np.float32)
    log(f"  stage2 known labels: {len(sup_lab)}/{n_train}")

    decoder = MLP(node_features.shape[1], drop=dropout).to(dev)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    n_sup = len(sup_lab)
    num_batch_sup = math.ceil(n_sup / batch_size)
    for epoch in range(n_epoch_sup):
        tgn.memory.__init_memory__()
        tgn.eval()
        decoder.train()
        losses = []
        for k in range(num_batch_sup):
            s = k * batch_size
            e = min(n_sup, s + batch_size)
            dec_opt.zero_grad()
            with torch.no_grad():
                src_emb, _, _ = tgn.compute_temporal_embeddings(
                    sup_src[s:e], sup_dst[s:e], sup_dst[s:e], sup_ts[s:e],
                    sup_idx[s:e], n_neighbors)
            lab = torch.from_numpy(sup_lab[s:e]).float().to(dev)
            pred = decoder(src_emb).sigmoid().reshape(-1)
            loss = criterion(pred, lab)
            loss.backward()
            dec_opt.step()
            losses.append(loss.item())
        log(f"  stage2 epoch {epoch} loss={np.mean(losses):.4f}")

    # ---- score all splits (past-only) -------------------------------------
    tgn.set_neighbor_finder(full_ngh_finder)
    scores = score_events(tgn, decoder, full_data, batch_size, n_neighbors, dev)
    return scores


def events_with_dataset(events: dict, dataset: str) -> dict:
    events["dataset"] = dataset
    return events
