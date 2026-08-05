"""DyGLib adapter — GraphMixer (ICLR 2023) and DyGFormer (NeurIPS 2023)
dynamic-graph link predictors, using the official DyGLib implementation
(https://github.com/yule-buaa/dyglib, MIT License).

The official model code (models/GraphMixer.py, models/DyGFormer.py,
models/modules.py MergeLayer) and utilities (NeighborSampler,
NegativeEdgeSampler) are imported as-is. Only the data plumbing is adapted:

- events.parquet -> DyGLib ``Data`` (src/dst node ids re-mapped to a
  contiguous range, interact times, edge ids, labels);
- node features: random N(0,1) (DyGLib's non-attributed-data convention);
- edge features: numeric ``f_*`` columns when present (NaN -> 0), else a
  dummy feature; both padded to feat_dim;
- self-supervised link-prediction training (BCE with negative edges) on the
  train split — the official DyGLib protocol; labels are never used in
  training;
- all splits scored past-only (NeighborSampler uses searchsorted, so only
  interactions strictly before the query time are sampled).

Score: P(link) from the link predictor; anomaly_score = 1 - P(link)
(larger = more anomalous), i.e. the self-supervised link-anomaly view.
"""

from __future__ import annotations

import math
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
DYGLIB_DIR = _REPO / "source code" / "dyglib"
if str(DYGLIB_DIR) not in sys.path:
    sys.path.insert(0, str(DYGLIB_DIR))

from models.GraphMixer import GraphMixer  # noqa: E402
from models.DyGFormer import DyGFormer  # noqa: E402
from models.modules import MergeLayer  # noqa: E402
from utils.utils import (  # noqa: E402
    NegativeEdgeSampler,
    convert_to_gpu,
    get_neighbor_sampler,
    set_random_seed,
)
from utils.DataLoader import Data  # noqa: E402


def build_data(event_id, timestamp, src, dst, label, label_available):
    uniq = np.unique(np.concatenate([src, dst]))
    mapping = {int(n): i for i, n in enumerate(uniq)}
    src_map = np.array([mapping[int(n)] for n in src], dtype=np.int64)
    dst_map = np.array([mapping[int(n)] for n in dst], dtype=np.int64)
    return (Data(src_node_ids=src_map, dst_node_ids=dst_map,
                 node_interact_times=timestamp.astype(np.float64),
                 edge_ids=np.arange(len(src), dtype=np.int64),
                 labels=label.astype(np.int64)),
            int(len(uniq)) - 1)


def build_features(dataset: str, n_events: int, n_nodes: int,
                   feat_dim: int) -> tuple[np.ndarray, np.ndarray]:
    from script.common.events import event_feature_columns, load_event_features

    cols = event_feature_columns(dataset)
    if cols:
        arr = np.stack([load_event_features(dataset)[c][:n_events]
                        for c in cols], axis=1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        arr = np.ones((n_events, 1), dtype=np.float32)
    if arr.shape[1] < feat_dim:
        pad = np.zeros((n_events, feat_dim - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    node_feat = np.random.randn(n_nodes, feat_dim).astype(np.float32)
    return node_feat, arr


def _make_model(model_name, node_features, edge_features, neighbor_sampler,
                device, time_feat_dim, num_layers, num_heads, dropout,
                num_neighbors, patch_size, max_input_sequence_length,
                channel_embedding_dim, time_gap):
    if model_name == "GraphMixer":
        backbone = GraphMixer(
            node_raw_features=node_features, edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim,
            num_tokens=num_neighbors, num_layers=num_layers, dropout=dropout,
            device=device)
    elif model_name == "DyGFormer":
        backbone = DyGFormer(
            node_raw_features=node_features, edge_raw_features=edge_features,
            neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim,
            channel_embedding_dim=channel_embedding_dim, patch_size=patch_size,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            max_input_sequence_length=max_input_sequence_length, device=device)
    else:
        raise ValueError(model_name)
    link_predictor = MergeLayer(input_dim1=node_features.shape[1],
                                input_dim2=node_features.shape[1],
                                hidden_dim=node_features.shape[1], output_dim=1)
    return nn.Sequential(backbone, link_predictor)


def score_events(model, full_data: Data, full_sampler, batch_size: int,
                 num_neighbors: int, time_gap: int, model_name: str,
                 device: torch.device) -> np.ndarray:
    """P(link) for every event; anomaly = 1 - P(link)."""
    model[0].set_neighbor_sampler(full_sampler)
    n = len(full_data.src_node_ids)
    pred = np.zeros(n, dtype=np.float64)
    num_batch = math.ceil(n / batch_size)
    model.eval()
    with torch.no_grad():
        for k in range(num_batch):
            s = k * batch_size
            e = min(n, s + batch_size)
            src_b = full_data.src_node_ids[s:e]
            dst_b = full_data.dst_node_ids[s:e]
            ts_b = full_data.node_interact_times[s:e]
            if model_name == "GraphMixer":
                src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=dst_b,
                    node_interact_times=ts_b, num_neighbors=num_neighbors,
                    time_gap=time_gap)
            else:
                src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=dst_b,
                    node_interact_times=ts_b)
            logits = model[1](src_emb, dst_emb)
            pred[s:e] = torch.sigmoid(logits).reshape(-1).cpu().numpy()
    return 1.0 - pred  # larger = more anomalous


def run_dyglib(model_name: str, events: dict, splits: dict, *,
               dataset: str, device: str = "cpu", seed: int = 42,
               batch_size: int = 200, num_neighbors: int = 20,
               time_gap: int = 2000, num_layers: int = 2, num_heads: int = 2,
               dropout: float = 0.1, feat_dim: int = 172,
               num_epochs: int = 5, learning_rate: float = 1e-4,
               patch_size: int = 1, log=print, return_timing: bool = False) -> np.ndarray | tuple[np.ndarray, dict]:
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit_started = time.perf_counter()

    tr_s, tr_e = splits["train"]
    full_data, max_idx = build_data(
        events["event_id"], events["timestamp"], events["src"], events["dst"],
        events["label"], events["label_available"])
    tr = slice(tr_s, tr_e)
    train_data = Data(src_node_ids=full_data.src_node_ids[tr],
                      dst_node_ids=full_data.dst_node_ids[tr],
                      node_interact_times=full_data.node_interact_times[tr],
                      edge_ids=full_data.edge_ids[tr],
                      labels=full_data.labels[tr])

    dev = torch.device(device)
    node_features, edge_features = build_features(
        dataset, len(full_data.src_node_ids), max_idx + 1, feat_dim)

    train_sampler = get_neighbor_sampler(data=train_data,
                                         sample_neighbor_strategy="uniform",
                                         time_scaling_factor=0.0, seed=0)
    full_sampler = get_neighbor_sampler(data=full_data,
                                        sample_neighbor_strategy="uniform",
                                        time_scaling_factor=0.0, seed=1)
    train_neg = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                    dst_node_ids=train_data.dst_node_ids)

    model = _make_model(
        model_name, node_features, edge_features, train_sampler, dev,
        time_feat_dim=feat_dim, num_layers=num_layers, num_heads=num_heads,
        dropout=dropout, num_neighbors=num_neighbors, patch_size=patch_size,
        max_input_sequence_length=num_neighbors,
        channel_embedding_dim=feat_dim, time_gap=time_gap)
    model = convert_to_gpu(model, device=device)

    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=1e-4)
    n_train = len(train_data.src_node_ids)
    num_batch = math.ceil(n_train / batch_size)
    for epoch in range(num_epochs):
        model.train()
        model[0].set_neighbor_sampler(train_sampler)
        losses = []
        for k in range(num_batch):
            s = k * batch_size
            e = min(n_train, s + batch_size)
            src_b = train_data.src_node_ids[s:e]
            dst_b = train_data.dst_node_ids[s:e]
            ts_b = train_data.node_interact_times[s:e]
            _, neg_dst_b = train_neg.sample(size=len(src_b))
            optimizer.zero_grad()
            if model_name == "GraphMixer":
                src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=dst_b,
                    node_interact_times=ts_b, num_neighbors=num_neighbors,
                    time_gap=time_gap)
                neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=neg_dst_b,
                    node_interact_times=ts_b, num_neighbors=num_neighbors,
                    time_gap=time_gap)
            else:
                src_emb, dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=dst_b,
                    node_interact_times=ts_b)
                neg_src_emb, neg_dst_emb = model[0].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_b, dst_node_ids=neg_dst_b,
                    node_interact_times=ts_b)
            pos_prob = torch.sigmoid(model[1](src_emb, dst_emb)).reshape(-1)
            neg_prob = torch.sigmoid(model[1](neg_src_emb, neg_dst_emb)).reshape(-1)
            loss = (loss_func(pos_prob, torch.ones_like(pos_prob))
                    + loss_func(neg_prob, torch.zeros_like(neg_prob)))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        log(f"  epoch {epoch} loss={np.mean(losses):.4f}")

    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    scores = score_events(model, full_data, full_sampler, batch_size,
                          num_neighbors, time_gap, model_name, dev)
    timing = {"fit_seconds": fit_seconds, "score_seconds": time.perf_counter() - score_started}
    return (scores, timing) if return_timing else scores
