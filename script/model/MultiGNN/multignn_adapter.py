"""Multi-GNN adapter — Provably Powerful Graph Neural Networks for Directed
Multigraphs (AAAI 2024, arXiv:2306.11586), using the official IBM
implementation (https://github.com/IBM/Multi-GNN, Apache-2.0).

The official model (models.py ``GINe``) and training machinery
(LinkNeighborLoader with edge-id masking, CrossEntropy with class weights)
are used as-is; only the data plumbing is adapted to the canonical event
table:

- events.parquet -> directed multigraph PyG ``Data``: edge_index = [src, dst]
  preserving parallel edges (a core property of the method), edge features
  [Timestamp, Amount Received, Received Currency, Payment Format] following
  the official data_loading.py choice (categoricals ordinal-encoded since
  our events table stores them as strings), node features = all-ones
  placeholder (official choice), labels = transaction labels.
- train graph contains only train edges (official protocol: val = train+val,
  test = all edges); chronological splits from splits.json (60/20/20)
  instead of the official day-balanced split.
- unknown (-1) labels are excluded from the loss but all edges are scored.

Score: P(fraud | edge) = softmax[1] of the GNN output (larger = more
anomalous).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
MGN_DIR = _REPO / "source code" / "Multi-GNN"
if str(MGN_DIR) not in sys.path:
    sys.path.insert(0, str(MGN_DIR))

from models import GINe  # noqa: E402
from data_util import GraphData  # noqa: E402
from torch_geometric.loader import LinkNeighborLoader  # noqa: E402


def build_graph(events: dict, n: int, feats: dict):
    """Directed multigraph Data for the first n events."""
    src = events["src"][:n].astype(np.int64)
    dst = events["dst"][:n].astype(np.int64)
    label = events["label"][:n].astype(np.int64)
    ts = events["timestamp"][:n].astype(np.float64)
    ts_min = ts.min()
    n_nodes = int(max(src.max(), dst.max())) + 1

    cols = []
    for c, arr in feats.items():
        if arr.dtype.kind in ("f", "i"):
            v = np.nan_to_num(arr.astype(np.float64), nan=0.0)
        else:
            codes, _ = pd.factorize(arr, sort=True)
            v = codes.astype(np.float64)
        cols.append(v)
    if not cols:
        cols = [np.zeros(n, dtype=np.float64)]  # dummy feature
    edge_attr = torch.tensor(np.stack(cols, axis=1), dtype=torch.float)
    x = torch.ones((n_nodes, 1), dtype=torch.float)
    edge_index = torch.LongTensor(np.stack([src, dst], axis=0))
    y = torch.LongTensor(label)
    timestamps = torch.Tensor(ts - ts_min)
    return GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                     timestamps=timestamps)


def _add_arange_ids(data: GraphData) -> None:
    data.edge_attr = torch.cat(
        [torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr],
        dim=1)


@torch.no_grad()
def score_edges(loader, inds, model, device, n_edges: int) -> np.ndarray:
    """P(fraud) for every seed edge (replicates official evaluate_homo)."""
    probs = np.zeros(n_edges, dtype=np.float64)
    for batch in loader:
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(),
                          loader.data.edge_attr.detach().cpu()[inds.numpy(), 0])
        seed_edge_ids = batch.edge_attr[mask, 0].detach().cpu().numpy().astype(int)
        batch.edge_attr = batch.edge_attr[:, 1:]
        batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        out = out[mask]
        soft = torch.softmax(out, dim=-1)
        probs[seed_edge_ids] = soft[:, 1].cpu().numpy()
    return probs


def run_multignn(events: dict, splits: dict, *,
                 device: str = "cpu", seed: int = 42,
                 batch_size: int = 8192, n_epochs: int = 100,
                 num_neighs: list[int] | None = None,
                 n_hidden: int = 66, n_gnn_layers: int = 2,
                 dropout: float = 0.01, final_dropout: float = 0.105,
                 lr: float = 0.0062, w_ce1: float = 1.0, w_ce2: float = 6.275,
                 log=print) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_neighs = num_neighs or [100, 100]

    n = len(events["event_id"])
    # per-dataset edge features: numeric f_* columns + categorical string
    # columns (ordinal-encoded), mirroring the official feature choice
    from script.common.events import event_feature_columns, load_event_features
    from script.dataset.linkstream.utils import read_json
    from script.common.results import REPO_ROOT as _RR
    import polars as _pl

    feats = {}
    for c in event_feature_columns(events["dataset"]):
        feats[c] = load_event_features(events["dataset"])[c][:n]
    proc = _RR / "data" / "processed" / events["dataset"]
    schema = _pl.scan_parquet(proc / "events.parquet").collect_schema().names()
    str_cols = [c for c in ("edge_type", "payment_currency",
                            "receiving_currency") if c in schema]
    if str_cols:
        df = _pl.scan_parquet(proc / "events.parquet") \
            .select(str_cols).head(n).collect()
        for c in str_cols:
            feats[c] = df[c].to_numpy()
    full = build_graph(events, n, feats)
    edge_dim = full.edge_attr.shape[1]

    tr_s, tr_e = splits["train"]
    va_s, va_e = splits["val"]
    tr_inds = torch.arange(tr_s, tr_e)
    val_inds = torch.arange(va_s, va_e)
    te_inds = torch.arange(n)

    # official protocol: train graph = train edges, val = train+val, test = all
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds.numpy(), val_inds.numpy()])
    tr_data = GraphData(x=full.x, y=full.y[e_tr],
                        edge_index=full.edge_index[:, e_tr],
                        edge_attr=full.edge_attr[e_tr],
                        timestamps=full.timestamps[e_tr])
    val_data = GraphData(x=full.x, y=full.y[e_val],
                         edge_index=full.edge_index[:, e_val],
                         edge_attr=full.edge_attr[e_val],
                         timestamps=full.timestamps[e_val])
    te_data = full
    _add_arange_ids(tr_data)
    _add_arange_ids(val_data)
    _add_arange_ids(te_data)

    dev = torch.device(device)
    model = GINe(num_features=1, num_gnn_layers=n_gnn_layers, n_classes=2,
                 n_hidden=n_hidden, edge_updates=False, residual=True,
                 edge_dim=edge_dim, dropout=dropout,
                 final_dropout=final_dropout).to(dev)

    tr_loader = LinkNeighborLoader(tr_data, num_neighbors=num_neighs,
                                   batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([w_ce1, w_ce2]).to(dev))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train on train edges with known labels only
    known_loader = LinkNeighborLoader(
        tr_data, num_neighbors=num_neighs, batch_size=batch_size, shuffle=True)
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in known_loader:
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(),
                              tr_data.edge_attr.detach().cpu()[
                                  tr_inds.numpy(), 0])
            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(dev)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out_m = out[mask]
            gt_m = batch.y[mask]
            known = gt_m != -1
            if known.sum().item() == 0:
                continue
            optimizer.zero_grad()
            loss = loss_fn(out_m[known], gt_m[known])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        log(f"  epoch {epoch} loss={np.mean(losses) if losses else 0:.4f}")

    # score all edges
    te_loader = LinkNeighborLoader(
        te_data, num_neighbors=num_neighs,
        edge_label_index=te_data.edge_index[:, te_inds.numpy()],
        edge_label=te_data.y[te_inds.numpy()], batch_size=batch_size,
        shuffle=False)
    model.eval()
    probs = score_edges(te_loader, te_inds, model, dev, n)
    return probs
