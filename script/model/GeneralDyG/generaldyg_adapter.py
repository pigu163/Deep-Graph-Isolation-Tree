"""GeneralDyG adapter — A Generalizable Anomaly Detection Method in Dynamic
Graphs (AAAI 2025, arXiv:2412.16447), using the official implementation
(https://github.com/YXNTU/GeneralDyG).

The official model code is imported as-is, with two CPU-compatibility copies
that only replace hard-coded ``.to('cuda')`` device strings (the upstream
code cannot run on CPU otherwise): the CensNet ``GraphConvolution`` layer and
the ``CombinedModel`` forward. The Transformer backbone is used unmodified.

Task adaptation:
- events.parquet -> one sample per transaction edge, exactly as the paper's
  per-edge ego-graph construction (k=1 hop subgraph, max_mask_len=24).
- Leakage fix: the paper extracts ego-graphs from the *full* graph; for our
  chronological task the ego-graph is built from strictly past events
  (event_id < current) plus the current edge itself, so no future
  information enters a sample. This is recorded as a deviation.
- labels are the transaction labels; unknown (-1) events are excluded from
  training but still scored.
- chronological 60/20/20 splits from splits.json.

Score: P(fraud|ego-graph) from the classifier (larger = more anomalous).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3]
GDG_DIR = _REPO / "source code" / "GeneralDyG"
if str(GDG_DIR) not in sys.path:
    sys.path.insert(0, str(GDG_DIR))

from model.CensNet import CensNet  # noqa: E402
from model.Transformer import TransformerBinaryClassifier  # noqa: E402


# --------------------------------------------------------------------------
# CPU-compatible copy of layers.GraphConvolution (hard-coded cuda removed)
# --------------------------------------------------------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features_v, out_features_v, in_features_e,
                 out_features_e, bias=True, node_layer=True):
        super().__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        if node_layer:
            self.node_layer = True
            self.weight = nn.Parameter(
                torch.FloatTensor(in_features_v, out_features_v))
            self.p = nn.Parameter(
                torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter("bias", None)
        else:
            self.node_layer = False
            self.weight = nn.Parameter(
                torch.FloatTensor(in_features_e, out_features_e))
            self.p = nn.Parameter(
                torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        dev = H_v.device
        if self.node_layer:
            multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.t()
            mask1 = torch.eye(multiplier1.shape[0], device=dev)
            M1 = mask1 * torch.ones(multiplier1.shape[0], device=dev) + (1. - mask1) * multiplier1
            adjusted_A = torch.mul(M1, adj_v)
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e
        else:
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T
            mask2 = torch.eye(multiplier2.shape[0], device=dev)
            M3 = mask2 * torch.ones(multiplier2.shape[0], device=dev) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e)
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret


class CensNetCPU(CensNet):
    """CensNet with CPU-compatible GraphConvolution."""

    def __init__(self, nfeat, dropout):
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=True)
        self.gc2 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=False)
        self.dropout = dropout


class CombinedModel(nn.Module):
    """CombinedModel with CPU-compatible device handling."""

    def __init__(self, gnn, transformer):
        super().__init__()
        self.GNN = gnn
        self.Transformer = transformer

    def forward(self, H_v, H_e, H_e_pad, adj_e, adj_v, T, mask):
        output = self.GNN(H_v, H_e, adj_e, adj_v, T)
        pad_output = torch.zeros(
            (H_e_pad.shape[0], H_e_pad.shape[1], H_e_pad.shape[2]),
            device=H_e_pad.device)
        for i in range(H_e_pad.shape[0]):
            length = output[i].shape[0]
            pad_output[i, :length, :] = output[i]
        output = self.Transformer(H_e_pad, pad_output, mask)
        return output


# --------------------------------------------------------------------------
# ego-graph construction (paper's BatchGraphSample, past-only graph)
# --------------------------------------------------------------------------
class PastEgoGraphBuilder:
    def __init__(self, src: np.ndarray, dst: np.ndarray, max_mask_len: int = 24):
        self.src = src
        self.dst = dst
        self.max_mask_len = max_mask_len
        self.G = nx.Graph()

    @staticmethod
    def _remove_random(nodes, max_mask_len, keep: set):
        nodes = list(nodes)
        rng = random.Random(0)
        while len(nodes) > max_mask_len:
            to_remove = rng.choice(nodes)
            if to_remove not in keep:
                nodes.remove(to_remove)
        return set(nodes)

    def _extract_k_hop(self, graph, src_node, dest_node, k, max_mask_len):
        nodes = {src_node}
        frontier = {src_node}
        for _ in range(k):
            new_nodes = set()
            for n in frontier:
                if graph.has_node(n):
                    new_nodes.update(graph.neighbors(n))
            frontier = new_nodes - nodes
            nodes.update(new_nodes)
        return self._remove_random(nodes, max_mask_len, {src_node, dest_node})

    def build(self, i: int):
        """Ego-graph for event i using strictly past edges + current edge."""
        s, d = int(self.src[i]), int(self.dst[i])
        # add past edges up to i-1 (G is updated incrementally by caller)
        src_nodes = self._extract_k_hop(self.G, s, d, 1, self.max_mask_len)
        dest_nodes = self._extract_k_hop(self.G, d, s, 1, self.max_mask_len)
        sub_nodes = src_nodes | dest_nodes
        sub = self.G.subgraph(sub_nodes).copy()
        # current edge (weight = event id)
        if sub.has_edge(s, d):
            sub[s][d]["weight"] = i
        else:
            sub.add_edge(s, d, weight=i)
        # reorder nodes by min edge weight (paper's reorder_nodes)
        node_weights = {}
        for node in sub.nodes:
            edges = sub.edges(node, data="weight")
            node_weights[node] = (min(w for _, _, w in edges) if edges
                                  else float("inf"))
        order = sorted(node_weights.items(), key=lambda kv: kv[1])
        node_mapping = {old: new for new, (old, _) in enumerate(order)}
        return sub, node_mapping, s, d, i


def build_samples(events: dict, n: int, max_mask_len: int = 24):
    """Per-event ego-graph samples (past-only). Returns a dict of arrays
    ready for the training loop (fields mirror the official DygDataset)."""
    src = events["src"][:n]
    dst = events["dst"][:n]
    labels = events["label"][:n]

    builder = PastEgoGraphBuilder(src, dst, max_mask_len)
    node_feat_list, edge_feat_list = [], []
    Tmats, adjs, eadjs = [], [], []
    masks = []

    def sparse_mx_to_dense(mx):
        mx = mx.tocoo().astype(np.float32)
        indices = np.vstack((mx.row, mx.col)).astype(np.int64)
        values = mx.data
        shape = mx.shape
        return torch.sparse_coo_tensor(indices, values, shape).to_dense()

    def create_transition_matrix(vertex_adj):
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = list(zip(edge_index[0], edge_index[1]))
        row_index = [i for sub in edge_name for i in sub]
        col_index = np.repeat(np.arange(num_edge), 2)
        data = np.ones(num_edge * 2)
        return sp.csr_matrix((data, (row_index, col_index)),
                             shape=(vertex_adj.shape[0], num_edge))

    def create_edge_adj(vertex_adj):
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = list(zip(edge_index[0], edge_index[1]))
        edge_adj = np.zeros((num_edge, num_edge))
        for i in range(num_edge):
            for j in range(i, num_edge):
                if len(set(edge_name[i]) & set(edge_name[j])) != 0:
                    edge_adj[i, j] = 1
        adj = edge_adj + edge_adj.T
        np.fill_diagonal(adj, 1)
        return sp.csr_matrix(adj)

    def normalize(mx):
        rowsum = np.array(mx.sum(1)).astype("float")
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        return sp.diags(r_inv).dot(mx)

    for i in range(n):
        # G holds strictly past edges [0, i): add the previous event's edge
        if i > 0:
            builder.G.add_edge(int(src[i - 1]), int(dst[i - 1]), weight=i - 1)
        sub, node_mapping, s, d, _ = builder.build(i)
        # replace subgraph with new (contiguous) ids, paper's replace_subgraph
        new_sub = nx.Graph()
        new_node_features = np.zeros(len(sub.nodes))
        new_edge_features = {}
        for node, new_id in node_mapping.items():
            new_node_features[new_id] = int(node)
            new_sub.add_node(new_id)
        for u, v, data in sub.edges(data=True):
            ns, nt = node_mapping[u], node_mapping[v]
            new_sub.add_edge(ns, nt, weight=1)
            new_edge_features[(ns, nt)] = int(data["weight"])
        mask_len = new_node_features.shape[0]

        adj = nx.adjacency_matrix(new_sub)
        T = create_transition_matrix(adj.copy())
        T_dense = sparse_mx_to_dense(T)
        eadj = create_edge_adj(adj.copy())
        # edge feature matrix per T column order
        temp_T = T.toarray()
        num_edges = eadj.shape[0]
        edge_feature_matrix = np.zeros(num_edges)
        edge_idx = 0
        for col in range(temp_T.shape[1]):
            indices = np.where(temp_T[:, col] == 1)[0]
            i0, j0 = indices[0], indices[1]
            if (i0, j0) in new_edge_features:
                edge_feature_matrix[edge_idx] = new_edge_features[(i0, j0)]
            else:
                edge_feature_matrix[edge_idx] = new_edge_features[(j0, i0)]
            edge_idx += 1

        eadj_dense = sparse_mx_to_dense(normalize(eadj))
        adj_dense = sparse_mx_to_dense(normalize(adj + sp.eye(adj.shape[0])))

        node_feat_list.append(new_node_features)
        edge_feat_list.append(edge_feature_matrix)
        Tmats.append(T_dense)
        adjs.append(adj_dense)
        eadjs.append(eadj_dense)
        masks.append(mask_len)

    return {
        "nodefeatures": node_feat_list,
        "edgefeatures": edge_feat_list,
        "labels": labels.astype(np.int64),
        "Tmats": Tmats,
        "adjs": adjs,
        "eadjs": eadjs,
        "masks": masks,
    }


def _embed_features(samples: dict, input_dim: int, device: torch.device):
    """Build input_nodes_feature / input_edges_feature / pads (paper's
    DygDataset.get_data feature mapping)."""
    rng = np.random.RandomState(0)
    node_feats = samples["nodefeatures"]
    edge_feats = samples["edgefeatures"]
    flat_node = np.concatenate([a.astype(int) for a in node_feats])
    flat_edge = np.concatenate([a.astype(int) for a in edge_feats])
    uniq_node = np.unique(flat_node)
    uniq_edge = np.unique(flat_edge)
    n_nodes = int(max(uniq_node.size, np.max(uniq_node) + 1)) if uniq_node.size else 0
    n_edges = int(max(uniq_edge.size, np.max(uniq_edge) + 1)) if uniq_edge.size else 0
    Nfeatures = rng.uniform(0.0, 1.0, size=(max(n_nodes, 1), input_dim))
    Efeatures = rng.uniform(0.0, 1.0, size=(max(n_edges, 1), input_dim))

    max_mask_edge = max(len(a) for a in edge_feats)
    masks_edge = [len(a) for a in edge_feats]
    NS = len(edge_feats)
    mask_edge = np.ones((NS, max_mask_edge), dtype=bool)
    for i in range(NS):
        mask_edge[i, :masks_edge[i]] = False

    input_edges_pad = np.zeros((NS, max_mask_edge, input_dim))
    for i, indices in enumerate(edge_feats):
        indices = indices.astype(int)
        input_edges_pad[i, :len(indices), :] = Efeatures[indices]
    input_edges_feature = [torch.tensor(input_edges_pad[i, :masks_edge[i], :])
                           for i in range(NS)]

    max_mask_node = max(len(a) for a in node_feats)
    masks_node = [len(a) for a in node_feats]
    input_nodes_pad = np.zeros((NS, max_mask_node, input_dim))
    for i, indices in enumerate(node_feats):
        indices = indices.astype(int)
        input_nodes_pad[i, :len(indices), :] = Nfeatures[indices]
    input_nodes_feature = [torch.tensor(input_nodes_pad[i, :masks_node[i], :])
                           for i in range(NS)]

    out = {
        "input_nodes_feature": input_nodes_feature,
        "input_edges_feature": input_edges_feature,
        "input_edges_pad": torch.tensor(input_edges_pad, dtype=torch.float32),
        "labels": torch.tensor(samples["labels"], dtype=torch.float32),
        "Tmats": [t.float() for t in samples["Tmats"]],
        "adjs": [a.float() for a in samples["adjs"]],
        "eadjs": [e.float() for e in samples["eadjs"]],
        "mask_edge": torch.tensor(mask_edge, dtype=torch.bool),
    }
    return out, masks_edge


def _to_device(sample: dict, device: torch.device) -> dict:
    return {
        "input_nodes_feature": [t.to(device) for t in sample["input_nodes_feature"]],
        "input_edges_feature": [t.to(device) for t in sample["input_edges_feature"]],
        "input_edges_pad": sample["input_edges_pad"].to(device),
        "labels": sample["labels"].to(device),
        "Tmats": [t.to(device) for t in sample["Tmats"]],
        "adjs": [a.to(device) for a in sample["adjs"]],
        "eadjs": [e.to(device) for e in sample["eadjs"]],
        "mask_edge": sample["mask_edge"].to(device),
    }


def run_generaldyg(events: dict, splits: dict, *,
                   dataset: str, device: str = "cpu", seed: int = 42,
                   batch_size: int = 32, n_epochs: int = 20,
                   input_dim: int = 128, hidden_dim: int = 258,
                   n_heads: int = 4, n_layer: int = 6, drop_out: float = 0.3,
                   learning_rate: float = 1e-4, max_mask_len: int = 24,
                   log=print) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(events["event_id"])
    samples = build_samples(events, n, max_mask_len)
    data, masks_edge = _embed_features(samples, input_dim, torch.device("cpu"))

    dev = torch.device(device)
    config = argparse_like(input_dim, hidden_dim, n_heads, n_layer, drop_out,
                           learning_rate, batch_size, seed)
    gnn = CensNetCPU(input_dim, drop_out)
    transformer = TransformerBinaryClassifier(config, dev, hidden_size=hidden_dim)
    model = CombinedModel(gnn, transformer).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # official uses with_logits
    # NOTE: official train.py computes BCE-with-logits on the sigmoid output
    # (double sigmoid); we follow the official loss call to stay faithful.

    tr_s, tr_e = splits["train"]
    known = (data["labels"][:n].numpy() != -1)
    train_idx = np.where(known)[0]
    train_idx = train_idx[(train_idx >= tr_s) & (train_idx < tr_e)]
    eval_idx = np.arange(n)
    log(f"  train events (known, in-split): {len(train_idx)}/{n}")

    def run_epoch(idx, train: bool):
        order = idx.copy()
        if train:
            np.random.shuffle(order)
        model.train(train)
        pred = np.zeros(len(idx), dtype=np.float64)
        losses = []
        for s in range(0, len(order), batch_size):
            b = order[s:s + batch_size]
            batch = {k: [v[i] for i in b] if isinstance(v, list) else v[b]
                     for k, v in data.items()}
            batch = _to_device(batch, dev)
            optimizer.zero_grad()
            logits = model(batch["input_nodes_feature"],
                           batch["input_edges_feature"],
                           batch["input_edges_pad"], batch["eadjs"],
                           batch["adjs"], batch["Tmats"], batch["mask_edge"])
            y = batch["labels"]
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            pred[s:s + len(b)] = logits.detach().cpu().numpy().flatten()
        return pred, (float(np.mean(losses)) if losses else 0.0)

    for epoch in range(n_epochs):
        _, tr_loss = run_epoch(train_idx, train=True)
        log(f"  epoch {epoch} loss={tr_loss:.4f}")

    # score every event (known and unknown), map back to event order
    pred, _ = run_epoch(eval_idx, train=False)
    scores = np.full(n, 0.0, dtype=np.float64)
    scores[eval_idx] = pred
    return scores


def argparse_like(input_dim, hidden_dim, n_heads, n_layer, drop_out,
                  learning_rate, batch_size, seed):
    import argparse
    return argparse.Namespace(
        input_dim=input_dim, hidden_dim=hidden_dim, n_heads=n_heads,
        n_layer=n_layer, drop_out=drop_out, learning_rate=learning_rate,
        batch_size=batch_size, seed=seed)
