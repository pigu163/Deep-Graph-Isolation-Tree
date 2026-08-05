"""F-FADE: Frequency Factorization for Anomaly Detection in Edge Streams.

Faithful translation of the official repository
(https://github.com/snap-stanford/F-FADE, WSDM 2021, arXiv:2011.04723):
- ``UNION``: exponential-decay frequency table update + pruning (M, alpha).
- ``DETECT``: per-edge likelihood-ratio style score combining the edge
  frequency, its source/destination neighborhood frequencies, and the
  Poisson-rate embedding (frequency factorization) lambda_sd.
- ``F_FAC``: Poisson-likelihood embedding training (global at setup, local
  online updates) using RMSprop on static node embeddings.

Score direction: larger = more anomalous (ratio of observed frequency to
predicted Poisson rate).

Task adaptation (recorded in config):
- Events are grouped into discrete time slices instead of the repository's
  integer-tick assumption. Slice policy matches the MAD baseline
  (paysim/elliptic/midas native steps; IBM AML 1-hour bins).
- The original ``global`` F_FAC enumerates V x V candidate pairs; that is
  intractable on 500k-node graphs, so global F_FAC runs only when |V|^2 is
  small; otherwise the local (sampled) F_FAC is used for every update.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FF

DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------
# Poisson-rate embedding (frequency factorization), faithful to model.py
# --------------------------------------------------------------------------
class FadeEmbedding(nn.Module):
    def __init__(self, num_nodes: int, embedding_size: int) -> None:
        super().__init__()
        h_initial = FF.normalize(torch.rand(num_nodes + 1, embedding_size), dim=1)
        self.h_static = nn.Parameter(h_initial)
        self.embedding_size = embedding_size
        self.num_nodes = num_nodes
        self.Q = nn.Parameter(torch.rand(embedding_size, embedding_size))
        self.Q.requires_grad_(False)

    def forward(self, s_id, d_id):
        h_s = self.h_static[s_id, :]
        h_d = self.h_static[d_id, :]
        lambda_sd = torch.exp((torch.matmul(h_s, self.Q) * h_d).sum(dim=-1))
        return lambda_sd


# --------------------------------------------------------------------------
# UNION — frequency table update (faithful translation)
# --------------------------------------------------------------------------
def union_update(F, Act_S, delta_F, alpha, M, t_0, f_th):
    for (s, d), (t, f) in delta_F.items():
        Act_S.add((s, d))
        if (s, d) in F:
            t_, f_ = F[(s, d)]
            F[(s, d)] = (t, alpha ** (t - t_) * f_ + (1 - alpha) * f)
        else:
            F[(s, d)] = (t, (1 - alpha) * f)
    if M > 0:
        F_ = {k: v for k, v in sorted(
            F.items(), key=lambda item: -1 * alpha ** (t_0 - item[1][0]) * item[1][1])}
        F_value_list = list(F_.values())
        if len(F_value_list) >= M:
            f_th = alpha ** (t_0 - F_value_list[M - 1][0]) * F_value_list[M - 1][1]
    F_ = F.copy()
    for (s, d), (t, f) in F.items():
        if alpha ** (t_0 - t) * f < f_th:
            del F_[(s, d)]
            Act_S.discard((s, d))
    return F_, Act_S, f_th


# --------------------------------------------------------------------------
# DETECT — per-edge scoring (faithful translation)
# --------------------------------------------------------------------------
def detect_edges(model, F, delta_F, H, f_th):
    """Score every event of the current slice.

    Returns dict {(s, d): list[float]} with one score per event of that pair
    in the current slice (first occurrence -> first score, repeats -> others).
    """
    Xi_out: dict[int, set] = {}
    Xi_in: dict[int, set] = {}
    for (s, d) in delta_F.keys():
        Xi_out.setdefault(s, set()).add((s, d))
        Xi_in.setdefault(d, set()).add((s, d))

    s_set = set(s for s, _ in delta_F)
    d_set = set(d for _, d in delta_F)
    out: dict[tuple[int, int], list[float]] = {}

    for (s, d), (t, f) in delta_F.items():
        # f_sd (observed frequency)
        if (s, d) not in F:
            f_sd_first = f_th
        else:
            t_prime = F[(s, d)][0]
            f_sd_first = 1 / (t - t_prime - 1 + (1 / f))
        f_sd_others = f if f > 1 else 0.0

        # lambda_sd (predicted Poisson rate)
        with torch.no_grad():
            if s in H and d in H:
                lambda_sd = model(torch.tensor([s]), torch.tensor([d])).item()
            else:
                lambda_sd = f_th
        Sc_sd_first = 1 * f_sd_first / lambda_sd
        Sc_sd_others = 1 * f_sd_others / lambda_sd if f > 1 else 0.0

        # source neighborhood
        f_Xi_out_total = 0
        t_Xi_out_prime = 0
        in_F = False
        for (s_, d_) in Xi_out[s]:
            f_Xi_out_total += delta_F[(s_, d_)][1]
            if (s_, d_) in F:
                in_F = True
                t_Xi_out_prime = max(t_Xi_out_prime, F[(s_, d_)][0])
        if s in s_set:
            f_Xi_out_first = f_Xi_out_total
            f_Xi_out_others = f_Xi_out_total if f > 1 else 0.0
        else:
            if not in_F:
                f_Xi_out_first = f_th if s not in s_set else f_Xi_out_total
            else:
                f_Xi_out_first = (f_Xi_out_total if s not in s_set
                                  else 1 / (t - t_Xi_out_prime - 1 + (1 / f_Xi_out_total)))
            f_Xi_out_others = f_Xi_out_total if f > 1 else 0.0

        # destination neighborhood
        f_Xi_in_total = 0
        t_Xi_in_prime = 0
        in_F = False
        for (s_, d_) in Xi_in[d]:
            f_Xi_in_total += delta_F[(s_, d_)][1]
            if (s_, d_) in F:
                in_F = True
                t_Xi_in_prime = max(t_Xi_in_prime, F[(s_, d_)][0])
        if d in d_set:
            f_Xi_in_first = f_Xi_in_total
            f_Xi_in_others = f_Xi_in_total if f > 1 else 0.0
        else:
            if not in_F:
                f_Xi_in_first = f_th if d not in d_set else f_Xi_in_total
            else:
                f_Xi_in_first = (f_Xi_in_total if d not in d_set
                                 else 1 / (t - t_Xi_in_prime - 1 + (1 / f_Xi_in_total)))
            f_Xi_in_others = f_Xi_in_total if f > 1 else 0.0

        # predicted rates for neighborhoods
        lambda_out = 0.0
        with torch.no_grad():
            for (s_, d_) in Xi_out[s]:
                if s_ in H and d_ in H:
                    lambda_out += model(torch.tensor([s_]), torch.tensor([d_])).item()
                else:
                    lambda_out += f_th
        lambda_in = 0.0
        with torch.no_grad():
            for (s_, d_) in Xi_in[d]:
                if s_ in H and d_ in H:
                    lambda_in += model(torch.tensor([s_]), torch.tensor([d_])).item()
                else:
                    lambda_in += f_th
        Sc_out_first = 1 * f_Xi_out_first / lambda_out if lambda_out > 0 else 0.0
        Sc_out_others = 1 * f_Xi_out_others / lambda_out if lambda_out > 0 else 0.0
        Sc_in_first = 1 * f_Xi_in_first / lambda_in if lambda_in > 0 else 0.0
        Sc_in_others = 1 * f_Xi_in_others / lambda_in if lambda_in > 0 else 0.0

        Sc_first = max(Sc_sd_first, Sc_out_first, Sc_in_first)
        Sc_others = max(Sc_sd_others, Sc_out_others, Sc_in_others)
        scores = [Sc_first] + [Sc_others] * (int(f) - 1)
        out[(s, d)] = scores
    return out


# --------------------------------------------------------------------------
# F_FAC — embedding training (global at setup / local online)
# --------------------------------------------------------------------------
def _train_batch(model, optimizer, batch, device):
    batch_size = len(batch)
    src = torch.from_numpy(np.array([x[0] for x in batch], dtype=np.int64))
    dst = torch.from_numpy(np.array([x[1] for x in batch], dtype=np.int64))
    freq = torch.from_numpy(np.array([x[2] for x in batch], dtype=np.float32)).view(batch_size, 1)
    optimizer.zero_grad()
    lam = model(src, dst).view(batch_size, 1)
    loss = (torch.log(lam) + freq / lam).sum() / batch_size
    loss.backward()
    optimizer.step()
    return loss.item()


def f_fac_global(model, F, H, V, f_th, epochs, max_batch_size, device):
    """Global training over V x V pairs (only feasible for small |V|)."""
    H_ = H.copy()
    for v in V:
        if v not in H_:
            H_.add(v)
            h_tmp = FF.normalize(torch.rand(1, model.embedding_size), dim=1).view(-1)
            model.h_static[v].data.copy_(h_tmp)
    optimizer = optim_rmsprop([model.h_static], lr=1e-3)
    V_list = list(V)
    pair_set = set(F.keys())
    train_data = [(s, d, f) for (s, d), (t, f) in F.items()]
    for s in V_list:
        for d in V_list:
            if s != d and (s, d) not in pair_set:
                train_data.append((s, d, f_th))
    train_num = len(train_data)
    for _ in range(epochs):
        random.shuffle(train_data)
        for bs in range(0, train_num, max_batch_size):
            _train_batch(model, optimizer, train_data[bs:bs + max_batch_size], device)
    return H_


def f_fac_local(model, F, Act_S, H, V, f_th, online_train_steps, max_batch_size, device):
    """Local online training over sampled node pairs."""
    H_ = H.copy()
    for v in V:
        if v not in H_:
            H_.add(v)
            h_tmp = FF.normalize(torch.rand(1, model.embedding_size), dim=1).view(-1)
            model.h_static[v].data.copy_(h_tmp)
    optimizer = optim_rmsprop([model.h_static], lr=2e-3)
    V_Omega_p = set()
    for (s, d) in Act_S:
        V_Omega_p.add(s)
        V_Omega_p.add(d)
    V_prime = set(V) - V_Omega_p
    for _ in range(online_train_steps):
        V_tmp = set()
        if len(V_Omega_p) > 0:
            V_tmp.update(random.sample(list(V_Omega_p),
                                       min(max_batch_size, len(V_Omega_p))))
        if len(V_prime) > 0:
            V_tmp.update(random.sample(list(V_prime),
                                       min(4 * max_batch_size, len(V_prime))))
        train_data = []
        for s in V_tmp:
            for d in V_tmp:
                if s != d:
                    train_data.append((s, d, F[(s, d)][1] if (s, d) in F else f_th))
        _train_batch(model, optimizer, train_data, device)
    return H_


def optim_rmsprop(params, lr):
    return torch.optim.RMSprop(params, lr=lr)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def run_f_fade(events: dict, slice_ids: np.ndarray, n_slices: int,
               embedding_size: int = 200, batch_size: int = 32,
               t_setup: int | None = None, w_upd: int | None = None,
               alpha: float = 0.999, M: int = 100, t_th: int = 120,
               epochs: int = 5, online_train_steps: int = 10,
               seed: int = 42, max_global_pairs: int = 4_000_000,
               log=print) -> np.ndarray:
    """Run F-FADE over the event stream; return per-event scores.

    events: dict with 'src', 'dst' arrays (int64, same length).
    slice_ids: per-event slice index (0-based, sorted by time).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    src = events["src"]
    dst = events["dst"]
    n = len(src)
    # node id -> contiguous (keep original ids; model indexes by id + 1)
    node_ids = np.unique(np.concatenate([src, dst]))
    num_nodes = int(node_ids.max()) + 1  # F-FADE uses raw ids as indices

    # group events per slice (preserving in-slice event order)
    slice_events: dict[int, list[int]] = {}
    for i in range(n):
        slice_events.setdefault(int(slice_ids[i]), []).append(i)

    T_max = int(n_slices)
    if t_setup is None:
        t_setup = min(8000, max(1, int(0.3 * T_max)))
    if w_upd is None:
        w_upd = min(720, max(1, int(0.05 * T_max)))
    f_th = 1.0 / t_th

    model = FadeEmbedding(num_nodes=num_nodes, embedding_size=embedding_size)
    model.eval()

    Act_S = set()
    F: dict[tuple[int, int], tuple[int, float]] = {}
    H: set[int] = set()
    V: set[int] = set()
    scores = np.full(n, 0.0, dtype=np.float64)
    k = 0

    for t in range(1, T_max + 1):
        idxs = slice_events.get(t - 1, [])
        # Step 3: aggregate per-pair frequency within the slice
        delta_F: dict[tuple[int, int], tuple[int, int]] = {}
        pair_order: dict[tuple[int, int], list[int]] = {}
        for i in idxs:
            key = (int(src[i]), int(dst[i]))
            if key not in delta_F:
                delta_F[key] = (t, 1)
                pair_order[key] = [i]
            else:
                _, f_ = delta_F[key]
                delta_F[key] = (t, f_ + 1)
                pair_order[key].append(i)
        if not delta_F:
            continue

        # Step 4: score events after the setup phase
        if t > t_setup:
            pair_scores = detect_edges(model, F, delta_F, H, f_th)
            for key, sc_list in pair_scores.items():
                for i, sc in zip(pair_order[key], sc_list):
                    scores[i] = sc

        # Step 5: frequency table update
        F, Act_S, f_th = union_update(F, Act_S, delta_F, alpha, M, t, f_th)

        # Steps 6-9: embedding (re)training
        if t == t_setup + k * w_upd:
            V = set()
            for (s, d) in F.keys():
                V.add(s)
                V.add(d)
            if k == 0 and len(V) ** 2 <= max_global_pairs:
                H = f_fac_global(model, F, H, V, f_th, epochs, batch_size, DEVICE)
                log(f"  global F_FAC at slice {t}: |V|={len(V)}")
            else:
                if k == 0:
                    log(f"  global F_FAC skipped at slice {t}: |V|^2="
                        f"{len(V) ** 2} > {max_global_pairs} (intractable); "
                        "using local F_FAC instead")
                H = f_fac_local(model, F, Act_S, H, V, f_th,
                                online_train_steps, batch_size, DEVICE)
            k += 1
            Act_S = set()

    return scores


def make_slice_ids(timestamps: np.ndarray, policy_kind: str,
                   bin_width: float = 3600.0) -> np.ndarray:
    """Map continuous timestamps to 0-based ordered slice ids.

    policy_kind: 'native-step' (timestamp is already the step) or
    'fixed-width' (floor((ts - min) / bin_width)).
    """
    if policy_kind == "native-step":
        t = timestamps.astype(np.int64)
        return t - t.min()
    if policy_kind == "fixed-width":
        tmin = float(timestamps.min())
        return np.floor((timestamps - tmin) / bin_width).astype(np.int64)
    raise ValueError(f"unknown policy {policy_kind!r}")
