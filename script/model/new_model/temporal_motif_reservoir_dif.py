"""Causal, training-free Temporal Motif Reservoir Isolation Forest.

The implementation deliberately consumes the repository's canonical event
arrays.  It has no optimiser or learnt readout: all reservoir/projection
weights are seeded once, frozen NumPy arrays.  The detector returns one score
per *transaction*; node states are private streaming context.
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - CUDA backend is optional
    torch = None

try:
    from numba import njit
except ImportError:  # pragma: no cover - the project environment has numba
    njit = None


class MotifType(str, Enum):
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    RAPID_CHAIN = "rapid_chain"
    CYCLE = "cycle"


MOTIFS = tuple(MotifType)


@dataclass
class ReservoirConfig:
    num_reservoirs: int = 8
    state_dim: int = 64
    embedding_dim: int = 64
    time_dim: int = 16
    leak_rate: float = .20
    target_operator_norm: float = .95
    contraction_tolerance: float = 1e-6
    recurrent_layout: str = "orthogonal"
    recurrent_density: float = .10
    recurrent_group_count: int = 4
    grouped_base_layout: str = "random_dense_uniform"
    input_scale: float = .50
    bias_scale: float = .10
    projection_scale: float = 1.0
    max_recent_edges_per_node: int = 128
    max_recent_edges_per_pair: int = 32
    temporal_window_seconds: float = 604800.
    rapid_chain_window_seconds: float = 86400.
    cycle_window_seconds: float = 604800.
    context_decay_beta: float = 1e-5
    context_shrinkage_alpha: float = 5.
    motif_neighbor_limit: int = 64
    trees_per_reservoir: int = 25
    subsample_size: int = 256
    max_depth: int | None = None
    max_split_retries: int = 8
    split_eps: float = 1e-8
    # Contiguous chronological mini-batches share the pre-batch state. Repeated
    # endpoint updates are averaged before the batch is committed.
    batch_size: int = 256
    state_update_mode: str = "batch"
    device: str = "cpu"


def _operator_norm(matrix: np.ndarray) -> float:
    value = float(np.linalg.svd(matrix.astype(np.float64), compute_uv=False)[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("recurrent matrix has an invalid or zero operator norm")
    return value


def _raw_layout(name: str, size: int, rng: np.random.Generator, cfg: ReservoirConfig) -> np.ndarray:
    scale = 1.0 / math.sqrt(size)
    if name == "random_dense_uniform":
        return rng.uniform(-math.sqrt(3) * scale, math.sqrt(3) * scale, (size, size))
    if name == "random_dense_gaussian":
        return rng.normal(0., scale, (size, size))
    if name == "orthogonal":
        q, r = np.linalg.qr(rng.normal(size=(size, size)))
        return q * np.sign(np.diag(r)).clip(min=-1, max=1)[None, :]
    if name in {"symmetric", "antisymmetric"}:
        raw = rng.normal(0., scale, (size, size))
        return (raw + raw.T) / 2 if name == "symmetric" else (raw - raw.T) / 2
    if name == "ring":
        raw = np.zeros((size, size), dtype=np.float64)
        raw[np.arange(size), (np.arange(size) - 1) % size] = rng.choice([-1., 1.], size=size)
        return raw
    if name == "sparse_random":
        raw = rng.normal(0., scale, (size, size))
        raw[rng.random((size, size)) > cfg.recurrent_density] = 0.
        # A nonzero diagonal avoids accidental zero rows at very low density.
        raw[np.arange(size), np.arange(size)] += 1e-3
        return raw
    if name == "block_diagonal_grouped":
        groups = max(1, min(cfg.recurrent_group_count, size))
        boundaries = np.linspace(0, size, groups + 1, dtype=int)
        raw = np.zeros((size, size), dtype=np.float64)
        base = cfg.grouped_base_layout
        if base == "block_diagonal_grouped":
            base = "random_dense_uniform"
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            raw[start:end, start:end] = _raw_layout(base, end - start, rng, cfg)
        return raw
    raise ValueError(f"unknown recurrent layout: {name}")


class ContractiveReservoir:
    """One independently seeded, frozen endpoint-wise echo-state reservoir."""
    def __init__(self, edge_dim: int, cfg: ReservoirConfig, seed: int):
        if not (0 < cfg.leak_rate <= 1 and 0 < cfg.target_operator_norm < 1):
            raise ValueError("leak_rate and target_operator_norm must be strictly in (0, 1)")
        self.cfg, self.seed, self.edge_dim = cfg, int(seed), int(edge_dim)
        self.rng = np.random.default_rng(seed)
        self.event_input_dim = edge_dim + 2 * cfg.time_dim + 2
        self.endpoint_input_dim = self.event_input_dim + 1  # final value is the endpoint role
        size = cfg.state_dim
        raw = _raw_layout(cfg.recurrent_layout, size, self.rng, cfg)
        self.w_res = cfg.target_operator_norm * raw / _operator_norm(raw)
        self.w_in = self.rng.normal(
            0., cfg.input_scale / math.sqrt(self.endpoint_input_dim),
            (size, self.endpoint_input_dim))
        self.bias = self.rng.uniform(-cfg.bias_scale, cfg.bias_scale, size)
        qdim = 2 * size + self.event_input_dim
        self.w_event = self.rng.normal(0., cfg.projection_scale / math.sqrt(qdim), (cfg.embedding_dim, qdim))
        self.b_event = self.rng.uniform(-cfg.bias_scale, cfg.bias_scale, cfg.embedding_dim)
        self.states: dict[int, np.ndarray] = {}
        self.report = self.verify()

    def verify(self) -> dict[str, Any]:
        norm = _operator_norm(self.w_res)
        kappa = (1. - self.cfg.leak_rate) + self.cfg.leak_rate * norm
        passed = norm <= self.cfg.target_operator_norm + self.cfg.contraction_tolerance and norm < 1. and kappa < 1.
        if not passed:
            raise ValueError(f"reservoir contraction verification failed: norm={norm}, kappa={kappa}")
        return {"seed": self.seed, "layout": self.cfg.recurrent_layout,
                "leak_rate": self.cfg.leak_rate, "target_operator_norm": self.cfg.target_operator_norm,
                "measured_operator_norm": norm, "analytical_kappa": kappa,
                "verification_tolerance": self.cfg.contraction_tolerance, "verification_passed": True}

    def reset_state(self) -> None:
        self.states.clear()

    def _time(self, delta: np.ndarray) -> np.ndarray:
        half = self.cfg.time_dim // 2
        freqs = np.geomspace(1e-4, 1., num=half)
        x = np.log1p(np.maximum(delta, 0.))[:, None] * freqs[None, :]
        enc = np.concatenate([np.sin(x), np.cos(x)], axis=1)
        return enc[:, :self.cfg.time_dim]

    def encode_group(self, src: np.ndarray, dst: np.ndarray, feat: np.ndarray,
                     src_gap: np.ndarray, dst_gap: np.ndarray,
                     src_new: np.ndarray, dst_new: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Encode one chronological mini-batch and return deferred updates.

        Shapes: B=batch size, D=node state, A=event input, H=embedding.
        Endpoint recurrence is node-separable, so averaging repeated-node
        candidates preserves the verified ESN contraction bound.
        """
        zero = np.zeros(self.cfg.state_dim)
        old_src = np.stack([self.states.get(int(x), zero) for x in src])  # [B, D]
        old_dst = np.stack([self.states.get(int(x), zero) for x in dst])  # [B, D]
        event_input = np.concatenate([
            feat, self._time(src_gap), self._time(dst_gap),
            src_new[:, None], dst_new[:, None],
        ], axis=1)  # [B, A]

        # Pair embedding retains ordered sender/receiver history: [B, 2D + A] -> [B, H].
        pair_input = np.concatenate([old_src, old_dst, event_input], axis=1)
        embed = np.tanh(pair_input @ self.w_event.T + self.b_event)
        embed = (embed - embed.mean(axis=1, keepdims=True)) / np.sqrt(embed.var(axis=1, keepdims=True) + 1e-5)

        # Flatten endpoints: [B,D] + [B,D] -> [2B,D]. Role 0=sender, 1=receiver.
        old_endpoint = np.concatenate([old_src, old_dst], axis=0)
        role = np.concatenate([np.zeros(len(src)), np.ones(len(dst))])[:, None]
        endpoint_input = np.concatenate([
            np.concatenate([event_input, event_input], axis=0), role,
        ], axis=1)  # [2B, A + 1]
        candidate = np.tanh(
            endpoint_input @ self.w_in.T + old_endpoint @ self.w_res.T + self.bias)
        fresh = ((1. - self.cfg.leak_rate) * old_endpoint
                 + self.cfg.leak_rate * candidate)  # [2B, D]
        pending_nodes = np.concatenate([src, dst]).astype(np.int64, copy=False)
        return embed, (pending_nodes, fresh)

    def commit(self, pending: tuple[np.ndarray, np.ndarray]) -> None:
        """Mean repeated-node candidates and commit one state per active node."""
        nodes, values = pending
        if len(nodes) == 0:
            return
        order = np.argsort(nodes, kind="stable")
        sorted_nodes = nodes[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_nodes)) + 1]
        unique_nodes = sorted_nodes[starts]
        totals = np.add.reduceat(values[order], starts, axis=0)
        counts = np.diff(np.r_[starts, len(nodes)])
        for node, state in zip(unique_nodes, totals / counts[:, None]):
            self.states[int(node)] = state


@dataclass
class _Record:
    ts: float
    src: int
    dst: int
    embedding: np.ndarray  # [R, H]


class TemporalMotifState:
    """Bounded, past-only topology and embedding context.  One shared graph."""
    def __init__(self, cfg: ReservoirConfig, reservoirs: int, embedding_dim: int):
        self.cfg, self.r, self.h = cfg, reservoirs, embedding_dim
        self.last: dict[int, float] = {}
        self.out: dict[int, deque[_Record]] = defaultdict(lambda: deque(maxlen=cfg.max_recent_edges_per_node))
        self.inc: dict[int, deque[_Record]] = defaultdict(lambda: deque(maxlen=cfg.max_recent_edges_per_node))
        self.pair: dict[tuple[int, int], deque[_Record]] = defaultdict(lambda: deque(maxlen=cfg.max_recent_edges_per_pair))
        self.global_sum = np.zeros((reservoirs, embedding_dim)); self.global_count = 0

    def _recent(self, records: deque[_Record], now: float) -> list[_Record]:
        cutoff = now - self.cfg.temporal_window_seconds
        return [x for x in records if x.ts >= cutoff]

    def _context(self, motif: MotifType, u: int, v: int, now: float) -> list[_Record]:
        if motif is MotifType.FAN_OUT: return self._recent(self.out[u], now)
        if motif is MotifType.FAN_IN: return self._recent(self.inc[v], now)
        if motif is MotifType.RAPID_CHAIN: return [x for x in self.inc[u] if x.ts >= now - self.cfg.rapid_chain_window_seconds]
        ans = self._recent(self.pair[(u, v)], now)
        for edge in list(self.out[v])[-self.cfg.motif_neighbor_limit:]:
            if edge.ts >= now - self.cfg.cycle_window_seconds:
                ans.append(edge)
        return ans

    def features(self, u: int, v: int, now: float, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, increments = np.empty((self.r, 4)), np.empty(4)
        # Shared by all four motifs for this query; computing it once avoids
        # repeated division/allocation in the per-event hot path.
        global_ctx = self.global_sum / max(self.global_count, 1)
        for mi, motif in enumerate(MOTIFS):
            records = self._context(motif, u, v, now)
            if motif is MotifType.FAN_OUT:
                increments[mi] = float(v not in {x.dst for x in records})
            elif motif is MotifType.FAN_IN:
                increments[mi] = float(u not in {x.src for x in records})
            elif motif is MotifType.RAPID_CHAIN:
                increments[mi] = sum(math.exp(-1e-5 * max(0., now-x.ts)) for x in records)
            else:
                # v -> w and w -> u, searched in bounded historical adjacency.
                increments[mi] = sum(1. for a in records[-self.cfg.motif_neighbor_limit:]
                    for b in list(self.out[a.dst])[-self.cfg.motif_neighbor_limit:]
                    if a.src == v and b.dst == u and b.ts < now)
            if records:
                ages = np.array([max(0., now-x.ts) for x in records])
                weights = np.exp(-self.cfg.context_decay_beta * ages); weights /= weights.sum()
                ctx = np.sum(np.stack([x.embedding for x in records]) * weights[:, None, None], axis=0)
                count = len(records)
            else:
                ctx = np.zeros((self.r, self.h)); count = 0
            rho = count / (count + self.cfg.context_shrinkage_alpha)
            ctx = rho * ctx + (1. - rho) * global_ctx
            distances[:, mi] = np.linalg.norm(embeds - ctx, axis=1) / math.sqrt(self.h)
        return distances, increments

    def commit_batch(self, src: np.ndarray, dst: np.ndarray, ts: np.ndarray,
                     embeds: np.ndarray) -> None:
        """Commit one complete chronological mini-batch after scoring it."""
        for i, (u, v) in enumerate(zip(src, dst)):
            now = float(ts[i])
            rec = _Record(now, int(u), int(v), embeds[:, i].copy())
            self.out[int(u)].append(rec); self.inc[int(v)].append(rec); self.pair[(int(u), int(v))].append(rec)
            self.last[int(u)] = now; self.last[int(v)] = now
            self.global_sum += rec.embedding; self.global_count += 1


@dataclass
class Leaf: size: int
@dataclass
class Internal: motif: int; threshold: float; left: Any; right: Any


def _c(n: int) -> float:
    return 0. if n <= 1 else 2. * (math.log(n - 1) + 0.5772156649) - 2. * (n - 1) / n


class MotifIsolationTree:
    def __init__(self, cfg: ReservoirConfig, rng: np.random.Generator): self.cfg, self.rng, self.root = cfg, rng, None
    def fit(self, dist: np.ndarray, inc: np.ndarray) -> "MotifIsolationTree":
        max_depth = self.cfg.max_depth if self.cfg.max_depth is not None else math.ceil(math.log2(max(2, len(dist))))
        self.root = self._build(np.arange(len(dist)), dist, inc, 0, max_depth); return self
    def _build(self, ids, dist, inc, depth, max_depth):
        if len(ids) <= 1 or depth >= max_depth: return Leaf(len(ids))
        for _ in range(self.cfg.max_split_retries):
            motif = int(self.rng.integers(4)); values = dist[ids, motif] + np.log1p(np.maximum(inc[ids, motif], 0.))
            lo, hi = float(values.min()), float(values.max())
            if hi - lo < self.cfg.split_eps: continue
            threshold = float(self.rng.uniform(lo, hi)); left = ids[values < threshold]
            if 0 < len(left) < len(ids):
                right = ids[values >= threshold]
                return Internal(motif, threshold, self._build(left, dist, inc, depth + 1, max_depth), self._build(right, dist, inc, depth + 1, max_depth))
        return Leaf(len(ids))
    def path(self, d: np.ndarray, inc: np.ndarray) -> float:
        node, depth = self.root, 0
        while isinstance(node, Internal):
            value = d[node.motif] + math.log1p(max(0., inc[node.motif]))
            node = node.left if value < node.threshold else node.right; depth += 1
        return depth + _c(node.size)


def _pack_forest(forests: list[list[MotifIsolationTree]]) -> tuple[np.ndarray, ...]:
    """Lower Python tree objects to contiguous arrays for batch inference."""
    motif: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leaf_size: list[int] = []
    roots: list[int] = []
    tree_reservoir: list[int] = []

    def add(node: Any) -> int:
        pos = len(motif)
        motif.append(-1); threshold.append(0.); left.append(-1); right.append(-1); leaf_size.append(0)
        if isinstance(node, Leaf):
            leaf_size[pos] = node.size
        else:
            motif[pos] = node.motif; threshold[pos] = node.threshold
            left[pos] = add(node.left); right[pos] = add(node.right)
        return pos

    for reservoir_id, forest in enumerate(forests):
        for tree in forest:
            roots.append(add(tree.root)); tree_reservoir.append(reservoir_id)
    return (
        np.asarray(motif, dtype=np.int8), np.asarray(threshold, dtype=np.float64),
        np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32),
        np.asarray(leaf_size, dtype=np.int32), np.asarray(roots, dtype=np.int32),
        np.asarray(tree_reservoir, dtype=np.int16),
    )


if njit is not None:
    @njit(cache=True)
    def _score_packed_forest(dist, inc, motif, threshold, left, right, leaf_size,
                             roots, tree_reservoir, normalizer):
        out = np.empty(dist.shape[0], dtype=np.float64)
        for row in range(dist.shape[0]):
            path_sum = 0.0
            for tree_id in range(roots.shape[0]):
                node = roots[tree_id]
                depth = 0
                while motif[node] >= 0:
                    m = motif[node]
                    value = dist[row, tree_reservoir[tree_id], m]
                    x = inc[row, m]
                    if x > 0.0:
                        value += math.log1p(x)
                    node = left[node] if value < threshold[node] else right[node]
                    depth += 1
                size = leaf_size[node]
                correction = 0.0 if size <= 1 else 2.0 * (math.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size
                path_sum += depth + correction
            out[row] = 2.0 ** (-(path_sum / roots.shape[0]) / normalizer)
        return out
else:
    _score_packed_forest = None


class TemporalMotifReservoirDIF:
    """Fit trees on chronological fit events; score any chronological stream."""
    def __init__(self, edge_dim: int, config: ReservoirConfig | None = None, seed: int = 42):
        self.cfg = config or ReservoirConfig(); self.edge_dim, self.seed = edge_dim, seed
        if self.cfg.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if self.cfg.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is unavailable")
        self.reservoirs = [ContractiveReservoir(edge_dim, self.cfg, seed + 1009*i) for i in range(self.cfg.num_reservoirs)]
        self.forests: list[list[MotifIsolationTree]] = []
        self._packed_forest: tuple[np.ndarray, ...] | None = None
        # Constructed lazily because it is a runtime cache, not model state to pickle.
        self._cuda_weights: tuple[Any, Any, Any, Any, Any] | None = None
        self._cuda_staging: dict[str, Any] = {}

    def _score_batch(self, distances: np.ndarray, increments: np.ndarray) -> np.ndarray:
        if self._packed_forest is None:
            self._packed_forest = _pack_forest(self.forests)
        if _score_packed_forest is not None:
            return _score_packed_forest(distances, increments, *self._packed_forest,
                                        max(_c(self.cfg.subsample_size), 1e-12))
        return np.asarray([2. ** (-np.mean([tree.path(d[ri], inc) for ri, forest in enumerate(self.forests) for tree in forest]) /
                            max(_c(self.cfg.subsample_size), 1e-12))
                           for d, inc in zip(distances, increments)])

    def _cuda_from_numpy(self, name: str, values: np.ndarray):
        """Stage a dense batch through reusable pinned host memory."""
        if torch is None:
            raise RuntimeError("torch is required for CUDA execution")
        values = np.ascontiguousarray(values, dtype=np.float32)
        stage = self._cuda_staging.get(name)
        if stage is None or tuple(stage.shape) != tuple(values.shape):
            stage = torch.empty(values.shape, dtype=torch.float32, pin_memory=True)
            self._cuda_staging[name] = stage
        stage.copy_(torch.from_numpy(values))
        return stage.to("cuda", non_blocking=True)

    def _encode_cuda_bank(self, src: np.ndarray, dst: np.ndarray, feat: np.ndarray,
                          src_gap: np.ndarray, dst_gap: np.ndarray,
                          src_new: np.ndarray, dst_new: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Run all independent endpoint-wise reservoirs in one CUDA bank.

        Shapes: R=reservoirs, B=events, D=node state, A=event input,
        P=A+1 endpoint input, H=embedding. Only dense algebra runs on CUDA;
        sparse node dictionaries and motif context remain on CPU.
        """
        if torch is None:  # defensive; __init__ already validates CUDA
            raise RuntimeError("torch is required for CUDA execution")
        r_count, state_dim = len(self.reservoirs), self.cfg.state_dim
        zero = np.zeros(state_dim)
        old_src = np.stack([
            np.stack([reservoir.states.get(int(node), zero) for node in src])
            for reservoir in self.reservoirs
        ]).astype(np.float32, copy=False)  # [R, B, D]
        old_dst = np.stack([
            np.stack([reservoir.states.get(int(node), zero) for node in dst])
            for reservoir in self.reservoirs
        ]).astype(np.float32, copy=False)  # [R, B, D]

        # Event inputs are shared by the R frozen reservoirs: [B, A].
        base = self.reservoirs[0]
        event_input = np.concatenate([
            feat, base._time(src_gap), base._time(dst_gap),
            src_new[:, None], dst_new[:, None],
        ], axis=1).astype(np.float32, copy=False)

        if self._cuda_weights is None:
            self._cuda_weights = tuple(
                torch.as_tensor(np.stack(values).astype(np.float32), device="cuda")
                for values in (
                    [reservoir.w_event for reservoir in self.reservoirs],
                    [reservoir.b_event for reservoir in self.reservoirs],
                    [reservoir.w_in for reservoir in self.reservoirs],
                    [reservoir.w_res for reservoir in self.reservoirs],
                    [reservoir.bias for reservoir in self.reservoirs],
                )
            )
        w_event, b_event, w_in, w_res, bias = self._cuda_weights

        # Ordered pair projection: [R,B,2D+A] @ [R,2D+A,H] -> [R,B,H].
        pair_state = self._cuda_from_numpy(
            "pair_state", np.concatenate([old_src, old_dst], axis=2))
        event_a = self._cuda_from_numpy("event_input", event_input).unsqueeze(0).expand(r_count, -1, -1)
        pair_input = torch.cat([pair_state, event_a], dim=2)
        embeds = torch.tanh(torch.bmm(pair_input, w_event.transpose(1, 2)) + b_event[:, None, :])
        embeds = (embeds - embeds.mean(dim=2, keepdim=True)) / torch.sqrt(embeds.var(dim=2, keepdim=True, correction=0) + 1e-5)

        # Endpoint recurrence: [R,2B,P] and [R,2B,D] -> [R,2B,D].
        old_endpoint = self._cuda_from_numpy(
            "endpoint_state", np.concatenate([old_src, old_dst], axis=1))
        role = np.concatenate([np.zeros(len(src)), np.ones(len(dst))])[:, None]
        endpoint_input = np.concatenate([
            np.concatenate([event_input, event_input], axis=0), role,
        ], axis=1).astype(np.float32, copy=False)
        endpoint_a = self._cuda_from_numpy(
            "endpoint_input", endpoint_input).unsqueeze(0).expand(r_count, -1, -1)
        candidate = torch.tanh(
            torch.bmm(endpoint_a, w_in.transpose(1, 2))
            + torch.bmm(old_endpoint, w_res.transpose(1, 2)) + bias[:, None, :])
        fresh = ((1. - self.cfg.leak_rate) * old_endpoint
                 + self.cfg.leak_rate * candidate).cpu().numpy()
        pending_nodes = np.concatenate([src, dst]).astype(np.int64, copy=False)
        pending = [(pending_nodes, fresh[i]) for i in range(r_count)]
        return embeds.cpu().numpy(), pending

    @property
    def contraction_report(self):
        return [dict(reservoir_id=i, **r.verify()) for i, r in enumerate(self.reservoirs)]

    def _stream(self, data: dict, collect: bool, score: bool) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
        n = len(data["event_id"]); feat = np.asarray(data["edge_feat"], dtype=np.float64)
        temporal = TemporalMotifState(self.cfg, len(self.reservoirs), self.cfg.embedding_dim)
        for r in self.reservoirs: r.reset_state(); r.verify()
        cache_d = [np.empty((n, 4)) for _ in self.reservoirs] if collect else None
        cache_i = np.empty((n, 4)) if collect else None
        scores = np.empty(n) if score else None
        order = np.lexsort((data["event_id"], data["timestamp"]))
        batch_size = int(self.cfg.batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        pos = 0
        while pos < n:
            end = min(pos + batch_size, n)
            # All links read the pre-batch state. Endpoint candidates are
            # averaged by node and committed only after batch scoring.
            ix = order[pos:end]; src, dst = data["src"][ix], data["dst"][ix]
            now = np.asarray(data["timestamp"][ix], dtype=np.float64)
            src_gap = np.array([float(now[i]) - temporal.last.get(int(x), float(now[i])) for i, x in enumerate(src)])
            dst_gap = np.array([float(now[i]) - temporal.last.get(int(x), float(now[i])) for i, x in enumerate(dst)])
            src_new = np.array([float(int(x) not in temporal.last) for x in src]); dst_new = np.array([float(int(x) not in temporal.last) for x in dst])
            if self.cfg.device == "cuda":
                embeds, pending_groups = self._encode_cuda_bank(
                    src, dst, feat[ix], src_gap, dst_gap, src_new, dst_new)
            else:
                emb_groups, pending_groups = [], []
                for r in self.reservoirs:
                    emb, pending = r.encode_group(src, dst, feat[ix], src_gap, dst_gap, src_new, dst_new)
                    emb_groups.append(emb); pending_groups.append(pending)
                embeds = np.stack(emb_groups)
            batch_d = np.empty((len(ix), len(self.reservoirs), 4))
            batch_i = np.empty((len(ix), 4))
            for j, row in enumerate(ix):
                d, inc = temporal.features(int(src[j]), int(dst[j]), float(now[j]), embeds[:, j])
                batch_d[j] = d; batch_i[j] = inc
                if collect:
                    for ri in range(len(self.reservoirs)): cache_d[ri][row] = d[ri]
                    cache_i[row] = inc
            if score:
                scores[ix] = self._score_batch(batch_d, batch_i)
            for r, pending in zip(self.reservoirs, pending_groups): r.commit(pending)
            temporal.commit_batch(src, dst, now, embeds)
            pos = end
        return scores, [*cache_d, cache_i] if collect else None

    def fit(self, data: dict) -> "TemporalMotifReservoirDIF":
        self.contraction_report
        _, cached = self._stream(data, collect=True, score=False); assert cached is not None
        increments = cached[-1]; self.forests = []
        for ri, distances in enumerate(cached[:-1]):
            rng = np.random.default_rng(self.seed + 7919 * (ri + 1)); trees = []
            for _ in range(self.cfg.trees_per_reservoir):
                take = min(self.cfg.subsample_size, len(distances)); ids = rng.choice(len(distances), size=take, replace=False)
                trees.append(MotifIsolationTree(self.cfg, rng).fit(distances[ids], increments[ids]))
            self.forests.append(trees)
        self._packed_forest = _pack_forest(self.forests)
        return self

    def score(self, data: dict) -> np.ndarray:
        if not self.forests: raise RuntimeError("model must be fitted before scoring")
        scores, _ = self._stream(data, collect=False, score=True); assert scores is not None
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)): raise FloatingPointError("non-finite score")
        return scores

    def save(self, path: str | Path) -> None:
        # CUDA tensors are a derived runtime cache; keep serialized models portable.
        cache, staging = self._cuda_weights, self._cuda_staging
        self._cuda_weights, self._cuda_staging = None, {}
        try:
            with open(path, "wb") as f: pickle.dump(self, f)
        finally:
            self._cuda_weights, self._cuda_staging = cache, staging
    @classmethod
    def load(cls, path: str | Path) -> "TemporalMotifReservoirDIF":
        with open(path, "rb") as f: model = pickle.load(f)
        for r in model.reservoirs: r.verify()
        return model
    def metadata(self) -> dict[str, Any]:
        return {"model_name": "TemporalMotifReservoirDIF", "version": "0.5.0", "detection_level": "transaction", "causal": True,
                "trained_neural_parameters": False, "strict_active_endpoint_contraction": True,
                "execution_device": self.cfg.device, "config": asdict(self.cfg)}
