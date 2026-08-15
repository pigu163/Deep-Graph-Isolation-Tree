"""Tests for the endpoint-wise batched ESN bank."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from script.model.new_model import temporal_motif_reservoir_dif as model_module
from script.model.new_model.temporal_motif_reservoir_dif import (
    ContractiveReservoir,
    LocalLinkStreamConfig,
    LocalLinkStreamESNIF,
    ReservoirConfig,
    TemporalMotifReservoirDIF,
)


def _config(**changes) -> ReservoirConfig:
    base = ReservoirConfig(
        num_reservoirs=2,
        state_dim=8,
        embedding_dim=8,
        time_dim=4,
        trees_per_reservoir=2,
        subsample_size=8,
        batch_size=4,
    )
    return replace(base, **changes)


def _batch():
    src = np.array([0, 0, 0, 0], dtype=np.int64)
    dst = np.array([1, 2, 3, 4], dtype=np.int64)
    feat = np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0
    gap = np.array([0.0, 1.0, 2.0, 3.0])
    is_new = np.zeros(4)
    return src, dst, feat, gap, gap + 0.5, is_new, is_new


def test_endpoint_weight_and_output_shapes():
    cfg = _config()
    reservoir = ContractiveReservoir(edge_dim=3, cfg=cfg, seed=7)
    src, dst, feat, src_gap, dst_gap, src_new, dst_new = _batch()

    embedding, (nodes, candidates) = reservoir.encode_group(
        src, dst, feat, src_gap, dst_gap, src_new, dst_new)

    event_input_dim = 3 + 2 * cfg.time_dim + 2
    assert reservoir.w_res.shape == (cfg.state_dim, cfg.state_dim)
    assert reservoir.w_in.shape == (cfg.state_dim, event_input_dim + 1)
    assert reservoir.w_event.shape == (
        cfg.embedding_dim, 2 * cfg.state_dim + event_input_dim)
    assert embedding.shape == (len(src), cfg.embedding_dim)
    assert nodes.shape == (2 * len(src),)
    assert candidates.shape == (2 * len(src), cfg.state_dim)


def test_repeated_nodes_preserve_batch_contraction():
    """A star batch must not replicate one node's state perturbation."""
    cfg = _config()
    left = ContractiveReservoir(edge_dim=3, cfg=cfg, seed=11)
    right = ContractiveReservoir(edge_dim=3, cfg=cfg, seed=11)
    rng = np.random.default_rng(9)
    active_nodes = np.arange(5)
    for node in active_nodes:
        left.states[int(node)] = rng.uniform(-0.5, 0.5, cfg.state_dim)
        right.states[int(node)] = rng.uniform(-0.5, 0.5, cfg.state_dim)
    before = np.concatenate([
        left.states[int(node)] - right.states[int(node)] for node in active_nodes])

    args = _batch()
    _, pending_left = left.encode_group(*args)
    _, pending_right = right.encode_group(*args)
    left.commit(pending_left)
    right.commit(pending_right)
    after = np.concatenate([
        left.states[int(node)] - right.states[int(node)] for node in active_nodes])

    kappa = left.report["analytical_kappa"]
    assert np.linalg.norm(after) <= kappa * np.linalg.norm(before) + 1e-12


def test_small_stream_fit_and_score():
    cfg = _config(device="cpu")
    model = TemporalMotifReservoirDIF(edge_dim=3, config=cfg, seed=13)
    n = 12
    data = {
        "event_id": np.arange(n, dtype=np.int64),
        "timestamp": np.arange(n, dtype=np.float64),
        "src": np.arange(n, dtype=np.int64) % 3,
        "dst": (np.arange(n, dtype=np.int64) + 1) % 5,
        "edge_feat": np.arange(3 * n, dtype=np.float64).reshape(n, 3) / 20.0,
    }

    scores = model.fit(data).score(data)

    assert scores.shape == (n,)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert model.metadata()["strict_active_endpoint_contraction"] is True


@pytest.mark.skipif(
    model_module.torch is None or not model_module.torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_bank_matches_cpu():
    cpu = TemporalMotifReservoirDIF(
        edge_dim=3, config=_config(device="cpu"), seed=17)
    cuda = TemporalMotifReservoirDIF(
        edge_dim=3, config=_config(device="cuda"), seed=17)
    rng = np.random.default_rng(5)
    for cpu_reservoir, cuda_reservoir in zip(cpu.reservoirs, cuda.reservoirs):
        for node in range(5):
            state = rng.uniform(-0.5, 0.5, cpu.cfg.state_dim)
            cpu_reservoir.states[node] = state.copy()
            cuda_reservoir.states[node] = state.copy()

    args = _batch()
    cpu_embeddings = []
    cpu_pending = []
    for reservoir in cpu.reservoirs:
        embedding, pending = reservoir.encode_group(*args)
        cpu_embeddings.append(embedding)
        cpu_pending.append(pending)
    cuda_embeddings, cuda_pending = cuda._encode_cuda_bank(*args)

    np.testing.assert_allclose(
        cuda_embeddings, np.stack(cpu_embeddings), rtol=2e-4, atol=2e-5)
    for (cpu_nodes, cpu_values), (cuda_nodes, cuda_values) in zip(
            cpu_pending, cuda_pending):
        np.testing.assert_array_equal(cuda_nodes, cpu_nodes)
        np.testing.assert_allclose(
            cuda_values, cpu_values, rtol=2e-4, atol=2e-5)


def test_local_esn_equal_timestamp_context_is_order_invariant():
    torch = pytest.importorskip("torch")

    class Context:
        def __init__(self, features):
            self.event_ids = torch.tensor([0, 1])
            self.edge_feat = torch.tensor(features, dtype=torch.float32)
            self.ts = torch.tensor([1.0, 1.0], dtype=torch.float64)

    class Collator:
        def __init__(self, context):
            self.context = context

        def sample_event_ids(self, ids):
            return [self.context for _ in ids]

    cfg = LocalLinkStreamConfig(state_dim=8, projection_count=3, device="cpu")
    feature_a = np.array([[1.0, 2.0, 0.5], [3.0, 1.0, -0.5]])
    feature_b = feature_a[::-1].copy()
    edge_features = np.vstack((feature_a, [[2.0, 2.0, 0.0]])).astype(np.float32)
    timestamps = np.array([1.0, 1.0, 2.0])

    first = LocalLinkStreamESNIF(edge_dim=3, config=cfg, seed=23)
    second = LocalLinkStreamESNIF(edge_dim=3, config=cfg, seed=23)
    first.fit_normalizer(edge_features, timestamps, np.arange(3))
    second.fit_normalizer(edge_features, timestamps, np.arange(3))

    encoded_a = first._encode(np.array([2]), edge_features, timestamps, Collator(Context(feature_a)))
    encoded_b = second._encode(np.array([2]), edge_features, timestamps, Collator(Context(feature_b)))
    for name in ("departure", "surprise", "recurrence"):
        np.testing.assert_allclose(encoded_a[name], encoded_b[name], rtol=1e-6, atol=1e-6)
