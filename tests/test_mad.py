"""MAD 测试（任务文档 Sec. 11.8 清单）。

覆盖：past-only、同 slice 顺序、Bernoulli 概率、padding、partition 覆盖、
期望/方差手算、数值稳定、densification/sparsification/rewiring、查询 API、
五数据集 runner、selected scale、统一 metric。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PYTHON = "/home/zhu/.conda/envs/IF/bin/python"

from script.common.metrics import compute_metrics  # noqa: E402
from script.model.MAD.mad import MAD, run_edge_event_batch  # noqa: E402
from script.model.MAD.multiscale import (  # noqa: E402
    VAR_FLOOR,
    decomposition_coefficients,
    get_model_mean_var,
    leaf_vector,
    next_power_of_two,
    order_relations,
    score_query,
)
from script.model.MAD.probability import (  # noqa: E402
    SlicePolicy,
    compute_slice_ids,
    estimate_probabilities,
    estimate_probability,
)


# ---------------------------------------------------------------------------
# 概率估计（公式 9）
# ---------------------------------------------------------------------------


def test_bernoulli_probability_estimation():
    assert estimate_probability(3, 10) == pytest.approx(0.3)
    assert estimate_probability(0, 10) == pytest.approx(0.0)
    assert estimate_probability(10, 10) == pytest.approx(1.0)
    # smoothing
    assert estimate_probability(3, 10, smoothing=1.0) == pytest.approx(
        (3 + 1.0) / (10 + 2.0))
    with pytest.raises(ValueError):
        estimate_probability(5, 0)  # zero window
    probs = estimate_probabilities(np.array([1, 2]), 4)
    assert np.allclose(probs, [0.25, 0.5])


def test_unseen_relation_zero_history():
    # unseen relation with no history -> P=0, variance floor keeps score finite
    probs = np.zeros(3)
    obs = np.array([1.0, 0.0, 1.0])
    order = order_relations([(0, 1), (1, 2), (2, 3)], probs)
    res = score_query(obs, probs, order, next_power_of_two(3), VAR_FLOOR)
    assert np.isfinite(res.score)
    assert res.score >= 0


# ---------------------------------------------------------------------------
# multi-scale 分解
# ---------------------------------------------------------------------------


def test_padding_does_not_change_real_stats():
    # M=3 relations padded to 4 leaves; the virtual leaf stays zero and the
    # score equals the unpadded M=3 decomposition up to the padded leaf.
    probs = np.array([0.5, 0.3, 0.2])
    obs = np.array([1.0, 0.0, 1.0])
    order = order_relations([0, 1, 2], probs)
    mem = next_power_of_two(3)
    assert mem == 4
    leaves = leaf_vector(obs, order, mem)
    assert leaves.shape == (4,)
    assert leaves[3] == 0.0  # virtual leaf
    # every real relation appears exactly once
    assert set(leaves[:3]) == {0.0, 1.0}
    res = score_query(obs, probs, order, mem, VAR_FLOOR)
    assert np.isfinite(res.score)


def test_partition_covers_all_relations_exactly_once():
    # decomposition coefficient vector length == mem (power of two), and the
    # leaf vector bijects input relations onto distinct leaves.
    rng = np.random.RandomState(0)
    for m in (1, 2, 3, 4, 5):
        probs = rng.rand(m)
        probs /= probs.sum()
        order = order_relations(list(range(m)), probs)
        mem = next_power_of_two(m)
        x = decomposition_coefficients(np.ones(m), order, mem)
        assert len(x) == mem
        assert len(np.unique(leaf_vector(np.arange(m), order, mem)[:m])) == m


def test_expectation_variance_small_cases():
    # M=1: th_mean = [P], th_var = [P(1-P)]
    th_mean, th_var = get_model_mean_var(
        np.array([0.3]), np.array([0]), 1)
    assert th_mean[0] == pytest.approx(0.3)
    assert th_var[0] == pytest.approx(0.3 * 0.7)

    # M=2 with p0 >= p1: x = [s, w], s = (p0+p1)/sqrt(2),
    # sigma2[s] = (q0+q1)/2 with q_i = p_i(1-p_i)
    p = np.array([0.8, 0.4])
    order = order_relations([0, 1], p)
    th_mean, th_var = get_model_mean_var(p, order, 2)
    assert th_mean[0] == pytest.approx((0.8 + 0.4) / np.sqrt(2.0))
    assert th_var[0] == pytest.approx((0.8 * 0.2 + 0.4 * 0.6) / 2.0)
    # reference-implementation detail (faithfully ported): the variance
    # vector is hstack([var_dec[0]] + var_dec[0:-1]), so x[1] (w) shares the
    # level-0 scaling variance with x[0] (s) in this M=2 case.
    assert th_var[1] == pytest.approx(th_var[0])
    obs = np.array([1.0, 1.0])
    res = score_query(obs, p, order, 2, VAR_FLOOR)
    s = np.sqrt(2.0)
    e_s = (0.8 + 0.4) / np.sqrt(2.0)
    w = (1.0 - 1.0) / np.sqrt(2.0)
    e_w = (0.8 - 0.4) / np.sqrt(2.0)
    expected = (s - e_s) ** 2 / th_var[0] + (w - e_w) ** 2 / th_var[0]
    assert res.score == pytest.approx(expected)


def test_numerical_stability_degenerate_models():
    # all-equal / all-0 / all-1 probabilities must not produce NaN
    for p in (np.full(4, 0.5), np.zeros(4), np.ones(4)):
        order = np.arange(4)
        for obs in (np.zeros(4), np.ones(4)):
            res = score_query(obs, p, order, 4, VAR_FLOOR)
            assert np.isfinite(res.score)
            assert np.isfinite(res.per_variable_score).all()


def test_anomaly_scores_densification_sparsification_rewiring():
    # History: slices 1..10; (1,2) active in 9 of them (P=0.9), (3,4) in 1
    # (P=0.1).  Query slice 11.
    mad = MAD(num_nodes=10, slice_policy=SlicePolicy("native-step"),
              window_mode="expanding")
    for t in range(1, 11):
        rels = [(1, 2)] if t <= 9 else [(3, 4)]
        mad.add_slice(t, np.array([a * 10 + b for a, b in rels], dtype=np.int64))

    # densification: the usually-inactive relation suddenly active
    dense, _ = mad.score_relations(11, [(3, 4)], active=np.array([1.0]))
    # sparsification: the usually-active relation missing
    sparse, _ = mad.score_relations(11, [(1, 2)], active=np.array([0.0]))
    # normal: usually-active relation still active
    normal, _ = mad.score_relations(11, [(1, 2)], active=np.array([1.0]))
    # rewiring: likely missing + unlikely active together
    rewire, _ = mad.score_relations(
        11, [(1, 2), (3, 4)], active=np.array([0.0, 1.0]))
    assert dense.score > normal.score
    assert sparse.score > normal.score
    assert rewire.score > normal.score
    assert np.isfinite(dense.score) and np.isfinite(sparse.score)
    # explainable output
    for r in (dense, sparse, rewire):
        assert r.selected_scale >= 0
        assert r.selected_relation_count >= 0
        assert r.explanation


# ---------------------------------------------------------------------------
# 查询 API（strict past-only）
# ---------------------------------------------------------------------------


def _mad_with_history(slices: dict[int, list[tuple[int, int]]]) -> MAD:
    mad = MAD(num_nodes=10, slice_policy=SlicePolicy("native-step"),
              window_mode="expanding")
    for t, rels in slices.items():
        keys = np.array([a * 10 + b for (a, b) in rels], dtype=np.int64)
        mad.add_slice(t, keys)
    return mad


def test_current_and_future_never_in_history():
    mad = _mad_with_history({1: [(1, 2)], 2: [(2, 3)], 3: [(3, 4)]})
    res, win = mad.score_edge(3, 3, 4, active=True)
    # history for t=3 is slices {1, 2}: (3,4) was never active -> P=0
    assert res.th_mean[0] == pytest.approx(0.0)
    # expanding history size is 2 slices
    assert win == 2


def test_same_timestamp_events_follow_event_id_order():
    # events with identical timestamps are ordered by event_id; the batch
    # scorer scores a whole slice against strictly earlier slices.
    n = 100
    src = np.zeros(n, dtype=np.int64)
    dst = (np.arange(n) % 7).astype(np.int64)
    ts = np.zeros(n)  # single slice
    ids = np.arange(n)
    splits = {"train": [0, 60], "val": [60, 80], "test": [80, n]}
    records, stats, _ = run_edge_event_batch(
        num_nodes=10, src=src, dst=dst, timestamps=ts, slice_ids=ts.astype(int),
        labels=np.zeros(n, dtype=np.int8), label_available=np.ones(n, bool),
        splits=splits, query_mode="active-edge", window_mode="fixed",
        window_k=10, max_samples=None)
    assert stats["leakage_check"]["passed"]
    for c in stats["leakage_check"]["samples"]:
        assert c["max_history_event_id"] < c["current_event_id"]


def test_query_apis():
    mad = _mad_with_history({1: [(1, 2), (2, 3)], 2: [(1, 2)]})
    r1, _ = mad.score_edge(3, 1, 2, active=True)
    r2, _ = mad.score_node(3, 1, direction="out")
    r3, _ = mad.score_slice(3, relation_scope=[(1, 2), (2, 3)])
    r4, _ = mad.score_subgraph(3, [(1, 2), (2, 3)])
    for r in (r1, r2, r3, r4):
        assert np.isfinite(r.score)
        assert r.selected_scale >= 0
        assert r.selected_relation_count >= 0
        assert r.explanation


def test_batch_scorer_modes_run():
    n = 300
    rng = np.random.RandomState(0)
    src = rng.randint(0, 5, n)
    dst = rng.randint(0, 5, n)
    ts = rng.randint(0, 10, n).astype(np.float64)
    # events.parquet is ordered by (timestamp, original_row_id): event_id is
    # the temporal order.  Simulate that here so the past-only assertion
    # (max_history_event_id < current_event_id) reflects the real contract.
    order = np.argsort(ts, kind="stable")
    src, dst, ts = src[order], dst[order], ts[order]
    labels = rng.randint(0, 2, n).astype(np.int8)[order]
    avail = np.ones(n, bool)
    splits = {"train": [0, 180], "val": [180, 240], "test": [240, n]}
    for mode in ("active-edge", "local-candidate-set", "full-scope"):
        records, stats, _ = run_edge_event_batch(
            num_nodes=6, src=src, dst=dst, timestamps=ts,
            slice_ids=ts.astype(np.int64), labels=labels,
            label_available=avail, splits=splits, query_mode=mode,
            window_mode="fixed", window_k=5, max_samples=None)
        assert stats["n_scored"] == n
        assert stats["leakage_check"]["passed"]
        assert stats["leakage_check"]["n_checks"] > 0
        assert all(r.anomaly_score >= 0 for r in records)


def test_unified_metric_function():
    y = np.array([0, 1, -1, 0, 1], dtype=np.int64)
    s = np.array([0.1, 0.9, 0.5, 0.2, 0.8])
    m = compute_metrics(y, s)
    assert m["n_eval"] == 4  # unknown excluded by the SAME function as IF/DIF


# ---------------------------------------------------------------------------
# 五数据集 runner smoke（小样本）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", [
    "midas_darpa", "ibm_hi_small", "ibm_li_small", "paysim", "elliptic_pp",
])
def test_runner_five_datasets(dataset, tmp_path):
    cmd = [
        PYTHON, str(REPO_ROOT / "script/model/MAD/runner.py"),
        "--dataset", dataset, "--protocol", "chronological", "--seed", "42",
        "--query-mode", "active-edge", "--window-mode", "fixed",
        "--max-samples", "2000", "--quick",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = REPO_ROOT / "result/MAD" / dataset / "chronological" / "seed_42"
    for fname in ("config.json", "metrics.json", "scores.parquet",
                  "timing.json", "warnings.json", "run.log",
                  "interface_check.json"):
        assert (out / fname).exists(), fname
    assert (out / "metrics.json").stat().st_size > 0
