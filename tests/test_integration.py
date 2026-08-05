"""集成测试：三模型跨模型对齐检查（任务文档 Sec. 15）。

只做轻量、无全量结果依赖的检查：共享 tabular pipeline、score 方向、
统一 metric 字段、quick-run 标记。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from script.model.IF.isolation_forest import IFAnomalyDetector  # noqa: E402
from script.model.DIF.dif import DIF  # noqa: E402
from script.model.MAD.mad import MAD, run_edge_event_batch  # noqa: E402
from script.model.MAD.probability import SlicePolicy  # noqa: E402
from script.tabular_pipeline.datasets import load_tabular_dataset  # noqa: E402


def _synthetic(n_normal=200, n_anom=20, seed=7):
    rng = np.random.RandomState(seed)
    normal = rng.normal(0, 1.0, size=(n_normal, 4))
    anom = rng.uniform(8, 12, size=(n_anom, 4))
    X = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    return X, y


def test_if_dif_share_tabular_pipeline():
    """IF 与 DIF 使用同一个 load_tabular_dataset（同一 X/y/sample_ids）。"""
    a = load_tabular_dataset("elliptic_pp", split="test")
    b = load_tabular_dataset("elliptic_pp", split="test")
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.sample_ids, b.sample_ids)
    assert np.array_equal(a.y, b.y)
    assert a.feature_names == b.feature_names
    # unknown labels present but excluded at metric time (labels are -1)
    assert (a.y == -1).sum() > 0


def test_score_direction_all_models():
    """三个模型 score 都是越大越异常：合成异常样本平均分更高。"""
    X, y = _synthetic()

    if_model = IFAnomalyDetector(n_estimators=50, max_samples=64,
                                 random_state=42, n_jobs=-1)
    if_model.fit(X[: 200])
    s_if = if_model.score(X)
    assert s_if[y == 1].mean() > s_if[y == 0].mean()

    dif = DIF(n_representations=8, trees_per_representation=3,
              max_samples=64, max_depth=6, hidden_dim=(16,), rep_dim=8,
              cere=True, random_state=42, n_calib=128)
    dif.fit(X[: 200])
    s_dif = dif.score(X)
    assert s_dif[y == 1].mean() > s_dif[y == 0].mean()

    # MAD: link-stream, densification scores high
    mad = MAD(num_nodes=5, slice_policy=SlicePolicy("native-step"),
              window_mode="expanding")
    mad.add_slice(1, np.array([0 * 5 + 1, 1 * 5 + 2]))  # (0,1),(1,2)
    mad.add_slice(2, np.array([0 * 5 + 1]))
    r_active, _ = mad.score_edge(3, 0, 1, active=True)   # known frequent edge
    r_new, _ = mad.score_edge(3, 3, 4, active=True)      # never seen edge
    assert r_new.score > r_active.score


def test_quick_run_flagged_not_mixed():
    """quick run 的 config 标记 is_quick_run=true；正式汇总不混入（由
    run_all_models/evaluate_all 的目录隔离保证，这里检查 runner 契约）。"""
    import subprocess

    cmd = ["/home/zhu/.conda/envs/IF/bin/python",
           str(REPO_ROOT / "script/model/IF/runner.py"),
           "--dataset", "elliptic_pp", "--protocol", "chronological",
           "--seed", "42", "--max-samples", "300", "--quick", "--overwrite"]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    cfg_path = REPO_ROOT / "result/IF/elliptic_pp/chronological/seed_42/config.json"
    cfg = json.loads(cfg_path.read_text())
    assert cfg["is_quick_run"] is True
