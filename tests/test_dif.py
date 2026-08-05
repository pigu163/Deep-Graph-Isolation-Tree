"""Tests for DIF (random encoder / DIT / DEAS / reproducibility)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "script")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from script.model.DIF.deep_isolation_tree import DeepIsolationTree  # noqa: E402
from script.model.DIF.dif import DIF  # noqa: E402
from script.model.DIF.random_encoder import CereEncoder, IndependentEncoder  # noqa: E402
from script.model.DIF.scoring import (  # noqa: E402
    average_path_length,
    deas_score,
    deas_score_from_means,
)


def _synthetic(n_normal: int = 300, n_anom: int = 30, seed: int = 7):
    rng = np.random.RandomState(seed)
    normal = rng.normal(0, 1.0, size=(n_normal, 4))
    anom = rng.uniform(10, 15, size=(n_anom, 4))
    X = np.vstack([normal, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anom)])
    return X, y


class TestRandomEncoder:
    def test_no_optimizer_no_backward(self):
        enc = CereEncoder(n_features=4, n_ensemble=5, hidden_dim=(8,),
                          rep_dim=4, seed=1)
        x = torch.randn(6, 4)
        enc.encode_all(x)
        for tensors in (enc.W0, enc.P, enc.Q):
            for t in tensors:
                assert t.grad is None
        assert not hasattr(enc, "optimizer")

    def test_same_seed_same_repr(self):
        x = torch.randn(5, 4)
        a = CereEncoder(n_features=4, n_ensemble=5, hidden_dim=(8,),
                        rep_dim=4, seed=42).encode_all(x)
        b = CereEncoder(n_features=4, n_ensemble=5, hidden_dim=(8,),
                        rep_dim=4, seed=42).encode_all(x)
        assert torch.equal(a, b)

    def test_different_members_different(self):
        enc = CereEncoder(n_features=4, n_ensemble=5, hidden_dim=(8,),
                          rep_dim=4, seed=1)
        out = enc.encode_all(torch.randn(6, 4))
        assert out.shape == (5, 6, 4)
        assert not torch.equal(out[0], out[1])

    def test_cere_matches_naive_member(self):
        enc = CereEncoder(n_features=4, n_ensemble=5, hidden_dim=(8, 6),
                          rep_dim=4, seed=3)
        x = torch.randn(7, 4)
        all_reps = enc.encode_all(x)
        for u in range(enc.r):
            assert torch.allclose(all_reps[u], enc.encode_member(x, u),
                                  atol=1e-6)

    def test_independent_encoder_runs(self):
        enc = IndependentEncoder(n_features=4, n_ensemble=3, hidden_dim=(8,),
                                 rep_dim=4, seed=1)
        out = enc.encode_all(torch.randn(5, 4))
        assert out.shape == (3, 5, 4)


class TestDeepIsolationTree:
    def test_path_length(self):
        rng = np.random.RandomState(0)
        proj = rng.randn(64, 3)
        tree = DeepIsolationTree(max_depth=8, rng=rng).build(proj)
        path, g = tree.traverse(proj)
        assert path.shape == (64,) and g.shape == (64,)
        assert (path >= 0).all() and (path <= 8).all()
        assert np.isfinite(g).all()

    def test_isolated_point_short_path(self):
        rng = np.random.RandomState(1)
        normal = rng.normal(0, 1, (120, 2))
        weird = np.array([[20.0, -20.0]])
        proj = np.vstack([normal, weird])
        tree = DeepIsolationTree(max_depth=6, rng=rng).build(proj)
        path, _ = tree.traverse(proj)
        assert path[-1] <= path[:-1].mean()  # outlier is isolated sooner


class TestScoring:
    def test_average_path_length(self):
        assert average_path_length(1) == 0.0
        assert average_path_length(2) == 1.0
        assert average_path_length(256) > 0.0

    def test_deas_finite_and_direction(self):
        paths = np.array([[1.0, 1.0], [8.0, 8.0], [4.0, 5.0]])
        devs = np.array([[2.0, 2.0], [0.1, 0.2], [0.5, 0.5]])
        scores = deas_score(paths, devs, n_subsample=256)
        assert np.isfinite(scores).all()
        # shorter paths + larger deviation -> higher score
        assert scores[0] > scores[1]

    def test_deas_from_means_matches(self):
        paths = np.array([[1.0, 3.0], [7.0, 7.0]])
        devs = np.array([[2.0, 1.0], [0.2, 0.4]])
        s1 = deas_score(paths, devs, 256)
        s2 = deas_score_from_means(paths.mean(1), devs.mean(1), 256)
        assert np.allclose(s1, s2)


class TestDIF:
    def _small_dif(self, cere=True, **kw):
        return DIF(n_representations=kw.pop("r", 16),
                   trees_per_representation=kw.pop("t", 4),
                   max_samples=64, max_depth=6,
                   hidden_dim=(16,), rep_dim=8,
                   cere=cere, random_state=kw.pop("seed", 42),
                   n_calib=128, score_batch_size=128, **kw)

    def test_fit_score_runs(self):
        X, _ = _synthetic()
        model = self._small_dif()
        model.fit(X)
        s = model.score(X)
        assert s.shape == (X.shape[0],)
        assert np.isfinite(s).all()

    def test_anomaly_higher_score(self):
        X, y = _synthetic()
        model = self._small_dif()
        model.fit(X)
        s = model.score(X)
        assert s[y == 1].mean() > s[y == 0].mean()

    def test_batch_matches_full(self):
        X, _ = _synthetic()
        model = self._small_dif()
        model.fit(X)
        full = model.score(X)
        model.score_batch_size = 16
        batched = model.score(X)
        assert np.allclose(full, batched)

    def test_reproducible(self):
        X, _ = _synthetic()
        s1 = self._small_dif(seed=42); s1.fit(X); sc1 = s1.score(X)
        s2 = self._small_dif(seed=42); s2.fit(X); sc2 = s2.score(X)
        assert np.allclose(sc1, sc2)

    def test_different_seed_different(self):
        X, _ = _synthetic()
        a = self._small_dif(seed=1); a.fit(X); sa = a.score(X)
        b = self._small_dif(seed=2); b.fit(X); sb = b.score(X)
        assert not np.allclose(sa, sb)

    def test_simplified_mode(self):
        X, _ = _synthetic()
        model = self._small_dif(cere=False)
        model.fit(X)
        s = model.score(X)
        assert np.isfinite(s).all()
        assert model.cere is False
        assert model.config()["cere_enabled"] is False

    def test_paper_defaults(self):
        model = DIF()
        assert model.n_representations == 50
        assert model.trees_per_representation == 6
        assert model.max_samples == 256
        assert model.max_depth == 8
        assert model.T == 300
        assert model.cere is True and model.deas is True

    def test_no_deas_ablation_runs(self):
        X, _ = _synthetic()
        model = self._small_dif(deas=False)
        model.fit(X)
        assert np.isfinite(model.score(X)).all()

    def test_requires_fit_first(self):
        model = self._small_dif()
        with pytest.raises(RuntimeError):
            model.score(_synthetic()[0])


class TestMetricContract:
    def test_unknown_labels_excluded(self):
        """Shared metrics must ignore -1 (unknown) labels (task doc 11.2)."""
        from script.common.metrics import compute_metrics

        y = np.array([0, 1, -1, 0, 1, -1], dtype=np.int64)
        s = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.99])
        m = compute_metrics(y, s)
        assert m["n_eval"] == 4
        assert m["n_pos"] == 2
        assert m["roc_auc"] is not None
        assert m["pr_auc"] is not None
