"""DIF — Deep Isolation Forest (paper-guided reproduction).

Pipeline (paper Sec. 5):

1. **Random representation ensemble** via optimisation-free neural networks
   (never trained, no optimiser / backward).  `faithful` mode uses CERE
   (Eq. 3-6): a shared base weight per layer modulated by rank-one random
   vectors.  `simplified` mode uses one independent random MLP per member
   (tagged `paper_fidelity="simplified"`).
2. **Deep isolation trees** per representation space (Algorithm 1): random
   axis-parallel splits, depth limit ``J``, subsample ``n`` per tree.
3. **DEAS scoring** (Eq. 7-8): path length + average split deviation along
   the traversed path, combined as
   ``F = 2^{-E|p|/C} * E[g]`` with ``C = c(n)``.

Representation post-processing follows the authors' implementation: each
member's outputs are z-scored (fitted on a random calibration subset of the
train split, never on test data) and passed through ``tanh``.

Inference reuses the exact random weights drawn at fit time; all randomness
is deterministic given ``random_state``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .deep_isolation_tree import DeepIsolationTree
from .random_encoder import make_encoder
from .scoring import average_path_length, deas_score_from_means

__all__ = ["DIF"]


class DIF:
    def __init__(
        self,
        n_representations: int = 50,
        trees_per_representation: int = 6,
        max_samples: int = 256,
        max_depth: int = 8,
        hidden_dim: Sequence[int] = (500, 100),
        rep_dim: int = 20,
        activation: str = "tanh",
        cere: bool = True,
        deas: bool = True,
        random_state: int = 42,
        device: str = "cpu",
        n_calib: int = 4096,
        score_batch_size: int = 4096,
    ):
        if n_representations < 1 or trees_per_representation < 1:
            raise ValueError("n_representations and trees_per_representation >= 1")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2 (subsample per tree)")
        self.n_representations = int(n_representations)
        self.trees_per_representation = int(trees_per_representation)
        self.max_samples = int(max_samples)
        self.max_depth = int(max_depth)
        self.hidden_dim = tuple(int(h) for h in hidden_dim)
        self.rep_dim = int(rep_dim)
        self.activation = activation
        self.cere = bool(cere)
        self.deas = bool(deas)
        self.random_state = int(random_state)
        self.device = device
        self.n_calib = int(n_calib)
        self.score_batch_size = int(score_batch_size)

        self.encoder = None
        self.scalers: list[tuple[np.ndarray, np.ndarray]] = []
        self.trees: list[list[DeepIsolationTree]] = []
        self.c_n = 0.0
        self.n_sub = 0
        self.T = self.n_representations * self.trees_per_representation
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray):
        """Fit on the train split.  y is ignored (unsupervised)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix (n_samples, n_features)")
        n, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        ensemble_seeds = [
            int(s) for s in rng.randint(0, 100_000, size=self.n_representations)
        ]

        self.encoder = make_encoder(
            n_features=n_features,
            n_ensemble=self.n_representations,
            hidden_dim=self.hidden_dim,
            rep_dim=self.rep_dim,
            activation=self.activation,
            device=self.device,
            seed=ensemble_seeds[0],
            cere=self.cere,
        )

        # ---- per-member calibration scaler (fit on train only) -----------
        n_calib = min(self.n_calib, n)
        calib_idx = rng.choice(n, n_calib, replace=False)
        calib_reps = self.encoder.encode_all(
            torch.from_numpy(X[calib_idx])).detach().cpu().numpy()
        self.scalers = []
        for u in range(self.n_representations):
            arr = calib_reps[u]
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            sd = np.where(sd < 1e-8, 1.0, sd)
            self.scalers.append((mu, sd))

        # ---- grow t deep isolation trees per representation --------------
        n_sub = min(self.max_samples, n)
        self.trees = []
        for u in range(self.n_representations):
            member_trees = []
            for _ in range(self.trees_per_representation):
                idx = rng.choice(n, size=n_sub, replace=False)
                rep = self.encoder.encode_member(
                    torch.from_numpy(X[idx]), u).detach().cpu().numpy()
                rep_t = self._postprocess(rep, u)
                tree = DeepIsolationTree(
                    max_depth=self.max_depth, rng=rng).build(rep_t)
                member_trees.append(tree)
            self.trees.append(member_trees)

        self.c_n = average_path_length(n_sub)
        self.n_sub = n_sub
        self.T = self.n_representations * self.trees_per_representation
        self._fitted = True
        return self

    def _postprocess(self, rep: np.ndarray, u: int) -> np.ndarray:
        mu, sd = self.scalers[u]
        return np.tanh((rep - mu) / sd)

    # ---------------------------------------------------------------- score
    def score(self, X: np.ndarray) -> np.ndarray:
        """Batched anomaly scoring; larger = more anomalous (Eq. 8)."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before score()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix")
        n = X.shape[0]
        r = self.n_representations
        path_sum = np.zeros(n, dtype=np.float64)
        g_sum = np.zeros(n, dtype=np.float64)
        bsize = max(1, self.score_batch_size)

        for start in range(0, n, bsize):
            end = min(start + bsize, n)
            reps = self.encoder.encode_all(
                torch.from_numpy(X[start:end])).detach().cpu().numpy()
            for u in range(r):
                ru = self._postprocess(reps[u], u)
                for tree in self.trees[u]:
                    path, g = tree.traverse(ru)
                    path_sum[start:end] += path
                    g_sum[start:end] += g

        if self.deas:
            scores = deas_score_from_means(
                path_sum / self.T, g_sum / self.T, self.n_sub)
        else:
            # ablation w/o DEAS: iForest-style depth term only
            avg_path = path_sum / self.T
            c_n = average_path_length(self.n_sub)
            scores = np.power(2.0, -avg_path / c_n) if c_n > 0 else \
                np.power(2.0, -avg_path)
        return np.asarray(scores, dtype=np.float64)

    def config(self) -> dict:
        enc_cfg = self.encoder.config() if self.encoder is not None else {}
        return {
            "model": "DIF",
            "n_representations": self.n_representations,
            "trees_per_representation": self.trees_per_representation,
            "total_trees": self.T,
            "subsample_size": self.max_samples,
            "max_depth": self.max_depth,
            "rep_dim": self.rep_dim,
            "activation": self.activation,
            "cere_enabled": self.cere,
            "deas_enabled": self.deas,
            "random_state": self.random_state,
            "device": self.device,
            "n_calib": self.n_calib,
            "score_batch_size": self.score_batch_size,
            "score_function": "DEAS (Eq. 8)" if self.deas else "depth-only (ablation)",
            "encoder": enc_cfg,
        }
