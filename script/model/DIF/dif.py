"""DIF — paper-guided, temporal-safe Deep Isolation Forest."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Sequence

import numpy as np
import torch

from .deep_isolation_tree import DeepIsolationTree
from .random_encoder import make_encoder
from .scoring import average_path_length, deas_score_from_means

__all__ = ["DIF"]


class DIF:
    """Paper-algorithm DIF with bounded batches and internal stage profiling.

    This is deliberately a temporal-safe reimplementation rather than a
    byte-for-byte execution of the authors' sklearn-based source code.
    """

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
        batch_size: int = 4096,
        score_batch_size: int = 4096,
        num_workers: int = 0,
        n_calib: int | None = None,
    ):
        if n_representations < 1 or trees_per_representation < 1:
            raise ValueError("n_representations and trees_per_representation >= 1")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2 (subsample per tree)")
        if batch_size < 1 or score_batch_size < 1:
            raise ValueError("batch_size and score_batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
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
        self.batch_size = int(batch_size)
        self.score_batch_size = int(score_batch_size)
        self.num_workers = int(num_workers)
        # Compatibility only: old callers may pass it, but approximation is gone.
        self.n_calib = n_calib

        self.encoder = None
        self.scalers: list[tuple[np.ndarray, np.ndarray]] = []
        self.trees: list[list[DeepIsolationTree]] = []
        self.c_n = 0.0
        self.n_sub = 0
        self.T = self.n_representations * self.trees_per_representation
        self._fitted = False
        self.profile_: dict[str, float | int] = {}

    def _reset_profile(self) -> None:
        self.profile_ = {"fit_samples": 0, "score_samples": 0}

    def _add_profile(self, name: str, started: float) -> None:
        self.profile_[name] = float(self.profile_.get(name, 0.0)) + time.perf_counter() - started

    def profile(self) -> dict[str, float | int]:
        """Cumulative stage timing. Collection has no effect on scores."""
        return dict(self.profile_)

    def fit(self, X: np.ndarray):
        """Fit only on the supplied split; labels are intentionally ignored."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix (n_samples, n_features)")
        n, n_features = X.shape
        if n == 0:
            raise ValueError("cannot fit DIF on an empty matrix")
        self._reset_profile()
        self.profile_["fit_samples"] = int(n)
        rng = np.random.RandomState(self.random_state)
        ensemble_seeds = [int(s) for s in rng.randint(0, 100_000, size=self.n_representations)]

        started = time.perf_counter()
        self.encoder = make_encoder(
            n_features=n_features, n_ensemble=self.n_representations,
            hidden_dim=self.hidden_dim, rep_dim=self.rep_dim,
            activation=self.activation, device=self.device,
            seed=ensemble_seeds[0], cere=self.cere,
        )
        self._add_profile("fit_encoder_initialization_seconds", started)

        # Exact train-only z-score statistics, streamed to bound memory.  This
        # replaces the old random 4096-row calibration approximation.
        started = time.perf_counter()
        rep_sum = np.zeros((self.n_representations, self.rep_dim), dtype=np.float64)
        rep_sq_sum = np.zeros_like(rep_sum)
        for offset in range(0, n, self.batch_size):
            reps = self.encoder.encode_all(
                torch.from_numpy(X[offset:offset + self.batch_size])
            ).detach().cpu().numpy()
            rep_sum += reps.sum(axis=1, dtype=np.float64)
            rep_sq_sum += np.einsum("rbd,rbd->rd", reps, reps, dtype=np.float64)
        self.scalers = []
        for u in range(self.n_representations):
            mu = rep_sum[u] / n
            sd = np.sqrt(np.maximum(rep_sq_sum[u] / n - mu * mu, 0.0))
            self.scalers.append((mu, np.where(sd < 1e-8, 1.0, sd)))
        self._add_profile("fit_scaler_encoding_and_statistics_seconds", started)

        n_sub = min(self.max_samples, n)
        self.trees = []
        encode_seconds = 0.0
        build_seconds = 0.0
        for u in range(self.n_representations):
            member_trees = []
            for _ in range(self.trees_per_representation):
                idx = rng.choice(n, size=n_sub, replace=False)
                started = time.perf_counter()
                rep = self.encoder.encode_member(
                    torch.from_numpy(X[idx]), u).detach().cpu().numpy()
                encode_seconds += time.perf_counter() - started
                started = time.perf_counter()
                member_trees.append(DeepIsolationTree(
                    max_depth=self.max_depth, rng=rng).build(self._postprocess(rep, u)))
                build_seconds += time.perf_counter() - started
            self.trees.append(member_trees)

        self.c_n = average_path_length(n_sub)
        self.n_sub = n_sub
        self.T = self.n_representations * self.trees_per_representation
        self.profile_["fit_tree_representation_encoding_seconds"] = encode_seconds
        self.profile_["fit_tree_build_seconds"] = build_seconds
        self._fitted = True
        return self

    def _postprocess(self, rep: np.ndarray, u: int) -> np.ndarray:
        mu, sd = self.scalers[u]
        return np.tanh((rep - mu) / sd)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return larger-is-more-anomalous scores with stage-level profiling."""
        if not self._fitted:
            raise RuntimeError("fit() must be called before score()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix")
        n = X.shape[0]
        path_sum = np.zeros(n, dtype=np.float64)
        g_sum = np.zeros(n, dtype=np.float64)
        # Both public batch controls are active: encoding must honour batch_size
        # and score_batch_size can further cap an evaluation block.
        bsize = min(self.batch_size, self.score_batch_size)
        self.profile_["score_samples"] = int(n)

        for start in range(0, n, bsize):
            end = min(start + bsize, n)
            started = time.perf_counter()
            reps = self.encoder.encode_all(
                torch.from_numpy(X[start:end])
            ).detach().cpu().numpy()
            self._add_profile("score_representation_encoding_seconds", started)

            started = time.perf_counter()
            def traverse_member(u: int) -> tuple[np.ndarray, np.ndarray]:
                ru = self._postprocess(reps[u], u)
                member_path = np.zeros(end - start, dtype=np.float64)
                member_g = np.zeros(end - start, dtype=np.float64)
                for tree in self.trees[u]:
                    path, g = tree.traverse(ru, num_workers=self.num_workers)
                    member_path += path
                    member_g += g
                return member_path, member_g

            # Numba releases the GIL inside traversal; unlike the old NumPy
            # indexing kernel, representation-level workers can run concurrently.
            if self.num_workers > 1:
                with ThreadPoolExecutor(max_workers=min(self.num_workers, self.n_representations)) as pool:
                    member_scores = list(pool.map(traverse_member, range(self.n_representations)))
            else:
                member_scores = [traverse_member(u) for u in range(self.n_representations)]
            for member_path, member_g in member_scores:
                path_sum[start:end] += member_path
                g_sum[start:end] += member_g
            self._add_profile("score_tree_traversal_seconds", started)

        avg_path = path_sum / self.T
        if self.deas:
            return np.asarray(deas_score_from_means(avg_path, g_sum / self.T, self.n_sub), dtype=np.float64)
        c_n = average_path_length(self.n_sub)
        return np.power(2.0, -avg_path / c_n) if c_n > 0 else np.power(2.0, -avg_path)

    def config(self) -> dict:
        enc_cfg = self.encoder.config() if self.encoder is not None else {}
        return {
            "model": "DIF", "n_representations": self.n_representations,
            "trees_per_representation": self.trees_per_representation,
            "total_trees": self.T, "subsample_size": self.max_samples,
            "max_depth": self.max_depth, "rep_dim": self.rep_dim,
            "activation": self.activation, "cere_enabled": self.cere,
            "deas_enabled": self.deas, "random_state": self.random_state,
            "device": self.device, "batch_size": self.batch_size,
            "score_batch_size": self.score_batch_size, "num_workers": self.num_workers,
            "scaler_fit": "complete_fit_split_streamed",
            "score_function": "DEAS (Eq. 8)" if self.deas else "depth-only (ablation)",
            "encoder": enc_cfg,
        }
