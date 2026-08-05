"""Deep SVDD (one-class) with an MLP encoder for tabular data.

Paper: Ruff et al., "Deep One-Class Classification" (ICML 2018,
arXiv:1802.06360); PyTorch reference implementation:
https://github.com/lukasruff/Deep-SVDD-PyTorch (MIT).

Adaptation: the official repository is image-oriented (LeNet encoders,
MNIST/CIFAR loaders). We keep the Deep-SVDD objective unchanged
(minimize ||phi(x; W) - c||^2 with a fixed hypersphere center c) and replace
the CNN encoder with an MLP suitable for transaction feature rows.

Objective (one-class Deep SVDD):
    min_W (1/n) * sum_i || phi(x_i; W) - c ||^2
c is initialized as the mean of phi(x_i; W0) over the train set after the
encoder is warmed up with a few forward passes (official initialization),
then kept fixed during training. Score = ||phi(x; W) - c||^2
(larger = more anomalous).

The official pipeline also pre-trains an auto-encoder; that stage is optional
here (--pretrain-ae, off by default) and is recorded in config.json.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SVDDEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], rep_dim: int,
                 bias: bool = True) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h, bias=bias))
            layers.append(nn.LeakyReLU(0.2))
            prev = h
        layers.append(nn.Linear(prev, rep_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSVDD:
    """One-class Deep SVDD on tabular rows; larger score = more anomalous."""

    def __init__(self, input_dim: int, hidden: tuple[int, ...] = (64, 32),
                 rep_dim: int = 16, epochs: int = 50, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-6,
                 pretrain_ae: bool = False, ae_epochs: int = 10,
                 device: str = "cpu", seed: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden = list(hidden)
        self.rep_dim = rep_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_ae = pretrain_ae
        self.ae_epochs = ae_epochs
        self.device = torch.device(device)
        self.seed = seed
        self.net: SVDDEncoder | None = None
        self.center: torch.Tensor | None = None
        self.c_ = torch.zeros(rep_dim)

    def _reset(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.net = SVDDEncoder(self.input_dim, self.hidden, self.rep_dim).to(self.device)

    def fit(self, X: np.ndarray) -> "DeepSVDD":
        X = np.asarray(X, dtype=np.float32)
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("DeepSVDD.fit got non-finite values in X")
        self._reset()
        assert self.net is not None
        Xt = torch.from_numpy(X).to(self.device)

        # --- optional AE pre-training (official two-stage pipeline) ---
        if self.pretrain_ae:
            self._pretrain_autoencoder(Xt)

        # --- center initialization (official: mean output after warmup) ---
        with torch.no_grad():
            out = self.net(Xt[: min(Xt.shape[0], 10_000)])
            self.c_ = out.mean(dim=0)
        self.center = self.c_.clone().detach()
        assert self.center is not None

        # --- one-class training: min sum ||phi(x) - c||^2, c fixed ---
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n = Xt.shape[0]
        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            self.net.train()
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                batch = Xt[b]
                optimizer.zero_grad()
                phi = self.net(batch)
                loss = torch.mean(torch.sum((phi - self.center) ** 2, dim=1))
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def _pretrain_autoencoder(self, Xt: torch.Tensor) -> None:
        """Optional AE pre-training that initializes the encoder weights."""
        assert self.net is not None
        # mirror decoder: rep_dim -> reversed(hidden) -> input_dim
        dec_layers: list[nn.Module] = []
        dims = [self.rep_dim, *reversed(self.hidden)]
        for a, b in zip(dims[:-1], dims[1:]):
            dec_layers.append(nn.Linear(a, b))
            dec_layers.append(nn.LeakyReLU(0.2))
        dec_layers.append(nn.Linear(dims[-1], self.input_dim))
        decoder = nn.Sequential(*dec_layers)
        ae = nn.Sequential(self.net, decoder).to(self.device)
        opt = torch.optim.Adam(ae.parameters(), lr=self.lr)
        n = Xt.shape[0]
        idx = np.arange(n)
        for _ in range(self.ae_epochs):
            np.random.shuffle(idx)
            ae.train()
            for s in range(0, n, self.batch_size):
                b = idx[s:s + self.batch_size]
                opt.zero_grad()
                rec = ae(Xt[b])
                loss = torch.mean((rec - Xt[b]) ** 2)
                loss.backward()
                opt.step()
        ae.eval()

    def score(self, X: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
        if self.net is None or self.center is None:
            raise RuntimeError("fit() must be called before score()")
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(X.shape[0], dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            for s in range(0, X.shape[0], batch_size):
                batch = torch.from_numpy(X[s:s + batch_size]).to(self.device)
                phi = self.net(batch)
                out[s:s + batch_size] = torch.sum(
                    (phi - self.center) ** 2, dim=1).cpu().numpy()
        return out
