"""Random representation encoders for DIF (optimisation-free).

Two implementations:

* :class:`CereEncoder` — faithful CERE (paper Eq. 3-6).  All ensemble
  members share a base weight matrix ``W0`` per layer, modulated by rank-one
  random vectors ``(p_i, q_i)``:

      W_i = W0 ◦ (p_i q_i^T)          (Eq. 3)
      W_i^T x = W0^T (x ◦ p_i) ◦ q_i  (Eq. 4)

  batched for a whole mini-batch (Eq. 5):

      [X W_1; ...; X W_r] = ((X repeated) ◦ P) W0 ◦ Q

* :class:`IndependentEncoder` — simplified variant: every ensemble member is
  its own randomly initialised MLP (no CERE sharing).  Used only when
  ``--implementation simplified``; results are marked
  ``paper_fidelity = "simplified"`` and are **not** paper reproduction.

Neither encoder is trained: there are no optimisers and no backward passes.
Weights are drawn once at construction and reused for both fit and scoring.
"""

from __future__ import annotations

from typing import Sequence

import torch

_ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "leaky_relu": torch.nn.functional.leaky_relu,
}


def _act(name: str):
    if name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}")
    return _ACTIVATIONS[name]


class CereEncoder:
    """Computation-efficient Representation Ensemble (paper Eq. 3-6)."""

    def __init__(
        self,
        n_features: int,
        n_ensemble: int,
        hidden_dim: Sequence[int] = (500, 100),
        rep_dim: int = 20,
        activation: str = "tanh",
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.n_features = n_features
        self.r = int(n_ensemble)
        self.rep_dim = int(rep_dim)
        self.activation = activation
        self.act = _act(activation)
        self.device = device
        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        sizes = [int(n_features)] + [int(h) for h in hidden_dim] + [int(rep_dim)]
        self.L = len(sizes) - 1
        # base weights W0^(l) ~ N(0,1) and member random vectors p^(l,i), q^(l,i)
        self.W0 = [
            torch.randn(out, inp, generator=gen) for inp, out in zip(sizes, sizes[1:])
        ]
        self.P = [torch.randn(self.r, inp, generator=gen) for inp in sizes[:-1]]
        self.Q = [torch.randn(self.r, out, generator=gen) for out in sizes[1:]]
        self._to_device()

    def _to_device(self) -> None:
        self.W0 = [w.to(self.device) for w in self.W0]
        self.P = [p.to(self.device) for p in self.P]
        self.Q = [q.to(self.device) for q in self.Q]

    def encode_all(self, x: torch.Tensor) -> torch.Tensor:
        """Map a mini-batch into all r representation spaces.

        Returns a tensor of shape ``(r, b, rep_dim)`` where row ``u`` is the
        projection of the batch into member u's space (paper Eq. 5).
        """
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"expected (batch, {self.n_features}) got {tuple(x.shape)}")
        b = x.shape[0]
        # batch-ensemble layout: block i (rows i*b:(i+1)*b) is member i's
        # copy of the batch; P/Q rows are repeated b times to align (Eq. 5)
        h = x.to(self.device).repeat(self.r, 1)        # (b*r, in)
        for l in range(self.L):
            W = self.W0[l]
            p = self.P[l].repeat_interleave(b, dim=0)  # (b*r, in)
            hw = (h * p) @ W.t()                       # (b*r, out)
            q = self.Q[l].repeat_interleave(b, dim=0)  # (b*r, out)
            hw = hw * q
            if l < self.L - 1:
                hw = self.act(hw)
            h = hw
        return h.view(self.r, b, self.rep_dim)

    def encode_member(self, x: torch.Tensor, u: int) -> torch.Tensor:
        """Project a batch into a single representation space ``u`` (Eq. 4)."""
        h = x.to(self.device)
        for l in range(self.L):
            W = self.W0[l]
            pu = self.P[l][u]
            qu = self.Q[l][u]
            h = (h * pu) @ W.t()  # (b, out)
            h = h * qu
            if l < self.L - 1:
                h = self.act(h)
        return h

    def config(self) -> dict:
        return {
            "encoder_type": "cere",
            "cere_enabled": True,
            "layers": [self.n_features] + [int(x.shape[1]) for x in self.W0],
            "rep_dim": self.rep_dim,
            "activation": self.activation,
            "weight_init": "normal(0,1) for W0 and p,q (as in author code)",
            "bias": False,
        }


class IndependentEncoder:
    """Simplified: one randomly initialised MLP per ensemble member.

    This is NOT CERE.  It is offered as ``--implementation simplified`` so
    that the effect of the CERE weight sharing can be studied separately;
    its results are tagged ``paper_fidelity="simplified"``.
    """

    def __init__(
        self,
        n_features: int,
        n_ensemble: int,
        hidden_dim: Sequence[int] = (500, 100),
        rep_dim: int = 20,
        activation: str = "tanh",
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.n_features = n_features
        self.r = int(n_ensemble)
        self.rep_dim = int(rep_dim)
        self.activation = activation
        self.act = _act(activation)
        self.device = device

        sizes = [int(n_features)] + [int(h) for h in hidden_dim] + [int(rep_dim)]
        self.L = len(sizes) - 1
        # per-member weights W^(l,u) ~ N(0,1)
        self.weights = []
        for m in range(self.r):
            gen = torch.Generator().manual_seed(
                (seed or 0) * (self.r + 1) + m + 1
            ) if seed is not None else None
            self.weights.append(
                [torch.randn(out, inp, generator=gen)
                 for inp, out in zip(sizes, sizes[1:])]
            )
        self._to_device()

    def _to_device(self) -> None:
        self.weights = [
            [w.to(self.device) for w in member] for member in self.weights
        ]

    def _member_forward(self, x: torch.Tensor, m: int) -> torch.Tensor:
        h = x.to(self.device)
        for l in range(self.L):
            h = h @ self.weights[m][l].t()
            if l < self.L - 1:
                h = self.act(h)
        return h

    def encode_all(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"expected (batch, {self.n_features}) got {tuple(x.shape)}")
        return torch.stack([self._member_forward(x, m) for m in range(self.r)])

    def encode_member(self, x: torch.Tensor, u: int) -> torch.Tensor:
        return self._member_forward(x, u)

    def config(self) -> dict:
        return {
            "encoder_type": "independent_mlp",
            "cere_enabled": False,
            "layers": [self.n_features] + [int(w.shape[1]) for w in self.weights[0]],
            "rep_dim": self.rep_dim,
            "activation": self.activation,
            "weight_init": "normal(0,1) per member (NOT CERE)",
            "bias": False,
        }


def make_encoder(
    n_features: int,
    n_ensemble: int,
    hidden_dim: Sequence[int],
    rep_dim: int,
    activation: str,
    device: str,
    seed: int,
    cere: bool,
):
    cls = CereEncoder if cere else IndependentEncoder
    return cls(
        n_features=n_features,
        n_ensemble=n_ensemble,
        hidden_dim=hidden_dim,
        rep_dim=rep_dim,
        activation=activation,
        device=device,
        seed=seed,
    )
