"""Multi-scale analysis of sub-graph activity (paper Section 3.1).

This module implements the random-variable decomposition that is the core of
MAD:

    phi            : arbitrary set of M relations (directed (u, v) pairs)
    phi_hat        : set of relations that are ACTIVE in the queried slice
    1_{phi_hat}    : indicator / weight function, 1 if active else 0
    P(e)           : estimated probability of relation e being active

Random variables (paper eq. 4-5):

    s      = (1/sqrt(M)) * 1_{phi_hat}(phi)                     (eq. 4)
    w_k^(l) = sqrt(M/2^l) * [1_{phi_hat}(E_{2k}^{l+1})
                             - 1_{phi_hat}(E_{2k+1}^{l+1})]     (eq. 5)

Their expectation / variance under the independent-Bernoulli generative
model are given by Proposition 2 (a)-(d) and the per-variable anomaly
score by eq. (7), aggregated into a single score by eq. (8).

Implementation follows the *author's reference code* (GitLab
publications1/mad, EUPL-1.2) which computes the decomposition with a
normalised Haar wavelet transform (``pywt.wavedec`` / ``pywt.idwt``).  The
wavelet coefficients equal the random variables of eq. (4)-(5) up to a
per-level constant factor that cancels out in the standardised score, so the
paper formulas and the reference implementation yield the same anomaly
score.  A numpy-only Haar transform is provided here (no PyWavelets
dependency).

Reference-implementation details faithfully reproduced:

* relations are indexed in decreasing order of P (stable sort);
* the leaf vector is padded with zeros to the next power of two
  (virtual elements with P=0 are always inactive; they do not affect the
  real-relation statistics, because every score term only depends on the
  weighted sums of real leaves);
* ``th_var`` is obtained by the reference ``multiscale_approximation``
  (scaling coefficients of the variance graph p*(1-p));
* the standardised score uses ``var_floor = 1e-2`` exactly as in the
  reference code, which also keeps fully-degenerate models (all p=0,
  all p=1, or p constant) numerically stable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "VAR_FLOOR",
    "next_power_of_two",
    "order_relations",
    "leaf_vector",
    "haar_wavedec",
    "haar_idwt",
    "multiscale_approximation",
    "decomposition_coefficients",
    "get_model_mean_var",
    "score_query",
    "random_variable_names",
    "explain_contribution",
    "MultiScaleResult",
]

#: Variance floor used by the reference implementation to keep the
#: standardised score well-defined when the theoretical variance vanishes.
VAR_FLOOR = 1e-2


def next_power_of_two(x: int) -> int:
    """Smallest power of two >= x (0 -> 0, matching the reference code)."""
    if x <= 0:
        return 0
    return 1 << math.ceil(math.log2(x))


# ---------------------------------------------------------------------------
# Relation ordering (paper: indexed in decreasing order of probability)
# ---------------------------------------------------------------------------


def order_relations(
    relations: list[tuple[int, int]],
    probs: np.ndarray,
) -> np.ndarray:
    """Stable descending order of relation indices by probability.

    ``relations[i]`` has probability ``probs[i]``.  Returns an array of
    indices such that ``probs[order]`` is non-increasing; ties keep their
    input order (stable sort, like the reference ``sorted(key=..., reverse=)``).
    """
    return np.argsort(-np.asarray(probs, dtype=np.float64), kind="stable")


def leaf_vector(
    weights: np.ndarray,
    order: np.ndarray,
    mem_size: int,
) -> np.ndarray:
    """Map per-relation weights onto the padded leaf vector.

    ``weights`` is indexed by the input relation order, ``order`` maps input
    index -> rank.  The returned array has ``mem_size`` entries
    (power of two); virtual leaves stay zero.  ``weights`` may also be a
    probability or variance array (same shape).
    """
    w = np.asarray(weights, dtype=np.float64)
    leaves = np.zeros(mem_size, dtype=np.float64)
    leaves[np.arange(len(order))] = w[order]
    return leaves


# ---------------------------------------------------------------------------
# Normalised Haar wavelet transform (PyWavelets-compatible, numpy-only)
# ---------------------------------------------------------------------------


def haar_wavedec(data: np.ndarray) -> list[np.ndarray]:
    """Multi-level Haar decomposition, returns [cA_n, cD_n, ..., cD_1].

    Exactly like ``pywt.wavedec(data, 'haar')``: level 1 applies
    a = (x0+x1)/sqrt(2), d = (x0-x1)/sqrt(2) to consecutive pairs, then
    recurses on the approximation.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    if n == 0:
        return [data]
    if n == 1:
        return [data.copy()]
    current = data.copy()
    details: list[np.ndarray] = []
    while len(current) > 1:
        a = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        d = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        details.append(d)
        current = a
    out = [current]
    out.extend(reversed(details))  # [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    return out


def haar_idwt(c_approx: np.ndarray, c_detail: np.ndarray) -> np.ndarray:
    """Single-level inverse Haar transform (``pywt.idwt`` 'haar')."""
    ca = np.asarray(c_approx, dtype=np.float64)
    cd = np.asarray(c_detail, dtype=np.float64)
    n = len(ca)
    out = np.empty(2 * n, dtype=np.float64)
    out[0::2] = (ca + cd) / np.sqrt(2.0)
    out[1::2] = (ca - cd) / np.sqrt(2.0)
    return out


def multiscale_approximation(dec: list[np.ndarray]) -> list[np.ndarray]:
    """Reference ``multiscale_approximation``: scaling coefficients of every
    level, reconstructed by successive inverse transforms starting from the
    coarsest approximation coefficient."""
    scaling = [dec[0]]
    for l in range(1, len(dec)):
        scaling.append(haar_idwt(scaling[-1], dec[l]))
    return scaling


def decomposition_coefficients(
    weights: np.ndarray,
    order: np.ndarray,
    mem_size: int,
) -> np.ndarray:
    """Random-variable vector x for a weight function (eq. 4-5, reference scale).

    ``weights``: per-relation observed values (0/1 for a query, p for a
    probability graph, p*(1-p) for a variance graph).
    """
    leaves = leaf_vector(weights, order, mem_size)
    return np.hstack(haar_wavedec(leaves))


# ---------------------------------------------------------------------------
# Theoretical mean and variance (Proposition 2, reference scale)
# ---------------------------------------------------------------------------


def get_model_mean_var(
    probs: np.ndarray,
    order: np.ndarray,
    mem_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (th_mean, th_var) of the random-variable vector under the
    independent-Bernoulli model, computed exactly as the reference
    ``get_model_mean_var``.

    Args:
        probs: per-relation probabilities P(e) (input relation order).
        order: descending-probability order (see :func:`order_relations`).
        mem_size: power-of-two leaf-vector length.
    """
    probs = np.asarray(probs, dtype=np.float64)
    th_mean = decomposition_coefficients(probs, order, mem_size)

    # Variance graph: e -> p(e) * (1 - p(e))
    q = probs * (1.0 - probs)
    var_dec = haar_wavedec(leaf_vector(q, order, mem_size))
    var_dec = multiscale_approximation(var_dec)
    for level in range(len(var_dec)):
        var_dec[level] = var_dec[level] / np.sqrt(mem_size / (2 ** level))
    # Reference: th_var = hstack([var_dec[0]] + var_dec[0:-1])
    th_var = np.hstack([var_dec[0]] + var_dec[0:-1])
    return th_mean, th_var


# ---------------------------------------------------------------------------
# Scoring (paper eq. 7-8 + reference var floor)
# ---------------------------------------------------------------------------


@dataclass
class MultiScaleResult:
    """One multi-scale query evaluation."""

    score: float
    n_relations: int
    mem_size: int
    x: np.ndarray
    th_mean: np.ndarray
    th_var: np.ndarray
    per_variable_score: np.ndarray
    selected_scale: int
    selected_relation_count: int
    explanation: str
    notes: list[str] = field(default_factory=list)


def random_variable_names(mem_size: int) -> list[str]:
    """Human-readable names for the M random variables (reference order)."""
    if mem_size <= 1:
        return ["s"]
    names = ["s"]
    level = 0
    while (1 << level) < mem_size:
        count = 1 << level
        for k in range(count):
            names.append(f"w^{level}({k})")
        level += 1
    return names


def explain_contribution(
    per_variable_score: np.ndarray,
    mem_size: int,
) -> tuple[int, int, str]:
    """Map the largest per-variable contribution to a scale and a relation
    count.

    Reference variable ordering: x[0] = s (scale 0, all M relations), the
    following blocks of size 2^l correspond to w^{(l)} whose parent node
    E_k^{(l)} contains M / 2^l relations.
    """
    per_variable_score = np.asarray(per_variable_score, dtype=np.float64)
    if len(per_variable_score) == 0:
        return 0, 0, "no variables"
    i = int(np.argmax(per_variable_score))
    if i == 0 or mem_size <= 1:
        return 0, int(mem_size), f"scale s (all {mem_size} relations)"
    level = int(math.floor(math.log2(i)))
    rel_count = mem_size // (1 << level)
    return level, rel_count, f"scale w^({level}) (subset of {rel_count} relations)"


def score_query(
    observed: np.ndarray,
    probs: np.ndarray,
    order: np.ndarray,
    mem_size: int,
    var_floor: float = VAR_FLOOR,
) -> MultiScaleResult:
    """Anomaly score of one query (paper eq. 8).

    Args:
        observed: per-relation binary activity (1 = active, 0 = inactive).
        probs: per-relation probabilities P(e) estimated from the history.
        order: descending-probability relation order.
        mem_size: power-of-two padded number of relations.
        var_floor: variance floor (reference default 1e-2).
    """
    observed = np.asarray(observed, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    x = decomposition_coefficients(observed, order, mem_size)
    th_mean, th_var = get_model_mean_var(probs, order, mem_size)

    denom = np.maximum(th_var, var_floor)
    per_var = (x - th_mean) ** 2 / denom
    total = float(np.sum(per_var))

    selected_scale, rel_count, expl = explain_contribution(per_var, mem_size)
    return MultiScaleResult(
        score=total,
        n_relations=int(len(probs)),
        mem_size=int(mem_size),
        x=x,
        th_mean=th_mean,
        th_var=th_var,
        per_variable_score=per_var,
        selected_scale=int(selected_scale),
        selected_relation_count=int(rel_count),
        explanation=expl,
    )
