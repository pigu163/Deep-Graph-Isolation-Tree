# MAD — Multi-Scale Anomaly Detection in Link Streams

Paper-guided reproduction of *MAD: Multi-Scale Anomaly Detection in Link
Streams* (Bautista, Brisson, Bothorel, Smits — WSDM '24).  The algorithm is
ported from the authors' reference implementation
(https://gitlab.imt-atlantique.fr/publications1/mad, EUPL-1.2) and wired to
the repository's canonical link-stream pipeline (`script/dataset/linkstream`).

## Input / data model

MAD consumes the canonical event stream `(t, u, v)` from
`data/processed/<dataset>/events.parquet` (ordered by `event_id`, stable
temporal sort).  It never converts the stream into a static graph or a
tabular feature matrix.

* History is **strictly past-only**: the window of a query at slice `t`
  ends at `t-1`; current and future events never enter it.  Within the same
  timestamp, events are processed in `event_id` order.
* Per-slice processing order (task Sec. 9.4):
  `query history -> estimate P -> score current query -> update state`.

## Slice policy (discrete time)

| dataset       | policy           | justification |
|---------------|------------------|--------------|
| paysim        | native-step      | timestamp IS the fraud step (1..743) |
| elliptic_pp   | native-step      | timestamp is the time step (1..49) |
| midas_darpa   | native-step      | timestamp is an integer tick (1..46572) |
| ibm_hi_small  | fixed-width 3600s | Unix-seconds timestamps spanning ~18 days; 1h bins give ~424 slices of ~12K transactions each (day bins would leave only ~18 slices; minute bins would be mostly empty). Time-scale choice, not label-tuned. |
| ibm_li_small  | fixed-width 3600s | same as ibm_hi_small |

`--window-mode event-count` is supported by `SlicePolicy` but not used by
the five datasets (their timestamps already define natural discrete time).

## Paper equations implemented

| component | status |
|-----------|--------|
| (1) subgraph joint probability (motivation) | implemented as documentation/derivation only — MAD deliberately does not use it for scoring |
| (2)(3) recursive power-of-two bisection, virtual P=0 elements | implemented in `multiscale.py` (leaf vector padded to `2^ceil(log2 M)`, virtual leaves always 0) |
| (4) `s = 1/sqrt(M) * 1_{phi_hat}(phi)` | implemented via Haar decomposition coefficient `cA` |
| (5) `w_k^(l) = sqrt(M/2^l)[1(E_{2k}^{l+1}) - 1(E_{2k+1}^{l+1})]` | implemented via Haar detail coefficients (see note below) |
| Prop. 2 (a)-(d) expectation/variance | `get_model_mean_var` (reference `multiscale_approximation` scaling of the variance graph `p(1-p)`) |
| (7) `score(x_i) = (x_i - E[x_i])^2 / sigma^2[x_i]` | implemented |
| (8) aggregated `score(1_{phi_hat})` | implemented |
| (9) `P(e) = (1/K) sum_k 1_{L(t-k, phi)}(e)` | `probability.py` (active-slice fraction, indicator semantics) |

**Scale note.**  The reference code computes the random variables with a
normalised Haar wavelet transform (`pywt.wavedec`).  Its coefficients are
exactly the random variables of (4)-(5) up to a per-level constant factor
that cancels out in the standardised score (7)/(8); the theoretical mean and
variance (`get_model_mean_var`) are computed on the same scale, so the
reference implementation and the paper formulas yield the same anomaly
score.  Our `multiscale.py` provides a numpy-only Haar transform with the
same conventions.

**Numerical stability.**  The standardised score uses a variance floor of
`1e-2` exactly as in the reference code, so all-p=0, all-p=1 and
constant-p models never produce NaN / division by zero.  `--smoothing`
(default 0.0) optionally adds `P=(count+a)/(K+2a)`; at 0.0 the paper
definition is exact.

## Window selection / stationarity test

`stationarity.py` implements the paper's stationarity test (Sec. 3.2),
restored from the reference `MAD.model.fit` / `stationarity_test`:

* for every window length k ending at the most recent slice, estimate P on
  that window (eq. 9), decompose every slice into the multi-scale random
  variables, and compare the **sample variance** (population variance over
  the window, empty slices treated as all-zero) with the theoretical
  variance of Prop. 2;
* the fitness is `|| th_var - sample_var ||_2` and the window with the
  smallest fitness is retained (ties keep the largest window, matching the
  reference `fitness <= best_fit`).

Three window modes: `fixed` (default, `--window K`), `expanding`, and
`stationary`.  Only `stationary` is the paper's automatic selection.  The
main edge-event experiment defaults to `fixed` because the stationary scan
is O(#windows × #slices × M) per query and does not scale to the 5-7M event
datasets; the stationary mode is fully implemented and can be exercised on
small data (`--window-mode stationary --max-samples N`).

## Query modes

| mode | phi | status |
|------|-----|--------|
| `active-edge` (default) | `{(u,v)}` — edge-query special case, M=1 | fully implemented, O(1) per event |
| `local-candidate-set` | `{(u,v)}` + up to `--max-candidates` past out-relations of u and in-relations of v (from the temporal index) | implemented; slower (per-event multi-scale), use `--max-samples` on large data |
| `full-scope` | all relations active in the queried slice (restricted scope) | implemented as a restricted scope; a global `E = V x V` space is never enumerated and no dense matrix is materialised (infeasible for PaySim's ~9M accounts) |

`active-edge` detects unexpected-active relations (densification).  Since it
only emits scores for present events, expected-but-missing relations
(sparsification) are captured by `local-candidate-set` / `full-scope`, where
missing relations of the query set are scored with `observed = 0`.

Query API on the `MAD` object: `score_edge(t, src, dst)`,
`score_node(t, node_id, direction="out"|"in"|"both")`,
`score_slice(t, relation_scope=None)` and
`score_subgraph(t, relations)`.

## Interpretability

Every scored query returns, in addition to the aggregate score (eq. 8), the
per-variable standardised contributions; the column `selected_scale` /
`selected_relation_count` report the scale `s` / `w^(l)` and the size of the
relation subset that contributed the most, and `explanation` is a
human-readable description (also stored in `scores.parquet`).

## Complexity

* Reference claim: O(N·M) per query (N history length, M = |phi|).
* Our batch scorer processes events **slice by slice** with an incremental
  sliding-window count (active slices per relation) and numpy vectorisation:
  `active-edge` is O(1) per event plus O(|active slice|) for the window
  update, no per-event history scan, no dense matrix.
* `local-candidate-set` costs O(M log M) per event (M <= max-candidates).
* `stationary` window selection costs O(#windows × #slices × M).

## Running

```bash
cd /home/zhu/Deep Graph Isolation Tree
/home/zhu/.conda/envs/IF/bin/python script/model/MAD/runner.py \
    --dataset paysim --protocol chronological --seed 42 --device cpu \
    --query-mode active-edge
```

Outputs: `result/MAD/<dataset>/<protocol>/seed_<seed>/{config.json,
metrics.json, scores.parquet, timing.json, warnings.json, run.log}`.

`scores.parquet` columns: `sample_id, timestamp_or_step, label,
label_available, anomaly_score, split, query_mode, selected_scale,
selected_relation_count, history_window, explanation`.

## Fidelity statement

* **Faithful**: equations (2)-(9), Proposition 2, the standardised score
  (7)-(8) with the reference variance floor, the stationarity test of
  Sec. 3.2, and the reference Haar-based decomposition.
* **Partial / documented deviations**:
  1. The main edge-event experiment defaults to a `fixed` window rather than
     the (faithful, but expensive) `stationary` selection; this is a
     scalability choice, recorded in `config.json`.
  2. `full-scope` is restricted to the observed relation space of each slice
     (the paper's arbitrary-subgraph framework is otherwise fully supported
     through `score_subgraph` / `score_relations`).
  3. Paper eq. (1) is not used for scoring (neither does the reference
     implementation); it is the motivating derivation only.
* MAD is deterministic; the `--seed` is recorded for reproducibility checks.
