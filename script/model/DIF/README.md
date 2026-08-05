# DIF — Deep Isolation Forest

Paper-guided reproduction of *Deep Isolation Forest for Anomaly Detection*
(IEEE TKDE), following paper Sec. 5 and the authors' public code
(`xuhongzuo/deep-iforest`).

## Components

| Component | Status | Where |
|---|---|---|
| Random representation ensemble | ✅ paper-faithful | `random_encoder.py` |
| CERE (Eq. 3-6) | ✅ implemented (faithful mode) | `random_encoder.CereEncoder` |
| Deep isolation trees (Alg. 1) | ✅ implemented (own DIT, no sklearn dependency) | `deep_isolation_tree.py` |
| Path length (Alg. 2) | ✅ implemented | `deep_isolation_tree.traverse` |
| Average split deviation `g` (Eq. 7) | ✅ implemented | `deep_isolation_tree.traverse` |
| DEAS score (Eq. 8) | ✅ implemented | `scoring.py` |
| Batched inference | ✅ implemented | `DIF.score` |
| Deterministic seeding | ✅ implemented | `DIF.fit` / `set_seed` |

### CERE (paper Eq. 3-6)

Every layer `l` keeps one shared base weight `W0^(l) ~ N(0,1)` and, per
ensemble member `i`, random vectors `p^(l,i)`, `q^(l,i) ~ N(0,1)`:

    W_i = W0 ◦ (p_i q_i^T)                       (Eq. 3)
    W_i^T x = W0^T (x ◦ p_i) ◦ q_i               (Eq. 4)
    [XW_1;…;XW_r] = ((X⊗1_r) ◦ P) W0 ◦ Q         (Eq. 5)

`encode_all` implements the batched form (Eq. 5) for scoring; `encode_member`
implements the single-member form (Eq. 4) used when growing trees on a
subsample. Both are identical for the same member (covered by a unit test).

### DEAS (paper Eq. 7-8)

    g(o|τ)      = (1/|p|) Σ_k |o(j_k) − η_k|       (Eq. 7)
    F_DEAS(o|T) = 2^{−E|p|/C(T)} × E[g]            (Eq. 8)

`C(T) = c(n)` is the iForest normalising factor at the subsample size. The
aggregation follows the paper formula: means are taken over **all** trees.

## Modes

* `--implementation faithful` (default): CERE + DIT + DEAS.
  `paper_fidelity = "faithful"`.
* `--implementation simplified`: independent random MLPs (no CERE) + DIT +
  DEAS.  `paper_fidelity = "simplified"` — **not** a paper reproduction.
* `--no-deas`: depth-only ablation score (iForest-style).

## Config defaults and deviations

Paper-specified: `r=50`, `t=6`, `T=300`, `n=256`, mini-batch `b=64`,
max depth `J=8`.

Paper **does not** specify the MLP depth / width / activation /
initialisation / representation post-processing.  Documented fallback
(following the authors' public code, configurable via CLI):

* hidden layers `[500, 100]`, representation dim `20`, `tanh` activation;
* weights `~ N(0,1)`, no bias;
* per-representation z-score (fitted on a random **train** subset) then
  `tanh` post-processing — scaler never sees test data;
* no leaf-size path correction in DEAS (paper Eq. 8 uses raw path length).

All deviations are written into every `result/DIF/.../config.json`
(`deviation_notes`).

## Elliptic++ transaction features

`script/tabular_pipeline/datasets.py` rebuilds transaction-node samples from
`data/elliptic_pp/txs_features.csv` (no header: `column_0` = txId,
`column_1` = time_step, `column_2..` = features).  The dataset ships 165
features; we assume the original Elliptic layout — first 93 features are
*local*, remaining 72 are *aggregated*.  The pipeline builds the **`local`**
subset (93 features) by default and offers `elliptic_features="all"` for the
full 165.  The choice is recorded in `data/processed_tabular/elliptic_pp/
metadata.json` / `feature_schema.json`.  **Assumption caveat**: the exact
local/aggregated boundary could not be verified at build time without a
runtime check; if the shipped files differ from the 93/72 split, the boundary
in `script/tabular_pipeline/schema.py` (`ELLIPTIC_LOCAL_COUNT`) must be
updated accordingly.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/DIF/runner.py \
  --dataset paysim --protocol chronological --seed 42 [--overwrite] \
  [--implementation faithful|simplified] [--device cpu] [--max-samples N] [--quick]
```

Outputs to `result/DIF/<DATASET>/<PROTOCOL>/seed_<SEED>/` (config.json,
metrics.json, scores.parquet, timing.json, warnings.json, run.log).

## Fairness

IF and DIF consume the **same** tabular samples/features/splits from
`script/tabular_pipeline/`; only the anomaly-scoring algorithm differs.
