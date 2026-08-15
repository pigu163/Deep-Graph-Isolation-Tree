# Supervised vs Unsupervised on IBM AML HI-Small — Why GFP+XGBoost Works and IsolationForest Fails

> **Audience:** AI coding agents (Codex, etc.) that must understand *why* the
> supervised GFP+XGBoost pipeline achieves Test F1 ≈ 0.61 while the unsupervised
> GFP+IsolationForest pipeline is indistinguishable from random (ROC-AUC ≈ 0.51),
> on the **same feature set**.
>
> **Companion docs:** `GFP_XGBOOST_PIPELINE.md` (how the supervised pipeline is
> built), `xgboost_test/notebook_pipeline.py` (supervised, cell-per-stage),
> `xgboost_test/notebook_if_unsupervised.py` (unsupervised variant).
>
> **Headline conclusion:** the gap is **not** a code bug, not a feature bug, and
> not a tuning problem. It is a fundamental mismatch between the *algorithmic
> assumption* of IsolationForest ("anomaly = outlier in feature space") and the
> *statistical nature* of money-laundering labels ("anomaly = conditional
> deviation from normal behaviour, embedded inside the normal distribution").

---

## 1. The two experiments (identical features, different supervision)

| | Supervised | Unsupervised |
|---|---|---|
| Pipeline | `xgboost_test/notebook_pipeline.py` | `xgboost_test/notebook_if_unsupervised.py` |
| Features | GFP (124 cols) + raw (12 cols) = 132 | **identical** GFP (124) + raw (12) = 132 |
| GFP config | fan/degree=False, 2-day / 12h windows | **identical** |
| Split | temporal 60/20/20 | temporal 70/30 |
| Model | XGBoost (`scale_pos_weight`≈20, Optuna) | `sklearn.ensemble.IsolationForest` (300 trees, max_samples=256, contamination="auto") |
| Uses labels | yes (train + val threshold) | **no** |

Both pipelines stream the same edges through SNAP ML's GFP in timestamp order
(batch=128, no future leakage), so the input features are essentially identical.
The **only** difference is the model and whether labels are used.

---

## 2. Results

| Metric | Supervised GFP+XGBoost | Unsupervised GFP+IsolationForest |
|---|---|---|
| Test F1 (threshold on val) | **0.61** | — (no threshold; top-k instead) |
| AUPRC / PR-AUC | **0.60** | **0.0016** |
| ROC-AUC | — | **0.5075** |
| Precision@k (k = #pos in test) | — | **0.0009** (2 hits out of 2321) |
| Recall@k | — | 0.0009 |
| Random baseline (PR-AUC) | 0.00152 (class prior) | 0.00152 |

Interpretation of the unsupervised numbers:

- **ROC-AUC 0.5075 ≈ 0.5** → ranking ability ≈ coin flip.
- **PR-AUC 0.0016 ≈ class prior (0.00152)** → the model extracts *zero*
  discrimination; it is equivalent to not predicting at all.
- **Precision@2321 = 0.0009** → of the 2321 "most anomalous" transactions,
  only **2** are actually laundering. Random sampling of 2321 rows would already
  expect ≈3.5 hits — i.e. the unsupervised top-k is *worse than random draw*.

---

## 3. Diagnostic evidence (measured, not speculated)

All diagnostics below were run on the real HI-Small data (test split, 1,523,504
transactions, 2,321 positives).

### 3.1 The anomaly scores barely separate the classes

```
positive-score median percentile (vs all test scores): 0.537   (0.5 = random)
```

The median laundering transaction is only ~3.7 percentile points above the
median normal transaction. The signal is so weak that any ranking is dominated
by noise.

### 3.2 What IS in the top-k — the only "isolated" positives are extreme-value ACH

The few positives that do rank high are all **ultra-large ACH transfers**:

```
score=0.786  pay_fmt=ACH  amount=876,934,779
score=0.786  pay_fmt=ACH  amount=930,030,659
score=0.775  pay_fmt=ACH  amount=244,971,628
...
```

IsolationForest can only isolate positives that are *statistical outliers in
amount* (hundreds of millions). But:

- 100% of the top-0.1% largest normal transactions also rank in the top-10% of
  anomaly scores, i.e. the "extreme amount" slot is flooded by normal money
  movement;
- the majority of laundering transactions are **not** extreme in amount, so they
  never get isolated.

### 3.3 Laundering transactions are inside the normal distribution, not outside it

```
laundering log10(amount) median : 4.00
normal     log10(amount) median : 3.10
fraction of laundering amounts inside the normal IQR : 0.454
```

- 45% of laundering transactions have amounts squarely inside the inter-quartile
  range of normal transactions.
- ACH is the dominant laundering channel (≈86% of positives) but is also a
  common payment method overall — not a rare attribute.
- The known discriminating signals are *conditional*: ACH **given** laundering
  patterns, graph cycles, short-time multi-hop flows, amount sequences. None of
  these make a single transaction *sparse* in feature space.

### 3.4 Ablation: amount-only IsolationForest is even below random

```
1-feature (log amount) IsolationForest: ROC-AUC = 0.476
```

Even on the single most skewed raw feature, density-based isolation fails.

---

## 4. Root-cause analysis

### 4.1 IsolationForest's assumption is violated by this data

IsolationForest scores a point by how *easily it can be separated by random
axis-aligned cuts* — i.e. how **isolated/sparse** the point is. Its implicit
model is:

> anomaly ⇔ far from the bulk of the data in feature space.

Money laundering violates this in both directions:

1. **Most laundering is not far from the bulk.** 45% of laundering amounts sit
   inside the normal IQR; laundering is a *behavioural pattern* (who pays whom,
   how often, in what sequence), not a single-feature extreme.
2. **Much of what *is* far from the bulk is normal.** Huge transfers, odd hours,
   rare currencies — these are mostly legitimate. The "isolated" region of
   feature space is dominated by normal-but-extreme activity, which floods the
   top-k.

Net effect: the IF ranking ≈ random w.r.t. the laundering label (ROC-AUC 0.51),
and its top-k is dominated by normal outliers (Precision@k < random draw).

### 4.2 Why XGBoost succeeds on the same features

XGBoost is a **conditional supervised model**: it optimizes a loss against the
label, so it can learn rules of the form

> P(laundering | amount bucket, pay_fmt=ACH, src_fan_out high, cycle count ≥ 1,
>   sum_col4_in high, ...)

It does not need laundering to be *isolated*; it needs laundering to be
*linearly/conditionally separable in aggregate*. The GFP features are exactly
the conditional discriminators (graph-pattern histograms, vertex statistics,
ACH channel) — they are weak in isolation but highly predictive in combination
with the label. With `scale_pos_weight` and a validation-tuned threshold, the
supervised model reaches Test F1 0.61 / AUPRC 0.60.

### 4.3 The feature set is not the problem

Both pipelines used identical GFP + raw features. The same GFP features that
yield AUPRC 0.60 under XGBoost yield AUPRC 0.0016 under IsolationForest.
**Conclusion: the features carry the signal; density-based unsupervised models
cannot extract conditional signal from them.** Adding more/better features would
not fix IsolationForest — the bottleneck is the model's assumption, not the
feature representation.

### 4.4 Extreme class imbalance amplifies the failure

0.10% positive rate (2321 of 1.5M in test). Even a *small* ranking advantage
gets swamped: top-2321 is filled by normal "looks-anomalous" rows. Under
supervision the model can calibrate toward precision/recall on the minority
class; unsupervised has no such lever.

---

## 5. Actionable implications for agents

1. **Do not "fix" IsolationForest by adding features or tuning** on this task —
   the gap is architectural. Expect any density-based detector (ECOD, LOF,
   OCSVM on raw features) to behave similarly on HI-Small.
2. **If an unsupervised baseline is required** (e.g. for a paper comparison),
   use graph/temporal anomaly scores instead of feature density: e.g. MIDAS /
   StreamSpot-style edge-anomaly scoring over the temporal edge stream, or
   GFP-feature *deviation* scores computed per-vertex against historical
   distributions — anything that models *behavioural deviation* rather than
   *point isolation*.
3. **The supervised pipeline is the correct primary baseline** for this
   benchmark: GFP (streaming, no future leak) + raw features + XGBoost with
   `scale_pos_weight`, Optuna on validation F1, validation threshold, single
   frozen test evaluation.
4. **When reporting unsupervised results**, use the corrected top-k convention:
   `precision@k = hits/k`, `recall@k = hits/#pos` (do **not** use
   `sklearn.recall_score(y_top, ones)`, which is identically 1.0 and
   meaningless — see `notebook_if_unsupervised.py` Cell 9 for the corrected
   implementation).
5. **Interpretation template** for the final report: supervised ≈ 0.61 F1;
   unsupervised ≈ random (ROC-AUC ≈ 0.51, PR-AUC ≈ class prior). State that the
   gap is due to conditional-vs-density modelling, cite §3 diagnostics.

---

## 6. One-line summary

> GFP+XGBoost works because laundering is a **conditional** deviation that
> supervised models can learn from the label; IsolationForest fails because
> laundering is **not an outlier in feature space** — it hides inside the normal
> distribution, and the extreme region of feature space is occupied by normal
> activity. The negative unsupervised result is a property of the problem, not
> of the implementation.

---

## 7. 2026-08-10 — Orchestra, Link Stream, Velocity and SHAP Update

### 7.1 Agent hand-off: scope and protocol

This section contains only experiments completed on local IBM AMLWorld files
under `data/ibm_aml/SMALL`. New feature experiments ran in memory: no new
Python files were created.

- Every new result uses stable timestamp ordering and earliest 70% / latest 30%
  temporal split.
- IF: 300 trees, `max_samples=256`, seed 42, all CPU cores. In Orchestra-style
  tests, `contamination` is the actual positive rate of the 70% train period.
  It affects the threshold, **not** ROC-AUC, AP, or score ordering.
- `k = number of test positives`; `Precision@k = hits/k`; `Recall@k =
  hits/number_of_test_positives`.

| Dataset | Rows | Unique raw `From Bank ∪ To Bank` IDs |
|---|---:|---:|
| HI-Small | 5,078,345 | 30,470 |
| LI-Small | 6,924,049 | 41,815 |

### 7.2 Feature implementations and meanings

#### A. Causality rule

For link-stream features, each transaction is a directed temporal edge
`e_i=(t_i,u_i,v_i,a_i)`: time, source account, destination account, paid amount.
For all events with the same timestamp, calculate features from history `< t_i`
for the complete group, then add that group to the state. This is strict
score-before-update: no future or same-timestamp transaction can contribute.

SNAP ML GFP is related but not identical. Its directed sliding graph exposes
fan/degree (temporal-neighbourhood statistics) plus `scatter-gather`,
`temp-cycle`, and `lc-cycle` (temporal-motif histograms). `transform(batch)`
inserts a full batch before calculating output, so `batch=128` prevents leakage
between batches but not inside one batch. Do not call GFP output strict pre-event
state.

#### B. Velocity: 80 features

For entity `g ∈ {source account, destination account, source bank, destination
bank}`, and each historical window `Δ ∈ {1h,6h,24h,7d}`, calculate:

| Statistic | Definition | Meaning |
|---|---|---|
| `count` | prior transaction count | activity burst |
| `sum` | sum of prior paid amounts | recent value volume |
| `mean` | mean prior paid amount | normal ticket size |
| `std` | standard deviation of prior amounts | ticket-size volatility |
| `max` | maximum prior paid amount | recent extreme-value context |

`4 entities × 4 windows × 5 statistics = 80` dimensions. Counts/sums use
`log1p` where appropriate.

#### C. Local causal link stream: 10 features

All windows are `[t_i-Δ,t_i)`.

| Feature | Definition | Meaning |
|---|---|---|
| `pair_count_Δ` (4) | prior `u_i → v_i` edges | recurrence of the exact ordered account pair |
| `reverse_pair_count_Δ` (4) | prior `v_i → u_i` edges | reciprocity / return-flow motif |
| `src_gap` | `log1p(min(t_i-last_out(u_i),7d))`; 7d if unseen | source inter-contact time |
| `dst_gap` | `log1p(min(t_i-last_in(v_i),7d))`; 7d if unseen | destination inter-contact time |

Do not use cumulative source/destination counts, unique-neighbour counts,
cumulative pair counts, or in/out balances in IF. They grow monotonically over
time, so later normal activity becomes artificially isolated.

### 7.3 IF results: Orchestra and link-stream ablations

[Orchestra](paper/Orchestra-%20A%20Heterogeneous%20Ensemble%20Architecture%20for%20Interbank%20Financial%20Anomaly%20Detection.pdf)
states temporal/transaction/categorical features, 80 velocity dimensions, and
128 source/destination-bank Node2Vec dimensions. The listed components total
228, although the paper also says 146; this is unresolved.

The local Orchestra approximation used transaction/time features + 80 velocity
+ 128 training-period-only bank random-walk embeddings. It is scope-adapted,
not byte-for-byte reproduction.

| IF features | Dataset | ROC-AUC | AP | Hits@k | P@k | R@100 | R@200 |
|---|---|---:|---:|---:|---:|---:|---:|
| velocity + transaction, no Node2Vec (78 dims) | HI | 0.690666 | 0.010893 | 83 | 0.035760 | 0.002154 | 0.004739 |
| same | LI | 0.598812 | 0.001741 | 11 | 0.008246 | 0.000000 | 0.000750 |
| Orchestra-like + 128 bank embeddings (228 dims) | HI | 0.694348 | 0.016452 | 139 | 0.059888 | 0.000862 | 0.002154 |
| same | LI | 0.579712 | 0.001405 | 9 | 0.006747 | 0.000000 | 0.000750 |
| raw + windows + cumulative link-stream quantities | HI | 0.415855 | 0.001192 | 1 | 0.000431 | 0 | 0 |
| same | LI | 0.415373 | 0.000630 | 0 | 0.000000 | 0 | 0 |
| raw + local Δ-counts + gaps only | HI | 0.469004 | 0.001310 | 0 | 0.000000 | 0 | 0 |
| same | LI | 0.501802 | 0.000697 | 0 | 0.000000 | 0 | 0 |

Interpretation: Node2Vec helps HI AP but hurts LI AP/top-200 recall. Pure
link-stream quantities are not sparse point-cloud signals, so IF cannot use
them as a replacement for supervision. They are still useful conditional
features for XGBoost.

### 7.4 Supervised GFP + velocity + link stream

HI-Small used `120 GFP + 12 raw + 80 velocity + 10 link-stream = 222`
features. Parameters were fixed, no Optuna: 2,000 trees, depth 7, learning rate
0.04, subsample 0.786, colsample 0.804, min-child-weight 14, gamma 1.26,
scale-pos-weight 20, reg-lambda 10.57, `hist`, 64 threads, seed 42.

| Dataset | ROC-AUC | AP | Best F1 on test | Test-derived threshold |
|---|---:|---:|---:|---:|
| HI-Small | **0.986969** | **0.654571** | 0.663447 | 0.866983 |

The F1 is an optimistic diagnostic because the threshold was selected on test.
For a reportable F1, use early train / middle validation / late test, choose
threshold on validation, then evaluate the test period once.

SHAP used `TreeExplainer` on all held-out positives plus 20,000 fixed random
held-out negatives. Mean absolute SHAP leaders:

| Rank | Feature | Mean abs SHAP | Meaning |
|---:|---|---:|---|
| 1 | `pay_fmt` | 1.336931 | payment-channel conditional discriminator |
| 2 | `pair_count_7d` | 1.310390 | one-week exact-pair recurrence |
| 3 | `gfp_084` | 0.621084 | GFP source-incoming vertex time statistic |
| 4 | `gfp_070` | 0.515542 | GFP source-outgoing fan/degree ratio |
| 5 | `src_count_1h` | 0.440709 | one-hour source activity burst |
| 9 | `Amount Paid` | 0.301125 | amount context |
| 11 | `reverse_pair_count_7d` | 0.260431 | one-week reciprocity / return flow |
| 13 | `src_gap` | 0.203903 | source inter-contact time |
| 14 | `pair_count_1h` | 0.196807 | one-hour pair recurrence |

Practical conclusion: `pair_count_7d` is second overall, so pair-level
link-stream recurrence is highly useful under XGBoost supervision; IF cannot
exploit it by itself.

### 7.5 Orchestra reproducibility warning and next action

Orchestra reports LI-Small has 68 banks and 4,018 positives. Local LI has the
matching row/label scale (6,924,049 rows and 4,018 positives) but 41,815 raw
bank IDs. Thus 68 is not the raw count. It may be undisclosed aggregation, a
subgraph, another data variant, or a reporting error; no intent can be inferred.
Do not claim exact Orchestra IF reproduction until its bank mapping, graph
training period, filters, and final 146-vs-228 feature list are resolved.

Recommended supervised candidate: `GFP + raw + 80 velocity + 10 local causal
link-stream`. Future features must be bounded and past-only (pair gaps,
reverse-pair gaps, causal two-hop counts) and tested as ablations against this
222-feature baseline.
