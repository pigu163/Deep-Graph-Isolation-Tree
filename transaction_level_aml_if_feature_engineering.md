# Transaction-Level AML Feature Engineering for Isolation Forest

## 1. Objective

This document specifies a feature-engineering pipeline for **transaction-level anti-money-laundering (AML) anomaly detection**.

The prediction unit is always a single transaction/event.

For each transaction

\[
e_t = (u_t, v_t, t, x_t)
\]

where:

- \(u_t\): sender account
- \(v_t\): receiver account
- \(t\): transaction timestamp
- \(x_t\): raw transaction attributes

the pipeline must produce:

\[
e_t \rightarrow \phi(e_t) \rightarrow \text{Isolation Forest} \rightarrow s_t
\]

where:

- \(\phi(e_t)\) is the engineered feature vector for the **current transaction**
- \(s_t\) is the anomaly score for the **current transaction**
- the final label, when evaluated against ground truth, is transaction-level AML / non-AML

This project is **not** a node anomaly detection task.

Do not convert the task into:

- suspicious-account detection
- sender classification
- receiver classification
- graph-level classification
- community-level anomaly detection

Node- and graph-derived information may be used only as historical/contextual information for generating features attached to the current transaction.

---

## 2. Core Design Principle

Raw transaction rows are often weak inputs for Isolation Forest because laundering transactions may have ordinary-looking:

- amounts
- currencies
- timestamps
- payment formats
- sender/receiver IDs

The main goal of feature engineering is therefore:

> Convert hidden contextual or behavioral irregularity into explicit transaction-level numeric features that Isolation Forest can isolate.

The target representation is:

\[
\phi(e_t)
=
[
x_t^{raw},
x_t^{temporal},
x_t^{fan},
x_t^{degree},
x_t^{amount},
x_t^{velocity},
x_t^{historical\_deviation},
x_t^{flow},
x_t^{counterparty},
x_t^{pair},
x_t^{motif}
]
\]

Every row of the final feature table must correspond to exactly one original transaction.

---

## 3. Critical Temporal Constraint: Past-Only Features

For transaction \(e_t\), feature extraction must use only information strictly available before \(t\).

Use:

\[
\mathcal{H}_{t^-}
=
\{e_i : t_i < t\}
\]

Never use:

- future transactions
- statistics computed using the complete dataset
- graph edges occurring after the current transaction
- target labels during feature construction

This rule is mandatory to avoid temporal leakage.

For events with identical timestamps, define and document a deterministic ordering policy.

Recommended default:

1. sort by timestamp
2. preserve original row order as secondary key
3. process transactions sequentially
4. update state only after features for the current row are computed

Pattern:

```python
for tx in transactions_sorted:
    features = compute_features(tx, state)   # past-only
    output.append(features)
    state.update(tx)                         # current tx becomes history
```

---

# 4. Feature Families

## 4.1 Raw Transaction Features

Purpose:

- preserve original transaction information
- provide a baseline
- allow Isolation Forest to combine raw and contextual signals

Recommended features:

```text
log_amount
hour_of_day
day_of_week
is_weekend
sender_bank
receiver_bank
same_bank
sender_currency
receiver_currency
same_currency
payment_format
```

### Amount transform

Use:

\[
\text{log\_amount}
=
\log(1+\text{amount})
\]

Prefer log amount over raw amount because transaction amounts are often highly skewed.

Raw amount may optionally be retained as an additional feature.

### Categorical variables

Isolation Forest requires numeric inputs.

Possible encodings:

- integer/frequency encoding
- one-hot encoding for low-cardinality features
- hashing for high-cardinality categories

Do **not** directly use sender or receiver IDs as arbitrary integer values.

---

# 4.2 Transaction Frequency Features

For current transaction:

\[
u \rightarrow v
\]

count historical activity over multiple trailing windows.

Recommended windows:

```text
10 minutes
1 hour
6 hours
24 hours
7 days
```

Optional:

```text
5 minutes
30 minutes
3 days
30 days
```

### Sender

```text
sender_out_tx_count_10m
sender_out_tx_count_1h
sender_out_tx_count_6h
sender_out_tx_count_24h
sender_out_tx_count_7d

sender_in_tx_count_10m
sender_in_tx_count_1h
...
```

### Receiver

```text
receiver_in_tx_count_10m
receiver_in_tx_count_1h
receiver_in_tx_count_6h
receiver_in_tx_count_24h
receiver_in_tx_count_7d

receiver_out_tx_count_10m
receiver_out_tx_count_1h
...
```

### Total activity

Optional:

```text
sender_total_tx_count_1h
receiver_total_tx_count_1h
sender_total_tx_count_24h
receiver_total_tx_count_24h
```

These features expose bursts that may be invisible in a single raw transaction.

---

# 4.3 Fan-In / Fan-Out Features

Fan counts **unique counterparties**, not total transactions.

For sender \(u\):

\[
fan^{out}_{u,\Delta}
=
|\{v' : (u,v') \in \mathcal{H}_{t^-}^{\Delta}\}|
\]

For receiver \(v\):

\[
fan^{in}_{v,\Delta}
=
|\{u' : (u',v) \in \mathcal{H}_{t^-}^{\Delta}\}|
\]

Recommended columns:

```text
sender_fan_out_10m
sender_fan_out_1h
sender_fan_out_6h
sender_fan_out_24h
sender_fan_out_7d

sender_fan_in_10m
sender_fan_in_1h
...

receiver_fan_in_10m
receiver_fan_in_1h
receiver_fan_in_6h
receiver_fan_in_24h
receiver_fan_in_7d

receiver_fan_out_10m
receiver_fan_out_1h
...
```

Fan features are especially relevant for patterns resembling:

- fan-out
- fan-in
- scatter
- gather
- account dispersion
- account consolidation

---

# 4.4 Degree Features

Degree counts transactions rather than unique counterparties.

For sender:

\[
degree^{out}_{u,\Delta}
=
|\{e_i : source(e_i)=u,\; t_i\in[t-\Delta,t)\}|
\]

Recommended columns:

```text
sender_degree_out_10m
sender_degree_out_1h
sender_degree_out_6h
sender_degree_out_24h
sender_degree_out_7d

sender_degree_in_10m
...

receiver_degree_in_10m
receiver_degree_in_1h
...
```

### Degree-to-fan derived features

Very useful:

\[
repeatRatio^{out}
=
\frac{degree^{out}}
{fan^{out}+\epsilon}
\]

Recommended:

```text
sender_repeat_ratio_out_1h
sender_repeat_ratio_out_24h
receiver_repeat_ratio_in_1h
receiver_repeat_ratio_in_24h
```

Interpretation:

High degree + low fan:

```text
A -> B
A -> B
A -> B
A -> B
```

High degree + high fan:

```text
A -> B
A -> C
A -> D
A -> E
```

These behaviors should not be treated as equivalent.

---

# 4.5 Amount Aggregation Features

For each relevant account and time window, compute historical transaction amount statistics.

Recommended statistics:

```text
sum
mean
median
std
min
max
```

Example sender outgoing features:

```text
sender_out_amount_sum_1h
sender_out_amount_mean_1h
sender_out_amount_std_1h
sender_out_amount_max_1h

sender_out_amount_sum_24h
sender_out_amount_mean_24h
sender_out_amount_std_24h
sender_out_amount_max_24h
```

Receiver incoming equivalents:

```text
receiver_in_amount_sum_1h
receiver_in_amount_mean_1h
receiver_in_amount_std_1h
receiver_in_amount_max_1h
```

Recommended multi-scale windows:

```text
1h
6h
24h
7d
```

Avoid excessive duplication if memory becomes a problem.

---

# 4.6 Interarrival and Velocity Features

## Time since previous transaction

For sender:

\[
\Delta t_u
=
t -
t^{prev}_{u}
\]

Recommended:

```text
sender_time_since_last_out_tx
sender_time_since_last_in_tx
sender_time_since_last_any_tx

receiver_time_since_last_in_tx
receiver_time_since_last_out_tx
receiver_time_since_last_any_tx
```

Use log transform if values span a large range:

\[
\log(1+\Delta t)
\]

## Transaction rate

Example:

\[
txRate_{u,\Delta}
=
\frac{N_{u,\Delta}}{\Delta}
\]

Recommended:

```text
sender_out_tx_rate_10m
sender_out_tx_rate_1h
receiver_in_tx_rate_10m
receiver_in_tx_rate_1h
```

## Amount velocity

\[
amountVelocity_{u,\Delta}
=
\frac{
\sum amount_{u,\Delta}
}{
\Delta+\epsilon
}
\]

Recommended:

```text
sender_out_amount_velocity_10m
sender_out_amount_velocity_1h
receiver_in_amount_velocity_10m
receiver_in_amount_velocity_1h
```

These features are designed to expose rapid transaction bursts.

---

# 4.7 Historical Deviation / Surprise Features

This feature family is high priority.

The idea is:

> A transaction does not need to be globally extreme. It may be abnormal relative to the sender's or receiver's own historical behavior.

For current amount \(a_t\), compute sender-relative deviation:

\[
z^{amount}_{u,t}
=
\frac{
a_t-\mu_{u,t^-}
}{
\sigma_{u,t^-}+\epsilon
}
\]

Receiver equivalent:

\[
z^{amount}_{v,t}
=
\frac{
a_t-\mu_{v,t^-}
}{
\sigma_{v,t^-}+\epsilon
}
\]

Recommended columns:

```text
sender_amount_zscore
receiver_amount_zscore

sender_amount_ratio_to_mean
receiver_amount_ratio_to_mean

sender_amount_ratio_to_median
receiver_amount_ratio_to_median
```

For count/fan/degree features:

```text
sender_degree_out_1h_zscore
sender_fan_out_1h_zscore
sender_tx_count_1h_zscore

receiver_degree_in_1h_zscore
receiver_fan_in_1h_zscore
receiver_tx_count_1h_zscore
```

General form:

\[
residual_t
=
current_t
-
expected_t
\]

or:

\[
z_t
=
\frac{current_t-\mu_{history}}
{\sigma_{history}+\epsilon}
\]

## Robust alternative

Because financial features may be heavy-tailed, optionally use:

\[
robustZ
=
\frac{x-\operatorname{median}(x)}
{1.4826 \cdot MAD(x)+\epsilon}
\]

This may be preferable to mean/std for some features.

Recommended experiment:

```text
standard_zscore
robust_zscore
```

Do not use future observations when estimating these values.

---

# 4.8 Flow Features

The purpose is to expose pass-through or flow imbalance behavior.

For account \(u\) over window \(\Delta\):

\[
I_u(\Delta)
=
\sum incoming\ amount
\]

\[
O_u(\Delta)
=
\sum outgoing\ amount
\]

## Flow imbalance

\[
flowImbalance
=
\frac{
O-I
}{
O+I+\epsilon
}
\]

## Flow-through ratio

\[
flowThroughRatio
=
\frac{
\min(I,O)
}{
\max(I,O)+\epsilon
}
\]

Recommended:

```text
sender_in_amount_1h
sender_out_amount_1h
sender_flow_imbalance_1h
sender_flow_through_ratio_1h

sender_in_amount_24h
sender_out_amount_24h
sender_flow_imbalance_24h
sender_flow_through_ratio_24h

receiver_in_amount_1h
receiver_out_amount_1h
receiver_flow_imbalance_1h
receiver_flow_through_ratio_1h
```

Possible interpretation:

A short-lived intermediary account may show:

\[
incoming \approx outgoing
\]

within a short period.

---

# 4.9 New-Counterparty Features

For current transaction:

\[
u \rightarrow v
\]

define:

\[
isNewReceiver
=
\mathbb{1}[v \notin N_u(t^-)]
\]

\[
isNewSender
=
\mathbb{1}[u \notin N_v(t^-)]
\]

Recommended:

```text
is_new_receiver_for_sender
is_new_sender_for_receiver
```

Also compute new counterparties over recent windows:

```text
sender_new_receivers_1h
sender_new_receivers_24h
sender_new_receivers_7d

receiver_new_senders_1h
receiver_new_senders_24h
receiver_new_senders_7d
```

Ratios:

\[
newCounterpartyRatio
=
\frac{
newCounterparties
}{
allCounterparties+\epsilon
}
\]

Recommended:

```text
sender_new_receiver_ratio_1h
sender_new_receiver_ratio_24h
receiver_new_sender_ratio_1h
receiver_new_sender_ratio_24h
```

---

# 4.10 Pair-Level Sender-Receiver Features

These describe the historical relationship between the specific sender and receiver.

For current pair \((u,v)\):

```text
pair_u_to_v_previous_count
pair_v_to_u_previous_count

pair_u_to_v_amount_sum
pair_v_to_u_amount_sum

pair_u_to_v_amount_mean
pair_v_to_u_amount_mean

time_since_previous_u_to_v
time_since_previous_v_to_u
```

Recommended multi-scale counts:

```text
pair_u_to_v_count_1h
pair_u_to_v_count_24h
pair_u_to_v_count_7d

pair_v_to_u_count_1h
pair_v_to_u_count_24h
pair_v_to_u_count_7d
```

Optional reciprocity feature:

\[
reciprocity_{uv}
=
\frac{
N(u\rightarrow v)
}{
N(u\rightarrow v)+N(v\rightarrow u)+\epsilon
}
\]

Other useful features:

```text
pair_is_first_transaction
pair_bidirectional_history
pair_recent_bidirectional_activity
```

---

# 4.11 Local Motif / Structural Features

These remain transaction-level features.

For current transaction:

\[
u \rightarrow v
\]

inspect the graph formed only by previous transactions.

Possible features:

```text
num_common_neighbors_uv
num_two_hop_paths_u_to_v
num_two_hop_paths_v_to_u

creates_directed_triangle
creates_short_cycle
num_short_cycles_if_added

sender_local_clustering
receiver_local_clustering
```

Important:

The feature is attached to the current transaction.

Example:

```text
transaction_12345.creates_short_cycle = 1
```

This is still transaction-level AML detection.

## Suggested priority

Start with cheap features:

```text
num_common_neighbors_uv
num_two_hop_paths_u_to_v
num_two_hop_paths_v_to_u
creates_triangle
creates_short_cycle
```

Do not begin with computationally expensive global graph algorithms.

---

# 5. Recommended Minimal Feature Set

Before building hundreds of features, implement a compact baseline.

## Tier 1: Raw

```text
log_amount
hour_of_day
day_of_week
same_bank
same_currency
payment_format
```

## Tier 2: Temporal activity

```text
sender_out_tx_count_10m
sender_out_tx_count_1h
sender_out_tx_count_24h

receiver_in_tx_count_10m
receiver_in_tx_count_1h
receiver_in_tx_count_24h
```

## Tier 3: Fan / degree

```text
sender_fan_out_1h
sender_fan_out_24h
receiver_fan_in_1h
receiver_fan_in_24h

sender_degree_out_1h
sender_degree_out_24h
receiver_degree_in_1h
receiver_degree_in_24h
```

## Tier 4: Amount

```text
sender_out_amount_sum_1h
sender_out_amount_sum_24h
receiver_in_amount_sum_1h
receiver_in_amount_sum_24h
```

## Tier 5: Velocity

```text
sender_time_since_last_out_tx
receiver_time_since_last_in_tx
sender_out_amount_velocity_1h
receiver_in_amount_velocity_1h
```

## Tier 6: Historical deviation

```text
sender_amount_zscore
receiver_amount_zscore
sender_degree_out_1h_zscore
sender_fan_out_1h_zscore
receiver_degree_in_1h_zscore
receiver_fan_in_1h_zscore
```

## Tier 7: Flow

```text
sender_flow_imbalance_1h
sender_flow_through_ratio_1h
receiver_flow_imbalance_1h
receiver_flow_through_ratio_1h
```

## Tier 8: Counterparty / pair

```text
is_new_receiver_for_sender
is_new_sender_for_receiver
sender_new_receiver_ratio_24h
receiver_new_sender_ratio_24h

pair_u_to_v_count_24h
pair_v_to_u_count_24h
time_since_previous_u_to_v
time_since_previous_v_to_u
```

Expected initial scale:

```text
~50 to 150 engineered features
```

This is enough for the first Isolation Forest experiment.

---

# 6. Recommended Implementation Priority

Implement in this order.

## Phase 1

```text
raw
temporal counts
fan
degree
amount aggregation
interarrival
```

Goal:

Establish a fast feature-engineered IF baseline.

## Phase 2

```text
historical deviation
robust z-scores
flow imbalance
flow-through ratio
new-counterparty features
pair-level history
```

Goal:

Expose contextual anomalies that raw features miss.

## Phase 3

```text
two-hop paths
common neighbors
triangle closure
short-cycle features
other local motifs
```

Goal:

Capture laundering structures that require graph context.

---

# 7. Feature Table Contract

The output must preserve one row per transaction.

Recommended schema:

```text
transaction_id
timestamp
sender_id
receiver_id

# raw
log_amount
hour_of_day
...

# temporal
sender_out_tx_count_1h
receiver_in_tx_count_1h
...

# fan / degree
sender_fan_out_1h
receiver_fan_in_1h
...

# amount aggregation
sender_out_amount_sum_1h
...

# velocity
sender_time_since_last_out_tx
...

# historical deviation
sender_amount_zscore
...

# flow
sender_flow_imbalance_1h
...

# counterparty
is_new_receiver_for_sender
...

# pair
pair_u_to_v_count_24h
...

# motif
num_two_hop_paths_u_to_v
creates_short_cycle
...

# label retained only for evaluation
is_laundering
```

The detector input should exclude:

```text
transaction_id
timestamp
sender_id
receiver_id
is_laundering
```

unless a transformed version is explicitly intended as a feature.

---

# 8. Isolation Forest Baseline

Use a standard implementation first.

Suggested baseline:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=300,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
```

Do not optimize the model too early.

The immediate scientific question is:

> Does transaction-level feature engineering make AML transactions more isolatable?

The first comparison should therefore keep the IF model fixed.

---

# 9. Required Experiments

## Experiment A: Raw IF

\[
X_{raw}
\rightarrow IF
\]

Purpose:

Establish the weak baseline already observed.

---

## Experiment B: Engineered IF

\[
X_{engineered}
\rightarrow IF
\]

Purpose:

Measure whether representation quality is the primary bottleneck.

---

# 10. Required Ablation Study

Run cumulative feature-family ablations.

Recommended sequence:

```text
A0 = Raw

A1 = Raw
   + Temporal

A2 = Raw
   + Temporal
   + Fan/Degree

A3 = Raw
   + Temporal
   + Fan/Degree
   + Amount Aggregation

A4 = A3
   + Velocity

A5 = A4
   + Historical Deviation

A6 = A5
   + Flow

A7 = A6
   + New Counterparty
   + Pair Features

A8 = A7
   + Motif Features
```

This allows the project to answer:

```text
Which feature families actually separate laundering transactions
from normal transactions for Isolation Forest?
```

---

# 11. Evaluation

Do not rely only on ROC-AUC.

AML is highly imbalanced.

Recommended metrics:

```text
ROC-AUC
PR-AUC

Precision
Recall
F1

Precision@0.1%
Precision@0.5%
Precision@1%

Recall@0.1%
Recall@0.5%
Recall@1%

F1@0.1%
F1@0.5%
F1@1%

Lift@0.1%
Lift@0.5%
Lift@1%
```

Interpretation of top-\(k\)% metrics:

The model ranks transactions by anomaly score.

For:

```text
Precision@0.1%
```

evaluate only the top 0.1% most anomalous transactions.

This is often operationally more meaningful than global F1 for AML because investigators can inspect only a small fraction of all transactions.

---

# 12. Data Leakage Checks

Before accepting any result, verify all of the following.

```text
[ ] Features use only transactions with timestamp < current transaction
[ ] Label is never used for feature extraction
[ ] Global statistics do not include future data
[ ] Graph neighborhoods are past-only
[ ] Pair history is past-only
[ ] Motif detection uses only historical edges
[ ] Current transaction is added to state only after its features are generated
[ ] Train/test split respects time when applicable
```

---

# 13. Performance Engineering

IBM AML datasets can be large.

Avoid repeatedly scanning the full historical dataframe.

Maintain incremental states.

Possible structures:

```python
sender_state: dict
receiver_state: dict
pair_state: dict
neighbor_state: dict
window_queues: dict
```

For rolling windows, prefer:

```text
deque
incremental counters
incremental sums
incremental unique-counterparty structures
```

Avoid:

```python
df[df.timestamp < current_timestamp]
```

inside the per-row loop.

That will be too slow.

For multiple windows:

```text
10m
1h
6h
24h
7d
```

maintain rolling queues/counters or an equivalent optimized implementation.

---

# 14. Important Modeling Principle

The feature extractor must convert:

```text
"this account is suspicious"
```

into:

```text
"this current transaction occurs under suspicious historical conditions"
```

Example:

Bad target:

```text
sender_is_abnormal = True
```

Preferred transaction features:

```text
sender_out_tx_count_1h = 31
sender_fan_out_1h = 27
sender_amount_zscore = 8.4
is_new_receiver_for_sender = 1
sender_flow_through_ratio_1h = 0.96
```

The output remains a transaction representation.

---

# 15. Conceptual Example

Current transaction:

```text
A -> B
amount = 480
```

Raw amount may appear normal.

Historical context:

```text
A -> B   480
A -> C   510
A -> D   495
A -> E   505
A -> F   490
...
```

Possible engineered features:

```text
log_amount = 6.175

sender_out_tx_count_10m = 18
sender_fan_out_10m = 16
sender_degree_out_10m = 18

sender_new_receiver_ratio_10m = 0.89

sender_amount_zscore = 0.25

sender_tx_count_10m_zscore = 11.4
sender_fan_out_10m_zscore = 13.2

sender_out_amount_velocity_10m = very_high
```

The AML signal is therefore not:

```text
amount = abnormal
```

but:

```text
transaction context = abnormal
```

This is exactly the kind of representation that should improve Isolation Forest.

---

# 16. Research Hypothesis

Primary hypothesis:

> Isolation Forest performs poorly on raw AML transaction rows because laundering anomalies are primarily contextual, temporal, and structural rather than simple point anomalies in raw feature space.

Secondary hypothesis:

> Past-only transaction-level behavioral features can transform laundering transactions into more isolatable points without modifying the Isolation Forest algorithm itself.

Expected pipeline:

\[
\boxed{
Raw\ Transaction
\rightarrow
Past\text{-}Only\ Transaction\ Feature\ Engineering
\rightarrow
Isolation\ Forest
\rightarrow
Transaction\ AML\ Score
}
\]

---

# 17. High-Priority Features

If only a small number of feature families can be implemented initially, prioritize:

```text
1. fan_in / fan_out
2. transaction velocity / interarrival
3. incoming / outgoing flow features
4. historical deviation / robust z-score
5. new-counterparty features
```

Then add:

```text
6. pair-level history
7. local motifs / short-cycle features
```

---

# 18. Non-Goals for This Stage

Do not implement yet unless explicitly required:

```text
TGN
GNN
ESN
DIF
deep autoencoder
supervised XGBoost
node classification
graph classification
learned embeddings
```

The purpose of this stage is intentionally narrow:

```text
Feature Engineering
        +
Standard Isolation Forest
```

This provides a clean baseline before introducing more complex anomaly detectors or link-stream encoders.

---

# 19. Deliverables Expected from a Coding Agent

A coding agent implementing this specification should produce:

```text
1. A past-only transaction feature extractor
2. One output row per original transaction
3. Configurable rolling windows
4. Incremental sender/receiver/pair state
5. Feature-family toggles for ablation
6. Isolation Forest baseline
7. Evaluation script
8. Feature-family ablation script
9. Runtime/memory logging
10. Leakage tests
```

Recommended configuration interface:

```yaml
windows:
  - 10m
  - 1h
  - 6h
  - 24h
  - 7d

features:
  raw: true
  temporal: true
  fan: true
  degree: true
  amount_aggregation: true
  velocity: true
  historical_deviation: true
  flow: true
  counterparty: true
  pair: true
  motif: false
```

Do not hard-code feature families into the evaluation pipeline.

---

# 20. Final Requirement

At every stage, preserve the following invariant:

\[
\boxed{
\text{One transaction}
\rightarrow
\text{One engineered feature vector}
\rightarrow
\text{One Isolation Forest anomaly score}
}
\]

All historical account, pair, temporal, and graph information exists only to improve the representation of the current transaction.

The final scientific target is **transaction-level AML detection**.
