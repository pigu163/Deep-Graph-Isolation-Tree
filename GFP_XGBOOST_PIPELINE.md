# GFP + XGBoost AML Pipeline — Implementation & SNAP ML Reference

> **Audience:** AI coding agents (Codex, etc.) that need to understand, modify, or
> extend this pipeline without prior knowledge of SNAP ML.
>
> **Source of truth:** the installed SNAP ML package at
> `site-packages/snapml/GraphFeaturePreprocessor.py`, `include/GfpEncoder.hpp`,
> the official example notebook under `xgboost_test/snapml-examples/`, and the
> reference pipeline in `xgboost_test/notebook_pipeline.py`.
>
> **Task context:** reproduce the IBM AML HI-Small benchmark — GFP + XGBoost,
> minority-class F1 ≥ 0.60 (paper reports 63.23). Best-known local result:
> **Test F1 = 0.6105** (see `xgboost_test/results/experiments.csv`).

---

## 1. Dataset

- `data/ibm_aml/SMALL/HI-Small_Trans.csv` — 5,078,345 transactions × 11 columns.
- `data/ibm_aml/SMALL/HI-Small_accounts.csv` — 518,581 accounts × 5 columns
  (**not used** by the current pipeline; see §7).
- Each transaction is an **edge** in a directed temporal graph:
  `(From Bank, Account) -> (To Bank, Account_duplicated_0)` at `Timestamp`,
  with `Amount Paid` as edge weight.
- Label: `Is Laundering` (1 = laundering). Only **0.10%** of rows are positive.

---

## 2. Pipeline overview

```
CSV (polars) → rename + datetime → node encoding (bank:account → int)
→ edge_list [edge_id, src, dst, ts, amount] → stable sort by ts
→ GFP streaming (batch=128, fit_transform first batch, transform rest)
→ raw feature set (polars) → X = hstack(raw, gfp[:, 4:]) → temporal 60/20/20 split
→ Optuna (maximize val F1) → best params → threshold on val → evaluate test once
```

End-to-end code: `xgboost_test/notebook_pipeline.py` (cell-per-stage, meant to be
pasted into a Jupyter notebook).

---

## 3. Stage-by-stage description

### 3.1 Load & preprocess (polars)

```python
hi_trans = pl.read_csv(DATA_DIR / "SMALL" / "HI-Small_Trans.csv")
hi_trans = hi_trans.rename({"Account": "src", "Account_duplicated_0": "dst"})
hi_trans = hi_trans.with_columns(
    pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M").alias("datetime")
)
```

- Parse timestamps with format `%Y/%m/%d %H:%M` (minutes granularity).
- Extract `hour`, `weekday`, `month` as raw time features.

### 3.2 Node encoding

Accounts are encoded with the composite key `(Bank ID, Account Number)`:

```python
src_key = hi_trans["From Bank"].cast(pl.Utf8) + ":" + hi_trans["src"]
dst_key = hi_trans["To Bank"].cast(pl.Utf8) + ":" + hi_trans["dst"]
accounts = pl.concat([src_key, dst_key]).unique()
node_id = np.arange(len(accounts))
```

- **Why `(bank, account)` and not bare `account`:** a handful of account numbers
  repeat across banks (4 in HI-Small); the composite key is unambiguous and
  matches the reference pipeline (515,088 nodes).
- Node IDs are dense integers `0..N-1`; both `src` and `dst` roles of the same
  account must map to the same integer (graph connectivity depends on it).

### 3.3 Edge list construction

SNAP ML GFP requires a 2-D **float64** array, one row per edge:

```
[Edge ID, Source Vertex ID, Target Vertex ID, Timestamp, <other raw features>]
```

```python
edge_list = np.column_stack([
    np.arange(len(hi_trans), dtype=np.float64),  # unique edge id
    src_id, dst_id,                               # integer node ids
    timestamp_epoch_seconds,                      # float64
    hi_trans["Amount Paid"].to_numpy(),           # extra raw feature (col 4)
]).astype(np.float64, copy=False)
```

- Timestamp unit is **epoch seconds** — every `_tw` window parameter below is in
  the same unit.
- `Amount Paid` is the only extra raw feature fed to GFP (`vertex_stats_cols=[3,4]`
  then computes statistics over columns 3=timestamp and 4=amount).

### 3.4 Temporal sort

```python
order = np.argsort(edge_list[:, 3], kind="stable")
edge_list = edge_list[order]
labels = hi_trans["Is Laundering"].to_numpy()[order].astype(np.int8)
```

- Stable sort by timestamp. Required so successive batches advance the GFP
  graph chronologically. This prevents cross-batch future leakage; the
  within-batch approximation is described in §4.2.1.
- `labels` and every raw feature array must be re-ordered with the same `order`.

### 3.5 GFP feature construction (streaming)

```python
gfp = GraphFeaturePreprocessor()
gfp.set_params({...})                      # see §5 for the full parameter reference
BATCH_SIZE = 128
first = gfp.fit_transform(edge_list[:128]) # build graph from first batch + transform it
out = np.empty((N, first.shape[1]), dtype=first.dtype)
out[:128] = first
for s in range(128, N, 128):
    out[s:s+128] = gfp.transform(edge_list[s:s+128])   # insert batch, then compute features
```

- `fit_transform` on the first batch returns the **output width** (124 columns for
  the reference config) → pre-allocate `out` once, avoid `np.vstack` copies.
- Every subsequent `transform` both **inserts** the batch into the in-memory graph
  and **computes features** on the updated graph (§4.2). Batching is a throughput
  choice, not a per-account or per-pair grouping: the same dynamic graph persists
  across all calls.

### 3.6 Raw feature selection (feature set C)

The raw (non-GFP) features kept for XGBoost, per transaction:

| Feature | Construction |
|---|---|
| `amount_received`, `amount_paid` | raw float columns |
| `rcv_cur`, `pay_cur` | label-encode `Receiving/Payment Currency` (15 categories) |
| `pay_fmt` | label-encode `Payment Format` (7 categories; ACH dominates laundering) |
| `hour`, `weekday`, `month` | extracted from parsed datetime |
| `log_amt_recv`, `log_amt_paid` | `log1p(...)` — amounts are heavy-tailed (max ≈ 1e12) |
| `same_currency` | `rcv_cur == pay_cur` (0/1) |
| `amount_gap` | `|amount_received - amount_paid|` (FX spread) |

All arrays are indexed by `order` so they align with `edge_list` / `labels`.

> Feature set C = raw (12) + GFP engineered (119). The GFP engineered block is
> obtained as `gfp_features[:, 4:]` (drop `edge_id, src, dst, timestamp`; keep
> `amount` + engineered). IDs are never given to XGBoost (per the paper).

### 3.7 Temporal split (60 / 20 / 20)

```python
n = len(y); train_end = int(n*0.60); val_end = int(n*0.80)
```

- **Never a random split.** Rows are already time-sorted, so the split is
  temporal: train = first 60% of time, val = next 20%, test = last 20%.
- The GFP graph is built once over the whole stream; because edges are fed in
  time order, an edge's features only reflect earlier edges — validation/test
  edges never leak into earlier features, and no future edge is visible.

### 3.8 Optuna hyper-parameter search

Objective: **maximize minority-class F1 on validation** (never accuracy).

```python
def objective(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 300, 1500),
        "max_depth":     trial.suggest_int("max_depth", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "subsample":     trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
        "gamma":         trial.suggest_float("gamma", 0, 3),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 30),
        "reg_lambda":    trial.suggest_float("reg_lambda", 0.5, 50, log=True),
        "tree_method": "hist", "n_jobs": -1, "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=60, verbose=False)
    p = model.predict_proba(X_val)[:, 1]
    # best threshold on validation:
    prec, rec, thrs = precision_recall_curve(y_val, p)
    f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    i = int(np.nanargmax(f1s))
    trial.set_user_attr("threshold", float(thrs[i]))
    return float(f1s[i])
```

- Early stopping on validation inside each trial (60 rounds patience).
- The best validation threshold found inside the trial is carried to evaluation.

### 3.9 Threshold optimization & final evaluation

- **Never** use the default 0.5 threshold — with 0.1% positives it misses almost
  everything.
- Select the threshold on **validation only**, freeze it, evaluate **test once**:

```python
test_pred = (test_prob >= best_threshold).astype(np.int8)
f1_score(y_test, test_pred); precision_score(...); recall_score(...)
average_precision_score(y_test, test_prob)   # threshold-free AUPRC
```

---

## 4. SNAP ML `GraphFeaturePreprocessor` — how it works

### 4.1 Input format (hard contract)

- 2-D `numpy.ndarray` of **float64** (the wrapper calls
  `np.ascontiguousarray(features_in, dtype=np.float64)` internally).
- Row layout, in this exact order:

```
[Edge ID | Source Vertex ID | Target Vertex ID | Timestamp | <extra raw features...>]
```

- `Edge ID` must be **unique** per edge (use a running index).
- Timestamps must share the same unit as every `*_tw` parameter.
- Sorting is *not* required, but pattern detection is most effective when edges
  are fed in increasing timestamp order (the pipeline sorts first).

### 4.2 Lifecycle: `fit` / `transform` / `fit_transform` / `partial_fit`

From the source docstring (`GraphFeaturePreprocessor.py`):

| Method | Behaviour |
|---|---|
| `fit(features)` | **Clear** the in-memory graph, then insert all edges of `features`. |
| `partial_fit(features)` | Insert edges of `features` into the existing graph (no clearing). |
| `transform(features)` | **Insert** `features` into the graph, then compute graph features on the **updated** graph; returns input + appended engineered columns. |
| `fit_transform(features)` | `fit(features)` followed by `transform(features)`. |

Key consequences:

1. **`transform` mutates the graph.** Calling it repeatedly accumulates edges.
   A later batch can use any matching source, destination, pair, or motif edge
   from an earlier batch, provided that edge is still inside `time_window`.
   Thus a pair's related transactions do **not** need to appear in the same batch.
2. Edges that already exist in the graph (same edge id) are ignored on re-insertion.
3. **Sliding window:** when an edge with timestamp `ts_now` is inserted, all edges
   with timestamp `≤ ts_now - time_window` are removed from the graph. So the
   in-memory graph only ever contains the most recent `time_window` of activity.

### 4.2.1 Batch semantics versus time windows

`batch=128` means “make one native GFP call for 128 chronologically ordered
edges”; it does **not** mean “only compare the 128 edges with each other.” The
dynamic graph survives from one call to the next. For example, with
`time_window=24h`, an `A → B` edge at 09:00 in batch 1 remains available to an
`A → B` edge at 14:00 in batch 37. Conversely, it is removed before a later edge
once it is more than 24 hours old.

There are two distinct window controls:

| Control | Purpose |
|---|---|
| `time_window` | retention window of the in-memory dynamic graph; old edges are evicted on insertion. |
| `vertex_stats_tw`, `scatter-gather_tw`, `temp-cycle_tw`, etc. | feature-specific lookback windows used when calculating that feature. |

Always set `time_window >= max(enabled *_tw)`. In the paper-style configuration,
the graph retains two days, while scatter-gather only looks back 12 hours and
the other enabled features look back two days.

The trade-off is temporal resolution. SNAP ML documents `transform(batch)` as
inserting the batch and then calculating features on the updated graph. Treat a
batch as an atomic micro-batch: an earlier edge can be influenced by later edges
in that same batch. Sorting prevents leakage **between** batches. `batch=1`
removes look-ahead to later events, but `transform` still inserts the current
edge before computing its features; strict pre-event `t_i^-` semantics need an
explicit score-then-commit implementation. `batch=128` is the paper's practical
throughput/temporal-resolution compromise; keep each batch's timestamp span
small relative to the smallest enabled `*_tw`.

### 4.3 Graph patterns & vertex statistics — two families of engineered features

Output layout per row (from `transform` docstring):

```
<raw features> <graph-pattern features> <vertex statistics features>
```

**Family 1 — graph patterns (histograms).** For the current edge, GFP counts how
many patterns of each *size* the edge participates in, then buckets the counts:

| Pattern | Meaning | Directional? |
|---|---|---|
| `fan` | number of *distinct neighbours* of an endpoint | yes: fan-in / fan-out |
| `degree` | number of *incident edges* of an endpoint (repeats counted) | yes: in / out |
| `scatter-gather` | 2-hop structure `A → {intermediaries} → C` (money fan-out then fan-in) | no |
| `temp-cycle` | directed cycle `A→B→C→A` with **strictly increasing** timestamps | no |
| `lc-cycle` | simple cycle (no repeated vertices) of length ≤ `lc-cycle_len` | no |

Each enabled pattern emits `len(bins)` histogram columns; bucket *i* holds the
count of patterns of size `S` with `bins[i] <= S < bins[i+1]`, the last bucket
holds `S >= bins[-1]`.

**Family 2 — vertex statistics (scalars).** For the edge's `src` and `dst`
endpoints, for *outgoing* and *incoming* edges separately (4 groups), compute:

| feat id | name | definition |
|---|---|---|
| 0 | `fan` | distinct neighbours |
| 1 | `degree` | number of incident edges |
| 2 | `ratio` | fan / degree |
| 3 | `avg` | mean of the selected raw columns |
| 4 | `sum` | sum of the selected raw columns |
| 5 | `min` | min of the selected raw columns |
| 6 | `max` | max of the selected raw columns |
| 7 | `median` | median of the selected raw columns |
| 8 | `var` | variance of the selected raw columns |
| 9 | `skew` | skewness of the selected raw columns |
| 10 | `kurtosis` | kurtosis of the selected raw columns |

`avg..kurtosis` are computed per raw column listed in `vertex_stats_cols`, and
only for ids present in `vertex_stats_feats`.

---

## 5. GFP parameters — complete reference

Listed via `gfp.get_params()`; set via `gfp.set_params({...})`. `set_params`
clears the in-memory graph, so configure **before** feeding any edges.
Unknown keys raise `KeyError` — parameter names are exact, including the hyphens
in `scatter-gather`, `temp-cycle`, `lc-cycle`.

### 5.1 Global controls

| Parameter | Default | Meaning |
|---|---|---|
| `num_threads` | 12 | software threads for feature computation |
| `time_window` | -1 | dynamic-graph window. If negative, overwritten with the **largest** `*_tw` among enabled features. Unit = timestamp unit. |
| `max_no_edges` | -1 | if positive, keep only the `max_no_edges` most recently added edges (in addition to the time window). |

### 5.2 Vertex statistics

| Parameter | Default | Meaning |
|---|---|---|
| `vertex_stats` | True | enable vertex-statistics features |
| `vertex_stats_tw` | 480*3600 = 20 days | window for vertex statistics |
| `vertex_stats_cols` | [3] | input columns used for `avg..kurtosis` (your config: [3,4] → timestamp & amount) |
| `vertex_stats_feats` | [0,1,2,3,4,8,9,10] | which statistics to emit (fan, deg, ratio, avg, sum, var, skew, kurtosis; min/max/median excluded by default) |

### 5.3 Graph patterns (`<pattern>` ∈ {`fan`, `degree`, `scatter-gather`, `temp-cycle`, `lc-cycle`})

| Parameter | Default | Meaning |
|---|---|---|
| `<pattern>` | fan=True, others False | enable/disable the pattern |
| `<pattern>_tw` | 12h (fan/degree), 5d (scatter-gather), 20d (temp-cycle), 10d (lc-cycle) | time window for this pattern; larger → more patterns found but slower |
| `<pattern>_bins` | [2..30] for fan/degree/scatter-gather/temp-cycle; [2..10] for lc-cycle | histogram bucket boundaries; `bins[i] <= S < bins[i+1]`, last bucket `>= bins[-1]` |
| `lc-cycle_len` | 10 | max simple-cycle length to search |

### 5.4 Reference configuration (paper-aligned, best-known)

```python
gfp.set_params({
    "num_threads": 16,
    "vertex_stats": True,
    "vertex_stats_cols": [3, 4],
    "vertex_stats_tw": 172800,          # 2 days (epoch seconds)
    "scatter-gather": True,
    "scatter-gather_tw": 43200,         # 12 hours
    "temp-cycle": True,
    "temp-cycle_tw": 172800,            # 2 days
    "lc-cycle": True,
    "lc-cycle_tw": 172800,              # 2 days
    "lc-cycle_len": 10,
    "fan": False,                       # <-- intentional
    "degree": False,                    # <-- intentional
    "time_window": 172800,              # global window = 2 days
})
```

- **Why `fan=False, degree=False`:** the vertex-statistics block already emits
  per-endpoint fan/degree scalars; enabling the fan/degree *histograms* adds 116
  columns (124 → 240 total) that did not improve the result. The best
  experiments in `results/experiments.csv` all use `fan=False, degree=False`.
- Window sizes: the paper's Appendix D uses 1 day / 6 hours (86400 / 21600);
  the local best result used **2 days / 12 hours** (172800 / 43200). Both are
  valid; the 2-day windows won empirically.
- Expected output width for this config: **124 columns** (5 raw + 119 engineered),
  i.e. `gfp_features.shape == (n_edges, 124)`.

### 5.5 Column naming of the output (for debugging / interpretation)

From the official example notebook (`print_enriched_transaction`):

- Raw columns keep their position: `edge_id, src, dst, timestamp, amount, ...`.
- Pattern histograms: `fan_in_bins_2-3 ... fan_in_bins_30-inf`, `fan_out_...`,
  `degree_in/out_...`, `scatter-gather_bins_...`, `temp-cycle_bins_...`,
  `lc-cycle_bins_...`.
- Vertex statistics: `<orig>_<stat>_<direction>` for fan/deg/ratio, and
  `<orig>_<stat>_col<k>_<direction>` for the column-wise statistics, where
  `orig ∈ {source, dest}`, `direction ∈ {out, in}`.
  Examples: `source_fan_out`, `dest_sum_col4_in`, `source_avg_col3_out`.

---

## 6. Known pitfalls

1. **Forgetting the temporal sort** → future edges leak into earlier features
   (the graph is only correct when edges arrive in time order).
2. **Setting `time_window` smaller than a `*_tw`** → that pattern silently has
   nothing to search (its window exceeds what the graph retains). Keep
   `time_window >= max(all enabled *_tw)`.
3. **Passing int arrays** → GFP silently coerces to float64, but `edge_id`/
   node ids are then float; that is fine, but build the array as float64 up
   front to avoid copies.
4. **Reusing one `GraphFeaturePreprocessor` across experiments** → `set_params`
   clears the graph, but `fit_transform`/`transform` accumulate edges; create a
   fresh instance (or call `fit`) for each experiment.
5. **0.5 threshold** → always select the threshold on validation via
   `precision_recall_curve` (see §3.8).
6. **`scale_pos_weight=5` is too weak** for 0.1% positives — the search range
   10–30 with best ≈ 20 consistently outperformed smaller values.

---

## 7. The accounts table (currently unused)

`HI-Small_accounts.csv` adds `Entity ID` / `Entity Name` (the "owner" of one or
more accounts; mean 3.1 accounts/entity, max ~7800) and `Bank Name`. The current
pipeline does **not** need it — node keys `(bank, account)` already exist in the
transactions table. Possible (optional) enhancement: per-entity account counts
and a `same_entity` flag joined via `(Bank ID, Account Number)`. Not part of the
baseline; treat as an ablation experiment.

---

## 8. File map

| File | Purpose |
|---|---|
| `xgboost_test/notebook_pipeline.py` | reference pipeline, cell-per-stage, notebook-ready |
| `xgboost_test/scripts/prepare_data.py` | batch preparation (edge_list/labels/raw.parquet) |
| `xgboost_test/scripts/build_gfp_features.py` | batch GFP feature generation + configs |
| `xgboost_test/scripts/optuna_search.py` | batch Optuna search |
| `xgboost_test/scripts/train_xgb.py` | batch train + evaluate + log to experiments.csv |
| `xgboost_test/results/experiments.csv` | every recorded experiment (best: test F1 0.6105) |
| `xgboost_test/results/final_report.md` | reproduction summary (F1 0.6059 vs paper 63.23) |
| `xgboost_test/GFP_XGBOOST_REPRODUCTION_LOOP.md` | task spec this pipeline implements |
