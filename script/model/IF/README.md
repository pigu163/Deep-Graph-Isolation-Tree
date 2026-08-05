# IF — Isolation Forest baseline

Paper-fair baseline for the DIF reproduction (task doc Sec. 8.2).

## Implementation

```python
from sklearn.ensemble import IsolationForest

IsolationForest(
    n_estimators=300,
    max_samples=256,
    contamination="auto",
    random_state=seed,
    n_jobs=-1,
)
```

* **Score direction**: `anomaly_score = -model.score_samples(X)` → larger =
  more anomalous (verified by test `test_score_direction`).
* **Train-only fit**: fitted exclusively on the chronological train split;
  val/test are scored with the fitted model.
* **No label tuning**: `contamination="auto"`; labels never set the
  contamination or any hyper-parameter.
* **Batched scoring** for memory-heavy datasets (PaySim, IBM).

## Input

Uses the shared tabular pipeline (`script/tabular_pipeline/`):

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/IF/runner.py \
  --dataset paysim --protocol chronological --seed 42 [--overwrite] \
  [--max-samples N] [--quick] [--device cpu] [--batch-size N]
```

Outputs to `result/IF/<DATASET>/<PROTOCOL>/seed_<SEED>/`:
`config.json`, `metrics.json`, `scores.parquet`, `timing.json`,
`warnings.json`, `run.log`.

`paper_fidelity = "baseline"`.

## Faithfulness notes

IF is a baseline, not a paper reproduction. The tabular inputs, splits and
score direction are exactly the same as DIF so that comparisons are fair.
