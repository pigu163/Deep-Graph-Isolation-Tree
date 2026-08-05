# ECOD — Empirical-CDF-based Outlier Detection

- Paper: Li, Zheng & Zhao, Yue et al., "ECOD: Unsupervised Outlier Detection
  Using Empirical Cumulative Distribution Functions", IEEE TKDE 2022
  (arXiv:2201.00382).
- Source: PyOD (https://github.com/yzhao062/pyod, BSD-2-Clause),
  `pyod/models/ecod.py`. Installed via `pip install pyod` (3.6.4);
  no fork of the algorithm is used.
- Category: tabular, unsupervised (no labels used for fitting).
- Score: `ECOD.decision_function`, larger = more anomalous.

## Adaptation notes (task-specific)

- Input: the shared tabular view (`script/tabular_pipeline`), identical to
  IF/DIF — each transaction is one row; integer sender/receiver IDs are
  excluded from the feature matrix (frequency-encoded or dropped there), so
  ECOD never sees raw node IDs as continuous values.
- Chronological 60/20/20 split from `data/processed/*/splits.json`; CDFs are
  fitted on the train rows only.
- ECOD is parameter-free; `contamination` only affects thresholding, not the
  score ordering, and is recorded in `config.json`.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/ECOD/runner.py \
  --dataset ibm_hi_small --protocol chronological --seed 42
```
