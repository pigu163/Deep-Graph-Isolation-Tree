# SLADE — Self-supervised Learning for Anomaly Detection in Edge Streams

- Paper: JooHyong et al., "SLADE: Self-supervised Learning for Anomaly
  Detection in Edge Streams", KDD 2024 (arXiv:2402.11933).
- Source: official implementation imported as-is
  (https://github.com/jhsk777/SLADE); no modifications to the model code.
  (No LICENSE file in the upstream repository.)
- Category: edge-stream, self-supervised (labels used for evaluation only).
- Score: `(2 - drift_sim - recovery_sim) / 4` following the official
  evaluation (larger = more anomalous).

## What is adapted (data plumbing only)

- `events.parquet` -> SLADE `Data(sources, destinations, timestamps,
  edge_idxs, labels)`; node ids re-mapped to a contiguous 0..N-1 range
  (the official Memory arrays index nodes directly).
- Chronological splits come from `data/processed/*/splits.json` (60/20/20)
  instead of the official single training-ratio quantile cut.
- The train split (60%) is used for self-supervised training (drift +
  recovery contrastive losses, official defaults); train/val/test are all
  scored with a full past-only neighbor finder.
- Evaluation uses the shared metric code (unknown labels excluded).

## Leakage

`NeighborFinder.find_before` uses `np.searchsorted(timestamps, cut_time,
'left')`, i.e. only events with timestamp < query timestamp are returned;
current and future events (including same-timestamp later events) never enter
the history.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/SLADE/runner.py \
  --dataset paysim --protocol chronological --seed 42 --n-epochs 10
```
