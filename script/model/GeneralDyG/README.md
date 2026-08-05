# GeneralDyG — A Generalizable Anomaly Detection Method in Dynamic Graphs

- Paper: Ye et al., "GeneralDyG: A Generalizable Anomaly Detection Method in
  Dynamic Graphs", AAAI 2025 (arXiv:2412.16447).
- Source: official implementation imported as-is
  (https://github.com/YXNTU/GeneralDyG); two small CPU-compat copies
  (`CensNetCPU` and `CombinedModel`) only remove hard-coded `.to('cuda')`
  device strings that prevent the upstream code from running on CPU.
- Category: dynamic-graph anomaly detection, supervised.
- Score: `P(fraud | ego-graph)` from the CensNet + Transformer classifier
  (larger = more anomalous).

## Adaptation notes (task-specific)

- One sample per transaction edge, matching the paper's per-edge ego-graph
  construction (k=1 hop, `max_mask_len=24`): source and destination
  neighborhoods are extracted and re-ordered by edge-id weights exactly as
  in the official `BatchGraphSample` / `replace_subgraph`.
- **Leakage fix**: the paper extracts ego-graphs from the *full* graph; for
  our chronological task the ego-graph is built from strictly past events
  (event_id < current) plus the current edge itself. This is a deliberate
  deviation recorded in `config.json`.
- Labels: transaction labels; unknown (-1) events are excluded from training
  but still scored. Chronological 60/20/20 splits from `splits.json`.
- Official training loop preserved (BCE loss, Adam, same model config
  defaults: input_dim 128, hidden 258, 4 heads, 6 layers).

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/GeneralDyG/runner.py \
  --dataset paysim --protocol chronological --seed 42 --n-epochs 20
```
