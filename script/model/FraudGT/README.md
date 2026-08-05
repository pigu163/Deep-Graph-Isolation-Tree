# FraudGT — A Simple, Effective, and Efficient Graph Transformer for
# Financial Fraud Detection (ACM ICAIF 2024)

- Paper: Shun et al., "FraudGT: A Simple, Effective, and Efficient Graph
  Transformer for Financial Fraud Detection", ICAIF 2024.
- Source: official implementation imported as-is
  (https://github.com/junhongmit/FraudGT); the graphgym pipeline (cfg,
  create_dataset / create_loader / create_model, HomoGNNEdgeModel,
  weighted cross-entropy) is used unmodified.
- Category: transaction graph, supervised (labels used in train split only).
- Score: `P(fraud | edge)` (larger = more anomalous).

## What is adapted (data plumbing only)

- `events.parquet` -> the official AML processed format `data.pt`
  (train/val/test `HeteroData` with `node` / `to` / `rev_to` edge types and
  `split_mask`), written under `tmp/fraudgt_data/` so the official
  `AMLDataset` skips its CSV processing.
- Edge features follow the official AML choice: Timestamp, Amount Received,
  Received Currency, Payment Format (categoricals ordinal-encoded), z-normed
  like the official `process()`; node features are the all-ones placeholder.
- cfg is loaded from the official `AML-Small-HI-GINE+ports.yaml`, with
  unknown keys pruned (the yaml is out of sync with this repo's cfg schema:
  `wandb`, `add_ports` etc.), `add_ports=False`, `reverse_mp=False`,
  wandb disabled, CPU.
- Chronological 60/20/20 splits from `splits.json` (the official process
  uses a day-balanced 60/20/20 split; boundaries may differ).
- unknown (-1) labels are excluded from the loss but all edges are scored.

## Compatibility fixes (recorded in config.json)

- `torch.load` patched to `weights_only=False` (torch >= 2.6 default breaks
  the upstream loader for PyG `HeteroData`).
- Official yaml keys not present in the repo's cfg schema are pruned.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/FraudGT/runner.py \
  --dataset ibm_hi_small --protocol chronological --seed 42
```
