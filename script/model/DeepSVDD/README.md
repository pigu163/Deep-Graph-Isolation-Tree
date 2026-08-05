# Deep SVDD (MLP) — Deep Support Vector Data Description

- Paper: Ruff et al., "Deep One-Class Classification", ICML 2018
  (arXiv:1802.06360).
- Source: https://github.com/lukasruff/Deep-SVDD-PyTorch (MIT License).
- Category: tabular, one-class unsupervised (no labels used for fitting).
- Score: `||phi(x) - c||^2` distance to the hypersphere center
  (larger = more anomalous).

## Adaptation notes (task-specific)

- The official repository is image-oriented (LeNet encoders, MNIST/CIFAR
  datasets). The Deep-SVDD objective is kept unchanged; the encoder is an MLP
  (`--hidden`, `--rep-dim`) suitable for transaction feature rows.
- Center initialization follows the official scheme: `c` = mean of the
  encoder output over the train rows after a forward pass, kept fixed during
  training.
- The official two-stage pipeline pre-trains an auto-encoder first; that
  stage is available via `--pretrain-ae` but is off by default
  (recorded in `config.json`).
- Input is the shared tabular view (same as IF/DIF/ECOD); raw sender/receiver
  IDs are excluded there (frequency-encoded), so the network never sees
  integer IDs as continuous features.
- Chronological 60/20/20 split; training uses train rows only.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/DeepSVDD/runner.py \
  --dataset paysim --protocol chronological --seed 42 --epochs 50
```
