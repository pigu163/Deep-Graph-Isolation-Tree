# F-FADE — Frequency Factorization for Anomaly Detection in Edge Streams

- Paper: Zhou et al., "F-FADE: Frequency Factorization for Anomaly Detection
  in Edge Streams", WSDM 2021 (arXiv:2011.04723).
- Source: https://github.com/snap-stanford/F-FADE — faithful translation of
  `model.py` (UNION / DETECT / F_FAC). No LICENSE file in the repository;
  code is ported with attribution.
- Category: edge-stream, unsupervised (labels used for evaluation only).
- Score: ratio of observed frequency to predicted Poisson rate
  (larger = more anomalous).

## Algorithm (as implemented)

1. Events are grouped into discrete time slices. `delta_F` aggregates
   per-(src,dst) frequency within a slice.
2. After a setup phase, every edge is scored by `DETECT`: it combines the
   edge frequency (with exponential-decay history `F`), the source and
   destination neighborhood frequencies, and the Poisson-rate embedding
   `lambda_sd` from the frequency-factorization model.
3. `UNION` updates the decaying frequency table (alpha) and prunes to the
   top-M table entries; `F_FAC` trains the embedding with the Poisson
   negative log-likelihood (global at setup, local online afterwards).

## Task adaptation (recorded in config.json)

- The repository assumes integer time ticks; our events use real timestamps.
  Slices follow the MAD-baseline policy: paysim / elliptic_pp / midas_darpa
  use native steps (1..743 / 1..49 / 1..46572), IBM AML uses 1-hour bins
  (3600 s). This is a time-scale choice, not label-tuned.
- The original `global` F_FAC enumerates all V x V candidate pairs, which is
  intractable on 500k-node transaction graphs; it runs only when |V|^2 <=
  4,000,000, otherwise the `local` (sampled) F_FAC is used for every update.
- Default hyper-parameters follow the repository (embedding_size 200,
  alpha 0.999, M 100, T_th 120, epochs 5, online_train_steps 10); t_setup and
  w_upd are scaled to the slice count when the defaults exceed it.

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/FFADE/runner.py \
  --dataset paysim --protocol chronological --seed 42
```
