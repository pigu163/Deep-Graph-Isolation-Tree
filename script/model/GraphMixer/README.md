# GraphMixer — dynamic-graph link predictor (ICLR 2023)

- Paper: Cong et al., "GraphMixer: A Thoroughly Empirical Study of Dynamic
  Graph Prediction via MLPs", ICLR 2023 (arXiv:2302.11636).
- Source: official DyGLib implementation imported as-is
  (https://github.com/yule-buaa/dyglib, MIT License); model files
  `models/GraphMixer.py` are not modified.
- Category: dynamic graph, self-supervised link prediction (labels unused in
  training).
- Score: `1 - P(link)` from the link predictor (larger = more anomalous).

## Adaptation notes

- `events.parquet` -> DyGLib `Data`; node ids re-mapped to a contiguous
  range; random node features (DyGLib's convention for non-attributed
  graphs); edge features from the numeric `f_*` columns when present (NaN ->
  0), otherwise a dummy feature; both padded to `--feat-dim`.
- Self-supervised link-prediction training on the train split (BCE with
  negative edges, official protocol); chronological splits from
  `splits.json` (60/20/20).
- `NeighborSampler.find_neighbors_before` uses `np.searchsorted(..., 'left')`
  so only interactions strictly before the query time are sampled
  (past-only, no leakage).

## Run

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/GraphMixer/runner.py \
  --dataset paysim --protocol chronological --seed 42 --num-epochs 5
```
