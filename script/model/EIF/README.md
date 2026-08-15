# EIF — Extended Isolation Forest

Faithful wrapper around the local `source code/eif` Cython/C++ implementation of Hariri et al. (TKDE 2020).

- Category: tabular, unsupervised; labels are used only for evaluation.
- Input: the shared transaction tabular pipeline used by IF and DIF.
- Fit: chronological train split only; score direction is larger = more anomalous.
- Default: 300 trees, 256-row subsamples, fully extended random hyperplanes (`extension_level = n_features - 1`).

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/EIF/runner.py \
  --dataset paysim --protocol chronological --seed 42
```
