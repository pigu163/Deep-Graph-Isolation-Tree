# Tabular pipeline（IF 与 DIF 共用）

共享的 tabular 数据接口：IF 与 DIF 使用**完全相同**的样本、特征与 split
（任务文档 Sec. 5.1 / 7 / 8）。

## 数据源与样本定义

| dataset | 样本 | 标签 | 特征 |
|---|---|---|---|
| `paysim` | 一笔交易（event） | `isFraud` | `timestamp, f_amount, f_oldbalanceOrg, f_newbalanceOrig, f_oldbalanceDest, f_newbalanceDest` + `edge_type` one-hot + `src/dst` freq + strict past-only account/pair temporal features；`is_flagged_fraud` 被删除 |
| `ibm_hi_small` / `ibm_li_small` | 一笔交易（event） | `Is Laundering` | `timestamp, f_amount_paid, f_amount_received, f_same_bank, f_same_currency` + `edge_type/payment_currency/receiving_currency` one-hot + `src/dst` freq + strict past-only account/pair temporal features |
| `midas_darpa` | 一条通信事件 | ground-truth（按行对齐） | `timestamp` + `src/dst` freq + strict past-only account/pair temporal features |
| `elliptic_pp` | **transaction node**（非 money-flow edge） | `illicit=1 / licit=0 / unknown=-1` | `time_step` + 93 个 local 特征（`txs_features.csv`，`--features local` 默认；`all` = 165） |

## 编码规则（train-only fit）

- 数值列 → float32，train 均值 impute + train z-score；
- 高基数 ID（账户 / `src` / `dst`）→ train-set **frequency encoding**（绝不 one-hot），unseen → 0；
- 类别列 → train 确定性 one-hot（排序类别），unknown → 全零行；
- scaler / imputer / encoder **只在 train split 拟合**，val/test 严格复用；
- 标签从不用于拟合预处理或模型超参数。

## 输出

`data/processed_tabular/<dataset>/`：

```
features.parquet  labels.npy  label_available.npy  sample_ids.npy
timestamps.npy    splits.json  feature_schema.json  preprocessing.json  metadata.json
```

## API

```python
from script.tabular_pipeline.datasets import load_tabular_dataset, TabularData
data: TabularData = load_tabular_dataset("paysim", split="all")
# data.X (float32), data.y, data.label_available, data.sample_ids,
# data.feature_names, data.timestamps
```

split 为行序 60/20/20（chronological；事件数据集行序 = event_id 序，
elliptic 按 `(time_step, txId)` 排序）。

## Temporal features

For IBM, PaySim and MIDAS, the shared table also includes past-only sender, receiver and directed-pair counts, unique-counterparty counts and last-seen gaps. Each timestamp group is scored before its events update state, so same-time events cannot leak into one another.
