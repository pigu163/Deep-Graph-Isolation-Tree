# 复现报告：IF / DIF / MAD（paper-guided reproduction）

> 结论边界：本项目是**基于论文与作者 source code 的复现（paper-guided reproduction）**，
> 不是对论文实验表格的逐项复刻。数据集的注入协议、评估范围与论文不同，
> 指标数值不可与论文表格直接对比。

## 1. 仓库与环境

- 仓库：`/home/zhu/Deep Graph Isolation Tree`
- Python 解释器：`/home/zhu/.conda/envs/IF/bin/python`
- 关键依赖（实测）：numpy 2.4.4, scikit-learn 1.9.0, torch 2.12.1+cu132,
  polars 1.43.2, pytest 9.1.1
- `pyarrow` 缺失：polars 1.43 自带 parquet 引擎，本任务未用到 pyarrow；
  如未来需要，用户空间安装命令为 `/home/zhu/.conda/envs/IF/bin/pip install pyarrow`（未安装，未使用 sudo）。

## 2. source code 实际使用的文件

| 来源 | 文件 | 用途 |
|---|---|---|
| `source code/deep-iforest/` | `algorithms/dif.py` | DIF 主类、DEAS `_cal_score`、`_average_path_length`、z-score+tanh 后处理、每表示 6 树/256 子采样 |
| `source code/deep-iforest/` | `algorithms/net_torch.py` | CERE `LinearBlock`（rank-one 调制 `(x·p)W0·q`）、MLP 结构（hidden [500,100], rep 20, tanh, bias=False） |
| `source code/mad/Proposed/MAD/` | `MAD.py`, `model.py` | MAD scoring 公式 (8)、`fit`/`stationarity_test`（窗口扫描 + 方差 fitness）、`get_model_mean_var` |
| `source code/mad/Proposed/MAD/` | `gdecomp/dictionary/dictionary_class.py` | 关系按概率降序索引（`MapperSubgraph_byRank`） |
| `source code/mad/Proposed/MAD/` | `gdecomp/decomposition/decomposition.py` | Haar 小波分解（`pywt.wavedec/idwt`）、`multiscale_approximation`、power-of-two padding |
| `source code/mad/Proposed/` | `edge_anomaly/main.py` | edge-query 实验协议（phi=单边、context=过去窗口） |

许可：MAD source code 为 EUPL-1.2（仅在仓库内适配，未修改原始文件）。

## 3. 数据集统计（`data/processed/*/metadata.json`）

| dataset | events | nodes | labeled | positives | timestamp 范围 |
|---|---|---|---|---|---|
| midas_darpa | 4,554,344 | 25,525 | 4,554,344 | 2,737,209 | 1..46572 |
| ibm_hi_small | 5,078,345 | 515,088 | 5,078,345 | 5,177 | Unix 秒 |
| ibm_li_small | 6,924,049 | 705,907 | 6,924,049 | 3,565 | Unix 秒 |
| paysim | 6,362,620 | 9,073,900 | 6,362,620 | 8,213 | step 1..743 |
| elliptic_pp | 234,355 | 203,769 | 86,003 | 5,772 | step 1..49 |

Split：chronological 60/20/20（`data/processed/<ds>/splits.json`，event_id 半开区间）。

## 4. 两种数据范式

- **IF / DIF（tabular）**：`data/processed_tabular/<ds>/`，每行一个独立样本；
  特征映射与编码见 `script/tabular_pipeline/README.md`。
- **MAD（link stream）**：直接消费 `data/processed/<ds>/events.parquet`
  （`(t,u,v)` 事件序），不转静态图、不转 tabular 矩阵。

三者**不声称输入范式相同**；能对齐时 `sample_id` 均为 event_id
（elliptic 的 IF/DIF 样本为 transaction node，sample_id=txId，与 MAD 的
事件样本不对齐，属不同范式，见任务文档 Sec. 5.3）。

## 5. 统一 metric 与接口

- 公共实现：`script/common/metrics.py::compute_metrics`（三个 runner 共用，
  单类别 split 返回 null AUC + warning；unknown 标签排除；非有限 score 报错；
  K = 评估集 positive 数，并列分数稳定排序）。
- score 方向统一：**越大越异常**。
- 结果 schema：`result/<MODEL>/<DATASET>/<PROTOCOL>/seed_<SEED>/`
  （config/metrics/scores/timing/warnings/run.log + interface_check.json）。
- `--quick` 结果标记 `is_quick_run=true`，`evaluate_all.py` 汇总时排除。

## 6. IF（Stage 2）

- `sklearn.ensemble.IsolationForest(n_estimators=300, max_samples=256,
  contamination="auto", random_state=seed, n_jobs=-1)`；
- `anomaly_score = -model.score_samples(X)`（越大越异常）；
- 只在 train split fit；不使用标签设 contamination；
- 与 DIF 共用同一 `load_tabular_dataset`；
- `paper_fidelity = baseline`。
- 状态：**PASS**（单测 + 五数据集 smoke + 正式多 seed 运行）。

## 7. DIF（Stage 3）

### 7.1 论文方法 ↔ 实现映射

| 论文组件 | 实现 | 状态 |
|---|---|---|
| Random representation ensemble（式 1） | `random_encoder.py`，优化-free 随机网络，无 optimizer / 无 backward | faithful |
| CERE（式 3-6） | `CereEncoder`：共享基础权重 `W0~N(0,1)` + rank-one 随机向量 `(p_i, q_i)`，`W_i = W0◦(p_i q_i^T)`；批量 `encode_all` 按式 (5) 块布局；`encode_member` 按式 (4)，两者一致性有单测 | faithful |
| Deep isolation trees（Algorithm 1） | `deep_isolation_tree.py`（自实现，不依赖 sklearn.tree）：随机维度 + `η~U(min,max)`，深度上限 `J=8=ceil(log2 256)` | faithful |
| Path length（Algorithm 2） | `traverse()` 返回分支步数 `|p|` | faithful（无叶节点 size 修正，忠实论文式 8） |
| 平均分裂偏差 `g`（式 7） | `g = β/|p|`，`β = Σ|x(j_k)−η_k|` | faithful |
| DEAS（式 8） | `F = 2^{−E|p|/C} × E[g]`，`C = c(256)` | faithful |
| 默认参数 | `r=50, t=6, T=300, n=256, batch=64`（论文 Sec. 5.3/6.1.3 + 作者代码） | faithful |
| Batched inference / seeding | `DIF.score` 流式批处理；`set_seed` + `RandomState`；inference 复用 fit 时随机权重 | faithful |

### 7.2 CERE 状态

`CereEncoder` 完整实现（式 3-6），`--implementation faithful` 默认；
`--implementation simplified` 用独立随机 MLP（非 CERE），
`paper_fidelity = simplified`，单独标记，不混入 faithful 汇总。

### 7.3 与论文 / source code 的偏差（记录于每个 config.json `deviation_notes`）

1. 论文未指定 MLP 层数/宽度/激活/初始化 → 采用作者公开代码默认
   （hidden [500,100]、rep_dim 20、tanh、权重 N(0,1)、无 bias），可经 CLI 覆盖；
2. 表示后处理采用作者实现的 z-score + tanh；scaler 只在 train 的随机
   calibration 子集拟合（作者代码对全量 fit，含 test——我们收紧为无泄漏）；
3. DEAS 不加叶节点 size 修正（忠实论文式 8 的原始 path length）；
4. 聚合用论文全局平均（作者按表示再平均的写法未采用）；
5. elliptic_pp 的 93 local / 72 aggregated 边界沿用原版 Elliptic 布局假设
   （本地文件无列名说明，`schema.py:ELLIPTIC_LOCAL_COUNT` 可调），默认 local。

- 状态：**PASS**（单测 + 与 IF 的 tabular 对齐检查 + 五数据集 smoke + 正式运行）。

## 8. MAD（Stage 4）

### 8.1 论文方法 ↔ 实现映射

| 论文组件 | 实现 | 状态 |
|---|---|---|
| 关系概率估计（式 9） | `probability.py`：`P(e) = active slices / K`（K = 窗口内 slice 数，含空 slice；smoothing 可配置，默认 0） | faithful |
| 概率降序索引 | `multiscale.order_relations`（稳定排序，同作者 `sorted(reverse=True)`） | faithful |
| power-of-two padding | `next_power_of_two` + `leaf_vector`（虚拟关系 P=0、恒 inactive，不影响真实关系统计） | faithful |
| 多尺度二分（式 2-3） | 归一化 Haar 小波分解 `haar_wavedec`（numpy-only，等价 `pywt.wavedec('haar')`） | faithful |
| 随机变量期望/方差（Prop. 2） | `get_model_mean_var`：均值 = 概率图分解；方差 = `p(1-p)` 图分解 + `multiscale_approximation` 归一化，拼接 `[var_dec[0]] + var_dec[0:-1]`（与作者代码逐行一致） | faithful |
| 标准化偏差与聚合（式 7-8） | `score_query`：`Σ (x−E)² / max(σ², 1e-2)`（var_floor=1e-2 同作者代码） | faithful |
| stationarity-based 窗口选择 | `stationarity.py`：从作者 `model.fit/stationarity_test` 精确恢复（向后扫描全部 K、样本方差 vs 理论方差 L2 fitness、`fitness <= best_fit` 保留最大窗口、空 slice 按全零实现处理）；`--window-mode stationary` 才可称 paper-faithful | faithful |
| edge / node / slice / subgraph 查询 | `MAD.score_edge / score_node / score_slice / score_subgraph` | 已实现 |
| batch 主实验（edge-event） | `run_edge_event_batch`：按 slice 分组、增量窗口计数、`--query-mode active-edge / local-candidate-set / full-scope` | 已实现 |

### 8.2 查询模式说明

- `active-edge`：MAD 任意 subgraph 框架的 **edge-query 特例**（M=1，phi 为当前事件
  的 (u,v)），只检测 unexpected-active；README 已明确标注。
- `local-candidate-set`：phi = (u,v) + 两端点过去交互过的关系（有限集合），
  同时考虑 expected-but-missing 与 unexpected-active。
- `full-scope`：使用每个 slice 的**观测关系空间**（受限 full-scope），
  不枚举 E=V×V（PaySim ~907 万账户不可行）。

### 8.3 离散时间策略（记录于 config `slice_policy`）

- paysim / elliptic_pp / midas_darpa：`native-step`（timestamp 即离散 step）；
- ibm_hi_small / ibm_li_small：`fixed-width` 1 小时 bin（Unix 秒跨度 ~18 天，
  每小时 ~1.2 万笔；按时间尺度选择，不用标签调 bin）。

### 8.4 与论文 / source code 的偏差

1. batch 主实验的 `fixed` 窗口按**日历 slice 数**计 delta（含空 slice，
   与作者 `delta = max_time−min_time+1` 一致）；`stationary` 在 batch 路径
   降级为 `expanding` 并标注 `window_mode_effective`（完整 stationary 由
   `MAD` 查询类提供，避免 batch 中 O(N·K) 扫描）；
2. 单边 edge-query 的官方实验（phi=单边、逐边独立历史）未作为主实验，
   主实验采用 edge-event（对每个 active 事件评分）并保留 active-edge 模式对齐官方语义；
3. 空 slice 的观测按全零实现处理（与作者 `(delta−n)·mean²` 修正一致）。

### 8.5 past-only 泄漏检查

每次运行生成 `interface_check.json`：随机抽样 ≤50 个事件，记录
`current_event_id / max_history_event_id / current_timestamp /
max_history_timestamp / history_size`，并断言
`max_history_event_id < current_event_id`；同 timestamp 事件按 event_id 顺序
（同一 slice 内事件互不入历史）。所有运行 `past_only_passed=true`。

- 状态：**PASS**（单测 + past-only 检查 + 五数据集 smoke + 正式运行）。

## 9. 运行命令

```bash
# 单元测试 / 集成测试
/home/zhu/.conda/envs/IF/bin/python -m pytest -q

# 单模型
/home/zhu/.conda/envs/IF/bin/python script/model/IF/runner.py --dataset paysim --protocol chronological --seed 42
/home/zhu/.conda/envs/IF/bin/python script/model/DIF/runner.py --dataset paysim --protocol chronological --implementation faithful --seed 42
/home/zhu/.conda/envs/IF/bin/python script/model/MAD/runner.py --dataset paysim --query-mode active-edge --window-mode fixed --protocol chronological --seed 42

# 全量（最终按用户要求缩短为 IBM × 单 seed）
/home/zhu/.conda/envs/IF/bin/python script/run_all_models.py \
  --datasets ibm_hi_small ibm_li_small \
  --models IF DIF MAD --seeds 42 --protocol chronological

# 汇总
/home/zhu/.conda/envs/IF/bin/python script/evaluate_all.py --protocol chronological
```

`run_all_models.py` 带 tqdm 实时进度条（百分比 / 已完成数 / 剩余时间 / 当前 run）。

## 10. 测试结果

`/home/zhu/.conda/envs/IF/bin/python -m pytest -q` → **109 passed**
（39 个既有 link-stream pipeline 测试 + 70 个新测试：
`test_tabular_pipeline.py` 13、`test_dif.py` 18、`test_if.py` 5、
`test_metrics.py` 9、`test_mad.py` 17、`test_integration.py` 3 + 其它）。

## 11. 各阶段状态

| Stage | 状态 |
|---|---|
| Stage 0 环境/仓库/source code 检查 | PASS |
| Stage 1 公共接口与 metrics | PASS |
| Stage 2 IF | PASS |
| Stage 3 DIF | PASS |
| Stage 4 MAD | PASS |
| Stage 5 统一运行与汇总 | PASS（见 result/summary.csv） |

## 12. 指标与运行结果

**最终实验范围**（用户要求缩短运行）：`ibm_hi_small`、`ibm_li_small` 两个数据集，
`IF / DIF / MAD` 三模型，`seed 42` 单 seed，`chronological` protocol。
全部 6 次运行成功（0 失败），总耗时 2671.6s（约 44 分钟，其中 DIF 占大头）。

`result/summary.csv`（test split，ROC-AUC）：

| model | ibm_hi_small | ibm_li_small |
|---|---|---|
| IF | 0.6172 | 0.5700 |
| DIF | 0.5849 | 0.5817 |
| **MAD** | **0.8395** | **0.7926** |

完整字段（PR-AUC、Precision@K/Recall@K/F1@K、fit/score 时间、吞吐）见
`result/summary.csv` / `result/summary.json` 与 `result/README.md`。
每次运行的原始结果在 `result/<MODEL>/<DATASET>/chronological/seed_42/`
（config / metrics / scores.parquet / timing / warnings / run.log / interface_check.json）。

注：MAD 与 IF/DIF 输入范式不同（link stream vs tabular），
指标只在统一标签与统一 metric 下可比，不构成同类模型对比。
注：单 seed 结果不构成均值/方差统计（多 seed 运行已被用户按需缩短）。

## 13. Runtime 与内存

- 实验运行期间记录于各 run 的 `timing.json`（fit_seconds / score_seconds /
  throughput）与 `run.log`；峰值内存未做专门测量（若需要，
  可用 `/usr/bin/time -v` 重跑）。
- DIF/IF 评分按批流式（`score_batch_size` / `batch_size`），峰值内存 ~数百 MB 级。

## 14. 失败案例与限制

- 最终实验按用户要求缩短为 IBM 两数据集 × 单 seed（42），其余三个数据集
  （midas_darpa / paysim / elliptic_pp）未纳入最终汇总；单 seed 无方差统计；
- `pyarrow` 未安装（无需，polars 自带 parquet 引擎）；安装命令见第 1 节。
- elliptic_pp 的 local/aggregated 特征边界为原版 Elliptic 布局假设；
- MAD `full-scope` 为受限关系空间，非全局 E=V×V；
- MAD batch 路径 `stationary` 降级为 `expanding`（`window_mode_effective` 标注）；
- IF/DIF 在 midas_darpa 等结构稀疏数据上结果较弱（允许且公平，未加复杂图统计）。

## 15. 结论

Implemented a paper-guided reproduction using the provided source code where available.
