# Reasonix 单 Agent 顺序复现任务：IF、DIF 与 MAD

## 0. 核心目标

在仓库：

```bash
cd "/home/zhu/Deep Graph Isolation Tree"
```

中，由 **一个 Reasonix Agent 独立、顺序地** 完成三个异常检测模型的实现、测试和统一评估：

1. **Isolation Forest（IF）**：使用 `scikit-learn` 的 `IsolationForest` 作为基础基线；
2. **Deep Isolation Forest（DIF）**：依据论文与已有 source code 进行复现；
3. **MAD**：依据论文与已有 source code 复现 link-stream 异常检测。

**禁止使用 subagent、并行 agent、后台 agent 或 skill 的 `runAs: subagent`。**

必须按照以下顺序执行：

```text
Stage 0: 仓库、环境、论文和 source code 检查
Stage 1: 公共数据接口与统一 metrics
Stage 2: IF 实现与验收
Stage 3: DIF 实现与验收
Stage 4: MAD 实现与验收
Stage 5: 三模型统一运行、汇总和报告
```

每完成一个模型，必须先完成该模型的：

```text
代码检查 → 数据接口检查 → 单元测试 → smoke test → 指标输出检查 → 阶段报告
```

只有当前阶段全部通过后，才允许进入下一个模型。不得一次性把三个模型都写完后再统一排错。

Python 解释器必须显式使用：

```bash
/home/zhu/.conda/envs/IF/bin/python
```

不得假设 `python` 或 `pip` 指向正确环境。

---

# 1. 可信实现依据与 source code

用户已经将两篇论文的相关 source code 放在：

```text
/home/zhu/Deep Graph Isolation Tree/source code/
```

开始编码前必须：

1. 递归列出 `source code/` 的目录结构；
2. 确认哪些目录对应 MAD、DIF，以及是否包含 IF 或数据处理代码；
3. 阅读 README、requirements、入口脚本、核心模型文件和评估脚本；
4. 记录源代码的许可证、依赖、预期输入和输出；
5. 优先复用或适配作者实现，不要在未检查 source code 的情况下凭记忆重写；
6. 不要直接覆盖原始 source code；项目实现写入 `script/`，必要时保留清晰的适配层；
7. 若 source code 与论文不一致，必须记录差异，不得静默选择其中一方；
8. 若 source code 不完整或无法运行，允许进行 paper-guided reconstruction，但必须标记为 `partial`。

研究依据：

```text
Deep Isolation Forest for Anomaly Detection(10).pdf
MAD- Multi-Scale Anomaly Detection in Link Streams(20260805-121035).pdf
source code/
```

DIF 的关键不是训练神经网络，而是使用未经优化的随机初始化网络产生多个随机表示，并在这些表示空间中执行 isolation。

MAD 的标准输入是离散时间 link stream：

```text
(t, u, v)
```

它应依据历史上下文估计关系出现概率，并在多个尺度上分析当前 active/inactive 通信模式。

---

# 2. 总体执行规则

## 2.1 单 Agent 工作方式

当前 Agent 负责全部工作：

```text
仓库检查
论文和 source code 阅读
公共接口设计
数据适配
三个模型实现
单元测试
集成测试
运行实验
结果核对
复现报告
```

禁止：

- 创建或调用 subagent；
- 使用 `/skill new` 创建并行复现 skill；
- 把任务拆给隔离 loop；
- 在没有亲自运行测试的情况下相信已有输出；
- 同时大规模修改三个模型，导致无法定位问题。

## 2.2 阶段化提交与检查

每个 Stage 结束时输出阶段报告：

```text
Stage name
Files created/modified
Source files consulted
Commands executed
Tests passed/failed
Data interface status
Metric output status
Known deviations
Blocking issues
Decision: PASS / FAIL
```

当阶段状态为 `FAIL` 时，不得进入下一阶段。

如果仓库使用 Git，建议每个通过的阶段形成一个独立 commit，但未经用户要求不得强制 push。

---

# 3. 目标目录结构

保持实现简单，优先复用仓库已有结构。推荐：

```text
Deep Graph Isolation Tree/
├── source code/                    # 用户提供的论文 source code，只读参考
├── script/
│   ├── model/
│   │   ├── IF/
│   │   │   ├── __init__.py
│   │   │   ├── isolation_forest.py
│   │   │   ├── runner.py
│   │   │   └── README.md
│   │   ├── DIF/
│   │   │   ├── __init__.py
│   │   │   ├── dif.py
│   │   │   ├── random_encoder.py
│   │   │   ├── deep_isolation_tree.py
│   │   │   ├── scoring.py
│   │   │   ├── runner.py
│   │   │   └── README.md
│   │   └── MAD/
│   │       ├── __init__.py
│   │       ├── mad.py
│   │       ├── probability.py
│   │       ├── multiscale.py
│   │       ├── stationarity.py
│   │       ├── runner.py
│   │       └── README.md
│   ├── tabular_pipeline/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── preprocessing.py
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── common/
│   │   ├── metrics.py
│   │   ├── results.py
│   │   ├── seeds.py
│   │   ├── timing.py
│   │   └── validation.py
│   ├── run_all_models.py
│   └── evaluate_all.py
├── result/
│   ├── IF/
│   ├── DIF/
│   ├── MAD/
│   ├── summary.csv
│   ├── summary.json
│   └── README.md
├── tests/
│   ├── test_tabular_pipeline.py
│   ├── test_if.py
│   ├── test_dif.py
│   ├── test_mad.py
│   ├── test_metrics.py
│   └── test_integration.py
└── REPRODUCTION.md
```

不要创建大型框架，不使用 Hydra、数据库、Web API、分布式训练或复杂插件系统。

---

# 4. 数据集范围

必须检查并支持当前 pipeline 中实际存在的以下数据集：

```text
midas_darpa
ibm_hi_small
ibm_li_small
paysim
elliptic_pp
```

默认不得处理 IBM Medium 或 Large。

若某个数据集实际不存在或 schema 与预期不同：

1. 不得静默跳过；
2. 在阶段报告中列出实际路径和缺失内容；
3. 使 runner 返回明确的 `missing_dataset` 或 `unsupported_schema` 状态；
4. 不得伪造实验结果。

---

# 5. 两种数据范式必须严格分开

## 5.1 IF 与 DIF：tabular

IF 和 DIF 必须共用完全相同的 tabular pipeline、样本、特征和 split。

每一行是一个独立样本：

```text
X[i] -> y[i]
```

`X` 只能包含当前原始记录可获得的 tabular 特征。

默认禁止加入：

- 使用未来事件计算的统计量；
- 全图 PageRank、中心性、社区等图特征；
- 基于完整数据集标签构造的特征；
- 目标标签本身；
- PaySim 的 `isFlaggedFraud` 作为 ground truth；
- Elliptic++ 中时间有效性不明确的未来聚合图特征。

## 5.2 MAD：link stream

MAD 必须使用已有 processed link-stream pipeline 作为唯一标准数据入口，不要复制一套独立事件预处理逻辑。

标准事件：

```text
(t, u, v)
```

必须保证：

```text
history_event_id < current_event_id
```

即便两个事件时间戳相同，也不得让当前事件或未来事件进入历史上下文。

MAD 不得默认把完整数据集转换为静态图或普通 tabular feature matrix。

## 5.3 比较边界

三者可以使用统一标签范围和统一 metrics，但不能声称输入范式完全相同：

```text
IF/DIF: tabular anomaly detection
MAD: temporal link-stream anomaly detection
```

能通过 `sample_id` 对齐的样本应保存对齐关系；不能对齐时必须解释原因。

---

# 6. Stage 0：环境、仓库、论文和 source code 检查

首先执行：

```bash
pwd
git status --short
tree -L 4 .
tree -L 4 "source code"
```

检查 Python 环境：

```bash
/home/zhu/.conda/envs/IF/bin/python -c \
"import sys, numpy, sklearn, torch; print(sys.executable); print(numpy.__version__, sklearn.__version__, torch.__version__)"
```

检查数据依赖：

```bash
/home/zhu/.conda/envs/IF/bin/python -c \
"import polars, pyarrow, pandas; print(polars.__version__, pyarrow.__version__, pandas.__version__)"
```

若缺依赖：

- 不使用 `sudo`；
- 不静默安装；
- 先报告准确的用户空间安装命令；
- 只有用户允许或项目任务明确授权后才安装。

必须检查：

1. 五个数据集的真实路径与列名；
2. 现有 link-stream pipeline 的公共入口；
3. temporal index 如何保证 past-only；
4. source code 中 MAD 与 DIF 的入口及依赖；
5. 已有 tests 是否通过；
6. 是否已有同名模块，避免重复造轮子。

Stage 0 验收：

- [ ] 环境解释器正确；
- [ ] source code 已完整定位；
- [ ] 数据路径与 schema 已记录；
- [ ] link-stream 入口已定位；
- [ ] 当前测试基线已运行；
- [ ] 没有开始盲目编码。

---

# 7. Stage 1：公共数据接口与统一 metrics

必须先完成公共接口，再实现模型。

## 7.1 Tabular API

输出目录：

```text
data/processed_tabular/<dataset_name>/
├── features.parquet
├── labels.npy
├── label_available.npy
├── sample_ids.npy
├── splits.json
├── feature_schema.json
├── preprocessing.json
└── metadata.json
```

公共 API：

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TabularData:
    X: np.ndarray
    y: np.ndarray
    label_available: np.ndarray
    sample_ids: np.ndarray
    feature_names: list[str]
    timestamps: np.ndarray | None = None


def load_tabular_dataset(
    dataset_name: str,
    split: str = "all",
    fit_preprocessor_on: str = "train",
) -> TabularData:
    ...
```

基本要求：

1. 数值特征转为 `float32`；
2. `NaN` 和无穷值显式处理；
3. scaler、imputer、categorical encoder 只在 train 拟合；
4. 标签只用于评估；
5. 保存 preprocessing state，val/test 严格复用；
6. 高基数 ID 不进行巨大 one-hot；
7. 类别编码确定性且支持 unknown category；
8. 所有特征、删除列和编码方法写入 schema；
9. 同样输入和 seed 两次预处理结果一致；
10. 禁止静默猜测原始 schema。

## 7.2 Link-stream API

优先复用现有 API。若缺统一包装，可提供轻量适配层：

```python
@dataclass
class LinkStreamData:
    event_ids: np.ndarray
    timestamps: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    y: np.ndarray
    label_available: np.ndarray
    sample_ids: np.ndarray


def load_link_stream_dataset(
    dataset_name: str,
    split: str = "all",
) -> LinkStreamData:
    ...
```

适配层不得重做已有 preprocessing；只负责标准化字段和验证顺序。

必须验证：

```text
event_ids strictly define order
history lookup is past-only
labels align with events
sample_ids are stable
timestamps are non-decreasing after sorting
```

## 7.3 统一 metrics

所有模型统一：

```text
larger score = more anomalous
```

对已知标签样本报告：

```text
ROC-AUC
PR-AUC / Average Precision
Precision@K
Recall@K
F1@K
number of evaluated samples
number of positives
fit time
score time
throughput
peak memory if practical
```

其中：

```text
K = number of positive labels in evaluated split
```

统一函数建议：

```python
def evaluate_anomaly_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_available: np.ndarray | None = None,
) -> dict:
    ...
```

要求：

- unknown label 不进入指标；
- 如果只有一个类别，ROC-AUC 返回 `null` 并产生 warning；
- score、label 或 mask 长度不一致时直接报错；
- 非有限 score 必须报错；
- Precision@K、Recall@K、F1@K 对并列分数采用确定性排序；
- 三个模型不得各自实现不同版本的 metric。

## 7.4 Stage 1 必须测试

```bash
/home/zhu/.conda/envs/IF/bin/python -m pytest -q \
  tests/test_tabular_pipeline.py \
  tests/test_metrics.py
```

至少测试：

- train-only preprocessing；
- unknown category；
- unknown labels 被排除；
- 单类别 ROC-AUC 行为；
- score 方向约定；
- 重复执行结果一致；
- link-stream 事件与标签对齐；
- past-only history validator 能捕获泄漏。

Stage 1 状态必须为 `PASS` 才可进入 IF。

---

# 8. 各数据集的 tabular 映射

## 8.1 PaySim

样本：一笔交易。

标签：

```text
isFraud
```

候选特征：

```text
step
type
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

`isFlaggedFraud` 默认关闭，不能作为 ground truth 的替代。

`nameOrig`、`nameDest` 不得直接 one-hot；默认删除或使用训练集 frequency encoding，并记录选择。

## 8.2 IBM AML

样本：一笔交易。

标签：

```text
Is Laundering
```

候选特征依据真实 schema 核查，通常包括：

```text
Timestamp
From Bank
To Bank
Amount Received
Receiving Currency
Amount Paid
Payment Currency
Payment Format
same_bank
same_currency
```

账户字符串可删除或使用 train-only frequency encoding；不得产生超大 one-hot。

HI 和 LI 分开输出：

```text
ibm_hi_small
ibm_li_small
```

## 8.3 MIDAS-DARPA

样本：一条通信事件。

标签：与事件行严格对齐的 ground truth。

基础特征必须依据实际 CSV schema，至少核查：

```text
timestamp
src
dst
```

高基数 `src`、`dst` 使用 train-only frequency encoding 或稳定 ordinal encoding。

禁止加入由未来历史计算的动态图统计。

## 8.4 Elliptic++

IF/DIF 样本采用 transaction node，而不是 money-flow edge。

标签：

```text
illicit -> 1
licit   -> 0
unknown -> -1
```

要求：

- transaction ID 用于 join 和 sample ID，不作为连续特征；
- unknown 保留但不进入指标；
- 区分 local 与 aggregated features；
- 若 aggregated features 的时间安全性不明确，默认使用 `local`；
- 提供 `local` 与 `all` 配置并记录。

## 8.5 结构字段很少的数据

不得为了提升 IF/DIF 结果而偷偷加入复杂图统计。结果较弱是允许的，数据范式必须公平、可解释。

---

# 9. Stage 2：Isolation Forest

IF 是第一个模型，用于首先验证 tabular pipeline 和统一评估协议。

必须使用：

```python
from sklearn.ensemble import IsolationForest
```

默认配置：

```python
IsolationForest(
    n_estimators=300,
    max_samples=256,
    contamination="auto",
    random_state=seed,
    n_jobs=-1,
)
```

统一 anomaly score：

```python
anomaly_score = -model.score_samples(X)
```

要求：

- 只在 train split fit；
- 不使用测试标签设置 contamination；
- IF 与后续 DIF 使用同一 tabular 输入；
- 保存模型配置、score、timing 和 warnings；
- fit 与 score API 清晰分离；
- batch 与全量 score 一致。

## 9.1 IF 数据接口验收

在实现完成后，必须对每个数据集执行快速接口检查：

```text
load train/val/test
print X shape, y shape, label mask shape
check finite X
check feature names count == X.shape[1]
check sample IDs unique within split
fit only train
score train/val/test
check score finite
check score length == sample count
run unified metrics on labeled test rows
```

必须先至少测试：

```text
paysim
ibm_hi_small
elliptic_pp
```

然后对所有五个数据集运行 smoke test。

## 9.2 IF 测试

至少测试：

- score 越大越异常；
- 只在 train fit；
- 不读取测试标签；
- 相同 seed 结果一致；
- score 数量和 sample 数一致；
- unknown labels 不进入 metric；
- synthetic anomalies 平均得分高于正常样本；
- 五个数据集 runner 均能启动并产生结构正确的结果文件。

运行：

```bash
/home/zhu/.conda/envs/IF/bin/python -m pytest -q \
  tests/test_if.py \
  tests/test_tabular_pipeline.py \
  tests/test_metrics.py
```

Smoke test：

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/IF/runner.py \
  --dataset paysim \
  --protocol chronological \
  --seed 42 \
  --quick
```

IF 阶段只有在以下内容均正常时才可 PASS：

- [ ] tabular data interface；
- [ ] score 方向；
- [ ] metrics 输出；
- [ ] result schema；
- [ ] unit tests；
- [ ] 五数据集 smoke records。

---

# 10. Stage 3：Deep Isolation Forest

DIF 必须在 IF 与 tabular pipeline 已通过后实现。

DIF 不是“训练一个神经网络然后再跑 IF”。其核心包括：

1. 未经优化或训练的随机初始化神经网络；
2. 多个随机表示；
3. 每个表示空间中的多棵 isolation trees；
4. 随机表示与随机 partition 的 ensemble；
5. CERE；
6. DEAS。

优先依据论文 source code 恢复以下配置；若 source code 与论文默认实验一致，可采用：

```text
r = 50 random representations
t = 6 isolation trees per representation
T = 300 total trees
subsample size = 256
```

DIF tabular encoder 使用 MLP。

## 10.1 必须实现或精确核查

```text
random representation ensemble
CERE-style efficient ensemble mapping
isolation-tree construction in each projected space
maximum tree depth
path-length computation
average split deviation along traversed path
DEAS anomaly score
batched inference
deterministic seeding
fit-time random state persistence
```

不得只将多个随机 MLP 输出拼接后调用一次 sklearn `IsolationForest`，并声称这是完整 DIF。

允许保留：

```text
--implementation faithful
--implementation simplified
```

但：

- `faithful` 是默认目标；
- `simplified` 单独标记；
- simplified 结果不得混入 faithful 汇总；
- 若 CERE 或 DEAS 未实现，`paper_fidelity` 不能写 `faithful`。

## 10.2 CERE

必须从论文公式与 source code 核查：

```text
shared base weights
rank-one random vectors
ensemble-member modulation
batched representation generation
```

若首版使用独立随机 MLP：

1. 网络仍不得训练；
2. README 标为非完整 CERE；
3. `paper_fidelity = partial`；
4. 不得提前将 DIF 阶段标为完整 PASS，除非任务明确接受 partial。

## 10.3 DEAS

至少保留：

```text
tree path length
+
average quantitative deviation from split thresholds
```

必须根据论文与 source code 恢复准确公式、归一化和聚合方式。

## 10.4 DIF 数据接口验收

DIF 必须复用 IF 已验证的 tabular loader，不得创建另一套数据解析。

对每个数据集检查：

```text
IF X shape == DIF X shape
IF sample_ids == DIF sample_ids
IF labels == DIF labels
IF split == DIF split
random encoder input dimension == X.shape[1]
representation shape is stable
all representations finite
all scores finite
score count == sample count
unified metric function works unchanged
```

必须对相同 dataset、split 和 seed 保存 IF/DIF 的接口对照报告。

## 10.5 DIF 测试

至少测试：

- 随机 encoder 没有 optimizer；
- 不进行 backward 或参数更新；
- 相同 seed 表示一致；
- 不同 member 表示不同；
- 每个 representation 的 tree 数正确；
- total trees 与配置一致；
- isolation path length 小例子正确；
- DEAS score 有限；
- batch 与 single-sample score 一致；
- inference 复用 fit 时随机权重；
- anomaly score 越大越异常；
- synthetic anomalies 平均得分较高；
- 五个数据集均能运行；
- 与 IF 完全共用 tabular interface。

运行：

```bash
/home/zhu/.conda/envs/IF/bin/python -m pytest -q \
  tests/test_dif.py \
  tests/test_if.py \
  tests/test_tabular_pipeline.py \
  tests/test_metrics.py
```

Smoke test：

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/DIF/runner.py \
  --dataset paysim \
  --protocol chronological \
  --seed 42 \
  --implementation faithful \
  --quick
```

DIF Stage 报告必须明确：

```text
paper-faithful components
source-code-derived components
approximated components
CERE status
DEAS status
network architecture
runtime per dataset
known deviations
```

---

# 11. Stage 4：MAD

只有 IF 和 DIF 的 tabular 路径均稳定后，才进入 MAD。

MAD 接受：

```text
Query Q: 当前时间 t 上任意关系集合 phi 的 active/inactive 状态
History H: 查询前长度 N 的历史窗口
```

每条候选关系依据历史被建模为 Bernoulli trial：

```text
P(e)
```

查询异常性不是简单计算整个 subgraph realization 的联合概率，而应执行论文的 multi-scale probabilistic analysis。

## 11.1 必须实现或精确核查

```text
edge probability estimation from history
probability-based relation ordering
power-of-two padding
multi-scale binary partition
random-variable expectation
random-variable variance
standardized deviation or paper tail score
aggregation across scales
most anomalous scale selection
stationarity-based history selection
edge query
node query
slice query
arbitrary subgraph query
```

必须优先依据 MAD source code 恢复公式和默认参数。

## 11.2 时间与历史约束

正确顺序必须是：

```text
1. Query history
2. Estimate probabilities
3. Score current query
4. Update state with current event/slice
```

严格禁止：

- 当前事件进入自己的历史；
- 当前 slice 在评分前更新概率；
- 未来事件进入历史；
- 对完整数据集先统计概率再回头评分。

相同时间戳事件必须依据稳定 `event_id` 顺序处理。

## 11.3 离散时间策略

支持：

```text
native discrete step
fixed-width time bins
event-count bins
```

默认优先：

- PaySim：原始 `step`；
- Elliptic++：原始 time step；
- IBM AML：依据真实 timestamp 使用可配置 fixed-width bin；
- MIDAS-DARPA：依据实际 schema 选择 native 或 fixed-width bin。

不得查看测试标签后选择 bin size。

每个数据集的 policy 必须写入 metadata。

## 11.4 查询 API

至少支持：

```python
score_edge(t, src, dst)
score_node(t, node_id, direction="out")
score_slice(t, relation_scope=None)
score_subgraph(t, relations)
```

主实验由于数据集通常提供 event labels，可使用：

```text
edge-event anomaly detection
```

并提供：

```text
--query-mode active-edge
--query-mode local-candidate-set
--query-mode full-scope
```

`active-edge` 是完整 MAD 框架的 edge-query 特例，不能写成完整 arbitrary-subgraph 实验。

`local-candidate-set` 应考虑过去与当前局部范围中的 active 和 inactive relations，从而表达 unexpected-active 与 expected-but-missing。

## 11.5 概率估计

基础定义：

```text
P(e) = active historical slices / number of historical slices
```

必须处理：

```text
zero history
unseen relation
probability exactly 0 or 1
short history
non-stationary history
```

若使用 smoothing：

- 参数可配置；
- 默认值记录；
- 标明是否与论文一致；
- 不得改变定义却不说明。

## 11.6 Stationarity window

必须从论文和 source code 精确恢复 stationarity-based window selection。

若无法精确恢复：

```text
--history-mode fixed
--history-mode expanding
--history-mode stationary
```

其中只有精确恢复后，`stationary` 才可标记为 faithful。否则：

```text
paper_fidelity = partial
```

不得自创统计检验后称为论文原算法。

## 11.7 MAD 数据接口验收

MAD 不能复用 tabular loader 代替 link-stream loader，但必须使用相同的：

```text
sample_id convention
label definition
split protocol
metric function
result schema
score direction
```

对每个数据集检查：

```text
events sorted by (timestamp, event_id)
labels align with event IDs
history excludes current and future
score count matches evaluated events
scores finite
score direction is larger = more anomalous
unified metric function works unchanged
```

必须额外生成泄漏检查日志，例如随机抽取若干 event：

```text
current_event_id
max_history_event_id
current_timestamp
max_history_timestamp
history_size
```

并断言：

```text
max_history_event_id < current_event_id
```

## 11.8 MAD 测试

至少测试：

- 当前和未来事件永不进入历史；
- 同时间戳遵守 event ID 顺序；
- Bernoulli probability estimation 正确；
- padding 不改变真实关系统计；
- multi-scale partition 覆盖全部真实关系且不重复；
- expectation 与 variance 在手算小例子中正确；
- 概率全相同、全 0、全 1 时稳定；
- densification 得到较高异常分数；
- sparsification 得到较高异常分数；
- rewiring 得到较高异常分数；
- edge/node/slice/subgraph API 工作；
- 五个数据集 runner 均能运行；
- 输出 selected scale 和 relation subset；
- metric 输出与 IF/DIF 使用同一函数。

运行：

```bash
/home/zhu/.conda/envs/IF/bin/python -m pytest -q \
  tests/test_mad.py \
  tests/test_metrics.py \
  tests/test_integration.py
```

Smoke test：

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/MAD/runner.py \
  --dataset paysim \
  --query-mode active-edge \
  --history-mode fixed \
  --protocol chronological \
  --seed 42 \
  --quick
```

MAD Stage 报告必须明确：

```text
exact equations implemented
source-code-derived components
stationarity method status
query modes implemented
past-only verification
paper-faithful vs partial components
runtime and bottlenecks
```

---

# 12. 统一 split 和评估协议

## 12.1 默认协议

使用 chronological split：

```text
train = first 60%
val   = next 20%
test  = final 20%
```

IF/DIF：

- train fit；
- train、val、test score；
- 主要报告 test。

MAD：

- 顺序在线评分；
- train 可用于 warm-up / probability estimation；
- val/test 不得读取未来；
- 记录每个数据集 warm-up 长度。

可额外提供：

```text
--protocol transductive
```

但默认正式结果必须是 leakage-safe chronological protocol，两种结果分开保存。

## 12.2 Seeds

至少运行：

```text
42
43
44
45
46
```

保存每个 seed 原始结果并汇总 mean/std。

MAD 若完全确定性，也执行一致性检查；不得人为增加无意义随机性。

## 12.3 公平性

1. IF 和 DIF 使用完全相同的 tabular samples、features 和 split；
2. MAD 保持 link-stream 输入；
3. 三者使用相同标签定义和统一 metric；
4. 能对齐时使用相同 test sample/event 范围；
5. 不使用 test labels 调参；
6. 不只报告最优 seed；
7. 失败或超时运行保留状态；
8. PaySim 提供 chunked/batched 模式；
9. 所有近似实现明确标记；
10. quick run 不进入正式汇总。

---

# 13. 统一命令行

## 13.1 IF

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/IF/runner.py \
  --dataset paysim \
  --protocol chronological \
  --seed 42
```

## 13.2 DIF

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/DIF/runner.py \
  --dataset paysim \
  --protocol chronological \
  --implementation faithful \
  --seed 42
```

## 13.3 MAD

```bash
/home/zhu/.conda/envs/IF/bin/python script/model/MAD/runner.py \
  --dataset paysim \
  --query-mode active-edge \
  --history-mode fixed \
  --protocol chronological \
  --seed 42
```

## 13.4 全模型运行

只有三个模型各自 Stage 均 PASS 后，才创建或运行：

```bash
/home/zhu/.conda/envs/IF/bin/python script/run_all_models.py \
  --datasets midas_darpa ibm_hi_small ibm_li_small paysim elliptic_pp \
  --models IF DIF MAD \
  --seeds 42 43 44 45 46 \
  --protocol chronological
```

支持：

```text
--overwrite
--resume
--device cpu|cuda
--num-workers
--batch-size
--max-samples
--quick
```

`--quick` 结果必须包含：

```text
is_quick_run = true
```

不得混入正式汇总。

---

# 14. 结果格式

每次运行保存：

```text
result/<MODEL>/<DATASET>/<PROTOCOL>/seed_<SEED>/
├── config.json
├── metrics.json
├── scores.parquet
├── timing.json
├── warnings.json
├── interface_check.json
└── run.log
```

`scores.parquet` 至少包含：

```text
sample_id
timestamp_or_step
label
label_available
anomaly_score
split
```

MAD 额外包含：

```text
query_mode
selected_scale
selected_relation_count
history_window
explanation
```

DIF 额外记录：

```text
implementation_mode
num_representations
trees_per_representation
total_trees
subsample_size
encoder_architecture
cere_enabled
deas_enabled
```

总汇总：

```text
result/summary.csv
result/summary.json
```

字段至少包括：

```text
model
dataset
protocol
seed
roc_auc
pr_auc
precision_at_k
recall_at_k
f1_at_k
num_evaluated
num_positives
fit_seconds
score_seconds
throughput
status
paper_fidelity
is_quick_run
```

`paper_fidelity` 只允许：

```text
faithful
partial
simplified
baseline
```

IF 使用：

```text
baseline
```

---

# 15. 最终集成检查

三个模型都通过各自 Stage 后，必须亲自运行：

```bash
/home/zhu/.conda/envs/IF/bin/python -m pytest -q
```

并执行至少一个跨模型对齐检查：

```text
同一 dataset 的 IF/DIF sample_ids 完全一致
同一 dataset 的 IF/DIF labels 完全一致
三个模型 anomaly score 都是越大越异常
三个模型 metrics.json 使用同一字段与实现
unknown labels 在三个模型中均被排除
所有正式 test 结果不包含 quick run
```

不得只相信单模型阶段报告。

---

# 16. 复现报告

创建：

```text
REPRODUCTION.md
```

必须包含：

1. 仓库与环境；
2. Python 解释器；
3. `source code/` 中实际使用的文件；
4. 数据集统计与真实 schema；
5. Tabular 与 link-stream 两种范式；
6. 公共 metric 定义；
7. IF 实现与接口验收；
8. DIF 论文方法、source code 映射和实现；
9. MAD 论文方法、source code 映射和实现；
10. 每个阶段的 PASS/FAIL 记录；
11. 精确实现的组件；
12. 未能精确恢复的组件；
13. 与论文/source code 的偏差；
14. 完整运行命令；
15. 测试结果；
16. 每个数据集的指标；
17. runtime 与 memory；
18. 失败案例和限制；
19. 结论边界。

不得写：

```text
successfully reproduced the paper
```

除非主要算法、实验协议和结果趋势均有充分证据。

更稳妥的表述：

```text
Implemented a paper-guided reproduction using the provided source code where available.
```

或：

```text
Implemented a partial reproduction with documented deviations.
```

---

# 17. 必须防止的错误

- 使用 subagent 或并行 agent；
- 未检查 `source code/` 就凭记忆重写；
- 同时实现三个模型，最后才检查接口；
- IF 未通过数据接口验收就开始 DIF；
- DIF 未通过 tabular 对齐检查就开始 MAD；
- 把 DIF 写成随机 MLP + 单个 sklearn IF 后声称完整复现；
- 对 DIF 随机网络执行训练；
- 不复用 fit 时的随机权重；
- 不实现 DEAS 却标记 faithful；
- 把 MAD 转成静态图分类；
- MAD 使用当前或未来事件；
- 只检测 active edge，却声称完整 arbitrary-subgraph MAD；
- 自创 stationarity test 后称为论文原算法；
- 用标签拟合 scaler、encoder、contamination、bin size 或 history window；
- IF 与 DIF 使用不同 features；
- Elliptic++ 把 unknown 当正常；
- PaySim 对账户 ID 做巨大 one-hot；
- 三个模型各自实现不同 metrics；
- 静默跳过失败数据集；
- 只运行一个 seed；
- 只报告 ROC-AUC；
- quick run 混入正式结果；
- 未亲自运行全部测试就宣布完成。

---

# 18. 最终验收标准

只有满足以下全部条件才可结束：

- [ ] 未使用任何 subagent；
- [ ] 已检查论文和 `source code/`；
- [ ] 公共 tabular、link-stream 适配和 metrics 已先完成；
- [ ] IF 已独立实现、测试并通过数据接口验收；
- [ ] DIF 在 IF 之后实现，并与 IF 完全共用 tabular pipeline；
- [ ] DIF 的 CERE、DEAS 状态被准确记录；
- [ ] MAD 在 DIF 之后实现，并使用现有 link-stream pipeline；
- [ ] MAD 已通过 past-only 泄漏检查；
- [ ] 每完成一个模型均生成 `interface_check.json`；
- [ ] 五个数据集均有明确运行记录或明确失败状态；
- [ ] 所有模型 score 方向一致；
- [ ] 所有模型使用同一个 metric 实现；
- [ ] chronological protocol 无未来泄漏；
- [ ] 单元测试和 integration tests 通过；
- [ ] 每个 seed 的配置和结果已保存；
- [ ] `result/summary.csv` 已生成；
- [ ] `REPRODUCTION.md` 已完成；
- [ ] partial/simplified 实现均被明确标记；
- [ ] 最终报告包含实际命令、测试结果、耗时和未解决问题。

---

# 19. 给 Reasonix Agent 的启动指令

请严格按顺序开始：

1. 阅读本文件；
2. 明确禁止使用 subagent；
3. 检查 Git 状态、仓库结构、五个数据集和现有 pipeline；
4. 递归检查 `/home/zhu/Deep Graph Isolation Tree/source code/`；
5. 阅读两篇论文和对应 source code；
6. 创建公共数据接口、统一 metrics 和验证器；
7. 运行 Stage 1 测试；
8. 实现 IF；
9. 对 IF 执行数据接口检查、测试和 smoke run，生成阶段报告；
10. IF PASS 后实现 DIF；
11. 对 DIF 执行与 IF 的输入对齐检查、测试和 smoke run，生成阶段报告；
12. DIF PASS 后实现 MAD；
13. 对 MAD 执行 link-stream 接口、past-only、公式和 smoke run 检查，生成阶段报告；
14. 三个模型均 PASS 后运行全部测试；
15. 执行五数据集、多 seed 正式实验；
16. 生成统一汇总与 `REPRODUCTION.md`。

最终回复必须包含：

```text
Execution mode: single agent, sequential
Source code inspected
Stage 0 status
Stage 1 status
IF stage status
DIF stage status
MAD stage status
Files created/modified
Commands executed
Test summary
Data-interface checks
Metric consistency checks
Dataset run summary
Result paths
Paper-faithful components
Partial/simplified components
Remaining limitations
```

不得只回复“完成了”。
