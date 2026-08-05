# AML/Fraud Transaction Benchmark Baselines

> Purpose: reproducible baseline shortlist for transaction-level fraud / money-laundering detection.
>
> Data representations:
>
> 1. **Tabular view**: each transaction is one feature row.
> 2. **Link-stream view**: each transaction is a directed timestamped edge  
>    `event = (sender, receiver, timestamp, edge_features, label)`.
>
> Existing baselines in the project: **IF**, **DIF**, **MAD**.
>
> Constraints:
>
> - No LLM-based methods.
> - Every method below has a paper and a publicly accessible implementation repository.
> - The list contains classic baselines and recent 2024–2025 methods.
> - Unsupervised and supervised results must be reported separately.

---

## 0. Canonical task definition

```python
TransactionEvent = {
    "src": int,                 # sender/account/entity ID
    "dst": int,                 # receiver/account/entity ID
    "timestamp": float,         # Unix time or monotonically ordered time
    "edge_features": list[float],
    "label": int,               # 0 = normal, 1 = fraud/AML
}
```

Recommended evaluation tracks:

```yaml
tracks:
  tabular_unsupervised:
    training_labels_used: false
    input: edge_features

  link_stream_unsupervised:
    training_labels_used: false
    input: [src, dst, timestamp, optional_edge_features]

  link_stream_supervised:
    training_labels_used: true
    input: [src, dst, timestamp, edge_features]

  transaction_graph_supervised:
    training_labels_used: true
    input: directed_multigraph_with_edge_features
```

Do not place supervised and unsupervised methods in the same ranking table without explicitly separating their training protocols.

---

# 1. ECOD

```yaml
name: ECOD
full_name: Empirical-CDF-based Outlier Detection
category: tabular_anomaly_detection
learning: unsupervised
year: 2022
venue: IEEE TKDE
priority: high
resource_cost: low
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

ECOD is a parameter-light statistical anomaly detector based on empirical cumulative distribution tails. It complements IF/DIF because it is not isolation-based, density-neighbor-based, or reconstruction-based.

## Input

```text
X shape: [number_of_transactions, number_of_tabular_features]
```

Do not feed integer-encoded sender and receiver IDs as continuous values. Either exclude them or replace them with leakage-safe historical statistics.

## Repository

- PyOD: https://github.com/yzhao062/pyod
- ECOD implementation: https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py
- Example: https://github.com/yzhao062/pyod/blob/master/examples/ecod_example.py

## Paper

- arXiv: https://arxiv.org/abs/2201.00382

## Minimal use

```python
from pyod.models.ecod import ECOD

model = ECOD()
model.fit(X_train)
scores = model.decision_function(X_test)
```

## Adaptation effort

```yaml
adaptation_effort: very_low
expected_changes:
  - preprocessing
  - categorical_encoding
  - temporal_train_test_split
```

---

# 2. Deep SVDD

```yaml
name: Deep-SVDD
full_name: Deep Support Vector Data Description
category: deep_tabular_anomaly_detection
learning: unsupervised_or_one_class
year: 2018
venue: ICML
priority: high
resource_cost: medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

Deep SVDD is a canonical deep one-class baseline. It provides a non-tree neural comparison against IF and DIF.

## Input

```text
X shape: [number_of_transactions, number_of_tabular_features]
```

The official repository is image-oriented. For transaction tables, replace the CNN encoder with an MLP encoder while keeping the Deep-SVDD objective unchanged.

## Repository

- Official PyTorch implementation: https://github.com/lukasruff/Deep-SVDD-PyTorch
- Original repository: https://github.com/lukasruff/Deep-SVDD

## Paper

- PMLR: https://proceedings.mlr.press/v80/ruff18a.html
- arXiv: https://arxiv.org/abs/1802.06360

## Required adaptation

```yaml
adaptation_effort: medium
expected_changes:
  - replace_image_network_with_mlp
  - implement_tabular_dataset_loader
  - preserve_one_class_or_soft_boundary_objective
```

---

# 3. F-FADE

```yaml
name: F-FADE
full_name: Frequency Factorization for Anomaly Detection in Edge Streams
category: edge_stream_anomaly_detection
learning: unsupervised
year: 2021
venue: WSDM
priority: high
resource_cost: low_to_medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

F-FADE is a classic edge-stream anomaly detector that models time-evolving interaction frequencies between node pairs. It is a strong structural-frequency counterpart to MIDAS-style methods.

## Input

```text
src, dst, timestamp
```

The original method primarily models interaction frequency. Transaction attributes such as amount or currency may require separate preprocessing or model extension.

## Repository

- Official repository: https://github.com/snap-stanford/F-FADE

## Paper

- arXiv: https://arxiv.org/abs/2011.04723
- ACM DL: https://dl.acm.org/doi/10.1145/3437963.3441809

## Adaptation effort

```yaml
adaptation_effort: low_to_medium
expected_changes:
  - convert_transactions_to_sorted_edge_stream
  - map_sender_receiver_ids_to_contiguous_integers
  - align_output_scores_with_transaction_labels
```

---

# 4. SLADE

```yaml
name: SLADE
full_name: Self-supervised Learning for Anomaly Detection in Edge Streams
category: edge_stream_anomaly_detection
learning: self_supervised
year: 2024
venue: KDD
priority: critical
resource_cost: medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

SLADE is one of the most directly relevant recent baselines for label-free anomaly detection on dynamic edge streams. It detects abnormal changes in node interaction states and is designed for online processing.

## Input

```text
src, dst, timestamp, label_for_evaluation_only
```

Check the repository loader before adding edge attributes. The original method is primarily driven by temporal interaction structure.

## Repository

- Official repository: https://github.com/jhsk777/SLADE

## Paper

- ACM DL: https://dl.acm.org/doi/10.1145/3637528.3671845
- arXiv: https://arxiv.org/abs/2402.11933

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - implement_common_transaction_loader
  - preserve_chronological_order
  - ensure_labels_are_used_only_for_evaluation
```

---

# 5. TGN

```yaml
name: TGN
full_name: Temporal Graph Networks
category: continuous_time_dynamic_graph
learning: supervised_or_self_supervised
year: 2020
venue: arXiv_and_workshop_lineage
priority: critical
resource_cost: medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

TGN is a canonical architecture for continuous-time graphs represented as timed events. It supports node memory, temporal encoding, graph aggregation, and edge features.

## Valid experiment modes

```yaml
modes:
  supervised_edge_classification:
    target: fraud_or_aml_label
    score: predicted_probability

  self_supervised_link_anomaly:
    training_target: normal_link_prediction
    anomaly_score: negative_link_likelihood
```

These two modes must not be mixed in the same result table.

## Repository

- Official repository: https://github.com/twitter-research/tgn
- Unified DyGLib implementation: https://github.com/yule-buaa/dyglib
- Link-anomaly benchmark implementation: https://github.com/timpostuvan/TGB-link-anomaly-detection

## Paper

- arXiv: https://arxiv.org/abs/2006.10637

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - transaction_csv_to_temporal_event_loader
  - edge_feature_dimension_configuration
  - replace_or_extend_link_predictor_with_edge_classifier
```

---

# 6. GraphMixer

```yaml
name: GraphMixer
full_name: GraphMixer
category: continuous_time_dynamic_graph
learning: supervised_or_self_supervised
year: 2023
venue: ICLR
priority: high
resource_cost: medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

GraphMixer is a simpler temporal-graph architecture based largely on MLP mixing, temporal link encoding, neighbor pooling, and link classification. It gives a useful contrast against memory-based TGN.

## Repository

- Unified official research code in DyGLib: https://github.com/yule-buaa/dyglib
- TGB-compatible continuation: https://github.com/yule-buaa/DyGLib_TGB
- Link-anomaly adaptation: https://github.com/timpostuvan/TGB-link-anomaly-detection

## Paper

- ICLR: https://openreview.net/forum?id=ayPPc0SyLv1
- arXiv: https://arxiv.org/abs/2302.11636

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - common_temporal_graph_format
  - fraud_edge_classifier_or_anomaly_scoring_head
  - tune_history_length_and_number_of_neighbors
```

---

# 7. DyGFormer

```yaml
name: DyGFormer
full_name: Dynamic Graph Transformer
category: continuous_time_dynamic_graph
learning: supervised_or_self_supervised
year: 2023
venue: NeurIPS
priority: high
resource_cost: medium_to_high
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
```

## Why include it

DyGFormer is a strong Transformer-style dynamic-graph baseline that operates on historical first-hop interaction sequences, node co-occurrence encodings, and patched histories. It is not an LLM.

## Repository

- DyGLib, containing DyGFormer and common temporal baselines: https://github.com/yule-buaa/dyglib
- TGB-compatible repository: https://github.com/yule-buaa/DyGLib_TGB
- Link-anomaly benchmark adaptation: https://github.com/timpostuvan/TGB-link-anomaly-detection

## Paper

- NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/d611019afba70d547bd595e8a4158f55-Abstract-Conference.html
- OpenReview: https://openreview.net/forum?id=xHNzWHbklj
- arXiv: https://arxiv.org/abs/2303.13047

## Resource controls

```yaml
memory_controls:
  - batch_size
  - num_neighbors
  - history_length
  - patch_size
  - embedding_dimension
```

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - common_temporal_graph_format
  - transaction_level_decoder
  - reduce_history_and_batch_size_for_limited_gpu
```

---

# 8. GeneralDyG

```yaml
name: GeneralDyG
full_name: A Generalizable Anomaly Detection Method in Dynamic Graphs
category: dynamic_graph_anomaly_detection
learning: anomaly_detection
year: 2025
venue: AAAI
priority: high
resource_cost: medium
recommended_datasets:
  - elliptic_pp
  - ibm_aml
  - midas_darpa
  - paysim
special_value:
  - cross_dataset_generalization
```

## Why include it

GeneralDyG is a recent 2025 dynamic-graph anomaly detector. It samples temporal ego-graphs and extracts structural and temporal information, with emphasis on generalization across graph datasets.

## Repository

- Official repository: https://github.com/YXNTU/GeneralDyG

## Paper

- AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/35508
- arXiv: https://arxiv.org/abs/2412.16447

## Important compatibility check

The repository's original anomaly construction and task setup may differ from transaction-level fraud labels. Preserve the original model, but replace the data adapter and ensure that every output score corresponds to one transaction edge.

## Adaptation effort

```yaml
adaptation_effort: medium_to_high
expected_changes:
  - temporal_ego_graph_dataset_adapter
  - transaction_edge_score_alignment
  - possible_cross_dataset_feature_harmonization
```

---

# 9. Multi-GNN

```yaml
name: Multi-GNN
full_name: Provably Powerful Graph Neural Networks for Directed Multigraphs
category: transaction_graph_fraud_detection
learning: supervised
year: 2024
venue: AAAI
priority: critical
resource_cost: medium
recommended_datasets:
  - ibm_aml
  - paysim
  - elliptic_pp
```

## Why include it

Multi-GNN was designed for directed multigraphs and is highly relevant to financial transaction networks containing repeated directed transactions and edge attributes. The codebase includes AML experiments and variants based on GIN, GAT, PNA, and RGCN.

## Repository

- Official IBM repository: https://github.com/IBM/Multi-GNN

## Paper

- AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/29069
- IBM Research page: https://research.ibm.com/publications/provably-powerful-graph-neural-networks-for-directed-multigraphs
- arXiv: https://arxiv.org/abs/2306.11586

## Recommended variants

```yaml
variants:
  - GIN
  - PNA
  - Multi-GIN_or_Multi-PNA
```

Avoid running every architecture × augmentation combination unless performing an ablation study.

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - convert_dataset_to_directed_multigraph
  - preserve_parallel_edges
  - map_transaction_labels_to_edge_labels
  - configure_edge_features
```

---

# 10. FraudGT

```yaml
name: FraudGT
full_name: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection
category: transaction_graph_fraud_detection
learning: supervised
year: 2024
venue: ACM ICAIF
priority: critical
resource_cost: medium
recommended_datasets:
  - ibm_aml
  - paysim
  - elliptic_pp
```

## Why include it

FraudGT is a recent graph Transformer built specifically for fraud detection in financial transaction graphs. It is a direct modern comparison for graph-based transaction classification.

## Repository

- Official repository: https://github.com/junhongmit/FraudGT

## Paper

- ACM DL: https://dl.acm.org/doi/10.1145/3677052.3698648
- Author-hosted PDF: https://jshun.csail.mit.edu/FraudGT.pdf

## Adaptation effort

```yaml
adaptation_effort: medium
expected_changes:
  - convert_transaction_data_to_repository_graph_format
  - preserve_edge_attributes_and_direction
  - use_chronological_train_validation_test_split
```

---

# Recommended experiment matrix

Legend:

- `R` = recommended
- `O` = optional / requires task adaptation
- `-` = not a natural fit

| Baseline | Tabular | Link stream | Directed transaction graph | Unsupervised | Supervised |
|---|---:|---:|---:|---:|---:|
| ECOD | R | - | - | R | - |
| Deep SVDD | R | O via engineered features | - | R | - |
| F-FADE | - | R | - | R | - |
| SLADE | - | R | - | R/self-supervised | - |
| TGN | - | R | R | O | R |
| GraphMixer | - | R | R | O | R |
| DyGFormer | - | R | R | O | R |
| GeneralDyG | - | R | O | R/task-dependent | O |
| Multi-GNN | - | O/static snapshots | R | - | R |
| FraudGT | - | O/static snapshots | R | - | R |

---

# Dataset applicability

```yaml
elliptic_pp:
  tabular:
    - ECOD
    - Deep-SVDD
  link_stream:
    - F-FADE
    - SLADE
    - TGN
    - GraphMixer
    - DyGFormer
    - GeneralDyG
  graph_supervised:
    - Multi-GNN
    - FraudGT

ibm_aml:
  tabular:
    - ECOD
    - Deep-SVDD
  link_stream:
    - F-FADE
    - SLADE
    - TGN
    - GraphMixer
    - DyGFormer
    - GeneralDyG
  graph_supervised:
    - Multi-GNN
    - FraudGT

midas_darpa:
  tabular:
    - ECOD
    - Deep-SVDD
  link_stream:
    - F-FADE
    - SLADE
    - TGN
    - GraphMixer
    - DyGFormer
    - GeneralDyG
  graph_supervised:
    - Multi-GNN_optional
    - FraudGT_optional

paysim:
  tabular:
    - ECOD
    - Deep-SVDD
  link_stream:
    - F-FADE
    - SLADE
    - TGN
    - GraphMixer
    - DyGFormer
    - GeneralDyG
  graph_supervised:
    - Multi-GNN
    - FraudGT
```

---

# Recommended implementation order

```yaml
phase_1_low_effort:
  - ECOD
  - F-FADE

phase_2_recent_unsupervised:
  - SLADE
  - GeneralDyG

phase_3_temporal_graph:
  - TGN
  - GraphMixer
  - DyGFormer

phase_4_financial_graph_supervised:
  - Multi-GNN
  - FraudGT

phase_5_deep_tabular:
  - Deep-SVDD
```

A practical first submission-ready subset is:

```yaml
minimal_new_baselines:
  tabular_unsupervised:
    - ECOD
    - Deep-SVDD

  link_stream_unsupervised:
    - F-FADE
    - SLADE
    - GeneralDyG

  temporal_graph:
    - TGN
    - GraphMixer
    - DyGFormer

  transaction_graph_supervised:
    - Multi-GNN
    - FraudGT
```

Together with the existing methods:

```yaml
existing:
  - IF
  - DIF
  - MAD

total_baseline_pool:
  - IF
  - DIF
  - MAD
  - ECOD
  - Deep-SVDD
  - F-FADE
  - SLADE
  - TGN
  - GraphMixer
  - DyGFormer
  - GeneralDyG
  - Multi-GNN
  - FraudGT
```

---

# Reproducibility rules

## Temporal split

Use chronological splits, for example:

```yaml
split:
  train: earliest_70_percent
  validation: following_15_percent
  test: latest_15_percent
```

Do not randomly shuffle transactions before splitting.

## Historical feature leakage

Every historical feature for transaction `e_t` must be calculated using events strictly earlier than `t`.

Invalid:

```python
df["sender_total_count"] = df.groupby("sender")["sender"].transform("count")
```

Valid:

```python
df = df.sort_values("timestamp")
df["sender_count_before"] = df.groupby("sender").cumcount()
df["pair_count_before"] = df.groupby(["sender", "receiver"]).cumcount()
```

## Labels

```yaml
unsupervised_training:
  labels_allowed: false
  labels_allowed_for:
    - validation_threshold_if_explicitly_declared
    - final_evaluation

supervised_training:
  labels_allowed: true
```

## Metrics

Because fraud/AML labels are severely imbalanced, report:

```yaml
primary_metrics:
  - average_precision
  - precision_recall_auc
  - recall_at_fixed_false_positive_rate
  - precision_at_k

secondary_metrics:
  - roc_auc
  - f1_at_declared_threshold
```

Do not rely on accuracy as a primary metric.

## Random seeds

```yaml
seeds:
  recommended: [1, 2, 3, 4, 5]
report:
  - mean
  - standard_deviation
```

## Common output contract

Every model adapter should produce:

```python
{
    "transaction_id": np.ndarray,
    "timestamp": np.ndarray,
    "label": np.ndarray,
    "anomaly_score": np.ndarray,
    "prediction": np.ndarray | None,
}
```

---

# Suggested repository structure

```text
baselines/
├── tabular/
│   ├── ecod/
│   └── deep_svdd/
├── edge_stream/
│   ├── f_fade/
│   ├── slade/
│   └── generaldyg/
├── temporal_graph/
│   ├── tgn/
│   ├── graphmixer/
│   └── dygformer/
├── transaction_graph/
│   ├── multi_gnn/
│   └── fraudgt/
├── adapters/
│   ├── elliptic_pp.py
│   ├── ibm_aml.py
│   ├── midas_darpa.py
│   └── paysim.py
├── common/
│   ├── schema.py
│   ├── temporal_split.py
│   ├── metrics.py
│   └── result_writer.py
└── configs/
    ├── elliptic_pp.yaml
    ├── ibm_aml.yaml
    ├── midas_darpa.yaml
    └── paysim.yaml
```

---

# Agent execution checklist

```yaml
task: implement_and_run_baseline

required_steps:
  - verify_repository_license
  - pin_commit_hash
  - create_isolated_environment
  - record_python_pytorch_cuda_versions
  - inspect_original_task_definition
  - implement_dataset_adapter_without_modifying_core_model
  - enforce_chronological_split
  - prevent_future_information_leakage
  - save_raw_transaction_level_scores
  - evaluate_with_shared_metric_code
  - repeat_over_fixed_random_seeds
  - record_failures_and_incompatibilities

required_artifacts:
  - environment.yml_or_requirements.txt
  - config.yaml
  - commit_hash.txt
  - train.log
  - predictions.csv
  - metrics.json
  - runtime.json
  - README.md
```

---

# Important exclusions

The following are intentionally not included in the ten-method list:

```yaml
excluded:
  llm_based_methods:
    reason: hardware_and_scope_constraint

  graph_feature_preprocessor:
    reason: paper_and_commercial_software_are_documented_but_a_clear_public_research_repository_was_not_verified

  methods_without_public_code:
    reason: insufficient_reproducibility

  static_node_anomaly_methods:
    reason: primary_label_is_transaction_edge_level

  generic_time_series_methods:
    reason: do_not_naturally_model_sender_receiver_relationships
```

---

# Reference index

```yaml
repositories:
  ECOD_PyOD: https://github.com/yzhao062/pyod
  Deep_SVDD: https://github.com/lukasruff/Deep-SVDD-PyTorch
  F_FADE: https://github.com/snap-stanford/F-FADE
  SLADE: https://github.com/jhsk777/SLADE
  TGN: https://github.com/twitter-research/tgn
  DyGLib: https://github.com/yule-buaa/dyglib
  DyGLib_TGB: https://github.com/yule-buaa/DyGLib_TGB
  CTDG_Link_Anomaly: https://github.com/timpostuvan/TGB-link-anomaly-detection
  GeneralDyG: https://github.com/YXNTU/GeneralDyG
  Multi_GNN: https://github.com/IBM/Multi-GNN
  FraudGT: https://github.com/junhongmit/FraudGT
```
