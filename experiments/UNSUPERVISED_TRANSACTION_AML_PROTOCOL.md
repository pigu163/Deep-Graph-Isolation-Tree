# Unsupervised Transaction-Level AML/Fraud Experiment Protocol

## Material Passport

- **Origin skill:** ARS experiment-agent (run mode)
- **Version:** `unsupervised_txn_aml_v1`
- **Status:** executable protocol; results are not yet available
- **Execution boundary:** New code may be written only in `experiments/`. Do **not** create, edit, format, or delete files under `script/`. All generated artifacts belong under `experiments_result/`.

## Objective and scope

Evaluate label-free transaction anomaly ranking on the three canonical datasets:

| Dataset | Detection label | Unit | Train/test protocol |
|---|---|---|---|
| `ibm_hi_small` | `Is Laundering` | directed transaction event | chronological 70% / 30% |
| `ibm_li_small` | `Is Laundering` | directed transaction event | chronological 70% / 30% |
| `paysim` | `isFraud` | directed transaction event | chronological 70% / 30% |

The detector receives no AML/fraud labels while fitting, sampling local history,
calibrating scores, or choosing a configuration. Labels are accessed only after
test scoring by the reporting program.

This is a transaction-level task. It is not account classification, node
classification, community detection, or link prediction.

## Read-only input contract

`data/processed/<dataset>` and `data/processed_tabular/<dataset>` must have
identical chronological `train=[0,floor(0.7N))` and `test=[floor(0.7N),N)`
ranges. The event `sample_id` is the canonical `event_id`.

- Tabular baselines consume the existing train-fitted tabular view. Its encoders
  and scalers are fitted on train only; its temporal statistics are past-only.
- Link-stream methods consume canonical directed events. A history query may use
  earlier unlabeled test events online, but never the seed, a future event, or
  another event sharing the seed timestamp.
- The proposed model is the active `LocalLinkStreamESNIF` implementation and
  `TRANSACTION_LEVEL_LOCAL_LINKSTREAM_ESN_FOREST_SPEC.md`, not the legacy
  unconnected motif-conditioned design note.

The experiment driver rejects `--quick`, `--max-samples`, and `transductive`.
Old `result/` artifacts are never read. Historical 60/20/20 wording in old
README files is documentation debt, not the current protocol.

## Main benchmark

Run each `(method, dataset, seed)` once for seeds `42`, `43`, and `44`.
Configurations are current runner defaults or the pre-declared proposed-model
specification; there is no Optuna, validation split, label-based threshold, or
feature selection.

| View | Methods |
|---|---|
| Tabular, unsupervised | IF, EIF, DIF, ECOD |
| Link stream, unsupervised/self-supervised | FFADE, MAD, SLADE |
| Proposed link stream | LocalLinkStreamESNIF |

SILIF remains available in the repository but is intentionally excluded from
the primary table: it is an IF derivative and adds less coverage than EIF,
DIF, and ECOD in a nine-page paper. Supervised XGBoost and other labelled
upper bounds are excluded altogether.

Use the following command sequence from repository root:

```bash
PY=/home/zhu/.conda/envs/IF/bin/python
$PY experiments/run_unsupervised_aml.py preflight
$PY experiments/run_unsupervised_aml.py main --workers 64
$PY experiments/run_unsupervised_aml.py ablation --workers 64
$PY experiments/summarize_unsupervised_aml.py
```

`run_unsupervised_aml.py` sets `ACM_RESULT_ROOT` to
`experiments_result/unsupervised_txn_aml_v1/runs`, so unchanged runners write
only beneath `experiments_result/`. It stops at the first failed subprocess and
does not retry a failed model automatically. Add `--overwrite` only to replace
an existing dedicated experiment run.

## Compact proposed-model ablation

Run only on IBM HI and LI, still with the three fixed seeds and exactly 300
trees. The implementation changes only the pre-declared ESN split-family mix:

| Variant | Departure | Surprise | Recurrence |
|---|---:|---:|---:|
| `D` | 300 | 0 | 0 |
| `DS` | 150 | 150 | 0 |
| `DSR` (main model) | 100 | 100 | 100 |

All variants keep the same local 7-day, timestamp-strict context sampler,
frozen ESN weights, 64-seed tree population, normalization policy, and total
tree budget. The ablation is descriptive: no variant is selected using a
label-derived score.

## Metrics and reporting

The shared scoring convention is larger score = more suspicious.

- **Primary:** test PR-AUC and Recall@0.5% review budget, reported as mean ±
  standard deviation over seeds.
- **Secondary:** ROC-AUC; Precision@K, Recall@K, F1@K where `K` equals the
  known positive count; Precision, Recall, and Lift at review budgets 0.1%,
  0.5%, and 1.0%.
- **Efficiency:** fit seconds, score seconds, and score throughput. Compare
  only runs recorded on the same CPU environment.
- **Proposed-model coverage:** empty-context rate and median context size.

`summarize_unsupervised_aml.py` independently re-computes metrics from the
stored test scores, validates schema and labels, writes CSV/JSON tables, and
draws one PR panel per dataset. The panels contain the proposed model, the
best tabular model, the best link-stream baseline, and the prevalence line.
“Best” is a post-hoc display choice only; it never alters a fitted model.

## Required result layout

```text
experiments_result/unsupervised_txn_aml_v1/
  preflight.json
  environment.json
  manifest.json
  runs/<MODEL>/<DATASET>/chronological/seed_<SEED>/...
  ablations/LocalLinkStreamESNIF_<VARIANT>/<DATASET>/chronological/seed_<SEED>/...
  summary/validation_report.json
  summary/metrics_long.csv
  summary/main_table.csv
  summary/ablation_table.csv
  summary/summary.json
  summary/pr_curves.png
```

The summarizer fails rather than silently aggregating if a score file has a
wrong split, missing event IDs, non-finite scores, quick/subsampled metadata,
or an unexpected test label vector.

## Literature rationale

- **AMLWorld / Realistic Synthetic Financial Transactions** motivates HI/LI
  transaction-edge evaluation, complete synthetic labels, and avoiding account
  ID memorization.
- **Network Analytics systematic review** motivates the unsupervised setting
  under scarce labels and imbalance-aware ranking evaluation.
- **MAD** supplies a direct edge-event/link-stream unsupervised comparator.
- **Anti-Money Laundering in Bitcoin** is intentionally not a baseline because
  it uses transactions as graph nodes and performs supervised classification.
- **2026 mobile AML review** motivates temporal and relational context under
  high volume and scarce confirmed cases; it does not justify adding labelled
  deep baselines here.

For a nine-page double-column paper, use one data/protocol paragraph, the
main table, the three-row ablation table, a three-panel PR figure, and a small
efficiency table. Do not add a hyperparameter sweep merely to fill space.
