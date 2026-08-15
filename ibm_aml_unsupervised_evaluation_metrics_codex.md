# Evaluation Metrics for Unsupervised AML Anomaly Detection on IBM AML

## Purpose

This document summarizes the evaluation logic, metric choices, empirical observations, and research implications discussed for unsupervised anti-money-laundering (AML) anomaly detection on the IBM Transactions for Anti-Money Laundering dataset.

It is designed to be **highly readable by coding agents such as Codex** and can be used as:

- an implementation specification for evaluation code;
- a guide for reproducing existing IBM AML baselines;
- a metric-selection policy for unsupervised transaction anomaly detection;
- a checklist for interpreting low F1 / precision / recall values correctly;
- a guardrail against misleading evaluation caused by arbitrary thresholds or extreme class imbalance.

---

# 1. Core Problem Setting

The target setting is unsupervised anomaly detection for AML.

A typical model receives a transaction or graph-derived representation and produces a continuous anomaly score:

```text
input transaction / node / graph representation
        |
        v
unsupervised detector
        |
        v
continuous anomaly score
        |
        v
ranking from most suspicious to least suspicious
```

Examples include:

```text
raw transaction features -> Isolation Forest
graph features -> Isolation Forest
DeepWalk embedding -> Isolation Forest
Node2Vec embedding -> Isolation Forest
GARG-AML structural features -> Isolation Forest
graph autoencoder -> reconstruction error
link-stream representation -> anomaly detector
```

The key point is:

> An unsupervised anomaly detector is naturally a **ranking model**, not necessarily a calibrated binary classifier.

Therefore, evaluation should not rely only on a single fixed binary threshold.

---

# 2. Why F1, Precision, and Recall Can Be Extremely Low

## 2.1 Extreme class imbalance

AML datasets can contain far below 1% illicit observations.

Example:

```text
Total transactions:       1,000,000
Laundering transactions:      2,000
Prevalence:                    0.2%
```

Even a model with useful ranking ability can have low absolute precision because the positive class is extremely rare.

If a model selects 10,000 suspicious transactions and 500 are truly illicit:

```text
Precision = 500 / 10,000 = 5%
Recall    = 500 / 2,000  = 25%
F1        = 8.33%
```

An F1 of roughly 0.083 looks small, but operationally the model has found 25% of all illicit transactions by reviewing only 1% of all transactions.

This distinction is essential.

---

## 2.2 A good ranking does not imply a good fixed threshold

A model can have:

```text
ROC-AUC = 0.82
F1      ~= 0
```

without contradiction.

ROC-AUC evaluates ranking over all possible thresholds.

F1 evaluates binary predictions at one specific operating point.

If a score threshold is poorly chosen, for example:

```text
score >= 0.5 -> anomaly
```

but almost all anomaly scores are below 0.5, then:

```text
almost everything -> normal
Recall -> approximately 0
F1     -> approximately 0
```

The ranking can still be informative.

This was explicitly observed in GARG-AML, where the authors note that precision and F1 can be close to zero because almost no observations cross the chosen 0.5 threshold.

---

## 2.3 Unsupervised scores are usually not calibrated probabilities

A score such as:

```text
0.43
0.61
-0.18
2.17
```

does not necessarily mean:

```text
43% probability of laundering
61% probability of laundering
```

For Isolation Forest specifically, `score_samples()` produces an anomaly-related score whose scale is not a calibrated probability.

Therefore, a universal threshold such as `0.5` usually has no intrinsic statistical meaning unless the method explicitly defines one.

---

## 2.4 AML anomalies are not necessarily generic statistical outliers

Isolation Forest is based on the intuition that anomalies:

- occur in sparse regions;
- have unusual feature values;
- can be isolated with fewer random partitions.

But money laundering can be intentionally designed to resemble legitimate behavior.

Therefore:

```text
statistically unusual != necessarily laundering
laundering != necessarily statistically unusual
```

This explains why supervised models can dramatically outperform unsupervised anomaly detectors when labels are available.

A supervised model such as XGBoost learns features that separate known laundering cases from normal transactions.

An unsupervised detector only learns what appears unusual without knowing which unusual behaviors correspond to AML.

---

# 3. Metrics: Definitions and Interpretation

## 3.1 Precision

```text
Precision = TP / (TP + FP)
```

Interpretation:

> Among the transactions that the model flags, what fraction are actually laundering?

Example:

```text
Model flags:       1,000
True laundering:     100

Precision = 100 / 1,000 = 10%
```

For AML investigation, precision measures analyst workload efficiency.

Higher precision means fewer false alerts for each true illicit transaction found.

---

## 3.2 Recall / True Positive Rate

```text
Recall = TP / (TP + FN)
```

Interpretation:

> Among all true laundering transactions, what fraction did the model retrieve?

Example:

```text
True laundering total: 2,000
Found by model:           500

Recall = 500 / 2,000 = 25%
```

Recall measures coverage.

---

## 3.3 F1 score

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1 balances precision and recall.

However, in unsupervised AML it has two important limitations:

1. it depends on the chosen binary threshold;
2. it can look numerically very low even when the ranking provides substantial operational lift.

Therefore:

> F1 should be reported, but should not be the only primary metric for highly imbalanced unsupervised AML detection.

---

# 4. ROC-AUC

## 4.1 Definition

ROC plots:

```text
x-axis: False Positive Rate
y-axis: True Positive Rate / Recall
```

across all possible score thresholds.

ROC-AUC is the area under this curve.

A useful probabilistic interpretation is:

> ROC-AUC is the probability that a randomly selected positive example receives a more suspicious score than a randomly selected negative example.

Typical interpretation:

```text
ROC-AUC = 1.0 -> perfect ranking
ROC-AUC = 0.5 -> random ranking
ROC-AUC < 0.5 -> ranking direction is effectively reversed
```

---

## 4.2 Why ROC-AUC is useful

It is threshold-independent.

Therefore it answers:

> Does the model generally rank laundering cases ahead of legitimate cases?

This is especially useful when anomaly scores are not calibrated.

---

## 4.3 Why ROC-AUC can be misleading in AML

AML datasets are extremely imbalanced.

Because the number of true negatives is enormous, the false-positive rate can remain numerically small even when the number of false positives is operationally huge.

For this reason:

> ROC-AUC should be reported, but it should not be the only main metric.

---

# 5. PR-AUC

PR-AUC is the area under the precision-recall curve.

It directly reflects the tradeoff between:

```text
Precision
Recall
```

across thresholds.

For highly imbalanced AML problems, PR-AUC is usually more informative than ROC-AUC because false positives directly reduce precision.

Important property:

```text
Random PR-AUC baseline ~= positive-class prevalence
```

Example:

```text
Laundering prevalence = 0.2%
Random PR-AUC baseline ~= 0.002
```

Thus a PR-AUC of 0.02 can already represent a large improvement over random ranking.

Recommended interpretation:

> For extremely imbalanced AML experiments, PR-AUC should usually be treated as one of the most important global ranking metrics.

---

# 6. Top-K / Top-Percentage Evaluation

Instead of choosing an arbitrary anomaly-score threshold, rank all observations by anomaly score.

Then define a review budget.

Example with 1,000,000 transactions:

```text
Top 0.1% ->  1,000 transactions
Top 0.5% ->  5,000 transactions
Top 1.0% -> 10,000 transactions
```

These percentages mean:

> the most suspicious X% of all observations according to the anomaly score.

They are **not anomaly-score thresholds**.

---

# 7. Precision@K

Example:

```text
Total transactions = 1,000,000
Top 1% reviewed     =    10,000
True laundering in top 1% = 500
```

Then:

```text
Precision@1% = 500 / 10,000 = 5%
```

Interpretation:

> Among the most suspicious 1% of transactions, 5% are truly laundering.

This metric directly reflects investigation efficiency.

---

# 8. Recall@K

Using the same example:

```text
Total true laundering = 2,000
Found in top 1%       =   500
```

Then:

```text
Recall@1% = 500 / 2,000 = 25%
```

Interpretation:

> By investigating only 1% of all transactions, the system captures 25% of all laundering transactions.

This is often much more operationally meaningful than a default binary threshold.

---

# 9. F1@K

F1@K uses Precision@K and Recall@K:

```text
F1@K =
2 * Precision@K * Recall@K
/
(Precision@K + Recall@K)
```

Example:

```text
Precision@1% = 5%
Recall@1%    = 25%

F1@1% ~= 8.33%
```

Again, the numerical F1 may appear small while the ranking can still have substantial operational value.

---

# 10. Lift@K

Lift measures how much better model-guided investigation is than random selection.

Definition:

```text
Lift@K = Precision@K / positive prevalence
```

Example:

```text
Positive prevalence = 0.2%
Precision@1%        = 5%

Lift@1% = 5% / 0.2% = 25
```

Interpretation:

> Investigating the model's top 1% is 25 times more efficient than randomly selecting the same number of transactions.

This is highly intuitive for AML.

Random selection of 10,000 transactions:

```text
Expected laundering = 10,000 * 0.2% = 20
```

Model-guided top 1%:

```text
Detected laundering = 500
```

Therefore:

```text
500 / 20 = 25x lift
```

Lift is particularly useful when raw F1 scores look small.

---

# 11. Expected Behavior Across Review Budgets

Typically:

```text
Top 0.1%
-> very selective
-> precision tends to be highest
-> recall tends to be low

Top 0.5%
-> larger investigation budget
-> precision may decrease
-> recall increases

Top 1%
-> still larger budget
-> recall increases further
-> precision usually decreases

Top 5%
-> much larger coverage
-> recall may be high
-> many more false positives
```

This tradeoff is not a flaw.

It represents different operational investigation budgets.

---

# 12. Recommended Metric Set for Unsupervised IBM AML Experiments

## 12.1 Primary metrics

Use:

```text
PR-AUC
ROC-AUC
```

Priority recommendation:

```text
PR-AUC > ROC-AUC
```

for judging performance under extreme class imbalance.

---

## 12.2 Operational ranking metrics

Report at multiple review budgets:

```text
Precision@0.1%
Recall@0.1%
F1@0.1%
Lift@0.1%

Precision@0.5%
Recall@0.5%
F1@0.5%
Lift@0.5%

Precision@1%
Recall@1%
F1@1%
Lift@1%
```

Optional:

```text
Precision@5%
Recall@5%
F1@5%
Lift@5%
```

These metrics should be computed from the same anomaly ranking.

---

# 13. Recommended Metric Priority

For highly imbalanced unsupervised AML:

```text
1. PR-AUC
2. Recall@K / Precision@K
3. Lift@K
4. F1@K
5. ROC-AUC
```

This does not mean ROC-AUC is unimportant.

It means:

> ROC-AUC should be interpreted together with metrics that explicitly reflect rare-positive retrieval and investigation workload.

---

# 14. What the Three Important IBM AML Papers Emphasize

## 14.1 Network Analytics for Anti-Money Laundering

Unsupervised Isolation Forest experiments report:

```text
ROC-AUC
PR-AUC
Precision
Recall
F1
```

For IBM AML, the paper also reports **top-1%** threshold-dependent metrics.

Reported IBM AML unsupervised results include approximately:

```text
Intrinsic features + IF
Precision@1% = 0.042
Recall@1%    = 0.156
F1@1%        = 0.067

Egonet features + IF
Precision@1% = 0.027
Recall@1%    = 0.098
F1@1%        = 0.042

DeepWalk + IF
Precision@1% = 0.055
Recall@1%    = 0.204
F1@1%        = 0.087

Node2Vec + IF
Precision@1% = 0.007
Recall@1%    = 0.025
F1@1%        = 0.010
```

Main lesson:

> The F1 values are low, but ranking metrics and top-budget retrieval still provide useful information.

DeepWalk + IF performs best among these unsupervised variants.

---

## 14.2 GARG-AML

Metrics include:

```text
Precision
F1
ROC-AUC
PR-AUC
```

The authors emphasize ROC-AUC and PR-AUC because fixed-threshold precision and F1 can collapse under severe imbalance.

A key observation is that a GARG-AML configuration can have relatively high ROC-AUC while F1 is close to zero.

The paper explicitly attributes near-zero F1 partly to the choice of a fixed threshold such as 0.5, where very few observations exceed the threshold.

Main lesson:

> Threshold-dependent metrics can look terrible even when ranking quality remains meaningful.

---

## 14.3 ReDiRect

ReDiRect focuses less on standard fixed-threshold F1.

Its evaluation emphasizes:

```text
TPR
TPR-AUC
top-X% anomalies
contextual anomaly evaluation
computational efficiency
investigation lead time
```

It evaluates review budgets such as:

```text
Top 0.1%
Top 0.2%
Top 0.5%
Top 1%
```

Main lesson:

> AML anomaly detection is naturally an investigation-prioritization problem.

The useful question is often not:

```text
Can the model perfectly classify every transaction?
```

but:

```text
Given a limited investigation budget,
how many true laundering cases can the model surface?
```

---

# 15. Important Interpretation of Low F1

Do not automatically conclude:

```text
F1 = 0.08 -> model is useless
```

Instead ask:

```text
What is the base prevalence?
What review budget is used?
What is Precision@K?
What is Recall@K?
What is Lift@K?
What is PR-AUC?
Does the model strongly improve over random ranking?
```

Example:

```text
Prevalence      = 0.2%
Precision@1%    = 5%
Recall@1%       = 25%
F1@1%           = 8.3%
Lift@1%         = 25x
```

The F1 is low, but the model may still be operationally useful because:

```text
review only 1% of transactions
-> retrieve 25% of all laundering
-> 25x higher hit rate than random review
```

---

# 16. When a Low-F1 Model Is Actually Not Useful

A ranking model should not be defended solely because ROC-AUC is above 0.5.

Example:

```text
ROC-AUC        = 0.82
PR-AUC         ~= random baseline
Precision@1%   ~= prevalence
Recall@1%      very low
Lift@1%        ~= 1
```

This would indicate:

> The ranking may look globally reasonable under ROC-AUC, but it provides little practical concentration of illicit cases at the top of the ranking.

Therefore:

```text
high ROC-AUC alone != useful AML detector
```

A useful unsupervised AML model should demonstrate substantial improvement in:

```text
PR-AUC
Precision@K
Recall@K
Lift@K
```

relative to random ranking and baseline methods.

---

# 17. Why PR-AUC Is Usually More Important Than ROC-AUC Here

Suppose:

```text
99.8% normal
0.2% laundering
```

ROC-AUC uses:

```text
FPR = FP / total negatives
```

Because total negatives are enormous, thousands of false positives can still correspond to a small FPR.

PR-AUC uses precision:

```text
Precision = TP / (TP + FP)
```

so false positives directly hurt the score.

Therefore PR-AUC more strongly reflects whether the model can concentrate rare positives near the top.

---

# 18. Recommended Evaluation Protocol for Codex

The implementation should follow this procedure.

## Step 1 — Produce anomaly scores

For each evaluation observation:

```text
y_true[i]   in {0,1}
score[i]    continuous anomaly score
```

Ensure:

```text
larger score = more anomalous
```

For scikit-learn Isolation Forest:

```python
anomaly_score = -model.score_samples(X)
```

because `score_samples()` assigns lower values to more abnormal points.

---

## Step 2 — Compute global ranking metrics

```python
roc_auc = roc_auc_score(y_true, anomaly_score)
pr_auc = average_precision_score(y_true, anomaly_score)
```

---

## Step 3 — Sort by anomaly score

```python
order = np.argsort(-anomaly_score)
```

---

## Step 4 — Evaluate multiple review budgets

Required:

```text
K = 0.1%
K = 0.5%
K = 1.0%
```

Optional:

```text
K = 5.0%
```

For each K:

```text
n_selected = ceil(N * K)
selected = highest-scoring n_selected observations
```

---

## Step 5 — Compute threshold-at-budget predictions

```text
predicted positive = selected observations
predicted negative = all remaining observations
```

Then compute:

```text
TP
FP
FN
TN

Precision@K
Recall@K
F1@K
Lift@K
```

---

## Step 6 — Compute random baseline

For each K:

```text
Expected Precision@K for random ranking = prevalence
Expected Lift@K for random ranking = 1
Expected Recall@K for random ranking ~= K
```

For example:

```text
K = 1%
random Recall@1% ~= 1%
```

This provides an intuitive baseline.

---

# 19. Suggested Output Schema

Coding agents should produce a machine-readable table such as:

```csv
method,roc_auc,pr_auc,precision_at_0.1,recall_at_0.1,f1_at_0.1,lift_at_0.1,precision_at_0.5,recall_at_0.5,f1_at_0.5,lift_at_0.5,precision_at_1,recall_at_1,f1_at_1,lift_at_1
```

Recommended JSON equivalent:

```json
{
  "method": "IsolationForest",
  "global": {
    "roc_auc": 0.0,
    "pr_auc": 0.0,
    "prevalence": 0.0
  },
  "top_k": {
    "0.001": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "lift": 0.0
    },
    "0.005": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "lift": 0.0
    },
    "0.01": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "lift": 0.0
    }
  }
}
```

---

# 20. Important Reproducibility Rules

## Rule 1 — Never tune an unsupervised threshold on the test labels

Do not choose the final threshold by maximizing test-set F1.

That leaks labels into evaluation.

Allowed approaches include:

```text
fixed investigation budget
validation-set threshold
known expected contamination
domain-defined review capacity
```

For unsupervised AML benchmarking, fixed Top-K budgets are especially clean.

---

## Rule 2 — Do not compare metrics across different evaluation units without saying so

Existing IBM AML papers often evaluate:

```text
node/account level
community level
```

whereas a new model may evaluate:

```text
transaction/edge level
```

These are different tasks.

Do not directly compare F1 values unless the evaluation unit is aligned.

---

## Rule 3 — Do not compare F1 values with different threshold policies as if they were equivalent

Examples:

```text
F1 using score >= 0.5
F1 using IsolationForest.predict()
F1@Top1%
F1 at oracle-optimal threshold
```

are not directly comparable.

Always report the threshold policy.

---

## Rule 4 — Report class prevalence

Every result table should include or clearly state:

```text
positive prevalence
number of positives
number of negatives
```

Without prevalence, PR-AUC, precision, and lift are difficult to interpret.

---

## Rule 5 — Report score direction

For every anomaly detector, explicitly document:

```text
higher score = more anomalous
```

or

```text
lower score = more anomalous
```

Normalize all models to one convention before metric computation.

---

# 21. Recommended Paper-Level Evaluation Table

A strong experimental table would contain:

| Method | ROC-AUC | PR-AUC | P@0.1% | R@0.1% | Lift@0.1% | P@0.5% | R@0.5% | Lift@0.5% | P@1% | R@1% | F1@1% | Lift@1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Recommended emphasis:

```text
PR-AUC
Recall@K
Precision@K
Lift@K
```

Use ROC-AUC for compatibility with prior work.

Use F1@K as a supplementary operating-point metric.

---

# 22. Recommended Visualization

Useful plots include:

```text
Precision-Recall curve
ROC curve
Recall vs review budget
Precision vs review budget
Lift vs review budget
cumulative gains curve
anomaly-score distributions by label
```

Especially useful:

```text
x-axis: fraction of transactions reviewed
y-axis: fraction of laundering transactions retrieved
```

This directly visualizes operational investigation efficiency.

---

# 23. Cumulative Gains Interpretation

If:

```text
review top 1% -> retrieve 25% of laundering cases
```

then the cumulative gains point is:

```text
(0.01, 0.25)
```

Random ranking would approximately follow:

```text
y = x
```

so random review of 1% would retrieve approximately 1% of positives.

Therefore:

```text
25% recall at 1% review budget
```

is a strong concentration effect even if F1 remains numerically small.

---

# 24. Deep Research Insight: AML Evaluation Is a Resource Allocation Problem

For real AML systems, analyst capacity is finite.

The practical optimization problem is often:

```text
Given that investigators can inspect only K transactions,
which K should be selected?
```

This makes ranking quality central.

Therefore, anomaly detection should be framed as:

```text
risk prioritization
```

rather than only:

```text
binary classification
```

This framing explains why Top-K metrics and lift are operationally natural.

---

# 25. Deep Research Insight: Supervised and Unsupervised Models Solve Different Problems

Supervised XGBoost learns:

```text
Which feature combinations predict known laundering labels?
```

Isolation Forest learns:

```text
Which observations look statistically unusual?
```

These objectives are fundamentally different.

Therefore, a large performance gap between supervised XGBoost and unsupervised IF is expected and does not by itself indicate an implementation bug.

---

# 26. Deep Research Insight: High AUC With Low F1 Is Not a Paradox

This combination usually indicates:

```text
ranking is somewhat informative
+
binary threshold is poorly aligned with the score distribution
```

or:

```text
positives are concentrated near the top
+
class imbalance keeps absolute precision low
```

To diagnose, inspect:

```text
PR-AUC
score histograms
Precision@K
Recall@K
Lift@K
```

before concluding that the model is useless.

---

# 27. Deep Research Insight: Threshold Choice Is Part of the Evaluation Protocol

In unsupervised anomaly detection, the threshold is not merely a technical implementation detail.

It defines the operational budget.

Examples:

```text
Top 0.1% -> very conservative investigation budget
Top 0.5% -> moderate budget
Top 1%   -> larger budget
```

Therefore a paper should explicitly define:

```text
score generation
ranking direction
budget selection
binary conversion rule
metric computation
```

---

# 28. Deep Research Insight: Lift Can Make Low F1 Interpretable

F1 values such as:

```text
0.05
0.08
0.10
```

may look weak in conventional classification.

But if prevalence is:

```text
0.2%
```

and Precision@1% is:

```text
5%
```

then:

```text
Lift@1% = 25x
```

This reveals a substantial concentration of risk.

Therefore lift is a useful complementary measure when presenting unsupervised AML results.

---

# 29. Deep Research Insight: Use Multiple Budgets, Not One Arbitrary K

Do not report only:

```text
Top 1%
```

Recommended:

```text
0.1%
0.5%
1%
```

because different models may dominate at different review capacities.

One method may be excellent in the extreme top 0.1% but degrade quickly.

Another may have lower precision at 0.1% but much stronger recall by 1%.

This matters operationally.

---

# 30. Final Recommended Experimental Policy

For every unsupervised AML model:

```text
1. Generate one continuous anomaly score per evaluation item.

2. Ensure:
   higher score = more anomalous.

3. Report:
   - ROC-AUC
   - PR-AUC
   - positive prevalence

4. Rank all observations by anomaly score.

5. Evaluate:
   - Precision@0.1%
   - Recall@0.1%
   - F1@0.1%
   - Lift@0.1%

   - Precision@0.5%
   - Recall@0.5%
   - F1@0.5%
   - Lift@0.5%

   - Precision@1%
   - Recall@1%
   - F1@1%
   - Lift@1%

6. Optionally report @5%.

7. Compare against:
   - random ranking
   - vanilla Isolation Forest
   - other unsupervised baselines

8. Never choose an oracle test-set threshold.

9. Clearly state:
   - task unit: transaction / edge / node / community
   - threshold policy
   - prevalence
   - score direction

10. Interpret low F1 together with:
    - PR-AUC
    - Recall@K
    - Precision@K
    - Lift@K
```

---

# 31. Short Codex Implementation Checklist

```text
[ ] Load y_true and anomaly_score.
[ ] Convert score direction so larger = more anomalous.
[ ] Validate y_true is binary.
[ ] Calculate positive prevalence.
[ ] Calculate ROC-AUC.
[ ] Calculate PR-AUC / Average Precision.
[ ] Sort descending by anomaly score.
[ ] Evaluate top 0.1%.
[ ] Evaluate top 0.5%.
[ ] Evaluate top 1%.
[ ] Compute TP / FP / FN / TN for each budget.
[ ] Compute Precision@K.
[ ] Compute Recall@K.
[ ] Compute F1@K.
[ ] Compute Lift@K = Precision@K / prevalence.
[ ] Save results in CSV/JSON.
[ ] Record threshold policy.
[ ] Record evaluation unit.
[ ] Record random baseline.
[ ] Do not tune on test labels.
```

---

# 32. Bottom Line

For unsupervised AML detection on IBM AML:

> Low F1 does not automatically imply that the anomaly ranking is useless.

The decisive questions are:

```text
Does the model rank illicit cases toward the top?

Does PR-AUC substantially exceed prevalence?

How many illicit cases are retrieved when only 0.1%, 0.5%, or 1% of observations can be reviewed?

How much better is this than random review?

What is the Lift@K?
```

A strong evaluation should therefore emphasize:

```text
PR-AUC
ROC-AUC
Precision@K
Recall@K
F1@K
Lift@K
```

with particular attention to:

```text
PR-AUC
Recall@K
Precision@K
Lift@K
```

for highly imbalanced AML anomaly detection.

The goal is not merely to produce a binary prediction.

The goal is to produce a risk ranking that efficiently concentrates true laundering activity into the limited set of observations that investigators can realistically review.
