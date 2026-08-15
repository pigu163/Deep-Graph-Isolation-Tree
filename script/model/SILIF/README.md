# SILIF — Silhouette-Augmented Isolation Forest

Tabular baseline adapted from `source code/silif-anomaly-detection`. It trains a 100-tree Isolation Forest, represents each row by its per-tree path-depth fingerprint, clusters fingerprints into 8 structural groups, and adds a centroid-proxy silhouette anomaly term (`alpha=0.5`) to the IF score.

For chronological evaluation, IF, fingerprint scaling, K-means and score normalisation fit on the train split only; this is a documented no-future-leakage adaptation of the source script, which fits and scores one whole table. CPU only.
