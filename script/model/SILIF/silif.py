"""Temporal-safe adaptation of the source SilIF tabular algorithm."""
from __future__ import annotations
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _node_depths(tree):
    depths = np.zeros(tree.node_count, dtype=np.float32)
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        depths[node] = depth
        left, right = tree.children_left[node], tree.children_right[node]
        if left != right:
            stack.append((left, depth + 1)); stack.append((right, depth + 1))
    return depths


class SILIFDetector:
    """Source SilIF: IF score + alpha times structural-silhouette anomaly."""
    def __init__(self, n_estimators=100, n_clusters=8, alpha=0.5,
                 max_samples='auto', random_state=42, n_jobs=-1):
        self.n_estimators = int(n_estimators); self.n_clusters = int(n_clusters)
        self.alpha = float(alpha); self.max_samples = max_samples
        self.random_state = int(random_state); self.n_jobs = int(n_jobs)
        self.iforest = self.fp_scaler = self.kmeans = None
        self.base_mean = self.base_std = self.sil_mean = self.sil_std = None
        self.profile_ = {}

    def _fingerprints(self, X):
        fp = np.empty((len(X), len(self.iforest.estimators_)), dtype=np.float32)
        for i, est in enumerate(self.iforest.estimators_):
            fp[:, i] = _node_depths(est.tree_)[est.apply(X)]
        return fp

    @staticmethod
    def _silhouette_anomaly(Z, labels, centers):
        # Source code uses centroid-proxy silhouette; vectorised equivalent.
        d = np.sqrt(np.maximum(((Z[:, None, :] - centers[None, :, :]) ** 2).sum(2), 0.0))
        own = d[np.arange(len(Z)), labels]
        d[np.arange(len(Z)), labels] = np.inf
        other = d.min(axis=1)
        denom = np.maximum(own, other)
        sil = np.divide(other - own, denom, out=np.zeros_like(own), where=denom > 1e-12)
        return 1.0 - sil

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        if len(X) < 2: raise ValueError('SilIF needs at least two training rows')
        t = time.perf_counter()
        self.iforest = IsolationForest(n_estimators=self.n_estimators,
            max_samples=self.max_samples, contamination='auto',
            random_state=self.random_state, n_jobs=self.n_jobs).fit(X)
        self.profile_['fit_iforest_seconds'] = time.perf_counter() - t
        t = time.perf_counter(); fp = self._fingerprints(X)
        self.fp_scaler = StandardScaler().fit(fp); Z = self.fp_scaler.transform(fp)
        k = min(self.n_clusters, len(X))
        cls = MiniBatchKMeans if len(X) > 50_000 else KMeans
        kwargs = dict(n_clusters=k, random_state=self.random_state, n_init=3)
        if cls is MiniBatchKMeans: kwargs['batch_size'] = 4096
        self.kmeans = cls(**kwargs).fit(Z)
        self.profile_['fit_fingerprint_and_cluster_seconds'] = time.perf_counter() - t
        base = -self.iforest.score_samples(X)
        sil = self._silhouette_anomaly(Z, self.kmeans.labels_, self.kmeans.cluster_centers_)
        self.base_mean, self.base_std = float(base.mean()), float(base.std() + 1e-9)
        self.sil_mean, self.sil_std = float(sil.mean()), float(sil.std() + 1e-9)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float32); t = time.perf_counter()
        base = -self.iforest.score_samples(X); fp = self._fingerprints(X)
        Z = self.fp_scaler.transform(fp); labels = self.kmeans.predict(Z)
        sil = self._silhouette_anomaly(Z, labels, self.kmeans.cluster_centers_)
        self.profile_['score_seconds_internal'] = time.perf_counter() - t
        return (base-self.base_mean)/self.base_std + self.alpha*(sil-self.sil_mean)/self.sil_std

    def config(self):
        return {'n_estimators':self.n_estimators,'n_structure_clusters':self.n_clusters,
          'alpha':self.alpha,'max_samples':self.max_samples,'n_jobs':self.n_jobs,
          'scoring':'IF z-score + alpha * centroid-proxy silhouette anomaly',
          'source_fidelity':'core algorithm reproduced; chronological train-only fitting adaptation'}
