"""Extended Isolation Forest baseline runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "script"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from script.common.metrics import compute_metrics
from script.common.results import run_dir, save_scores, write_json
from script.common.seeds import set_seed
from script.common.timing import Timer
from script.common.validation import tabular_interface_check
from script.tabular_pipeline.datasets import load_tabular_dataset
from script.model.EIF.eiforest import EIFAnomalyDetector

MODEL = "EIF"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended Isolation Forest baseline")
    p.add_argument("--dataset", required=True, choices=["midas_darpa", "ibm_hi_small", "ibm_li_small", "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological", choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--tree-max-samples", type=int, default=256)
    p.add_argument("--extension-level", type=int, default=-1,
                   help="-1 means fully extended (n_features - 1)")
    return p.parse_args()


def _splits(n: int):
    train = int(n * 0.7)
    return {"train": (0, train), "test": (train, n)}


def main() -> None:
    args = _args()
    out_dir = run_dir(MODEL, args.dataset, args.protocol, args.seed)
    if out_dir.exists() and not args.overwrite:
        print(f"skip (exists, no --overwrite): {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    data = load_tabular_dataset(args.dataset, split="all")
    if args.max_samples is not None:
        n = min(args.max_samples, data.X.shape[0])
        for name in ("X", "y", "label_available", "sample_ids", "timestamps"):
            setattr(data, name, getattr(data, name)[:n])
    splits = _splits(len(data.X))
    fit_timer = Timer()
    with fit_timer:
        model = EIFAnomalyDetector(args.n_estimators, args.tree_max_samples,
                                   args.extension_level, args.seed, args.batch_size)
        model.fit(data.X[:splits["train"][1]])
    score_timer = Timer()
    with score_timer:
        scores = model.score(data.X)
    metrics, warnings = {}, []
    for name, (start, end) in splits.items():
        metric = compute_metrics(data.y[start:end], scores[start:end])
        warnings.extend(metric.pop("warnings", []))
        metrics[name] = metric
    throughput = len(scores) / max(score_timer.seconds, 1e-9)
    metrics.update(status="ok", paper_fidelity="official_eif_extension", timing={"fit_seconds": round(fit_timer.seconds, 6), "score_seconds": round(score_timer.seconds, 6), "throughput": round(throughput, 3)})
    config = {"model": MODEL, "dataset": args.dataset, "protocol": args.protocol, "seed": args.seed,
              "n_estimators": args.n_estimators, "tree_max_samples": args.tree_max_samples,
              "extension_level": model.extension_level_, "batch_size": args.batch_size,
              "max_samples_used": args.max_samples, "is_quick_run": bool(args.quick),
              "score_direction": "larger=more anomalous", "learning": "unsupervised; labels used only for evaluation",
              "paper_fidelity": "official_eif_extension", "source": "source code/eif (Hariri et al., TKDE 2020)"}
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "timing.json", {"fit_seconds": fit_timer.seconds, "score_seconds": score_timer.seconds, "throughput": throughput})
    write_json(out_dir / "warnings.json", {"warnings": warnings})
    split = np.empty(len(scores), dtype=object)
    for name, (start, end) in splits.items():
        split[start:end] = name
    save_scores(out_dir / "scores.parquet", pl.DataFrame({"sample_id": data.sample_ids, "timestamp_or_step": data.timestamps, "label": data.y, "label_available": data.label_available, "anomaly_score": scores, "split": split}))
    write_json(out_dir / "interface_check.json", tabular_interface_check(data.X, data.y, data.label_available, data.sample_ids, data.feature_names, scores, fitted_on="train"))
    (out_dir / "run.log").write_text(json.dumps({"status": "ok", "test_roc_auc": metrics["test"].get("roc_auc"), "test_pr_auc": metrics["test"].get("pr_auc")}, indent=2) + "\n", encoding="utf-8")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
