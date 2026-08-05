"""IF runner CLI.

Usage (task doc Sec. 12):

    /home/zhu/.conda/envs/IF/bin/python script/model/IF/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--num-workers N] [--batch-size N] [--max-samples N] [--quick] [--device cpu]

Outputs (task doc Sec. 13):

    result/IF/<DATASET>/<PROTOCOL>/seed_<SEED>/
        config.json metrics.json scores.parquet timing.json warnings.json run.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "script"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from script.common.metrics import compute_metrics  # noqa: E402
from script.common.results import run_dir, save_scores, write_json, write_text  # noqa: E402
from script.common.seeds import set_seed  # noqa: E402
from script.common.timing import Timer  # noqa: E402
from script.common.validation import tabular_interface_check  # noqa: E402
from script.tabular_pipeline.datasets import (  # noqa: E402
    load_tabular_dataset,
    load_tabular_splits,
)

MODEL = "IF"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolation Forest baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--max-samples", type=int, default=None,
                   help="restrict scoring to the first N samples (smoke tests)")
    p.add_argument("--quick", action="store_true", help="smoke run flag")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--tree-max-samples", type=int, default=256)
    return p.parse_args()


def _split_indices(n: int):
    """60/20/20 chronological split indices on the scored array."""
    t = int(round(n * 0.6))
    v = int(round(n * 0.2))
    return {"train": (0, t), "val": (t, t + v), "test": (t + v, n)}


def main() -> None:
    args = _parse_args()
    out_dir = run_dir(MODEL, args.dataset, args.protocol, args.seed)
    if out_dir.exists() and not args.overwrite:
        print(f"skip (exists, no --overwrite): {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"IF runner | dataset={args.dataset} protocol={args.protocol} "
                  f"seed={args.seed}\n")
        log.flush()

        set_seed(args.seed)
        data = load_tabular_dataset(args.dataset, split="all")
        log.write(f"loaded tabular data: {data.X.shape}\n"); log.flush()

        if args.max_samples is not None:
            n = min(args.max_samples, data.X.shape[0])
            data.X = data.X[:n]
            data.y = data.y[:n]
            data.label_available = data.label_available[:n]
            data.sample_ids = data.sample_ids[:n]
            data.timestamps = data.timestamps[:n]
        n_total = data.X.shape[0]
        splits = _split_indices(n_total)

        # ---- fit on train only -------------------------------------------
        from script.model.IF.isolation_forest import IFAnomalyDetector

        fit_timer = Timer()
        with fit_timer:
            model = IFAnomalyDetector(
                n_estimators=args.n_estimators,
                max_samples=args.tree_max_samples,
                contamination="auto",
                random_state=args.seed,
                n_jobs=-1,
                batch_size=args.batch_size,
            )
            model.fit(data.X[splits["train"][0]:splits["train"][1]])
        log.write(f"fit done in {fit_timer.seconds:.2f}s\n"); log.flush()

        # ---- score every split -------------------------------------------
        score_timer = Timer()
        with score_timer:
            scores = model.score(data.X)
        throughput = n_total / max(score_timer.seconds, 1e-9)

        # ---- metrics per split --------------------------------------------
        warnings_all: list[str] = []
        metrics = {}
        for name, (s, e) in splits.items():
            m = compute_metrics(data.y[s:e], scores[s:e])
            warnings_all.extend(m.pop("warnings", []))
            metrics[name] = m
        metrics["status"] = "ok"
        metrics["paper_fidelity"] = "baseline"
        metrics["timing"] = {
            "fit_seconds": round(fit_timer.seconds, 6),
            "score_seconds": round(score_timer.seconds, 6),
            "throughput": round(throughput, 3),
        }

        config = {
            "model": MODEL,
            "dataset": args.dataset,
            "protocol": args.protocol,
            "seed": args.seed,
            "device": args.device,
            "n_estimators": args.n_estimators,
            "tree_max_samples": args.tree_max_samples,
            "contamination": "auto",
            "n_jobs": -1,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous",
            "paper_fidelity": "baseline",
        }
        write_json(out_dir / "config.json", config)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "timing.json", {
            "fit_seconds": fit_timer.seconds,
            "score_seconds": score_timer.seconds,
            "throughput": throughput,
        })
        write_json(out_dir / "warnings.json", {"warnings": warnings_all})

        # ---- scores.parquet ----------------------------------------------
        split_arr = np.empty(n_total, dtype=object)
        for name, (s, e) in splits.items():
            split_arr[s:e] = name
        df = pl.DataFrame({
            "sample_id": data.sample_ids,
            "timestamp_or_step": data.timestamps,
            "label": data.y,
            "label_available": data.label_available,
            "anomaly_score": scores,
            "split": split_arr,
        })
        save_scores(out_dir / "scores.parquet", df)

        # ---- data-interface check (task Sec. 9.1 / 14) -------------------
        write_json(out_dir / "interface_check.json", tabular_interface_check(
            data.X, data.y, data.label_available, data.sample_ids,
            data.feature_names, scores, fitted_on="train"))

        log.write(json.dumps({
            "status": "ok",
            "test_roc_auc": metrics["test"].get("roc_auc"),
            "test_pr_auc": metrics["test"].get("pr_auc"),
            "test_n_eval": metrics["test"].get("n_eval"),
        }, indent=2))
        log.write("\n")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
