"""ECOD runner CLI (tabular, unsupervised).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/ECOD/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--max-samples N] [--quick] [--device cpu]

Outputs under result/ECOD/<dataset>/<protocol>/seed_<seed>/:
    config.json metrics.json scores.parquet timing.json warnings.json run.log

Input: the shared tabular view (script/tabular_pipeline), same as IF/DIF.
Score: PyOD ECOD decision_function, larger = more anomalous.
Fitting uses train rows only; labels are not used by the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from script.common.metrics import compute_metrics  # noqa: E402
from script.common.results import run_dir, save_scores, write_json  # noqa: E402
from script.common.seeds import set_seed  # noqa: E402
from script.common.timing import Timer  # noqa: E402
from script.common.validation import tabular_interface_check  # noqa: E402
from script.tabular_pipeline.datasets import load_tabular_dataset  # noqa: E402
from script.model.ECOD.ecod import ECODDetector  # noqa: E402

MODEL = "ECOD"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECOD baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=200_000)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--contamination", type=float, default=0.1)
    return p.parse_args()


def _split_indices(n: int):
    t = int(n * 0.7)
    return {"train": (0, t), "test": (t, n)}


def main() -> None:
    args = _parse_args()
    out_dir = run_dir(MODEL, args.dataset, args.protocol, args.seed)
    if out_dir.exists() and not args.overwrite:
        print(f"skip (exists, no --overwrite): {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"ECOD runner | dataset={args.dataset} protocol={args.protocol} "
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

        fit_timer = Timer()
        with fit_timer:
            model = ECODDetector(contamination=args.contamination,
                                 n_jobs=-1, seed=args.seed)
            model.fit(data.X[splits["train"][0]:splits["train"][1]])
        log.write(f"fit done in {fit_timer.seconds:.2f}s\n"); log.flush()

        score_timer = Timer()
        with score_timer:
            scores = model.score(data.X, batch_size=args.batch_size)
        throughput = n_total / max(score_timer.seconds, 1e-9)

        warnings_all: list[str] = []
        metrics = {}
        for name, (s, e) in splits.items():
            m = compute_metrics(data.y[s:e], scores[s:e])
            warnings_all.extend(m.pop("warnings", []))
            metrics[name] = m
        metrics["status"] = "ok"
        metrics["paper_fidelity"] = "faithful"
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
            "contamination": args.contamination,
            "n_jobs": -1,
            "batch_size": args.batch_size,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous",
            "paper_fidelity": "faithful",
            "source": "PyOD ECOD (BSD-2)",
            "learning": "unsupervised",
        }
        write_json(out_dir / "config.json", config)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "timing.json", {
            "fit_seconds": fit_timer.seconds,
            "score_seconds": score_timer.seconds,
            "throughput": throughput,
        })
        write_json(out_dir / "warnings.json", {"warnings": warnings_all})

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
