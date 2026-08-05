"""Deep SVDD runner CLI (tabular, one-class unsupervised).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/DeepSVDD/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--max-samples N] [--quick] [--device cpu]
        [--hidden 64 32] [--rep-dim 16] [--epochs 50] [--batch-size 256]
        [--lr 1e-3] [--pretrain-ae]

Outputs under result/DeepSVDD/<dataset>/<protocol>/seed_<seed>/:
    config.json metrics.json scores.parquet timing.json warnings.json run.log

Input: the shared tabular view (script/tabular_pipeline), same as IF/DIF/ECOD.
Score = ||phi(x) - c||^2 (larger = more anomalous); fitting uses train rows
and no labels.
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
from script.model.DeepSVDD.deepsvdd import DeepSVDD  # noqa: E402

MODEL = "DeepSVDD"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep SVDD baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--rep-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--pretrain-ae", action="store_true")
    p.add_argument("--ae-epochs", type=int, default=10)
    return p.parse_args()


def _split_indices(n: int):
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
    with open(out_dir / "run.log", "w", encoding="utf-8") as log:
        log.write(f"DeepSVDD runner | dataset={args.dataset} protocol={args.protocol} "
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
            model = DeepSVDD(
                input_dim=data.X.shape[1],
                hidden=tuple(args.hidden),
                rep_dim=args.rep_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                pretrain_ae=args.pretrain_ae,
                ae_epochs=args.ae_epochs,
                device=args.device,
                seed=args.seed,
            )
            model.fit(data.X[splits["train"][0]:splits["train"][1]])
        log.write(f"fit done in {fit_timer.seconds:.2f}s\n"); log.flush()

        score_timer = Timer()
        with score_timer:
            scores = model.score(data.X)
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
            "hidden": args.hidden,
            "rep_dim": args.rep_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pretrain_ae": args.pretrain_ae,
            "ae_epochs": args.ae_epochs,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous",
            "paper_fidelity": "faithful",
            "source": "https://github.com/lukasruff/Deep-SVDD-PyTorch (ICML 2018)",
            "learning": "one-class unsupervised",
            "adaptation": "CNN encoder replaced by MLP; AE pre-training "
                          "optional (off by default)",
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
        }, indent=2))
        log.write("\n")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
