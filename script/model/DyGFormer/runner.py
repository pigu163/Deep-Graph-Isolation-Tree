"""DyGFormer runner CLI (dynamic graph, self-supervised link prediction).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/DyGFormer/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--max-samples N] [--quick] [--device cpu]
        [--num-epochs 5] [--batch-size 200] [--num-neighbors 20]
        [--num-layers 2] [--feat-dim 172]

Outputs under result/DyGFormer/<dataset>/<protocol>/seed_<seed>/:
    config.json metrics.json scores.parquet timing.json warnings.json run.log

Score: 1 - P(link) from the self-supervised link predictor
(larger = more anomalous). Training uses the train split and no labels.
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

from script.common.events import load_event_table  # noqa: E402
from script.common.metrics import compute_metrics  # noqa: E402
from script.common.results import run_dir, save_scores, write_json  # noqa: E402
from script.common.timing import Timer  # noqa: E402
from script.model.DyGLib.dyglib_adapter import run_dyglib  # noqa: E402

MODEL = "DyGFormer"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DyGFormer baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--num-neighbors", type=int, default=20)
    p.add_argument("--time-gap", type=int, default=2000)
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--feat-dim", type=int, default=172)
    p.add_argument("--learning-rate", type=float, default=1e-4)
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
        log.write(f"DyGFormer runner | dataset={args.dataset} "
                  f"protocol={args.protocol} seed={args.seed}\n")
        log.flush()

        data = load_event_table(args.dataset)
        n_total = len(data["event_id"])
        if args.max_samples is not None:
            n_total = min(args.max_samples, n_total)
        for k in ("event_id", "timestamp", "src", "dst", "label",
                  "label_available"):
            data[k] = data[k][:n_total]
        splits = _split_indices(n_total)
        log.write(f"loaded {n_total} events; splits={splits}\n"); log.flush()

        fit_timer = Timer()
        with fit_timer:
            scores = run_dyglib(
                "DyGFormer", data, splits, dataset=args.dataset,
                device=args.device, seed=args.seed,
                batch_size=args.batch_size, num_neighbors=args.num_neighbors,
                time_gap=args.time_gap, num_layers=args.num_layers,
                num_heads=args.num_heads, dropout=args.dropout,
                feat_dim=args.feat_dim, num_epochs=args.num_epochs,
                patch_size=args.patch_size,
                learning_rate=args.learning_rate,
                log=lambda m: (log.write(m + "\n"), log.flush()))
        log.write(f"fit+score done in {fit_timer.seconds:.2f}s\n"); log.flush()

        warnings_all: list[str] = []
        metrics = {}
        for name, (s, e) in splits.items():
            m = compute_metrics(data["label"][s:e], scores[s:e])
            warnings_all.extend(m.pop("warnings", []))
            metrics[name] = m
        metrics["status"] = "ok"
        metrics["paper_fidelity"] = "faithful"
        metrics["timing"] = {
            "fit_seconds": round(fit_timer.seconds, 6),
            "score_seconds": round(fit_timer.seconds, 6),
            "throughput": round(n_total / max(fit_timer.seconds, 1e-9), 3),
        }

        config = {
            "model": MODEL,
            "dataset": args.dataset,
            "protocol": args.protocol,
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "num_neighbors": args.num_neighbors,
            "time_gap": args.time_gap,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "feat_dim": args.feat_dim,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous (1 - P(link))",
            "paper_fidelity": "faithful",
            "source": "https://github.com/yule-buaa/dyglib (NeurIPS 2023)",
            "learning": "self-supervised link prediction (labels unused in "
                        "training)",
            "adaptation": "events.parquet -> DyGLib Data; contiguous node "
                          "remap; random node features; f_* edge features or "
                          "dummy; past-only NeighborSampler",
        }
        write_json(out_dir / "config.json", config)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "timing.json", {
            "fit_seconds": fit_timer.seconds,
            "score_seconds": fit_timer.seconds,
            "throughput": n_total / max(fit_timer.seconds, 1e-9),
        })
        write_json(out_dir / "warnings.json", {"warnings": warnings_all})

        split_arr = np.empty(n_total, dtype=object)
        for name, (s, e) in splits.items():
            split_arr[s:e] = name
        df = pl.DataFrame({
            "sample_id": data["event_id"],
            "timestamp_or_step": data["timestamp"],
            "label": data["label"],
            "label_available": data["label_available"],
            "anomaly_score": scores,
            "split": split_arr,
        })
        save_scores(out_dir / "scores.parquet", df)

        log.write(json.dumps({
            "status": "ok",
            "test_roc_auc": metrics["test"].get("roc_auc"),
            "test_pr_auc": metrics["test"].get("pr_auc"),
        }, indent=2))
        log.write("\n")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
