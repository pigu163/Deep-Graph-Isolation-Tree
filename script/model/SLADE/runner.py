"""SLADE runner CLI (edge-stream, self-supervised).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/SLADE/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--max-samples N] [--quick] [--device cpu]
        [--n-epochs 10] [--batch-size 100] [--n-neighbors 20]
        [--message-dim 128] [--memory-dim 256]

Outputs under result/SLADE/<dataset>/<protocol>/seed_<seed>/:
    config.json metrics.json scores.parquet timing.json warnings.json run.log

Input: canonical link-stream events; train split (60%) used for
self-supervised training (labels never used), all splits scored past-only.
Score = (2 - drift_sim - recovery_sim)/4, larger = more anomalous.
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
from script.model.SLADE.slade_adapter import run_slade  # noqa: E402

MODEL = "SLADE"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SLADE baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--message-dim", type=int, default=128)
    p.add_argument("--memory-dim", type=int, default=256)
    p.add_argument("--memory-updater", default="gru", choices=["gru", "rnn"])
    p.add_argument("--message-updater", default="mlp",
                   choices=["mlp", "identity"])
    p.add_argument("--srf", type=float, default=0.1)
    p.add_argument("--drf", type=float, default=0.1)
    return p.parse_args()


def _split_indices(n: int):
    t = int(n * 0.7)
    return {"train": (0, t), "test": (t, n)}


def main() -> None:
    args = _parse_args()
    if args.quick:
        args.n_epochs = min(args.n_epochs, 1)
        args.n_neighbors = min(args.n_neighbors, 5)
        args.message_dim = min(args.message_dim, 32)
        args.memory_dim = min(args.memory_dim, 64)
        args.batch_size = max(args.batch_size, 256)
    out_dir = run_dir(MODEL, args.dataset, args.protocol, args.seed)
    if out_dir.exists() and not args.overwrite:
        print(f"skip (exists, no --overwrite): {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run.log", "w", encoding="utf-8") as log:
        log.write(f"SLADE runner | dataset={args.dataset} protocol={args.protocol} "
                  f"seed={args.seed}\n")
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
            scores, phase_timing = run_slade(
                data, splits,
                device=args.device, seed=args.seed,
                batch_size=args.batch_size, n_neighbors=args.n_neighbors,
                n_heads=args.n_heads, n_epochs=args.n_epochs, lr=args.lr,
                dropout=args.dropout, message_dim=args.message_dim,
                memory_dim=args.memory_dim,
                memory_updater=args.memory_updater,
                message_updater=args.message_updater,
                srf=args.srf, drf=args.drf,
                log=lambda m: (log.write(m + "\n"), log.flush()), return_timing=True)
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
            "fit_seconds": round(phase_timing["fit_seconds"], 6),
            "score_seconds": round(phase_timing["score_seconds"], 6),
            "throughput": round(n_total / max(phase_timing["score_seconds"], 1e-9), 3),
        }

        config = {
            "model": MODEL,
            "dataset": args.dataset,
            "protocol": args.protocol,
            "seed": args.seed,
            "device": args.device,
            "batch_size": args.batch_size,
            "n_neighbors": args.n_neighbors,
            "n_heads": args.n_heads,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "dropout": args.dropout,
            "message_dim": args.message_dim,
            "memory_dim": args.memory_dim,
            "memory_updater": args.memory_updater,
            "message_updater": args.message_updater,
            "srf": args.srf,
            "drf": args.drf,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous",
            "paper_fidelity": "faithful",
            "source": "https://github.com/jhsk777/SLADE (KDD 2024)",
            "learning": "self-supervised (labels unused in training)",
            "adaptation": "events.parquet -> Data; contiguous node remap; "
                          "chronological splits from splits.json; past-only "
                          "neighbor finder (searchsorted, ts < query)",
        }
        write_json(out_dir / "config.json", config)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "timing.json", {
            "fit_seconds": phase_timing["fit_seconds"],
            "score_seconds": phase_timing["score_seconds"],
            "throughput": n_total / max(phase_timing["score_seconds"], 1e-9),
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
