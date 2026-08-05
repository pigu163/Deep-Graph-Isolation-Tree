"""F-FADE runner CLI (edge-stream, unsupervised).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/FFADE/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite]
        [--max-samples N] [--quick] [--device cpu]
        [--embedding-size 200] [--alpha 0.999] [--M 100] [--T-th 120]
        [--epochs 5] [--online-train-steps 10]

Outputs under result/FFADE/<dataset>/<protocol>/seed_<seed>/:
    config.json metrics.json scores.parquet timing.json warnings.json run.log

Input: canonical link-stream events (data/processed/<dataset>/events.parquet);
events are grouped into time slices (same policy as MAD:
paysim/elliptic/midas native steps, IBM AML 1-hour bins).
Labels are used for evaluation only.
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
from script.model.FFADE.ffade import make_slice_ids, run_f_fade  # noqa: E402

MODEL = "FFADE"

# Slice policy mirrors the MAD baseline (documented there):
# native step for step-datasets, 1-hour bins for Unix-time IBM AML.
SLICE_POLICY = {
    "paysim": "native-step",
    "elliptic_pp": "native-step",
    "midas_darpa": "native-step",
    "ibm_hi_small": "fixed-width",
    "ibm_li_small": "fixed-width",
}
BIN_WIDTH = 3600.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F-FADE baseline runner")
    p.add_argument("--dataset", required=True,
                   choices=list(SLICE_POLICY))
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--embedding-size", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.999)
    p.add_argument("--M", type=int, default=100)
    p.add_argument("--T-th", type=int, default=120)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--online-train-steps", type=int, default=10)
    p.add_argument("--t-setup", type=int, default=None,
                   help="override setup-slice count (default: "
                        "min(8000, 30 percent of slices))")
    p.add_argument("--w-upd", type=int, default=None,
                   help="override embedding-update interval (default: "
                        "min(720, 5 percent of slices))")
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
        log.write(f"FFADE runner | dataset={args.dataset} protocol={args.protocol} "
                  f"seed={args.seed}\n")
        log.flush()

        data = load_event_table(args.dataset)
        n_total = len(data["event_id"])
        if args.max_samples is not None:
            n_total = min(args.max_samples, n_total)
        for k in ("event_id", "timestamp", "src", "dst", "label",
                  "label_available"):
            data[k] = data[k][:n_total]
        log.write(f"loaded {n_total} events\n"); log.flush()

        policy = SLICE_POLICY[args.dataset]
        slice_ids = make_slice_ids(data["timestamp"], policy, BIN_WIDTH)
        n_slices = int(slice_ids.max()) + 1
        log.write(f"slice policy={policy} n_slices={n_slices}\n"); log.flush()

        score_timer = Timer()
        with score_timer:
            scores = run_f_fade(
                data, slice_ids, n_slices,
                embedding_size=args.embedding_size,
                batch_size=args.batch_size,
                t_setup=args.t_setup, w_upd=args.w_upd,
                alpha=args.alpha, M=args.M, t_th=args.T_th,
                epochs=args.epochs,
                online_train_steps=args.online_train_steps,
                seed=args.seed, log=lambda m: (log.write(m + "\n"), log.flush()))
        log.write(f"scored in {score_timer.seconds:.2f}s\n"); log.flush()

        splits = _split_indices(n_total)
        warnings_all: list[str] = []
        metrics = {}
        for name, (s, e) in splits.items():
            m = compute_metrics(data["label"][s:e], scores[s:e])
            warnings_all.extend(m.pop("warnings", []))
            metrics[name] = m
        metrics["status"] = "ok"
        metrics["paper_fidelity"] = "faithful"
        metrics["timing"] = {
            "fit_seconds": 0.0,
            "score_seconds": round(score_timer.seconds, 6),
            "throughput": round(n_total / max(score_timer.seconds, 1e-9), 3),
        }

        config = {
            "model": MODEL,
            "dataset": args.dataset,
            "protocol": args.protocol,
            "seed": args.seed,
            "device": args.device,
            "embedding_size": args.embedding_size,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "M": args.M,
            "T_th": args.T_th,
            "epochs": args.epochs,
            "online_train_steps": args.online_train_steps,
            "t_setup": args.t_setup,
            "w_upd": args.w_upd,
            "slice_policy": {"kind": policy, "bin_width": BIN_WIDTH},
            "n_slices": n_slices,
            "max_samples_used": args.max_samples,
            "is_quick_run": bool(args.quick),
            "score_direction": "larger=more anomalous",
            "paper_fidelity": "faithful",
            "source": "https://github.com/snap-stanford/F-FADE (WSDM 2021)",
            "learning": "unsupervised",
            "adaptation": "events grouped into discrete slices (native step / "
                          "1h bins for IBM); global F_FAC limited to |V|^2 "
                          "<= 4e6 for tractability on large graphs",
        }
        write_json(out_dir / "config.json", config)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "timing.json", {
            "fit_seconds": 0.0,
            "score_seconds": score_timer.seconds,
            "throughput": n_total / max(score_timer.seconds, 1e-9),
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
            "slice_id": slice_ids,
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
