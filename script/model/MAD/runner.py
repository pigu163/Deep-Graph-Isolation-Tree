"""MAD runner: edge-event anomaly detection on the canonical link-stream data.

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/model/MAD/runner.py \
        --dataset paysim --protocol chronological --seed 42 --device cpu \
        [--query-mode active-edge|local-candidate-set|full-scope] \
        [--window-mode fixed|expanding|stationary] [--window K] \
        [--max-samples N] [--quick] [--overwrite]

Outputs under result/MAD/<dataset>/<protocol>/seed_<seed>/:
    config.json  metrics.json  scores.parquet  timing.json  warnings.json  run.log

The dataset root ``data/processed`` is resolved from the repository root
(``script.common.results.REPO_ROOT``), so the runner must be executed from
the repository root (which ``run_all_models.py`` does).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# Make `script.common.*` importable when invoked from the repository root.
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from script.common.metrics import compute_metrics  # noqa: E402
from script.common.results import (  # noqa: E402
    RESULT_ROOT,
    default_log,
    run_dir,
    save_scores,
    write_json,
    write_text,
)
from script.dataset.linkstream import schema as ls_schema  # noqa: E402
from script.dataset.linkstream.temporal_index import TemporalNeighborIndex  # noqa: E402
from script.dataset.linkstream.utils import read_json  # noqa: E402

from script.model.MAD.mad import QUERY_MODES, run_edge_event_batch  # noqa: E402
from script.model.MAD.probability import SlicePolicy, compute_slice_ids  # noqa: E402

PROCESSED_ROOT = _REPO / "data" / "processed"

#: Per-dataset discrete-time policy (task Sec. 9.2).
#: - PaySim  : timestamp IS the fraud step (1..743) -> native step.
#: - Elliptic++: timestamp is the time step (1..49) -> native step.
#: - MIDAS-DARPA: timestamp is an integer tick (1..46572) -> native step.
#: - IBM AML: timestamp is Unix seconds spanning ~18 days; a 1-hour bin is
#:   chosen so that every slice carries a sizeable number of transactions
#:   (~12K/slice) without merging distinct days (a day bin would leave only
#:   ~18 slices).  The choice is a time-scale decision, not a label-tuned one.
DEFAULT_POLICY = {
    "paysim": SlicePolicy("native-step"),
    "elliptic_pp": SlicePolicy("native-step"),
    "midas_darpa": SlicePolicy("native-step"),
    "ibm_hi_small": SlicePolicy("fixed-width", bin_width=3600.0),
    "ibm_li_small": SlicePolicy("fixed-width", bin_width=3600.0),
}

#: Default historical window in slices (fixed mode).  Recorded in config;
#: not tuned on test labels (task Sec. 9.4 / 14).
DEFAULT_WINDOW_K = 100


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=sorted(DEFAULT_POLICY))
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--query-mode", default="active-edge", choices=QUERY_MODES)
    p.add_argument("--window-mode", "--history-mode", dest="window_mode",
                   default="fixed", choices=["fixed", "expanding", "stationary"],
                   help="historical-window selection (alias --history-mode)")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW_K,
                   help="window length in slices (fixed mode)")
    p.add_argument("--smoothing", type=float, default=0.0,
                   help="additive smoothing alpha in P=(count+a)/(K+2a); "
                        "0.0 = exact paper definition")
    p.add_argument("--max-candidates", type=int, default=256,
                   help="max relations per query (local-candidate-set)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="smoke-test mode; results marked is_quick_run=true")
    return p


def load_events(dataset: str) -> dict:
    """Load the canonical event table columns from events.parquet."""
    proc = PROCESSED_ROOT / dataset
    meta = read_json(proc / ls_schema.METADATA_JSON)
    splits = read_json(proc / ls_schema.SPLITS_JSON)
    cols = [ls_schema.EVENT_ID, ls_schema.TIMESTAMP, ls_schema.SRC,
            ls_schema.DST, ls_schema.LABEL, ls_schema.LABEL_AVAILABLE]
    df = pl.scan_parquet(proc / ls_schema.EVENTS_PARQUET).select(cols).collect()
    return {
        "meta": meta,
        "splits": {k: [int(v[0]), int(v[1])] for k, v in splits.items()},
        "event_id": df[ls_schema.EVENT_ID].to_numpy().astype(np.int64),
        "timestamp": df[ls_schema.TIMESTAMP].to_numpy().astype(np.float64),
        "src": df[ls_schema.SRC].to_numpy().astype(np.int64),
        "dst": df[ls_schema.DST].to_numpy().astype(np.int64),
        "label": df[ls_schema.LABEL].to_numpy().astype(np.int8),
        "label_available": df[ls_schema.LABEL_AVAILABLE].to_numpy(),
    }


def load_temporal_index(dataset: str) -> TemporalNeighborIndex | None:
    idx_dir = PROCESSED_ROOT / dataset / ls_schema.TEMPORAL_INDEX_DIR
    if not (idx_dir / ls_schema.TEMP_INDEX_FILES[0]).exists():
        return None
    try:
        return TemporalNeighborIndex.load(idx_dir)
    except Exception:
        return None


def slice_policy_for(dataset: str, meta: dict) -> SlicePolicy:
    policy = DEFAULT_POLICY[dataset]
    if policy.kind == "fixed-width" and policy.min_time == 0.0:
        policy.min_time = float(meta.get("timestamp_min", 0.0))
    return policy


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = run_dir("MAD", args.dataset, args.protocol, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    default_log(out / "run.log")

    if not args.overwrite and (out / "metrics.json").exists():
        print(f"skip (exists): {out}", flush=True)
        return 0

    t_fit = time.perf_counter()
    data = load_events(args.dataset)
    meta = data["meta"]
    policy = slice_policy_for(args.dataset, meta)

    t_idx = load_temporal_index(args.dataset)
    src, dst = data["src"], data["dst"]
    n_orig = len(src)

    # If requested, keep only the last max_samples events (so the window
    # effects and split proportions remain representative).
    max_samples = args.max_samples
    if max_samples is not None and max_samples > 0 and max_samples < n_orig:
        keep = np.zeros(n_orig, dtype=bool)
        keep[n_orig - max_samples:] = True
        keep = np.flatnonzero(keep)
        src, dst = src[keep], dst[keep]
        data["timestamp"] = data["timestamp"][keep]
        data["label"] = data["label"][keep]
        data["label_available"] = data["label_available"][keep]
        data["event_id"] = data["event_id"][keep]
        # Re-derive splits from the kept event-id range.
        spl = data["splits"]
        new_spl = {}
        for name in ("train", "val", "test"):
            lo, hi = spl[name]
            lo2 = int(np.searchsorted(data["event_id"], lo, side="left"))
            hi2 = int(np.searchsorted(data["event_id"], hi, side="left"))
            new_spl[name] = [lo2, hi2]
        data["splits"] = new_spl
        n = len(src)
    else:
        n = n_orig

    slice_ids = compute_slice_ids(data["timestamp"], policy,
                                  event_ids=data["event_id"])
    fit_seconds = time.perf_counter() - t_fit

    warnings: list[str] = []
    if args.quick:
        warnings.append("is_quick_run=true: smoke test, not a formal run")
    if args.query_mode == "local-candidate-set" and max_samples is None:
        warnings.append(
            "local-candidate-set on the full stream is O(N*M) in Python and "
            "slow on the largest datasets; use --max-samples for smoke runs")
    if args.query_mode == "full-scope":
        warnings.append(
            "full-scope uses the *observed* relation space of each slice "
            "(restricted scope); the unseen relation space E=V x V is not "
            "enumerated (would be infeasible for PaySim's ~9M accounts)")

    t_score = time.perf_counter()
    records, stats, _mad = run_edge_event_batch(
        num_nodes=int(meta["num_nodes"]),
        src=src,
        dst=dst,
        timestamps=data["timestamp"],
        slice_ids=slice_ids,
        labels=data["label"],
        label_available=data["label_available"],
        splits=data["splits"],
        query_mode=args.query_mode,
        window_mode=args.window_mode,
        window_k=args.window,
        smoothing=args.smoothing,
        max_candidates=args.max_candidates,
        max_samples=None,          # already subsampled above
        temporal_index=t_idx,
    )
    score_seconds = time.perf_counter() - t_score

    n_scored = stats["n_scored"]
    throughput = n_scored / score_seconds if score_seconds > 0 else 0.0

    # ---- metrics on the test split ----------------------------------------
    rows = [r.as_row() for r in records]
    test = [r for r in rows
            if r["split"] == "test" and r["label_available"]]
    if test:
        y_true = np.array([r["label"] for r in test], dtype=np.int64)
        y_score = np.array([r["anomaly_score"] for r in test], dtype=np.float64)
        test_metrics = compute_metrics(y_true, y_score)
        warnings.extend(test_metrics.pop("warnings", []))
    else:
        test_metrics = {
            "roc_auc": None, "pr_auc": None,
            "precision_at_k": None, "recall_at_k": None, "f1_at_k": None,
            "n_eval": 0, "n_pos": 0,
        }
        warnings.append("no labeled test events scored")

    config = {
        "model": "MAD",
        "dataset": args.dataset,
        "protocol": args.protocol,
        "seed": args.seed,
        "query_mode": args.query_mode,
        "window_mode": args.window_mode,
        "window_k": args.window,
        "smoothing": args.smoothing,
        "max_candidates": args.max_candidates,
        "var_floor": 1e-2,
        "slice_policy": policy.to_dict(),
        "paper_fidelity": "faithful",
        "deterministic": True,
        "is_quick_run": bool(args.quick),
        "max_samples": max_samples,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "num_events_loaded": int(stats["n_events_loaded"]),
        "num_events_processed": int(stats["n_events_processed"]),
        "num_slices": int(stats["n_slices"]),
        "dataset_metadata": {
            "num_nodes": int(meta["num_nodes"]),
            "timestamp_min": float(meta["timestamp_min"]),
            "timestamp_max": float(meta["timestamp_max"]),
        },
    }

    metrics = {
        "status": "ok",
        "paper_fidelity": "faithful",
        "test": test_metrics,
        "timing": {
            "fit_seconds": round(fit_seconds, 4),
            "score_seconds": round(score_seconds, 4),
            "throughput": round(float(throughput), 2),
        },
    }

    # ---- persist -----------------------------------------------------------
    write_json(out / "config.json", config)
    write_json(out / "metrics.json", metrics)
    write_json(out / "timing.json", metrics["timing"])
    write_json(out / "warnings.json", warnings)

    if rows:
        df = pl.DataFrame(rows)
        save_scores(out / "scores.parquet", df)
    else:
        warnings.append("no rows produced")

    # ---- leakage check (task Sec. 11.7) ----------------------------------
    write_json(out / "interface_check.json", {
        "past_only_passed": bool(stats["leakage_check"]["passed"]),
        "n_checks": int(stats["leakage_check"]["n_checks"]),
        "samples": stats["leakage_check"]["samples"],
        "window_mode_effective": stats["window_mode_effective"],
    })

    write_json(out / "warnings.json", warnings)

    log_lines = [
        f"dataset={args.dataset} protocol={args.protocol} seed={args.seed}",
        f"query_mode={args.query_mode} window_mode={args.window_mode} "
        f"window_k={args.window} smoothing={args.smoothing}",
        f"slice_policy={policy.to_dict()}",
        f"events_loaded={stats['n_events_loaded']} "
        f"events_processed={stats['n_events_processed']} scored={n_scored} "
        f"slices={stats['n_slices']}",
        f"fit={fit_seconds:.2f}s score={score_seconds:.2f}s "
        f"throughput={throughput:.0f} ev/s",
        f"test_metrics={json.dumps(test_metrics, sort_keys=True)}",
        "warnings: " + ("; ".join(warnings) if warnings else "none"),
    ]
    write_text(out / "run.log", "\n".join(log_lines) + "\n")

    print("\n".join(log_lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
