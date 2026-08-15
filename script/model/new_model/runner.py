#!/usr/bin/env python
"""Run the pair-centric local-link-stream ESN isolation forest on IBM AML."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from script.common.metrics import compute_metrics
from script.common.results import run_dir, save_scores, write_json
from script.common.timing import Timer
from script.dataset.linkstream.collate import TemporalContextCollator
from script.dataset.linkstream.datasets import LinkStreamDataset
from script.dataset.linkstream.temporal_index import TemporalNeighborIndex
from script.model.new_model.temporal_motif_reservoir_dif import (
    LocalLinkStreamConfig, LocalLinkStreamESNIF,
)

MODEL = "LocalLinkStreamESNIF"
DATASETS = ("ibm_hi_small", "ibm_li_small", "paysim")
# IBM timestamps are seconds; PaySim uses hourly simulation steps.  These are
# equivalent causal history horizons (seven days) in each dataset's own unit.
TIME_WINDOWS = {"ibm_hi_small": 7 * 24 * 3600.0, "ibm_li_small": 7 * 24 * 3600.0,
                "paysim": 7 * 24.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quick", action="store_true", help="small smoke configuration only")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="restrict to the first N chronological events (smoke tests)")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--context-workers", type=int, default=0,
                        help="0 is fastest on IBM Small; positive values use experimental thread parallelism")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--max-context-events", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="seed contexts encoded together; tree population remains 64")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--projection-count", type=int, default=16)
    parser.add_argument("--trees-per-family", type=int, default=100)
    return parser.parse_args()


def _split_indices(n: int) -> dict[str, tuple[int, int]]:
    """Canonical chronological 70/30 train/test split."""
    train_end = int(n * 0.70)
    return {"train": (0, train_end), "test": (train_end, n)}


def main() -> None:
    args = parse_args()
    out = run_dir(MODEL, args.dataset, "chronological", args.seed)
    if out.exists() and not args.overwrite:
        print(f"skip (exists, no --overwrite): {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    processed = REPO / "data" / "processed" / args.dataset
    dataset = LinkStreamDataset(processed, split="all", include_edge_features=True)
    index = TemporalNeighborIndex.load(processed / "temporal_index")
    collator = TemporalContextCollator(
        index, dataset=dataset, num_hops=args.num_hops,
        max_neighbors=args.max_neighbors, max_context_events=args.max_context_events,
        min_context_events=0, direction="both", context_workers=args.context_workers,
        time_window=TIME_WINDOWS[args.dataset],
    )
    if collator._edge_feat is None:
        raise SystemExit("canonical data has no numeric edge features")
    n_total = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
    splits = _split_indices(n_total)
    train_end = splits["train"][1]
    test_ids = np.arange(*splits["test"], dtype=np.int64)
    if args.max_test_samples is not None:
        test_ids = test_ids[:args.max_test_samples]
    cfg = LocalLinkStreamConfig(
        state_dim=args.state_dim, projection_count=args.projection_count,
        trees_per_family=args.trees_per_family, encode_batch_size=args.batch_size,
        device=args.device,
    )
    if args.quick:
        cfg.state_dim = min(cfg.state_dim, 32)
        cfg.trees_per_family = min(cfg.trees_per_family, 3)
        cfg.calibration_size = 256
    model = LocalLinkStreamESNIF(collator._edge_feat.shape[1], cfg, args.seed)
    train_ids = np.arange(train_end, dtype=np.int64)
    with Timer() as fit_timer:
        model.fit(train_ids, collator._edge_feat, collator._ts, collator)
    with Timer() as score_timer:
        score, detail = model.score(test_ids, collator._edge_feat, collator._ts,
                                    collator, return_details=True)
    labels = dataset.label[test_ids]
    metric = compute_metrics(labels, score)
    positives = labels == 1
    coverage = {
        "n_test": int(len(test_ids)),
        "empty_context_rate": float((detail["context_size"] == 0).mean()),
        "median_context_size": float(np.median(detail["context_size"])),
        "positive_empty_context_rate": float((detail["context_size"][positives] == 0).mean()) if positives.any() else None,
    }
    timing = {"fit_seconds": fit_timer.seconds, "score_seconds": score_timer.seconds,
              "throughput_transactions_per_second": len(test_ids) / max(score_timer.seconds, 1e-9)}
    metadata = model.metadata()
    config = {**metadata, "dataset": args.dataset, "seed": args.seed,
              "chronological_split": {name: list(bounds) for name, bounds in splits.items()},
              "sampling": {"num_hops": args.num_hops, "time_window_seconds": TIME_WINDOWS[args.dataset],
                           "max_neighbors": args.max_neighbors, "max_context_events": args.max_context_events,
                           "direction": "incident_both", "strict_past": True,
                           "timestamp_bucket_atomic": True,
                           "context_workers": args.context_workers, "encode_batch_size": args.batch_size},
              "paper_fidelity": "DIF-guided_Normal_initialization_with_ESN_stability_rescaling",
              "labels_used_for_fit": False, "is_quick_run": bool(args.quick)}
    metrics = {"test": metric, "context_coverage": coverage, "timing": timing,
               "status": "ok", "score_direction": "larger=more anomalous"}
    write_json(out / "config.json", config)
    write_json(out / "metrics.json", metrics)
    write_json(out / "timing.json", timing)
    with (out / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    save_scores(out / "scores.parquet", pl.DataFrame({
        "sample_id": test_ids, "timestamp_or_step": dataset.ts[test_ids],
        "label": labels, "label_available": dataset.label_available[test_ids],
        "anomaly_score": score, "split": np.full(len(test_ids), "test"),
        "departure_score": detail["departure"], "surprise_score": detail["surprise"],
        "recurrence_score": detail["recurrence"], "context_size": detail["context_size"],
    }))
    (out / "run.log").write_text(json.dumps({"status": "ok", "metrics": metric,
        "coverage": coverage, "timing": timing}, indent=2), encoding="utf-8")
    print(f"done: {out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
