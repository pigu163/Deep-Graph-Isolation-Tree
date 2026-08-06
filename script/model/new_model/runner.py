#!/usr/bin/env python
"""CLI runner for the Temporal Motif-Conditioned Deep Isolation Forest."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from script.common.events import event_feature_columns, load_event_features, load_event_table
from script.common.metrics import compute_metrics
from script.common.results import run_dir, save_scores, write_json
from script.common.timing import Timer
from script.model.new_model.temporal_motif_reservoir_dif import ReservoirConfig, TemporalMotifReservoirDIF

MODEL = "TemporalMotifReservoirDIF"
DATASETS = ["midas_darpa", "ibm_hi_small", "ibm_li_small", "paysim", "elliptic_pp"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS); p.add_argument("--protocol", default="chronological", choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42); p.add_argument("--device", default="cpu", choices=["cpu", "cuda"]); p.add_argument("--overwrite", action="store_true"); p.add_argument("--quick", action="store_true")
    p.add_argument("--max-samples", type=int); p.add_argument("--num-workers", type=int, default=0); p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-reservoirs", type=int); p.add_argument("--state-dim", type=int); p.add_argument("--embedding-dim", type=int); p.add_argument("--trees-per-reservoir", type=int); p.add_argument("--subsample-size", type=int); p.add_argument("--recurrent-layout")
    return p.parse_args()

def features(dataset, n):
    cols = event_feature_columns(dataset)
    if not cols: return np.zeros((n, 1), dtype=np.float64), []
    loaded = load_event_features(dataset); x = np.column_stack([loaded[c] for c in cols])[:n].astype(np.float64)
    x[~np.isfinite(x)] = 0.; return np.clip(np.sign(x) * np.log1p(np.abs(x)), -30., 30.), cols

def main():
    args = parse_args(); out = run_dir(MODEL, args.dataset, args.protocol, args.seed)
    if out.exists() and not args.overwrite: print(f"skip (exists, no --overwrite): {out}"); return
    out.mkdir(parents=True, exist_ok=True); data = load_event_table(args.dataset)
    n = min(len(data["event_id"]), args.max_samples) if args.max_samples else len(data["event_id"])
    for key in ("event_id", "timestamp", "src", "dst", "label", "label_available"): data[key] = data[key][:n]
    data["edge_feat"], feature_names = features(args.dataset, n); cfg = ReservoirConfig(device=args.device)
    if args.quick: cfg.num_reservoirs, cfg.state_dim, cfg.embedding_dim, cfg.trees_per_reservoir, cfg.subsample_size = 2, 16, 16, 4, 64
    for name in ("num_reservoirs", "state_dim", "embedding_dim", "trees_per_reservoir", "subsample_size", "recurrent_layout", "batch_size"):
        if getattr(args, name) is not None: setattr(cfg, name, getattr(args, name))
    train_end = min(int(data["splits"].get("train", [0, round(.6*n)])[1]), n)
    fit_data = data if args.protocol == "transductive" else {k: (v[:train_end] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    with Timer() as fit_timer: model = TemporalMotifReservoirDIF(data["edge_feat"].shape[1], cfg, args.seed).fit(fit_data)
    with Timer() as score_timer: scores = model.score(data)
    metrics, warnings = {}, []
    for name in ("train", "val", "test"):
        s, e = data["splits"].get(name, [0, n]); s, e = min(s,n), min(e,n); m = compute_metrics(data["label"][s:e], scores[s:e]); warnings += m.pop("warnings", []); metrics[name] = m
    timing = {"fit_seconds":fit_timer.seconds, "score_seconds":score_timer.seconds, "throughput":n/max(score_timer.seconds, 1e-9)}; metrics.update(status="ok", paper_fidelity="specification_implementation", timing=timing)
    config = {**model.metadata(), "model":MODEL, "dataset":args.dataset, "protocol":args.protocol, "seed":args.seed, "edge_feature_names":feature_names, "max_samples_used":args.max_samples, "is_quick_run":bool(args.quick), "score_direction":"larger=more anomalous", "contraction_report":model.contraction_report}
    write_json(out / "config.json", config); write_json(out / "metrics.json", metrics); write_json(out / "timing.json", timing); write_json(out / "warnings.json", {"warnings":warnings}); write_json(out / "reservoir_contraction_report.json", model.contraction_report); write_json(out / "metadata.json", model.metadata()); model.save(out / "model.pkl")
    split = np.full(n, "test", dtype=object)
    for name, (s,e) in data["splits"].items(): split[min(s,n):min(e,n)] = name
    save_scores(out / "scores.parquet", pl.DataFrame({"sample_id":data["event_id"], "timestamp_or_step":data["timestamp"], "label":data["label"], "label_available":data["label_available"], "anomaly_score":scores, "split":split}))
    (out / "run.log").write_text(json.dumps({"status":"ok", "n_events":n, "test":metrics["test"]}, indent=2), encoding="utf-8"); print(f"done: {out}")
if __name__ == "__main__": main()
