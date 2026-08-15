#!/usr/bin/env python3
"""Run the locked unsupervised transaction-level AML protocol.

This file is deliberately an orchestration layer. It imports or launches
existing code but never edits anything under ``script/``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments_result" / "unsupervised_txn_aml_v1"
RUNS = OUT / "runs"
ABLATIONS = OUT / "ablations"
DATASETS = ("ibm_hi_small", "ibm_li_small", "paysim")
ABLATION_DATASETS = ("ibm_hi_small", "ibm_li_small")
SEEDS = (42, 43, 44)
RUNNERS = {
    "IF": "script/model/IF/runner.py",
    "EIF": "script/model/EIF/runner.py",
    "DIF": "script/model/DIF/runner.py",
    "ECOD": "script/model/ECOD/runner.py",
    "FFADE": "script/model/FFADE/runner.py",
    "MAD": "script/model/MAD/runner.py",
    "SLADE": "script/model/SLADE/runner.py",
    "LocalLinkStreamESNIF": "script/model/new_model/runner.py",
}
VARIANTS = {
    "D": (("departure",), 300),
    "DS": (("departure", "surprise"), 150),
    "DSR": (("departure", "surprise", "recurrence"), 100),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def split_for(n: int) -> dict[str, list[int]]:
    train_end = int(n * 0.7)
    return {"train": [0, train_end], "test": [train_end, n]}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> dict:
    """Verify that tabular and link-stream views define the same 70/30 task."""
    checks: dict[str, object] = {"status": "ok", "datasets": {}}
    for dataset in DATASETS:
        link = ROOT / "data" / "processed" / dataset
        tab = ROOT / "data" / "processed_tabular" / dataset
        meta, link_split = read_json(link / "metadata.json"), read_json(link / "splits.json")
        tab_meta, tab_split = read_json(tab / "metadata.json"), read_json(tab / "splits.json")
        expected = split_for(int(meta["num_events"]))
        if link_split != expected or tab_split != expected:
            raise RuntimeError(f"{dataset}: expected matching chronological 70/30 splits")
        if int(tab_meta["n_samples"]) != int(meta["num_events"]):
            raise RuntimeError(f"{dataset}: tabular/link-stream event count mismatch")
        table = pl.scan_parquet(link / "events.parquet").select([
            pl.len().alias("n"), pl.col("event_id").min().alias("min_id"),
            pl.col("event_id").max().alias("max_id"),
            pl.col("timestamp").min().alias("min_ts"), pl.col("timestamp").max().alias("max_ts"),
        ]).collect().row(0, named=True)
        if table["n"] != meta["num_events"] or table["min_id"] != 0 or table["max_id"] != table["n"] - 1:
            raise RuntimeError(f"{dataset}: canonical event IDs are not 0..N-1")
        checks["datasets"][dataset] = {
            "n_events": int(table["n"]), "n_train": expected["train"][1],
            "n_test": expected["test"][1] - expected["test"][0],
            "n_positive_total": int(meta["num_positive_labels"]),
            "timestamp_min": float(table["min_ts"]), "timestamp_max": float(table["max_ts"]),
            "link_edge_features": list(meta["edge_feature_names"]),
            "tabular_features": int(tab_meta["n_features"]),
        }
    write_json(OUT / "preflight.json", checks)
    return checks


def record_environment(workers: int) -> None:
    try:
        import psutil
        memory_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        memory_bytes = None
    write_json(OUT / "environment.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "requested_workers": workers, "memory_bytes": memory_bytes,
    })


def run_process(command: list[str], result_root: Path, workers: int) -> None:
    env = os.environ.copy()
    env.update({"ACM_RESULT_ROOT": str(result_root), "OMP_NUM_THREADS": str(workers),
                "MKL_NUM_THREADS": str(workers), "OPENBLAS_NUM_THREADS": str(workers),
                "NUMEXPR_NUM_THREADS": str(workers)})
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main_command(method: str, dataset: str, seed: int, overwrite: bool) -> list[str]:
    command = [sys.executable, str(ROOT / RUNNERS[method]), "--dataset", dataset, "--seed", str(seed)]
    if method == "LocalLinkStreamESNIF":
        command += ["--device", "cpu"]
    else:
        command += ["--protocol", "chronological"]
    if overwrite:
        command.append("--overwrite")
    return command


def run_main(workers: int, overwrite: bool) -> None:
    for method in RUNNERS:
        for dataset in DATASETS:
            for seed in SEEDS:
                run_process(main_command(method, dataset, seed, overwrite), RUNS, workers)


def run_ablation_one(dataset: str, seed: int, variant: str, overwrite: bool) -> None:
    """Run a split-family ablation without altering proposed-model source code."""
    families, trees_per_family = VARIANTS[variant]
    out = ABLATIONS / f"LocalLinkStreamESNIF_{variant}" / dataset / "chronological" / f"seed_{seed}"
    if out.exists() and not overwrite:
        print(f"skip (exists, no --overwrite): {out}")
        return
    from script.common.metrics import compute_metrics
    from script.common.results import save_scores
    from script.common.timing import Timer
    from script.dataset.linkstream.collate import TemporalContextCollator
    from script.dataset.linkstream.datasets import LinkStreamDataset
    from script.dataset.linkstream.temporal_index import TemporalNeighborIndex
    from script.model.new_model.runner import TIME_WINDOWS
    from script.model.new_model.temporal_motif_reservoir_dif import LocalLinkStreamConfig, LocalLinkStreamESNIF

    out.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data" / "processed" / dataset
    source = LinkStreamDataset(processed, split="all", include_edge_features=True)
    index = TemporalNeighborIndex.load(processed / "temporal_index")
    collator = TemporalContextCollator(index, dataset=source, num_hops=1, max_neighbors=20,
                                       max_context_events=500, min_context_events=0, direction="both",
                                       context_workers=0, time_window=TIME_WINDOWS[dataset])
    if collator._edge_feat is None:
        raise RuntimeError(f"{dataset}: no numeric edge features")
    n_total, train_end = len(source), int(len(source) * 0.7)
    config = LocalLinkStreamConfig(trees_per_family=trees_per_family, device="cpu")
    model_type = type(f"LocalLinkStreamESNIF_{variant}", (LocalLinkStreamESNIF,), {"families": families})
    model = model_type(collator._edge_feat.shape[1], config, seed)
    train_ids, test_ids = np.arange(train_end), np.arange(train_end, n_total)
    with Timer() as fit_timer:
        model.fit(train_ids, collator._edge_feat, collator._ts, collator)
    with Timer() as score_timer:
        score, detail = model.score(test_ids, collator._edge_feat, collator._ts, collator, return_details=True)
    labels = source.label[test_ids]
    metrics = compute_metrics(labels, score)
    positive = labels == 1
    coverage = {
        "n_test": int(len(test_ids)), "empty_context_rate": float((detail["context_size"] == 0).mean()),
        "median_context_size": float(np.median(detail["context_size"])),
        "positive_empty_context_rate": float((detail["context_size"][positive] == 0).mean()) if positive.any() else None,
    }
    timing = {"fit_seconds": fit_timer.seconds, "score_seconds": score_timer.seconds,
              "throughput_transactions_per_second": len(test_ids) / max(score_timer.seconds, 1e-9)}
    write_json(out / "config.json", {**model.metadata(), "model": f"LocalLinkStreamESNIF_{variant}",
        "dataset": dataset, "seed": seed, "protocol": "chronological", "variant": variant,
        "families": list(families), "trees_per_family": trees_per_family, "total_trees": 300,
        "chronological_split": split_for(n_total), "labels_used_for_fit": False,
        "is_quick_run": False, "max_samples": None})
    write_json(out / "metrics.json", {"test": metrics, "context_coverage": coverage,
        "timing": timing, "status": "ok", "score_direction": "larger=more anomalous"})
    write_json(out / "timing.json", timing)
    save_scores(out / "scores.parquet", pl.DataFrame({
        "sample_id": test_ids, "timestamp_or_step": source.ts[test_ids], "label": labels,
        "label_available": source.label_available[test_ids], "anomaly_score": score,
        "split": np.full(len(test_ids), "test"), "departure_score": detail["departure"],
        "surprise_score": detail.get("surprise"), "recurrence_score": detail.get("recurrence"),
        "context_size": detail["context_size"],
    }))
    (out / "run.log").write_text(json.dumps({"status": "ok", "metrics": metrics,
        "coverage": coverage, "timing": timing}, indent=2), encoding="utf-8")


def run_ablation(workers: int, overwrite: bool) -> None:
    os.environ.update({"OMP_NUM_THREADS": str(workers), "MKL_NUM_THREADS": str(workers),
                       "OPENBLAS_NUM_THREADS": str(workers), "NUMEXPR_NUM_THREADS": str(workers)})
    for variant in VARIANTS:
        for dataset in ABLATION_DATASETS:
            for seed in SEEDS:
                run_ablation_one(dataset, seed, variant, overwrite)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "main", "ablation", "all"))
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true", help="replace only dedicated experiment outputs")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    preflight()
    record_environment(args.workers)
    write_json(OUT / "manifest.json", {
        "protocol": "unsupervised_txn_aml_v1", "datasets": list(DATASETS), "seeds": list(SEEDS),
        "methods": list(RUNNERS), "ablation_variants": {k: [list(v[0]), v[1]] for k, v in VARIANTS.items()},
        "script_write_policy": "forbidden", "created_at": datetime.now(timezone.utc).isoformat(),
    })
    if args.stage in ("main", "all"):
        run_main(args.workers, args.overwrite)
    if args.stage in ("ablation", "all"):
        run_ablation(args.workers, args.overwrite)


if __name__ == "__main__":
    run()
