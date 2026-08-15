#!/usr/bin/env python3
"""Validate and summarize outputs from UNSUPERVISED_TRANSACTION_AML_PROTOCOL.md."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments_result" / "unsupervised_txn_aml_v1"
SUMMARY = OUT / "summary"
DATASETS = ("ibm_hi_small", "ibm_li_small", "paysim")
SEEDS = {42, 43, 44}
MAIN_METHODS = {"IF", "EIF", "DIF", "ECOD", "FFADE", "MAD", "SLADE", "LocalLinkStreamESNIF"}
TABULAR = {"IF", "EIF", "DIF", "ECOD"}
LINK_STREAM = {"FFADE", "MAD", "SLADE"}
REQUIRED = {"sample_id", "timestamp_or_step", "label", "label_available", "anomaly_score", "split"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def expected_test(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    root = ROOT / "data" / "processed" / dataset
    split = json.loads((root / "splits.json").read_text(encoding="utf-8"))["test"]
    expected = pl.scan_parquet(root / "events.parquet").filter(
        pl.col("event_id").is_between(split[0], split[1], closed="left")
    ).select(["event_id", "label"]).collect().sort("event_id")
    return expected["event_id"].to_numpy(), expected["label"].to_numpy()


def result_dirs() -> list[tuple[str, Path, str]]:
    found = []
    for section in ("runs", "ablations"):
        base = OUT / section
        if not base.exists():
            continue
        for scores in base.glob("*/*/chronological/seed_*/scores.parquet"):
            found.append((section, scores.parent, scores.relative_to(base).parts[0]))
    return sorted(found, key=lambda item: str(item[1]))


def validate_one(section: str, run_dir: Path, model: str,
                 cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[dict, np.ndarray, np.ndarray]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset, seed = config["dataset"], int(config["seed"])
    if config.get("protocol", "chronological") != "chronological":
        raise ValueError(f"{run_dir}: non-chronological protocol")
    if config.get("is_quick_run") or config.get("max_samples") is not None or config.get("max_samples_used") is not None:
        raise ValueError(f"{run_dir}: quick/subsampled output is not formal")
    if config.get("labels_used_for_fit") is True:
        raise ValueError(f"{run_dir}: labels_used_for_fit=true")
    scores = pl.read_parquet(run_dir / "scores.parquet")
    missing = REQUIRED - set(scores.columns)
    if missing:
        raise ValueError(f"{run_dir}: missing score fields {sorted(missing)}")
    scores = scores.filter(pl.col("split") == "test").sort("sample_id")
    sample, labels = cache.setdefault(dataset, expected_test(dataset))
    got_ids, got_labels = scores["sample_id"].to_numpy(), scores["label"].to_numpy()
    values = scores["anomaly_score"].to_numpy().astype(np.float64)
    if not np.array_equal(got_ids, sample):
        raise ValueError(f"{run_dir}: test sample_id is not the canonical 30% range")
    if not np.array_equal(got_labels, labels):
        raise ValueError(f"{run_dir}: test labels differ from canonical events")
    if not np.isfinite(values).all():
        raise ValueError(f"{run_dir}: non-finite anomaly scores")
    from script.common.metrics import compute_metrics
    m = compute_metrics(labels, values)
    timing = json.loads((run_dir / "timing.json").read_text(encoding="utf-8"))
    context = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")).get("context_coverage", {})
    row = {
        "section": section, "model": model, "dataset": dataset, "seed": seed,
        "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "precision_at_k": m["precision_at_k"],
        "recall_at_k": m["recall_at_k"], "f1_at_k": m["f1_at_k"], "n_eval": m["n_eval"], "n_pos": m["n_pos"],
        "precision_at_0.1pct": m["top_k"]["0.001"]["precision"], "recall_at_0.1pct": m["top_k"]["0.001"]["recall"],
        "lift_at_0.1pct": m["top_k"]["0.001"]["lift"], "precision_at_0.5pct": m["top_k"]["0.005"]["precision"],
        "recall_at_0.5pct": m["top_k"]["0.005"]["recall"], "lift_at_0.5pct": m["top_k"]["0.005"]["lift"],
        "precision_at_1pct": m["top_k"]["0.01"]["precision"], "recall_at_1pct": m["top_k"]["0.01"]["recall"],
        "lift_at_1pct": m["top_k"]["0.01"]["lift"], "fit_seconds": timing.get("fit_seconds"),
        "score_seconds": timing.get("score_seconds"), "throughput": timing.get("throughput", timing.get("throughput_transactions_per_second")),
        "empty_context_rate": context.get("empty_context_rate"), "median_context_size": context.get("median_context_size"),
    }
    return row, labels, values


def aggregate(rows: list[dict], columns: list[str]) -> pl.DataFrame:
    return (pl.DataFrame(rows).group_by(["section", "model", "dataset"]).agg(
        pl.len().alias("n_seeds"), *[pl.col(c).mean().alias(f"{c}_mean") for c in columns],
        *[pl.col(c).std(ddof=1).fill_null(0.0).alias(f"{c}_std") for c in columns]
    ).sort(["section", "dataset", "pr_auc_mean"], descending=[False, False, True]))


def pr_figure(score_cache: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]], summary: pl.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        table = summary.filter((pl.col("section") == "runs") & (pl.col("dataset") == dataset))
        best_tab = table.filter(pl.col("model").is_in(TABULAR)).sort("pr_auc_mean", descending=True)[0, "model"]
        best_link = table.filter(pl.col("model").is_in(LINK_STREAM)).sort("pr_auc_mean", descending=True)[0, "model"]
        for model, style in (("LocalLinkStreamESNIF", "-"), (best_tab, "--"), (best_link, ":")):
            runs = score_cache[(dataset, model)]
            y = runs[0][0]
            mean_score = np.mean(np.stack([score for _, score in runs]), axis=0)
            precision, recall, _ = precision_recall_curve(y, mean_score)
            axis.plot(recall, precision, style, linewidth=2, label=model)
        axis.axhline(float(y.mean()), color="0.5", linewidth=1, label="prevalence")
        axis.set(title=dataset, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, None))
        axis.legend(fontsize=8)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUMMARY / "pr_curves.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true", help="permit incomplete matrices for debugging only")
    args = parser.parse_args()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    score_cache: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    failures: list[str] = []
    for section, run_dir, model in result_dirs():
        try:
            row, y, score = validate_one(section, run_dir, model, cache)
            rows.append(row)
            if section == "runs":
                score_cache[(row["dataset"], model)].append((y, score))
        except Exception as exc:
            failures.append(f"{run_dir}: {exc}")
    if failures:
        write_json(SUMMARY / "validation_report.json", {"status": "failed", "failures": failures})
        raise SystemExit("\n".join(failures))
    if not rows:
        raise SystemExit("no experiment results found")
    observed = {(r["section"], r["model"], r["dataset"], r["seed"]) for r in rows}
    expected = {("runs", m, d, s) for m in MAIN_METHODS for d in DATASETS for s in SEEDS}
    if not args.allow_partial and not expected.issubset(observed):
        missing = sorted(expected - observed)
        raise SystemExit(f"incomplete main matrix ({len(missing)} missing); use --allow-partial only for debugging")
    numeric = [c for c in rows[0] if c not in {"section", "model", "dataset", "seed"}]
    long = pl.DataFrame(rows).sort(["section", "dataset", "model", "seed"])
    summary = aggregate(rows, numeric)
    main_table, ablation_table = summary.filter(pl.col("section") == "runs"), summary.filter(pl.col("section") == "ablations")
    SUMMARY.mkdir(parents=True, exist_ok=True)
    long.write_csv(SUMMARY / "metrics_long.csv")
    main_table.write_csv(SUMMARY / "main_table.csv")
    ablation_table.write_csv(SUMMARY / "ablation_table.csv")
    write_json(SUMMARY / "summary.json", {"status": "ok", "n_runs": len(rows),
        "primary_metrics": ["pr_auc", "recall_at_0.5pct"], "seeds": sorted(SEEDS)})
    write_json(SUMMARY / "validation_report.json", {"status": "ok", "validated_runs": len(rows),
        "checks": ["chronological protocol", "not quick/subsampled", "canonical test IDs", "canonical labels", "finite scores", "required schema"]})
    if expected.issubset(observed):
        pr_figure(score_cache, summary)
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
