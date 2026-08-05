#!/usr/bin/env python
"""Aggregate result/<MODEL>/<DATASET>/<PROTOCOL>/seed_* runs into summary tables.

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/evaluate_all.py [--protocol chronological]

Reads every metrics.json under result/ and writes:
    result/summary.csv   (per-seed rows)
    result/summary.json  (same rows, JSON lines)
    result/README.md     (mean/std over seeds per model-dataset)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common.results import RESULT_ROOT, SUMMARY_FIELDS, write_summary  # noqa: E402

MEAN_STD_FIELDS = [
    "roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "f1_at_k",
    "fit_seconds", "score_seconds", "throughput",
]


def collect_rows(protocol: str) -> list[dict]:
    rows = []
    for metrics_path in sorted(RESULT_ROOT.glob(f"*/*/{protocol}/seed_*/metrics.json")):
        parts = metrics_path.parts
        model, dataset, seed = parts[-5], parts[-4], parts[-2].removeprefix("seed_")
        # quick runs (is_quick_run=true) must never enter the formal summary
        cfg_path = metrics_path.parent / "config.json"
        if cfg_path.exists():
            try:
                if json.loads(cfg_path.read_text()).get("is_quick_run"):
                    continue
            except Exception:
                pass
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        test = m.get("test", {})
        timing = m.get("timing", {})
        row = {
            "model": model,
            "dataset": dataset,
            "protocol": protocol,
            "seed": int(seed),
            "roc_auc": test.get("roc_auc"),
            "pr_auc": test.get("pr_auc"),
            "precision_at_k": test.get("precision_at_k"),
            "recall_at_k": test.get("recall_at_k"),
            "f1_at_k": test.get("f1_at_k"),
            "fit_seconds": timing.get("fit_seconds"),
            "score_seconds": timing.get("score_seconds"),
            "throughput": timing.get("throughput"),
            "status": m.get("status", "ok"),
            "paper_fidelity": m.get("paper_fidelity"),
        }
        rows.append(row)
    return rows


def write_mean_std_table(rows: list[dict], protocol: str) -> None:
    """Mean/std over seeds, grouped by (model, dataset)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["status"] != "ok":
            continue
        groups[(r["model"], r["dataset"])].append(r)

    lines = [
        f"# Result summary (protocol: {protocol})",
        "",
        "Grouped by (model, dataset); values are mean ± std over seeds "
        "for the **test** split. Null metrics (single-class splits) are excluded.",
        "",
        "| model | dataset | seeds | roc_auc | pr_auc | p@k | r@k | f1@k | fit_s | score_s | throughput |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, dataset), runs in sorted(groups.items()):
        n = len(runs)
        cells = []
        for field in MEAN_STD_FIELDS:
            vals = [r[field] for r in runs if r[field] is not None]
            if not vals:
                cells.append("-")
                continue
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                std = var ** 0.5
                cells.append(f"{mean:.4f}±{std:.4f}")
            else:
                cells.append(f"{mean:.4f}")
        lines.append(f"| {model} | {dataset} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    (RESULT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="chronological")
    args = ap.parse_args()
    rows = collect_rows(args.protocol)
    if not rows:
        sys.exit(f"no runs found under {RESULT_ROOT}/<model>/{args.protocol}/seed_*/")
    write_summary(rows)
    write_mean_std_table(rows, args.protocol)
    print(f"wrote {len(rows)} rows -> {RESULT_ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
