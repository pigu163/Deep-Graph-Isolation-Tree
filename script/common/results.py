"""统一结果保存与汇总。

目录约定（任务文档 Sec. 13）：

    result/<MODEL>/<DATASET>/<PROTOCOL>/seed_<SEED>/
        config.json  metrics.json  scores.parquet  timing.json  warnings.json  run.log

`scores.parquet` 至少包含：sample_id, timestamp_or_step, label,
label_available, anomaly_score, split；模型可追加额外列。

总汇总写入 result/summary.csv 与 result/summary.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

__all__ = [
    "REPO_ROOT",
    "RESULT_ROOT",
    "run_dir",
    "write_json",
    "write_text",
    "save_scores",
    "append_summary_row",
    "write_summary",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "result"

SCORES_REQUIRED_COLUMNS = [
    "sample_id", "timestamp_or_step", "label", "label_available",
    "anomaly_score", "split",
]

SUMMARY_FIELDS = [
    "model", "dataset", "protocol", "seed",
    "roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "f1_at_k",
    "fit_seconds", "score_seconds", "throughput",
    "status", "paper_fidelity",
]


def run_dir(model: str, dataset: str, protocol: str, seed: int) -> Path:
    return RESULT_ROOT / model / dataset / protocol / f"seed_{seed}"


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_scores(path: Path, df: pl.DataFrame) -> None:
    """Validate and persist the scores table."""
    missing = [c for c in SCORES_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"scores.parquet missing required columns: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def append_summary_row(row: dict) -> None:
    """Append one summary row (all SUMMARY_FIELDS) to summary.csv/json."""
    row = {k: row.get(k) for k in SUMMARY_FIELDS}
    write_summary_rows([row], append=True)


def write_summary(rows: list[dict]) -> None:
    write_summary_rows(rows, append=False)


def _json_value(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def write_summary_rows(rows: list[dict], *, append: bool) -> None:
    normalized = [{k: _json_value(r.get(k)) for k in SUMMARY_FIELDS} for r in rows]
    if append and RESULT_ROOT.joinpath("summary.csv").exists():
        existing = pl.read_csv(RESULT_ROOT / "summary.csv")
        new = pl.DataFrame(normalized)
        df = pl.concat([existing, new], how="vertical")
    else:
        df = pl.DataFrame(normalized)
    df.write_csv(RESULT_ROOT / "summary.csv")

    # JSON: keep one JSON object per line (easy to append / stream).
    json_path = RESULT_ROOT / "summary.json"
    mode = "a" if append and json_path.exists() else "w"
    with open(json_path, mode, encoding="utf-8") as f:
        for r in normalized:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def default_log(path: Path) -> None:
    """Point the runner's log output at ``path`` (used by run_all_models)."""
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
