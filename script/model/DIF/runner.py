"""DIF runner CLI.

Usage (task doc Sec. 12):

    /home/zhu/.conda/envs/IF/bin/python script/model/DIF/runner.py \
        --dataset paysim --protocol chronological --seed 42 [--overwrite] \
        [--implementation faithful|simplified] [--num-workers N] \
        [--batch-size N] [--max-samples N] [--quick] [--device cpu]

Outputs (task doc Sec. 13):

    result/DIF/<DATASET>/<PROTOCOL>/seed_<SEED>/
        config.json metrics.json scores.parquet timing.json warnings.json run.log

`--implementation simplified` disables CERE (independent random MLPs) and
must be tagged `paper_fidelity="simplified"`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT, REPO_ROOT / "script"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from script.common.metrics import compute_metrics  # noqa: E402
from script.common.results import run_dir, save_scores, write_json  # noqa: E402
from script.common.seeds import set_seed  # noqa: E402
from script.common.timing import Timer  # noqa: E402
from script.common.validation import tabular_interface_check  # noqa: E402
from script.tabular_pipeline.datasets import load_tabular_dataset  # noqa: E402

MODEL = "DIF"

# Paper hyper-parameters (paper Sec. 5.3/6.1.3 + author code) and the
# documented deviations (paper does not specify network structure).
DEFAULT_CONFIG = {
    "n_representations": 50,
    "trees_per_representation": 6,
    "total_trees": 300,
    "subsample_size": 256,
    "max_depth": 8,               # ceil(log2(256)) iForest convention
    "hidden_dim": [500, 100],     # deviation: paper leaves architecture open;
    "rep_dim": 20,                #  values follow the authors' public DIF code.
    "activation": "tanh",
    "weight_init": "normal(0,1)",
    "bias": False,
    "batch_size": 4096,          # scalable default; author code used 64
    "score_batch_size": 4096,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DIF runner")
    p.add_argument("--dataset", required=True,
                   choices=["midas_darpa", "ibm_hi_small", "ibm_li_small",
                            "paysim", "elliptic_pp"])
    p.add_argument("--protocol", default="chronological",
                   choices=["chronological", "transductive"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-workers", type=int, default=0,
                   help="parallel JIT representation traversals; 0/1 is sequential")
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                   help="CERE encoding batch size (author-code default: 64)")
    p.add_argument("--score-batch-size", type=int,
                   default=DEFAULT_CONFIG["score_batch_size"],
                   help="DIF scoring batch size (default: 4096)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="restrict scoring to the first N samples (smoke tests)")
    p.add_argument("--quick", action="store_true", help="smoke run flag")
    p.add_argument("--implementation", default="faithful",
                   choices=["faithful", "simplified"])
    p.add_argument("--n-representations", type=int, default=DEFAULT_CONFIG["n_representations"])
    p.add_argument("--trees-per-representation", type=int,
                   default=DEFAULT_CONFIG["trees_per_representation"])
    p.add_argument("--subsample-size", type=int, default=DEFAULT_CONFIG["subsample_size"])
    p.add_argument("--max-depth", type=int, default=DEFAULT_CONFIG["max_depth"])
    p.add_argument("--hidden-dim", type=int, nargs="+", default=None)
    p.add_argument("--rep-dim", type=int, default=DEFAULT_CONFIG["rep_dim"])
    p.add_argument("--no-deas", action="store_true",
                   help="disable DEAS (depth-only ablation score)")
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
    log_path = out_dir / "run.log"

    set_seed(args.seed)
    data = load_tabular_dataset(args.dataset, split="all")

    if args.max_samples is not None:
        n = min(args.max_samples, data.X.shape[0])
        data.X = data.X[:n]
        data.y = data.y[:n]
        data.label_available = data.label_available[:n]
        data.sample_ids = data.sample_ids[:n]
        data.timestamps = data.timestamps[:n]
    n_total = data.X.shape[0]
    splits = _split_indices(n_total)

    cere = args.implementation == "faithful"
    deas = not args.no_deas
    hidden_dim = args.hidden_dim if args.hidden_dim is not None \
        else DEFAULT_CONFIG["hidden_dim"]
    paper_fidelity = "paper_guided_temporal_reimplementation" if cere else "simplified"

    from script.model.DIF.dif import DIF

    fit_timer = Timer()
    with fit_timer:
        model = DIF(
            n_representations=args.n_representations,
            trees_per_representation=args.trees_per_representation,
            max_samples=args.subsample_size,
            max_depth=args.max_depth,
            hidden_dim=hidden_dim,
            rep_dim=args.rep_dim,
            activation=DEFAULT_CONFIG["activation"],
            cere=cere,
            deas=deas,
            random_state=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            score_batch_size=args.score_batch_size,
            num_workers=args.num_workers,
        )
        fit_X = data.X if args.protocol == "transductive" else data.X[splits["train"][0]:splits["train"][1]]
        model.fit(fit_X)

    score_timer = Timer()
    with score_timer:
        scores = model.score(data.X)
    throughput = n_total / max(score_timer.seconds, 1e-9)

    warnings_all: list[str] = []
    metrics = {}
    for name, (s, e) in splits.items():
        m = compute_metrics(data.y[s:e], scores[s:e])
        warnings_all.extend(m.pop("warnings", []))
        metrics[name] = m
    metrics["status"] = "ok"
    metrics["paper_fidelity"] = paper_fidelity
    metrics["timing"] = {
        "fit_seconds": round(fit_timer.seconds, 6),
        "score_seconds": round(score_timer.seconds, 6),
        "throughput": round(throughput, 3),
    }

    model_cfg = model.config()
    profile = model.profile()
    config = {
        "model": MODEL,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "seed": args.seed,
        "device": args.device,
        "implementation_mode": args.implementation,
        "paper_fidelity": paper_fidelity,
        "cere_enabled": cere,
        "deas_enabled": deas,
        "is_quick_run": bool(args.quick),
        "batch_size": args.batch_size,
        "score_batch_size": args.score_batch_size,
        "num_workers": args.num_workers,
        "max_samples_used": args.max_samples,
        "n_representations": args.n_representations,
        "trees_per_representation": args.trees_per_representation,
        "total_trees": args.n_representations * args.trees_per_representation,
        "subsample_size": args.subsample_size,
        "max_depth": args.max_depth,
        "hidden_dim": list(hidden_dim),
        "rep_dim": args.rep_dim,
        "activation": model_cfg.get("activation"),
        "encoder_architecture": model_cfg.get("encoder", {}),
        "deviation_notes": [
            "Paper specifies r=50, t=6, n=256 and typical J=8; author code uses b=64, "
            "while this scalable runner defaults to b=4096. The paper does NOT specify "
            "the MLP depth/width/activation/initialisation. We use the "
            "authors' public DIF code defaults (hidden [500,100], rep 20, tanh, "
            "weights N(0,1), no bias) as a documented fallback.",
            "Per-representation z-score + tanh post-processing follows the "
            "authors' implementation; scaler statistics are fitted exactly on the complete "
            "model-fit split, streamed in batches (no test leakage in chronological mode).",
            "DEAS follows paper Eq. (8): F = 2^{-E|p|/C} * E[g], C = c(256). "
            "No leaf-size path correction is added.",
            "Paper aggregates over all trees (Eq. 8); no per-representation "
            "score averaging is used.",
            "This custom-DIT raw-path implementation is not a byte-for-byte reproduction "
            "of the authors' sklearn IsolationForest plus leaf-size correction.",
            "elliptic_pp uses transaction-node samples; features 'local' "
            "subset by default (93 local features, see "
            "script/tabular_pipeline/schema.py and result metadata.json).",
        ],
        "score_direction": "larger=more anomalous",
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "timing.json", {
        "fit_seconds": fit_timer.seconds,
        "score_seconds": score_timer.seconds,
        "throughput": throughput,
    })
    write_json(out_dir / "profile.json", profile)
    write_json(out_dir / "warnings.json", {"warnings": warnings_all})

    split_arr = np.empty(n_total, dtype=object)
    for name, (s, e) in splits.items():
        split_arr[s:e] = name
    df = pl.DataFrame({
        "sample_id": data.sample_ids,
        "timestamp_or_step": data.timestamps,
        "label": data.y,
        "label_available": data.label_available,
        "anomaly_score": scores,
        "split": split_arr,
        # DIF-specific metadata columns (task doc Sec. 13)
        "implementation_mode": args.implementation,
        "num_representations": args.n_representations,
        "trees_per_representation": args.trees_per_representation,
        "total_trees": args.n_representations * args.trees_per_representation,
        "subsample_size": args.subsample_size,
        "encoder_architecture": str(hidden_dim),
        "cere_enabled": cere,
        "deas_enabled": deas,
    })
    save_scores(out_dir / "scores.parquet", df)

    # ---- data-interface check (task Sec. 10.4 / 14) ----------------------
    write_json(out_dir / "interface_check.json", tabular_interface_check(
        data.X, data.y, data.label_available, data.sample_ids,
        data.feature_names, scores,
        fitted_on="all" if args.protocol == "transductive" else "train"))

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"DIF runner | dataset={args.dataset} protocol={args.protocol} "
                  f"seed={args.seed} implementation={args.implementation}\n")
        log.write(f"X shape: {data.X.shape}\n")
        log.write(f"fit={fit_timer.seconds:.2f}s score={score_timer.seconds:.2f}s "
                  f"throughput={throughput:.1f}\n")
        log.write(json.dumps({
            "status": "ok",
            "paper_fidelity": paper_fidelity,
            "test_roc_auc": metrics["test"].get("roc_auc"),
            "test_pr_auc": metrics["test"].get("pr_auc"),
            "test_n_eval": metrics["test"].get("n_eval"),
        }, indent=2))
        log.write("\n")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
