#!/usr/bin/env python
"""Run every model on every dataset for every seed (task document Sec. 12.2).

Usage:
    /home/zhu/.conda/envs/IF/bin/python script/run_all_models.py \
        --datasets midas_darpa ibm_hi_small ibm_li_small paysim elliptic_pp \
        --models IF DIF MAD --seeds 42 43 44 45 46 --protocol chronological

Options: --overwrite --resume --device cpu|cuda --num-workers N
         --batch-size N --max-samples N --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zhu/.conda/envs/IF/bin/python"

DEFAULT_DATASETS = ["midas_darpa", "ibm_hi_small", "ibm_li_small", "paysim", "elliptic_pp"]
DEFAULT_MODELS = ["IF", "DIF", "MAD"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# model name -> runner script (relative to repo root)
RUNNERS = {
    "IF": "script/model/IF/runner.py",
    "DIF": "script/model/DIF/runner.py",
    "MAD": "script/model/MAD/runner.py",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--protocol", default="chronological")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="skip runs whose result directory already exists")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--quick", action="store_true",
                   help="smoke-test mode; results marked is_quick_run=true")
    p.add_argument("--query-mode", default=None,
                   help="MAD query mode (default: active-edge)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    unknown = [m for m in args.models if m not in RUNNERS]
    if unknown:
        sys.exit(f"unknown models: {unknown} (available: {sorted(RUNNERS)})")

    failures: list[tuple] = []
    total = len(args.datasets) * len(args.models) * len(args.seeds)
    done = 0
    t0 = time.time()
    # tqdm progress bar: updates once per completed run (subprocess blocks in
    # between), showing the current command and the running elapsed time.
    from tqdm import tqdm

    bar = tqdm(total=total, desc="runs", unit="run",
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                cmd = [PYTHON, str(REPO_ROOT / RUNNERS[model]),
                       "--dataset", dataset, "--protocol", args.protocol,
                       "--seed", str(seed), "--device", args.device]
                if args.overwrite:
                    cmd.append("--overwrite")
                if args.num_workers:
                    cmd += ["--num-workers", str(args.num_workers)]
                if args.batch_size:
                    cmd += ["--batch-size", str(args.batch_size)]
                if args.max_samples:
                    cmd += ["--max-samples", str(args.max_samples)]
                if args.quick:
                    cmd.append("--quick")
                if model == "MAD" and args.query_mode:
                    cmd += ["--query-mode", args.query_mode]

                done += 1
                bar.set_postfix_str(f"{model}/{dataset}/seed_{seed}")
                if args.resume and not args.quick:
                    from script.common.results import run_dir
                    out = run_dir(model, dataset, args.protocol, seed)
                    if (out / "metrics.json").exists():
                        bar.update(1)
                        continue
                rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
                bar.update(1)
                if rc != 0:
                    failures.append((model, dataset, seed))
                    bar.write(f"  FAILED rc={rc}: {model}/{dataset}/seed_{seed}")

    bar.close()
    print(f"\nfinished {total - len(failures)}/{total} runs in "
          f"{time.time() - t0:.1f}s; {len(failures)} failures")
    for f in failures:
        print("  failed:", f)


if __name__ == "__main__":
    main()
