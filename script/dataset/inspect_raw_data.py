#!/usr/bin/env python
"""Inspect raw dataset files: paths, sizes, columns, dtypes, samples, nulls.

Usage:
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/inspect_raw_data.py \
        --dataset paysim --num-rows 5
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/inspect_raw_data.py \
        --dataset ibm_aml --size SMALL --regime HI --num-rows 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from linkstream import utils

DEFAULT_DATA_ROOT = "/home/zhu/Deep Graph Isolation Tree/data"

# dataset -> list of (label, path) pairs; size/regime are resolved below.
_RAW_FILES = {
    "midas_darpa": lambda d: [
        ("processed events", d / "midas_darpa" / "darpa_processed.csv"),
        ("ground truth", d / "midas_darpa" / "darpa_ground_truth.csv"),
        ("shape", d / "midas_darpa" / "darpa_shape.txt"),
    ],
    "ibm_aml": lambda d, size="SMALL", regime="HI": [
        ("transactions", d / "ibm_aml" / size / f"{regime}-{size}_Trans.csv"),
        ("accounts", d / "ibm_aml" / size / f"{regime}-{size}_accounts.csv"),
    ],
    "paysim": lambda d: [("transactions", d / "paysim" / "paysim.csv")],
    "elliptic_pp": lambda d: [
        ("classes", d / "elliptic_pp" / "txs_classes.csv"),
        ("edgelist", d / "elliptic_pp" / "txs_edgelist.csv"),
        ("features", d / "elliptic_pp" / "txs_features.csv"),
    ],
}

# (column, csv path) to report label value counts when available.
_LABEL_COLUMNS = {
    "midas_darpa": ("label", "darpa_ground_truth.csv"),
    "paysim": ("isFraud", "paysim.csv"),
    "ibm_aml": ("Is Laundering", "{regime}-{size}_Trans.csv"),
    "elliptic_pp": ("class", "txs_classes.csv"),
}


def _human_size(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _inspect_csv(path: Path, num_rows: int) -> None:
    if utils.is_lfs_pointer(path):
        print(f"    !! Git LFS pointer (real data not checked out): "
              f"{path.read_text()[:80].strip()!r}")
        return

    size = path.stat().st_size
    print(f"  file      : {path}")
    print(f"  size      : {_human_size(size)}")

    schema = pl.scan_csv(path).collect_schema()
    print(f"  columns   : {list(schema.names())}")
    print(f"  dtypes    : {[str(schema[d]) for d in schema.names()]}")

    sample = pl.scan_csv(path, n_rows=max(num_rows, 1)).collect()
    print(f"  first rows ({min(num_rows, len(sample))}):")
    for row in sample.head(num_rows).to_dicts():
        print(f"    {row}")

    # Cheap null estimate over a bounded sample.
    sample_nulls = pl.scan_csv(path, n_rows=10000).collect().null_count()
    print(f"  nulls (first 10k rows): "
          f"{dict((c, int(sample_nulls[c][0])) for c in sample_nulls.columns)}")

    # Estimated row count from average line length (order of magnitude).
    with open(path, "rb") as f:
        first = f.readline()
        f.seek(0, 2)
        total = f.tell()
    if first:
        est = int(total / max(len(first), 1))
        print(f"  est. rows : ~{est:,} (by average line length)")


def _inspect_shape_txt(path: Path) -> None:
    print(f"  file      : {path}")
    print(f"  content   : {path.read_text().strip()!r} (expected: number of events)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(_RAW_FILES))
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--size", default="SMALL", help="IBM AML size")
    parser.add_argument("--regime", default="HI", help="IBM AML regime (HI/LI)")
    parser.add_argument("--num-rows", type=int, default=5)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.dataset == "ibm_aml":
        files = _RAW_FILES[args.dataset](data_root, args.size.upper(), args.regime.upper())
    else:
        files = _RAW_FILES[args.dataset](data_root)

    print(f"== dataset: {args.dataset} ==")
    for label, path in files:
        print(f"-- {label} --")
        if not path.exists():
            print(f"  !! missing: {path}")
            continue
        if path.suffix == ".txt":
            _inspect_shape_txt(path)
        else:
            _inspect_csv(path, args.num_rows)

    # Label value counts when the dataset provides a label column.
    if args.dataset in _LABEL_COLUMNS:
        col, fname = _LABEL_COLUMNS[args.dataset]
        if args.dataset == "ibm_aml":
            fname = fname.format(regime=args.regime.upper(), size=args.size.upper())
        path = data_root / {
            "midas_darpa": "midas_darpa", "paysim": "paysim",
            "ibm_aml": "ibm_aml", "elliptic_pp": "elliptic_pp",
        }[args.dataset] / fname
        if path.exists() and not utils.is_lfs_pointer(path):
            if args.dataset == "midas_darpa":
                # headerless single-column file (0/1 per row)
                col_data = pl.scan_csv(
                    path, has_header=False, new_columns=[col]).select(col).collect()
            else:
                col_data = pl.scan_csv(path).select(col).collect()
            counts = col_data[col].value_counts()
            print(f"-- label distribution ({col}) --")
            print(counts)


if __name__ == "__main__":
    main()
