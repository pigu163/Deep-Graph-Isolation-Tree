#!/usr/bin/env python
"""Convert a raw tabular dataset into the canonical processed link-stream format.

Usage:
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py \
        --dataset midas_darpa
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py \
        --dataset ibm_aml --size SMALL --regime HI
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py \
        --dataset paysim
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py \
        --dataset elliptic_pp --elliptic-features none

Output layout (per dataset):
    <output-root>/<dataset>/
        events.parquet  nodes.parquet  metadata.json  splits.json
        temporal_index/{src,dst,node}_{offsets,event_ids}.npy timestamps.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkstream import preprocessing as pp
from linkstream import utils

DEFAULT_DATA_ROOT = "/home/zhu/Deep Graph Isolation Tree/data"
DEFAULT_OUTPUT_ROOT = "/home/zhu/Deep Graph Isolation Tree/data/processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        choices=sorted(pp.PREPROCESSORS),
        help="raw dataset to convert",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"directory containing raw data (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help=f"directory for processed data (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing processed output directory")
    parser.add_argument("--build-pair-index", action="store_true",
                        help="also build the directed (src, dst) pair index")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="fraction of events in the train split (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="fraction of events in the validation split (default: 0.0)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="reserved for future chunked CSV reading; lazy "
                             "scanning is used by default")
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "lz4", "gzip"],
                        help="Parquet compression (default: zstd)")
    parser.add_argument("--size", default="SMALL",
                        help="IBM AML size: SMALL | MEDIUM | LARGE (default: SMALL)")
    parser.add_argument("--regime", default="HI",
                        help="IBM AML regime: HI | LI (default: HI)")
    parser.add_argument("--elliptic-features", default="none", choices=["none", "all"],
                        help="Elliptic++ feature mode: none = structure-only "
                             "(default), all = include raw per-node features")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    utils.set_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.output_root)

    kwargs = dict(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        compression=args.compression,
        build_pair_index=args.build_pair_index,
        overwrite=args.overwrite,
    )
    if args.dataset == "ibm_aml":
        kwargs.update(size=args.size, regime=args.regime)
    if args.dataset == "elliptic_pp":
        kwargs.update(features_mode=args.elliptic_features)

    meta = pp.PREPROCESSORS[args.dataset](data_root, out_root, **kwargs)

    print(f"[preprocess] dataset       : {meta['dataset_name']}")
    print(f"[preprocess] num events    : {meta['num_events']}")
    print(f"[preprocess] num nodes     : {meta['num_nodes']}")
    print(f"[preprocess] labeled events: {meta['num_labeled_events']} "
          f"(positive: {meta['num_positive_labels']})")
    print(f"[preprocess] timestamps    : [{meta['timestamp_min']:.3g}, {meta['timestamp_max']:.3g}]")
    print(f"[preprocess] edge features : {len(meta['edge_feature_names'])} "
          f"{meta['edge_feature_names'] if meta['edge_feature_names'] else ''}")
    print(f"[preprocess] splits        : {meta['splits']}")
    print(f"[preprocess] output        : {Path(out_root) / meta['dataset_name']}")


if __name__ == "__main__":
    main()
