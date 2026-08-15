#!/usr/bin/env python
"""Demonstrate PyTorch DataLoader loading of a processed link stream.

Prints dataset stats, batch shapes, label distribution and loading
throughput (events/second) for base loading and, optionally, for loading
with temporal-context extraction.

Usage:
    /home/zhu/.conda/envs/GRAG/bin/python script/dataset/demo_dataloader.py \
        --dataset paysim --batch-size 1024 --num-workers 0 --seed 42
    ... --with-context --num-hops 1 --max-neighbors 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from linkstream import schema, utils
from linkstream.collate import TemporalContextCollator, make_link_stream_loader
from linkstream.datasets import LinkStreamDataset
from linkstream.temporal_index import TemporalNeighborIndex

DEFAULT_OUTPUT_ROOT = "/home/zhu/Deep Graph Isolation Tree/data/processed"


def _report_dataset(dataset: LinkStreamDataset) -> None:
    print(f"dataset name        : {dataset.dataset_name}")
    print(f"number of events    : {len(dataset)}")
    print(f"number of nodes     : {dataset.num_nodes}")
    print(f"edge feature dim    : {dataset.edge_feature_dim} "
          f"{dataset.edge_feature_names if dataset.edge_feature_names else ''}")
    print(f"first timestamp     : {dataset.ts.min():.6g}")
    print(f"last timestamp      : {dataset.ts.max():.6g}")


def _report_batch(batch) -> None:
    print("batch tensor shapes:")
    print(f"  event_id      : {tuple(batch.event_id.shape)} {batch.event_id.dtype}")
    print(f"  src           : {tuple(batch.src.shape)} {batch.src.dtype}")
    print(f"  dst           : {tuple(batch.dst.shape)} {batch.dst.dtype}")
    print(f"  ts            : {tuple(batch.ts.shape)} {batch.ts.dtype}")
    print(f"  label         : {tuple(batch.label.shape)} {batch.label.dtype}")
    print(f"  label_avail   : {tuple(batch.label_available.shape)} {batch.label_available.dtype}")
    print(f"  edge_feat     : "
          f"{'None' if batch.edge_feat is None else tuple(batch.edge_feat.shape)}")


def _report_labels(dataset: LinkStreamDataset) -> None:
    labels = torch.from_numpy(dataset.label)
    avail = torch.from_numpy(dataset.label_available)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    unk = int((~avail).sum())
    print(f"label distribution : positive={pos} negative={neg} unknown={unk}")


def _throughput(loader, desc: str) -> float:
    start = time.perf_counter()
    total = 0
    for batch in loader:
        total += len(batch.event_id)
    elapsed = time.perf_counter() - start
    rate = total / elapsed if elapsed > 0 else float("inf")
    print(f"loading throughput ({desc}): {rate:,.0f} events/second "
          f"({total} events in {elapsed:.2f}s)")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=schema.PROCESSED_DATASET_NAMES)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--time-window", type=float, default=7 * 24 * 60 * 60)
    args = parser.parse_args()

    utils.set_seed(args.seed)
    processed_dir = Path(args.output_root) / args.dataset
    if not processed_dir.exists():
        print(f"error: {processed_dir} does not exist; run preprocess.py first")
        sys.exit(1)

    dataset = LinkStreamDataset(processed_dir, split="all")
    _report_dataset(dataset)
    _report_labels(dataset)

    # ---- base event loading --------------------------------------------------
    loader = make_link_stream_loader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    batch = next(iter(loader))
    _report_batch(batch)
    _throughput(loader, "base event loading")

    # ---- optional temporal-context loading -----------------------------------
    if args.with_context:
        index = TemporalNeighborIndex.load(processed_dir / schema.TEMPORAL_INDEX_DIR)
        collator = TemporalContextCollator(
            index, dataset=dataset, num_hops=args.num_hops,
            max_neighbors=args.max_neighbors, time_window=args.time_window)
        ctx_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collator)
        _throughput(ctx_loader,
                    f"loading with {args.num_hops}-hop temporal context")

        # Show a sample context and verify the no-leakage rule.
        out = collator([dataset[i] for i in range(min(8, len(dataset)))])
        n = sum(len(c.event_ids) for c in out["contexts"])
        print(f"context demo: {len(out['contexts'])} events, "
              f"{n} total historical events (avg {n / max(len(out['contexts']), 1):.1f})")
        for i, c in enumerate(out["contexts"]):
            eid = int(out["events"].event_id[i])
            assert torch.all(c.event_ids < eid), \
                f"temporal leakage detected in context of event {eid}"
        print("context demo: no temporal leakage (all history ids < event ids)")


if __name__ == "__main__":
    main()
