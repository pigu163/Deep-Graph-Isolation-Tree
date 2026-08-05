"""Tests for the PyTorch Dataset / DataLoader layer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from linkstream import schema
from linkstream.collate import (
    TemporalContextCollator,
    collate_link_stream,
    make_link_stream_loader,
)
from linkstream.datasets import LinkStreamDataset
from linkstream.temporal_index import TemporalNeighborIndex
from tests import preprocess_midas


def build_processed(tmp: str) -> Path:
    root = Path(tmp) / "data"
    out_root = Path(tmp) / "out"
    meta = preprocess_midas(root, out_root, build_pair_index=True)
    return out_root / meta["dataset_name"]


class TestLinkStreamDataset(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.processed_dir = build_processed(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_len_and_item(self):
        ds = LinkStreamDataset(self.processed_dir, split="all")
        self.assertEqual(len(ds), 4)
        item = ds[0]
        self.assertEqual(set(item.keys()),
                         {"event_id", "src", "dst", "ts", "label",
                          "label_available", "edge_feat"})
        self.assertEqual(item["event_id"], 0)
        self.assertIsNone(item["edge_feat"])  # midas has no features

    def test_split_ranges(self):
        ds = LinkStreamDataset(self.processed_dir, split="train")
        self.assertEqual(len(ds), 2)
        self.assertEqual([ds[i]["event_id"] for i in range(len(ds))], [0, 1])
        ds = LinkStreamDataset(self.processed_dir, split="test")
        self.assertEqual([ds[i]["event_id"] for i in range(len(ds))], [3])

    def test_unknown_split_raises(self):
        with self.assertRaises(KeyError):
            LinkStreamDataset(self.processed_dir, split="nope")

    def test_edge_features_off(self):
        ds = LinkStreamDataset(self.processed_dir, include_edge_features=False)
        self.assertEqual(ds.edge_feature_dim, 0)


class TestCollate(unittest.TestCase):
    def test_collate_preserves_order_and_dtypes(self):
        samples = [
            {"event_id": 0, "src": 1, "dst": 2, "ts": 1.0, "label": 0,
             "label_available": True, "edge_feat": None},
            {"event_id": 1, "src": 3, "dst": 4, "ts": 2.0, "label": 1,
             "label_available": True, "edge_feat": None},
        ]
        batch = collate_link_stream(samples)
        self.assertEqual(batch.event_id.tolist(), [0, 1])
        self.assertEqual(batch.src.tolist(), [1, 3])
        self.assertEqual(batch.ts.tolist(), [1.0, 2.0])
        self.assertEqual(batch.src.dtype, torch.long)
        self.assertEqual(batch.ts.dtype, torch.float32)
        self.assertEqual(batch.label.dtype, torch.long)
        self.assertEqual(batch.label_available.dtype, torch.bool)
        self.assertIsNone(batch.edge_feat)


class TestLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.processed_dir = build_processed(self._tmp.name)
        self.dataset = LinkStreamDataset(self.processed_dir, split="all")

    def tearDown(self):
        self._tmp.cleanup()

    def test_configurable_batch_size(self):
        for bs in (1, 2, 3, 4):
            loader = make_link_stream_loader(self.dataset, batch_size=bs)
            batches = list(loader)
            sizes = [len(b.event_id) for b in batches]
            self.assertEqual(sizes, [bs] * (4 // bs) + ([4 % bs] if 4 % bs else []))

    def test_no_shuffle_event_ids_increasing(self):
        loader = make_link_stream_loader(self.dataset, batch_size=2)
        all_ids = []
        for batch in loader:
            all_ids.extend(batch.event_id.tolist())
        self.assertEqual(all_ids, [0, 1, 2, 3])

    def test_incomplete_final_batch_kept_by_default(self):
        loader = make_link_stream_loader(self.dataset, batch_size=3)
        batches = list(loader)
        self.assertEqual([len(b.event_id) for b in batches], [3, 1])

    def test_drop_last(self):
        loader = make_link_stream_loader(self.dataset, batch_size=3, drop_last=True)
        batches = list(loader)
        self.assertEqual([len(b.event_id) for b in batches], [3])

    def test_shapes_and_label_alignment(self):
        loader = make_link_stream_loader(self.dataset, batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(batch.src.shape, (2,))
        self.assertEqual(batch.dst.shape, (2,))
        self.assertEqual(batch.ts.shape, (2,))
        self.assertEqual(batch.label.shape, (2,))
        for i in range(2):
            item = self.dataset[i]
            self.assertEqual(int(batch.label[i]), item["label"])
            self.assertEqual(int(batch.src[i]), item["src"])
            self.assertEqual(int(batch.event_id[i]), item["event_id"])


class TestContextCollator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.processed_dir = build_processed(self._tmp.name)
        self.dataset = LinkStreamDataset(self.processed_dir, split="all")
        self.index = TemporalNeighborIndex.load(
            self.processed_dir / schema.TEMPORAL_INDEX_DIR)

    def tearDown(self):
        self._tmp.cleanup()

    def test_context_batch_no_leakage(self):
        collator = TemporalContextCollator(
            self.index, dataset=self.dataset, num_hops=1, max_neighbors=20)
        samples = [self.dataset[i] for i in range(4)]
        out = collator(samples)
        batch, contexts = out["events"], out["contexts"]
        self.assertEqual(len(contexts), 4)
        for c, eid in zip(contexts, batch.event_id.tolist()):
            self.assertTrue(torch.all(c.event_ids < eid),
                            f"leak in context of event {eid}")

    def test_requires_dataset(self):
        collator = TemporalContextCollator(self.index, num_hops=1, max_neighbors=5)
        with self.assertRaises(RuntimeError):
            collator([self.dataset[0]])


if __name__ == "__main__":
    unittest.main()
