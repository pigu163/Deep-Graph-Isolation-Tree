"""Tests for dataset parsing, stable sorting and deterministic node mapping."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from linkstream import schema
from linkstream.preprocessing import (
    _map_elliptic_class,
    preprocess_elliptic_pp,
    preprocess_ibm_aml,
    preprocess_midas_darpa,
    preprocess_paysim,
)
from tests import (
    preprocess_elliptic,
    preprocess_ibm,
    preprocess_midas,
    preprocess_paysim,
    write_elliptic_raw,
    write_ibm_raw,
)


def _read(out_dir: Path):
    events = pl.read_parquet(out_dir / schema.EVENTS_PARQUET)
    nodes = pl.read_parquet(out_dir / schema.NODES_PARQUET)
    import json

    splits = json.loads((out_dir / schema.SPLITS_JSON).read_text())
    return events, nodes, splits


class TestMidasDarpa(unittest.TestCase):
    def test_parsing_sorted_mapping_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_midas(root, out_root)
            events, nodes, splits = _read(out_root / meta["dataset_name"])

        # stable sort: (t, original_row_id)
        self.assertEqual(events[schema.TIMESTAMP].to_list(), [10.0, 10.0, 20.0, 30.0])
        self.assertEqual(events[schema.ORIGINAL_ROW_ID].to_list(), [1, 2, 0, 3])
        # labels stay aligned with their original rows
        self.assertEqual(events[schema.LABEL].to_list(), [1, 0, 0, 1])
        self.assertTrue(all(events[schema.LABEL_AVAILABLE].to_list()))
        # first-appearance node mapping: 1->0, 2->1, 3->2, 4->3
        self.assertEqual(events[schema.SRC].to_list(), [0, 2, 1, 3])
        self.assertEqual(events[schema.DST].to_list(), [1, 3, 3, 0])
        self.assertEqual(nodes[schema.RAW_ID].to_list(), [1, 2, 3, 4])
        self.assertEqual(nodes[schema.FIRST_EVENT_ID].to_list(), [0, 0, 1, 1])
        # contiguous splits
        self.assertEqual(splits["train"], [0, 2])
        self.assertEqual(splits["val"], [2, 3])
        self.assertEqual(splits["test"], [3, 4])
        self.assertEqual(meta["num_events"], 4)
        self.assertEqual(meta["num_nodes"], 4)

    def test_label_count_mismatch_raises(self):
        import tempfile

        from tests import write_midas_raw

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_midas_raw(root)
            # overwrite ground truth with a wrong number of rows
            (root / "midas_darpa" / "darpa_ground_truth.csv").write_text(
                "0\n1\n0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                preprocess_midas_darpa(root, Path(tmp) / "out", train_ratio=0.6,
                                       val_ratio=0.2, compression="zstd",
                                       build_pair_index=False, overwrite=False)


class TestPaysim(unittest.TestCase):
    def test_mapping_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_paysim(root, out_root)
            events, _, _ = _read(out_root / meta["dataset_name"])

        # rows already ordered by step; equal step keeps original order
        self.assertEqual(events[schema.TIMESTAMP].to_list(), [1.0, 1.0, 2.0])
        self.assertEqual(events[schema.LABEL].to_list(), [0, 0, 1])
        # isFlaggedFraud is NOT the label (row 1: isFraud=0, isFlaggedFraud=1)
        self.assertEqual(events["is_flagged_fraud"].to_list(), [0, 1, 0])
        # node prefixes preserved: C1->0, M2->1, C3->2, C4->3, C5->4, M6->5
        self.assertEqual(events[schema.SRC].to_list(), [0, 2, 4])
        self.assertEqual(events[schema.DST].to_list(), [1, 3, 5])
        self.assertEqual(events[schema.AMOUNT].to_list(), [100.0, 50.0, 30.0])
        self.assertEqual(events[schema.EDGE_TYPE].to_list(),
                         ["PAYMENT", "TRANSFER", "CASH_OUT"])
        feats = [c for c in events.columns if c.startswith(schema.FEATURE_PREFIX)]
        self.assertIn(schema.FEATURE_PREFIX + "amount", feats)


class TestIbmAml(unittest.TestCase):
    def test_bank_account_nodes_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_ibm(root, out_root, size="SMALL", regime="HI")
            events, nodes, _ = _read(out_root / meta["dataset_name"])
            out_dir = out_root / meta["dataset_name"]

        # timestamp: parsed and sorted (00:00, 00:10, 00:20)
        epoch_0000 = datetime(2022, 9, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(events[schema.TIMESTAMP].to_list(),
                         [epoch_0000, epoch_0000 + 600, epoch_0000 + 1200])
        # original row order: row1 (00:00) first, row0 (00:20) last
        self.assertEqual(events[schema.ORIGINAL_ROW_ID].to_list(), [1, 2, 0])
        # bank:account composite raw ids
        self.assertEqual(events[schema.LABEL].to_list(), [1, 0, 0])
        self.assertEqual(events[schema.AMOUNT].to_list(), [0.01, 1858.96, 3697.34])
        # same_bank / same_currency features
        feats = {c: events[c].to_list() for c in events.columns
                 if c.startswith(schema.FEATURE_PREFIX)}
        self.assertEqual(feats[schema.FEATURE_PREFIX + "same_bank"], [0, 1, 1])
        self.assertEqual(feats[schema.FEATURE_PREFIX + "same_currency"], [1, 0, 1])
        # node table keeps bank+account composite ids
        raw_ids = set(nodes[schema.RAW_ID].to_list())
        self.assertIn("03208:8000F4580", raw_ids)
        self.assertIn("010:8000EBD30", raw_ids)
        self.assertEqual(meta["dataset_name"], "ibm_hi_small")
        self.assertEqual(out_dir.name, "ibm_hi_small")

    def test_li_regime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_ibm(root, out_root, size="SMALL", regime="LI")
            events, _, _ = _read(out_root / meta["dataset_name"])
            self.assertEqual(len(events), 3)
            self.assertEqual(meta["dataset_name"], "ibm_li_small")

    def test_unsupported_regime_raises(self):
        with self.assertRaises(ValueError):
            preprocess_ibm_aml(Path("/nonexistent"), Path("/nonexistent"),
                               size="SMALL", regime="XX", train_ratio=0.6,
                               val_ratio=0.2, compression="zstd",
                               build_pair_index=False, overwrite=False)


class TestElliptic(unittest.TestCase):
    def test_destination_label_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_elliptic(root, out_root, features_mode="none")
            events, nodes, _ = _read(out_root / meta["dataset_name"])

        # timestamp = destination time_step; label = destination class
        self.assertEqual(events[schema.TIMESTAMP].to_list(), [2.0, 3.0, 4.0, 5.0])
        self.assertEqual(events[schema.LABEL].to_list(), [1, 0, -1, 1])
        self.assertEqual(events[schema.LABEL_AVAILABLE].to_list(),
                         [True, True, False, True])
        # raw tx ids live in nodes.parquet; first-appearance order after sort:
        # x, a, b, c, y, d  ->  node ids 0..5
        self.assertEqual(nodes[schema.RAW_ID].to_list(), ["x", "a", "b", "c", "y", "d"])
        self.assertEqual(events[schema.SRC].to_list(), [0, 1, 2, 4])
        self.assertEqual(events[schema.DST].to_list(), [1, 2, 3, 5])
        self.assertEqual(len(nodes), 6)
        self.assertEqual(meta["event_label_definition"],
                         "Event labels are derived from destination transaction node labels.")

    def test_features_mode_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            out_root = Path(tmp) / "out"
            meta = preprocess_elliptic(root, out_root, features_mode="all")
            events, _, _ = _read(out_root / meta["dataset_name"])

        # Headerless features table: polars auto-names the feature column
        # "column_3"; it belongs to the DESTINATION transaction.
        f1 = schema.FEATURE_PREFIX + "column_3"
        self.assertIn(f1, events.columns)
        self.assertEqual(events[f1].to_list(), [20.0, 30.0, 40.0, 60.0])

    def test_lfs_pointer_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            write_elliptic_raw(root)
            (root / "elliptic_pp" / "txs_features.csv").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\nsize 123\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                preprocess_elliptic_pp(root, Path(tmp) / "out",
                                       features_mode="none", train_ratio=0.6,
                                       val_ratio=0.2, compression="zstd",
                                       build_pair_index=False, overwrite=False)


class TestClassMapping(unittest.TestCase):
    def test_numeric_classes(self):
        df = pl.DataFrame({"class": [1, 2, 3, 1, 2]})
        out = df.select(_map_elliptic_class(pl.col("class"), df.schema["class"])).to_series()
        self.assertEqual(out.to_list(), [1, 0, -1, 1, 0])

    def test_string_classes(self):
        df = pl.DataFrame({"class": ["illicit", "licit", "unknown", "1", "2"]})
        out = df.select(_map_elliptic_class(pl.col("class"), df.schema["class"])).to_series()
        self.assertEqual(out.to_list(), [1, 0, -1, 1, 0])


if __name__ == "__main__":
    unittest.main()
