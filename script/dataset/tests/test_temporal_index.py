"""Tests for the CSR temporal neighbor index and subgraph sampling.

Uses the tiny stream from task section 22.2:

    event 0: t=1, 0 -> 1
    event 1: t=2, 1 -> 2
    event 2: t=2, 0 -> 2
    event 3: t=3, 2 -> 0
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from linkstream.temporal_index import (
    TemporalNeighborIndex,
    sample_temporal_neighbors,
    sample_temporal_subgraph,
)

SRC = np.array([0, 1, 0, 2], dtype=np.int64)
DST = np.array([1, 2, 2, 0], dtype=np.int64)
TS = np.array([1.0, 2.0, 2.0, 3.0])
N_NODES = 3


def build_index(**kwargs) -> TemporalNeighborIndex:
    return TemporalNeighborIndex.build(
        src=SRC, dst=DST, timestamps=TS, num_nodes=N_NODES,
        build_pair_index=kwargs.get("build_pair_index", True),
    )


class TestIndexQueries(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_history_of_first_event_is_empty(self):
        self.assertEqual(self.index.get_past_incident_events(0, 0).tolist(), [])
        self.assertEqual(self.index.get_past_incident_events(1, 0).tolist(), [])
        self.assertEqual(self.index.get_past_out_events(0, 0).tolist(), [])
        self.assertEqual(self.index.get_past_in_events(1, 0).tolist(), [])

    def test_incident_history(self):
        # node 0 before event 2 contains event 0 only (event 2 is current)
        self.assertEqual(self.index.get_past_incident_events(0, 2).tolist(), [0])
        # node 2 before event 3: events 1 (1->2) and 2 (0->2)
        self.assertEqual(self.index.get_past_incident_events(2, 3).tolist(), [1, 2])

    def test_directed_histories(self):
        # node 0 as source: events 0 (0->1) and 2 (0->2), both before event 3
        self.assertEqual(self.index.get_past_out_events(0, 3).tolist(), [0, 2])
        # node 2 as destination: events 1 and 2
        self.assertEqual(self.index.get_past_in_events(2, 3).tolist(), [1, 2])

    def test_no_future_leakage(self):
        for e in range(4):
            for node in range(3):
                for hist in (self.index.get_past_incident_events(node, e),
                             self.index.get_past_out_events(node, e),
                             self.index.get_past_in_events(node, e)):
                    self.assertTrue(np.all(hist < e),
                                    f"leak at node {node} event {e}: {hist}")

    def test_equal_timestamp_events_are_atomic(self):
        # events 1 and 2 share t=2: neither is history for the other.
        self.assertEqual(self.index.get_past_incident_events(2, 2).tolist(), [])
        # A later timestamp can read both t=2 events.
        self.assertEqual(self.index.get_past_incident_events(0, 3).tolist(), [0, 2])

    def test_max_events(self):
        # node 0's incident history before event 3 is [0, 2]; keep most recent
        self.assertEqual(
            self.index.get_past_incident_events(0, 3, max_events=1).tolist(), [2])

    def test_time_window(self):
        # node 2 before event 3 with window 0: only ts >= 3 -> none
        self.assertEqual(
            self.index.get_past_incident_events(2, 3, time_window=0.0).tolist(), [])
        # window 1: ts >= 2 -> events 1 and 2
        self.assertEqual(
            self.index.get_past_incident_events(2, 3, time_window=1.0).tolist(),
            [1, 2])
        # window 2: ts >= 1 -> still events 1 and 2 (node 2 never touches event 0)
        self.assertEqual(
            self.index.get_past_incident_events(2, 3, time_window=2.0).tolist(),
            [1, 2])

    def test_pair_history(self):
        # first interaction of (0, 2) is event 2 itself
        self.assertEqual(self.index.get_past_pair_events(0, 2, 2).tolist(), [])
        self.assertEqual(self.index.get_past_pair_events(0, 2, 3).tolist(), [2])
        self.assertEqual(self.index.get_past_pair_events(0, 1, 2).tolist(), [0])
        # unseen pair
        self.assertEqual(self.index.get_past_pair_events(1, 0, 3).tolist(), [])


class TestDenseTemporalNeighborBatch(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_recent_neighbors_are_padded_and_exclude_equal_timestamp(self):
        batch = sample_temporal_neighbors(
            self.index,
            node_ids=np.array([0, 2, 2]),
            cut_timestamps=np.array([2.0, 2.0, 3.0]),
            n_neighbors=2,
            direction="both",
        )
        # Node 0 has only event 0 before t=2. Node 2 must not see either
        # event at its own t=2, while at t=3 it sees both event 1 and 2.
        self.assertEqual(batch.event_ids.tolist(), [[-1, 0], [-1, -1], [1, 2]])
        self.assertEqual(batch.valid.tolist(), [[False, True], [False, False], [True, True]])


class TestPairIndexOptional(unittest.TestCase):
    def test_absent_pair_index_raises(self):
        index = build_index(build_pair_index=False)
        self.assertFalse(index.has_pair_index())
        with self.assertRaises(RuntimeError):
            index.get_past_pair_events(0, 1, 2)


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        from pathlib import Path

        index = build_index(build_pair_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            index.save(Path(tmp) / "idx")
            loaded = TemporalNeighborIndex.load(Path(tmp) / "idx")
        for attr in ("src_offsets", "src_event_ids", "dst_offsets", "dst_event_ids",
                     "node_offsets", "node_event_ids", "timestamps"):
            self.assertTrue(np.array_equal(getattr(index, attr), getattr(loaded, attr)))
        self.assertTrue(np.array_equal(index.pair_keys, loaded.pair_keys))
        self.assertEqual(loaded.get_past_incident_events(2, 3).tolist(), [1, 2])
        self.assertEqual(loaded.get_past_pair_events(0, 1, 2).tolist(), [0])


class TestTemporalSubgraph(unittest.TestCase):
    def setUp(self):
        self.index = build_index(build_pair_index=True)

    def _sample(self, event_id, **kwargs):
        return sample_temporal_subgraph(
            self.index, event_id,
            src_arr=SRC, dst_arr=DST, ts_arr=TS,
            edge_feat=np.arange(4 * 2, dtype=np.float32).reshape(4, 2),
            **kwargs)

    def test_one_hop_both(self):
        # event 2 (0->2) shares t=2 with event 1, so only t=1 is history.
        sg = self._sample(2, num_hops=1, max_neighbors=20)
        self.assertEqual(sg.event_ids.tolist(), [0])
        self.assertEqual(sg.node_ids.tolist(), [0, 1, 2])  # endpoints + history nodes
        self.assertEqual(sg.src.tolist(), [0])
        self.assertEqual(sg.dst.tolist(), [1])
        self.assertEqual(sg.ts.dtype, torch.float64)
        self.assertEqual(tuple(sg.edge_feat.shape), (1, 2))

    def test_current_and_equal_timestamp_events_excluded(self):
        sg = self._sample(2, num_hops=1)
        self.assertTrue(torch.all(sg.ts < TS[2]))

    def test_direction_out(self):
        # event 3 (2->0): out history of node 2 = [] (events 1, 2 are in-edges),
        # out history of node 0 = [0, 2]
        sg = self._sample(3, num_hops=1, direction="out")
        self.assertEqual(sg.event_ids.tolist(), [0, 2])

    def test_two_hops(self):
        # event 3: hop 1 = incident of 2 and 0 -> {1, 2} and {0, 2} = [0, 1, 2];
        # hop 2 expands from nodes of those events; still strictly past.
        sg = self._sample(3, num_hops=2, max_neighbors=5)
        self.assertTrue(torch.all(sg.event_ids < 3))
        self.assertIn(0, sg.event_ids.tolist())
        self.assertIn(1, sg.event_ids.tolist())

    def test_no_edge_feat(self):
        sg = sample_temporal_subgraph(
            self.index, 1, src_arr=SRC, dst_arr=DST, ts_arr=TS,
            edge_feat=None, num_hops=1, max_neighbors=5)
        self.assertIsNone(sg.edge_feat)
        self.assertEqual(sg.event_ids.tolist(), [0])

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            self._sample(1, num_hops=0)
        with self.assertRaises(ValueError):
            self._sample(1, direction="sideways")
        with self.assertRaises(IndexError):
            self._sample(99)


if __name__ == "__main__":
    unittest.main()
