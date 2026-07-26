#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name("m23_70a_causal_branch_capacity.py")
spec = importlib.util.spec_from_file_location("m23_70a", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["m23_70a"] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def test_causal_teacher_cut() -> None:
    fixed = pd.DataFrame([{"chunk_id": 0}])
    chunks = {0: list(range(7))}
    rows = [{"frame": f} for f in [1, 2, 3, 4, 5, 6, 7]]
    gt = np.asarray([1, 1, 1, 2, 0, 2, 2])
    cuts, transitions, report = module.causal_teacher_cuts(fixed, chunks, rows, gt, {})
    assert cuts == {0: [3]}
    assert int(transitions.iloc[0].causal_eligible) == 1
    assert report["causal_eligible_transitions"] == 1


def test_noncausal_cut_rejected() -> None:
    fixed = pd.DataFrame([{"chunk_id": 0}])
    chunks = {0: list(range(5))}
    rows = [{"frame": f} for f in [1, 2, 3, 20, 30]]
    gt = np.asarray([1, 1, 1, 2, 2])
    cuts, _, report = module.causal_teacher_cuts(fixed, chunks, rows, gt, {})
    assert cuts == {}
    assert report["causal_eligible_transitions"] == 0


def test_k3_and_destination_horizon() -> None:
    nodes = pd.DataFrame([
        {"chunk_id": 0, "first_frame": 1, "last_frame": 3, "first_cx": 0.0, "first_cy": 0.0, "last_cx": 2.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 10},
        {"chunk_id": 1, "first_frame": 4, "last_frame": 5, "first_cx": 3.0, "first_cy": 0.0, "last_cx": 4.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 10},
        {"chunk_id": 2, "first_frame": 5, "last_frame": 6, "first_cx": 4.0, "first_cy": 0.0, "last_cx": 5.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 20},
        {"chunk_id": 3, "first_frame": 6, "last_frame": 7, "first_cx": 5.0, "first_cy": 0.0, "last_cx": 6.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 30},
        {"chunk_id": 4, "first_frame": 7, "last_frame": 8, "first_cx": 6.0, "first_cy": 0.0, "last_cx": 7.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 40},
        {"chunk_id": 5, "first_frame": 20, "last_frame": 21, "first_cx": 20.0, "first_cy": 0.0, "last_cx": 21.0, "last_cy": 0.0, "first_h": 10.0, "last_h": 10.0, "end_vx": 1.0, "end_vy": 0.0, "parent_tracker_id": 50},
    ])
    source_rows = [{"frame": f} for f in range(1, 22)]
    row_map = {0: [0, 1, 2], 1: [3, 4], 2: [4, 5], 3: [5, 6], 4: [6, 7], 5: [19, 20]}
    emb = np.zeros((21, 2), np.float32)
    emb[:, 0] = 1.0
    mapped = np.ones(21, bool)
    edges, branches, report = module.causal_candidate_edges(nodes, row_map, source_rows, emb, mapped)
    from_zero = branches[branches.src_chunk == 0]
    assert len(from_zero) == 2
    assert int(from_zero.branch_rank.max()) == 2
    assert not ((edges.src_chunk == 0) & (edges.dst_chunk == 5)).any()
    assert report["max_alternatives_per_source"] <= 2
    assert report["complete_future_destination_descriptor_used"] is False


if __name__ == "__main__":
    test_causal_teacher_cut()
    test_noncausal_cut_rejected()
    test_k3_and_destination_horizon()
    print("M23-70A tests passed")
