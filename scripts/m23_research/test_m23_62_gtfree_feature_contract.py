#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/m23_research/m23_62_gtfree_source_regeneration.py"


def load_module():
    specification = importlib.util.spec_from_file_location("m23_62_under_test", SCRIPT)
    if specification is None or specification.loader is None:
        raise RuntimeError(SCRIPT)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_contract(module) -> dict:
    rows, aggregate = module.feature_contract("0" * 64)
    assert len(rows) == 144
    assert aggregate["feature_count"] == 144
    assert aggregate["unique_index_count"] == 144
    assert aggregate["gt_free_count"] == 144
    assert [int(row["zero_based_index"]) for row in rows] == list(range(144))
    visibility = rows[134]
    assert visibility["geometry_local_index"] == 6
    assert visibility["exact_formula"] == "1.0"
    assert "not observed visibility" in visibility["physical_meaning"]
    feature_143 = rows[143]
    assert feature_143["geometry_local_index"] == 15
    assert feature_143["one_based_display_index"] == 144
    assert "nearest" in feature_143["physical_meaning"]
    assert all(row["GT_free"] is True for row in rows)
    return aggregate


def fixture_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"row_index": 0, "line_index": 0, "frame": 1, "track_id": 7, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0},
        {"row_index": 1, "line_index": 1, "frame": 1, "track_id": 8, "x1": 30.0, "y1": 0.0, "x2": 40.0, "y2": 20.0},
        {"row_index": 2, "line_index": 2, "frame": 2, "track_id": 7, "x1": 10.0, "y1": 0.0, "x2": 20.0, "y2": 20.0},
        {"row_index": 3, "line_index": 3, "frame": 3, "track_id": 7, "x1": 25.0, "y1": 0.0, "x2": 35.0, "y2": 20.0},
    ])


def test_geometry(module) -> dict:
    frame = fixture_frame()
    geometry = module.canonical_geometry(frame, width=100, height=100)
    assert geometry.shape == (4, 16)
    assert np.array_equal(geometry[:, 6], np.ones(4, np.float32))
    assert np.allclose(geometry[:2, 14], 0.02)
    assert np.allclose(geometry[2:, 14], 0.01)
    expected_nearest = np.float32(0.3)
    assert np.isclose(geometry[0, 15], expected_nearest)
    assert np.isclose(geometry[1, 15], expected_nearest)
    assert geometry[2, 15] == 1.0
    assert geometry[3, 15] == 1.0
    assert np.isclose(geometry[2, 7], 0.5)
    assert np.isclose(geometry[3, 7], 0.75)
    assert np.isclose(geometry[3, 12], 0.25)
    roundtrip = geometry.astype(np.float16).astype(np.float32)
    assert np.array_equal(roundtrip[:, 15], geometry[:, 15].astype(np.float16).astype(np.float32))
    return {
        "visibility_values": sorted(set(map(float, geometry[:, 6]))),
        "feature_143": geometry[:, 15].tolist(),
        "velocity_x": geometry[:, 7].tolist(),
    }


def test_assignment(module) -> dict:
    source_boxes = np.asarray([[0, 0, 10, 10], [20, 0, 30, 10]], np.float32)
    source_keys = np.asarray([[9, 4], [3, 7]], np.int64)
    detection_boxes = np.asarray([[20, 0, 30, 10], [0, 0, 10, 10]], np.float32)
    global_ids = np.asarray([101, 100], np.int64)
    has_reid = np.asarray([True, True])
    mapped, quality = module.stable_frame_assignment(source_boxes, source_keys, detection_boxes, global_ids, has_reid)
    assert mapped.tolist() == [1, 0]
    assert np.allclose(quality, 1.0)
    source_permutation = np.asarray([1, 0])
    detection_permutation = np.asarray([1, 0])
    mapped_permuted, quality_permuted = module.stable_frame_assignment(
        source_boxes[source_permutation], source_keys[source_permutation],
        detection_boxes[detection_permutation], global_ids[detection_permutation], has_reid[detection_permutation],
    )
    resolved_global = global_ids[detection_permutation][mapped_permuted]
    restored = np.empty(2, np.int64)
    restored[source_permutation] = resolved_global
    assert restored.tolist() == [100, 101]
    assert np.allclose(quality_permuted, 1.0)
    no_reid = np.asarray([True, False])
    mapped_missing, _ = module.stable_frame_assignment(source_boxes, source_keys, detection_boxes, global_ids, no_reid)
    assert mapped_missing.tolist() == [-1, 0]
    return {"base_mapping": mapped.tolist(), "permuted_resolved_global": restored.tolist()}


def test_projection(module) -> dict:
    first = module.projection_matrix()
    second = module.projection_matrix()
    assert first.shape == (2048, 128)
    assert np.array_equal(first, second)
    raw = np.zeros((3, 2048), np.float32)
    raw[0, 0] = 1.0
    raw[1, 1] = 2.0
    raw[2, :4] = np.asarray([1, 2, 3, 4], np.float32)
    projected_first = module.project_embeddings(raw, first)
    projected_second = module.project_embeddings(raw, second)
    assert np.array_equal(projected_first, projected_second)
    assert np.allclose(np.linalg.norm(projected_first, axis=1), 1.0, atol=1e-6)
    return {"matrix_checksum": float(first[:8, :8].sum()), "norms": np.linalg.norm(projected_first, axis=1).tolist()}


def main() -> None:
    module = load_module()
    payload = {
        "passed": True,
        "contract": test_contract(module),
        "geometry": test_geometry(module),
        "assignment": test_assignment(module),
        "projection": test_projection(module),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
