from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import dmm_base_tracker as base  # noqa: E402

SEQ = "MOT20-01"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"
ROOT = REPO / "outputs/mot20_m28_20260726/m28_a1_online_external_identity_parity_m01"
A0 = REPO / "outputs/mot20_m28_20260726/m28_a0_deferred_identity_inheritance_m01_v1"
SELECTED = A0 / "teacher_capacity/selected_actions.csv"
STATIC_TEACHER = A0 / "teacher_capacity/track_results/MOT20-01.txt"
BASELINE_REFERENCE = REPO / ".mcp_tmp/m24_online_b0_parity/run1/track_results/MOT20-01.txt"
DELAY = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_class():
    for cls in base.DMMTrack.mro():
        if hasattr(cls, "_count"):
            return cls
    raise RuntimeError("counter missing")


def frame_data(detections, features, offsets, column, frame: int):
    start, end = int(offsets[frame - 1]), int(offsets[frame])
    rows = detections[start:end]
    if len(rows):
        boxes = rows[:, [column["x1"], column["y1"], column["x2"], column["y2"]]].astype(np.float32)
        scores = rows[:, column["score"]].astype(np.float32)
        feats = np.asarray(features[start:end], dtype=np.float32)
        gids = rows[:, column["global_det_idx"]].astype(np.int64)
    else:
        dim = int(features.shape[1]) if features.ndim == 2 else 0
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        feats = np.zeros((0, dim), dtype=np.float32)
        gids = np.zeros((0,), dtype=np.int64)
    return boxes, scores, feats, gids


def visible_rows(tracks, frame: int, cfg):
    rows = []
    for track in tracks:
        x, y, width, height = [float(value) for value in track.tlwh]
        if width * height < float(cfg.min_box_area) or width <= 0 or height <= 0:
            continue
        if width / max(height, 1e-12) > float(cfg.aspect_ratio_thresh):
            continue
        rows.append({
            "frame": frame, "internal_id": int(track.track_id), "external_id": int(track.track_id),
            "x": x, "y": y, "w": width, "h": height, "score": float(track.score),
        })
    return rows


def render(rows):
    return [(row["frame"], row["external_id"], row["x"], row["y"], row["w"], row["h"], row["score"]) for row in rows]


def duplicates(rows) -> int:
    keys = [(int(row["frame"]), int(row["external_id"])) for row in rows]
    return len(keys) - len(set(keys))


def run(actions: pd.DataFrame, path: Path):
    dump = base.load_dump(DUMP)
    detections = np.asarray(dump["detections"], dtype=np.float32)
    features = np.asarray(dump["features"])
    offsets = np.asarray(dump["frame_offsets"], dtype=np.int64)
    columns = [str(value) for value in dump["columns"].tolist()]
    column = {name: index for index, name in enumerate(columns)}
    frames = len(offsets) - 1
    cfg = base.TrackerConfig(dmm_v3_enable=False)
    count_class()._count = 0
    tracker = base.DMMBaseTracker(cfg)
    pending = defaultdict(list)
    for action in actions.itertuples(index=False):
        pending[int(action.frame) + DELAY].append({
            "event_frame": int(action.frame), "young": int(action.young_track_id), "old": int(action.old_track_id),
        })
    identity_map = {}
    buffer = []
    emitted = []
    commits = []
    for frame in range(1, frames + 1):
        boxes, scores, feats, gids = frame_data(detections, features, offsets, column, frame)
        tracks = tracker.update(boxes, scores, feats, gids)
        for row in visible_rows(tracks, frame, cfg):
            row["external_id"] = int(identity_map.get(row["internal_id"], row["internal_id"]))
            buffer.append(row)
        for action in pending.get(frame, []):
            young, old, event_frame = action["young"], action["old"], action["event_frame"]
            if young in identity_map:
                raise RuntimeError(f"young identity committed twice: {young}")
            if old in identity_map.values():
                raise RuntimeError(f"old identity inherited twice: {old}")
            identity_map[young] = old
            rewritten = 0
            for row in buffer:
                if int(row["internal_id"]) == young and int(row["frame"]) >= event_frame:
                    row["external_id"] = old
                    rewritten += 1
            commits.append({
                "event_frame": event_frame, "decision_frame": frame, "young_track_id": young,
                "old_identity_id": old, "buffered_rows_rewritten": rewritten,
            })
        cutoff = frame - DELAY
        keep = []
        for row in buffer:
            if int(row["frame"]) <= cutoff:
                emitted.append(row)
            else:
                keep.append(row)
        buffer = keep
    if buffer:
        emitted.extend(buffer)
    if duplicates(emitted):
        raise RuntimeError("duplicate external frame/identity rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    base.write_mot_results(path, render(emitted))
    return emitted, pd.DataFrame(commits), {
        "frames": frames, "rows": len(emitted), "actions": len(actions), "commits": len(commits),
        "maximum_output_delay_frames": DELAY, "duplicates": duplicates(emitted),
    }


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    selected = pd.read_csv(SELECTED)
    b0_rows, b0_commits, b0 = run(selected.iloc[:0], ROOT / "b0/track_results/MOT20-01.txt")
    teacher_rows, commits, teacher = run(selected, ROOT / "teacher/track_results/MOT20-01.txt")
    b0_commits.to_csv(ROOT / "b0/commit_log.csv", index=False)
    commits.to_csv(ROOT / "teacher/commit_log.csv", index=False)
    b0_path = ROOT / "b0/track_results/MOT20-01.txt"
    teacher_path = ROOT / "teacher/track_results/MOT20-01.txt"
    b0_equal = b0_path.read_bytes() == BASELINE_REFERENCE.read_bytes()
    teacher_equal = teacher_path.read_bytes() == STATIC_TEACHER.read_bytes()
    report = {
        "experiment_id": "M28-A1", "status": "completed",
        "teacher_only": True, "deployable": False, "mot20_test_reads": 0, "test_submission": False,
        "mechanism": "internal geometric track ID is unchanged; external identity ID is committed after a bounded delay",
        "delay_frames": DELAY, "selected_teacher_actions": len(selected),
        "b0": b0 | {"sha256": sha256(b0_path), "byte_identical_reference": b0_equal},
        "online_teacher": teacher | {"sha256": sha256(teacher_path), "byte_identical_static_teacher": teacher_equal},
        "static_teacher_sha256": sha256(STATIC_TEACHER),
        "all_decisions_within_sequence": bool((commits.decision_frame <= 429).all()),
        "all_commit_delays_exact": bool(((commits.decision_frame - commits.event_frame) == DELAY).all()),
        "gate_pass": bool(b0_equal and teacher_equal and teacher["duplicates"] == 0 and len(commits) == len(selected)),
    }
    (ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{
        "experiment_id": "M28-A1", "actions": len(selected), "delay_frames": DELAY,
        "b0_byte_exact": int(b0_equal), "teacher_byte_exact": int(teacher_equal),
        "duplicates": teacher["duplicates"], "gate_pass": int(report["gate_pass"]),
    }]).to_csv(ROOT / "summary.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(commits.to_string(index=False))


if __name__ == "__main__":
    main()
