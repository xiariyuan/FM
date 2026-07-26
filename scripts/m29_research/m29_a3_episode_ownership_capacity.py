from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
SEQ = "MOT20-01"
ROOT = REPO / "outputs/mot20_m29_20260726/m29_a3_episode_ownership_m01_v1"
GEOMETRY = REPO / "outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/track_results/MOT20-01.txt"
C0_TRACKER = REPO / "outputs/mot20_m28_20260726/m28_b0_m23_46_deferred_identity_m01/teacher_capacity/track_results/MOT20-01.txt"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"

DECISION_OBSERVATIONS = 3
MIN_MATURE_PAST_ROWS = 9
MIN_REID = 2
MAX_GAP = 120
TOPK_REENTRY = 8
MAX_PAIR_ACTIONS = 128
PAIR_DISTANCE_MAX = 2.5
PAIR_LOOKBACK = 120
MAP_IOU = 0.5


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M10 = load_module("m29_a3_m10", "scripts/m23_research/m23_10_build_micrograph.py")
M28 = load_module("m29_a3_m28", "scripts/m28_research/m28_a0_deferred_identity_inheritance.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_tracker(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(",")
            if len(fields) < 7:
                continue
            frame = int(float(fields[0]))
            track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            rows.append(
                {
                    "row_index": len(rows),
                    "frame": frame,
                    "track_id": track_id,
                    "x": x,
                    "y": y,
                    "w": width,
                    "h": height,
                    "x1": x,
                    "y1": y,
                    "x2": x + width,
                    "y2": y + height,
                    "cx": x + 0.5 * width,
                    "cy": y + 0.5 * height,
                    "fields": fields,
                }
            )
    return rows


def assert_same_geometry(left: Path, right: Path) -> None:
    rows_left = parse_tracker(left)
    rows_right = parse_tracker(right)
    if len(rows_left) != len(rows_right):
        raise RuntimeError("C0/geometry row-count mismatch")
    for left_row, right_row in zip(rows_left, rows_right):
        if left_row["fields"][0] != right_row["fields"][0] or left_row["fields"][2:] != right_row["fields"][2:]:
            raise RuntimeError("C0 does not preserve M23-46 geometry/order")


def unit(value) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float32)
    if not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-12 else vector / norm


def prototype(gids: list[int], features, limit: int, from_end: bool) -> tuple[np.ndarray | None, int]:
    selected = gids[-limit:] if from_end else gids[:limit]
    vectors = []
    for gid in selected:
        if gid < 0:
            continue
        vector = unit(features[int(gid)])
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        return None, 0
    return unit(np.mean(np.stack(vectors), axis=0)), len(vectors)


def velocity(rows: list[dict]) -> tuple[float, float]:
    tail = rows[-5:]
    if len(tail) < 2:
        return 0.0, 0.0
    frames = np.asarray([row["frame"] for row in tail], dtype=float)
    if len(np.unique(frames)) < 2:
        return 0.0, 0.0
    centers_x = np.asarray([row["cx"] for row in tail], dtype=float)
    centers_y = np.asarray([row["cy"] for row in tail], dtype=float)
    return float(np.polyfit(frames, centers_x, 1)[0]), float(np.polyfit(frames, centers_y, 1)[0])


def normalized_distance(left: dict, right: dict) -> float:
    scale = max(0.5 * (left["h"] + right["h"]), 1.0)
    return math.hypot(left["cx"] - right["cx"], left["cy"] - right["cy"]) / scale


def box_iou(left: dict, right: dict) -> float:
    x1 = max(left["x1"], right["x1"])
    y1 = max(left["y1"], right["y1"])
    x2 = min(left["x2"], right["x2"])
    y2 = min(left["y2"], right["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left["w"]) * max(0.0, left["h"])
    area_right = max(0.0, right["w"]) * max(0.0, right["h"])
    return intersection / max(area_left + area_right - intersection, 1e-12)


def map_phase_features(rows: list[dict]):
    detections = M10.member(DUMP, "detections.npy")
    columns = M10.member(DUMP, "columns.npy", True).tolist()
    column = {name: index for index, name in enumerate(columns)}
    offsets = M10.member(DUMP, "frame_offsets.npy")
    features = M10.mmap_member(DUMP, "features.npy")
    phase = np.full(len(rows), -1, dtype=np.int64)
    match_iou = np.zeros(len(rows), dtype=np.float32)
    by_frame: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_frame[int(row["frame"])].append(index)
    for frame, indices_list in by_frame.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        start = int(offsets[frame - 1])
        end = int(offsets[frame]) if frame < len(offsets) else len(detections)
        detections_frame = detections[start:end]
        if not len(detections_frame):
            continue
        tracker_boxes = np.asarray(
            [[rows[index][key] for key in ("x1", "y1", "x2", "y2")] for index in indices],
            dtype=np.float32,
        )
        detection_boxes = detections_frame[:, [column["x1"], column["y1"], column["x2"], column["y2"]]].astype(np.float32)
        overlap = M10.iou_matrix(tracker_boxes, detection_boxes)
        row_indices, detection_indices = linear_sum_assignment(-overlap)
        for row_index, detection_index in zip(row_indices, detection_indices):
            if overlap[row_index, detection_index] >= MAP_IOU and detections_frame[detection_index, column["has_reid"]] > 0.5:
                phase[indices[row_index]] = start + detection_index
                match_iou[indices[row_index]] = overlap[row_index, detection_index]
    return phase, match_iou, features


def build_episodes(rows: list[dict], phase: np.ndarray):
    by_track: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_track[int(row["track_id"])].append(index)
    episode_records: list[dict] = []
    episode_objects: list[dict] = []
    for track_id, indices in sorted(by_track.items()):
        indices.sort(key=lambda index: (rows[index]["frame"], index))
        start = 0
        ordinal = 0
        for position in range(1, len(indices) + 1):
            split = position == len(indices) or rows[indices[position]]["frame"] > rows[indices[position - 1]]["frame"] + 1
            if not split:
                continue
            episode_indices = indices[start:position]
            past_indices = indices[:start]
            episode = {
                "track_id": int(track_id),
                "episode_ordinal": int(ordinal),
                "start_frame": int(rows[episode_indices[0]]["frame"]),
                "end_frame": int(rows[episode_indices[-1]]["frame"]),
                "rows": int(len(episode_indices)),
                "past_rows": int(len(past_indices)),
                "row_indices": episode_indices,
                "past_row_indices": past_indices,
                "mapped_rows": int(sum(phase[index] >= 0 for index in episode_indices)),
            }
            episode_objects.append(episode)
            episode_records.append({key: value for key, value in episode.items() if key not in ("row_indices", "past_row_indices")})
            start = position
            ordinal += 1
    return pd.DataFrame(episode_records), by_track, episode_objects


def freeze_candidates() -> None:
    output = ROOT / "frozen_candidates"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    assert_same_geometry(GEOMETRY, C0_TRACKER)
    rows = parse_tracker(GEOMETRY)
    phase, match_iou, features = map_phase_features(rows)
    episodes_frame, by_track, episodes = build_episodes(rows, phase)
    episodes_frame.to_parquet(output / "episodes.parquet", index=False)
    frame_lookup = {(int(row["track_id"]), int(row["frame"])): int(row["row_index"]) for row in rows}

    # C1: re-certify each mature visibility episode after exactly three observations.
    c1_events: list[dict] = []
    c1_candidates: list[dict] = []
    for episode in episodes:
        if episode["episode_ordinal"] == 0 or episode["past_rows"] < MIN_MATURE_PAST_ROWS or episode["rows"] < DECISION_OBSERVATIONS:
            continue
        decision_row_index = int(episode["row_indices"][DECISION_OBSERVATIONS - 1])
        decision_frame = int(rows[decision_row_index]["frame"])
        current_gids = [
            int(phase[index])
            for index in episode["row_indices"][:DECISION_OBSERVATIONS]
            if phase[index] >= 0
        ]
        current_prototype, current_reid_count = prototype(current_gids, features, DECISION_OBSERVATIONS, False)
        if current_prototype is None or current_reid_count < MIN_REID:
            continue
        current_row = rows[decision_row_index]
        event_index = len(c1_events)
        candidates: list[dict] = []
        for old_track_id, old_indices_all in by_track.items():
            if int(old_track_id) == int(episode["track_id"]):
                continue
            old_indices = [index for index in old_indices_all if rows[index]["frame"] < episode["start_frame"]]
            if len(old_indices) < MIN_MATURE_PAST_ROWS:
                continue
            old_last_index = int(old_indices[-1])
            old_last_row = rows[old_last_index]
            gap = int(episode["start_frame"] - old_last_row["frame"])
            if gap < 1 or gap > MAX_GAP:
                continue
            old_gids = [int(phase[index]) for index in old_indices if phase[index] >= 0]
            old_prototype, old_reid_count = prototype(old_gids, features, 8, True)
            if old_prototype is None or old_reid_count < MIN_REID:
                continue
            old_rows = [rows[index] for index in old_indices]
            velocity_x, velocity_y = velocity(old_rows)
            predicted_x = old_last_row["cx"] + velocity_x * gap
            predicted_y = old_last_row["cy"] + velocity_y * gap
            scale = max(0.5 * (old_last_row["h"] + current_row["h"]), 1.0)
            motion_error = math.hypot(current_row["cx"] - predicted_x, current_row["cy"] - predicted_y) / scale
            appearance_cos = float(old_prototype @ current_prototype)
            height_ratio = max(current_row["h"], 1e-3) / max(old_last_row["h"], 1e-3)
            score = appearance_cos - 0.35 * motion_error - 0.03 * math.log1p(gap) - 0.08 * abs(math.log(height_ratio))
            candidates.append(
                {
                    "event_index": event_index,
                    "action_type": "episode_inherit",
                    "track_id": int(episode["track_id"]),
                    "episode_ordinal": int(episode["episode_ordinal"]),
                    "episode_start": int(episode["start_frame"]),
                    "episode_end_replay_only": int(episode["end_frame"]),
                    "decision_frame": decision_frame,
                    "candidate_old_track_id": int(old_track_id),
                    "old_last_row_index": old_last_index,
                    "gap": gap,
                    "appearance_cos": appearance_cos,
                    "motion_error": float(motion_error),
                    "height_ratio": float(height_ratio),
                    "candidate_score": float(score),
                    "current_reid_count": int(current_reid_count),
                    "old_reid_count": int(old_reid_count),
                    "episode_row_indices_replay_only": json.dumps(episode["row_indices"]),
                    "max_candidate_scoring_frame": decision_frame,
                    "max_current_feature_frame": max((rows[index]["frame"] for index in episode["row_indices"][:DECISION_OBSERVATIONS] if phase[index] >= 0), default=-1),
                    "max_old_feature_frame": max((rows[index]["frame"] for index in old_indices if phase[index] >= 0), default=-1),
                    "gt_opened": False,
                }
            )
        candidates.sort(key=lambda row: (-row["candidate_score"], -row["appearance_cos"], row["motion_error"], row["candidate_old_track_id"]))
        selected = candidates[:TOPK_REENTRY]
        for rank, candidate in enumerate(selected, start=1):
            candidate["candidate_rank"] = rank
            c1_candidates.append(candidate)
        c1_events.append(
            {
                "event_index": event_index,
                "track_id": int(episode["track_id"]),
                "episode_ordinal": int(episode["episode_ordinal"]),
                "episode_start": int(episode["start_frame"]),
                "decision_frame": decision_frame,
                "candidate_count": len(selected),
                "gt_opened": False,
            }
        )

    # C2: reciprocal ownership arbitration for two mature overlapping episodes.
    raw_pairs: list[dict] = []
    for left_index, left in enumerate(episodes):
        if left["past_rows"] < MIN_MATURE_PAST_ROWS or left["rows"] < DECISION_OBSERVATIONS:
            continue
        for right in episodes[left_index + 1 :]:
            if right["track_id"] == left["track_id"] or right["past_rows"] < MIN_MATURE_PAST_ROWS or right["rows"] < DECISION_OBSERVATIONS:
                continue
            overlap_start = max(int(left["start_frame"]), int(right["start_frame"]))
            decision_frame = overlap_start + DECISION_OBSERVATIONS - 1
            left_decision = frame_lookup.get((int(left["track_id"]), decision_frame))
            right_decision = frame_lookup.get((int(right["track_id"]), decision_frame))
            if left_decision is None or right_decision is None:
                continue
            left_anchor_gids = [int(phase[index]) for index in left["past_row_indices"] if rows[index]["frame"] < left["start_frame"] and phase[index] >= 0]
            right_anchor_gids = [int(phase[index]) for index in right["past_row_indices"] if rows[index]["frame"] < right["start_frame"] and phase[index] >= 0]
            left_current_indices = [frame_lookup.get((int(left["track_id"]), frame)) for frame in range(overlap_start, decision_frame + 1)]
            right_current_indices = [frame_lookup.get((int(right["track_id"]), frame)) for frame in range(overlap_start, decision_frame + 1)]
            if any(index is None for index in left_current_indices + right_current_indices):
                continue
            left_current_gids = [int(phase[int(index)]) for index in left_current_indices if phase[int(index)] >= 0]
            right_current_gids = [int(phase[int(index)]) for index in right_current_indices if phase[int(index)] >= 0]
            left_anchor, left_anchor_count = prototype(left_anchor_gids, features, 8, True)
            right_anchor, right_anchor_count = prototype(right_anchor_gids, features, 8, True)
            left_current, left_current_count = prototype(left_current_gids, features, DECISION_OBSERVATIONS, False)
            right_current, right_current_count = prototype(right_current_gids, features, DECISION_OBSERVATIONS, False)
            if any(value is None for value in (left_anchor, right_anchor, left_current, right_current)):
                continue
            if min(left_anchor_count, right_anchor_count, left_current_count, right_current_count) < MIN_REID:
                continue
            left_row = rows[int(left_decision)]
            right_row = rows[int(right_decision)]
            distance = normalized_distance(left_row, right_row)
            overlap_iou = box_iou(left_row, right_row)
            if distance > PAIR_DISTANCE_MAX and overlap_iou <= 0.0:
                continue
            own_affinity = float(left_current @ left_anchor + right_current @ right_anchor)
            swap_affinity = float(left_current @ right_anchor + right_current @ left_anchor)
            cross_gain = swap_affinity - own_affinity
            recurrent_close_frames = 0
            for frame in range(max(1, decision_frame - PAIR_LOOKBACK), decision_frame):
                left_index_frame = frame_lookup.get((int(left["track_id"]), frame))
                right_index_frame = frame_lookup.get((int(right["track_id"]), frame))
                if left_index_frame is not None and right_index_frame is not None:
                    if normalized_distance(rows[left_index_frame], rows[right_index_frame]) <= PAIR_DISTANCE_MAX:
                        recurrent_close_frames += 1
            score = cross_gain + 0.15 * min(recurrent_close_frames, 10) / 10.0 + 0.10 / (1.0 + distance) + 0.05 * overlap_iou
            raw_pairs.append(
                {
                    "action_type": "reciprocal_episode_swap",
                    "track_a": int(left["track_id"]),
                    "episode_a": int(left["episode_ordinal"]),
                    "track_b": int(right["track_id"]),
                    "episode_b": int(right["episode_ordinal"]),
                    "episode_a_start": int(left["start_frame"]),
                    "episode_a_end_replay_only": int(left["end_frame"]),
                    "episode_b_start": int(right["start_frame"]),
                    "episode_b_end_replay_only": int(right["end_frame"]),
                    "overlap_start": overlap_start,
                    "decision_frame": decision_frame,
                    "distance": float(distance),
                    "box_iou": float(overlap_iou),
                    "recurrent_close_frames": int(recurrent_close_frames),
                    "own_affinity": own_affinity,
                    "swap_affinity": swap_affinity,
                    "cross_gain": cross_gain,
                    "candidate_score": float(score),
                    "episode_a_row_indices_replay_only": json.dumps(left["row_indices"]),
                    "episode_b_row_indices_replay_only": json.dumps(right["row_indices"]),
                    "max_candidate_scoring_frame": decision_frame,
                    "max_a_current_feature_frame": max((rows[int(index)]["frame"] for index in left_current_indices if phase[int(index)] >= 0), default=-1),
                    "max_b_current_feature_frame": max((rows[int(index)]["frame"] for index in right_current_indices if phase[int(index)] >= 0), default=-1),
                    "max_a_anchor_feature_frame": max((rows[index]["frame"] for index in left["past_row_indices"] if rows[index]["frame"] < left["start_frame"] and phase[index] >= 0), default=-1),
                    "max_b_anchor_feature_frame": max((rows[index]["frame"] for index in right["past_row_indices"] if rows[index]["frame"] < right["start_frame"] and phase[index] >= 0), default=-1),
                    "gt_opened": False,
                }
            )

    pair_frame = pd.DataFrame(raw_pairs)
    selected_pairs: list[dict] = []
    if len(pair_frame):
        channels = [
            pair_frame.sort_values(["cross_gain", "recurrent_close_frames"], ascending=[False, False]),
            pair_frame.sort_values(["recurrent_close_frames", "cross_gain"], ascending=[False, False]),
            pair_frame.sort_values(["distance", "cross_gain"], ascending=[True, False]),
            pair_frame.sort_values(["candidate_score", "distance"], ascending=[False, True]),
        ]
        cursors = [0] * len(channels)
        seen = set()
        while len(selected_pairs) < MAX_PAIR_ACTIONS:
            advanced = False
            for channel_index, channel in enumerate(channels):
                while cursors[channel_index] < len(channel):
                    row = channel.iloc[cursors[channel_index]]
                    cursors[channel_index] += 1
                    key = (int(row.track_a), int(row.episode_a), int(row.track_b), int(row.episode_b))
                    if key in seen:
                        continue
                    seen.add(key)
                    selected_pairs.append(row.to_dict())
                    advanced = True
                    break
                if len(selected_pairs) >= MAX_PAIR_ACTIONS:
                    break
            if not advanced:
                break
    for event_index, candidate in enumerate(selected_pairs):
        candidate["event_index"] = event_index

    c1_frame = pd.DataFrame(c1_candidates)
    c1_event_frame = pd.DataFrame(c1_events)
    c2_frame = pd.DataFrame(selected_pairs)
    if c1_frame.empty and c2_frame.empty:
        raise RuntimeError("no M29-A3 candidates")
    if len(c1_frame):
        causal = (
            (c1_frame.max_current_feature_frame <= c1_frame.decision_frame)
            & (c1_frame.max_old_feature_frame < c1_frame.episode_start)
            & (c1_frame.max_candidate_scoring_frame <= c1_frame.decision_frame)
        )
        if not causal.all():
            raise RuntimeError("C1 causal feature invariant failed")
        c1_frame.to_parquet(output / "c1_episode_inherit_candidates.parquet", index=False)
        c1_event_frame.to_parquet(output / "c1_events.parquet", index=False)
    if len(c2_frame):
        causal = (
            (c2_frame.max_a_current_feature_frame <= c2_frame.decision_frame)
            & (c2_frame.max_b_current_feature_frame <= c2_frame.decision_frame)
            & (c2_frame.max_a_anchor_feature_frame < c2_frame.episode_a_start)
            & (c2_frame.max_b_anchor_feature_frame < c2_frame.episode_b_start)
            & (c2_frame.max_candidate_scoring_frame <= c2_frame.decision_frame)
        )
        if not causal.all():
            raise RuntimeError("C2 causal feature invariant failed")
        c2_frame.to_parquet(output / "c2_reciprocal_episode_candidates.parquet", index=False)

    manifest = {
        "experiment_id": "M29-A3",
        "stage": "candidates_frozen",
        "seq": SEQ,
        "host_geometry": "M23-46",
        "capacity_baseline": "M28-B1 teacher on M23-46",
        "gt_opened": False,
        "future_feature_reads": 0,
        "replay_boundary_note": "episode end is stored only to replay an action that persists causally until the next observed gap; no end-frame field enters candidate scoring",
        "decision_observations": DECISION_OBSERVATIONS,
        "max_gap": MAX_GAP,
        "topk_reentry": TOPK_REENTRY,
        "max_pair_actions": MAX_PAIR_ACTIONS,
        "pair_distance_max": PAIR_DISTANCE_MAX,
        "pair_lookback": PAIR_LOOKBACK,
        "geometry_rows": len(rows),
        "mapped_rows": int((phase >= 0).sum()),
        "mapping_rate": float((phase >= 0).mean()),
        "median_mapping_iou": float(np.median(match_iou[phase >= 0])),
        "episodes": int(len(episodes_frame)),
        "c1_events": int(len(c1_event_frame)),
        "c1_actions": int(len(c1_frame)),
        "c2_raw_actions": int(len(pair_frame)),
        "c2_frozen_actions": int(len(c2_frame)),
        "geometry_sha256": sha256(GEOMETRY),
        "c0_sha256": sha256(C0_TRACKER),
        "dump_sha256": sha256(DUMP),
        "episodes_sha256": sha256(output / "episodes.parquet"),
        "c1_sha256": sha256(output / "c1_episode_inherit_candidates.parquet") if len(c1_frame) else None,
        "c2_sha256": sha256(output / "c2_reciprocal_episode_candidates.parquet") if len(c2_frame) else None,
        "script_sha256": sha256(Path(__file__)),
        "mot20_test_reads": 0,
        "action_equivalence_note": "single transforms overlap historical interval/suffix/transaction families; only a strong composed C1+C2 capacity pass can authorize a new ownership-state mechanism",
    }
    write_json(output / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if len(c2_frame):
        print("\nC2 TOP\n", c2_frame.sort_values("candidate_score", ascending=False).head(25).to_string(index=False))


def duplicate_free(ids: np.ndarray, frames: np.ndarray, changed_rows: np.ndarray) -> bool:
    if not len(changed_rows):
        return False
    for frame in np.unique(frames[changed_rows]):
        frame_ids = ids[frames == frame]
        if len(frame_ids) != len(np.unique(frame_ids)):
            return False
    return True


def write_empty_or_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def evaluate_teacher() -> None:
    frozen = ROOT / "frozen_candidates"
    output = ROOT / "teacher_capacity"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    m37 = load_module("m29_a3_m37", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    prepared = m37.PreparedExactHOTA(SEQ, C0_TRACKER, output / "exact_cache")
    base_ids = prepared.parent_row_ids.copy()
    base_metrics = prepared.evaluate_row_ids_incremental(base_ids)
    frames = np.asarray([int(float(fields[0])) for fields in prepared.parent_rows], dtype=np.int32)

    c1_labels_list: list[dict] = []
    c1_path = frozen / "c1_episode_inherit_candidates.parquet"
    if c1_path.exists():
        c1_candidates = pd.read_parquet(c1_path)
        for ordinal, candidate in enumerate(c1_candidates.itertuples(index=False), start=1):
            row_indices = np.asarray(json.loads(candidate.episode_row_indices_replay_only), dtype=np.int64)
            target_identity = int(base_ids[int(candidate.old_last_row_index)])
            ids = base_ids.copy()
            ids[row_indices] = target_identity
            if not duplicate_free(ids, frames, row_indices):
                c1_labels_list.append({**candidate._asdict(), "status": "invalid_duplicate", "target_identity": target_identity, "modified_rows": int(len(row_indices)), "delta_HOTA": math.nan})
                continue
            metrics = prepared.evaluate_row_ids_incremental(ids)
            c1_labels_list.append(
                {
                    **candidate._asdict(),
                    "status": "success",
                    "target_identity": target_identity,
                    "modified_rows": int(len(row_indices)),
                    "HOTA": float(metrics["HOTA"]),
                    "DetA": float(metrics["DetA"]),
                    "AssA": float(metrics["AssA"]),
                    "delta_HOTA": float(metrics["HOTA"] - base_metrics["HOTA"]),
                }
            )
            if ordinal % 100 == 0:
                print("C1 labeled", ordinal, "/", len(c1_candidates), flush=True)
    c1_labels = pd.DataFrame(c1_labels_list)
    c1_labels.to_parquet(output / "c1_exact_labels.parquet", index=False)

    c2_labels_list: list[dict] = []
    c2_path = frozen / "c2_reciprocal_episode_candidates.parquet"
    if c2_path.exists():
        c2_candidates = pd.read_parquet(c2_path)
        for candidate in c2_candidates.itertuples(index=False):
            rows_a = np.asarray(json.loads(candidate.episode_a_row_indices_replay_only), dtype=np.int64)
            rows_b = np.asarray(json.loads(candidate.episode_b_row_indices_replay_only), dtype=np.int64)
            identity_a = int(base_ids[rows_a[0]])
            identity_b = int(base_ids[rows_b[0]])
            changed_rows = np.unique(np.concatenate([rows_a, rows_b]))
            if identity_a == identity_b:
                c2_labels_list.append({**candidate._asdict(), "status": "invalid_same_identity", "identity_a": identity_a, "identity_b": identity_b, "modified_rows": int(len(changed_rows)), "delta_HOTA": math.nan})
                continue
            ids = base_ids.copy()
            ids[rows_a] = identity_b
            ids[rows_b] = identity_a
            if not duplicate_free(ids, frames, changed_rows):
                c2_labels_list.append({**candidate._asdict(), "status": "invalid_duplicate", "identity_a": identity_a, "identity_b": identity_b, "modified_rows": int(len(changed_rows)), "delta_HOTA": math.nan})
                continue
            metrics = prepared.evaluate_row_ids_incremental(ids)
            c2_labels_list.append(
                {
                    **candidate._asdict(),
                    "status": "success",
                    "identity_a": identity_a,
                    "identity_b": identity_b,
                    "modified_rows": int(len(changed_rows)),
                    "HOTA": float(metrics["HOTA"]),
                    "DetA": float(metrics["DetA"]),
                    "AssA": float(metrics["AssA"]),
                    "delta_HOTA": float(metrics["HOTA"] - base_metrics["HOTA"]),
                }
            )
    c2_labels = pd.DataFrame(c2_labels_list)
    c2_labels.to_parquet(output / "c2_exact_labels.parquet", index=False)

    current_ids = base_ids.copy()
    current_metrics = dict(base_metrics)
    occupied_rows: set[int] = set()
    selected_c1: list[dict] = []
    if len(c1_labels):
        c1_success = c1_labels[c1_labels.status == "success"].sort_values(["delta_HOTA", "candidate_score"], ascending=[False, False])
        for candidate in c1_success[c1_success.delta_HOTA > 0].itertuples(index=False):
            row_indices = np.asarray(json.loads(candidate.episode_row_indices_replay_only), dtype=np.int64)
            if any(int(index) in occupied_rows for index in row_indices):
                continue
            proposal = current_ids.copy()
            proposal[row_indices] = int(candidate.target_identity)
            if not duplicate_free(proposal, frames, row_indices):
                continue
            metrics = prepared.evaluate_row_ids_incremental(proposal)
            gain = float(metrics["HOTA"] - current_metrics["HOTA"])
            if gain <= 0:
                continue
            current_ids = proposal
            current_metrics = metrics
            occupied_rows.update(map(int, row_indices))
            selected_c1.append(
                {
                    "event_index": int(candidate.event_index),
                    "track_id": int(candidate.track_id),
                    "episode_ordinal": int(candidate.episode_ordinal),
                    "target_identity": int(candidate.target_identity),
                    "individual_delta_HOTA": float(candidate.delta_HOTA),
                    "step_delta_HOTA": gain,
                    "modified_rows": int(len(row_indices)),
                }
            )
    c1_ids = current_ids.copy()
    c1_metrics = dict(current_metrics)

    selected_c2: list[dict] = []
    if len(c2_labels):
        c2_success = c2_labels[c2_labels.status == "success"].sort_values(["delta_HOTA", "candidate_score"], ascending=[False, False])
        for candidate in c2_success[c2_success.delta_HOTA > 0].itertuples(index=False):
            rows_a = np.asarray(json.loads(candidate.episode_a_row_indices_replay_only), dtype=np.int64)
            rows_b = np.asarray(json.loads(candidate.episode_b_row_indices_replay_only), dtype=np.int64)
            changed_rows = np.unique(np.concatenate([rows_a, rows_b]))
            if any(int(index) in occupied_rows for index in changed_rows):
                continue
            identity_a = int(current_ids[rows_a[0]])
            identity_b = int(current_ids[rows_b[0]])
            if identity_a == identity_b:
                continue
            proposal = current_ids.copy()
            proposal[rows_a] = identity_b
            proposal[rows_b] = identity_a
            if not duplicate_free(proposal, frames, changed_rows):
                continue
            metrics = prepared.evaluate_row_ids_incremental(proposal)
            gain = float(metrics["HOTA"] - current_metrics["HOTA"])
            if gain <= 0:
                continue
            current_ids = proposal
            current_metrics = metrics
            occupied_rows.update(map(int, changed_rows))
            selected_c2.append(
                {
                    "event_index": int(candidate.event_index),
                    "track_a": int(candidate.track_a),
                    "track_b": int(candidate.track_b),
                    "individual_delta_HOTA": float(candidate.delta_HOTA),
                    "step_delta_HOTA": gain,
                    "modified_rows": int(len(changed_rows)),
                }
            )

    selected_c1_frame = pd.DataFrame(selected_c1)
    selected_c2_frame = pd.DataFrame(selected_c2)
    write_empty_or_frame(selected_c1_frame, output / "selected_c1.csv")
    write_empty_or_frame(selected_c2_frame, output / "selected_c2.csv")
    c1_tracker = output / "c1_track_results" / f"{SEQ}.txt"
    c2_tracker = output / "c2_track_results" / f"{SEQ}.txt"
    M28.write_tracker(c1_tracker, prepared, c1_ids)
    M28.write_tracker(c2_tracker, prepared, current_ids)
    M28.SEQ = SEQ
    official_c1 = M28.official_eval(c1_tracker.parent, "m29_a3_c1_teacher", output / "official_eval_c1")
    official_c2 = M28.official_eval(c2_tracker.parent, "m29_a3_c2_teacher", output / "official_eval_c2")

    c1_gain = float(c1_metrics["HOTA"] - base_metrics["HOTA"])
    c2_increment = float(current_metrics["HOTA"] - c1_metrics["HOTA"])
    total_gain = float(current_metrics["HOTA"] - base_metrics["HOTA"])
    gate = {
        "c1_gain_at_least_0p15": c1_gain >= 0.15,
        "c2_increment_at_least_0p15": c2_increment >= 0.15,
        "final_HOTA_at_least_80": float(current_metrics["HOTA"]) >= 80.0,
        "IDSW_nonincrease": int(official_c2["IDSW"]) <= 42,
    }
    gate["pass"] = all(gate.values())
    report = {
        "experiment_id": "M29-A3",
        "status": "completed",
        "decision": "PASS_M29_A3_EXTEND_ALL4" if gate["pass"] else "FAIL_M29_A3_CLOSE_EPISODE_OWNERSHIP",
        "teacher_only": True,
        "deployable": False,
        "gt_opened_after_candidate_freeze": True,
        "baseline_C0": {key: float(base_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "C1": {key: float(c1_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "C2": {key: float(current_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "c1_gain": c1_gain,
        "c2_increment": c2_increment,
        "total_gain": total_gain,
        "c1_actions": int(len(c1_labels)),
        "c1_valid": int((c1_labels.status == "success").sum()) if len(c1_labels) else 0,
        "c1_positive": int((c1_labels.delta_HOTA > 0).sum()) if len(c1_labels) else 0,
        "c1_selected": int(len(selected_c1_frame)),
        "c2_actions": int(len(c2_labels)),
        "c2_valid": int((c2_labels.status == "success").sum()) if len(c2_labels) else 0,
        "c2_positive": int((c2_labels.delta_HOTA > 0).sum()) if len(c2_labels) else 0,
        "c2_selected": int(len(selected_c2_frame)),
        "official_C1": official_c1,
        "official_C2": official_c2,
        "gate": gate,
        "mot20_test_reads": 0,
        "test_submission": False,
        "freeze_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "final_tracker_sha256": sha256(c2_tracker),
        "equivalence_audit": "single episode inherit/swap output transforms overlap prior interval/suffix/transaction families; failure closes this packaging rather than supporting a novelty claim",
    }
    write_json(ROOT / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if len(c1_labels):
        print("\nC1 TOP\n", c1_labels[c1_labels.status == "success"].sort_values("delta_HOTA", ascending=False).head(30).to_string(index=False))
    if len(c2_labels):
        print("\nC2 TOP\n", c2_labels[c2_labels.status == "success"].sort_values("delta_HOTA", ascending=False).head(30).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-candidates", "teacher"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-candidates":
        freeze_candidates()
    else:
        evaluate_teacher()


if __name__ == "__main__":
    main()
