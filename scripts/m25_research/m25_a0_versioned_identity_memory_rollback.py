from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import dmm_base_tracker as base  # noqa: E402

SEQ = "MOT20-01"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"
ROOT = REPO / "outputs/mot20_m25_20260726/m25_a0_versioned_identity_memory_rollback_m01_v1"
BASELINE_REFERENCE = REPO / ".mcp_tmp/m24_online_b0_parity/run1/track_results/MOT20-01.txt"
EVENT_BUDGET = 24
TRACK_NMS = 24
MAX_PER_FRAME = 2
MIN_VERSIONS = 16
HOLD_FRAMES = 8
ACTIONS = ("freeze0", "rollback4", "rollback8", "rollback16")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_class():
    for cls in base.DMMTrack.mro():
        if hasattr(cls, "_count"):
            return cls
    raise RuntimeError("track ID counter not found")


def implementation_globals() -> dict:
    return base.DMMBaseTracker.update.__globals__


def load_dump_arrays():
    dump = base.load_dump(DUMP)
    detections = np.asarray(dump["detections"], dtype=np.float32)
    features = np.asarray(dump["features"])
    offsets = np.asarray(dump["frame_offsets"], dtype=np.int64)
    columns = [str(value) for value in dump["columns"].tolist()]
    column = {name: index for index, name in enumerate(columns)}
    return detections, features, offsets, column


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


def unit(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return None if norm <= 1e-12 else array / norm


def cosine(left, right, default: float = -1.0) -> float:
    a, b = unit(left), unit(right)
    return default if a is None or b is None else float(a @ b)


def appearance_snapshot(track) -> dict | None:
    if track is None:
        return None
    return {
        "smooth_feat": None if track.smooth_feat is None else np.asarray(track.smooth_feat, dtype=np.float32).copy(),
        "curr_feat": None if track.curr_feat is None else np.asarray(track.curr_feat, dtype=np.float32).copy(),
        # Feature vectors already stored in DMMTrack.features are immutable copies.
        # Keep shared references across versions; materialize fresh copies only on restore.
        "features": tuple(track.features),
        "feature_maxlen": track.features.maxlen,
        "alpha": float(track.alpha),
    }


def restore_appearance(track, snapshot: dict | None) -> bool:
    if track is None or snapshot is None:
        return False
    track.smooth_feat = None if snapshot["smooth_feat"] is None else snapshot["smooth_feat"].copy()
    track.curr_feat = None if snapshot["curr_feat"] is None else snapshot["curr_feat"].copy()
    track.features = deque(
        [np.asarray(value, dtype=np.float32).copy() for value in snapshot["features"]],
        maxlen=snapshot.get("feature_maxlen", 50),
    )
    track.alpha = float(snapshot["alpha"])
    return True


def install_version_logger(versions: dict[int, list[dict]]):
    cls = implementation_globals()["DMMTrack"]
    original_activate = cls.activate
    original_update = cls.update
    original_reactivate = cls.re_activate

    def append(track):
        snap = appearance_snapshot(track)
        if snap is not None:
            versions[int(track.track_id)].append(snap)

    def activate(self, *args, **kwargs):
        result = original_activate(self, *args, **kwargs)
        append(self)
        return result

    def update(self, *args, **kwargs):
        result = original_update(self, *args, **kwargs)
        append(self)
        return result

    def reactivate(self, *args, **kwargs):
        result = original_reactivate(self, *args, **kwargs)
        append(self)
        return result

    cls.activate = activate
    cls.update = update
    cls.re_activate = reactivate
    return cls, original_activate, original_update, original_reactivate


def uninstall_version_logger(token) -> None:
    cls, activate, update, reactivate = token
    cls.activate = activate
    cls.update = update
    cls.re_activate = reactivate


def output_rows(tracks, frame: int, cfg) -> list[tuple]:
    rows = []
    for track in tracks:
        x, y, width, height = [float(value) for value in track.tlwh]
        if width * height < float(cfg.min_box_area) or width <= 0 or height <= 0:
            continue
        if width / max(height, 1e-12) > float(cfg.aspect_ratio_thresh):
            continue
        rows.append((frame, int(track.track_id), x, y, width, height, float(track.score)))
    return rows


def parse_tracker(path: Path) -> list[tuple]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(",")
            if len(fields) >= 7:
                rows.append((int(float(fields[0])), int(float(fields[1])), *[float(x) for x in fields[2:7]]))
    return rows


def duplicate_count(rows: Iterable[tuple]) -> int:
    keys = [(int(row[0]), int(row[1])) for row in rows]
    return len(keys) - len(set(keys))


def risk_percentile(values: pd.Series, higher_is_risk: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    rank = numeric.fillna(fill).rank(method="average", pct=True)
    return rank if higher_is_risk else 1.0 - rank


def freeze_events() -> None:
    event_dir = ROOT / "frozen_events"
    if event_dir.exists():
        raise FileExistsError(event_dir)
    event_dir.mkdir(parents=True)
    detections, features, offsets, column = load_dump_arrays()
    frames = len(offsets) - 1
    cfg = base.TrackerConfig(dmm_v3_enable=False)
    count_class()._count = 0
    tracker = base.DMMBaseTracker(cfg)
    versions: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=MIN_VERSIONS))
    logger = install_version_logger(versions)
    globals_ = implementation_globals()
    original_cost = globals_["assoc_cost"]
    original_assignment = globals_["matching"].linear_assignment
    records: list[dict] = []
    baseline_rows: list[tuple] = []

    try:
        for frame in range(1, frames + 1):
            boxes, scores, feats, gids = frame_data(detections, features, offsets, column, frame)
            calls: list[dict] = []

            def cost_hook(tracks, dets, tracker_cfg):
                cost, debug = original_cost(tracks, dets, tracker_cfg)
                track_rows = []
                for track in tracks:
                    history = [np.asarray(value, dtype=np.float32).copy() for value in list(track.features)]
                    history_versions = versions.get(int(track.track_id), ())
                    checkpoint_smooth = {
                        depth: (
                            None
                            if len(history_versions) < depth or history_versions[-depth]["smooth_feat"] is None
                            else np.asarray(history_versions[-depth]["smooth_feat"], dtype=np.float32).copy()
                        )
                        for depth in (4, 8, 16)
                    }
                    track_rows.append({
                        "track_id": int(track.track_id),
                        "tracklet_len": int(track.tracklet_len),
                        "smooth_feat": None if track.smooth_feat is None else np.asarray(track.smooth_feat, dtype=np.float32).copy(),
                        "curr_feat": None if track.curr_feat is None else np.asarray(track.curr_feat, dtype=np.float32).copy(),
                        "history": history,
                        "version_count": len(history_versions),
                        "checkpoint_smooth": checkpoint_smooth,
                    })
                det_rows = [{
                    "det_global_idx": int(det.det_global_idx),
                    "score": float(det.score),
                    "curr_feat": None if det.curr_feat is None else np.asarray(det.curr_feat, dtype=np.float32).copy(),
                } for det in dets]
                calls.append({
                    "cost": np.asarray(cost, dtype=np.float32).copy(),
                    "debug": {key: np.asarray(value).copy() for key, value in debug.items()},
                    "tracks": track_rows,
                    "detections": det_rows,
                })
                return cost, debug

            def assignment_hook(cost, thresh):
                output = original_assignment(cost, thresh=thresh)
                if calls and "matches" not in calls[-1] and np.asarray(cost).shape == calls[-1]["cost"].shape:
                    calls[-1]["matches"] = np.asarray(output[0], dtype=np.int64).copy()
                    calls[-1]["threshold"] = float(thresh)
                return output

            globals_["assoc_cost"] = cost_hook
            globals_["matching"].linear_assignment = assignment_hook
            try:
                tracks = tracker.update(boxes, scores, feats, gids)
            finally:
                globals_["assoc_cost"] = original_cost
                globals_["matching"].linear_assignment = original_assignment
            baseline_rows.extend(output_rows(tracks, frame, cfg))

            if not calls or frame > frames - HOLD_FRAMES:
                continue
            call = calls[0]
            cost = call["cost"]
            debug = call["debug"]
            matches = call.get("matches", np.empty((0, 2), dtype=np.int64))
            threshold = float(call.get("threshold", cfg.match_thresh))
            emb = debug.get("emb", np.full_like(cost, np.nan, dtype=np.float32))
            raw_iou = debug.get("raw_iou", np.full_like(cost, np.nan, dtype=np.float32))
            for row_index, det_index in matches:
                i, j = int(row_index), int(det_index)
                if i >= len(call["tracks"]) or j >= len(call["detections"]):
                    continue
                track = call["tracks"][i]
                det = call["detections"][j]
                if track["tracklet_len"] < MIN_VERSIONS or track["version_count"] < MIN_VERSIONS:
                    continue
                if track["smooth_feat"] is None or det["curr_feat"] is None:
                    continue
                chosen = float(cost[i, j])
                if not np.isfinite(chosen) or chosen > threshold:
                    continue
                row_values = np.delete(cost[i], j)
                col_values = np.delete(cost[:, j], i)
                row_alt = float(np.min(row_values)) if len(row_values) else threshold
                col_alt = float(np.min(col_values)) if len(col_values) else threshold
                row_margin = row_alt - chosen
                col_margin = col_alt - chosen
                pair_margin = min(row_margin, col_margin)
                smooth_cos = cosine(track["smooth_feat"], det["curr_feat"])
                history = track["history"]
                recent = history[-4:] if len(history) >= 4 else history
                older = history[:-4] if len(history) > 4 else history
                recent_mean = unit(np.mean(np.stack(recent), axis=0)) if recent else None
                older_mean = unit(np.mean(np.stack(older), axis=0)) if older else None
                recent_old_cos = cosine(recent_mean, older_mean)
                checkpoint_cos = {
                    depth: cosine(track["checkpoint_smooth"][depth], det["curr_feat"])
                    for depth in (4, 8, 16)
                }
                checkpoint_advantage = max(checkpoint_cos.values()) - smooth_cos
                records.append({
                    "seq": SEQ,
                    "frame": frame,
                    "track_id": int(track["track_id"]),
                    "det_global_idx": int(det["det_global_idx"]),
                    "tracklet_len": int(track["tracklet_len"]),
                    "version_count": int(track["version_count"]),
                    "chosen_cost": chosen,
                    "embedding_cost": float(emb[i, j]) if emb.shape == cost.shape else math.nan,
                    "raw_iou_cost": float(raw_iou[i, j]) if raw_iou.shape == cost.shape else math.nan,
                    "det_score": float(det["score"]),
                    "row_margin": row_margin,
                    "col_margin": col_margin,
                    "pair_margin": pair_margin,
                    "smooth_current_cos": smooth_cos,
                    "recent_old_cos": recent_old_cos,
                    "checkpoint4_current_cos": checkpoint_cos[4],
                    "checkpoint8_current_cos": checkpoint_cos[8],
                    "checkpoint16_current_cos": checkpoint_cos[16],
                    "checkpoint_advantage": checkpoint_advantage,
                    "primary_threshold": threshold,
                    "gt_opened": False,
                })
    finally:
        globals_["assoc_cost"] = original_cost
        globals_["matching"].linear_assignment = original_assignment
        uninstall_version_logger(logger)

    baseline_path = event_dir / "baseline_online.txt"
    base.write_mot_results(baseline_path, baseline_rows)
    if not BASELINE_REFERENCE.is_file() or baseline_path.read_bytes() != BASELINE_REFERENCE.read_bytes():
        raise RuntimeError("B0 replay is not byte-identical to the independent baseline")
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no eligible memory events")
    frame["risk_low_margin"] = risk_percentile(frame.pair_margin, False)
    frame["risk_memory_conflict"] = risk_percentile(frame.smooth_current_cos, False)
    frame["risk_recent_drift"] = risk_percentile(frame.recent_old_cos, False)
    frame["risk_checkpoint_advantage"] = risk_percentile(frame.checkpoint_advantage, True)
    channels = {
        "low_margin": frame.sort_values(["risk_low_margin", "frame", "track_id"], ascending=[False, True, True]).index.tolist(),
        "memory_conflict": frame.sort_values(["risk_memory_conflict", "frame", "track_id"], ascending=[False, True, True]).index.tolist(),
        "recent_drift": frame.sort_values(["risk_recent_drift", "frame", "track_id"], ascending=[False, True, True]).index.tolist(),
        "checkpoint_advantage": frame.sort_values(["risk_checkpoint_advantage", "frame", "track_id"], ascending=[False, True, True]).index.tolist(),
    }
    cursors = {name: 0 for name in channels}
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_meta: dict[int, tuple[str, int]] = {}
    per_frame: dict[int, int] = defaultdict(int)
    active = list(channels)
    while active and len(selected) < EVENT_BUDGET:
        progressed = False
        for name in list(active):
            order = channels[name]
            cursor = cursors[name]
            chosen = None
            while cursor < len(order):
                index = int(order[cursor])
                cursor += 1
                if index in selected_set:
                    continue
                row = frame.loc[index]
                event_frame, track_id = int(row.frame), int(row.track_id)
                if per_frame[event_frame] >= MAX_PER_FRAME:
                    continue
                if any(int(frame.loc[other].track_id) == track_id and abs(event_frame - int(frame.loc[other].frame)) <= TRACK_NMS for other in selected):
                    continue
                chosen = index
                break
            cursors[name] = cursor
            if chosen is None:
                active.remove(name)
                continue
            selected.append(chosen)
            selected_set.add(chosen)
            selected_meta[chosen] = (name, cursor)
            per_frame[int(frame.loc[chosen].frame)] += 1
            progressed = True
            if len(selected) >= EVENT_BUDGET:
                break
        if not progressed:
            break
    events = frame.loc[selected].copy()
    events["selection_channel"] = [selected_meta[int(index)][0] for index in events.index]
    events["selection_rank"] = [selected_meta[int(index)][1] for index in events.index]
    events.sort_values(["frame", "track_id", "det_global_idx"], inplace=True)
    events.reset_index(drop=True, inplace=True)
    events.insert(0, "event_index", np.arange(len(events), dtype=int))
    events["event_id"] = [f"{SEQ}_f{int(f):04d}_t{int(t)}_d{int(d)}" for f, t, d in zip(events.frame, events.track_id, events.det_global_idx)]
    events_path = event_dir / "events.parquet"
    events.to_parquet(events_path, index=False)
    manifest = {
        "experiment_id": "M25-A0",
        "stage": "events_frozen",
        "seq": SEQ,
        "gt_opened": False,
        "mot20_test_reads": 0,
        "frames": frames,
        "baseline_rows": len(baseline_rows),
        "baseline_sha256": sha256(baseline_path),
        "reference_sha256": sha256(BASELINE_REFERENCE),
        "raw_eligible_events": len(frame),
        "frozen_events": len(events),
        "event_sha256": sha256(events_path),
        "event_budget": EVENT_BUDGET,
        "track_nms_frames": TRACK_NMS,
        "max_per_frame": MAX_PER_FRAME,
        "minimum_versions": MIN_VERSIONS,
        "hold_frames": HOLD_FRAMES,
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": sha256(DUMP),
    }
    json_write(event_dir / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(events.to_string(index=False))


def install_action_logger(target_track: int, target_det: int, event_frame: int, hold_snapshot: dict, action_stats: dict):
    cls = implementation_globals()["DMMTrack"]
    original_activate = cls.activate
    original_update = cls.update
    original_reactivate = cls.re_activate

    def maybe_restore(track, new_track, frame_id: int):
        if int(track.track_id) != int(target_track):
            return
        gid = int(getattr(new_track, "det_global_idx", -1))
        if int(frame_id) == int(event_frame) and gid == int(target_det):
            action_stats["event_update_hits"] += 1
        if int(event_frame) <= int(frame_id) < int(event_frame) + HOLD_FRAMES:
            if restore_appearance(track, hold_snapshot):
                action_stats["restore_hits"] += 1

    def activate(self, *args, **kwargs):
        return original_activate(self, *args, **kwargs)

    def update(self, new_track, frame_id):
        result = original_update(self, new_track, frame_id)
        maybe_restore(self, new_track, frame_id)
        return result

    def reactivate(self, new_track, frame_id, new_id=False):
        result = original_reactivate(self, new_track, frame_id, new_id=new_id)
        maybe_restore(self, new_track, frame_id)
        return result

    cls.activate = activate
    cls.update = update
    cls.re_activate = reactivate
    return cls, original_activate, original_update, original_reactivate


def run_frame(tracker, detections, features, offsets, column, frame: int):
    boxes, scores, feats, gids = frame_data(detections, features, offsets, column, frame)
    tracks = tracker.update(boxes, scores, feats, gids)
    return output_rows(tracks, frame, tracker.cfg)


def find_track(tracker, track_id: int):
    for track in list(tracker.tracked_stracks) + list(tracker.lost_stracks) + list(tracker.removed_stracks):
        if int(track.track_id) == int(track_id):
            return track
    return None


def generate_candidates() -> None:
    event_dir = ROOT / "frozen_events"
    candidate_dir = ROOT / "frozen_candidates"
    if candidate_dir.exists():
        raise FileExistsError(candidate_dir)
    candidate_dir.mkdir(parents=True)
    events = pd.read_parquet(event_dir / "events.parquet")
    baseline = parse_tracker(event_dir / "baseline_online.txt")
    detections, features, offsets, column = load_dump_arrays()
    frames = len(offsets) - 1
    count_cls = count_class()
    reports: list[dict] = []

    for event in events.itertuples(index=False):
        event_index = int(event.event_index)
        event_frame = int(event.frame)
        track_id = int(event.track_id)
        det_id = int(event.det_global_idx)
        count_cls._count = 0
        prefix_tracker = base.DMMBaseTracker(base.TrackerConfig(dmm_v3_enable=False))
        prefix_versions: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=MIN_VERSIONS))
        logger = install_version_logger(prefix_versions)
        try:
            for frame in range(1, event_frame):
                run_frame(prefix_tracker, detections, features, offsets, column, frame)
        finally:
            uninstall_version_logger(logger)
        track = find_track(prefix_tracker, track_id)
        if track is None:
            raise RuntimeError(f"event track missing before frame {event_frame}: {track_id}")
        pre_snapshot = appearance_snapshot(track)
        if len(prefix_versions.get(track_id, [])) < MIN_VERSIONS:
            raise RuntimeError("version history shorter than frozen eligibility")
        snapshots = {
            "freeze0": pre_snapshot,
            "rollback4": copy.deepcopy(prefix_versions[track_id][-4]),
            "rollback8": copy.deepcopy(prefix_versions[track_id][-8]),
            "rollback16": copy.deepcopy(prefix_versions[track_id][-16]),
        }
        prefix_count = int(count_cls._count)

        for action in ACTIONS:
            tracker = copy.deepcopy(prefix_tracker)
            count_cls._count = prefix_count
            stats = {"event_update_hits": 0, "restore_hits": 0}
            token = install_action_logger(track_id, det_id, event_frame, snapshots[action], stats)
            branch_rows: list[tuple] = []
            try:
                for frame in range(event_frame, frames + 1):
                    branch_rows.extend(run_frame(tracker, detections, features, offsets, column, frame))
            finally:
                uninstall_version_logger(token)
            candidate_rows = [row for row in baseline if int(row[0]) < event_frame] + branch_rows
            name = f"{action}_e{event_index:02d}"
            path = candidate_dir / "trackers" / name / "data" / f"{SEQ}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            base.write_mot_results(path, candidate_rows)
            record = {
                "event_index": event_index,
                "event_id": str(event.event_id),
                "selection_channel": str(event.selection_channel),
                "frame": event_frame,
                "track_id": track_id,
                "det_global_idx": det_id,
                "action": action,
                "depth": int(action.replace("rollback", "")) if action.startswith("rollback") else 0,
                "tracker": name,
                "rows": len(candidate_rows),
                "row_delta_vs_b0": len(candidate_rows) - len(baseline),
                "duplicates": duplicate_count(candidate_rows),
                "event_update_hits": int(stats["event_update_hits"]),
                "restore_hits": int(stats["restore_hits"]),
                "sha256": sha256(path),
                "gt_opened": False,
            }
            if record["event_update_hits"] != 1:
                raise RuntimeError(f"frozen event did not hit exactly one update: {record}")
            if record["duplicates"]:
                raise RuntimeError(f"duplicate frame/track rows: {record}")
            reports.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    summary = pd.DataFrame(reports)
    summary_path = candidate_dir / "candidate_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "experiment_id": "M25-A0",
        "stage": "candidates_frozen",
        "seq": SEQ,
        "gt_opened": False,
        "mot20_test_reads": 0,
        "events_sha256": sha256(event_dir / "events.parquet"),
        "event_manifest_sha256": sha256(event_dir / "freeze_manifest.json"),
        "candidate_count": len(summary),
        "actions": list(ACTIONS),
        "hold_frames": HOLD_FRAMES,
        "all_event_updates_exactly_once": bool((summary.event_update_hits == 1).all()),
        "all_no_duplicates": bool((summary.duplicates == 0).all()),
        "candidate_summary_sha256": sha256(summary_path),
        "tracker_hashes": dict(zip(summary.tracker, summary.sha256)),
        "script_sha256": sha256(Path(__file__)),
    }
    json_write(candidate_dir / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def run_trackeval() -> None:
    candidate_dir = ROOT / "frozen_candidates"
    eval_dir = ROOT / "official_eval"
    if eval_dir.exists():
        raise FileExistsError(eval_dir)
    summary = pd.read_csv(candidate_dir / "candidate_summary.csv")
    baseline_name = "baseline"
    tracker_root = eval_dir / "trackers"
    baseline_data = tracker_root / baseline_name / "data"
    baseline_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "frozen_events/baseline_online.txt", baseline_data / f"{SEQ}.txt")
    for name in summary.tracker:
        source = candidate_dir / "trackers" / str(name) / "data" / f"{SEQ}.txt"
        destination = tracker_root / str(name) / "data"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / f"{SEQ}.txt")
    seqmap = eval_dir / "seqmaps/MOT20_train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\nMOT20-01\n", encoding="utf-8")
    names = [baseline_name] + summary.tracker.astype(str).tolist()
    command = [
        sys.executable,
        "TrackEval/scripts/run_mot_challenge.py",
        "--GT_FOLDER", "datasets/MOT20/train",
        "--TRACKERS_FOLDER", str(tracker_root),
        "--OUTPUT_FOLDER", str(eval_dir / "eval"),
        "--TRACKERS_TO_EVAL", *names,
        "--BENCHMARK", "MOT20",
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(seqmap),
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "",
        "--PRINT_ONLY_COMBINED", "True",
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (eval_dir / "trackeval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-8000:])
    rows = []
    for name in names:
        path = eval_dir / "eval" / name / "pedestrian_summary.txt"
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = dict(zip(lines[0].split(), lines[1].split()))
        rows.append({
            "tracker": name,
            "HOTA": float(values["HOTA"]),
            "DetA": float(values["DetA"]),
            "AssA": float(values["AssA"]),
            "IDF1": float(values["IDF1"]),
            "IDSW": int(float(values["IDSW"])),
            "Dets": int(float(values["Dets"])),
        })
    metrics = pd.DataFrame(rows)
    baseline = metrics[metrics.tracker == baseline_name].iloc[0]
    joined = summary.merge(metrics[metrics.tracker != baseline_name], on="tracker", how="left", validate="one_to_one")
    for metric in ("HOTA", "DetA", "AssA", "IDF1"):
        joined[f"delta_{metric}"] = joined[metric] - float(baseline[metric])
    joined["delta_IDSW"] = joined.IDSW - int(baseline.IDSW)
    joined.to_csv(ROOT / "candidate_official_metrics.csv", index=False)
    paired = joined.pivot(index="event_index", columns="action", values="delta_HOTA")
    rollback_columns = [column for column in paired.columns if str(column).startswith("rollback")]
    paired["best_rollback_delta"] = paired[rollback_columns].max(axis=1)
    paired["best_rollback_action"] = paired[rollback_columns].idxmax(axis=1)
    paired["paired_gain_over_freeze0"] = paired.best_rollback_delta - paired.freeze0
    paired.reset_index().to_csv(ROOT / "paired_action_metrics.csv", index=False)
    rollback = joined[joined.action.str.startswith("rollback")].copy()
    best_rollback = rollback.sort_values(["delta_HOTA", "delta_AssA"], ascending=[False, False]).iloc[0]
    best_freeze = joined[joined.action == "freeze0"].sort_values(["delta_HOTA", "delta_AssA"], ascending=[False, False]).iloc[0]
    positive_rollback = int((rollback.delta_HOTA > 0).sum())
    max_paired_gain = float(paired.paired_gain_over_freeze0.max())
    gate_pass = bool(
        float(best_rollback.delta_HOTA) >= 0.15
        and positive_rollback >= 4
        and max_paired_gain >= 0.10
        and (summary.duplicates == 0).all()
        and (ROOT / "frozen_events/baseline_online.txt").read_bytes() == BASELINE_REFERENCE.read_bytes()
    )
    decision = "PASS_M25_A0_AUTHORIZE_MULTI_SEQUENCE_CAPACITY" if gate_pass else "FAIL_M25_A0_CLOSE_VERSIONED_MEMORY_ROLLBACK"
    report = {
        "experiment_id": "M25-A0",
        "status": "completed",
        "decision": decision,
        "teacher_only": True,
        "deployable": False,
        "gt_opened_after_candidate_freeze": True,
        "mot20_test_reads": 0,
        "test_submission": False,
        "baseline": {key: (int(baseline[key]) if key in {"IDSW", "Dets"} else float(baseline[key])) for key in ("HOTA", "DetA", "AssA", "IDF1", "IDSW", "Dets")},
        "best_rollback": best_rollback.to_dict(),
        "best_freeze0": best_freeze.to_dict(),
        "positive_rollback_candidates": positive_rollback,
        "maximum_paired_gain_over_freeze0": max_paired_gain,
        "gate": {
            "required_best_rollback_delta_HOTA": 0.15,
            "required_positive_rollback_candidates": 4,
            "required_paired_gain_over_freeze0": 0.10,
            "pass": gate_pass,
        },
        "candidate_count": len(joined),
        "candidate_freeze_manifest_sha256": sha256(ROOT / "frozen_candidates/freeze_manifest.json"),
        "event_freeze_manifest_sha256": sha256(ROOT / "frozen_events/freeze_manifest.json"),
        "conclusion": (
            "Exact version rollback creates sufficient single-sequence capacity for expansion."
            if gate_pass else
            "Exact version rollback does not create sufficient corrective capacity beyond one-update freezing on the M01 kill gate."
        ),
    }
    json_write(ROOT / "report.json", report)
    pd.DataFrame([{
        "experiment_id": "M25-A0",
        "baseline_HOTA": float(baseline.HOTA),
        "best_rollback_HOTA": float(best_rollback.HOTA),
        "best_rollback_delta_HOTA": float(best_rollback.delta_HOTA),
        "positive_rollback_candidates": positive_rollback,
        "maximum_paired_gain_over_freeze0": max_paired_gain,
        "gate_pass": int(gate_pass),
        "decision": decision,
    }]).to_csv(ROOT / "summary.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("\nTOP ROLLBACK\n", rollback.sort_values("delta_HOTA", ascending=False).head(12).to_string(index=False))
    print("\nPAIRED\n", paired.sort_values("paired_gain_over_freeze0", ascending=False).head(12).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-events", "generate-candidates", "evaluate"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-events":
        freeze_events()
    elif args.stage == "generate-candidates":
        generate_candidates()
    else:
        run_trackeval()


if __name__ == "__main__":
    main()
