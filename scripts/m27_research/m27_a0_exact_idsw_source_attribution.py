from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import dmm_base_tracker as base  # noqa: E402

SEQ = "MOT20-01"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"
REFERENCE = REPO / ".mcp_tmp/m24_online_b0_parity/run1/track_results/MOT20-01.txt"
ROOT = REPO / "outputs/mot20_m27_20260726/m27_a0_exact_idsw_source_attribution_m01"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def count_class():
    for cls in base.DMMTrack.mro():
        if hasattr(cls, "_count"):
            return cls
    raise RuntimeError("track counter missing")


def globals_() -> dict:
    return base.DMMBaseTracker.update.__globals__


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


def output_rows(tracks, frame: int, cfg):
    rows = []
    for track in tracks:
        x, y, width, height = [float(value) for value in track.tlwh]
        if width * height < float(cfg.min_box_area) or width <= 0 or height <= 0:
            continue
        if width / max(height, 1e-12) > float(cfg.aspect_ratio_thresh):
            continue
        rows.append((frame, int(track.track_id), x, y, width, height, float(track.score)))
    return rows


def freeze_runtime() -> None:
    frozen = ROOT / "frozen_runtime"
    if frozen.exists():
        raise FileExistsError(frozen)
    frozen.mkdir(parents=True)
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
    implementation = globals_()
    cls = implementation["DMMTrack"]
    original_assignment = implementation["matching"].linear_assignment
    original_update = cls.update
    original_reactivate = cls.re_activate
    original_activate = cls.activate
    context = {"frame": 0, "call_index": 0, "stage": "none", "threshold": None}
    logs: list[dict] = []
    rows = []

    def assignment_hook(cost, thresh):
        index = int(context["call_index"])
        stages = ("primary", "second_low_score", "unconfirmed")
        context["stage"] = stages[index] if index < len(stages) else f"extra_assignment_{index}"
        context["threshold"] = float(thresh)
        context["call_index"] = index + 1
        return original_assignment(cost, thresh=thresh)

    def record(method: str, track, detection, frame_id: int, stage: str, before_start: int, before_len: int, before_state: int):
        logs.append({
            "seq": SEQ,
            "frame": int(frame_id),
            "method": method,
            "stage": stage,
            "assignment_call_index": int(context["call_index"] - 1),
            "assignment_threshold": context["threshold"],
            "track_id": int(track.track_id),
            "det_global_idx": int(getattr(detection, "det_global_idx", -1)),
            "det_score": float(getattr(detection, "score", 0.0)),
            "track_start_frame_before": int(before_start),
            "tracklet_len_before": int(before_len),
            "track_state_before": int(before_state),
            "track_start_frame_after": int(track.start_frame),
            "tracklet_len_after": int(track.tracklet_len),
            "track_state_after": int(track.state),
            "gt_opened": False,
        })

    def update(self, new_track, frame_id):
        before = (int(self.start_frame), int(self.tracklet_len), int(self.state))
        result = original_update(self, new_track, frame_id)
        record("update", self, new_track, frame_id, str(context["stage"]), *before)
        return result

    def reactivate(self, new_track, frame_id, new_id=False):
        before = (int(self.start_frame), int(self.tracklet_len), int(self.state))
        result = original_reactivate(self, new_track, frame_id, new_id=new_id)
        record("re_activate_new_id" if new_id else "re_activate", self, new_track, frame_id, str(context["stage"]), *before)
        return result

    def activate(self, kalman_filter, frame_id, activate_new_after_first=False):
        before = (int(getattr(self, "start_frame", 0)), int(self.tracklet_len), int(self.state))
        result = original_activate(self, kalman_filter, frame_id, activate_new_after_first=activate_new_after_first)
        record("activate", self, self, frame_id, "newborn", *before)
        return result

    implementation["matching"].linear_assignment = assignment_hook
    cls.update = update
    cls.re_activate = reactivate
    cls.activate = activate
    try:
        for frame in range(1, frames + 1):
            context.update({"frame": frame, "call_index": 0, "stage": "none", "threshold": None})
            boxes, scores, feats, gids = frame_data(detections, features, offsets, column, frame)
            tracks = tracker.update(boxes, scores, feats, gids)
            rows.extend(output_rows(tracks, frame, cfg))
    finally:
        implementation["matching"].linear_assignment = original_assignment
        cls.update = original_update
        cls.re_activate = original_reactivate
        cls.activate = original_activate

    baseline = frozen / "baseline_online.txt"
    base.write_mot_results(baseline, rows)
    if baseline.read_bytes() != REFERENCE.read_bytes():
        raise RuntimeError("B0 runtime replay mismatch")
    frame = pd.DataFrame(logs)
    frame.to_parquet(frozen / "association_updates.parquet", index=False)
    manifest = {
        "experiment_id": "M27-A0",
        "stage": "runtime_frozen",
        "gt_opened": False,
        "mot20_test_reads": 0,
        "frames": frames,
        "output_rows": len(rows),
        "association_updates": len(frame),
        "baseline_sha256": sha256(baseline),
        "reference_sha256": sha256(REFERENCE),
        "association_updates_sha256": sha256(frozen / "association_updates.parquet"),
        "input_sha256": sha256(DUMP),
        "script_sha256": sha256(Path(__file__)),
        "methods": frame.method.value_counts().to_dict(),
        "stages": frame.stage.value_counts().to_dict(),
    }
    json_write(frozen / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def original_tracker_mapping(prepared):
    mapping: dict[int, int] = {}
    frame_original_ids = []
    for remapped, parent_rows in zip(prepared.data["tracker_ids"], prepared.processed_parent_row_indices):
        original = prepared.parent_row_ids[parent_rows]
        if len(remapped) != len(original):
            raise RuntimeError("processed tracker alignment mismatch")
        frame_original_ids.append(original.astype(np.int64))
        for new, old in zip(remapped, original):
            new, old = int(new), int(old)
            if new in mapping and mapping[new] != old:
                raise RuntimeError("non-bijective tracker remap")
            mapping[new] = old
    return mapping, frame_original_ids


def reconstruct_clear_idsw(data, tracker_map: dict[int, int]):
    num_gt_ids = int(data["num_gt_ids"])
    prev_tracker_id = np.nan * np.zeros(num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(num_gt_ids)
    last_match_frame = np.full(num_gt_ids, -1, dtype=np.int64)
    events = []
    matched_rows = []
    threshold = 0.5
    for timestep, (gt_ids_t, tracker_ids_t) in enumerate(zip(data["gt_ids"], data["tracker_ids"]), start=1):
        if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
            prev_timestep_tracker_id[:] = np.nan
            continue
        similarity = data["similarity_scores"][timestep - 1]
        score = (tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]])
        score = 1000 * score + similarity
        score[similarity < threshold - np.finfo("float").eps] = 0
        rows, cols = linear_sum_assignment(-score)
        keep = score[rows, cols] > np.finfo("float").eps
        rows, cols = rows[keep], cols[keep]
        matched_gt = gt_ids_t[rows]
        matched_tracker = tracker_ids_t[cols]
        previous = prev_tracker_id[matched_gt]
        switches = (~np.isnan(previous)) & (matched_tracker != previous)
        for local, (gt_id, tracker_id, sim) in enumerate(zip(matched_gt, matched_tracker, similarity[rows, cols])):
            gt_id, tracker_id = int(gt_id), int(tracker_id)
            original = int(tracker_map[tracker_id])
            previous_remap = None if np.isnan(previous[local]) else int(previous[local])
            previous_original = None if previous_remap is None else int(tracker_map[previous_remap])
            gap = None if last_match_frame[gt_id] < 0 else int(timestep - last_match_frame[gt_id])
            matched_rows.append({
                "frame": timestep,
                "gt_id": gt_id,
                "tracker_id_remapped": tracker_id,
                "tracker_id": original,
                "previous_tracker_id": previous_original,
                "gap_since_last_match": gap,
                "similarity": float(sim),
                "is_idsw": int(switches[local]),
            })
            if switches[local]:
                events.append(dict(matched_rows[-1]))
            last_match_frame[gt_id] = timestep
        prev_tracker_id[matched_gt] = matched_tracker
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt] = matched_tracker
    return pd.DataFrame(events), pd.DataFrame(matched_rows)


def attribute() -> None:
    frozen = ROOT / "frozen_runtime"
    out = ROOT / "attribution"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    m37 = load_module("m27_exact_hota", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    prepared = m37.PreparedExactHOTA(SEQ, frozen / "baseline_online.txt", out / "trackeval_cache")
    from trackeval.metrics import CLEAR
    official = CLEAR({"PRINT_CONFIG": False}).eval_sequence(prepared.data)
    tracker_map, _ = original_tracker_mapping(prepared)
    switches, matches = reconstruct_clear_idsw(prepared.data, tracker_map)
    if int(official["IDSW"]) != len(switches):
        raise RuntimeError(f"CLEAR reconstruction mismatch: {official['IDSW']} != {len(switches)}")
    updates = pd.read_parquet(frozen / "association_updates.parquet")
    grouped = {(int(frame), int(track_id)): group.copy() for (frame, track_id), group in updates.groupby(["frame", "track_id"], sort=False)}
    records = []
    for switch in switches.itertuples(index=False):
        key = (int(switch.frame), int(switch.tracker_id))
        candidates = grouped.get(key, pd.DataFrame())
        if len(candidates):
            preferred = candidates.sort_values(
                ["method", "assignment_call_index"],
                key=lambda values: values.map({"re_activate": 0, "re_activate_new_id": 0, "update": 1, "activate": 2}).fillna(3) if values.name == "method" else values,
            ).iloc[0]
            update = preferred.to_dict()
            source = f"{update['method']}:{update['stage']}"
            track_age = int(switch.frame - int(update["track_start_frame_after"]))
            det_gid = int(update["det_global_idx"])
            update_count = len(candidates)
        else:
            source = "no_current_frame_update"
            track_age = None
            det_gid = None
            update_count = 0
        gap = int(switch.gap_since_last_match) if switch.gap_since_last_match is not None else None
        records.append({
            **switch._asdict(),
            "source": source,
            "update_candidates": update_count,
            "det_global_idx": det_gid,
            "current_tracker_age": track_age,
            "temporal_type": "immediate" if gap == 1 else "after_gap",
            "age_type": "unknown" if track_age is None else ("newborn_age0" if track_age == 0 else ("young_age1_5" if track_age <= 5 else "established_age6plus")),
        })
    attributed = pd.DataFrame(records)
    attributed.to_csv(out / "idsw_events.csv", index=False)
    matches.to_parquet(out / "clear_matches.parquet", index=False)
    stage_table = attributed.groupby(["temporal_type", "source"], dropna=False).size().reset_index(name="IDSW")
    age_table = attributed.groupby(["temporal_type", "age_type"], dropna=False).size().reset_index(name="IDSW")
    stage_table.to_csv(out / "idsw_by_source.csv", index=False)
    age_table.to_csv(out / "idsw_by_age.csv", index=False)
    summary = {
        "experiment_id": "M27-A0",
        "status": "completed",
        "diagnostic_only": True,
        "gt_opened_after_runtime_freeze": True,
        "mot20_test_reads": 0,
        "test_submission": False,
        "official_clear_idsw": int(official["IDSW"]),
        "reconstructed_idsw": len(attributed),
        "immediate_idsw": int((attributed.temporal_type == "immediate").sum()),
        "after_gap_idsw": int((attributed.temporal_type == "after_gap").sum()),
        "by_source": attributed.source.value_counts().to_dict(),
        "by_age": attributed.age_type.value_counts().to_dict(),
        "source_temporal": {
            f"{row.temporal_type}|{row.source}": int(row.IDSW)
            for row in stage_table.itertuples(index=False)
        },
        "runtime_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "idsw_events_sha256": sha256(out / "idsw_events.csv"),
        "script_sha256": sha256(Path(__file__)),
    }
    json_write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nBY SOURCE\n", stage_table.to_string(index=False))
    print("\nBY AGE\n", age_table.to_string(index=False))
    print("\nEVENTS\n", attributed.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-runtime", "attribute"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-runtime":
        freeze_runtime()
    else:
        attribute()


if __name__ == "__main__":
    main()
