from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEQ = "MOT20-01"
ROOT = REPO / "outputs/mot20_m28_20260726/m28_a0_deferred_identity_inheritance_m01_v1"
M27 = REPO / "outputs/mot20_m27_20260726/m27_a0_exact_idsw_source_attribution_m01/frozen_runtime"
BASELINE = M27 / "baseline_online.txt"
UPDATES = M27 / "association_updates.parquet"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"
GT = REPO / "datasets/MOT20/train/MOT20-01/gt/gt.txt"
TOPK = 8
MAX_GAP = 120
MIN_ROWS = 4
MIN_REID = 2


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


def parse_tracker(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            fields = line.rstrip("\n").split(",")
            if len(fields) < 7:
                continue
            frame = int(float(fields[0])); track_id = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            rows.append({
                "row_index": len(rows), "line_index": line_index, "frame": frame, "track_id": track_id,
                "x": x, "y": y, "w": width, "h": height,
                "cx": x + 0.5 * width, "cy": y + 0.5 * height,
                "fields": fields,
            })
    return rows


def unit(value):
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return None if norm <= 1e-12 or not np.isfinite(array).all() else array / norm


def prototype(feature_rows, features, limit: int, from_end: bool):
    selected = feature_rows[-limit:] if from_end else feature_rows[:limit]
    vectors = []
    for gid in selected:
        if gid < 0 or gid >= len(features):
            continue
        vector = unit(features[int(gid)])
        if vector is not None:
            vectors.append(vector)
    if len(vectors) < 1:
        return None, len(vectors)
    mean = np.mean(np.stack(vectors), axis=0)
    return unit(mean), len(vectors)


def velocity(rows):
    tail = rows[-5:]
    if len(tail) < 2:
        return 0.0, 0.0
    frames = np.asarray([row["frame"] for row in tail], dtype=float)
    if len(np.unique(frames)) < 2:
        return 0.0, 0.0
    cx = np.asarray([row["cx"] for row in tail], dtype=float)
    cy = np.asarray([row["cy"] for row in tail], dtype=float)
    return float(np.polyfit(frames, cx, 1)[0]), float(np.polyfit(frames, cy, 1)[0])


def duplicate_count(ids: np.ndarray, rows) -> int:
    keys = [(int(row["frame"]), int(ids[index])) for index, row in enumerate(rows)]
    return len(keys) - len(set(keys))


def freeze_candidates() -> None:
    out = ROOT / "frozen_candidates"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rows = parse_tracker(BASELINE)
    updates = pd.read_parquet(UPDATES)
    events = updates[(updates.method.astype(str) == "update") & (updates.stage.astype(str) == "unconfirmed")].copy()
    events.sort_values(["frame", "track_id"], inplace=True)
    events.drop_duplicates(["frame", "track_id"], inplace=True)
    events.reset_index(drop=True, inplace=True)
    events.insert(0, "event_index", np.arange(len(events), dtype=int))
    dump = np.load(DUMP, allow_pickle=True)
    features = np.asarray(dump["features"])

    by_track_rows = defaultdict(list)
    for row in rows:
        by_track_rows[int(row["track_id"])].append(row)
    by_track_updates = defaultdict(list)
    for update in updates.itertuples(index=False):
        gid = int(update.det_global_idx)
        if gid >= 0:
            by_track_updates[int(update.track_id)].append((int(update.frame), gid))
    track_summary = {}
    for track_id, track_rows in by_track_rows.items():
        track_rows.sort(key=lambda row: (row["frame"], row["row_index"]))
        updates_for_track = sorted(by_track_updates.get(track_id, []))
        valid_gids = [gid for _frame, gid in updates_for_track if unit(features[gid]) is not None]
        proto_end, count_end = prototype(valid_gids, features, 8, True)
        vx, vy = velocity(track_rows)
        track_summary[track_id] = {
            "first_frame": int(track_rows[0]["frame"]), "last_frame": int(track_rows[-1]["frame"]),
            "rows": len(track_rows), "first": track_rows[0], "last": track_rows[-1],
            "frames": {int(row["frame"]) for row in track_rows},
            "valid_gids": valid_gids, "prototype_end": proto_end, "reid_count": count_end,
            "vx": vx, "vy": vy,
        }

    candidates = []
    event_records = []
    for event in events.itertuples(index=False):
        event_index = int(event.event_index); frame = int(event.frame); young_id = int(event.track_id)
        young = track_summary.get(young_id)
        if young is None:
            continue
        young_updates = [gid for update_frame, gid in sorted(by_track_updates.get(young_id, [])) if update_frame <= frame]
        young_proto, young_count = prototype(young_updates, features, 2, False)
        if young_proto is None:
            continue
        young_suffix_frames = {int(row["frame"]) for row in by_track_rows[young_id] if int(row["frame"]) >= frame}
        event_row = next((row for row in by_track_rows[young_id] if int(row["frame"]) >= frame), young["first"])
        records = []
        for old_id, old in track_summary.items():
            if old_id == young_id or old["last_frame"] >= frame:
                continue
            gap = frame - int(old["last_frame"])
            if gap < 1 or gap > MAX_GAP or int(old["rows"]) < MIN_ROWS or int(old["reid_count"]) < MIN_REID:
                continue
            if old["frames"] & young_suffix_frames:
                continue
            old_proto = old["prototype_end"]
            if old_proto is None:
                continue
            appearance = float(old_proto @ young_proto)
            dt = float(frame - old["last_frame"])
            predicted_x = float(old["last"]["cx"]) + float(old["vx"]) * dt
            predicted_y = float(old["last"]["cy"]) + float(old["vy"]) * dt
            scale = max(0.5 * (float(old["last"]["h"]) + float(event_row["h"])), 1.0)
            motion = math.hypot(float(event_row["cx"]) - predicted_x, float(event_row["cy"]) - predicted_y) / scale
            height_ratio = max(float(event_row["h"]), 1e-3) / max(float(old["last"]["h"]), 1e-3)
            score = appearance - 0.35 * motion - 0.03 * math.log1p(gap) - 0.08 * abs(math.log(height_ratio))
            records.append({
                "event_index": event_index, "frame": frame, "young_track_id": young_id,
                "old_track_id": int(old_id), "gap": gap, "old_rows": int(old["rows"]),
                "old_reid_count": int(old["reid_count"]), "young_reid_count": int(young_count),
                "appearance_cos": appearance, "motion_error": motion,
                "height_ratio": height_ratio, "candidate_score": score,
                "gt_opened": False,
            })
        records.sort(key=lambda item: (-item["candidate_score"], -item["appearance_cos"], item["motion_error"], item["old_track_id"]))
        selected = records[:TOPK]
        for rank, record in enumerate(selected, start=1):
            record["candidate_rank"] = rank
            candidates.append(record)
        event_records.append({
            "event_index": event_index, "frame": frame, "young_track_id": young_id,
            "candidate_count": len(selected), "young_suffix_rows": sum(int(row["frame"]) >= frame for row in by_track_rows[young_id]),
            "gt_opened": False,
        })
    candidate_frame = pd.DataFrame(candidates)
    event_frame = pd.DataFrame(event_records)
    if candidate_frame.empty:
        raise RuntimeError("no deferred identity candidates")
    candidate_frame.to_parquet(out / "candidates.parquet", index=False)
    event_frame.to_parquet(out / "events.parquet", index=False)
    manifest = {
        "experiment_id": "M28-A0", "stage": "candidates_frozen", "gt_opened": False,
        "mot20_test_reads": 0, "unconfirmed_events": len(events), "frozen_events": len(event_frame),
        "candidate_actions": len(candidate_frame), "candidate_topk": TOPK, "max_gap": MAX_GAP,
        "baseline_sha256": sha256(BASELINE), "runtime_updates_sha256": sha256(UPDATES),
        "candidates_sha256": sha256(out / "candidates.parquet"), "events_sha256": sha256(out / "events.parquet"),
        "input_sha256": sha256(DUMP), "script_sha256": sha256(Path(__file__)),
        "forbidden_gt_columns": [],
    }
    json_write(out / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(candidate_frame.groupby("event_index").agg(candidates=("old_track_id", "size"), best_score=("candidate_score", "max"), best_app=("appearance_cos", "max"), min_gap=("gap", "min")).head(60).to_string())


def write_tracker(path: Path, prepared, row_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for fields, track_id in zip(prepared.parent_rows, row_ids):
            output = list(fields)
            output[1] = str(int(track_id))
            handle.write(",".join(output) + "\n")


def official_eval(track_dir: Path, tracker_name: str, out: Path):
    command = [
        sys.executable, "scripts/eval_motstyle_trackeval.py",
        "--benchmark-name", "MOT20", "--split-to-eval", "train",
        "--gt-root", "datasets/MOT20/train", "--results-dir", str(track_dir),
        "--tracker-name", tracker_name, "--work-dir", str(out), "--keep-workdir", "--seqs", SEQ,
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out.parent / "official_trackeval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-8000:])
    summary = out / "eval" / tracker_name / "pedestrian_summary.txt"
    lines = [line.strip() for line in summary.read_text().splitlines() if line.strip()]
    values = dict(zip(lines[0].split(), lines[1].split()))
    return {key: float(values[key]) for key in ("HOTA", "DetA", "AssA", "IDF1")} | {"IDSW": int(float(values["IDSW"])), "Dets": int(float(values["Dets"]))}


def label_capacity() -> None:
    frozen = ROOT / "frozen_candidates"
    out = ROOT / "teacher_capacity"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    candidates = pd.read_parquet(frozen / "candidates.parquet")
    m37 = load_module("m28_exact_hota", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    prepared = m37.PreparedExactHOTA(SEQ, BASELINE, out / "exact_cache")
    baseline_ids = prepared.parent_row_ids.copy()
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)
    parent_frames = np.asarray([int(float(fields[0])) for fields in prepared.parent_rows], dtype=np.int32)
    labels = []
    for ordinal, candidate in enumerate(candidates.itertuples(index=False), start=1):
        ids = baseline_ids.copy()
        mask = (ids == int(candidate.young_track_id)) & (parent_frames >= int(candidate.frame))
        ids[mask] = int(candidate.old_track_id)
        duplicate = False
        for frame in np.unique(parent_frames[mask]):
            frame_ids = ids[parent_frames == frame]
            if len(frame_ids) != len(np.unique(frame_ids)):
                duplicate = True
                break
        if duplicate or not mask.any():
            labels.append({**candidate._asdict(), "status": "invalid", "modified_rows": int(mask.sum()), "delta_HOTA": math.nan})
            continue
        metrics = prepared.evaluate_row_ids_incremental(ids)
        labels.append({
            **candidate._asdict(), "status": "success", "modified_rows": int(mask.sum()),
            "HOTA": float(metrics["HOTA"]), "DetA": float(metrics["DetA"]), "AssA": float(metrics["AssA"]),
            "delta_HOTA": float(metrics["HOTA"] - baseline_metrics["HOTA"]),
            "delta_DetA": float(metrics["DetA"] - baseline_metrics["DetA"]),
            "delta_AssA": float(metrics["AssA"] - baseline_metrics["AssA"]),
        })
        if ordinal % 100 == 0:
            print("labeled", ordinal, "/", len(candidates), flush=True)
    label_frame = pd.DataFrame(labels)
    label_frame.to_parquet(out / "exact_labels.parquet", index=False)
    successful = label_frame[label_frame.status == "success"].copy()
    successful.sort_values(["delta_HOTA", "candidate_score"], ascending=[False, False], inplace=True)
    current_ids = baseline_ids.copy()
    current_metrics = dict(baseline_metrics)
    used_young = set(); used_old = set(); selected = []
    for candidate in successful[successful.delta_HOTA > 0].itertuples(index=False):
        young, old = int(candidate.young_track_id), int(candidate.old_track_id)
        if young in used_young or old in used_old:
            continue
        proposal = current_ids.copy()
        mask = (proposal == young) & (parent_frames >= int(candidate.frame))
        if not mask.any():
            continue
        proposal[mask] = old
        valid = True
        for frame in np.unique(parent_frames[mask]):
            frame_ids = proposal[parent_frames == frame]
            if len(frame_ids) != len(np.unique(frame_ids)):
                valid = False; break
        if not valid:
            continue
        metrics = prepared.evaluate_row_ids_incremental(proposal)
        step_gain = float(metrics["HOTA"] - current_metrics["HOTA"])
        if step_gain <= 0:
            continue
        current_ids = proposal; current_metrics = metrics
        used_young.add(young); used_old.add(old)
        selected.append({
            "event_index": int(candidate.event_index), "frame": int(candidate.frame),
            "young_track_id": young, "old_track_id": old,
            "individual_delta_HOTA": float(candidate.delta_HOTA), "step_delta_HOTA": step_gain,
            "modified_rows": int(mask.sum()),
        })
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(out / "selected_actions.csv", index=False)
    tracker_path = out / "track_results" / f"{SEQ}.txt"
    write_tracker(tracker_path, prepared, current_ids)
    official = official_eval(tracker_path.parent, "m28_a0_deferred_identity_teacher", out / "official_eval")
    positive = int((successful.delta_HOTA > 0).sum())
    best_single = float(successful.delta_HOTA.max()) if len(successful) else 0.0
    combined_delta = float(current_metrics["HOTA"] - baseline_metrics["HOTA"])
    gate_pass = bool(positive >= 8 and best_single >= 0.10 and combined_delta >= 0.50 and abs(float(official["HOTA"]) - float(current_metrics["HOTA"])) < 0.002)
    decision = "PASS_M28_A0_AUTHORIZE_TRUE_ONLINE_ANONYMOUS_IDENTITY" if gate_pass else "FAIL_M28_A0_CLOSE_DEFERRED_IDENTITY_INHERITANCE"
    report = {
        "experiment_id": "M28-A0", "status": "completed", "decision": decision,
        "teacher_only": True, "deployable": False, "gt_opened_after_candidate_freeze": True,
        "mot20_test_reads": 0, "test_submission": False,
        "baseline_metrics": {key: float(baseline_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "candidate_actions": len(candidates), "successful_actions": len(successful), "positive_actions": positive,
        "best_single_delta_HOTA": best_single, "selected_actions": len(selected),
        "combined_exact_metrics": {key: float(current_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "combined_delta_HOTA": combined_delta, "official_trackeval": official,
        "gate": {"minimum_positive_actions": 8, "minimum_best_single_delta_HOTA": 0.10, "minimum_combined_delta_HOTA": 0.50, "pass": gate_pass},
        "candidate_manifest_sha256": sha256(frozen / "freeze_manifest.json"),
        "labels_sha256": sha256(out / "exact_labels.parquet"), "tracker_sha256": sha256(tracker_path),
    }
    json_write(ROOT / "report.json", report)
    pd.DataFrame([{
        "experiment_id": "M28-A0", "baseline_HOTA": baseline_metrics["HOTA"], "positive_actions": positive,
        "best_single_delta_HOTA": best_single, "selected_actions": len(selected), "combined_HOTA": current_metrics["HOTA"],
        "combined_delta_HOTA": combined_delta, "official_HOTA": official["HOTA"], "official_IDSW": official["IDSW"],
        "gate_pass": int(gate_pass), "decision": decision,
    }]).to_csv(ROOT / "summary.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nTOP LABELS\n", successful.head(30).to_string(index=False))
    print("\nSELECTED\n", selected_frame.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-candidates", "label-capacity"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-candidates":
        freeze_candidates()
    else:
        label_capacity()


if __name__ == "__main__":
    main()
