"""M23-5-0 mechanism-decomposed NMS-conflict action-space oracle on MOT20 train.

The audit starts from the exact M23-3-1 dense replacement context and uses the
M23-4 bridge candidates whose box overlaps an existing context row at IoU>=0.5.
Each conflict is classified with train GT into one of three fixed mechanisms:

* other_target: the host row best overlaps a different valid GT; retain the
  host and add the suppressed candidate under the inherited existing ID.
* same_target: host and candidate represent the same GT; migrate the host row
  to the inherited ID and retain whichever box has higher target IoU.
* unmatched: the host row does not overlap any valid GT at IoU>=0.5; replace it
  with the candidate under the inherited ID.

This is a GT-derived action-space ceiling, not a deployable result. No model,
threshold sweep, parameter sweep, or locked P15 evaluation is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "dense_replacement_context",
    "safe_gap_context",
    "all_additions_reference",
    "distinct_person_dual_retention_oracle",
    "mechanism_decomposed_oracle",
    "safe_gap_plus_mechanism_oracle",
)
CONFLICT_IOU = 0.50
VALID_GT_IOU = 0.50
MECHANISM_GATE_HOTA = 78.20
COMBINED_GATE_HOTA = 78.25
METRIC_FIELDS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def parse_tracker_line(line: str) -> Tuple[int, int, List[str]]:
    fields = line.rstrip("\n").split(",")
    if len(fields) < 7:
        raise ValueError(f"invalid MOT row: {line!r}")
    return int(fields[0]), int(fields[1]), fields


def row_box(fields: Sequence[str]) -> Tuple[float, float, float, float]:
    x, y, width, height = map(float, fields[2:6])
    return x, y, x + width, y + height


def iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def tracker_sort_key(line: str) -> Tuple[int, int, float, float, str]:
    frame, identity, fields = parse_tracker_line(line)
    return frame, identity, float(fields[2]), float(fields[3]), line


def validate_tracker(lines: Sequence[str], label: str) -> None:
    seen: set[Tuple[int, int]] = set()
    previous_frame = -1
    for line in lines:
        frame, identity, _ = parse_tracker_line(line)
        if frame < previous_frame:
            raise RuntimeError(f"{label}: frames not monotonic")
        previous_frame = frame
        key = (frame, identity)
        if key in seen:
            raise RuntimeError(f"{label}: duplicate frame/ID {key}")
        seen.add(key)


def replace_identity(line: str, identity: int) -> str:
    _, _, fields = parse_tracker_line(line)
    fields[1] = str(int(identity))
    return ",".join(fields)


def candidate_line(action: Mapping[str, object], candidate_row: np.void) -> str:
    x1 = float(candidate_row["candidate_x1"])
    y1 = float(candidate_row["candidate_y1"])
    x2 = float(candidate_row["candidate_x2"])
    y2 = float(candidate_row["candidate_y2"])
    return ",".join([
        str(int(action["frame"])), str(int(action["inherited_track_id"])),
        f"{x1:.6f}", f"{y1:.6f}", f"{x2 - x1:.6f}", f"{y2 - y1:.6f}",
        f"{float(candidate_row['candidate_score']):.6f}", "-1", "-1", "-1",
    ])


def collision_priority(action: Mapping[str, object]) -> Tuple[float, float, int, float, int, int, int]:
    return (
        float(action["candidate_target_iou"]), float(action["candidate_host_iou"]),
        -int(action["gap_frames"]), float(action["minimum_segment_purity"]),
        int(action["matched_segment_support"]), -int(action["inherited_track_id"]),
        -int(action["candidate_row_index"]),
    )


def hash_action_plan(actions: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    fields = (
        "sequence", "frame", "source_line_index", "source_line_sha256", "source_track_id",
        "inherited_track_id", "target_gt_id", "mechanism", "box_choice", "candidate_row_index",
        "global_pre_idx", "candidate_target_iou", "source_target_iou", "source_best_other_iou",
        "source_best_other_gt_id", "candidate_host_iou",
    )
    for action in sorted(actions, key=lambda row: (str(row["sequence"]), int(row["source_line_index"]))):
        value = {field: action[field] for field in fields}
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def classify_host(
    source_box: Sequence[float],
    target_gt_id: int,
    valid_gt: Sequence[object],
) -> Tuple[str, float, float, int]:
    target_rows = [item for item in valid_gt if int(item.gt_id) == int(target_gt_id)]
    if len(target_rows) != 1:
        raise RuntimeError(f"expected one target GT {target_gt_id}, got {len(target_rows)}")
    source_target_iou = iou_box(source_box, target_rows[0].box)
    other = [
        (iou_box(source_box, item.box), int(item.gt_id))
        for item in valid_gt if int(item.gt_id) != int(target_gt_id)
    ]
    source_best_other_iou, source_best_other_gt_id = max(other, default=(0.0, 0))
    if source_target_iou >= VALID_GT_IOU and source_target_iou >= source_best_other_iou:
        mechanism = "same_target"
    elif source_best_other_iou >= VALID_GT_IOU:
        mechanism = "other_target"
    else:
        mechanism = "unmatched"
    return mechanism, source_target_iou, source_best_other_iou, source_best_other_gt_id


def build_plans(
    repo: Path,
    appearance_dir: Path,
    compact_dir: Path,
    gap_dir: Path,
    replacement_dir: Path,
    gt_root: Path,
) -> dict:
    m23 = load_module(repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py", "m23_for_m23_5_mechanism")
    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    candidate_keys_path = compact_dir / "candidate_keys.npy"
    candidates = np.load(candidate_manifest_path, mmap_mode="r", allow_pickle=False)
    keys = np.load(candidate_keys_path, mmap_mode="r", allow_pickle=False)
    if len(candidates) != len(keys):
        raise RuntimeError("candidate manifest/key row mismatch")
    bridge_events = [
        row for row in read_csv_rows(gap_dir / "bridge_events.csv")
        if float(row["maximum_context_iou"]) >= CONFLICT_IOU
    ]
    by_sequence: MutableMapping[str, List[dict]] = defaultdict(list)
    for row in bridge_events:
        by_sequence[row["sequence"]].append(row)

    source_hashes = {
        "candidate_manifest": sha256_file(candidate_manifest_path),
        "candidate_keys": sha256_file(candidate_keys_path),
        "bridge_events": sha256_file(gap_dir / "bridge_events.csv"),
        "gap_report": sha256_file(gap_dir / "report.json"),
        "gap_manifest": sha256_file(gap_dir / "manifest.json"),
        "dense_trackers": {}, "safe_trackers": {}, "all_addition_trackers": {}, "gt": {},
    }
    selected_by_sequence: Dict[str, List[dict]] = {}
    summary_rows: List[dict] = []
    all_counts = Counter()

    for sequence in SEQUENCES:
        dense_path = replacement_dir / "eval_work/trackers/dense_replacement_oracle/data" / f"{sequence}.txt"
        safe_path = gap_dir / "eval_work/trackers/safe_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        all_path = gap_dir / "eval_work/trackers/all_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        gt_path = gt_root / sequence / "gt/gt.txt"
        source_hashes["dense_trackers"][sequence] = sha256_file(dense_path)
        source_hashes["safe_trackers"][sequence] = sha256_file(safe_path)
        source_hashes["all_addition_trackers"][sequence] = sha256_file(all_path)
        source_hashes["gt"][sequence] = sha256_file(gt_path)
        gt = m23.load_gt(gt_path)
        dense_lines = dense_path.read_text(encoding="utf-8").splitlines()
        validate_tracker(dense_lines, f"{sequence}/dense")
        dense_ids: MutableMapping[int, set[int]] = defaultdict(set)
        rows_by_frame: MutableMapping[int, List[dict]] = defaultdict(list)
        for line_index, line in enumerate(dense_lines):
            frame, identity, fields = parse_tracker_line(line)
            dense_ids[frame].add(identity)
            rows_by_frame[frame].append({
                "line_index": line_index, "line": line, "track_id": identity, "box": row_box(fields),
            })

        raw_actions: List[dict] = []
        counts = Counter()
        for event in by_sequence[sequence]:
            frame = int(event["frame"])
            target_gt_id = int(event["target_gt_id"])
            candidate_row_index = int(event["candidate_row_index"])
            candidate = candidates[candidate_row_index]
            candidate_box = tuple(float(candidate[name]) for name in (
                "candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2"
            ))
            scored = [
                (iou_box(candidate_box, row["box"]), -int(row["track_id"]), -int(row["line_index"]), row)
                for row in rows_by_frame[frame]
            ]
            candidate_host_iou, _, _, source = max(scored, key=lambda item: (item[0], item[1], item[2]))
            if abs(candidate_host_iou - float(event["maximum_context_iou"])) > 1e-9:
                raise RuntimeError(f"{sequence}/frame {frame}: host IoU mismatch")
            inherited_track_id = int(event["inherited_track_id"])
            if inherited_track_id in dense_ids[frame]:
                raise RuntimeError(f"{sequence}/frame {frame}: inherited ID already present")
            valid_gt = [item for item in gt.get(frame, []) if item.marked != 0 and item.cls == 1]
            mechanism, source_target_iou, source_best_other_iou, source_best_other_gt_id = classify_host(
                source["box"], target_gt_id, valid_gt
            )
            candidate_target_iou = float(event["target_iou"])
            if mechanism == "other_target":
                action_type = "dual_retention_add"
                box_choice = "candidate"
            elif mechanism == "same_target":
                action_type = "migrate_replace"
                box_choice = "candidate" if candidate_target_iou > source_target_iou else "source"
            else:
                action_type = "replace_unmatched_host"
                box_choice = "candidate"
            action = {
                "sequence": sequence,
                "frame": frame,
                "target_gt_id": target_gt_id,
                "inherited_track_id": inherited_track_id,
                "source_track_id": int(source["track_id"]),
                "source_line_index": int(source["line_index"]),
                "source_line_sha256": hashlib.sha256(str(source["line"]).encode("utf-8")).hexdigest(),
                "mechanism": mechanism,
                "action_type": action_type,
                "box_choice": box_choice,
                "candidate_row_index": candidate_row_index,
                "global_pre_idx": int(keys[candidate_row_index]["global_pre_idx"]),
                "candidate_track_id": int(candidate["candidate_track_id"]),
                "candidate_score": float(candidate["candidate_score"]),
                "candidate_target_iou": candidate_target_iou,
                "source_target_iou": float(source_target_iou),
                "source_best_other_iou": float(source_best_other_iou),
                "source_best_other_gt_id": int(source_best_other_gt_id),
                "candidate_host_iou": float(candidate_host_iou),
                "gap_frames": int(event["gap_frames"]),
                "minimum_segment_purity": float(event["minimum_segment_purity"]),
                "matched_segment_support": int(event["matched_segment_support"]),
                "original_source_line": str(source["line"]),
            }
            raw_actions.append(action)
            counts["raw_conflict_events"] += 1
            counts[f"raw_{mechanism}"] += 1

        grouped: MutableMapping[int, List[dict]] = defaultdict(list)
        for action in raw_actions:
            grouped[int(action["source_line_index"])].append(action)
        selected = [max(actions, key=collision_priority) for actions in grouped.values()]
        selected.sort(key=lambda row: (int(row["frame"]), int(row["source_line_index"])))
        if len({int(row["source_line_index"]) for row in selected}) != len(selected):
            raise RuntimeError(f"{sequence}: duplicate selected source row")
        if len({(int(row["frame"]), int(row["inherited_track_id"])) for row in selected}) != len(selected):
            raise RuntimeError(f"{sequence}: duplicate selected target frame/ID")
        counts["unique_source_rows"] = len(grouped)
        counts["source_row_collisions"] = len(raw_actions) - len(grouped)
        counts["selected_actions"] = len(selected)
        for action in selected:
            counts[f"selected_{action['mechanism']}"] += 1
            counts[f"selected_box_{action['box_choice']}"] += 1
        selected_by_sequence[sequence] = selected
        summary_rows.append({"sequence": sequence, **counts})
        all_counts.update(counts)

    summary_rows.append({"sequence": "COMBINED", **all_counts})
    combined = [action for sequence in SEQUENCES for action in selected_by_sequence[sequence]]
    return {
        "candidates": candidates,
        "selected_by_sequence": selected_by_sequence,
        "combined_actions": combined,
        "summary_rows": summary_rows,
        "source_hashes": source_hashes,
        "plan_hashes": {
            **{sequence: hash_action_plan(selected_by_sequence[sequence]) for sequence in SEQUENCES},
            "COMBINED": hash_action_plan(combined),
        },
    }


def run_trackeval(repo: Path, gt_root: Path, eval_work: Path) -> None:
    seqmap = eval_work / "seqmaps/MOT20_train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(SEQUENCES) + "\n", encoding="utf-8")
    command = [
        sys.executable, str(repo / "TrackEval/scripts/run_mot_challenge.py"),
        "--GT_FOLDER", str(gt_root), "--TRACKERS_FOLDER", str(eval_work / "trackers"),
        "--OUTPUT_FOLDER", str(eval_work / "eval"), "--TRACKERS_TO_EVAL", *VARIANTS,
        "--BENCHMARK", "MOT20", "--SPLIT_TO_EVAL", "train", "--SEQMAP_FILE", str(seqmap),
        "--SKIP_SPLIT_FOL", "True", "--DO_PREPROC", "True", "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "", "--PRINT_ONLY_COMBINED", "True", "--PLOT_CURVES", "False",
        "--OUTPUT_DETAILED", "True", "--USE_PARALLEL", "False", "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (eval_work / "trackeval.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"TrackEval failed ({process.returncode})\n{process.stdout[-8000:]}")


def parse_summary(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header, values = lines[0].split(), lines[1].split()
    result = {}
    for key, value in zip(header, values):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def parse_detailed(path: Path, variant: str) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "variant": variant, "sequence": row["seq"],
                "HOTA": float(row["HOTA___AUC"]) * 100.0,
                "DetA": float(row["DetA___AUC"]) * 100.0,
                "AssA": float(row["AssA___AUC"]) * 100.0,
                "IDF1": float(row["IDF1"]) * 100.0,
                "MOTA": float(row["MOTA"]) * 100.0,
                "IDSW": int(float(row["IDSW"])), "CLR_FP": int(float(row["CLR_FP"])),
                "CLR_FN": int(float(row["CLR_FN"])), "Dets": int(float(row["Dets"])),
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--appearance-dir", default="outputs/mot20_m23_20260717/appearance_manifest_v1")
    parser.add_argument("--compact-dir", default="outputs/mot20_m23_20260717/compact_appearance_features_v1")
    parser.add_argument("--gap-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1")
    parser.add_argument("--replacement-dir", default="outputs/mot20_m23_20260717/deployable_replacement_oracle_v1")
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument("--preregister", default="outputs/mot20_m23_20260717/mechanism_decomposed_nms_conflict_preregister_v1.json")
    parser.add_argument("--prepare-preregister")
    parser.add_argument("--validate-trackers-only", action="store_true")
    parser.add_argument("--out-dir", default="outputs/mot20_m23_20260717/mechanism_decomposed_nms_conflict_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    appearance_dir, compact_dir = resolve(repo, args.appearance_dir), resolve(repo, args.compact_dir)
    gap_dir, replacement_dir = resolve(repo, args.gap_dir), resolve(repo, args.replacement_dir)
    gt_root, prereg_path = resolve(repo, args.gt_root), resolve(repo, args.preregister)
    output_dir = resolve(repo, args.out_dir)
    plans = build_plans(repo, appearance_dir, compact_dir, gap_dir, replacement_dir, gt_root)
    script_sha = sha256_file(Path(__file__).resolve())

    if args.prepare_preregister:
        references = {}
        for row in read_csv_rows(gap_dir / "per_sequence_metrics.csv"):
            if row["variant"] in {
                "dense_replacement_context", "safe_existing_id_gap_bridge_oracle",
                "all_existing_id_gap_bridge_oracle",
            }:
                references[f"{row['variant']}::{row['sequence']}"] = row
        prereg = {
            "schema": "fmtrack.m23_5.mechanism_decomposed_nms_conflict.preregister.v1",
            "script_sha256": script_sha,
            "sources": plans["source_hashes"],
            "protocol": {
                "sequences": list(SEQUENCES), "variants": list(VARIANTS),
                "conflict_iou": CONFLICT_IOU, "valid_gt_iou": VALID_GT_IOU,
                "host_selection": "maximum candidate/host IoU, lower host ID, lower source line index",
                "collision_resolution": "higher candidate target IoU, host IoU, shorter gap, inherited purity/support, lower inherited ID/candidate index",
                "other_target_action": "retain host and add candidate under inherited existing ID",
                "same_target_action": "migrate host to inherited ID and use higher-target-IoU source/candidate box; ties keep source",
                "unmatched_action": "replace unmatched host with candidate under inherited existing ID",
                "new_identity_namespace": False, "models_trained": 0,
                "threshold_sweeps": 0, "parameter_sweeps": 0,
                "locked_label_reads": 0, "locked_trackeval_calls": 0,
            },
            "expected": {
                "summary_rows": plans["summary_rows"], "plan_hashes": plans["plan_hashes"],
                "selected_actions": len(plans["combined_actions"]),
            },
            "references": references,
            "gates": {
                "mechanism_combined_hota_min": MECHANISM_GATE_HOTA,
                "mechanism_all_sequence_hota_nonnegative_vs_dense": True,
                "mechanism_idsw_nonincreasing_vs_dense": True,
                "safe_plus_mechanism_hota_min": COMBINED_GATE_HOTA,
                "safe_plus_mechanism_all_sequence_hota_nonnegative_vs_safe": True,
                "safe_plus_mechanism_idsw_nonincreasing_vs_safe": True,
            },
        }
        path = resolve(repo, args.prepare_preregister)
        canonical_json_dump(prereg, path)
        print(json.dumps({
            "script_sha256": script_sha, "preregister": str(path),
            "selected_actions": len(plans["combined_actions"]),
            "summary": plans["summary_rows"][-1], "plan_hashes": plans["plan_hashes"],
        }, indent=2, sort_keys=True))
        return

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg["script_sha256"] != script_sha:
        raise RuntimeError("script hash mismatch")
    if prereg["sources"] != plans["source_hashes"]:
        raise RuntimeError("source hashes changed")
    if prereg["expected"]["summary_rows"] != plans["summary_rows"]:
        raise RuntimeError("structural counts changed")
    if prereg["expected"]["plan_hashes"] != plans["plan_hashes"]:
        raise RuntimeError("mechanism plan changed")
    if prereg["protocol"]["variants"] != list(VARIANTS):
        raise RuntimeError("variant protocol changed")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    eval_work = output_dir / "eval_work"
    tracker_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in VARIANTS}
    action_rows: List[dict] = []

    for sequence in SEQUENCES:
        dense_path = replacement_dir / "eval_work/trackers/dense_replacement_oracle/data" / f"{sequence}.txt"
        safe_path = gap_dir / "eval_work/trackers/safe_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        all_path = gap_dir / "eval_work/trackers/all_existing_id_gap_bridge_oracle/data" / f"{sequence}.txt"
        dense_lines = dense_path.read_text(encoding="utf-8").splitlines()
        safe_lines = safe_path.read_text(encoding="utf-8").splitlines()
        all_lines = all_path.read_text(encoding="utf-8").splitlines()
        dual_lines = list(dense_lines)
        mechanism_lines = list(dense_lines)
        replacement_by_original: Dict[str, str] = {}
        additions: List[str] = []
        actions = plans["selected_by_sequence"][sequence]
        for action in actions:
            source_index = int(action["source_line_index"])
            source_line = dense_lines[source_index]
            if hashlib.sha256(source_line.encode("utf-8")).hexdigest() != action["source_line_sha256"]:
                raise RuntimeError(f"{sequence}: source row hash mismatch")
            candidate = plans["candidates"][int(action["candidate_row_index"])]
            if action["mechanism"] == "other_target":
                line = candidate_line(action, candidate)
                additions.append(line)
                dual_lines.append(line)
                mechanism_lines.append(line)
            elif action["mechanism"] == "same_target":
                line = (
                    candidate_line(action, candidate)
                    if action["box_choice"] == "candidate"
                    else replace_identity(source_line, int(action["inherited_track_id"]))
                )
                mechanism_lines[source_index] = line
                replacement_by_original[source_line] = line
            else:
                line = candidate_line(action, candidate)
                mechanism_lines[source_index] = line
                replacement_by_original[source_line] = line
            action_rows.append({key: value for key, value in action.items() if key != "original_source_line"})

        combined_lines: List[str] = []
        replaced = 0
        for line in safe_lines:
            if line in replacement_by_original:
                combined_lines.append(replacement_by_original[line])
                replaced += 1
            else:
                combined_lines.append(line)
        combined_lines.extend(additions)
        expected_replaced = sum(action["mechanism"] != "other_target" for action in actions)
        expected_additions = sum(action["mechanism"] == "other_target" for action in actions)
        if replaced != expected_replaced:
            raise RuntimeError(f"{sequence}: combined replaced {replaced}/{expected_replaced}")
        expected_lengths = {
            "dense_replacement_context": len(dense_lines),
            "safe_gap_context": len(safe_lines),
            "all_additions_reference": len(all_lines),
            "distinct_person_dual_retention_oracle": len(dense_lines) + expected_additions,
            "mechanism_decomposed_oracle": len(dense_lines) + expected_additions,
            "safe_gap_plus_mechanism_oracle": len(safe_lines) + expected_additions,
        }

        variants = {
            "dense_replacement_context": dense_lines,
            "safe_gap_context": safe_lines,
            "all_additions_reference": all_lines,
            "distinct_person_dual_retention_oracle": dual_lines,
            "mechanism_decomposed_oracle": mechanism_lines,
            "safe_gap_plus_mechanism_oracle": combined_lines,
        }
        reference_variants = {
            "dense_replacement_context",
            "safe_gap_context",
            "all_additions_reference",
        }
        for variant, lines in variants.items():
            if len(lines) != expected_lengths[variant]:
                raise RuntimeError(
                    f"{sequence}/{variant}: row count {len(lines)} != {expected_lengths[variant]}"
                )
            output_lines = list(lines) if variant in reference_variants else sorted(lines, key=tracker_sort_key)
            validate_tracker(output_lines, f"{sequence}/{variant}")
            out_path = eval_work / "trackers" / variant / "data" / f"{sequence}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
            tracker_hashes[variant][sequence] = sha256_file(out_path)
        for variant, expected in (
            ("dense_replacement_context", plans["source_hashes"]["dense_trackers"][sequence]),
            ("safe_gap_context", plans["source_hashes"]["safe_trackers"][sequence]),
            ("all_additions_reference", plans["source_hashes"]["all_addition_trackers"][sequence]),
        ):
            if tracker_hashes[variant][sequence] != expected:
                raise RuntimeError(f"{sequence}/{variant}: source tracker reproduction failed")

    if args.validate_trackers_only:
        print(json.dumps({
            "validation_only": True,
            "selected_actions": len(plans["combined_actions"]),
            "summary": plans["summary_rows"][-1],
            "plan_hashes": plans["plan_hashes"],
            "tracker_hashes": tracker_hashes,
        }, indent=2, sort_keys=True))
        return

    run_trackeval(repo, gt_root, eval_work)
    metric_rows: List[dict] = []
    summary_rows: List[dict] = []
    for variant in VARIANTS:
        metric_rows.extend(parse_detailed(eval_work / "eval" / variant / "pedestrian_detailed.csv", variant))
        summary = parse_summary(eval_work / "eval" / variant / "pedestrian_summary.txt")
        summary_rows.append({"variant": variant, **{field: summary[field] for field in METRIC_FIELDS}})
    metric_map = {(row["variant"], row["sequence"]): row for row in metric_rows}

    reference_map = {
        "dense_replacement_context": "dense_replacement_context",
        "safe_gap_context": "safe_existing_id_gap_bridge_oracle",
        "all_additions_reference": "all_existing_id_gap_bridge_oracle",
    }
    for variant, reference_variant in reference_map.items():
        for sequence in (*SEQUENCES, "COMBINED"):
            reference = prereg["references"][f"{reference_variant}::{sequence}"]
            actual = metric_map[(variant, sequence)]
            for field in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
                if float(actual[field]) != float(reference[field]):
                    raise RuntimeError(f"{variant}/{sequence}/{field}: reference mismatch")
            for field in ("IDSW", "CLR_FP", "CLR_FN", "Dets"):
                if int(actual[field]) != int(reference[field]):
                    raise RuntimeError(f"{variant}/{sequence}/{field}: reference mismatch")

    dense = metric_map[("dense_replacement_context", "COMBINED")]
    safe = metric_map[("safe_gap_context", "COMBINED")]
    mechanism = metric_map[("mechanism_decomposed_oracle", "COMBINED")]
    combined = metric_map[("safe_gap_plus_mechanism_oracle", "COMBINED")]
    mechanism_deltas = {
        sequence: metric_map[("mechanism_decomposed_oracle", sequence)]["HOTA"] - metric_map[("dense_replacement_context", sequence)]["HOTA"]
        for sequence in SEQUENCES
    }
    combined_deltas = {
        sequence: metric_map[("safe_gap_plus_mechanism_oracle", sequence)]["HOTA"] - metric_map[("safe_gap_context", sequence)]["HOTA"]
        for sequence in SEQUENCES
    }
    mechanism_pass = (
        mechanism["HOTA"] >= MECHANISM_GATE_HOTA
        and all(delta >= 0.0 for delta in mechanism_deltas.values())
        and mechanism["IDSW"] <= dense["IDSW"]
    )
    combined_pass = (
        combined["HOTA"] >= COMBINED_GATE_HOTA
        and all(delta >= 0.0 for delta in combined_deltas.values())
        and combined["IDSW"] <= safe["IDSW"]
    )
    ceiling_passed = mechanism_pass and combined_pass
    decision = {
        "dense_context_reproduced": True, "safe_context_reproduced": True,
        "all_additions_reference_reproduced": True,
        "mechanism_combined_hota": mechanism["HOTA"],
        "mechanism_hota_gain_over_dense": mechanism["HOTA"] - dense["HOTA"],
        "mechanism_idsw_change_vs_dense": mechanism["IDSW"] - dense["IDSW"],
        "mechanism_sequence_hota_deltas": mechanism_deltas,
        "mechanism_gate_passed": mechanism_pass,
        "safe_plus_mechanism_hota": combined["HOTA"],
        "safe_plus_mechanism_hota_gain_over_safe": combined["HOTA"] - safe["HOTA"],
        "safe_plus_mechanism_idsw_change_vs_safe": combined["IDSW"] - safe["IDSW"],
        "safe_plus_mechanism_sequence_hota_deltas": combined_deltas,
        "safe_plus_mechanism_gate_passed": combined_pass,
        "mechanism_action_space_ceiling_passed": ceiling_passed,
        "deployment_allowed": False, "locked_manifest_created": False,
        "next_stage": (
            "nested cross-sequence mechanism-aware dual-retention/migration scorer with abstention"
            if ceiling_passed else
            "close local NMS-conflict actions and audit longer-horizon tracklet graph evidence"
        ),
    }

    write_csv(output_dir / "conflict_summary.csv", plans["summary_rows"], [
        "sequence", "raw_conflict_events", "unique_source_rows", "source_row_collisions", "selected_actions",
        "raw_other_target", "raw_same_target", "raw_unmatched",
        "selected_other_target", "selected_same_target", "selected_unmatched",
        "selected_box_candidate", "selected_box_source",
    ])
    write_csv(output_dir / "mechanism_actions.csv", action_rows, [
        "sequence", "frame", "target_gt_id", "inherited_track_id", "source_track_id", "source_line_index",
        "source_line_sha256", "mechanism", "action_type", "box_choice", "candidate_row_index", "global_pre_idx",
        "candidate_track_id", "candidate_score", "candidate_target_iou", "source_target_iou",
        "source_best_other_iou", "source_best_other_gt_id", "candidate_host_iou", "gap_frames",
        "minimum_segment_purity", "matched_segment_support",
    ])
    write_csv(output_dir / "per_sequence_metrics.csv", metric_rows, ["variant", "sequence", *METRIC_FIELDS])
    write_csv(output_dir / "variant_metrics.csv", summary_rows, ["variant", *METRIC_FIELDS])
    report = {
        "schema": "fmtrack.m23_5.mechanism_decomposed_nms_conflict.report.v1",
        "protocol": prereg["protocol"], "gates": prereg["gates"],
        "counts": plans["summary_rows"][-1], "plan_hashes": plans["plan_hashes"],
        "source_hashes": plans["source_hashes"], "decision": decision,
        "locked_policy": {
            "p15_no_op": True, "remaining_locked_rows_unread": 156,
            "locked_label_reads": 0, "locked_trackeval_calls": 0,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    compact_files = (
        "conflict_summary.csv", "mechanism_actions.csv", "per_sequence_metrics.csv",
        "variant_metrics.csv", "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_5.mechanism_decomposed_nms_conflict.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
        "tracker_hashes": tracker_hashes, "plan_hashes": plans["plan_hashes"],
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
