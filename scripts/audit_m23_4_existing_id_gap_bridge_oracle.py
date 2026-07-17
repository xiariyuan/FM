"""M23-4-0 existing-ID internal-gap bridge oracle audit on MOT20 train.

The audit starts from the exact M23-3-1 dense-replacement tracker. It preserves
all context rows and identities. The only new action is to add one frozen-budget
pre-NMS observation to an internal gap between two consecutive contiguous
segments of the same baseline tracker ID when both segments have the same
positive modal train-GT support identity and the gap is at most 30 frames.

For each visible support-GT frame inside such a gap, competing bridge targets
are deduplicated by a fixed priority: shorter gap, higher minimum segment
purity, more matched segment support, then lower inherited tracker ID. Frozen
pre-NMS candidates and bridge targets are matched globally within each frame
using the fixed MOT20 IoU>=0.5 Hungarian rule.

Four fixed tracker variants are evaluated:
  * baseline_raw: exact frozen raw baseline reproduction;
  * dense_replacement_context: exact M23-3-1 dense replacement reproduction;
  * all_existing_id_gap_bridge_oracle: add every GT-matched bridge candidate;
  * safe_existing_id_gap_bridge_oracle: add only candidates whose maximum IoU
    with any dense-context row in the same frame is strictly below 0.5.

No new identity namespace, identity relabeling, suppressor routing, threshold
sweep, gap sweep, or model training is used. This is a train-GT oracle ceiling,
not a deployable result.
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
SEQUENCE_TO_ID = {name: index for index, name in enumerate(SEQUENCES)}
MAX_GAP_FRAMES = 30
CONTEXT_EXCLUSIVE_IOU = 0.50
VARIANTS = (
    "baseline_raw",
    "dense_replacement_context",
    "all_existing_id_gap_bridge_oracle",
    "safe_existing_id_gap_bridge_oracle",
)


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
        raise ValueError(f"invalid MOT tracker line: {line!r}")
    return int(fields[0]), int(fields[1]), fields


def validate_tracker(lines: Sequence[str], label: str) -> None:
    previous_frame = -1
    seen = set()
    for line in lines:
        frame, identity, _ = parse_tracker_line(line)
        if frame < previous_frame:
            raise RuntimeError(f"{label}: frames are not monotonic")
        previous_frame = frame
        key = (frame, identity)
        if key in seen:
            raise RuntimeError(f"{label}: duplicate tracker ID {identity} in frame {frame}")
        seen.add(key)


def tracker_boxes(lines: Sequence[str]) -> Tuple[Dict[int, List[Tuple[float, float, float, float]]], Dict[int, set[int]]]:
    boxes: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    identities: Dict[int, set[int]] = defaultdict(set)
    for line in lines:
        frame, identity, fields = parse_tracker_line(line)
        x = float(fields[2])
        y = float(fields[3])
        width = float(fields[4])
        height = float(fields[5])
        boxes[frame].append((x, y, x + width, y + height))
        identities[frame].add(identity)
    return boxes, identities


def iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def target_priority(record: Mapping[str, object]) -> Tuple[int, float, int, int]:
    return (
        -int(record["gap_frames"]),
        float(record["minimum_segment_purity"]),
        int(record["matched_segment_support"]),
        -int(record["inherited_track_id"]),
    )


def addition_line(event: Mapping[str, object], candidate_row: np.void) -> str:
    x1 = float(candidate_row["candidate_x1"])
    y1 = float(candidate_row["candidate_y1"])
    x2 = float(candidate_row["candidate_x2"])
    y2 = float(candidate_row["candidate_y2"])
    return ",".join(
        [
            str(int(event["frame"])),
            str(int(event["inherited_track_id"])),
            f"{x1:.6f}",
            f"{y1:.6f}",
            f"{x2 - x1:.6f}",
            f"{y2 - y1:.6f}",
            f"{float(candidate_row['candidate_score']):.6f}",
            "-1",
            "-1",
            "-1",
        ]
    )


def tracker_sort_key(line: str) -> Tuple[int, int, float, float, str]:
    frame, identity, fields = parse_tracker_line(line)
    return frame, identity, float(fields[2]), float(fields[3]), line


def hash_event_plan(events: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for event in sorted(
        events,
        key=lambda item: (
            str(item["sequence"]),
            int(item["frame"]),
            int(item["inherited_track_id"]),
        ),
    ):
        value = {
            "sequence": str(event["sequence"]),
            "frame": int(event["frame"]),
            "target_gt_id": int(event["target_gt_id"]),
            "inherited_track_id": int(event["inherited_track_id"]),
            "candidate_row_index": int(event["candidate_row_index"]),
            "global_pre_idx": int(event["global_pre_idx"]),
            "target_iou": float(event["target_iou"]),
            "maximum_context_iou": float(event["maximum_context_iou"]),
            "gap_frames": int(event["gap_frames"]),
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_trackeval(repo: Path, gt_root: Path, eval_work: Path) -> None:
    seqmap = eval_work / "seqmaps" / "MOT20_train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(SEQUENCES) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(repo / "TrackEval" / "scripts" / "run_mot_challenge.py"),
        "--GT_FOLDER", str(gt_root),
        "--TRACKERS_FOLDER", str(eval_work / "trackers"),
        "--OUTPUT_FOLDER", str(eval_work / "eval"),
        "--TRACKERS_TO_EVAL", *VARIANTS,
        "--BENCHMARK", "MOT20",
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(seqmap),
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "",
        "--PRINT_ONLY_COMBINED", "True",
        "--PLOT_CURVES", "False",
        "--OUTPUT_DETAILED", "True",
        "--USE_PARALLEL", "False",
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (eval_work / "trackeval.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"TrackEval failed with code {process.returncode}\n{process.stdout[-8000:]}")


def parse_summary(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = lines[0].split()
    values = lines[1].split()
    if len(header) != len(values):
        raise ValueError(f"summary width mismatch: {path}")
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
            rows.append(
                {
                    "variant": variant,
                    "sequence": row["seq"],
                    "HOTA": float(row["HOTA___AUC"]) * 100.0,
                    "DetA": float(row["DetA___AUC"]) * 100.0,
                    "AssA": float(row["AssA___AUC"]) * 100.0,
                    "IDF1": float(row["IDF1"]) * 100.0,
                    "MOTA": float(row["MOTA"]) * 100.0,
                    "IDSW": int(float(row["IDSW"])),
                    "CLR_FP": int(float(row["CLR_FP"])),
                    "CLR_FN": int(float(row["CLR_FN"])),
                    "Dets": int(float(row["Dets"])),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--appearance-dir", required=True)
    parser.add_argument("--compact-dir", required=True)
    parser.add_argument("--segment-script", required=True)
    parser.add_argument("--replacement-dir", required=True)
    parser.add_argument(
        "--baseline-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data",
    )
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument("--preregister", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    appearance_dir = resolve(repo, args.appearance_dir)
    compact_dir = resolve(repo, args.compact_dir)
    segment_script_path = resolve(repo, args.segment_script)
    replacement_dir = resolve(repo, args.replacement_dir)
    baseline_dir = resolve(repo, args.baseline_dir)
    gt_root = resolve(repo, args.gt_root)
    prereg_path = resolve(repo, args.preregister)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    if prereg["script_sha256"] != sha256_file(script_path):
        raise RuntimeError("preregistered script hash mismatch")
    if prereg["sources"]["segment_script"]["sha256"] != sha256_file(segment_script_path):
        raise RuntimeError("segment script hash mismatch")
    if tuple(prereg["protocol"]["variants"]) != VARIANTS:
        raise RuntimeError("preregistered variants mismatch")
    if int(prereg["protocol"]["maximum_gap_frames"]) != MAX_GAP_FRAMES:
        raise RuntimeError("maximum gap mismatch")
    if float(prereg["protocol"]["context_exclusive_iou"]) != CONTEXT_EXCLUSIVE_IOU:
        raise RuntimeError("context-exclusive IoU mismatch")

    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    candidate_keys_path = compact_dir / "candidate_keys.npy"
    replacement_report_path = replacement_dir / "report.json"
    replacement_manifest_path = replacement_dir / "manifest.json"
    for label, path in (
        ("candidate_manifest", candidate_manifest_path),
        ("candidate_keys", candidate_keys_path),
        ("replacement_report", replacement_report_path),
        ("replacement_manifest", replacement_manifest_path),
    ):
        expected = prereg["sources"][label]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")

    m23 = load_module(repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py", "m23_for_m23_4")
    segment_module = load_module(segment_script_path, "m23_3_segment_for_m23_4")
    candidates = np.load(candidate_manifest_path, allow_pickle=False, mmap_mode="r")
    keys = np.load(candidate_keys_path, allow_pickle=False, mmap_mode="r")
    if len(candidates) != len(keys):
        raise RuntimeError("candidate manifest and key row counts differ")

    eval_work = output_dir / "eval_work"
    gap_pair_rows: List[dict] = []
    bridge_event_rows: List[dict] = []
    summary_rows: List[dict] = []
    tracker_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in VARIANTS}
    event_plan_hashes: Dict[str, Dict[str, str]] = {
        "all_existing_id_gap_bridge_oracle": {},
        "safe_existing_id_gap_bridge_oracle": {},
    }
    source_hashes = {"baseline": {}, "gt": {}, "dense_trackers": {}}
    all_counts = Counter()
    all_events: List[dict] = []
    safe_events: List[dict] = []
    gap_pair_counter = 0

    for sequence_id, sequence in enumerate(SEQUENCES):
        print(f"[M23-4-0] building internal-gap bridge {sequence}", flush=True)
        baseline_path = baseline_dir / f"{sequence}.txt"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        dense_path = replacement_dir / "eval_work" / "trackers" / "dense_replacement_oracle" / "data" / f"{sequence}.txt"
        if sha256_file(baseline_path) != prereg["sources"]["baseline"][sequence]:
            raise RuntimeError(f"{sequence}: baseline hash mismatch")
        if sha256_file(gt_path) != prereg["sources"]["gt"][sequence]:
            raise RuntimeError(f"{sequence}: GT hash mismatch")
        if sha256_file(dense_path) != prereg["sources"]["dense_trackers"][sequence]:
            raise RuntimeError(f"{sequence}: dense tracker hash mismatch")
        source_hashes["baseline"][sequence] = sha256_file(baseline_path)
        source_hashes["gt"][sequence] = sha256_file(gt_path)
        source_hashes["dense_trackers"][sequence] = sha256_file(dense_path)

        baseline_lines = baseline_path.read_text(encoding="utf-8").splitlines()
        dense_lines = dense_path.read_text(encoding="utf-8").splitlines()
        validate_tracker(baseline_lines, f"{sequence}/baseline")
        validate_tracker(dense_lines, f"{sequence}/dense")
        dense_boxes, dense_ids = tracker_boxes(dense_lines)
        gt = m23.load_gt(gt_path)
        _, _, segment_rows = segment_module.build_tracklet_segments(
            m23, sequence, baseline_path, gt_path
        )
        segments_by_track: MutableMapping[int, List[dict]] = defaultdict(list)
        for row in segment_rows:
            segments_by_track[int(row["baseline_track_id"])].append(row)

        counts = Counter()
        raw_targets: List[dict] = []
        sequence_gap_pairs: List[dict] = []
        for track_id in sorted(segments_by_track):
            segments = sorted(segments_by_track[track_id], key=lambda item: int(item["first_frame"]))
            for left, right in zip(segments, segments[1:]):
                counts["consecutive_segment_pairs"] += 1
                gap_frames = int(right["first_frame"]) - int(left["last_frame"]) - 1
                pair = {
                    "sequence": sequence,
                    "gap_pair_id": gap_pair_counter,
                    "baseline_track_id": track_id,
                    "left_segment_id": int(left["segment_id"]),
                    "right_segment_id": int(right["segment_id"]),
                    "left_last_frame": int(left["last_frame"]),
                    "right_first_frame": int(right["first_frame"]),
                    "gap_frames": gap_frames,
                    "left_modal_gt_id": int(left["modal_gt_id"]),
                    "right_modal_gt_id": int(right["modal_gt_id"]),
                    "minimum_segment_purity": min(float(left["modal_purity"]), float(right["modal_purity"])),
                    "matched_segment_support": int(left["matched_frames"]) + int(right["matched_frames"]),
                    "eligible_gap": False,
                    "visible_target_frames": 0,
                    "matched_bridge_events": 0,
                    "safe_bridge_events": 0,
                }
                gap_pair_counter += 1
                if gap_frames <= 0 or gap_frames > MAX_GAP_FRAMES:
                    gap_pair_rows.append(pair)
                    continue
                counts["gap_le30_pairs"] += 1
                left_gt = int(left["modal_gt_id"])
                right_gt = int(right["modal_gt_id"])
                if left_gt <= 0 or left_gt != right_gt:
                    gap_pair_rows.append(pair)
                    continue
                counts["same_support_gap_pairs"] += 1
                pair["eligible_gap"] = True
                gap_pair_rows.append(pair)
                sequence_gap_pairs.append(pair)
                for frame in range(int(left["last_frame"]) + 1, int(right["first_frame"])):
                    counts["gap_frames"] += 1
                    targets = [
                        item
                        for item in gt.get(frame, [])
                        if item.marked != 0 and item.cls == 1 and int(item.gt_id) == left_gt
                    ]
                    if not targets:
                        counts["target_gt_absent"] += 1
                        continue
                    counts["target_gt_visible"] += 1
                    pair["visible_target_frames"] += 1
                    raw_targets.append(
                        {
                            "sequence": sequence,
                            "frame": frame,
                            "target_gt_id": left_gt,
                            "inherited_track_id": track_id,
                            "gap_pair_id": int(pair["gap_pair_id"]),
                            "gap_frames": gap_frames,
                            "minimum_segment_purity": float(pair["minimum_segment_purity"]),
                            "matched_segment_support": int(pair["matched_segment_support"]),
                            "target_box": targets[0].box,
                        }
                    )

        target_by_frame: MutableMapping[int, Dict[int, Tuple[Tuple[int, float, int, int], dict]]] = defaultdict(dict)
        for target in raw_targets:
            frame = int(target["frame"])
            gt_id = int(target["target_gt_id"])
            rank = target_priority(target)
            previous = target_by_frame[frame].get(gt_id)
            if previous is None or rank > previous[0]:
                target_by_frame[frame][gt_id] = (rank, target)
        targets_by_frame = {
            frame: [item[1] for _, item in sorted(per_gt.items())]
            for frame, per_gt in target_by_frame.items()
        }
        counts["unique_target_frames"] = sum(len(items) for items in targets_by_frame.values())
        counts["deduped_competing_targets"] = len(raw_targets) - counts["unique_target_frames"]

        candidate_indices_by_frame: MutableMapping[int, List[int]] = defaultdict(list)
        for row_index in np.flatnonzero(candidates["sequence_id"] == sequence_id).tolist():
            candidate_indices_by_frame[int(candidates[row_index]["frame"])].append(row_index)

        pair_by_id = {int(row["gap_pair_id"]): row for row in sequence_gap_pairs}
        sequence_events: List[dict] = []
        sequence_safe_events: List[dict] = []
        for frame in sorted(targets_by_frame):
            targets = targets_by_frame[frame]
            candidate_indices = candidate_indices_by_frame.get(frame, [])
            if not candidate_indices:
                counts["target_no_candidates"] += len(targets)
                continue
            candidate_objects = []
            for row_index in candidate_indices:
                row = candidates[row_index]
                box = (
                    float(row["candidate_x1"]),
                    float(row["candidate_y1"]),
                    float(row["candidate_x2"]),
                    float(row["candidate_y2"]),
                )
                candidate_objects.append(
                    m23.Candidate(
                        frame=frame,
                        source="prenms",
                        uid=row_index,
                        original_id=int(row["candidate_track_id"]),
                        box=box,
                        score=float(row["candidate_score"]),
                    )
                )
            target_objects = [
                m23.GTObject(
                    gt_id=int(target["target_gt_id"]),
                    box=target["target_box"],
                    marked=1,
                    cls=1,
                )
                for target in targets
            ]
            matches = m23.match_candidates(candidate_objects, target_objects)
            matched_target_indices = set()
            for candidate_local_index, target_local_index, target_iou in matches:
                matched_target_indices.add(target_local_index)
                target = targets[target_local_index]
                row_index = candidate_indices[candidate_local_index]
                row = candidates[row_index]
                inherited_track_id = int(target["inherited_track_id"])
                if inherited_track_id in dense_ids.get(frame, set()):
                    raise RuntimeError(
                        f"{sequence}: inherited ID {inherited_track_id} unexpectedly present in frame {frame}"
                    )
                candidate_box = candidate_objects[candidate_local_index].box
                maximum_context_iou = max(
                    [0.0] + [iou_box(candidate_box, box) for box in dense_boxes.get(frame, [])]
                )
                safe = maximum_context_iou < CONTEXT_EXCLUSIVE_IOU
                event = {
                    "sequence": sequence,
                    "frame": frame,
                    "target_gt_id": int(target["target_gt_id"]),
                    "inherited_track_id": inherited_track_id,
                    "gap_pair_id": int(target["gap_pair_id"]),
                    "gap_frames": int(target["gap_frames"]),
                    "minimum_segment_purity": float(target["minimum_segment_purity"]),
                    "matched_segment_support": int(target["matched_segment_support"]),
                    "candidate_row_index": row_index,
                    "global_pre_idx": int(keys[row_index]["global_pre_idx"]),
                    "candidate_track_id": int(row["candidate_track_id"]),
                    "candidate_score": float(row["candidate_score"]),
                    "target_iou": float(target_iou),
                    "maximum_context_iou": maximum_context_iou,
                    "context_exclusive": safe,
                }
                sequence_events.append(event)
                pair_by_id[int(target["gap_pair_id"])]["matched_bridge_events"] += 1
                counts["matched_additions"] += 1
                candidate_track_id = int(row["candidate_track_id"])
                if candidate_track_id <= 0:
                    counts["candidate_track_zero"] += 1
                elif candidate_track_id == inherited_track_id:
                    counts["candidate_track_same"] += 1
                else:
                    counts["candidate_track_other"] += 1
                if maximum_context_iou >= 0.999:
                    counts["context_iou_ge_0_999"] += 1
                if maximum_context_iou >= 0.90:
                    counts["context_iou_ge_0_90"] += 1
                if maximum_context_iou >= CONTEXT_EXCLUSIVE_IOU:
                    counts["context_iou_ge_0_50"] += 1
                else:
                    counts["context_iou_lt_0_50"] += 1
                    pair_by_id[int(target["gap_pair_id"])]["safe_bridge_events"] += 1
                    sequence_safe_events.append(event)
            counts["unmatched_targets"] += len(targets) - len(matched_target_indices)

        for pair in sequence_gap_pairs:
            # The same dictionaries are already present in gap_pair_rows.
            pass
        dense_lines_set = set(dense_lines)
        if len(dense_lines_set) != len(dense_lines):
            raise RuntimeError(f"{sequence}: duplicate dense tracker lines")
        if len({(int(event["frame"]), int(event["inherited_track_id"])) for event in sequence_events}) != len(sequence_events):
            raise RuntimeError(f"{sequence}: all-addition plan has duplicate frame/inherited-ID actions")
        if len({(int(event["frame"]), int(event["inherited_track_id"])) for event in sequence_safe_events}) != len(sequence_safe_events):
            raise RuntimeError(f"{sequence}: safe-addition plan has duplicate frame/inherited-ID actions")
        variants = {
            "baseline_raw": list(baseline_lines),
            "dense_replacement_context": list(dense_lines),
        }
        for variant, events in (
            ("all_existing_id_gap_bridge_oracle", sequence_events),
            ("safe_existing_id_gap_bridge_oracle", sequence_safe_events),
        ):
            addition_lines = [addition_line(event, candidates[int(event["candidate_row_index"])]) for event in events]
            if len(addition_lines) != len(set(addition_lines)):
                raise RuntimeError(f"{sequence}/{variant}: duplicate addition lines")
            lines = sorted([*dense_lines, *addition_lines], key=tracker_sort_key)
            validate_tracker(lines, f"{sequence}/{variant}")
            if len(lines) != len(dense_lines) + len(events):
                raise RuntimeError(f"{sequence}/{variant}: row-count mismatch")
            variants[variant] = lines
            event_plan_hashes[variant][sequence] = hash_event_plan(events)

        for variant, lines in variants.items():
            tracker_path = eval_work / "trackers" / variant / "data" / f"{sequence}.txt"
            tracker_path.parent.mkdir(parents=True, exist_ok=True)
            tracker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tracker_hashes[variant][sequence] = sha256_file(tracker_path)
        if tracker_hashes["baseline_raw"][sequence] != sha256_file(baseline_path):
            raise RuntimeError(f"{sequence}: baseline tracker was not reproduced")
        if tracker_hashes["dense_replacement_context"][sequence] != sha256_file(dense_path):
            raise RuntimeError(f"{sequence}: dense replacement tracker was not reproduced")

        counts["all_plan_events"] = len(sequence_events)
        counts["safe_plan_events"] = len(sequence_safe_events)
        summary_rows.append({"sequence": sequence, **counts})
        all_counts.update(counts)
        all_events.extend(sequence_events)
        safe_events.extend(sequence_safe_events)
        bridge_event_rows.extend(sequence_events)

    # Eligible gap-pair rows were mutated through pair_by_id references; all rows are now complete.
    summary_rows.append({"sequence": "COMBINED", **all_counts})
    event_plan_hashes["all_existing_id_gap_bridge_oracle"]["COMBINED"] = hash_event_plan(all_events)
    event_plan_hashes["safe_existing_id_gap_bridge_oracle"]["COMBINED"] = hash_event_plan(safe_events)

    for field in (
        "consecutive_segment_pairs",
        "gap_le30_pairs",
        "same_support_gap_pairs",
        "gap_frames",
        "target_gt_visible",
        "unique_target_frames",
        "matched_additions",
        "context_iou_ge_0_50",
        "context_iou_lt_0_50",
        "all_plan_events",
        "safe_plan_events",
    ):
        actual = int(all_counts[field])
        expected = int(prereg["structural_preflight"][field])
        if actual != expected:
            raise RuntimeError(f"structural preflight mismatch for {field}: {actual} != {expected}")

    run_trackeval(repo, gt_root, eval_work)
    variant_rows: List[dict] = []
    sequence_metric_rows: List[dict] = []
    for variant in VARIANTS:
        summary = parse_summary(eval_work / "eval" / variant / "pedestrian_summary.txt")
        variant_rows.append(
            {
                "variant": variant,
                "HOTA": summary["HOTA"],
                "DetA": summary["DetA"],
                "AssA": summary["AssA"],
                "IDF1": summary["IDF1"],
                "MOTA": summary["MOTA"],
                "IDSW": int(summary["IDSW"]),
                "CLR_FP": int(summary["CLR_FP"]),
                "CLR_FN": int(summary["CLR_FN"]),
                "Dets": int(summary["Dets"]),
            }
        )
        sequence_metric_rows.extend(
            parse_detailed(eval_work / "eval" / variant / "pedestrian_detailed.csv", variant)
        )

    by_variant = {row["variant"]: row for row in variant_rows}
    by_sequence_variant = {(row["sequence"], row["variant"]): row for row in sequence_metric_rows}
    exact_combined = {variant: by_sequence_variant[("COMBINED", variant)] for variant in VARIANTS}
    tolerance = float(prereg["acceptance"]["context_reproduction_tolerance"])
    baseline_deltas = {
        metric: float(
            exact_combined["baseline_raw"][metric]
            - prereg["references"]["baseline_combined"][metric]
        )
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    }
    dense_deltas = {
        metric: float(
            exact_combined["dense_replacement_context"][metric]
            - prereg["references"]["dense_combined"][metric]
        )
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    }
    baseline_reproduced = all(abs(value) <= tolerance for value in baseline_deltas.values())
    dense_reproduced = all(abs(value) <= tolerance for value in dense_deltas.values())
    safe_sequence_deltas = {
        sequence: float(
            by_sequence_variant[(sequence, "safe_existing_id_gap_bridge_oracle")]["HOTA"]
            - by_sequence_variant[(sequence, "dense_replacement_context")]["HOTA"]
        )
        for sequence in SEQUENCES
    }
    all_sequence_deltas = {
        sequence: float(
            by_sequence_variant[(sequence, "all_existing_id_gap_bridge_oracle")]["HOTA"]
            - by_sequence_variant[(sequence, "dense_replacement_context")]["HOTA"]
        )
        for sequence in SEQUENCES
    }
    acceptance = prereg["acceptance"]
    safe_hota = float(exact_combined["safe_existing_id_gap_bridge_oracle"]["HOTA"])
    decision = {
        "baseline_reproduced": baseline_reproduced,
        "dense_context_reproduced": dense_reproduced,
        "baseline_metric_deltas": baseline_deltas,
        "dense_metric_deltas": dense_deltas,
        "all_additions_combined_hota": float(exact_combined["all_existing_id_gap_bridge_oracle"]["HOTA"]),
        "all_additions_sequence_hota_deltas": all_sequence_deltas,
        "safe_additions_combined_hota": safe_hota,
        "safe_additions_combined_hota_passed": bool(
            safe_hota >= acceptance["minimum_safe_combined_hota"]
        ),
        "safe_additions_sequence_hota_deltas": safe_sequence_deltas,
        "safe_additions_all_sequence_hota_nonnegative": bool(min(safe_sequence_deltas.values()) >= 0.0),
        "safe_additions_minimum_sequence_gain": min(safe_sequence_deltas.values()),
        "safe_additions_minimum_sequence_gain_passed": bool(
            min(safe_sequence_deltas.values()) >= acceptance["minimum_safe_sequence_hota_gain"]
        ),
        "internal_gap_bridge_ceiling_passed": False,
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": "set below after gate evaluation",
    }
    decision["internal_gap_bridge_ceiling_passed"] = bool(
        baseline_reproduced
        and dense_reproduced
        and decision["safe_additions_combined_hota_passed"]
        and decision["safe_additions_all_sequence_hota_nonnegative"]
        and decision["safe_additions_minimum_sequence_gain_passed"]
    )
    if decision["internal_gap_bridge_ceiling_passed"]:
        decision["next_stage"] = "preregister sequence-LOSO context-exclusive internal-gap bridge scorer"
    else:
        decision["next_stage"] = "close same-ID internal gap fill and audit cross-ID tracklet relinking ceiling"

    write_csv(
        output_dir / "gap_pairs.csv",
        gap_pair_rows,
        [
            "sequence", "gap_pair_id", "baseline_track_id", "left_segment_id", "right_segment_id",
            "left_last_frame", "right_first_frame", "gap_frames", "left_modal_gt_id",
            "right_modal_gt_id", "minimum_segment_purity", "matched_segment_support", "eligible_gap",
            "visible_target_frames", "matched_bridge_events", "safe_bridge_events",
        ],
    )
    write_csv(
        output_dir / "bridge_events.csv",
        bridge_event_rows,
        [
            "sequence", "frame", "target_gt_id", "inherited_track_id", "gap_pair_id", "gap_frames",
            "minimum_segment_purity", "matched_segment_support", "candidate_row_index", "global_pre_idx",
            "candidate_track_id", "candidate_score", "target_iou", "maximum_context_iou",
            "context_exclusive",
        ],
    )
    write_csv(
        output_dir / "gap_summary.csv",
        summary_rows,
        [
            "sequence", "consecutive_segment_pairs", "gap_le30_pairs", "same_support_gap_pairs",
            "gap_frames", "target_gt_absent", "target_gt_visible", "unique_target_frames",
            "deduped_competing_targets", "target_no_candidates", "matched_additions", "unmatched_targets",
            "candidate_track_zero", "candidate_track_same", "candidate_track_other",
            "context_iou_ge_0_999", "context_iou_ge_0_90", "context_iou_ge_0_50",
            "context_iou_lt_0_50", "all_plan_events", "safe_plan_events",
        ],
    )
    write_csv(
        output_dir / "variant_metrics.csv",
        variant_rows,
        ["variant", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets"],
    )
    write_csv(
        output_dir / "per_sequence_metrics.csv",
        sequence_metric_rows,
        ["variant", "sequence", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets"],
    )

    report = {
        "schema": "fmtrack.m23_4.existing_id_gap_bridge_oracle.v1",
        "protocol": {
            "starting_tracker": "M23-3-1 dense replacement oracle context",
            "eligible_gap": "consecutive segments of same baseline tracker ID, 1..30 missing frames, same positive modal train-GT support",
            "target_deduplication": "per frame and GT: shorter gap, higher minimum purity, more matched support, lower inherited ID",
            "candidate_target_matching": "global frame-level Hungarian at IoU>=0.5",
            "all_additions": "all GT-matched candidates",
            "safe_additions": "maximum IoU with every dense-context row in the frame is strictly below 0.5",
            "new_identity_namespace": False,
            "identity_relabeling": False,
            "suppressor_routing": False,
            "models_trained": 0,
            "parameter_sweeps": 0,
            "gap_sweeps": 0,
            "threshold_sweeps": 0,
            "mot20_train_trackeval_invocations": 1,
            "locked_trackeval_calls": 0,
            "locked_label_reads": 0,
        },
        "counts": {key: int(value) for key, value in all_counts.items()},
        "acceptance": acceptance,
        "variant_metrics": by_variant,
        "exact_combined_metrics": exact_combined,
        "event_plan_hashes": event_plan_hashes,
        "tracker_hashes": tracker_hashes,
        "decision": decision,
        "sources": {
            "preregister": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
            "candidate_manifest": {"path": str(candidate_manifest_path), "sha256": sha256_file(candidate_manifest_path)},
            "candidate_keys": {"path": str(candidate_keys_path), "sha256": sha256_file(candidate_keys_path)},
            "segment_script": {"path": str(segment_script_path), "sha256": sha256_file(segment_script_path)},
            "replacement_report": {"path": str(replacement_report_path), "sha256": sha256_file(replacement_report_path)},
            "replacement_manifest": {"path": str(replacement_manifest_path), "sha256": sha256_file(replacement_manifest_path)},
            **source_hashes,
        },
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    compact_files = (
        "gap_pairs.csv",
        "bridge_events.csv",
        "gap_summary.csv",
        "variant_metrics.csv",
        "per_sequence_metrics.csv",
        "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_4.existing_id_gap_bridge_oracle.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
        "event_plan_hashes": event_plan_hashes,
        "tracker_hashes": tracker_hashes,
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
