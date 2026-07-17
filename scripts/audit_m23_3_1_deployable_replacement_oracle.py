"""M23-3-1 deployable-form observation-replacement oracle audit.

This audit starts from the frozen raw MOT20 baseline tracker and preserves its
row count, frame sequence, tracker IDs, and tail columns exactly. The only
allowed action is to replace the box and score of an existing baseline row by a
frozen-budget pre-NMS candidate whose ``candidate_track_id`` equals that
baseline row's tracker ID in the same frame. No detection is added, removed, or
relabelled; suppressor routing and spawn are disabled.

A baseline contiguous segment obtains an oracle support identity from the
unique modal MOT20-train GT identity among its same-frame IoU>=0.5 Hungarian
matches. For every candidate with an observable baseline row and target GT in
the frame, utility is candidate IoU to that support GT minus the original
baseline-row IoU. Within each (sequence, frame, baseline track ID) group, the
fixed oracle chooses the candidate with maximum strictly positive utility;
otherwise the baseline row is left unchanged.

Three fixed variants are evaluated:
  * baseline_raw: exact frozen baseline reproduction;
  * m23_selected_replacement_oracle: positive replacements restricted to
    suppressed keys selected by the frozen M23-1 budget oracle;
  * dense_replacement_oracle: positive replacements over all frozen-budget
    candidates.

This is a train-GT oracle ceiling, not a deployable result. No model, threshold
sweep, graph-family sweep, or locked P15 evaluation is performed.
"""
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_TO_ID = {name: index for index, name in enumerate(SEQUENCES)}
VARIANTS = (
    "baseline_raw",
    "m23_selected_replacement_oracle",
    "dense_replacement_oracle",
)
UTILITY_BIN_EDGES = (-math.inf, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, math.inf)
UTILITY_BIN_LABELS = (
    "(-inf,0]",
    "(0,0.01]",
    "(0.01,0.02]",
    "(0.02,0.05]",
    "(0.05,0.10]",
    "(0.10,0.20]",
    "(0.20,0.40]",
    "(0.40,inf)",
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


def tracker_map(lines: Sequence[str], label: str) -> Dict[Tuple[int, int], Tuple[int, str]]:
    mapping: Dict[Tuple[int, int], Tuple[int, str]] = {}
    previous_frame = -1
    for index, line in enumerate(lines):
        frame, identity, _ = parse_tracker_line(line)
        if frame < previous_frame:
            raise RuntimeError(f"{label}: frames are not monotonic")
        previous_frame = frame
        key = (frame, identity)
        if key in mapping:
            raise RuntimeError(f"{label}: duplicate tracker ID {identity} in frame {frame}")
        mapping[key] = (index, line)
    return mapping


def identity_skeleton(lines: Sequence[str]) -> List[Tuple[int, int, Tuple[str, ...]]]:
    skeleton = []
    for line in lines:
        frame, identity, fields = parse_tracker_line(line)
        skeleton.append((frame, identity, tuple(fields[7:])))
    return skeleton


def iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    intersection = width * height
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def utility_bin(value: float) -> str:
    if value <= 0.0:
        return UTILITY_BIN_LABELS[0]
    for index in range(1, len(UTILITY_BIN_EDGES) - 1):
        if value <= UTILITY_BIN_EDGES[index + 1]:
            return UTILITY_BIN_LABELS[index]
    return UTILITY_BIN_LABELS[-1]


def candidate_rank(record: Mapping[str, object]) -> Tuple[float, float, float, int]:
    return (
        float(record["utility"]),
        float(record["candidate_iou"]),
        float(record["candidate_score"]),
        -int(record["global_pre_idx"]),
    )


def replacement_line(original_line: str, candidate_row: np.void) -> str:
    frame, identity, fields = parse_tracker_line(original_line)
    x1 = float(candidate_row["candidate_x1"])
    y1 = float(candidate_row["candidate_y1"])
    x2 = float(candidate_row["candidate_x2"])
    y2 = float(candidate_row["candidate_y2"])
    replaced = [
        str(frame),
        str(identity),
        f"{x1:.6f}",
        f"{y1:.6f}",
        f"{x2 - x1:.6f}",
        f"{y2 - y1:.6f}",
        f"{float(candidate_row['candidate_score']):.6f}",
    ]
    replaced.extend(fields[7:])
    return ",".join(replaced)


def hash_replacement_plan(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (
            str(item["sequence"]),
            int(item["frame"]),
            int(item["baseline_track_id"]),
        ),
    ):
        canonical = {
            "sequence": str(record["sequence"]),
            "frame": int(record["frame"]),
            "baseline_track_id": int(record["baseline_track_id"]),
            "global_pre_idx": int(record["global_pre_idx"]),
            "candidate_row_index": int(record["candidate_row_index"]),
            "segment_id": int(record["segment_id"]),
            "target_gt_id": int(record["target_gt_id"]),
            "baseline_iou": float(record["baseline_iou"]),
            "candidate_iou": float(record["candidate_iou"]),
            "utility": float(record["utility"]),
        }
        digest.update(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8"))
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
            rows.append({
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
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--appearance-dir", required=True)
    parser.add_argument("--compact-dir", required=True)
    parser.add_argument("--m23-1-events", required=True)
    parser.add_argument("--segment-script", required=True)
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
    event_path = resolve(repo, args.m23_1_events)
    segment_script_path = resolve(repo, args.segment_script)
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
        raise RuntimeError("preregistered action script hash mismatch")
    if prereg["sources"]["segment_script"]["sha256"] != sha256_file(segment_script_path):
        raise RuntimeError("segment script hash mismatch")
    if tuple(prereg["protocol"]["variants"]) != VARIANTS:
        raise RuntimeError("preregistered variants mismatch")
    if tuple(float(value) for value in prereg["protocol"]["positive_utility_bin_edges"]) != tuple(
        value for value in UTILITY_BIN_EDGES if math.isfinite(value) and value >= 0.0
    ):
        raise RuntimeError("positive utility-bin edges mismatch")
    if tuple(prereg["protocol"]["utility_bin_labels"]) != UTILITY_BIN_LABELS:
        raise RuntimeError("utility-bin labels mismatch")

    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    candidate_keys_path = compact_dir / "candidate_keys.npy"
    for label, path in (
        ("candidate_manifest", candidate_manifest_path),
        ("candidate_keys", candidate_keys_path),
        ("m23_1_events", event_path),
    ):
        expected = prereg["sources"][label]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")

    m23 = load_module(repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py", "m23_for_m23_3_1")
    segment_module = load_module(segment_script_path, "m23_3_segment_for_m23_3_1")
    candidates = np.load(candidate_manifest_path, allow_pickle=False, mmap_mode="r")
    keys = np.load(candidate_keys_path, allow_pickle=False, mmap_mode="r")
    if len(candidates) != len(keys):
        raise RuntimeError("candidate manifest and key rows differ")
    key_lookup = {
        (int(sequence_id), int(global_pre_idx)): index
        for index, (sequence_id, global_pre_idx) in enumerate(
            zip(keys["sequence_id"], keys["global_pre_idx"])
        )
    }
    if len(key_lookup) != len(keys):
        raise RuntimeError("candidate keys are not unique")

    selected_keys = set()
    with event_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "prenms_budget_replace_add_oracle" or row["source"] != "prenms_suppressed":
                continue
            selected_keys.add((SEQUENCE_TO_ID[row["sequence"]], int(row["global_pre_idx"])))
    if len(selected_keys) != int(prereg["structural_preflight"]["m23_selected_keys"]):
        raise RuntimeError("selected-key count differs from preregistration")
    missing_selected = selected_keys - set(key_lookup)
    if missing_selected:
        raise RuntimeError(f"selected keys outside frozen candidate budget: {list(sorted(missing_selected))[:5]}")

    inventory_rows: List[dict] = []
    utility_counter: Counter[Tuple[str, str, str]] = Counter()
    tracker_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in VARIANTS}
    replacement_plan_hashes: Dict[str, Dict[str, str]] = {
        "m23_selected_replacement_oracle": {},
        "dense_replacement_oracle": {},
    }
    source_hashes = {"baseline": {}, "gt": {}}
    total_counts = Counter()
    total_selected_records: List[dict] = []
    total_dense_records: List[dict] = []
    eval_work = output_dir / "eval_work"

    for sequence_id, sequence in enumerate(SEQUENCES):
        print(f"[M23-3-1] building replacement oracle {sequence}", flush=True)
        baseline_path = baseline_dir / f"{sequence}.txt"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        if sha256_file(baseline_path) != prereg["sources"]["baseline"][sequence]:
            raise RuntimeError(f"{sequence}: baseline hash mismatch")
        if sha256_file(gt_path) != prereg["sources"]["gt"][sequence]:
            raise RuntimeError(f"{sequence}: GT hash mismatch")
        source_hashes["baseline"][sequence] = sha256_file(baseline_path)
        source_hashes["gt"][sequence] = sha256_file(gt_path)

        baseline_lines = baseline_path.read_text(encoding="utf-8").splitlines()
        baseline_line_map = tracker_map(baseline_lines, f"{sequence}/baseline")
        baseline_skeleton = identity_skeleton(baseline_lines)
        baseline = m23.load_baseline(baseline_path)
        baseline_candidates = {
            (int(frame), int(candidate.original_id)): candidate
            for frame, rows in baseline.items()
            for candidate in rows
        }
        gt = m23.load_gt(gt_path)
        valid_gt = {
            (int(frame), int(item.gt_id)): item
            for frame, rows in gt.items()
            for item in rows
            if item.marked != 0 and item.cls == 1
        }
        frame_segment, segment_support, segment_rows = segment_module.build_tracklet_segments(
            m23, sequence, baseline_path, gt_path
        )

        dense_best: Dict[Tuple[int, int], dict] = {}
        selected_best: Dict[Tuple[int, int], dict] = {}
        counts = Counter()
        sequence_indices = np.flatnonzero(candidates["sequence_id"] == sequence_id)
        counts["candidate_rows"] = len(sequence_indices)

        for candidate_row_index in sequence_indices.tolist():
            candidate_row = candidates[candidate_row_index]
            global_pre_idx = int(keys[candidate_row_index]["global_pre_idx"])
            candidate_key = (sequence_id, global_pre_idx)
            selected = candidate_key in selected_keys
            if selected:
                counts["selected_rows"] += 1
            track_id = int(candidate_row["candidate_track_id"])
            if track_id <= 0:
                counts["candidate_track_missing"] += 1
                continue
            counts["candidate_track_observed"] += 1
            frame = int(candidate_row["frame"])
            group_key = (frame, track_id)
            baseline_candidate = baseline_candidates.get(group_key)
            baseline_line_entry = baseline_line_map.get(group_key)
            if baseline_candidate is None or baseline_line_entry is None:
                counts["baseline_row_missing"] += 1
                continue
            segment_id = frame_segment.get((track_id, frame), -1)
            target_gt_id = segment_support.get(segment_id, 0)
            if target_gt_id <= 0:
                counts["segment_support_missing"] += 1
                continue
            target = valid_gt.get((frame, target_gt_id))
            if target is None:
                counts["target_gt_absent"] += 1
                continue

            candidate_box = (
                float(candidate_row["candidate_x1"]),
                float(candidate_row["candidate_y1"]),
                float(candidate_row["candidate_x2"]),
                float(candidate_row["candidate_y2"]),
            )
            baseline_iou = iou_box(baseline_candidate.box, target.box)
            candidate_iou = iou_box(candidate_box, target.box)
            utility = candidate_iou - baseline_iou
            record = {
                "sequence": sequence,
                "frame": frame,
                "baseline_track_id": track_id,
                "segment_id": segment_id,
                "target_gt_id": target_gt_id,
                "global_pre_idx": global_pre_idx,
                "candidate_row_index": candidate_row_index,
                "candidate_score": float(candidate_row["candidate_score"]),
                "baseline_iou": baseline_iou,
                "candidate_iou": candidate_iou,
                "utility": utility,
                "baseline_line_index": int(baseline_line_entry[0]),
            }
            counts["eligible_candidates"] += 1
            if utility > 0.0:
                counts["positive_utility_candidates"] += 1
            utility_counter[(sequence, "dense_candidates", utility_bin(utility))] += 1
            previous = dense_best.get(group_key)
            if previous is None or candidate_rank(record) > candidate_rank(previous):
                dense_best[group_key] = record
            if selected:
                counts["selected_eligible_candidates"] += 1
                if utility > 0.0:
                    counts["selected_positive_utility_candidates"] += 1
                utility_counter[(sequence, "selected_candidates", utility_bin(utility))] += 1
                previous = selected_best.get(group_key)
                if previous is None or candidate_rank(record) > candidate_rank(previous):
                    selected_best[group_key] = record

        dense_records = [record for record in dense_best.values() if float(record["utility"]) > 0.0]
        selected_records = [record for record in selected_best.values() if float(record["utility"]) > 0.0]
        dense_records.sort(key=lambda item: (int(item["frame"]), int(item["baseline_track_id"])))
        selected_records.sort(key=lambda item: (int(item["frame"]), int(item["baseline_track_id"])))
        counts["dense_groups"] = len(dense_best)
        counts["dense_positive_groups"] = len(dense_records)
        counts["selected_groups"] = len(selected_best)
        counts["selected_positive_groups"] = len(selected_records)
        for record in dense_best.values():
            utility_counter[(sequence, "dense_group_best", utility_bin(float(record["utility"])))] += 1
        for record in selected_best.values():
            utility_counter[(sequence, "selected_group_best", utility_bin(float(record["utility"])))] += 1

        variants = {"baseline_raw": list(baseline_lines)}
        for variant, records in (
            ("m23_selected_replacement_oracle", selected_records),
            ("dense_replacement_oracle", dense_records),
        ):
            lines = list(baseline_lines)
            used_line_indices = set()
            for record in records:
                line_index = int(record["baseline_line_index"])
                if line_index in used_line_indices:
                    raise RuntimeError(f"{sequence}/{variant}: more than one replacement for baseline row {line_index}")
                used_line_indices.add(line_index)
                candidate_row = candidates[int(record["candidate_row_index"])]
                lines[line_index] = replacement_line(lines[line_index], candidate_row)
            if len(lines) != len(baseline_lines):
                raise RuntimeError(f"{sequence}/{variant}: row count changed")
            if identity_skeleton(lines) != baseline_skeleton:
                raise RuntimeError(f"{sequence}/{variant}: identity/frame/tail skeleton changed")
            tracker_map(lines, f"{sequence}/{variant}")
            variants[variant] = lines
            replacement_plan_hashes[variant][sequence] = hash_replacement_plan(records)

        for variant, lines in variants.items():
            tracker_path = eval_work / "trackers" / variant / "data" / f"{sequence}.txt"
            tracker_path.parent.mkdir(parents=True, exist_ok=True)
            tracker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tracker_hashes[variant][sequence] = sha256_file(tracker_path)
        if tracker_hashes["baseline_raw"][sequence] != sha256_file(baseline_path):
            raise RuntimeError(f"{sequence}: baseline tracker was not reproduced exactly")

        inventory_rows.append({"sequence": sequence, **counts})
        total_counts.update(counts)
        total_selected_records.extend(selected_records)
        total_dense_records.extend(dense_records)

    inventory_rows.append({"sequence": "COMBINED", **total_counts})
    for sequence, scope, label in list(utility_counter):
        utility_counter[("COMBINED", scope, label)] += utility_counter[(sequence, scope, label)]
    utility_rows = []
    for sequence in (*SEQUENCES, "COMBINED"):
        for scope in ("dense_candidates", "selected_candidates", "dense_group_best", "selected_group_best"):
            for label in UTILITY_BIN_LABELS:
                utility_rows.append({
                    "sequence": sequence,
                    "scope": scope,
                    "utility_bin": label,
                    "count": utility_counter[(sequence, scope, label)],
                })
    replacement_plan_hashes["m23_selected_replacement_oracle"]["COMBINED"] = hash_replacement_plan(total_selected_records)
    replacement_plan_hashes["dense_replacement_oracle"]["COMBINED"] = hash_replacement_plan(total_dense_records)

    # Structural counts were inspected before TrackEval and are bound in the preregistration.
    for field in (
        "candidate_rows",
        "eligible_candidates",
        "dense_positive_groups",
        "selected_positive_groups",
    ):
        actual = int(total_counts[field])
        expected = int(prereg["structural_preflight"][field])
        if actual != expected:
            raise RuntimeError(f"structural preflight mismatch for {field}: {actual} != {expected}")

    run_trackeval(repo, gt_root, eval_work)
    variant_rows: List[dict] = []
    sequence_metric_rows: List[dict] = []
    for variant in VARIANTS:
        summary = parse_summary(eval_work / "eval" / variant / "pedestrian_summary.txt")
        variant_rows.append({
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
        })
        sequence_metric_rows.extend(parse_detailed(eval_work / "eval" / variant / "pedestrian_detailed.csv", variant))

    by_variant = {row["variant"]: row for row in variant_rows}
    by_sequence_variant = {(row["sequence"], row["variant"]): row for row in sequence_metric_rows}
    exact_combined = {
        variant: by_sequence_variant[("COMBINED", variant)]
        for variant in VARIANTS
    }
    tolerance = float(prereg["acceptance"]["baseline_reproduction_tolerance"])
    reference_combined = prereg["references"]["baseline_combined"]
    baseline_metric_deltas = {
        metric: float(exact_combined["baseline_raw"][metric] - reference_combined[metric])
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    }
    baseline_combined_reproduced = all(abs(delta) <= tolerance for delta in baseline_metric_deltas.values())
    baseline_sequence_deltas = {}
    for sequence in SEQUENCES:
        baseline_sequence_deltas[sequence] = {
            metric: float(
                by_sequence_variant[(sequence, "baseline_raw")][metric]
                - prereg["references"]["baseline_per_sequence"][sequence][metric]
            )
            for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
        }
    baseline_sequences_reproduced = all(
        abs(delta) <= tolerance
        for values in baseline_sequence_deltas.values()
        for delta in values.values()
    )

    selected_hota_deltas = {
        sequence: float(
            by_sequence_variant[(sequence, "m23_selected_replacement_oracle")]["HOTA"]
            - by_sequence_variant[(sequence, "baseline_raw")]["HOTA"]
        )
        for sequence in SEQUENCES
    }
    dense_hota_deltas = {
        sequence: float(
            by_sequence_variant[(sequence, "dense_replacement_oracle")]["HOTA"]
            - by_sequence_variant[(sequence, "baseline_raw")]["HOTA"]
        )
        for sequence in SEQUENCES
    }
    acceptance = prereg["acceptance"]
    decision = {
        "baseline_combined_reproduced": baseline_combined_reproduced,
        "baseline_sequences_reproduced": baseline_sequences_reproduced,
        "baseline_metric_deltas": baseline_metric_deltas,
        "baseline_sequence_metric_deltas": baseline_sequence_deltas,
        "selected_combined_hota": float(exact_combined["m23_selected_replacement_oracle"]["HOTA"]),
        "selected_combined_hota_passed": bool(
            exact_combined["m23_selected_replacement_oracle"]["HOTA"]
            >= acceptance["minimum_selected_combined_hota"]
        ),
        "selected_sequence_hota_deltas": selected_hota_deltas,
        "selected_all_sequence_hota_nonnegative": bool(min(selected_hota_deltas.values()) >= 0.0),
        "dense_combined_hota": float(exact_combined["dense_replacement_oracle"]["HOTA"]),
        "dense_combined_hota_passed": bool(
            exact_combined["dense_replacement_oracle"]["HOTA"]
            >= acceptance["minimum_dense_combined_hota"]
        ),
        "dense_sequence_hota_deltas": dense_hota_deltas,
        "dense_all_sequence_hota_nonnegative": bool(min(dense_hota_deltas.values()) >= 0.0),
        "replacement_graph_ceiling_passed": False,
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": "set below after gate evaluation",
    }
    decision["replacement_graph_ceiling_passed"] = bool(
        decision["baseline_combined_reproduced"]
        and decision["baseline_sequences_reproduced"]
        and decision["selected_combined_hota_passed"]
        and decision["selected_all_sequence_hota_nonnegative"]
        and decision["dense_combined_hota_passed"]
        and decision["dense_all_sequence_hota_nonnegative"]
    )
    if decision["replacement_graph_ceiling_passed"]:
        decision["next_stage"] = "preregister sequence-LOSO segment-conditioned replacement scorer with baseline no-op and fixed outer acceptance"
    else:
        decision["next_stage"] = "close direct replacement graph and acquire stronger deployable evidence before training"

    write_csv(
        output_dir / "candidate_inventory.csv",
        inventory_rows,
        [
            "sequence", "candidate_rows", "selected_rows", "candidate_track_missing", "candidate_track_observed",
            "baseline_row_missing", "segment_support_missing", "target_gt_absent", "eligible_candidates",
            "positive_utility_candidates", "selected_eligible_candidates", "selected_positive_utility_candidates",
            "dense_groups", "dense_positive_groups", "selected_groups", "selected_positive_groups",
        ],
    )
    write_csv(
        output_dir / "utility_bins.csv",
        utility_rows,
        ["sequence", "scope", "utility_bin", "count"],
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
        "schema": "fmtrack.m23_3_1.deployable_replacement_oracle.v1",
        "protocol": {
            "starting_tracker": "frozen raw baseline",
            "allowed_action": "replace box and score at the same frame and baseline tracker ID",
            "row_additions": 0,
            "row_deletions": 0,
            "identity_changes": 0,
            "tail_column_changes": 0,
            "segment_definition": "baseline original track ID split whenever observed frame gap > 1",
            "segment_support": "unique modal MOT20-train GT identity among same-frame IoU>=0.5 Hungarian matches; ties unlabeled",
            "utility": "candidate IoU to segment-support GT minus baseline-row IoU to the same GT",
            "group": "sequence, frame, baseline tracker ID",
            "selection": "maximum strictly positive utility; deterministic ties by candidate IoU, score, then lower global_pre_idx",
            "m23_selected_scope": "suppressed keys in frozen M23-1 prenms_budget_replace_add_oracle",
            "suppressor_enabled": False,
            "spawn_enabled": False,
            "models_trained": 0,
            "parameter_sweeps": 0,
            "threshold_sweeps": 0,
            "graph_variant_sweeps": 0,
            "mot20_train_trackeval_invocations": 1,
            "locked_trackeval_calls": 0,
            "locked_label_reads": 0,
        },
        "counts": {
            **{key: int(value) for key, value in total_counts.items()},
            "selected_replacement_plan_rows": len(total_selected_records),
            "dense_replacement_plan_rows": len(total_dense_records),
        },
        "acceptance": acceptance,
        "variant_metrics": by_variant,
        "exact_combined_metrics": exact_combined,
        "replacement_plan_hashes": replacement_plan_hashes,
        "tracker_hashes": tracker_hashes,
        "decision": decision,
        "sources": {
            "preregister": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
            "candidate_manifest": {"path": str(candidate_manifest_path), "sha256": sha256_file(candidate_manifest_path)},
            "candidate_keys": {"path": str(candidate_keys_path), "sha256": sha256_file(candidate_keys_path)},
            "m23_1_events": {"path": str(event_path), "sha256": sha256_file(event_path)},
            "segment_script": {"path": str(segment_script_path), "sha256": sha256_file(segment_script_path)},
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
        "candidate_inventory.csv",
        "utility_bins.csv",
        "variant_metrics.csv",
        "per_sequence_metrics.csv",
        "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_3_1.deployable_replacement_oracle.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
        "replacement_plan_hashes": replacement_plan_hashes,
        "tracker_hashes": tracker_hashes,
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
