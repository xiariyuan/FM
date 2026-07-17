"""M23-6-0 long-horizon tracklet identity-chain oracle on MOT20 train.

The audit starts from the exact M23-4 all-additions tracker. It compares four
GT-derived identity action spaces while preserving every source row and box:

* pure_segment_chain: weighted non-overlapping chains of purity-1 segments;
* modal_segment_chain: weighted non-overlapping chains of all positive-modal
  segments, relabeling each selected segment as one identity chain;
* modal_core_chain: relabel every row whose frame-level Hungarian GT match is
  equal to its segment's positive modal GT, allowing frame-level switching
  between overlapping candidate segments while leaving non-modal rows intact;
* full_matched_identity: relabel every Hungarian-matched row to its GT chain ID.

All variants use sequence-separated synthetic IDs below 2^24. This is an oracle
ceiling audit, not a deployable result. No model, threshold/parameter sweep, or
locked P15 evaluation is performed.
"""
from __future__ import annotations

import argparse
import bisect
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

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_INDEX = {sequence: index for index, sequence in enumerate(SEQUENCES)}
VARIANTS = (
    "all_additions_context",
    "pure_segment_chain_oracle",
    "modal_segment_chain_oracle",
    "modal_core_chain_oracle",
    "full_matched_identity_ceiling",
)
SYNTHETIC_BASE = 10_000_000
SYNTHETIC_SEQUENCE_STRIDE = 1_000_000
MAX_EXACT_FLOAT32_INTEGER = 1 << 24
TARGET_HOTA = 82.50
MODAL_SEGMENT_HOTA_GATE = 80.00
MODAL_CORE_HOTA_GATE = 82.00
FULL_IDENTITY_HOTA_GATE = 82.50
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


def replace_identity(line: str, identity: int) -> str:
    _, _, fields = parse_tracker_line(line)
    fields[1] = str(int(identity))
    return ",".join(fields)


def validate_tracker(lines: Sequence[str], label: str) -> None:
    seen: set[Tuple[int, int]] = set()
    previous_frame = -1
    for line in lines:
        frame, identity, _ = parse_tracker_line(line)
        if frame < previous_frame:
            raise RuntimeError(f"{label}: frames are not monotonic")
        previous_frame = frame
        key = (frame, identity)
        if key in seen:
            raise RuntimeError(f"{label}: duplicate frame/ID {key}")
        seen.add(key)


def synthetic_chain_id(sequence: str, gt_id: int) -> int:
    identity = SYNTHETIC_BASE + SEQUENCE_INDEX[sequence] * SYNTHETIC_SEQUENCE_STRIDE + int(gt_id)
    if not (0 < identity < MAX_EXACT_FLOAT32_INTEGER):
        raise RuntimeError(f"synthetic ID out of exact float32 range: {identity}")
    return identity


def weighted_nonoverlap_chain(segments: Sequence[Mapping[str, object]]) -> List[dict]:
    items = sorted(
        (dict(row) for row in segments),
        key=lambda row: (int(row["last_frame"]), int(row["first_frame"]), int(row["segment_id"])),
    )
    ends = [int(row["last_frame"]) for row in items]
    predecessors = [
        bisect.bisect_left(ends, int(row["first_frame"])) - 1 for row in items
    ]
    # Lexicographic objective: modal matches, rows, fewer segments, stable IDs.
    dp: List[Tuple[int, int, int, Tuple[int, ...]]] = [(0, 0, 0, ())]
    for index, row in enumerate(items, 1):
        base = dp[predecessors[index - 1] + 1]
        take = (
            base[0] + int(row["modal_count"]),
            base[1] + int(row["frames"]),
            base[2] - 1,
            base[3] + (index - 1,),
        )
        skip = dp[index - 1]
        dp.append(take if take[:3] > skip[:3] else skip)
    return [items[index] for index in dp[-1][3]]


def hash_segment_plan(rows: Sequence[Mapping[str, object]], selected_field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["sequence"]), int(item["segment_id"]))):
        value = {
            "sequence": str(row["sequence"]),
            "segment_id": int(row["segment_id"]),
            "baseline_track_id": int(row["baseline_track_id"]),
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "frames": int(row["frames"]),
            "matched_frames": int(row["matched_frames"]),
            "modal_gt_id": int(row["modal_gt_id"]),
            "modal_count": int(row["modal_count"]),
            "modal_purity": float(row["modal_purity"]),
            "selected": bool(row[selected_field]),
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_row_plan(rows: Sequence[Mapping[str, object]], target_field: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: (str(item["sequence"]), int(item["line_index"]))):
        target = int(row[target_field])
        if target <= 0:
            continue
        value = {
            "sequence": str(row["sequence"]),
            "line_index": int(row["line_index"]),
            "frame": int(row["frame"]),
            "source_track_id": int(row["source_track_id"]),
            "segment_id": int(row["segment_id"]),
            "matched_gt_id": int(row["matched_gt_id"]),
            "segment_modal_gt_id": int(row["segment_modal_gt_id"]),
            "target_chain_gt_id": target,
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_plans(
    repo: Path,
    source_dir: Path,
    gt_root: Path,
    m23_script: Path,
    segment_script: Path,
) -> dict:
    m23 = load_module(m23_script, "m23_for_m23_6")
    segment_module = load_module(segment_script, "segment_for_m23_6")
    source_hashes = {
        "m23_script": sha256_file(m23_script),
        "segment_script": sha256_file(segment_script),
        "source_trackers": {},
        "gt": {},
    }
    segment_rows_all: List[dict] = []
    row_rows_all: List[dict] = []
    summary_rows: List[dict] = []
    plans_by_sequence: Dict[str, dict] = {}
    combined_counts = Counter()

    for sequence in SEQUENCES:
        tracker_path = source_dir / f"{sequence}.txt"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        source_hashes["source_trackers"][sequence] = sha256_file(tracker_path)
        source_hashes["gt"][sequence] = sha256_file(gt_path)
        source_lines = tracker_path.read_text(encoding="utf-8").splitlines()
        validate_tracker(source_lines, f"{sequence}/source")
        frame_to_segment, segment_support, raw_segment_rows = segment_module.build_tracklet_segments(
            m23, sequence, tracker_path, gt_path
        )
        segments = [dict(row) for row in raw_segment_rows]
        segment_meta = {int(row["segment_id"]): row for row in segments}
        positive_groups: MutableMapping[int, List[dict]] = defaultdict(list)
        pure_groups: MutableMapping[int, List[dict]] = defaultdict(list)
        for row in segments:
            modal_gt = int(row["modal_gt_id"])
            if modal_gt > 0:
                positive_groups[modal_gt].append(row)
                if float(row["modal_purity"]) == 1.0:
                    pure_groups[modal_gt].append(row)
        modal_selected: set[int] = set()
        pure_selected: set[int] = set()
        modal_relink_edges = 0
        pure_relink_edges = 0
        duplicate_gt_frames = Counter()
        overlapping_segment_pairs = 0
        for gt_id, rows in positive_groups.items():
            ordered = sorted(rows, key=lambda item: (int(item["first_frame"]), int(item["last_frame"])))
            active: List[dict] = []
            for row in ordered:
                start, end = int(row["first_frame"]), int(row["last_frame"])
                active = [item for item in active if int(item["last_frame"]) >= start]
                overlapping_segment_pairs += len(active)
                active.append(row)
                for frame in range(start, end + 1):
                    duplicate_gt_frames[(gt_id, frame)] += 1
            selected = weighted_nonoverlap_chain(rows)
            modal_selected.update(int(row["segment_id"]) for row in selected)
            modal_relink_edges += max(0, len(selected) - 1)
        for gt_id, rows in pure_groups.items():
            selected = weighted_nonoverlap_chain(rows)
            pure_selected.update(int(row["segment_id"]) for row in selected)
            pure_relink_edges += max(0, len(selected) - 1)

        baseline = m23.load_baseline(tracker_path)
        gt = m23.load_gt(gt_path)
        row_records: Dict[int, dict] = {}
        preproc_removed = 0
        for frame in sorted(baseline):
            frame_rows = baseline[frame]
            kept, valid_gt, removed = m23.valid_and_distractor_filtered(frame_rows, gt.get(frame, []))
            preproc_removed += len(removed)
            kept_uids = {int(candidate.uid) for candidate in kept}
            for candidate in frame_rows:
                line_index = int(candidate.uid & ((1 << 32) - 1))
                segment_id = int(frame_to_segment[(int(candidate.original_id), int(frame))])
                modal_gt = int(segment_support[segment_id])
                row_records[line_index] = {
                    "sequence": sequence,
                    "line_index": line_index,
                    "frame": int(frame),
                    "source_track_id": int(candidate.original_id),
                    "segment_id": segment_id,
                    "segment_modal_gt_id": modal_gt,
                    "matched_gt_id": 0,
                    "match_iou": 0.0,
                    "preproc_kept": int(candidate.uid) in kept_uids,
                    "modal_core_target_gt_id": 0,
                    "full_identity_target_gt_id": 0,
                }
            for candidate_index, gt_index, overlap in m23.match_candidates(kept, valid_gt):
                candidate = kept[candidate_index]
                line_index = int(candidate.uid & ((1 << 32) - 1))
                matched_gt = int(valid_gt[gt_index].gt_id)
                row_records[line_index]["matched_gt_id"] = matched_gt
                row_records[line_index]["match_iou"] = float(overlap)
                row_records[line_index]["full_identity_target_gt_id"] = matched_gt
                if matched_gt == int(row_records[line_index]["segment_modal_gt_id"]):
                    row_records[line_index]["modal_core_target_gt_id"] = matched_gt
        if len(row_records) != len(source_lines):
            raise RuntimeError(f"{sequence}: row plan count mismatch")

        segment_output_rows: List[dict] = []
        for row in segments:
            sid = int(row["segment_id"])
            output = {
                **row,
                "selected_pure_chain": sid in pure_selected,
                "selected_modal_chain": sid in modal_selected,
                "synthetic_chain_id": synthetic_chain_id(sequence, int(row["modal_gt_id"])) if int(row["modal_gt_id"]) > 0 else 0,
            }
            segment_output_rows.append(output)
            segment_rows_all.append(output)
        row_output_rows = [row_records[index] for index in sorted(row_records)]
        row_rows_all.extend(row_output_rows)

        positive_segments = [row for row in segments if int(row["modal_gt_id"]) > 0]
        pure_segments = [row for row in positive_segments if float(row["modal_purity"]) == 1.0]
        selected_modal_segments = [segment_meta[sid] for sid in modal_selected]
        selected_pure_segments = [segment_meta[sid] for sid in pure_selected]
        counts = Counter({
            "source_rows": len(source_lines),
            "segments_total": len(segments),
            "segments_positive": len(positive_segments),
            "segments_pure": len(pure_segments),
            "segments_support_zero_or_tied": len(segments) - len(positive_segments),
            "positive_modal_matches": sum(int(row["modal_count"]) for row in positive_segments),
            "positive_segment_rows": sum(int(row["frames"]) for row in positive_segments),
            "pure_chain_segments": len(pure_selected),
            "pure_chain_rows": sum(int(row["frames"]) for row in selected_pure_segments),
            "pure_chain_modal_matches": sum(int(row["modal_count"]) for row in selected_pure_segments),
            "pure_chain_relink_edges": pure_relink_edges,
            "modal_chain_segments": len(modal_selected),
            "modal_chain_rows": sum(int(row["frames"]) for row in selected_modal_segments),
            "modal_chain_modal_matches": sum(int(row["modal_count"]) for row in selected_modal_segments),
            "modal_chain_relink_edges": modal_relink_edges,
            "overlapping_segment_pairs": overlapping_segment_pairs,
            "duplicate_gt_frames": sum(value > 1 for value in duplicate_gt_frames.values()),
            "duplicate_gt_frame_excess": sum(max(0, value - 1) for value in duplicate_gt_frames.values()),
            "modal_core_row_actions": sum(int(row["modal_core_target_gt_id"]) > 0 for row in row_output_rows),
            "full_identity_row_actions": sum(int(row["full_identity_target_gt_id"]) > 0 for row in row_output_rows),
            "matched_rows_not_modal_core": sum(
                int(row["full_identity_target_gt_id"]) > 0 and int(row["modal_core_target_gt_id"]) == 0
                for row in row_output_rows
            ),
            "preproc_removed_rows": preproc_removed,
        })
        counts["pure_chain_modal_match_recall_pp"] = 100.0 * counts["pure_chain_modal_matches"] / max(1, counts["positive_modal_matches"])
        counts["modal_chain_modal_match_recall_pp"] = 100.0 * counts["modal_chain_modal_matches"] / max(1, counts["positive_modal_matches"])
        summary_rows.append({"sequence": sequence, **counts})
        for key, value in counts.items():
            if not key.endswith("_pp"):
                combined_counts[key] += value
        plans_by_sequence[sequence] = {
            "source_lines": source_lines,
            "segment_rows": segment_output_rows,
            "row_rows": row_output_rows,
            "pure_selected": pure_selected,
            "modal_selected": modal_selected,
        }

    combined_counts["pure_chain_modal_match_recall_pp"] = 100.0 * combined_counts["pure_chain_modal_matches"] / max(1, combined_counts["positive_modal_matches"])
    combined_counts["modal_chain_modal_match_recall_pp"] = 100.0 * combined_counts["modal_chain_modal_matches"] / max(1, combined_counts["positive_modal_matches"])
    summary_rows.append({"sequence": "COMBINED", **combined_counts})
    return {
        "plans_by_sequence": plans_by_sequence,
        "segment_rows": segment_rows_all,
        "row_rows": row_rows_all,
        "summary_rows": summary_rows,
        "source_hashes": source_hashes,
        "plan_hashes": {
            "pure_segment_chain": hash_segment_plan(segment_rows_all, "selected_pure_chain"),
            "modal_segment_chain": hash_segment_plan(segment_rows_all, "selected_modal_chain"),
            "modal_core_rows": hash_row_plan(row_rows_all, "modal_core_target_gt_id"),
            "full_identity_rows": hash_row_plan(row_rows_all, "full_identity_target_gt_id"),
        },
    }


def build_tracker_variants(plans: dict, output_dir: Path) -> Dict[str, Dict[str, str]]:
    tracker_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in VARIANTS}
    for sequence in SEQUENCES:
        plan = plans["plans_by_sequence"][sequence]
        source_lines = list(plan["source_lines"])
        pure_lines = list(source_lines)
        modal_lines = list(source_lines)
        modal_core_lines = list(source_lines)
        full_lines = list(source_lines)
        segment_by_id = {int(row["segment_id"]): row for row in plan["segment_rows"]}
        for row in plan["row_rows"]:
            index = int(row["line_index"])
            segment_id = int(row["segment_id"])
            segment = segment_by_id[segment_id]
            if bool(segment["selected_pure_chain"]):
                pure_lines[index] = replace_identity(
                    source_lines[index], synthetic_chain_id(sequence, int(segment["modal_gt_id"]))
                )
            if bool(segment["selected_modal_chain"]):
                modal_lines[index] = replace_identity(
                    source_lines[index], synthetic_chain_id(sequence, int(segment["modal_gt_id"]))
                )
            modal_target = int(row["modal_core_target_gt_id"])
            if modal_target > 0:
                modal_core_lines[index] = replace_identity(source_lines[index], synthetic_chain_id(sequence, modal_target))
            full_target = int(row["full_identity_target_gt_id"])
            if full_target > 0:
                full_lines[index] = replace_identity(source_lines[index], synthetic_chain_id(sequence, full_target))
        variants = {
            "all_additions_context": source_lines,
            "pure_segment_chain_oracle": pure_lines,
            "modal_segment_chain_oracle": modal_lines,
            "modal_core_chain_oracle": modal_core_lines,
            "full_matched_identity_ceiling": full_lines,
        }
        for variant, lines in variants.items():
            validate_tracker(lines, f"{sequence}/{variant}")
            if len(lines) != len(source_lines):
                raise RuntimeError(f"{sequence}/{variant}: row count changed")
            path = output_dir / "eval_work" / "trackers" / variant / "data" / f"{sequence}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tracker_hashes[variant][sequence] = sha256_file(path)
        if tracker_hashes["all_additions_context"][sequence] != plans["source_hashes"]["source_trackers"][sequence]:
            raise RuntimeError(f"{sequence}: source tracker byte reproduction failed")
    return tracker_hashes


def run_trackeval(repo: Path, gt_root: Path, output_dir: Path) -> None:
    eval_work = output_dir / "eval_work"
    seqmap = eval_work / "seqmaps" / "MOT20_train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(SEQUENCES) + "\n", encoding="utf-8")
    command = [
        sys.executable, str(repo / "TrackEval" / "scripts" / "run_mot_challenge.py"),
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
    parser.add_argument("--source-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1/eval_work/trackers/all_existing_id_gap_bridge_oracle/data")
    parser.add_argument("--parent-dir", default="outputs/mot20_m23_20260717/existing_id_gap_bridge_oracle_v1")
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument("--m23-script", default="scripts/audit_m23_mot20_expanded_evidence_oracle.py")
    parser.add_argument("--segment-script", default="scripts/audit_m23_3_candidate_default_action_space.py")
    parser.add_argument("--preregister", default="outputs/mot20_m23_20260717/long_horizon_tracklet_chain_preregister_v1.json")
    parser.add_argument("--prepare-preregister")
    parser.add_argument("--validate-trackers-only", action="store_true")
    parser.add_argument("--out-dir", default="outputs/mot20_m23_20260717/long_horizon_tracklet_chain_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    source_dir = resolve(repo, args.source_dir)
    parent_dir = resolve(repo, args.parent_dir)
    gt_root = resolve(repo, args.gt_root)
    m23_script = resolve(repo, args.m23_script)
    segment_script = resolve(repo, args.segment_script)
    prereg_path = resolve(repo, args.preregister)
    output_dir = resolve(repo, args.out_dir)
    plans = build_plans(repo, source_dir, gt_root, m23_script, segment_script)
    plans["source_hashes"]["parent_report"] = sha256_file(parent_dir / "report.json")
    plans["source_hashes"]["parent_manifest"] = sha256_file(parent_dir / "manifest.json")
    script_sha = sha256_file(Path(__file__).resolve())

    if args.prepare_preregister:
        references = {
            f"{row['variant']}::{row['sequence']}": row
            for row in read_csv_rows(parent_dir / "per_sequence_metrics.csv")
            if row["variant"] == "all_existing_id_gap_bridge_oracle"
        }
        source_combined = references["all_existing_id_gap_bridge_oracle::COMBINED"]
        source_hota = float(source_combined["HOTA"])
        source_idsw = int(source_combined["IDSW"])
        prereg = {
            "schema": "fmtrack.m23_6.long_horizon_tracklet_chain.preregister.v1",
            "script_sha256": script_sha,
            "sources": plans["source_hashes"],
            "protocol": {
                "sequences": list(SEQUENCES), "variants": list(VARIANTS),
                "segment_definition": "contiguous tracker ID; frame gap >1 starts a new segment",
                "segment_support": "unique modal valid-GT identity among frame-level threshold-before-Hungarian matches",
                "chain_solver": "per-GT weighted interval scheduling; maximize modal matches, then rows, then fewer segments",
                "pure_chain": "purity exactly 1.0; relabel every row in selected segments",
                "modal_chain": "all positive-modal segments; relabel every row in selected non-overlapping segments",
                "modal_core_chain": "relabel rows only when their Hungarian match equals their segment modal GT",
                "full_identity": "relabel every Hungarian-matched row to its matched GT chain",
                "synthetic_id_base": SYNTHETIC_BASE,
                "synthetic_sequence_stride": SYNTHETIC_SEQUENCE_STRIDE,
                "row_additions": 0, "row_deletions": 0, "box_changes": 0,
                "models_trained": 0, "threshold_sweeps": 0, "parameter_sweeps": 0,
                "locked_label_reads": 0, "locked_trackeval_calls": 0,
            },
            "expected": {
                "summary_rows": plans["summary_rows"],
                "plan_hashes": plans["plan_hashes"],
            },
            "references": references,
            "gate_rationale": {
                "source_hota": source_hota, "target_hota": TARGET_HOTA,
                "source_to_target_gap": TARGET_HOTA - source_hota,
                "modal_segment_gate_recovered_gap_fraction": (MODAL_SEGMENT_HOTA_GATE - source_hota) / (TARGET_HOTA - source_hota),
                "modal_core_gate_residual_to_target": TARGET_HOTA - MODAL_CORE_HOTA_GATE,
                "idsw_reduction_thresholds": "modal segment <=50% source; modal core <=25% source",
            },
            "gates": {
                "modal_segment_combined_hota_min": MODAL_SEGMENT_HOTA_GATE,
                "modal_segment_all_sequence_hota_nonnegative_vs_source": True,
                "modal_segment_idsw_max": source_idsw // 2,
                "modal_core_combined_hota_min": MODAL_CORE_HOTA_GATE,
                "modal_core_all_sequence_hota_nonnegative_vs_source": True,
                "modal_core_idsw_max": source_idsw // 4,
                "full_identity_combined_hota_min": FULL_IDENTITY_HOTA_GATE,
                "full_identity_all_sequence_hota_nonnegative_vs_source": True,
            },
        }
        path = resolve(repo, args.prepare_preregister)
        canonical_json_dump(prereg, path)
        print(json.dumps({
            "script_sha256": script_sha,
            "preregister": str(path),
            "summary": plans["summary_rows"][-1],
            "plan_hashes": plans["plan_hashes"],
            "gates": prereg["gates"],
        }, indent=2, sort_keys=True))
        return

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg["script_sha256"] != script_sha:
        raise RuntimeError("script hash mismatch")
    if prereg["sources"] != plans["source_hashes"]:
        raise RuntimeError("source hashes changed")
    if prereg["expected"]["summary_rows"] != plans["summary_rows"]:
        raise RuntimeError("structural summary changed")
    if prereg["expected"]["plan_hashes"] != plans["plan_hashes"]:
        raise RuntimeError("chain plan changed")
    if prereg["protocol"]["variants"] != list(VARIANTS):
        raise RuntimeError("variant protocol changed")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    tracker_hashes = build_tracker_variants(plans, output_dir)
    if args.validate_trackers_only:
        print(json.dumps({
            "validation_only": True,
            "summary": plans["summary_rows"][-1],
            "plan_hashes": plans["plan_hashes"],
            "tracker_hashes": tracker_hashes,
        }, indent=2, sort_keys=True))
        return

    run_trackeval(repo, gt_root, output_dir)
    metric_rows: List[dict] = []
    summary_metrics: List[dict] = []
    for variant in VARIANTS:
        metric_rows.extend(parse_detailed(output_dir / "eval_work" / "eval" / variant / "pedestrian_detailed.csv", variant))
        summary = parse_summary(output_dir / "eval_work" / "eval" / variant / "pedestrian_summary.txt")
        summary_metrics.append({"variant": variant, **{field: summary[field] for field in METRIC_FIELDS}})
    metric_map = {(row["variant"], row["sequence"]): row for row in metric_rows}
    for sequence in (*SEQUENCES, "COMBINED"):
        reference = prereg["references"][f"all_existing_id_gap_bridge_oracle::{sequence}"]
        actual = metric_map[("all_additions_context", sequence)]
        for field in ("HOTA", "DetA", "AssA", "IDF1", "MOTA"):
            if float(actual[field]) != float(reference[field]):
                raise RuntimeError(f"source/{sequence}/{field}: reference mismatch")
        for field in ("IDSW", "CLR_FP", "CLR_FN", "Dets"):
            if int(actual[field]) != int(reference[field]):
                raise RuntimeError(f"source/{sequence}/{field}: reference mismatch")

    source = metric_map[("all_additions_context", "COMBINED")]
    modal = metric_map[("modal_segment_chain_oracle", "COMBINED")]
    core = metric_map[("modal_core_chain_oracle", "COMBINED")]
    full = metric_map[("full_matched_identity_ceiling", "COMBINED")]
    modal_deltas = {
        sequence: metric_map[("modal_segment_chain_oracle", sequence)]["HOTA"] - metric_map[("all_additions_context", sequence)]["HOTA"]
        for sequence in SEQUENCES
    }
    core_deltas = {
        sequence: metric_map[("modal_core_chain_oracle", sequence)]["HOTA"] - metric_map[("all_additions_context", sequence)]["HOTA"]
        for sequence in SEQUENCES
    }
    full_deltas = {
        sequence: metric_map[("full_matched_identity_ceiling", sequence)]["HOTA"] - metric_map[("all_additions_context", sequence)]["HOTA"]
        for sequence in SEQUENCES
    }
    gates = prereg["gates"]
    modal_pass = (
        modal["HOTA"] >= float(gates["modal_segment_combined_hota_min"])
        and all(delta >= 0.0 for delta in modal_deltas.values())
        and modal["IDSW"] <= int(gates["modal_segment_idsw_max"])
    )
    core_pass = (
        core["HOTA"] >= float(gates["modal_core_combined_hota_min"])
        and all(delta >= 0.0 for delta in core_deltas.values())
        and core["IDSW"] <= int(gates["modal_core_idsw_max"])
    )
    full_pass = (
        full["HOTA"] >= float(gates["full_identity_combined_hota_min"])
        and all(delta >= 0.0 for delta in full_deltas.values())
    )
    ceiling_passed = modal_pass and core_pass and full_pass
    decision = {
        "source_context_reproduced": True,
        "modal_segment_hota": modal["HOTA"],
        "modal_segment_hota_gain": modal["HOTA"] - source["HOTA"],
        "modal_segment_idsw": modal["IDSW"],
        "modal_segment_sequence_hota_deltas": modal_deltas,
        "modal_segment_gate_passed": modal_pass,
        "modal_core_hota": core["HOTA"],
        "modal_core_hota_gain": core["HOTA"] - source["HOTA"],
        "modal_core_idsw": core["IDSW"],
        "modal_core_sequence_hota_deltas": core_deltas,
        "modal_core_gate_passed": core_pass,
        "full_identity_hota": full["HOTA"],
        "full_identity_hota_gain": full["HOTA"] - source["HOTA"],
        "full_identity_idsw": full["IDSW"],
        "full_identity_sequence_hota_deltas": full_deltas,
        "full_identity_gate_passed": full_pass,
        "long_horizon_chain_ceiling_passed": ceiling_passed,
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": (
            "train nested sequence-LOSO long-horizon tracklet graph with row-level duplicate arbitration and abstention"
            if ceiling_passed else
            "close current segment-chain representation and acquire stronger tracklet purification/evidence before graph training"
        ),
    }

    write_csv(output_dir / "chain_summary.csv", plans["summary_rows"], [
        "sequence", "source_rows", "segments_total", "segments_positive", "segments_pure",
        "segments_support_zero_or_tied", "positive_modal_matches", "positive_segment_rows",
        "pure_chain_segments", "pure_chain_rows", "pure_chain_modal_matches", "pure_chain_relink_edges",
        "pure_chain_modal_match_recall_pp", "modal_chain_segments", "modal_chain_rows",
        "modal_chain_modal_matches", "modal_chain_relink_edges", "modal_chain_modal_match_recall_pp",
        "overlapping_segment_pairs", "duplicate_gt_frames", "duplicate_gt_frame_excess",
        "modal_core_row_actions", "full_identity_row_actions", "matched_rows_not_modal_core",
        "preproc_removed_rows",
    ])
    write_csv(output_dir / "chain_segments.csv", plans["segment_rows"], [
        "sequence", "segment_id", "baseline_track_id", "local_segment_index", "first_frame", "last_frame",
        "frames", "matched_frames", "unique_matched_gt", "modal_gt_id", "modal_count", "modal_purity",
        "modal_tied", "mean_match_iou", "selected_pure_chain", "selected_modal_chain", "synthetic_chain_id",
    ])
    write_csv(output_dir / "per_sequence_metrics.csv", metric_rows, ["variant", "sequence", *METRIC_FIELDS])
    write_csv(output_dir / "variant_metrics.csv", summary_metrics, ["variant", *METRIC_FIELDS])
    report = {
        "schema": "fmtrack.m23_6.long_horizon_tracklet_chain.report.v1",
        "protocol": prereg["protocol"], "gates": prereg["gates"],
        "gate_rationale": prereg["gate_rationale"],
        "counts": plans["summary_rows"][-1], "plan_hashes": plans["plan_hashes"],
        "source_hashes": plans["source_hashes"], "decision": decision,
        "locked_policy": {
            "p15_no_op": True, "remaining_locked_rows_unread": 156,
            "locked_label_reads": 0, "locked_trackeval_calls": 0,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    compact_files = ("chain_summary.csv", "chain_segments.csv", "per_sequence_metrics.csv", "variant_metrics.csv", "report.json")
    manifest = {
        "schema": "fmtrack.m23_6.long_horizon_tracklet_chain.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
        "tracker_hashes": tracker_hashes, "plan_hashes": plans["plan_hashes"],
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
