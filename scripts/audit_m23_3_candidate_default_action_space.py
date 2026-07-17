"""M23-3-0 candidate-default graph action-space oracle audit on MOT20 train.

This audit does not train a selector. It holds the M23-1 post-NMS oracle context
fixed and changes only the identity action applied to pre-NMS suppressed rows
selected by the frozen M23-1 budgeted replace/add oracle.

Baseline tracker IDs are split into contiguous segments (frame gap > 1 starts a
new segment). Each segment receives an oracle support identity from the unique
modal MOT20-train GT identity among its same-frame Hungarian matches. Every
selected suppressed observation is then classified, in fixed priority order,
as candidate-default, suppressor-override, or spawn-required.

Five tracker variants are evaluated:
  * postnms_context: unchanged M23-1 post-NMS replace/add oracle.
  * candidate_default_safe: candidate-supported suppressed rows are retained;
    all others fall back to the post-NMS context row or abstain.
  * candidate_suppressor_safe: candidate- or suppressor-supported rows are
    retained; spawn-required rows fall back or abstain.
  * candidate_suppressor_episode_spawn: spawn-required rows are retained under
    a new synthetic identity shared within a fixed 30-frame oracle episode.
  * oracle_linked_spawn: unchanged M23-1 pre-NMS budget replace/add oracle.

The GT-derived action labels make this an action-space ceiling, not a deployable
result. No model, threshold sweep, or locked P15 evaluation is performed.
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_TO_ID = {name: index for index, name in enumerate(SEQUENCES)}
SPAWN_EPISODE_MAX_GAP = 30
SYNTHETIC_SPAWN_ID_BASE = 10_000_000
SYNTHETIC_SEQUENCE_STRIDE = 1_000_000
MAX_EXACT_FLOAT32_INTEGER = 1 << 24
VARIANTS = (
    "postnms_context",
    "candidate_default_safe",
    "candidate_suppressor_safe",
    "candidate_suppressor_episode_spawn",
    "oracle_linked_spawn",
)
ACTION_PRIORITY = ("candidate_default", "suppressor_override", "spawn")


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


def replace_tracker_identity(line: str, identity: int) -> str:
    _, _, fields = parse_tracker_line(line)
    fields[1] = str(int(identity))
    return ",".join(fields)


def tracker_frame_identity_map(lines: Sequence[str], label: str) -> Dict[Tuple[int, int], str]:
    mapping: Dict[Tuple[int, int], str] = {}
    for line in lines:
        frame, identity, _ = parse_tracker_line(line)
        key = (frame, identity)
        if key in mapping:
            raise RuntimeError(f"{label}: duplicate identity {identity} in frame {frame}")
        mapping[key] = line
    return mapping


def validate_tracker(lines: Sequence[str], label: str) -> None:
    tracker_frame_identity_map(lines, label)
    previous_frame = -1
    for line in lines:
        frame, _, _ = parse_tracker_line(line)
        if frame < previous_frame:
            raise RuntimeError(f"{label}: tracker frames are not monotonic")
        previous_frame = frame


def modal_support(labels: Sequence[int]) -> Tuple[int, int, float, bool]:
    if not labels:
        return 0, 0, 0.0, False
    counts = Counter(int(value) for value in labels)
    maximum = max(counts.values())
    winners = sorted(identity for identity, count in counts.items() if count == maximum)
    tied = len(winners) != 1
    support = 0 if tied else int(winners[0])
    return support, maximum, float(maximum / len(labels)), tied


def build_tracklet_segments(m23, sequence: str, baseline_path: Path, gt_path: Path):
    baseline = m23.load_baseline(baseline_path)
    gt = m23.load_gt(gt_path)
    frames_by_track: MutableMapping[int, List[int]] = defaultdict(list)
    matched_gt: Dict[Tuple[int, int], int] = {}
    matched_iou: Dict[Tuple[int, int], float] = {}

    for frame in sorted(baseline):
        rows = baseline[frame]
        for candidate in rows:
            frames_by_track[int(candidate.original_id)].append(int(frame))
        kept, valid_gt, _ = m23.valid_and_distractor_filtered(rows, gt.get(frame, []))
        for candidate_index, gt_index, overlap in m23.match_candidates(kept, valid_gt):
            track_id = int(kept[candidate_index].original_id)
            matched_gt[(int(frame), track_id)] = int(valid_gt[gt_index].gt_id)
            matched_iou[(int(frame), track_id)] = float(overlap)

    frame_to_segment: Dict[Tuple[int, int], int] = {}
    segment_support: Dict[int, int] = {}
    segment_rows: List[dict] = []
    segment_id = 0
    for track_id in sorted(frames_by_track):
        frames = sorted(set(frames_by_track[track_id]))
        groups: List[List[int]] = []
        current: List[int] = []
        for frame in frames:
            if current and frame > current[-1] + 1:
                groups.append(current)
                current = []
            current.append(frame)
        if current:
            groups.append(current)

        for local_segment_index, group in enumerate(groups):
            labels = [matched_gt[(frame, track_id)] for frame in group if (frame, track_id) in matched_gt]
            overlaps = [matched_iou[(frame, track_id)] for frame in group if (frame, track_id) in matched_iou]
            support, modal_count, purity, tied = modal_support(labels)
            for frame in group:
                frame_to_segment[(track_id, frame)] = segment_id
            segment_support[segment_id] = support
            segment_rows.append({
                "sequence": sequence,
                "segment_id": segment_id,
                "baseline_track_id": track_id,
                "local_segment_index": local_segment_index,
                "first_frame": group[0],
                "last_frame": group[-1],
                "frames": len(group),
                "matched_frames": len(labels),
                "unique_matched_gt": len(set(labels)),
                "modal_gt_id": support,
                "modal_count": modal_count,
                "modal_purity": purity,
                "modal_tied": tied,
                "mean_match_iou": float(np.mean(overlaps)) if overlaps else 0.0,
            })
            segment_id += 1
    return frame_to_segment, segment_support, segment_rows


def event_tracker_line(row: Mapping[str, str], candidate_row: np.void) -> str:
    x1 = float(candidate_row["candidate_x1"])
    y1 = float(candidate_row["candidate_y1"])
    x2 = float(candidate_row["candidate_x2"])
    y2 = float(candidate_row["candidate_y2"])
    fields = [
        str(int(row["frame"])),
        str(int(row["gt_id"])),
        f"{x1:.6f}",
        f"{y1:.6f}",
        f"{x2 - x1:.6f}",
        f"{y2 - y1:.6f}",
        f"{float(candidate_row['candidate_score']):.6f}",
        "-1", "-1", "-1",
    ]
    return ",".join(fields)


def apply_replacements(
    full_lines: Sequence[str],
    replacement_by_line: Mapping[str, str | None],
    *,
    label: str,
) -> List[str]:
    seen_replacement_keys = set(replacement_by_line)
    output: List[str] = []
    replaced = 0
    for line in full_lines:
        if line not in replacement_by_line:
            output.append(line)
            continue
        replaced += 1
        replacement = replacement_by_line[line]
        if replacement is not None:
            output.append(replacement)
    if replaced != len(seen_replacement_keys):
        missing = sorted(seen_replacement_keys - set(full_lines))[:5]
        raise RuntimeError(f"{label}: replaced {replaced}/{len(seen_replacement_keys)} lines; missing={missing}")
    validate_tracker(output, label)
    return output


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
    parser.add_argument("--oracle-events", required=True)
    parser.add_argument("--m23-1-dir", required=True)
    parser.add_argument("--preregister", required=True)
    parser.add_argument(
        "--baseline-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data",
    )
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    appearance_dir = resolve(repo, args.appearance_dir)
    compact_dir = resolve(repo, args.compact_dir)
    oracle_path = resolve(repo, args.oracle_events)
    m23_1_dir = resolve(repo, args.m23_1_dir)
    prereg_path = resolve(repo, args.preregister)
    baseline_dir = resolve(repo, args.baseline_dir)
    gt_root = resolve(repo, args.gt_root)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    if prereg.get("script_sha256") != sha256_file(script_path):
        raise RuntimeError("preregistered script hash mismatch")
    if int(prereg["protocol"]["spawn_episode_max_gap"]) != SPAWN_EPISODE_MAX_GAP:
        raise RuntimeError("preregistered spawn gap mismatch")
    if int(prereg["protocol"]["synthetic_spawn_id_base"]) != SYNTHETIC_SPAWN_ID_BASE:
        raise RuntimeError("preregistered synthetic spawn namespace mismatch")
    if tuple(prereg["protocol"]["variants"]) != VARIANTS:
        raise RuntimeError("preregistered variants mismatch")

    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    candidate_keys_path = compact_dir / "candidate_keys.npy"
    m23_1_report_path = m23_1_dir / "report.json"
    m23_1_manifest_path = m23_1_dir / "manifest.json"
    for label, path in (
        ("candidate_manifest", candidate_manifest_path),
        ("candidate_keys", candidate_keys_path),
        ("oracle_events", oracle_path),
        ("m23_1_report", m23_1_report_path),
        ("m23_1_manifest", m23_1_manifest_path),
    ):
        expected = prereg["sources"][label]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")

    m23 = load_module(repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py", "m23_for_m23_3")
    keys = np.load(candidate_keys_path, allow_pickle=False, mmap_mode="r")
    candidates = np.load(candidate_manifest_path, allow_pickle=False, mmap_mode="r")
    if len(keys) != len(candidates):
        raise RuntimeError("candidate key and manifest row counts differ")
    key_lookup = {
        (int(sequence_id), int(global_pre_idx)): row_index
        for row_index, (sequence_id, global_pre_idx) in enumerate(
            zip(keys["sequence_id"], keys["global_pre_idx"])
        )
    }
    if len(key_lookup) != len(keys):
        raise RuntimeError("candidate keys are not unique")

    all_segment_rows: List[dict] = []
    action_events: Dict[str, List[dict]] = {sequence: [] for sequence in SEQUENCES}
    tracker_source_hashes: Dict[str, dict] = {}
    tracker_lines: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    tracker_maps: Dict[str, Dict[str, Dict[Tuple[int, int], str]]] = defaultdict(dict)
    tracker_line_sets: Dict[str, Dict[str, set[str]]] = defaultdict(dict)
    frame_segment_by_sequence: Dict[str, Dict[Tuple[int, int], int]] = {}
    segment_support_by_sequence: Dict[str, Dict[int, int]] = {}

    for sequence in SEQUENCES:
        baseline_path = baseline_dir / f"{sequence}.txt"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        if sha256_file(baseline_path) != prereg["sources"]["baseline"][sequence]:
            raise RuntimeError(f"{sequence}: baseline hash mismatch")
        if sha256_file(gt_path) != prereg["sources"]["gt"][sequence]:
            raise RuntimeError(f"{sequence}: GT hash mismatch")
        frame_segment, segment_support, segment_rows = build_tracklet_segments(
            m23,
            sequence,
            baseline_path,
            gt_path,
        )
        frame_segment_by_sequence[sequence] = frame_segment
        segment_support_by_sequence[sequence] = segment_support
        all_segment_rows.extend(segment_rows)

        paths = {
            "context": m23_1_dir / "eval_work" / "trackers" / "postnms_replace_add_oracle" / "data" / f"{sequence}.txt",
            "full": m23_1_dir / "eval_work" / "trackers" / "prenms_budget_replace_add_oracle" / "data" / f"{sequence}.txt",
        }
        tracker_source_hashes[sequence] = {}
        for label, path in paths.items():
            expected = prereg["sources"][f"{label}_trackers"][sequence]
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"{sequence}/{label} tracker hash mismatch")
            lines = path.read_text(encoding="utf-8").splitlines()
            validate_tracker(lines, f"{sequence}/{label}")
            tracker_lines[label][sequence] = lines
            tracker_maps[label][sequence] = tracker_frame_identity_map(lines, f"{sequence}/{label}")
            tracker_line_sets[label][sequence] = set(lines)
            tracker_source_hashes[sequence][label] = {"path": str(path), "sha256": actual, "rows": len(lines)}

    with oracle_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "prenms_budget_replace_add_oracle" or row["source"] != "prenms_suppressed":
                continue
            sequence = row["sequence"]
            sequence_id = SEQUENCE_TO_ID[sequence]
            frame = int(row["frame"])
            target_gt_id = int(row["gt_id"])
            global_pre_idx = int(row["global_pre_idx"])
            key = (sequence_id, global_pre_idx)
            if key not in key_lookup:
                raise RuntimeError(f"oracle event key outside frozen budget: {key}")
            candidate_row_index = key_lookup[key]
            candidate_row = candidates[candidate_row_index]
            if int(candidate_row["frame"]) != frame:
                raise RuntimeError(f"event frame mismatch for {sequence}/{global_pre_idx}")
            full_line = event_tracker_line(row, candidate_row)
            if full_line not in tracker_line_sets["full"][sequence]:
                raise RuntimeError(f"event line not found exactly in full tracker: {sequence}/{global_pre_idx}")

            candidate_track_id = int(candidate_row["candidate_track_id"])
            suppressor_track_id = int(candidate_row["suppressor_track_id"])
            candidate_segment_id = frame_segment_by_sequence[sequence].get((candidate_track_id, frame), -1)
            suppressor_segment_id = frame_segment_by_sequence[sequence].get((suppressor_track_id, frame), -1)
            candidate_support_gt = segment_support_by_sequence[sequence].get(candidate_segment_id, 0)
            suppressor_support_gt = segment_support_by_sequence[sequence].get(suppressor_segment_id, 0)
            if candidate_support_gt == target_gt_id:
                action = "candidate_default"
            elif suppressor_support_gt == target_gt_id:
                action = "suppressor_override"
            else:
                action = "spawn"
            context_line = tracker_maps["context"][sequence].get((frame, target_gt_id))
            action_events[sequence].append({
                "sequence": sequence,
                "frame": frame,
                "target_gt_id": target_gt_id,
                "global_pre_idx": global_pre_idx,
                "candidate_row_index": candidate_row_index,
                "candidate_track_id": candidate_track_id,
                "candidate_segment_id": candidate_segment_id,
                "candidate_support_gt": candidate_support_gt,
                "suppressor_track_id": suppressor_track_id,
                "suppressor_segment_id": suppressor_segment_id,
                "suppressor_support_gt": suppressor_support_gt,
                "action": action,
                "context_available": context_line is not None,
                "increment_type": "replacement" if context_line is not None else "addition",
                "full_line": full_line,
                "context_line": context_line,
                "spawn_episode_id": -1,
                "spawn_synthetic_id": -1,
            })

    spawn_episode_rows: List[dict] = []
    spawn_episode_counter = 0
    for sequence in SEQUENCES:
        grouped: MutableMapping[int, List[dict]] = defaultdict(list)
        for event in action_events[sequence]:
            if event["action"] == "spawn":
                grouped[int(event["target_gt_id"])].append(event)
        for target_gt_id in sorted(grouped):
            events = sorted(grouped[target_gt_id], key=lambda item: (int(item["frame"]), int(item["global_pre_idx"])))
            episodes: List[List[dict]] = []
            current: List[dict] = []
            for event in events:
                if current and int(event["frame"]) - int(current[-1]["frame"]) > SPAWN_EPISODE_MAX_GAP:
                    episodes.append(current)
                    current = []
                current.append(event)
            if current:
                episodes.append(current)
            for local_episode_index, episode in enumerate(episodes):
                synthetic_id = (
                    SYNTHETIC_SPAWN_ID_BASE
                    + SEQUENCE_TO_ID[sequence] * SYNTHETIC_SEQUENCE_STRIDE
                    + spawn_episode_counter
                )
                if synthetic_id >= MAX_EXACT_FLOAT32_INTEGER:
                    raise RuntimeError(f"synthetic tracker ID is not exactly representable in float32: {synthetic_id}")
                for event in episode:
                    event["spawn_episode_id"] = spawn_episode_counter
                    event["spawn_synthetic_id"] = synthetic_id
                frames = sorted({int(event["frame"]) for event in episode})
                spawn_episode_rows.append({
                    "sequence": sequence,
                    "target_gt_id": target_gt_id,
                    "local_episode_index": local_episode_index,
                    "spawn_episode_id": spawn_episode_counter,
                    "synthetic_tracker_id": synthetic_id,
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "span_frames": frames[-1] - frames[0] + 1,
                    "observed_frames": len(frames),
                    "events": len(episode),
                    "replacement_events": sum(event["increment_type"] == "replacement" for event in episode),
                    "addition_events": sum(event["increment_type"] == "addition" for event in episode),
                })
                spawn_episode_counter += 1

    eval_work = output_dir / "eval_work"
    source_tracker_ids = {
        identity
        for label in ("context", "full")
        for sequence in SEQUENCES
        for _, identity, _ in map(parse_tracker_line, tracker_lines[label][sequence])
    }
    synthetic_ids = {int(row["synthetic_tracker_id"]) for row in spawn_episode_rows}
    collisions = sorted(source_tracker_ids & synthetic_ids)
    if collisions:
        raise RuntimeError(f"synthetic tracker namespace collides with source IDs: {collisions[:10]}")
    tracker_hashes: Dict[str, Dict[str, str]] = {variant: {} for variant in VARIANTS}
    action_rows: List[dict] = []
    reason_rows: List[dict] = []
    for sequence in SEQUENCES:
        events = action_events[sequence]
        full_lines = tracker_lines["full"][sequence]
        context_lines = tracker_lines["context"][sequence]
        full_line_set = set(full_lines)
        if len(full_line_set) != len(full_lines):
            raise RuntimeError(f"{sequence}: duplicate full tracker lines")
        if len({event["full_line"] for event in events}) != len(events):
            raise RuntimeError(f"{sequence}: duplicate event tracker lines")

        replacements_candidate: Dict[str, str | None] = {}
        replacements_candidate_suppressor: Dict[str, str | None] = {}
        replacements_episode: Dict[str, str | None] = {}
        for event in events:
            full_line = str(event["full_line"])
            fallback = event["context_line"]
            replacements_candidate[full_line] = full_line if event["action"] == "candidate_default" else fallback
            replacements_candidate_suppressor[full_line] = (
                full_line if event["action"] in ("candidate_default", "suppressor_override") else fallback
            )
            replacements_episode[full_line] = (
                full_line
                if event["action"] in ("candidate_default", "suppressor_override")
                else replace_tracker_identity(full_line, int(event["spawn_synthetic_id"]))
            )

        variants = {
            "postnms_context": list(context_lines),
            "candidate_default_safe": apply_replacements(
                full_lines, replacements_candidate, label=f"{sequence}/candidate_default_safe"
            ),
            "candidate_suppressor_safe": apply_replacements(
                full_lines, replacements_candidate_suppressor, label=f"{sequence}/candidate_suppressor_safe"
            ),
            "candidate_suppressor_episode_spawn": apply_replacements(
                full_lines, replacements_episode, label=f"{sequence}/candidate_suppressor_episode_spawn"
            ),
            "oracle_linked_spawn": list(full_lines),
        }
        for variant, lines in variants.items():
            validate_tracker(lines, f"{sequence}/{variant}")
            path = eval_work / "trackers" / variant / "data" / f"{sequence}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tracker_hashes[variant][sequence] = sha256_file(path)

        total = len(events)
        counts = Counter(str(event["action"]) for event in events)
        for action in ACTION_PRIORITY:
            action_rows.append({
                "sequence": sequence,
                "action": action,
                "events": counts[action],
                "fraction": float(counts[action] / max(1, total)),
                "replacement_events": sum(
                    event["action"] == action and event["increment_type"] == "replacement" for event in events
                ),
                "addition_events": sum(
                    event["action"] == action and event["increment_type"] == "addition" for event in events
                ),
            })
        reason_counter = Counter()
        for event in events:
            candidate_reason = (
                "candidate_support_target" if event["candidate_support_gt"] == event["target_gt_id"]
                else "candidate_segment_missing" if event["candidate_segment_id"] < 0
                else "candidate_segment_unlabeled" if event["candidate_support_gt"] == 0
                else "candidate_support_other_gt"
            )
            suppressor_reason = (
                "suppressor_support_target" if event["suppressor_support_gt"] == event["target_gt_id"]
                else "suppressor_segment_missing" if event["suppressor_segment_id"] < 0
                else "suppressor_segment_unlabeled" if event["suppressor_support_gt"] == 0
                else "suppressor_support_other_gt"
            )
            reason_counter[(event["action"], candidate_reason, suppressor_reason)] += 1
        for (action, candidate_reason, suppressor_reason), count in sorted(reason_counter.items()):
            reason_rows.append({
                "sequence": sequence,
                "action": action,
                "candidate_reason": candidate_reason,
                "suppressor_reason": suppressor_reason,
                "events": count,
            })

    combined_events = [event for sequence in SEQUENCES for event in action_events[sequence]]
    combined_counts = Counter(str(event["action"]) for event in combined_events)
    for action in ACTION_PRIORITY:
        action_rows.append({
            "sequence": "COMBINED",
            "action": action,
            "events": combined_counts[action],
            "fraction": float(combined_counts[action] / max(1, len(combined_events))),
            "replacement_events": sum(
                event["action"] == action and event["increment_type"] == "replacement"
                for event in combined_events
            ),
            "addition_events": sum(
                event["action"] == action and event["increment_type"] == "addition"
                for event in combined_events
            ),
        })

    # Source tracker reproduction is checked before running TrackEval.
    for sequence in SEQUENCES:
        if tracker_hashes["postnms_context"][sequence] != tracker_source_hashes[sequence]["context"]["sha256"]:
            raise RuntimeError(f"{sequence}: post-NMS context tracker was not reproduced")
        if tracker_hashes["oracle_linked_spawn"][sequence] != tracker_source_hashes[sequence]["full"]["sha256"]:
            raise RuntimeError(f"{sequence}: full M23-1 tracker was not reproduced")

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
    gates = prereg["acceptance"]
    full_reference = prereg["references"]["m23_1_budget_replace_add"]
    full_reproduction_delta = {
        metric: float(by_variant["oracle_linked_spawn"][metric] - full_reference[metric])
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    }
    full_reproduced = all(
        abs(value) <= float(gates["oracle_linked_reproduction_tolerance"])
        for value in full_reproduction_delta.values()
    )
    candidate_suppressor_sequence_deltas = {
        sequence: float(
            by_sequence_variant[(sequence, "candidate_suppressor_safe")]["HOTA"]
            - by_sequence_variant[(sequence, "postnms_context")]["HOTA"]
        )
        for sequence in SEQUENCES
    }
    episode_sequence_hota = {
        sequence: float(by_sequence_variant[(sequence, "candidate_suppressor_episode_spawn")]["HOTA"])
        for sequence in SEQUENCES
    }
    candidate_suppressor_sequence_hota = {
        sequence: float(by_sequence_variant[(sequence, "candidate_suppressor_safe")]["HOTA"])
        for sequence in SEQUENCES
    }
    decision = {
        "oracle_linked_spawn_reproduced": full_reproduced,
        "oracle_linked_spawn_metric_deltas": full_reproduction_delta,
        "candidate_suppressor_combined_hota": float(by_variant["candidate_suppressor_safe"]["HOTA"]),
        "candidate_suppressor_combined_hota_passed": bool(
            by_variant["candidate_suppressor_safe"]["HOTA"] >= gates["minimum_candidate_suppressor_combined_hota"]
        ),
        "candidate_suppressor_worst_sequence_hota": min(candidate_suppressor_sequence_hota.values()),
        "candidate_suppressor_worst_sequence_hota_passed": bool(
            min(candidate_suppressor_sequence_hota.values()) >= gates["minimum_candidate_suppressor_worst_sequence_hota"]
        ),
        "candidate_suppressor_all_sequence_increment_nonnegative": bool(
            min(candidate_suppressor_sequence_deltas.values()) >= 0.0
        ),
        "candidate_suppressor_sequence_hota_delta_vs_postnms": candidate_suppressor_sequence_deltas,
        "episode_spawn_combined_hota": float(by_variant["candidate_suppressor_episode_spawn"]["HOTA"]),
        "episode_spawn_combined_hota_passed": bool(
            by_variant["candidate_suppressor_episode_spawn"]["HOTA"] >= gates["minimum_episode_spawn_combined_hota"]
        ),
        "episode_spawn_worst_sequence_hota": min(episode_sequence_hota.values()),
        "episode_spawn_worst_sequence_hota_passed": bool(
            min(episode_sequence_hota.values()) >= gates["minimum_episode_spawn_worst_sequence_hota"]
        ),
        "candidate_default_fraction": float(combined_counts["candidate_default"] / len(combined_events)),
        "suppressor_override_fraction": float(combined_counts["suppressor_override"] / len(combined_events)),
        "spawn_fraction": float(combined_counts["spawn"] / len(combined_events)),
        "direct_candidate_suppressor_graph_sufficient": False,
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "next_stage": "set below after gate evaluation",
    }
    decision["direct_candidate_suppressor_graph_sufficient"] = bool(
        decision["oracle_linked_spawn_reproduced"]
        and decision["candidate_suppressor_combined_hota_passed"]
        and decision["candidate_suppressor_worst_sequence_hota_passed"]
        and decision["candidate_suppressor_all_sequence_increment_nonnegative"]
    )
    if decision["direct_candidate_suppressor_graph_sufficient"]:
        decision["next_stage"] = "cross-fitted candidate-default graph scoring with spawn linkage as a separate abstaining module"
    elif decision["episode_spawn_combined_hota_passed"] and decision["episode_spawn_worst_sequence_hota_passed"]:
        decision["next_stage"] = "learn candidate-default graph plus explicit spawn-episode linkage; do not train suppressor override alone"
    else:
        decision["next_stage"] = "expand graph evidence with short-gap propagation before learning global decisions"

    write_csv(
        output_dir / "tracklet_segments.csv",
        all_segment_rows,
        [
            "sequence", "segment_id", "baseline_track_id", "local_segment_index", "first_frame", "last_frame",
            "frames", "matched_frames", "unique_matched_gt", "modal_gt_id", "modal_count", "modal_purity",
            "modal_tied", "mean_match_iou",
        ],
    )
    write_csv(
        output_dir / "action_summary.csv",
        action_rows,
        ["sequence", "action", "events", "fraction", "replacement_events", "addition_events"],
    )
    write_csv(
        output_dir / "action_reason_summary.csv",
        reason_rows,
        ["sequence", "action", "candidate_reason", "suppressor_reason", "events"],
    )
    write_csv(
        output_dir / "spawn_episodes.csv",
        spawn_episode_rows,
        [
            "sequence", "target_gt_id", "local_episode_index", "spawn_episode_id", "synthetic_tracker_id",
            "first_frame", "last_frame", "span_frames", "observed_frames", "events", "replacement_events",
            "addition_events",
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
        "schema": "fmtrack.m23_3.candidate_default_action_space.v1",
        "protocol": {
            "audit_scope": "MOT20-train oracle action-space audit under fixed post-NMS oracle context",
            "segment_definition": "baseline original track ID split whenever consecutive observed frames have gap > 1",
            "segment_support": "unique modal GT identity among same-frame IoU>=0.5 Hungarian matches; ties are unlabeled",
            "action_priority": list(ACTION_PRIORITY),
            "spawn_episode_max_gap": SPAWN_EPISODE_MAX_GAP,
            "synthetic_spawn_id_base": SYNTHETIC_SPAWN_ID_BASE,
            "synthetic_sequence_stride": SYNTHETIC_SEQUENCE_STRIDE,
            "synthetic_ids_below_exact_float32_limit": True,
            "spawn_episode_grouping": "same sequence and oracle target GT; new episode when frame gap > 30",
            "safe_fallback": "restore post-NMS oracle-context target row when present, otherwise abstain",
            "models_trained": 0,
            "parameter_sweeps": 0,
            "threshold_sweeps": 0,
            "mot20_train_trackeval_invocations": 1,
            "locked_trackeval_calls": 0,
            "locked_label_reads": 0,
        },
        "counts": {
            "candidate_rows": len(candidates),
            "selected_suppressed_events": len(combined_events),
            "tracklet_segments": len(all_segment_rows),
            "segments_with_support": sum(int(row["modal_gt_id"]) > 0 for row in all_segment_rows),
            "segments_with_tied_mode": sum(bool(row["modal_tied"]) for row in all_segment_rows),
            "spawn_episodes": len(spawn_episode_rows),
            "actions": dict(combined_counts),
        },
        "acceptance": gates,
        "variant_metrics": by_variant,
        "decision": decision,
        "sources": {
            "preregister": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
            "candidate_manifest": {"path": str(candidate_manifest_path), "sha256": sha256_file(candidate_manifest_path)},
            "candidate_keys": {"path": str(candidate_keys_path), "sha256": sha256_file(candidate_keys_path)},
            "oracle_events": {"path": str(oracle_path), "sha256": sha256_file(oracle_path)},
            "m23_1_report": {"path": str(m23_1_report_path), "sha256": sha256_file(m23_1_report_path)},
            "m23_1_manifest": {"path": str(m23_1_manifest_path), "sha256": sha256_file(m23_1_manifest_path)},
            "tracker_sources": tracker_source_hashes,
        },
        "tracker_hashes": tracker_hashes,
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    compact_files = (
        "tracklet_segments.csv",
        "action_summary.csv",
        "action_reason_summary.csv",
        "spawn_episodes.csv",
        "variant_metrics.csv",
        "per_sequence_metrics.csv",
        "report.json",
    )
    manifest = {
        "schema": "fmtrack.m23_3.candidate_default_action_space.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in compact_files},
        "tracker_hashes": tracker_hashes,
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
