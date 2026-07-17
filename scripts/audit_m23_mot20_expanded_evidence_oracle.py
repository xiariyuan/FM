"""M23-0 MOT20 post-NMS low-confidence expanded-evidence oracle audit.

This diagnostic answers whether the existing low-confidence YOLOX Phase-0 pool
can raise the ceiling above the current tracker-output ID oracle. It does not
train or deploy a model. Ground truth is used only to construct explicit oracle
variants.

Fixed protocol:
- MOT20 train sequences 01/02/03/05.
- Baseline rows are the frozen all4 baseline inputs from the TrustTrack audit.
- Expanded candidates are Phase-0 YOLOX postprocess outputs with score >= 0.09.
- A Phase-0 candidate is geometrically novel when its maximum IoU with any
  baseline row in the same frame is < 0.90.
- MOTChallenge distractor preprocessing and valid-pedestrian matching use IoU
  0.50 and threshold-before-Hungarian assignment.
- No threshold or variant is selected from results.

Oracle variants:
1. baseline_raw: unchanged baseline tracker output.
2. baseline_id_oracle: preserve every baseline row/box; GT-reassign IDs only.
3. expanded_additive_oracle: preserve baseline rows and add only novel Phase-0
   candidates that recover valid GT missed by baseline at IoU 0.50.
4. expanded_replace_add_oracle: preserve baseline unmatched/FP burden, replace
   baseline GT-matched rows with the best expanded candidate, and add newly
   recovered GT.
5. expanded_selected_ceiling: retain only the best expanded-pool candidate for
   each matched valid GT. This is an optimistic pool ceiling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
VARIANTS = (
    "baseline_raw",
    "baseline_id_oracle",
    "expanded_additive_oracle",
    "expanded_replace_add_oracle",
    "expanded_selected_ceiling",
)
DISTRACTOR_CLASSES_MOT20 = {2, 6, 7, 8, 12}
VALID_PEDESTRIAN_CLASS = 1
GT_MATCH_IOU = 0.50
PREPROC_IOU = 0.50
NOVELTY_IOU = 0.90
UNMATCHED_OFFSET = 1_000_000
PHASE0_UID_OFFSET = 10_000_000

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Candidate:
    frame: int
    source: str
    uid: int
    original_id: int
    box: Box
    score: float
    tail: Tuple[str, ...] = ("-1", "-1", "-1")


@dataclass(frozen=True)
class GTObject:
    gt_id: int
    box: Box
    marked: int
    cls: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Box:
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box: Box) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    xx1 = np.maximum(aa[:, None, 0], bb[None, :, 0])
    yy1 = np.maximum(aa[:, None, 1], bb[None, :, 1])
    xx2 = np.minimum(aa[:, None, 2], bb[None, :, 2])
    yy2 = np.minimum(aa[:, None, 3], bb[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, aa[:, 2] - aa[:, 0]) * np.maximum(0.0, aa[:, 3] - aa[:, 1])
    area_b = np.maximum(0.0, bb[:, 2] - bb[:, 0]) * np.maximum(0.0, bb[:, 3] - bb[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-12))


def thresholded_hungarian(similarities: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    if similarities.size == 0:
        return []
    eps = np.finfo(float).eps
    valid = similarities + eps >= float(threshold)
    scores = np.where(valid, similarities, 0.0)
    rows, cols = linear_sum_assignment(-scores)
    return [
        (int(row), int(col), float(similarities[row, col]))
        for row, col in zip(rows.tolist(), cols.tolist())
        if bool(valid[row, col])
    ]


def load_gt(path: Path) -> Dict[int, List[GTObject]]:
    by_frame: Dict[int, List[GTObject]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected >=6 columns")
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            marked = int(float(parts[6])) if len(parts) > 6 else 1
            cls = int(float(parts[7])) if len(parts) > 7 else 1
            by_frame[frame].append(GTObject(gt_id, xywh_to_xyxy(x, y, w, h), marked, cls))
    return by_frame


def load_baseline(path: Path) -> Dict[int, List[Candidate]]:
    by_frame: Dict[int, List[Candidate]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected >=6 columns")
            frame = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            tail = tuple(parts[7:]) if len(parts) > 7 else ("-1", "-1", "-1")
            uid = (frame << 32) + (line_no - 1)
            by_frame[frame].append(
                Candidate(frame, "baseline", uid, track_id, xywh_to_xyxy(x, y, w, h), score, tail)
            )
    return by_frame


def read_npz_member(path: Path, member: str, *, allow_pickle: bool = False) -> Tuple[np.ndarray, str]:
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle), sha256_bytes(data)


def load_phase0(path: Path) -> Tuple[Dict[int, List[Candidate]], dict]:
    detections, detections_hash = read_npz_member(path, "detections.npy")
    columns, columns_hash = read_npz_member(path, "columns.npy", allow_pickle=True)
    names = [str(value) for value in columns.tolist()]
    index = {name: idx for idx, name in enumerate(names)}
    required = {"frame", "global_det_idx", "x1", "y1", "x2", "y2", "score"}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    by_frame: Dict[int, List[Candidate]] = defaultdict(list)
    for row in detections:
        frame = int(row[index["frame"]])
        global_id = int(row[index["global_det_idx"]])
        score = float(row[index["score"]])
        box = tuple(float(row[index[name]]) for name in ("x1", "y1", "x2", "y2"))
        by_frame[frame].append(
            Candidate(frame, "phase0_postnms", PHASE0_UID_OFFSET + global_id, global_id, box, score)
        )
    metadata = {
        "npz_path": str(path),
        "npz_size_bytes": path.stat().st_size,
        "detections_member_sha256": detections_hash,
        "columns_member_sha256": columns_hash,
        "columns": names,
        "detections": int(detections.shape[0]),
        "score_min": float(np.min(detections[:, index["score"]])) if detections.size else None,
        "score_max": float(np.max(detections[:, index["score"]])) if detections.size else None,
    }
    return by_frame, metadata


def valid_and_distractor_filtered(
    candidates: Sequence[Candidate], gt_rows: Sequence[GTObject]
) -> Tuple[List[Candidate], List[GTObject], set[int]]:
    all_gt_boxes = [item.box for item in gt_rows]
    candidate_boxes = [item.box for item in candidates]
    preproc_matches = thresholded_hungarian(iou_matrix(all_gt_boxes, candidate_boxes), PREPROC_IOU)
    removed = {
        candidate_index
        for gt_index, candidate_index, _ in preproc_matches
        if gt_rows[gt_index].cls in DISTRACTOR_CLASSES_MOT20
    }
    kept = [candidate for index, candidate in enumerate(candidates) if index not in removed]
    valid_gt = [
        item for item in gt_rows
        if item.marked != 0 and item.cls == VALID_PEDESTRIAN_CLASS
    ]
    return kept, valid_gt, removed


def match_candidates(
    candidates: Sequence[Candidate], valid_gt: Sequence[GTObject]
) -> List[Tuple[int, int, float]]:
    return thresholded_hungarian(
        iou_matrix([candidate.box for candidate in candidates], [item.box for item in valid_gt]),
        GT_MATCH_IOU,
    )


def novel_phase0(
    phase0: Sequence[Candidate], baseline: Sequence[Candidate]
) -> Tuple[List[Candidate], np.ndarray]:
    if not phase0:
        return [], np.zeros((0,), dtype=np.float64)
    if not baseline:
        return list(phase0), np.zeros((len(phase0),), dtype=np.float64)
    overlaps = iou_matrix([item.box for item in phase0], [item.box for item in baseline])
    maximum = overlaps.max(axis=1) if overlaps.shape[1] else np.zeros((len(phase0),), dtype=np.float64)
    return [item for item, value in zip(phase0, maximum.tolist()) if value < NOVELTY_IOU], maximum


def unmatched_oracle_id(candidate: Candidate) -> int:
    if candidate.source == "baseline":
        return UNMATCHED_OFFSET + int(candidate.original_id)
    return UNMATCHED_OFFSET + int(candidate.uid)


def format_candidate(candidate: Candidate, oracle_id: int) -> str:
    x, y, width, height = xyxy_to_xywh(candidate.box)
    fields = [
        str(candidate.frame),
        str(int(oracle_id)),
        f"{x:.6f}",
        f"{y:.6f}",
        f"{width:.6f}",
        f"{height:.6f}",
        f"{candidate.score:.6f}",
    ]
    fields.extend(candidate.tail if candidate.tail else ("-1", "-1", "-1"))
    return ",".join(fields)


def score_bin(score: float) -> str:
    if score >= 0.60:
        return "high_ge_0p60"
    if score >= 0.10:
        return "low_0p10_0p60"
    return "fringe_0p09_0p10"


def build_sequence_variants(
    sequence: str,
    baseline_by_frame: Mapping[int, List[Candidate]],
    phase0_by_frame: Mapping[int, List[Candidate]],
    gt_by_frame: Mapping[int, List[GTObject]],
) -> Tuple[Dict[str, List[str]], dict, List[dict]]:
    outputs: Dict[str, List[str]] = {variant: [] for variant in VARIANTS}
    stats = defaultdict(int)
    selected_rows: List[dict] = []
    frames = sorted(set(baseline_by_frame) | set(phase0_by_frame) | set(gt_by_frame))

    for frame in frames:
        baseline = list(baseline_by_frame.get(frame, []))
        phase0 = list(phase0_by_frame.get(frame, []))
        gt_rows = list(gt_by_frame.get(frame, []))
        stats["frames"] += 1
        stats["baseline_rows"] += len(baseline)
        stats["phase0_rows"] += len(phase0)

        for candidate in baseline:
            outputs["baseline_raw"].append(format_candidate(candidate, candidate.original_id))

        base_kept, valid_gt, base_removed = valid_and_distractor_filtered(baseline, gt_rows)
        base_matches = match_candidates(base_kept, valid_gt)
        base_match_gt_ids = {valid_gt[gt_index].gt_id for _, gt_index, _ in base_matches}
        base_matched_uids = {base_kept[candidate_index].uid for candidate_index, _, _ in base_matches}
        base_gt_to_uid = {
            valid_gt[gt_index].gt_id: base_kept[candidate_index].uid
            for candidate_index, gt_index, _ in base_matches
        }
        stats["valid_gt"] += len(valid_gt)
        stats["baseline_matched_gt"] += len(base_matches)
        stats["baseline_preproc_removed"] += len(base_removed)

        base_gt_id_by_uid = {
            base_kept[candidate_index].uid: valid_gt[gt_index].gt_id
            for candidate_index, gt_index, _ in base_matches
        }
        for candidate in baseline:
            outputs["baseline_id_oracle"].append(
                format_candidate(
                    candidate,
                    base_gt_id_by_uid.get(candidate.uid, unmatched_oracle_id(candidate)),
                )
            )

        novel, max_iou_to_base = novel_phase0(phase0, baseline)
        stats["phase0_novel_rows"] += len(novel)
        stats["phase0_duplicate_rows_iou_ge_0p90"] += len(phase0) - len(novel)
        for maximum in max_iou_to_base.tolist():
            if maximum >= 0.90:
                stats["phase0_duplicate_ge_0p90"] += 1
            if maximum >= 0.95:
                stats["phase0_duplicate_ge_0p95"] += 1
            if maximum >= 0.99:
                stats["phase0_duplicate_ge_0p99"] += 1

        phase_kept, _, phase_removed = valid_and_distractor_filtered(novel, gt_rows)
        stats["phase0_novel_preproc_removed"] += len(phase_removed)

        missed_gt = [item for item in valid_gt if item.gt_id not in base_match_gt_ids]
        additive_matches = match_candidates(phase_kept, missed_gt)
        for candidate in baseline:
            outputs["expanded_additive_oracle"].append(
                format_candidate(
                    candidate,
                    base_gt_id_by_uid.get(candidate.uid, unmatched_oracle_id(candidate)),
                )
            )
        for candidate_index, gt_index, overlap in additive_matches:
            candidate = phase_kept[candidate_index]
            gt_item = missed_gt[gt_index]
            outputs["expanded_additive_oracle"].append(format_candidate(candidate, gt_item.gt_id))
            stats["additive_recovered_gt"] += 1
            stats[f"additive_{score_bin(candidate.score)}"] += 1
            selected_rows.append({
                "sequence": sequence,
                "frame": frame,
                "variant": "expanded_additive_oracle",
                "gt_id": gt_item.gt_id,
                "source": candidate.source,
                "source_uid": candidate.uid,
                "score": round(candidate.score, 12),
                "iou": round(overlap, 12),
                "operation": "add_missing",
                "score_bin": score_bin(candidate.score),
            })

        combined = base_kept + phase_kept
        combined_matches = match_candidates(combined, valid_gt)
        for candidate in baseline:
            if candidate.uid not in base_matched_uids:
                outputs["expanded_replace_add_oracle"].append(
                    format_candidate(candidate, unmatched_oracle_id(candidate))
                )
        for candidate_index, gt_index, overlap in combined_matches:
            candidate = combined[candidate_index]
            gt_item = valid_gt[gt_index]
            outputs["expanded_replace_add_oracle"].append(format_candidate(candidate, gt_item.gt_id))
            baseline_uid = base_gt_to_uid.get(gt_item.gt_id)
            if candidate.source == "phase0_postnms":
                operation = "replace_baseline" if baseline_uid is not None else "add_missing"
                stats[f"replace_add_{operation}"] += 1
                stats[f"replace_add_{score_bin(candidate.score)}"] += 1
                selected_rows.append({
                    "sequence": sequence,
                    "frame": frame,
                    "variant": "expanded_replace_add_oracle",
                    "gt_id": gt_item.gt_id,
                    "source": candidate.source,
                    "source_uid": candidate.uid,
                    "score": round(candidate.score, 12),
                    "iou": round(overlap, 12),
                    "operation": operation,
                    "score_bin": score_bin(candidate.score),
                })
            else:
                stats["replace_add_keep_baseline"] += 1
        stats["replace_add_matched_gt"] += len(combined_matches)

        for candidate_index, gt_index, _ in combined_matches:
            candidate = combined[candidate_index]
            gt_item = valid_gt[gt_index]
            outputs["expanded_selected_ceiling"].append(format_candidate(candidate, gt_item.gt_id))
            stats[f"selected_source_{candidate.source}"] += 1
            if candidate.source == "phase0_postnms":
                stats[f"selected_{score_bin(candidate.score)}"] += 1

    for variant, lines in outputs.items():
        seen: Dict[int, set[int]] = defaultdict(set)
        for line in lines:
            parts = line.split(",")
            frame = int(parts[0])
            identity = int(parts[1])
            if identity in seen[frame]:
                raise RuntimeError(f"{sequence}/{variant}: duplicate ID {identity} in frame {frame}")
            seen[frame].add(identity)

    stats["expanded_pool_rows"] = stats["baseline_rows"] + stats["phase0_novel_rows"]
    stats["expanded_pool_ratio"] = (
        stats["expanded_pool_rows"] / stats["baseline_rows"]
        if stats["baseline_rows"] else 0.0
    )
    stats["baseline_gt_coverage"] = (
        stats["baseline_matched_gt"] / stats["valid_gt"]
        if stats["valid_gt"] else 0.0
    )
    stats["replace_add_gt_coverage"] = (
        stats["replace_add_matched_gt"] / stats["valid_gt"]
        if stats["valid_gt"] else 0.0
    )
    stats["sequence"] = sequence
    return outputs, dict(stats), selected_rows


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
    process = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (eval_work / "trackeval.log").write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(
            f"TrackEval failed with code {process.returncode}\n{process.stdout[-8000:]}"
        )


def parse_summary(path: Path) -> dict:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"invalid summary: {path}")
    header = lines[0].split()
    values = lines[1].split()
    if len(header) != len(values):
        raise ValueError(f"summary width mismatch: {path}: {len(header)} != {len(values)}")
    result = {}
    for key, value in zip(header, values):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def aggregate_inventory(per_sequence: Sequence[dict]) -> dict:
    integer_keys = sorted({
        key
        for row in per_sequence
        for key, value in row.items()
        if isinstance(value, int)
    })
    total = {
        key: int(sum(int(row.get(key, 0)) for row in per_sequence))
        for key in integer_keys
    }
    total["sequence"] = "COMBINED"
    total["expanded_pool_ratio"] = (
        total.get("expanded_pool_rows", 0) / max(1, total.get("baseline_rows", 0))
    )
    total["baseline_gt_coverage"] = (
        total.get("baseline_matched_gt", 0) / max(1, total.get("valid_gt", 0))
    )
    total["replace_add_gt_coverage"] = (
        total.get("replace_add_matched_gt", 0) / max(1, total.get("valid_gt", 0))
    )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--baseline-dir",
        default=(
            "outputs/trusttrack_review_20260710/all4_baseline_oracle/"
            "baseline/eval_work/trackers/all4_baseline/data"
        ),
    )
    parser.add_argument(
        "--reference-oracle-dir",
        default=(
            "outputs/trusttrack_review_20260710/all4_baseline_oracle/"
            "oracle/eval_work/trackers/all4_oracle/data"
        ),
    )
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_under(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    baseline_dir = resolve_under(repo, args.baseline_dir)
    reference_oracle_dir = resolve_under(repo, args.reference_oracle_dir)
    phase0_root = resolve_under(repo, args.phase0_root)
    gt_root = resolve_under(repo, args.gt_root)
    output_dir = resolve_under(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    eval_work = output_dir / "eval_work"

    sources = {"baseline": {}, "reference_oracle": {}, "phase0": {}, "gt": {}}
    per_sequence_stats: List[dict] = []
    selected_rows: List[dict] = []

    for sequence in SEQUENCES:
        print(f"[M23-0] building {sequence}", flush=True)
        baseline_path = baseline_dir / f"{sequence}.txt"
        reference_path = reference_oracle_dir / f"{sequence}.txt"
        phase0_path = phase0_root / sequence / "dump_yolox_reid.npz"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        for path in (baseline_path, reference_path, phase0_path, gt_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        baseline = load_baseline(baseline_path)
        phase0, phase0_metadata = load_phase0(phase0_path)
        gt_rows = load_gt(gt_path)
        outputs, stats, selected = build_sequence_variants(
            sequence,
            baseline,
            phase0,
            gt_rows,
        )
        per_sequence_stats.append(stats)
        selected_rows.extend(selected)
        for variant in VARIANTS:
            write_lines(
                eval_work / "trackers" / variant / "data" / f"{sequence}.txt",
                outputs[variant],
            )
        sources["baseline"][sequence] = {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "rows": stats["baseline_rows"],
        }
        sources["reference_oracle"][sequence] = {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        }
        sources["phase0"][sequence] = phase0_metadata
        sources["gt"][sequence] = {
            "path": str(gt_path),
            "sha256": sha256_file(gt_path),
        }

    combined = aggregate_inventory(per_sequence_stats)
    inventory_rows = list(per_sequence_stats) + [combined]
    inventory_fields = ["sequence"] + sorted({
        key for row in inventory_rows for key in row if key != "sequence"
    })
    write_csv(output_dir / "candidate_inventory.csv", inventory_rows, inventory_fields)

    selected_fields = [
        "sequence",
        "frame",
        "variant",
        "gt_id",
        "source",
        "source_uid",
        "score",
        "iou",
        "operation",
        "score_bin",
    ]
    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            row["sequence"],
            row["frame"],
            row["variant"],
            row["gt_id"],
            row["source_uid"],
        ),
    )
    write_csv(
        output_dir / "expanded_selected_events.csv",
        selected_rows,
        selected_fields,
    )

    print("[M23-0] running combined TrackEval", flush=True)
    run_trackeval(repo, gt_root, eval_work)
    metrics = {
        variant: parse_summary(
            eval_work / "eval" / variant / "pedestrian_summary.txt"
        )
        for variant in VARIANTS
    }

    reproduction = {}
    reference_equal = True
    for sequence in SEQUENCES:
        generated = (
            eval_work / "trackers" / "baseline_id_oracle" / "data" / f"{sequence}.txt"
        )
        reference = reference_oracle_dir / f"{sequence}.txt"
        generated_hash = sha256_file(generated)
        reference_hash = sha256_file(reference)
        reproduction[sequence] = {
            "generated_sha256": generated_hash,
            "reference_sha256": reference_hash,
            "byte_identical": generated_hash == reference_hash,
        }
        reference_equal = reference_equal and generated_hash == reference_hash

    baseline_metrics = metrics["baseline_raw"]
    comparisons = {}
    for variant, row in metrics.items():
        comparisons[variant] = {
            key: row.get(key)
            for key in (
                "HOTA",
                "DetA",
                "AssA",
                "IDF1",
                "MOTA",
                "IDSW",
                "CLR_FP",
                "CLR_FN",
                "Dets",
            )
        }
        comparisons[variant]["delta_vs_baseline_HOTA"] = (
            float(row.get("HOTA", 0.0)) - float(baseline_metrics.get("HOTA", 0.0))
        )
        comparisons[variant]["delta_vs_baseline_IDF1"] = (
            float(row.get("IDF1", 0.0)) - float(baseline_metrics.get("IDF1", 0.0))
        )
        comparisons[variant]["delta_vs_baseline_MOTA"] = (
            float(row.get("MOTA", 0.0)) - float(baseline_metrics.get("MOTA", 0.0))
        )

    metric_rows = []
    for variant in VARIANTS:
        row = {"variant": variant}
        row.update(comparisons[variant])
        metric_rows.append(row)
    metric_fields = [
        "variant",
        "HOTA",
        "DetA",
        "AssA",
        "IDF1",
        "MOTA",
        "IDSW",
        "CLR_FP",
        "CLR_FN",
        "Dets",
        "delta_vs_baseline_HOTA",
        "delta_vs_baseline_IDF1",
        "delta_vs_baseline_MOTA",
    ]
    write_csv(output_dir / "variant_metrics.csv", metric_rows, metric_fields)

    additive_hota = float(metrics["expanded_additive_oracle"]["HOTA"])
    replace_hota = float(metrics["expanded_replace_add_oracle"]["HOTA"])
    selected_hota = float(metrics["expanded_selected_ceiling"]["HOTA"])
    target_hota = 84.5
    decision = {
        "target_hota": target_hota,
        "baseline_hota": float(metrics["baseline_raw"]["HOTA"]),
        "baseline_id_oracle_hota": float(metrics["baseline_id_oracle"]["HOTA"]),
        "expanded_additive_oracle_hota": additive_hota,
        "expanded_replace_add_oracle_hota": replace_hota,
        "expanded_selected_ceiling_hota": selected_hota,
        "expanded_pool_ratio": combined["expanded_pool_ratio"],
        "candidate_budget_passed": combined["expanded_pool_ratio"] <= 1.5,
        "additive_target_passed": additive_hota >= target_hota,
        "replace_add_target_passed": replace_hota >= target_hota,
        "selected_ceiling_target_passed": selected_hota >= target_hota,
        "baseline_id_oracle_byte_identical_to_reference": reference_equal,
        "pre_nms_evidence_included": False,
        "propagation_evidence_included": False,
        "phase0_evidence_semantics": (
            "YOLOX postprocess/NMS output filtered at dump score >=0.09"
        ),
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "p15_policy": "no_op",
        "locked_label_reads": 0,
        "locked_trackeval_calls": 0,
        "remaining_locked_rows_untouched": 156,
    }

    report = {
        "protocol": {
            "sequences": list(SEQUENCES),
            "variants": list(VARIANTS),
            "gt_match_iou": GT_MATCH_IOU,
            "preproc_iou": PREPROC_IOU,
            "phase0_novelty_iou": NOVELTY_IOU,
            "phase0_min_score": 0.09,
            "fixed_before_evaluation": True,
            "notes": [
                "Phase-0 evidence is post-NMS, not pre-NMS.",
                "expanded_selected_ceiling removes unmatched rows and is optimistic.",
                "Row-preserving variants retain baseline false-positive burden.",
            ],
        },
        "inventory": inventory_rows,
        "metrics": metrics,
        "comparisons": comparisons,
        "reference_oracle_reproduction": reproduction,
        "decision": decision,
        "sources": sources,
    }
    canonical_json_dump(report, output_dir / "report.json")

    generated_track_hashes = {}
    for variant in VARIANTS:
        generated_track_hashes[variant] = {}
        for sequence in SEQUENCES:
            path = eval_work / "trackers" / variant / "data" / f"{sequence}.txt"
            generated_track_hashes[variant][sequence] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    compact_files = [
        "candidate_inventory.csv",
        "expanded_selected_events.csv",
        "variant_metrics.csv",
        "report.json",
    ]
    compact_hashes = {
        name: sha256_file(output_dir / name)
        for name in compact_files
    }
    manifest = {
        "schema": "fmtrack.m23.expanded_evidence_oracle.v1",
        "generated_track_hashes": generated_track_hashes,
        "compact_file_hashes": compact_hashes,
        "report_sha256": compact_hashes["report.json"],
        "decision": decision,
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
