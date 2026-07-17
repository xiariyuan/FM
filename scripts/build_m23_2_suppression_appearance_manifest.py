"""Build the GT-free M23-2 appearance manifest for the frozen pre-NMS budget.

Inputs are restricted to observable evidence:
- the frozen M23-1 candidate plan,
- verified pre-NMS suppression dumps,
- frozen post-NMS Phase-0 detector geometry,
- frozen baseline tracker outputs and MOT20 image dimensions.

The script does not accept a GT path and never runs TrackEval. It maps each
budgeted suppressed candidate to its exact post-NMS suppressor row, constructs
the integer crop consumed by FastReID, deduplicates identical candidate crops,
and records best same-frame baseline-track correspondences for both candidate
and suppressor boxes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_TO_ID = {sequence: index for index, sequence in enumerate(SEQUENCES)}
BASELINE_MATCH_IOU = 0.50
EXPECTED_PLAN_MANIFEST_SHA256 = "c9dab3959da330ef5ca5bebc85a0d87807666cc6db82e3d70f07e724e2b9a7ad"

CANDIDATE_DTYPE = np.dtype([
    ("sequence_id", "u1"),
    ("frame", "<i4"),
    ("suppressed_index", "<i8"),
    ("crop_index", "<i8"),
    ("phase0_suppressor_global_index", "<i8"),
    ("candidate_track_id", "<i4"),
    ("candidate_track_iou", "<f4"),
    ("suppressor_track_id", "<i4"),
    ("suppressor_track_iou", "<f4"),
    ("candidate_x1", "<f4"),
    ("candidate_y1", "<f4"),
    ("candidate_x2", "<f4"),
    ("candidate_y2", "<f4"),
    ("candidate_score", "<f4"),
    ("suppressor_x1", "<f4"),
    ("suppressor_y1", "<f4"),
    ("suppressor_x2", "<f4"),
    ("suppressor_y2", "<f4"),
    ("suppressor_score", "<f4"),
    ("suppressor_iou", "<f4"),
    ("priority", "<f4"),
    ("family_rank", "<i2"),
    ("local_rank", "<i2"),
])

CROP_DTYPE = np.dtype([
    ("sequence_id", "u1"),
    ("frame", "<i4"),
    ("x1", "<i4"),
    ("y1", "<i4"),
    ("x2", "<i4"),
    ("y2", "<i4"),
    ("first_suppressed_index", "<i8"),
])


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


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_npz_member(path: Path, member: str, *, allow_pickle: bool = False) -> Tuple[np.ndarray, str]:
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle), sha256_bytes(data)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
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
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def best_track_matches(query_boxes: np.ndarray, baseline_rows: Sequence[object]) -> Tuple[np.ndarray, np.ndarray]:
    track_ids = np.zeros((len(query_boxes),), dtype=np.int32)
    best_ious = np.zeros((len(query_boxes),), dtype=np.float32)
    if len(query_boxes) == 0 or not baseline_rows:
        return track_ids, best_ious
    baseline_boxes = np.asarray([row.box for row in baseline_rows], dtype=np.float32)
    overlaps = iou_matrix(query_boxes, baseline_boxes)
    best = np.argmax(overlaps, axis=1)
    values = overlaps[np.arange(len(query_boxes)), best]
    best_ious[:] = values.astype(np.float32)
    for index, (column, value) in enumerate(zip(best.tolist(), values.tolist())):
        if value + np.finfo(float).eps >= BASELINE_MATCH_IOU:
            track_ids[index] = int(baseline_rows[column].original_id)
    return track_ids, best_ious


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data",
    )
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--prenms-root", required=True)
    parser.add_argument("--candidate-plan-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fast-reid-config", default="external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", default="external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth")
    parser.add_argument("--fast-reid-interface", default="external/BoT-SORT-main/fast_reid/fast_reid_interfece.py")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    baseline_dir = resolve(repo, args.baseline_dir)
    phase0_root = resolve(repo, args.phase0_root)
    prenms_root = resolve(repo, args.prenms_root)
    plan_root = resolve(repo, args.candidate_plan_root)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    plan_manifest_path = plan_root / "manifest.json"
    plan_manifest_sha = sha256_file(plan_manifest_path)
    if plan_manifest_sha != EXPECTED_PLAN_MANIFEST_SHA256:
        raise RuntimeError(f"candidate plan manifest changed: {plan_manifest_sha}")
    plan_report = json.loads((plan_root / "report.json").read_text(encoding="utf-8"))
    if not bool(plan_report["decision"]["candidate_plan_frozen"]):
        raise RuntimeError("candidate plan is not frozen")
    if not bool(plan_report["decision"]["candidate_budget_passed"]):
        raise RuntimeError("candidate budget did not pass")

    suppression_report = json.loads((prenms_root / "report.json").read_text(encoding="utf-8"))
    if not bool(suppression_report["all_reference_equivalence_passed"]):
        raise RuntimeError("pre-NMS dump did not reproduce frozen Phase-0")

    m23 = load_module(repo / "scripts/audit_m23_mot20_expanded_evidence_oracle.py", "m23_0_for_m23_2_manifest")

    total_rows = sum(len(np.load(plan_root / sequence / "budget_indices.npy", allow_pickle=False)) for sequence in SEQUENCES)
    if total_rows != 459_899:
        raise RuntimeError(f"unexpected frozen budget size: {total_rows}")
    candidate_path = output_dir / "candidate_manifest.npy"
    # The workspace is backed by a network filesystem. Field-by-field writes to
    # an on-disk memmap trigger severe dirty-page throttling, even though the
    # complete structured array is only about 45 MB. Build it in RAM and emit a
    # single deterministic NPY write after all observable mappings validate.
    candidates = np.empty((total_rows,), dtype=CANDIDATE_DTYPE)

    crop_lookup: Dict[tuple, int] = {}
    crop_rows = []
    sequence_summary = []
    source_hashes: Dict[str, dict] = {}
    cursor = 0
    max_suppressor_box_diff = 0.0
    max_suppressor_score_diff = 0.0
    suppressor_mapping_failures = 0
    invalid_crops = 0

    for sequence in SEQUENCES:
        sequence_id = SEQUENCE_TO_ID[sequence]
        print(f"[M23-2 manifest] {sequence}", flush=True)
        budget_indices = np.load(plan_root / sequence / "budget_indices.npy", allow_pickle=False).astype(np.int64)
        priority = np.load(plan_root / sequence / "budget_priority.npy", allow_pickle=False).astype(np.float32)
        family_rank = np.load(plan_root / sequence / "budget_family_rank.npy", allow_pickle=False).astype(np.int16)
        local_rank = np.load(plan_root / sequence / "budget_local_rank.npy", allow_pickle=False).astype(np.int16)
        if not (len(budget_indices) == len(priority) == len(family_rank) == len(local_rank)):
            raise RuntimeError(f"{sequence}: frozen plan arrays differ in length")

        sequence_report = json.loads((prenms_root / sequence / "report.json").read_text(encoding="utf-8"))
        columns = json.loads((prenms_root / sequence / "columns.json").read_text(encoding="utf-8"))
        column_index = {name: index for index, name in enumerate(columns)}
        raw_shape = tuple(int(value) for value in sequence_report["files"]["suppressed_candidates"]["shape"])
        raw = np.memmap(prenms_root / sequence / "suppressed_candidates.f32", dtype="<f4", mode="r", shape=raw_shape)
        selected = np.asarray(raw[budget_indices])

        phase0_path = phase0_root / sequence / "dump_yolox_reid.npz"
        detections, detection_hash = read_npz_member(phase0_path, "detections.npy")
        offsets, offsets_hash = read_npz_member(phase0_path, "frame_offsets.npy")
        names_array, columns_hash = read_npz_member(phase0_path, "columns.npy", allow_pickle=True)
        image_wh, image_wh_hash = read_npz_member(phase0_path, "image_wh.npy")
        names = [str(value) for value in names_array.tolist()]
        phase_index = {name: index for index, name in enumerate(names)}
        required = {"x1", "y1", "x2", "y2", "score", "has_reid"}
        missing = sorted(required - set(phase_index))
        if missing:
            raise ValueError(f"{sequence}: Phase-0 missing {missing}")

        frames = selected[:, column_index["frame"]].astype(np.int64)
        ranks = selected[:, column_index["suppressor_nms_rank"]].astype(np.int64)
        if np.any(frames < 1) or np.any(frames >= len(offsets)):
            raise RuntimeError(f"{sequence}: invalid frame in frozen budget")
        frame_lengths = offsets[frames] - offsets[frames - 1]
        if np.any(ranks < 0) or np.any(ranks >= frame_lengths):
            raise RuntimeError(f"{sequence}: suppressor NMS rank is out of range")
        suppressor_global = offsets[frames - 1].astype(np.int64) + ranks
        phase_rows = detections[suppressor_global]
        suppressor_boxes = selected[:, [column_index[name] for name in ("suppressor_x1", "suppressor_y1", "suppressor_x2", "suppressor_y2")]].astype(np.float32)
        phase_boxes = phase_rows[:, [phase_index[name] for name in ("x1", "y1", "x2", "y2")]].astype(np.float32)
        box_diff = np.max(np.abs(suppressor_boxes - phase_boxes), axis=1)
        score_diff = np.abs(selected[:, column_index["suppressor_score"]] - phase_rows[:, phase_index["score"]])
        max_suppressor_box_diff = max(max_suppressor_box_diff, float(np.max(box_diff)) if len(box_diff) else 0.0)
        max_suppressor_score_diff = max(max_suppressor_score_diff, float(np.max(score_diff)) if len(score_diff) else 0.0)
        bad = (box_diff > 2e-4) | (score_diff > 1e-5)
        suppressor_mapping_failures += int(np.sum(bad))
        if np.any(bad):
            raise RuntimeError(f"{sequence}: {int(np.sum(bad))} suppressors fail exact Phase-0 mapping")
        if not np.all(phase_rows[:, phase_index["has_reid"]] > 0.5):
            raise RuntimeError(f"{sequence}: mapped suppressors lack Phase-0 ReID")

        baseline_path = baseline_dir / f"{sequence}.txt"
        baseline = m23.load_baseline(baseline_path)
        candidate_boxes = selected[:, [column_index[name] for name in ("x1", "y1", "x2", "y2")]].astype(np.float32)
        candidate_track_ids = np.zeros((len(selected),), dtype=np.int32)
        candidate_track_ious = np.zeros((len(selected),), dtype=np.float32)
        suppressor_track_ids = np.zeros((len(selected),), dtype=np.int32)
        suppressor_track_ious = np.zeros((len(selected),), dtype=np.float32)

        for frame in np.unique(frames):
            mask = np.flatnonzero(frames == frame)
            base_rows = baseline.get(int(frame), [])
            cid, ciou = best_track_matches(candidate_boxes[mask], base_rows)
            sid, siou = best_track_matches(suppressor_boxes[mask], base_rows)
            candidate_track_ids[mask] = cid
            candidate_track_ious[mask] = ciou
            suppressor_track_ids[mask] = sid
            suppressor_track_ious[mask] = siou

        start = cursor
        end = cursor + len(selected)
        view = candidates[start:end]
        view["sequence_id"] = sequence_id
        view["frame"] = frames.astype(np.int32)
        view["suppressed_index"] = budget_indices
        view["phase0_suppressor_global_index"] = suppressor_global
        view["candidate_track_id"] = candidate_track_ids
        view["candidate_track_iou"] = candidate_track_ious
        view["suppressor_track_id"] = suppressor_track_ids
        view["suppressor_track_iou"] = suppressor_track_ious
        for field, name in zip(
            ("candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2"),
            ("x1", "y1", "x2", "y2"),
        ):
            view[field] = selected[:, column_index[name]]
        view["candidate_score"] = selected[:, column_index["score"]]
        for field, name in zip(
            ("suppressor_x1", "suppressor_y1", "suppressor_x2", "suppressor_y2"),
            ("suppressor_x1", "suppressor_y1", "suppressor_x2", "suppressor_y2"),
        ):
            view[field] = selected[:, column_index[name]]
        view["suppressor_score"] = selected[:, column_index["suppressor_score"]]
        view["suppressor_iou"] = selected[:, column_index["suppressor_iou"]]
        view["priority"] = priority
        view["family_rank"] = family_rank
        view["local_rank"] = local_rank

        crop_indices = np.empty((len(selected),), dtype=np.int64)
        for local, (frame, box, suppressed_index) in enumerate(zip(frames.tolist(), candidate_boxes, budget_indices.tolist())):
            width, height = map(int, image_wh[frame - 1].tolist())
            x1, y1, x2, y2 = box.astype(np.int64).tolist()
            x1 = max(0, min(width - 1, x1)); x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1)); y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                invalid_crops += 1
                raise RuntimeError(f"{sequence} frame {frame}: invalid integer crop {(x1, y1, x2, y2)}")
            key = (sequence_id, int(frame), x1, y1, x2, y2)
            crop_index = crop_lookup.get(key)
            if crop_index is None:
                crop_index = len(crop_rows)
                crop_lookup[key] = crop_index
                crop_rows.append((sequence_id, int(frame), x1, y1, x2, y2, int(suppressed_index)))
            crop_indices[local] = crop_index
        view["crop_index"] = crop_indices
        cursor = end

        summary = {
            "sequence": sequence,
            "budget_candidates": int(len(selected)),
            "unique_candidate_crops_added": int(len(set(crop_indices.tolist()))),
            "candidate_track_match_ge_0p50": int(np.sum(candidate_track_ids > 0)),
            "candidate_track_match_rate": float(np.mean(candidate_track_ids > 0)),
            "suppressor_track_match_ge_0p50": int(np.sum(suppressor_track_ids > 0)),
            "suppressor_track_match_rate": float(np.mean(suppressor_track_ids > 0)),
            "unique_suppressor_phase0_rows": int(len(np.unique(suppressor_global))),
            "max_suppressor_box_abs_diff": float(np.max(box_diff)) if len(box_diff) else 0.0,
            "max_suppressor_score_abs_diff": float(np.max(score_diff)) if len(score_diff) else 0.0,
        }
        sequence_summary.append(summary)
        source_hashes[sequence] = {
            "baseline_sha256": sha256_file(baseline_path),
            "candidate_plan_manifest_sha256": sha256_file(plan_root / sequence / "manifest.json"),
            "suppression_manifest_sha256": sha256_file(prenms_root / sequence / "manifest.json"),
            "suppression_data_sha256": sequence_report["files"]["suppressed_candidates"]["sha256"],
            "phase0_detections_member_sha256": detection_hash,
            "phase0_offsets_member_sha256": offsets_hash,
            "phase0_columns_member_sha256": columns_hash,
            "phase0_image_wh_member_sha256": image_wh_hash,
        }

    if cursor != total_rows:
        raise RuntimeError(f"manifest cursor {cursor} != expected {total_rows}")
    candidate_tmp = output_dir / "candidate_manifest.tmp.npy"
    np.save(candidate_tmp, candidates, allow_pickle=False)
    candidate_tmp.replace(candidate_path)
    del candidates
    crops = np.asarray(crop_rows, dtype=CROP_DTYPE)
    crop_path = output_dir / "unique_candidate_crops.npy"
    np.save(crop_path, crops, allow_pickle=False)

    combined = {
        "sequence": "COMBINED",
        "budget_candidates": int(total_rows),
        "unique_candidate_crops_added": int(len(crops)),
        "candidate_track_match_ge_0p50": int(sum(row["candidate_track_match_ge_0p50"] for row in sequence_summary)),
        "suppressor_track_match_ge_0p50": int(sum(row["suppressor_track_match_ge_0p50"] for row in sequence_summary)),
        "unique_suppressor_phase0_rows": int(sum(row["unique_suppressor_phase0_rows"] for row in sequence_summary)),
        "max_suppressor_box_abs_diff": float(max_suppressor_box_diff),
        "max_suppressor_score_abs_diff": float(max_suppressor_score_diff),
    }
    combined["candidate_track_match_rate"] = combined["candidate_track_match_ge_0p50"] / max(1, total_rows)
    combined["suppressor_track_match_rate"] = combined["suppressor_track_match_ge_0p50"] / max(1, total_rows)
    summary_rows = sequence_summary + [combined]
    write_csv(
        output_dir / "sequence_summary.csv",
        summary_rows,
        [
            "sequence", "budget_candidates", "unique_candidate_crops_added",
            "candidate_track_match_ge_0p50", "candidate_track_match_rate",
            "suppressor_track_match_ge_0p50", "suppressor_track_match_rate",
            "unique_suppressor_phase0_rows", "max_suppressor_box_abs_diff",
            "max_suppressor_score_abs_diff",
        ],
    )

    config = resolve(repo, args.fast_reid_config)
    weights = resolve(repo, args.fast_reid_weights)
    interface = resolve(repo, args.fast_reid_interface)
    report = {
        "schema": "fmtrack.m23_2.suppression_appearance_manifest.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "candidate_plan_recomputed": False,
            "candidate_plan_manifest_sha256": plan_manifest_sha,
            "candidate_pool_changed": False,
            "baseline_match_iou": BASELINE_MATCH_IOU,
            "candidate_crop_semantics": "FastReID integer truncation and image-bound clipping",
            "suppressor_feature_source": "frozen Phase-0 FastReID feature row addressed by frame offset plus NMS rank",
        },
        "counts": combined,
        "sequence_summary": sequence_summary,
        "validation": {
            "suppressor_mapping_failures": int(suppressor_mapping_failures),
            "invalid_candidate_crops": int(invalid_crops),
            "candidate_rows_equal_frozen_budget": int(total_rows) == 459_899,
            "all_suppressors_have_phase0_reid": True,
        },
        "assets": {
            "fast_reid_config": {"path": "external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml", "sha256": sha256_file(config)},
            "fast_reid_weights": {"path": "external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth", "sha256": sha256_file(weights)},
            "fast_reid_interface": {"path": "external/BoT-SORT-main/fast_reid/fast_reid_interfece.py", "sha256": sha256_file(interface)},
        },
        "sources": source_hashes,
        "files": {
            "candidate_manifest.npy": {"sha256": sha256_file(candidate_path), "dtype": CANDIDATE_DTYPE.descr, "shape": [total_rows], "size_bytes": candidate_path.stat().st_size},
            "unique_candidate_crops.npy": {"sha256": sha256_file(crop_path), "dtype": CROP_DTYPE.descr, "shape": [len(crops)], "size_bytes": crop_path.stat().st_size},
            "sequence_summary.csv": {"sha256": sha256_file(output_dir / "sequence_summary.csv"), "rows": len(summary_rows)},
        },
        "decision": {
            "appearance_manifest_ready": True,
            "candidate_plan_verified": True,
            "appearance_features_ready": False,
            "deployment_allowed": False,
            "locked_manifest_created": False,
        },
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    manifest = {
        "schema": "fmtrack.m23_2.suppression_appearance_manifest.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {
            name: sha256_file(output_dir / name)
            for name in ("candidate_manifest.npy", "unique_candidate_crops.npy", "sequence_summary.csv", "report.json")
        },
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(report["counts"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
