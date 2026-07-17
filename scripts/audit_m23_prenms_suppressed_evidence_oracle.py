"""M23-1 MOT20 pre-NMS suppressed-evidence oracle audit.

This diagnostic combines the frozen baseline, the frozen post-NMS Phase-0 pool,
and a fixed-budget subset of NMS-suppressed YOLOX candidates. Candidate
selection is completely GT-free and frozen before evaluation.

Fixed candidate policy per sequence:
- Preserve the M23-0 baseline + post-NMS novel pool.
- The total observable pool may not exceed 1.50 times baseline rows.
- Discard suppressed candidates with suppressor IoU >= 0.99.
- Within each suppression family, greedily deduplicate at IoU >= 0.95.
- Priority is score times normalized geometric novelty from the suppressor.
- Diversify first by suppression-family rank, then create a local frame rank.
- Fill the sequence budget round-robin by local frame rank and priority.

Ground truth is used only after candidate selection to construct explicit oracle
variants and to run TrackEval. No selector is trained or tuned.
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
    "postnms_replace_add_oracle",
    "prenms_budget_additive_oracle",
    "prenms_budget_replace_add_oracle",
    "prenms_budget_selected_ceiling",
    "prenms_full_selected_ceiling",
)
POOL_RATIO_LIMIT = 1.50
EXACT_DUPLICATE_IOU = 0.99
FAMILY_DEDUPE_IOU = 0.95
TARGET_COMBINED_HOTA = 84.50
TARGET_WORST_SEQUENCE_HOTA = 84.00
PRENMS_UID_OFFSET = 30_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_under(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
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


@dataclass
class PreNMSStore:
    sequence: str
    data: np.memmap
    offsets: np.ndarray
    columns: List[str]
    index: Dict[str, int]
    report: dict
    manifest: dict


@dataclass
class BudgetPlan:
    full_mask: np.ndarray
    budget_mask: np.ndarray
    priority: np.ndarray
    family_rank: np.ndarray
    local_rank: np.ndarray
    stats: dict


def load_prenms_store(root: Path, sequence: str) -> PreNMSStore:
    sequence_dir = root / sequence
    report = json.loads((sequence_dir / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((sequence_dir / "manifest.json").read_text(encoding="utf-8"))
    columns = json.loads((sequence_dir / "columns.json").read_text(encoding="utf-8"))
    index = {name: idx for idx, name in enumerate(columns)}
    shape = tuple(int(value) for value in report["files"]["suppressed_candidates"]["shape"])
    data = np.memmap(
        sequence_dir / "suppressed_candidates.f32",
        dtype="<f4",
        mode="r",
        shape=shape,
    )
    offsets = np.load(sequence_dir / "frame_offsets.npy", allow_pickle=False).astype(np.int64)
    if len(offsets) != int(report["counts"]["frames"]) + 1:
        raise RuntimeError(f"{sequence}: invalid pre-NMS frame offsets")
    if not bool(report["decision"]["reference_equivalence_passed"]):
        raise RuntimeError(f"{sequence}: pre-NMS dump failed frozen Phase-0 equivalence")
    required = {
        "frame",
        "raw_anchor_idx",
        "global_pre_idx",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "suppressor_frame_pre_idx",
        "suppressor_iou",
        "suppressor_score",
    }
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"{sequence}: missing pre-NMS columns {missing}")
    return PreNMSStore(sequence, data, offsets, columns, index, report, manifest)


def load_frozen_plan(
    root: Path,
    sequence: str,
    store: PreNMSStore,
    *,
    baseline_path: Path,
    phase0_metadata: Mapping[str, object],
) -> Tuple[BudgetPlan, dict, str]:
    """Load and verify a GT-free candidate plan frozen by the prior stage."""
    sequence_dir = root / sequence
    report_path = sequence_dir / "report.json"
    manifest_path = sequence_dir / "manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("schema") != "fmtrack.m23.prenms_candidate_plan.v1":
        raise RuntimeError(f"{sequence}: unexpected frozen-plan schema")
    if report.get("sequence") != sequence:
        raise RuntimeError(f"{sequence}: frozen-plan sequence mismatch")
    protocol = report.get("protocol", {})
    if bool(protocol.get("ground_truth_read", True)):
        raise RuntimeError(f"{sequence}: candidate plan was not generated GT-free")
    if int(protocol.get("trackeval_calls", -1)) != 0:
        raise RuntimeError(f"{sequence}: candidate plan includes TrackEval calls")
    if float(protocol.get("pool_ratio_limit")) != float(POOL_RATIO_LIMIT):
        raise RuntimeError(f"{sequence}: pool-ratio policy drift")
    if float(protocol.get("exact_duplicate_iou")) != float(EXACT_DUPLICATE_IOU):
        raise RuntimeError(f"{sequence}: exact-duplicate policy drift")
    if float(protocol.get("family_dedupe_iou")) != float(FAMILY_DEDUPE_IOU):
        raise RuntimeError(f"{sequence}: family-dedupe policy drift")

    sources = report.get("sources", {})
    if sources.get("baseline", {}).get("sha256") != sha256_file(baseline_path):
        raise RuntimeError(f"{sequence}: frozen-plan baseline hash mismatch")
    expected_phase0 = sources.get("phase0", {})
    for key in ("detections_member_sha256", "columns_member_sha256", "detections"):
        if expected_phase0.get(key) != phase0_metadata.get(key):
            raise RuntimeError(f"{sequence}: frozen-plan Phase-0 {key} mismatch")
    if sources.get("prenms_manifest", {}).get("sha256") != sha256_file(
        Path(store.report["files"]["suppressed_candidates"]["path"]).parent / "manifest.json"
    ):
        raise RuntimeError(f"{sequence}: frozen-plan pre-NMS manifest mismatch")
    if sources.get("prenms_data", {}).get("sha256") != store.manifest["file_hashes"][
        "suppressed_candidates.f32"
    ]:
        raise RuntimeError(f"{sequence}: frozen-plan pre-NMS data mismatch")

    required_names = (
        "full_indices",
        "full_priority",
        "full_family_rank",
        "full_local_rank",
        "budget_indices",
        "budget_priority",
        "budget_family_rank",
        "budget_local_rank",
    )
    arrays = {}
    for name in required_names:
        path = sequence_dir / f"{name}.npy"
        expected_hash = report.get("files", {}).get(name, {}).get("sha256")
        if expected_hash != sha256_file(path):
            raise RuntimeError(f"{sequence}: frozen-plan file hash mismatch for {name}")
        arrays[name] = np.load(path, allow_pickle=False)

    full_indices = arrays["full_indices"].astype(np.int64, copy=False)
    budget_indices = arrays["budget_indices"].astype(np.int64, copy=False)
    total_rows = len(store.data)
    for label, indices in (("full", full_indices), ("budget", budget_indices)):
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= total_rows):
            raise RuntimeError(f"{sequence}: invalid {label} frozen indices")
        if len(indices) and (np.any(indices[1:] <= indices[:-1])):
            raise RuntimeError(f"{sequence}: {label} frozen indices are not strictly sorted")
    if len(np.setdiff1d(budget_indices, full_indices, assume_unique=True)):
        raise RuntimeError(f"{sequence}: budget plan is not a subset of full survivors")

    for prefix, indices in (("full", full_indices), ("budget", budget_indices)):
        for suffix in ("priority", "family_rank", "local_rank"):
            values = arrays[f"{prefix}_{suffix}"]
            if values.ndim != 1 or len(values) != len(indices):
                raise RuntimeError(f"{sequence}: frozen {prefix}_{suffix} length mismatch")

    full_mask = np.zeros(total_rows, dtype=bool)
    budget_mask = np.zeros(total_rows, dtype=bool)
    priority = np.full(total_rows, -np.inf, dtype=np.float32)
    family_rank = np.full(total_rows, -1, dtype=np.int16)
    local_rank = np.full(total_rows, -1, dtype=np.int16)
    full_mask[full_indices] = True
    budget_mask[budget_indices] = True
    priority[full_indices] = arrays["full_priority"].astype(np.float32, copy=False)
    family_rank[full_indices] = arrays["full_family_rank"].astype(np.int16, copy=False)
    local_rank[full_indices] = arrays["full_local_rank"].astype(np.int16, copy=False)
    if not np.array_equal(priority[budget_indices], arrays["budget_priority"].astype(np.float32)):
        raise RuntimeError(f"{sequence}: budget priority is inconsistent with full plan")
    if not np.array_equal(family_rank[budget_indices], arrays["budget_family_rank"].astype(np.int16)):
        raise RuntimeError(f"{sequence}: budget family rank is inconsistent with full plan")
    if not np.array_equal(local_rank[budget_indices], arrays["budget_local_rank"].astype(np.int16)):
        raise RuntimeError(f"{sequence}: budget local rank is inconsistent with full plan")

    stats = dict(report.get("inventory", {}))
    if int(stats.get("full_survivor_rows", -1)) != len(full_indices):
        raise RuntimeError(f"{sequence}: frozen full-survivor count mismatch")
    if int(stats.get("budget_selected_rows", -1)) != len(budget_indices):
        raise RuntimeError(f"{sequence}: frozen budget count mismatch")
    if not bool(stats.get("budget_passed")):
        raise RuntimeError(f"{sequence}: frozen candidate budget failed")
    if manifest.get("report_sha256") != sha256_file(report_path):
        raise RuntimeError(f"{sequence}: frozen-plan report hash mismatch")
    return (
        BudgetPlan(full_mask, budget_mask, priority, family_rank, local_rank, stats),
        report,
        sha256_file(manifest_path),
    )


def greedy_family_dedupe(rows: np.ndarray, candidates: np.ndarray, box_columns: Sequence[int]) -> List[int]:
    if len(candidates) <= 1:
        return candidates.tolist()
    selected: List[int] = []
    selected_boxes: List[np.ndarray] = []
    for local_index in candidates.tolist():
        box = rows[local_index, box_columns].astype(np.float64)
        if selected_boxes:
            maximum = float(np.max(iou_matrix(box[None, :], np.asarray(selected_boxes))))
            if maximum >= FAMILY_DEDUPE_IOU:
                continue
        selected.append(int(local_index))
        selected_boxes.append(box)
    return selected


def build_budget_plan(
    store: PreNMSStore,
    *,
    baseline_rows: int,
    postnms_novel_rows: int,
) -> BudgetPlan:
    total_rows = len(store.data)
    full_mask = np.zeros(total_rows, dtype=bool)
    priority = np.full(total_rows, -np.inf, dtype=np.float32)
    family_rank = np.full(total_rows, -1, dtype=np.int16)
    local_rank = np.full(total_rows, -1, dtype=np.int16)
    idx = store.index
    box_columns = [idx[name] for name in ("x1", "y1", "x2", "y2")]
    nms_threshold = float(store.report["protocol"]["nms_threshold"])
    denominator = max(EXACT_DUPLICATE_IOU - nms_threshold, 1e-6)

    survivor_indices: List[np.ndarray] = []
    survivor_local_rank: List[np.ndarray] = []
    survivor_priority: List[np.ndarray] = []
    survivor_frames: List[np.ndarray] = []
    exact_duplicate_removed = 0
    family_duplicate_removed = 0
    families = 0

    for frame in range(1, len(store.offsets)):
        start = int(store.offsets[frame - 1])
        end = int(store.offsets[frame])
        if end <= start:
            continue
        rows = np.asarray(store.data[start:end])
        suppressor_iou = rows[:, idx["suppressor_iou"]].astype(np.float64)
        score = rows[:, idx["score"]].astype(np.float64)
        candidate_mask = suppressor_iou < EXACT_DUPLICATE_IOU
        exact_duplicate_removed += int(np.sum(~candidate_mask))
        candidate_local = np.flatnonzero(candidate_mask)
        if not len(candidate_local):
            continue
        novelty = np.clip((EXACT_DUPLICATE_IOU - suppressor_iou) / denominator, 0.0, 1.0)
        frame_priority = score * novelty
        suppressor = rows[:, idx["suppressor_frame_pre_idx"]].astype(np.int64)
        kept_local: List[int] = []
        frame_family_rank = np.full(len(rows), -1, dtype=np.int16)

        for family_id in np.unique(suppressor[candidate_local]):
            family_candidates = candidate_local[suppressor[candidate_local] == family_id]
            families += 1
            order = np.lexsort((
                rows[family_candidates, idx["raw_anchor_idx"]],
                suppressor_iou[family_candidates],
                -score[family_candidates],
                -frame_priority[family_candidates],
            ))
            ordered = family_candidates[order]
            deduplicated = greedy_family_dedupe(rows, ordered, box_columns)
            family_duplicate_removed += len(ordered) - len(deduplicated)
            for rank, local_index in enumerate(deduplicated):
                frame_family_rank[local_index] = rank
                kept_local.append(local_index)

        if not kept_local:
            continue
        kept_array = np.asarray(kept_local, dtype=np.int64)
        frame_order = np.lexsort((
            rows[kept_array, idx["raw_anchor_idx"]],
            suppressor_iou[kept_array],
            -score[kept_array],
            -frame_priority[kept_array],
            frame_family_rank[kept_array],
        ))
        ranked = kept_array[frame_order]
        global_indices = ranked + start
        ranks = np.arange(len(ranked), dtype=np.int16)
        full_mask[global_indices] = True
        priority[global_indices] = frame_priority[ranked].astype(np.float32)
        family_rank[global_indices] = frame_family_rank[ranked]
        local_rank[global_indices] = ranks
        survivor_indices.append(global_indices)
        survivor_local_rank.append(ranks)
        survivor_priority.append(frame_priority[ranked].astype(np.float32))
        survivor_frames.append(np.full(len(ranked), frame, dtype=np.int32))

    if survivor_indices:
        all_indices = np.concatenate(survivor_indices)
        all_local_rank = np.concatenate(survivor_local_rank)
        all_priority = np.concatenate(survivor_priority)
        all_frames = np.concatenate(survivor_frames)
    else:
        all_indices = np.zeros((0,), dtype=np.int64)
        all_local_rank = np.zeros((0,), dtype=np.int16)
        all_priority = np.zeros((0,), dtype=np.float32)
        all_frames = np.zeros((0,), dtype=np.int32)

    max_total = int(math.floor(POOL_RATIO_LIMIT * baseline_rows))
    available_budget = max(0, max_total - baseline_rows - postnms_novel_rows)
    selection_order = np.lexsort((
        all_indices,
        all_frames,
        -all_priority,
        all_local_rank,
    ))
    chosen = all_indices[selection_order[:available_budget]]
    budget_mask = np.zeros(total_rows, dtype=bool)
    budget_mask[chosen] = True
    selected_count = int(np.sum(budget_mask))
    pool_rows = baseline_rows + postnms_novel_rows + selected_count
    full_indices_sorted = np.flatnonzero(full_mask).astype("<i8", copy=False)
    budget_indices_sorted = np.flatnonzero(budget_mask).astype("<i8", copy=False)
    stats = {
        "raw_suppressed_rows": int(total_rows),
        "full_survivor_index_sha256": hashlib.sha256(full_indices_sorted.tobytes()).hexdigest(),
        "budget_selected_index_sha256": hashlib.sha256(budget_indices_sorted.tobytes()).hexdigest(),
        "exact_duplicate_removed_iou_ge_0p99": int(exact_duplicate_removed),
        "family_duplicate_removed_iou_ge_0p95": int(family_duplicate_removed),
        "suppression_families": int(families),
        "full_survivor_rows": int(np.sum(full_mask)),
        "available_suppressed_budget": int(available_budget),
        "budget_selected_rows": selected_count,
        "baseline_rows": int(baseline_rows),
        "postnms_novel_rows": int(postnms_novel_rows),
        "budget_pool_rows": int(pool_rows),
        "budget_pool_ratio": pool_rows / max(1, baseline_rows),
        "budget_limit_rows": max_total,
        "budget_passed": pool_rows <= max_total,
        "policy": (
            "discard suppressor_iou>=0.99; within-family greedy IoU<0.95; "
            "priority=score*normalized suppressor novelty; family-rank diversity; "
            "sequence round-robin by frame-local rank"
        ),
    }
    return BudgetPlan(full_mask, budget_mask, priority, family_rank, local_rank, stats)


def suppressed_candidates_for_frame(
    m23,
    store: PreNMSStore,
    plan: BudgetPlan,
    frame: int,
    *,
    budgeted: bool,
) -> Tuple[List[object], Dict[int, dict]]:
    start = int(store.offsets[frame - 1])
    end = int(store.offsets[frame])
    if end <= start:
        return [], {}
    mask = plan.budget_mask[start:end] if budgeted else plan.full_mask[start:end]
    positions = np.flatnonzero(mask) + start
    idx = store.index
    candidates = []
    metadata: Dict[int, dict] = {}
    for position in positions.tolist():
        row = store.data[position]
        global_pre_idx = int(row[idx["global_pre_idx"]])
        uid = PRENMS_UID_OFFSET + global_pre_idx
        candidate = m23.Candidate(
            frame=frame,
            source="prenms_suppressed",
            uid=uid,
            original_id=global_pre_idx,
            box=tuple(float(row[idx[name]]) for name in ("x1", "y1", "x2", "y2")),
            score=float(row[idx["score"]]),
        )
        candidates.append(candidate)
        metadata[uid] = {
            "raw_anchor_idx": int(row[idx["raw_anchor_idx"]]),
            "global_pre_idx": global_pre_idx,
            "score": float(row[idx["score"]]),
            "suppressor_iou": float(row[idx["suppressor_iou"]]),
            "suppressor_score": float(row[idx["suppressor_score"]]),
            "priority": float(plan.priority[position]),
            "family_rank": int(plan.family_rank[position]),
            "local_rank": int(plan.local_rank[position]),
        }
    return candidates, metadata


def emit_replace_add(
    m23,
    variant: str,
    baseline: Sequence[object],
    base_kept: Sequence[object],
    base_matched_uids: set,
    valid_gt: Sequence[object],
    combined: Sequence[object],
    outputs: Dict[str, List[str]],
    selected_rows: List[dict],
    sequence: str,
    frame: int,
    metadata: Mapping[int, dict],
    stats: defaultdict,
) -> List[Tuple[int, int, float]]:
    matches = m23.match_candidates(combined, valid_gt)
    for candidate in baseline:
        if candidate.uid not in base_matched_uids:
            outputs[variant].append(m23.format_candidate(candidate, m23.unmatched_oracle_id(candidate)))
    for candidate_index, gt_index, overlap in matches:
        candidate = combined[candidate_index]
        gt_item = valid_gt[gt_index]
        outputs[variant].append(m23.format_candidate(candidate, gt_item.gt_id))
        stats[f"{variant}_source_{candidate.source}"] += 1
        if candidate.source == "prenms_suppressed":
            row = {
                "sequence": sequence,
                "frame": frame,
                "variant": variant,
                "gt_id": gt_item.gt_id,
                "source": candidate.source,
                "source_uid": candidate.uid,
                "score": round(candidate.score, 12),
                "iou": round(overlap, 12),
                "operation": "replace_or_add",
            }
            row.update(metadata.get(candidate.uid, {}))
            selected_rows.append(row)
    return matches


def validate_unique_ids(sequence: str, outputs: Mapping[str, Sequence[str]]) -> None:
    for variant, lines in outputs.items():
        seen: Dict[int, set] = defaultdict(set)
        for line in lines:
            parts = line.split(",")
            frame = int(parts[0])
            identity = int(parts[1])
            if identity in seen[frame]:
                raise RuntimeError(f"{sequence}/{variant}: duplicate ID {identity} in frame {frame}")
            seen[frame].add(identity)


def build_sequence_variants(
    m23,
    sequence: str,
    baseline_by_frame: Mapping[int, List[object]],
    phase0_by_frame: Mapping[int, List[object]],
    gt_by_frame: Mapping[int, List[object]],
    store: PreNMSStore,
    plan: BudgetPlan,
) -> Tuple[Dict[str, List[str]], dict, List[dict]]:
    outputs = {variant: [] for variant in VARIANTS}
    stats = defaultdict(int)
    selected_rows: List[dict] = []
    frames = sorted(set(baseline_by_frame) | set(phase0_by_frame) | set(gt_by_frame))

    for frame in frames:
        baseline = list(baseline_by_frame.get(frame, []))
        phase0 = list(phase0_by_frame.get(frame, []))
        gt_rows = list(gt_by_frame.get(frame, []))
        stats["frames"] += 1
        stats["baseline_rows"] += len(baseline)

        for candidate in baseline:
            outputs["baseline_raw"].append(m23.format_candidate(candidate, candidate.original_id))
        base_kept, valid_gt, base_removed = m23.valid_and_distractor_filtered(baseline, gt_rows)
        base_matches = m23.match_candidates(base_kept, valid_gt)
        base_matched_uids = {base_kept[candidate_index].uid for candidate_index, _, _ in base_matches}
        base_gt_id_by_uid = {
            base_kept[candidate_index].uid: valid_gt[gt_index].gt_id
            for candidate_index, gt_index, _ in base_matches
        }
        base_match_gt_ids = {valid_gt[gt_index].gt_id for _, gt_index, _ in base_matches}
        stats["valid_gt"] += len(valid_gt)
        stats["baseline_matched_gt"] += len(base_matches)
        stats["baseline_preproc_removed"] += len(base_removed)
        for candidate in baseline:
            outputs["baseline_id_oracle"].append(
                m23.format_candidate(candidate, base_gt_id_by_uid.get(candidate.uid, m23.unmatched_oracle_id(candidate)))
            )

        novel_postnms, _ = m23.novel_phase0(phase0, baseline)
        postnms_kept, _, postnms_removed = m23.valid_and_distractor_filtered(novel_postnms, gt_rows)
        stats["postnms_novel_rows"] += len(novel_postnms)
        stats["postnms_preproc_removed"] += len(postnms_removed)
        post_metadata: Dict[int, dict] = {}
        post_combined = list(base_kept) + list(postnms_kept)
        post_matches = emit_replace_add(
            m23,
            "postnms_replace_add_oracle",
            baseline,
            base_kept,
            base_matched_uids,
            valid_gt,
            post_combined,
            outputs,
            selected_rows,
            sequence,
            frame,
            post_metadata,
            stats,
        )
        stats["postnms_matched_gt"] += len(post_matches)

        budget_suppressed, budget_metadata = suppressed_candidates_for_frame(
            m23, store, plan, frame, budgeted=True
        )
        full_suppressed, full_metadata = suppressed_candidates_for_frame(
            m23, store, plan, frame, budgeted=False
        )
        budget_kept, _, budget_removed = m23.valid_and_distractor_filtered(budget_suppressed, gt_rows)
        full_kept, _, full_removed = m23.valid_and_distractor_filtered(full_suppressed, gt_rows)
        stats["budget_suppressed_rows"] += len(budget_suppressed)
        stats["full_suppressed_rows"] += len(full_suppressed)
        stats["budget_suppressed_preproc_removed"] += len(budget_removed)
        stats["full_suppressed_preproc_removed"] += len(full_removed)

        evidence_budget = list(postnms_kept) + list(budget_kept)
        missed_gt = [item for item in valid_gt if item.gt_id not in base_match_gt_ids]
        additive_matches = m23.match_candidates(evidence_budget, missed_gt)
        for candidate in baseline:
            outputs["prenms_budget_additive_oracle"].append(
                m23.format_candidate(candidate, base_gt_id_by_uid.get(candidate.uid, m23.unmatched_oracle_id(candidate)))
            )
        all_metadata = dict(budget_metadata)
        for candidate_index, gt_index, overlap in additive_matches:
            candidate = evidence_budget[candidate_index]
            gt_item = missed_gt[gt_index]
            outputs["prenms_budget_additive_oracle"].append(m23.format_candidate(candidate, gt_item.gt_id))
            stats[f"budget_additive_source_{candidate.source}"] += 1
            if candidate.source == "prenms_suppressed":
                row = {
                    "sequence": sequence,
                    "frame": frame,
                    "variant": "prenms_budget_additive_oracle",
                "gt_id": gt_item.gt_id,
                "source": candidate.source,
                "source_uid": candidate.uid,
                "score": round(candidate.score, 12),
                "iou": round(overlap, 12),
                    "operation": "add_missing",
                }
                row.update(all_metadata.get(candidate.uid, {}))
                selected_rows.append(row)
        stats["budget_additive_recovered_gt"] += len(additive_matches)

        budget_combined = list(base_kept) + evidence_budget
        budget_matches = emit_replace_add(
            m23,
            "prenms_budget_replace_add_oracle",
            baseline,
            base_kept,
            base_matched_uids,
            valid_gt,
            budget_combined,
            outputs,
            selected_rows,
            sequence,
            frame,
            budget_metadata,
            stats,
        )
        stats["budget_replace_add_matched_gt"] += len(budget_matches)
        for candidate_index, gt_index, overlap in budget_matches:
            candidate = budget_combined[candidate_index]
            gt_item = valid_gt[gt_index]
            outputs["prenms_budget_selected_ceiling"].append(m23.format_candidate(candidate, gt_item.gt_id))
            stats[f"budget_selected_source_{candidate.source}"] += 1
            # Selection is identical to prenms_budget_replace_add_oracle; avoid duplicate event rows.

        full_combined = list(base_kept) + list(postnms_kept) + list(full_kept)
        full_matches = m23.match_candidates(full_combined, valid_gt)
        stats["full_selected_matched_gt"] += len(full_matches)
        for candidate_index, gt_index, overlap in full_matches:
            candidate = full_combined[candidate_index]
            gt_item = valid_gt[gt_index]
            outputs["prenms_full_selected_ceiling"].append(m23.format_candidate(candidate, gt_item.gt_id))
            stats[f"full_selected_source_{candidate.source}"] += 1
            # Full-dedup ceiling is summarized by counts and TrackEval; detailed rows are omitted.

    validate_unique_ids(sequence, outputs)
    stats.update(plan.stats)
    stats["sequence"] = sequence
    stats["baseline_gt_coverage"] = stats["baseline_matched_gt"] / max(1, stats["valid_gt"])
    stats["postnms_gt_coverage"] = stats["postnms_matched_gt"] / max(1, stats["valid_gt"])
    stats["budget_gt_coverage"] = stats["budget_replace_add_matched_gt"] / max(1, stats["valid_gt"])
    stats["full_gt_coverage"] = stats["full_selected_matched_gt"] / max(1, stats["valid_gt"])
    return outputs, dict(stats), selected_rows


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
            sequence = row["seq"]
            rows.append({
                "variant": variant,
                "sequence": sequence,
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


def aggregate_inventory(per_sequence: Sequence[dict]) -> dict:
    total = {"sequence": "COMBINED"}
    integer_keys = sorted({key for row in per_sequence for key, value in row.items() if isinstance(value, int)})
    for key in integer_keys:
        total[key] = int(sum(int(row.get(key, 0)) for row in per_sequence))
    total["budget_pool_ratio"] = total.get("budget_pool_rows", 0) / max(1, total.get("baseline_rows", 0))
    for prefix in ("baseline", "postnms", "budget", "full"):
        numerator = total.get(f"{prefix}_matched_gt", total.get(f"{prefix}_replace_add_matched_gt", total.get(f"{prefix}_selected_matched_gt", 0)))
        if prefix == "budget":
            numerator = total.get("budget_replace_add_matched_gt", 0)
        if prefix == "full":
            numerator = total.get("full_selected_matched_gt", 0)
        total[f"{prefix}_gt_coverage"] = numerator / max(1, total.get("valid_gt", 0))
    total["budget_passed"] = all(bool(row["budget_passed"]) for row in per_sequence) and total["budget_pool_ratio"] <= POOL_RATIO_LIMIT
    return total


def score_bin(score: float) -> str:
    if score >= 0.60:
        return "score_ge_0p60"
    if score >= 0.30:
        return "score_0p30_0p60"
    if score >= 0.10:
        return "score_0p10_0p30"
    return "score_0p09_0p10"


def suppressor_iou_bin(value: float) -> str:
    if value < 0.75:
        return "iou_0p70_0p75"
    if value < 0.80:
        return "iou_0p75_0p80"
    if value < 0.90:
        return "iou_0p80_0p90"
    if value < 0.95:
        return "iou_0p90_0p95"
    return "iou_0p95_0p99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--baseline-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/baseline/eval_work/trackers/all4_baseline/data",
    )
    parser.add_argument(
        "--reference-oracle-dir",
        default="outputs/trusttrack_review_20260710/all4_baseline_oracle/oracle/eval_work/trackers/all4_oracle/data",
    )
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--prenms-root", required=True)
    parser.add_argument("--candidate-plan-root", required=True)
    parser.add_argument("--gt-root", default="datasets/MOT20/train")
    parser.add_argument(
        "--m23-0-manifest",
        default="outputs/mot20_m23_20260717/expanded_evidence_oracle_v1/manifest.json",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    baseline_dir = resolve_under(repo, args.baseline_dir)
    reference_oracle_dir = resolve_under(repo, args.reference_oracle_dir)
    phase0_root = resolve_under(repo, args.phase0_root)
    prenms_root = resolve_under(repo, args.prenms_root)
    candidate_plan_root = resolve_under(repo, args.candidate_plan_root)
    gt_root = resolve_under(repo, args.gt_root)
    m23_0_manifest_path = resolve_under(repo, args.m23_0_manifest)
    output_dir = resolve_under(repo, args.out_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    eval_work = output_dir / "eval_work"
    m23 = load_module(repo / "scripts" / "audit_m23_mot20_expanded_evidence_oracle.py", "fmtrack_m23_0")
    m23_0_manifest = json.loads(m23_0_manifest_path.read_text(encoding="utf-8"))

    per_sequence_stats: List[dict] = []
    selected_rows: List[dict] = []
    source_hashes = {
        "baseline": {},
        "phase0": {},
        "prenms": {},
        "candidate_plan": {},
        "gt": {},
    }
    reference_oracle_reproduction = {}
    postnms_track_reproduction = {}

    candidate_plan_report_path = candidate_plan_root / "report.json"
    candidate_plan_manifest_path = candidate_plan_root / "manifest.json"
    candidate_plan_report = json.loads(candidate_plan_report_path.read_text(encoding="utf-8"))
    candidate_plan_manifest = json.loads(candidate_plan_manifest_path.read_text(encoding="utf-8"))
    if candidate_plan_report.get("schema") != "fmtrack.m23.prenms_candidate_plan.combined.v1":
        raise RuntimeError("unexpected combined candidate-plan schema")
    if not bool(candidate_plan_report.get("protocol", {}).get("candidate_plan_frozen_before_oracle")):
        raise RuntimeError("candidate plan was not frozen before oracle evaluation")
    if bool(candidate_plan_report.get("protocol", {}).get("ground_truth_read", True)):
        raise RuntimeError("candidate-plan stage read ground truth")
    if int(candidate_plan_report.get("protocol", {}).get("trackeval_calls", -1)) != 0:
        raise RuntimeError("candidate-plan stage called TrackEval")
    if not bool(candidate_plan_report.get("decision", {}).get("candidate_plan_frozen")):
        raise RuntimeError("candidate plan is not marked frozen")
    if not bool(candidate_plan_report.get("decision", {}).get("candidate_budget_passed")):
        raise RuntimeError("frozen candidate plan exceeds the fixed budget")
    if candidate_plan_manifest.get("report_sha256") != sha256_file(candidate_plan_report_path):
        raise RuntimeError("combined candidate-plan report hash mismatch")

    # Phase A: load and verify all evidence and all frozen plans. No GT file is
    # opened anywhere in this phase. This establishes a strict information
    # barrier before oracle construction begins.
    prepared = {}
    for sequence in SEQUENCES:
        print(f"[M23-1 audit] verifying frozen plan {sequence}", flush=True)
        baseline_path = baseline_dir / f"{sequence}.txt"
        reference_oracle_path = reference_oracle_dir / f"{sequence}.txt"
        phase0_path = phase0_root / sequence / "dump_yolox_reid.npz"
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        baseline = m23.load_baseline(baseline_path)
        phase0, phase0_metadata = m23.load_phase0(phase0_path)
        store = load_prenms_store(prenms_root, sequence)
        plan, frozen_report, frozen_manifest_sha = load_frozen_plan(
            candidate_plan_root,
            sequence,
            store,
            baseline_path=baseline_path,
            phase0_metadata=phase0_metadata,
        )
        if not bool(plan.stats["budget_passed"]):
            raise RuntimeError(f"{sequence}: fixed candidate budget exceeded: {plan.stats}")

        expected_sequence_manifest = candidate_plan_report.get("sequence_manifests", {}).get(
            sequence, {}
        ).get("sha256")
        if expected_sequence_manifest != frozen_manifest_sha:
            raise RuntimeError(f"{sequence}: combined candidate-plan manifest mismatch")
        prepared[sequence] = {
            "baseline_path": baseline_path,
            "reference_oracle_path": reference_oracle_path,
            "phase0_path": phase0_path,
            "gt_path": gt_path,
            "baseline": baseline,
            "phase0": phase0,
            "phase0_metadata": phase0_metadata,
            "store": store,
            "plan": plan,
            "frozen_report": frozen_report,
        }
        source_hashes["baseline"][sequence] = {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        }
        source_hashes["phase0"][sequence] = phase0_metadata
        source_hashes["prenms"][sequence] = {
            "manifest_sha256": sha256_file(prenms_root / sequence / "manifest.json"),
            "data_sha256": store.manifest["file_hashes"]["suppressed_candidates.f32"],
        }
        source_hashes["candidate_plan"][sequence] = {
            "manifest_sha256": frozen_manifest_sha,
            "report_sha256": sha256_file(candidate_plan_root / sequence / "report.json"),
        }

    print(
        f"[M23-1 audit] candidate plan frozen and verified: "
        f"{sha256_file(candidate_plan_manifest_path)}",
        flush=True,
    )

    # Phase B: only after all four plans are verified do we open GT and create
    # diagnostic oracle outputs. Candidate masks are never recomputed here.
    for sequence in SEQUENCES:
        print(f"[M23-1 audit] constructing oracle variants {sequence}", flush=True)
        item = prepared[sequence]
        baseline_path = item["baseline_path"]
        reference_oracle_path = item["reference_oracle_path"]
        gt_path = item["gt_path"]
        baseline = item["baseline"]
        phase0 = item["phase0"]
        store = item["store"]
        plan = item["plan"]
        gt = m23.load_gt(gt_path)
        outputs, stats, selected = build_sequence_variants(m23, sequence, baseline, phase0, gt, store, plan)
        per_sequence_stats.append(stats)
        selected_rows.extend(selected)
        for variant in VARIANTS:
            write_lines(eval_work / "trackers" / variant / "data" / f"{sequence}.txt", outputs[variant])

        baseline_oracle_path = eval_work / "trackers" / "baseline_id_oracle" / "data" / f"{sequence}.txt"
        baseline_oracle_hash = sha256_file(baseline_oracle_path)
        reference_hash = sha256_file(reference_oracle_path)
        reference_oracle_reproduction[sequence] = {
            "generated_sha256": baseline_oracle_hash,
            "reference_sha256": reference_hash,
            "byte_identical": baseline_oracle_hash == reference_hash,
        }
        post_path = eval_work / "trackers" / "postnms_replace_add_oracle" / "data" / f"{sequence}.txt"
        post_hash = sha256_file(post_path)
        expected_post_hash = m23_0_manifest["generated_track_hashes"]["expanded_replace_add_oracle"][sequence]["sha256"]
        postnms_track_reproduction[sequence] = {
            "generated_sha256": post_hash,
            "m23_0_sha256": expected_post_hash,
            "byte_identical": post_hash == expected_post_hash,
        }
        source_hashes["gt"][sequence] = {"path": str(gt_path), "sha256": sha256_file(gt_path)}

    all_baseline_reproduced_pre_eval = all(
        value["byte_identical"] for value in reference_oracle_reproduction.values()
    )
    all_postnms_reproduced_pre_eval = all(
        value["byte_identical"] for value in postnms_track_reproduction.values()
    )
    if not all_baseline_reproduced_pre_eval:
        raise RuntimeError(f"baseline ID-oracle reproduction failed: {reference_oracle_reproduction}")
    if not all_postnms_reproduced_pre_eval:
        raise RuntimeError(f"M23-0 post-NMS track reproduction failed: {postnms_track_reproduction}")

    combined_inventory = aggregate_inventory(per_sequence_stats)
    if not bool(combined_inventory["budget_passed"]):
        raise RuntimeError(f"combined fixed candidate budget exceeded: {combined_inventory}")
    frozen_combined_inventory = next(
        row
        for row in candidate_plan_report.get("inventory", [])
        if row.get("sequence") == "COMBINED"
    )
    for key in (
        "baseline_rows",
        "postnms_novel_rows",
        "raw_suppressed_rows",
        "full_survivor_rows",
        "budget_selected_rows",
        "budget_pool_rows",
    ):
        if int(combined_inventory.get(key, -1)) != int(frozen_combined_inventory.get(key, -2)):
            raise RuntimeError(f"combined candidate-plan count drift for {key}")
    if not math.isclose(
        float(combined_inventory["budget_pool_ratio"]),
        float(frozen_combined_inventory["budget_pool_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("combined candidate-plan pool-ratio drift")
    inventory_rows = list(per_sequence_stats) + [combined_inventory]
    inventory_fields = ["sequence"] + sorted({key for row in inventory_rows for key in row if key != "sequence"})
    write_csv(output_dir / "candidate_inventory.csv", inventory_rows, inventory_fields)

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (
            row["sequence"], row["frame"], row["variant"], row["gt_id"], row["source_uid"]
        ),
    )
    selected_fields = [
        "sequence", "frame", "variant", "gt_id", "source", "source_uid", "score", "iou", "operation",
        "raw_anchor_idx", "global_pre_idx", "suppressor_iou", "suppressor_score", "priority", "family_rank", "local_rank",
    ]
    write_csv(output_dir / "oracle_selected_events.csv", selected_rows, selected_fields)

    mechanism_counts = defaultdict(int)
    for row in selected_rows:
        if row.get("source") != "prenms_suppressed":
            continue
        key = (
            row["sequence"],
            row["variant"],
            score_bin(float(row["score"])),
            suppressor_iou_bin(float(row["suppressor_iou"])),
        )
        mechanism_counts[key] += 1
    mechanism_rows = [
        {
            "sequence": key[0],
            "variant": key[1],
            "score_bin": key[2],
            "suppressor_iou_bin": key[3],
            "selected_events": value,
        }
        for key, value in sorted(mechanism_counts.items())
    ]
    write_csv(
        output_dir / "suppression_mechanism_summary.csv",
        mechanism_rows,
        ["sequence", "variant", "score_bin", "suppressor_iou_bin", "selected_events"],
    )

    print("[M23-1 audit] running combined TrackEval", flush=True)
    run_trackeval(repo, gt_root, eval_work)
    metrics = {
        variant: parse_summary(eval_work / "eval" / variant / "pedestrian_summary.txt")
        for variant in VARIANTS
    }
    per_sequence_metrics: List[dict] = []
    for variant in VARIANTS:
        per_sequence_metrics.extend(
            parse_detailed(eval_work / "eval" / variant / "pedestrian_detailed.csv", variant)
        )
    metric_fields = ["variant", "sequence", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets"]
    write_csv(output_dir / "per_sequence_metrics.csv", per_sequence_metrics, metric_fields)

    baseline_metrics = metrics["baseline_raw"]
    metric_rows = []
    for variant in VARIANTS:
        row = {"variant": variant}
        for key in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets"):
            row[key] = metrics[variant].get(key)
        row["delta_vs_baseline_HOTA"] = float(metrics[variant]["HOTA"]) - float(baseline_metrics["HOTA"])
        row["delta_vs_m23_0_replace_add_HOTA"] = float(metrics[variant]["HOTA"]) - float(metrics["postnms_replace_add_oracle"]["HOTA"])
        metric_rows.append(row)
    write_csv(
        output_dir / "variant_metrics.csv",
        metric_rows,
        [
            "variant", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "CLR_FP", "CLR_FN", "Dets",
            "delta_vs_baseline_HOTA", "delta_vs_m23_0_replace_add_HOTA",
        ],
    )

    budget_sequence_rows = [
        row for row in per_sequence_metrics
        if row["variant"] == "prenms_budget_selected_ceiling" and row["sequence"] != "COMBINED"
    ]
    worst_sequence_row = min(budget_sequence_rows, key=lambda row: row["HOTA"])
    combined_budget_hota = float(metrics["prenms_budget_selected_ceiling"]["HOTA"])
    full_ceiling_hota = float(metrics["prenms_full_selected_ceiling"]["HOTA"])
    all_baseline_reproduced = all(value["byte_identical"] for value in reference_oracle_reproduction.values())
    all_postnms_reproduced = all(value["byte_identical"] for value in postnms_track_reproduction.values())
    decision = {
        "target_combined_hota": TARGET_COMBINED_HOTA,
        "target_worst_sequence_hota": TARGET_WORST_SEQUENCE_HOTA,
        "baseline_hota": float(metrics["baseline_raw"]["HOTA"]),
        "baseline_id_oracle_hota": float(metrics["baseline_id_oracle"]["HOTA"]),
        "m23_0_postnms_replace_add_hota": float(metrics["postnms_replace_add_oracle"]["HOTA"]),
        "prenms_budget_additive_hota": float(metrics["prenms_budget_additive_oracle"]["HOTA"]),
        "prenms_budget_replace_add_hota": float(metrics["prenms_budget_replace_add_oracle"]["HOTA"]),
        "prenms_budget_selected_ceiling_hota": combined_budget_hota,
        "prenms_full_selected_ceiling_hota": full_ceiling_hota,
        "worst_sequence": worst_sequence_row["sequence"],
        "worst_sequence_hota": float(worst_sequence_row["HOTA"]),
        "candidate_pool_ratio": combined_inventory["budget_pool_ratio"],
        "candidate_budget_passed": bool(combined_inventory["budget_passed"]),
        "combined_target_passed": combined_budget_hota >= TARGET_COMBINED_HOTA,
        "worst_sequence_target_passed": float(worst_sequence_row["HOTA"]) >= TARGET_WORST_SEQUENCE_HOTA,
        "baseline_oracle_reproduced": all_baseline_reproduced,
        "m23_0_postnms_tracks_reproduced": all_postnms_reproduced,
        "candidate_plan_manifest_verified": True,
        "candidate_plan_manifest_sha256": sha256_file(candidate_plan_manifest_path),
        "expanded_evidence_ceiling_sufficient": (
            combined_budget_hota >= TARGET_COMBINED_HOTA
            and float(worst_sequence_row["HOTA"]) >= TARGET_WORST_SEQUENCE_HOTA
            and bool(combined_inventory["budget_passed"])
            and all_baseline_reproduced
            and all_postnms_reproduced
        ),
        "next_stage": (
            "appearance-conditioned global tracklet graph on the fixed budgeted evidence pool"
            if (
                combined_budget_hota >= TARGET_COMBINED_HOTA
                and float(worst_sequence_row["HOTA"]) >= TARGET_WORST_SEQUENCE_HOTA
                and bool(combined_inventory["budget_passed"])
                and all_baseline_reproduced
                and all_postnms_reproduced
            )
            else "short-horizon propagation-evidence oracle audit"
        ),
        "deployment_allowed": False,
        "locked_manifest_created": False,
        "p15_policy": "no_op",
        "locked_label_reads": 0,
        "locked_trackeval_calls": 0,
        "remaining_locked_rows_untouched": 156,
    }

    report = {
        "schema": "fmtrack.m23.prenms_suppressed_evidence_oracle.v1",
        "protocol": {
            "sequences": list(SEQUENCES),
            "variants": list(VARIANTS),
            "pool_ratio_limit": POOL_RATIO_LIMIT,
            "exact_duplicate_iou": EXACT_DUPLICATE_IOU,
            "family_dedupe_iou": FAMILY_DEDUPE_IOU,
            "priority": "score * clip((0.99 - suppressor_iou) / (0.99 - nms_threshold), 0, 1)",
            "budget_order": "frame local rank, descending priority, frame, stable global index",
            "fixed_before_gt_evaluation": True,
            "candidate_plan_manifest_sha256": sha256_file(candidate_plan_manifest_path),
            "candidate_plan_recomputed_during_oracle": False,
            "reid_extracted": False,
            "selector_trained": False,
            "threshold_sweep": False,
        },
        "inventory": inventory_rows,
        "metrics": metrics,
        "per_sequence_metrics": per_sequence_metrics,
        "reference_oracle_reproduction": reference_oracle_reproduction,
        "m23_0_postnms_track_reproduction": postnms_track_reproduction,
        "decision": decision,
        "sources": source_hashes,
    }
    canonical_json_dump(report, output_dir / "report.json")

    generated_track_hashes = {
        variant: {
            sequence: {
                "sha256": sha256_file(eval_work / "trackers" / variant / "data" / f"{sequence}.txt"),
                "size_bytes": (eval_work / "trackers" / variant / "data" / f"{sequence}.txt").stat().st_size,
            }
            for sequence in SEQUENCES
        }
        for variant in VARIANTS
    }
    compact_files = (
        "candidate_inventory.csv",
        "oracle_selected_events.csv",
        "suppression_mechanism_summary.csv",
        "per_sequence_metrics.csv",
        "variant_metrics.csv",
        "report.json",
    )
    compact_hashes = {name: sha256_file(output_dir / name) for name in compact_files}
    manifest = {
        "schema": "fmtrack.m23.prenms_suppressed_evidence_oracle.manifest.v1",
        "decision": decision,
        "generated_track_hashes": generated_track_hashes,
        "trackeval_summary_hashes": {
            variant: sha256_file(eval_work / "eval" / variant / "pedestrian_summary.txt")
            for variant in VARIANTS
        },
        "candidate_plan_manifest_sha256": sha256_file(candidate_plan_manifest_path),
        "compact_file_hashes": compact_hashes,
        "report_sha256": compact_hashes["report.json"],
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
