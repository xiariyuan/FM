#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
BOTSORT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BOTSORT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOTSORT_ROOT))

from fast_reid.fast_reid_interfece import FastReIDInterface
from tracker import matching
from tracker.bot_sort import STrack
from tracker.kalman_filter import KalmanFilter

from projects.fcaa.fcaa.model.freq_dwt import BandDescriptor, cosine_similarity, extract_band_descriptor


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

SUMMARY_FIELDS = [
    "dataset_name",
    "dataset_root",
    "benchmark",
    "split",
    "subset",
    "grouping",
    "sequences",
    "rows",
    "groups",
    "ambiguous_groups",
    "multi_candidate_groups",
    "top1_conflict_tracks",
    "topk_conflict_tracks",
    "top1_conflict_detections",
    "topk_conflict_detections",
    "positive_rows",
    "negative_rows",
    "top_k",
    "track_high_thresh",
    "proximity_thresh",
    "appearance_thresh",
    "ambiguity_margin",
    "label_iou_thresh",
    "reid_config",
    "reid_weights",
    "status",
    "error",
]


@dataclass
class GTBox:
    gt_id: int
    tlbr: np.ndarray
    visibility: float


@dataclass
class DetBox:
    det_id: int
    tlbr: np.ndarray
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an FCAA pair-bank from detector-driven BoT-SORT candidates.")
    parser.add_argument("--dataset-root", default="/gemini/code/datasets")
    parser.add_argument("--benchmark", default="MOT17")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", choices=["full", "train_half", "val_half"], default="full")
    parser.add_argument("--grouping", choices=["row", "shared_det_top1"], default="row")
    parser.add_argument("--seq-names", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="fcaa_mot17_pairbank")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--track-high-thresh", type=float, default=0.6)
    parser.add_argument("--proximity-thresh", type=float, default=0.5)
    parser.add_argument("--appearance-thresh", type=float, default=0.25)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--label-iou-thresh", type=float, default=0.5)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-visibility", type=float, default=0.0)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--reid-config", default=str(BOTSORT_ROOT / "fast_reid/configs/MOT17/sbs_S50.yml"))
    parser.add_argument("--reid-weights", default=str(BOTSORT_ROOT / "pretrained/mot17_sbs_S50.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, run_root: Path, status: str, notes: str = "") -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/build_fcaa_pairbank.py",
        "--dataset",
        args.benchmark,
        "--split",
        f"{args.split}:{args.subset}",
        "--tracker-family",
        "botsort_fcaa",
        "--variant",
        args.dataset_name,
        "--tag",
        args.dataset_name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, check=False)


def _parse_box(parts: Sequence[str], start_idx: int = 2) -> np.ndarray:
    x, y, w, h = [float(parts[idx]) for idx in range(start_idx, start_idx + 4)]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def load_gt(path: Path, min_visibility: float) -> Dict[int, List[GTBox]]:
    frames: Dict[int, List[GTBox]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame_id = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            mark = int(float(parts[6]))
            cls = int(float(parts[7]))
            visibility = float(parts[8])
            if mark <= 0 or cls != 1 or visibility < float(min_visibility):
                continue
            frames[frame_id].append(
                GTBox(
                    gt_id=gt_id,
                    tlbr=_parse_box(parts),
                    visibility=visibility,
                )
            )
    return frames


def load_detections(path: Path) -> Dict[int, List[DetBox]]:
    frames: Dict[int, List[DetBox]] = defaultdict(list)
    det_id = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_id = int(float(parts[0]))
            score = float(parts[6])
            tlbr = _parse_box(parts)
            if score <= 0.0 or (tlbr[2] - tlbr[0]) <= 1.0 or (tlbr[3] - tlbr[1]) <= 1.0:
                continue
            frames[frame_id].append(
                DetBox(
                    det_id=det_id,
                    tlbr=tlbr,
                    score=score,
                )
            )
            det_id += 1
    for items in frames.values():
        items.sort(key=lambda det: float(det.score), reverse=True)
    return frames


def count_unique_frames(path: Path) -> int:
    frame_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if parts and parts[0]:
                frame_ids.add(int(float(parts[0])))
    return len(frame_ids)


def resolve_subset_paths(seq_dir: Path, subset: str) -> Tuple[Path, Path, int]:
    if subset == "full":
        return seq_dir / "gt" / "gt.txt", seq_dir / "det" / "det.txt", 0
    if subset == "train_half":
        return seq_dir / "gt" / "gt_train_half.txt", seq_dir / "det" / "det_train_half.txt", 0
    train_half_gt = seq_dir / "gt" / "gt_train_half.txt"
    train_half_det = seq_dir / "det" / "det_train_half.txt"
    train_frame_count = count_unique_frames(train_half_gt if train_half_gt.is_file() else train_half_det)
    return seq_dir / "gt" / "gt_val_half.txt", seq_dir / "det" / "det_val_half.txt", train_frame_count


def load_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def encode_detections(
    encoder: FastReIDInterface,
    image: np.ndarray,
    detections: Sequence[DetBox],
    *,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, List[BandDescriptor]]:
    boxes = np.asarray([det.tlbr for det in detections], dtype=np.float32)
    reid_feats = encoder.inference(image, boxes)
    band_descs = [
        extract_band_descriptor(
            image,
            det.tlbr,
            image_height=image_height,
            image_width=image_width,
        )
        for det in detections
    ]
    return np.asarray(reid_feats, dtype=np.float32), band_descs


def normalize_features(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features.astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return (features / norms).astype(np.float32)


def build_detection_tracks(
    detections: Sequence[DetBox],
    reid_feats: np.ndarray,
    band_descs: Sequence[BandDescriptor],
) -> List[STrack]:
    det_tracks: List[STrack] = []
    for det, feat, desc in zip(detections, reid_feats, band_descs):
        det_track = STrack(STrack.tlbr_to_tlwh(det.tlbr), det.score, feat.astype(np.float32))
        det_track.fcaa_band_desc = desc
        det_track.update_fcaa_bands(desc, momentum=0.0)
        det_tracks.append(det_track)
    return det_tracks


def assign_detection_labels(
    detections: Sequence[DetBox],
    gt_boxes: Sequence[GTBox],
    label_iou_thresh: float,
) -> Tuple[List[int], List[float]]:
    if not detections:
        return [], []
    if not gt_boxes:
        return [-1 for _ in detections], [0.0 for _ in detections]
    det_tlbrs = np.asarray([det.tlbr for det in detections], dtype=np.float32)
    gt_tlbrs = np.asarray([gt.tlbr for gt in gt_boxes], dtype=np.float32)
    iou_matrix = matching.ious(det_tlbrs, gt_tlbrs)
    det_gt_ids: List[int] = []
    det_gt_ious: List[float] = []
    for det_idx in range(iou_matrix.shape[0]):
        gt_idx = int(np.argmax(iou_matrix[det_idx]))
        best_iou = float(iou_matrix[det_idx, gt_idx])
        if best_iou >= float(label_iou_thresh):
            det_gt_ids.append(int(gt_boxes[gt_idx].gt_id))
            det_gt_ious.append(best_iou)
        else:
            det_gt_ids.append(-1)
            det_gt_ious.append(best_iou)
    return det_gt_ids, det_gt_ious


def select_update_candidates(det_gt_ids: Sequence[int], det_gt_ious: Sequence[float], detections: Sequence[DetBox]) -> Dict[int, int]:
    chosen: Dict[int, int] = {}
    for det_idx, gt_id in enumerate(det_gt_ids):
        if int(gt_id) <= 0:
            continue
        current = chosen.get(int(gt_id))
        if current is None:
            chosen[int(gt_id)] = int(det_idx)
            continue
        current_key = (float(det_gt_ious[current]), float(detections[current].score))
        candidate_key = (float(det_gt_ious[det_idx]), float(detections[det_idx].score))
        if candidate_key > current_key:
            chosen[int(gt_id)] = int(det_idx)
    return chosen


def expire_old_tracks(track_memory: Dict[int, STrack], frame_id: int, track_buffer: int) -> None:
    expired = [
        gt_id
        for gt_id, track in track_memory.items()
        if int(frame_id) - int(getattr(track, "frame_id", frame_id)) > int(track_buffer)
    ]
    for gt_id in expired:
        del track_memory[gt_id]


def image_path_for_frame(seq_dir: Path, image_frame_id: int) -> Path:
    candidate = seq_dir / "img1" / f"{int(image_frame_id):06d}.jpg"
    if candidate.is_file():
        return candidate
    candidate = seq_dir / "img1" / f"{int(image_frame_id):08d}.jpg"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Unable to resolve image for frame {image_frame_id} under {seq_dir / 'img1'}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    pairbank_jsonl = out_dir / "pairbank.jsonl"
    status_row = {
        "dataset_name": args.dataset_name,
        "dataset_root": str(Path(args.dataset_root) / args.benchmark / args.split),
        "benchmark": args.benchmark,
        "split": args.split,
        "subset": args.subset,
        "grouping": args.grouping,
        "sequences": "",
        "rows": 0,
        "groups": 0,
        "ambiguous_groups": 0,
        "multi_candidate_groups": 0,
        "top1_conflict_tracks": 0,
        "topk_conflict_tracks": 0,
        "top1_conflict_detections": 0,
        "topk_conflict_detections": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "top_k": args.top_k,
        "track_high_thresh": args.track_high_thresh,
        "proximity_thresh": args.proximity_thresh,
        "appearance_thresh": args.appearance_thresh,
        "ambiguity_margin": args.ambiguity_margin,
        "label_iou_thresh": args.label_iou_thresh,
        "reid_config": args.reid_config,
        "reid_weights": args.reid_weights,
        "status": "running",
        "error": "",
    }
    write_rows(summary_csv, SUMMARY_FIELDS, [status_row])

    rows: List[Dict[str, object]] = []
    try:
        dataset_dir = Path(args.dataset_root) / args.benchmark / args.split
        seq_names = list(args.seq_names) if args.seq_names else sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
        encoder = FastReIDInterface(args.reid_config, args.reid_weights, args.device)
        total_groups = 0
        ambiguous_groups = 0
        multi_candidate_groups = 0
        top1_conflict_tracks = 0
        topk_conflict_tracks = 0
        top1_conflict_detections = 0
        topk_conflict_detections = 0
        shared_kalman = KalmanFilter()

        for seq_name in seq_names:
            seq_dir = dataset_dir / seq_name
            gt_path, det_path, image_offset = resolve_subset_paths(seq_dir, str(args.subset))
            gt_frames = load_gt(gt_path, float(args.min_visibility))
            det_frames = load_detections(det_path)
            frame_ids = sorted(set(gt_frames.keys()) | set(det_frames.keys()))
            if int(args.max_frames) > 0:
                frame_ids = frame_ids[: int(args.max_frames)]
            if not frame_ids:
                continue
            print(
                f"[fcaa-pairbank] seq={seq_name} subset={args.subset} frames={len(frame_ids)} "
                f"det_file={det_path.name} gt_file={gt_path.name}",
                flush=True,
            )
            verbose_every_frame = len(frame_ids) <= 50
            track_memory: Dict[int, STrack] = {}
            for frame_offset_idx, frame_id in enumerate(frame_ids, start=1):
                expire_old_tracks(track_memory, int(frame_id), int(args.track_buffer))
                if verbose_every_frame or frame_offset_idx == 1 or frame_offset_idx % 100 == 0 or frame_offset_idx == len(frame_ids):
                    print(
                        f"[fcaa-pairbank] seq={seq_name} frame={frame_id} progress={frame_offset_idx}/{len(frame_ids)} "
                        f"rows={len(rows)} groups={total_groups} live_tracks={len(track_memory)}",
                        flush=True,
                    )

                image_frame_id = int(frame_id) + int(image_offset)
                image_path = image_path_for_frame(seq_dir, image_frame_id)
                frame_dets_all = det_frames.get(int(frame_id), [])
                frame_dets = [det for det in frame_dets_all if float(det.score) > float(args.track_high_thresh)]
                frame_gts = gt_frames.get(int(frame_id), [])
                det_gt_ids: List[int] = []
                det_gt_ious: List[float] = []
                det_reid = np.zeros((0, 0), dtype=np.float32)
                det_bands: List[BandDescriptor] = []
                det_tracks: List[STrack] = []

                if frame_dets:
                    image = load_frame(image_path)
                    det_reid, det_bands = encode_detections(
                        encoder,
                        image,
                        frame_dets,
                        image_height=int(args.image_height),
                        image_width=int(args.image_width),
                    )
                    det_reid = normalize_features(det_reid)
                    det_gt_ids, det_gt_ious = assign_detection_labels(frame_dets, frame_gts, float(args.label_iou_thresh))
                    det_tracks = build_detection_tracks(frame_dets, det_reid, det_bands)
                if verbose_every_frame:
                    print(
                        f"[fcaa-pairbank] seq={seq_name} frame={frame_id} "
                        f"high_dets={len(frame_dets)} positives={sum(int(gt_id > 0) for gt_id in det_gt_ids)}",
                        flush=True,
                    )

                active_gt_ids = sorted(
                    gt_id
                    for gt_id, track in track_memory.items()
                    if int(frame_id) - int(getattr(track, "frame_id", frame_id)) <= int(args.track_buffer)
                )
                track_pool = [track_memory[gt_id] for gt_id in active_gt_ids]

                if track_pool and det_tracks:
                    STrack.multi_predict(track_pool)
                    raw_ious = matching.iou_distance(track_pool, det_tracks)
                    invalid_mask = raw_ious > float(args.proximity_thresh)
                    fused_iou_cost = raw_ious.copy()
                    if str(args.benchmark).upper() != "MOT20":
                        fused_iou_cost = matching.fuse_score(fused_iou_cost, det_tracks)
                    emb_dists = matching.embedding_distance(track_pool, det_tracks) / 2.0
                    clipped_emb = emb_dists.copy()
                    clipped_emb[clipped_emb > float(args.appearance_thresh)] = 1.0
                    clipped_emb[invalid_mask] = 1.0
                    base_cost = np.minimum(fused_iou_cost, clipped_emb)
                    base_cost[invalid_mask] = 1.0
                    base_similarity = 1.0 - base_cost
                    frame_top1_counts: Dict[int, int] = defaultdict(int)
                    frame_topk_counts: Dict[int, int] = defaultdict(int)
                    row_groups: List[Dict[str, object]] = []

                    for track_idx, gt_id in enumerate(active_gt_ids):
                        valid_det_idx = np.where(~invalid_mask[track_idx])[0]
                        if valid_det_idx.size == 0:
                            continue
                        order = valid_det_idx[np.argsort(base_cost[track_idx, valid_det_idx])]
                        order = order[: max(1, int(args.top_k))]
                        if order.size == 0:
                            continue
                        frame_top1_counts[int(order[0])] += 1
                        for det_idx in order.tolist():
                            frame_topk_counts[int(det_idx)] += 1
                        best = float(base_similarity[track_idx, order[0]])
                        second = float(base_similarity[track_idx, order[1]]) if order.size > 1 else 0.0
                        ambiguous_flag = int(order.size > 1 and (best - second) < float(args.ambiguity_margin))
                        if order.size > 1:
                            multi_candidate_groups += 1
                        row_groups.append(
                            {
                                "track_idx": int(track_idx),
                                "track_gt_id": int(gt_id),
                                "order": [int(det_idx) for det_idx in order.tolist()],
                                "ambiguous_flag": int(ambiguous_flag),
                            }
                        )
                    frame_top1_conflicts = [count for count in frame_top1_counts.values() if int(count) > 1]
                    frame_topk_conflicts = [count for count in frame_topk_counts.values() if int(count) > 1]
                    top1_conflict_detections += len(frame_top1_conflicts)
                    topk_conflict_detections += len(frame_topk_conflicts)
                    top1_conflict_tracks += int(sum(count for count in frame_top1_conflicts))
                    topk_conflict_tracks += int(sum(count for count in frame_topk_conflicts))

                    if str(args.grouping) == "row":
                        for group in row_groups:
                            gt_id = int(group["track_gt_id"])
                            order = list(group["order"])
                            if not any(int(det_gt_ids[det_idx]) == int(gt_id) for det_idx in order):
                                continue
                            total_groups += 1
                            if int(group["ambiguous_flag"]):
                                ambiguous_groups += 1
                            group_key = f"{seq_name}:{image_frame_id}:{gt_id}"
                            track = track_memory[gt_id]
                            track_idx = int(group["track_idx"])
                            for rank, det_idx in enumerate(order):
                                det = frame_dets[det_idx]
                                bands = det_bands[det_idx]
                                rows.append(
                                    {
                                        "group_key": group_key,
                                        "seq_name": seq_name,
                                        "frame_id": int(image_frame_id),
                                        "subset_frame_id": int(frame_id),
                                        "track_gt_id": int(gt_id),
                                        "det_gt_id": int(det_gt_ids[det_idx]),
                                        "det_gt_iou": float(det_gt_ious[det_idx]),
                                        "candidate_rank": int(rank),
                                        "label": int(int(det_gt_ids[det_idx]) == int(gt_id)),
                                        "ambiguous_flag": int(group["ambiguous_flag"]),
                                        "det_score": float(det.score),
                                        "s_reid": float(1.0 - emb_dists[track_idx, det_idx]),
                                        "s_low": cosine_similarity(np.asarray(track.fcaa_low), bands.low),
                                        "s_mid": cosine_similarity(np.asarray(track.fcaa_mid), bands.mid),
                                        "s_high": cosine_similarity(np.asarray(track.fcaa_high), bands.high),
                                        "base_cost": float(base_cost[track_idx, det_idx]),
                                        "base_similarity": float(base_similarity[track_idx, det_idx]),
                                        "raw_iou_cost": float(raw_ious[track_idx, det_idx]),
                                        "fused_iou_cost": float(fused_iou_cost[track_idx, det_idx]),
                                        "track_age": int(frame_id) - int(getattr(track, "frame_id", frame_id)),
                                        "det_source": det_path.name,
                                        "track_source": "gt_aligned_pseudo_track",
                                        "track_box": [float(v) for v in track.tlbr.tolist()],
                                        "det_box": [float(v) for v in det.tlbr.tolist()],
                                    }
                                )
                    else:
                        shared_det_groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
                        for group in row_groups:
                            order = list(group["order"])
                            if not order:
                                continue
                            shared_det_groups[int(order[0])].append(group)
                        for det_idx, contenders in shared_det_groups.items():
                            if len(contenders) < 2:
                                continue
                            det_gt_id = int(det_gt_ids[det_idx])
                            if det_gt_id <= 0:
                                continue
                            if not any(int(group["track_gt_id"]) == det_gt_id for group in contenders):
                                continue
                            total_groups += 1
                            ambiguous_groups += 1
                            det = frame_dets[det_idx]
                            bands = det_bands[det_idx]
                            ordered_contenders = sorted(
                                contenders,
                                key=lambda group: float(base_similarity[int(group["track_idx"]), det_idx]),
                                reverse=True,
                            )
                            group_key = f"{seq_name}:{image_frame_id}:shared_det_top1:{det_idx}"
                            for rank, group in enumerate(ordered_contenders):
                                gt_id = int(group["track_gt_id"])
                                track_idx = int(group["track_idx"])
                                track = track_memory[gt_id]
                                rows.append(
                                    {
                                        "group_key": group_key,
                                        "seq_name": seq_name,
                                        "frame_id": int(image_frame_id),
                                        "subset_frame_id": int(frame_id),
                                        "track_gt_id": int(gt_id),
                                        "det_gt_id": int(det_gt_id),
                                        "det_gt_iou": float(det_gt_ious[det_idx]),
                                        "candidate_rank": int(rank),
                                        "label": int(det_gt_id == gt_id),
                                        "ambiguous_flag": 1,
                                        "det_score": float(det.score),
                                        "s_reid": float(1.0 - emb_dists[track_idx, det_idx]),
                                        "s_low": cosine_similarity(np.asarray(track.fcaa_low), bands.low),
                                        "s_mid": cosine_similarity(np.asarray(track.fcaa_mid), bands.mid),
                                        "s_high": cosine_similarity(np.asarray(track.fcaa_high), bands.high),
                                        "base_cost": float(base_cost[track_idx, det_idx]),
                                        "base_similarity": float(base_similarity[track_idx, det_idx]),
                                        "raw_iou_cost": float(raw_ious[track_idx, det_idx]),
                                        "fused_iou_cost": float(fused_iou_cost[track_idx, det_idx]),
                                        "track_age": int(frame_id) - int(getattr(track, "frame_id", frame_id)),
                                        "det_source": det_path.name,
                                        "track_source": "gt_aligned_pseudo_track",
                                        "track_box": [float(v) for v in track.tlbr.tolist()],
                                        "det_box": [float(v) for v in det.tlbr.tolist()],
                                    }
                                )

                update_candidates = select_update_candidates(det_gt_ids, det_gt_ious, frame_dets)
                for gt_id, det_idx in update_candidates.items():
                    det_track = det_tracks[det_idx]
                    det_track.analysis_gt_id = int(gt_id)
                    if gt_id in track_memory and int(frame_id) - int(getattr(track_memory[gt_id], "frame_id", frame_id)) <= int(args.track_buffer):
                        track_memory[gt_id].update(det_track, int(frame_id))
                        track_memory[gt_id].analysis_gt_id = int(gt_id)
                    else:
                        det_track.activate(shared_kalman, int(frame_id))
                        det_track.analysis_gt_id = int(gt_id)
                        track_memory[gt_id] = det_track

        with pairbank_jsonl.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        status_row.update(
            {
                "sequences": " ".join(seq_names),
                "rows": int(len(rows)),
                "groups": int(total_groups),
                "ambiguous_groups": int(ambiguous_groups),
                "multi_candidate_groups": int(multi_candidate_groups),
                "top1_conflict_tracks": int(top1_conflict_tracks),
                "topk_conflict_tracks": int(topk_conflict_tracks),
                "top1_conflict_detections": int(top1_conflict_detections),
                "topk_conflict_detections": int(topk_conflict_detections),
                "positive_rows": int(sum(int(row["label"]) for row in rows)),
                "negative_rows": int(sum(1 - int(row["label"]) for row in rows)),
                "status": "success",
                "error": "",
            }
        )
        print(
            f"[fcaa-pairbank] finished rows={len(rows)} groups={total_groups} ambiguous_groups={ambiguous_groups}",
            flush=True,
        )
        write_rows(summary_csv, SUMMARY_FIELDS, [status_row])
        append_registry(args, summary_csv, out_dir, "success", notes="fcaa pair-bank build")
    except Exception as exc:
        status_row["status"] = "failed"
        status_row["error"] = str(exc)
        write_rows(summary_csv, SUMMARY_FIELDS, [status_row])
        append_registry(args, summary_csv, out_dir, "failed", notes=f"fcaa pair-bank build failed: {exc}")
        raise


if __name__ == "__main__":
    main()
