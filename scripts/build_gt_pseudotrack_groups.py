#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import csv
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPO_ROOT / "external/BoT-SORT-main"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: E402


PAIR_CSV_FIELDNAMES = [
    "seq",
    "frame",
    "assoc_stage",
    "track_id",
    "det_index",
    "gap",
    "history_len",
    "chosen",
    "is_true_match",
    "track_gt_id",
    "det_gt_id",
    "pair_rel",
    "learned_alpha",
    "learned_r",
    "appearance_sim",
    "fused_sim",
    "motion_sim",
    "spatial_sim",
    "laplace_sim",
    "agreement",
    "stability",
    "coherence",
    "det_score",
    "prod_sim",
    "amb_spa",
    "amb_lap",
    "amb_mot",
]


@dataclass
class Detection:
    frame: int
    det_index: int
    tlwh: np.ndarray
    tlbr: np.ndarray
    score: float
    feat: np.ndarray
    gt_id: int
    ignore: bool


@dataclass
class PseudoTrackState:
    gt_id: int
    frames: deque
    boxes_tlbr: deque
    feats: deque
    smooth_feat: np.ndarray | None = None

    def update(self, frame: int, tlbr: np.ndarray, feat: np.ndarray, smooth_alpha: float) -> None:
        self.frames.append(int(frame))
        self.boxes_tlbr.append(np.asarray(tlbr, dtype=np.float32))
        feat = _normalize_rows(np.asarray(feat, dtype=np.float32))[0]
        self.feats.append(feat)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = smooth_alpha * self.smooth_feat + (1.0 - smooth_alpha) * feat
            self.smooth_feat = self.smooth_feat / np.clip(np.linalg.norm(self.smooth_feat), 1e-12, None)

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1])

    @property
    def last_box_tlbr(self) -> np.ndarray:
        return np.asarray(self.boxes_tlbr[-1], dtype=np.float32)

    @property
    def history_feats(self) -> np.ndarray:
        if not self.feats:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(list(self.feats), axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GT pseudo-track training groups for current alpha/r LTRA and future learnable pole-bank LTRA."
    )
    parser.add_argument("--dataset", choices=["MOT17", "MOT20"], required=True)
    parser.add_argument("--data-root", default="/gemini/code/datasets", help="Root containing MOT17/ or MOT20/")
    parser.add_argument("--seqs", nargs="+", required=True, help="Sequence names, e.g. MOT17-02-FRCNN MOT17-04-FRCNN")
    parser.add_argument(
        "--split-part",
        choices=["full", "train_half", "val_half"],
        default="full",
        help="Use full det/gt files or the MOT half-split files.",
    )
    parser.add_argument("--fast-reid-config", required=True)
    parser.add_argument("--fast-reid-weights", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="FastReID crop batch size")
    parser.add_argument("--max-history", type=int, default=30, help="Max stored history features per GT pseudo-track")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum history before multi-scale Laplace prototypes are used")
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16", help="Storage dtype for det_feat/hist_feat in the output NPZ")
    parser.add_argument("--seed", type=int, default=123, help="Random seed used for negative subsampling")
    parser.add_argument("--smooth-alpha", type=float, default=0.9, help="EMA alpha for pseudo-track smooth appearance")
    parser.add_argument("--laplace-decay-scales", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--iou-pos", type=float, default=0.7, help="Detection-to-GT IoU threshold for positive assignment")
    parser.add_argument("--iou-ignore", type=float, default=0.5, help="IoU threshold for ignore GT overlap")
    parser.add_argument("--max-gap", type=int, default=30, help="Maximum allowed frame gap from last GT history")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick smoke tests (0 means all frames)")
    parser.add_argument("--candidate-topk", type=int, default=32, help="Maximum pseudo-track candidates kept per detection")
    parser.add_argument("--max-hard-negatives", type=int, default=4, help="Maximum hard negatives retained per detection group")
    parser.add_argument("--max-random-negatives", type=int, default=2, help="Maximum additional random negatives retained per detection group")
    parser.add_argument(
        "--candidate-min-motion",
        type=float,
        default=0.0,
        help="Keep negative candidates whose last-box IoU with the detection is at least this value; positive is always kept if present.",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="Keep background groups with no positive candidate. Recommended for training.",
    )
    parser.add_argument("--out-csv", default="", help="Optional pair-style CSV output compatible with train_ltra_calibrator_from_pairs.py")
    parser.add_argument("--out-npz", default="", help="Optional NPZ output for future pole-bank training")
    return parser.parse_args()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def _cosine_sim01(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_rows(np.asarray(a, dtype=np.float32))[0]
    b = _normalize_rows(np.asarray(b, dtype=np.float32))[0]
    return float(np.clip((np.dot(a, b) + 1.0) * 0.5, 0.0, 1.0))


def _tlwh_to_tlbr(tlwh: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in tlwh]
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def _box_iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.zeros((0,), dtype=np.float32)
    box = np.asarray(box, dtype=np.float32).reshape(1, 4)
    others = np.asarray(others, dtype=np.float32).reshape(-1, 4)
    x1 = np.maximum(box[:, 0], others[:, 0])
    y1 = np.maximum(box[:, 1], others[:, 1])
    x2 = np.minimum(box[:, 2], others[:, 2])
    y2 = np.minimum(box[:, 3], others[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(box[:, 2] - box[:, 0], 0.0, None) * np.clip(box[:, 3] - box[:, 1], 0.0, None)
    area_b = np.clip(others[:, 2] - others[:, 0], 0.0, None) * np.clip(others[:, 3] - others[:, 1], 0.0, None)
    union = np.clip(area_a + area_b - inter, 1e-12, None)
    return (inter / union).astype(np.float32).reshape(-1)


def _build_exp_prototype(hist: np.ndarray, tau: float) -> np.ndarray:
    hist = _normalize_rows(hist)
    length = int(hist.shape[0])
    age = np.arange(length - 1, -1, -1, dtype=np.float32)
    tau = max(float(tau), 1e-3)
    w = np.exp(-age / tau).astype(np.float32)
    w = w / np.clip(w.sum(), 1e-12, None)
    proto = (w[:, None] * hist).sum(axis=0)
    return _normalize_rows(proto)[0]


def _track_stability(hist: np.ndarray) -> float:
    hist = _normalize_rows(hist)
    if hist.shape[0] < 3:
        return 1.0
    delta2 = hist[2:] - 2.0 * hist[1:-1] + hist[:-2]
    curvature = np.sqrt(np.mean(delta2 ** 2, axis=1)).mean()
    return float(np.exp(-curvature))


def _build_laplace_stats(hist: np.ndarray, decay_scales: Sequence[float], min_history: int) -> tuple[np.ndarray, float, float]:
    hist = _normalize_rows(hist)
    last = hist[-1]
    if hist.shape[0] < max(int(min_history), 1):
        return last[None, :], 1.0, 1.0
    protos = []
    proto_sims = []
    for scale in decay_scales:
        proto = _build_exp_prototype(hist, tau=float(scale))
        protos.append(proto)
        proto_sims.append(((proto * last).sum() + 1.0) * 0.5)
    protos = np.stack(protos, axis=0).astype(np.float32)
    stability = _track_stability(hist)
    coherence = float(np.mean(proto_sims))
    return protos, stability, coherence


def _agreement(spatial_sim: float, laplace_sim: float) -> float:
    return float(np.clip(1.0 - abs(float(spatial_sim) - float(laplace_sim)), 0.0, 1.0))


def _margin(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(np.clip(values[0], 0.0, 1.0))
    top2 = sorted((float(v) for v in values), reverse=True)[:2]
    return float(np.clip(top2[0] - top2[1], 0.0, 1.0))


def _heuristic_pair_rel(stability: float, coherence: float, agreement: float, det_score: float) -> float:
    pair_rel = 0.35 * float(stability) + 0.35 * float(coherence) + 0.30 * float(agreement)
    pair_rel = pair_rel * (0.5 + 0.5 * np.clip(float(det_score), 0.0, 1.0))
    return float(np.clip(pair_rel, 0.0, 1.0))


def _subsample_group_records(
    group_records: Sequence[dict],
    max_hard_negatives: int,
    max_random_negatives: int,
    rng: np.random.Generator,
) -> list[dict]:
    positives = [rec for rec in group_records if int(rec["label"]) == 1]
    negatives = [rec for rec in group_records if int(rec["label"]) == 0]
    if not negatives:
        return list(group_records)

    negatives.sort(
        key=lambda rec: max(float(rec["spatial_sim"]), float(rec["laplace_sim"]), float(rec["motion_sim"])),
        reverse=True,
    )

    keep_negatives: list[dict] = []
    hard_keep = max(int(max_hard_negatives), 0)
    rand_keep = max(int(max_random_negatives), 0)

    if hard_keep > 0:
        keep_negatives.extend(negatives[:hard_keep])
    remaining = negatives[hard_keep:]
    if rand_keep > 0 and remaining:
        choose_count = min(rand_keep, len(remaining))
        choose_idx = np.sort(rng.choice(len(remaining), size=choose_count, replace=False))
        keep_negatives.extend([remaining[int(idx)] for idx in choose_idx])

    selected = positives + keep_negatives
    if not selected:
        selected = keep_negatives
    selected.sort(key=lambda rec: float(rec["motion_sim"]), reverse=True)
    return selected


def _history_file_name(base_dir: Path, kind: str, split_part: str) -> Path:
    if kind not in {"det", "gt"}:
        raise ValueError(kind)
    if split_part == "full":
        return base_dir / kind / f"{kind}.txt"
    suffix = "train_half" if split_part == "train_half" else "val_half"
    return base_dir / kind / f"{kind}_{suffix}.txt"


def _read_seqinfo(seq_dir: Path) -> dict[str, str]:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    if "Sequence" not in parser:
        raise ValueError(f"Invalid seqinfo.ini in {seq_dir}")
    return dict(parser["Sequence"])


def _read_det_rows(det_path: Path) -> dict[int, list[tuple[int, np.ndarray, float]]]:
    frames: dict[int, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    with det_path.open("r") as f:
        for det_index, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(float(parts[0]))
            tlwh = np.asarray([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])], dtype=np.float32)
            score = float(parts[6])
            frames[frame].append((det_index, tlwh, score))
    return frames


def _read_gt_rows(gt_path: Path) -> dict[int, list[dict[str, float]]]:
    frames: dict[int, list[dict[str, float]]] = defaultdict(list)
    with gt_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame = int(float(parts[0]))
            row = {
                "frame": frame,
                "id": int(float(parts[1])),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "w": float(parts[4]),
                "h": float(parts[5]),
                "conf": float(parts[6]),
                "class": int(float(parts[7])),
                "vis": float(parts[8]) if len(parts) >= 9 else 1.0,
            }
            frames[frame].append(row)
    return frames


def _split_gt_frame(rows: Iterable[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    positives = []
    ignores = []
    for row in rows:
        if int(row["class"]) == 1 and float(row["conf"]) > 0.5:
            positives.append(row)
        else:
            ignores.append(row)
    return positives, ignores


def _assign_detections_to_gt(
    det_tlbrs: np.ndarray,
    gt_positive: Sequence[dict[str, float]],
    gt_ignore: Sequence[dict[str, float]],
    iou_pos: float,
    iou_ignore: float,
) -> tuple[np.ndarray, np.ndarray]:
    assigned_gt = np.full((det_tlbrs.shape[0],), -1, dtype=np.int32)
    ignore_mask = np.zeros((det_tlbrs.shape[0],), dtype=bool)

    if gt_positive:
        gt_boxes = np.asarray(
            [[row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]] for row in gt_positive],
            dtype=np.float32,
        )
        all_pairs = []
        for det_idx in range(det_tlbrs.shape[0]):
            ious = _box_iou_one_to_many(det_tlbrs[det_idx], gt_boxes)
            for gt_idx, iou in enumerate(ious):
                if float(iou) >= float(iou_pos):
                    all_pairs.append((float(iou), det_idx, gt_idx))
        all_pairs.sort(reverse=True)
        used_det = set()
        used_gt = set()
        for _, det_idx, gt_idx in all_pairs:
            if det_idx in used_det or gt_idx in used_gt:
                continue
            used_det.add(det_idx)
            used_gt.add(gt_idx)
            assigned_gt[det_idx] = int(gt_positive[gt_idx]["id"])

    if gt_ignore:
        ignore_boxes = np.asarray(
            [[row["x"], row["y"], row["x"] + row["w"], row["y"] + row["h"]] for row in gt_ignore],
            dtype=np.float32,
        )
        for det_idx in range(det_tlbrs.shape[0]):
            if assigned_gt[det_idx] > 0:
                continue
            ious = _box_iou_one_to_many(det_tlbrs[det_idx], ignore_boxes)
            if ious.size and float(np.max(ious)) >= float(iou_ignore):
                ignore_mask[det_idx] = True

    return assigned_gt, ignore_mask


def _choose_candidates(
    histories: Dict[int, PseudoTrackState],
    det_tlbr: np.ndarray,
    det_gt_id: int,
    frame: int,
    max_gap: int,
    candidate_topk: int,
    candidate_min_motion: float,
) -> list[tuple[int, float]]:
    active = []
    for gt_id, state in histories.items():
        gap = int(frame) - int(state.last_frame)
        if gap <= 0 or gap > int(max_gap):
            continue
        motion_sim = float(_box_iou_one_to_many(det_tlbr, state.last_box_tlbr.reshape(1, 4))[0])
        active.append((gt_id, motion_sim))

    if not active:
        return []

    active.sort(key=lambda x: x[1], reverse=True)
    chosen: list[tuple[int, float]] = []
    positive_item = None
    for gt_id, motion_sim in active:
        if det_gt_id > 0 and gt_id == det_gt_id:
            positive_item = (gt_id, motion_sim)
            break

    for gt_id, motion_sim in active:
        if motion_sim < float(candidate_min_motion):
            continue
        chosen.append((gt_id, motion_sim))
        if len(chosen) >= int(candidate_topk):
            break

    if positive_item is not None and all(gt_id != positive_item[0] for gt_id, _ in chosen):
        if len(chosen) >= int(candidate_topk) and chosen:
            chosen = chosen[:-1]
        chosen.append(positive_item)

    chosen.sort(key=lambda x: x[1], reverse=True)
    return chosen


def _frame_image_path(seq_dir: Path, frame: int, imext: str) -> Path:
    return seq_dir / "img1" / f"{frame:06d}{imext}"


def _collect_detections_for_frame(
    encoder: FastReIDInterface,
    image: np.ndarray,
    frame: int,
    rows: Sequence[tuple[int, np.ndarray, float]],
    gt_positive: Sequence[dict[str, float]],
    gt_ignore: Sequence[dict[str, float]],
    iou_pos: float,
    iou_ignore: float,
) -> list[Detection]:
    if not rows:
        return []

    valid_meta = []
    tlbrs = []
    H, W = image.shape[:2]
    for det_index, tlwh, score in rows:
        tlbr = _tlwh_to_tlbr(tlwh)
        tlbr[0] = np.clip(tlbr[0], 0.0, W - 1.0)
        tlbr[1] = np.clip(tlbr[1], 0.0, H - 1.0)
        tlbr[2] = np.clip(tlbr[2], 0.0, W - 1.0)
        tlbr[3] = np.clip(tlbr[3], 0.0, H - 1.0)
        if tlbr[2] <= tlbr[0] + 1.0 or tlbr[3] <= tlbr[1] + 1.0:
            continue
        valid_meta.append((det_index, tlwh.astype(np.float32), tlbr.astype(np.float32), float(score)))
        tlbrs.append(tlbr.astype(np.float32))

    if not valid_meta:
        return []

    tlbrs_arr = np.stack(tlbrs, axis=0).astype(np.float32)
    feats = encoder.inference(image, tlbrs_arr)
    feats = _normalize_rows(np.asarray(feats, dtype=np.float32))
    assigned_gt, ignore_mask = _assign_detections_to_gt(
        det_tlbrs=tlbrs_arr,
        gt_positive=gt_positive,
        gt_ignore=gt_ignore,
        iou_pos=iou_pos,
        iou_ignore=iou_ignore,
    )

    detections = []
    for idx, (det_index, tlwh, tlbr, score) in enumerate(valid_meta):
        detections.append(
            Detection(
                frame=int(frame),
                det_index=int(det_index),
                tlwh=np.asarray(tlwh, dtype=np.float32),
                tlbr=np.asarray(tlbr, dtype=np.float32),
                score=float(score),
                feat=feats[idx].astype(np.float32),
                gt_id=int(assigned_gt[idx]),
                ignore=bool(ignore_mask[idx]),
            )
        )
    return detections


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    if not args.out_csv and not args.out_npz:
        raise ValueError("At least one of --out-csv or --out-npz must be provided.")
    if str(args.device).lower().startswith("cpu"):
        raise ValueError(
            "This repository's FastReIDInterface is only reliable on the CUDA path. "
            "Please run the GT pseudo-track builder with --device cuda."
        )

    dataset_root = Path(args.data_root) / args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    encoder = FastReIDInterface(
        args.fast_reid_config,
        args.fast_reid_weights,
        args.device,
        batch_size=int(args.batch_size),
    )

    feat_dtype_np = np.float16 if args.feature_dtype == "float16" else np.float32
    csv_row_count = 0
    npz_det_feat: list[np.ndarray] = []
    npz_hist_feat: list[np.ndarray] = []
    npz_hist_mask: list[np.ndarray] = []
    npz_track_feat: list[np.ndarray] = []
    npz_ctx_feat: list[np.ndarray] = []
    npz_group_id: list[int] = []
    npz_label: list[float] = []
    npz_gap: list[int] = []
    npz_history_len: list[int] = []
    npz_track_gt_id: list[int] = []
    npz_det_gt_id: list[int] = []
    npz_frame: list[int] = []
    npz_det_index: list[int] = []
    npz_seq: list[str] = []

    global_group_id = 0
    csv_file = None
    csv_writer = None
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = out_csv.open("w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=PAIR_CSV_FIELDNAMES)
        csv_writer.writeheader()

    for seq_name in args.seqs:
        split = "train"
        seq_dir = dataset_root / split / seq_name
        if not seq_dir.exists():
            raise FileNotFoundError(f"Missing sequence directory: {seq_dir}")

        seqinfo = _read_seqinfo(seq_dir)
        imext = seqinfo.get("imExt", ".jpg")
        det_path = _history_file_name(seq_dir, "det", args.split_part)
        gt_path = _history_file_name(seq_dir, "gt", args.split_part)
        if not det_path.exists():
            raise FileNotFoundError(f"Missing det file: {det_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing gt file: {gt_path}")

        det_rows = _read_det_rows(det_path)
        gt_rows = _read_gt_rows(gt_path)
        seq_histories: Dict[int, PseudoTrackState] = {}
        all_frames = sorted(set(det_rows.keys()) | set(gt_rows.keys()))
        if int(args.max_frames) > 0:
            all_frames = all_frames[: int(args.max_frames)]
        print(f"[seq] {seq_name} frames={len(all_frames)} det={det_path.name} gt={gt_path.name}", flush=True)

        for frame in all_frames:
            image_path = _frame_image_path(seq_dir, frame, imext)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Failed to read image: {image_path}")

            gt_positive, gt_ignore = _split_gt_frame(gt_rows.get(frame, []))
            frame_dets = _collect_detections_for_frame(
                encoder=encoder,
                image=image,
                frame=frame,
                rows=det_rows.get(frame, []),
                gt_positive=gt_positive,
                gt_ignore=gt_ignore,
                iou_pos=float(args.iou_pos),
                iou_ignore=float(args.iou_ignore),
            )

            frame_updates: list[Detection] = []
            for det in frame_dets:
                if det.ignore:
                    continue

                candidates = _choose_candidates(
                    histories=seq_histories,
                    det_tlbr=det.tlbr,
                    det_gt_id=det.gt_id,
                    frame=frame,
                    max_gap=int(args.max_gap),
                    candidate_topk=int(args.candidate_topk),
                    candidate_min_motion=float(args.candidate_min_motion),
                )

                if not candidates:
                    if det.gt_id > 0:
                        frame_updates.append(det)
                    continue

                group_records = []
                for track_gt_id, motion_sim in candidates:
                    state = seq_histories[track_gt_id]
                    hist = state.history_feats
                    protos, stability, coherence = _build_laplace_stats(
                        hist=hist,
                        decay_scales=args.laplace_decay_scales,
                        min_history=int(args.min_history),
                    )
                    spatial_sim = _cosine_sim01(det.feat, state.smooth_feat if state.smooth_feat is not None else hist[-1])
                    laplace_sim = float(np.mean(np.matmul(protos, det.feat.reshape(-1, 1)).reshape(-1) * 0.5 + 0.5))
                    agreement = _agreement(spatial_sim, laplace_sim)
                    pair_rel = _heuristic_pair_rel(
                        stability=stability,
                        coherence=coherence,
                        agreement=agreement,
                        det_score=det.score,
                    )
                    appearance_sim = (1.0 - 0.35) * spatial_sim + 0.35 * laplace_sim
                    fused_sim = pair_rel * appearance_sim + (1.0 - pair_rel) * float(motion_sim)
                    gap = int(frame - state.last_frame)
                    hist_len = int(hist.shape[0])
                    group_records.append(
                        {
                            "track_gt_id": int(track_gt_id),
                            "det_gt_id": int(det.gt_id),
                            "gap": gap,
                            "history_len": hist_len,
                            "pair_rel": float(pair_rel),
                            "appearance_sim": float(appearance_sim),
                            "fused_sim": float(fused_sim),
                            "motion_sim": float(motion_sim),
                            "spatial_sim": float(spatial_sim),
                            "laplace_sim": float(laplace_sim),
                            "agreement": float(agreement),
                            "stability": float(stability),
                            "coherence": float(coherence),
                            "det_score": float(det.score),
                            "prod_sim": float(np.clip(spatial_sim * laplace_sim, 0.0, 1.0)),
                            "label": 1 if det.gt_id > 0 and track_gt_id == det.gt_id else 0,
                            "hist_feats": hist[-int(args.max_history) :].astype(feat_dtype_np, copy=False),
                        }
                    )

                has_positive = any(int(rec["label"]) == 1 for rec in group_records)
                if not has_positive and not args.include_background:
                    if det.gt_id > 0:
                        frame_updates.append(det)
                    continue

                group_records = _subsample_group_records(
                    group_records=group_records,
                    max_hard_negatives=int(args.max_hard_negatives),
                    max_random_negatives=int(args.max_random_negatives),
                    rng=rng,
                )

                spatial_vals = [float(rec["spatial_sim"]) for rec in group_records]
                laplace_vals = [float(rec["laplace_sim"]) for rec in group_records]
                motion_vals = [float(rec["motion_sim"]) for rec in group_records]

                amb_spa = _margin(spatial_vals)
                amb_lap = _margin(laplace_vals)
                amb_mot = _margin(motion_vals)
                chosen_idx = int(np.argmax([float(rec["fused_sim"]) for rec in group_records]))

                for rec_idx, rec in enumerate(group_records):
                    row_dict = {
                            "seq": seq_name,
                            "frame": int(frame),
                            "assoc_stage": "primary",
                            "track_id": int(rec["track_gt_id"]),
                            "det_index": int(det.det_index),
                            "gap": int(rec["gap"]),
                            "history_len": int(rec["history_len"]),
                            "chosen": 1 if rec_idx == chosen_idx else 0,
                            "is_true_match": int(rec["label"]),
                            "track_gt_id": int(rec["track_gt_id"]),
                            "det_gt_id": int(rec["det_gt_id"]),
                            "pair_rel": float(rec["pair_rel"]),
                            "learned_alpha": float("nan"),
                            "learned_r": float("nan"),
                            "appearance_sim": float(rec["appearance_sim"]),
                            "fused_sim": float(rec["fused_sim"]),
                            "motion_sim": float(rec["motion_sim"]),
                            "spatial_sim": float(rec["spatial_sim"]),
                            "laplace_sim": float(rec["laplace_sim"]),
                            "agreement": float(rec["agreement"]),
                            "stability": float(rec["stability"]),
                            "coherence": float(rec["coherence"]),
                            "det_score": float(rec["det_score"]),
                            "prod_sim": float(rec["prod_sim"]),
                            "amb_spa": float(amb_spa),
                            "amb_lap": float(amb_lap),
                            "amb_mot": float(amb_mot),
                        }
                    if csv_writer is not None:
                        csv_writer.writerow(row_dict)
                    csv_row_count += 1

                    hist_feats = rec["hist_feats"]
                    hist_pad = np.zeros((int(args.max_history), hist_feats.shape[1]), dtype=feat_dtype_np)
                    hist_mask = np.zeros((int(args.max_history),), dtype=np.float32)
                    keep = min(int(args.max_history), hist_feats.shape[0])
                    hist_pad[-keep:] = hist_feats[-keep:]
                    hist_mask[-keep:] = 1.0

                    gap_log1p = float(np.log1p(max(int(rec["gap"]), 0)))
                    hist_norm = float(min(1.0, float(rec["history_len"]) / float(max(int(args.min_history), 1))))
                    track_feat = np.asarray(
                        [gap_log1p, hist_norm, float(rec["stability"]), float(rec["coherence"])],
                        dtype=np.float32,
                    )
                    # Keep the legacy prefix intact for old loaders, then append richer context
                    # so new training code can consume a closer-to-runtime anchor.
                    ctx_feat = np.asarray(
                        [
                            float(rec["spatial_sim"]),
                            float(rec["motion_sim"]),
                            float(rec["det_score"]),
                            float(amb_spa),
                            float(amb_mot),
                            float(rec["stability"]),
                            float(rec["coherence"]),
                            float(rec["appearance_sim"]),
                            float(rec["fused_sim"]),
                            float(rec["laplace_sim"]),
                            float(rec["pair_rel"]),
                            float(amb_lap),
                            float(rec["agreement"]),
                            float(rec["prod_sim"]),
                        ],
                        dtype=np.float32,
                    )
                    npz_det_feat.append(np.asarray(det.feat, dtype=feat_dtype_np))
                    npz_hist_feat.append(hist_pad)
                    npz_hist_mask.append(hist_mask)
                    npz_track_feat.append(track_feat)
                    npz_ctx_feat.append(ctx_feat)
                    npz_group_id.append(int(global_group_id))
                    npz_label.append(float(rec["label"]))
                    npz_gap.append(int(rec["gap"]))
                    npz_history_len.append(int(rec["history_len"]))
                    npz_track_gt_id.append(int(rec["track_gt_id"]))
                    npz_det_gt_id.append(int(rec["det_gt_id"]))
                    npz_frame.append(int(frame))
                    npz_det_index.append(int(det.det_index))
                    npz_seq.append(seq_name)

                global_group_id += 1

                if det.gt_id > 0:
                    frame_updates.append(det)

            for det in frame_updates:
                state = seq_histories.get(det.gt_id)
                if state is None:
                    state = PseudoTrackState(
                        gt_id=int(det.gt_id),
                        frames=deque(maxlen=int(args.max_history)),
                        boxes_tlbr=deque(maxlen=int(args.max_history)),
                        feats=deque(maxlen=int(args.max_history)),
                        smooth_feat=None,
                    )
                    seq_histories[int(det.gt_id)] = state
                state.update(
                    frame=int(frame),
                    tlbr=det.tlbr,
                    feat=det.feat,
                    smooth_alpha=float(args.smooth_alpha),
                )

    if csv_file is not None:
        csv_file.close()
        print(f"[saved] csv={args.out_csv} rows={csv_row_count}")

    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        if not npz_det_feat:
            raise ValueError("No NPZ training candidates were built.")
        np.savez_compressed(
            out_npz,
            det_feat=np.stack(npz_det_feat, axis=0).astype(feat_dtype_np),
            hist_feat=np.stack(npz_hist_feat, axis=0).astype(feat_dtype_np),
            hist_mask=np.stack(npz_hist_mask, axis=0).astype(np.float32),
            track_feat=np.stack(npz_track_feat, axis=0).astype(np.float32),
            ctx_feat=np.stack(npz_ctx_feat, axis=0).astype(np.float32),
            group_id=np.asarray(npz_group_id, dtype=np.int64),
            label=np.asarray(npz_label, dtype=np.float32),
            gap=np.asarray(npz_gap, dtype=np.int32),
            history_len=np.asarray(npz_history_len, dtype=np.int32),
            track_gt_id=np.asarray(npz_track_gt_id, dtype=np.int32),
            det_gt_id=np.asarray(npz_det_gt_id, dtype=np.int32),
            frame=np.asarray(npz_frame, dtype=np.int32),
            det_index=np.asarray(npz_det_index, dtype=np.int32),
            seq=np.asarray(npz_seq, dtype=object),
        )
        print(f"[saved] npz={out_npz} groups={global_group_id} candidates={len(npz_group_id)}")


if __name__ == "__main__":
    main()
