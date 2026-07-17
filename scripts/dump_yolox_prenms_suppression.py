"""Dump YOLOX pre-NMS suppressed detections for the M23-1 oracle audit.

The script preserves the detector configuration used by the frozen Phase-0
post-NMS dumps. It runs the detector in batches, reconstructs the exact YOLOX
confidence filter and class-aware NMS, and writes only candidates removed by
NMS. No ReID features and no ground-truth information are used.

For every suppressed candidate the dump records its suppressing kept box,
NMS rank, IoU, confidence, and stable pre-NMS indices. Generated kept boxes are
matched back to the frozen Phase-0 post-NMS dump to detect configuration or
numerical drift.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment

SEQUENCE_IDS = (1, 2, 3, 5)
SEQUENCES = tuple(f"MOT20-{value:02d}" for value in SEQUENCE_IDS)
COLUMNS = (
    "frame",
    "raw_anchor_idx",
    "frame_pre_idx",
    "global_pre_idx",
    "frame_suppressed_idx",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "obj_conf",
    "cls_conf",
    "class_id",
    "suppressor_frame_pre_idx",
    "suppressor_global_pre_idx",
    "suppressor_nms_rank",
    "suppressor_iou",
    "suppressor_score",
    "suppressor_x1",
    "suppressor_y1",
    "suppressor_x2",
    "suppressor_y2",
)


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


def load_phase0_module(repo: Path):
    script = repo / "scripts" / "dump_yolox_reid_phase0.py"
    spec = importlib.util.spec_from_file_location("fmtrack_phase0", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_npz_member(path: Path, member: str, *, allow_pickle: bool = False) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle)


def iou_matrix_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def match_kept_to_reference(generated: np.ndarray, reference: np.ndarray) -> dict:
    result = {
        "generated": int(len(generated)),
        "reference": int(len(reference)),
        "count_equal": len(generated) == len(reference),
        "order_equal": False,
        "min_matched_iou": None,
        "max_box_abs_diff": None,
        "max_score_abs_diff": None,
        "class_mismatch": 0,
    }
    if len(generated) != len(reference):
        return result
    if len(generated) == 0:
        result.update({
            "order_equal": True,
            "min_matched_iou": 1.0,
            "max_box_abs_diff": 0.0,
            "max_score_abs_diff": 0.0,
        })
        return result
    result["order_equal"] = bool(np.allclose(generated, reference, rtol=0.0, atol=2e-4))
    if result["order_equal"]:
        a = generated[:, :4].astype(np.float64)
        b = reference[:, :4].astype(np.float64)
        xx1 = np.maximum(a[:, 0], b[:, 0])
        yy1 = np.maximum(a[:, 1], b[:, 1])
        xx2 = np.minimum(a[:, 2], b[:, 2])
        yy2 = np.minimum(a[:, 3], b[:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        diagonal_iou = inter / np.maximum(area_a + area_b - inter, 1e-12)
        generated_score = generated[:, 4] * generated[:, 5]
        reference_score = reference[:, 4] * reference[:, 5]
        result["min_matched_iou"] = float(np.min(diagonal_iou))
        result["max_box_abs_diff"] = float(np.max(np.abs(generated[:, :4] - reference[:, :4])))
        result["max_score_abs_diff"] = float(np.max(np.abs(generated_score - reference_score)))
        result["class_mismatch"] = int(np.sum(
            generated[:, 6].astype(np.int64) != reference[:, 6].astype(np.int64)
        ))
        return result
    similarities = iou_matrix_numpy(generated[:, :4], reference[:, :4])
    generated_score_all = generated[:, 4] * generated[:, 5]
    reference_score_all = reference[:, 4] * reference[:, 5]
    score_difference = np.abs(generated_score_all[:, None] - reference_score_all[None, :])
    class_mismatch_matrix = (
        generated[:, 6].astype(np.int64)[:, None]
        != reference[:, 6].astype(np.int64)[None, :]
    )
    # Dense MOT20 frames contain many nearly identical boxes. Pure-IoU Hungarian
    # can cross-pair duplicates even when the detector output is exactly reproduced.
    # The frozen post-NMS row is identified jointly by class, score and geometry.
    cost = (1.0 - similarities) + 20.0 * score_difference + 1000.0 * class_mismatch_matrix
    rows, cols = linear_sum_assignment(cost)
    matched = reference[cols]
    result["min_matched_iou"] = float(np.min(similarities[rows, cols]))
    result["max_box_abs_diff"] = float(np.max(np.abs(generated[rows, :4] - matched[:, :4])))
    generated_score = generated_score_all[rows]
    reference_score = reference_score_all[cols]
    result["max_score_abs_diff"] = float(np.max(np.abs(generated_score - reference_score)))
    result["class_mismatch"] = int(np.sum(class_mismatch_matrix[rows, cols]))
    return result


def resolve_under(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--seq-ids", nargs="+", type=int, default=list(SEQUENCE_IDS))
    parser.add_argument("--bot-sort-root", default="external/BoT-SORT-main")
    parser.add_argument("--exp-file", default="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py")
    parser.add_argument("--ckpt", default="./pretrained/bytetrack_x_mot20.pth.tar")
    parser.add_argument("--track-low-thresh", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--nms", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--refresh-reports-only",
        action="store_true",
        help="rebuild deterministic reports/manifests from existing raw dumps without detector inference",
    )
    parser.add_argument("--equivalence-min-iou", type=float, default=0.999)
    parser.add_argument("--equivalence-max-score-diff", type=float, default=1e-4)
    return parser.parse_args()


def build_reference_view(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], dict]:
    with zipfile.ZipFile(npz_path) as archive:
        detections_bytes = archive.read("detections.npy")
        offsets_bytes = archive.read("frame_offsets.npy")
        columns_bytes = archive.read("columns.npy")
    detections = np.load(io.BytesIO(detections_bytes), allow_pickle=False)
    frame_offsets = np.load(io.BytesIO(offsets_bytes), allow_pickle=False)
    columns = [
        str(value)
        for value in np.load(io.BytesIO(columns_bytes), allow_pickle=True).tolist()
    ]
    index = {name: idx for idx, name in enumerate(columns)}
    required = ("x1", "y1", "x2", "y2", "obj_conf", "cls_conf", "class_id")
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"{npz_path}: missing reference columns {missing}")
    metadata = {
        "path": str(npz_path),
        "size_bytes": npz_path.stat().st_size,
        "detections_member_sha256": hashlib.sha256(detections_bytes).hexdigest(),
        "frame_offsets_member_sha256": hashlib.sha256(offsets_bytes).hexdigest(),
        "columns_member_sha256": hashlib.sha256(columns_bytes).hexdigest(),
        "detections": int(len(detections)),
        "columns": columns,
        "full_npz_hash_omitted": "feature member is multi-GB and is not read by this geometry-only audit",
    }
    return detections, frame_offsets.astype(np.int64), index, metadata


def process_prediction(
    raw: torch.Tensor,
    *,
    frame: int,
    height: int,
    width: int,
    test_size: Tuple[int, int],
    num_classes: int,
    conf_threshold: float,
    nms_threshold: float,
    global_pre_offset: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    prediction = raw.float()
    corners = prediction.new_empty((prediction.shape[0], 4))
    corners[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    corners[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    corners[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    corners[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    class_conf, class_pred = torch.max(prediction[:, 5:5 + num_classes], 1)
    scores = prediction[:, 4] * class_conf
    confidence_mask = scores >= float(conf_threshold)
    raw_anchor_indices = torch.nonzero(confidence_mask, as_tuple=False).flatten()
    if raw_anchor_indices.numel() == 0:
        return np.zeros((0, len(COLUMNS)), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), {
            "pre": 0,
            "kept": 0,
            "suppressed": 0,
            "orphan_suppressed": 0,
        }

    boxes = corners[confidence_mask]
    obj_conf = prediction[confidence_mask, 4]
    cls_conf = class_conf[confidence_mask]
    class_id = class_pred[confidence_mask].float()
    filtered_scores = scores[confidence_mask]
    keep = torchvision.ops.batched_nms(boxes, filtered_scores, class_id, float(nms_threshold))
    keep_mask = torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)
    keep_mask[keep] = True
    suppressed_indices = torch.nonzero(~keep_mask, as_tuple=False).flatten()

    scale = min(float(test_size[0]) / float(height), float(test_size[1]) / float(width))
    original_boxes = boxes / max(scale, 1e-12)
    original_boxes[:, 0] = torch.clamp(original_boxes[:, 0], 0, max(width - 1, 0))
    original_boxes[:, 1] = torch.clamp(original_boxes[:, 1], 0, max(height - 1, 0))
    original_boxes[:, 2] = torch.clamp(original_boxes[:, 2], 0, max(width - 1, 0))
    original_boxes[:, 3] = torch.clamp(original_boxes[:, 3], 0, max(height - 1, 0))

    kept_boxes = original_boxes[keep]
    kept_view = torch.cat(
        [kept_boxes, obj_conf[keep, None], cls_conf[keep, None], class_id[keep, None]],
        dim=1,
    ).detach().cpu().numpy().astype(np.float32)

    if suppressed_indices.numel() == 0:
        return np.zeros((0, len(COLUMNS)), dtype=np.float32), kept_view, {
            "pre": int(boxes.shape[0]),
            "kept": int(keep.numel()),
            "suppressed": 0,
            "orphan_suppressed": 0,
        }

    suppressed_boxes_network = boxes[suppressed_indices]
    kept_boxes_network = boxes[keep]
    overlaps = torchvision.ops.box_iou(suppressed_boxes_network, kept_boxes_network)
    same_class = class_id[suppressed_indices, None] == class_id[keep][None, :]
    valid_suppressor = same_class & (overlaps > float(nms_threshold))
    has_suppressor = valid_suppressor.any(dim=1)
    first_rank = torch.argmax(valid_suppressor.to(torch.int64), dim=1)
    if not bool(torch.all(has_suppressor)):
        fallback = torch.where(same_class, overlaps, overlaps.new_full(overlaps.shape, -1.0))
        first_rank = torch.where(has_suppressor, first_rank, torch.argmax(fallback, dim=1))
    suppressor_indices = keep[first_rank]
    suppressor_iou = overlaps[torch.arange(len(suppressed_indices), device=overlaps.device), first_rank]

    frame_pre_idx = torch.arange(boxes.shape[0], device=boxes.device, dtype=torch.int64)
    global_pre_idx = frame_pre_idx + int(global_pre_offset)
    frame_suppressed_idx = torch.arange(len(suppressed_indices), device=boxes.device, dtype=torch.int64)
    suppressed_boxes = original_boxes[suppressed_indices]
    suppressor_boxes = original_boxes[suppressor_indices]

    rows = torch.cat(
        [
            torch.full((len(suppressed_indices), 1), float(frame), device=boxes.device),
            raw_anchor_indices[suppressed_indices, None].float(),
            frame_pre_idx[suppressed_indices, None].float(),
            global_pre_idx[suppressed_indices, None].float(),
            frame_suppressed_idx[:, None].float(),
            suppressed_boxes,
            filtered_scores[suppressed_indices, None],
            obj_conf[suppressed_indices, None],
            cls_conf[suppressed_indices, None],
            class_id[suppressed_indices, None],
            suppressor_indices[:, None].float(),
            global_pre_idx[suppressor_indices, None].float(),
            first_rank[:, None].float(),
            suppressor_iou[:, None],
            filtered_scores[suppressor_indices, None],
            suppressor_boxes,
        ],
        dim=1,
    ).detach().cpu().numpy().astype("<f4", copy=False)
    if rows.shape[1] != len(COLUMNS):
        raise RuntimeError(f"suppressed row width {rows.shape[1]} != {len(COLUMNS)}")
    return rows, kept_view, {
        "pre": int(boxes.shape[0]),
        "kept": int(keep.numel()),
        "suppressed": int(len(suppressed_indices)),
        "orphan_suppressed": int((~has_suppressor).sum().item()),
    }


def dump_sequence(args: argparse.Namespace, repo: Path, phase0_module, sequence: str, sequence_id: int) -> dict:
    output_dir = resolve_under(repo, args.out_root) / sequence
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    phase0_path = resolve_under(repo, args.phase0_root) / sequence / "dump_yolox_reid.npz"
    if not phase0_path.is_file():
        raise FileNotFoundError(phase0_path)
    reference, reference_offsets, reference_index, reference_metadata = build_reference_view(phase0_path)

    detector_args = argparse.Namespace(
        bot_sort_root=str(resolve_under(repo, args.bot_sort_root)),
        exp_file=args.exp_file,
        ckpt=args.ckpt,
        benchmark="MOT20",
        conf=args.conf,
        nms=args.nms,
        tsize=None,
        track_low_thresh=float(args.track_low_thresh),
        device=args.device,
        fuse=bool(args.fuse),
        fp16=bool(args.fp16),
    )
    exp, predictor, model = phase0_module.init_detector(detector_args, sequence)
    from yolox.data.data_augment import preproc

    specs = phase0_module.build_sequence_specs(
        resolve_under(repo, args.data_root), "MOT20", "train", [sequence_id]
    )
    if len(specs) != 1:
        raise RuntimeError(f"unexpected sequence spec count for {sequence}: {len(specs)}")
    image_files = phase0_module.image_list(Path(str(specs[0]["img_dir"])))
    full_frame_count = len(image_files)
    if len(reference_offsets) != full_frame_count + 1:
        raise RuntimeError(
            f"{sequence}: reference offsets {len(reference_offsets)} != frames+1 {full_frame_count+1}"
        )
    if int(args.limit_frames) > 0:
        image_files = image_files[: int(args.limit_frames)]

    raw_path = output_dir / "suppressed_candidates.f32"
    frame_offsets: List[int] = [0]
    frame_rows: List[dict] = []
    total_pre = 0
    total_kept = 0
    total_suppressed = 0
    total_orphans = 0
    count_mismatch_frames = 0
    set_mismatch_frames = 0
    order_mismatch_frames = 0
    min_matched_iou = 1.0
    max_box_abs_diff = 0.0
    max_score_abs_diff = 0.0
    class_mismatch = 0

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    batch_size = max(1, int(args.batch_size))
    with raw_path.open("wb") as raw_handle:
        for batch_start in range(0, len(image_files), batch_size):
            selected = image_files[batch_start:batch_start + batch_size]
            tensors = []
            infos = []
            for local_index, image_path in enumerate(selected):
                frame = batch_start + local_index + 1
                raw_image = phase0_module.read_image_robust(image_path)
                if raw_image is None:
                    raise RuntimeError(f"failed to read {image_path}")
                height, width = raw_image.shape[:2]
                image, _ratio = preproc(
                    raw_image,
                    predictor.test_size,
                    predictor.rgb_means,
                    predictor.std,
                )
                tensors.append(torch.from_numpy(image))
                infos.append((frame, height, width))
            tensor = torch.stack(tensors).float().to(predictor.device)
            if predictor.fp16:
                tensor = tensor.half()
            with torch.no_grad():
                predictions = model(tensor)

            for batch_index, (frame, height, width) in enumerate(infos):
                rows, kept_view, counts = process_prediction(
                    predictions[batch_index],
                    frame=frame,
                    height=height,
                    width=width,
                    test_size=tuple(predictor.test_size),
                    num_classes=int(exp.num_classes),
                    conf_threshold=float(predictor.confthre),
                    nms_threshold=float(predictor.nmsthre),
                    global_pre_offset=total_pre,
                )
                start = int(reference_offsets[frame - 1])
                end = int(reference_offsets[frame])
                reference_view = reference[start:end][:, [
                    reference_index["x1"],
                    reference_index["y1"],
                    reference_index["x2"],
                    reference_index["y2"],
                    reference_index["obj_conf"],
                    reference_index["cls_conf"],
                    reference_index["class_id"],
                ]].astype(np.float32, copy=False)
                equivalence = match_kept_to_reference(kept_view, reference_view)
                if not equivalence["count_equal"]:
                    count_mismatch_frames += 1
                if not equivalence["order_equal"]:
                    order_mismatch_frames += 1
                frame_min_iou = equivalence["min_matched_iou"]
                frame_score_diff = equivalence["max_score_abs_diff"]
                frame_box_diff = equivalence["max_box_abs_diff"]
                if frame_min_iou is not None:
                    min_matched_iou = min(min_matched_iou, float(frame_min_iou))
                if frame_score_diff is not None:
                    max_score_abs_diff = max(max_score_abs_diff, float(frame_score_diff))
                if frame_box_diff is not None:
                    max_box_abs_diff = max(max_box_abs_diff, float(frame_box_diff))
                class_mismatch += int(equivalence["class_mismatch"])
                set_ok = (
                    bool(equivalence["count_equal"])
                    and frame_min_iou is not None
                    and float(frame_min_iou) >= float(args.equivalence_min_iou)
                    and frame_score_diff is not None
                    and float(frame_score_diff) <= float(args.equivalence_max_score_diff)
                    and int(equivalence["class_mismatch"]) == 0
                )
                if not set_ok:
                    set_mismatch_frames += 1

                rows.tofile(raw_handle)
                total_pre += int(counts["pre"])
                total_kept += int(counts["kept"])
                total_suppressed += int(counts["suppressed"])
                total_orphans += int(counts["orphan_suppressed"])
                frame_offsets.append(total_suppressed)
                frame_rows.append({
                    "sequence": sequence,
                    "frame": frame,
                    "pre_candidates": counts["pre"],
                    "postnms_kept": counts["kept"],
                    "suppressed": counts["suppressed"],
                    "orphan_suppressed": counts["orphan_suppressed"],
                    "reference_kept": len(reference_view),
                    "reference_count_equal": equivalence["count_equal"],
                    "reference_order_equal": equivalence["order_equal"],
                    "reference_min_matched_iou": equivalence["min_matched_iou"],
                    "reference_max_box_abs_diff": equivalence["max_box_abs_diff"],
                    "reference_max_score_abs_diff": equivalence["max_score_abs_diff"],
                    "reference_class_mismatch": equivalence["class_mismatch"],
                    "reference_set_ok": set_ok,
                })
            processed = min(batch_start + len(selected), len(image_files))
            if batch_start == 0 or processed % 200 == 0 or processed == len(image_files):
                print(
                    f"[M23-1 dump] {sequence} frames={processed}/{len(image_files)} "
                    f"pre={total_pre} kept={total_kept} suppressed={total_suppressed}",
                    flush=True,
                )
        raw_handle.flush()
        os.fsync(raw_handle.fileno())

    np.save(output_dir / "frame_offsets.npy", np.asarray(frame_offsets, dtype="<i8"), allow_pickle=False)
    canonical_json_dump(list(COLUMNS), output_dir / "columns.json")
    frame_fields = [
        "sequence",
        "frame",
        "pre_candidates",
        "postnms_kept",
        "suppressed",
        "orphan_suppressed",
        "reference_kept",
        "reference_count_equal",
        "reference_order_equal",
        "reference_min_matched_iou",
        "reference_max_box_abs_diff",
        "reference_max_score_abs_diff",
        "reference_class_mismatch",
        "reference_set_ok",
    ]
    write_csv(output_dir / "frame_summary.csv", frame_rows, frame_fields)

    if total_pre != total_kept + total_suppressed:
        raise RuntimeError(f"{sequence}: pre != kept + suppressed")
    if raw_path.stat().st_size != total_suppressed * len(COLUMNS) * 4:
        raise RuntimeError(f"{sequence}: raw byte size mismatch")

    decision = {
        "reference_count_mismatch_frames": count_mismatch_frames,
        "reference_set_mismatch_frames": set_mismatch_frames,
        "reference_order_mismatch_frames": order_mismatch_frames,
        "reference_min_matched_iou": min_matched_iou,
        "reference_max_box_abs_diff": max_box_abs_diff,
        "reference_max_score_abs_diff": max_score_abs_diff,
        "reference_class_mismatch": class_mismatch,
        "suppression_orphans": total_orphans,
        "reference_equivalence_passed": (
            count_mismatch_frames == 0
            and set_mismatch_frames == 0
            and class_mismatch == 0
            and total_orphans == 0
        ),
    }
    report = {
        "schema": "fmtrack.m23.prenms_suppression_dump.v1",
        "sequence": sequence,
        "protocol": {
            "detector_precision": "fp16" if args.fp16 else "fp32",
            "fused_model": bool(args.fuse),
            "batch_size": batch_size,
            "confidence_threshold": float(predictor.confthre),
            "nms_threshold": float(predictor.nmsthre),
            "reid_extracted": False,
            "ground_truth_used": False,
            "suppression_parent": "first selected NMS box of the same class with IoU above the NMS threshold",
            "equivalence_min_iou": float(args.equivalence_min_iou),
            "equivalence_max_score_diff": float(args.equivalence_max_score_diff),
        },
        "counts": {
            "frames": len(image_files),
            "pre_candidates": total_pre,
            "postnms_kept": total_kept,
            "suppressed_candidates": total_suppressed,
            "pre_to_post_ratio": total_pre / max(1, total_kept),
        },
        "decision": decision,
        "reference_phase0": reference_metadata,
        "files": {
            "suppressed_candidates": {
                "path": str(raw_path),
                "sha256": sha256_file(raw_path),
                "size_bytes": raw_path.stat().st_size,
                "dtype": "little-endian float32",
                "shape": [total_suppressed, len(COLUMNS)],
            },
            "frame_offsets": {
                "sha256": sha256_file(output_dir / "frame_offsets.npy"),
                "shape": [len(frame_offsets)],
            },
            "columns": {
                "sha256": sha256_file(output_dir / "columns.json"),
                "columns": list(COLUMNS),
            },
            "frame_summary": {
                "sha256": sha256_file(output_dir / "frame_summary.csv"),
                "rows": len(frame_rows),
            },
        },
        "locked_state": {
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
            "p15_policy": "no_op",
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    manifest = {
        "schema": "fmtrack.m23.prenms_suppression_dump.manifest.v1",
        "sequence": sequence,
        "decision": decision,
        "file_hashes": {
            "suppressed_candidates.f32": report["files"]["suppressed_candidates"]["sha256"],
            "frame_offsets.npy": report["files"]["frame_offsets"]["sha256"],
            "columns.json": report["files"]["columns"]["sha256"],
            "frame_summary.csv": report["files"]["frame_summary"]["sha256"],
            "report.json": sha256_file(output_dir / "report.json"),
        },
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    if not decision["reference_equivalence_passed"]:
        raise RuntimeError(f"{sequence}: generated kept set does not match frozen Phase-0 reference: {decision}")
    return report



def refresh_sequence_report(
    args: argparse.Namespace,
    repo: Path,
    sequence: str,
) -> dict:
    output_dir = resolve_under(repo, args.out_root) / sequence
    report_path = output_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    phase0_path = resolve_under(repo, args.phase0_root) / sequence / "dump_yolox_reid.npz"
    detections, frame_offsets, _index, reference_metadata = build_reference_view(phase0_path)
    if int(report["counts"]["postnms_kept"]) != int(len(detections)):
        raise RuntimeError(
            f"{sequence}: report kept count {report['counts']['postnms_kept']} "
            f"!= frozen Phase-0 {len(detections)}"
        )
    if int(report["counts"]["frames"]) + 1 != int(len(frame_offsets)):
        raise RuntimeError(
            f"{sequence}: report frame count {report['counts']['frames']} "
            f"!= frozen Phase-0 offsets {len(frame_offsets)-1}"
        )
    raw_path = output_dir / "suppressed_candidates.f32"
    offsets_path = output_dir / "frame_offsets.npy"
    columns_path = output_dir / "columns.json"
    frame_summary_path = output_dir / "frame_summary.csv"
    required = (raw_path, offsets_path, columns_path, frame_summary_path)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    shape = [int(value) for value in report["files"]["suppressed_candidates"]["shape"]]
    columns = json.loads(columns_path.read_text(encoding="utf-8"))
    if shape != [int(report["counts"]["suppressed_candidates"]), len(columns)]:
        raise RuntimeError(f"{sequence}: raw shape metadata mismatch: {shape}")
    expected_size = shape[0] * shape[1] * 4
    if raw_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"{sequence}: raw size {raw_path.stat().st_size} != expected {expected_size}"
        )
    report["reference_phase0"] = reference_metadata
    report["files"] = {
        "suppressed_candidates": {
            "path": str(raw_path),
            "sha256": sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "dtype": "little-endian float32",
            "shape": shape,
        },
        "frame_offsets": {
            "sha256": sha256_file(offsets_path),
            "shape": [int(len(np.load(offsets_path, allow_pickle=False)))],
        },
        "columns": {
            "sha256": sha256_file(columns_path),
            "columns": columns,
        },
        "frame_summary": {
            "sha256": sha256_file(frame_summary_path),
            "rows": int(report["counts"]["frames"]),
        },
    }
    report["report_refresh"] = {
        "detector_rerun": False,
        "raw_dump_unchanged": True,
        "purpose": "replace whole-NPZ hashing with hashes of the geometry members actually read",
    }
    canonical_json_dump(report, report_path)
    manifest = {
        "schema": "fmtrack.m23.prenms_suppression_dump.manifest.v1",
        "sequence": sequence,
        "decision": report["decision"],
        "file_hashes": {
            "suppressed_candidates.f32": report["files"]["suppressed_candidates"]["sha256"],
            "frame_offsets.npy": report["files"]["frame_offsets"]["sha256"],
            "columns.json": report["files"]["columns"]["sha256"],
            "frame_summary.csv": report["files"]["frame_summary"]["sha256"],
            "report.json": sha256_file(report_path),
        },
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    return report

def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    out_root = resolve_under(repo, args.out_root)
    # Overwrite is sequence-scoped in dump_sequence so completed sequences can
    # be preserved when a later sequence is resumed or rerun with a new batch size.
    out_root.mkdir(parents=True, exist_ok=True)
    selected_ids = [value for value in args.seq_ids if value in SEQUENCE_IDS]
    if not selected_ids:
        raise ValueError(f"no valid MOT20 train sequence ids: {args.seq_ids}")
    phase0_module = load_phase0_module(repo)
    sequence_reports = []
    for sequence_id in selected_ids:
        sequence = f"MOT20-{sequence_id:02d}"
        if args.refresh_reports_only:
            print(f"[M23-1 dump] refreshing report {sequence}", flush=True)
            sequence_reports.append(refresh_sequence_report(args, repo, sequence))
        else:
            print(f"[M23-1 dump] initializing {sequence}", flush=True)
            sequence_reports.append(dump_sequence(args, repo, phase0_module, sequence, sequence_id))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    combined = {
        "schema": "fmtrack.m23.prenms_suppression_dump.combined.v1",
        "sequences": [report["sequence"] for report in sequence_reports],
        "counts": {
            key: int(sum(int(report["counts"][key]) for report in sequence_reports))
            for key in ("frames", "pre_candidates", "postnms_kept", "suppressed_candidates")
        },
        "all_reference_equivalence_passed": all(
            bool(report["decision"]["reference_equivalence_passed"])
            for report in sequence_reports
        ),
        "sequence_manifest_hashes": {
            report["sequence"]: sha256_file(out_root / report["sequence"] / "manifest.json")
            for report in sequence_reports
        },
        "locked_state": {
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
            "p15_policy": "no_op",
        },
    }
    combined["counts"]["pre_to_post_ratio"] = (
        combined["counts"]["pre_candidates"] / max(1, combined["counts"]["postnms_kept"])
    )
    canonical_json_dump(combined, out_root / "report.json")
    canonical_json_dump(
        {
            "schema": "fmtrack.m23.prenms_suppression_dump.combined_manifest.v1",
            "report_sha256": sha256_file(out_root / "report.json"),
            "sequence_manifest_hashes": combined["sequence_manifest_hashes"],
        },
        out_root / "manifest.json",
    )
    print(json.dumps(combined, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
