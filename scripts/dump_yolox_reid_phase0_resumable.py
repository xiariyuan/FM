#!/usr/bin/env python3
"""Resumable Phase 0 dump for long MOT sequences.

Loads YOLOX/FastReID once per sequence and writes chunk npz files every N frames.
If interrupted, rerun with the same out-root and it will skip existing chunks.
After all chunks exist, run scripts/merge_phase0_chunks.py to build the standard
per-sequence dump.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "scripts" / "dump_yolox_reid_phase0.py"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_base():
    spec = importlib.util.spec_from_file_location("phase0_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase0_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resumable chunked Phase0 dump.")
    ap.add_argument("--data-root", default="/gemini/code/datasets")
    ap.add_argument("--benchmark", default="MOT20", choices=["MOT20", "MOT17"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seq-id", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--max-chunks", type=int, default=0, help="debug/resume cap; 0 means all missing chunks")
    ap.add_argument("--bot-sort-root", default=str(REPO_ROOT / "external" / "BoT-SORT-main"))
    ap.add_argument("--exp-file", default="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py")
    ap.add_argument("--ckpt", default="./pretrained/bytetrack_x_mot20.pth.tar")
    ap.add_argument("--fast-reid-config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    ap.add_argument("--fast-reid-weights", default="pretrained/mot20_sbs_S50.pth")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--nms", type=float, default=None)
    ap.add_argument("--tsize", type=int, default=None)
    ap.add_argument("--track-low-thresh", type=float, default=0.10)
    ap.add_argument("--dump-min-score", type=float, default=0.0)
    ap.add_argument("--reid-min-score", type=float, default=0.0)
    ap.add_argument("--reid-batch-size", type=int, default=64)
    ap.add_argument("--no-reid", action="store_true")
    ap.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--compress", action="store_true")
    # Dummy attrs expected by base helpers.
    ap.set_defaults(write_csv=False, limit_frames=0, overwrite=True)
    return ap.parse_args()


def write_chunk(base, args: argparse.Namespace, seq: str, spec: Dict[str, object], out_root: Path, start_frame: int, end_frame: int, det_chunks: List[np.ndarray], feat_chunks: List[np.ndarray], image_files: List[str], image_wh: List[Tuple[int, int]], frame_offsets: List[int], detections_with_reid: int, failed_images: int, feature_dim: int, feature_dtype) -> None:
    chunk_dir = out_root / f"{seq}_{start_frame:04d}_{end_frame:04d}" / seq
    chunk_dir.mkdir(parents=True, exist_ok=True)
    detections = np.concatenate(det_chunks, axis=0) if det_chunks else np.zeros((0, len(base.DETECTION_COLUMNS)), dtype=np.float32)
    if feat_chunks:
        features = np.concatenate(feat_chunks, axis=0)
    elif feature_dim > 0:
        features = np.zeros((0, feature_dim), dtype=feature_dtype)
    else:
        features = np.zeros((detections.shape[0], 0), dtype=np.float32)
    npz_path = chunk_dir / "dump_yolox_reid.npz"
    save_kwargs = {
        "detections": detections,
        "features": features,
        "frame_offsets": np.asarray(frame_offsets, dtype=np.int64),
        "columns": np.asarray(base.DETECTION_COLUMNS),
        "image_files": np.asarray(image_files),
        "image_wh": np.asarray(image_wh, dtype=np.int32),
    }
    if args.compress:
        np.savez_compressed(npz_path, **save_kwargs)
    else:
        np.savez(npz_path, **save_kwargs)
    manifest = {
        "status": "completed",
        "phase": "DMM_PHASE0_RESUMABLE_CHUNK",
        "seq": seq,
        "spec": spec,
        "chunk_frame_start": int(start_frame),
        "chunk_frame_end": int(end_frame),
        "chunk_frame_count": int(end_frame - start_frame + 1),
        "npz_path": str(npz_path),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "detections": int(detections.shape[0]),
        "detections_with_reid": int(detections_with_reid),
        "failed_images": int(failed_images),
        "args": vars(args),
        "command": " ".join(shlex.quote(v) for v in sys.argv),
        "finished_at": now_iso(),
    }
    (chunk_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[resumable] wrote {seq} {start_frame}-{end_frame} dets={detections.shape[0]}", flush=True)


def main() -> None:
    base = load_base()
    args = parse_args()
    args.seq_ids = [int(args.seq_id)]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    specs = base.build_sequence_specs(Path(args.data_root), args.benchmark, args.split, args.seq_ids)
    if len(specs) != 1:
        raise RuntimeError(f"Expected one spec, got {len(specs)}")
    spec = specs[0]
    seq = str(spec["seq"])
    img_dir = Path(str(spec["img_dir"]))
    files = base.image_list(img_dir)
    if not files:
        raise FileNotFoundError(f"No images under {img_dir}")
    chunk_size = max(1, int(args.chunk_size))
    ranges = []
    for start in range(1, len(files) + 1, chunk_size):
        end = min(len(files), start + chunk_size - 1)
        ranges.append((start, end))
    missing = []
    for start, end in ranges:
        npz = out_root / f"{seq}_{start:04d}_{end:04d}" / seq / "dump_yolox_reid.npz"
        if not npz.is_file():
            missing.append((start, end))
    if args.max_chunks and args.max_chunks > 0:
        missing = missing[: int(args.max_chunks)]
    print(f"[resumable] seq={seq} total_frames={len(files)} chunk_size={chunk_size} total_chunks={len(ranges)} missing_to_run={len(missing)}", flush=True)
    if not missing:
        return

    encoder = base.init_encoder(args)
    exp, predictor, model = base.init_detector(args, seq)
    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32
    try:
        for start, end in missing:
            selected = files[start - 1:end]
            det_chunks: List[np.ndarray] = []
            feat_chunks: List[np.ndarray] = []
            image_files: List[str] = []
            image_wh: List[Tuple[int, int]] = []
            frame_offsets = [0]
            global_det_idx = 0
            feature_dim = 2048 if not args.no_reid else 0
            detections_with_reid = 0
            failed_images = 0
            for local_idx, img_path in enumerate(selected, start=0):
                frame_id = start + local_idx
                if local_idx == 0 or (local_idx + 1) % 50 == 0 or frame_id == end:
                    print(f"[resumable] seq={seq} frame={frame_id} local={local_idx+1}/{len(selected)} dets_so_far={global_det_idx}", flush=True)
                outputs, info = predictor.inference(img_path)
                raw_img = info.get("raw_img")
                width = int(info.get("width", 0) or 0)
                height = int(info.get("height", 0) or 0)
                image_files.append(str(img_path))
                image_wh.append((width, height))
                if raw_img is None:
                    failed_images += 1
                    frame_offsets.append(global_det_idx)
                    continue
                if outputs is None or outputs.size == 0:
                    frame_offsets.append(global_det_idx)
                    continue
                scale = min(float(predictor.test_size[0]) / float(height), float(predictor.test_size[1]) / float(width))
                boxes = outputs[:, :4].astype(np.float32) / max(scale, 1e-12)
                obj_conf = outputs[:, 4].astype(np.float32)
                cls_conf = outputs[:, 5].astype(np.float32)
                cls_id = outputs[:, 6].astype(np.float32)
                score = obj_conf * cls_conf
                keep = score >= float(args.dump_min_score)
                boxes = boxes[keep]
                obj_conf = obj_conf[keep]
                cls_conf = cls_conf[keep]
                cls_id = cls_id[keep]
                score = score[keep]
                k = int(boxes.shape[0])
                if k == 0:
                    frame_offsets.append(global_det_idx)
                    continue
                boxes[:, 0] = np.clip(boxes[:, 0], 0, max(width - 1, 0))
                boxes[:, 1] = np.clip(boxes[:, 1], 0, max(height - 1, 0))
                boxes[:, 2] = np.clip(boxes[:, 2], 0, max(width - 1, 0))
                boxes[:, 3] = np.clip(boxes[:, 3], 0, max(height - 1, 0))
                frame_det_idx = np.arange(k, dtype=np.float32)
                global_ids = np.arange(global_det_idx, global_det_idx + k, dtype=np.float32)
                has_reid = np.zeros((k,), dtype=np.float32)
                feats = np.zeros((k, feature_dim), dtype=feature_dtype) if feature_dim > 0 else np.zeros((k, 0), dtype=np.float32)
                if encoder is not None and feature_dim > 0:
                    valid_box = (boxes[:, 2] > boxes[:, 0] + 1.0) & (boxes[:, 3] > boxes[:, 1] + 1.0)
                    reid_mask = valid_box & (score >= float(args.reid_min_score))
                    if np.any(reid_mask):
                        reid_feats = np.asarray(encoder.inference(raw_img, boxes[reid_mask].astype(np.float32)), dtype=np.float32)
                        reid_feats = base.l2norm(reid_feats)
                        if reid_feats.ndim != 2 or reid_feats.shape[0] != int(np.sum(reid_mask)):
                            raise RuntimeError(f"Unexpected ReID shape {reid_feats.shape} for {seq} frame {frame_id}")
                        feature_dim = int(reid_feats.shape[1])
                        if feats.shape[1] != feature_dim:
                            feats = np.zeros((k, feature_dim), dtype=feature_dtype)
                        feats[reid_mask] = reid_feats.astype(feature_dtype)
                        has_reid[reid_mask] = 1.0
                        detections_with_reid += int(np.sum(reid_mask))
                det_table = np.column_stack([
                    np.full((k,), frame_id, dtype=np.float32), frame_det_idx, global_ids,
                    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], score,
                    obj_conf, cls_conf, cls_id, has_reid,
                ]).astype(np.float32)
                det_chunks.append(det_table)
                feat_chunks.append(feats)
                global_det_idx += k
                frame_offsets.append(global_det_idx)
            write_chunk(base, args, seq, spec, out_root, start, end, det_chunks, feat_chunks, image_files, image_wh, frame_offsets, detections_with_reid, failed_images, feature_dim, feature_dtype)
    finally:
        del predictor
        del model
        if encoder is not None:
            del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
