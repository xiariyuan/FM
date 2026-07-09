#!/usr/bin/env python3
"""Phase 0 dump: YOLOX detections + FastReID features for independent tracker work.

This script intentionally does NOT instantiate BoTSORT. It reuses the same YOLOX
predictor path and FastReID encoder used by external/BoT-SORT-main/tools/track.py,
then writes per-sequence arrays that Phase 1/2 trackers can replay without doing
GPU detector/ReID inference again.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import os.path as osp
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPO_ROOT / "external" / "BoT-SORT-main"
IMAGE_EXTS = {".jpg", ".jpeg", ".webp", ".bmp", ".png"}
DETECTION_COLUMNS = [
    "frame", "frame_det_idx", "global_det_idx", "x1", "y1", "x2", "y2",
    "score", "obj_conf", "cls_conf", "class_id", "has_reid",
]
SUMMARY_FIELDS = [
    "seq", "status", "frames_total", "frames_processed", "detections",
    "detections_with_reid", "feature_dim", "npz_path", "csv_path",
    "manifest_path", "started_at", "finished_at", "notes",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Dump MOT YOLOX detections and aligned ReID features for DMM Phase 0."
    )
    parser.add_argument("--data-root", default="/gemini/code/datasets", help="dataset root containing MOT20/MOT17")
    parser.add_argument("--benchmark", default="MOT20", choices=["MOT20", "MOT17"], help="benchmark name")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="train/val/test split; val maps to MOT train sequences")
    parser.add_argument("--seq-ids", nargs="+", type=int, default=[1], help="sequence ids to dump, e.g. --seq-ids 1 2 3 5")
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "outputs" / f"dmm_phase0_yolox_reid_{ts}"),
        help="output root for per-sequence dumps",
    )
    parser.add_argument("--bot-sort-root", default=str(BOT_ROOT), help="external/BoT-SORT-main root")
    parser.add_argument("--exp-file", default="./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py")
    parser.add_argument("--ckpt", default="./pretrained/bytetrack_x_mot20.pth.tar")
    parser.add_argument("--fast-reid-config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", default="pretrained/mot20_sbs_S50.pth")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fp16", action="store_true", help="run YOLOX detector in fp16")
    parser.add_argument("--fuse", action="store_true", help="fuse YOLOX conv/bn before inference")
    parser.add_argument("--conf", type=float, default=None, help="override YOLOX postprocess confidence")
    parser.add_argument("--nms", type=float, default=None, help="override YOLOX NMS threshold")
    parser.add_argument("--tsize", type=int, default=None, help="override square test size")
    parser.add_argument("--track-low-thresh", type=float, default=0.10, help="used to derive default detector conf = low - 0.01")
    parser.add_argument("--dump-min-score", type=float, default=0.0, help="minimum obj*cls score to keep in dump")
    parser.add_argument("--reid-min-score", type=float, default=0.0, help="minimum obj*cls score to extract ReID features")
    parser.add_argument("--reid-batch-size", type=int, default=64)
    parser.add_argument("--no-reid", action="store_true", help="dump detections only")
    parser.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--limit-frames", type=int, default=0, help="debug/smoke limit; 0 means all frames")
    parser.add_argument("--frame-start", type=int, default=1, help="1-based first frame to dump after sorting image files")
    parser.add_argument("--frame-end", type=int, default=0, help="1-based inclusive last frame to dump; 0 means sequence end")
    parser.add_argument("--frame-start", type=int, default=1, help="1-based first frame to dump after sorting image files")
    parser.add_argument("--frame-end", type=int, default=0, help="1-based inclusive last frame to dump; 0 means sequence end")
    parser.add_argument("--write-csv", action="store_true", default=True, help="write human-readable detections.csv")
    parser.add_argument("--no-csv", dest="write_csv", action="store_false")
    parser.add_argument("--compress", action="store_true", help="use np.savez_compressed; slower but smaller")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_under(root: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return root / path


def read_seqinfo(seq_dir: Path) -> Dict[str, object]:
    seqinfo = seq_dir / "seqinfo.ini"
    result: Dict[str, object] = {
        "im_dir": "img1",
        "im_ext": ".jpg",
        "seq_length": 0,
        "frame_rate": 30,
        "width": 0,
        "height": 0,
    }
    if not seqinfo.is_file():
        return result
    cfg = configparser.ConfigParser()
    cfg.read(seqinfo)
    if not cfg.has_section("Sequence"):
        return result
    sec = cfg["Sequence"]
    result["im_dir"] = sec.get("imDir", "img1")
    result["im_ext"] = sec.get("imExt", ".jpg")
    result["seq_length"] = int(sec.get("seqLength", "0"))
    result["frame_rate"] = int(sec.get("frameRate", "30"))
    result["width"] = int(sec.get("imWidth", "0"))
    result["height"] = int(sec.get("imHeight", "0"))
    return result


def build_sequence_specs(data_root: Path, benchmark: str, split: str, seq_ids: Sequence[int]) -> List[Dict[str, object]]:
    if benchmark == "MOT20":
        train_ids = [1, 2, 3, 5]
        test_ids = [4, 6, 7, 8]
        prefix = "MOT20"
    elif benchmark == "MOT17":
        train_ids = [2, 4, 5, 9, 10, 11, 13]
        test_ids = [1, 3, 6, 7, 8, 12, 14]
        prefix = "MOT17"
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

    allowed = train_ids if split in {"train", "val"} else test_ids
    selected = [sid for sid in allowed if sid in set(seq_ids)]
    if not selected:
        raise ValueError(f"No selected sequences for benchmark={benchmark} split={split} seq_ids={seq_ids}")

    specs: List[Dict[str, object]] = []
    for sid in selected:
        seq = f"{prefix}-{sid:02d}"
        physical_split = "train" if sid in train_ids else "test"
        seq_dir = data_root / benchmark / physical_split / seq
        info = read_seqinfo(seq_dir)
        img_dir = seq_dir / str(info["im_dir"])
        specs.append({
            "seq": seq,
            "seq_id": sid,
            "seq_dir": str(seq_dir),
            "img_dir": str(img_dir),
            "physical_split": physical_split,
            "frame_rate": int(info["frame_rate"]),
            "seqinfo": info,
        })
    return specs


def image_list(img_dir: Path) -> List[Path]:
    files: List[Path] = []
    for root, _dirs, names in os.walk(img_dir):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
    return sorted(files)


def read_image_robust(img_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(img_path))
    if img is not None:
        return img
    try:
        data = np.fromfile(str(img_path), dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        return None
    return None


def l2norm(feats: np.ndarray) -> np.ndarray:
    if feats.size == 0:
        return feats
    denom = np.linalg.norm(feats, axis=1, keepdims=True)
    denom[denom < 1e-12] = 1.0
    return feats / denom


class YoloXPredictor:
    def __init__(self, model: torch.nn.Module, exp, device: torch.device, fp16: bool) -> None:
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img_path: Path) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        raw_img = read_image_robust(img_path)
        if raw_img is None:
            return None, {"img_path": str(img_path), "height": 0, "width": 0, "ratio": 0.0, "raw_img": None}
        h, w = raw_img.shape[:2]
        img, ratio = preproc(raw_img, self.test_size, self.rgb_means, self.std)
        tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            tensor = tensor.half()
        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        arr = None if outputs[0] is None else outputs[0].detach().cpu().numpy()
        return arr, {"img_path": str(img_path), "height": h, "width": w, "ratio": ratio, "raw_img": raw_img}


def init_detector(args: argparse.Namespace, seq_name: str):
    bot_root = Path(args.bot_sort_root).resolve()
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    from yolox.exp import get_exp
    from yolox.utils import fuse_model, get_model_info

    exp_file = resolve_under(bot_root, args.exp_file)
    ckpt_file = resolve_under(bot_root, args.ckpt)
    exp = get_exp(str(exp_file), seq_name)
    if args.conf is None:
        exp.test_conf = max(0.001, float(args.track_low_thresh) - 0.01)
    else:
        exp.test_conf = float(args.conf)
    if args.nms is not None:
        exp.nmsthre = float(args.nms)
    if args.tsize is not None:
        exp.test_size = (int(args.tsize), int(args.tsize))
    elif args.benchmark == "MOT20" and seq_name in {"MOT20-06", "MOT20-08"}:
        # Match original BoT-SORT MOT20 defaults in tools/track.py.
        exp.test_size = (736, 1920)
        print(f"[phase0] seq={seq_name} using MOT20 special test_size={exp.test_size}", flush=True)

    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was requested, but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = exp.get_model().to(device)
    model.eval()
    print(f"[phase0] model={get_model_info(model, exp.test_size)}", flush=True)
    print(f"[phase0] loading detector checkpoint: {ckpt_file}", flush=True)
    ckpt = torch.load(str(ckpt_file), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()
    return exp, YoloXPredictor(model, exp, device, args.fp16), model


def init_encoder(args: argparse.Namespace):
    if args.no_reid:
        return None
    bot_root = Path(args.bot_sort_root).resolve()
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    from fast_reid.fast_reid_interfece import FastReIDInterface

    cfg = resolve_under(bot_root, args.fast_reid_config)
    weights = resolve_under(bot_root, args.fast_reid_weights)
    print(f"[phase0] loading FastReID config={cfg} weights={weights}", flush=True)
    return FastReIDInterface(str(cfg), str(weights), args.device, batch_size=int(args.reid_batch_size))


def write_csv(path: Path, columns: Sequence[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row.tolist())


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def dump_sequence(args: argparse.Namespace, predictor: YoloXPredictor, encoder, spec: Dict[str, object], out_root: Path) -> Dict[str, object]:
    seq = str(spec["seq"])
    started = now_iso()
    seq_out = out_root / seq
    if seq_out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {seq_out}. Pass --overwrite to replace/update.")
    seq_out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(str(spec["img_dir"]))
    all_files = image_list(img_dir)
    if not all_files:
        raise FileNotFoundError(f"No image files under {img_dir}")
    frame_start = max(1, int(getattr(args, "frame_start", 1)))
    frame_end_arg = int(getattr(args, "frame_end", 0) or 0)
    frame_end = len(all_files) if frame_end_arg <= 0 else min(len(all_files), frame_end_arg)
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame range for {seq}: frame_start={frame_start} frame_end={frame_end}")
    files = all_files[frame_start - 1:frame_end]
    if args.limit_frames and args.limit_frames > 0:
        files = files[: int(args.limit_frames)]

    det_chunks: List[np.ndarray] = []
    feat_chunks: List[np.ndarray] = []
    image_files: List[str] = []
    image_wh: List[Tuple[int, int]] = []
    frame_offsets = [0]
    global_det_idx = 0
    feature_dim = 2048 if not args.no_reid else 0
    detections_with_reid = 0
    failed_images = 0
    feature_dtype = np.float16 if args.feature_dtype == "float16" else np.float32

    for local_idx, img_path in enumerate(files, start=0):
        frame_id = frame_start + local_idx
        if local_idx == 0 or (local_idx + 1) % 100 == 0 or (local_idx + 1) == len(files):
            print(f"[phase0] seq={seq} frame={frame_id} local={local_idx + 1}/{len(files)} dets_so_far={global_det_idx}", flush=True)
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
                reid_boxes = boxes[reid_mask].astype(np.float32)
                reid_feats = np.asarray(encoder.inference(raw_img, reid_boxes), dtype=np.float32)
                reid_feats = l2norm(reid_feats)
                if reid_feats.ndim == 2 and reid_feats.shape[0] == int(np.sum(reid_mask)):
                    feature_dim = int(reid_feats.shape[1])
                    if feats.shape[1] != feature_dim:
                        feats = np.zeros((k, feature_dim), dtype=feature_dtype)
                    feats[reid_mask] = reid_feats.astype(feature_dtype)
                    has_reid[reid_mask] = 1.0
                    detections_with_reid += int(np.sum(reid_mask))
                else:
                    raise RuntimeError(f"FastReID returned unexpected shape for {seq} frame {frame_id}: {reid_feats.shape}")

        det_table = np.column_stack([
            np.full((k,), frame_id, dtype=np.float32), frame_det_idx, global_ids,
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], score, obj_conf,
            cls_conf, cls_id, has_reid,
        ]).astype(np.float32)
        det_chunks.append(det_table)
        feat_chunks.append(feats)
        global_det_idx += k
        frame_offsets.append(global_det_idx)

    detections = np.concatenate(det_chunks, axis=0) if det_chunks else np.zeros((0, len(DETECTION_COLUMNS)), dtype=np.float32)
    if feature_dim > 0:
        features = np.concatenate(feat_chunks, axis=0) if feat_chunks else np.zeros((0, feature_dim), dtype=feature_dtype)
    else:
        features = np.zeros((detections.shape[0], 0), dtype=np.float32)

    npz_path = seq_out / "dump_yolox_reid.npz"
    csv_path = seq_out / "detections.csv"
    manifest_path = seq_out / "manifest.json"
    save_kwargs = {
        "detections": detections,
        "features": features,
        "frame_offsets": np.asarray(frame_offsets, dtype=np.int64),
        "columns": np.asarray(DETECTION_COLUMNS),
        "image_files": np.asarray(image_files),
        "image_wh": np.asarray(image_wh, dtype=np.int32),
    }
    if args.compress:
        np.savez_compressed(npz_path, **save_kwargs)
    else:
        np.savez(npz_path, **save_kwargs)
    csv_path_text = str(csv_path)
    if args.write_csv:
        write_csv(csv_path, DETECTION_COLUMNS, detections)
    else:
        csv_path_text = ""

    manifest = {
        "status": "completed",
        "phase": "DMM_PHASE0_YOLOX_REID_DUMP",
        "seq": seq,
        "spec": spec,
        "npz_path": str(npz_path),
        "csv_path": csv_path_text,
        "columns": DETECTION_COLUMNS,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "detections": int(detections.shape[0]),
        "detections_with_reid": int(detections_with_reid),
        "frames_total": int(len(files)),
        "frame_start": int(frame_start),
        "frame_end": int(frame_start + len(files) - 1),
        "failed_images": int(failed_images),
        "detector_test_size": list(predictor.test_size),
        "args": vars(args),
        "command": " ".join(shlex.quote(v) for v in sys.argv),
        "started_at": started,
        "finished_at": now_iso(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {
        "seq": seq,
        "status": "completed",
        "frames_total": len(files),
        "frames_processed": len(files) - failed_images,
        "detections": int(detections.shape[0]),
        "detections_with_reid": int(detections_with_reid),
        "feature_dim": int(features.shape[1]),
        "npz_path": str(npz_path),
        "csv_path": csv_path_text,
        "manifest_path": str(manifest_path),
        "started_at": started,
        "finished_at": manifest["finished_at"],
        "notes": f"failed_images={failed_images}; feature_dtype={features.dtype}; test_size={tuple(predictor.test_size)}",
    }


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bot_root = Path(args.bot_sort_root).resolve()
    if not bot_root.is_dir():
        raise FileNotFoundError(f"BoT-SORT root not found: {bot_root}")
    data_root = Path(args.data_root)
    specs = build_sequence_specs(data_root, args.benchmark, args.split, args.seq_ids)
    print(f"[phase0] output root: {out_root}", flush=True)
    print(f"[phase0] sequences: {[s['seq'] for s in specs]}", flush=True)

    encoder = init_encoder(args)
    rows: List[Dict[str, object]] = []
    try:
        for spec in specs:
            exp, predictor, model = init_detector(args, str(spec["seq"]))
            try:
                rows.append(dump_sequence(args, predictor, encoder, spec, out_root))
                write_summary(out_root / "summary.csv", rows)
            finally:
                del predictor
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if encoder is not None:
            del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    root_manifest = {
        "status": "completed",
        "phase": "DMM_PHASE0_YOLOX_REID_DUMP",
        "out_root": str(out_root),
        "summary_csv": str(out_root / "summary.csv"),
        "sequences": [row["seq"] for row in rows],
        "rows": rows,
        "args": vars(args),
        "command": " ".join(shlex.quote(v) for v in sys.argv),
        "finished_at": now_iso(),
    }
    with (out_root / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(root_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_summary(out_root / "summary.csv", rows)
    print(f"[phase0] completed. summary={out_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
