#!/usr/bin/env python3
"""Chunked Phase 0 dump helper.

Runs scripts/dump_yolox_reid_phase0.py functions on a frame slice so long MOT20
test sequences can be dumped in resumable chunks under tool/runtime limits.
Chunk detections store GLOBAL frame numbers but chunk-local frame_offsets.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "scripts" / "dump_yolox_reid_phase0.py"


def load_base():
    spec = importlib.util.spec_from_file_location("phase0_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase0_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dump one sequence frame slice for Phase0.")
    ap.add_argument("--data-root", default="/gemini/code/datasets")
    ap.add_argument("--benchmark", default="MOT20", choices=["MOT20", "MOT17"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seq-id", type=int, required=True)
    ap.add_argument("--frame-start", type=int, required=True)
    ap.add_argument("--frame-end", type=int, required=True)
    ap.add_argument("--out-root", required=True)
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
    ap.add_argument("--write-csv", action="store_true", default=False)
    ap.add_argument("--no-csv", dest="write_csv", action="store_false")
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    # Attributes expected by base.dump_sequence.
    ap.set_defaults(limit_frames=0)
    return ap.parse_args()


def main() -> None:
    base = load_base()
    args = parse_args()
    if args.frame_start < 1 or args.frame_end < args.frame_start:
        raise ValueError(f"Invalid frame range: {args.frame_start}-{args.frame_end}")
    # base.build_sequence_specs expects seq_ids.
    args.seq_ids = [int(args.seq_id)]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    specs = base.build_sequence_specs(Path(args.data_root), args.benchmark, args.split, args.seq_ids)
    if len(specs) != 1:
        raise RuntimeError(f"Expected one spec, got {len(specs)}")
    spec = specs[0]
    seq = str(spec["seq"])
    img_dir = Path(str(spec["img_dir"]))
    all_files = base.image_list(img_dir)
    if not all_files:
        raise FileNotFoundError(f"No images under {img_dir}")
    frame_end = min(int(args.frame_end), len(all_files))
    frame_start = int(args.frame_start)
    if frame_start > frame_end:
        raise ValueError(f"Empty slice after clipping: {frame_start}-{frame_end} total={len(all_files)}")
    selected = all_files[frame_start - 1:frame_end]
    print(f"[chunk] seq={seq} frames={frame_start}-{frame_end} count={len(selected)} out={out_root}", flush=True)

    original_image_list = base.image_list
    base.image_list = lambda _img_dir: list(selected)
    encoder = base.init_encoder(args)
    exp, predictor, model = base.init_detector(args, seq)
    try:
        row = base.dump_sequence(args, predictor, encoder, spec, out_root)
    finally:
        base.image_list = original_image_list
        del predictor
        del model
        if encoder is not None:
            del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seq_out = out_root / seq
    npz_path = seq_out / "dump_yolox_reid.npz"
    z = np.load(npz_path, allow_pickle=True)
    arrays = {k: z[k] for k in z.files}
    det = arrays["detections"].copy()
    if det.size:
        det[:, 0] += float(frame_start - 1)
        arrays["detections"] = det
    np.savez(npz_path, **arrays)

    manifest_path = seq_out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["chunk_frame_start"] = int(frame_start)
    manifest["chunk_frame_end"] = int(frame_end)
    manifest["chunk_frame_count"] = int(len(selected))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    row["notes"] = str(row.get("notes", "")) + f"; chunk={frame_start}-{frame_end}"
    base.write_summary(out_root / "summary.csv", [row])
    print(f"[chunk] completed seq={seq} frames={frame_start}-{frame_end} dets={row['detections']}", flush=True)


if __name__ == "__main__":
    main()
