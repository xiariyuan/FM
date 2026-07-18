#!/usr/bin/env python3
from __future__ import annotations

"""Replace cached Phase-0 appearance features with an M23-24 LOSO checkpoint.

Detector outputs and their row order are copied byte-for-value from the source
dump.  Only the GT-free per-detection FastReID embedding is recomputed.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BOT_ROOT = REPO / "external" / "BoT-SORT-main"
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--held-seq", required=True, choices=SEQUENCES)
    parser.add_argument("--source-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seq", action="append", choices=SEQUENCES)
    parser.add_argument("--bot-sort-root", default=str(DEFAULT_BOT_ROOT))
    parser.add_argument("--config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--limit-frames", type=int, default=0, help="Smoke only; 0 processes the full sequence.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    bot_root = Path(args.bot_sort_root).resolve()
    config = (bot_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(f"checkpoint={checkpoint} config={config}")
    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    from fast_reid.fast_reid_interfece import FastReIDInterface

    encoder = FastReIDInterface(str(config), str(checkpoint), "gpu", batch_size=args.batch_size)
    dtype = np.float16 if args.feature_dtype == "float16" else np.float32
    sequences = args.seq or list(SEQUENCES)
    reports: List[Dict[str, object]] = []
    checkpoint_hash = sha256(checkpoint)

    for seq in sequences:
        source = source_root / seq / "dump_yolox_reid.npz"
        destination_dir = output_root / seq
        destination = destination_dir / "dump_yolox_reid.npz"
        if destination.is_file() and not args.overwrite:
            print(f"[M23-24 reembed] skip existing {destination}", flush=True)
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        destination_dir.mkdir(parents=True, exist_ok=True)
        with np.load(source, allow_pickle=False) as archive:
            detections = archive["detections"]
            frame_offsets = archive["frame_offsets"]
            columns = archive["columns"]
            image_files = archive["image_files"]
            image_wh = archive["image_wh"]
        column_index = {str(name): index for index, name in enumerate(columns.tolist())}
        required = {"x1", "y1", "x2", "y2", "has_reid"}
        if not required.issubset(column_index):
            raise RuntimeError(f"missing columns in {source}: {sorted(required.difference(column_index))}")
        total_frames = len(frame_offsets) - 1
        process_frames = total_frames if args.limit_frames <= 0 else min(total_frames, args.limit_frames)
        features = np.zeros((len(detections), 2048), dtype=dtype)
        encoded = 0
        failed_images = 0
        for zero_frame in range(process_frames):
            start = int(frame_offsets[zero_frame])
            end = int(frame_offsets[zero_frame + 1])
            if start == end:
                continue
            image_path = Path(str(image_files[zero_frame]))
            image = cv2.imread(str(image_path))
            if image is None:
                failed_images += 1
                continue
            rows = detections[start:end]
            mask = rows[:, column_index["has_reid"]] > 0.5
            if not np.any(mask):
                continue
            boxes = rows[mask][:, [column_index["x1"], column_index["y1"], column_index["x2"], column_index["y2"]]].astype(np.float32)
            embedding = np.asarray(encoder.inference(image, boxes), dtype=np.float32)
            if embedding.shape != (int(mask.sum()), 2048):
                raise RuntimeError(f"unexpected embedding shape {embedding.shape} seq={seq} frame={zero_frame + 1}")
            row_indices = np.flatnonzero(mask) + start
            features[row_indices] = embedding.astype(dtype)
            encoded += len(row_indices)
            if zero_frame == 0 or (zero_frame + 1) % 100 == 0 or zero_frame + 1 == process_frames:
                print(
                    f"[M23-24 reembed] held={args.held_seq} seq={seq} frame={zero_frame + 1}/{process_frames} encoded={encoded}",
                    flush=True,
                )
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                detections=detections,
                features=features,
                frame_offsets=frame_offsets,
                columns=columns,
                image_files=image_files,
                image_wh=image_wh,
            )
        os.replace(temporary, destination)
        report = {
            "held_sequence": args.held_seq,
            "sequence": seq,
            "source": str(source),
            "destination": str(destination),
            "detector_rows_preserved": int(len(detections)),
            "frames_total": int(total_frames),
            "frames_processed": int(process_frames),
            "features_encoded": int(encoded),
            "failed_images": int(failed_images),
            "feature_dim": 2048,
            "feature_dtype": str(dtype),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "held_gt_read": False,
            "complete": bool(process_frames == total_frames and failed_images == 0),
        }
        (destination_dir / "m23_24_reembed_protocol.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "m23_24_reembed_summary.json").write_text(
        json.dumps({"reports": reports}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
