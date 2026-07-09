#!/usr/bin/env python3
"""Merge chunked Phase0 npz dumps into a standard per-sequence dump."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

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
    ap = argparse.ArgumentParser(description="Merge chunked Phase0 dumps.")
    ap.add_argument("--chunks-root", required=True, help="Root containing chunk dirs, each with <seq>/dump_yolox_reid.npz")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def load_manifest(chunk_seq_dir: Path) -> Dict:
    p = chunk_seq_dir / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})


def main() -> None:
    args = parse_args()
    chunks_root = Path(args.chunks_root)
    seq = args.seq
    candidates = sorted(chunks_root.glob(f"*/{seq}/dump_yolox_reid.npz"))
    if not candidates:
        raise FileNotFoundError(f"No chunks found under {chunks_root} for {seq}")
    chunk_items = []
    for npz in candidates:
        seq_dir = npz.parent
        manifest = load_manifest(seq_dir)
        start = int(manifest.get("chunk_frame_start", 0))
        end = int(manifest.get("chunk_frame_end", 0))
        if start <= 0 or end < start:
            raise ValueError(f"Invalid chunk frame range in {seq_dir}: {start}-{end}")
        chunk_items.append((start, end, npz, manifest))
    chunk_items.sort(key=lambda x: x[0])

    # Basic continuity check.
    expected = chunk_items[0][0]
    for start, end, npz, _manifest in chunk_items:
        if start != expected:
            raise ValueError(f"Non-contiguous chunks for {seq}: expected start {expected}, got {start} at {npz}")
        expected = end + 1

    out_seq = Path(args.out_root) / seq
    if out_seq.exists() and any(out_seq.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_seq}. Pass --overwrite")
    out_seq.mkdir(parents=True, exist_ok=True)

    det_chunks = []
    feat_chunks = []
    image_files = []
    image_wh = []
    frame_offsets = [0]
    global_base = 0
    feature_dim = 0
    feature_dtype = None
    det_with_reid = 0
    started = now_iso()

    for start, end, npz, manifest in chunk_items:
        z = np.load(npz, allow_pickle=True)
        det = z["detections"].astype(np.float32).copy()
        feat = z["features"]
        offsets = z["frame_offsets"].astype(np.int64)
        if det.size:
            det[:, 2] += float(global_base)
            det_chunks.append(det)
        if feat.size:
            feat_chunks.append(feat)
            feature_dim = int(feat.shape[1])
            feature_dtype = feat.dtype
        else:
            feature_dim = int(feat.shape[1]) if feat.ndim == 2 else feature_dim
        image_files.extend([str(x) for x in z["image_files"].tolist()])
        image_wh.append(z["image_wh"].astype(np.int32))
        counts = np.diff(offsets)
        for c in counts.tolist():
            frame_offsets.append(frame_offsets[-1] + int(c))
        global_base += int(det.shape[0])
        det_with_reid += int(manifest.get("detections_with_reid", 0))

    detections = np.concatenate(det_chunks, axis=0) if det_chunks else np.zeros((0, len(DETECTION_COLUMNS)), dtype=np.float32)
    if feat_chunks:
        features = np.concatenate(feat_chunks, axis=0)
    else:
        features = np.zeros((detections.shape[0], feature_dim), dtype=np.float16)
    image_wh_arr = np.concatenate(image_wh, axis=0) if image_wh else np.zeros((0, 2), dtype=np.int32)
    frame_offsets_arr = np.asarray(frame_offsets, dtype=np.int64)
    if int(frame_offsets_arr[-1]) != int(detections.shape[0]):
        raise RuntimeError(f"frame_offsets end {frame_offsets_arr[-1]} != detections {detections.shape[0]}")
    if features.shape[0] != detections.shape[0]:
        raise RuntimeError(f"features rows {features.shape[0]} != detections {detections.shape[0]}")

    npz_path = out_seq / "dump_yolox_reid.npz"
    np.savez(
        npz_path,
        detections=detections,
        features=features,
        frame_offsets=frame_offsets_arr,
        columns=np.asarray(DETECTION_COLUMNS),
        image_files=np.asarray(image_files),
        image_wh=image_wh_arr,
    )
    manifest_path = out_seq / "manifest.json"
    manifest = {
        "status": "completed",
        "phase": "DMM_PHASE0_CHUNK_MERGE",
        "seq": seq,
        "chunks_root": str(chunks_root),
        "chunks": [
            {"start": int(s), "end": int(e), "npz": str(p)} for s, e, p, _m in chunk_items
        ],
        "npz_path": str(npz_path),
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "detections": int(detections.shape[0]),
        "detections_with_reid": int(det_with_reid),
        "frames_total": int(len(frame_offsets_arr) - 1),
        "started_at": started,
        "finished_at": now_iso(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    row = {
        "seq": seq,
        "status": "completed",
        "frames_total": int(len(frame_offsets_arr) - 1),
        "frames_processed": int(len(frame_offsets_arr) - 1),
        "detections": int(detections.shape[0]),
        "detections_with_reid": int(det_with_reid),
        "feature_dim": int(features.shape[1]),
        "npz_path": str(npz_path),
        "csv_path": "",
        "manifest_path": str(manifest_path),
        "started_at": started,
        "finished_at": manifest["finished_at"],
        "notes": f"merged_chunks={len(chunk_items)}; feature_dtype={features.dtype}",
    }
    write_summary(Path(args.out_root) / "summary.csv", [row])
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
