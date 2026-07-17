"""Finalize a complete-frame prefix of an interrupted M23-2 ReID shard.

The raw embedding bytes and progress boundary already exist. This GT-free tool
verifies the byte length, reconstructs frame-level norm summaries from the
frozen crop order, and emits the same report/manifest contract as a completed
extractor shard.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 2048
EXPECTED_APPEARANCE_MANIFEST_SHA256 = "26c55dd63fb912ef3900869fe14c8687b08220fcca8b307848ece1d5f1092dd7"
SEQUENCE_BY_ID = {0: "MOT20-01", 1: "MOT20-02", 2: "MOT20-03", 3: "MOT20-05"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dump(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(repo: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".")
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--start-crop", type=int, required=True)
    p.add_argument("--end-crop", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--engine", default="efficient")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    manifest_dir = resolve(repo, args.manifest_dir)
    shard_dir = resolve(repo, args.shard_dir)
    manifest_sha = sha256_file(manifest_dir / "manifest.json")
    if manifest_sha != EXPECTED_APPEARANCE_MANIFEST_SHA256:
        raise RuntimeError(f"appearance manifest changed: {manifest_sha}")
    crops = np.load(manifest_dir / "unique_candidate_crops.npy", allow_pickle=False, mmap_mode="r")
    start, end = int(args.start_crop), int(args.end_crop)
    if start < 0 or end > len(crops) or start >= end:
        raise RuntimeError("invalid crop range")
    if start > 0 and int(crops[start - 1]["sequence_id"]) == int(crops[start]["sequence_id"]) and int(crops[start - 1]["frame"]) == int(crops[start]["frame"]):
        raise RuntimeError("start splits a frame")
    if end < len(crops) and int(crops[end - 1]["sequence_id"]) == int(crops[end]["sequence_id"]) and int(crops[end - 1]["frame"]) == int(crops[end]["frame"]):
        raise RuntimeError("end splits a frame")

    progress_path = shard_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if int(progress["completed_crops"]) != end - start:
        raise RuntimeError("progress does not match requested prefix")
    embedding_path = shard_dir / "candidate_embeddings.f16"
    expected_bytes = (end - start) * EMBEDDING_DIM * np.dtype("<f2").itemsize
    if embedding_path.stat().st_size != expected_bytes:
        raise RuntimeError("embedding byte size does not match prefix")
    embeddings = np.memmap(embedding_path, dtype="<f2", mode="r", shape=(end - start, EMBEDDING_DIM))

    rows = []
    norm_min = float("inf")
    norm_max = 0.0
    norm_sum = 0.0
    cursor = 0
    selected = crops[start:end]
    while cursor < len(selected):
        sid = int(selected[cursor]["sequence_id"])
        frame = int(selected[cursor]["frame"])
        block_end = cursor + 1
        while block_end < len(selected) and int(selected[block_end]["sequence_id"]) == sid and int(selected[block_end]["frame"]) == frame:
            block_end += 1
        values = np.asarray(embeddings[cursor:block_end], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite prefix embedding")
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum())
        rows.append({
            "sequence": SEQUENCE_BY_ID[sid],
            "frame": frame,
            "crop_start": start + cursor,
            "crop_end": start + block_end,
            "crops": block_end - cursor,
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
        })
        cursor = block_end

    summary_path = shard_dir / "frame_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "frame", "crop_start", "crop_end", "crops", "norm_min", "norm_max", "norm_mean"], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    final_progress = {
        "completed_crops": end - start,
        "total_crops": end - start,
        "global_crop_start": start,
        "global_crop_end": end,
        "global_completed_crops": end,
        "completed_frames": len(rows),
        "sequence": rows[-1]["sequence"],
        "frame": int(rows[-1]["frame"]),
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "little-endian float16",
    }
    canonical_json_dump(final_progress, progress_path)
    report = {
        "schema": "fmtrack.m23_2.candidate_reid_extraction.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "candidate_plan_recomputed": False,
            "candidate_pool_changed": False,
            "appearance_manifest_sha256": manifest_sha,
            "crop_order": "unique_candidate_crops.npy row order",
            "global_crop_range": [start, end],
            "crop_semantics": "FastReID integer truncation, image-bound clipping, x2/y2-exclusive slicing",
            "embedding_dtype": "little-endian float16",
            "embedding_dim": EMBEDDING_DIM,
            "device": "gpu",
            "batch_size": int(args.batch_size),
            "engine": args.engine,
            "finalized_from_complete_frame_progress": True,
            "start_crop": start,
            "end_crop": end,
            "limit_crops": 0,
        },
        "counts": {
            "available_crops": int(len(crops)),
            "extracted_crops": end - start,
            "global_crop_start": start,
            "global_crop_end": end,
            "frames_processed": len(rows),
            "nonfinite_values": 0,
        },
        "validation": {
            "embedding_size_exact": True,
            "all_features_finite": True,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "norm_mean": norm_sum / (end - start),
        },
        "sources": {
            "appearance_manifest": {"path": str(manifest_dir / "manifest.json"), "sha256": manifest_sha},
            "candidate_crops": {"path": str(manifest_dir / "unique_candidate_crops.npy"), "sha256": sha256_file(manifest_dir / "unique_candidate_crops.npy")},
        },
        "files": {
            "candidate_embeddings.f16": {"sha256": sha256_file(embedding_path), "size_bytes": expected_bytes, "dtype": "little-endian float16", "shape": [end - start, EMBEDDING_DIM]},
            "frame_summary.csv": {"sha256": sha256_file(summary_path), "rows": len(rows)},
            "progress.json": {"sha256": sha256_file(progress_path)},
        },
        "decision": {"candidate_embeddings_ready": True, "compact_appearance_features_ready": False, "deployment_allowed": False, "locked_manifest_created": False},
        "locked_state": {"p15_policy": "no_op", "locked_label_reads": 0, "locked_trackeval_calls": 0, "remaining_locked_rows_untouched": 156},
    }
    canonical_json_dump(report, shard_dir / "report.json")
    manifest = {
        "schema": "fmtrack.m23_2.candidate_reid_extraction.manifest.v1",
        "report_sha256": sha256_file(shard_dir / "report.json"),
        "file_hashes": {name: sha256_file(shard_dir / name) for name in ("candidate_embeddings.f16", "frame_summary.csv", "progress.json", "report.json")},
    }
    canonical_json_dump(manifest, shard_dir / "manifest.json")
    print(json.dumps({"start": start, "end": end, "crops": end - start, "frames": len(rows), "embedding_sha256": report["files"]["candidate_embeddings.f16"]["sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
