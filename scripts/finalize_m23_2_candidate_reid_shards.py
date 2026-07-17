"""Finalize the complete M23-2 FastReID cache as contiguous raw shards.

This GT-free finalizer validates the frozen execution plan, every completed
shard report/manifest, byte size, and frame-range continuity. It emits a small
catalog and Merkle-style digest without copying the 1.88 GB raw cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EMBEDDING_DIM = 2048
DTYPE_BYTES = 2
EXPECTED_PLAN_SHA256 = "1fb8306151ed9227acd5c2af8d095eb2e1f21acd1ad3a88b931e9fc068864695"
EXPECTED_APPEARANCE_MANIFEST_SHA256 = "26c55dd63fb912ef3900869fe14c8687b08220fcca8b307848ece1d5f1092dd7"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve(repo: Path, value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".")
    p.add_argument("--shard-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    shard_root = resolve(repo, args.shard_root)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = shard_root / "shard_plan.json"
    plan_sha = sha256_file(plan_path)
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RuntimeError(f"shard plan changed: {plan_sha}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if bool(plan.get("ground_truth_read")) or int(plan.get("trackeval_calls", -1)) != 0:
        raise RuntimeError("shard plan is not GT-free")
    if bool(plan.get("candidate_pool_changed")):
        raise RuntimeError("candidate pool changed")
    if plan.get("appearance_manifest_sha256") != EXPECTED_APPEARANCE_MANIFEST_SHA256:
        raise RuntimeError("appearance manifest changed")

    expected_start = 0
    previous_frame_end = 0
    total_rows = 0
    total_frames = 0
    catalog = []
    combined_frame_rows = []
    frame_columns = None
    merkle = hashlib.sha256()
    norm_min = float("inf")
    norm_max = 0.0
    weighted_norm_sum = 0.0

    for shard in plan["shards"]:
        index = int(shard["shard"])
        start = int(shard["start_crop"])
        end = int(shard["end_crop"])
        rows = end - start
        if start != expected_start or rows <= 0:
            raise RuntimeError(f"non-contiguous shard {index}: [{start}, {end})")
        expected_start = end
        shard_dir = resolve(repo, shard["output_dir"])
        report_path = shard_dir / "report.json"
        manifest_path = shard_dir / "manifest.json"
        embedding_path = shard_dir / "candidate_embeddings.f16"
        summary_path = shard_dir / "frame_summary.csv"
        progress_path = shard_dir / "progress.json"
        for path in (report_path, manifest_path, embedding_path, summary_path, progress_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if report["protocol"]["global_crop_range"] != [start, end]:
            raise RuntimeError(f"shard {index}: report range mismatch")
        if int(report["counts"]["extracted_crops"]) != rows:
            raise RuntimeError(f"shard {index}: extracted count mismatch")
        if int(report["counts"]["nonfinite_values"]) != 0:
            raise RuntimeError(f"shard {index}: non-finite values")
        if not bool(report["validation"]["embedding_size_exact"]) or not bool(report["validation"]["all_features_finite"]):
            raise RuntimeError(f"shard {index}: validation failed")
        if not bool(report["decision"]["candidate_embeddings_ready"]):
            raise RuntimeError(f"shard {index}: embeddings not ready")
        if int(progress["completed_crops"]) != rows or int(progress["global_crop_start"]) != start or int(progress["global_crop_end"]) != end:
            raise RuntimeError(f"shard {index}: progress mismatch")
        expected_bytes = rows * EMBEDDING_DIM * DTYPE_BYTES
        if embedding_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"shard {index}: byte size mismatch")
        report_sha = sha256_file(report_path)
        if report_sha != manifest["report_sha256"]:
            raise RuntimeError(f"shard {index}: report hash mismatch")
        embedding_sha = report["files"]["candidate_embeddings.f16"]["sha256"]
        if embedding_sha != manifest["file_hashes"]["candidate_embeddings.f16"]:
            raise RuntimeError(f"shard {index}: embedding hash contract mismatch")
        if int(report["files"]["candidate_embeddings.f16"]["size_bytes"]) != expected_bytes:
            raise RuntimeError(f"shard {index}: report byte size mismatch")

        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if frame_columns is None:
                frame_columns = reader.fieldnames
            elif reader.fieldnames != frame_columns:
                raise RuntimeError(f"shard {index}: frame columns changed")
            frame_rows = list(reader)
        if not frame_rows or int(frame_rows[0]["crop_start"]) != start or int(frame_rows[-1]["crop_end"]) != end:
            raise RuntimeError(f"shard {index}: frame coverage mismatch")
        for row in frame_rows:
            row_start = int(row["crop_start"])
            row_end = int(row["crop_end"])
            if row_start != previous_frame_end or row_end <= row_start:
                raise RuntimeError(f"frame range [{row_start}, {row_end}) after {previous_frame_end}")
            previous_frame_end = row_end
        combined_frame_rows.extend(frame_rows)
        total_frames += len(frame_rows)
        total_rows += rows
        norm_min = min(norm_min, float(report["validation"]["norm_min"]))
        norm_max = max(norm_max, float(report["validation"]["norm_max"]))
        weighted_norm_sum += float(report["validation"]["norm_mean"]) * rows
        token = f"{index}:{start}:{end}:{embedding_sha}:{expected_bytes}\n".encode("utf-8")
        merkle.update(token)
        catalog.append({
            "shard": index,
            "start_crop": start,
            "end_crop": end,
            "crops": rows,
            "frames": len(frame_rows),
            "embedding_path": str(Path(shard["output_dir"]) / "candidate_embeddings.f16"),
            "embedding_sha256": embedding_sha,
            "embedding_size_bytes": expected_bytes,
            "report_sha256": report_sha,
            "manifest_sha256": sha256_file(manifest_path),
            "frame_summary_sha256": sha256_file(summary_path),
        })

    if total_rows != int(plan["total_crops"]) or expected_start != total_rows or previous_frame_end != total_rows:
        raise RuntimeError("shards do not cover the complete crop set")

    catalog_path = output_dir / "shard_catalog.csv"
    with catalog_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(catalog[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(catalog)
    summary_path = output_dir / "frame_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frame_columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(combined_frame_rows)
    progress = {
        "completed_crops": total_rows,
        "total_crops": total_rows,
        "global_crop_start": 0,
        "global_crop_end": total_rows,
        "global_completed_crops": total_rows,
        "completed_frames": total_frames,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "little-endian float16",
        "layout": "contiguous_shards",
        "shard_count": len(catalog),
        "sequence": combined_frame_rows[-1]["sequence"],
        "frame": int(combined_frame_rows[-1]["frame"]),
    }
    canonical_json_dump(progress, output_dir / "progress.json")
    report = {
        "schema": "fmtrack.m23_2.candidate_reid_sharded_cache.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "candidate_pool_changed": False,
            "candidate_plan_recomputed": False,
            "appearance_manifest_sha256": EXPECTED_APPEARANCE_MANIFEST_SHA256,
            "shard_plan_sha256": plan_sha,
            "raw_cache_layout": "contiguous complete-frame shards",
            "global_embedding_order": "unique_candidate_crops.npy row order",
            "embedding_dtype": "little-endian float16",
            "embedding_dim": EMBEDDING_DIM,
            "engine": plan["engine"],
            "batch_size": int(plan["batch_size"]),
        },
        "counts": {
            "candidate_crops": total_rows,
            "frames_processed": total_frames,
            "shards": len(catalog),
            "nonfinite_values": 0,
            "raw_bytes": total_rows * EMBEDDING_DIM * DTYPE_BYTES,
        },
        "validation": {
            "all_shards_complete": True,
            "all_shards_finite": True,
            "ranges_contiguous": True,
            "frame_summaries_contiguous": True,
            "byte_sizes_exact": True,
            "report_manifest_contracts_valid": True,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "norm_mean": weighted_norm_sum / total_rows,
            "sharded_content_merkle_sha256": merkle.hexdigest(),
        },
        "shards": catalog,
        "files": {
            "shard_catalog.csv": {"sha256": sha256_file(catalog_path), "rows": len(catalog)},
            "frame_summary.csv": {"sha256": sha256_file(summary_path), "rows": total_frames},
            "progress.json": {"sha256": sha256_file(output_dir / "progress.json")},
        },
        "decision": {
            "candidate_embeddings_ready": True,
            "compact_appearance_features_ready": False,
            "deployment_allowed": False,
            "locked_manifest_created": False,
        },
        "locked_state": {
            "p15_policy": "no_op",
            "locked_label_reads": 0,
            "locked_trackeval_calls": 0,
            "remaining_locked_rows_untouched": 156,
        },
    }
    canonical_json_dump(report, output_dir / "report.json")
    manifest = {
        "schema": "fmtrack.m23_2.candidate_reid_sharded_cache.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {name: sha256_file(output_dir / name) for name in ("shard_catalog.csv", "frame_summary.csv", "progress.json", "report.json")},
        "sharded_content_merkle_sha256": merkle.hexdigest(),
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps({
        "candidate_crops": total_rows,
        "frames_processed": total_frames,
        "shards": len(catalog),
        "raw_bytes": total_rows * EMBEDDING_DIM * DTYPE_BYTES,
        "sharded_content_merkle_sha256": merkle.hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
