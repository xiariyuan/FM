"""Extract GT-free FastReID embeddings for the frozen M23-2 candidate crops.

The crop order is fixed by ``appearance_manifest_v1/unique_candidate_crops.npy``.
Embeddings are emitted as a raw little-endian float16 matrix in that exact order.
The extractor never accepts a ground-truth path and never invokes TrackEval.

The raw cache is intentionally separated from compact deployable appearance
features. It can remain outside Git while its hash, model assets, crop manifest,
and deterministic progress metadata are retained in formal reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
SEQUENCE_BY_ID = {index: sequence for index, sequence in enumerate(SEQUENCES)}
EMBEDDING_DIM = 2048
EXPECTED_APPEARANCE_MANIFEST_SHA256 = "26c55dd63fb912ef3900869fe14c8687b08220fcca8b307848ece1d5f1092dd7"


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


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bot-sort-root", default="external/BoT-SORT-main")
    parser.add_argument("--fast-reid-config", default="fast_reid/configs/MOT20/sbs_S50.yml")
    parser.add_argument("--fast-reid-weights", default="pretrained/mot20_sbs_S50.pth")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--engine", choices=("interface", "efficient"), default="efficient")
    parser.add_argument("--start-crop", type=int, default=0)
    parser.add_argument("--end-crop", type=int, default=0)
    parser.add_argument("--limit-crops", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def infer_frame_efficient(encoder, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Replicate FastReIDInterface.inference with batched host preprocessing.

    The original interface transfers every crop to CUDA separately before
    stacking. Here the identical RGB uint8 patches are stacked on the host and
    transferred once per model batch. Frame-local batch boundaries are kept
    unchanged so outputs can be checked byte-for-byte against the reference
    interface.
    """
    if boxes is None or np.size(boxes) == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    height, width = image.shape[:2]
    outputs = []
    patches = []

    def run_batch(batch_patches: list[np.ndarray]) -> None:
        array = np.stack(batch_patches, axis=0).astype("float32", copy=False)
        tensor = torch.as_tensor(array.transpose(0, 3, 1, 2))
        tensor = tensor.to(device=encoder.device)
        tensor = tensor.half() if encoder.device != "cpu" else tensor.float()
        # The reference interface moves each non-contiguous CHW patch to CUDA
        # and then stacks the patches. ``torch.stack`` materializes a contiguous
        # NCHW batch. Preserve that exact model-input layout while still using a
        # single host-to-device transfer for the complete batch.
        tensor = tensor.contiguous()
        with torch.no_grad():
            pred = encoder.model(tensor)
            pred[torch.isinf(pred)] = 1.0
            feat = F.normalize(pred).cpu().numpy()
        if not np.isfinite(feat).all():
            raise RuntimeError("non-finite FastReID output in efficient engine")
        outputs.append(feat.astype(np.float32, copy=False))

    for detection in boxes:
        tlbr = detection[:4].astype(np.int_)
        tlbr[0] = max(0, tlbr[0])
        tlbr[1] = max(0, tlbr[1])
        tlbr[2] = min(width - 1, tlbr[2])
        tlbr[3] = min(height - 1, tlbr[3])
        patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], :]
        if patch.size == 0:
            raise RuntimeError(f"invalid FastReID crop {tuple(tlbr.tolist())}")
        patch = patch[:, :, ::-1]
        patch = cv2.resize(
            patch,
            tuple(encoder.cfg.INPUT.SIZE_TEST[::-1]),
            interpolation=cv2.INTER_LINEAR,
        )
        patches.append(patch)
        if len(patches) == int(encoder.batch_size):
            run_batch(patches)
            patches = []
    if patches:
        run_batch(patches)
    return np.concatenate(outputs, axis=0)


def load_progress(path: Path) -> dict:
    if not path.is_file():
        return {"completed_crops": 0, "completed_frames": 0, "sequence": None, "frame": None}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    manifest_dir = resolve(repo, args.manifest_dir)
    data_root = resolve(repo, args.data_root)
    output_dir = resolve(repo, args.out_dir)
    bot_root = resolve(repo, args.bot_sort_root)

    if output_dir.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != EXPECTED_APPEARANCE_MANIFEST_SHA256:
        raise RuntimeError(f"appearance manifest changed: {manifest_sha}")
    manifest_report = json.loads((manifest_dir / "report.json").read_text(encoding="utf-8"))
    if not bool(manifest_report["decision"]["appearance_manifest_ready"]):
        raise RuntimeError("appearance manifest is not ready")
    if bool(manifest_report["protocol"]["ground_truth_read"]):
        raise RuntimeError("appearance manifest unexpectedly used ground truth")

    crop_path = manifest_dir / "unique_candidate_crops.npy"
    all_crops = np.load(crop_path, allow_pickle=False, mmap_mode="r")
    total_available = int(len(all_crops))
    global_start = int(args.start_crop)
    global_end = int(args.end_crop) if int(args.end_crop) > 0 else total_available
    if int(args.limit_crops) > 0:
        global_end = min(global_end, global_start + int(args.limit_crops))
    if global_start < 0 or global_end > total_available or global_start >= global_end:
        raise RuntimeError(
            f"invalid crop range [{global_start}, {global_end}) for {total_available} crops"
        )

    def same_frame(left, right) -> bool:
        return (
            int(left["sequence_id"]) == int(right["sequence_id"])
            and int(left["frame"]) == int(right["frame"])
        )

    if global_start > 0 and same_frame(all_crops[global_start - 1], all_crops[global_start]):
        raise RuntimeError(f"start crop {global_start} splits a frame")
    if global_end < total_available and same_frame(all_crops[global_end - 1], all_crops[global_end]):
        raise RuntimeError(f"end crop {global_end} splits a frame")
    crops = all_crops[global_start:global_end]
    total_crops = int(len(crops))
    if total_crops <= 0:
        raise RuntimeError("no candidate crops selected")

    config = resolve(bot_root, args.fast_reid_config)
    weights = resolve(bot_root, args.fast_reid_weights)
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f"FastReID assets missing: {config}, {weights}")

    if str(bot_root) not in sys.path:
        sys.path.insert(0, str(bot_root))
    from fast_reid.fast_reid_interfece import FastReIDInterface

    print(f"[M23-2 ReID] loading config={config} weights={weights}", flush=True)
    encoder = FastReIDInterface(str(config), str(weights), args.device, batch_size=max(1, int(args.batch_size)))

    embedding_path = output_dir / "candidate_embeddings.f16"
    progress_path = output_dir / "progress.json"
    progress = load_progress(progress_path) if args.resume else {
        "completed_crops": 0,
        "global_crop_start": global_start,
        "global_crop_end": global_end,
        "completed_frames": 0,
        "sequence": None,
        "frame": None,
    }
    completed = int(progress.get("completed_crops", 0))
    if args.resume:
        if int(progress.get("global_crop_start", global_start)) != global_start:
            raise RuntimeError("resume global crop start differs from requested shard")
        if int(progress.get("global_crop_end", global_end)) != global_end:
            raise RuntimeError("resume global crop end differs from requested shard")
    if completed < 0 or completed > total_crops:
        raise RuntimeError(f"invalid resume count {completed} for {total_crops} crops")

    expected_bytes = completed * EMBEDDING_DIM * np.dtype("<f2").itemsize
    if args.resume:
        if not embedding_path.is_file():
            if completed != 0:
                raise RuntimeError("progress exists but embedding cache is missing")
        else:
            actual = embedding_path.stat().st_size
            if actual < expected_bytes:
                raise RuntimeError(f"embedding cache shorter than progress: {actual} < {expected_bytes}")
            if actual != expected_bytes:
                with embedding_path.open("r+b") as handle:
                    handle.truncate(expected_bytes)
    elif embedding_path.exists():
        if not args.overwrite:
            raise FileExistsError(embedding_path)
        embedding_path.unlink()

    frame_rows = []
    norm_min = float("inf")
    norm_max = 0.0
    norm_sum = 0.0
    nonfinite_values = 0
    frames_processed = int(progress.get("completed_frames", 0))

    mode = "ab" if completed else "wb"
    with embedding_path.open(mode) as raw_handle:
        cursor = completed
        while cursor < total_crops:
            sequence_id = int(crops[cursor]["sequence_id"])
            frame = int(crops[cursor]["frame"])
            if sequence_id not in SEQUENCE_BY_ID:
                raise RuntimeError(f"invalid sequence id {sequence_id} at crop {cursor}")
            end = cursor + 1
            while (
                end < total_crops
                and int(crops[end]["sequence_id"]) == sequence_id
                and int(crops[end]["frame"]) == frame
            ):
                end += 1

            sequence = SEQUENCE_BY_ID[sequence_id]
            image_path = data_root / "MOT20" / "train" / sequence / "img1" / f"{frame:06d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"failed to read image {image_path}")
            block = crops[cursor:end]
            boxes = np.column_stack((block["x1"], block["y1"], block["x2"], block["y2"])).astype(np.float32)
            if args.engine == "efficient":
                features = infer_frame_efficient(encoder, image, boxes)
            else:
                features = np.asarray(encoder.inference(image, boxes), dtype=np.float32)
            if features.shape != (len(block), EMBEDDING_DIM):
                raise RuntimeError(
                    f"{sequence} frame {frame}: feature shape {features.shape} != {(len(block), EMBEDDING_DIM)}"
                )
            finite = np.isfinite(features)
            nonfinite_values += int(features.size - int(finite.sum()))
            if not bool(np.all(finite)):
                raise RuntimeError(f"{sequence} frame {frame}: non-finite FastReID output")
            norms = np.linalg.norm(features, axis=1)
            norm_min = min(norm_min, float(norms.min()))
            norm_max = max(norm_max, float(norms.max()))
            norm_sum += float(norms.sum())
            features.astype("<f2", copy=False).tofile(raw_handle)
            raw_handle.flush()

            cursor = end
            frames_processed += 1
            progress = {
                "completed_crops": int(cursor),
                "total_crops": int(total_crops),
                "global_crop_start": int(global_start),
                "global_crop_end": int(global_end),
                "global_completed_crops": int(global_start + cursor),
                "completed_frames": int(frames_processed),
                "sequence": sequence,
                "frame": int(frame),
                "embedding_dim": EMBEDDING_DIM,
                "dtype": "little-endian float16",
            }
            canonical_json_dump(progress, progress_path)
            frame_rows.append({
                "sequence": sequence,
                "frame": frame,
                "crop_start": global_start + cursor - len(block),
                "crop_end": global_start + cursor,
                "crops": len(block),
                "norm_min": float(norms.min()),
                "norm_max": float(norms.max()),
                "norm_mean": float(norms.mean()),
            })
            if frames_processed == 1 or frames_processed % 100 == 0 or cursor == total_crops:
                print(
                    f"[M23-2 ReID] frames={frames_processed} crops={cursor}/{total_crops} "
                    f"global={global_start + cursor}/{global_end} sequence={sequence} frame={frame}",
                    flush=True,
                )

        raw_handle.flush()
        os.fsync(raw_handle.fileno())

    final_size = embedding_path.stat().st_size
    required_size = total_crops * EMBEDDING_DIM * np.dtype("<f2").itemsize
    if final_size != required_size:
        raise RuntimeError(f"embedding byte size {final_size} != expected {required_size}")

    write_csv(
        output_dir / "frame_summary.csv",
        frame_rows,
        ["sequence", "frame", "crop_start", "crop_end", "crops", "norm_min", "norm_max", "norm_mean"],
    )
    report = {
        "schema": "fmtrack.m23_2.candidate_reid_extraction.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "candidate_plan_recomputed": False,
            "candidate_pool_changed": False,
            "appearance_manifest_sha256": manifest_sha,
            "crop_order": "unique_candidate_crops.npy row order",
            "global_crop_range": [global_start, global_end],
            "crop_semantics": "FastReID integer truncation, image-bound clipping, x2/y2-exclusive slicing",
            "embedding_dtype": "little-endian float16",
            "embedding_dim": EMBEDDING_DIM,
            "device": args.device,
            "batch_size": int(args.batch_size),
            "engine": args.engine,
            "limit_crops": int(args.limit_crops),
            "start_crop": global_start,
            "end_crop": global_end,
        },
        "counts": {
            "available_crops": total_available,
            "extracted_crops": total_crops,
            "global_crop_start": global_start,
            "global_crop_end": global_end,
            "frames_processed": frames_processed,
            "nonfinite_values": nonfinite_values,
        },
        "validation": {
            "embedding_size_exact": final_size == required_size,
            "all_features_finite": nonfinite_values == 0,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "norm_mean": norm_sum / max(1, total_crops),
        },
        "assets": {
            "fast_reid_config": {"path": str(Path(args.fast_reid_config)), "sha256": sha256_file(config)},
            "fast_reid_weights": {"path": str(Path(args.fast_reid_weights)), "sha256": sha256_file(weights)},
            "fast_reid_interface": {
                "path": "fast_reid/fast_reid_interfece.py",
                "sha256": sha256_file(bot_root / "fast_reid/fast_reid_interfece.py"),
            },
        },
        "sources": {
            "appearance_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
            "candidate_crops": {"path": str(crop_path), "sha256": sha256_file(crop_path)},
        },
        "files": {
            "candidate_embeddings.f16": {
                "sha256": sha256_file(embedding_path),
                "size_bytes": final_size,
                "dtype": "little-endian float16",
                "shape": [total_crops, EMBEDDING_DIM],
            },
            "frame_summary.csv": {
                "sha256": sha256_file(output_dir / "frame_summary.csv"),
                "rows": len(frame_rows),
            },
            "progress.json": {"sha256": sha256_file(progress_path)},
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
        "schema": "fmtrack.m23_2.candidate_reid_extraction.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {
            name: sha256_file(output_dir / name)
            for name in ("candidate_embeddings.f16", "frame_summary.csv", "progress.json", "report.json")
        },
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps(report["counts"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
