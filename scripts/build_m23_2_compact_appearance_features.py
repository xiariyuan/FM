"""Build GT-free compact appearance features for the frozen M23-2 candidates.

Candidate embeddings come from the deterministic merged FastReID cache. Each
candidate is compared with its exact frozen post-NMS suppressor embedding and
with suppression-context track prototypes built only from observable Phase-0
suppressor rows mapped to baseline track IDs. The script has no GT argument and
never runs TrackEval.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import struct
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
EXPECTED_APPEARANCE_MANIFEST_SHA256 = "26c55dd63fb912ef3900869fe14c8687b08220fcca8b307848ece1d5f1092dd7"
EMBEDDING_DIM = 2048
CHUNK = 2048
MAX_TRACK_SAMPLES = 16

KEY_DTYPE = np.dtype([
    ("sequence_id", "u1"),
    ("frame", "<i4"),
    ("suppressed_index", "<i8"),
    ("global_pre_idx", "<i8"),
    ("crop_index", "<i8"),
    ("phase0_suppressor_global_index", "<i8"),
    ("candidate_track_id", "<i4"),
    ("suppressor_track_id", "<i4"),
])

FEATURE_COLUMNS = [
    "candidate_score",
    "suppressor_score",
    "score_diff",
    "score_ratio",
    "suppressor_iou",
    "priority",
    "family_rank",
    "local_rank",
    "candidate_track_iou",
    "suppressor_track_iou",
    "candidate_has_track",
    "suppressor_has_track",
    "same_track",
    "different_tracks",
    "candidate_suppressor_cos",
    "candidate_suppressor_l2",
    "candidate_to_candidate_track_cos",
    "candidate_to_suppressor_track_cos",
    "suppressor_to_candidate_track_cos",
    "suppressor_to_suppressor_track_cos",
    "candidate_track_margin",
    "suppressor_track_margin",
    "candidate_best_track_cos",
    "candidate_second_track_margin",
    "candidate_track_count_log1p",
    "suppressor_track_count_log1p",
    "candidate_track_coherence",
    "suppressor_track_coherence",
    "center_dx_norm",
    "center_dy_norm",
    "center_distance_norm",
    "log_area_ratio",
    "log_aspect_ratio",
    "candidate_log_area",
    "suppressor_log_area",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
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


def read_npz_member(path: Path, member: str, allow_pickle: bool = False) -> np.ndarray:
    import io
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member)
    return np.load(io.BytesIO(data), allow_pickle=allow_pickle)


def mmap_stored_npy_member(path: Path, member: str) -> np.memmap:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError(f"{path}:{member} is compressed and cannot be memory-mapped")
    with path.open("rb") as handle:
        handle.seek(info.header_offset + 26)
        filename_len, extra_len = struct.unpack("<HH", handle.read(4))
        member_offset = info.header_offset + 30 + filename_len + extra_len
        handle.seek(member_offset)
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
        array_offset = handle.tell()
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=array_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def take_rows_sequential(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary mmap rows through contiguous runs, preserving order.

    Fancy indexing a network-backed mmap issues many small page reads even when
    the requested indices are nearly sorted. De-duplicate and coalesce adjacent
    rows first, then restore the original request order deterministically.
    """
    requested = np.asarray(indices, dtype=np.int64)
    if requested.ndim != 1:
        raise ValueError("row indices must be one-dimensional")
    if len(requested) == 0:
        return np.empty((0, array.shape[1]), dtype=np.float32)
    unique, inverse = np.unique(requested, return_inverse=True)
    if unique[0] < 0 or unique[-1] >= len(array):
        raise IndexError("sequential row request is outside the array")
    values = np.empty((len(unique), array.shape[1]), dtype=np.float32)
    run_start = 0
    for run_end in np.r_[np.flatnonzero(np.diff(unique) > 1) + 1, len(unique)]:
        first = int(unique[run_start])
        last = int(unique[run_end - 1]) + 1
        values[run_start:run_end] = np.asarray(array[first:last], dtype=np.float32)
        run_start = int(run_end)
    return values[inverse]


def take_column_sequential(array: np.ndarray, indices: np.ndarray, column: int) -> np.ndarray:
    requested = np.asarray(indices, dtype=np.int64)
    if len(requested) == 0:
        return np.empty((0,), dtype=np.float32)
    unique, inverse = np.unique(requested, return_inverse=True)
    if unique[0] < 0 or unique[-1] >= len(array):
        raise IndexError("sequential column request is outside the array")
    values = np.empty((len(unique),), dtype=np.float32)
    run_start = 0
    for run_end in np.r_[np.flatnonzero(np.diff(unique) > 1) + 1, len(unique)]:
        first = int(unique[run_start])
        last = int(unique[run_end - 1]) + 1
        values[run_start:run_end] = np.asarray(array[first:last, column], dtype=np.float32)
        run_start = int(run_end)
    return values[inverse]


class ShardedEmbeddingReader:
    def __init__(self, repo: Path, shard_root: Path) -> None:
        plan_path = shard_root / "shard_plan.json"
        self.plan_sha256 = sha256_file(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        self.total_crops = int(plan["total_crops"])
        shard_specs = []
        expected_start = 0
        for shard in plan["shards"]:
            start = int(shard["start_crop"])
            end = int(shard["end_crop"])
            if start != expected_start or end <= start:
                raise RuntimeError("candidate ReID shard ranges are not contiguous")
            expected_start = end
            shard_dir = resolve(repo, shard["output_dir"])
            path = shard_dir / "candidate_embeddings.f16"
            rows = end - start
            required = rows * EMBEDDING_DIM * np.dtype("<f2").itemsize
            if path.stat().st_size != required:
                raise RuntimeError(f"candidate ReID shard byte size changed: {path}")
            shard_specs.append((start, end, path))
        if expected_start != self.total_crops:
            raise RuntimeError("candidate ReID shards do not cover all crops")
        # Network-backed random mmap access is prohibitively slow. The complete
        # frozen cache is only 1.88 GB, so stage it once into RAM through twelve
        # sequential reads and perform all later lookup in local memory.
        self.values = np.empty((self.total_crops, EMBEDDING_DIM), dtype="<f2")

        def read_shard(spec: tuple[int, int, Path]) -> tuple[int, int, np.ndarray]:
            start, end, path = spec
            values = np.fromfile(path, dtype="<f2")
            expected = (end - start) * EMBEDDING_DIM
            if values.size != expected:
                raise RuntimeError(
                    f"candidate ReID shard values changed: {path} {values.size} != {expected}"
                )
            return start, end, values.reshape(end - start, EMBEDDING_DIM)

        # Independent files can be read concurrently without changing any
        # numerical operation. Eight readers were already validated during raw
        # extraction and avoid serial NFS throughput collapse.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(read_shard, spec) for spec in shard_specs]
            for future in concurrent.futures.as_completed(futures):
                start, end, values = future.result()
                self.values[start:end] = values
                print(
                    f"[M23-2 compact] preloaded candidate ReID {start}:{end}",
                    flush=True,
                )

    def get(self, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if np.any(requested < 0) or np.any(requested >= self.total_crops):
            raise RuntimeError("candidate crop index is outside sharded ReID cache")
        return np.asarray(self.values[requested], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--appearance-manifest-dir", required=True)
    parser.add_argument("--candidate-reid-cache-dir", required=True)
    parser.add_argument("--candidate-reid-shard-root", required=True)
    parser.add_argument("--phase0-root", default="outputs/alink_train_inputs/phase0_root")
    parser.add_argument("--prenms-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    appearance_dir = resolve(repo, args.appearance_manifest_dir)
    reid_cache_dir = resolve(repo, args.candidate_reid_cache_dir)
    reid_shard_root = resolve(repo, args.candidate_reid_shard_root)
    phase0_root = resolve(repo, args.phase0_root)
    prenms_root = resolve(repo, args.prenms_root)
    output_dir = resolve(repo, args.out_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    appearance_manifest_path = appearance_dir / "manifest.json"
    appearance_sha = sha256_file(appearance_manifest_path)
    if appearance_sha != EXPECTED_APPEARANCE_MANIFEST_SHA256:
        raise RuntimeError(f"appearance manifest changed: {appearance_sha}")
    appearance_report = json.loads((appearance_dir / "report.json").read_text(encoding="utf-8"))
    if bool(appearance_report["protocol"]["ground_truth_read"]):
        raise RuntimeError("appearance manifest used GT")

    reid_report = json.loads((reid_cache_dir / "report.json").read_text(encoding="utf-8"))
    if bool(reid_report["protocol"]["ground_truth_read"]):
        raise RuntimeError("candidate ReID extraction used GT")
    if not bool(reid_report["decision"]["candidate_embeddings_ready"]):
        raise RuntimeError("candidate embeddings are not ready")
    candidate_count = int(appearance_report["counts"]["budget_candidates"])
    crop_count = int(appearance_report["counts"]["unique_candidate_crops_added"])
    if int(reid_report["counts"]["candidate_crops"]) != crop_count:
        raise RuntimeError("candidate ReID crop count differs from appearance manifest")

    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    candidate_manifest = np.load(candidate_manifest_path, allow_pickle=False, mmap_mode="r")
    if len(candidate_manifest) != candidate_count:
        raise RuntimeError("candidate manifest row count changed")
    candidate_embeddings = ShardedEmbeddingReader(repo, reid_shard_root)
    if candidate_embeddings.total_crops != crop_count:
        raise RuntimeError("candidate ReID shard plan differs from crop manifest")

    keys_path = output_dir / "candidate_keys.npy"
    feature_path = output_dir / "appearance_features.f32"
    keys = np.empty((candidate_count,), dtype=KEY_DTYPE)
    feature_memmap = np.memmap(
        feature_path,
        dtype="<f4",
        mode="w+",
        shape=(candidate_count, len(FEATURE_COLUMNS)),
    )

    summaries = []
    source_hashes = {}
    max_abs_feature = 0.0
    nonfinite_values = 0
    cursor = 0

    for sequence_id, sequence in enumerate(SEQUENCES):
        print(f"[M23-2 compact] {sequence}", flush=True)
        rows = np.flatnonzero(candidate_manifest["sequence_id"] == sequence_id)
        if len(rows) == 0:
            raise RuntimeError(f"{sequence}: no candidates")
        block = candidate_manifest[rows]
        if int(rows[0]) != cursor:
            raise RuntimeError(f"{sequence}: candidate rows are not sequence-contiguous")

        phase0_path = phase0_root / sequence / "dump_yolox_reid.npz"
        phase_features_mmap = mmap_stored_npy_member(phase0_path, "features.npy")
        print(
            f"[M23-2 compact] preload Phase-0 features {sequence} "
            f"shape={phase_features_mmap.shape}",
            flush=True,
        )
        phase_features = np.array(phase_features_mmap, copy=True)
        del phase_features_mmap
        if phase_features.shape[1] != EMBEDDING_DIM:
            raise RuntimeError(f"{sequence}: Phase-0 embedding dimension changed")
        supp_indices = block["phase0_suppressor_global_index"].astype(np.int64)
        if np.any(supp_indices < 0) or np.any(supp_indices >= len(phase_features)):
            raise RuntimeError(f"{sequence}: suppressor feature index out of range")

        supp_track_ids = block["suppressor_track_id"].astype(np.int64)
        valid_tracks = np.unique(supp_track_ids[supp_track_ids > 0])
        track_lookup = {int(track_id): index for index, track_id in enumerate(valid_tracks.tolist())}
        sums = np.zeros((len(valid_tracks), EMBEDDING_DIM), dtype=np.float32)
        full_counts = np.zeros((len(valid_tracks),), dtype=np.int64)
        sample_counts = np.zeros((len(valid_tracks),), dtype=np.int64)
        sample_positions = []
        sample_offsets = [0]
        for dense, track_id in enumerate(valid_tracks.tolist()):
            positions = np.flatnonzero(supp_track_ids == int(track_id))
            full_counts[dense] = len(positions)
            count = min(MAX_TRACK_SAMPLES, len(positions))
            take = np.linspace(0, len(positions) - 1, count, dtype=np.int64)
            selected_positions = positions[take]
            sample_positions.extend(selected_positions.tolist())
            sample_counts[dense] = len(selected_positions)
            sample_offsets.append(len(sample_positions))
        sample_positions = np.asarray(sample_positions, dtype=np.int64)
        sampled_phase_indices = supp_indices[sample_positions]
        phase_order = np.argsort(sampled_phase_indices, kind="mergesort")
        ordered_features = normalize_rows(
            np.asarray(phase_features[sampled_phase_indices[phase_order]], dtype=np.float32)
        )
        sample_features = np.empty_like(ordered_features)
        sample_features[phase_order] = ordered_features
        for dense in range(len(valid_tracks)):
            start = sample_offsets[dense]
            end = sample_offsets[dense + 1]
            sums[dense] = sample_features[start:end].sum(axis=0, dtype=np.float32)
        sampled_phase_to_dense = {
            int(phase_index): int(dense)
            for dense in range(len(valid_tracks))
            for phase_index in sampled_phase_indices[sample_offsets[dense]:sample_offsets[dense + 1]].tolist()
        }

        mean_vectors = sums / np.maximum(sample_counts[:, None], 1)
        coherence = np.linalg.norm(mean_vectors, axis=1).astype(np.float32)
        prototypes = normalize_rows(sums)

        suppression_report = json.loads((prenms_root / sequence / "report.json").read_text(encoding="utf-8"))
        columns = json.loads((prenms_root / sequence / "columns.json").read_text(encoding="utf-8"))
        column_index = {name: index for index, name in enumerate(columns)}
        raw_shape = tuple(int(value) for value in suppression_report["files"]["suppressed_candidates"]["shape"])
        raw_path = prenms_root / sequence / "suppressed_candidates.f32"
        raw_mmap = np.memmap(raw_path, dtype="<f4", mode="r", shape=raw_shape)
        print(
            f"[M23-2 compact] preload suppression rows {sequence} shape={raw_shape}",
            flush=True,
        )
        raw = np.array(raw_mmap, copy=True)
        del raw_mmap
        raw_indices = block["suppressed_index"].astype(np.int64)
        global_pre_idx = raw[raw_indices, column_index["global_pre_idx"]].astype(np.int64)

        for name in (
            "sequence_id", "frame", "suppressed_index", "crop_index",
            "phase0_suppressor_global_index", "candidate_track_id", "suppressor_track_id",
        ):
            keys[name][rows] = block[name]
        keys["global_pre_idx"][rows] = global_pre_idx

        for local_start in range(0, len(block), CHUNK):
            local_end = min(len(block), local_start + CHUNK)
            b = block[local_start:local_end]
            n = len(b)
            cand = normalize_rows(candidate_embeddings.get(b["crop_index"].astype(np.int64)))
            supp = normalize_rows(
                np.asarray(
                    phase_features[b["phase0_suppressor_global_index"].astype(np.int64)],
                    dtype=np.float32,
                )
            )

            candidate_ids = b["candidate_track_id"].astype(np.int64)
            suppressor_ids = b["suppressor_track_id"].astype(np.int64)
            candidate_dense = np.asarray([track_lookup.get(int(value), -1) for value in candidate_ids], dtype=np.int64)
            suppressor_dense = np.asarray([track_lookup.get(int(value), -1) for value in suppressor_ids], dtype=np.int64)
            candidate_proto = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
            suppressor_proto = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
            candidate_proto_count = np.zeros((n,), dtype=np.float32)
            suppressor_proto_count = np.zeros((n,), dtype=np.float32)
            candidate_coherence = np.zeros((n,), dtype=np.float32)
            suppressor_coherence = np.zeros((n,), dtype=np.float32)
            valid_candidate = candidate_dense >= 0
            valid_suppressor = suppressor_dense >= 0
            if np.any(valid_candidate):
                candidate_proto[valid_candidate] = prototypes[candidate_dense[valid_candidate]]
                candidate_proto_count[valid_candidate] = full_counts[candidate_dense[valid_candidate]]
                candidate_coherence[valid_candidate] = coherence[candidate_dense[valid_candidate]]
            if np.any(valid_suppressor):
                suppressor_proto[valid_suppressor] = prototypes[suppressor_dense[valid_suppressor]]
                suppressor_proto_count[valid_suppressor] = full_counts[suppressor_dense[valid_suppressor]]
                suppressor_coherence[valid_suppressor] = coherence[suppressor_dense[valid_suppressor]]

            # Leave the current suppressor row out of its own track prototype.
            sampled_dense = np.asarray(
                [sampled_phase_to_dense.get(int(value), -1) for value in b["phase0_suppressor_global_index"]],
                dtype=np.int64,
            )
            loo = (
                valid_suppressor
                & (sampled_dense == suppressor_dense)
                & (sample_counts[np.maximum(suppressor_dense, 0)] > 1)
            )
            if np.any(loo):
                dense = suppressor_dense[loo]
                loo_sums = sums[dense] - supp[loo]
                suppressor_proto[loo] = normalize_rows(loo_sums)
                same_candidate = loo & (candidate_ids == suppressor_ids)
                if np.any(same_candidate):
                    candidate_proto[same_candidate] = suppressor_proto[same_candidate]

            candidate_suppressor_cos = np.sum(cand * supp, axis=1)
            candidate_to_candidate = np.sum(cand * candidate_proto, axis=1)
            candidate_to_suppressor = np.sum(cand * suppressor_proto, axis=1)
            suppressor_to_candidate = np.sum(supp * candidate_proto, axis=1)
            suppressor_to_suppressor = np.sum(supp * suppressor_proto, axis=1)
            candidate_has = candidate_ids > 0
            suppressor_has = suppressor_ids > 0
            same_track = candidate_has & suppressor_has & (candidate_ids == suppressor_ids)
            different_tracks = candidate_has & suppressor_has & (candidate_ids != suppressor_ids)

            cx_candidate = (b["candidate_x1"] + b["candidate_x2"]) * 0.5
            cy_candidate = (b["candidate_y1"] + b["candidate_y2"]) * 0.5
            cx_suppressor = (b["suppressor_x1"] + b["suppressor_x2"]) * 0.5
            cy_suppressor = (b["suppressor_y1"] + b["suppressor_y2"]) * 0.5
            cw = np.maximum(b["candidate_x2"] - b["candidate_x1"], 1e-3)
            ch = np.maximum(b["candidate_y2"] - b["candidate_y1"], 1e-3)
            sw = np.maximum(b["suppressor_x2"] - b["suppressor_x1"], 1e-3)
            sh = np.maximum(b["suppressor_y2"] - b["suppressor_y1"], 1e-3)
            scale = np.sqrt(np.maximum(sw * sh, 1e-6))
            dx = (cx_candidate - cx_suppressor) / scale
            dy = (cy_candidate - cy_suppressor) / scale
            candidate_area = np.maximum(cw * ch, 1e-6)
            suppressor_area = np.maximum(sw * sh, 1e-6)

            values = np.column_stack([
                b["candidate_score"],
                b["suppressor_score"],
                b["candidate_score"] - b["suppressor_score"],
                b["candidate_score"] / np.maximum(b["suppressor_score"], 1e-6),
                b["suppressor_iou"],
                b["priority"],
                b["family_rank"],
                b["local_rank"],
                b["candidate_track_iou"],
                b["suppressor_track_iou"],
                candidate_has.astype(np.float32),
                suppressor_has.astype(np.float32),
                same_track.astype(np.float32),
                different_tracks.astype(np.float32),
                candidate_suppressor_cos,
                np.sqrt(np.maximum(2.0 - 2.0 * candidate_suppressor_cos, 0.0)),
                candidate_to_candidate,
                candidate_to_suppressor,
                suppressor_to_candidate,
                suppressor_to_suppressor,
                candidate_to_candidate - candidate_to_suppressor,
                suppressor_to_suppressor - suppressor_to_candidate,
                np.maximum(candidate_to_candidate, candidate_to_suppressor),
                np.abs(candidate_to_candidate - candidate_to_suppressor),
                np.log1p(candidate_proto_count),
                np.log1p(suppressor_proto_count),
                candidate_coherence,
                suppressor_coherence,
                dx,
                dy,
                np.sqrt(dx * dx + dy * dy),
                np.log(candidate_area / suppressor_area),
                np.log((cw / ch) / (sw / sh)),
                np.log(candidate_area),
                np.log(suppressor_area),
            ]).astype(np.float32)
            if values.shape[1] != len(FEATURE_COLUMNS):
                raise RuntimeError("feature column count changed")
            nonfinite_values += int(values.size - np.isfinite(values).sum())
            if not np.isfinite(values).all():
                raise RuntimeError(f"{sequence}: non-finite compact appearance feature")
            max_abs_feature = max(max_abs_feature, float(np.max(np.abs(values))))
            feature_memmap[rows[local_start:local_end], :] = values

        summaries.append({
            "sequence": sequence,
            "candidates": int(len(block)),
            "unique_suppressor_tracks": int(len(valid_tracks)),
            "candidate_has_track": int(np.sum(block["candidate_track_id"] > 0)),
            "suppressor_has_track": int(np.sum(block["suppressor_track_id"] > 0)),
            "same_track": int(np.sum((block["candidate_track_id"] > 0) & (block["candidate_track_id"] == block["suppressor_track_id"]))),
            "different_tracks": int(np.sum((block["candidate_track_id"] > 0) & (block["suppressor_track_id"] > 0) & (block["candidate_track_id"] != block["suppressor_track_id"]))),
            "mean_track_coherence": float(np.mean(coherence)) if len(coherence) else 0.0,
            "minimum_track_count": int(full_counts.min()) if len(full_counts) else 0,
            "maximum_track_count": int(full_counts.max()) if len(full_counts) else 0,
            "prototype_samples": int(sample_counts.sum()),
            "maximum_prototype_samples_per_track": int(sample_counts.max()) if len(sample_counts) else 0,
        })
        phase0_manifest_path = phase0_path.resolve().parent / "manifest.json"
        with zipfile.ZipFile(phase0_path) as archive:
            feature_info = archive.getinfo("features.npy")
        source_hashes[sequence] = {
            "phase0_manifest_sha256": sha256_file(phase0_manifest_path),
            "phase0_features_member": {
                "crc32": f"{feature_info.CRC:08x}",
                "file_size": int(feature_info.file_size),
                "compress_type": int(feature_info.compress_type),
                "shape": [int(value) for value in phase_features.shape],
                "dtype": str(phase_features.dtype),
            },
            "suppression_data_sha256": suppression_report["files"]["suppressed_candidates"]["sha256"],
            "suppression_manifest_sha256": sha256_file(prenms_root / sequence / "manifest.json"),
        }
        cursor += len(block)
        del raw, phase_features

    if cursor != candidate_count:
        raise RuntimeError("compact feature cursor differs from candidate count")
    feature_memmap.flush()
    del feature_memmap
    np.save(keys_path, keys, allow_pickle=False)
    canonical_json_dump(FEATURE_COLUMNS, output_dir / "feature_columns.json")

    combined = {
        "sequence": "COMBINED",
        "candidates": candidate_count,
        "unique_suppressor_tracks": int(sum(row["unique_suppressor_tracks"] for row in summaries)),
        "candidate_has_track": int(sum(row["candidate_has_track"] for row in summaries)),
        "suppressor_has_track": int(sum(row["suppressor_has_track"] for row in summaries)),
        "same_track": int(sum(row["same_track"] for row in summaries)),
        "different_tracks": int(sum(row["different_tracks"] for row in summaries)),
        "mean_track_coherence": float(np.mean([row["mean_track_coherence"] for row in summaries])),
        "minimum_track_count": int(min(row["minimum_track_count"] for row in summaries)),
        "maximum_track_count": int(max(row["maximum_track_count"] for row in summaries)),
        "prototype_samples": int(sum(row["prototype_samples"] for row in summaries)),
        "maximum_prototype_samples_per_track": int(max(row["maximum_prototype_samples_per_track"] for row in summaries)),
    }
    summary_rows = summaries + [combined]
    write_csv(
        output_dir / "sequence_summary.csv",
        summary_rows,
        [
            "sequence", "candidates", "unique_suppressor_tracks", "candidate_has_track",
            "suppressor_has_track", "same_track", "different_tracks", "mean_track_coherence",
            "minimum_track_count", "maximum_track_count",
            "prototype_samples", "maximum_prototype_samples_per_track",
        ],
    )

    feature_size = feature_path.stat().st_size
    required_feature_size = candidate_count * len(FEATURE_COLUMNS) * np.dtype("<f4").itemsize
    if feature_size != required_feature_size:
        raise RuntimeError("compact feature byte size changed")
    report = {
        "schema": "fmtrack.m23_2.compact_appearance_features.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "candidate_pool_changed": False,
            "candidate_plan_recomputed": False,
            "offline_observable_track_prototypes": True,
            "track_prototype_source": "up to 16 temporally ordered frozen Phase-0 suppressor embeddings per observable baseline suppressor_track_id",
            "maximum_track_prototype_samples": MAX_TRACK_SAMPLES,
            "track_prototype_sampling": "uniform index positions in observable suppressor-row order",
            "current_suppressor_leave_one_out_when_sampled": True,
            "appearance_manifest_sha256": appearance_sha,
            "candidate_reid_report_sha256": sha256_file(reid_cache_dir / "report.json"),
            "candidate_reid_shard_plan_sha256": candidate_embeddings.plan_sha256,
        },
        "counts": {
            "candidates": candidate_count,
            "features": len(FEATURE_COLUMNS),
            "nonfinite_values": nonfinite_values,
        },
        "validation": {
            "all_features_finite": nonfinite_values == 0,
            "feature_size_exact": feature_size == required_feature_size,
            "max_abs_feature": max_abs_feature,
            "key_rows_equal_candidates": len(keys) == candidate_count,
        },
        "sequence_summary": summaries,
        "sources": source_hashes,
        "files": {
            "candidate_keys.npy": {"sha256": sha256_file(keys_path), "dtype": KEY_DTYPE.descr, "shape": [candidate_count], "size_bytes": keys_path.stat().st_size},
            "appearance_features.f32": {"sha256": sha256_file(feature_path), "dtype": "little-endian float32", "shape": [candidate_count, len(FEATURE_COLUMNS)], "size_bytes": feature_size},
            "feature_columns.json": {"sha256": sha256_file(output_dir / "feature_columns.json")},
            "sequence_summary.csv": {"sha256": sha256_file(output_dir / "sequence_summary.csv"), "rows": len(summary_rows)},
        },
        "decision": {
            "compact_appearance_features_ready": True,
            "sequence_loso_audit_ready": True,
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
        "schema": "fmtrack.m23_2.compact_appearance_features.manifest.v1",
        "report_sha256": sha256_file(output_dir / "report.json"),
        "file_hashes": {
            name: sha256_file(output_dir / name)
            for name in (
                "candidate_keys.npy", "appearance_features.f32", "feature_columns.json",
                "sequence_summary.csv", "report.json",
            )
        },
    }
    canonical_json_dump(manifest, output_dir / "manifest.json")
    print(json.dumps({
        "candidates": candidate_count,
        "features": len(FEATURE_COLUMNS),
        "candidate_has_track": combined["candidate_has_track"],
        "suppressor_has_track": combined["suppressor_has_track"],
        "different_tracks": combined["different_tracks"],
        "nonfinite_values": nonfinite_values,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
