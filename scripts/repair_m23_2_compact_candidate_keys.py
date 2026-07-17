"""Repair and verify the M23-2 compact candidate key table without recomputing features.

The original compact builder used integer fancy indexing and assigned through the
resulting copy. This script reconstructs every key field from the frozen GT-free
candidate manifest and the frozen pre-NMS suppression dumps, validates uniqueness,
and atomically refreshes report/manifest hashes. It never reads GT and never runs
TrackEval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
EXPECTED_APPEARANCE_MANIFEST_SHA256 = "26c55dd63fb912ef3900869fe14c8687b08220fcca8b307848ece1d5f1092dd7"
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
COPIED_FIELDS = (
    "sequence_id",
    "frame",
    "suppressed_index",
    "crop_index",
    "phase0_suppressor_global_index",
    "candidate_track_id",
    "suppressor_track_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--appearance-manifest-dir", required=True)
    parser.add_argument("--prenms-root", required=True)
    parser.add_argument("--compact-dir", required=True)
    return parser.parse_args()


def resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_dump(value: object, path: Path) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def unique_pair_count(sequence_ids: np.ndarray, global_ids: np.ndarray) -> int:
    pair_dtype = np.dtype([("sequence_id", "u1"), ("global_pre_idx", "<i8")])
    pairs = np.empty((len(sequence_ids),), dtype=pair_dtype)
    pairs["sequence_id"] = sequence_ids
    pairs["global_pre_idx"] = global_ids
    return int(len(np.unique(pairs)))


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    appearance_dir = resolve(repo, args.appearance_manifest_dir)
    prenms_root = resolve(repo, args.prenms_root)
    compact_dir = resolve(repo, args.compact_dir)

    appearance_manifest_path = appearance_dir / "manifest.json"
    appearance_manifest_sha = sha256_file(appearance_manifest_path)
    if appearance_manifest_sha != EXPECTED_APPEARANCE_MANIFEST_SHA256:
        raise RuntimeError(f"appearance manifest changed: {appearance_manifest_sha}")

    appearance_report = json.loads((appearance_dir / "report.json").read_text(encoding="utf-8"))
    candidate_manifest_path = appearance_dir / "candidate_manifest.npy"
    expected_candidate_sha = appearance_report["files"]["candidate_manifest.npy"]["sha256"]
    actual_candidate_sha = sha256_file(candidate_manifest_path)
    if actual_candidate_sha != expected_candidate_sha:
        raise RuntimeError("candidate manifest hash changed")
    candidate_manifest = np.load(candidate_manifest_path, allow_pickle=False, mmap_mode="r")
    candidate_count = int(appearance_report["counts"]["budget_candidates"])
    if len(candidate_manifest) != candidate_count:
        raise RuntimeError("candidate manifest row count changed")

    compact_report_path = compact_dir / "report.json"
    compact_manifest_path = compact_dir / "manifest.json"
    compact_report = json.loads(compact_report_path.read_text(encoding="utf-8"))
    feature_path = compact_dir / "appearance_features.f32"
    feature_sha_before = sha256_file(feature_path)
    if feature_sha_before != compact_report["files"]["appearance_features.f32"]["sha256"]:
        raise RuntimeError("appearance feature file differs from compact report")

    keys = np.empty((candidate_count,), dtype=KEY_DTYPE)
    per_sequence = []
    cursor = 0
    source_hashes = {}

    for sequence_id, sequence in enumerate(SEQUENCES):
        rows = np.flatnonzero(candidate_manifest["sequence_id"] == sequence_id)
        if len(rows) == 0:
            raise RuntimeError(f"{sequence}: no candidate rows")
        if int(rows[0]) != cursor or int(rows[-1]) + 1 != cursor + len(rows):
            raise RuntimeError(f"{sequence}: candidate rows are not contiguous")
        block = candidate_manifest[rows]
        for field in COPIED_FIELDS:
            keys[field][rows] = block[field]

        report_path = prenms_root / sequence / "report.json"
        columns_path = prenms_root / sequence / "columns.json"
        raw_path = prenms_root / sequence / "suppressed_candidates.f32"
        suppression_report = json.loads(report_path.read_text(encoding="utf-8"))
        columns = json.loads(columns_path.read_text(encoding="utf-8"))
        column_index = {name: index for index, name in enumerate(columns)}
        if "global_pre_idx" not in column_index:
            raise RuntimeError(f"{sequence}: global_pre_idx column missing")
        raw_shape = tuple(int(value) for value in suppression_report["files"]["suppressed_candidates"]["shape"])
        raw = np.memmap(raw_path, dtype="<f4", mode="r", shape=raw_shape)
        # Read the complete strided column sequentially, then index the frozen sorted subset.
        global_column = np.array(raw[:, column_index["global_pre_idx"]], dtype=np.float32, copy=True)
        del raw
        if not np.isfinite(global_column).all():
            raise RuntimeError(f"{sequence}: non-finite global_pre_idx values")
        rounded = np.rint(global_column).astype(np.int64)
        if not np.array_equal(global_column, rounded.astype(np.float32)):
            raise RuntimeError(f"{sequence}: non-integer global_pre_idx values")
        suppressed_indices = block["suppressed_index"].astype(np.int64)
        if np.any(suppressed_indices < 0) or np.any(suppressed_indices >= len(rounded)):
            raise RuntimeError(f"{sequence}: suppressed_index outside frozen dump")
        selected_global = rounded[suppressed_indices]
        keys["global_pre_idx"][rows] = selected_global

        unique_local = int(len(np.unique(selected_global)))
        if unique_local != len(rows):
            raise RuntimeError(f"{sequence}: duplicate global_pre_idx in frozen budget")
        if not np.all(selected_global[1:] > selected_global[:-1]):
            raise RuntimeError(f"{sequence}: global_pre_idx is not strictly increasing")

        per_sequence.append({
            "sequence": sequence,
            "rows": int(len(rows)),
            "global_pre_idx_min": int(selected_global.min()),
            "global_pre_idx_max": int(selected_global.max()),
            "unique_global_pre_idx": unique_local,
            "frame_min": int(block["frame"].min()),
            "frame_max": int(block["frame"].max()),
        })
        source_hashes[sequence] = {
            "suppression_report_sha256": sha256_file(report_path),
            "suppression_columns_sha256": sha256_file(columns_path),
            "suppression_data_sha256": suppression_report["files"]["suppressed_candidates"]["sha256"],
        }
        cursor += len(rows)
        print(f"[M23-2 key repair] {sequence}: {len(rows)} rows", flush=True)

    if cursor != candidate_count:
        raise RuntimeError("not all candidate rows were rebuilt")

    field_matches = {
        field: bool(np.array_equal(keys[field], candidate_manifest[field]))
        for field in COPIED_FIELDS
    }
    if not all(field_matches.values()):
        raise RuntimeError(f"rebuilt copied fields differ: {field_matches}")
    unique_keys = unique_pair_count(keys["sequence_id"], keys["global_pre_idx"])
    duplicate_keys = candidate_count - unique_keys
    if duplicate_keys != 0:
        raise RuntimeError(f"candidate key duplicates remain: {duplicate_keys}")

    key_path = compact_dir / "candidate_keys.npy"
    atomic_save_npy(key_path, keys)
    key_sha = sha256_file(key_path)
    feature_sha_after = sha256_file(feature_path)
    if feature_sha_after != feature_sha_before:
        raise RuntimeError("feature matrix changed during key repair")

    repair_report = {
        "schema": "fmtrack.m23_2.compact_candidate_key_repair.v1",
        "protocol": {
            "ground_truth_read": False,
            "trackeval_calls": 0,
            "feature_matrix_recomputed": False,
            "candidate_pool_changed": False,
            "repair_reason": "integer fancy indexing returned a copy in the original compact builder",
            "repair_script_sha256": sha256_file(Path(__file__).resolve()),
            "appearance_manifest_sha256": appearance_manifest_sha,
            "candidate_manifest_sha256": actual_candidate_sha,
        },
        "counts": {
            "candidate_rows": candidate_count,
            "unique_candidate_keys": unique_keys,
            "duplicate_candidate_keys": duplicate_keys,
        },
        "validation": {
            "all_copied_fields_match_candidate_manifest": all(field_matches.values()),
            "field_matches": field_matches,
            "global_pre_idx_unique_within_sequence": True,
            "global_pre_idx_strictly_increasing_within_sequence": True,
            "feature_sha_unchanged": feature_sha_after == feature_sha_before,
        },
        "per_sequence": per_sequence,
        "sources": source_hashes,
        "files": {
            "candidate_keys.npy": {
                "sha256": key_sha,
                "size_bytes": key_path.stat().st_size,
                "shape": [candidate_count],
                "dtype": KEY_DTYPE.descr,
            },
            "appearance_features.f32": {
                "sha256": feature_sha_after,
                "unchanged": True,
            },
        },
        "decision": {
            "candidate_keys_ready": True,
            "sequence_loso_preflight_ready": True,
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
    repair_report_path = compact_dir / "key_rebuild_report.json"
    canonical_json_dump(repair_report, repair_report_path)

    compact_report["files"]["candidate_keys.npy"] = repair_report["files"]["candidate_keys.npy"]
    compact_report["files"]["key_rebuild_report.json"] = {
        "sha256": sha256_file(repair_report_path),
        "size_bytes": repair_report_path.stat().st_size,
    }
    compact_report.setdefault("protocol", {})["candidate_key_rebuilt"] = True
    compact_report["protocol"]["candidate_key_repair_script_sha256"] = repair_report["protocol"]["repair_script_sha256"]
    compact_report.setdefault("validation", {}).update({
        "candidate_key_rows": candidate_count,
        "unique_candidate_keys": unique_keys,
        "duplicate_candidate_keys": duplicate_keys,
        "all_key_fields_match_frozen_sources": True,
    })
    compact_report.setdefault("decision", {})["sequence_loso_audit_ready"] = True
    canonical_json_dump(compact_report, compact_report_path)

    manifest = {
        "schema": "fmtrack.m23_2.compact_appearance_features.manifest.v1",
        "report_sha256": sha256_file(compact_report_path),
        "file_hashes": {
            name: sha256_file(compact_dir / name)
            for name in (
                "appearance_features.f32",
                "candidate_keys.npy",
                "feature_columns.json",
                "sequence_summary.csv",
                "key_rebuild_report.json",
                "report.json",
            )
        },
    }
    canonical_json_dump(manifest, compact_manifest_path)

    print(json.dumps({
        "candidate_rows": candidate_count,
        "unique_candidate_keys": unique_keys,
        "duplicate_candidate_keys": duplicate_keys,
        "candidate_keys_sha256": key_sha,
        "feature_sha256_unchanged": feature_sha_after,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
