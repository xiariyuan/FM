#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.runtime_replay_assoc import RUNTIME_REPLAY_FEATURE_NAMES  # noqa: E402
from scripts.train_runtime_rerank_baseline import (  # noqa: E402
    _build_feature,
    _load_rows,
    _predict_scores,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build trainable runtime-replay group shards from labeled CSV + raw tensor shards.")
    ap.add_argument("--labeled-csv", required=True)
    ap.add_argument("--tensor-root", required=True, help="Root containing raw runtime tensor shard npz files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank-score-col", default="refined_score", choices=["base_score", "refined_score"])
    ap.add_argument("--teacher-model", default="", help="Optional baseline model.pkl to write teacher scores.")
    ap.add_argument("--groups-per-shard", type=int, default=1024)
    ap.add_argument("--min-candidates", type=int, default=1)
    return ap.parse_args()


def _load_teacher(path: str):
    if not path:
        return None
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _teacher_score_map(rows: list[dict[str, str]], model, rank_score_col: str) -> dict[tuple[str, int], float]:
    if model is None:
        return {}
    x = np.asarray([_build_feature(row, rank_score_col) for row in rows], dtype=np.float32)
    scores = _predict_scores(model, x)
    out: dict[tuple[str, int], float] = {}
    for row, score in zip(rows, scores.tolist()):
        gid = str(row["group_id"])
        rank = int(float(row["track_rank"]))
        out[(gid, rank)] = float(score)
    return out


def _sorted_tensor_shards(root: Path) -> list[Path]:
    return sorted(root.rglob("*.npz"))


def _load_raw_tensor_shard(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {key: z[key] for key in z.files}


def _append_group(
    buffer: dict[str, list[Any]],
    group_meta: dict[str, Any],
    rows: list[dict[str, str]],
    raw: dict[str, Any],
    raw_indices: list[int],
    teacher_scores: dict[tuple[str, int], float],
    rank_score_col: str,
) -> None:
    gid = str(group_meta["group_id"])
    start_offset = int(buffer["group_offsets"][-1])
    cand_count = len(rows)

    det_feat = np.asarray(group_meta["det_feat"], dtype=np.float16)
    det_box = np.asarray(group_meta["det_box"], dtype=np.float16)
    det_score = float(group_meta["det_score"])

    buffer["group_ids"].append(gid)
    buffer["seq"].append(str(rows[0].get("seq", "")))
    buffer["det_frame"].append(int(float(rows[0].get("frame", 0))))
    buffer["group_size"].append(int(rows[0].get("group_size", cand_count)))
    buffer["candidate_count_total"].append(int(rows[0].get("candidate_count_total", cand_count)))
    buffer["group_is_ambiguous"].append(int(rows[0].get("group_is_ambiguous", 0)))
    buffer["group_is_background"].append(int(rows[0].get("group_is_background", 0)))
    buffer["group_is_recoverable"].append(int(rows[0].get("group_is_recoverable", 0)))
    buffer["rank_top1_correct"].append(int(rows[0].get("rank_top1_correct", 0)))
    buffer["positive_in_topk"].append(int(rows[0].get("positive_in_topk", 0)))
    buffer["det_feat"].append(det_feat)
    buffer["det_box"].append(det_box)
    buffer["det_score"].append(det_score)

    for row, raw_idx in zip(rows, raw_indices):
        rank = int(float(row["track_rank"]))
        buffer["track_rank"].append(rank)
        buffer["track_id"].append(int(float(row["track_id"])))
        buffer["label"].append(int(float(row.get("label", 0))))
        buffer["valid_train_row"].append(int(float(row.get("valid_train_row", 1))))
        buffer["base_score"].append(float(row["base_score"]))
        buffer["refined_score"].append(float(row["refined_score"]))
        buffer["motion_score"].append(float(row["motion_score"]))
        buffer["teacher_score"].append(float(teacher_scores.get((gid, rank), 0.0)))
        buffer["scalar_feat"].append(np.asarray(_build_feature(row, rank_score_col), dtype=np.float32))
        buffer["hist_feat"].append(np.asarray(raw["cand_hist_feat"][raw_idx], dtype=np.float16))
        buffer["hist_mask"].append(np.asarray(raw["cand_hist_mask"][raw_idx], dtype=np.uint8))
        buffer["hist_time"].append(np.asarray(raw["cand_hist_time"][raw_idx], dtype=np.int32))
        buffer["track_box"].append(np.asarray(raw["cand_track_box"][raw_idx], dtype=np.float16))

    buffer["group_offsets"].append(start_offset + cand_count)


def _flush_buffer(buffer: dict[str, list[Any]], out_dir: Path, shard_index: int, rank_score_col: str) -> Path | None:
    if len(buffer["group_ids"]) == 0:
        return None

    payload = {
        "group_ids": np.asarray(buffer["group_ids"], dtype=object),
        "seq": np.asarray(buffer["seq"], dtype=object),
        "det_frame": np.asarray(buffer["det_frame"], dtype=np.int32),
        "group_offsets": np.asarray(buffer["group_offsets"], dtype=np.int64),
        "group_size": np.asarray(buffer["group_size"], dtype=np.int32),
        "candidate_count_total": np.asarray(buffer["candidate_count_total"], dtype=np.int32),
        "group_is_ambiguous": np.asarray(buffer["group_is_ambiguous"], dtype=np.uint8),
        "group_is_background": np.asarray(buffer["group_is_background"], dtype=np.uint8),
        "group_is_recoverable": np.asarray(buffer["group_is_recoverable"], dtype=np.uint8),
        "rank_top1_correct": np.asarray(buffer["rank_top1_correct"], dtype=np.uint8),
        "positive_in_topk": np.asarray(buffer["positive_in_topk"], dtype=np.uint8),
        "det_feat": np.stack(buffer["det_feat"], axis=0).astype(np.float16),
        "det_box": np.stack(buffer["det_box"], axis=0).astype(np.float16),
        "det_score": np.asarray(buffer["det_score"], dtype=np.float32),
        "track_rank": np.asarray(buffer["track_rank"], dtype=np.int16),
        "track_id": np.asarray(buffer["track_id"], dtype=np.int32),
        "label": np.asarray(buffer["label"], dtype=np.int8),
        "valid_train_row": np.asarray(buffer["valid_train_row"], dtype=np.uint8),
        "base_score": np.asarray(buffer["base_score"], dtype=np.float32),
        "refined_score": np.asarray(buffer["refined_score"], dtype=np.float32),
        "motion_score": np.asarray(buffer["motion_score"], dtype=np.float32),
        "teacher_score": np.asarray(buffer["teacher_score"], dtype=np.float32),
        "scalar_feat": np.stack(buffer["scalar_feat"], axis=0).astype(np.float32),
        "hist_feat": np.stack(buffer["hist_feat"], axis=0).astype(np.float16),
        "hist_mask": np.stack(buffer["hist_mask"], axis=0).astype(np.uint8),
        "hist_time": np.stack(buffer["hist_time"], axis=0).astype(np.int32),
        "track_box": np.stack(buffer["track_box"], axis=0).astype(np.float16),
        "feature_names": np.asarray(list(RUNTIME_REPLAY_FEATURE_NAMES), dtype=object),
        "rank_score_col": np.asarray([rank_score_col], dtype=object),
    }
    out_path = out_dir / f"runtime_replay_shard_{shard_index:05d}.npz"
    tmp_path = out_dir / f".runtime_replay_shard_{shard_index:05d}.npz.tmp"
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp_path, out_path)
    return out_path


def _new_buffer() -> dict[str, list[Any]]:
    return {
        "group_ids": [],
        "seq": [],
        "det_frame": [],
        "group_offsets": [0],
        "group_size": [],
        "candidate_count_total": [],
        "group_is_ambiguous": [],
        "group_is_background": [],
        "group_is_recoverable": [],
        "rank_top1_correct": [],
        "positive_in_topk": [],
        "det_feat": [],
        "det_box": [],
        "det_score": [],
        "track_rank": [],
        "track_id": [],
        "label": [],
        "valid_train_row": [],
        "base_score": [],
        "refined_score": [],
        "motion_score": [],
        "teacher_score": [],
        "scalar_feat": [],
        "hist_feat": [],
        "hist_mask": [],
        "hist_time": [],
        "track_box": [],
    }


def main() -> None:
    args = parse_args()
    labeled_csv = Path(args.labeled_csv).resolve()
    tensor_root = Path(args.tensor_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in out_dir.glob("runtime_replay_shard_*.npz"):
        old_path.unlink()

    rows = _load_rows(labeled_csv)
    group_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_rows[str(row["group_id"])].append(row)
    for gid in list(group_rows.keys()):
        group_rows[gid] = sorted(group_rows[gid], key=lambda r: int(float(r["track_rank"])))

    teacher_model = _load_teacher(args.teacher_model)
    teacher_scores = _teacher_score_map(rows, teacher_model, args.rank_score_col)

    shard_paths = _sorted_tensor_shards(tensor_root)
    if not shard_paths:
        raise FileNotFoundError(f"No tensor shard npz files found under {tensor_root}")

    out_paths: list[str] = []
    buffer = _new_buffer()
    shard_index = 0
    matched_groups = 0
    skipped_groups = 0
    matched_rows = 0

    for shard_path in shard_paths:
        raw = _load_raw_tensor_shard(shard_path)
        group_ids = [str(x) for x in raw["group_ids"].tolist()]
        offsets = np.asarray(raw["group_offsets"], dtype=np.int64)
        det_feats = np.asarray(raw["det_feat"])
        det_boxes = np.asarray(raw["det_box"])
        det_scores = np.asarray(raw["det_score"], dtype=np.float32)
        cand_ranks = np.asarray(raw["cand_track_rank"], dtype=np.int32)

        for group_idx, gid in enumerate(group_ids):
            rows_for_group = group_rows.get(gid, [])
            if len(rows_for_group) < int(args.min_candidates):
                skipped_groups += 1
                continue

            start = int(offsets[group_idx])
            end = int(offsets[group_idx + 1])
            raw_rank_map = {int(rank): int(start + local_idx) for local_idx, rank in enumerate(cand_ranks[start:end].tolist())}

            selected_rows: list[dict[str, str]] = []
            raw_indices: list[int] = []
            for row in rows_for_group:
                rank = int(float(row["track_rank"]))
                if rank not in raw_rank_map:
                    continue
                selected_rows.append(row)
                raw_indices.append(raw_rank_map[rank])

            if len(selected_rows) < int(args.min_candidates):
                skipped_groups += 1
                continue

            _append_group(
                buffer=buffer,
                group_meta={
                    "group_id": gid,
                    "det_feat": det_feats[group_idx],
                    "det_box": det_boxes[group_idx],
                    "det_score": det_scores[group_idx],
                },
                rows=selected_rows,
                raw=raw,
                raw_indices=raw_indices,
                teacher_scores=teacher_scores,
                rank_score_col=args.rank_score_col,
            )
            matched_groups += 1
            matched_rows += len(selected_rows)

            if len(buffer["group_ids"]) >= int(args.groups_per_shard):
                out_path = _flush_buffer(buffer, out_dir, shard_index, args.rank_score_col)
                if out_path is not None:
                    out_paths.append(str(out_path))
                    shard_index += 1
                buffer = _new_buffer()

    out_path = _flush_buffer(buffer, out_dir, shard_index, args.rank_score_col)
    if out_path is not None:
        out_paths.append(str(out_path))

    summary = {
        "labeled_csv": str(labeled_csv),
        "tensor_root": str(tensor_root),
        "out_dir": str(out_dir),
        "rank_score_col": str(args.rank_score_col),
        "teacher_model": str(args.teacher_model or ""),
        "groups_per_shard": int(args.groups_per_shard),
        "input_rows": int(len(rows)),
        "input_groups": int(len(group_rows)),
        "matched_groups": int(matched_groups),
        "skipped_groups": int(skipped_groups),
        "matched_rows": int(matched_rows),
        "output_shards": out_paths,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
