#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


GROUP_KEYS = [
    "group_ids",
    "seq",
    "det_frame",
    "group_size",
    "candidate_count_total",
    "group_is_ambiguous",
    "group_is_background",
    "group_is_recoverable",
    "rank_top1_correct",
    "positive_in_topk",
    "det_feat",
    "det_box",
    "det_score",
]

ROW_KEYS = [
    "track_rank",
    "track_id",
    "label",
    "valid_train_row",
    "base_score",
    "refined_score",
    "motion_score",
    "teacher_score",
    "scalar_feat",
    "hist_feat",
    "hist_mask",
    "hist_time",
    "track_box",
]

EXTRA_KEYS = ["feature_names", "rank_score_col"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize a small runtime-replay subset shard for fast pilot training.")
    ap.add_argument("--src-shard", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out-shard", required=True)
    ap.add_argument("--max-groups", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-first", action="store_true", help="Prefer recoverable / ambiguous / top1-wrong / background groups.")
    ap.add_argument("--compressed", action="store_true", help="Write compressed npz instead of plain npz.")
    return ap.parse_args()


def _load_npz(path: Path) -> dict[str, Any]:
    print(f"[load] {path}", flush=True)
    z = np.load(path, allow_pickle=True)
    data = {key: z[key] for key in z.files}
    print(f"[load-done] keys={len(data)}", flush=True)
    return data


def _select_group_indices(data: dict[str, Any], seq_name: str, max_groups: int, seed: int, hard_first: bool) -> list[int]:
    seqs = [str(x) for x in np.asarray(data["seq"]).tolist()]
    eligible = [idx for idx, seq in enumerate(seqs) if seq == seq_name]
    if not eligible:
        raise RuntimeError(f"No groups for sequence {seq_name}")

    rng = np.random.default_rng(seed)
    choose_count = min(int(max_groups), len(eligible))
    if not hard_first:
        chosen = sorted(rng.choice(np.asarray(eligible, dtype=np.int64), size=choose_count, replace=False).tolist())
        print(f"[select] seq={seq_name} eligible={len(eligible)} chosen={len(chosen)} mode=random", flush=True)
        return chosen

    amb = np.asarray(data["group_is_ambiguous"], dtype=np.uint8)
    rec = np.asarray(data["group_is_recoverable"], dtype=np.uint8)
    bg = np.asarray(data["group_is_background"], dtype=np.uint8)
    top1 = np.asarray(data["rank_top1_correct"], dtype=np.uint8)
    pos_in_topk = np.asarray(data["positive_in_topk"], dtype=np.uint8)

    priority: list[tuple[int, int, float]] = []
    fallback: list[int] = []
    for idx in eligible:
        is_rec = int(rec[idx] > 0)
        is_amb = int(amb[idx] > 0)
        is_wrong = int(top1[idx] <= 0 and pos_in_topk[idx] > 0)
        is_bg = int(bg[idx] > 0)
        score = 100 * is_rec + 10 * is_wrong + 5 * is_amb + 1 * is_bg
        if score > 0:
            priority.append((score, idx, float(rng.random())))
        else:
            fallback.append(idx)

    priority.sort(key=lambda x: (-x[0], x[2], x[1]))
    chosen = [idx for _, idx, _ in priority[:choose_count]]
    if len(chosen) < choose_count and fallback:
        remain = choose_count - len(chosen)
        extra = rng.choice(np.asarray(fallback, dtype=np.int64), size=min(remain, len(fallback)), replace=False).tolist()
        chosen.extend(int(x) for x in extra)
    chosen = sorted(chosen)
    print(
        f"[select] seq={seq_name} eligible={len(eligible)} chosen={len(chosen)} mode=hard_first "
        f"priority_groups={len(priority)}",
        flush=True,
    )
    return chosen


def _build_subset(data: dict[str, Any], seq_name: str, max_groups: int, seed: int, hard_first: bool) -> dict[str, np.ndarray]:
    group_offsets = np.asarray(data["group_offsets"], dtype=np.int64)
    chosen = _select_group_indices(data=data, seq_name=seq_name, max_groups=max_groups, seed=seed, hard_first=hard_first)

    row_indices: list[int] = []
    new_offsets = [0]
    for group_idx in chosen:
        start = int(group_offsets[group_idx])
        end = int(group_offsets[group_idx + 1])
        row_indices.extend(range(start, end))
        new_offsets.append(new_offsets[-1] + (end - start))

    row_indices_np = np.asarray(row_indices, dtype=np.int64)
    chosen_np = np.asarray(chosen, dtype=np.int64)
    payload: dict[str, np.ndarray] = {}
    for key in GROUP_KEYS:
        payload[key] = np.asarray(data[key][chosen_np])
    payload["group_offsets"] = np.asarray(new_offsets, dtype=np.int64)
    for key in ROW_KEYS:
        payload[key] = np.asarray(data[key][row_indices_np])
    for key in EXTRA_KEYS:
        payload[key] = np.asarray(data[key])
    return payload


def _write_npz(path: Path, payload: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"[save] {path} compressed={compressed}", flush=True)
    with tmp_path.open("wb") as f:
        if compressed:
            np.savez_compressed(f, **payload)
        else:
            np.savez(f, **payload)
    tmp_path.replace(path)
    print(f"[save-done] size_mb={path.stat().st_size / 1024.0 / 1024.0:.2f}", flush=True)


def main() -> None:
    args = parse_args()
    src_path = Path(args.src_shard).resolve()
    out_path = Path(args.out_shard).resolve()
    data = _load_npz(src_path)
    payload = _build_subset(
        data=data,
        seq_name=str(args.seq),
        max_groups=int(args.max_groups),
        seed=int(args.seed),
        hard_first=bool(args.hard_first),
    )
    _write_npz(path=out_path, payload=payload, compressed=bool(args.compressed))
    summary = {
        "src_shard": str(src_path),
        "out_shard": str(out_path),
        "seq": str(args.seq),
        "max_groups": int(args.max_groups),
        "seed": int(args.seed),
        "hard_first": bool(args.hard_first),
        "compressed": bool(args.compressed),
        "group_count": int(payload["group_ids"].shape[0]),
        "row_count": int(payload["track_rank"].shape[0]),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
