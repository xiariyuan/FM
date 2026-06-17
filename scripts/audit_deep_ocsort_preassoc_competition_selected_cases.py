#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit selected Deep-OC-SORT pre-association stale-competition cases and summarize their track-window structure."
    )
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--raw-track-dir", required=True)
    parser.add_argument("--competition-track-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=int, default=8)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_selected_cases(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("selected_final", 0) or 0) <= 0:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No selected_final>0 rows found in {path}")
    return rows


def load_track_presence(track_dir: Path) -> Dict[str, Dict[int, Set[int]]]:
    presence: Dict[str, Dict[int, Set[int]]] = {}
    for txt_path in sorted(track_dir.glob("*.txt")):
        seq_frames: Dict[int, Set[int]] = defaultdict(set)
        with txt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    frame_id = int(float(parts[0]))
                    track_id = int(float(parts[1]))
                except ValueError:
                    continue
                seq_frames[frame_id].add(track_id)
        presence[txt_path.stem] = seq_frames
    if not presence:
        raise FileNotFoundError(f"No MOT text files found in {track_dir}")
    return presence


def count_frames_with_track(frame_map: Dict[int, Set[int]], track_id: int, frame_ids: Iterable[int]) -> int:
    if track_id <= 0:
        return 0
    total = 0
    for frame_id in frame_ids:
        if track_id in frame_map.get(frame_id, set()):
            total += 1
    return total


def has_track(frame_map: Dict[int, Set[int]], track_id: int, frame_id: int) -> int:
    if track_id <= 0:
        return 0
    return int(track_id in frame_map.get(frame_id, set()))


def output_track_id(internal_track_id: int) -> int:
    if internal_track_id < 0:
        return -1
    return int(internal_track_id) + 1


def gt_available(case: Dict[str, Any]) -> int:
    gt_fields = [
        "det_gt_id",
        "track_gt_id",
        "raw_owner_track_gt_id",
        "final_det_gt_id_for_track",
        "raw_det_gt_id_for_track",
        "raw_owner_det_gt_id",
    ]
    for field in gt_fields:
        try:
            if int(case.get(field, -1) or -1) > 0:
                return 1
        except Exception:
            continue
    return 0


def seq_frame_maps(presence: Dict[str, Dict[int, Set[int]]], seq_name: str) -> Dict[int, Set[int]]:
    if seq_name not in presence:
        raise KeyError(f"Missing track text for sequence {seq_name}")
    return presence[seq_name]


def build_case_rows(
    selected_cases: List[Dict[str, Any]],
    raw_presence: Dict[str, Dict[int, Set[int]]],
    competition_presence: Dict[str, Dict[int, Set[int]]],
    window: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in selected_cases:
        seq_name = str(case.get("seq_name", ""))
        frame_id = int(case.get("frame_id", 0) or 0)
        reclaim_track_internal_id = int(case.get("track_internal_id", -1) or -1)
        owner_track_internal_id = int(case.get("raw_owner_internal_id", -1) or -1)
        reclaim_track_id = int(case.get("track_output_id", output_track_id(reclaim_track_internal_id)) or -1)
        owner_track_id = int(case.get("raw_owner_output_id", output_track_id(owner_track_internal_id)) or -1)
        raw_frames = seq_frame_maps(raw_presence, seq_name)
        competition_frames = seq_frame_maps(competition_presence, seq_name)
        prev_range = range(frame_id - int(window), frame_id)
        next_range = range(frame_id + 1, frame_id + int(window) + 1)
        row: Dict[str, object] = {
            "seq_name": seq_name,
            "frame_id": frame_id,
            "det_idx": int(case.get("det_idx", -1) or -1),
            "det_gt_id": int(case.get("det_gt_id", -1) or -1),
            "reclaim_track_internal_id": reclaim_track_internal_id,
            "reclaim_track_id": reclaim_track_id,
            "reclaim_track_gt_id": int(case.get("track_gt_id", -1) or -1),
            "raw_owner_track_internal_id": owner_track_internal_id,
            "raw_owner_track_id": owner_track_id,
            "raw_owner_track_gt_id": int(case.get("raw_owner_track_gt_id", -1) or -1),
            "selected_final": int(case.get("selected_final", 0) or 0),
            "selected_forced": int(case.get("selected_forced", 0) or 0),
            "selected_track_emitted_before_filter": int(case.get("selected_track_emitted_before_filter", 0) or 0),
            "raw_owner_emitted_before_filter": int(case.get("raw_owner_emitted_before_filter", 0) or 0),
            "det_rank": int(case.get("det_rank", -1) or -1),
            "track_hits": int(case.get("track_hits", 0) or 0),
            "owner_hits": int(case.get("owner_hits", 0) or 0),
            "track_age": int(case.get("track_age", 0) or 0),
            "owner_age": int(case.get("owner_age", 0) or 0),
            "age_gap": int(case.get("age_gap", 0) or 0),
            "best_box_iou": float(case.get("best_box_iou", 0.0) or 0.0),
            "edge_score": float(case.get("edge_score", 0.0) or 0.0),
            "owner_edge_score": float(case.get("owner_edge_score", 0.0) or 0.0),
            "edge_advantage_vs_owner": float(case.get("edge_advantage_vs_owner", 0.0) or 0.0),
            "gt_available": gt_available(case),
            "raw_prev_reclaim_frames": count_frames_with_track(raw_frames, reclaim_track_id, prev_range),
            "raw_prev_owner_frames": count_frames_with_track(raw_frames, owner_track_id, prev_range),
            "raw_curr_reclaim": has_track(raw_frames, reclaim_track_id, frame_id),
            "raw_curr_owner": has_track(raw_frames, owner_track_id, frame_id),
            "raw_next_reclaim_frames": count_frames_with_track(raw_frames, reclaim_track_id, next_range),
            "raw_next_owner_frames": count_frames_with_track(raw_frames, owner_track_id, next_range),
            "competition_prev_reclaim_frames": count_frames_with_track(competition_frames, reclaim_track_id, prev_range),
            "competition_prev_owner_frames": count_frames_with_track(competition_frames, owner_track_id, prev_range),
            "competition_curr_reclaim": has_track(competition_frames, reclaim_track_id, frame_id),
            "competition_curr_owner": has_track(competition_frames, owner_track_id, frame_id),
            "competition_next_reclaim_frames": count_frames_with_track(competition_frames, reclaim_track_id, next_range),
            "competition_next_owner_frames": count_frames_with_track(competition_frames, owner_track_id, next_range),
        }
        row["reclaim_continuity_shape"] = int(
            int(row["competition_prev_reclaim_frames"]) > 0
            and int(row["competition_curr_reclaim"]) > 0
            and int(row["competition_next_reclaim_frames"]) > 0
        )
        row["weak_owner_shape"] = int(int(row["owner_hits"]) <= 8)
        row["owner_not_stronger_by_edge"] = int(float(row["edge_advantage_vs_owner"]) >= -0.2)
        row["supports_main_claim_shape"] = int(
            int(row["reclaim_continuity_shape"]) > 0
            and int(row["weak_owner_shape"]) > 0
            and int(row["owner_not_stronger_by_edge"]) > 0
        )
        rows.append(row)
    return rows


def build_sequence_rows(case_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_seq: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_seq[str(row["seq_name"])].append(row)

    rows: List[Dict[str, object]] = []
    for seq_name, seq_rows in sorted(by_seq.items()):
        rows.append(
            {
                "seq_name": seq_name,
                "selected_cases": len(seq_rows),
                "gt_available_cases": sum(int(r["gt_available"]) for r in seq_rows),
                "rank2_cases": sum(int(int(r["det_rank"]) == 2) for r in seq_rows),
                "weak_owner_cases": sum(int(r["weak_owner_shape"]) for r in seq_rows),
                "within_owner_edge_deficit_cases": sum(int(r["owner_not_stronger_by_edge"]) for r in seq_rows),
                "selected_track_emitted_before_filter_cases": sum(int(r["selected_track_emitted_before_filter"]) for r in seq_rows),
                "negative_edge_adv_cases": sum(int(float(r["edge_advantage_vs_owner"]) < 0.0) for r in seq_rows),
                "continuity_shape_cases": sum(int(r["reclaim_continuity_shape"]) for r in seq_rows),
                "supports_main_claim_shape_cases": sum(int(r["supports_main_claim_shape"]) for r in seq_rows),
                "min_edge_advantage_vs_owner": min(float(r["edge_advantage_vs_owner"]) for r in seq_rows),
                "max_owner_hits": max(int(r["owner_hits"]) for r in seq_rows),
                "max_age_gap": max(int(r["age_gap"]) for r in seq_rows),
            }
        )
    return rows


def aggregate_row(case_rows: List[Dict[str, object]], sequence_rows: List[Dict[str, object]], window: int) -> Dict[str, object]:
    return {
        "window": int(window),
        "selected_cases": len(case_rows),
        "selected_sequences": len(sequence_rows),
        "gt_available_cases": sum(int(row["gt_available"]) for row in case_rows),
        "rank2_cases": sum(int(int(row["det_rank"]) == 2) for row in case_rows),
        "weak_owner_cases": sum(int(row["weak_owner_shape"]) for row in case_rows),
        "within_owner_edge_deficit_cases": sum(int(row["owner_not_stronger_by_edge"]) for row in case_rows),
        "selected_track_emitted_before_filter_cases": sum(int(row["selected_track_emitted_before_filter"]) for row in case_rows),
        "negative_edge_adv_cases": sum(int(float(row["edge_advantage_vs_owner"]) < 0.0) for row in case_rows),
        "continuity_shape_cases": sum(int(row["reclaim_continuity_shape"]) for row in case_rows),
        "supports_main_claim_shape_cases": sum(int(row["supports_main_claim_shape"]) for row in case_rows),
        "min_edge_advantage_vs_owner": min(float(row["edge_advantage_vs_owner"]) for row in case_rows),
        "max_owner_hits": max(int(row["owner_hits"]) for row in case_rows),
        "max_age_gap": max(int(row["age_gap"]) for row in case_rows),
    }


def main() -> int:
    args = parse_args()
    started_at = now_iso()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    log_path = out_dir / "audit.log"
    try:
        selected_cases = load_selected_cases(Path(args.candidate_jsonl).resolve())
        raw_presence = load_track_presence(Path(args.raw_track_dir).resolve())
        competition_presence = load_track_presence(Path(args.competition_track_dir).resolve())
        case_rows = build_case_rows(selected_cases, raw_presence, competition_presence, int(args.window))
        sequence_rows = build_sequence_rows(case_rows)
        aggregate = aggregate_row(case_rows, sequence_rows, int(args.window))

        write_rows(out_dir / "selected_cases.csv", list(case_rows[0].keys()), case_rows)
        write_rows(out_dir / "sequence_summary.csv", list(sequence_rows[0].keys()), sequence_rows)
        write_rows(out_dir / "aggregate.csv", list(aggregate.keys()), [aggregate])
        log_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "audit_selected_cases",
                    "name": out_dir.name,
                    "status": "success",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": (
                        f"selected_cases={aggregate['selected_cases']} selected_sequences={aggregate['selected_sequences']} "
                        f"rank2_cases={aggregate['rank2_cases']} weak_owner_cases={aggregate['weak_owner_cases']} "
                        f"within_owner_edge_deficit_cases={aggregate['within_owner_edge_deficit_cases']} "
                        f"selected_track_emitted_before_filter_cases={aggregate['selected_track_emitted_before_filter_cases']} "
                        f"gt_available_cases={aggregate['gt_available_cases']} uses_output_track_ids=1"
                    ),
                }
            ],
        )
        return 0
    except Exception as exc:
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "audit_selected_cases",
                    "name": out_dir.name,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": f"audit failed: {exc}",
                }
            ],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
