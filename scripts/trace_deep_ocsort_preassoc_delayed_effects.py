#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set


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
        description="Trace delayed output effects of selected Deep-OC-SORT pre-association stale-competition reclaim cases."
    )
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--competition-track-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20, 30])
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


def load_selected_cases(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                seq_frames[frame_id].add(track_id)
        presence[txt_path.stem] = seq_frames
    if not presence:
        raise FileNotFoundError(f"No MOT tracker txt files found in {track_dir}")
    return presence


def has_track(frame_map: Dict[int, Set[int]], frame_id: int, track_id: int) -> int:
    if track_id <= 0:
        return 0
    return int(track_id in frame_map.get(frame_id, set()))


def count_track_frames(frame_map: Dict[int, Set[int]], start_frame: int, horizon: int, track_id: int) -> int:
    if track_id <= 0:
        return 0
    total = 0
    for frame_id in range(start_frame + 1, start_frame + int(horizon) + 1):
        total += int(track_id in frame_map.get(frame_id, set()))
    return total


def first_emit_gap(frame_map: Dict[int, Set[int]], start_frame: int, horizon: int, track_id: int) -> int:
    if track_id <= 0:
        return -1
    for delta in range(1, int(horizon) + 1):
        if track_id in frame_map.get(start_frame + delta, set()):
            return int(delta)
    return -1


def max_consecutive_run(frame_map: Dict[int, Set[int]], start_frame: int, horizon: int, track_id: int) -> int:
    if track_id <= 0:
        return 0
    best = 0
    current = 0
    for frame_id in range(start_frame + 1, start_frame + int(horizon) + 1):
        if track_id in frame_map.get(frame_id, set()):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def winner_label(selected_gap: int, owner_gap: int) -> str:
    if selected_gap > 0 and (owner_gap < 0 or selected_gap < owner_gap):
        return "selected_delayed_win"
    if owner_gap > 0 and (selected_gap < 0 or owner_gap < selected_gap):
        return "owner_delayed_win"
    if selected_gap > 0 and owner_gap > 0 and selected_gap == owner_gap:
        return "tie_same_gap"
    return "neither_within_horizon"


def build_case_rows(
    selected_cases: List[Dict[str, object]],
    presence: Dict[str, Dict[int, Set[int]]],
    horizons: List[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for case in selected_cases:
        seq_name = str(case.get("seq_name", ""))
        frame_id = int(case.get("frame_id", 0) or 0)
        frame_map = presence[seq_name]
        selected_track_id = int(case.get("track_output_id", -1) or -1)
        owner_track_id = int(case.get("raw_owner_output_id", -1) or -1)
        row: Dict[str, object] = {
            "seq_name": seq_name,
            "frame_id": frame_id,
            "track_output_id": selected_track_id,
            "raw_owner_output_id": owner_track_id,
            "selected_final": int(case.get("selected_final", 0) or 0),
            "selected_forced": int(case.get("selected_forced", 0) or 0),
            "selected_track_emitted_before_filter": int(case.get("selected_track_emitted_before_filter", 0) or 0),
            "raw_owner_emitted_before_filter": int(case.get("raw_owner_emitted_before_filter", 0) or 0),
            "final_det_owner_output_id": int(case.get("final_det_owner_output_id", -1) or -1),
            "track_hits": int(case.get("track_hits", 0) or 0),
            "owner_hits": int(case.get("owner_hits", 0) or 0),
            "det_rank": int(case.get("det_rank", -1) or -1),
            "edge_advantage_vs_owner": float(case.get("edge_advantage_vs_owner", 0.0) or 0.0),
            "selected_emitted_at_frame": has_track(frame_map, frame_id, selected_track_id),
            "owner_emitted_at_frame": has_track(frame_map, frame_id, owner_track_id),
        }
        max_horizon = max(int(h) for h in horizons)
        row["selected_first_emit_gap_hmax"] = first_emit_gap(frame_map, frame_id, max_horizon, selected_track_id)
        row["owner_first_emit_gap_hmax"] = first_emit_gap(frame_map, frame_id, max_horizon, owner_track_id)
        row["selected_max_run_hmax"] = max_consecutive_run(frame_map, frame_id, max_horizon, selected_track_id)
        row["owner_max_run_hmax"] = max_consecutive_run(frame_map, frame_id, max_horizon, owner_track_id)
        for horizon in horizons:
            selected_gap = first_emit_gap(frame_map, frame_id, horizon, selected_track_id)
            owner_gap = first_emit_gap(frame_map, frame_id, horizon, owner_track_id)
            selected_count = count_track_frames(frame_map, frame_id, horizon, selected_track_id)
            owner_count = count_track_frames(frame_map, frame_id, horizon, owner_track_id)
            row[f"selected_first_emit_gap_h{horizon}"] = selected_gap
            row[f"owner_first_emit_gap_h{horizon}"] = owner_gap
            row[f"selected_emit_count_h{horizon}"] = selected_count
            row[f"owner_emit_count_h{horizon}"] = owner_count
            row[f"selected_max_run_h{horizon}"] = max_consecutive_run(frame_map, frame_id, horizon, selected_track_id)
            row[f"owner_max_run_h{horizon}"] = max_consecutive_run(frame_map, frame_id, horizon, owner_track_id)
            row[f"winner_h{horizon}"] = winner_label(selected_gap, owner_gap)
            row[f"selected_only_h{horizon}"] = int(selected_count > 0 and owner_count == 0)
            row[f"owner_only_h{horizon}"] = int(owner_count > 0 and selected_count == 0)
            row[f"selected_beats_owner_h{horizon}"] = int(winner_label(selected_gap, owner_gap) == "selected_delayed_win")
            row[f"owner_beats_selected_h{horizon}"] = int(winner_label(selected_gap, owner_gap) == "owner_delayed_win")
        rows.append(row)
    return rows


def build_aggregate(case_rows: List[Dict[str, object]], horizons: List[int]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "selected_cases": len(case_rows),
        "selected_track_emitted_before_filter_cases": sum(int(r["selected_track_emitted_before_filter"]) for r in case_rows),
        "selected_emitted_at_frame_cases": sum(int(r["selected_emitted_at_frame"]) for r in case_rows),
        "owner_emitted_at_frame_cases": sum(int(r["owner_emitted_at_frame"]) for r in case_rows),
    }
    for horizon in horizons:
        row[f"selected_emerges_h{horizon}"] = sum(int(int(r[f"selected_first_emit_gap_h{horizon}"]) > 0) for r in case_rows)
        row[f"owner_emerges_h{horizon}"] = sum(int(int(r[f"owner_first_emit_gap_h{horizon}"]) > 0) for r in case_rows)
        row[f"selected_only_h{horizon}"] = sum(int(r[f"selected_only_h{horizon}"]) for r in case_rows)
        row[f"owner_only_h{horizon}"] = sum(int(r[f"owner_only_h{horizon}"]) for r in case_rows)
        row[f"selected_beats_owner_h{horizon}"] = sum(int(r[f"selected_beats_owner_h{horizon}"]) for r in case_rows)
        row[f"owner_beats_selected_h{horizon}"] = sum(int(r[f"owner_beats_selected_h{horizon}"]) for r in case_rows)
        row[f"selected_emit_total_h{horizon}"] = sum(int(r[f"selected_emit_count_h{horizon}"]) for r in case_rows)
        row[f"owner_emit_total_h{horizon}"] = sum(int(r[f"owner_emit_count_h{horizon}"]) for r in case_rows)
    return row


def build_sequence_rows(case_rows: List[Dict[str, object]], horizons: List[int]) -> List[Dict[str, object]]:
    by_seq: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_seq[str(row["seq_name"])].append(row)

    seq_rows: List[Dict[str, object]] = []
    for seq_name, rows in sorted(by_seq.items()):
        seq_row: Dict[str, object] = {
            "seq_name": seq_name,
            "selected_cases": len(rows),
            "selected_track_emitted_before_filter_cases": sum(int(r["selected_track_emitted_before_filter"]) for r in rows),
        }
        for horizon in horizons:
            seq_row[f"selected_beats_owner_h{horizon}"] = sum(int(r[f"selected_beats_owner_h{horizon}"]) for r in rows)
            seq_row[f"owner_beats_selected_h{horizon}"] = sum(int(r[f"owner_beats_selected_h{horizon}"]) for r in rows)
            seq_row[f"selected_emit_total_h{horizon}"] = sum(int(r[f"selected_emit_count_h{horizon}"]) for r in rows)
            seq_row[f"owner_emit_total_h{horizon}"] = sum(int(r[f"owner_emit_count_h{horizon}"]) for r in rows)
        seq_rows.append(seq_row)
    return seq_rows


def main() -> int:
    args = parse_args()
    started_at = now_iso()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    log_path = out_dir / "trace.log"
    write_rows(
        summary_csv,
        SUMMARY_FIELDS,
        [
            {
                "step": "trace_delayed_effects",
                "name": out_dir.name,
                "status": "running",
                "out_dir": str(out_dir),
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "started_at": started_at,
                "finished_at": "",
                "notes": "trace started",
            }
        ],
    )
    try:
        horizons = sorted(set(int(h) for h in args.horizons))
        selected_cases = load_selected_cases(Path(args.candidate_jsonl).resolve())
        presence = load_track_presence(Path(args.competition_track_dir).resolve())
        case_rows = build_case_rows(selected_cases, presence, horizons)
        aggregate = build_aggregate(case_rows, horizons)
        sequence_rows = build_sequence_rows(case_rows, horizons)

        write_rows(out_dir / "trace_cases.csv", list(case_rows[0].keys()), case_rows)
        write_rows(out_dir / "aggregate.csv", list(aggregate.keys()), [aggregate])
        write_rows(out_dir / "sequence_summary.csv", list(sequence_rows[0].keys()), sequence_rows)
        log_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        notes = (
            f"selected_cases={aggregate['selected_cases']} "
            f"selected_beats_owner_h10={aggregate.get('selected_beats_owner_h10', 0)} "
            f"owner_beats_selected_h10={aggregate.get('owner_beats_selected_h10', 0)} "
            f"selected_beats_owner_h20={aggregate.get('selected_beats_owner_h20', 0)} "
            f"owner_beats_selected_h20={aggregate.get('owner_beats_selected_h20', 0)}"
        )
        write_rows(
            summary_csv,
            SUMMARY_FIELDS,
            [
                {
                    "step": "trace_delayed_effects",
                    "name": out_dir.name,
                    "status": "success",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": notes,
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
                    "step": "trace_delayed_effects",
                    "name": out_dir.name,
                    "status": "failed",
                    "out_dir": str(out_dir),
                    "summary_csv": str(summary_csv),
                    "log_path": str(log_path),
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "notes": f"trace failed: {exc}",
                }
            ],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
