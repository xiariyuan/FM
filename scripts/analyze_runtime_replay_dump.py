#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_groups(csv_path: Path) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["frame"] = int(row["frame"])
            row["candidate_count_total"] = int(row["candidate_count_total"])
            row["det_index"] = int(row["det_index"])
            row["track_rank"] = int(row["track_rank"])
            row["track_id"] = int(row["track_id"])
            row["is_selected"] = int(row["is_selected"])
            for key in ("det_score", "base_score", "refined_score", "motion_score", "det_cx", "det_cy", "det_w", "det_h", "track_cx", "track_cy", "track_w", "track_h"):
                row[key] = float(row[key])
            for key in ("track_gap", "track_hist_len"):
                row[key] = int(row[key])
            groups[str(row["group_id"])].append(row)
    return groups


def group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    seq = str(rows[0]["seq"])
    frame = int(rows[0]["frame"])
    det_index = int(rows[0]["det_index"])
    cand_total = int(rows[0]["candidate_count_total"])
    sorted_base = sorted(rows, key=lambda r: float(r["base_score"]), reverse=True)
    sorted_refined = sorted(rows, key=lambda r: float(r["refined_score"]), reverse=True)
    base_top1 = sorted_base[0]
    refined_top1 = sorted_refined[0]
    base_top2 = sorted_base[1] if len(sorted_base) > 1 else sorted_base[0]
    refined_top2 = sorted_refined[1] if len(sorted_refined) > 1 else sorted_refined[0]
    selected = next((r for r in rows if int(r["is_selected"]) == 1), None)
    selected_track = int(selected["track_id"]) if selected is not None else -1
    base_top1_track = int(base_top1["track_id"])
    refined_top1_track = int(refined_top1["track_id"])
    return {
        "seq": seq,
        "frame": frame,
        "det_index": det_index,
        "group_id": str(rows[0]["group_id"]),
        "candidate_count_total": cand_total,
        "base_top1_track": base_top1_track,
        "refined_top1_track": refined_top1_track,
        "selected_track": selected_track,
        "top1_flip": int(base_top1_track != refined_top1_track),
        "selected_equals_base_top1": int(selected_track == base_top1_track),
        "selected_equals_refined_top1": int(selected_track == refined_top1_track),
        "base_margin": float(base_top1["base_score"]) - float(base_top2["base_score"]),
        "refined_margin": float(refined_top1["refined_score"]) - float(refined_top2["refined_score"]),
        "selected_base_score": float(selected["base_score"]) if selected is not None else -1.0,
        "selected_refined_score": float(selected["refined_score"]) if selected is not None else -1.0,
        "mean_hist_len": sum(int(r["track_hist_len"]) for r in rows) / float(max(len(rows), 1)),
        "mean_track_gap": sum(int(r["track_gap"]) for r in rows) / float(max(len(rows), 1)),
        "det_score": float(rows[0]["det_score"]),
    }


def seq_aggregate(group_summaries: list[dict[str, object]]) -> dict[str, float]:
    n = len(group_summaries)
    if n == 0:
        return {}
    amb = [g for g in group_summaries if float(g["base_margin"]) < 0.05]
    very_amb = [g for g in group_summaries if float(g["base_margin"]) < 0.02]
    return {
        "groups": n,
        "flip_rate": sum(int(g["top1_flip"]) for g in group_summaries) / n,
        "base_selected_top1_rate": sum(int(g["selected_equals_base_top1"]) for g in group_summaries) / n,
        "refined_selected_top1_rate": sum(int(g["selected_equals_refined_top1"]) for g in group_summaries) / n,
        "mean_base_margin": sum(float(g["base_margin"]) for g in group_summaries) / n,
        "mean_refined_margin": sum(float(g["refined_margin"]) for g in group_summaries) / n,
        "mean_hist_len": sum(float(g["mean_hist_len"]) for g in group_summaries) / n,
        "mean_track_gap": sum(float(g["mean_track_gap"]) for g in group_summaries) / n,
        "ambiguous_rate_005": len(amb) / n,
        "ambiguous_flip_rate_005": (sum(int(g["top1_flip"]) for g in amb) / len(amb)) if amb else 0.0,
        "very_ambiguous_rate_002": len(very_amb) / n,
        "very_ambiguous_flip_rate_002": (sum(int(g["top1_flip"]) for g in very_amb) / len(very_amb)) if very_amb else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze runtime replay diagnostic dump CSV files.")
    ap.add_argument("--dump-root", required=True, help="Directory containing per-sequence runtime dump CSVs")
    ap.add_argument("--out-json", default="", help="Optional output json path")
    ap.add_argument("--top-groups", type=int, default=20, help="How many flipped groups to print")
    args = ap.parse_args()

    dump_root = Path(args.dump_root).resolve()
    csv_files = sorted(dump_root.rglob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No dump csv files found under {dump_root}")

    all_groups: list[dict[str, object]] = []
    seq_to_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for csv_path in csv_files:
        groups = load_groups(csv_path)
        for rows in groups.values():
            gs = group_summary(rows)
            all_groups.append(gs)
            seq_to_groups[str(gs["seq"])].append(gs)

    result = {
        "dump_root": str(dump_root),
        "sequence_summaries": {seq: seq_aggregate(groups) for seq, groups in sorted(seq_to_groups.items())},
        "top_flipped_groups": sorted(
            [g for g in all_groups if int(g["top1_flip"]) == 1],
            key=lambda g: (float(g["base_margin"]), -float(g["selected_refined_score"])),
        )[: max(int(args.top_groups), 0)],
    }

    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result["sequence_summaries"], indent=2, sort_keys=True))
    if result["top_flipped_groups"]:
        print("\nTop flipped groups:")
        for g in result["top_flipped_groups"]:
            print(
                f"{g['seq']} frame={g['frame']} det={g['det_index']} "
                f"base_margin={g['base_margin']:.4f} refined_margin={g['refined_margin']:.4f} "
                f"base_top1={g['base_top1_track']} refined_top1={g['refined_top1_track']} selected={g['selected_track']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
