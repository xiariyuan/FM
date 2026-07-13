#!/usr/bin/env python3
"""
Improved tracklet linking with gap-aware thresholds and competition filtering.

Improvements over link_tracks_by_score.py:
1. Gap-dependent score threshold: stricter for longer gaps
2. Competition filter: top candidate must dominate second-best by a margin
3. Hard appearance filter: explicit minimum cosine similarity
4. Track length filter: skip very short tracklets
5. Optimal 1:1 matching via greedy with competition awareness
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_mot(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append((int(float(parts[0])), int(float(parts[1])), parts))
    return rows


def find(parent, x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def gap_threshold(gap: int, base_thr: float) -> float:
    """Stricter threshold for longer gaps."""
    if gap <= 5:
        return max(base_thr - 0.05, 0.05)
    elif gap <= 10:
        return base_thr
    elif gap <= 20:
        return base_thr + 0.03
    elif gap <= 30:
        return base_thr + 0.06
    else:
        return base_thr + 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with MOT txt files")
    ap.add_argument("--scores", required=True, help="CSV with aflink predictions")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--base-thr", type=float, default=0.20, help="Base score threshold")
    ap.add_argument("--max-gap", type=int, default=40, help="Max gap to consider (default 40, was 60)")
    ap.add_argument("--min-len", type=int, default=5, help="Minimum tracklet length to link")
    ap.add_argument("--min-appearance", type=float, default=0.25, help="Minimum appearance_mean cosine")
    ap.add_argument("--competition-margin", type=float, default=0.03, help="Min score advantage over 2nd-best")
    ap.add_argument("--max-links-per-seq", type=int, default=999999)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_seq = defaultdict(list)
    with open(args.scores, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = float(row.get("aflink_score", 0.0) or 0.0)
            gap = int(float(row.get("gap", 0) or 0))
            if gap <= 0 or gap > args.max_gap:
                continue
            len_a = int(float(row.get("len_a", 0) or 0))
            len_b = int(float(row.get("len_b", 0) or 0))
            app_mean = float(row.get("appearance_mean", 0.0) or 0.0)
            if len_a < args.min_len or len_b < args.min_len:
                continue
            if app_mean < args.min_appearance:
                continue
            thr = gap_threshold(gap, args.base_thr)
            if score < thr:
                continue
            candidates_by_seq[row["seq"]].append(
                {
                    "seq": row["seq"],
                    "track_a": int(float(row["track_a"])),
                    "track_b": int(float(row["track_b"])),
                    "score": score,
                    "same_gt": int(float(row.get("same_gt", 0) or 0)),
                    "gap": gap,
                    "len_a": len_a,
                    "len_b": len_b,
                    "appearance_mean": app_mean,
                }
            )

    selected_all = []
    by_seq_report = []
    input_dir = Path(args.input_dir)

    for seq, candidates in sorted(candidates_by_seq.items()):
        candidates.sort(key=lambda r: -r["score"])

        # Build competition index: for each track_a and track_b, find top-2 candidates
        by_a = defaultdict(list)
        by_b = defaultdict(list)
        for c in candidates:
            by_a[c["track_a"]].append(c)
            by_b[c["track_b"]].append(c)

        # Apply competition filter
        filtered = []
        for c in candidates:
            a = c["track_a"]
            b = c["track_b"]
            # Check if this is the best candidate for track_a
            a_cands = by_a[a]
            if len(a_cands) > 1:
                second_best_a = a_cands[1]["score"] if a_cands[0] is not c else (a_cands[1]["score"] if len(a_cands) > 1 else 0.0)
                if a_cands[0] is c:
                    if c["score"] - second_best_a < args.competition_margin:
                        continue
                else:
                    continue  # not the best for track_a

            # Check if this is the best candidate for track_b
            b_cands = by_b[b]
            if len(b_cands) > 1:
                if b_cands[0] is not c:
                    continue  # not the best for track_b

            filtered.append(c)

        # Greedy 1:1 matching on filtered candidates
        used_successor = set()
        used_predecessor = set()
        parent = {}
        selected = []
        for cand in filtered:
            if len(selected) >= args.max_links_per_seq:
                break
            a = cand["track_a"]
            b = cand["track_b"]
            if a in used_successor or b in used_predecessor:
                continue
            root_a = find(parent, a)
            root_b = find(parent, b)
            if root_a == root_b:
                continue
            parent[root_b] = root_a
            used_successor.add(a)
            used_predecessor.add(b)
            selected.append(cand)

        involved_ids = set()
        for cand in selected:
            involved_ids.add(cand["track_a"])
            involved_ids.add(cand["track_b"])
        id_map = {tid: find(parent, tid) for tid in involved_ids}

        input_file = input_dir / f"{seq}.txt"
        output_file = out_dir / f"{seq}.txt"
        if input_file.exists():
            new_rows = []
            for _, tid, parts in read_mot(input_file):
                parts = list(parts)
                parts[1] = str(id_map.get(tid, tid))
                new_rows.append(parts)
            new_rows.sort(key=lambda p: (int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
            with output_file.open("w", encoding="utf-8") as f:
                for parts in new_rows:
                    f.write(",".join(parts) + "\n")

        selected_all.extend(selected)
        tp = sum(c["same_gt"] for c in selected)
        by_seq_report.append(
            {
                "seq": seq,
                "candidate_after_threshold": len(candidates),
                "candidate_after_competition": len(filtered),
                "selected_links": len(selected),
                "tp_train_label": tp,
                "precision_train_label": tp / len(selected) if selected else 0.0,
            }
        )

    for input_file in sorted(input_dir.glob("MOT20-*.txt")):
        output_file = out_dir / input_file.name
        if not output_file.exists():
            output_file.write_text(input_file.read_text(encoding="utf-8"), encoding="utf-8")

    total_tp = sum(c["same_gt"] for c in selected_all)
    summary = {
        "base_thr": args.base_thr,
        "max_gap": args.max_gap,
        "min_len": args.min_len,
        "min_appearance": args.min_appearance,
        "competition_margin": args.competition_margin,
        "selected_links": len(selected_all),
        "tp_train_label": total_tp,
        "precision_train_label": total_tp / len(selected_all) if selected_all else 0.0,
        "by_seq": by_seq_report,
    }
    parent_dir = out_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    (parent_dir / "link_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (parent_dir / "selected_links.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "track_a", "track_b", "score", "same_gt", "gap", "len_a", "len_b", "appearance_mean"])
        writer.writeheader()
        writer.writerows(selected_all)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
