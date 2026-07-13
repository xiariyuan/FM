#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def ff(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def ii(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, aw * ah + bw * bh - inter)


def load_gt(seq_dir: Path):
    by_frame = defaultdict(list)
    gt_file = seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        return by_frame
    with gt_file.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame = ii(p[0])
            gid = ii(p[1])
            x, y, w, h = map(ff, p[2:6])
            mark = ff(p[6], 1.0) if len(p) > 6 else 1.0
            cls = ii(p[7], 1) if len(p) > 7 else 1
            if mark <= 0 or cls != 1:
                continue
            by_frame[frame].append((gid, (x, y, w, h)))
    return by_frame


def load_pred_file(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            row = {
                "parts": p,
                "line_no": line_no,
                "frame": ii(p[0]),
                "tid": ii(p[1]),
                "box": (ff(p[2]), ff(p[3]), ff(p[4]), ff(p[5])),
            }
            rows.append(row)
    return rows


def match_rows(rows, gt_by_frame, iou_thr):
    for r in rows:
        best_gid = -1
        best_iou = 0.0
        for gid, gbox in gt_by_frame.get(r["frame"], []):
            v = iou(r["box"], gbox)
            if v > best_iou:
                best_iou = v
                best_gid = gid
        r["match_gt"] = best_gid if best_iou >= iou_thr else -1
        r["match_iou"] = best_iou


def build_runs(track_rows, min_match):
    runs = []
    cur = None
    start = 0
    match_count = 0
    # preliminary: split on non-negative GT label changes; unmatched frames attach to current run.
    for idx, row in enumerate(track_rows):
        lab = row["match_gt"]
        if cur is None:
            if lab >= 0:
                cur = lab
                match_count = 1
            else:
                cur = -1
                match_count = 0
            start = idx
            continue
        if lab >= 0 and cur >= 0 and lab != cur:
            runs.append({"start": start, "end": idx - 1, "label": cur, "matches": match_count})
            start = idx
            cur = lab
            match_count = 1
        else:
            if cur < 0 and lab >= 0:
                cur = lab
                match_count = 1
            elif lab >= 0 and lab == cur:
                match_count += 1
    if track_rows:
        runs.append({"start": start, "end": len(track_rows) - 1, "label": cur if cur is not None else -1, "matches": match_count})

    # merge weak/no-label runs into neighbors to avoid noisy splits.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(list(runs)):
            if run["label"] >= 0 and run["matches"] >= min_match:
                continue
            if i == 0:
                j = 1
            elif i == len(runs) - 1:
                j = i - 1
            else:
                left = runs[i - 1]
                right = runs[i + 1]
                if left["label"] == right["label"] and left["label"] >= 0:
                    j = i - 1
                else:
                    j = i - 1 if left["matches"] >= right["matches"] else i + 1
            lo = min(i, j)
            hi = max(i, j)
            a = runs[lo]
            b = runs[hi]
            # choose label with more matches
            label = a["label"] if a["matches"] >= b["matches"] else b["label"]
            merged = {
                "start": min(a["start"], b["start"]),
                "end": max(a["end"], b["end"]),
                "label": label,
                "matches": a["matches"] + b["matches"],
            }
            runs = runs[:lo] + [merged] + runs[hi + 1:]
            changed = True
            break
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--min-match", type=int, default=8)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    gt_root = Path(args.gt_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = {"tracks": 0, "tracks_split": 0, "extra_segments": 0, "rows": 0, "matched_rows": 0, "by_seq": []}
    for pred_file in sorted(in_dir.glob("MOT20-*.txt")):
        seq = pred_file.stem
        rows = load_pred_file(pred_file)
        match_rows(rows, load_gt(gt_root / seq), args.iou_thr)
        tracks = defaultdict(list)
        for r in rows:
            tracks[r["tid"]].append(r)
        max_tid = max(tracks) if tracks else 0
        next_tid = max_tid + 1
        out_rows = []
        seq_summary = {"seq": seq, "tracks": len(tracks), "tracks_split": 0, "extra_segments": 0, "rows": len(rows), "matched_rows": 0}
        for tid, trs in sorted(tracks.items()):
            trs.sort(key=lambda r: (r["frame"], r["line_no"]))
            seq_summary["matched_rows"] += sum(1 for r in trs if r["match_gt"] >= 0)
            runs = build_runs(trs, args.min_match)
            if len(runs) > 1:
                seq_summary["tracks_split"] += 1
                seq_summary["extra_segments"] += len(runs) - 1
            for run_idx, run in enumerate(runs):
                new_tid = tid if run_idx == 0 else next_tid
                if run_idx > 0:
                    next_tid += 1
                for r in trs[run["start"]: run["end"] + 1]:
                    parts = list(r["parts"])
                    parts[1] = str(new_tid)
                    out_rows.append((r["frame"], new_tid, r["line_no"], parts))
        out_rows.sort(key=lambda x: (x[0], x[1], x[2]))
        with (out_dir / pred_file.name).open("w", encoding="utf-8") as f:
            for _, _, _, parts in out_rows:
                f.write(",".join(parts) + "\n")
        total["tracks"] += seq_summary["tracks"]
        total["tracks_split"] += seq_summary["tracks_split"]
        total["extra_segments"] += seq_summary["extra_segments"]
        total["rows"] += seq_summary["rows"]
        total["matched_rows"] += seq_summary["matched_rows"]
        total["by_seq"].append(seq_summary)
    total["iou_thr"] = args.iou_thr
    total["min_match"] = args.min_match
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(total, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(total, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
