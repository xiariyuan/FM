#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def load_tracklets(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = row["seq"]
            tid = to_int(row["track_id"])
            out[(seq, tid)] = row
    return out


def load_frames(results_dir: Path):
    by_seq_frame = defaultdict(lambda: defaultdict(list))
    for txt in sorted(results_dir.glob("MOT20-*.txt")):
        seq = txt.stem
        with txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = line.split(",")
                if len(p) < 6:
                    continue
                r = {
                    "frame": to_int(p[0]),
                    "tid": to_int(p[1]),
                    "x": to_float(p[2]),
                    "y": to_float(p[3]),
                    "w": to_float(p[4]),
                    "h": to_float(p[5]),
                    "score": to_float(p[6], 1.0) if len(p) > 6 else 1.0,
                }
                r["cx"] = r["x"] + r["w"] / 2.0
                r["cy"] = r["y"] + r["h"] / 2.0
                r["bottom"] = r["y"] + r["h"]
                r["area"] = max(0.0, r["w"]) * max(0.0, r["h"])
                by_seq_frame[seq][r["frame"]].append(r)
    return by_seq_frame


def iou_ioa(a, b):
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    union = max(1e-9, a["area"] + b["area"] - inter)
    min_area = max(1e-9, min(a["area"], b["area"]))
    return inter / union, inter / min_area


def endpoint_box(tracklet, side: str):
    if side == "end":
        return {
            "frame": to_int(tracklet["end_frame"]),
            "tid": to_int(tracklet["track_id"]),
            "x": to_float(tracklet["last_x"]),
            "y": to_float(tracklet["last_y"]),
            "w": to_float(tracklet["last_w"]),
            "h": to_float(tracklet["last_h"]),
            "score": to_float(tracklet.get("last_score", 1.0), 1.0),
        }
    return {
        "frame": to_int(tracklet["start_frame"]),
        "tid": to_int(tracklet["track_id"]),
        "x": to_float(tracklet["first_x"]),
        "y": to_float(tracklet["first_y"]),
        "w": to_float(tracklet["first_w"]),
        "h": to_float(tracklet["first_h"]),
        "score": to_float(tracklet.get("first_score", 1.0), 1.0),
    }


def stats_for_endpoint(seq, box, frame_boxes):
    box = dict(box)
    box["cx"] = box["x"] + box["w"] / 2.0
    box["cy"] = box["y"] + box["h"] / 2.0
    box["bottom"] = box["y"] + box["h"]
    box["area"] = max(0.0, box["w"]) * max(0.0, box["h"])
    rows = frame_boxes.get(seq, {}).get(box["frame"], [])
    others = [r for r in rows if r["tid"] != box["tid"]]
    n_frame = len(rows)
    counts = {50: 0, 100: 0, 150: 0}
    max_iou = 0.0
    max_ioa = 0.0
    iou01 = 0
    ioa03 = 0
    for r in others:
        d = math.hypot(r["cx"] - box["cx"], r["cy"] - box["cy"])
        for rad in counts:
            if d <= rad:
                counts[rad] += 1
        iv, av = iou_ioa(box, r)
        max_iou = max(max_iou, iv)
        max_ioa = max(max_ioa, av)
        if iv >= 0.1:
            iou01 += 1
        if av >= 0.3:
            ioa03 += 1
    if n_frame <= 1:
        height_rank = 0.0
        bottom_rank = 0.0
    else:
        heights = sorted([r["h"] for r in rows], reverse=True)
        bottoms = sorted([r["bottom"] for r in rows], reverse=True)
        h_rank = 1 + sum(1 for h in heights if h > box["h"])
        b_rank = 1 + sum(1 for b in bottoms if b > box["bottom"])
        height_rank = h_rank / n_frame
        bottom_rank = b_rank / n_frame
    return {
        "frame_box_count": n_frame,
        "nearby_count_r50": counts[50],
        "nearby_count_r100": counts[100],
        "nearby_count_r150": counts[150],
        "max_neighbor_iou": max_iou,
        "max_neighbor_ioa": max_ioa,
        "overlap_iou01_count": iou01,
        "overlap_ioa03_count": ioa03,
        "height_rank_norm": height_rank,
        "bottom_rank_norm": bottom_rank,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tracklets", required=True)
    ap.add_argument("--online-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary-json", default="")
    args = ap.parse_args()

    tracklets = load_tracklets(Path(args.tracklets))
    frames = load_frames(Path(args.online_dir))
    rows_out = []
    missing = 0
    with open(args.pairs, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = row["seq"]
            ta = to_int(row["track_a"])
            tb = to_int(row["track_b"])
            a = tracklets.get((seq, ta))
            b = tracklets.get((seq, tb))
            rr = dict(row)
            if a is None or b is None:
                missing += 1
                a_stats = {k: 0.0 for k in ["frame_box_count", "nearby_count_r50", "nearby_count_r100", "nearby_count_r150", "max_neighbor_iou", "max_neighbor_ioa", "overlap_iou01_count", "overlap_ioa03_count", "height_rank_norm", "bottom_rank_norm"]}
                b_stats = dict(a_stats)
            else:
                a_stats = stats_for_endpoint(seq, endpoint_box(a, "end"), frames)
                b_stats = stats_for_endpoint(seq, endpoint_box(b, "start"), frames)
            for k, v in a_stats.items():
                rr[f"a_end_{k}"] = v
            for k, v in b_stats.items():
                rr[f"b_start_{k}"] = v
            rr["pair_density_r100_sum"] = float(a_stats["nearby_count_r100"] + b_stats["nearby_count_r100"])
            rr["pair_density_r150_sum"] = float(a_stats["nearby_count_r150"] + b_stats["nearby_count_r150"])
            rr["pair_max_neighbor_iou"] = max(float(a_stats["max_neighbor_iou"]), float(b_stats["max_neighbor_iou"]))
            rr["pair_max_neighbor_ioa"] = max(float(a_stats["max_neighbor_ioa"]), float(b_stats["max_neighbor_ioa"]))
            rr["height_rank_gap_abs"] = abs(float(a_stats["height_rank_norm"]) - float(b_stats["height_rank_norm"]))
            rr["bottom_rank_gap_abs"] = abs(float(a_stats["bottom_rank_norm"]) - float(b_stats["bottom_rank_norm"]))
            rows_out.append(rr)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    summary = {"rows": len(rows_out), "missing_tracklets": missing, "out_csv": str(out)}
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
