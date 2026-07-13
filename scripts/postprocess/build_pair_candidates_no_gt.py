#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else 0.0


def load_mot_by_track(path: Path):
    tracks = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            try:
                r = {
                    "frame": int(float(p[0])),
                    "tid": int(float(p[1])),
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "w": float(p[4]),
                    "h": float(p[5]),
                    "score": float(p[6]) if len(p) > 6 else 1.0,
                }
            except Exception:
                continue
            if r["w"] <= 2 or r["h"] <= 2:
                continue
            tracks[r["tid"]].append(r)
    for rows in tracks.values():
        rows.sort(key=lambda r: r["frame"])
    return tracks


def center(r):
    return r["x"] + r["w"] / 2.0, r["y"] + r["h"] / 2.0


def endpoint_velocity(rows, side, k=5):
    if len(rows) < 2:
        return 0.0, 0.0, 0.0
    sub = rows[-k:] if side == "end" else rows[:k]
    if len(sub) < 2:
        sub = rows[-2:] if side == "end" else rows[:2]
    a, b = sub[0], sub[-1]
    dt = max(1, b["frame"] - a["frame"])
    ca, cb = center(a), center(b)
    vx = (cb[0] - ca[0]) / dt
    vy = (cb[1] - ca[1]) / dt
    return vx, vy, math.hypot(vx, vy)


def track_stats(seq, tid, rows):
    frames = [r["frame"] for r in rows]
    scores = [r["score"] for r in rows]
    vxs, vys, _ = endpoint_velocity(rows, "start")
    vxe, vye, _ = endpoint_velocity(rows, "end")
    return {
        "seq": seq,
        "track_id": tid,
        "row_count": len(rows),
        "start_frame": min(frames),
        "end_frame": max(frames),
        "duration": max(frames) - min(frames) + 1,
        "avg_score": mean(scores),
        "first_x": rows[0]["x"],
        "first_y": rows[0]["y"],
        "first_w": rows[0]["w"],
        "first_h": rows[0]["h"],
        "first_score": rows[0]["score"],
        "last_x": rows[-1]["x"],
        "last_y": rows[-1]["y"],
        "last_w": rows[-1]["w"],
        "last_h": rows[-1]["h"],
        "last_score": rows[-1]["score"],
        "vx_start": vxs,
        "vy_start": vys,
        "vx_end": vxe,
        "vy_end": vye,
    }


def pair_features(a, b):
    gap = b["start_frame"] - a["end_frame"]
    ax = a["last_x"] + a["last_w"] / 2.0
    ay = a["last_y"] + a["last_h"] / 2.0
    bx = b["first_x"] + b["first_w"] / 2.0
    by = b["first_y"] + b["first_h"] / 2.0
    dist = math.hypot(bx - ax, by - ay)
    predx = ax + a["vx_end"] * gap
    predy = ay + a["vy_end"] * gap
    pd = math.hypot(bx - predx, by - predy)
    na = math.hypot(a["vx_end"], a["vy_end"])
    nb = math.hypot(b["vx_start"], b["vy_start"])
    vc = (a["vx_end"] * b["vx_start"] + a["vy_end"] * b["vy_start"]) / (na * nb) if na > 1e-6 and nb > 1e-6 else 0.0
    area_a = max(1e-6, a["last_w"] * a["last_h"])
    area_b = max(1e-6, b["first_w"] * b["first_h"])
    return {
        "seq": a["seq"],
        "track_a": a["track_id"],
        "track_b": b["track_id"],
        "gap": gap,
        "center_distance": dist,
        "center_distance_per_frame": dist / max(1, gap),
        "predicted_distance": pd,
        "predicted_distance_per_frame": pd / max(1, gap),
        "velocity_cosine": vc,
        "height_ratio": max(a["last_h"], b["first_h"]) / max(1e-6, min(a["last_h"], b["first_h"])),
        "area_ratio": max(area_a, area_b) / max(1e-6, min(area_a, area_b)),
        "bottom_y_gap": abs((a["last_y"] + a["last_h"]) - (b["first_y"] + b["first_h"])),
        "len_a": a["row_count"],
        "len_b": b["row_count"],
        "duration_a": a["duration"],
        "duration_b": b["duration"],
        "avg_score_a": a["avg_score"],
        "avg_score_b": b["avg_score"],
        "last_score_a": a["last_score"],
        "first_score_b": b["first_score"],
        "matched_ratio_a": 0.0,
        "matched_ratio_b": 0.0,
        "dominant_gt_a": -1,
        "dominant_gt_b": -1,
        "same_gt": 0,
        "quality_label_a": "unknown",
        "quality_label_b": "unknown",
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-link-gap", type=int, default=60)
    ap.add_argument("--max-center-step", type=float, default=80.0)
    ap.add_argument("--max-area-ratio", type=float, default=4.0)
    ap.add_argument("--min-track-len-for-link", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tracklets = []
    all_pairs = []
    by_seq = {}

    for txt in sorted(Path(args.online_dir).glob("MOT20-*.txt")):
        seq = txt.stem
        tracks = load_mot_by_track(txt)
        stats = [track_stats(seq, tid, rows) for tid, rows in sorted(tracks.items())]
        valid = [s for s in stats if s["row_count"] >= args.min_track_len_for_link]
        starts = sorted(valid, key=lambda r: r["start_frame"])
        pairs = []
        for a in valid:
            limit = a["end_frame"] + args.max_link_gap
            for b in starts:
                gap = b["start_frame"] - a["end_frame"]
                if gap <= 0:
                    continue
                if b["start_frame"] > limit:
                    break
                pf = pair_features(a, b)
                if pf["center_distance_per_frame"] <= args.max_center_step and pf["area_ratio"] <= args.max_area_ratio:
                    pairs.append(pf)
        all_tracklets.extend(stats)
        all_pairs.extend(pairs)
        by_seq[seq] = {"tracks": len(stats), "valid_tracks": len(valid), "pairs": len(pairs)}

    write_csv(out_dir / "tracklet_rows.csv", all_tracklets)
    write_csv(out_dir / "aflink_pair_candidates.csv", all_pairs)
    summary = {"candidate_pairs": len(all_pairs), "tracklets": len(all_tracklets), "by_seq": by_seq}
    (out_dir / "candidate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
