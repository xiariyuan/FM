#!/usr/bin/env python3
"""Conservative linear interpolation for MOTChallenge txt files.

Input/output rows keep MOT format columns:
frame,id,x,y,w,h,score,-1,-1,-1

Only fills missing frames inside the same track id when both endpoints exist and
simple sanity gates pass. This is intended as the first postprocess baseline, not
an aggressive AFLink/GSI replacement.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def fnum(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def inum(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def load_mot(path: Path) -> Dict[int, List[dict]]:
    tracks: Dict[int, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            row = {
                "frame": inum(parts[0]),
                "track_id": inum(parts[1]),
                "x": fnum(parts[2]),
                "y": fnum(parts[3]),
                "w": fnum(parts[4]),
                "h": fnum(parts[5]),
                "score": fnum(parts[6], 1.0) if len(parts) > 6 else 1.0,
                "tail": parts[7:] if len(parts) > 7 else ["-1", "-1", "-1"],
                "is_interp": False,
            }
            tracks[row["track_id"]].append(row)
    for tid in list(tracks):
        tracks[tid].sort(key=lambda r: r["frame"])
    return tracks


def area(row: dict) -> float:
    return max(0.0, row["w"]) * max(0.0, row["h"])


def center(row: dict) -> tuple[float, float]:
    return row["x"] + row["w"] / 2.0, row["y"] + row["h"] / 2.0


def pass_gates(a: dict, b: dict, gap: int, args) -> bool:
    if gap <= 1 or gap - 1 > args.max_gap:
        return False
    if a["score"] < args.min_endpoint_score or b["score"] < args.min_endpoint_score:
        return False
    aa, ab = area(a), area(b)
    if aa <= 1.0 or ab <= 1.0:
        return False
    ratio = max(aa, ab) / max(1e-6, min(aa, ab))
    if ratio > args.max_area_ratio:
        return False
    ca, cb = center(a), center(b)
    dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
    if dist / max(1, gap) > args.max_center_step:
        return False
    # Avoid obviously tiny/degenerate endpoints.
    if min(a["w"], a["h"], b["w"], b["h"]) < args.min_box_side:
        return False
    return True


def interpolate_rows(a: dict, b: dict, args) -> List[dict]:
    gap = b["frame"] - a["frame"]
    if not pass_gates(a, b, gap, args):
        return []
    out = []
    tail = a["tail"] if a.get("tail") else ["-1", "-1", "-1"]
    for fr in range(a["frame"] + 1, b["frame"]):
        t = (fr - a["frame"]) / gap
        score = min(a["score"], b["score"]) * args.interp_score_scale
        out.append({
            "frame": fr,
            "track_id": a["track_id"],
            "x": a["x"] + (b["x"] - a["x"]) * t,
            "y": a["y"] + (b["y"] - a["y"]) * t,
            "w": a["w"] + (b["w"] - a["w"]) * t,
            "h": a["h"] + (b["h"] - a["h"]) * t,
            "score": score,
            "tail": tail,
            "is_interp": True,
        })
    return out


def process_file(input_path: Path, output_path: Path, args) -> dict:
    tracks = load_mot(input_path)
    output_rows = []
    inserted_rows = []
    gaps_seen = 0
    gaps_filled = 0
    for tid, rows in tracks.items():
        if len(rows) < args.min_track_len:
            output_rows.extend(rows)
            continue
        output_rows.append(rows[0])
        for prev, cur in zip(rows, rows[1:]):
            gap = cur["frame"] - prev["frame"]
            if gap > 1:
                gaps_seen += 1
                new_rows = interpolate_rows(prev, cur, args)
                if new_rows:
                    gaps_filled += 1
                    inserted_rows.extend(new_rows)
                    output_rows.extend(new_rows)
            output_rows.append(cur)
    output_rows.sort(key=lambda r: (r["frame"], r["track_id"], r["x"], r["y"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for r in output_rows:
            tail = r.get("tail") or ["-1", "-1", "-1"]
            while len(tail) < 3:
                tail.append("-1")
            cols = [
                str(int(r["frame"])),
                str(int(r["track_id"])),
                f"{r['x']:.2f}",
                f"{r['y']:.2f}",
                f"{r['w']:.2f}",
                f"{r['h']:.2f}",
                f"{r['score']:.2f}",
                *tail[:3],
            ]
            f.write(",".join(cols) + "\n")
    return {
        "seq": input_path.stem,
        "input_rows": sum(len(v) for v in tracks.values()),
        "output_rows": len(output_rows),
        "inserted_rows": len(inserted_rows),
        "tracks": len(tracks),
        "gaps_seen": gaps_seen,
        "gaps_filled": gaps_filled,
        "max_gap": args.max_gap,
        "output_path": str(output_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pattern", default="*.txt")
    ap.add_argument("--max-gap", type=int, default=20)
    ap.add_argument("--min-track-len", type=int, default=2)
    ap.add_argument("--min-endpoint-score", type=float, default=0.0)
    ap.add_argument("--max-area-ratio", type=float, default=3.0)
    ap.add_argument("--max-center-step", type=float, default=80.0)
    ap.add_argument("--min-box-side", type=float, default=2.0)
    ap.add_argument("--interp-score-scale", type=float, default=1.0)
    ap.add_argument("--summary-json", default="")
    ap.add_argument("--summary-csv", default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    summaries = []
    for inp in sorted(in_dir.glob(args.pattern)):
        if inp.name.endswith(".partial"):
            continue
        summaries.append(process_file(inp, out_dir / inp.name, args))

    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = list(summaries[0].keys()) if summaries else ["seq"]
        with Path(args.summary_csv).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(summaries)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
