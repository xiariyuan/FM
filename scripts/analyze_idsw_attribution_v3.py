#!/usr/bin/env python3
"""IDSW Attribution v3: identity-continuity based labeling.

Labels each new ID switch by checking GT identity continuity over next K frames:
- beneficial: switch improves GT identity continuity vs baseline
- harmful: switch degrades GT identity continuity vs baseline
- neutral: no measurable difference

Tracks added/removed/net switches per sequence.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
K_EVAL = 30  # frames to evaluate continuity after switch


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def append_registry(summary_csv, status, notes):
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv", str(REGISTRY_CSV), "--kind", "analysis", "--status", status,
        "--script", "scripts/analyze_idsw_attribution_v3.py",
        "--dataset", "DanceTrack", "--split", "val",
        "--tracker-family", "idsw_attribution_v3",
        "--variant", summary_csv.parent.name, "--tag", summary_csv.parent.name,
        "--run-root", str(summary_csv.parent.resolve()),
        "--summary-csv", str(summary_csv.resolve()), "--notes", notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    denom = a[2]*a[3] + b[2]*b[3] - inter
    return inter / denom if denom > 0 else 0


def load_mot(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            tid = int(float(row[1]))
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            tracks[frame].append((tid, x, y, w, h))
    return tracks


def load_gt(path):
    data = json.loads(Path(path).read_text())
    images = {int(img["id"]): img for img in data["images"]}
    gt = defaultdict(lambda: defaultdict(list))
    for ann in data["annotations"]:
        img = images.get(int(ann["image_id"]))
        if img is None:
            continue
        seq = img["file_name"].split("/")[0]
        frame = int(img["frame_id"])
        tid = int(ann["track_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        gt[seq][frame].append((tid, x, y, w, h))
    return gt


def load_bc_traces(trace_dir):
    traces = defaultdict(lambda: defaultdict(list))
    if not trace_dir or not Path(trace_dir).is_dir():
        return traces
    for p in Path(trace_dir).glob("*_bc_traces.jsonl"):
        seq = p.name.replace("_bc_traces.jsonl", "")
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                traces[seq][int(r["frame"])].append(r)
    return traces


def load_reentry_events(event_csv, tracker_name):
    events = defaultdict(lambda: defaultdict(list))
    if not event_csv or not Path(event_csv).exists():
        return events
    with open(event_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["tracker_name"] == tracker_name:
                events[r["seq_name"]][int(r["gt_id"])].append(r)
    return events


def match_tid_to_gt(gt_bbox, tracks_at_frame, thresh=0.3):
    best_tid, best_iou = -1, 0
    for tid, x, y, w, h in tracks_at_frame:
        s = iou(gt_bbox, (x, y, w, h))
        if s > best_iou:
            best_iou = s
            best_tid = tid
    return best_tid if best_iou >= thresh else -1


def gt_identity_continuity(gt_id, gt_seq, tracks, start_frame, all_frames, k):
    """Count how many of the next K frames maintain consistent tracker assignment for a GT track."""
    consistent = 0
    ref_tid = -1
    checked = 0
    for f in all_frames:
        if f < start_frame:
            continue
        if f > start_frame + k:
            break
        gt_bbox = None
        for gid, gx, gy, gw, gh in gt_seq.get(f, []):
            if gid == gt_id:
                gt_bbox = (gx, gy, gw, gh)
                break
        if gt_bbox is None:
            continue
        tid = match_tid_to_gt(gt_bbox, tracks.get(f, []))
        if tid < 0:
            continue
        checked += 1
        if ref_tid < 0:
            ref_tid = tid
        if tid == ref_tid:
            consistent += 1
    return consistent, checked


def main():
    parser = argparse.ArgumentParser(description="IDSW Attribution v3")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--bc-trace-dir", default="")
    parser.add_argument("--event-details-csv", default="")
    parser.add_argument("--compare-tracker-name", default="bc_v2_conservative_fullval")
    parser.add_argument("--k-eval", type=int, default=K_EVAL)
    parser.add_argument("--seq-list", nargs="+", default=None,
                        help="Optional sequence filter, e.g. dancetrack0026 dancetrack0030")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    detail_csv = out_dir / "idsw_detail.csv"
    seq_summary_csv = out_dir / "seq_summary.csv"

    DETAIL_FIELDS = [
        "seq_name", "gt_id", "frame",
        "prev_compare_tid", "compare_tid", "prev_baseline_tid", "baseline_tid",
        "added_vs_baseline",  # 1=new switch, 0=switch also in baseline
        "label", "label_detail",
        "bl_continuity", "co_continuity", "continuity_delta",
        "nearest_reentry_gap",
        "has_promotion_at_frame", "promotion_time_since_update",
        "promotion_emb_dist", "promotion_cost_delta",
    ]
    SEQ_FIELDS = [
        "seq_name", "added_switches", "removed_switches", "net_switches",
        "beneficial", "neutral", "harmful",
    ]

    summary = {"status": "running", "error": ""}
    write_rows(summary_csv, list(summary) + ["total_added", "total_removed", "total_net",
                                              "total_beneficial", "total_neutral", "total_harmful"], [summary])
    for p, f in [(detail_csv, DETAIL_FIELDS), (seq_summary_csv, SEQ_FIELDS)]:
        write_rows(p, f, [])
    append_registry(summary_csv, "running", "IDSW attribution v3")

    try:
        gt_data = load_gt(args.gt_json)
        bc_traces = load_bc_traces(args.bc_trace_dir)
        reentry_events = load_reentry_events(args.event_details_csv, args.compare_tracker_name)
        baseline_dir = Path(args.baseline_dir)
        compare_dir = Path(args.compare_dir)
        k = args.k_eval

        all_switches = []
        total_added = 0
        total_removed = 0
        seq_stats = {}
        seq_filter = set(args.seq_list) if args.seq_list else None

        for seq_file in sorted(compare_dir.glob("*.txt")):
            seq_name = seq_file.stem
            if seq_filter and seq_name not in seq_filter:
                continue
            gt_seq = gt_data.get(seq_name, {})
            if not gt_seq:
                continue
            # Also check baseline has this sequence
            if not (baseline_dir / f"{seq_name}.txt").is_file():
                continue

            bl_tracks = load_mot(baseline_dir / f"{seq_name}.txt")
            co_tracks = load_mot(compare_dir / f"{seq_name}.txt")
            all_frames = sorted(set(list(bl_tracks.keys()) + list(co_tracks.keys())))

            gt_ids = set()
            for frame_tracks in gt_seq.values():
                for gt_id, *_ in frame_tracks:
                    gt_ids.add(gt_id)

            seq_added = 0
            seq_removed = 0
            seq_beneficial = 0
            seq_neutral = 0
            seq_harmful = 0

            for gt_id in sorted(gt_ids):
                prev_bl_tid = -1
                prev_co_tid = -1

                for frame in all_frames:
                    gt_bbox = None
                    for gid, gx, gy, gw, gh in gt_seq.get(frame, []):
                        if gid == gt_id:
                            gt_bbox = (gx, gy, gw, gh)
                            break
                    if gt_bbox is None:
                        prev_bl_tid = -1
                        prev_co_tid = -1
                        continue

                    bl_tid = match_tid_to_gt(gt_bbox, bl_tracks.get(frame, []))
                    co_tid = match_tid_to_gt(gt_bbox, co_tracks.get(frame, []))

                    co_switch = (prev_co_tid >= 0 and co_tid >= 0 and co_tid != prev_co_tid)
                    bl_switch = (prev_bl_tid >= 0 and bl_tid >= 0 and bl_tid != prev_bl_tid)

                    if co_switch and not bl_switch:
                        seq_added += 1
                        total_added += 1

                        # Compute GT identity continuity for both trackers
                        bl_cont, bl_checked = gt_identity_continuity(
                            gt_id, gt_seq, bl_tracks, frame, all_frames, k)
                        co_cont, co_checked = gt_identity_continuity(
                            gt_id, gt_seq, co_tracks, frame, all_frames, k)

                        bl_rate = bl_cont / bl_checked if bl_checked > 0 else 0
                        co_rate = co_cont / co_checked if co_checked > 0 else 0
                        delta = co_rate - bl_rate

                        # Promotion trace
                        traces_at = bc_traces.get(seq_name, {}).get(frame, [])
                        has_promo = len(traces_at) > 0
                        promo_tsu = traces_at[0]["time_since_update"] if traces_at else ""
                        promo_emb = traces_at[0]["emb_dist"] if traces_at else ""
                        promo_cd = traces_at[0]["cost_delta"] if traces_at else ""

                        # Reentry proximity
                        events_for_gt = reentry_events.get(seq_name, {}).get(gt_id, [])
                        nearest_gap = ""
                        for ev in events_for_gt:
                            ef, rf = int(ev.get("exit_frame", 0)), int(ev.get("reentry_frame", 0))
                            if ef <= frame <= rf + 5 or abs(frame - rf) < 10:
                                nearest_gap = int(ev.get("gap", 0))
                                break

                        # Label based on continuity delta
                        if delta > 0.1:
                            label = "beneficial"
                            label_detail = f"continuity_improved_{delta:.2f}"
                            seq_beneficial += 1
                        elif delta < -0.1:
                            label = "harmful"
                            label_detail = f"continuity_degraded_{delta:.2f}"
                            seq_harmful += 1
                        else:
                            label = "neutral"
                            label_detail = f"continuity_unchanged_{delta:.2f}"
                            seq_neutral += 1

                        all_switches.append({
                            "seq_name": seq_name, "gt_id": gt_id, "frame": frame,
                            "prev_compare_tid": prev_co_tid, "compare_tid": co_tid,
                            "prev_baseline_tid": prev_bl_tid, "baseline_tid": bl_tid,
                            "added_vs_baseline": 1,
                            "label": label, "label_detail": label_detail,
                            "bl_continuity": round(bl_rate, 4),
                            "co_continuity": round(co_rate, 4),
                            "continuity_delta": round(delta, 4),
                            "nearest_reentry_gap": nearest_gap,
                            "has_promotion_at_frame": int(has_promo),
                            "promotion_time_since_update": promo_tsu,
                            "promotion_emb_dist": promo_emb,
                            "promotion_cost_delta": promo_cd,
                        })

                    # Track removed switches (baseline had switch, compare doesn't)
                    if bl_switch and not co_switch:
                        seq_removed += 1
                        total_removed += 1

                    if bl_tid >= 0:
                        prev_bl_tid = bl_tid
                    if co_tid >= 0:
                        prev_co_tid = co_tid

            seq_stats[seq_name] = {
                "seq_name": seq_name,
                "added_switches": seq_added,
                "removed_switches": seq_removed,
                "net_switches": seq_added - seq_removed,
                "beneficial": seq_beneficial,
                "neutral": seq_neutral,
                "harmful": seq_harmful,
            }

        total_beneficial = sum(1 for s in all_switches if s["label"] == "beneficial")
        total_neutral = sum(1 for s in all_switches if s["label"] == "neutral")
        total_harmful = sum(1 for s in all_switches if s["label"] == "harmful")

        summary = {
            "status": "success", "error": "",
            "total_added": total_added, "total_removed": total_removed,
            "total_net": total_added - total_removed,
            "total_beneficial": total_beneficial,
            "total_neutral": total_neutral,
            "total_harmful": total_harmful,
        }
        write_rows(summary_csv, list(summary.keys()), [summary])
        write_rows(detail_csv, DETAIL_FIELDS, all_switches)
        write_rows(seq_summary_csv, SEQ_FIELDS, list(seq_stats.values()))
        append_registry(summary_csv, "success",
                        f"v3: added={total_added}, removed={total_removed}, "
                        f"net={total_added-total_removed}, "
                        f"beneficial={total_beneficial}, harmful={total_harmful}")
        return 0

    except Exception as exc:
        summary = {"status": "failed", "error": str(exc)}
        write_rows(summary_csv, list(summary.keys()), [summary])
        append_registry(summary_csv, "failed", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
