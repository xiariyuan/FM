#!/usr/bin/env python3
"""IDSW Attribution: classify each new ID switch as beneficial/neutral/harmful.

Cross-refs:
1. Per-frame track results (baseline vs compare)
2. Promotion trace logs from bc-v2
3. Reentry event paired flips (if available)

For each GT track, detects where the assigned tracker ID switches,
then classifies each switch.
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
        "--script", "scripts/analyze_idsw_attribution_v2.py",
        "--dataset", "DanceTrack", "--split", "val",
        "--tracker-family", "idsw_attribution_v2",
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
    """Load promotion traces: seq -> frame -> list of trace records."""
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
    """Load per-GT event status: seq -> gt_id -> list of event dicts."""
    events = defaultdict(lambda: defaultdict(list))
    if not event_csv or not Path(event_csv).exists():
        return events
    with open(event_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["tracker_name"] == tracker_name:
                events[r["seq_name"]][int(r["gt_id"])].append(r)
    return events


def main():
    parser = argparse.ArgumentParser(description="IDSW Attribution v2")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--compare-dir", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--bc-trace-dir", default="")
    parser.add_argument("--event-details-csv", default="")
    parser.add_argument("--compare-tracker-name", default="bc_v2_conservative_fullval")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    detail_csv = out_dir / "idsw_detail.csv"
    seq_summary_csv = out_dir / "seq_summary.csv"
    gap_summary_csv = out_dir / "gap_summary.csv"
    tsu_summary_csv = out_dir / "tsu_summary.csv"

    DETAIL_FIELDS = [
        "seq_name", "gt_id", "frame", "prev_compare_tid", "compare_tid",
        "prev_baseline_tid", "baseline_tid",
        "label", "label_detail",
        "nearest_reentry_gap", "has_promotion_at_frame", "promotion_time_since_update",
        "promotion_emb_dist", "promotion_cost_delta",
        "switch_persists_n_frames",
    ]
    SEQ_FIELDS = ["seq_name", "beneficial", "neutral", "harmful", "total"]
    GAP_FIELDS = ["gap_bucket", "beneficial", "neutral", "harmful", "total"]
    TSU_FIELDS = ["tsu_bucket", "beneficial", "neutral", "harmful", "total"]

    summary = {"status": "running", "error": ""}
    write_rows(summary_csv, ["status", "error", "total_new_switches",
                             "beneficial", "neutral", "harmful"], [summary])
    for p, f in [(detail_csv, DETAIL_FIELDS), (seq_summary_csv, SEQ_FIELDS),
                 (gap_summary_csv, GAP_FIELDS), (tsu_summary_csv, TSU_FIELDS)]:
        write_rows(p, f, [])
    append_registry(summary_csv, "running", "IDSW attribution v2")

    try:
        gt_data = load_gt(args.gt_json)
        bc_traces = load_bc_traces(args.bc_trace_dir)
        reentry_events = load_reentry_events(args.event_details_csv, args.compare_tracker_name)

        baseline_dir = Path(args.baseline_dir)
        compare_dir = Path(args.compare_dir)

        all_switches = []

        for seq_file in sorted(compare_dir.glob("*.txt")):
            seq_name = seq_file.stem
            gt_seq = gt_data.get(seq_name, {})
            if not gt_seq:
                continue

            bl_tracks = load_mot(baseline_dir / f"{seq_name}.txt")
            co_tracks = load_mot(compare_dir / f"{seq_name}.txt")

            gt_ids = set()
            for frame_tracks in gt_seq.values():
                for gt_id, *_ in frame_tracks:
                    gt_ids.add(gt_id)

            all_frames = sorted(set(list(bl_tracks.keys()) + list(co_tracks.keys())))

            for gt_id in sorted(gt_ids):
                prev_bl_tid = -1
                prev_co_tid = -1
                switch_persist_count = 0
                last_switch_frame = -999

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

                    bl_tid = -1
                    best_bl = 0
                    for tid, x, y, w, h in bl_tracks.get(frame, []):
                        s = iou(gt_bbox, (x, y, w, h))
                        if s > best_bl:
                            best_bl = s
                            bl_tid = tid
                    if best_bl < 0.3:
                        bl_tid = -1

                    co_tid = -1
                    best_co = 0
                    for tid, x, y, w, h in co_tracks.get(frame, []):
                        s = iou(gt_bbox, (x, y, w, h))
                        if s > best_co:
                            best_co = s
                            co_tid = tid
                    if best_co < 0.3:
                        co_tid = -1

                    co_switch = (prev_co_tid >= 0 and co_tid >= 0 and co_tid != prev_co_tid)
                    bl_switch = (prev_bl_tid >= 0 and bl_tid >= 0 and bl_tid != prev_bl_tid)

                    if co_switch and not bl_switch:
                        # Check persistence: does the new tid persist?
                        future_persist = 0
                        check_tid = co_tid
                        for f2 in all_frames:
                            if f2 <= frame:
                                continue
                            if f2 > frame + 20:
                                break
                            found = False
                            for tid, x, y, w, h in co_tracks.get(f2, []):
                                if tid == check_tid:
                                    gt2 = None
                                    for gid, gx, gy, gw, gh in gt_seq.get(f2, []):
                                        if gid == gt_id:
                                            gt2 = (gx, gy, gw, gh)
                                            break
                                    if gt2 and iou(gt2, (x, y, w, h)) >= 0.3:
                                        found = True
                                        break
                            if found:
                                future_persist += 1

                        # Check promotion trace
                        traces_at_frame = bc_traces.get(seq_name, {}).get(frame, [])
                        has_promo = len(traces_at_frame) > 0
                        promo_tsu = traces_at_frame[0]["time_since_update"] if traces_at_frame else ""
                        promo_emb = traces_at_frame[0]["emb_dist"] if traces_at_frame else ""
                        promo_cd = traces_at_frame[0]["cost_delta"] if traces_at_frame else ""

                        # Check nearest reentry event
                        events_for_gt = reentry_events.get(seq_name, {}).get(gt_id, [])
                        nearest_gap = ""
                        for ev in events_for_gt:
                            ef = int(ev.get("exit_frame", 0))
                            rf = int(ev.get("reentry_frame", 0))
                            if ef <= frame <= rf + 5:
                                nearest_gap = int(ev.get("gap", 0))
                                break
                            if abs(frame - rf) < 10:
                                nearest_gap = int(ev.get("gap", 0))

                        # Classify
                        if has_promo and nearest_gap:
                            label = "beneficial"
                            label_detail = "promotion_near_reentry"
                        elif has_promo and future_persist >= 10:
                            label = "beneficial"
                            label_detail = "promotion_persists"
                        elif has_promo and future_persist >= 5:
                            label = "neutral"
                            label_detail = "promotion_moderate_persist"
                        elif has_promo:
                            label = "harmful"
                            label_detail = "promotion_no_persist"
                        elif future_persist >= 10:
                            label = "neutral"
                            label_detail = "no_promotion_persists"
                        else:
                            label = "harmful"
                            label_detail = "no_promotion_no_persist"

                        all_switches.append({
                            "seq_name": seq_name,
                            "gt_id": gt_id,
                            "frame": frame,
                            "prev_compare_tid": prev_co_tid,
                            "compare_tid": co_tid,
                            "prev_baseline_tid": prev_bl_tid,
                            "baseline_tid": bl_tid,
                            "label": label,
                            "label_detail": label_detail,
                            "nearest_reentry_gap": nearest_gap,
                            "has_promotion_at_frame": int(has_promo),
                            "promotion_time_since_update": promo_tsu,
                            "promotion_emb_dist": promo_emb,
                            "promotion_cost_delta": promo_cd,
                            "switch_persists_n_frames": future_persist,
                        })

                    if prev_bl_tid >= 0 and bl_tid >= 0:
                        prev_bl_tid = bl_tid
                    if prev_co_tid >= 0 and co_tid >= 0:
                        prev_co_tid = co_tid
                    if bl_tid >= 0:
                        prev_bl_tid = bl_tid
                    if co_tid >= 0:
                        prev_co_tid = co_tid

        # Aggregate
        beneficial = sum(1 for s in all_switches if s["label"] == "beneficial")
        neutral = sum(1 for s in all_switches if s["label"] == "neutral")
        harmful = sum(1 for s in all_switches if s["label"] == "harmful")

        # By sequence
        seq_counts = Counter()
        seq_labels = defaultdict(Counter)
        for s in all_switches:
            seq_counts[s["seq_name"]] += 1
            seq_labels[s["seq_name"]][s["label"]] += 1
        seq_rows = []
        for seq in sorted(seq_counts.keys()):
            c = seq_labels[seq]
            seq_rows.append({"seq_name": seq, "beneficial": c["beneficial"],
                             "neutral": c["neutral"], "harmful": c["harmful"],
                             "total": seq_counts[seq]})

        # By gap bucket
        def gap_bucket(g):
            if g == "":
                return "no_reentry"
            g = int(g)
            if g < 30:
                return "10-30"
            elif g < 100:
                return "30-100"
            else:
                return "100+"

        gap_counts = defaultdict(Counter)
        for s in all_switches:
            b = gap_bucket(s["nearest_reentry_gap"])
            gap_counts[b][s["label"]] += 1
        gap_rows = []
        for b in ["10-30", "30-100", "100+", "no_reentry"]:
            if b in gap_counts:
                c = gap_counts[b]
                gap_rows.append({"gap_bucket": b, "beneficial": c["beneficial"],
                                 "neutral": c["neutral"], "harmful": c["harmful"],
                                 "total": sum(c.values())})

        # By time_since_update bucket
        def tsu_bucket(tsu):
            if tsu == "":
                return "no_promotion"
            tsu = int(tsu)
            if tsu <= 5:
                return "1-5"
            elif tsu <= 15:
                return "6-15"
            elif tsu <= 30:
                return "16-30"
            else:
                return "30+"

        tsu_counts = defaultdict(Counter)
        for s in all_switches:
            b = tsu_bucket(s["promotion_time_since_update"])
            tsu_counts[b][s["label"]] += 1
        tsu_rows = []
        for b in ["1-5", "6-15", "16-30", "30+", "no_promotion"]:
            if b in tsu_counts:
                c = tsu_counts[b]
                tsu_rows.append({"tsu_bucket": b, "beneficial": c["beneficial"],
                                 "neutral": c["neutral"], "harmful": c["harmful"],
                                 "total": sum(c.values())})

        summary = {
            "status": "success", "error": "",
            "total_new_switches": len(all_switches),
            "beneficial": beneficial, "neutral": neutral, "harmful": harmful,
        }
        write_rows(summary_csv, list(summary.keys()), [summary])
        write_rows(detail_csv, DETAIL_FIELDS, all_switches)
        write_rows(seq_summary_csv, SEQ_FIELDS, seq_rows)
        write_rows(gap_summary_csv, GAP_FIELDS, gap_rows)
        write_rows(tsu_summary_csv, TSU_FIELDS, tsu_rows)
        append_registry(summary_csv, "success",
                        f"IDSW attribution: {len(all_switches)} switches, "
                        f"beneficial={beneficial}, neutral={neutral}, harmful={harmful}")
        return 0

    except Exception as exc:
        summary = {"status": "failed", "error": str(exc)}
        write_rows(summary_csv, list(summary.keys()), [summary])
        append_registry(summary_csv, "failed", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
