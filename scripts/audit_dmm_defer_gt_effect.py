"""Audit DMM defer events against GT and the BASE (no-DMM) tracker output.

For each defer event (frame T, track_id X, det_global_idx D) this answers:
  - What GT identity does detection D correspond to?
  - What is track X's "canonical" GT identity (majority vote over base output)?
  - Did the base tracker match track X at frame T? To which GT?
  - Did base commit an ID switch at/after frame T?

Classification per defer:
  BLOCKED_CORRECT  : D's GT == X's canonical GT  (DMM paused a good match)
  BLOCKED_WRONG    : D's GT != X's canonical GT  (DMM paused an ID switch)
  BLOCKED_LOST     : base also lost X at frame T  (DMM paused nothing useful)
  NO_GT_FOR_DET    : detection D has no GT match (IoU<0.5)

Output: outputs/dmm_phase2_audit_defer_gt_mot20_01/audit_defer_gt.csv + summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def load_dump(path):
    d = np.load(path, allow_pickle=True)
    dets = d["detections"]  # cols: frame, frame_det_idx, global_det_idx, x1,y1,x2,y2, score,...
    feats = d["features"].astype(np.float32)  # (N, 2048)
    cols = list(d["columns"])
    gi = cols.index("global_det_idx")
    fi = cols.index("frame")
    bi = cols.index("x1")
    det_by_global = {}  # global_det_idx -> (frame, box)
    det_by_frame = defaultdict(list)  # frame -> [(global_det_idx, box, feature)]
    for ri, row in enumerate(dets):
        gidx = int(row[gi])
        frame = int(row[fi])
        box = [float(row[bi + 0]), float(row[bi + 1]), float(row[bi + 2]), float(row[bi + 3])]
        det_by_global[gidx] = (frame, box, feats[ri])
        det_by_frame[frame].append((gidx, box, feats[ri]))
    return det_by_global, det_by_frame


def find_track_prev_detection(track_rows, det_by_frame, upto_frame):
    """Find the detection the track matched at the last frame before upto_frame."""
    best = None
    for frame, box in reversed(track_rows):
        if frame >= upto_frame:
            continue
        cands = det_by_frame.get(frame, [])
        best_iou = 0.0
        best_feat = None
        for _gidx, dbox, dfeat in cands:
            i = iou_xyxy(box, dbox)
            if i > best_iou:
                best_iou = i
                best_feat = dfeat
        if best_feat is not None and best_iou > 0.5:
            return best_feat, best_iou, frame
        best = (best_feat, best_iou, frame)
    return best if best else (None, 0.0, -1)


def load_gt(path):
    """GT: frame, id, left, top, w, h, conf, class, vis. Returns {frame: [(gt_id, xyxy)]}"""
    gt_by_frame = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            gid = int(parts[1])
            left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            x1, y1, x2, y2 = left, top, left + w, top + h
            vis = float(parts[8]) if len(parts) >= 9 else 1.0
            gt_by_frame[frame].append((gid, (x1, y1, x2, y2), vis))
    return gt_by_frame


def load_base_tracks(path):
    """Base tracker output: frame, track_id, left, top, w, h, score, -1, -1, -1.
    Returns {track_id: [(frame, xyxy)]} sorted by frame."""
    tracks = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            left, top, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            x1, y1, x2, y2 = left, top, left + w, top + h
            tracks[tid].append((frame, (x1, y1, x2, y2)))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def match_to_gt(box, gt_list, iou_thresh=0.5):
    """Return (best_gt_id, best_iou) for a box against GT list at same frame."""
    best_id = -1
    best_iou = 0.0
    for gid, gbox, vis in gt_list:
        i = iou_xyxy(box, gbox)
        if i > best_iou:
            best_iou = i
            best_id = gid
    if best_iou < iou_thresh:
        return -1, best_iou
    return best_id, best_iou


def canonical_gt_id(track_rows, gt_by_frame, upto_frame):
    """Majority-vote GT id for a track over rows with frame < upto_frame."""
    counts = Counter()
    for frame, box in track_rows:
        if frame >= upto_frame:
            continue
        gts = gt_by_frame.get(frame, [])
        gid, _iou = match_to_gt(box, gts, iou_thresh=0.5)
        if gid > 0:
            counts[gid] += 1
    if not counts:
        return -1, 0
    gid, n = counts.most_common(1)[0]
    return gid, n


def track_has_idswitch_around(track_rows, gt_by_frame, frame_t, window=10):
    """Check if track's GT mapping changes in [frame_t, frame_t+window]."""
    before_id = -1
    after_ids = Counter()
    for frame, box in track_rows:
        gts = gt_by_frame.get(frame, [])
        gid, _iou = match_to_gt(box, gts, iou_thresh=0.5)
        if frame < frame_t:
            if gid > 0:
                before_id = gid
        elif frame >= frame_t:
            if gid > 0:
                after_ids[gid] += 1
    if before_id < 0:
        return False, before_id, -1
    if not after_ids:
        return False, before_id, -1
    top_after = after_ids.most_common(1)[0][0]
    return (top_after != before_id), before_id, top_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz")
    ap.add_argument("--gt", default="/gemini/code/datasets/MOT20/train/MOT20-01/gt/gt.txt")
    ap.add_argument("--base", default="outputs/dmm_phase1_base_mot20_01_reid/track_results/MOT20-01.txt")
    ap.add_argument("--events", default="outputs/dmm_phase2_v1_mot20_01_reid_m003/dmm_events.csv")
    ap.add_argument("--out-dir", default="outputs/dmm_phase2_audit_defer_gt_mot20_01")
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dump   : {args.dump}")
    det_by_global, det_by_frame = load_dump(args.dump)
    print(f"  dets loaded  : {len(det_by_global)}")

    print(f"loading GT     : {args.gt}")
    gt_by_frame = load_gt(args.gt)
    print(f"  GT frames    : {len(gt_by_frame)}")

    print(f"loading base   : {args.base}")
    base_tracks = load_base_tracks(args.base)
    print(f"  base tracks  : {len(base_tracks)}")

    print(f"loading events : {args.events}")
    events = []
    with open(args.events, "r") as f:
        for row in csv.DictReader(f):
            if row.get("event") != "defer":
                continue
            events.append(row)
    print(f"  defer events: {len(events)}")

    audit_rows = []
    counts = Counter()
    for ev in events:
        frame_t = int(ev["frame"])
        track_id = int(ev["track_id"])
        det_global = int(ev["det_global_idx"])
        cost = float(ev.get("cost", "nan"))
        margin = float(ev.get("row_margin", "nan"))

        if det_global not in det_by_global:
            print(f"  WARN: det_global_idx={det_global} not in dump, skip")
            continue
        _det_frame, det_box, det_feat = det_by_global[det_global]
        if _det_frame != frame_t:
            print(f"  WARN: det frame mismatch {_det_frame} != {frame_t}")

        gts_at_t = gt_by_frame.get(frame_t, [])
        det_gt_id, det_gt_iou = match_to_gt(det_box, gts_at_t, iou_thresh=args.iou_thresh)

        track_rows = base_tracks.get(track_id, [])
        canon_gt, canon_n = canonical_gt_id(track_rows, gt_by_frame, upto_frame=frame_t)

        # ReID distance between deferred detection and track's last-matched detection
        prev_feat, prev_iou, prev_frame = find_track_prev_detection(track_rows, det_by_frame, frame_t)
        reid_dist = -1.0
        if prev_feat is not None and det_feat is not None:
            pf = prev_feat / max(float(np.linalg.norm(prev_feat)), 1e-12)
            df = det_feat / max(float(np.linalg.norm(det_feat)), 1e-12)
            reid_dist = float(1.0 - np.dot(pf, df))

        base_at_t = None
        for fr, bx in track_rows:
            if fr == frame_t:
                base_at_t = (fr, bx)
                break
        base_gt_id = -1
        base_gt_iou = 0.0
        if base_at_t is not None:
            base_gt_id, base_gt_iou = match_to_gt(base_at_t[1], gts_at_t, iou_thresh=args.iou_thresh)

        idsw, idsw_before, idsw_after = track_has_idswitch_around(
            track_rows, gt_by_frame, frame_t, window=10
        )

        if det_gt_id < 0:
            verdict = "NO_GT_FOR_DET"
        elif canon_gt < 0:
            verdict = "NO_CANONICAL"
        elif det_gt_id == canon_gt:
            verdict = "BLOCKED_CORRECT"
        else:
            verdict = "BLOCKED_WRONG"
        counts[verdict] += 1

        audit_rows.append({
            "frame": frame_t,
            "track_id": track_id,
            "det_global_idx": det_global,
            "cost": cost,
            "row_margin": margin,
            "det_box": ",".join(f"{v:.1f}" for v in det_box),
            "det_gt_id": det_gt_id,
            "det_gt_iou": round(det_gt_iou, 3),
            "reid_dist": round(reid_dist, 4),
            "prev_match_frame": prev_frame,
            "prev_match_iou": round(prev_iou, 3),
            "canonical_gt": canon_gt,
            "canonical_n": canon_n,
            "base_present_at_t": int(base_at_t is not None),
            "base_gt_id_at_t": base_gt_id,
            "base_gt_iou_at_t": round(base_gt_iou, 3),
            "idsw_in_base_around_t": int(idsw),
            "idsw_before_gt": idsw_before,
            "idsw_after_gt": idsw_after,
            "verdict": verdict,
        })

    audit_csv = out_dir / "audit_defer_gt.csv"
    fieldnames = list(audit_rows[0].keys()) if audit_rows else ["frame"]
    with open(audit_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(audit_rows)
    print(f"\naudit rows -> {audit_csv}")

    summary = {
        "total_defer_events": len(audit_rows),
        "verdict_counts": dict(counts),
        "verdict_pct": {k: round(100.0 * v / max(1, len(audit_rows)), 1) for k, v in counts.items()},
        "idsw_in_base_around_t": sum(1 for r in audit_rows if r["idsw_in_base_around_t"]),
        "iou_thresh": args.iou_thresh,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    print("\n=== per-event verdicts ===")
    for r in audit_rows:
        print(
            f"  frame={r['frame']:3d} trk={r['track_id']:3d} det_gt={r['det_gt_id']:3d}"
            f" canon={r['canonical_gt']:3d} base_gt@t={r['base_gt_id_at_t']:3d}"
            f" reid={r['reid_dist']:.3f} idsw_base={r['idsw_in_base_around_t']} -> {r['verdict']}"
        )

    # ReID separation analysis
    print("\n=== ReID distance by verdict ===")
    by_verdict = defaultdict(list)
    for r in audit_rows:
        by_verdict[r["verdict"]].append(r["reid_dist"])
    for v, dists in sorted(by_verdict.items()):
        if dists:
            print(f"  {v}: n={len(dists)} min={min(dists):.3f} max={max(dists):.3f} mean={sum(dists)/len(dists):.3f}")
    correct_reid = by_verdict.get("BLOCKED_CORRECT", [])
    wrong_reid = by_verdict.get("BLOCKED_WRONG", [])
    if correct_reid and wrong_reid:
        print(f"\n  BLOCKED_CORRECT mean reid: {sum(correct_reid)/len(correct_reid):.3f}")
        print(f"  BLOCKED_WRONG   mean reid: {sum(wrong_reid)/len(wrong_reid):.3f}")
        # Try threshold sweep
        print("\n  threshold sweep (filter defers where reid_dist < thresh):")
        for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            kept = sum(1 for d in wrong_reid if d >= thresh)
            dropped = sum(1 for d in correct_reid if d < thresh)
            total_kept = sum(1 for d in wrong_reid + correct_reid if d >= thresh)
            print(f"    reid>={thresh:.2f}: keeps {kept}/{len(wrong_reid)} WRONG, drops {dropped}/{len(correct_reid)} CORRECT, total_defers={total_kept}")

    # Verdict-conditional implication
    print("\n=== implication ===")
    blocked_wrong = counts.get("BLOCKED_WRONG", 0)
    blocked_correct = counts.get("BLOCKED_CORRECT", 0)
    if blocked_wrong >= blocked_correct:
        print(f"  {blocked_wrong}/{len(audit_rows)} defers blocked ID-switches (WRONG matches).")
        print("  => trigger is well-targeted. v2 tracklet recovery is the correct next step.")
    else:
        print(f"  {blocked_correct}/{len(audit_rows)} defers blocked CORRECT matches.")
        print("  => trigger too aggressive; v2 recovery alone will not help. Tighten trigger first.")


if __name__ == "__main__":
    main()
