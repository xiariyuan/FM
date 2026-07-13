#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external/BoT-SORT-main"))
from fast_reid.fast_reid_interfece import FastReIDInterface  # noqa: E402


def ai(v, d=0):
    try:
        return int(float(v))
    except Exception:
        return d


def af(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return d


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def read_track(path: Path):
    rows = []
    by_frame = defaultdict(list)
    by_frame_tid = defaultdict(list)
    by_tid = defaultdict(list)
    img_width = 1.0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fr = ai(p[0], -1)
            tid = ai(p[1], -1)
            x = af(p[2])
            y = af(p[3])
            w = af(p[4])
            h = af(p[5])
            score = af(p[6], 1.0) if len(p) > 6 else 1.0
            if fr < 0 or tid < 0 or w <= 0 or h <= 0:
                continue
            img_width = max(img_width, x + w)
            r = {
                "idx": len(rows),
                "frame": fr,
                "track_id": tid,
                "box": np.array([x, y, x + w, y + h], dtype=np.float32),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": x + w / 2.0,
                "bottom_y": y + h,
                "height": h,
                "score": score,
                "parts": p,
            }
            rows.append(r)
            by_frame[fr].append(r)
            by_frame_tid[(fr, tid)].append(r)
            by_tid[tid].append(r)
    return rows, by_frame, by_frame_tid, by_tid, img_width


def read_gt(path: Path):
    by = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fr = ai(p[0], -1)
            gid = ai(p[1], -1)
            x = af(p[2])
            y = af(p[3])
            w = af(p[4])
            h = af(p[5])
            mark = ai(p[6], 1) if len(p) > 6 else 1
            cls = ai(p[7], 1) if len(p) > 7 else 1
            if fr < 0 or gid < 0 or w <= 0 or h <= 0 or mark <= 0 or cls != 1:
                continue
            by[fr].append({"frame": fr, "gt_id": gid, "box": np.array([x, y, x + w, y + h], dtype=np.float32)})
    return by


def pair_iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.stack([x["box"] for x in a])
    B = np.stack([x["box"] for x in b])
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]), 1e-6, None)
    bb = np.clip((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]), 1e-6, None)
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-6, None)


def match_rows(rows_by, gt_by, thr):
    row_gt = {}
    row_iou = {}
    for fr in sorted(set(rows_by) | set(gt_by)):
        rr = rows_by.get(fr, [])
        gg = gt_by.get(fr, [])
        if not rr or not gg:
            continue
        I = pair_iou(rr, gg)
        ri, ci = linear_sum_assignment(1 - I)
        for r, c in zip(ri, ci):
            val = float(I[r, c])
            if val >= thr:
                row_gt[rr[r]["idx"]] = int(gg[c]["gt_id"])
                row_iou[rr[r]["idx"]] = val
    return row_gt, row_iou


def read_tunnel(path: Path, tunnel_id: int):
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tid = ai(r.get("tunnel_id"), -1)
            if tid != tunnel_id:
                continue
            tracks = [ai(x, -1) for x in str(r.get("tracks", "")).split("|") if x != ""]
            tracks = [x for x in tracks if x >= 0]
            return {
                "tunnel_id": tid,
                "start": ai(r.get("start")),
                "end": ai(r.get("end")),
                "duration": ai(r.get("duration")),
                "tracks": set(tracks),
            }
    raise FileNotFoundError(f"tunnel_id={tunnel_id} not found in {path}")


def split_fragments(rows, max_gap=1):
    rows = sorted(rows, key=lambda r: r["frame"])
    if not rows:
        return []
    out = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r["frame"] - cur[-1]["frame"] <= max_gap:
            cur.append(r)
        else:
            out.append(cur)
            cur = [r]
    out.append(cur)
    return out


def choose_rows(rows, max_crops):
    rows = sorted(rows, key=lambda r: (-r.get("score", 1.0), r["frame"]))[:max_crops]
    return sorted(rows, key=lambda r: r["frame"])


def img_path(img_dir, frame):
    return Path(img_dir) / f"{frame:06d}.jpg"


def extract_features(groups, img_dir, encoder, max_crops):
    selected = {k: choose_rows(v, max_crops) for k, v in groups.items() if v}
    by_frame = defaultdict(list)
    for k, rr in selected.items():
        for r in rr:
            by_frame[r["frame"]].append((k, r))
    feats = defaultdict(list)
    for fr, items in sorted(by_frame.items()):
        img = cv2.imread(str(img_path(img_dir, fr)))
        if img is None:
            continue
        dets = np.stack([r["box"] for _, r in items]).astype(np.float32)
        out = encoder.inference(img, dets)
        for (k, r), feat in zip(items, out):
            feats[k].append(feat.astype(np.float32))
    proto = {}
    for k, fs in feats.items():
        if not fs:
            continue
        v = np.mean(np.stack(fs, axis=0), axis=0)
        n = max(float(np.linalg.norm(v)), 1e-12)
        proto[k] = v / n
    return proto, {k: len(v) for k, v in selected.items()}


def major_gt(rows, row_gt):
    c = Counter()
    for r in rows:
        gid = row_gt.get(r["idx"], -1)
        if gid >= 0:
            c[gid] += 1
    if not c:
        return -1, 0, 0.0
    gid, n = c.most_common(1)[0]
    return gid, n, n / max(1, sum(c.values()))


def frag_meta(key, tid, local_id, rows, row_gt, crop_counts):
    rows = sorted(rows, key=lambda r: r["frame"])
    gid, gt_n, purity = major_gt(rows, row_gt)
    return {
        "fragment_key": key,
        "track_id": tid,
        "local_fragment_id": local_id,
        "frame_start": rows[0]["frame"],
        "frame_end": rows[-1]["frame"],
        "rows": len(rows),
        "crops": crop_counts.get(key, 0),
        "first_cx": rows[0]["cx"],
        "first_bottom_y": rows[0]["bottom_y"],
        "first_height": rows[0]["height"],
        "last_cx": rows[-1]["cx"],
        "last_bottom_y": rows[-1]["bottom_y"],
        "last_height": rows[-1]["height"],
        "mean_cx": float(np.mean([r["cx"] for r in rows])),
        "mean_bottom_y": float(np.mean([r["bottom_y"] for r in rows])),
        "mean_height": float(np.mean([r["height"] for r in rows])),
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "major_gt": gid,
        "gt_count": gt_n,
        "gt_purity": purity,
        "row_indices": [r["idx"] for r in rows],
    }


def endpoint_from_rows(rows, which):
    rows = sorted(rows, key=lambda r: r["frame"])
    r = rows[0] if which == "first" else rows[-1]
    return {"frame": r["frame"], "cx": r["cx"], "bottom_y": r["bottom_y"], "height": r["height"]}


def endpoint_from_meta(m, which):
    if which == "first":
        return {"frame": m["frame_start"], "cx": m["first_cx"], "bottom_y": m["first_bottom_y"], "height": m["first_height"]}
    return {"frame": m["frame_end"], "cx": m["last_cx"], "bottom_y": m["last_bottom_y"], "height": m["last_height"]}


def predict_between(a, b, frame):
    span = max(1, b["frame"] - a["frame"])
    alpha = (frame - a["frame"]) / span
    return {
        "cx": a["cx"] + (b["cx"] - a["cx"]) * alpha,
        "bottom_y": a["bottom_y"] + (b["bottom_y"] - a["bottom_y"]) * alpha,
        "height": a["height"] + (b["height"] - a["height"]) * alpha,
    }


def norm_dist(row_like, pred):
    h = max(pred["height"], row_like.get("height", pred["height"]), 1.0)
    return math.hypot((row_like["cx"] - pred["cx"]) / h, (row_like["bottom_y"] - pred["bottom_y"]) / h)


def bridge_score(prev_ep, next_ep, m):
    first = {"cx": m["first_cx"], "bottom_y": m["first_bottom_y"], "height": m["first_height"]}
    last = {"cx": m["last_cx"], "bottom_y": m["last_bottom_y"], "height": m["last_height"]}
    pred_first = predict_between(prev_ep, next_ep, m["frame_start"])
    pred_last = predict_between(prev_ep, next_ep, m["frame_end"])
    d1 = norm_dist(first, pred_first)
    d2 = norm_dist(last, pred_last)
    h1 = first["height"] / max(pred_first["height"], 1e-6)
    h2 = last["height"] / max(pred_last["height"], 1e-6)
    height_pen = abs(math.log(max(h1, 1e-6))) + abs(math.log(max(h2, 1e-6)))
    return max(d1, d2) + 0.25 * height_pen, d1, d2, h1, h2


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="A39_03e0 path bridge rewrite for a single high-confidence anchor without GT-based selection.")
    ap.add_argument("--track-file", required=True)
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--tunnels-csv", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--fast-reid-config", required=True)
    ap.add_argument("--fast-reid-weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tunnel-id", type=int, required=True)
    ap.add_argument("--pre-anchor", type=int, required=True)
    ap.add_argument("--post-anchor", type=int, required=True)
    ap.add_argument("--target-id", type=int, default=None)
    ap.add_argument("--anchor-gt", type=int, default=None, help="diagnostic only")
    ap.add_argument("--pre-window", type=int, default=10)
    ap.add_argument("--post-window", type=int, default=10)
    ap.add_argument("--exit-window", type=int, default=10)
    ap.add_argument("--max-crops-per-fragment", type=int, default=8)
    ap.add_argument("--high-sim", type=float, default=0.65)
    ap.add_argument("--bridge-min-sim", type=float, default=0.35)
    ap.add_argument("--bridge-max-score", type=float, default=0.22)
    ap.add_argument("--bridge-max-gap", type=int, default=25)
    ap.add_argument("--gap-max-dist", type=float, default=0.12)
    ap.add_argument("--gap-height-min", type=float, default=0.80)
    ap.add_argument("--gap-height-max", type=float, default=1.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target_id = args.target_id if args.target_id is not None else args.pre_anchor

    rows, rows_by, by_frame_tid, by_tid, img_width = read_track(Path(args.track_file))
    gt_by = read_gt(Path(args.gt_file))
    row_gt, row_iou = match_rows(rows_by, gt_by, args.iou_thr)
    tunnel = read_tunnel(Path(args.tunnels_csv), args.tunnel_id)
    start, end = tunnel["start"], tunnel["end"]
    f0, f1 = start, end + args.exit_window

    pre_anchor_rows = [r for r in by_tid.get(args.pre_anchor, []) if start - args.pre_window <= r["frame"] < start]
    post_anchor_rows = [r for r in by_tid.get(args.post_anchor, []) if end < r["frame"] <= end + args.post_window]
    if not pre_anchor_rows or not post_anchor_rows:
        raise RuntimeError("missing pre or post anchor clean rows")

    groups = {"anchor_pre": pre_anchor_rows, "anchor_post": post_anchor_rows}
    fragments = []
    for tid in sorted(tunnel["tracks"]):
        frag_rows_all = [r for r in by_tid.get(tid, []) if f0 <= r["frame"] <= f1]
        for local_id, frag_rows in enumerate(split_fragments(frag_rows_all, max_gap=1)):
            if not frag_rows:
                continue
            key = f"frag_{tid}_{local_id}"
            fragments.append((key, tid, local_id, frag_rows))
            groups[key] = frag_rows

    encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device, batch_size=32)
    proto, crop_counts = extract_features(groups, args.img_dir, encoder, args.max_crops_per_fragment)
    if "anchor_pre" not in proto or "anchor_post" not in proto:
        raise RuntimeError("missing anchor features")
    anchor_proto = proto["anchor_pre"] + proto["anchor_post"]
    anchor_proto = anchor_proto / max(float(np.linalg.norm(anchor_proto)), 1e-12)

    metas = []
    meta_by_key = {}
    for key, tid, local_id, rr in fragments:
        if key not in proto:
            continue
        m = frag_meta(key, tid, local_id, rr, row_gt, crop_counts)
        m["target_id"] = target_id
        m["sim_to_pre"] = float(np.dot(proto[key], proto["anchor_pre"]))
        m["sim_to_post"] = float(np.dot(proto[key], proto["anchor_post"]))
        m["sim_to_anchor"] = float(np.dot(proto[key], anchor_proto))
        m["collision_rows_if_rewrite"] = sum(1 for r in rr if tid != target_id and by_frame_tid.get((r["frame"], target_id)))
        m["collision_ratio_if_rewrite"] = safe_div(m["collision_rows_if_rewrite"], len(rr))
        m["gt_same_as_anchor"] = int(args.anchor_gt is not None and m["major_gt"] == args.anchor_gt and m["gt_purity"] >= 0.6)
        m["selected_stage"] = ""
        m["bridge_score"] = ""
        metas.append(m)
        meta_by_key[key] = m

    # Stage 1: high-ReID fragments. This is deployable and should catch the clean exit/core fragment.
    selected = []
    for m in metas:
        if m["track_id"] == target_id:
            continue
        if m["collision_ratio_if_rewrite"] > 0:
            continue
        if max(m["sim_to_anchor"], m["sim_to_post"]) >= args.high_sim and m["rows"] >= 2:
            m["selected_stage"] = "high_reid"
            selected.append(m)

    selected = sorted(selected, key=lambda x: (x["frame_start"], x["frame_end"]))
    target_frags = [m for m in metas if m["track_id"] == target_id and m["frame_start"] <= (selected[0]["frame_start"] if selected else f1)]
    if target_frags:
        prev_ep = endpoint_from_meta(sorted(target_frags, key=lambda x: x["frame_end"])[-1], "last")
    else:
        prev_ep = endpoint_from_rows(pre_anchor_rows, "last")

    bridge_candidates = []
    # Stage 2: low-ReID temporal/geometric bridge into the first high-ReID fragment.
    if selected:
        next_m = selected[0]
        next_ep = endpoint_from_meta(next_m, "first")
        for m in metas:
            if m["track_id"] == target_id or m in selected:
                continue
            if m["collision_ratio_if_rewrite"] > 0:
                continue
            if not (prev_ep["frame"] < m["frame_start"] <= m["frame_end"] < next_ep["frame"]):
                continue
            if m["frame_start"] - prev_ep["frame"] > args.bridge_max_gap:
                continue
            if next_ep["frame"] - m["frame_end"] > args.bridge_max_gap:
                continue
            score, d1, d2, h1, h2 = bridge_score(prev_ep, next_ep, m)
            mrow = {
                "fragment_key": m["fragment_key"],
                "track_id": m["track_id"],
                "frame_start": m["frame_start"],
                "frame_end": m["frame_end"],
                "rows": m["rows"],
                "sim_to_anchor": m["sim_to_anchor"],
                "sim_to_pre": m["sim_to_pre"],
                "sim_to_post": m["sim_to_post"],
                "bridge_score": score,
                "start_dist": d1,
                "end_dist": d2,
                "height_ratio_start": h1,
                "height_ratio_end": h2,
                "accepted": 0,
                "major_gt": m["major_gt"],
                "gt_purity": m["gt_purity"],
                "gt_same_as_anchor": m["gt_same_as_anchor"],
            }
            if m["sim_to_anchor"] >= args.bridge_min_sim and score <= args.bridge_max_score:
                mrow["accepted"] = 1
            bridge_candidates.append(mrow)
        accepted_bridge = [r for r in bridge_candidates if r["accepted"]]
        if accepted_bridge:
            # Keep the most temporally advanced valid bridge with best score; for tunnel 12 this should be frag_67_0.
            best = sorted(accepted_bridge, key=lambda r: (r["bridge_score"], -r["frame_end"], -r["rows"]))[0]
            bm = meta_by_key[best["fragment_key"]]
            bm["selected_stage"] = "bridge_fragment"
            bm["bridge_score"] = best["bridge_score"]
            selected = sorted([bm] + selected, key=lambda x: (x["frame_start"], x["frame_end"]))

    # Stage 3: row-level gap bridge between adjacent selected fragments.
    gap_rows = []
    selected_for_gaps = sorted(selected, key=lambda x: (x["frame_start"], x["frame_end"]))
    for a, b in zip(selected_for_gaps, selected_for_gaps[1:]):
        a_ep = endpoint_from_meta(a, "last")
        b_ep = endpoint_from_meta(b, "first")
        if b_ep["frame"] - a_ep["frame"] <= 1:
            continue
        for fr in range(a_ep["frame"] + 1, b_ep["frame"]):
            pred = predict_between(a_ep, b_ep, fr)
            cand = []
            for r in rows_by.get(fr, []):
                if r["track_id"] not in tunnel["tracks"] or r["track_id"] == target_id:
                    continue
                if by_frame_tid.get((fr, target_id)):
                    continue
                dist = norm_dist(r, pred)
                height_ratio = r["height"] / max(pred["height"], 1e-6)
                if dist <= args.gap_max_dist and args.gap_height_min <= height_ratio <= args.gap_height_max:
                    cand.append((dist, abs(1.0 - height_ratio), r, height_ratio))
            if cand:
                dist, hpen, r, height_ratio = sorted(cand, key=lambda x: (x[0], x[1], -x[2]["score"]))[0]
                gap_rows.append({
                    "frame": fr,
                    "track_id": r["track_id"],
                    "idx": r["idx"],
                    "target_id": target_id,
                    "dist": dist,
                    "height_ratio": height_ratio,
                    "score": r["score"],
                    "gt_id": row_gt.get(r["idx"], -1),
                    "gt_same_as_anchor": int(args.anchor_gt is not None and row_gt.get(r["idx"], -1) == args.anchor_gt),
                    "bridge_from": a["fragment_key"],
                    "bridge_to": b["fragment_key"],
                })

    # Rewrite selected fragments + gap rows with conservative collision skip.
    selected_indices = []
    for m in selected_for_gaps:
        if m["track_id"] == target_id:
            continue
        selected_indices.extend(m["row_indices"])
    selected_indices.extend([r["idx"] for r in gap_rows])
    selected_indices = sorted(set(selected_indices), key=lambda idx: rows[idx]["frame"])

    final_id = {r["idx"]: r["track_id"] for r in rows}
    cur = defaultdict(set)
    for r in rows:
        cur[r["frame"]].add(r["track_id"])
    applied = 0
    skipped = 0
    for idx in selected_indices:
        r = rows[idx]
        if r["track_id"] == target_id:
            continue
        if target_id in cur[r["frame"]]:
            skipped += 1
            continue
        cur[r["frame"]].discard(r["track_id"])
        cur[r["frame"]].add(target_id)
        final_id[idx] = target_id
        applied += 1

    selected_frag_rows = []
    for m in sorted(selected_for_gaps, key=lambda x: (x["frame_start"], x["frame_end"])):
        row = {k: v for k, v in m.items() if k != "row_indices"}
        selected_frag_rows.append(row)
    all_frag_rows = [{k: v for k, v in m.items() if k != "row_indices"} for m in metas]

    # Write result file.
    track_out = out / "track_results" / "MOT20-02.txt"
    track_out.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["parts"][1] = str(final_id[r["idx"]])
    with track_out.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: (ai(r["parts"][0]), ai(r["parts"][1]), af(r["parts"][2]), af(r["parts"][3]))):
            f.write(",".join(r["parts"]) + "\n")

    def write_any_csv(path, records):
        if records:
            fields = list(records[0].keys())
        else:
            fields = ["empty"]
        write_csv(path, fields, records)

    write_any_csv(out / "all_fragments.csv", all_frag_rows)
    write_any_csv(out / "selected_fragments.csv", selected_frag_rows)
    write_any_csv(out / "bridge_candidates.csv", bridge_candidates)
    write_any_csv(out / "gap_rows.csv", gap_rows)

    correct_fragments = [m for m in selected_for_gaps if m.get("gt_same_as_anchor")]
    wrong_fragments = [m for m in selected_for_gaps if not m.get("gt_same_as_anchor") and m["track_id"] != target_id]
    correct_gap = [r for r in gap_rows if r["gt_same_as_anchor"]]
    wrong_gap = [r for r in gap_rows if not r["gt_same_as_anchor"]]
    summary = {
        "tunnel_id": args.tunnel_id,
        "pre_anchor": args.pre_anchor,
        "post_anchor": args.post_anchor,
        "target_id": target_id,
        "anchor_gt_diag": args.anchor_gt,
        "selected_fragments": [m["fragment_key"] for m in selected_for_gaps],
        "selected_fragment_count": len(selected_for_gaps),
        "correct_fragment_count_diag": len(correct_fragments),
        "wrong_fragment_count_diag": len(wrong_fragments),
        "gap_row_count": len(gap_rows),
        "correct_gap_rows_diag": len(correct_gap),
        "wrong_gap_rows_diag": len(wrong_gap),
        "planned_rows": len(selected_indices),
        "applied_rows": applied,
        "skipped_collision_rows": skipped,
        "high_sim": args.high_sim,
        "bridge_min_sim": args.bridge_min_sim,
        "bridge_max_score": args.bridge_max_score,
        "gap_max_dist": args.gap_max_dist,
        "gt_diag_selected_rows_same_anchor": sum(1 for idx in selected_indices if args.anchor_gt is not None and row_gt.get(idx, -1) == args.anchor_gt),
        "gt_diag_selected_rows_wrong_or_unknown": sum(1 for idx in selected_indices if args.anchor_gt is not None and row_gt.get(idx, -1) != args.anchor_gt),
    }
    (out / "rewrite_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# A39_03e0 path bridge rewrite", "", "```json", json.dumps(summary, indent=2, sort_keys=True), "```"]
    (out / "rewrite_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
