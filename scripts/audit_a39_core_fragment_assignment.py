#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    frags = []
    cur = [rows[0]]
    for r in rows[1:]:
        if r["frame"] - cur[-1]["frame"] <= max_gap:
            cur.append(r)
        else:
            frags.append(cur)
            cur = [r]
    frags.append(cur)
    return frags


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


def endpoint_stats(rows):
    rows = sorted(rows, key=lambda r: r["frame"])
    if not rows:
        return {}
    return {
        "first_frame": rows[0]["frame"],
        "last_frame": rows[-1]["frame"],
        "first_cx": rows[0]["cx"],
        "last_cx": rows[-1]["cx"],
        "first_bottom_y": rows[0]["bottom_y"],
        "last_bottom_y": rows[-1]["bottom_y"],
        "first_height": rows[0]["height"],
        "last_height": rows[-1]["height"],
        "mean_cx": float(np.mean([r["cx"] for r in rows])),
        "mean_bottom_y": float(np.mean([r["bottom_y"] for r in rows])),
        "mean_height": float(np.mean([r["height"] for r in rows])),
        "mean_score": float(np.mean([r["score"] for r in rows])),
    }


POLICIES = {
    "strict_core": {"sim_anchor": 0.65, "sim_side": 0.60, "rows": 2, "collision_ratio": 0.50, "purity": 0.80},
    "normal_core": {"sim_anchor": 0.55, "sim_side": 0.50, "rows": 2, "collision_ratio": 0.80, "purity": 0.80},
    "research_core": {"sim_anchor": 0.45, "sim_side": -1.00, "rows": 1, "collision_ratio": 1.01, "purity": 0.60},
    "ultra_precise_core": {"sim_anchor": 0.70, "sim_side": 0.65, "rows": 2, "collision_ratio": 0.50, "purity": 0.80},
}


def policy_accept(row, pol):
    return (
        row["rows"] >= pol["rows"]
        and row["sim_to_anchor"] >= pol["sim_anchor"]
        and max(row["sim_to_pre"], row["sim_to_post"]) >= pol["sim_side"]
        and row["collision_ratio_if_rewrite"] <= pol["collision_ratio"]
        and row["track_id"] != row["target_id"]
    )


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="A39_03c0: core fragment assignment feasibility for one high-confidence tunnel anchor.")
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
    ap.add_argument("--anchor-gt", type=int, default=None)
    ap.add_argument("--pre-window", type=int, default=10)
    ap.add_argument("--post-window", type=int, default=10)
    ap.add_argument("--exit-window", type=int, default=10)
    ap.add_argument("--max-crops-per-fragment", type=int, default=8)
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

    pre_rows = [r for r in by_tid.get(args.pre_anchor, []) if start - args.pre_window <= r["frame"] < start]
    post_rows = [r for r in by_tid.get(args.post_anchor, []) if end < r["frame"] <= end + args.post_window]
    pre_gt, pre_gt_count, pre_purity = major_gt(pre_rows, row_gt)
    post_gt, post_gt_count, post_purity = major_gt(post_rows, row_gt)
    anchor_gt = args.anchor_gt if args.anchor_gt is not None else (pre_gt if pre_gt >= 0 and pre_gt == post_gt else -1)

    groups = {"anchor_pre": pre_rows, "anchor_post": post_rows}
    fragments = []
    for tid in sorted(tunnel["tracks"]):
        frag_rows_all = [r for r in by_tid.get(tid, []) if f0 <= r["frame"] <= f1]
        for local_id, frag_rows in enumerate(split_fragments(frag_rows_all, max_gap=1)):
            if not frag_rows:
                continue
            key = f"frag_{tid}_{local_id}"
            groups[key] = frag_rows
            fragments.append((key, tid, local_id, frag_rows))

    encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device, batch_size=32)
    proto, crop_counts = extract_features(groups, args.img_dir, encoder, args.max_crops_per_fragment)
    if "anchor_pre" not in proto or "anchor_post" not in proto:
        raise RuntimeError("missing anchor pre/post features")
    anchor_proto = proto["anchor_pre"] + proto["anchor_post"]
    anchor_proto = anchor_proto / max(float(np.linalg.norm(anchor_proto)), 1e-12)

    frag_rows_out = []
    core_row_rows = []
    accepted_rows_by_policy = {name: [] for name in POLICIES}
    for key, tid, local_id, rr in fragments:
        if key not in proto:
            continue
        st = endpoint_stats(rr)
        gid, gt_n, purity = major_gt(rr, row_gt)
        sim_pre = float(np.dot(proto[key], proto["anchor_pre"]))
        sim_post = float(np.dot(proto[key], proto["anchor_post"]))
        sim_anchor = float(np.dot(proto[key], anchor_proto))
        collision = sum(1 for r in rr if r["track_id"] != target_id and by_frame_tid.get((r["frame"], target_id)))
        gt_same = int(anchor_gt >= 0 and gid == anchor_gt and purity >= 0.6)
        row = {
            "tunnel_id": args.tunnel_id,
            "fragment_key": key,
            "track_id": tid,
            "local_fragment_id": local_id,
            "target_id": target_id,
            "frame_start": st["first_frame"],
            "frame_end": st["last_frame"],
            "rows": len(rr),
            "crops": crop_counts.get(key, 0),
            "sim_to_pre": sim_pre,
            "sim_to_post": sim_post,
            "sim_to_anchor": sim_anchor,
            "mean_cx": st["mean_cx"],
            "mean_bottom_y": st["mean_bottom_y"],
            "mean_height": st["mean_height"],
            "mean_score": st["mean_score"],
            "gap_from_pre": st["first_frame"] - (max([r["frame"] for r in pre_rows]) if pre_rows else start),
            "gap_to_post": (min([r["frame"] for r in post_rows]) if post_rows else end) - st["last_frame"],
            "center_delta_pre_norm": abs(st["first_cx"] - (pre_rows[-1]["cx"] if pre_rows else st["first_cx"])) / max(img_width, 1.0),
            "center_delta_post_norm": abs((post_rows[0]["cx"] if post_rows else st["last_cx"]) - st["last_cx"]) / max(img_width, 1.0),
            "height_ratio_to_pre": st["first_height"] / max((pre_rows[-1]["height"] if pre_rows else st["first_height"]), 1e-6),
            "height_ratio_to_post": (post_rows[0]["height"] if post_rows else st["last_height"]) / max(st["last_height"], 1e-6),
            "collision_rows_if_rewrite": collision,
            "collision_ratio_if_rewrite": safe_div(collision, len(rr)),
            "major_gt": gid,
            "gt_count": gt_n,
            "gt_purity": purity,
            "anchor_gt": anchor_gt,
            "gt_same_as_anchor": gt_same,
        }
        for name, pol in POLICIES.items():
            acc = policy_accept(row, pol)
            row[f"accept_{name}"] = int(acc)
            if acc:
                accepted_rows_by_policy[name].append(row)
        frag_rows_out.append(row)
        for r in rr:
            core_row_rows.append({
                "fragment_key": key,
                "frame": r["frame"],
                "track_id": tid,
                "target_id": target_id,
                "gt_id": row_gt.get(r["idx"], -1),
                "gt_same_as_anchor": int(row_gt.get(r["idx"], -1) == anchor_gt if anchor_gt >= 0 else 0),
                "iou": row_iou.get(r["idx"], 0.0),
            })

    summary_rows = []
    for name, acc in accepted_rows_by_policy.items():
        pol = POLICIES[name]
        eval_known = [r for r in acc if r["gt_purity"] >= pol["purity"] and r["major_gt"] >= 0]
        correct = [r for r in eval_known if r["gt_same_as_anchor"]]
        wrong = [r for r in eval_known if not r["gt_same_as_anchor"]]
        accepted_row_count = sum(r["rows"] for r in acc)
        correct_row_count = sum(r["rows"] for r in correct)
        wrong_row_count = sum(r["rows"] for r in wrong)
        summary_rows.append({
            "policy": name,
            "accepted_fragments": len(acc),
            "eval_fragments": len(eval_known),
            "correct_fragments": len(correct),
            "wrong_fragments": len(wrong),
            "fragment_precision": safe_div(len(correct), len(eval_known)),
            "accepted_rows": accepted_row_count,
            "correct_rows": correct_row_count,
            "wrong_rows": wrong_row_count,
            "row_precision": safe_div(correct_row_count, correct_row_count + wrong_row_count),
            "collision_rows": sum(r["collision_rows_if_rewrite"] for r in acc),
            "collision_ratio": safe_div(sum(r["collision_rows_if_rewrite"] for r in acc), accepted_row_count),
            "sim_anchor_min": pol["sim_anchor"],
            "sim_side_min": pol["sim_side"],
            "collision_ratio_max": pol["collision_ratio"],
        })

    fields = list(frag_rows_out[0].keys()) if frag_rows_out else ["tunnel_id"]
    write_csv(out / "core_fragment_candidates.csv", fields, frag_rows_out)
    write_csv(out / "core_row_candidates.csv", list(core_row_rows[0].keys()) if core_row_rows else ["fragment_key"], core_row_rows)
    write_csv(out / "policy_summary.csv", list(summary_rows[0].keys()), summary_rows)
    anchor_payload = {
        "tunnel": {k: (sorted(v) if isinstance(v, set) else v) for k, v in tunnel.items()},
        "pre_anchor": args.pre_anchor,
        "post_anchor": args.post_anchor,
        "target_id": target_id,
        "pre_rows": len(pre_rows),
        "post_rows": len(post_rows),
        "pre_gt": pre_gt,
        "pre_gt_count": pre_gt_count,
        "pre_gt_purity": pre_purity,
        "post_gt": post_gt,
        "post_gt_count": post_gt_count,
        "post_gt_purity": post_purity,
        "anchor_gt": anchor_gt,
        "fragment_count": len(frag_rows_out),
        "policies": POLICIES,
        "summary": summary_rows,
    }
    (out / "anchor_summary.json").write_text(json.dumps(anchor_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_03c0 core fragment assignment feasibility",
        "",
        f"tunnel_id: `{args.tunnel_id}`",
        f"anchor: `{args.pre_anchor} -> {args.post_anchor}`",
        f"anchor_gt_diag: `{anchor_gt}`",
        "",
        "| policy | accepted_fragments | eval_fragments | frag_precision | accepted_rows | row_precision | collision_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(f"| {r['policy']} | {r['accepted_fragments']} | {r['eval_fragments']} | {r['fragment_precision']:.4f} | {r['accepted_rows']} | {r['row_precision']:.4f} | {r['collision_ratio']:.4f} |")
    (out / "policy_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
