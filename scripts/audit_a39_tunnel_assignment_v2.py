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
    max_x2 = 1.0
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
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            max_x2 = max(max_x2, float(x + w))
            r = {
                "idx": len(rows),
                "frame": fr,
                "track_id": tid,
                "box": box,
                "score": score,
                "cx": x + w / 2.0,
                "bottom_y": y + h,
                "height": h,
                "parts": p,
            }
            rows.append(r)
            by_frame[fr].append(r)
            by_frame_tid[(fr, tid)].append(r)
            by_tid[tid].append(r)
    return rows, by_frame, by_frame_tid, by_tid, max_x2


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


def pair_iou(rr, gg):
    if not rr or not gg:
        return np.zeros((len(rr), len(gg)), dtype=np.float32)
    A = np.stack([x["box"] for x in rr])
    B = np.stack([x["box"] for x in gg])
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip((A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]), 1e-6, None)
    bb = np.clip((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]), 1e-6, None)
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-6, None)


def match_row_gt(rows_by, gt_by, thr):
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


def read_tunnels(path: Path):
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tracks = [ai(x, -1) for x in str(r.get("tracks", "")).split("|") if x != ""]
            tracks = [x for x in tracks if x >= 0]
            out.append(
                {
                    "tunnel_id": ai(r.get("tunnel_id"), len(out)),
                    "start": ai(r.get("start")),
                    "end": ai(r.get("end")),
                    "duration": ai(r.get("duration")),
                    "tracks": set(tracks),
                }
            )
    return out


def img_path(img_dir, frame):
    return Path(img_dir) / f"{frame:06d}.jpg"


def choose_rows(rows, max_per_track):
    rows = sorted(rows, key=lambda r: (-r.get("score", 1.0), r["frame"]))[:max_per_track]
    return sorted(rows, key=lambda r: r["frame"])


def extract_group_features(groups, img_dir, encoder, max_per_track):
    selected = {k: choose_rows(v, max_per_track) for k, v in groups.items() if v}
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


def interval_count(by_tid, tid, lo, hi):
    if lo > hi:
        return 0
    return sum(1 for r in by_tid.get(tid, []) if lo <= r["frame"] <= hi)


def summarize_group(k, rr, row_gt, crop_counts, by_tid, tunnel, side, pre_window, post_window):
    gt_count = Counter()
    for r in rr:
        gid = row_gt.get(r["idx"], -1)
        if gid >= 0:
            gt_count[gid] += 1
    major_gt = -1
    gt_n = 0
    purity = 0.0
    if gt_count:
        major_gt, gt_n = gt_count.most_common(1)[0]
        purity = gt_n / max(1, sum(gt_count.values()))
    start, end = tunnel["start"], tunnel["end"]
    tid = k[2]
    pre_rows = interval_count(by_tid, tid, start - pre_window, start - 1)
    core_rows = interval_count(by_tid, tid, start, end)
    post_rows = interval_count(by_tid, tid, end + 1, end + post_window)
    return {
        "tunnel_id": k[0],
        "side": side,
        "track_id": tid,
        "rows": len(rr),
        "crops": crop_counts.get(k, 0),
        "major_gt": major_gt,
        "gt_count": gt_n,
        "gt_purity": purity,
        "first_frame": min(r["frame"] for r in rr),
        "last_frame": max(r["frame"] for r in rr),
        "mean_cx": float(np.mean([r["cx"] for r in rr])),
        "mean_bottom_y": float(np.mean([r["bottom_y"] for r in rr])),
        "mean_height": float(np.mean([r["height"] for r in rr])),
        "pre_rows": pre_rows,
        "core_rows": core_rows,
        "post_rows": post_rows,
        "has_feat": 1,
    }


POLICIES = {
    "strict_v2": {
        "sim": 0.68,
        "row_margin": 0.08,
        "col_margin": 0.05,
        "row_min": 2,
        "col_min": 2,
        "bottom_rank_delta": 1,
        "height_rank_delta": 1,
        "height_ratio_min": 0.75,
        "height_ratio_max": 1.35,
        "pre_before_min": 3,
        "post_after_min": 3,
        "pre_after_max": 1,
        "post_before_max": 1,
        "purity_min": 0.80,
    },
    "normal_v2": {
        "sim": 0.62,
        "row_margin": 0.05,
        "col_margin": 0.03,
        "row_min": 2,
        "col_min": 2,
        "bottom_rank_delta": 2,
        "height_rank_delta": 2,
        "height_ratio_min": 0.65,
        "height_ratio_max": 1.60,
        "pre_before_min": 2,
        "post_after_min": 2,
        "pre_after_max": 999999,
        "post_before_max": 999999,
        "purity_min": 0.80,
    },
    "research_v2": {
        "sim": 0.55,
        "row_margin": 0.03,
        "col_margin": 0.02,
        "row_min": 2,
        "col_min": 2,
        "bottom_rank_delta": 3,
        "height_rank_delta": 3,
        "height_ratio_min": 0.50,
        "height_ratio_max": 2.00,
        "pre_before_min": 1,
        "post_after_min": 1,
        "pre_after_max": 999999,
        "post_before_max": 999999,
        "purity_min": 0.60,
    },
}


def deploy_accept(row, pol):
    if row["pre_track"] == row["post_track"]:
        return False
    return (
        row["sim"] >= pol["sim"]
        and row["row_candidate_count"] >= pol["row_min"]
        and row["col_candidate_count"] >= pol["col_min"]
        and row["row_margin"] >= pol["row_margin"]
        and row["col_margin"] >= pol["col_margin"]
        and row["bottom_rank_delta"] <= pol["bottom_rank_delta"]
        and row["height_rank_delta"] <= pol["height_rank_delta"]
        and pol["height_ratio_min"] <= row["height_ratio"] <= pol["height_ratio_max"]
        and row["pre_track_pre_rows"] >= pol["pre_before_min"]
        and row["post_track_post_rows"] >= pol["post_after_min"]
        and row["pre_track_post_rows"] <= pol["pre_after_max"]
        and row["post_track_pre_rows"] <= pol["post_before_max"]
    )


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="A39_03a_v2 tunnel assignment dry-run with full ReID matrix and lifecycle/geometry/rank gates.")
    ap.add_argument("--track-file", required=True)
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--tunnels-csv", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--fast-reid-config", required=True)
    ap.add_argument("--fast-reid-weights", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pre-window", type=int, default=10)
    ap.add_argument("--post-window", type=int, default=10)
    ap.add_argument("--exit-window", type=int, default=10)
    ap.add_argument("--max-crops-per-track", type=int, default=5)
    ap.add_argument("--min-group-rows", type=int, default=2)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, rows_by, by_frame_tid, by_tid, img_width = read_track(Path(args.track_file))
    gt_by = read_gt(Path(args.gt_file))
    tunnels = read_tunnels(Path(args.tunnels_csv))
    tunnel_by_id = {t["tunnel_id"]: t for t in tunnels}
    row_gt, row_iou = match_row_gt(rows_by, gt_by, args.iou_thr)

    groups = defaultdict(list)
    for t in tunnels:
        start, end = t["start"], t["end"]
        for r in rows:
            if r["track_id"] not in t["tracks"]:
                continue
            side = None
            if start - args.pre_window <= r["frame"] < start:
                side = "pre"
            elif end < r["frame"] <= end + args.post_window:
                side = "post"
            if side is None:
                continue
            groups[(t["tunnel_id"], side, r["track_id"])].append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= args.min_group_rows}

    encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device, batch_size=32)
    proto, crop_counts = extract_group_features(groups, args.img_dir, encoder, args.max_crops_per_track)

    group_rows = []
    gmeta = {}
    for k, rr in groups.items():
        if k not in proto:
            continue
        meta = summarize_group(k, rr, row_gt, crop_counts, by_tid, tunnel_by_id[k[0]], k[1], args.pre_window, args.post_window)
        group_rows.append(meta)
        gmeta[k] = meta

    matrix_rows = []
    accepted_rows = []
    rejected_rows = []
    collision_rows = []
    summary = {name: Counter() for name in POLICIES}
    global_counts = Counter()

    for t in tunnels:
        tid = t["tunnel_id"]
        pre = [g for g in group_rows if g["tunnel_id"] == tid and g["side"] == "pre" and (tid, "pre", g["track_id"]) in proto]
        post = [g for g in group_rows if g["tunnel_id"] == tid and g["side"] == "post" and (tid, "post", g["track_id"]) in proto]
        if not pre or not post:
            continue
        global_counts["tunnels_with_prepost"] += 1
        P = np.stack([proto[(tid, "pre", g["track_id"])] for g in pre])
        Q = np.stack([proto[(tid, "post", g["track_id"])] for g in post])
        S = P @ Q.T
        row_sorted = np.sort(S, axis=1)[:, ::-1]
        col_sorted = np.sort(S, axis=0)[::-1, :]
        row_margins = row_sorted[:, 0] - row_sorted[:, 1] if S.shape[1] >= 2 else np.zeros(S.shape[0], dtype=np.float32)
        col_margins = col_sorted[0, :] - col_sorted[1, :] if S.shape[0] >= 2 else np.zeros(S.shape[1], dtype=np.float32)
        row_ranks = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        col_ranks = np.argsort(np.argsort(-S, axis=0), axis=0) + 1
        bottom_pre_rank = {g["track_id"]: r for r, g in enumerate(sorted(pre, key=lambda x: -x["mean_bottom_y"]), start=1)}
        bottom_post_rank = {g["track_id"]: r for r, g in enumerate(sorted(post, key=lambda x: -x["mean_bottom_y"]), start=1)}
        height_pre_rank = {g["track_id"]: r for r, g in enumerate(sorted(pre, key=lambda x: -x["mean_height"]), start=1)}
        height_post_rank = {g["track_id"]: r for r, g in enumerate(sorted(post, key=lambda x: -x["mean_height"]), start=1)}
        ri, ci = linear_sum_assignment(-S)
        hungarian = {(int(i), int(j)) for i, j in zip(ri, ci)}

        for i, pg in enumerate(pre):
            for j, qg in enumerate(post):
                sim = float(S[i, j])
                pre_id = pg["track_id"]
                post_id = qg["track_id"]
                pre_gt = pg["major_gt"]
                post_gt = qg["major_gt"]
                gt_known = int(pre_gt >= 0 and post_gt >= 0 and pg["gt_purity"] >= 0.6 and qg["gt_purity"] >= 0.6)
                gt_same = int(gt_known and pre_gt == post_gt)
                self_cont = int(pre_id == post_id)
                true_reconnect = int((not self_cont) and gt_same)
                false_reconnect = int((not self_cont) and gt_known and not gt_same)
                height_ratio = qg["mean_height"] / max(pg["mean_height"], 1e-6)
                center_delta_norm = abs(pg["mean_cx"] - qg["mean_cx"]) / max(img_width, 1.0)
                bottom_delta = abs(pg["mean_bottom_y"] - qg["mean_bottom_y"])
                bottom_rank_delta = abs(bottom_pre_rank[pre_id] - bottom_post_rank[post_id])
                height_rank_delta = abs(height_pre_rank[pre_id] - height_post_rank[post_id])
                # Forecast collision for post-only and oracle core+exit only when diagnostic GT says same person.
                post_rows = [r for fr in range(t["end"] + 1, t["end"] + args.exit_window + 1) for r in by_frame_tid.get((fr, post_id), [])]
                post_collision = sum(1 for r in post_rows if by_frame_tid.get((r["frame"], pre_id)))
                oracle_rows = []
                if gt_same:
                    for r in rows:
                        if t["start"] <= r["frame"] <= t["end"] + args.exit_window and r["track_id"] in t["tracks"] and row_gt.get(r["idx"], -1) == pre_gt and r["track_id"] != pre_id:
                            oracle_rows.append(r)
                oracle_collision = sum(1 for r in oracle_rows if by_frame_tid.get((r["frame"], pre_id)))
                row = {
                    "tunnel_id": tid,
                    "pre_track": pre_id,
                    "post_track": post_id,
                    "sim": sim,
                    "row_rank": int(row_ranks[i, j]),
                    "col_rank": int(col_ranks[i, j]),
                    "row_margin": float(row_margins[i]),
                    "col_margin": float(col_margins[j]),
                    "row_candidate_count": len(post),
                    "col_candidate_count": len(pre),
                    "hungarian": int((i, j) in hungarian),
                    "pre_gt": pre_gt,
                    "post_gt": post_gt,
                    "gt_known": gt_known,
                    "gt_same": gt_same,
                    "self_continuation": self_cont,
                    "true_reconnect": true_reconnect,
                    "false_reconnect": false_reconnect,
                    "pre_gt_purity": pg["gt_purity"],
                    "post_gt_purity": qg["gt_purity"],
                    "pre_track_pre_rows": pg["pre_rows"],
                    "pre_track_core_rows": pg["core_rows"],
                    "pre_track_post_rows": pg["post_rows"],
                    "post_track_pre_rows": qg["pre_rows"],
                    "post_track_core_rows": qg["core_rows"],
                    "post_track_post_rows": qg["post_rows"],
                    "pre_bottom_rank": bottom_pre_rank[pre_id],
                    "post_bottom_rank": bottom_post_rank[post_id],
                    "bottom_rank_delta": bottom_rank_delta,
                    "pre_height_rank": height_pre_rank[pre_id],
                    "post_height_rank": height_post_rank[post_id],
                    "height_rank_delta": height_rank_delta,
                    "center_delta_norm": center_delta_norm,
                    "bottom_delta": bottom_delta,
                    "height_ratio": height_ratio,
                    "post_rows_forecast": len(post_rows),
                    "post_collision_rows": post_collision,
                    "post_collision_ratio": safe_div(post_collision, len(post_rows)),
                    "oracle_core_exit_rows": len(oracle_rows),
                    "oracle_collision_rows": oracle_collision,
                    "oracle_collision_ratio": safe_div(oracle_collision, len(oracle_rows)),
                }
                reasons = []
                for name, pol in POLICIES.items():
                    dep = deploy_accept(row, pol)
                    purity_ok = row["pre_gt_purity"] >= pol["purity_min"] and row["post_gt_purity"] >= pol["purity_min"]
                    row[f"accept_{name}"] = int(dep)
                    row[f"eval_accept_{name}"] = int(dep and purity_ok)
                    if dep:
                        summary[name]["deploy_accept"] += 1
                        summary[name]["self_continuation"] += self_cont
                        summary[name]["true_reconnect"] += true_reconnect
                        summary[name]["false_reconnect"] += false_reconnect
                        summary[name]["gt_unknown_or_impure"] += int(not purity_ok or not gt_known)
                        summary[name]["post_rows_forecast"] += len(post_rows)
                        summary[name]["post_collision_rows"] += post_collision
                        summary[name]["oracle_core_exit_rows"] += len(oracle_rows)
                        summary[name]["oracle_collision_rows"] += oracle_collision
                    if dep and purity_ok:
                        summary[name]["eval_accept"] += 1
                        summary[name]["eval_gt_same"] += gt_same
                        summary[name]["eval_wrong"] += int(gt_known and not gt_same)
                if any(row[f"accept_{name}"] for name in POLICIES):
                    accepted_rows.append(row.copy())
                    collision_rows.append({k: row[k] for k in ["tunnel_id", "pre_track", "post_track", "post_rows_forecast", "post_collision_rows", "post_collision_ratio", "oracle_core_exit_rows", "oracle_collision_rows", "oracle_collision_ratio", "gt_same"]})
                elif row["hungarian"] and row["row_rank"] == 1:
                    rejected_rows.append(row.copy())
                matrix_rows.append(row)

    fields = list(matrix_rows[0].keys()) if matrix_rows else []
    write_csv(out / "full_pair_matrix.csv", fields, matrix_rows)
    write_csv(out / "accepted_assignments.csv", fields, accepted_rows)
    write_csv(out / "rejected_top1_hungarian.csv", fields, rejected_rows)
    write_csv(out / "collision_forecast.csv", list(collision_rows[0].keys()) if collision_rows else ["tunnel_id"], collision_rows)
    write_csv(out / "group_features.csv", list(group_rows[0].keys()) if group_rows else ["tunnel_id"], group_rows)

    summary_rows = []
    for name, c in summary.items():
        eval_precision = safe_div(c["eval_gt_same"], c["eval_accept"])
        deploy_known_precision = safe_div(c["true_reconnect"], c["true_reconnect"] + c["false_reconnect"])
        summary_rows.append(
            {
                "policy": name,
                "deploy_accept": c["deploy_accept"],
                "eval_accept": c["eval_accept"],
                "eval_gt_same": c["eval_gt_same"],
                "eval_wrong": c["eval_wrong"],
                "eval_precision": eval_precision,
                "true_reconnect": c["true_reconnect"],
                "false_reconnect": c["false_reconnect"],
                "deploy_known_precision": deploy_known_precision,
                "self_continuation": c["self_continuation"],
                "gt_unknown_or_impure": c["gt_unknown_or_impure"],
                "post_rows_forecast": c["post_rows_forecast"],
                "post_collision_rows": c["post_collision_rows"],
                "post_collision_ratio": safe_div(c["post_collision_rows"], c["post_rows_forecast"]),
                "oracle_core_exit_rows": c["oracle_core_exit_rows"],
                "oracle_collision_rows": c["oracle_collision_rows"],
                "oracle_collision_ratio": safe_div(c["oracle_collision_rows"], c["oracle_core_exit_rows"]),
            }
        )
    write_csv(out / "policy_summary.csv", list(summary_rows[0].keys()), summary_rows)
    payload = {"params": vars(args), "global_counts": dict(global_counts), "policies": POLICIES, "summary": summary_rows}
    (out / "policy_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_03a_v2 tunnel assignment dry-run",
        "",
        "| policy | deploy_accept | eval_accept | eval_precision | true_reconnect | false_reconnect | self_cont | unknown/impure | post_rows | post_collision_ratio | oracle_rows | oracle_collision_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['policy']} | {r['deploy_accept']} | {r['eval_accept']} | {r['eval_precision']:.4f} | {r['true_reconnect']} | {r['false_reconnect']} | {r['self_continuation']} | {r['gt_unknown_or_impure']} | {r['post_rows_forecast']} | {r['post_collision_ratio']:.4f} | {r['oracle_core_exit_rows']} | {r['oracle_collision_ratio']:.4f} |"
        )
    (out / "policy_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
