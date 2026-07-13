#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_track(path: Path):
    rows = []
    by_frame = defaultdict(list)
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
    return rows, by_frame


def read_gt(path: Path):
    by_frame = defaultdict(list)
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
            by_frame[fr].append({"frame": fr, "gt_id": gid, "box": np.array([x, y, x + w, y + h], dtype=np.float32)})
    return by_frame


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


def match_rows(rows_by_frame, gt_by_frame, iou_thr: float):
    row_gt = {}
    row_iou = {}
    for fr in sorted(set(rows_by_frame) | set(gt_by_frame)):
        rr = rows_by_frame.get(fr, [])
        gg = gt_by_frame.get(fr, [])
        if not rr or not gg:
            continue
        I = pair_iou(rr, gg)
        ri, ci = linear_sum_assignment(1 - I)
        for r, c in zip(ri, ci):
            val = float(I[r, c])
            if val >= iou_thr:
                row_gt[rr[r]["idx"]] = int(gg[c]["gt_id"])
                row_iou[rr[r]["idx"]] = val
    return row_gt, row_iou


def read_tunnels(path: Path):
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tid = ai(r.get("tunnel_id"), len(out))
            tracks = [ai(x, -1) for x in str(r.get("tracks", "")).split("|") if str(x).strip() != ""]
            out[tid] = {
                "tunnel_id": tid,
                "start": ai(r.get("start")),
                "end": ai(r.get("end")),
                "duration": ai(r.get("duration")),
                "tracks": set(x for x in tracks if x >= 0),
            }
    return out


def parse_list_int(s: str):
    return [ai(x, -1) for x in str(s).replace(";", ",").split(",") if str(x).strip() and ai(x, -1) >= 0]


def group_stats(rows, row_gt, target_gt):
    if not rows:
        return None
    c = Counter()
    for r in rows:
        gid = row_gt.get(r["idx"], -1)
        if gid >= 0:
            c[gid] += 1
    major_gt = -1
    gt_count = 0
    purity = 0.0
    if c:
        major_gt, gt_count = c.most_common(1)[0]
        purity = gt_count / max(1, sum(c.values()))
    frames = [r["frame"] for r in rows]
    return {
        "track_id": rows[0]["track_id"],
        "rows": len(rows),
        "matched_rows": sum(c.values()),
        "first_frame": min(frames),
        "last_frame": max(frames),
        "major_gt": major_gt,
        "gt_count": gt_count,
        "gt_purity": purity,
        "target_gt_rows": c.get(target_gt, 0),
        "target_gt_ratio": safe_div(c.get(target_gt, 0), len(rows)),
        "mean_cx": float(np.mean([r["cx"] for r in rows])),
        "mean_bottom_y": float(np.mean([r["bottom_y"] for r in rows])),
        "mean_height": float(np.mean([r["height"] for r in rows])),
        "mean_score": float(np.mean([r["score"] for r in rows])),
    }


def candidate_type(st, target_gt):
    if not st:
        return "none"
    if st["major_gt"] != target_gt:
        if st["target_gt_rows"] > 0:
            return "mixed_contains_target"
        return "wrong_gt"
    if st["rows"] < 3:
        return "too_short"
    if st["gt_purity"] < 0.8:
        return "impure"
    if st["rows"] >= 5 and st["gt_purity"] >= 0.9:
        return "strict_clean"
    return "loose_clean"


def is_loose(st, target_gt):
    return bool(st and st["major_gt"] == target_gt and st["rows"] >= 3 and st["gt_purity"] >= 0.8)


def is_strict(st, target_gt):
    return bool(st and st["major_gt"] == target_gt and st["rows"] >= 5 and st["gt_purity"] >= 0.9)


def get_side_groups(rows, row_gt, target_gt, tunnel, side, window, scope):
    start, end = tunnel["start"], tunnel["end"]
    if side == "pre":
        lo, hi = start - window, start - 1
    else:
        lo, hi = end + 1, end + window
    by_tid = defaultdict(list)
    for r in rows:
        if not (lo <= r["frame"] <= hi):
            continue
        if scope == "tunnel_tracks" and r["track_id"] not in tunnel["tracks"]:
            continue
        by_tid[r["track_id"]].append(r)
    stats = []
    for tid, rr in sorted(by_tid.items()):
        st = group_stats(rr, row_gt, target_gt)
        if not st:
            continue
        st.update({
            "side": side,
            "window": window,
            "scope": scope,
            "in_tunnel_tracks": int(tid in tunnel["tracks"]),
            "candidate_type": candidate_type(st, target_gt),
        })
        stats.append(st)
    # Best groups prioritize target gt rows, then purity, then rows.
    best_any_target = None
    targetish = [s for s in stats if s["target_gt_rows"] > 0]
    if targetish:
        best_any_target = max(targetish, key=lambda s: (s["target_gt_rows"], s["gt_purity"], s["rows"]))
    best_loose = None
    loose = [s for s in stats if is_loose(s, target_gt)]
    if loose:
        best_loose = max(loose, key=lambda s: (s["rows"], s["gt_purity"], s["target_gt_rows"]))
    best_strict = None
    strict = [s for s in stats if is_strict(s, target_gt)]
    if strict:
        best_strict = max(strict, key=lambda s: (s["rows"], s["gt_purity"], s["target_gt_rows"]))
    return stats, best_any_target, best_loose, best_strict


def best_type(best_any, best_loose, best_strict):
    if best_strict:
        return "strict_clean"
    if best_loose:
        return "loose_clean"
    if best_any:
        return best_any.get("candidate_type", "target_present_not_clean")
    return "absent"


def new_reason(pre_type, post_type, current_reason):
    pre_clean = pre_type in {"strict_clean", "loose_clean"}
    post_clean = post_type in {"strict_clean", "loose_clean"}
    if pre_clean and post_clean:
        return "WIDE_ANCHORS_AVAILABLE"
    if pre_clean and not post_clean:
        if post_type == "absent":
            return "TRUE_NO_POST_VISIBILITY"
        return f"POST_{post_type.upper()}"
    if post_clean and not pre_clean:
        if pre_type == "absent":
            return "TRUE_NO_PRE_VISIBILITY"
        return f"PRE_{pre_type.upper()}"
    if pre_type == "absent" and post_type == "absent":
        return "TRUE_NO_BOTH_VISIBILITY"
    if "mixed" in pre_type or "mixed" in post_type:
        return "MIXED_OR_ROW_LEVEL_ONLY"
    if "impure" in pre_type or "impure" in post_type:
        return "ANCHOR_IMPURE"
    if "too_short" in pre_type or "too_short" in post_type:
        return "ANCHOR_TOO_SHORT"
    return "NO_CLEAN_ANCHORS"


def main():
    ap = argparse.ArgumentParser(description="A39_03h: diagnose whether NO_PRE/NO_POST oracle misses are true anchor absence or extraction-window/scope misses.")
    ap.add_argument("--track-file", required=True)
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--tunnels-csv", required=True)
    ap.add_argument("--oracle-to-anchor", required=True)
    ap.add_argument("--tunnel-ids", default="104,22")
    ap.add_argument("--windows", default="10,20,30,50,75,100")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--focus-reasons", default="NO_PRE_ANCHOR,NO_POST_ANCHOR,LOW_REID_SIM_LT060")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tunnel_ids = set(parse_list_int(args.tunnel_ids))
    windows = parse_list_int(args.windows)
    focus_reasons = set(x.strip() for x in args.focus_reasons.split(",") if x.strip())

    rows, rows_by_frame = read_track(Path(args.track_file))
    gt_by_frame = read_gt(Path(args.gt_file))
    row_gt, row_iou = match_rows(rows_by_frame, gt_by_frame, args.iou_thr)
    tunnels = read_tunnels(Path(args.tunnels_csv))
    oracle = read_csv(Path(args.oracle_to_anchor))

    txs = [r for r in oracle if ai(r.get("tunnel_id"), -1) in tunnel_ids and r.get("miss_reason") in focus_reasons]
    group_records = []
    tx_records = []
    summary_counter = defaultdict(lambda: Counter())

    for tx in txs:
        tunnel_id = ai(tx["tunnel_id"], -1)
        tunnel = tunnels[tunnel_id]
        gt_id = ai(tx["gt_id"], -1)
        applied = ai(tx.get("row_audit_applied", tx.get("applied_rows", 0)), 0)
        for scope in ["tunnel_tracks", "boundary_all_tracks"]:
            for W in windows:
                pre_stats, pre_any, pre_loose, pre_strict = get_side_groups(rows, row_gt, gt_id, tunnel, "pre", W, scope)
                post_stats, post_any, post_loose, post_strict = get_side_groups(rows, row_gt, gt_id, tunnel, "post", W, scope)
                # Emit only relevant groups that contain target gt or are clean/wrong majority with target presence.
                for side, stats in [("pre", pre_stats), ("post", post_stats)]:
                    for st in stats:
                        if st["target_gt_rows"] <= 0 and st["major_gt"] != gt_id:
                            continue
                        rec = {
                            "transaction_id": tx["transaction_id"],
                            "tunnel_id": tunnel_id,
                            "gt_id": gt_id,
                            "target_id": tx.get("target_id", ""),
                            "applied_rows": applied,
                            "current_miss_reason": tx.get("miss_reason", ""),
                            **st,
                        }
                        group_records.append(rec)
                pre_type = best_type(pre_any, pre_loose, pre_strict)
                post_type = best_type(post_any, post_loose, post_strict)
                reason = new_reason(pre_type, post_type, tx.get("miss_reason", ""))
                rec = {
                    "transaction_id": tx["transaction_id"],
                    "tunnel_id": tunnel_id,
                    "gt_id": gt_id,
                    "target_id": tx.get("target_id", ""),
                    "applied_rows": applied,
                    "current_miss_reason": tx.get("miss_reason", ""),
                    "window": W,
                    "scope": scope,
                    "pre_best_type": pre_type,
                    "post_best_type": post_type,
                    "pre_loose_found": int(bool(pre_loose)),
                    "post_loose_found": int(bool(post_loose)),
                    "pre_strict_found": int(bool(pre_strict)),
                    "post_strict_found": int(bool(post_strict)),
                    "both_loose_found": int(bool(pre_loose and post_loose)),
                    "both_strict_found": int(bool(pre_strict and post_strict)),
                    "best_pre_track": pre_strict["track_id"] if pre_strict else (pre_loose["track_id"] if pre_loose else (pre_any["track_id"] if pre_any else "")),
                    "best_post_track": post_strict["track_id"] if post_strict else (post_loose["track_id"] if post_loose else (post_any["track_id"] if post_any else "")),
                    "best_pre_rows": pre_strict["rows"] if pre_strict else (pre_loose["rows"] if pre_loose else (pre_any["rows"] if pre_any else 0)),
                    "best_post_rows": post_strict["rows"] if post_strict else (post_loose["rows"] if post_loose else (post_any["rows"] if post_any else 0)),
                    "best_pre_purity": pre_strict["gt_purity"] if pre_strict else (pre_loose["gt_purity"] if pre_loose else (pre_any["gt_purity"] if pre_any else 0.0)),
                    "best_post_purity": post_strict["gt_purity"] if post_strict else (post_loose["gt_purity"] if post_loose else (post_any["gt_purity"] if post_any else 0.0)),
                    "new_reason": reason,
                }
                tx_records.append(rec)
                key = (W, scope)
                summary_counter[key]["transactions"] += 1
                summary_counter[key]["effective_applied_rows"] += applied
                if rec["both_loose_found"]:
                    summary_counter[key]["both_loose_found_rows"] += applied
                if rec["both_strict_found"]:
                    summary_counter[key]["both_strict_found_rows"] += applied
                if rec["pre_loose_found"] and not rec["post_loose_found"]:
                    summary_counter[key]["pre_only_rows"] += applied
                if rec["post_loose_found"] and not rec["pre_loose_found"]:
                    summary_counter[key]["post_only_rows"] += applied
                if not rec["pre_loose_found"] and not rec["post_loose_found"]:
                    summary_counter[key]["neither_rows"] += applied
                summary_counter[key][f"reason_{reason}"] += applied

    group_fields = [
        "transaction_id", "tunnel_id", "gt_id", "target_id", "applied_rows", "current_miss_reason", "side", "window", "scope", "track_id", "rows", "matched_rows", "first_frame", "last_frame", "major_gt", "gt_count", "gt_purity", "target_gt_rows", "target_gt_ratio", "mean_cx", "mean_bottom_y", "mean_height", "mean_score", "in_tunnel_tracks", "candidate_type"
    ]
    write_csv(out / "anchor_group_candidates.csv", group_fields, group_records)
    tx_fields = list(tx_records[0].keys()) if tx_records else ["transaction_id"]
    write_csv(out / "anchor_availability_by_transaction.csv", tx_fields, tx_records)

    summary_rows = []
    reason_keys = sorted({k for c in summary_counter.values() for k in c if k.startswith("reason_")})
    for (W, scope), c in sorted(summary_counter.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {
            "window": W,
            "scope": scope,
            "transactions": c["transactions"],
            "effective_applied_rows": c["effective_applied_rows"],
            "both_loose_found_rows": c["both_loose_found_rows"],
            "both_loose_found_rate": safe_div(c["both_loose_found_rows"], c["effective_applied_rows"]),
            "both_strict_found_rows": c["both_strict_found_rows"],
            "both_strict_found_rate": safe_div(c["both_strict_found_rows"], c["effective_applied_rows"]),
            "pre_only_rows": c["pre_only_rows"],
            "post_only_rows": c["post_only_rows"],
            "neither_rows": c["neither_rows"],
        }
        for rk in reason_keys:
            row[rk] = c[rk]
        summary_rows.append(row)
    write_csv(out / "anchor_absence_summary.csv", list(summary_rows[0].keys()) if summary_rows else ["window"], summary_rows)

    # Best per transaction: first scope/window that recovers both strict, else both loose, else best available.
    by_tx = defaultdict(list)
    for r in tx_records:
        by_tx[r["transaction_id"]].append(r)
    best_rows = []
    for txid, rr in by_tx.items():
        def score(r):
            return (
                ai(r["both_strict_found"]),
                ai(r["both_loose_found"]),
                ai(r["pre_loose_found"]) + ai(r["post_loose_found"]),
                -ai(r["window"]),
                1 if r["scope"] == "tunnel_tracks" else 0,
            )
        best = max(rr, key=score)
        best_rows.append(best)
    write_csv(out / "anchor_availability_best_by_transaction.csv", tx_fields, sorted(best_rows, key=lambda r: (-ai(r["applied_rows"]), ai(r["transaction_id"]))))

    payload = {
        "inputs": vars(args),
        "transactions_focus": len(txs),
        "group_records": len(group_records),
        "summary": summary_rows,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# A39_03h Anchor Absence Audit",
        "",
        f"focus transactions: `{len(txs)}`",
        f"focus tunnels: `{','.join(map(str, sorted(tunnel_ids)))}`",
        "",
        "## Window/scope summary",
        "",
        "| scope | window | rows | both_loose_rows | both_loose_rate | both_strict_rows | both_strict_rate | pre_only | post_only | neither |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(f"| {r['scope']} | {r['window']} | {r['effective_applied_rows']} | {r['both_loose_found_rows']} | {r['both_loose_found_rate']:.4f} | {r['both_strict_found_rows']} | {r['both_strict_found_rate']:.4f} | {r['pre_only_rows']} | {r['post_only_rows']} | {r['neither_rows']} |")
    md.extend(["", "## Best interpretation", ""])
    best_by_scope = {}
    for r in summary_rows:
        key = r["scope"]
        if key not in best_by_scope or af(r["both_strict_found_rate"]) > af(best_by_scope[key]["both_strict_found_rate"]):
            best_by_scope[key] = r
    for scope, r in best_by_scope.items():
        md.append(f"- `{scope}` best strict recovery: W={r['window']}, strict rows={r['both_strict_found_rows']} ({r['both_strict_found_rate']:.4f})")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
