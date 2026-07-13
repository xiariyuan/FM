#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


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


def lifecycle_suspension(pair):
    return (
        ai(pair.get("pre_track")) != ai(pair.get("post_track"))
        and ai(pair.get("pre_track_pre_rows")) >= 3
        and ai(pair.get("post_track_post_rows")) >= 3
        and ai(pair.get("pre_track_post_rows")) == 0
        and ai(pair.get("post_track_pre_rows")) == 0
    )


def top11(pair):
    return ai(pair.get("row_rank")) == 1 and ai(pair.get("col_rank")) == 1


def anchor_id(tunnel_id, pre_track, post_track):
    return f"{ai(tunnel_id)}_{ai(pre_track)}_{ai(post_track)}"


def parse_tracks(s):
    return [ai(x, -1) for x in str(s).split("|") if str(x).strip() != "" and ai(x, -1) >= 0]


def best_by_sim(rows):
    if not rows:
        return None
    return max(rows, key=lambda r: af(r.get("sim"), -999.0))


def pair_brief(p):
    if not p:
        return ""
    return f"{ai(p['pre_track'])}->{ai(p['post_track'])}@{af(p['sim']):.3f}/r{ai(p['row_rank'])}c{ai(p['col_rank'])}"


def miss_reason_for(tx, pre_pairs, post_pairs, true_pairs, self_pairs, best_true, path_summary_by_anchor):
    if not pre_pairs:
        return "NO_PRE_ANCHOR"
    if not post_pairs:
        return "NO_POST_ANCHOR"
    if not true_pairs:
        if self_pairs:
            return "SELF_CONTINUATION_ONLY"
        return "NO_TRUE_RECONNECT_PAIR"
    if best_true is None:
        return "NO_TRUE_RECONNECT_PAIR"
    if af(best_true.get("sim")) < 0.60:
        return "LOW_REID_SIM_LT060"
    if not top11(best_true):
        return "NOT_TOP1_TOP1"
    if not lifecycle_suspension(best_true):
        return "LIFECYCLE_NOT_CLEAN"
    aid = anchor_id(best_true["tunnel_id"], best_true["pre_track"], best_true["post_track"])
    hs = path_summary_by_anchor.get(aid)
    if hs:
        if str(hs.get("decision", "")).startswith("PASS"):
            return "DEPLOYABLE_GATE_PASS"
        if ai(hs.get("applied_rows")) < 30:
            return "PATH_BUILDER_TOO_FEW_ROWS"
        if ai(hs.get("selected_fragment_count")) < 2:
            return "PATH_BUILDER_INCOMPLETE_PATH"
        if ai(hs.get("diag_wrong_rows")) > 1:
            return "PATH_BUILDER_WRONG_ROWS_DIAG"
        return "PATH_BUILDER_REJECTED"
    # If the pair is lifecycle+top11+sim but was not audited in f0, it means it did not belong to the sim>=0.60 known audit set for some reason.
    return "PATH_BUILDER_NOT_RUN"


def main():
    ap = argparse.ArgumentParser(description="Audit the gap between A39 oracle tunnel transactions and deployable anchor/path-builder coverage.")
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--row-audit", required=True)
    ap.add_argument("--full-pair-matrix", required=True)
    ap.add_argument("--hard-negative-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-applied-rows", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txs = read_csv(Path(args.transactions))
    row_audit = read_csv(Path(args.row_audit))
    pairs = read_csv(Path(args.full_pair_matrix))
    hard = read_csv(Path(args.hard_negative_summary)) if Path(args.hard_negative_summary).exists() else []
    path_summary_by_anchor = {r["anchor_id"]: r for r in hard if r.get("anchor_id")}

    # Row-level transaction summary from row_audit. Use applied rows for actual oracle effect.
    tx_row_counts = defaultdict(Counter)
    tx_frame_ranges = defaultdict(lambda: {"min": None, "max": None})
    for r in row_audit:
        tid = ai(r.get("transaction_id"), -1)
        if tid < 0:
            continue
        action = r.get("action", "")
        tx_row_counts[tid][action] += 1
        fr = ai(r.get("frame"), -1)
        if fr >= 0:
            cur = tx_frame_ranges[tid]
            cur["min"] = fr if cur["min"] is None else min(cur["min"], fr)
            cur["max"] = fr if cur["max"] is None else max(cur["max"], fr)

    pairs_by_tunnel_gt_pre = defaultdict(list)
    pairs_by_tunnel_gt_post = defaultdict(list)
    true_pairs_by_tunnel_gt = defaultdict(list)
    self_pairs_by_tunnel_gt = defaultdict(list)
    all_pairs_by_tunnel_gt = defaultdict(list)
    for p in pairs:
        tunnel = ai(p.get("tunnel_id"), -1)
        pre_gt = ai(p.get("pre_gt"), -1)
        post_gt = ai(p.get("post_gt"), -1)
        if pre_gt >= 0:
            pairs_by_tunnel_gt_pre[(tunnel, pre_gt)].append(p)
            all_pairs_by_tunnel_gt[(tunnel, pre_gt)].append(p)
        if post_gt >= 0:
            pairs_by_tunnel_gt_post[(tunnel, post_gt)].append(p)
            all_pairs_by_tunnel_gt[(tunnel, post_gt)].append(p)
        if pre_gt >= 0 and pre_gt == post_gt:
            if ai(p.get("pre_track")) == ai(p.get("post_track")):
                self_pairs_by_tunnel_gt[(tunnel, pre_gt)].append(p)
            else:
                true_pairs_by_tunnel_gt[(tunnel, pre_gt)].append(p)

    oracle_rows = []
    oracle_to_anchor = []
    for tx in txs:
        txid = ai(tx.get("transaction_id"), -1)
        tunnel = ai(tx.get("tunnel_id"), -1)
        gid = ai(tx.get("gt_id"), -1)
        applied = ai(tx.get("applied_rows"), 0)
        if applied < args.min_applied_rows:
            continue
        key = (tunnel, gid)
        pre_pairs = pairs_by_tunnel_gt_pre.get(key, [])
        post_pairs = pairs_by_tunnel_gt_post.get(key, [])
        true_pairs = true_pairs_by_tunnel_gt.get(key, [])
        self_pairs = self_pairs_by_tunnel_gt.get(key, [])
        best_true = best_by_sim(true_pairs)
        best_self = best_by_sim(self_pairs)
        best_any_pre = best_by_sim(pre_pairs)
        best_any_post = best_by_sim(post_pairs)
        miss_reason = miss_reason_for(tx, pre_pairs, post_pairs, true_pairs, self_pairs, best_true, path_summary_by_anchor)
        best_anchor = ""
        path_decision = ""
        path_planned = 0
        path_wrong = 0
        if best_true:
            best_anchor = anchor_id(best_true["tunnel_id"], best_true["pre_track"], best_true["post_track"])
            hs = path_summary_by_anchor.get(best_anchor, {})
            path_decision = hs.get("decision", "")
            path_planned = ai(hs.get("planned_rows", 0))
            path_wrong = ai(hs.get("diag_wrong_rows", 0))
        actual_counts = tx_row_counts.get(txid, Counter())
        frame_range = tx_frame_ranges.get(txid, {"min": None, "max": None})
        source_tracks = parse_tracks(tx.get("source_track_ids", ""))
        row = {
            "transaction_id": txid,
            "tunnel_id": tunnel,
            "gt_id": gid,
            "target_id": ai(tx.get("target_id"), -1),
            "source_track_ids": "|".join(map(str, source_tracks)),
            "source_track_count": len(source_tracks),
            "frame_start": ai(tx.get("frame_start", -1), -1),
            "frame_end": ai(tx.get("frame_end", -1), -1),
            "audit_frame_min": frame_range["min"] if frame_range["min"] is not None else "",
            "audit_frame_max": frame_range["max"] if frame_range["max"] is not None else "",
            "planned_rows": ai(tx.get("planned_rows"), 0),
            "applied_rows": applied,
            "undone_rows": ai(tx.get("undone_rows"), 0),
            "temp_rows": ai(tx.get("temp_rows"), 0),
            "row_audit_applied": actual_counts.get("applied", 0),
            "row_audit_undone": actual_counts.get("undone", 0),
            "row_audit_temp": actual_counts.get("final_temp", 0),
            "status": tx.get("status", ""),
            "anchor_source": tx.get("anchor_source", ""),
            "has_pre_anchor": int(bool(pre_pairs)),
            "has_post_anchor": int(bool(post_pairs)),
            "pre_pair_count": len(pre_pairs),
            "post_pair_count": len(post_pairs),
            "true_reconnect_pair_count": len(true_pairs),
            "self_pair_count": len(self_pairs),
            "best_true_pair": pair_brief(best_true),
            "best_true_anchor_id": best_anchor,
            "best_true_sim": af(best_true.get("sim"), 0.0) if best_true else 0.0,
            "best_true_row_rank": ai(best_true.get("row_rank"), 0) if best_true else 0,
            "best_true_col_rank": ai(best_true.get("col_rank"), 0) if best_true else 0,
            "best_true_lifecycle": int(lifecycle_suspension(best_true)) if best_true else 0,
            "best_true_top11": int(top11(best_true)) if best_true else 0,
            "best_self_pair": pair_brief(best_self),
            "best_self_sim": af(best_self.get("sim"), 0.0) if best_self else 0.0,
            "best_any_pre_pair": pair_brief(best_any_pre),
            "best_any_post_pair": pair_brief(best_any_post),
            "path_builder_anchor_decision": path_decision,
            "path_builder_planned_rows": path_planned,
            "path_builder_wrong_rows_diag": path_wrong,
            "miss_reason": miss_reason,
        }
        oracle_to_anchor.append(row)
        oracle_rows.append(row)

    oracle_rank = sorted(oracle_rows, key=lambda r: (-ai(r["applied_rows"]), ai(r["transaction_id"])))
    write_csv(out / "oracle_transaction_rank.csv", list(oracle_rank[0].keys()) if oracle_rank else ["transaction_id"], oracle_rank)
    write_csv(out / "oracle_to_anchor_match.csv", list(oracle_to_anchor[0].keys()) if oracle_to_anchor else ["transaction_id"], oracle_to_anchor)

    reason_counter = defaultdict(lambda: Counter())
    tunnel_counter = defaultdict(lambda: Counter())
    total_applied = 0
    for r in oracle_to_anchor:
        reason = r["miss_reason"]
        applied = ai(r["applied_rows"], 0)
        total_applied += applied
        reason_counter[reason]["transaction_count"] += 1
        reason_counter[reason]["applied_rows_sum"] += applied
        reason_counter[reason]["planned_rows_sum"] += ai(r["planned_rows"], 0)
        tunnel_counter[r["tunnel_id"]]["transaction_count"] += 1
        tunnel_counter[r["tunnel_id"]]["applied_rows_sum"] += applied

    reason_rows = []
    for reason, c in sorted(reason_counter.items(), key=lambda kv: -kv[1]["applied_rows_sum"]):
        reason_rows.append({
            "miss_reason": reason,
            "transaction_count": c["transaction_count"],
            "applied_rows_sum": c["applied_rows_sum"],
            "planned_rows_sum": c["planned_rows_sum"],
            "percent_of_oracle_applied_rows": safe_div(c["applied_rows_sum"], total_applied),
        })
    write_csv(out / "miss_reason_summary.csv", list(reason_rows[0].keys()) if reason_rows else ["miss_reason"], reason_rows)

    top_missed = [r for r in oracle_rank if r["miss_reason"] != "DEPLOYABLE_GATE_PASS"][:50]
    write_csv(out / "top_missed_oracle_cases.csv", list(top_missed[0].keys()) if top_missed else ["transaction_id"], top_missed)

    tunnel_rows = []
    for tunnel, c in sorted(tunnel_counter.items(), key=lambda kv: -kv[1]["applied_rows_sum"]):
        tunnel_rows.append({"tunnel_id": tunnel, "transaction_count": c["transaction_count"], "applied_rows_sum": c["applied_rows_sum"], "percent_of_oracle_applied_rows": safe_div(c["applied_rows_sum"], total_applied)})
    write_csv(out / "oracle_tunnel_rank.csv", list(tunnel_rows[0].keys()) if tunnel_rows else ["tunnel_id"], tunnel_rows)

    payload = {
        "inputs": vars(args),
        "oracle_transaction_count": len(oracle_to_anchor),
        "oracle_applied_rows_total": total_applied,
        "miss_reason_summary": reason_rows,
        "top_tunnels": tunnel_rows[:20],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# A39_03g Oracle-to-deployable Gap Audit",
        "",
        f"oracle transactions audited: `{len(oracle_to_anchor)}`",
        f"oracle applied rows total: `{total_applied}`",
        "",
        "## Miss reason summary",
        "",
        "| miss_reason | transactions | applied_rows | percent |",
        "|---|---:|---:|---:|",
    ]
    for r in reason_rows:
        md.append(f"| {r['miss_reason']} | {r['transaction_count']} | {r['applied_rows_sum']} | {r['percent_of_oracle_applied_rows']:.4f} |")
    md.extend(["", "## Top missed oracle cases", "", "| tx | tunnel | gt | target | applied | best_true_pair | miss_reason |", "|---:|---:|---:|---:|---:|---|---|"])
    for r in top_missed[:20]:
        md.append(f"| {r['transaction_id']} | {r['tunnel_id']} | {r['gt_id']} | {r['target_id']} | {r['applied_rows']} | {r['best_true_pair']} | {r['miss_reason']} |")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
