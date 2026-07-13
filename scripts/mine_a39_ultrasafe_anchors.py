#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def bool_int(x):
    return 1 if x else 0


def lifecycle_ok(r, pre_min, post_min, pre_after_max, post_before_max):
    return (
        ai(r["pre_track"]) != ai(r["post_track"])
        and ai(r["pre_track_pre_rows"]) >= pre_min
        and ai(r["post_track_post_rows"]) >= post_min
        and ai(r["pre_track_post_rows"]) <= pre_after_max
        and ai(r["post_track_pre_rows"]) <= post_before_max
    )


def policy_accept(r, p):
    if not lifecycle_ok(r, p["pre_min"], p["post_min"], p["pre_after_max"], p["post_before_max"]):
        return False
    if af(r["sim"]) < p["sim"]:
        return False
    if ai(r["row_rank"]) != 1 or ai(r["col_rank"]) != 1:
        return False
    if ai(r["row_candidate_count"]) < p["row_candidate_min"]:
        return False
    if ai(r["col_candidate_count"]) < p["col_candidate_min"]:
        return False
    if af(r["post_collision_ratio"]) > p["post_collision_max"]:
        return False
    if af(r["row_margin"]) < p["row_margin_min"]:
        return False
    if af(r["col_margin"]) < p["col_margin_min"]:
        return False
    return True


POLICY_BASE = {
    "sim": 0.70,
    "pre_min": 3,
    "post_min": 3,
    "pre_after_max": 0,
    "post_before_max": 0,
    "row_candidate_min": 2,
    "col_candidate_min": 2,
    "post_collision_max": 0.25,
    "row_margin_min": 0.0,
    "col_margin_min": 0.0,
}


def build_policies(args):
    p0 = dict(POLICY_BASE)
    p0.update({
        "sim": args.min_sim,
        "pre_min": args.pre_min,
        "post_min": args.post_min,
        "pre_after_max": args.pre_after_max,
        "post_before_max": args.post_before_max,
        "post_collision_max": args.post_collision_max,
    })
    p_margin = dict(p0)
    p_margin.update({"row_margin_min": args.row_margin_min, "col_margin_min": args.col_margin_min})
    p_sim075 = dict(p_margin)
    p_sim075["sim"] = max(p_sim075["sim"], 0.75)
    p_no_collision_gate = dict(p_margin)
    p_no_collision_gate["post_collision_max"] = 1.01
    return {
        "ultra_safe": p0,
        "ultra_safe_margin": p_margin,
        "ultra_safe_sim075_margin": p_sim075,
        "ultra_safe_margin_no_collision_gate": p_no_collision_gate,
    }


def selected_fields():
    return [
        "anchor_id",
        "policy",
        "tunnel_id",
        "pre_track",
        "post_track",
        "sim",
        "row_rank",
        "col_rank",
        "row_margin",
        "col_margin",
        "row_candidate_count",
        "col_candidate_count",
        "pre_track_pre_rows",
        "pre_track_post_rows",
        "post_track_pre_rows",
        "post_track_post_rows",
        "post_rows_forecast",
        "post_collision_rows",
        "post_collision_ratio",
        "oracle_core_exit_rows",
        "oracle_collision_rows",
        "oracle_collision_ratio",
        "pre_gt",
        "post_gt",
        "gt_known",
        "gt_same",
        "true_reconnect",
        "false_reconnect",
        "pre_gt_purity",
        "post_gt_purity",
        "bottom_rank_delta",
        "height_rank_delta",
        "height_ratio",
        "center_delta_norm",
    ]


def main():
    ap = argparse.ArgumentParser(description="Mine A39 ultra-safe lifecycle anchors from A39_03a_v2 full_pair_matrix.csv.")
    ap.add_argument("--full-pair-matrix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sim", type=float, default=0.70)
    ap.add_argument("--pre-min", type=int, default=3)
    ap.add_argument("--post-min", type=int, default=3)
    ap.add_argument("--pre-after-max", type=int, default=0)
    ap.add_argument("--post-before-max", type=int, default=0)
    ap.add_argument("--post-collision-max", type=float, default=0.25)
    ap.add_argument("--row-margin-min", type=float, default=0.05)
    ap.add_argument("--col-margin-min", type=float, default=0.03)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.full_pair_matrix))
    policies = build_policies(args)

    all_records = []
    primary = []
    summary_rows = []
    for pname, pol in policies.items():
        selected = []
        for r in rows:
            if policy_accept(r, pol):
                rec = {k: r.get(k, "") for k in selected_fields() if k not in {"anchor_id", "policy"}}
                rec["policy"] = pname
                rec["anchor_id"] = f"{ai(r['tunnel_id'])}_{ai(r['pre_track'])}_{ai(r['post_track'])}"
                selected.append(rec)
        for rec in selected:
            all_records.append(rec)
            if pname == "ultra_safe_margin":
                primary.append(rec)
        c = Counter()
        for rec in selected:
            c["selected"] += 1
            c["gt_same"] += ai(rec.get("gt_same", 0))
            c["gt_known"] += ai(rec.get("gt_known", 0))
            c["true_reconnect"] += ai(rec.get("true_reconnect", 0))
            c["false_reconnect"] += ai(rec.get("false_reconnect", 0))
            c["post_rows"] += ai(rec.get("post_rows_forecast", 0))
            c["post_collision_rows"] += ai(rec.get("post_collision_rows", 0))
            c["oracle_rows"] += ai(rec.get("oracle_core_exit_rows", 0))
            c["oracle_collision_rows"] += ai(rec.get("oracle_collision_rows", 0))
        summary_rows.append({
            "policy": pname,
            "selected": c["selected"],
            "gt_known": c["gt_known"],
            "gt_same": c["gt_same"],
            "true_reconnect": c["true_reconnect"],
            "false_reconnect": c["false_reconnect"],
            "diagnostic_precision": safe_div(c["gt_same"], c["gt_known"]),
            "post_rows": c["post_rows"],
            "post_collision_rows": c["post_collision_rows"],
            "post_collision_ratio": safe_div(c["post_collision_rows"], c["post_rows"]),
            "oracle_rows": c["oracle_rows"],
            "oracle_collision_rows": c["oracle_collision_rows"],
            "oracle_collision_ratio": safe_div(c["oracle_collision_rows"], c["oracle_rows"]),
            "params": json.dumps(pol, sort_keys=True),
        })

    fields = selected_fields()
    with (out / "anchor_candidates_all_policies.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_records)
    with (out / "anchor_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(primary)
    sfields = list(summary_rows[0].keys()) if summary_rows else ["policy"]
    with (out / "anchor_candidates_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)

    payload = {"input": args.full_pair_matrix, "policies": policies, "summary": summary_rows, "primary_policy": "ultra_safe_margin"}
    (out / "anchor_candidates_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_03e1a ultra-safe anchor mining",
        "",
        "Primary policy: `ultra_safe_margin`",
        "",
        "| policy | selected | gt_known | gt_same | true_reconnect | false_reconnect | diag_precision | post_rows | post_collision_ratio | oracle_rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['policy']} | {r['selected']} | {r['gt_known']} | {r['gt_same']} | {r['true_reconnect']} | {r['false_reconnect']} | {r['diagnostic_precision']:.4f} | {r['post_rows']} | {r['post_collision_ratio']:.4f} | {r['oracle_rows']} |"
        )
    md.append("")
    md.append("## Primary candidates")
    md.append("")
    if primary:
        for rec in primary:
            md.append(
                f"- `{rec['anchor_id']}`: tunnel {rec['tunnel_id']}, {rec['pre_track']} -> {rec['post_track']}, sim={float(rec['sim']):.4f}, gt_same={rec['gt_same']}, oracle_rows={rec['oracle_core_exit_rows']}"
            )
    else:
        md.append("No primary candidates.")
    (out / "anchor_candidates_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
