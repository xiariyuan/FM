#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def anchor_id(tunnel_id, pre_track, post_track):
    return f"{ai(tunnel_id)}_{ai(pre_track)}_{ai(post_track)}"


def path_safety_label(row):
    wrong = ai(row.get("diag_wrong_rows", row.get("gt_diag_selected_rows_wrong_or_unknown", 0)))
    skipped = ai(row.get("skipped_collision_rows", 0))
    applied = ai(row.get("applied_rows", 0))
    idsw = ai(row.get("IDSW", 999999), 999999)
    hota = af(row.get("HOTA", 0.0), 0.0)
    idf1 = af(row.get("IDF1", 0.0), 0.0)
    if wrong <= 1 and skipped == 0 and applied >= 10 and idsw <= 443 and hota >= 68.430 and idf1 >= 74.413:
        return 1
    return 0


def path_reject_reason(row):
    if ai(row.get("diag_wrong_rows", 0)) > 1:
        return "wrong_rows"
    if ai(row.get("skipped_collision_rows", 0)) > 0:
        return "collision_skips"
    if ai(row.get("applied_rows", 0)) < 10:
        return "too_few_rows"
    if row.get("IDSW", "") and ai(row.get("IDSW"), 999999) > 443:
        return "idsw_worse"
    if row.get("HOTA", "") and af(row.get("HOTA"), 0.0) < 68.430:
        return "hota_drop"
    return "safe"


def lifecycle_suspension(r):
    return int(
        ai(r.get("pre_track")) != ai(r.get("post_track"))
        and ai(r.get("pre_track_pre_rows")) >= 3
        and ai(r.get("post_track_post_rows")) >= 3
        and ai(r.get("pre_track_post_rows")) == 0
        and ai(r.get("post_track_pre_rows")) == 0
    )


def top11(r):
    return int(ai(r.get("row_rank")) == 1 and ai(r.get("col_rank")) == 1)


def ultra_safe_margin(r):
    return int(
        lifecycle_suspension(r)
        and af(r.get("sim")) >= 0.70
        and top11(r)
        and ai(r.get("row_candidate_count")) >= 2
        and ai(r.get("col_candidate_count")) >= 2
        and af(r.get("row_margin")) >= 0.05
        and af(r.get("col_margin")) >= 0.03
        and af(r.get("post_collision_ratio")) <= 0.25
    )


def run_key_from_path(path: Path, root: Path):
    try:
        return str(path.parent.relative_to(root))
    except Exception:
        return str(path.parent)


def collect_path_runs(root: Path):
    runs = []
    # hard-negative summary has ready metrics for f0.
    hard = root / "A39_03f0_hard_negative_anchor_gate_audit" / "hard_negative_summary.csv"
    for r in read_csv(hard):
        r = dict(r)
        r["source_family"] = "A39_03f0_hard_negative"
        r["source_run_dir"] = r.get("run_dir", "")
        runs.append(r)
    # recovered unsafe summary from h.
    rec = root / "A39_03h_anchor_absence_audit_tunnels_104_22" / "micro_recovered_summary.csv"
    for r in read_csv(rec):
        r = dict(r)
        parts = str(r.get("anchor_id", "")).split("_")
        if len(parts) >= 3:
            r["tunnel_id"], r["pre_track"], r["post_track"] = parts[0], parts[1], parts[2]
            r["target_id"] = parts[1]
        r["source_family"] = "A39_03h_recovered_unsafe"
        r["source_run_dir"] = str(root / "A39_03h_anchor_absence_audit_tunnels_104_22" / "micro_recovered" / f"anchor_{r.get('anchor_id')}")
        runs.append(r)
    # e1 micro summary positive.
    e1 = root / "A39_03e1_ultrasafe_anchors_seq02" / "micro_summary.csv"
    for r in read_csv(e1):
        r = dict(r)
        r["source_family"] = "A39_03e1_ultrasafe_positive"
        r["source_run_dir"] = str(root / "A39_03e1_ultrasafe_anchors_seq02" / "micro" / "tunnel_12_9_71")
        runs.append(r)
    # e0 positive if separate.
    e0_json = root / "A39_03e0_path_bridge_tunnel12_no_gt" / "rewrite_summary.json"
    if e0_json.exists():
        s = json.loads(e0_json.read_text(encoding="utf-8"))
        # Metrics are known from eval if summary file exists.
        eval_sum = root / "A39_03e0_path_bridge_tunnel12_no_gt" / "eval_mot20_02" / "eval" / "A39_03e0_path_bridge_t12_no_gt" / "pedestrian_summary.txt"
        metrics = {}
        if eval_sum.exists():
            lines = [x.strip() for x in eval_sum.read_text(encoding="utf-8").splitlines() if x.strip()]
            if len(lines) >= 2:
                metrics = dict(zip(lines[0].split(), lines[1].split()))
        r = {
            "anchor_id": f"{s.get('tunnel_id')}_{s.get('pre_anchor')}_{s.get('post_anchor')}",
            "tunnel_id": s.get("tunnel_id"),
            "pre_track": s.get("pre_anchor"),
            "post_track": s.get("post_anchor"),
            "target_id": s.get("target_id"),
            "selected_fragments": "|".join(s.get("selected_fragments", [])),
            "selected_fragment_count": s.get("selected_fragment_count", 0),
            "gap_row_count": s.get("gap_row_count", 0),
            "planned_rows": s.get("planned_rows", 0),
            "applied_rows": s.get("applied_rows", 0),
            "skipped_collision_rows": s.get("skipped_collision_rows", 0),
            "diag_same_rows": s.get("gt_diag_selected_rows_same_anchor", 0),
            "diag_wrong_rows": s.get("gt_diag_selected_rows_wrong_or_unknown", 0),
            "HOTA": metrics.get("HOTA", ""),
            "AssA": metrics.get("AssA", ""),
            "IDF1": metrics.get("IDF1", ""),
            "MOTA": metrics.get("MOTA", ""),
            "IDSW": metrics.get("IDSW", ""),
            "Frag": metrics.get("Frag", ""),
            "source_family": "A39_03e0_positive",
            "source_run_dir": str(root / "A39_03e0_path_bridge_tunnel12_no_gt"),
        }
        runs.append(r)
    # Deduplicate exact same source run dir + anchor_id.
    seen = set()
    out = []
    for r in runs:
        aid = r.get("anchor_id") or anchor_id(r.get("tunnel_id"), r.get("pre_track"), r.get("post_track"))
        r["anchor_id"] = aid
        key = (r.get("source_family"), r.get("source_run_dir"), aid)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_anchor_examples(pair_matrix, path_runs):
    run_by_anchor = {}
    for r in path_runs:
        aid = r.get("anchor_id") or anchor_id(r.get("tunnel_id"), r.get("pre_track"), r.get("post_track"))
        # Prefer a run with actual applied rows / metrics.
        if aid not in run_by_anchor or ai(r.get("applied_rows")) > ai(run_by_anchor[aid].get("applied_rows")):
            run_by_anchor[aid] = r
    rows = []
    for p in pair_matrix:
        if ai(p.get("pre_track")) == ai(p.get("post_track")):
            pair_kind = "self_continuation"
        elif ai(p.get("gt_known")) and ai(p.get("gt_same")):
            pair_kind = "true_reconnect"
        elif ai(p.get("gt_known")) and not ai(p.get("gt_same")):
            pair_kind = "false_reconnect"
        else:
            pair_kind = "unknown_or_impure"
        aid = anchor_id(p.get("tunnel_id"), p.get("pre_track"), p.get("post_track"))
        run = run_by_anchor.get(aid, {})
        row = {
            "anchor_id": aid,
            "tunnel_id": ai(p.get("tunnel_id")),
            "pre_track": ai(p.get("pre_track")),
            "post_track": ai(p.get("post_track")),
            "pair_kind": pair_kind,
            "label_true_reconnect": int(pair_kind == "true_reconnect"),
            "label_false_reconnect": int(pair_kind == "false_reconnect"),
            "label_unknown_or_impure": int(pair_kind == "unknown_or_impure"),
            "sim": af(p.get("sim")),
            "row_rank": ai(p.get("row_rank")),
            "col_rank": ai(p.get("col_rank")),
            "row_margin": af(p.get("row_margin")),
            "col_margin": af(p.get("col_margin")),
            "row_candidate_count": ai(p.get("row_candidate_count")),
            "col_candidate_count": ai(p.get("col_candidate_count")),
            "hungarian": ai(p.get("hungarian")),
            "pre_gt": ai(p.get("pre_gt"), -1),
            "post_gt": ai(p.get("post_gt"), -1),
            "gt_known": ai(p.get("gt_known")),
            "gt_same": ai(p.get("gt_same")),
            "pre_gt_purity": af(p.get("pre_gt_purity")),
            "post_gt_purity": af(p.get("post_gt_purity")),
            "pre_track_pre_rows": ai(p.get("pre_track_pre_rows")),
            "pre_track_core_rows": ai(p.get("pre_track_core_rows")),
            "pre_track_post_rows": ai(p.get("pre_track_post_rows")),
            "post_track_pre_rows": ai(p.get("post_track_pre_rows")),
            "post_track_core_rows": ai(p.get("post_track_core_rows")),
            "post_track_post_rows": ai(p.get("post_track_post_rows")),
            "bottom_rank_delta": ai(p.get("bottom_rank_delta")),
            "height_rank_delta": ai(p.get("height_rank_delta")),
            "height_ratio": af(p.get("height_ratio")),
            "center_delta_norm": af(p.get("center_delta_norm")),
            "post_rows_forecast": ai(p.get("post_rows_forecast")),
            "post_collision_rows": ai(p.get("post_collision_rows")),
            "post_collision_ratio": af(p.get("post_collision_ratio")),
            "oracle_core_exit_rows": ai(p.get("oracle_core_exit_rows")),
            "oracle_collision_rows": ai(p.get("oracle_collision_rows")),
            "oracle_collision_ratio": af(p.get("oracle_collision_ratio")),
            "feature_lifecycle_suspension": lifecycle_suspension(p),
            "feature_top11": top11(p),
            "feature_ultra_safe_margin": ultra_safe_margin(p),
            "path_run_available": int(bool(run)),
            "path_planned_rows": ai(run.get("planned_rows")),
            "path_applied_rows": ai(run.get("applied_rows")),
            "path_skipped_collision_rows": ai(run.get("skipped_collision_rows")),
            "path_diag_same_rows": ai(run.get("diag_same_rows", run.get("gt_diag_selected_rows_same_anchor", 0))),
            "path_diag_wrong_rows": ai(run.get("diag_wrong_rows", run.get("gt_diag_selected_rows_wrong_or_unknown", 0))),
            "path_selected_fragment_count": ai(run.get("selected_fragment_count")),
            "path_high_reid_fragment_count": ai(run.get("high_reid_fragment_count")),
            "path_bridge_fragment_count": ai(run.get("bridge_fragment_count")),
            "path_gap_row_count": ai(run.get("gap_row_count")),
            "path_HOTA": run.get("HOTA", ""),
            "path_IDF1": run.get("IDF1", ""),
            "path_IDSW": run.get("IDSW", ""),
            "label_safe_to_rewrite": path_safety_label(run) if run else 0,
            "path_reject_reason": path_reject_reason(run) if run else "not_run",
        }
        rows.append(row)
    return rows


def build_path_examples(path_runs):
    rows = []
    for r in path_runs:
        aid = r.get("anchor_id") or anchor_id(r.get("tunnel_id"), r.get("pre_track"), r.get("post_track"))
        row = {
            "anchor_id": aid,
            "source_family": r.get("source_family", ""),
            "source_run_dir": r.get("source_run_dir", ""),
            "tunnel_id": ai(r.get("tunnel_id")),
            "pre_track": ai(r.get("pre_track")),
            "post_track": ai(r.get("post_track")),
            "target_id": ai(r.get("target_id", r.get("pre_track", 0))),
            "selected_fragments": r.get("selected_fragments", ""),
            "selected_fragment_count": ai(r.get("selected_fragment_count")),
            "high_reid_fragment_count": ai(r.get("high_reid_fragment_count")),
            "bridge_fragment_count": ai(r.get("bridge_fragment_count")),
            "gap_row_count": ai(r.get("gap_row_count")),
            "planned_rows": ai(r.get("planned_rows")),
            "applied_rows": ai(r.get("applied_rows")),
            "skipped_collision_rows": ai(r.get("skipped_collision_rows")),
            "diag_same_rows": ai(r.get("diag_same_rows", r.get("gt_diag_selected_rows_same_anchor", 0))),
            "diag_wrong_rows": ai(r.get("diag_wrong_rows", r.get("gt_diag_selected_rows_wrong_or_unknown", 0))),
            "diag_wrong_ratio": safe_div(ai(r.get("diag_wrong_rows", r.get("gt_diag_selected_rows_wrong_or_unknown", 0))), ai(r.get("planned_rows"))),
            "correct_fragment_count_diag": ai(r.get("correct_fragment_count_diag")),
            "wrong_fragment_count_diag": ai(r.get("wrong_fragment_count_diag")),
            "correct_gap_rows_diag": ai(r.get("correct_gap_rows_diag")),
            "wrong_gap_rows_diag": ai(r.get("wrong_gap_rows_diag")),
            "max_bridge_score_selected": af(r.get("max_bridge_score_selected")),
            "min_bridge_sim_selected": af(r.get("min_bridge_sim_selected")),
            "max_gap_dist": af(r.get("max_gap_dist")),
            "min_gap_height_ratio": af(r.get("min_gap_height_ratio")),
            "max_gap_height_ratio": af(r.get("max_gap_height_ratio")),
            "HOTA": r.get("HOTA", ""),
            "AssA": r.get("AssA", ""),
            "IDF1": r.get("IDF1", ""),
            "MOTA": r.get("MOTA", ""),
            "IDSW": r.get("IDSW", ""),
            "Frag": r.get("Frag", ""),
        }
        row["label_safe_to_rewrite"] = path_safety_label(row)
        row["reject_reason"] = path_reject_reason(row)
        rows.append(row)
    return rows


def build_fragment_examples(path_runs):
    rows = []
    for run in path_runs:
        run_dir = Path(run.get("source_run_dir", ""))
        if not run_dir.exists():
            continue
        aid = run.get("anchor_id") or anchor_id(run.get("tunnel_id"), run.get("pre_track"), run.get("post_track"))
        for src_name, selected_flag in [("all_fragments.csv", 0), ("selected_fragments.csv", 1)]:
            for f in read_csv(run_dir / src_name):
                row = dict(f)
                row.update({
                    "anchor_id": aid,
                    "source_family": run.get("source_family", ""),
                    "source_run_dir": str(run_dir),
                    "is_selected_in_path": selected_flag,
                    "label_fragment_same_anchor": ai(f.get("gt_same_as_anchor", 0)),
                    "label_fragment_wrong_or_unknown": int(not ai(f.get("gt_same_as_anchor", 0))),
                })
                rows.append(row)
    # Deduplicate selected/all duplicate rows by source_run_dir + anchor + fragment + selected flag kept as separate? Keep both maybe useful. 
    seen = set()
    dedup = []
    for r in rows:
        key = (r.get("source_run_dir"), r.get("anchor_id"), r.get("fragment_key"), r.get("is_selected_in_path"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def write_docs(out: Path):
    (out / "label_policy.md").write_text(
        "# A39_04a Label Policy\n\n"
        "## Anchor labels\n\n"
        "- `label_true_reconnect=1` when `pre_track != post_track`, GT is known, and `pre_gt == post_gt`.\n"
        "- `label_false_reconnect=1` when `pre_track != post_track`, GT is known, and `pre_gt != post_gt`.\n"
        "- `label_unknown_or_impure=1` when GT labels are missing/impure.\n\n"
        "## Path transaction labels\n\n"
        "`label_safe_to_rewrite=1` only if all of the following hold:\n\n"
        "```text\n"
        "diag_wrong_rows <= 1\n"
        "skipped_collision_rows == 0\n"
        "applied_rows >= 10\n"
        "IDSW <= baseline_IDSW=443\n"
        "HOTA >= baseline_HOTA=68.430\n"
        "IDF1 >= baseline_IDF1=74.413\n"
        "```\n\n"
        "This intentionally prioritizes precision and safety over recall.\n\n"
        "## Fragment labels\n\n"
        "- `label_fragment_same_anchor=1` when GT diagnostic says the fragment majority belongs to the anchor identity.\n"
        "- GT is diagnostic only; these labels are for training/evaluation, not deployment inference.\n",
        encoding="utf-8",
    )
    (out / "feature_schema.md").write_text(
        "# A39_04a Feature Schema\n\n"
        "## anchor_examples.csv\n\n"
        "Features include ReID similarity/rank/margins, lifecycle row counts, geometry deltas, collision forecast, and optional path-builder output features.\n\n"
        "## path_transaction_examples.csv\n\n"
        "Features describe the complete proposed rewrite transaction: selected fragment counts, high-ReID/bridge/gap counts, planned/applied rows, collision skips, diagnostics, and TrackEval metrics.\n\n"
        "## fragment_examples.csv\n\n"
        "Features describe each fragment under an anchor path-builder run: ReID-to-anchor scores, temporal extent, geometry, collision forecast, selected stage, and GT diagnostic label.\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser(description="Build A39_04a learned path gate dataset from accumulated A39 diagnostics.")
    ap.add_argument("--a39-root", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel")
    ap.add_argument("--out-dir", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04_learned_path_gate_dataset")
    args = ap.parse_args()
    root = Path(args.a39_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pair_matrix = read_csv(root / "A39_03a_v2_assignment_dryrun_seq02" / "full_pair_matrix.csv")
    path_runs = collect_path_runs(root)
    anchor_examples = build_anchor_examples(pair_matrix, path_runs)
    path_examples = build_path_examples(path_runs)
    fragment_examples = build_fragment_examples(path_runs)

    write_csv(out / "anchor_examples.csv", list(anchor_examples[0].keys()) if anchor_examples else ["anchor_id"], anchor_examples)
    write_csv(out / "path_transaction_examples.csv", list(path_examples[0].keys()) if path_examples else ["anchor_id"], path_examples)
    # Fragment rows may have many heterogeneous columns; use union of keys.
    ffields = []
    seen = set()
    for r in fragment_examples:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                ffields.append(k)
    write_csv(out / "fragment_examples.csv", ffields if ffields else ["anchor_id"], fragment_examples)

    summary = {
        "anchor_examples": len(anchor_examples),
        "path_transaction_examples": len(path_examples),
        "fragment_examples": len(fragment_examples),
        "anchor_true_reconnect": sum(r["label_true_reconnect"] for r in anchor_examples),
        "anchor_false_reconnect": sum(r["label_false_reconnect"] for r in anchor_examples),
        "anchor_unknown_or_impure": sum(r["label_unknown_or_impure"] for r in anchor_examples),
        "path_safe_to_rewrite": sum(r["label_safe_to_rewrite"] for r in path_examples),
        "path_unsafe_to_rewrite": sum(1 - r["label_safe_to_rewrite"] for r in path_examples),
        "fragment_same_anchor": sum(ai(r.get("label_fragment_same_anchor")) for r in fragment_examples),
        "fragment_wrong_or_unknown": sum(ai(r.get("label_fragment_wrong_or_unknown")) for r in fragment_examples),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_04a Learned Path Gate Dataset",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Files",
        "",
        "- `anchor_examples.csv`",
        "- `path_transaction_examples.csv`",
        "- `fragment_examples.csv`",
        "- `label_policy.md`",
        "- `feature_schema.md`",
        "",
        "## Decision",
        "",
        "This is a dataset scaffold, not a trained model. Path-level examples are still very small, so training should start with interpretable models and strict validation, not a large neural matcher.",
    ]
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_docs(out)
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    main()
