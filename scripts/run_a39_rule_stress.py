#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_HOTA = 68.430
BASE_IDF1 = 74.413
BASE_IDSW = 443


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


def fields_union(rows):
    out, seen = [], set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out or ["anchor_id"]


def anchor_id(tunnel_id, pre_track, post_track):
    return f"{ai(tunnel_id)}_{ai(pre_track)}_{ai(post_track)}"


def parse_metrics(path: Path):
    if not path.exists():
        return {}
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 2:
        return {}
    return dict(zip(lines[0].split(), lines[1].split()))


def run_cmd(cmd, stdout_path: Path, stderr_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        return subprocess.run(cmd, cwd=REPO, stdout=out, stderr=err, text=True, check=False)


def eval_run(run_dir: Path, tracker_name: str):
    if not (run_dir / "track_results" / "MOT20-02.txt").exists():
        return {}
    eval_root = run_dir / "eval_mot20_02"
    data_dir = eval_root / "trackers" / tracker_name / "data"
    seqmap_dir = eval_root / "seqmaps"
    data_dir.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "track_results" / "MOT20-02.txt", data_dir / "MOT20-02.txt")
    (seqmap_dir / "MOT20_train.txt").write_text("name\nMOT20-02\n", encoding="utf-8")
    cmd = [
        sys.executable,
        "TrackEval/scripts/run_mot_challenge.py",
        "--GT_FOLDER", "datasets/MOT20/train",
        "--TRACKERS_FOLDER", str(eval_root / "trackers"),
        "--OUTPUT_FOLDER", str(eval_root / "eval"),
        "--TRACKERS_TO_EVAL", tracker_name,
        "--BENCHMARK", "MOT20",
        "--SPLIT_TO_EVAL", "train",
        "--SEQMAP_FILE", str(seqmap_dir / "MOT20_train.txt"),
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "True",
        "--TRACKER_SUB_FOLDER", "data",
        "--OUTPUT_SUB_FOLDER", "",
        "--PRINT_ONLY_COMBINED", "True",
        "--METRICS", "HOTA", "CLEAR", "Identity",
    ]
    run_cmd(cmd, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log")
    for p in eval_root.glob("eval/*/pedestrian_summary.txt"):
        m = parse_metrics(p)
        if m:
            return m
    return {}


def path_label(row):
    return int(
        ai(row.get("diag_wrong_rows")) <= 1
        and ai(row.get("skipped_collision_rows")) == 0
        and ai(row.get("applied_rows")) >= 10
        and ai(row.get("IDSW", 999999), 999999) <= BASE_IDSW
        and af(row.get("HOTA")) >= BASE_HOTA
        and af(row.get("IDF1")) >= BASE_IDF1
    )


def reject_reason(row):
    if row.get("run_status") and str(row.get("run_status")).startswith("path_builder_failed"):
        return row.get("run_status")
    if ai(row.get("diag_wrong_rows")) > 1:
        return "wrong_rows"
    if ai(row.get("skipped_collision_rows")) > 0:
        return "collision_skips"
    if ai(row.get("applied_rows")) < 10:
        return "too_few_rows"
    if row.get("IDSW") and ai(row.get("IDSW"), 999999) > BASE_IDSW:
        return "idsw_worse"
    if row.get("HOTA") and af(row.get("HOTA")) < BASE_HOTA:
        return "hota_drop"
    if row.get("IDF1") and af(row.get("IDF1")) < BASE_IDF1:
        return "idf1_drop"
    return "safe"


def rule_current(row):
    return int(
        ai(row.get("planned_rows")) >= 30
        and ai(row.get("selected_fragment_count")) >= 2
        and ai(row.get("high_reid_fragment_count")) >= 1
        and ai(row.get("bridge_fragment_count")) >= 1
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_no_bridge(row):
    return int(
        ai(row.get("planned_rows")) >= 30
        and ai(row.get("selected_fragment_count")) >= 1
        and ai(row.get("high_reid_fragment_count")) >= 1
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_direct_mode(row):
    return int(
        ai(row.get("planned_rows")) >= 30
        and ai(row.get("selected_fragment_count")) == 1
        and ai(row.get("high_reid_fragment_count")) == 1
        and ai(row.get("bridge_fragment_count")) == 0
        and ai(row.get("gap_row_count")) == 0
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_bridge_mode(row):
    max_gap = af(row.get("max_gap_dist"), 0.0)
    max_gap_ok = (max_gap == 0.0) or (max_gap <= 0.12)
    return int(
        ai(row.get("planned_rows")) >= 30
        and ai(row.get("selected_fragment_count")) >= 2
        and ai(row.get("high_reid_fragment_count")) >= 1
        and ai(row.get("bridge_fragment_count")) >= 1
        and ai(row.get("gap_row_count")) <= 10
        and max_gap_ok
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_mode_union(row):
    return int(rule_direct_mode(row) or rule_bridge_mode(row))


def precheck_maybe(row):
    # Evaluate all rule candidates; if none pass, TrackEval can be skipped because it cannot be accepted by deployable rules.
    return bool(rule_no_bridge(row) or rule_current(row) or rule_mode_union(row))


def summarize_rule(rows, name, fn):
    tp = fp = tn = fnn = 0
    fps = []
    for r in rows:
        pred = fn(r)
        label = ai(r.get("label_safe_to_rewrite"))
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
            fps.append(r)
        elif not pred and not label:
            tn += 1
        else:
            fnn += 1
    return {
        "rule_name": name,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fnn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fnn),
        "accepted_count": tp + fp,
        "safe_count": tp + fnn,
    }, fps


def add_candidate(dst: OrderedDict, r: dict, source: str):
    aid = r.get("anchor_id") or anchor_id(r.get("tunnel_id"), r.get("pre_track"), r.get("post_track"))
    rec = dict(r)
    rec["anchor_id"] = aid
    rec.setdefault("tunnel_id", aid.split("_")[0])
    rec.setdefault("pre_track", aid.split("_")[1])
    rec.setdefault("post_track", aid.split("_")[2])
    if aid in dst:
        old = dst[aid]
        parts = old.get("candidate_source", "").split("|") if old.get("candidate_source") else []
        if source not in parts:
            parts.append(source)
        old["candidate_source"] = "|".join(parts)
    else:
        rec["candidate_source"] = source
        dst[aid] = rec


def select_candidates(anchor_examples, path_examples, false_top_n, false_topk_n, unknown_top_n):
    dst: OrderedDict[str, dict] = OrderedDict()
    for r in path_examples:
        add_candidate(dst, r, "A39_04b_existing")
    # Keep all true reconnects so stress set is self-contained.
    for r in sorted([x for x in anchor_examples if ai(x.get("label_true_reconnect"))], key=lambda x: -af(x.get("sim"))):
        add_candidate(dst, r, "true_reconnect_all")
    false_rows = [x for x in anchor_examples if ai(x.get("label_false_reconnect"))]
    for r in sorted(false_rows, key=lambda x: -af(x.get("sim")))[:false_top_n]:
        add_candidate(dst, r, "false_high_sim_topN")
    life_top11 = [x for x in false_rows if ai(x.get("feature_lifecycle_suspension")) == 1 and ai(x.get("feature_top11")) == 1]
    for r in sorted(life_top11, key=lambda x: -af(x.get("sim"))):
        add_candidate(dst, r, "false_lifecycle_top11")
    topk = [x for x in false_rows if (ai(x.get("row_rank")) <= 3 or ai(x.get("col_rank")) <= 3) and af(x.get("sim")) >= 0.45]
    for r in sorted(topk, key=lambda x: -af(x.get("sim")))[:false_topk_n]:
        add_candidate(dst, r, "false_topk_near")
    unk = [x for x in anchor_examples if ai(x.get("label_unknown_or_impure"))]
    for r in sorted(unk, key=lambda x: -af(x.get("sim")))[:unknown_top_n]:
        add_candidate(dst, r, "unknown_or_impure_high_sim")
    return list(dst.values())


def known_run(aid: str, a39_root: Path, b_root: Path):
    checks = [
        b_root / "micro" / f"anchor_{aid}",
        a39_root / "A39_03f0_hard_negative_anchor_gate_audit" / "micro" / f"anchor_{aid}",
        a39_root / "A39_03h_anchor_absence_audit_tunnels_104_22" / "micro_recovered" / f"anchor_{aid}",
    ]
    if aid == "12_9_71":
        checks.extend([
            a39_root / "A39_03e1_ultrasafe_anchors_seq02" / "micro" / "tunnel_12_9_71",
            a39_root / "A39_03e0_path_bridge_tunnel12_no_gt",
        ])
    for p in checks:
        if (p / "rewrite_summary.json").exists():
            return p
    return None


def summarize_run(anchor, run_dir: Path, status: str, source: str, eval_if_precheck: bool):
    aid = anchor.get("anchor_id") or anchor_id(anchor.get("tunnel_id"), anchor.get("pre_track"), anchor.get("post_track"))
    rw = {}
    if (run_dir / "rewrite_summary.json").exists():
        rw = json.loads((run_dir / "rewrite_summary.json").read_text(encoding="utf-8"))
    selected = read_csv(run_dir / "selected_fragments.csv")
    gaps = read_csv(run_dir / "gap_rows.csv")
    row = {
        "anchor_id": aid,
        "candidate_source": source,
        "run_status": status,
        "source_run_dir": str(run_dir),
        "tunnel_id": ai(anchor.get("tunnel_id")),
        "pre_track": ai(anchor.get("pre_track")),
        "post_track": ai(anchor.get("post_track")),
        "target_id": ai(anchor.get("pre_track")),
        "pre_gt": ai(anchor.get("pre_gt", -1), -1),
        "post_gt": ai(anchor.get("post_gt", -1), -1),
        "gt_same": ai(anchor.get("gt_same", anchor.get("label_true_reconnect", 0))),
        "sim": anchor.get("sim", ""),
        "row_rank": anchor.get("row_rank", ""),
        "col_rank": anchor.get("col_rank", ""),
        "row_margin": anchor.get("row_margin", ""),
        "col_margin": anchor.get("col_margin", ""),
        "feature_lifecycle_suspension": anchor.get("feature_lifecycle_suspension", ""),
        "feature_top11": anchor.get("feature_top11", ""),
        "selected_fragments": "|".join(rw.get("selected_fragments", [])),
        "selected_fragment_count": ai(rw.get("selected_fragment_count", len(selected))),
        "high_reid_fragment_count": sum(1 for s in selected if s.get("selected_stage") == "high_reid"),
        "bridge_fragment_count": sum(1 for s in selected if s.get("selected_stage") == "bridge_fragment"),
        "gap_row_count": ai(rw.get("gap_row_count", len(gaps))),
        "planned_rows": ai(rw.get("planned_rows", 0)),
        "applied_rows": ai(rw.get("applied_rows", 0)),
        "skipped_collision_rows": ai(rw.get("skipped_collision_rows", 0)),
        "diag_same_rows": ai(rw.get("gt_diag_selected_rows_same_anchor", 0)),
        "diag_wrong_rows": ai(rw.get("gt_diag_selected_rows_wrong_or_unknown", 0)),
        "diag_wrong_ratio": safe_div(ai(rw.get("gt_diag_selected_rows_wrong_or_unknown", 0)), ai(rw.get("planned_rows", 0))),
        "correct_fragment_count_diag": ai(rw.get("correct_fragment_count_diag", 0)),
        "wrong_fragment_count_diag": ai(rw.get("wrong_fragment_count_diag", 0)),
        "correct_gap_rows_diag": ai(rw.get("correct_gap_rows_diag", 0)),
        "wrong_gap_rows_diag": ai(rw.get("wrong_gap_rows_diag", 0)),
        "max_bridge_score_selected": "",
        "min_bridge_sim_selected": "",
        "max_gap_dist": "",
        "min_gap_height_ratio": "",
        "max_gap_height_ratio": "",
        "trackeval_run": 0,
        "HOTA": "",
        "AssA": "",
        "IDF1": "",
        "MOTA": "",
        "IDSW": "",
        "Frag": "",
    }
    bridge = [s for s in selected if s.get("selected_stage") == "bridge_fragment"]
    if bridge:
        row["max_bridge_score_selected"] = max(af(s.get("bridge_score")) for s in bridge)
        row["min_bridge_sim_selected"] = min(af(s.get("sim_to_anchor")) for s in bridge)
    if gaps:
        row["max_gap_dist"] = max(af(s.get("dist")) for s in gaps)
        row["min_gap_height_ratio"] = min(af(s.get("height_ratio")) for s in gaps)
        row["max_gap_height_ratio"] = max(af(s.get("height_ratio")) for s in gaps)
    should_eval = True
    if eval_if_precheck and not precheck_maybe(row):
        should_eval = False
    metrics = {}
    if should_eval and (run_dir / "track_results" / "MOT20-02.txt").exists():
        metrics = eval_run(run_dir, f"A39_04c_{aid}")
        row["trackeval_run"] = 1
    else:
        # Reuse existing metrics if already present; otherwise baseline for non-change/no-precheck rows.
        for p in run_dir.glob("eval_mot20_02/eval/*/pedestrian_summary.txt"):
            metrics = parse_metrics(p)
            if metrics:
                break
        if not metrics and ai(row.get("planned_rows")) == 0:
            metrics = {"HOTA": str(BASE_HOTA), "IDF1": str(BASE_IDF1), "IDSW": str(BASE_IDSW), "MOTA": "90.845", "Frag": "777"}
    for k in ["HOTA", "AssA", "IDF1", "MOTA", "IDSW", "Frag"]:
        if k in metrics:
            row[k] = metrics[k]
    row["label_safe_to_rewrite"] = path_label(row)
    row["reject_reason"] = reject_reason(row)
    row["rule_current"] = rule_current(row)
    row["rule_no_bridge"] = rule_no_bridge(row)
    row["rule_direct_mode"] = rule_direct_mode(row)
    row["rule_bridge_mode"] = rule_bridge_mode(row)
    row["rule_mode_union"] = rule_mode_union(row)
    return row


def main():
    ap = argparse.ArgumentParser(description="A39_04c: candidate expansion false TopN and rule stress.")
    ap.add_argument("--a39-root", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel")
    ap.add_argument("--a39-04b-root", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04b_rule_baseline_and_path_expansion")
    ap.add_argument("--anchor-examples", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04_learned_path_gate_dataset/anchor_examples.csv")
    ap.add_argument("--out-dir", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04c_candidate_expansion_false_topN_and_rule_stress")
    ap.add_argument("--false-top-n", type=int, default=20)
    ap.add_argument("--false-topk-n", type=int, default=20)
    ap.add_argument("--unknown-top-n", type=int, default=10)
    ap.add_argument("--max-new-runs", type=int, default=999)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-if-precheck", action="store_true", default=True)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    a39_root = Path(args.a39_root)
    b_root = Path(args.a39_04b_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor_examples = read_csv(Path(args.anchor_examples))
    path_examples_b = read_csv(b_root / "path_transaction_examples_v2.csv")
    candidates = select_candidates(anchor_examples, path_examples_b, args.false_top_n, args.false_topk_n, args.unknown_top_n)
    write_csv(out / "candidate_anchor_manifest.csv", fields_union(candidates), candidates)

    rows = []
    new_runs = 0
    for c in candidates:
        aid = c.get("anchor_id") or anchor_id(c.get("tunnel_id"), c.get("pre_track"), c.get("post_track"))
        run_dir = known_run(aid, a39_root, b_root)
        status = "reused_existing"
        if run_dir is None:
            run_dir = out / "micro" / f"anchor_{aid}"
            if not (args.skip_existing and (run_dir / "rewrite_summary.json").exists()):
                if new_runs >= args.max_new_runs:
                    row = {"anchor_id": aid, "candidate_source": c.get("candidate_source", ""), "run_status": "not_run_max_new_runs", "label_safe_to_rewrite": 0, "reject_reason": "not_run_max_new_runs"}
                    rows.append(row)
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    "scripts/simulate_a39_path_bridge_rewrite.py",
                    "--track-file", "outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt",
                    "--gt-file", "datasets/MOT20/train/MOT20-02/gt/gt.txt",
                    "--tunnels-csv", "outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv",
                    "--img-dir", "datasets/MOT20/train/MOT20-02/img1",
                    "--fast-reid-config", "external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml",
                    "--fast-reid-weights", "external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth",
                    "--out-dir", str(run_dir),
                    "--tunnel-id", str(ai(c.get("tunnel_id"))),
                    "--pre-anchor", str(ai(c.get("pre_track"))),
                    "--post-anchor", str(ai(c.get("post_track"))),
                    "--target-id", str(ai(c.get("pre_track"))),
                    "--anchor-gt", str(ai(c.get("pre_gt", -1), -1)),
                    "--device", args.device,
                ]
                ret = run_cmd(cmd, run_dir / "path_builder_stdout.log", run_dir / "path_builder_stderr.log")
                status = "ok" if ret.returncode == 0 else f"path_builder_failed_{ret.returncode}"
                new_runs += 1
            else:
                status = "reused_out_dir"
        row = summarize_run(c, run_dir, status, c.get("candidate_source", ""), args.eval_if_precheck) if (run_dir / "rewrite_summary.json").exists() else {"anchor_id": aid, "candidate_source": c.get("candidate_source", ""), "run_status": status, "label_safe_to_rewrite": 0, "reject_reason": status}
        rows.append(row)
        write_csv(out / "path_builder_summary.csv", fields_union(rows), rows)

    write_csv(out / "path_transaction_examples_stress.csv", fields_union(rows), rows)
    rule_defs = [
        ("rule_current", rule_current),
        ("rule_no_bridge", rule_no_bridge),
        ("rule_direct_mode", rule_direct_mode),
        ("rule_bridge_mode", rule_bridge_mode),
        ("rule_mode_union", rule_mode_union),
    ]
    rule_rows = []
    fp_cases = []
    for name, fn in rule_defs:
        rep, fps = summarize_rule(rows, name, fn)
        rule_rows.append(rep)
        for f in fps:
            rec = dict(f)
            rec["rule_name"] = name
            rec["why_dangerous"] = rec.get("reject_reason", "false_positive")
            fp_cases.append(rec)
    safe_cases = [r for r in rows if ai(r.get("label_safe_to_rewrite")) == 1]
    write_csv(out / "rule_stress_report.csv", fields_union(rule_rows), rule_rows)
    write_csv(out / "false_positive_cases.csv", fields_union(fp_cases), fp_cases)
    write_csv(out / "safe_path_cases.csv", fields_union(safe_cases), safe_cases)
    summary = {
        "candidate_count": len(candidates),
        "new_runs": new_runs,
        "path_examples": len(rows),
        "safe_path_examples": len(safe_cases),
        "unsafe_path_examples": len(rows) - len(safe_cases),
        "trackeval_run_count": sum(ai(r.get("trackeval_run")) for r in rows),
        "rule_report": rule_rows,
        "false_positive_cases": len(fp_cases),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_04c Candidate Expansion false TopN and Rule Stress",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Rule stress report",
        "",
        "| rule | tp | fp | tn | fn | precision | recall | accepted | safe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rule_rows:
        md.append(f"| {r['rule_name']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['accepted_count']} | {r['safe_count']} |")
    md.extend(["", "## Safe path cases", "", "| anchor | source | sim | planned | wrong | skip | HOTA | IDF1 | IDSW |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in safe_cases:
        md.append(f"| {r.get('anchor_id')} | {r.get('candidate_source')} | {r.get('sim')} | {r.get('planned_rows')} | {r.get('diag_wrong_rows')} | {r.get('skipped_collision_rows')} | {r.get('HOTA')} | {r.get('IDF1')} | {r.get('IDSW')} |")
    md.extend(["", "## False positives", ""])
    if fp_cases:
        md.extend(["| rule | anchor | source | sim | planned | wrong | skip | reason |", "|---|---|---|---:|---:|---:|---:|---|"])
        for r in fp_cases:
            md.append(f"| {r.get('rule_name')} | {r.get('anchor_id')} | {r.get('candidate_source')} | {r.get('sim')} | {r.get('planned_rows')} | {r.get('diag_wrong_rows')} | {r.get('skipped_collision_rows')} | {r.get('reject_reason')} |")
    else:
        md.append("No false positives in this stress set.")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "data_gap_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
