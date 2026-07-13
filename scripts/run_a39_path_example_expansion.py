#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
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


def anchor_id(tunnel_id, pre_track, post_track):
    return f"{ai(tunnel_id)}_{ai(pre_track)}_{ai(post_track)}"


def parse_metrics(path: Path):
    if not path.exists():
        return {}
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 2:
        return {}
    return dict(zip(lines[0].split(), lines[1].split()))


def run_cmd(cmd, cwd: Path, stdout_path: Path, stderr_path: Path):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        return subprocess.run(cmd, cwd=cwd, stdout=out, stderr=err, text=True, check=False)


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
    run_cmd(cmd, REPO, run_dir / "eval_stdout.log", run_dir / "eval_stderr.log")
    return parse_metrics(eval_root / "eval" / tracker_name / "pedestrian_summary.txt")


def path_label(row):
    return int(
        ai(row.get("diag_wrong_rows")) <= 1
        and ai(row.get("skipped_collision_rows")) == 0
        and ai(row.get("applied_rows")) >= 10
        and ai(row.get("IDSW"), 999999) <= BASE_IDSW
        and af(row.get("HOTA")) >= BASE_HOTA
        and af(row.get("IDF1")) >= BASE_IDF1
    )


def reject_reason(row):
    if row.get("run_status") and row.get("run_status") != "ok":
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


def rule_strict(row):
    return int(
        ai(row.get("planned_rows")) >= 50
        and ai(row.get("selected_fragment_count")) >= 2
        and ai(row.get("high_reid_fragment_count")) >= 1
        and ai(row.get("bridge_fragment_count")) >= 1
        and ai(row.get("gap_row_count")) <= 10
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_no_bridge(row):
    return int(
        ai(row.get("planned_rows")) >= 30
        and ai(row.get("selected_fragment_count")) >= 1
        and ai(row.get("high_reid_fragment_count")) >= 1
        and ai(row.get("skipped_collision_rows")) == 0
    )


def rule_oracle_diag(row):
    return int(rule_current(row) and ai(row.get("diag_wrong_rows")) <= 1)


def summarize_rule(rows, rule_name, fn):
    tp = fp = tn = fnn = 0
    for r in rows:
        pred = fn(r)
        label = ai(r.get("label_safe_to_rewrite"))
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        elif not pred and label:
            fnn += 1
    return {
        "rule_name": rule_name,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fnn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fnn),
        "accepted_count": tp + fp,
        "safe_count": tp + fnn,
    }


def select_candidates(anchor_examples, false_top_n: int):
    by_anchor = {r["anchor_id"]: r for r in anchor_examples}
    selected = {}

    def add(r, source):
        aid = r.get("anchor_id") or anchor_id(r.get("tunnel_id"), r.get("pre_track"), r.get("post_track"))
        rec = dict(r)
        rec["anchor_id"] = aid
        if aid in selected:
            if source not in selected[aid]["candidate_source"].split("|"):
                selected[aid]["candidate_source"] += "|" + source
        else:
            rec["candidate_source"] = source
            selected[aid] = rec

    for r in anchor_examples:
        if ai(r.get("label_true_reconnect")) == 1:
            add(r, "true_reconnect_all")

    known = {
        "47_104_112": {"tunnel_id": 47, "pre_track": 104, "post_track": 112, "pre_gt": 55, "post_gt": 183, "gt_same": 0},
        "104_26_266": {"tunnel_id": 104, "pre_track": 26, "post_track": 266, "pre_gt": 176, "post_gt": 178, "gt_same": 0},
        "148_373_394": {"tunnel_id": 148, "pre_track": 373, "post_track": 394, "pre_gt": 48, "post_gt": 91, "gt_same": 0},
        "202_530_542": {"tunnel_id": 202, "pre_track": 530, "post_track": 542, "pre_gt": 117, "post_gt": 122, "gt_same": 0},
        "104_17_266": {"tunnel_id": 104, "pre_track": 17, "post_track": 266, "pre_gt": 178, "post_gt": 178, "gt_same": 1},
        "104_26_489": {"tunnel_id": 104, "pre_track": 26, "post_track": 489, "pre_gt": 176, "post_gt": 176, "gt_same": 1},
    }
    for aid, info in known.items():
        if aid in by_anchor:
            add(by_anchor[aid], "known_hard_or_recovered")
        else:
            rec = dict(info)
            rec.update({"anchor_id": aid, "sim": "", "row_rank": "", "col_rank": "", "label_true_reconnect": int(info.get("gt_same", 0)), "label_false_reconnect": int(not info.get("gt_same", 0))})
            add(rec, "known_hard_or_recovered")

    if false_top_n > 0:
        false_rows = [r for r in anchor_examples if ai(r.get("label_false_reconnect")) == 1]
        # High sim false reconnects.
        for r in sorted(false_rows, key=lambda x: af(x.get("sim")), reverse=True)[:false_top_n]:
            add(r, "false_high_sim_top")
        # Lifecycle + top11 false reconnects.
        false_life = [r for r in false_rows if ai(r.get("feature_lifecycle_suspension")) == 1 and ai(r.get("feature_top11")) == 1]
        for r in sorted(false_life, key=lambda x: af(x.get("sim")), reverse=True)[:false_top_n]:
            add(r, "false_lifecycle_top11")

    return list(selected.values())


def existing_run_for_anchor(aid: str, a39_root: Path):
    candidates = [
        a39_root / "A39_03f0_hard_negative_anchor_gate_audit" / "micro" / f"anchor_{aid}",
        a39_root / "A39_03h_anchor_absence_audit_tunnels_104_22" / "micro_recovered" / f"anchor_{aid}",
    ]
    if aid == "12_9_71":
        candidates.extend([
            a39_root / "A39_03e1_ultrasafe_anchors_seq02" / "micro" / "tunnel_12_9_71",
            a39_root / "A39_03e0_path_bridge_tunnel12_no_gt",
        ])
    for c in candidates:
        if (c / "rewrite_summary.json").exists():
            return c
    return None


def summarize_run(anchor, run_dir: Path, source_run_status: str, candidate_source: str):
    aid = anchor.get("anchor_id") or anchor_id(anchor.get("tunnel_id"), anchor.get("pre_track"), anchor.get("post_track"))
    rw = {}
    if (run_dir / "rewrite_summary.json").exists():
        rw = json.loads((run_dir / "rewrite_summary.json").read_text(encoding="utf-8"))
    selected = read_csv(run_dir / "selected_fragments.csv")
    gaps = read_csv(run_dir / "gap_rows.csv")
    metrics = {}
    # Use any existing pedestrian summary.
    for p in run_dir.glob("eval_mot20_02/eval/*/pedestrian_summary.txt"):
        metrics = parse_metrics(p)
        if metrics:
            break
    row = {
        "anchor_id": aid,
        "candidate_source": candidate_source,
        "run_status": source_run_status,
        "source_run_dir": str(run_dir),
        "tunnel_id": ai(anchor.get("tunnel_id")),
        "pre_track": ai(anchor.get("pre_track")),
        "post_track": ai(anchor.get("post_track")),
        "target_id": ai(anchor.get("pre_track")),
        "pre_gt": ai(anchor.get("pre_gt", -1), -1),
        "post_gt": ai(anchor.get("post_gt", -1), -1),
        "gt_same": ai(anchor.get("gt_same", 0)),
        "sim": anchor.get("sim", ""),
        "row_rank": anchor.get("row_rank", ""),
        "col_rank": anchor.get("col_rank", ""),
        "row_margin": anchor.get("row_margin", ""),
        "col_margin": anchor.get("col_margin", ""),
        "feature_lifecycle_suspension": anchor.get("feature_lifecycle_suspension", ""),
        "feature_top11": anchor.get("feature_top11", ""),
        "selected_fragments": "|".join(rw.get("selected_fragments", [])) if rw else "",
        "selected_fragment_count": ai(rw.get("selected_fragment_count", len(selected))),
        "high_reid_fragment_count": sum(1 for r in selected if r.get("selected_stage") == "high_reid"),
        "bridge_fragment_count": sum(1 for r in selected if r.get("selected_stage") == "bridge_fragment"),
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
        "HOTA": metrics.get("HOTA", ""),
        "AssA": metrics.get("AssA", ""),
        "IDF1": metrics.get("IDF1", ""),
        "MOTA": metrics.get("MOTA", ""),
        "IDSW": metrics.get("IDSW", ""),
        "Frag": metrics.get("Frag", ""),
    }
    bridge = [r for r in selected if r.get("selected_stage") == "bridge_fragment"]
    if bridge:
        row["max_bridge_score_selected"] = max(af(r.get("bridge_score")) for r in bridge)
        row["min_bridge_sim_selected"] = min(af(r.get("sim_to_anchor")) for r in bridge)
    if gaps:
        row["max_gap_dist"] = max(af(r.get("dist")) for r in gaps)
        row["min_gap_height_ratio"] = min(af(r.get("height_ratio")) for r in gaps)
        row["max_gap_height_ratio"] = max(af(r.get("height_ratio")) for r in gaps)
    row["label_safe_to_rewrite"] = path_label(row)
    row["reject_reason"] = reject_reason(row)
    return row


def main():
    ap = argparse.ArgumentParser(description="A39_04b: expand path-level examples and evaluate rule baselines.")
    ap.add_argument("--a39-root", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel")
    ap.add_argument("--anchor-examples", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04_learned_path_gate_dataset/anchor_examples.csv")
    ap.add_argument("--out-dir", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_04b_rule_baseline_and_path_expansion")
    ap.add_argument("--false-top-n", type=int, default=0)
    ap.add_argument("--max-new-runs", type=int, default=999)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    a39_root = Path(args.a39_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    anchor_examples = read_csv(Path(args.anchor_examples))
    candidates = select_candidates(anchor_examples, args.false_top_n)
    # Sort: true reconnect first by sim desc, then known/recovered/hard.
    candidates = sorted(candidates, key=lambda r: (0 if ai(r.get("label_true_reconnect")) == 1 else 1, -af(r.get("sim"))))
    manifest_fields = []
    seen = set()
    for r in candidates:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                manifest_fields.append(k)
    write_csv(out / "candidate_anchor_manifest.csv", manifest_fields, candidates)

    summary_rows = []
    new_runs = 0
    for c in candidates:
        aid = c["anchor_id"]
        run_dir = existing_run_for_anchor(aid, a39_root)
        status = "reused_existing"
        if run_dir is None:
            run_dir = out / "micro" / f"anchor_{aid}"
            if not (args.skip_existing and (run_dir / "rewrite_summary.json").exists()):
                if new_runs >= args.max_new_runs:
                    summary_rows.append({"anchor_id": aid, "candidate_source": c.get("candidate_source", ""), "run_status": "not_run_max_new_runs", "tunnel_id": c.get("tunnel_id"), "pre_track": c.get("pre_track"), "post_track": c.get("post_track"), "pre_gt": c.get("pre_gt"), "post_gt": c.get("post_gt"), "gt_same": c.get("gt_same"), "sim": c.get("sim", "")})
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
                ret = run_cmd(cmd, REPO, run_dir / "path_builder_stdout.log", run_dir / "path_builder_stderr.log")
                status = "ok" if ret.returncode == 0 else f"path_builder_failed_{ret.returncode}"
                new_runs += 1
            else:
                status = "reused_out_dir"
        if (run_dir / "track_results" / "MOT20-02.txt").exists():
            eval_run(run_dir, f"A39_04b_{aid}")
        row = summarize_run(c, run_dir, status, c.get("candidate_source", "")) if (run_dir / "rewrite_summary.json").exists() else {"anchor_id": aid, "candidate_source": c.get("candidate_source", ""), "run_status": status, "tunnel_id": c.get("tunnel_id"), "pre_track": c.get("pre_track"), "post_track": c.get("post_track"), "pre_gt": c.get("pre_gt"), "post_gt": c.get("post_gt"), "gt_same": c.get("gt_same"), "sim": c.get("sim", ""), "label_safe_to_rewrite": 0, "reject_reason": status}
        summary_rows.append(row)
        # Incremental write for resume/visibility.
        fields = []
        seenf = set()
        for rr in summary_rows:
            for k in rr.keys():
                if k not in seenf:
                    seenf.add(k)
                    fields.append(k)
        write_csv(out / "path_expansion_summary.csv", fields, summary_rows)

    # Write final path examples v2.
    fields = []
    seenf = set()
    for r in summary_rows:
        for k in r.keys():
            if k not in seenf:
                seenf.add(k)
                fields.append(k)
    write_csv(out / "path_transaction_examples_v2.csv", fields, summary_rows)

    rule_rows = [
        summarize_rule(summary_rows, "rule_current", rule_current),
        summarize_rule(summary_rows, "rule_strict", rule_strict),
        summarize_rule(summary_rows, "rule_no_bridge", rule_no_bridge),
        summarize_rule(summary_rows, "rule_oracle_diag_upper", rule_oracle_diag),
    ]
    write_csv(out / "rule_baseline_report.csv", list(rule_rows[0].keys()), rule_rows)
    (out / "rule_confusion_matrix.json").write_text(json.dumps({r["rule_name"]: r for r in rule_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "candidate_count": len(candidates),
        "path_examples": len(summary_rows),
        "safe_path_examples": sum(ai(r.get("label_safe_to_rewrite")) for r in summary_rows),
        "unsafe_path_examples": sum(1 - ai(r.get("label_safe_to_rewrite")) for r in summary_rows),
        "new_runs": new_runs,
        "rule_report": rule_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# A39_04b Rule Baseline and Path Example Expansion",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Path examples",
        "",
        "| anchor | source | gt_same | sim | planned | applied | wrong | skip | HOTA | IDF1 | IDSW | label | reason |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        md.append(f"| {r.get('anchor_id','')} | {r.get('candidate_source','')} | {r.get('gt_same','')} | {r.get('sim','')} | {r.get('planned_rows','')} | {r.get('applied_rows','')} | {r.get('diag_wrong_rows','')} | {r.get('skipped_collision_rows','')} | {r.get('HOTA','')} | {r.get('IDF1','')} | {r.get('IDSW','')} | {r.get('label_safe_to_rewrite','')} | {r.get('reject_reason','')} |")
    md.extend(["", "## Rule report", "", "| rule | tp | fp | tn | fn | precision | recall |", "|---|---:|---:|---:|---:|---:|---:|"])
    for r in rule_rows:
        md.append(f"| {r['rule_name']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} |")
    (out / "data_gap_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
