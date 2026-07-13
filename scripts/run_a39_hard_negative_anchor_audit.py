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


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def lifecycle_top11_sim060(r):
    return (
        ai(r["pre_track"]) != ai(r["post_track"])
        and af(r["sim"]) >= 0.60
        and ai(r["row_rank"]) == 1
        and ai(r["col_rank"]) == 1
        and ai(r["pre_track_pre_rows"]) >= 3
        and ai(r["post_track_post_rows"]) >= 3
        and ai(r["pre_track_post_rows"]) == 0
        and ai(r["post_track_pre_rows"]) == 0
    )


def parse_trackeval_summary(path: Path):
    if not path.exists():
        return {}
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split()
    values = lines[1].split()
    return dict(zip(header, values))


def max_float(rows, key, default=0.0):
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, default)))
        except Exception:
            pass
    return max(vals) if vals else default


def min_float(rows, key, default=0.0):
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(key, default)))
        except Exception:
            pass
    return min(vals) if vals else default


def run_cmd(cmd, cwd: Path, stdout_path: Path | None = None):
    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as f:
            return subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.PIPE, text=True, check=False)
    return subprocess.run(cmd, cwd=cwd, text=True, check=False)


def eval_tracker(out_dir: Path, tracker_name: str):
    eval_root = out_dir / "eval_mot20_02"
    data_dir = eval_root / "trackers" / tracker_name / "data"
    seqmap_dir = eval_root / "seqmaps"
    data_dir.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "track_results" / "MOT20-02.txt", data_dir / "MOT20-02.txt")
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
    ret = run_cmd(cmd, REPO, out_dir / "eval_stdout.log")
    summary_path = eval_root / "eval" / tracker_name / "pedestrian_summary.txt"
    return ret.returncode, summary_path, parse_trackeval_summary(summary_path), ret.stderr


def decision_for_anchor(row):
    if row["diag_wrong_rows"] > 1:
        return "REJECT_WRONG_ROWS", "diagnostic wrong rows > 1"
    if row["applied_rows"] < 10:
        return "REJECT_TOO_FEW_ROWS", "applied rows < 10"
    if row["skipped_collision_rows"] > 0:
        return "REJECT_COLLISION", "collision skip rows > 0"
    if row["HOTA"] and float(row["HOTA"]) < 68.430:
        return "REJECT_HOTA_DROP", "HOTA below baseline"
    if row["IDSW"] and int(float(row["IDSW"])) > 444:
        return "REJECT_IDSW_SPIKE", "IDSW > baseline + 1"
    return "PASS_MICRO", "micro diagnostic safe"


def main():
    ap = argparse.ArgumentParser(description="A39_03f0 hard-negative anchor gate audit over lifecycle_top11_sim060 anchors.")
    ap.add_argument("--full-pair-matrix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--track-file", default="outputs/spot_runtime_gate_20260628/A37_identity_escrow_association/A37_00b_candidate_logger_seq02/track_results/MOT20-02.txt")
    ap.add_argument("--gt-file", default="datasets/MOT20/train/MOT20-02/gt/gt.txt")
    ap.add_argument("--tunnels-csv", default="outputs/spot_runtime_gate_20260628/A39_identity_suspension_tunnel/A39_00_tunnel_discovery_seq02_ioa050_dur3/tunnel_candidates.csv")
    ap.add_argument("--img-dir", default="datasets/MOT20/train/MOT20-02/img1")
    ap.add_argument("--fast-reid-config", default="external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml")
    ap.add_argument("--fast-reid-weights", default="external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_csv(Path(args.full_pair_matrix))
    anchors = [r for r in pairs if lifecycle_top11_sim060(r)]
    anchors = sorted(anchors, key=lambda r: (ai(r["tunnel_id"]), ai(r["pre_track"]), ai(r["post_track"])))

    anchor_rows = []
    for r in anchors:
        anchor_id = f"{ai(r['tunnel_id'])}_{ai(r['pre_track'])}_{ai(r['post_track'])}"
        ar = {
            "anchor_id": anchor_id,
            "tunnel_id": ai(r["tunnel_id"]),
            "pre_track": ai(r["pre_track"]),
            "post_track": ai(r["post_track"]),
            "target_id": ai(r["pre_track"]),
            "anchor_gt_diag": ai(r["pre_gt"], -1),
            "pre_gt": ai(r["pre_gt"], -1),
            "post_gt": ai(r["post_gt"], -1),
            "gt_same": ai(r["gt_same"], 0),
            "sim": af(r["sim"]),
            "row_margin": af(r["row_margin"]),
            "col_margin": af(r["col_margin"]),
            "row_candidate_count": ai(r["row_candidate_count"]),
            "col_candidate_count": ai(r["col_candidate_count"]),
            "pre_track_pre_rows": ai(r["pre_track_pre_rows"]),
            "pre_track_post_rows": ai(r["pre_track_post_rows"]),
            "post_track_pre_rows": ai(r["post_track_pre_rows"]),
            "post_track_post_rows": ai(r["post_track_post_rows"]),
            "post_collision_ratio": af(r["post_collision_ratio"]),
            "oracle_core_exit_rows_diag": ai(r.get("oracle_core_exit_rows", 0)),
        }
        anchor_rows.append(ar)
    write_csv(out / "anchors_lifecycle_top11_sim060.csv", list(anchor_rows[0].keys()) if anchor_rows else ["anchor_id"], anchor_rows)

    summary_rows = []
    for ar in anchor_rows:
        anchor_id = ar["anchor_id"]
        micro = out / "micro" / f"anchor_{anchor_id}"
        micro.mkdir(parents=True, exist_ok=True)
        if not (args.skip_existing and (micro / "rewrite_summary.json").exists()):
            cmd = [
                sys.executable,
                "scripts/simulate_a39_path_bridge_rewrite.py",
                "--track-file", args.track_file,
                "--gt-file", args.gt_file,
                "--tunnels-csv", args.tunnels_csv,
                "--img-dir", args.img_dir,
                "--fast-reid-config", args.fast_reid_config,
                "--fast-reid-weights", args.fast_reid_weights,
                "--out-dir", str(micro),
                "--tunnel-id", str(ar["tunnel_id"]),
                "--pre-anchor", str(ar["pre_track"]),
                "--post-anchor", str(ar["post_track"]),
                "--target-id", str(ar["target_id"]),
                "--anchor-gt", str(ar["anchor_gt_diag"]),
                "--device", args.device,
            ]
            ret = run_cmd(cmd, REPO, micro / "path_builder_stdout.log")
            if ret.returncode != 0:
                (micro / "path_builder_stderr.log").write_text(ret.stderr or "", encoding="utf-8")
        eval_code, eval_summary_path, metrics, eval_stderr = eval_tracker(micro, f"A39_03f0_{anchor_id}")
        if eval_stderr:
            (micro / "eval_stderr.log").write_text(eval_stderr, encoding="utf-8")

        rw = json.loads((micro / "rewrite_summary.json").read_text(encoding="utf-8")) if (micro / "rewrite_summary.json").exists() else {}
        selected_frags = read_csv(micro / "selected_fragments.csv") if (micro / "selected_fragments.csv").exists() else []
        gap_rows = read_csv(micro / "gap_rows.csv") if (micro / "gap_rows.csv").exists() else []
        bridge_candidates = read_csv(micro / "bridge_candidates.csv") if (micro / "bridge_candidates.csv").exists() else []
        selected_bridge = [r for r in selected_frags if r.get("selected_stage") == "bridge_fragment"]
        selected_high = [r for r in selected_frags if r.get("selected_stage") == "high_reid"]
        row = dict(ar)
        row.update({
            "run_dir": str(micro),
            "selected_fragments": "|".join(r.get("fragment_key", "") for r in selected_frags),
            "selected_fragment_count": int(rw.get("selected_fragment_count", 0)),
            "high_reid_fragment_count": len(selected_high),
            "bridge_fragment_count": len(selected_bridge),
            "gap_row_count": int(rw.get("gap_row_count", 0)),
            "planned_rows": int(rw.get("planned_rows", 0)),
            "applied_rows": int(rw.get("applied_rows", 0)),
            "skipped_collision_rows": int(rw.get("skipped_collision_rows", 0)),
            "diag_same_rows": int(rw.get("gt_diag_selected_rows_same_anchor", 0)),
            "diag_wrong_rows": int(rw.get("gt_diag_selected_rows_wrong_or_unknown", 0)),
            "diag_wrong_ratio": safe_div(int(rw.get("gt_diag_selected_rows_wrong_or_unknown", 0)), int(rw.get("planned_rows", 0))),
            "correct_fragment_count_diag": int(rw.get("correct_fragment_count_diag", 0)),
            "wrong_fragment_count_diag": int(rw.get("wrong_fragment_count_diag", 0)),
            "correct_gap_rows_diag": int(rw.get("correct_gap_rows_diag", 0)),
            "wrong_gap_rows_diag": int(rw.get("wrong_gap_rows_diag", 0)),
            "max_bridge_score_selected": max_float(selected_bridge, "bridge_score", 0.0),
            "min_bridge_sim_selected": min_float(selected_bridge, "sim_to_anchor", 0.0),
            "max_gap_dist": max_float(gap_rows, "dist", 0.0),
            "min_gap_height_ratio": min_float(gap_rows, "height_ratio", 0.0),
            "max_gap_height_ratio": max_float(gap_rows, "height_ratio", 0.0),
            "bridge_candidate_count": len(bridge_candidates),
            "accepted_bridge_candidate_count": sum(1 for r in bridge_candidates if ai(r.get("accepted", 0)) == 1),
            "HOTA": metrics.get("HOTA", ""),
            "AssA": metrics.get("AssA", ""),
            "IDF1": metrics.get("IDF1", ""),
            "MOTA": metrics.get("MOTA", ""),
            "IDSW": metrics.get("IDSW", ""),
            "Frag": metrics.get("Frag", ""),
        })
        dec, reason = decision_for_anchor(row)
        row["decision"] = dec
        row["reject_reason"] = reason
        summary_rows.append(row)

    fieldnames = list(summary_rows[0].keys()) if summary_rows else ["anchor_id"]
    write_csv(out / "hard_negative_summary.csv", fieldnames, summary_rows)

    true_anchor = [r for r in summary_rows if r["gt_same"] == 1]
    false_anchors = [r for r in summary_rows if r["gt_same"] == 0]
    false_rejected = [r for r in false_anchors if str(r["decision"]).startswith("REJECT")]
    payload = {
        "input": args.full_pair_matrix,
        "anchor_count": len(summary_rows),
        "true_anchor_count_diag": len(true_anchor),
        "false_anchor_count_diag": len(false_anchors),
        "false_rejected_count": len(false_rejected),
        "summary": summary_rows,
    }
    (out / "hard_negative_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# A39_03f0 hard-negative anchor gate audit",
        "",
        "Input anchors: lifecycle_top11_sim060 from A39_03a_v2 full pair matrix.",
        "",
        "| anchor | gt_same | sim | selected_fragments | planned | wrong_rows | gap_rows | HOTA | IDF1 | IDSW | decision |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['anchor_id']} | {r['gt_same']} | {float(r['sim']):.4f} | {r['selected_fragments']} | {r['planned_rows']} | {r['diag_wrong_rows']} | {r['gap_row_count']} | {r['HOTA']} | {r['IDF1']} | {r['IDSW']} | {r['decision']} |"
        )
    md.extend([
        "",
        "## Diagnostic conclusion",
        "",
        f"- total anchors: {len(summary_rows)}",
        f"- true anchors: {len(true_anchor)}",
        f"- false anchors: {len(false_anchors)}",
        f"- false anchors rejected by diagnostic safety rule: {len(false_rejected)} / {len(false_anchors)}",
    ])
    if len(false_rejected) == len(false_anchors) and true_anchor:
        md.append("- Result: hard-negative audit can separate the known true anchor from the four known false anchors using diagnostic wrong-row safety.")
    else:
        md.append("- Result: hard-negative audit does not yet separate true/false anchors cleanly.")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
