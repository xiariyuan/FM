#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TRAIN_SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]


def af(v, d=0.0):
    try:
        if v is None or v == "":
            return d
        return float(v)
    except Exception:
        return d


def ai(v, d=0):
    try:
        if v is None or v == "":
            return d
        return int(float(v))
    except Exception:
        return d


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    if not fields:
        fields = ["seq"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_mot(path: Path) -> List[Tuple[int, int, List[str]]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 6:
                out.append((int(float(parts[0])), int(float(parts[1])), parts))
    return out


def source_file(track_dir: Path, seq: str) -> Path:
    p = track_dir / seq / f"{seq}.txt"
    if p.exists():
        return p
    p = track_dir / f"{seq}.txt"
    if p.exists():
        return p
    raise FileNotFoundError(f"missing {seq} under {track_dir}")


def center_xy(row: dict, prefix: str) -> Tuple[float, float]:
    x, y, w, h = af(row[f"{prefix}_x"]), af(row[f"{prefix}_y"]), af(row[f"{prefix}_w"]), af(row[f"{prefix}_h"])
    return x + w / 2.0, y + h / 2.0


def build_candidates(tracklets: List[dict], max_gap: int) -> List[dict]:
    by_seq = defaultdict(list)
    for t in tracklets:
        if t.get("seq") in TRAIN_SEQS:
            by_seq[t["seq"]].append(t)
    rows = []
    for seq, ts in sorted(by_seq.items()):
        ts = sorted(ts, key=lambda r: (ai(r.get("start_frame")), ai(r.get("end_frame")), ai(r.get("track_id"))))
        for a in ts:
            ea = ai(a.get("end_frame"))
            for b in ts:
                sb = ai(b.get("start_frame"))
                if sb <= ea:
                    continue
                gap = sb - ea
                if gap > max_gap:
                    # b is start sorted only globally; once start exceeds max for this a, later b will too.
                    break
                ca = (af(a.get("last_x")) + af(a.get("last_w")) / 2.0, af(a.get("last_y")) + af(a.get("last_h")) / 2.0)
                cb = (af(b.get("first_x")) + af(b.get("first_w")) / 2.0, af(b.get("first_y")) + af(b.get("first_h")) / 2.0)
                center_distance = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                area_a = max(1e-6, af(a.get("last_w")) * af(a.get("last_h")))
                area_b = max(1e-6, af(b.get("first_w")) * af(b.get("first_h")))
                gt_a, gt_b = ai(a.get("dominant_gt"), -1), ai(b.get("dominant_gt"), -2)
                same_gt = int(gt_a > 0 and gt_a == gt_b and af(a.get("dominant_gt_ratio_total")) >= 0.5 and af(b.get("dominant_gt_ratio_total")) >= 0.5)
                bucket = "1_60" if gap <= 60 else ("61_150" if gap <= 150 else "151_300")
                rows.append({
                    "seq": seq,
                    "track_a": str(ai(a.get("track_id"))),
                    "track_b": str(ai(b.get("track_id"))),
                    "gap": gap,
                    "gap_bucket": bucket,
                    "same_gt": same_gt,
                    "dominant_gt_a": gt_a,
                    "dominant_gt_b": gt_b,
                    "matched_ratio_a": a.get("matched_ratio", ""),
                    "matched_ratio_b": b.get("matched_ratio", ""),
                    "dominant_gt_ratio_a": a.get("dominant_gt_ratio_total", ""),
                    "dominant_gt_ratio_b": b.get("dominant_gt_ratio_total", ""),
                    "quality_label_a": a.get("quality_label", ""),
                    "quality_label_b": b.get("quality_label", ""),
                    "len_a": a.get("row_count", ""),
                    "len_b": b.get("row_count", ""),
                    "duration_a": a.get("duration", ""),
                    "duration_b": b.get("duration", ""),
                    "avg_score_a": a.get("avg_score", ""),
                    "avg_score_b": b.get("avg_score", ""),
                    "last_score_a": a.get("last_score", ""),
                    "first_score_b": b.get("first_score", ""),
                    "height_ratio": af(b.get("first_h")) / max(1e-6, af(a.get("last_h"))),
                    "area_ratio": area_b / area_a,
                    "bottom_y_gap": (af(b.get("first_y")) + af(b.get("first_h"))) - (af(a.get("last_y")) + af(a.get("last_h"))),
                    "center_distance": center_distance,
                    "center_distance_per_frame": center_distance / max(1, gap),
                    "start_frame_a": a.get("start_frame", ""),
                    "end_frame_a": a.get("end_frame", ""),
                    "start_frame_b": b.get("start_frame", ""),
                    "end_frame_b": b.get("end_frame", ""),
                })
    return rows


def find(parent: Dict[int, int], x: int) -> int:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def select_links(candidates: List[dict], mode: str) -> List[dict]:
    if mode == "oracle_1_60":
        filt = lambda r: ai(r.get("same_gt")) == 1 and 1 <= ai(r.get("gap")) <= 60
    elif mode == "oracle_61_150_only":
        filt = lambda r: ai(r.get("same_gt")) == 1 and 61 <= ai(r.get("gap")) <= 150
    elif mode == "oracle_151_300_only":
        filt = lambda r: ai(r.get("same_gt")) == 1 and 151 <= ai(r.get("gap")) <= 300
    elif mode == "oracle_1_150":
        filt = lambda r: ai(r.get("same_gt")) == 1 and 1 <= ai(r.get("gap")) <= 150
    elif mode == "oracle_1_300":
        filt = lambda r: ai(r.get("same_gt")) == 1 and 1 <= ai(r.get("gap")) <= 300
    else:
        raise KeyError(mode)

    by_seq = defaultdict(list)
    for r in candidates:
        if filt(r):
            by_seq[r["seq"]].append(r)
    selected_all = []
    for seq, rows in sorted(by_seq.items()):
        # Oracle policy: shorter true edges first, with high-confidence GT ratio as tie-breaker.
        rows = sorted(rows, key=lambda r: (ai(r.get("gap")), -af(r.get("dominant_gt_ratio_a")), -af(r.get("dominant_gt_ratio_b")), ai(r.get("track_a")), ai(r.get("track_b"))))
        parent: Dict[int, int] = {}
        used_successor, used_predecessor = set(), set()
        for r in rows:
            a, b = ai(r["track_a"]), ai(r["track_b"])
            if a in used_successor or b in used_predecessor:
                continue
            ra, rb = find(parent, a), find(parent, b)
            if ra == rb:
                continue
            parent[rb] = ra
            used_successor.add(a)
            used_predecessor.add(b)
            q = dict(r)
            q["oracle_mode"] = mode
            q["score"] = 1.0
            selected_all.append(q)
    return selected_all


def link_outputs(selected: List[dict], track_dir: Path, out_dir: Path) -> dict:
    linked_dir = out_dir / "linked_results"
    linked_dir.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in selected:
        by[r["seq"]].append(r)
    by_seq = []
    for seq in TRAIN_SEQS:
        parent: Dict[int, int] = {}
        # Selected is already conflict guarded, but rebuild union for safety.
        for r in by.get(seq, []):
            a, b = ai(r["track_a"]), ai(r["track_b"])
            ra, rb = find(parent, a), find(parent, b)
            if ra != rb:
                parent[rb] = ra
        involved = set()
        for r in by.get(seq, []):
            involved.add(ai(r["track_a"]))
            involved.add(ai(r["track_b"]))
        id_map = {tid: find(parent, tid) for tid in involved}
        src = source_file(track_dir, seq)
        out = linked_dir / f"{seq}.txt"
        mot = []
        for _, tid, parts in read_mot(src):
            p = list(parts)
            p[1] = str(id_map.get(tid, tid))
            mot.append(p)
        mot.sort(key=lambda p: (int(float(p[0])), int(float(p[1])), float(p[2]), float(p[3])))
        with out.open("w", encoding="utf-8") as f:
            for p in mot:
                f.write(",".join(p) + "\n")
        by_seq.append({"seq": seq, "selected_links": len(by.get(seq, [])), "source_rows": len(read_mot(src)), "linked_rows": len(mot)})
    return {"selected_links": len(selected), "by_seq": by_seq}


def run_interp_and_eval(out_dir: Path, tracker_name: str) -> dict:
    interp_cmd = [sys.executable, "scripts/postprocess/linear_interpolate_mot.py", "--input-dir", str(out_dir / "linked_results"), "--output-dir", str(out_dir / "track_results"), "--max-gap", "30", "--summary-json", str(out_dir / "interp_summary.json"), "--summary-csv", str(out_dir / "interp_summary.csv")]
    p = subprocess.run(interp_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_dir / "interp_stdout.log").write_text(p.stdout, encoding="utf-8")
    eval_root = out_dir / "eval_mot20_all_train"
    data = eval_root / "trackers" / tracker_name / "data"
    data.mkdir(parents=True, exist_ok=True)
    for seq in TRAIN_SEQS:
        shutil.copy2(out_dir / "track_results" / f"{seq}.txt", data / f"{seq}.txt")
    seqmap = eval_root / "seqmaps" / "MOT20_train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(TRAIN_SEQS) + "\n", encoding="utf-8")
    eval_cmd = [sys.executable, "TrackEval/scripts/run_mot_challenge.py", "--GT_FOLDER", "datasets/MOT20/train", "--TRACKERS_FOLDER", str(eval_root / "trackers"), "--OUTPUT_FOLDER", str(eval_root / "eval"), "--TRACKERS_TO_EVAL", tracker_name, "--BENCHMARK", "MOT20", "--SPLIT_TO_EVAL", "train", "--SEQMAP_FILE", str(seqmap), "--SKIP_SPLIT_FOL", "True", "--DO_PREPROC", "True", "--TRACKER_SUB_FOLDER", "data", "--OUTPUT_SUB_FOLDER", "", "--PRINT_ONLY_COMBINED", "True", "--METRICS", "HOTA", "CLEAR", "Identity"]
    q = subprocess.run(eval_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_dir / "eval_stdout.log").write_text(q.stdout, encoding="utf-8")
    summary = eval_root / "eval" / tracker_name / "pedestrian_summary.txt"
    metrics = {}
    if summary.exists():
        lines = [x.strip() for x in summary.read_text().splitlines() if x.strip()]
        if len(lines) >= 2:
            metrics = dict(zip(lines[0].split(), lines[1].split()))
    metrics["interp_returncode"] = p.returncode
    metrics["trackeval_returncode"] = q.returncode
    metrics["summary_path"] = str(summary)
    return metrics


def parse_metrics(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return dict(zip(lines[0].split(), lines[1].split())) if len(lines) >= 2 else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracklet-rows", required=True)
    ap.add_argument("--track-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-gap", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracklets = read_csv(Path(args.tracklet_rows))
    candidates = build_candidates(tracklets, args.max_gap)
    write_csv(out / "long_gap_candidates_train.csv", candidates)

    bucket_rows = []
    for bucket_name, pred in [
        ("1_60", lambda r: 1 <= ai(r.get("gap")) <= 60),
        ("61_150", lambda r: 61 <= ai(r.get("gap")) <= 150),
        ("151_300", lambda r: 151 <= ai(r.get("gap")) <= 300),
        ("1_150", lambda r: 1 <= ai(r.get("gap")) <= 150),
        ("1_300", lambda r: 1 <= ai(r.get("gap")) <= 300),
    ]:
        rows = [r for r in candidates if pred(r)]
        true_rows = [r for r in rows if ai(r.get("same_gt")) == 1]
        bucket_rows.append({"bucket": bucket_name, "candidate_count": len(rows), "true_candidate_count": len(true_rows), "true_rate": len(true_rows) / len(rows) if rows else 0.0})
    write_csv(out / "gap_bucket_candidate_summary.csv", bucket_rows)

    modes = ["oracle_1_60", "oracle_61_150_only", "oracle_151_300_only", "oracle_1_150", "oracle_1_300"]
    eval_rows = []
    for mode in modes:
        mode_dir = out / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        selected = select_links(candidates, mode)
        write_csv(mode_dir / "selected_links.csv", selected)
        link_summary = link_outputs(selected, Path(args.track_dir), mode_dir)
        metrics = run_interp_and_eval(mode_dir, f"A42_00_{mode}")
        summary = {"mode": mode, "link_summary": link_summary, "metrics": metrics, "decision": "EVAL_DONE" if metrics.get("trackeval_returncode") == 0 else "EVAL_FAILED"}
        (mode_dir / "link_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        row = {"mode": mode, "selected_links": link_summary["selected_links"], **{k: metrics.get(k, "") for k in ["HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDSW", "Frag"]}, "summary_path": metrics.get("summary_path", "")}
        eval_rows.append(row)
    # Add refs for direct comparison.
    refs = {
        "A23_P14_thr015": "outputs/spot_runtime_gate_20260628/A23_appearance_aflink/A23_04_oof_apply_P14_thr015/eval_mot20_all_train/eval/P14_thr015/pedestrian_summary.txt",
        "A41_05_best": "outputs/spot_runtime_gate_20260628/A41_association_debt_global_tracker/A41_05_teacher_recovery_hybrid/A41_05b_hybrid_eval/hybrid_union_score0p15_app0p60_risk7_rank2/eval_mot20_all_train/eval/A41_05_hybrid_union_score0p15_app0p60_risk7_rank2/pedestrian_summary.txt",
        "A36_gap150_old": "outputs/spot_runtime_gate_20260628/A36_error_upper_bound_diagnosis/A36_02_gap150_oracle_true_candidates/eval_mot20_all_train/eval/A36_02_gap150_oracle_true_candidates/pedestrian_summary.txt",
    }
    for name, path in refs.items():
        m = parse_metrics(path)
        if m:
            eval_rows.append({"mode": name, "selected_links": "", **{k: m.get(k, "") for k in ["HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDSW", "Frag"]}, "summary_path": path})
    write_csv(out / "oracle_bound_summary.csv", eval_rows)

    decision = {
        "candidate_buckets": bucket_rows,
        "oracle_eval_rows": eval_rows,
        "decision": "A42_00_ORACLE_BOUND_DONE",
        "next": "If oracle_1_300 or oracle_1_150 shows meaningful room over A41_05, build A42_01 long-gap no-GT candidate manifest with ReID/zone/rank features.",
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = ["# A42_00 Long-Gap Oracle Bound", "", "## Decision", "", "```text", decision["decision"], "```", "", "## Bucket candidate summary", ""]
    for r in bucket_rows:
        md.append(f"- {r['bucket']}: candidates={r['candidate_count']}, true={r['true_candidate_count']}, true_rate={r['true_rate']:.6f}")
    md += ["", "## Eval rows", ""]
    for r in eval_rows:
        md.append(f"- {r['mode']}: HOTA={r.get('HOTA')}, IDF1={r.get('IDF1')}, AssA={r.get('AssA')}, IDSW={r.get('IDSW')}, Frag={r.get('Frag')}, links={r.get('selected_links')}")
    (out / "decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "bucket_rows": bucket_rows, "eval_rows": eval_rows, "decision": decision["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
