#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEEP_ROOT = REPO_ROOT / "external" / "Deep-OC-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]

PROBE_SUMMARY_FIELDS = [
    "seq_name",
    "frames_seen",
    "preassoc_stale_probe_rows",
    "preassoc_stale_probe_candidate_rows",
    "preassoc_stale_probe_positive_gt_rows",
    "preassoc_stale_probe_best_det_unmatched_rows",
    "preassoc_stale_probe_best_det_matched_other_rows",
    "exported_rows",
    "mean_best_box_iou",
    "max_best_box_iou",
    "mean_best_combined_score",
    "max_best_combined_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded Deep-OC-SORT pre-association stale-continuation probe on one or more MOT17 val sequences.")
    parser.add_argument("--seq-name", default="MOT17-05-FRCNN")
    parser.add_argument("--seq-names", nargs="+", default=None)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--preassoc-stale-probe-min-time-since-update", type=int, default=2)
    parser.add_argument("--preassoc-stale-probe-max-time-since-update", type=int, default=12)
    parser.add_argument("--preassoc-stale-probe-min-hits", type=int, default=6)
    parser.add_argument("--preassoc-stale-probe-min-box-iou", type=float, default=0.5)
    parser.add_argument("--preassoc-stale-probe-min-combined-score", type=float, default=0.0)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def run_step(cmd: List[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_preassoc_stale_probe.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_preassoc_probe",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_probe",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    return [str(args.seq_name)]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def ensure_success(step: str, return_code: int, rows: List[Dict[str, object]], summary_csv: Path, out_dir: Path, log_path: Path, notes: str) -> None:
    update_row(
        rows,
        step,
        status=("success" if return_code == 0 else "failed"),
        finished_at=now_iso(),
        out_dir=str(out_dir),
        summary_csv=str(summary_csv),
        log_path=str(log_path),
        notes=notes,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    if return_code != 0:
        raise RuntimeError(f"Step failed: {step}")


def analyze_probe(
    *,
    seq_names: List[str],
    probe_jsonl: Path,
    runtime_summary_csv: Path,
    probe_rows_csv: Path,
    probe_summary_csv: Path,
) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    if probe_jsonl.is_file():
        with probe_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    row_fieldnames = sorted({key for record in records for key in record.keys()}) if records else [
        "seq_name",
        "frame_id",
        "track_index",
        "track_internal_id",
        "track_gt_id",
        "best_box_det_index",
        "best_box_det_gt_id",
        "best_pred_iou",
        "best_last_iou",
        "best_box_iou",
        "best_combined_det_index",
        "best_combined_det_gt_id",
        "best_combined_score",
        "best_box_det_unmatched",
        "best_box_det_owner_track_index",
        "best_box_det_owner_track_internal_id",
        "best_box_det_owner_track_gt_id",
        "positive_gt_continuation",
        "time_since_update",
        "hit_streak",
        "hits",
        "age",
        "latest_observation_valid",
    ]
    write_rows(probe_rows_csv, row_fieldnames, records)

    runtime_rows = list(csv.DictReader(runtime_summary_csv.open("r", encoding="utf-8", newline=""))) if runtime_summary_csv.is_file() else []
    runtime_by_seq = {str(row.get("seq_name", "")): row for row in runtime_rows}
    records_by_seq: Dict[str, List[Dict[str, object]]] = {str(seq): [] for seq in seq_names}
    for record in records:
        records_by_seq.setdefault(str(record.get("seq_name", "")), []).append(record)

    summary_rows: List[Dict[str, object]] = []
    combined = {
        "seq_name": "COMBINED",
        "frames_seen": 0,
        "preassoc_stale_probe_rows": 0,
        "preassoc_stale_probe_candidate_rows": 0,
        "preassoc_stale_probe_positive_gt_rows": 0,
        "preassoc_stale_probe_best_det_unmatched_rows": 0,
        "preassoc_stale_probe_best_det_matched_other_rows": 0,
        "exported_rows": 0,
        "mean_best_box_iou": 0.0,
        "max_best_box_iou": 0.0,
        "mean_best_combined_score": 0.0,
        "max_best_combined_score": 0.0,
    }
    all_best_box_iou: List[float] = []
    all_best_combined: List[float] = []

    for seq_name in seq_names:
        runtime_row = runtime_by_seq.get(str(seq_name), {})
        seq_records = records_by_seq.get(str(seq_name), [])
        best_box_iou_values = [float(record.get("best_box_iou", 0.0) or 0.0) for record in seq_records]
        best_combined_values = [float(record.get("best_combined_score", 0.0) or 0.0) for record in seq_records]
        row = {
            "seq_name": str(seq_name),
            "frames_seen": int(runtime_row.get("frames_seen", 0) or 0),
            "preassoc_stale_probe_rows": int(runtime_row.get("preassoc_stale_probe_rows", 0) or 0),
            "preassoc_stale_probe_candidate_rows": int(runtime_row.get("preassoc_stale_probe_candidate_rows", 0) or 0),
            "preassoc_stale_probe_positive_gt_rows": int(runtime_row.get("preassoc_stale_probe_positive_gt_rows", 0) or 0),
            "preassoc_stale_probe_best_det_unmatched_rows": int(runtime_row.get("preassoc_stale_probe_best_det_unmatched_rows", 0) or 0),
            "preassoc_stale_probe_best_det_matched_other_rows": int(runtime_row.get("preassoc_stale_probe_best_det_matched_other_rows", 0) or 0),
            "exported_rows": len(seq_records),
            "mean_best_box_iou": (sum(best_box_iou_values) / len(best_box_iou_values)) if best_box_iou_values else 0.0,
            "max_best_box_iou": max(best_box_iou_values) if best_box_iou_values else 0.0,
            "mean_best_combined_score": (sum(best_combined_values) / len(best_combined_values)) if best_combined_values else 0.0,
            "max_best_combined_score": max(best_combined_values) if best_combined_values else 0.0,
        }
        summary_rows.append(row)
        for key in [
            "frames_seen",
            "preassoc_stale_probe_rows",
            "preassoc_stale_probe_candidate_rows",
            "preassoc_stale_probe_positive_gt_rows",
            "preassoc_stale_probe_best_det_unmatched_rows",
            "preassoc_stale_probe_best_det_matched_other_rows",
            "exported_rows",
        ]:
            combined[key] = int(combined[key]) + int(row[key])
        all_best_box_iou.extend(best_box_iou_values)
        all_best_combined.extend(best_combined_values)

    combined["mean_best_box_iou"] = (sum(all_best_box_iou) / len(all_best_box_iou)) if all_best_box_iou else 0.0
    combined["max_best_box_iou"] = max(all_best_box_iou) if all_best_box_iou else 0.0
    combined["mean_best_combined_score"] = (sum(all_best_combined) / len(all_best_combined)) if all_best_combined else 0.0
    combined["max_best_combined_score"] = max(all_best_combined) if all_best_combined else 0.0
    summary_rows.append(combined)
    write_rows(probe_summary_csv, PROBE_SUMMARY_FIELDS, summary_rows)
    return combined


def main() -> None:
    args = parse_args()
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_probe_{timestamp_tag()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = (run_root / "results" / "trackers").resolve()
    summary_csv = run_root / "summary.csv"
    probe_jsonl = run_root / "preassoc_stale_probe.jsonl"
    probe_rows_csv = run_root / "probe_rows.csv"
    probe_summary_csv = run_root / "probe_summary.csv"

    exp_name = f"{run_root.name}_probe"
    track_out = trackers_root / "MOT17-val" / exp_name
    runtime_summary_csv = track_out / "fgas_analysis" / f"{exp_name}_summary.csv"

    rows: List[Dict[str, object]] = [
        {
            "step": "track",
            "name": exp_name,
            "status": "running",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "track.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": f"Deep-OC-SORT raw tracking with pre-association stale probe on {seq_label}",
        },
        {
            "step": "analyze",
            "name": run_root.name,
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "analyze.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Analyze pre-association stale probe exports on {seq_label}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"started preassoc stale probe on {seq_label}", args.registry_csv)

    try:
        track_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            "mot17",
            "--result_folder",
            str(trackers_root),
            "--exp_name",
            exp_name,
            "--seq-filter",
            *seq_names,
            "--post",
            "--grid_off",
            "--new_kf_off",
            "--w_assoc_emb",
            "0.75",
            "--aw_param",
            "0.5",
            "--preassoc-stale-probe-export-jsonl",
            str(probe_jsonl),
            "--preassoc-stale-probe-min-time-since-update",
            str(args.preassoc_stale_probe_min_time_since_update),
            "--preassoc-stale-probe-max-time-since-update",
            str(args.preassoc_stale_probe_max_time_since_update),
            "--preassoc-stale-probe-min-hits",
            str(args.preassoc_stale_probe_min_hits),
            "--preassoc-stale-probe-min-box-iou",
            str(args.preassoc_stale_probe_min_box_iou),
            "--preassoc-stale-probe-min-combined-score",
            str(args.preassoc_stale_probe_min_combined_score),
        ]
        track_log = logs_dir / "track.log"
        return_code = run_step(track_cmd, track_log, cwd=DEEP_ROOT)
        ensure_success("track", return_code, rows, summary_csv, track_out, track_log, f"tracking complete for {seq_label}")

        update_row(rows, "analyze", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        combined = analyze_probe(
            seq_names=seq_names,
            probe_jsonl=probe_jsonl,
            runtime_summary_csv=runtime_summary_csv,
            probe_rows_csv=probe_rows_csv,
            probe_summary_csv=probe_summary_csv,
        )
        analyze_log = logs_dir / "analyze.log"
        analyze_log.write_text(
            "\n".join(
                [
                    f"runtime_summary_csv={runtime_summary_csv}",
                    f"probe_jsonl={probe_jsonl}",
                    f"probe_rows_csv={probe_rows_csv}",
                    f"probe_summary_csv={probe_summary_csv}",
                    f"combined_positive_gt_rows={combined['preassoc_stale_probe_positive_gt_rows']}",
                    f"combined_candidate_rows={combined['preassoc_stale_probe_candidate_rows']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        update_row(
            rows,
            "analyze",
            status="success",
            finished_at=now_iso(),
            out_dir=str(run_root),
            summary_csv=str(summary_csv),
            log_path=str(analyze_log),
            notes=(
                f"analyze complete for {seq_label} "
                f"candidate_rows={combined['preassoc_stale_probe_candidate_rows']} "
                f"positive_gt_rows={combined['preassoc_stale_probe_positive_gt_rows']}"
            ),
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, run_root, "success", f"completed preassoc stale probe on {seq_label}", args.registry_csv)
    except Exception as exc:
        for row in rows:
            if str(row.get("status", "")) == "running":
                row["status"] = "failed"
                row["finished_at"] = now_iso()
                row["notes"] = f"{row.get('notes', '')} | failed: {exc}".strip()
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        append_registry(summary_csv, run_root, "failed", f"preassoc stale probe failed on {seq_label}: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
