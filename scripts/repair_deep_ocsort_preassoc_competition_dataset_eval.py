#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_deep_ocsort_preassoc_competition_dataset_eval import (
    DELTA_FIELDS,
    DEEP_ROOT,
    LOCAL_CONTENTION_SUMMARY_FIELDS,
    METRIC_FIELDS,
    PER_SEQUENCE_FIELDS,
    QUEUE_FIELDS,
    RUNTIME_FIELDS,
    RUNTIME_PER_SEQUENCE_FIELDS,
    append_registry,
    ensure_eval_outputs,
    ensure_tracking_outputs,
    load_per_sequence_metrics,
    load_runtime_rows,
    now_iso,
    parse_summary_txt,
    resolve_runtime_summary,
    summarize_local_contention_export,
    summarize_runtime_rows,
    update_row,
    write_rows,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair and backfill a partially completed Deep-OC-SORT pre-association competition dataset-eval run."
    )
    parser.add_argument("--benchmark", choices=["MOT17", "MOT20", "DanceTrack"], required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_seq_names(track_out: Path) -> List[str]:
    data_dir = track_out / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing track data directory: {data_dir}")
    seq_names = sorted(path.stem for path in data_dir.glob("*.txt"))
    if not seq_names:
        raise FileNotFoundError(f"No sequence txt files under: {data_dir}")
    return seq_names


def max_mtime_iso(paths: List[Path]) -> str:
    latest = max((path.stat().st_mtime for path in paths if path.exists()), default=0.0)
    if latest <= 0.0:
        return now_iso()
    return datetime.fromtimestamp(latest).astimezone().isoformat(timespec="seconds")


def run_step(cmd: List[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def main() -> None:
    args = parse_args()
    benchmark = str(args.benchmark)
    eval_benchmark = {"MOT17": "MOT17", "MOT20": "MOT20", "DanceTrack": "DANCE"}[benchmark]
    split_label = "val" if benchmark == "DanceTrack" else "val_half"
    tracker_split = f"{eval_benchmark}-val"

    run_root = Path(args.run_root).resolve()
    summary_csv = run_root / "summary.csv"
    logs_dir = run_root / "logs"
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    runtime_compare_csv = run_root / "runtime_compare.csv"
    runtime_per_sequence_csv = run_root / "runtime_per_sequence.csv"
    local_contention_summary_csv = run_root / "local_contention_summary.csv"
    local_export_jsonl = run_root / "local_contention_units.jsonl"

    rows = read_rows(summary_csv)
    row_by_step = {str(row.get("step", "")): row for row in rows}
    raw_track_row = row_by_step["raw_track"]
    raw_eval_row = row_by_step["raw_eval"]
    competition_track_row = row_by_step["competition_track"]
    competition_eval_row = row_by_step["competition_eval"]
    compare_row = row_by_step["compare"]

    raw_track_out = Path(str(raw_track_row.get("out_dir", ""))).resolve()
    raw_eval_out = Path(str(raw_eval_row.get("out_dir", ""))).resolve()
    competition_exp = str(competition_track_row.get("name", "")).strip()
    if not competition_exp:
        raise ValueError("Missing competition experiment name in summary.csv")
    competition_track_out = run_root / "results" / "trackers" / tracker_split / competition_exp
    competition_eval_out = run_root / "results" / "trackers" / tracker_split / f"{competition_exp}_post"
    seq_names = infer_seq_names(competition_track_out)
    seq_label = "|".join(seq_names)

    ensure_tracking_outputs(competition_track_out, competition_eval_out, seq_names)

    if str(competition_track_row.get("status", "")) != "success":
        finished_at = max_mtime_iso(
            [
                competition_track_out,
                competition_eval_out,
                competition_track_out / "fgas_analysis",
                competition_track_out / "data",
                run_root / "local_contention_units.jsonl",
            ]
        )
        update_row(
            rows,
            "competition_track",
            status="success",
            out_dir=str(competition_track_out),
            summary_csv=str(summary_csv),
            log_path=str(logs_dir / "competition_track.log"),
            finished_at=finished_at,
            notes=f"repaired tracking success for {seq_label} from existing outputs",
        )
        write_rows(summary_csv, QUEUE_FIELDS, rows)

    competition_eval_log = logs_dir / "competition_eval.log"
    eval_summary_txt = competition_eval_out / "pedestrian_summary.txt"
    eval_detailed_csv = competition_eval_out / "pedestrian_detailed.csv"
    if not eval_summary_txt.is_file() or not eval_detailed_csv.is_file():
        update_row(rows, "competition_eval", status="running", started_at=now_iso(), out_dir=str(competition_eval_out))
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        eval_cmd = [
            sys.executable,
            "external/TrackEval/scripts/run_mot_challenge.py",
            "--BENCHMARK",
            eval_benchmark,
            "--SPLIT_TO_EVAL",
            "val",
            "--GT_FOLDER",
            str(DEEP_ROOT / "results" / "gt"),
            "--TRACKERS_FOLDER",
            str(run_root / "results" / "trackers"),
            "--TRACKERS_TO_EVAL",
            competition_exp + "_post",
            "--SEQ_INFO",
            *seq_names,
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
            "--USE_PARALLEL",
            "False",
            "--PRINT_ONLY_COMBINED",
            "True",
        ]
        rc = run_step(eval_cmd, competition_eval_log, cwd=DEEP_ROOT)
        if rc != 0:
            raise RuntimeError(f"competition eval repair failed rc={rc}")
        ensure_eval_outputs(competition_eval_out)

    update_row(
        rows,
        "competition_eval",
        status="success",
        out_dir=str(competition_eval_out),
        summary_csv=str(summary_csv),
        log_path=str(competition_eval_log),
        started_at=str(competition_eval_row.get("started_at", "") or now_iso()),
        finished_at=max_mtime_iso([eval_summary_txt, eval_detailed_csv]),
        notes=f"repaired competition eval complete for {seq_label}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)

    update_row(rows, "compare", status="running", started_at=now_iso(), out_dir=str(run_root))
    write_rows(summary_csv, QUEUE_FIELDS, rows)

    raw_summary_txt = raw_eval_out / "pedestrian_summary.txt"
    raw_detailed_csv = raw_eval_out / "pedestrian_detailed.csv"
    competition_summary_txt = competition_eval_out / "pedestrian_summary.txt"
    competition_detailed_csv = competition_eval_out / "pedestrian_detailed.csv"
    raw_metrics = parse_summary_txt(raw_summary_txt)
    competition_metrics = parse_summary_txt(competition_summary_txt)
    per_sequence_rows = load_per_sequence_metrics(raw_detailed_csv, "raw") + load_per_sequence_metrics(competition_detailed_csv, "competition")

    compare_rows = [
        {
            "name": "raw",
            "seq": seq_label,
            "HOTA": raw_metrics.get("HOTA", ""),
            "AssA": raw_metrics.get("AssA", ""),
            "IDF1": raw_metrics.get("IDF1", ""),
            "MOTA": raw_metrics.get("MOTA", ""),
            "IDs": raw_metrics.get("IDs", ""),
            "Frag": raw_metrics.get("Frag", ""),
            "summary_txt": str(raw_summary_txt),
            "detailed_csv": str(raw_detailed_csv),
            "tracker_dir": str(raw_track_out),
        },
        {
            "name": "competition",
            "seq": seq_label,
            "HOTA": competition_metrics.get("HOTA", ""),
            "AssA": competition_metrics.get("AssA", ""),
            "IDF1": competition_metrics.get("IDF1", ""),
            "MOTA": competition_metrics.get("MOTA", ""),
            "IDs": competition_metrics.get("IDs", ""),
            "Frag": competition_metrics.get("Frag", ""),
            "summary_txt": str(competition_summary_txt),
            "detailed_csv": str(competition_detailed_csv),
            "tracker_dir": str(competition_track_out),
        },
    ]
    delta_rows = [
        {
            "name": "competition_minus_raw",
            "seq": seq_label,
            "delta_HOTA": float(competition_metrics.get("HOTA", 0.0)) - float(raw_metrics.get("HOTA", 0.0)),
            "delta_AssA": float(competition_metrics.get("AssA", 0.0)) - float(raw_metrics.get("AssA", 0.0)),
            "delta_IDF1": float(competition_metrics.get("IDF1", 0.0)) - float(raw_metrics.get("IDF1", 0.0)),
            "delta_MOTA": float(competition_metrics.get("MOTA", 0.0)) - float(raw_metrics.get("MOTA", 0.0)),
            "delta_IDs": float(competition_metrics.get("IDs", 0.0)) - float(raw_metrics.get("IDs", 0.0)),
            "delta_Frag": float(competition_metrics.get("Frag", 0.0)) - float(raw_metrics.get("Frag", 0.0)),
        }
    ]
    write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
    write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
    write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

    raw_runtime_summary = resolve_runtime_summary(raw_track_out)
    competition_runtime_summary = resolve_runtime_summary(competition_track_out)
    raw_runtime_rows = load_runtime_rows(raw_runtime_summary, "raw") if raw_runtime_summary is not None else []
    competition_runtime_rows = load_runtime_rows(competition_runtime_summary, "competition") if competition_runtime_summary is not None else []
    runtime_compare_rows = [
        summarize_runtime_rows(runtime_rows=raw_runtime_rows, label="raw", seq_label=seq_label, summary_csv=raw_runtime_summary),
        summarize_runtime_rows(runtime_rows=competition_runtime_rows, label="competition", seq_label=seq_label, summary_csv=competition_runtime_summary),
    ]
    write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_compare_rows)
    write_rows(runtime_per_sequence_csv, RUNTIME_PER_SEQUENCE_FIELDS, raw_runtime_rows + competition_runtime_rows)
    if local_export_jsonl.is_file():
        write_rows(
            local_contention_summary_csv,
            LOCAL_CONTENTION_SUMMARY_FIELDS,
            [summarize_local_contention_export(local_export_jsonl, "competition")],
        )

    compare_log = logs_dir / "compare.log"
    compare_log.write_text(
        "\n".join(
            [
                f"raw_summary={raw_summary_txt}",
                f"competition_summary={competition_summary_txt}",
                f"metrics_compare={metrics_compare_csv}",
                f"metrics_delta={metrics_delta_csv}",
                f"per_sequence_metrics={per_sequence_csv}",
                f"runtime_compare={runtime_compare_csv}",
                f"runtime_per_sequence={runtime_per_sequence_csv}",
                (f"local_contention_summary={local_contention_summary_csv}" if local_export_jsonl.is_file() else ""),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    update_row(
        rows,
        "compare",
        status="success",
        out_dir=str(run_root),
        summary_csv=str(summary_csv),
        log_path=str(compare_log),
        finished_at=now_iso(),
        notes=f"repaired compare complete for {seq_label}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(
        summary_csv,
        run_root,
        "success",
        f"repaired and completed paired preassoc competition eval on {benchmark} {seq_label}",
        args.registry_csv,
        benchmark,
        split_label,
    )


if __name__ == "__main__":
    main()
