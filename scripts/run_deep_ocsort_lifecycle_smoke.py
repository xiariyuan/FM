#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

METRIC_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
    "summary_txt",
    "detailed_csv",
    "tracker_dir",
]

DELTA_FIELDS = [
    "name",
    "seq",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
]

PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
]

RUNTIME_FIELDS = [
    "name",
    "seq",
    "frames_seen",
    "frames_with_trigger_blocks",
    "frames_with_controller_actions",
    "lifecycle_reclaim_candidate_pairs",
    "lifecycle_reclaim_matches",
    "summary_csv",
]

RUNTIME_PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "frames_seen",
    "frames_with_trigger_blocks",
    "frames_with_controller_actions",
    "lifecycle_reclaim_candidate_pairs",
    "lifecycle_reclaim_matches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded Deep-OC-SORT raw vs raw+lifecycle paired eval on one or more MOT17 half-val sequences.")
    parser.add_argument("--seq-name", default="MOT17-05-FRCNN")
    parser.add_argument(
        "--seq-names",
        nargs="+",
        default=None,
        help="optional explicit sequence list; overrides --seq-name",
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument("--disable-lifecycle-reclaim", action="store_true", help="debug only: keep the second arm identical to the raw baseline")
    parser.add_argument("--lifecycle-reclaim-min-time-since-update", type=int, default=2)
    parser.add_argument("--lifecycle-reclaim-max-time-since-update", type=int, default=8)
    parser.add_argument("--lifecycle-reclaim-min-hits", type=int, default=15)
    parser.add_argument("--lifecycle-reclaim-min-box-iou", type=float, default=0.65)
    parser.add_argument("--lifecycle-reclaim-min-box-area-ratio", type=float, default=0.5)
    parser.add_argument("--lifecycle-reclaim-max-box-area-ratio", type=float, default=2.0)
    parser.add_argument("--lifecycle-reclaim-min-emb-similarity", type=float, default=0.0)
    parser.add_argument("--reuse-raw-from", default="", help="optional existing lifecycle-smoke run root whose raw arm should be reused")
    parser.add_argument("--compare-only", action="store_true", help="skip tracking/eval and only backfill compare artifacts from an existing run")
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


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_lifecycle_smoke.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_lifecycle",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_lifecycle_smoke",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def mark_running_rows_failed(rows: List[Dict[str, object]], summary_csv: Path, reason: str) -> None:
    finished_at = now_iso()
    changed = False
    for row in rows:
        if str(row.get("status", "")) == "running":
            row["status"] = "failed"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | failed: {reason}".strip()
            changed = True
    if changed:
        write_rows(summary_csv, QUEUE_FIELDS, rows)


def parse_summary_txt(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        rows = []
        for row in reader:
            filtered = [token for token in row if token != ""]
            if filtered:
                rows.append(filtered)
    if len(rows) < 2:
        raise RuntimeError(f"Unexpected TrackEval summary format: {path}")
    fields = rows[0]
    values = rows[1]
    data: Dict[str, float] = {}
    for key, value in zip(fields, values):
        try:
            data[key] = float(value)
        except ValueError:
            continue
    return data


def parse_step_log_metadata(log_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not log_path.is_file():
        return meta
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("[") or "] " not in line:
                continue
            key, value = line.split("] ", 1)
            meta[key[1:]] = value
    return meta


def resolve_existing_dir(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    if path.is_absolute():
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = None
        if rel is not None:
            alt = (DEEP_ROOT / rel).resolve()
            if alt.is_dir():
                return alt
    else:
        alt = (DEEP_ROOT / path).resolve()
        if alt.is_dir():
            return alt
    raise FileNotFoundError(f"Missing directory: {path}")


def resolve_existing_file(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    try:
        alt = resolve_existing_dir(path.parent) / path.name
    except FileNotFoundError:
        alt = None
    if alt is not None and alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"Missing file: {path}")


def repair_step_from_artifacts(
    rows: List[Dict[str, object]],
    summary_csv: Path,
    *,
    step: str,
    out_dir: Path,
    log_path: Path,
    notes: str,
) -> None:
    current_rows = {str(row["step"]): row for row in rows}
    row = current_rows.get(step)
    if row is None:
        return
    if str(row.get("status", "")) == "success":
        return
    meta = parse_step_log_metadata(log_path)
    if meta.get("return_code") != "0":
        return
    if not out_dir.is_dir():
        return
    row["status"] = "success"
    row["out_dir"] = str(out_dir)
    row["summary_csv"] = str(summary_csv)
    row["log_path"] = str(log_path)
    row["finished_at"] = meta.get("finished_at", row.get("finished_at", ""))
    if not row.get("started_at"):
        row["started_at"] = meta.get("started_at", "")
    row["notes"] = notes
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    return [str(args.seq_name)]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def load_per_sequence_metrics(detailed_csv: Path, label: str) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with detailed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = str(row.get("seq", ""))
            if not seq or seq == "COMBINED":
                continue
            rows.append(
                {
                    "name": label,
                    "seq": seq,
                    "HOTA": float(row["HOTA___AUC"]) * 100.0,
                    "AssA": float(row["AssA___AUC"]) * 100.0,
                    "IDF1": float(row["IDF1"]) * 100.0,
                    "MOTA": float(row["MOTA"]) * 100.0,
                    "IDs": int(round(float(row["IDs"]))),
                    "Frag": int(round(float(row["Frag"]))),
                }
            )
    return rows


def resolve_runtime_summary(track_dir: Path) -> Path | None:
    runtime_csv = track_dir / "fgas_analysis" / f"{track_dir.name}_summary.csv"
    if runtime_csv.is_file():
        return runtime_csv.resolve()
    return None


def load_runtime_rows(summary_csv: Path, label: str) -> List[Dict[str, int | str]]:
    rows: List[Dict[str, int | str]] = []
    if not summary_csv.is_file():
        return rows
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "name": label,
                    "seq": str(row.get("seq_name", "")),
                    "frames_seen": int(row.get("frames_seen", 0) or 0),
                    "frames_with_trigger_blocks": int(row.get("frames_with_trigger_blocks", 0) or 0),
                    "frames_with_controller_actions": int(row.get("frames_with_controller_actions", 0) or 0),
                    "lifecycle_reclaim_candidate_pairs": int(row.get("lifecycle_reclaim_candidate_pairs", 0) or 0),
                    "lifecycle_reclaim_matches": int(row.get("lifecycle_reclaim_matches", 0) or 0),
                }
            )
    return rows


def summarize_runtime_rows(
    *,
    runtime_rows: List[Dict[str, int | str]],
    label: str,
    seq_label: str,
    summary_csv: Path | None,
) -> Dict[str, int | str]:
    total = {
        "name": label,
        "seq": seq_label,
        "frames_seen": 0,
        "frames_with_trigger_blocks": 0,
        "frames_with_controller_actions": 0,
        "lifecycle_reclaim_candidate_pairs": 0,
        "lifecycle_reclaim_matches": 0,
        "summary_csv": str(summary_csv) if summary_csv is not None else "",
    }
    for row in runtime_rows:
        total["frames_seen"] = int(total["frames_seen"]) + int(row.get("frames_seen", 0))
        total["frames_with_trigger_blocks"] = int(total["frames_with_trigger_blocks"]) + int(row.get("frames_with_trigger_blocks", 0))
        total["frames_with_controller_actions"] = int(total["frames_with_controller_actions"]) + int(row.get("frames_with_controller_actions", 0))
        total["lifecycle_reclaim_candidate_pairs"] = int(total["lifecycle_reclaim_candidate_pairs"]) + int(
            row.get("lifecycle_reclaim_candidate_pairs", 0)
        )
        total["lifecycle_reclaim_matches"] = int(total["lifecycle_reclaim_matches"]) + int(row.get("lifecycle_reclaim_matches", 0))
    return total


def ensure_success(step: str, return_code: int, rows: List[Dict[str, object]], summary_csv: Path, out_dir: Path, log_path: Path, notes: str) -> None:
    finished_at = now_iso()
    status = "success" if return_code == 0 else "failed"
    update_row(
        rows,
        step,
        status=status,
        finished_at=finished_at,
        out_dir=str(out_dir),
        summary_csv=str(summary_csv),
        log_path=str(log_path),
        notes=notes,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    if return_code != 0:
        raise RuntimeError(f"Step failed: {step}")


def backfill_compare(
    *,
    rows: List[Dict[str, object]],
    summary_csv: Path,
    logs_dir: Path,
    run_root: Path,
    seq_label: str,
    raw_track_out: Path,
    raw_eval_out: Path,
    lifecycle_track_out: Path,
    lifecycle_eval_out: Path,
    metrics_compare_csv: Path,
    metrics_delta_csv: Path,
    per_sequence_csv: Path,
    runtime_compare_csv: Path,
    runtime_per_sequence_csv: Path,
) -> None:
    update_row(rows, "compare", status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, rows)

    raw_track_dir = resolve_existing_dir(raw_track_out)
    lifecycle_track_dir = resolve_existing_dir(lifecycle_track_out)
    raw_eval_dir = resolve_existing_dir(raw_eval_out)
    lifecycle_eval_dir = resolve_existing_dir(lifecycle_eval_out)

    raw_summary_txt = resolve_existing_file(raw_eval_dir / "pedestrian_summary.txt")
    raw_detailed_csv = resolve_existing_file(raw_eval_dir / "pedestrian_detailed.csv")
    lifecycle_summary_txt = resolve_existing_file(lifecycle_eval_dir / "pedestrian_summary.txt")
    lifecycle_detailed_csv = resolve_existing_file(lifecycle_eval_dir / "pedestrian_detailed.csv")
    raw_metrics = parse_summary_txt(raw_summary_txt)
    lifecycle_metrics = parse_summary_txt(lifecycle_summary_txt)
    per_sequence_rows = load_per_sequence_metrics(raw_detailed_csv, "raw") + load_per_sequence_metrics(lifecycle_detailed_csv, "lifecycle")

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
            "tracker_dir": str(raw_track_dir),
        },
        {
            "name": "lifecycle",
            "seq": seq_label,
            "HOTA": lifecycle_metrics.get("HOTA", ""),
            "AssA": lifecycle_metrics.get("AssA", ""),
            "IDF1": lifecycle_metrics.get("IDF1", ""),
            "MOTA": lifecycle_metrics.get("MOTA", ""),
            "IDs": lifecycle_metrics.get("IDs", ""),
            "Frag": lifecycle_metrics.get("Frag", ""),
            "summary_txt": str(lifecycle_summary_txt),
            "detailed_csv": str(lifecycle_detailed_csv),
            "tracker_dir": str(lifecycle_track_dir),
        },
    ]
    delta_rows = [
        {
            "name": "lifecycle_minus_raw",
            "seq": seq_label,
            "delta_HOTA": float(lifecycle_metrics.get("HOTA", 0.0)) - float(raw_metrics.get("HOTA", 0.0)),
            "delta_AssA": float(lifecycle_metrics.get("AssA", 0.0)) - float(raw_metrics.get("AssA", 0.0)),
            "delta_IDF1": float(lifecycle_metrics.get("IDF1", 0.0)) - float(raw_metrics.get("IDF1", 0.0)),
            "delta_MOTA": float(lifecycle_metrics.get("MOTA", 0.0)) - float(raw_metrics.get("MOTA", 0.0)),
            "delta_IDs": float(lifecycle_metrics.get("IDs", 0.0)) - float(raw_metrics.get("IDs", 0.0)),
            "delta_Frag": float(lifecycle_metrics.get("Frag", 0.0)) - float(raw_metrics.get("Frag", 0.0)),
        }
    ]
    write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
    write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
    write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

    raw_runtime_summary = resolve_runtime_summary(raw_track_dir)
    lifecycle_runtime_summary = resolve_runtime_summary(lifecycle_track_dir)
    raw_runtime_rows = load_runtime_rows(raw_runtime_summary, "raw") if raw_runtime_summary is not None else []
    lifecycle_runtime_rows = load_runtime_rows(lifecycle_runtime_summary, "lifecycle") if lifecycle_runtime_summary is not None else []
    runtime_compare_rows = [
        summarize_runtime_rows(runtime_rows=raw_runtime_rows, label="raw", seq_label=seq_label, summary_csv=raw_runtime_summary),
        summarize_runtime_rows(runtime_rows=lifecycle_runtime_rows, label="lifecycle", seq_label=seq_label, summary_csv=lifecycle_runtime_summary),
    ]
    runtime_per_sequence_rows = raw_runtime_rows + lifecycle_runtime_rows
    write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_compare_rows)
    write_rows(runtime_per_sequence_csv, RUNTIME_PER_SEQUENCE_FIELDS, runtime_per_sequence_rows)

    compare_log = logs_dir / "compare.log"
    compare_log.write_text(
        "\n".join(
            [
                f"raw_summary={raw_summary_txt}",
                f"lifecycle_summary={lifecycle_summary_txt}",
                f"metrics_compare={metrics_compare_csv}",
                f"metrics_delta={metrics_delta_csv}",
                f"per_sequence_metrics={per_sequence_csv}",
                f"runtime_compare={runtime_compare_csv}",
                f"runtime_per_sequence={runtime_per_sequence_csv}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    update_row(
        rows,
        "compare",
        status="success",
        finished_at=now_iso(),
        out_dir=str(run_root),
        summary_csv=str(summary_csv),
        log_path=str(compare_log),
        notes=f"compare complete for {seq_label}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def main() -> None:
    args = parse_args()
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_lifecycle_smoke_{timestamp_tag()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = (run_root / "results" / "trackers").resolve()
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    runtime_compare_csv = run_root / "runtime_compare.csv"
    runtime_per_sequence_csv = run_root / "runtime_per_sequence.csv"
    summary_csv = run_root / "summary.csv"

    reuse_raw_root = Path(str(args.reuse_raw_from)).resolve() if args.reuse_raw_from else None
    raw_exp = f"{run_root.name}_raw"
    lifecycle_exp = f"{run_root.name}_lifecycle"

    if reuse_raw_root is not None:
        raw_exp = f"{reuse_raw_root.name}_raw"
        raw_track_out = reuse_raw_root / "results" / "trackers" / "MOT17-val" / raw_exp
        raw_eval_out = reuse_raw_root / "results" / "trackers" / "MOT17-val" / (raw_exp + "_post")
    else:
        raw_track_out = trackers_root / "MOT17-val" / raw_exp
        raw_eval_out = trackers_root / "MOT17-val" / (raw_exp + "_post")
    lifecycle_track_out = trackers_root / "MOT17-val" / lifecycle_exp
    lifecycle_eval_out = trackers_root / "MOT17-val" / (lifecycle_exp + "_post")

    if args.compare_only:
        rows = read_rows(summary_csv)
        if not rows:
            raise FileNotFoundError(f"Missing summary.csv for compare-only run: {summary_csv}")
        try:
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="raw_track",
                out_dir=raw_track_out,
                log_path=logs_dir / "raw_track.log",
                notes=f"raw tracking complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="raw_eval",
                out_dir=raw_eval_out,
                log_path=logs_dir / "raw_eval.log",
                notes=f"raw eval complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="lifecycle_track",
                out_dir=lifecycle_track_out,
                log_path=logs_dir / "lifecycle_track.log",
                notes=f"lifecycle tracking complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="lifecycle_eval",
                out_dir=lifecycle_eval_out,
                log_path=logs_dir / "lifecycle_eval.log",
                notes=f"lifecycle eval complete for {seq_label}",
            )
            backfill_compare(
                rows=rows,
                summary_csv=summary_csv,
                logs_dir=logs_dir,
                run_root=run_root,
                seq_label=seq_label,
                raw_track_out=raw_track_out,
                raw_eval_out=raw_eval_out,
                lifecycle_track_out=lifecycle_track_out,
                lifecycle_eval_out=lifecycle_eval_out,
                metrics_compare_csv=metrics_compare_csv,
                metrics_delta_csv=metrics_delta_csv,
                per_sequence_csv=per_sequence_csv,
                runtime_compare_csv=runtime_compare_csv,
                runtime_per_sequence_csv=runtime_per_sequence_csv,
            )
            append_registry(summary_csv, run_root, "success", f"compare-only backfill completed on {seq_label}", args.registry_csv)
            return
        except Exception as exc:
            mark_running_rows_failed(rows, summary_csv, str(exc))
            append_registry(summary_csv, run_root, "failed", f"compare-only backfill failed on {seq_label}: {exc}", args.registry_csv)
            raise

    rows: List[Dict[str, object]] = [
        {
            "step": "raw_track",
            "name": raw_exp,
            "status": "success" if reuse_raw_root is not None else "running",
            "out_dir": str(raw_track_out) if reuse_raw_root is not None else "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_track.log"),
            "started_at": now_iso() if reuse_raw_root is None else "",
            "finished_at": now_iso() if reuse_raw_root is not None else "",
            "notes": (
                f"reuse raw tracking from {reuse_raw_root}"
                if reuse_raw_root is not None
                else f"Deep-OC-SORT raw tracking on {seq_label}"
            ),
        },
        {
            "step": "raw_eval",
            "name": raw_exp,
            "status": "success" if reuse_raw_root is not None else "pending",
            "out_dir": str(raw_eval_out) if reuse_raw_root is not None else "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_eval.log"),
            "started_at": "",
            "finished_at": now_iso() if reuse_raw_root is not None else "",
            "notes": (
                f"reuse raw eval from {reuse_raw_root}"
                if reuse_raw_root is not None
                else f"TrackEval for {raw_exp}"
            ),
        },
        {
            "step": "lifecycle_track",
            "name": lifecycle_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "lifecycle_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Deep-OC-SORT lifecycle tracking on {seq_label}",
        },
        {
            "step": "lifecycle_eval",
            "name": lifecycle_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "lifecycle_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"TrackEval for {lifecycle_exp}",
        },
        {
            "step": "compare",
            "name": run_root.name,
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "compare.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Compare raw vs lifecycle on {seq_label}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"started paired lifecycle eval on {seq_label}", args.registry_csv)
    try:
        if reuse_raw_root is None:
            raw_track_cmd = [
                sys.executable,
                "main.py",
                "--dataset",
                "mot17",
                "--result_folder",
                str(trackers_root),
                "--exp_name",
                raw_exp,
                "--seq-filter",
                *seq_names,
                "--post",
                "--grid_off",
                "--new_kf_off",
                "--w_assoc_emb",
                "0.75",
                "--aw_param",
                "0.5",
            ]
            raw_track_log = logs_dir / "raw_track.log"
            return_code = run_step(raw_track_cmd, raw_track_log, cwd=DEEP_ROOT)
            ensure_success("raw_track", return_code, rows, summary_csv, raw_track_out, raw_track_log, f"raw tracking complete for {seq_label}")

            update_row(rows, "raw_eval", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            raw_eval_cmd = [
                sys.executable,
                "external/TrackEval/scripts/run_mot_challenge.py",
                "--BENCHMARK",
                "MOT17",
                "--SPLIT_TO_EVAL",
                "val",
                "--GT_FOLDER",
                str(DEEP_ROOT / "results" / "gt"),
                "--TRACKERS_FOLDER",
                str(trackers_root),
                "--TRACKERS_TO_EVAL",
                raw_exp + "_post",
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
            raw_eval_log = logs_dir / "raw_eval.log"
            return_code = run_step(raw_eval_cmd, raw_eval_log, cwd=DEEP_ROOT)
            ensure_success("raw_eval", return_code, rows, summary_csv, raw_eval_out, raw_eval_log, f"raw eval complete for {seq_label}")
        else:
            resolve_existing_dir(raw_track_out)
            resolve_existing_dir(raw_eval_out)
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "lifecycle_track", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        lifecycle_track_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            "mot17",
            "--result_folder",
            str(trackers_root),
            "--exp_name",
            lifecycle_exp,
            "--seq-filter",
            *seq_names,
            "--post",
            "--grid_off",
            "--new_kf_off",
            "--w_assoc_emb",
            "0.75",
            "--aw_param",
            "0.5",
        ]
        if not args.disable_lifecycle_reclaim:
            lifecycle_track_cmd.extend(
                [
                    "--fgas-lifecycle-reclaim-enable",
                    "--fgas-lifecycle-reclaim-min-time-since-update",
                    str(args.lifecycle_reclaim_min_time_since_update),
                    "--fgas-lifecycle-reclaim-max-time-since-update",
                    str(args.lifecycle_reclaim_max_time_since_update),
                    "--fgas-lifecycle-reclaim-min-hits",
                    str(args.lifecycle_reclaim_min_hits),
                    "--fgas-lifecycle-reclaim-min-box-iou",
                    str(args.lifecycle_reclaim_min_box_iou),
                    "--fgas-lifecycle-reclaim-min-box-area-ratio",
                    str(args.lifecycle_reclaim_min_box_area_ratio),
                    "--fgas-lifecycle-reclaim-max-box-area-ratio",
                    str(args.lifecycle_reclaim_max_box_area_ratio),
                ]
            )
            if args.lifecycle_reclaim_min_emb_similarity > 0.0:
                lifecycle_track_cmd.extend(
                    [
                        "--fgas-lifecycle-reclaim-min-emb-similarity",
                        str(args.lifecycle_reclaim_min_emb_similarity),
                    ]
                )
        lifecycle_track_log = logs_dir / "lifecycle_track.log"
        return_code = run_step(lifecycle_track_cmd, lifecycle_track_log, cwd=DEEP_ROOT)
        ensure_success(
            "lifecycle_track",
            return_code,
            rows,
            summary_csv,
            lifecycle_track_out,
            lifecycle_track_log,
            f"lifecycle tracking complete for {seq_label}",
        )

        update_row(rows, "lifecycle_eval", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        lifecycle_eval_cmd = [
            sys.executable,
            "external/TrackEval/scripts/run_mot_challenge.py",
            "--BENCHMARK",
            "MOT17",
            "--SPLIT_TO_EVAL",
            "val",
            "--GT_FOLDER",
            str(DEEP_ROOT / "results" / "gt"),
            "--TRACKERS_FOLDER",
            str(trackers_root),
            "--TRACKERS_TO_EVAL",
            lifecycle_exp + "_post",
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
        lifecycle_eval_log = logs_dir / "lifecycle_eval.log"
        return_code = run_step(lifecycle_eval_cmd, lifecycle_eval_log, cwd=DEEP_ROOT)
        ensure_success(
            "lifecycle_eval",
            return_code,
            rows,
            summary_csv,
            lifecycle_eval_out,
            lifecycle_eval_log,
            f"lifecycle eval complete for {seq_label}",
        )

        backfill_compare(
            rows=rows,
            summary_csv=summary_csv,
            logs_dir=logs_dir,
            run_root=run_root,
            seq_label=seq_label,
            raw_track_out=raw_track_out,
            raw_eval_out=raw_eval_out,
            lifecycle_track_out=lifecycle_track_out,
            lifecycle_eval_out=lifecycle_eval_out,
            metrics_compare_csv=metrics_compare_csv,
            metrics_delta_csv=metrics_delta_csv,
            per_sequence_csv=per_sequence_csv,
            runtime_compare_csv=runtime_compare_csv,
            runtime_per_sequence_csv=runtime_per_sequence_csv,
        )
        append_registry(summary_csv, run_root, "success", f"completed paired lifecycle eval on {seq_label}", args.registry_csv)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        append_registry(summary_csv, run_root, "failed", f"paired lifecycle eval failed on {seq_label}: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
