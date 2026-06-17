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
    "preassoc_stale_competition_rows",
    "preassoc_stale_competition_candidate_rows",
    "preassoc_stale_competition_biased_edges",
    "preassoc_stale_competition_takeover_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_biased_edges",
    "preassoc_stale_competition_owner_alt_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_released_owners",
    "preassoc_stale_competition_selected_matches",
    "summary_csv",
]

RUNTIME_PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "frames_seen",
    "preassoc_stale_competition_rows",
    "preassoc_stale_competition_candidate_rows",
    "preassoc_stale_competition_biased_edges",
    "preassoc_stale_competition_takeover_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_biased_edges",
    "preassoc_stale_competition_owner_alt_risk_rejected_rows",
    "preassoc_stale_competition_owner_alt_released_owners",
    "preassoc_stale_competition_selected_matches",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded Deep-OC-SORT raw vs pre-association stale-competition paired eval on one or more MOT17 half-val sequences.")
    parser.add_argument("--seq-name", default="MOT17-05-FRCNN")
    parser.add_argument("--seq-names", nargs="+", default=None)
    parser.add_argument("--out-root", default="")
    parser.add_argument("--reuse-raw-from", default="", help="optional existing run root whose raw arm should be reused")
    parser.add_argument("--disable-preassoc-stale-competition", action="store_true", help="debug only: keep the second arm identical to the raw baseline")
    parser.add_argument("--preassoc-stale-competition-min-time-since-update", type=int, default=2)
    parser.add_argument("--preassoc-stale-competition-max-time-since-update", type=int, default=8)
    parser.add_argument("--preassoc-stale-competition-min-hits", type=int, default=15)
    parser.add_argument("--preassoc-stale-competition-min-box-iou", type=float, default=0.65)
    parser.add_argument("--preassoc-stale-competition-min-edge-score", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-bias", type=float, default=0.2)
    parser.add_argument("--preassoc-stale-competition-iou-scale", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-require-raw-owner", action="store_true")
    parser.add_argument("--preassoc-stale-competition-min-hit-gap-vs-owner", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-min-age-gap-vs-owner", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-owner-max-hits", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-owner-max-age", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-owner-edge-penalty", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-bias", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-score", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-box-iou", type=float, default=0.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-ranker-score", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-ranker-margin", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-owner-alt-det-min-edge-advantage", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-soft-margin-floor", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-soft-edge-advantage-floor", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-takeover-min-force-risk-scale", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-min-ranker-margin-to-second", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-min-edge-advantage-vs-owner", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-max-owner-edge-deficit", type=float, default=-1.0)
    parser.add_argument("--preassoc-stale-competition-block-owner-on-reclaim", action="store_true")
    parser.add_argument("--preassoc-stale-competition-require-det-top1", action="store_true")
    parser.add_argument("--preassoc-stale-competition-max-det-rank", type=int, default=0)
    parser.add_argument("--preassoc-stale-competition-export-jsonl", default="")
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
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_preassoc_competition_smoke.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_preassoc_competition",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_competition_smoke",
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


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    return [str(args.seq_name)]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


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
                    "preassoc_stale_competition_rows": int(row.get("preassoc_stale_competition_rows", 0) or 0),
                    "preassoc_stale_competition_candidate_rows": int(row.get("preassoc_stale_competition_candidate_rows", 0) or 0),
                    "preassoc_stale_competition_biased_edges": int(row.get("preassoc_stale_competition_biased_edges", 0) or 0),
                    "preassoc_stale_competition_takeover_risk_rejected_rows": int(
                        row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_biased_edges": int(
                        row.get("preassoc_stale_competition_owner_alt_biased_edges", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_risk_rejected_rows": int(
                        row.get("preassoc_stale_competition_owner_alt_risk_rejected_rows", 0) or 0
                    ),
                    "preassoc_stale_competition_owner_alt_released_owners": int(
                        row.get("preassoc_stale_competition_owner_alt_released_owners", 0) or 0
                    ),
                    "preassoc_stale_competition_selected_matches": int(row.get("preassoc_stale_competition_selected_matches", 0) or 0),
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
        "preassoc_stale_competition_rows": 0,
        "preassoc_stale_competition_candidate_rows": 0,
        "preassoc_stale_competition_biased_edges": 0,
        "preassoc_stale_competition_takeover_risk_rejected_rows": 0,
        "preassoc_stale_competition_owner_alt_biased_edges": 0,
        "preassoc_stale_competition_owner_alt_risk_rejected_rows": 0,
        "preassoc_stale_competition_owner_alt_released_owners": 0,
        "preassoc_stale_competition_selected_matches": 0,
        "summary_csv": str(summary_csv) if summary_csv is not None else "",
    }
    for row in runtime_rows:
        total["frames_seen"] = int(total["frames_seen"]) + int(row.get("frames_seen", 0))
        total["preassoc_stale_competition_rows"] = int(total["preassoc_stale_competition_rows"]) + int(
            row.get("preassoc_stale_competition_rows", 0)
        )
        total["preassoc_stale_competition_candidate_rows"] = int(total["preassoc_stale_competition_candidate_rows"]) + int(
            row.get("preassoc_stale_competition_candidate_rows", 0)
        )
        total["preassoc_stale_competition_biased_edges"] = int(total["preassoc_stale_competition_biased_edges"]) + int(
            row.get("preassoc_stale_competition_biased_edges", 0)
        )
        total["preassoc_stale_competition_takeover_risk_rejected_rows"] = int(
            total["preassoc_stale_competition_takeover_risk_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_takeover_risk_rejected_rows", 0))
        total["preassoc_stale_competition_owner_alt_biased_edges"] = int(
            total["preassoc_stale_competition_owner_alt_biased_edges"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_biased_edges", 0))
        total["preassoc_stale_competition_owner_alt_risk_rejected_rows"] = int(
            total["preassoc_stale_competition_owner_alt_risk_rejected_rows"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_risk_rejected_rows", 0))
        total["preassoc_stale_competition_owner_alt_released_owners"] = int(
            total["preassoc_stale_competition_owner_alt_released_owners"]
        ) + int(row.get("preassoc_stale_competition_owner_alt_released_owners", 0))
        total["preassoc_stale_competition_selected_matches"] = int(total["preassoc_stale_competition_selected_matches"]) + int(
            row.get("preassoc_stale_competition_selected_matches", 0)
        )
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


def main() -> None:
    args = parse_args()
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_competition_{timestamp_tag()}").resolve()
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
    intervention_exp = f"{run_root.name}_competition"

    if reuse_raw_root is not None:
        raw_exp = f"{reuse_raw_root.name}_raw"
        raw_track_out = reuse_raw_root / "results" / "trackers" / "MOT17-val" / raw_exp
        raw_eval_out = reuse_raw_root / "results" / "trackers" / "MOT17-val" / (raw_exp + "_post")
    else:
        raw_track_out = trackers_root / "MOT17-val" / raw_exp
        raw_eval_out = trackers_root / "MOT17-val" / (raw_exp + "_post")
    intervention_track_out = trackers_root / "MOT17-val" / intervention_exp
    intervention_eval_out = trackers_root / "MOT17-val" / (intervention_exp + "_post")

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
            "step": "competition_track",
            "name": intervention_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "competition_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Deep-OC-SORT preassoc competition tracking on {seq_label}",
        },
        {
            "step": "competition_eval",
            "name": intervention_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "competition_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"TrackEval for {intervention_exp}",
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
            "notes": f"Compare raw vs preassoc competition on {seq_label}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"started paired preassoc competition eval on {seq_label}", args.registry_csv)
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
            if not raw_track_out.is_dir():
                raise FileNotFoundError(f"Missing raw tracker dir: {raw_track_out}")
            if not raw_eval_out.is_dir():
                raise FileNotFoundError(f"Missing raw eval dir: {raw_eval_out}")
            write_rows(summary_csv, QUEUE_FIELDS, rows)

        update_row(rows, "competition_track", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        intervention_track_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            "mot17",
            "--result_folder",
            str(trackers_root),
            "--exp_name",
            intervention_exp,
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
        if not args.disable_preassoc_stale_competition:
            intervention_track_cmd.extend(
                [
                    "--preassoc-stale-competition-enable",
                    "--preassoc-stale-competition-min-time-since-update",
                    str(args.preassoc_stale_competition_min_time_since_update),
                    "--preassoc-stale-competition-max-time-since-update",
                    str(args.preassoc_stale_competition_max_time_since_update),
                    "--preassoc-stale-competition-min-hits",
                    str(args.preassoc_stale_competition_min_hits),
                    "--preassoc-stale-competition-min-box-iou",
                    str(args.preassoc_stale_competition_min_box_iou),
                    "--preassoc-stale-competition-min-edge-score",
                    str(args.preassoc_stale_competition_min_edge_score),
                    "--preassoc-stale-competition-bias",
                    str(args.preassoc_stale_competition_bias),
                    "--preassoc-stale-competition-iou-scale",
                    str(args.preassoc_stale_competition_iou_scale),
                ]
            )
            if args.preassoc_stale_competition_require_raw_owner:
                intervention_track_cmd.append("--preassoc-stale-competition-require-raw-owner")
            if args.preassoc_stale_competition_min_hit_gap_vs_owner > 0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-min-hit-gap-vs-owner",
                        str(args.preassoc_stale_competition_min_hit_gap_vs_owner),
                    ]
                )
            if args.preassoc_stale_competition_min_age_gap_vs_owner > 0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-min-age-gap-vs-owner",
                        str(args.preassoc_stale_competition_min_age_gap_vs_owner),
                    ]
                )
            if args.preassoc_stale_competition_owner_max_hits > 0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-max-hits",
                        str(args.preassoc_stale_competition_owner_max_hits),
                    ]
                )
            if args.preassoc_stale_competition_owner_max_age > 0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-max-age",
                        str(args.preassoc_stale_competition_owner_max_age),
                    ]
                )
            if args.preassoc_stale_competition_owner_edge_penalty > 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-edge-penalty",
                        str(args.preassoc_stale_competition_owner_edge_penalty),
                    ]
                )
            if args.preassoc_stale_competition_takeover_soft_margin_floor >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-takeover-soft-margin-floor",
                        str(args.preassoc_stale_competition_takeover_soft_margin_floor),
                    ]
                )
            if args.preassoc_stale_competition_takeover_soft_edge_advantage_floor >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-takeover-soft-edge-advantage-floor",
                        str(args.preassoc_stale_competition_takeover_soft_edge_advantage_floor),
                    ]
                )
            if args.preassoc_stale_competition_takeover_min_force_risk_scale >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-takeover-min-force-risk-scale",
                        str(args.preassoc_stale_competition_takeover_min_force_risk_scale),
                    ]
                )
            if args.preassoc_stale_competition_min_ranker_margin_to_second >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-min-ranker-margin-to-second",
                        str(args.preassoc_stale_competition_min_ranker_margin_to_second),
                    ]
                )
            if args.preassoc_stale_competition_min_edge_advantage_vs_owner >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-min-edge-advantage-vs-owner",
                        str(args.preassoc_stale_competition_min_edge_advantage_vs_owner),
                    ]
                )
            if args.preassoc_stale_competition_owner_alt_det_bias > 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-owner-alt-det-bias",
                        str(args.preassoc_stale_competition_owner_alt_det_bias),
                        "--preassoc-stale-competition-owner-alt-det-min-score",
                        str(args.preassoc_stale_competition_owner_alt_det_min_score),
                        "--preassoc-stale-competition-owner-alt-det-min-box-iou",
                        str(args.preassoc_stale_competition_owner_alt_det_min_box_iou),
                    ]
                )
                if args.preassoc_stale_competition_owner_alt_det_min_ranker_score >= 0.0:
                    intervention_track_cmd.extend(
                        [
                            "--preassoc-stale-competition-owner-alt-det-min-ranker-score",
                            str(args.preassoc_stale_competition_owner_alt_det_min_ranker_score),
                        ]
                    )
                if args.preassoc_stale_competition_owner_alt_det_min_ranker_margin >= 0.0:
                    intervention_track_cmd.extend(
                        [
                            "--preassoc-stale-competition-owner-alt-det-min-ranker-margin",
                            str(args.preassoc_stale_competition_owner_alt_det_min_ranker_margin),
                        ]
                    )
                if args.preassoc_stale_competition_owner_alt_det_min_edge_advantage >= 0.0:
                    intervention_track_cmd.extend(
                        [
                            "--preassoc-stale-competition-owner-alt-det-min-edge-advantage",
                            str(args.preassoc_stale_competition_owner_alt_det_min_edge_advantage),
                        ]
                    )
            if args.preassoc_stale_competition_max_owner_edge_deficit >= 0.0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-max-owner-edge-deficit",
                        str(args.preassoc_stale_competition_max_owner_edge_deficit),
                    ]
                )
            if args.preassoc_stale_competition_block_owner_on_reclaim:
                intervention_track_cmd.append("--preassoc-stale-competition-block-owner-on-reclaim")
            if args.preassoc_stale_competition_require_det_top1:
                intervention_track_cmd.append("--preassoc-stale-competition-require-det-top1")
            if args.preassoc_stale_competition_max_det_rank > 0:
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-max-det-rank",
                        str(args.preassoc_stale_competition_max_det_rank),
                    ]
                )
            if args.preassoc_stale_competition_export_jsonl:
                export_jsonl_path = Path(str(args.preassoc_stale_competition_export_jsonl)).expanduser()
                if not export_jsonl_path.is_absolute():
                    export_jsonl_path = (REPO_ROOT / export_jsonl_path).resolve()
                intervention_track_cmd.extend(
                    [
                        "--preassoc-stale-competition-export-jsonl",
                        str(export_jsonl_path),
                    ]
                )
        intervention_track_log = logs_dir / "competition_track.log"
        return_code = run_step(intervention_track_cmd, intervention_track_log, cwd=DEEP_ROOT)
        ensure_success(
            "competition_track",
            return_code,
            rows,
            summary_csv,
            intervention_track_out,
            intervention_track_log,
            f"preassoc competition tracking complete for {seq_label}",
        )

        update_row(rows, "competition_eval", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        intervention_eval_cmd = [
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
            intervention_exp + "_post",
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
        intervention_eval_log = logs_dir / "competition_eval.log"
        return_code = run_step(intervention_eval_cmd, intervention_eval_log, cwd=DEEP_ROOT)
        ensure_success(
            "competition_eval",
            return_code,
            rows,
            summary_csv,
            intervention_eval_out,
            intervention_eval_log,
            f"preassoc competition eval complete for {seq_label}",
        )

        update_row(rows, "compare", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)

        raw_summary_txt = raw_eval_out / "pedestrian_summary.txt"
        raw_detailed_csv = raw_eval_out / "pedestrian_detailed.csv"
        intervention_summary_txt = intervention_eval_out / "pedestrian_summary.txt"
        intervention_detailed_csv = intervention_eval_out / "pedestrian_detailed.csv"
        raw_metrics = parse_summary_txt(raw_summary_txt)
        intervention_metrics = parse_summary_txt(intervention_summary_txt)
        per_sequence_rows = load_per_sequence_metrics(raw_detailed_csv, "raw") + load_per_sequence_metrics(
            intervention_detailed_csv, "competition"
        )
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
                "HOTA": intervention_metrics.get("HOTA", ""),
                "AssA": intervention_metrics.get("AssA", ""),
                "IDF1": intervention_metrics.get("IDF1", ""),
                "MOTA": intervention_metrics.get("MOTA", ""),
                "IDs": intervention_metrics.get("IDs", ""),
                "Frag": intervention_metrics.get("Frag", ""),
                "summary_txt": str(intervention_summary_txt),
                "detailed_csv": str(intervention_detailed_csv),
                "tracker_dir": str(intervention_track_out),
            },
        ]
        delta_rows = [
            {
                "name": "competition_minus_raw",
                "seq": seq_label,
                "delta_HOTA": float(intervention_metrics.get("HOTA", 0.0)) - float(raw_metrics.get("HOTA", 0.0)),
                "delta_AssA": float(intervention_metrics.get("AssA", 0.0)) - float(raw_metrics.get("AssA", 0.0)),
                "delta_IDF1": float(intervention_metrics.get("IDF1", 0.0)) - float(raw_metrics.get("IDF1", 0.0)),
                "delta_MOTA": float(intervention_metrics.get("MOTA", 0.0)) - float(raw_metrics.get("MOTA", 0.0)),
                "delta_IDs": float(intervention_metrics.get("IDs", 0.0)) - float(raw_metrics.get("IDs", 0.0)),
                "delta_Frag": float(intervention_metrics.get("Frag", 0.0)) - float(raw_metrics.get("Frag", 0.0)),
            }
        ]
        write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
        write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
        write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

        raw_runtime_summary = resolve_runtime_summary(raw_track_out)
        intervention_runtime_summary = resolve_runtime_summary(intervention_track_out)
        raw_runtime_rows = load_runtime_rows(raw_runtime_summary, "raw") if raw_runtime_summary is not None else []
        intervention_runtime_rows = (
            load_runtime_rows(intervention_runtime_summary, "competition")
            if intervention_runtime_summary is not None
            else []
        )
        runtime_compare_rows = [
            summarize_runtime_rows(runtime_rows=raw_runtime_rows, label="raw", seq_label=seq_label, summary_csv=raw_runtime_summary),
            summarize_runtime_rows(
                runtime_rows=intervention_runtime_rows,
                label="competition",
                seq_label=seq_label,
                summary_csv=intervention_runtime_summary,
            ),
        ]
        write_rows(runtime_compare_csv, RUNTIME_FIELDS, runtime_compare_rows)
        write_rows(runtime_per_sequence_csv, RUNTIME_PER_SEQUENCE_FIELDS, raw_runtime_rows + intervention_runtime_rows)

        compare_log = logs_dir / "compare.log"
        compare_log.write_text(
            "\n".join(
                [
                    f"raw_summary={raw_summary_txt}",
                    f"competition_summary={intervention_summary_txt}",
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
        append_registry(summary_csv, run_root, "success", f"completed paired preassoc competition eval on {seq_label}", args.registry_csv)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        append_registry(summary_csv, run_root, "failed", f"paired preassoc competition eval failed on {seq_label}: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
