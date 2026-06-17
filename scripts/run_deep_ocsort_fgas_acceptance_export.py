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

SUMMARY_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "export_jsonl",
    "runtime_summary_csv",
    "exported_rows",
    "started_at",
    "finished_at",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded Deep-OC-SORT FGAS acceptance export job.")
    parser.add_argument("--seq-name", default="MOT17-05-FRCNN")
    parser.add_argument("--seq-names", nargs="+", default=None, help="optional explicit sequence list; overrides --seq-name")
    parser.add_argument("--annotation-json", default="val_half.json", help="annotation json under data/mot/annotations/")
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--block-primitive-checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_primitive_smoke_20260401_1" / "best.pt"),
    )
    parser.add_argument("--fgas-block-primitive-conf-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-assignment-mode", default="blend", choices=["blend", "replace"])
    parser.add_argument("--fgas-blend-weight", type=float, default=0.5)
    parser.add_argument("--fgas-pair-ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--fgas-soft-enable", action="store_true")
    parser.add_argument("--fgas-soft-lambda", type=float, default=0.5)
    parser.add_argument("--fgas-soft-allow-fallback", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-rows", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-frontier", action="store_true")
    parser.add_argument("--fgas-soft-row-base-margin-thresh", type=float, default=1.0)
    parser.add_argument("--fgas-soft-changed-row-flip-gap-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-soft-changed-row-refined-margin-thresh", type=float, default=0.0)
    parser.add_argument("--disable-controller", action="store_true")
    parser.add_argument("--fgas-acceptance-export-iou-thresh", type=float, default=0.5)
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing summary step: {step}")


def append_registry(args: argparse.Namespace, summary_csv: Path, run_root: Path, status: str, notes: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(args.registry_csv),
        "--kind",
        "analysis",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_fgas_acceptance_export.py",
        "--dataset",
        "MOT17",
        "--split",
        str(args.annotation_json).replace(".json", ""),
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        run_root.name,
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        str(Path(args.block_primitive_checkpoint)),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


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


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    return [str(args.seq_name)]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def main() -> int:
    args = parse_args()
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_fgas_acceptance_export_{timestamp_tag()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = run_root / "results" / "trackers"
    summary_csv = run_root / "summary.csv"
    export_jsonl = run_root / "acceptance_rows.jsonl"
    exp_name = run_root.name
    runtime_summary_csv = trackers_root / "MOT17-val" / exp_name / "fgas_analysis" / f"{exp_name}_summary.csv"

    rows: List[Dict[str, object]] = [
        {
            "step": "fgas_export",
            "name": exp_name,
            "status": "running",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "fgas_export.log"),
            "export_jsonl": str(export_jsonl),
            "runtime_summary_csv": str(runtime_summary_csv),
            "exported_rows": 0,
            "started_at": now_iso(),
            "finished_at": "",
            "notes": f"Deep-OC-SORT FGAS acceptance export on {seq_label} using {args.annotation_json}",
        }
    ]
    write_rows(summary_csv, SUMMARY_FIELDS, rows)
    append_registry(args, summary_csv, run_root, "running", f"started FGAS acceptance export on {seq_label}")

    track_cmd = [
        sys.executable,
        "main.py",
        "--dataset",
        "mot17",
        "--annotation-json",
        str(args.annotation_json),
        "--result_folder",
        str(trackers_root),
        "--exp_name",
        exp_name,
        "--seq-filter",
        *seq_names,
        "--grid_off",
        "--new_kf_off",
        "--w_assoc_emb",
        "0.75",
        "--aw_param",
        "0.5",
        "--fgas-enable",
        "--fgas-topk",
        "5",
        "--fgas-max-rows",
        "3",
        "--fgas-max-cols",
        "3",
        "--fgas-block-primitive-checkpoint",
        str(args.block_primitive_checkpoint),
        "--fgas-block-primitive-conf-thresh",
        str(args.fgas_block_primitive_conf_thresh),
        "--fgas-blend-weight",
        str(args.fgas_blend_weight),
        "--fgas-assignment-mode",
        str(args.fgas_assignment_mode),
        "--fgas-row-nomatch-weight",
        "0.0",
        "--fgas-pair-ambiguity-margin",
        str(args.fgas_pair_ambiguity_margin),
        "--fgas-acceptance-export-jsonl",
        str(export_jsonl),
        "--fgas-acceptance-export-iou-thresh",
        str(args.fgas_acceptance_export_iou_thresh),
    ]
    if args.fgas_soft_enable:
        track_cmd.extend(
            [
                "--fgas-soft-enable",
                "--fgas-soft-lambda",
                str(args.fgas_soft_lambda),
                "--fgas-soft-row-base-margin-thresh",
                str(args.fgas_soft_row_base_margin_thresh),
                "--fgas-soft-changed-row-flip-gap-thresh",
                str(args.fgas_soft_changed_row_flip_gap_thresh),
                "--fgas-soft-changed-row-refined-margin-thresh",
                str(args.fgas_soft_changed_row_refined_margin_thresh),
            ]
        )
    if args.fgas_soft_allow_fallback:
        track_cmd.append("--fgas-soft-allow-fallback")
    if args.fgas_soft_only_changed_rows:
        track_cmd.append("--fgas-soft-only-changed-rows")
    if args.fgas_soft_only_changed_frontier:
        track_cmd.append("--fgas-soft-only-changed-frontier")
    if not args.disable_controller:
        track_cmd.extend(
            [
                "--fgas-controller-enable",
                "--fgas-controller-edge-thresh",
                "0.7",
                "--fgas-controller-row-defer-thresh",
                "0.7",
                "--fgas-controller-col-newborn-thresh",
                "0.7",
                "--fgas-controller-margin-thresh",
                "0.1",
                "--fgas-controller-ambiguity-margin",
                "0.04",
            ]
        )

    log_path = logs_dir / "fgas_export.log"
    try:
        return_code = run_step(track_cmd, log_path, cwd=DEEP_ROOT)
        if return_code != 0:
            raise RuntimeError(f"export command failed with return code {return_code}")
        exported_rows = count_jsonl_rows(export_jsonl)
        runtime_rows = read_csv_rows(runtime_summary_csv)
        notes = f"FGAS acceptance export complete for {seq_label}"
        if runtime_rows:
            notes += f"; runtime summary rows={len(runtime_rows)}"
        update_row(
            rows,
            "fgas_export",
            status="success",
            finished_at=now_iso(),
            out_dir=str(run_root),
            summary_csv=str(summary_csv),
            log_path=str(log_path),
            export_jsonl=str(export_jsonl),
            runtime_summary_csv=str(runtime_summary_csv),
            exported_rows=int(exported_rows),
            notes=notes,
        )
        write_rows(summary_csv, SUMMARY_FIELDS, rows)
        append_registry(args, summary_csv, run_root, "success", f"completed FGAS acceptance export on {seq_label}; exported_rows={exported_rows}")
        return 0
    except Exception as exc:
        update_row(
            rows,
            "fgas_export",
            status="failed",
            finished_at=now_iso(),
            out_dir=str(run_root),
            summary_csv=str(summary_csv),
            log_path=str(log_path),
            export_jsonl=str(export_jsonl),
            runtime_summary_csv=str(runtime_summary_csv),
            exported_rows=int(count_jsonl_rows(export_jsonl)),
            notes=f"FGAS acceptance export failed on {seq_label}: {exc}",
        )
        write_rows(summary_csv, SUMMARY_FIELDS, rows)
        append_registry(args, summary_csv, run_root, "failed", f"FGAS acceptance export failed on {seq_label}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
