#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEEP_ROOT = REPO_ROOT / "external" / "Deep-OC-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

TRAIN_SEQS = [2, 4, 5, 9, 10, 11, 13]
TEST_SEQS = [1, 3, 6, 7, 8, 12, 14]
DETECTORS = ["DPM", "FRCNN", "SDP"]

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "artifact_path",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Deep-OC-SORT + FGAS best config and build a MOT17 submission zip.")
    parser.add_argument(
        "--out-root",
        default="",
        help="Output directory root. Defaults to outputs/deep_ocsort_fgas_submit_<timestamp>.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_resolver_v3_nofreq_20260331_1" / "best.pt"),
        help="FGAS resolver checkpoint.",
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=["full", "test"],
        help="Submission profile: full(42 files) or test(21 files).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing train outputs under --out-root.")
    parser.add_argument("--skip-test", action="store_true", help="Reuse existing test outputs under --out-root.")
    parser.add_argument(
        "--fgas-profile-mode",
        default="legacy_controller",
        choices=["legacy_controller", "soft_acceptance"],
        help="FGAS runtime profile used for export.",
    )
    parser.add_argument("--fgas-soft-lambda", type=float, default=0.5)
    parser.add_argument("--fgas-soft-only-changed-blocks", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-frontier", action="store_true")
    parser.add_argument("--fgas-soft-only-changed-rows", action="store_true")
    parser.add_argument("--fgas-soft-row-base-margin-thresh", type=float, default=1.0)
    parser.add_argument("--fgas-soft-changed-row-flip-gap-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-soft-changed-row-refined-margin-thresh", type=float, default=0.0)
    parser.add_argument("--fgas-acceptance-gate-checkpoint", default="")
    parser.add_argument("--fgas-acceptance-gate-thresh", type=float, default=0.5)
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


def append_registry(
    summary_csv: Path,
    run_root: Path,
    status: str,
    notes: str,
    registry_csv: str,
    profile: str,
) -> None:
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
        "scripts/make_mot17_full_submission_deep_ocsort_fgas.py",
        "--dataset",
        "MOT17",
        "--split",
        "full_submission" if profile == "full" else "test_submission",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_fgas_submission",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--checkpoint",
        "",
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def ensure_success(
    step: str,
    return_code: int,
    rows: List[Dict[str, object]],
    summary_csv: Path,
    out_dir: Path,
    log_path: Path,
    notes: str,
    artifact_path: str = "",
) -> None:
    finished_at = now_iso()
    status = "success" if return_code == 0 else "failed"
    note_text = notes if return_code == 0 else f"{notes} | failed rc={return_code}"
    update_row(
        rows,
        step,
        status=status,
        finished_at=finished_at,
        out_dir=str(out_dir),
        summary_csv=str(summary_csv),
        log_path=str(log_path),
        artifact_path=str(artifact_path),
        notes=note_text,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    if return_code != 0:
        raise RuntimeError(f"Step failed: {step}")


def mark_running_rows_failed(rows: List[Dict[str, object]], summary_csv: Path, reason: str) -> None:
    finished_at = now_iso()
    changed = False
    for row in rows:
        if row.get("status") == "running":
            row["status"] = "failed"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | failed: {reason}".strip()
            changed = True
    if changed:
        write_rows(summary_csv, QUEUE_FIELDS, rows)


def _fgas_args(args: argparse.Namespace) -> List[str]:
    base = [
        "--fgas-enable",
        "--fgas-resolver-checkpoint",
        str(args.checkpoint),
        "--fgas-topk",
        "5",
        "--fgas-max-rows",
        "3",
        "--fgas-max-cols",
        "3",
        "--fgas-blend-weight",
        "0.5",
        "--fgas-assignment-mode",
        "blend",
        "--fgas-row-nomatch-weight",
        "0.0",
    ]
    if str(args.fgas_profile_mode) == "soft_acceptance":
        base.extend(
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
        if bool(args.fgas_soft_only_changed_blocks):
            base.append("--fgas-soft-only-changed-blocks")
        if bool(args.fgas_soft_only_changed_frontier):
            base.append("--fgas-soft-only-changed-frontier")
        if bool(args.fgas_soft_only_changed_rows):
            base.append("--fgas-soft-only-changed-rows")
        if str(args.fgas_acceptance_gate_checkpoint):
            base.extend(
                [
                    "--fgas-acceptance-gate-checkpoint",
                    str(args.fgas_acceptance_gate_checkpoint),
                    "--fgas-acceptance-gate-thresh",
                    str(args.fgas_acceptance_gate_thresh),
                ]
            )
        return base
    base.extend(
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
    return base


def _seq_names(seq_ids: List[int]) -> List[str]:
    return [f"MOT17-{seq_id:02d}-FRCNN" for seq_id in seq_ids]


def _expected_frcnn_names(seq_ids: List[int]) -> List[str]:
    return [f"MOT17-{seq_id:02d}-FRCNN.txt" for seq_id in seq_ids]


def _resolve_existing_dir(path: Path) -> Path:
    if path.is_dir():
        return path
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
    raise RuntimeError(f"Missing source directory: {path}")


def _copy_expected(src_dir: Path, merge_dir: Path, expected_names: List[str]) -> None:
    src_dir = _resolve_existing_dir(src_dir)
    for name in expected_names:
        src = src_dir / name
        if not src.is_file():
            raise RuntimeError(f"Missing expected result file: {src}")
        shutil.copy2(src, merge_dir / name)


def _duplicate_frcnn_to_private_det(merge_dir: Path, seq_ids: List[int]) -> None:
    for seq_id in seq_ids:
        src = merge_dir / f"MOT17-{seq_id:02d}-FRCNN.txt"
        if not src.is_file():
            raise RuntimeError(f"Missing FRCNN base file for duplication: {src}")
        for det in DETECTORS:
            dst = merge_dir / f"MOT17-{seq_id:02d}-{det}.txt"
            if det == "FRCNN":
                continue
            shutil.copy2(src, dst)


def _zip_root_files(files: List[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files, key=lambda p: p.name):
            zf.write(path, arcname=path.name)


def main() -> None:
    args = parse_args()
    run_root = Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_fgas_submit_{args.profile}_{timestamp_tag()}"
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = (run_root / "results" / "trackers").resolve()
    merge_dir = run_root / "merge_txt"
    summary_csv = run_root / "summary.csv"

    train_exp = f"{run_root.name}_train"
    test_exp = f"{run_root.name}_test"
    zip_path = run_root / f"mot17_{args.profile}_submission_{timestamp_tag()}.zip"
    precheck_log = logs_dir / "precheck.log"

    rows: List[Dict[str, object]] = []
    if not args.skip_train and args.profile == "full":
        rows.append(
            {
                "step": "train_track",
                "name": train_exp,
                "status": "pending",
                "out_dir": "",
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "train_track.log"),
                "started_at": "",
                "finished_at": "",
                "artifact_path": "",
                "notes": "Deep-OC-SORT + FGAS full-train FRCNN export using train.json",
            }
        )
    if not args.skip_test:
        rows.append(
            {
                "step": "test_track",
                "name": test_exp,
                "status": "pending",
                "out_dir": "",
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "test_track.log"),
                "started_at": "",
                "finished_at": "",
                "artifact_path": "",
                "notes": "Deep-OC-SORT + FGAS test FRCNN export using test.json",
            }
        )
    rows.extend(
        [
            {
                "step": "package_zip",
                "name": run_root.name,
                "status": "pending",
                "out_dir": str(run_root),
                "summary_csv": str(summary_csv),
                "log_path": str(logs_dir / "package_zip.log"),
                "started_at": "",
                "finished_at": "",
                "artifact_path": str(zip_path),
                "notes": f"Build MOT17 {args.profile} submission zip by duplicating FRCNN outputs to DPM/SDP",
            },
            {
                "step": "precheck",
                "name": run_root.name,
                "status": "pending",
                "out_dir": str(run_root),
                "summary_csv": str(summary_csv),
                "log_path": str(precheck_log),
                "started_at": "",
                "finished_at": "",
                "artifact_path": str(zip_path),
                "notes": "Validate submission zip structure",
            },
        ]
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"submission start profile={args.profile}", args.registry_csv, args.profile)

    try:
        if not args.skip_train and args.profile == "full":
            update_row(rows, "train_track", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            train_cmd = [
                sys.executable,
                "main.py",
                "--dataset",
                "mot17",
                "--annotation-json",
                "train.json",
                "--result_folder",
                str(trackers_root),
                "--exp_name",
                train_exp,
                "--seq-filter",
                *_seq_names(TRAIN_SEQS),
                "--post",
                "--grid_off",
                "--new_kf_off",
                "--w_assoc_emb",
                "0.75",
                "--aw_param",
                "0.5",
                *_fgas_args(args),
            ]
            train_log = logs_dir / "train_track.log"
            train_out = trackers_root / "MOT17-val" / f"{train_exp}_post" / "data"
            return_code = run_step(train_cmd, train_log, cwd=DEEP_ROOT)
            ensure_success(
                "train_track",
                return_code,
                rows,
                summary_csv,
                train_out,
                train_log,
                "full-train export complete",
                artifact_path=str(train_out),
            )

        if not args.skip_test:
            update_row(rows, "test_track", status="running", started_at=now_iso())
            write_rows(summary_csv, QUEUE_FIELDS, rows)
            test_cmd = [
                sys.executable,
                "main.py",
                "--dataset",
                "mot17",
                "--test_dataset",
                "--annotation-json",
                "test.json",
                "--result_folder",
                str(trackers_root),
                "--exp_name",
                test_exp,
                "--seq-filter",
                *_seq_names(TEST_SEQS),
                "--post",
                "--grid_off",
                "--new_kf_off",
                "--w_assoc_emb",
                "0.75",
                "--aw_param",
                "0.5",
                *_fgas_args(args),
            ]
            test_log = logs_dir / "test_track.log"
            test_out = trackers_root / "MOT17-test" / f"{test_exp}_post" / "data"
            return_code = run_step(test_cmd, test_log, cwd=DEEP_ROOT)
            ensure_success(
                "test_track",
                return_code,
                rows,
                summary_csv,
                test_out,
                test_log,
                "test export complete",
                artifact_path=str(test_out),
            )

        update_row(rows, "package_zip", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        package_log = logs_dir / "package_zip.log"
        if merge_dir.exists():
            shutil.rmtree(merge_dir)
        merge_dir.mkdir(parents=True, exist_ok=True)
        with package_log.open("w", encoding="utf-8") as handle:
            handle.write(f"[started_at] {now_iso()}\n")
            handle.write(f"[run_root] {run_root}\n\n")
            if args.profile == "full":
                train_src = trackers_root / "MOT17-val" / f"{train_exp}_post" / "data"
                _copy_expected(train_src, merge_dir, _expected_frcnn_names(TRAIN_SEQS))
                handle.write(f"copied train FRCNN files from {train_src}\n")
            test_src = trackers_root / "MOT17-test" / f"{test_exp}_post" / "data"
            _copy_expected(test_src, merge_dir, _expected_frcnn_names(TEST_SEQS))
            handle.write(f"copied test FRCNN files from {test_src}\n")

            seq_ids = sorted(TRAIN_SEQS + TEST_SEQS) if args.profile == "full" else list(TEST_SEQS)
            _duplicate_frcnn_to_private_det(merge_dir, seq_ids)
            files = sorted(merge_dir.glob("MOT17-*.txt"))
            _zip_root_files(files, zip_path)
            handle.write(f"wrote zip: {zip_path}\n")
            handle.write(f"[finished_at] {now_iso()}\n")
        ensure_success(
            "package_zip",
            0,
            rows,
            summary_csv,
            run_root,
            package_log,
            f"built {args.profile} submission zip",
            artifact_path=str(zip_path),
        )

        update_row(rows, "precheck", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        precheck_profile = "mot17_full_42" if args.profile == "full" else "mot17_test_public_21"
        precheck_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "check_mot17_submission.py"),
            "--zip-path",
            str(zip_path),
            "--profile",
            precheck_profile,
        ]
        return_code = run_step(precheck_cmd, precheck_log, cwd=REPO_ROOT)
        ensure_success(
            "precheck",
            return_code,
            rows,
            summary_csv,
            run_root,
            precheck_log,
            f"precheck passed profile={precheck_profile}",
            artifact_path=str(zip_path),
        )
        (run_root / "latest_zip.txt").write_text(str(zip_path) + "\n", encoding="utf-8")
        append_registry(summary_csv, run_root, "success", f"submission ready zip={zip_path.name}", args.registry_csv, args.profile)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        append_registry(summary_csv, run_root, "failed", f"submission failed: {exc}", args.registry_csv, args.profile)
        raise


if __name__ == "__main__":
    main()
