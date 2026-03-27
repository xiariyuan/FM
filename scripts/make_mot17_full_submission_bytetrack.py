#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a MOT17 Codabench submission zip (42 files) for FM-Track ByteTrack runtime.

Codabench competition 10049 ("MOTChallenge-17 (MOT17)") expects a single .zip archive
containing ALL sequences (01..14) × detector variants (DPM/FRCNN/SDP) = 42 txt files,
directly in the zip root (no subfolders).

This helper runs inference on both:
  - MOT17/train (21 sequences)
  - MOT17/test  (21 sequences)
and then packs the 42 result files into a single zip.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

def _is_mot17_result_txt(p: Path) -> bool:
    """
    Accept only per-sequence MOT17 result files:
      MOT17-XX-{DPM|FRCNN|SDP}.txt

    This intentionally excludes TrackEval summary files such as:
      pedestrian_summary.txt
    """
    name = p.name
    if not (name.startswith("MOT17-") and name.endswith(".txt")):
        return False
    parts = name.split("-")
    if len(parts) != 3:
        return False
    seq = parts[1]
    det = parts[2].replace(".txt", "").upper()
    if not (seq.isdigit() and len(seq) == 2):
        return False
    return det in {"DPM", "FRCNN", "SDP"}


def _collect_txt_files(out_dir: Path) -> list[Path]:
    txts: list[Path] = []
    for sub in ["MOT17-train", "MOT17-test"]:
        d = out_dir / "tracker" / sub
        if d.is_dir():
            txts.extend(sorted([p for p in d.glob("*.txt") if _is_mot17_result_txt(p)]))
    return txts


def _zip_root_files(files: list[Path], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files, key=lambda x: x.name):
            zf.write(p, arcname=p.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True, help="Submission yaml config (public det, tuned thresholds).")
    ap.add_argument("--checkpoint", required=True, help="Path to tracking checkpoint (.pth).")
    ap.add_argument("--data-root", default="/gemini/code/datasets", help="Datasets root (contains MOT17/).")
    ap.add_argument("--out-dir", default=None, help="Output directory root.")
    ap.add_argument("--det-source", default="public", choices=["public", "txt", "external", "model"])
    ap.add_argument(
        "--reuse-across-detectors",
        action="store_true",
        help=(
            "Run only one detector-variant sequence per base video and copy the results to the other detector "
            "filenames. This is useful for det_source=model (private detections) because MOT17-{XX}-DPM/FRCNN/SDP "
            "share the same images/GT."
        ),
    )
    ap.add_argument(
        "--reuse-detector",
        default="FRCNN",
        choices=["DPM", "FRCNN", "SDP"],
        help="Which detector variant to actually run when --reuse-across-detectors is enabled.",
    )
    ap.add_argument(
        "--precheck-profile",
        default=None,
        choices=["mot17_test_public_21", "mot17_full_42", "mot17_train_frcnn_7", "skip"],
        help="Submission profile for scripts/check_mot17_submission.py (default inferred from skip flags).",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"submit_mot17_full_public_tuned_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_filter = None
    if args.reuse_across_detectors:
        detector_filter = str(args.reuse_detector).upper()
        if args.det_source != "model":
            print(f"[WARN] --reuse-across-detectors is primarily meant for det_source=model (got {args.det_source}).")

    # Run train + test into the same out_dir; they write into tracker/MOT17-train and tracker/MOT17-test respectively.
    if not args.skip_train:
        cmd = [
            sys.executable, "-u", "submit_bytetrack.py",
            "--config-path", args.config_path,
            "--inference-model", args.checkpoint,
            "--inference-dataset", "MOT17",
            "--inference-split", "train",
            "--data-root", args.data_root,
            "--output-dir", str(out_dir),
            "--det-source", args.det_source,
        ]
        if detector_filter:
            cmd += ["--detector-filter", detector_filter]
        _run(cmd, out_dir / "run_train.log")

    if not args.skip_test:
        cmd = [
            sys.executable, "-u", "submit_bytetrack.py",
            "--config-path", args.config_path,
            "--inference-model", args.checkpoint,
            "--inference-dataset", "MOT17",
            "--inference-split", "test",
            "--data-root", args.data_root,
            "--output-dir", str(out_dir),
            "--det-source", args.det_source,
        ]
        if detector_filter:
            cmd += ["--detector-filter", detector_filter]
        _run(cmd, out_dir / "run_test.log")

    # Collect and pack.
    txts = _collect_txt_files(out_dir)
    if len(txts) == 0:
        raise RuntimeError(f"No txt results found under {out_dir}/tracker. Did inference run?")

    merge_dir = out_dir / "merge_txt"
    merge_dir.mkdir(parents=True, exist_ok=True)
    for p in txts:
        shutil.copy2(p, merge_dir / p.name)

    # Optional: reuse across detectors by duplicating the one-run variant to the missing variants.
    if detector_filter:
        present = {p.name for p in merge_dir.glob("*.txt")}
        for name in sorted(present):
            # Expect "MOT17-XX-DET.txt"
            parts = name.split("-")
            if len(parts) != 3:
                continue
            seq = parts[1]
            det = parts[2].replace(".txt", "").upper()
            if det != detector_filter:
                continue
            src = merge_dir / name
            for other in ["DPM", "FRCNN", "SDP"]:
                dst = merge_dir / f"MOT17-{seq}-{other}.txt"
                if not dst.exists():
                    shutil.copy2(src, dst)

    if args.skip_train and (not args.skip_test):
        zip_path = out_dir / f"mot17_test_submission_{_timestamp()}.zip"
    else:
        zip_path = out_dir / f"mot17_full_submission_{_timestamp()}.zip"
    _zip_root_files(sorted([p for p in merge_dir.glob("*.txt") if _is_mot17_result_txt(p)]), zip_path)

    inferred_profile = None
    if (not args.skip_train) and (not args.skip_test):
        inferred_profile = "mot17_full_42"
    elif args.skip_train and (not args.skip_test):
        inferred_profile = "mot17_test_public_21"

    profile = args.precheck_profile if args.precheck_profile is not None else inferred_profile
    if profile and profile != "skip":
        precheck_log = out_dir / "precheck.log"
        cmd = [
            sys.executable, "-u", "scripts/check_mot17_submission.py",
            "--zip-path", str(zip_path),
            "--profile", str(profile),
        ]
        _run(cmd, precheck_log)
        print(f"[OK] Precheck log:   {precheck_log}")

    (out_dir / "latest_zip.txt").write_text(str(zip_path) + "\n", encoding="utf-8")
    print(f"[OK] Submission zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
