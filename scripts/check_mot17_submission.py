#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate MOT17 submission zip format for Codabench/MOTChallenge.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


MOT17_TEST_SEQS = [1, 3, 6, 7, 8, 12, 14]
MOT17_TRAIN_SEQS = [2, 4, 5, 9, 10, 11, 13]
MOT17_ALL_SEQS = list(range(1, 15))
MOT17_DETECTORS = ["DPM", "FRCNN", "SDP"]


def _expected_files_for_profile(profile: str) -> Set[str]:
    if profile == "mot17_test_public_21":
        return {
            f"MOT17-{seq:02d}-{det}.txt"
            for seq in MOT17_TEST_SEQS
            for det in MOT17_DETECTORS
        }
    if profile == "mot17_full_42":
        return {
            f"MOT17-{seq:02d}-{det}.txt"
            for seq in MOT17_ALL_SEQS
            for det in MOT17_DETECTORS
        }
    if profile == "mot17_train_frcnn_7":
        return {f"MOT17-{seq:02d}-FRCNN.txt" for seq in MOT17_TRAIN_SEQS}
    raise ValueError(f"Unknown profile: {profile}")


def _parse_custom_expected(raw: str) -> Set[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return set(names)


def _is_root_file(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return "/" not in normalized and not name.endswith("/")


def _validate_csv_content(
    zf: zipfile.ZipFile,
    filename: str,
    max_errors: int = 20,
) -> List[str]:
    errors: List[str] = []
    try:
        data = zf.read(filename)
    except Exception as exc:
        return [f"{filename}: read failed ({exc})"]

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:
            return [f"{filename}: decode failed (utf-8/latin-1)"]

    fobj = io.StringIO(text)
    reader = csv.reader(fobj)
    line_no = 0
    for row in reader:
        line_no += 1
        if not row:
            continue
        if len(row) != 10:
            errors.append(f"{filename}: line {line_no} has {len(row)} columns (expected 10)")
            if len(errors) >= max_errors:
                return errors
            continue
        values = [item.strip() for item in row]
        frame, track_id = values[0], values[1]
        x_left, y_top, width, height = values[2], values[3], values[4], values[5]
        conf, xw, yw, zw = values[6], values[7], values[8], values[9]

        try:
            frame_int = int(float(frame))
            if frame_int < 1:
                errors.append(f"{filename}: line {line_no} frame must be >=1, got {frame}")
        except ValueError:
            errors.append(f"{filename}: line {line_no} invalid frame '{frame}'")

        try:
            track_int = int(float(track_id))
            if track_int < 1:
                errors.append(f"{filename}: line {line_no} id must be >=1, got {track_id}")
        except ValueError:
            errors.append(f"{filename}: line {line_no} invalid id '{track_id}'")

        for field_name, raw in [("bb_left", x_left), ("bb_top", y_top), ("bb_width", width), ("bb_height", height), ("conf", conf), ("x", xw), ("y", yw), ("z", zw)]:
            try:
                float(raw)
            except ValueError:
                errors.append(f"{filename}: line {line_no} invalid {field_name} '{raw}'")

        try:
            if float(width) <= 0 or float(height) <= 0:
                errors.append(
                    f"{filename}: line {line_no} bb_width/bb_height must be >0, got ({width}, {height})"
                )
        except ValueError:
            pass

        if len(errors) >= max_errors:
            return errors
    return errors


def _list_root_txt_files(zf: zipfile.ZipFile) -> List[str]:
    names = [name for name in zf.namelist() if _is_root_file(name)]
    return sorted([name for name in names if name.lower().endswith(".txt")])


def _list_non_root_entries(zf: zipfile.ZipFile) -> List[str]:
    bad: List[str] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        normalized = name.replace("\\", "/")
        if "/" in normalized:
            bad.append(name)
    return sorted(bad)


def _print_name_list(title: str, names: Iterable[str], limit: int = 30) -> None:
    names_list = list(names)
    if not names_list:
        return
    print(f"\n{title} ({len(names_list)}):")
    for name in names_list[:limit]:
        print(f"  - {name}")
    if len(names_list) > limit:
        print(f"  ... ({len(names_list) - limit} more)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MOT17 submission zip file.")
    parser.add_argument("--zip-path", required=True, help="Path to submission zip.")
    parser.add_argument(
        "--profile",
        default="mot17_test_public_21",
        choices=["mot17_test_public_21", "mot17_full_42", "mot17_train_frcnn_7", "custom"],
        help="Expected filename profile.",
    )
    parser.add_argument(
        "--custom-expected",
        default="",
        help="Comma-separated expected txt names when profile=custom.",
    )
    parser.add_argument(
        "--check-content",
        action="store_true",
        default=True,
        help="Validate each txt line format (10 columns, numeric fields).",
    )
    parser.add_argument(
        "--skip-content-check",
        action="store_true",
        help="Skip line-by-line content checks.",
    )
    parser.add_argument("--max-errors", type=int, default=30, help="Max content errors to print.")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        print(f"[FAIL] zip not found: {zip_path}")
        return 2
    if zip_path.suffix.lower() != ".zip":
        print(f"[WARN] file is not .zip by suffix: {zip_path.name}")

    if args.profile == "custom":
        expected_files = _parse_custom_expected(args.custom_expected)
        if not expected_files:
            print("[FAIL] profile=custom requires --custom-expected")
            return 2
    else:
        expected_files = _expected_files_for_profile(args.profile)

    do_content_check = args.check_content and not args.skip_content_check
    failure = False
    all_content_errors: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        root_txt_files = _list_root_txt_files(zf)
        non_root_entries = _list_non_root_entries(zf)
        root_txt_set = set(root_txt_files)

        missing = sorted(expected_files - root_txt_set)
        unexpected = sorted(root_txt_set - expected_files)

        print("==== MOT17 Submission Precheck ====")
        print(f"Zip: {zip_path}")
        print(f"Profile: {args.profile}")
        print(f"Expected txt files: {len(expected_files)}")
        print(f"Found root txt files: {len(root_txt_files)}")

        if non_root_entries:
            failure = True
            _print_name_list("Files not in zip root", non_root_entries)

        if missing:
            failure = True
            _print_name_list("Missing expected files", missing)

        if unexpected:
            failure = True
            _print_name_list("Unexpected txt files", unexpected)

        if do_content_check and not missing:
            for txt_name in sorted(expected_files):
                if txt_name not in root_txt_set:
                    continue
                errors = _validate_csv_content(zf, txt_name, max_errors=max(5, args.max_errors))
                all_content_errors.extend(errors)
                if len(all_content_errors) >= args.max_errors:
                    break

            if all_content_errors:
                failure = True
                _print_name_list("Content format errors", all_content_errors, limit=args.max_errors)

    if failure:
        print("\n[FAIL] Submission package does NOT meet the selected profile.")
        return 2
    print("\n[PASS] Submission package meets the selected profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
