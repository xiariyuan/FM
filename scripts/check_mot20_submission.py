#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate MOT20 submission package (zip or plain result directory).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


MOT20_TEST_SEQS = [4, 6, 7, 8]
MOT20_TRAIN_SEQS = [1, 2, 3, 5]
MOT20_ALL_SEQS = [1, 2, 3, 4, 5, 6, 7, 8]

# Official test sequence lengths (frames), used for quick sanity checks.
MOT20_TEST_MAX_FRAME: Dict[str, int] = {
    "MOT20-04.txt": 2080,
    "MOT20-06.txt": 1008,
    "MOT20-07.txt": 585,
    "MOT20-08.txt": 806,
}


def _expected_files_for_profile(profile: str) -> Set[str]:
    if profile == "mot20_test_4":
        return {f"MOT20-{seq:02d}.txt" for seq in MOT20_TEST_SEQS}
    if profile == "mot20_train_4":
        return {f"MOT20-{seq:02d}.txt" for seq in MOT20_TRAIN_SEQS}
    if profile == "mot20_full_8":
        return {f"MOT20-{seq:02d}.txt" for seq in MOT20_ALL_SEQS}
    raise ValueError(f"Unknown profile: {profile}")


def _parse_custom_expected(raw: str) -> Set[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return set(names)


def _is_root_file(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return "/" not in normalized and not name.endswith("/")


def _print_name_list(title: str, names: Iterable[str], limit: int = 30) -> None:
    names_list = list(names)
    if not names_list:
        return
    print(f"\n{title} ({len(names_list)}):")
    for name in names_list[:limit]:
        print(f"  - {name}")
    if len(names_list) > limit:
        print(f"  ... ({len(names_list) - limit} more)")


def _validate_rows(rows: Iterable[List[str]], filename: str, max_errors: int = 50) -> List[str]:
    errors: List[str] = []
    last_frame = -1
    min_frame = None
    max_frame = None
    line_no = 0

    for row in rows:
        line_no += 1
        if not row:
            continue
        if len(row) != 10:
            errors.append(f"{filename}: line {line_no} has {len(row)} columns (expected 10)")
            if len(errors) >= max_errors:
                return errors
            continue

        vals = [c.strip() for c in row]
        frame_s, track_s = vals[0], vals[1]
        x_s, y_s, w_s, h_s = vals[2], vals[3], vals[4], vals[5]
        conf_s = vals[6]
        xyz = vals[7:10]

        # frame
        frame_i = None
        try:
            frame_i = int(float(frame_s))
            if frame_i < 1:
                errors.append(f"{filename}: line {line_no} frame must be >=1, got {frame_s}")
            if frame_i is not None and frame_i < last_frame:
                errors.append(
                    f"{filename}: line {line_no} frame order not non-decreasing ({last_frame} -> {frame_i})"
                )
            if frame_i is not None:
                last_frame = frame_i
                min_frame = frame_i if min_frame is None else min(min_frame, frame_i)
                max_frame = frame_i if max_frame is None else max(max_frame, frame_i)
        except ValueError:
            errors.append(f"{filename}: line {line_no} invalid frame '{frame_s}'")

        # id
        try:
            tid = int(float(track_s))
            if tid < 1:
                errors.append(f"{filename}: line {line_no} id must be >=1, got {track_s}")
        except ValueError:
            errors.append(f"{filename}: line {line_no} invalid id '{track_s}'")

        # numeric fields
        fields = [("bb_left", x_s), ("bb_top", y_s), ("bb_width", w_s), ("bb_height", h_s), ("conf", conf_s)]
        fields.extend([("x", xyz[0]), ("y", xyz[1]), ("z", xyz[2])])
        parsed: Dict[str, float] = {}
        for name, raw in fields:
            try:
                v = float(raw)
                if math.isnan(v) or math.isinf(v):
                    errors.append(f"{filename}: line {line_no} {name} is non-finite ({raw})")
                parsed[name] = v
            except ValueError:
                errors.append(f"{filename}: line {line_no} invalid {name} '{raw}'")

        if "bb_width" in parsed and "bb_height" in parsed:
            if parsed["bb_width"] <= 0 or parsed["bb_height"] <= 0:
                errors.append(
                    f"{filename}: line {line_no} bb_width/bb_height must be >0, got ({w_s}, {h_s})"
                )

        if len(errors) >= max_errors:
            return errors

    # Optional sequence length sanity for MOT20 test set.
    known_max = MOT20_TEST_MAX_FRAME.get(filename, None)
    if known_max is not None and max_frame is not None:
        if max_frame > known_max:
            errors.append(
                f"{filename}: max frame {max_frame} exceeds expected test length {known_max}"
            )
        if min_frame is not None and min_frame != 1:
            errors.append(f"{filename}: first frame is {min_frame}, expected to start from 1")

    return errors


def _read_zip_rows(zf: zipfile.ZipFile, filename: str) -> Tuple[List[List[str]] | None, str | None]:
    try:
        data = zf.read(filename)
    except Exception as exc:
        return None, f"{filename}: read failed ({exc})"

    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None, f"{filename}: decode failed (utf-8/latin-1)"
    rows = list(csv.reader(io.StringIO(text)))
    return rows, None


def _list_root_txt_files_zip(zf: zipfile.ZipFile) -> List[str]:
    names = [n for n in zf.namelist() if _is_root_file(n)]
    return sorted([n for n in names if n.lower().endswith(".txt")])


def _list_non_root_entries_zip(zf: zipfile.ZipFile) -> List[str]:
    bad: List[str] = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        normalized = name.replace("\\", "/")
        if "/" in normalized:
            bad.append(name)
    return sorted(bad)


def _list_root_txt_files_dir(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])


def _list_non_root_entries_dir(root: Path) -> List[str]:
    bad: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.parent != root:
            bad.append(str(p.relative_to(root)))
    return sorted(bad)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate MOT20 submission (zip or result directory).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip-path", default="", help="Path to submission zip.")
    group.add_argument("--results-dir", default="", help="Path to directory containing MOT20-xx.txt files.")
    ap.add_argument(
        "--profile",
        default="mot20_test_4",
        choices=["mot20_test_4", "mot20_train_4", "mot20_full_8", "custom"],
        help="Expected filename profile.",
    )
    ap.add_argument("--custom-expected", default="", help="Comma-separated expected txt names for profile=custom.")
    ap.add_argument("--max-errors", type=int, default=40, help="Max errors to print.")
    ap.add_argument("--skip-content-check", action="store_true", help="Skip line-by-line content checks.")
    args = ap.parse_args()

    if args.profile == "custom":
        expected = _parse_custom_expected(args.custom_expected)
        if not expected:
            print("[FAIL] profile=custom requires --custom-expected")
            return 2
    else:
        expected = _expected_files_for_profile(args.profile)

    do_content = not args.skip_content_check
    failure = False
    all_errors: List[str] = []

    if args.zip_path:
        zip_path = Path(args.zip_path)
        if not zip_path.exists():
            print(f"[FAIL] zip not found: {zip_path}")
            return 2
        with zipfile.ZipFile(zip_path, "r") as zf:
            found = _list_root_txt_files_zip(zf)
            non_root = _list_non_root_entries_zip(zf)
            found_set = set(found)

            missing = sorted(expected - found_set)
            unexpected = sorted(found_set - expected)

            print("==== MOT20 Submission Precheck ====")
            print(f"Input: {zip_path}")
            print(f"Profile: {args.profile}")
            print(f"Expected txt files: {len(expected)}")
            print(f"Found root txt files: {len(found)}")

            if non_root:
                failure = True
                _print_name_list("Files not in zip root", non_root)
            if missing:
                failure = True
                _print_name_list("Missing expected files", missing)
            if unexpected:
                failure = True
                _print_name_list("Unexpected txt files", unexpected)

            if do_content and not missing:
                for name in sorted(expected):
                    rows, err = _read_zip_rows(zf, name)
                    if err:
                        all_errors.append(err)
                        continue
                    if rows is None:
                        continue
                    all_errors.extend(_validate_rows(rows, filename=name, max_errors=max(5, args.max_errors)))
                    if len(all_errors) >= args.max_errors:
                        break

    else:
        root = Path(args.results_dir)
        if not root.exists() or not root.is_dir():
            print(f"[FAIL] results dir not found: {root}")
            return 2
        found = _list_root_txt_files_dir(root)
        non_root = _list_non_root_entries_dir(root)
        found_set = set(found)

        missing = sorted(expected - found_set)
        unexpected = sorted(found_set - expected)

        print("==== MOT20 Submission Precheck ====")
        print(f"Input: {root}")
        print(f"Profile: {args.profile}")
        print(f"Expected txt files: {len(expected)}")
        print(f"Found root txt files: {len(found)}")

        if non_root:
            failure = True
            _print_name_list("Nested files (should be flat)", non_root)
        if missing:
            failure = True
            _print_name_list("Missing expected files", missing)
        if unexpected:
            failure = True
            _print_name_list("Unexpected txt files", unexpected)

        if do_content and not missing:
            for name in sorted(expected):
                p = root / name
                try:
                    text = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = p.read_text(encoding="latin-1")
                except Exception as exc:
                    all_errors.append(f"{name}: read failed ({exc})")
                    continue
                rows = list(csv.reader(io.StringIO(text)))
                all_errors.extend(_validate_rows(rows, filename=name, max_errors=max(5, args.max_errors)))
                if len(all_errors) >= args.max_errors:
                    break

    if all_errors:
        failure = True
        _print_name_list("Content format errors", all_errors, limit=args.max_errors)

    if failure:
        print("\n[FAIL] Submission package does NOT meet the selected profile.")
        return 2
    print("\n[PASS] Submission package meets the selected profile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

