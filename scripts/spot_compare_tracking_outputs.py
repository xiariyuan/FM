#!/usr/bin/env python3
"""Strict parity checker for SPOT paired tracking outputs.

Compares two result directories, usually:
  00_baseline/track_results
  01_spot_noop/track_results

Default mode is byte-exact. Use --numeric to parse MOT txt rows and compare
numeric values with a small tolerance when a runner rewrites formatting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_result_files(root: Path, pattern: str) -> Dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"missing directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    files = {}
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            files[rel] = path
    return files


def parse_mot_rows(path: Path) -> List[Tuple[float, ...]]:
    rows: List[Tuple[float, ...]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            try:
                rows.append(tuple(float(cell.strip()) for cell in row))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: non-numeric MOT row: {row}") from exc
    return rows


def numeric_equal(a: Path, b: Path, tol: float) -> Tuple[bool, str]:
    rows_a = parse_mot_rows(a)
    rows_b = parse_mot_rows(b)
    if len(rows_a) != len(rows_b):
        return False, f"row_count {len(rows_a)} != {len(rows_b)}"
    for idx, (ra, rb) in enumerate(zip(rows_a, rows_b), start=1):
        if len(ra) != len(rb):
            return False, f"row {idx} width {len(ra)} != {len(rb)}"
        for col, (va, vb) in enumerate(zip(ra, rb), start=1):
            if math.isnan(va) or math.isnan(vb) or abs(va - vb) > tol:
                return False, f"row {idx} col {col} {va} != {vb} tol={tol}"
    return True, ""


def compare_dirs(baseline: Path, candidate: Path, pattern: str, numeric: bool, tol: float) -> dict:
    base_files = list_result_files(baseline, pattern)
    cand_files = list_result_files(candidate, pattern)
    base_keys = set(base_files)
    cand_keys = set(cand_files)
    report = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "pattern": pattern,
        "mode": "numeric" if numeric else "byte_exact",
        "tolerance": tol if numeric else 0.0,
        "baseline_file_count": len(base_files),
        "candidate_file_count": len(cand_files),
        "missing_in_candidate": sorted(base_keys - cand_keys),
        "extra_in_candidate": sorted(cand_keys - base_keys),
        "changed": [],
        "matched": [],
        "parity_ok": False,
    }
    for key in sorted(base_keys & cand_keys):
        a = base_files[key]
        b = cand_files[key]
        if numeric:
            ok, reason = numeric_equal(a, b, tol)
            if ok:
                report["matched"].append({"file": key})
            else:
                report["changed"].append({"file": key, "reason": reason})
        else:
            ha = sha256_file(a)
            hb = sha256_file(b)
            if ha == hb:
                report["matched"].append({"file": key, "sha256": ha})
            else:
                report["changed"].append({"file": key, "baseline_sha256": ha, "candidate_sha256": hb})
    report["parity_ok"] = not report["missing_in_candidate"] and not report["extra_in_candidate"] and not report["changed"]
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict SPOT paired-output parity checker")
    parser.add_argument("--baseline", required=True, type=Path, help="baseline result directory")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate result directory")
    parser.add_argument("--pattern", default="*.txt", help="glob pattern under each directory")
    parser.add_argument("--numeric", action="store_true", help="compare parsed MOT numeric rows instead of exact bytes")
    parser.add_argument("--tol", type=float, default=1e-6, help="numeric tolerance for --numeric")
    parser.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = compare_dirs(args.baseline, args.candidate, args.pattern, args.numeric, args.tol)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["parity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
