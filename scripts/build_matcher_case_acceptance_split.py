#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.features.matcher_acceptance_features import (
    MATCHER_ACCEPTANCE_FEATURE_NAMES,
    build_matcher_acceptance_feature_vector,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "filter_desc",
    "val_run_names",
    "source_files",
    "train_rows",
    "train_pos",
    "train_neg",
    "val_rows",
    "val_pos",
    "val_neg",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val JSONL splits from matcher_case_rows exports for matcher acceptance-gate prototyping."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--glob",
        default="outputs/*/matcher_case_rows.jsonl",
        help="glob pattern, resolved relative to repo root",
    )
    parser.add_argument(
        "--val-run-name",
        dest="val_run_names",
        nargs="+",
        required=True,
        help="output directory names reserved for validation",
    )
    parser.add_argument(
        "--require-owner-other-track",
        action="store_true",
        help="keep only rows where base_best_det_raw_owned_by_other_track=1",
    )
    parser.add_argument(
        "--solver-changed-row",
        type=int,
        choices=[0, 1],
        default=0,
        help="row type to export: 0 keeps solver-unchanged rows, 1 keeps solver-changed rows",
    )
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_registry(args: argparse.Namespace, summary_csv: Path, status: str, notes: str) -> None:
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
        "scripts/build_matcher_case_acceptance_split.py",
        "--dataset",
        "MOT17",
        "--split",
        "matcher_case_acceptance_split",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        Path(args.out_dir).name,
        "--tag",
        Path(args.out_dir).name,
        "--run-root",
        str(Path(args.out_dir)),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def _keep_row(
    row: Dict[str, object],
    *,
    require_owner_other_track: bool,
    solver_changed_row: int,
) -> bool:
    keep = bool(
        int(row.get("takeover_applied", 0)) == 1
        and int(row.get("solver_changed_row", 0)) == int(solver_changed_row)
        and int(row.get("raw_det_index", -1)) >= 0
        and int(row.get("changed_match", 0)) == 1
    )
    if keep and require_owner_other_track:
        keep = bool(int(row.get("base_best_det_raw_owned_by_other_track", 0)) == 1)
    return keep


def _dedup_key(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        str(row.get("seq_name", "")),
        int(row.get("frame_id", -1)),
        int(row.get("track_gt_id", -1)),
        int(row.get("raw_det_index", -1)),
        int(row.get("fgas_det_index", -1)),
        int(row.get("base_best_det_index", -1)),
        int(row.get("base_best_det_raw_owner_track_gt_id", -1)),
        int(row.get("row_no_match", 0)),
        int(row.get("component_row_count", 0)),
        int(row.get("component_col_count", 0)),
        str(row.get("flip_type", "")),
        int(row.get("label", 0)),
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"

    summary_row: Dict[str, object] = {
        "filter_desc": f"takeover_applied=1|solver_changed_row={int(args.solver_changed_row)}|raw_det_index>=0|changed_match=1",
        "val_run_names": "|".join(str(name) for name in args.val_run_names),
        "source_files": 0,
        "train_rows": 0,
        "train_pos": 0,
        "train_neg": 0,
        "val_rows": 0,
        "val_pos": 0,
        "val_neg": 0,
        "status": "running",
        "error": "",
    }
    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building matcher-case acceptance split")

    try:
        val_run_names: Set[str] = {str(name) for name in args.val_run_names}
        if bool(args.require_owner_other_track):
            summary_row["filter_desc"] = str(summary_row["filter_desc"]) + "|base_best_det_raw_owned_by_other_track=1"
        source_paths = sorted(REPO_ROOT.glob(str(args.glob)))
        summary_row["source_files"] = int(len(source_paths))
        seen_keys = {"train": set(), "val": set()}
        with train_jsonl.open("w", encoding="utf-8") as train_handle, val_jsonl.open("w", encoding="utf-8") as val_handle:
            for path in source_paths:
                split = "val" if path.parent.name in val_run_names else "train"
                target = val_handle if split == "val" else train_handle
                with path.open("r", encoding="utf-8") as source_handle:
                    for line in source_handle:
                        row = json.loads(line)
                        if not _keep_row(
                            row,
                            require_owner_other_track=bool(args.require_owner_other_track),
                            solver_changed_row=int(args.solver_changed_row),
                        ):
                            continue
                        row_key = (str(path.parent.name),) + _dedup_key(row)
                        if row_key in seen_keys[split]:
                            continue
                        seen_keys[split].add(row_key)
                        row["feature_names"] = list(MATCHER_ACCEPTANCE_FEATURE_NAMES)
                        row["features"] = build_matcher_acceptance_feature_vector(row)
                        target.write(json.dumps(row))
                        target.write("\n")
                        summary_row[f"{split}_rows"] = int(summary_row[f"{split}_rows"]) + 1
                        if int(row.get("label", 0)) == 1:
                            summary_row[f"{split}_pos"] = int(summary_row[f"{split}_pos"]) + 1
        summary_row["train_neg"] = int(summary_row["train_rows"]) - int(summary_row["train_pos"])
        summary_row["val_neg"] = int(summary_row["val_rows"]) - int(summary_row["val_pos"])
        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"matcher-case acceptance split built: train={summary_row['train_rows']} val={summary_row['val_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"matcher-case acceptance split failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
