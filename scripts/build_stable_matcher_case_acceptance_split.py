#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "rows_jsonl",
    "case_summary_csv",
    "filter_desc",
    "val_base_run_names",
    "val_seq_names",
    "stable_cases",
    "source_rows",
    "kept_rows",
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
        description="Build train/val JSONL splits from stable matcher-case consensus rows."
    )
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--case-summary-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-base-run-name", dest="val_base_run_names", nargs="*", default=[])
    parser.add_argument("--val-seq-name", dest="val_seq_names", nargs="*", default=[])
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def write_single_row_csv(path: Path, fieldnames: Iterable[str], row: Mapping[str, object]) -> None:
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
        "scripts/build_stable_matcher_case_acceptance_split.py",
        "--dataset",
        "MOT17",
        "--split",
        "stable_matcher_case_acceptance_split",
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


def _case_key(row: Mapping[str, object]) -> Tuple[object, ...]:
    return (
        str(row.get("base_run_name", "")),
        str(row.get("seq_name", "")),
        int(row.get("frame_id", -1)),
        int(row.get("track_gt_id", -1)),
        int(row.get("raw_det_index", -1)),
        int(row.get("fgas_det_index", -1)),
        int(row.get("base_best_det_index", -1)),
        int(row.get("base_best_det_raw_owner_track_gt_id", -1)),
    )


def _load_stable_case_keys(case_summary_csv: Path) -> Set[Tuple[object, ...]]:
    keys: Set[Tuple[object, ...]] = set()
    with case_summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row.get("stable_binary_case", 0)) != 1:
                continue
            keys.add(_case_key(row))
    return keys


def _split_name(
    row: Mapping[str, object],
    val_base_run_names: Set[str],
    val_seq_names: Set[str],
) -> str:
    if str(row.get("base_run_name", "")) in val_base_run_names:
        return "val"
    if str(row.get("seq_name", "")) in val_seq_names:
        return "val"
    return "train"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"

    val_base_run_names = {str(name) for name in list(args.val_base_run_names or [])}
    val_seq_names = {str(name) for name in list(args.val_seq_names or [])}

    summary_row: Dict[str, object] = {
        "rows_jsonl": str(Path(args.rows_jsonl)),
        "case_summary_csv": str(Path(args.case_summary_csv)),
        "filter_desc": "stable_binary_case=1",
        "val_base_run_names": "|".join(sorted(val_base_run_names)),
        "val_seq_names": "|".join(sorted(val_seq_names)),
        "stable_cases": 0,
        "source_rows": 0,
        "kept_rows": 0,
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
    append_registry(args, summary_csv, "running", "building stable matcher-case acceptance split")

    try:
        stable_case_keys = _load_stable_case_keys(Path(args.case_summary_csv))
        summary_row["stable_cases"] = int(len(stable_case_keys))
        seen_keys: Dict[str, Set[Tuple[object, ...]]] = {"train": set(), "val": set()}

        with Path(args.rows_jsonl).open("r", encoding="utf-8") as source_handle, train_jsonl.open(
            "w", encoding="utf-8"
        ) as train_handle, val_jsonl.open("w", encoding="utf-8") as val_handle:
            for line in source_handle:
                row = json.loads(line)
                summary_row["source_rows"] = int(summary_row["source_rows"]) + 1
                case_key = _case_key(row)
                if case_key not in stable_case_keys:
                    continue
                split = _split_name(
                    row,
                    val_base_run_names=val_base_run_names,
                    val_seq_names=val_seq_names,
                )
                dedup_key = (
                    case_key,
                    str(row.get("source_analysis_name", "")),
                    str(row.get("alt_run_name", "")),
                )
                if dedup_key in seen_keys[split]:
                    continue
                seen_keys[split].add(dedup_key)
                export_row = dict(row)
                export_row["label"] = int(export_row.get("utility_label", export_row.get("label", 0)))
                target = val_handle if split == "val" else train_handle
                target.write(json.dumps(export_row))
                target.write("\n")
                summary_row["kept_rows"] = int(summary_row["kept_rows"]) + 1
                summary_row[f"{split}_rows"] = int(summary_row[f"{split}_rows"]) + 1
                if int(export_row.get("label", -1)) == 1:
                    summary_row[f"{split}_pos"] = int(summary_row[f"{split}_pos"]) + 1

        summary_row["train_neg"] = int(summary_row["train_rows"]) - int(summary_row["train_pos"])
        summary_row["val_neg"] = int(summary_row["val_rows"]) - int(summary_row["val_pos"])
        if int(summary_row["train_rows"]) <= 0 or int(summary_row["val_rows"]) <= 0:
            raise ValueError("Empty train or val split after stable-case filtering.")
        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"stable matcher-case acceptance split built: train={summary_row['train_rows']} val={summary_row['val_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"stable matcher-case acceptance split failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
