#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.fgas.fgas.features.matcher_acceptance_features import (
    MATCHER_ACCEPTANCE_FEATURE_NAMES,
    build_matcher_acceptance_feature_vector,
)


REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"
SUMMARY_FIELDS = [
    "label_source",
    "filter_desc",
    "source_files",
    "source_rows",
    "kept_rows",
    "aligned_rows",
    "misaligned_rows",
    "beneficial_rows",
    "harmful_rows",
    "tie_rows",
    "pos_rows",
    "neg_rows",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a divergence-aligned matcher-case dataset from suffix TrackEval analysis rows."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--glob",
        default="outputs/*/suffix_trackeval_rows.jsonl",
        help="glob pattern, resolved relative to repo root",
    )
    parser.add_argument(
        "--label-source",
        choices=["rank", "hota", "assa", "idf1", "hota_idf1_consensus"],
        default="rank",
        help="suffix utility label source to export",
    )
    parser.add_argument(
        "--allow-misaligned",
        action="store_true",
        help="keep rows that are not the sequence's first output-diff frame",
    )
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="keep tie rows instead of dropping them",
    )
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
        "scripts/build_matcher_case_divergence_dataset.py",
        "--dataset",
        "MOT17",
        "--split",
        "matcher_case_divergence_dataset",
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


def _read_summary_row(summary_csv: Path) -> Dict[str, str]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle))


def _dedup_key(row: Mapping[str, object], source_analysis_name: str) -> Tuple[object, ...]:
    return (
        str(source_analysis_name),
        str(row.get("seq_name", "")),
        int(row.get("frame_id", -1)),
        int(row.get("track_index", -1)),
        int(row.get("track_gt_id", -1)),
        int(row.get("raw_det_index", -1)),
        int(row.get("fgas_det_index", -1)),
        int(row.get("base_best_det_index", -1)),
        int(row.get("base_best_det_raw_owner_track_gt_id", -1)),
    )


def _label_status(row: Mapping[str, object], label_source: str) -> str:
    if str(label_source) == "hota_idf1_consensus":
        hota_status = str(row.get("suffix_hota_status", "tie"))
        idf1_status = str(row.get("suffix_idf1_status", "tie"))
        if hota_status == idf1_status:
            return hota_status
        return "tie"
    field_name = {
        "rank": "suffix_rank_status",
        "hota": "suffix_hota_status",
        "assa": "suffix_assa_status",
        "idf1": "suffix_idf1_status",
    }[str(label_source)]
    return str(row.get(field_name, "tie"))


def _label_value(status: str) -> int:
    if str(status) == "beneficial":
        return 1
    if str(status) == "harmful":
        return 0
    raise ValueError(f"Unsupported non-binary status: {status}")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    rows_jsonl = out_dir / "rows.jsonl"

    summary_row: Dict[str, object] = {
        "label_source": str(args.label_source),
        "filter_desc": "suffix_trackeval_rows",
        "source_files": 0,
        "source_rows": 0,
        "kept_rows": 0,
        "aligned_rows": 0,
        "misaligned_rows": 0,
        "beneficial_rows": 0,
        "harmful_rows": 0,
        "tie_rows": 0,
        "pos_rows": 0,
        "neg_rows": 0,
        "status": "running",
        "error": "",
    }
    if not bool(args.allow_misaligned):
        summary_row["filter_desc"] = str(summary_row["filter_desc"]) + "|is_seq_first_output_diff_frame=1"
    if not bool(args.keep_ties):
        summary_row["filter_desc"] = str(summary_row["filter_desc"]) + "|drop_ties=1"

    write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
    append_registry(args, summary_csv, "running", "building divergence-aligned matcher-case dataset")

    try:
        source_paths = sorted(REPO_ROOT.glob(str(args.glob)))
        summary_row["source_files"] = int(len(source_paths))
        seen_keys = set()

        with rows_jsonl.open("w", encoding="utf-8") as out_handle:
            for path in source_paths:
                source_analysis_name = str(path.parent.name)
                source_summary_csv = path.parent / "summary.csv"
                summary_info = _read_summary_row(source_summary_csv) if source_summary_csv.is_file() else {}
                base_run_root = str(summary_info.get("base_run_root", ""))
                alt_run_root = str(summary_info.get("alt_run_root", ""))
                base_run_name = Path(base_run_root).name if base_run_root else ""
                alt_run_name = Path(alt_run_root).name if alt_run_root else ""

                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        summary_row["source_rows"] = int(summary_row["source_rows"]) + 1

                        is_aligned = int(row.get("is_seq_first_output_diff_frame", 0)) == 1
                        if is_aligned:
                            summary_row["aligned_rows"] = int(summary_row["aligned_rows"]) + 1
                        else:
                            summary_row["misaligned_rows"] = int(summary_row["misaligned_rows"]) + 1
                            if not bool(args.allow_misaligned):
                                continue

                        status = _label_status(row, str(args.label_source))
                        if status == "beneficial":
                            summary_row["beneficial_rows"] = int(summary_row["beneficial_rows"]) + 1
                        elif status == "harmful":
                            summary_row["harmful_rows"] = int(summary_row["harmful_rows"]) + 1
                        else:
                            summary_row["tie_rows"] = int(summary_row["tie_rows"]) + 1
                            if not bool(args.keep_ties):
                                continue

                        row_key = _dedup_key(row, source_analysis_name)
                        if row_key in seen_keys:
                            continue
                        seen_keys.add(row_key)

                        export_row = dict(row)
                        export_row["source_analysis_name"] = source_analysis_name
                        export_row["source_summary_csv"] = str(source_summary_csv)
                        export_row["base_run_name"] = base_run_name
                        export_row["alt_run_name"] = alt_run_name
                        export_row["utility_label_source"] = str(args.label_source)
                        export_row["utility_label_status"] = str(status)
                        if status in {"beneficial", "harmful"}:
                            export_row["utility_label"] = int(_label_value(status))
                        else:
                            export_row["utility_label"] = -1
                        export_row["original_label"] = int(export_row.get("label", 0))
                        export_row["feature_names"] = list(MATCHER_ACCEPTANCE_FEATURE_NAMES)
                        export_row["features"] = build_matcher_acceptance_feature_vector(export_row)
                        out_handle.write(json.dumps(export_row))
                        out_handle.write("\n")

                        summary_row["kept_rows"] = int(summary_row["kept_rows"]) + 1
                        if int(export_row["utility_label"]) == 1:
                            summary_row["pos_rows"] = int(summary_row["pos_rows"]) + 1
                        elif int(export_row["utility_label"]) == 0:
                            summary_row["neg_rows"] = int(summary_row["neg_rows"]) + 1

        summary_row["status"] = "success"
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(
            args,
            summary_csv,
            "success",
            f"divergence-aligned matcher-case dataset built: rows={summary_row['kept_rows']}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, SUMMARY_FIELDS, summary_row)
        append_registry(args, summary_csv, "failed", f"divergence-aligned matcher-case dataset failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
