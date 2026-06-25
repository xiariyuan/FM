#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import (
    append_registry,
    ensure_dir,
    upsert_plan,
    write_manifest,
    write_markdown,
    write_rows,
    write_single_row_csv,
    write_json,
)
from scripts.spot_common.metrics import mean_or_none, safe_ratio
from scripts.spot_common.mot_format import alignment_diagnostics, build_alignment, read_mot_txt


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "tracker_rows",
    "aligned_rows",
    "matched_rows",
    "match_rate",
    "mean_gt_iou",
    "num_gt_rows",
    "unmatched_tracker_rows",
    "unmatched_gt_rows",
    "duplicate_gt_assignments",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPOT GT alignment rows from tracker and GT MOT text.")
    parser.add_argument("--tracker-txt", required=True)
    parser.add_argument("--gt-txt", required=True)
    parser.add_argument("--dataset", default="MOT20")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seq-name", default="unknown_seq")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--out-dir", default="outputs/p0_dump")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    reports_dir = ensure_dir(out_dir / "reports")
    summary_csv = out_dir / "summary.csv"
    summary_row = {
        "status": "running",
        "error": "",
        "dataset": args.dataset,
        "split": args.split,
        "seq_name": args.seq_name,
        "tracker_rows": 0,
        "aligned_rows": 0,
        "matched_rows": 0,
        "match_rate": 0.0,
        "mean_gt_iou": 0.0,
        "num_gt_rows": 0,
        "unmatched_tracker_rows": 0,
        "unmatched_gt_rows": 0,
        "duplicate_gt_assignments": 0,
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)

    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    variant = out_dir.name
    tag = variant
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_p0_alignment",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"building GT alignment for {args.seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=args.dataset,
        split=args.split,
        tracker_family="spot_p0_alignment",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"building GT alignment for {args.seq_name}",
        key=f"spot_p0_alignment:{out_dir}",
    )

    try:
        tracker_rows = read_mot_txt(args.tracker_txt, treat_second_col_as_gt=False)
        gt_rows = read_mot_txt(args.gt_txt, treat_second_col_as_gt=True)
        aligned_rows = build_alignment(
            tracker_rows,
            gt_rows,
            iou_thresh=args.iou_thresh,
            dataset=args.dataset,
            split=args.split,
            seq_name=args.seq_name,
        )
        matched_rows = [row for row in aligned_rows if int(row["is_match"]) == 1]
        diagnostics = alignment_diagnostics(aligned_rows, gt_rows)
        gt_alignment_json = reports_dir / "gt_alignment.json"
        gt_alignment_csv = reports_dir / "gt_alignment_rows.csv"
        write_json(
            {
                "dataset": args.dataset,
                "split": args.split,
                "seq_name": args.seq_name,
                "tracker_txt": str(Path(args.tracker_txt).expanduser().resolve()),
                "gt_txt": str(Path(args.gt_txt).expanduser().resolve()),
                "iou_thresh": args.iou_thresh,
                "diagnostics": diagnostics,
                "rows": aligned_rows,
            },
            gt_alignment_json,
        )
        if aligned_rows:
            write_rows(gt_alignment_csv, aligned_rows[0].keys(), aligned_rows)
        mean_iou = mean_or_none(float(row["gt_iou"]) for row in matched_rows) or 0.0
        summary_row.update(
            {
                "status": "completed",
                "tracker_rows": len(tracker_rows),
                "aligned_rows": len(aligned_rows),
                "matched_rows": len(matched_rows),
                "match_rate": round(safe_ratio(len(matched_rows), len(aligned_rows)), 6),
                "mean_gt_iou": round(mean_iou, 6),
                "num_gt_rows": diagnostics["num_gt_rows"],
                "unmatched_tracker_rows": diagnostics["unmatched_tracker_rows"],
                "unmatched_gt_rows": diagnostics["unmatched_gt_rows"],
                "duplicate_gt_assignments": diagnostics["duplicate_gt_assignments"],
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        write_markdown(
            "\n".join(
                [
                    f"# GT Alignment Report: {args.seq_name}",
                    "",
                    f"- tracker_rows: {len(tracker_rows)}",
                    f"- aligned_rows: {len(aligned_rows)}",
                    f"- matched_rows: {len(matched_rows)}",
                    f"- match_rate: {summary_row['match_rate']}",
                    f"- mean_gt_iou: {summary_row['mean_gt_iou']}",
                    f"- num_gt_rows: {diagnostics['num_gt_rows']}",
                    f"- unmatched_tracker_rows: {diagnostics['unmatched_tracker_rows']}",
                    f"- unmatched_gt_rows: {diagnostics['unmatched_gt_rows']}",
                    f"- duplicate_gt_assignments: {diagnostics['duplicate_gt_assignments']}",
                ]
            ),
            reports_dir / "gt_alignment_report.md",
        )
        write_manifest(
            out_dir,
            phase="spot_p0_alignment",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics={
                "tracker_rows": len(tracker_rows),
                "aligned_rows": len(aligned_rows),
                "matched_rows": len(matched_rows),
                "match_rate": summary_row["match_rate"],
                "num_gt_rows": diagnostics["num_gt_rows"],
                "unmatched_tracker_rows": diagnostics["unmatched_tracker_rows"],
                "unmatched_gt_rows": diagnostics["unmatched_gt_rows"],
                "duplicate_gt_assignments": diagnostics["duplicate_gt_assignments"],
            },
            artifacts={
                "summary_csv": str(summary_csv),
                "gt_alignment_json": str(gt_alignment_json),
                "gt_alignment_csv": str(gt_alignment_csv),
            },
            notes=f"alignment for {args.seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_p0_alignment",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"GT alignment complete for {args.seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_p0_alignment",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"GT alignment complete for {args.seq_name}",
            key=f"spot_p0_alignment:{out_dir}",
        )
        return 0
    except Exception as exc:
        summary_row["status"] = "failed"
        summary_row["error"] = str(exc)
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        append_registry(
            kind="analysis",
            status="failed",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_p0_alignment",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"GT alignment failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=args.dataset,
            split=args.split,
            tracker_family="spot_p0_alignment",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"GT alignment failed: {exc}",
            key=f"spot_p0_alignment:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
