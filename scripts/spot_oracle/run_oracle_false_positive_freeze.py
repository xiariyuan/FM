#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spot_common.io_utils import (
    append_registry,
    ensure_dir,
    read_json,
    upsert_plan,
    write_json,
    write_manifest,
    write_markdown,
    write_single_row_csv,
)
from scripts.spot_common.metrics import safe_ratio


SUMMARY_FIELDS = [
    "status",
    "error",
    "dataset",
    "split",
    "seq_name",
    "num_correct_matches",
    "num_false_freeze",
    "false_freeze_rate",
    "protected_update_loss_count",
    "potential_fn_risk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle 0D proxy for false positive freeze damage.")
    parser.add_argument("--alignment-json", required=True)
    parser.add_argument("--out-dir", default="outputs/oracle_gate/0D_false_positive_damage")
    parser.add_argument("--freeze-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    summary_csv = out_dir / "summary.csv"
    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    payload = read_json(args.alignment_json)
    dataset = str(payload.get("dataset", "unknown"))
    split = str(payload.get("split", "unknown"))
    seq_name = str(payload.get("seq_name", "unknown_seq"))
    variant = out_dir.name
    tag = variant
    summary_row = {
        "status": "running",
        "error": "",
        "dataset": dataset,
        "split": split,
        "seq_name": seq_name,
        "num_correct_matches": 0,
        "num_false_freeze": 0,
        "false_freeze_rate": float(args.freeze_rate),
        "protected_update_loss_count": 0,
        "potential_fn_risk": 0.0,
    }
    write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
    append_registry(
        kind="analysis",
        status="running",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0D",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"false positive freeze oracle running for {seq_name}",
    )
    upsert_plan(
        status="running",
        kind="analysis",
        script=script_path,
        dataset=dataset,
        split=split,
        tracker_family="spot_oracle_0D",
        variant=variant,
        tag=tag,
        run_root=out_dir,
        summary_csv=summary_csv,
        notes=f"false positive freeze oracle running for {seq_name}",
        key=f"spot_oracle_0D:{out_dir}",
    )

    try:
        rows = list(payload.get("rows", []))
        correct = [row for row in rows if int(row.get("gt_id", -1)) > 0]
        sample_n = min(len(correct), int(round(len(correct) * float(args.freeze_rate))))
        sampled = random.sample(correct, sample_n) if sample_n > 0 else []
        by_track = Counter(int(row["track_id"]) for row in sampled)
        protected_update_loss_count = sum(1 for count in by_track.values() if count > 0)
        potential_fn_risk = round(safe_ratio(protected_update_loss_count, len(correct)), 6)
        summary_row.update(
            {
                "status": "completed",
                "num_correct_matches": len(correct),
                "num_false_freeze": len(sampled),
                "protected_update_loss_count": protected_update_loss_count,
                "potential_fn_risk": potential_fn_risk,
            }
        )
        write_single_row_csv(summary_csv, summary_row, SUMMARY_FIELDS)
        metrics = {
            "num_correct_matches": len(correct),
            "num_false_freeze": len(sampled),
            "false_freeze_rate": float(args.freeze_rate),
            "protected_update_loss_count": protected_update_loss_count,
            "potential_fn_risk": potential_fn_risk,
            "num_sampled_tracks": len(by_track),
        }
        write_json(metrics, out_dir / "oracle_false_positive_freeze_metrics.json")
        write_json({"sampled_events": sampled}, out_dir / "sampled_false_freeze_events.json")
        write_markdown(
            "\n".join(
                [
                    "# Oracle 0D False Positive Freeze Damage",
                    "",
                    f"- seq_name: {seq_name}",
                    f"- num_correct_matches: {len(correct)}",
                    f"- num_false_freeze: {len(sampled)}",
                    f"- protected_update_loss_count: {protected_update_loss_count}",
                    f"- potential_fn_risk: {potential_fn_risk}",
                ]
            ),
            out_dir / "oracle_false_positive_freeze_report.md",
        )
        write_manifest(
            out_dir,
            phase="oracle_0D_false_positive_freeze",
            script=script_path,
            args=vars(args),
            status="ok",
            metrics=metrics,
            artifacts={
                "summary_csv": str(summary_csv),
                "metrics_json": str(out_dir / "oracle_false_positive_freeze_metrics.json"),
                "sampled_json": str(out_dir / "sampled_false_freeze_events.json"),
            },
            notes=f"false positive freeze oracle for {seq_name}",
        )
        append_registry(
            kind="analysis",
            status="success",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0D",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"false positive freeze oracle complete for {seq_name}",
        )
        upsert_plan(
            status="completed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0D",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"false positive freeze oracle complete for {seq_name}",
            key=f"spot_oracle_0D:{out_dir}",
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
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0D",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"false positive freeze oracle failed: {exc}",
        )
        upsert_plan(
            status="failed",
            kind="analysis",
            script=script_path,
            dataset=dataset,
            split=split,
            tracker_family="spot_oracle_0D",
            variant=variant,
            tag=tag,
            run_root=out_dir,
            summary_csv=summary_csv,
            notes=f"false positive freeze oracle failed: {exc}",
            key=f"spot_oracle_0D:{out_dir}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
