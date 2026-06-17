#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
PAIR_SCRIPT = REPO_ROOT / "scripts" / "run_official_bytetrack_local_conflict_halfval_pair.py"
DEFAULT_SCORECARD = (
    REPO_ROOT
    / "outputs"
    / "official_bytetrack_posthost_rule_scorecard_noprefilter_20260329_022800"
    / "coefficients.json"
)

SUMMARY_FIELDS = [
    "queue_name",
    "run_name",
    "scorecard_json",
    "score_thresh",
    "use_legacy_prefilter",
    "status",
    "run_dir",
    "result_csv",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDSW",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially run oracle-guided posthost rule scorecard follow-up evals."
    )
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--queue-name", required=True)
    parser.add_argument("--scorecard-json", default=str(DEFAULT_SCORECARD))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.95, 0.90])
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})


def main() -> None:
    args = parse_args()
    queue_dir = Path(args.queue_dir).resolve()
    queue_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = queue_dir / "summary.csv"
    log_dir = queue_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for threshold in args.thresholds:
        threshold_tag = f"{threshold:.2f}".replace(".", "")
        run_name = f"{args.queue_name}_t{threshold_tag}"
        run_dir = REPO_ROOT / "outputs" / run_name
        rows.append(
            {
                "queue_name": args.queue_name,
                "run_name": run_name,
                "scorecard_json": str(Path(args.scorecard_json).resolve()),
                "score_thresh": float(threshold),
                "use_legacy_prefilter": False,
                "status": "pending",
                "run_dir": str(run_dir),
                "result_csv": str(run_dir / "result.csv"),
                "delta_HOTA": "",
                "delta_AssA": "",
                "delta_IDF1": "",
                "delta_MOTA": "",
                "delta_IDSW": "",
                "error": "",
            }
        )
    write_summary(summary_csv, rows)

    for row in rows:
        row["status"] = "running"
        write_summary(summary_csv, rows)
        log_path = log_dir / f"{row['run_name']}.log"
        cmd = [
            args.python_bin,
            str(PAIR_SCRIPT),
            "--out-dir",
            row["run_dir"],
            "--experiment-name",
            row["run_name"],
            "--protocol-tag",
            "official_bytetrack_posthost_one_edit_rule_scorecard_halfval",
            "--plugin-mode",
            "posthost_one_edit_rule",
            "--posthost-rule-large-only",
            "--posthost-rule-scorecard-json",
            str(Path(args.scorecard_json).resolve()),
            "--posthost-rule-score-thresh",
            str(row["score_thresh"]),
            "--posthost-rule-no-legacy-prefilter",
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log_fp:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=REPO_ROOT,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                )
            result_path = Path(row["result_csv"])
            if result_path.exists():
                with result_path.open("r", encoding="utf-8", newline="") as f:
                    result_row = next(csv.DictReader(f))
                row["delta_HOTA"] = result_row.get("delta_HOTA", "")
                row["delta_AssA"] = result_row.get("delta_AssA", "")
                row["delta_IDF1"] = result_row.get("delta_IDF1", "")
                row["delta_MOTA"] = result_row.get("delta_MOTA", "")
                row["delta_IDSW"] = result_row.get("delta_IDSW", "")
                row["status"] = result_row.get("status", "success") or "success"
                row["error"] = result_row.get("error", "")
            else:
                row["status"] = "failed"
                row["error"] = "missing result.csv"
        except subprocess.CalledProcessError as exc:
            row["status"] = "failed"
            row["error"] = f"subprocess_exit_{exc.returncode}"
            write_summary(summary_csv, rows)
            raise
        write_summary(summary_csv, rows)


if __name__ == "__main__":
    main()
