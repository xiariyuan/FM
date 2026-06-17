#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from queue_deep_ocsort_preassoc_force_recovery_anchor_gate_dance3 import (
    NEW_GATE_H16DO01,
    OLD_GATE_CKPT,
    anchored_variant_base,
)
from queue_deep_ocsort_preassoc_force_rewrite_next2h import (
    DECISION_FIELDS,
    QUEUE_FIELDS,
    REPO_ROOT,
    REGISTRY_CSV,
    build_cmd,
    child_finished_at,
    ensure_child_success,
    now_iso,
    read_rows,
    record_decision_from_child,
    run_step,
    timestamp_tag,
    update_row,
    write_rows,
)


ANALYSIS_FIELDS = [
    "step",
    "threshold",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "selected_matches",
    "force_rewrite_accepted_rows",
    "candidate_rows",
    "is_reference",
    "is_borderline",
    "is_low",
    "decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the current DanceTrack recovery-anchor threshold sweep, analyze it, and optionally launch a narrower refinement queue."
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--wait-summary-csv",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3_20260410_1"
            / "summary.csv"
        ),
    )
    parser.add_argument(
        "--wait-decision-csv",
        default=str(
            REPO_ROOT
            / "outputs"
            / "deep_ocsort_preassoc_force_recovery_anchor_threshold_sweep_dance3_20260410_1"
            / "decision_summary.csv"
        ),
    )
    parser.add_argument(
        "--reuse-raw-from",
        default=str(REPO_ROOT / "outputs" / "deep_ocsort_preassoc_acceptgate_smoke_20260408_2"),
    )
    parser.add_argument(
        "--seq-names",
        nargs="*",
        default=["dancetrack0081", "dancetrack0090", "dancetrack0094"],
    )
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=3 * 3600)
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "other",
        "--status",
        status,
        "--script",
        "scripts/queue_deep_ocsort_preassoc_force_recovery_anchor_autopilot_next3h.py",
        "--dataset",
        "DanceTrack",
        "--split",
        "val",
        "--tracker-family",
        "deep_ocsort_preassoc_force_recovery_anchor_autopilot",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_preassoc_force_recovery_anchor_autopilot_next3h",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def write_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).rstrip() + "\n")


def wait_for_queue_completion(summary_csv: Path, poll_sec: int, timeout_sec: int, log_path: Path) -> List[Dict[str, str]]:
    deadline = time.time() + float(timeout_sec)
    last_statuses: List[str] = []
    while time.time() < deadline:
        rows = read_rows(summary_csv)
        statuses = [str(row.get("status", "") or "") for row in rows]
        if rows and all(status not in {"", "pending", "running"} for status in statuses):
            write_log(
                log_path,
                [
                    f"[finished_at] {now_iso()}",
                    f"[source_summary_csv] {summary_csv}",
                    f"[statuses] {','.join(statuses)}",
                ],
            )
            return rows
        last_statuses = statuses
        write_log(
            log_path,
            [
                f"[heartbeat_at] {now_iso()}",
                f"[source_summary_csv] {summary_csv}",
                f"[statuses] {','.join(statuses) if statuses else 'missing'}",
            ],
        )
        time.sleep(float(poll_sec))
    raise TimeoutError(
        f"Timed out waiting for {summary_csv}; last_statuses={','.join(last_statuses) if last_statuses else 'missing'}"
    )


def parse_float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def find_row_by_threshold(rows: List[Dict[str, str]], target: float) -> Dict[str, str]:
    if not rows:
        raise ValueError("No decision rows to search.")
    return min(rows, key=lambda row: abs(parse_float(row, "recovery_anchor_gate_thresh") - target))


def analyze_source_decisions(source_rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not source_rows:
        raise FileNotFoundError("Missing source decision rows.")

    sorted_rows = sorted(source_rows, key=lambda row: parse_float(row, "recovery_anchor_gate_thresh"))
    ref_row = find_row_by_threshold(sorted_rows, 0.57)
    borderline_row = find_row_by_threshold(sorted_rows, 0.385)
    low_row = find_row_by_threshold(sorted_rows, 0.29)

    ref_hota = parse_float(ref_row, "delta_HOTA")
    ref_assa = parse_float(ref_row, "delta_AssA")
    ref_idf1 = parse_float(ref_row, "delta_IDF1")

    borderline_hota = parse_float(borderline_row, "delta_HOTA")
    borderline_assa = parse_float(borderline_row, "delta_AssA")
    borderline_idf1 = parse_float(borderline_row, "delta_IDF1")

    low_hota = parse_float(low_row, "delta_HOTA")
    low_assa = parse_float(low_row, "delta_AssA")
    low_idf1 = parse_float(low_row, "delta_IDF1")

    borderline_wins = (
        borderline_assa > ref_assa + 1e-9
        and borderline_hota >= ref_hota - 0.001
        and borderline_idf1 >= ref_idf1 - 0.001
    )
    low_is_worse = (
        low_hota < ref_hota - 0.005
        or low_assa < ref_assa - 0.005
        or low_idf1 < ref_idf1 - 0.005
    )

    action = "launch_refine_0385_band" if borderline_wins else "stop_threshold_search"
    reason = (
        "0.385 在不掉 HOTA/IDF1 的前提下让 AssA 回升，继续围绕 0.385 做窄带细调"
        if borderline_wins
        else "0.385 没有同时满足 AssA 回升且 HOTA/IDF1 不掉，阈值不是当前主瓶颈，停止继续向阈值发散"
    )

    analysis_rows: List[Dict[str, object]] = []
    for row in sorted_rows:
        step = str(row.get("step", ""))
        threshold = parse_float(row, "recovery_anchor_gate_thresh")
        analysis_rows.append(
            {
                "step": step,
                "threshold": f"{threshold:.4f}",
                "delta_HOTA": parse_float(row, "delta_HOTA"),
                "delta_AssA": parse_float(row, "delta_AssA"),
                "delta_IDF1": parse_float(row, "delta_IDF1"),
                "selected_matches": int(parse_float(row, "selected_matches")),
                "force_rewrite_accepted_rows": int(parse_float(row, "force_rewrite_accepted_rows")),
                "candidate_rows": int(parse_float(row, "candidate_rows")),
                "is_reference": int(step == str(ref_row.get("step", ""))),
                "is_borderline": int(step == str(borderline_row.get("step", ""))),
                "is_low": int(step == str(low_row.get("step", ""))),
                "decision": action if step == str(borderline_row.get("step", "")) else "",
                "notes": reason if step == str(borderline_row.get("step", "")) else "",
            }
        )

    decision = {
        "action": action,
        "reason": reason,
        "reference_threshold": parse_float(ref_row, "recovery_anchor_gate_thresh"),
        "reference_step": str(ref_row.get("step", "")),
        "borderline_threshold": parse_float(borderline_row, "recovery_anchor_gate_thresh"),
        "borderline_step": str(borderline_row.get("step", "")),
        "low_threshold": parse_float(low_row, "recovery_anchor_gate_thresh"),
        "low_step": str(low_row.get("step", "")),
        "borderline_wins": int(borderline_wins),
        "low_is_worse": int(low_is_worse),
    }
    return analysis_rows, decision


def build_refine_variants(seq_names: List[str]) -> List[Dict[str, object]]:
    base = anchored_variant_base(seq_names)
    thresholds = [0.375, 0.385, 0.395]
    variants: List[Dict[str, object]] = []
    for threshold in thresholds:
        tag = f"{threshold:.3f}".replace(".", "")
        variants.append(
            {
                **base,
                "step": f"anchor_h16_t{tag}_dance3",
                "name": f"anchor_h16_t{tag}_dance3",
                "notes": f"narrow refinement around the 0.385 recovery-anchor threshold with threshold={threshold:.3f}",
                "acceptance_gate_checkpoint": str(OLD_GATE_CKPT),
                "acceptance_gate_thresh": 0.9995,
                "recovery_anchor_gate_checkpoint": str(NEW_GATE_H16DO01),
                "recovery_anchor_gate_thresh": threshold,
            }
        )
    return variants


def run_refinement_queue(
    *,
    run_root: Path,
    reuse_raw_from: Path,
    seq_names: List[str],
    queue_rows: List[Dict[str, object]],
    summary_csv: Path,
    decision_csv: Path,
) -> Tuple[str, str]:
    runs_dir = run_root / "runs"
    logs_dir = run_root / "logs"
    decision_rows: List[Dict[str, object]] = []
    write_rows(decision_csv, DECISION_FIELDS, decision_rows)

    variants = build_refine_variants(seq_names)
    for variant in variants:
        step = str(variant["step"])
        queue_rows.append(
            {
                "step": step,
                "name": f"{run_root.name}_{step}",
                "status": "pending",
                "out_dir": str((runs_dir / step).resolve()),
                "summary_csv": str(((runs_dir / step) / "summary.csv").resolve()),
                "log_path": str((logs_dir / f"{step}.log").resolve()),
                "started_at": "",
                "finished_at": "",
                "notes": str(variant["notes"]),
            }
        )
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

    overall_status = "success"
    overall_notes = "完成 0.385 邻域的二阶段阈值细调"

    for variant in variants:
        step = str(variant["step"])
        update_row(queue_rows, step, status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        status = "success"
        finished_at = now_iso()
        notes = str(variant["notes"])

        try:
            out_dir = (runs_dir / step).resolve()
            log_path = (logs_dir / f"{step}.log").resolve()
            cmd = build_cmd(
                out_dir=out_dir,
                reuse_raw_from=reuse_raw_from,
                seq_names=list(seq_names),
                variant=variant,
            )
            rc = run_step(cmd, log_path, cwd=REPO_ROOT)
            child_summary = out_dir / "summary.csv"
            finished_at = now_iso()
            if rc != 0:
                status = "failed"
                notes = f"child return code {rc}"
            else:
                ensure_child_success(child_summary)
                finished_at = child_finished_at(child_summary)
                decision = record_decision_from_child(
                    decision_rows=decision_rows,
                    decision_csv=decision_csv,
                    step=step,
                    variant_name=str(variant["name"]),
                    child_root=out_dir,
                    variant=variant,
                    notes=(
                        f"{variant['notes']} "
                        f"gate_ckpt={Path(str(variant['acceptance_gate_checkpoint'])).name} "
                        f"gate_thresh={variant['acceptance_gate_thresh']} "
                        f"recovery_gate_ckpt={Path(str(variant['recovery_anchor_gate_checkpoint'])).name} "
                        f"recovery_gate_thresh={variant['recovery_anchor_gate_thresh']}"
                    ),
                )
                notes = (
                    f"delta_HOTA={float(decision['delta_HOTA']):+.3f} "
                    f"delta_AssA={float(decision['delta_AssA']):+.3f} "
                    f"delta_IDF1={float(decision['delta_IDF1']):+.3f} "
                    f"selected_matches={int(decision['selected_matches'])} "
                    f"force_rewrite_accepted={int(decision['force_rewrite_accepted_rows'])}"
                )
        except Exception as exc:
            status = "failed"
            finished_at = now_iso()
            notes = f"run step failed: {exc}"

        update_row(queue_rows, step, status=status, finished_at=finished_at, notes=notes)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        if status == "failed":
            overall_status = "failed"
            overall_notes = f"{step} failed: {notes}"

    return overall_status, overall_notes


def main() -> int:
    args = parse_args()
    run_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else (
            REPO_ROOT / "outputs" / f"deep_ocsort_preassoc_force_recovery_anchor_autopilot_next3h_{timestamp_tag()}"
        ).resolve()
    )
    wait_summary_csv = Path(args.wait_summary_csv).expanduser().resolve()
    wait_decision_csv = Path(args.wait_decision_csv).expanduser().resolve()
    reuse_raw_from = Path(args.reuse_raw_from).expanduser().resolve()
    seq_names = list(args.seq_names)

    summary_csv = run_root / "summary.csv"
    decision_csv = run_root / "decision_summary.csv"
    analysis_csv = run_root / "analysis.csv"
    logs_dir = run_root / "logs"

    queue_rows: List[Dict[str, object]] = [
        {
            "step": "wait_current_queue",
            "name": f"{run_root.name}_wait_current_queue",
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(wait_summary_csv),
            "log_path": str((logs_dir / "wait_current_queue.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "等待当前三序列二阶段阈值扫结束",
        },
        {
            "step": "decide_followup",
            "name": f"{run_root.name}_decide_followup",
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(analysis_csv.resolve()),
            "log_path": str((logs_dir / "decide_followup.log").resolve()),
            "started_at": "",
            "finished_at": "",
            "notes": "读取当前阈值扫结果并决定是否继续做 0.385 邻域细调",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
    write_rows(analysis_csv, ANALYSIS_FIELDS, [])
    write_rows(decision_csv, DECISION_FIELDS, [])
    append_registry(
        summary_csv,
        run_root,
        "running",
        "开始等待当前三序列二阶段阈值扫，并准备自动决定后续 2 到 3 小时实验",
        args.registry_csv,
    )

    overall_status = "success"
    overall_notes = "自动接力完成"

    try:
        update_row(queue_rows, "wait_current_queue", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        wait_rows = wait_for_queue_completion(
            wait_summary_csv,
            poll_sec=args.poll_sec,
            timeout_sec=args.timeout_sec,
            log_path=logs_dir / "wait_current_queue.log",
        )
        wait_statuses = {str(row.get("status", "") or "") for row in wait_rows}
        wait_failed = any(status == "failed" for status in wait_statuses)
        wait_notes = (
            "上游阈值扫已结束，但存在失败步骤"
            if wait_failed
            else "上游阈值扫已全部结束，可进入结果决策"
        )
        update_row(queue_rows, "wait_current_queue", status="failed" if wait_failed else "success", finished_at=now_iso(), notes=wait_notes)
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)
        if wait_failed:
            overall_status = "failed"
            overall_notes = wait_notes
            append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
            return 1

        update_row(queue_rows, "decide_followup", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        source_decisions = read_rows(wait_decision_csv)
        analysis_rows, followup_decision = analyze_source_decisions(source_decisions)
        write_rows(analysis_csv, ANALYSIS_FIELDS, analysis_rows)
        write_log(
            logs_dir / "decide_followup.log",
            [
                f"[finished_at] {now_iso()}",
                f"[source_decision_csv] {wait_decision_csv}",
                f"[action] {followup_decision['action']}",
                f"[reason] {followup_decision['reason']}",
                f"[borderline_wins] {followup_decision['borderline_wins']}",
                f"[low_is_worse] {followup_decision['low_is_worse']}",
            ],
        )
        update_row(
            queue_rows,
            "decide_followup",
            status="success",
            finished_at=now_iso(),
            notes=str(followup_decision["reason"]),
        )
        write_rows(summary_csv, QUEUE_FIELDS, queue_rows)

        if str(followup_decision["action"]) == "launch_refine_0385_band":
            overall_status, overall_notes = run_refinement_queue(
                run_root=run_root,
                reuse_raw_from=reuse_raw_from,
                seq_names=seq_names,
                queue_rows=queue_rows,
                summary_csv=summary_csv,
                decision_csv=decision_csv,
            )
        else:
            overall_status = "success"
            overall_notes = str(followup_decision["reason"])
    except Exception as exc:
        overall_status = "failed"
        overall_notes = f"自动接力失败: {exc}"

    append_registry(summary_csv, run_root, overall_status, overall_notes, args.registry_csv)
    return 0 if overall_status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
