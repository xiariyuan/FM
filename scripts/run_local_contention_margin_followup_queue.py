#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_deep_ocsort_preassoc_competition_dataset_eval.py"
QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]


@dataclass
class MetricRow:
    name: str
    seq: str
    HOTA: float
    AssA: float
    IDF1: float
    MOTA: float
    IDs: float
    Frag: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch the current local-contention margin smoke run and automatically queue follow-up experiments."
    )
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--watch-run-root", required=True)
    parser.add_argument("--ranker-smoke-root", required=True)
    parser.add_argument("--heuristic-smoke-root", required=True)
    parser.add_argument("--raw-reuse-root", required=True)
    parser.add_argument("--ranker-checkpoint", required=True)
    parser.add_argument("--eval-script", default=str(DEFAULT_EVAL_SCRIPT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--poll-sec", type=int, default=120)
    parser.add_argument("--full-batch-frame-budget", type=int, default=4000)
    return parser.parse_args()


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"missing queue step: {step}")


def cancel_pending_rows(rows: List[Dict[str, object]], *, reason: str) -> None:
    finished_at = now_iso()
    for row in rows:
        status = str(row.get("status", "")).strip()
        if status == "pending":
            row["status"] = "cancelled"
            row["finished_at"] = finished_at
            row["notes"] = reason


def read_summary_status(summary_csv: Path, step: str) -> str:
    if not summary_csv.is_file():
        return "missing"
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("step", "")) == step:
                return str(row.get("status", "")).strip() or "missing"
    return "missing"


def wait_for_run_completion(summary_csv: Path, poll_sec: int) -> str:
    while True:
        status = read_summary_status(summary_csv, "compare")
        if status in {"success", "failed", "cancelled"}:
            return status
        time.sleep(max(5, poll_sec))


def load_metric_row(metrics_compare_csv: Path, label: str) -> MetricRow:
    with metrics_compare_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("name", "")) != label:
                continue
            return MetricRow(
                name=str(row.get("name", "")),
                seq=str(row.get("seq", "")),
                HOTA=float(row.get("HOTA", 0.0) or 0.0),
                AssA=float(row.get("AssA", 0.0) or 0.0),
                IDF1=float(row.get("IDF1", 0.0) or 0.0),
                MOTA=float(row.get("MOTA", 0.0) or 0.0),
                IDs=float(row.get("IDs", 0.0) or 0.0),
                Frag=float(row.get("Frag", 0.0) or 0.0),
            )
    raise FileNotFoundError(f"missing label {label} in {metrics_compare_csv}")


def compare_candidates_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "tag",
        "run_root",
        "seq",
        "HOTA",
        "AssA",
        "IDF1",
        "MOTA",
        "IDs",
        "Frag",
        "delta_HOTA_vs_ranker_smoke",
        "delta_IDF1_vs_ranker_smoke",
        "delta_HOTA_vs_heuristic_smoke",
        "delta_IDF1_vs_heuristic_smoke",
    ]
    write_rows(path, fieldnames, rows)


def run_experiment(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def build_common_args(
    *,
    python_bin: str,
    eval_script: Path,
    out_root: Path,
    raw_reuse_root: Path,
    ranker_checkpoint: Path,
    min_margin_to_second: float,
    margin_bias: float,
    seq_names: List[str] | None,
    full_batch_frame_budget: int,
) -> List[str]:
    cmd = [
        python_bin,
        str(eval_script),
        "--benchmark",
        "DanceTrack",
        "--out-root",
        str(out_root),
        "--reuse-raw-from",
        str(raw_reuse_root),
        "--preassoc-stale-competition-min-time-since-update",
        "2",
        "--preassoc-stale-competition-max-time-since-update",
        "8",
        "--preassoc-stale-competition-min-hits",
        "8",
        "--preassoc-stale-competition-min-box-iou",
        "0.5",
        "--preassoc-stale-competition-min-edge-score",
        "0.0",
        "--preassoc-stale-competition-bias",
        "0.1",
        "--preassoc-stale-competition-iou-scale",
        "0.0",
        "--preassoc-stale-competition-require-raw-owner",
        "--preassoc-stale-competition-min-age-gap-vs-owner",
        "10",
        "--preassoc-stale-competition-owner-max-hits",
        "12",
        "--preassoc-stale-competition-owner-edge-penalty",
        "0.05",
        "--preassoc-stale-competition-max-owner-edge-deficit",
        "-1.0",
        "--preassoc-stale-competition-force-owner-edge-deficit-arg",
        "--preassoc-stale-competition-block-owner-on-reclaim",
        "--local-contention-export-jsonl",
        str(out_root / "local_contention_units.jsonl"),
        "--local-contention-topk",
        "3",
        "--local-contention-min-box-iou",
        "0.5",
        "--local-contention-max-time-since-update",
        "8",
        "--local-contention-min-challenger-hits",
        "3",
        "--local-contention-owner-weak-hits",
        "8",
        "--local-contention-ranker-checkpoint",
        str(ranker_checkpoint),
        "--local-contention-ranker-thresh",
        "0.99",
        "--local-contention-ranker-bias",
        "0.0",
        "--local-contention-ranker-min-margin-to-second",
        str(min_margin_to_second),
        "--local-contention-ranker-margin-bias",
        str(margin_bias),
    ]
    if seq_names:
        cmd.extend(["--seq-names", *seq_names])
    else:
        cmd.extend(
            [
                "--competition-track-max-frames-per-batch",
                str(full_batch_frame_budget),
            ]
        )
    return cmd


def choose_best_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object]:
    def sort_key(row: Dict[str, object]) -> tuple[float, float, float]:
        return (
            float(row["delta_HOTA_vs_ranker_smoke"]),
            float(row["delta_IDF1_vs_ranker_smoke"]),
            float(row["HOTA"]),
        )

    return max(candidates, key=sort_key)


def main() -> None:
    args = parse_args()
    queue_root = Path(args.queue_root).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    queue_summary = queue_root / "summary.csv"
    queue_log_dir = queue_root / "logs"
    decision_csv = queue_root / "decision_candidates.csv"
    decision_txt = queue_root / "decision.txt"

    watch_run_root = Path(args.watch_run_root).resolve()
    ranker_smoke_root = Path(args.ranker_smoke_root).resolve()
    heuristic_smoke_root = Path(args.heuristic_smoke_root).resolve()
    raw_reuse_root = Path(args.raw_reuse_root).resolve()
    ranker_checkpoint = Path(args.ranker_checkpoint).resolve()
    eval_script = Path(args.eval_script).resolve()

    rows: List[Dict[str, object]] = [
        {
            "step": "watch_current",
            "name": watch_run_root.name,
            "status": "running",
            "out_dir": str(watch_run_root),
            "summary_csv": str(queue_summary),
            "log_path": str(queue_log_dir / "watch_current.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": "等待当前冒烟分差实验完成并读取结果",
        },
        {
            "step": "followup_1",
            "name": "",
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(queue_summary),
            "log_path": str(queue_log_dir / "followup_1.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
        {
            "step": "followup_2",
            "name": "",
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(queue_summary),
            "log_path": str(queue_log_dir / "followup_2.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
        {
            "step": "followup_3",
            "name": "",
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(queue_summary),
            "log_path": str(queue_log_dir / "followup_3.log"),
            "started_at": "",
            "finished_at": "",
            "notes": "",
        },
    ]
    write_rows(queue_summary, QUEUE_FIELDS, rows)

    current_status = wait_for_run_completion(watch_run_root / "summary.csv", args.poll_sec)
    update_row(rows, "watch_current", status=current_status, finished_at=now_iso(), notes=f"当前实验结束状态: {current_status}")
    write_rows(queue_summary, QUEUE_FIELDS, rows)
    if current_status != "success":
        cancel_pending_rows(rows, reason="上游当前实验未成功，后续步骤不执行")
        write_rows(queue_summary, QUEUE_FIELDS, rows)
        decision_txt.write_text(f"[stopped_at] {now_iso()}\n当前实验未成功结束，状态为 {current_status}，自动队列停止。\n", encoding="utf-8")
        return

    ranker_smoke = load_metric_row(ranker_smoke_root / "metrics_compare.csv", "competition")
    heuristic_smoke = load_metric_row(heuristic_smoke_root / "metrics_compare.csv", "competition")
    current_smoke = load_metric_row(watch_run_root / "metrics_compare.csv", "competition")

    candidate_rows: List[Dict[str, object]] = []
    full_result_lines: List[str] = []

    def add_candidate(tag: str, run_root: Path, metric: MetricRow) -> None:
        candidate_rows.append(
            {
                "tag": tag,
                "run_root": str(run_root),
                "seq": metric.seq,
                "HOTA": metric.HOTA,
                "AssA": metric.AssA,
                "IDF1": metric.IDF1,
                "MOTA": metric.MOTA,
                "IDs": metric.IDs,
                "Frag": metric.Frag,
                "delta_HOTA_vs_ranker_smoke": metric.HOTA - ranker_smoke.HOTA,
                "delta_IDF1_vs_ranker_smoke": metric.IDF1 - ranker_smoke.IDF1,
                "delta_HOTA_vs_heuristic_smoke": metric.HOTA - heuristic_smoke.HOTA,
                "delta_IDF1_vs_heuristic_smoke": metric.IDF1 - heuristic_smoke.IDF1,
            }
        )

    add_candidate("current_margin", watch_run_root, current_smoke)
    compare_candidates_csv(decision_csv, candidate_rows)

    launch_plan: List[Dict[str, object]] = []
    current_beats_ranker_smoke = current_smoke.HOTA >= ranker_smoke.HOTA and current_smoke.IDF1 >= ranker_smoke.IDF1
    current_beats_heuristic_smoke = current_smoke.HOTA >= heuristic_smoke.HOTA and current_smoke.IDF1 >= heuristic_smoke.IDF1

    if current_beats_ranker_smoke and current_beats_heuristic_smoke:
        launch_plan.append(
            {
                "step": "followup_1",
                "name": "deep_ocsort_local_contention_ranker_dance_margin_full_20260407_1",
                "min_margin_to_second": 0.05,
                "margin_bias": 0.5,
                "seq_names": None,
                "notes": "当前冒烟同时优于旧学习版和启发式，直接扩到全量分批",
            }
        )
    else:
        launch_plan.extend(
            [
                {
                    "step": "followup_1",
                    "name": "deep_ocsort_local_contention_ranker_smoke_margin_bias02_20260407_1",
                    "min_margin_to_second": 0.05,
                    "margin_bias": 0.2,
                    "seq_names": ["dancetrack0019", "dancetrack0047", "dancetrack0063", "dancetrack0090"],
                    "notes": "当前冒烟未稳定优于旧版本，先试更保守的分差增益",
                },
                {
                    "step": "followup_2",
                    "name": "deep_ocsort_local_contention_ranker_smoke_margin_min002_20260407_1",
                    "min_margin_to_second": 0.02,
                    "margin_bias": 0.5,
                    "seq_names": ["dancetrack0019", "dancetrack0047", "dancetrack0063", "dancetrack0090"],
                    "notes": "若上一轮仍不够稳，再放宽分差门槛",
                },
            ]
        )

    for item in launch_plan:
        step = str(item["step"])
        run_root = REPO_ROOT / "outputs" / str(item["name"])
        log_path = queue_log_dir / f"{step}.log"
        update_row(
            rows,
            step,
            name=run_root.name,
            status="running",
            out_dir=str(run_root),
            started_at=now_iso(),
            notes=str(item["notes"]),
        )
        write_rows(queue_summary, QUEUE_FIELDS, rows)

        if not (run_root / "summary.csv").is_file() or read_summary_status(run_root / "summary.csv", "compare") not in {"success", "running"}:
            cmd = build_common_args(
                python_bin=args.python_bin,
                eval_script=eval_script,
                out_root=run_root,
                raw_reuse_root=raw_reuse_root,
                ranker_checkpoint=ranker_checkpoint,
                min_margin_to_second=float(item["min_margin_to_second"]),
                margin_bias=float(item["margin_bias"]),
                seq_names=item["seq_names"],  # type: ignore[arg-type]
                full_batch_frame_budget=int(args.full_batch_frame_budget),
            )
            rc = run_experiment(cmd, log_path)
            if rc != 0:
                update_row(rows, step, status="failed", finished_at=now_iso(), log_path=str(log_path), notes=f"{item['notes']} | return_code={rc}")
                cancel_pending_rows(rows, reason=f"前序步骤 {step} 执行失败，后续步骤不执行")
                write_rows(queue_summary, QUEUE_FIELDS, rows)
                decision_txt.write_text(f"[stopped_at] {now_iso()}\n步骤 {step} 执行失败，自动队列停止。\n", encoding="utf-8")
                return
        else:
            log_path.write_text(f"[reused_at] {now_iso()}\n复用已有实验目录 {run_root}\n", encoding="utf-8")

        final_status = wait_for_run_completion(run_root / "summary.csv", args.poll_sec)
        update_row(rows, step, status=final_status, finished_at=now_iso(), log_path=str(log_path), notes=str(item["notes"]))
        write_rows(queue_summary, QUEUE_FIELDS, rows)
        if final_status != "success":
            cancel_pending_rows(rows, reason=f"前序步骤 {step} 结束状态为 {final_status}，后续步骤不执行")
            write_rows(queue_summary, QUEUE_FIELDS, rows)
            decision_txt.write_text(f"[stopped_at] {now_iso()}\n步骤 {step} 结束状态为 {final_status}，自动队列停止。\n", encoding="utf-8")
            return

        if item["seq_names"] is not None:
            add_candidate(run_root.name, run_root, load_metric_row(run_root / "metrics_compare.csv", "competition"))
            compare_candidates_csv(decision_csv, candidate_rows)
        else:
            full_metric = load_metric_row(run_root / "metrics_compare.csv", "competition")
            full_result_lines.extend(
                [
                    f"全量结果 {run_root.name}: HOTA={full_metric.HOTA:.3f} AssA={full_metric.AssA:.3f} IDF1={full_metric.IDF1:.3f} MOTA={full_metric.MOTA:.3f} IDs={full_metric.IDs:.0f} Frag={full_metric.Frag:.0f}",
                ]
            )

    expanded_to_full = False
    if len(launch_plan) >= 2:
        best_candidate = choose_best_candidate(candidate_rows)
        best_run_root = Path(str(best_candidate["run_root"]))
        best_hota_gain = float(best_candidate["delta_HOTA_vs_ranker_smoke"])
        best_idf1_gain = float(best_candidate["delta_IDF1_vs_ranker_smoke"])
        best_beats_heuristic = (
            float(best_candidate["delta_HOTA_vs_heuristic_smoke"]) >= 0.0
            and float(best_candidate["delta_IDF1_vs_heuristic_smoke"]) >= 0.0
        )
        if best_hota_gain >= 0.0 and best_idf1_gain >= 0.0 and best_beats_heuristic:
            expanded_to_full = True
            step = "followup_3"
            full_name = "deep_ocsort_local_contention_ranker_dance_margin_best_full_20260407_1"
            if best_run_root.name == watch_run_root.name:
                min_margin_to_second = 0.05
                margin_bias = 0.5
            elif "bias02" in best_run_root.name:
                min_margin_to_second = 0.05
                margin_bias = 0.2
            else:
                min_margin_to_second = 0.02
                margin_bias = 0.5
            full_run_root = REPO_ROOT / "outputs" / full_name
            log_path = queue_log_dir / f"{step}.log"
            note = f"冒烟最优方案为 {best_run_root.name}，扩到全量分批"
            update_row(
                rows,
                step,
                name=full_run_root.name,
                status="running",
                out_dir=str(full_run_root),
                started_at=now_iso(),
                notes=note,
            )
            write_rows(queue_summary, QUEUE_FIELDS, rows)
            if not (full_run_root / "summary.csv").is_file() or read_summary_status(full_run_root / "summary.csv", "compare") not in {"success", "running"}:
                cmd = build_common_args(
                    python_bin=args.python_bin,
                    eval_script=eval_script,
                    out_root=full_run_root,
                    raw_reuse_root=raw_reuse_root,
                    ranker_checkpoint=ranker_checkpoint,
                    min_margin_to_second=min_margin_to_second,
                    margin_bias=margin_bias,
                    seq_names=None,
                    full_batch_frame_budget=int(args.full_batch_frame_budget),
                )
                rc = run_experiment(cmd, log_path)
                if rc != 0:
                    update_row(rows, step, status="failed", finished_at=now_iso(), log_path=str(log_path), notes=f"{note} | return_code={rc}")
                    cancel_pending_rows(rows, reason="全量扩展执行失败，队列结束")
                    write_rows(queue_summary, QUEUE_FIELDS, rows)
                    decision_txt.write_text(f"[stopped_at] {now_iso()}\n全量扩展失败，自动队列停止。\n", encoding="utf-8")
                    return
            else:
                log_path.write_text(f"[reused_at] {now_iso()}\n复用已有实验目录 {full_run_root}\n", encoding="utf-8")
            final_status = wait_for_run_completion(full_run_root / "summary.csv", args.poll_sec)
            update_row(rows, step, status=final_status, finished_at=now_iso(), log_path=str(log_path), notes=note)
            write_rows(queue_summary, QUEUE_FIELDS, rows)
            if final_status == "success":
                full_metric = load_metric_row(full_run_root / "metrics_compare.csv", "competition")
                full_result_lines.extend(
                    [
                        f"全量结果 {full_run_root.name}: HOTA={full_metric.HOTA:.3f} AssA={full_metric.AssA:.3f} IDF1={full_metric.IDF1:.3f} MOTA={full_metric.MOTA:.3f} IDs={full_metric.IDs:.0f} Frag={full_metric.Frag:.0f}",
                    ]
                )

    if not expanded_to_full:
        cancel_pending_rows(rows, reason="冒烟最优候选未达到扩全量条件，队列在冒烟阶段结束")
        write_rows(queue_summary, QUEUE_FIELDS, rows)

    best_candidate = choose_best_candidate(candidate_rows)
    decision_lines = [
        f"[finished_at] {now_iso()}",
        f"当前参考学习版冒烟: HOTA={ranker_smoke.HOTA:.3f} IDF1={ranker_smoke.IDF1:.3f}",
        f"当前参考启发式冒烟: HOTA={heuristic_smoke.HOTA:.3f} IDF1={heuristic_smoke.IDF1:.3f}",
        f"当前自动队列最优候选: {best_candidate['tag']} ({best_candidate['run_root']})",
        f"最优候选相对旧学习版冒烟: HOTA {float(best_candidate['delta_HOTA_vs_ranker_smoke']):+.3f}, IDF1 {float(best_candidate['delta_IDF1_vs_ranker_smoke']):+.3f}",
        f"最优候选相对启发式冒烟: HOTA {float(best_candidate['delta_HOTA_vs_heuristic_smoke']):+.3f}, IDF1 {float(best_candidate['delta_IDF1_vs_heuristic_smoke']):+.3f}",
    ]
    decision_lines.extend(full_result_lines)
    decision_txt.write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
