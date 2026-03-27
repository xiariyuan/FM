#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


_EPOCH_RE = re.compile(r"---> start train epoch(\d+)\b")
_AP_5095_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)"
)
_AP_50_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)"
)
_AP_75_RE = re.compile(
    r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)"
)
_AR_100_RE = re.compile(
    r"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9.]+)"
)


@dataclass
class EvalMetrics:
    epoch: int
    ap50_95: Optional[float] = None
    ap50: Optional[float] = None
    ap75: Optional[float] = None
    ar100: Optional[float] = None


@dataclass
class EarlyStopState:
    run_name: str
    run_dir: str
    metric: str
    min_delta: float
    min_evals: int
    patience_evals: int
    best_value: float
    best_epoch: int
    evals_seen: int
    evals_since_improve: int
    last_update_unix: float
    stop_requested: bool
    stop_reason: str


def _run_cmd(args: list[str]) -> str:
    out = subprocess.check_output(args, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def _tmux_pane_pid(tmux_session: str) -> Optional[int]:
    try:
        pid_s = _run_cmd(["tmux", "list-panes", "-t", tmux_session, "-F", "#{pane_pid}"])
    except Exception:
        return None
    pid_s = pid_s.strip().splitlines()[0].strip() if pid_s.strip() else ""
    if not pid_s.isdigit():
        return None
    return int(pid_s)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _find_metrics(train_log: Path) -> list[EvalMetrics]:
    metrics: list[EvalMetrics] = []
    current_epoch = 0
    last_block: Optional[EvalMetrics] = None

    if not train_log.exists():
        return metrics

    with train_log.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            m_epoch = _EPOCH_RE.search(line)
            if m_epoch is not None:
                try:
                    current_epoch = int(m_epoch.group(1))
                except ValueError:
                    pass
                continue

            m_ap5095 = _AP_5095_RE.search(line)
            if m_ap5095 is not None:
                last_block = EvalMetrics(epoch=current_epoch, ap50_95=float(m_ap5095.group(1)))
                metrics.append(last_block)
                continue

            if last_block is None:
                continue

            m_ap50 = _AP_50_RE.search(line)
            if m_ap50 is not None:
                last_block.ap50 = float(m_ap50.group(1))
                continue

            m_ap75 = _AP_75_RE.search(line)
            if m_ap75 is not None:
                last_block.ap75 = float(m_ap75.group(1))
                continue

            m_ar100 = _AR_100_RE.search(line)
            if m_ar100 is not None:
                last_block.ar100 = float(m_ar100.group(1))
                continue

    return metrics


def _write_metrics_csv(path: Path, metrics: Iterable[EvalMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "ap50_95", "ap50", "ap75", "ar100"],
        )
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))


def _stop_process(pid: int, grace_sec: int) -> tuple[bool, str]:
    if not _pid_alive(pid):
        return True, "process already exited"

    try:
        os.kill(pid, signal.SIGINT)
    except Exception as e:
        return False, f"failed to SIGINT: {type(e).__name__}: {e}"

    t0 = time.time()
    while time.time() - t0 < grace_sec:
        if not _pid_alive(pid):
            return True, "stopped by SIGINT"
        time.sleep(1)

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        return False, f"failed to SIGTERM: {type(e).__name__}: {e}"

    time.sleep(5)
    if not _pid_alive(pid):
        return True, "stopped by SIGTERM"

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception as e:
        return False, f"failed to SIGKILL: {type(e).__name__}: {e}"
    return (not _pid_alive(pid)), "stopped by SIGKILL"


def _get_metric_value(m: EvalMetrics, metric: str) -> Optional[float]:
    if metric == "ap50_95":
        return m.ap50_95
    if metric == "ap50":
        return m.ap50
    if metric == "ap75":
        return m.ap75
    if metric == "ar100":
        return m.ar100
    raise ValueError(f"unknown metric: {metric}")


def _load_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Watch a YOLOX (ByteTrack) detector training log and early-stop if metrics plateau.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Detector run name under outputs/detector/.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Full path to the detector run dir (contains train_log.txt). Overrides --run-name.",
    )
    parser.add_argument(
        "--tmux-session",
        type=str,
        default=None,
        help="tmux session that runs the detector training; used to find the process pid.",
    )
    parser.add_argument("--pid", type=int, default=None, help="Training process pid to stop.")
    parser.add_argument("--metric", type=str, default="ap50_95", choices=["ap50_95", "ap50", "ap75", "ar100"])
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum improvement to reset patience.")
    parser.add_argument("--min-evals", type=int, default=4, help="Do not early-stop before this many evals.")
    parser.add_argument("--patience-evals", type=int, default=6, help="Stop after this many non-improving evals.")
    parser.add_argument("--check-interval-sec", type=int, default=120)
    parser.add_argument("--grace-sec", type=int, default=60, help="Grace period after SIGINT before SIGTERM.")
    parser.add_argument("--dry-run", action="store_true", help="Never stop training; only write status files.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    detector_root = repo_root / "outputs" / "detector"

    run_name = args.run_name
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if run_name is None:
            run_name = run_dir.name
    else:
        if run_name is None:
            run_name = _load_text(detector_root / "latest_session.txt")
        if not run_name:
            print("[watch] ERROR: missing run name (pass --run-name or create outputs/detector/latest_session.txt)")
            return 2
        run_dir = detector_root / run_name

    if args.tmux_session is None:
        args.tmux_session = _load_text(detector_root / "latest_tmux_session.txt")

    status_json = run_dir / "early_stop_status.json"
    metrics_csv = run_dir / "val_metrics.csv"
    train_log = run_dir / "train_log.txt"

    best_value = float("-inf")
    best_epoch = -1
    best_eval_idx = 0  # 1-based index into the eval list
    evals_seen = 0
    evals_since_improve = 0
    stop_requested = False
    stop_reason = ""

    print(f"[watch] run={run_name} log={train_log} metric={args.metric}")
    while True:
        pid = args.pid
        if pid is None and args.tmux_session:
            pid = _tmux_pane_pid(args.tmux_session)

        metrics = _find_metrics(train_log)
        _write_metrics_csv(metrics_csv, metrics)

        prev_evals_seen = evals_seen
        evals_seen = len(metrics)

        if evals_seen > prev_evals_seen:
            for i in range(prev_evals_seen, evals_seen):
                m = metrics[i]
                value = _get_metric_value(m, args.metric)
                if value is None:
                    continue
                eval_idx = i + 1  # 1-based
                if value > best_value + args.min_delta:
                    best_value = value
                    best_epoch = m.epoch
                    best_eval_idx = eval_idx

        if best_eval_idx > 0:
            evals_since_improve = evals_seen - best_eval_idx
            if (not stop_requested) and evals_seen >= args.min_evals and evals_since_improve >= args.patience_evals:
                stop_requested = True
                stop_reason = (
                    f"no improvement for {evals_since_improve} evals "
                    f"(best {args.metric}={best_value:.4f} at epoch {best_epoch})"
                )

        state = EarlyStopState(
            run_name=run_name,
            run_dir=str(run_dir),
            metric=args.metric,
            min_delta=args.min_delta,
            min_evals=args.min_evals,
            patience_evals=args.patience_evals,
            best_value=best_value if best_value != float("-inf") else -1.0,
            best_epoch=best_epoch,
            evals_seen=evals_seen,
            evals_since_improve=evals_since_improve,
            last_update_unix=time.time(),
            stop_requested=stop_requested,
            stop_reason=stop_reason,
        )
        status_json.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if pid is not None and not _pid_alive(pid):
            print(f"[watch] training pid {pid} not alive; exiting.")
            return 0

        if stop_requested:
            msg = f"[watch] early-stop triggered: {stop_reason}"
            print(msg)
            if args.dry_run:
                return 0
            if pid is None:
                print("[watch] ERROR: cannot stop (no pid and no tmux session).")
                return 3
            ok, detail = _stop_process(pid, grace_sec=args.grace_sec)
            print(f"[watch] stop result ok={ok} detail={detail}")
            return 0 if ok else 4

        time.sleep(max(10, args.check_interval_sec))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
