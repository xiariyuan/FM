#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


_EPOCH_START_RE = re.compile(r"---> start train epoch(\d+)\b")


def _run_cmd(args: list[str]) -> str:
    out = subprocess.check_output(args, stderr=subprocess.STDOUT)
    return out.decode("utf-8", errors="replace").strip()


def _tmux_has_session(name: str) -> bool:
    try:
        subprocess.check_output(["tmux", "has-session", "-t", name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _tmux_pane_pid(session: str) -> Optional[int]:
    try:
        pid_s = _run_cmd(["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"])
    except Exception:
        return None
    pid_s = pid_s.strip().splitlines()[0].strip() if pid_s.strip() else ""
    return int(pid_s) if pid_s.isdigit() else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_pid(pid: int, grace_sec: int = 120) -> None:
    if not _pid_alive(pid):
        return
    os.kill(pid, signal.SIGINT)
    t0 = time.time()
    while time.time() - t0 < grace_sec:
        if not _pid_alive(pid):
            return
        time.sleep(1)
    os.kill(pid, signal.SIGTERM)


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _tail_lines(path: Path, n: int = 3000) -> list[str]:
    try:
        out = subprocess.check_output(["tail", "-n", str(n), str(path)])
        return out.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []


def _last_epoch_started(train_log: Path) -> Optional[int]:
    last: Optional[int] = None
    for line in _tail_lines(train_log, n=5000):
        m = _EPOCH_START_RE.search(line)
        if m is None:
            continue
        try:
            last = int(m.group(1))
        except ValueError:
            continue
    return last


def _wait_for_epoch_start(train_log: Path, epoch: int, train_pid: Optional[int], poll_sec: int = 2) -> bool:
    needle = f"---> start train epoch{epoch}"
    # If already present, we can proceed immediately.
    if any(needle in line for line in _tail_lines(train_log, n=8000)):
        return True

    # Follow the log from end.
    while not train_log.exists():
        if train_pid is not None and not _pid_alive(train_pid):
            return False
        time.sleep(poll_sec)

    with train_log.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            if train_pid is not None and not _pid_alive(train_pid):
                return False
            line = f.readline()
            if not line:
                time.sleep(poll_sec)
                continue
            if needle in line:
                return True


def _start_tmux_session(name: str, command: str) -> None:
    subprocess.check_call(["tmux", "new-session", "-d", "-s", name, command])


def _kill_tmux_session(name: str) -> None:
    if not _tmux_has_session(name):
        return
    subprocess.check_call(["tmux", "kill-session", "-t", name])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Wait for the next YOLOX eval to finish, then restart training with eval_interval=1 (per-epoch eval).",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-tmux-session", type=str, default=None)
    parser.add_argument("--watch-tmux-session", type=str, default=None)
    parser.add_argument("--exp-file", type=str, default="third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py")
    parser.add_argument("--output-dir", type=str, default="outputs/detector")
    parser.add_argument("--new-run-prefix", type=str, default="det_yolox_mixdet_restart")
    parser.add_argument(
        "--resume-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, pass --resume to YOLOX (resume optimizer/scheduler). If false, only load weights via -c and restart from epoch 0.",
    )
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--no-aug-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-ann", type=str, default="train_mot17_trainhalf.json")
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--metric", type=str, default="ap50_95")
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--min-evals", type=int, default=4)
    parser.add_argument("--patience-evals", type=int, default=6)
    parser.add_argument("--check-interval-sec", type=int, default=300)
    parser.add_argument("--grace-sec", type=int, default=120)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    detector_root = repo_root / "outputs" / "detector"

    run_name = args.run_name or _read_text(detector_root / "latest_session.txt")
    if not run_name:
        print("[restart] ERROR: missing run name.")
        return 2
    run_dir = detector_root / run_name
    train_log = run_dir / "train_log.txt"

    train_tmux = args.train_tmux_session or _read_text(detector_root / "latest_tmux_session.txt")
    watch_tmux = args.watch_tmux_session or _read_text(detector_root / "latest_watch_tmux_session.txt")
    if not train_tmux:
        print("[restart] ERROR: missing training tmux session name.")
        return 2

    train_pid = _tmux_pane_pid(train_tmux)
    cur_epoch = _last_epoch_started(train_log)
    if cur_epoch is None:
        print(f"[restart] ERROR: cannot find current epoch in log: {train_log}")
        return 3
    target_epoch = cur_epoch + 1

    print(f"[restart] run={run_name} tmux={train_tmux} pid={train_pid} current_epoch={cur_epoch}")
    print(f"[restart] waiting until epoch {target_epoch} starts (means previous epoch eval finished)...")

    ok = _wait_for_epoch_start(train_log, target_epoch, train_pid=train_pid, poll_sec=2)
    if not ok:
        print("[restart] training process exited before reaching the next epoch; aborting restart.")
        return 4

    print("[restart] detected next epoch start; stopping current training...")
    if train_pid is None:
        train_pid = _tmux_pane_pid(train_tmux)
    if train_pid is not None:
        _stop_pid(train_pid, grace_sec=args.grace_sec)

    # Wait for pid to exit.
    if train_pid is not None:
        for _ in range(180):
            if not _pid_alive(train_pid):
                break
            time.sleep(1)

    # Clean up old tmux sessions (best-effort).
    _kill_tmux_session(train_tmux)
    if watch_tmux:
        _kill_tmux_session(watch_tmux)

    ckpt_latest = run_dir / "latest_ckpt.pth.tar"
    ckpt_fallback = run_dir / "last_epoch_ckpt.pth.tar"
    ckpt_path = ckpt_latest if ckpt_latest.exists() else ckpt_fallback
    if not ckpt_path.exists():
        print(f"[restart] ERROR: cannot find resume checkpoint: {ckpt_latest} or {ckpt_fallback}")
        return 5

    ts = time.strftime("%Y%m%d_%H%M%S")
    new_run = f"{args.new_run_prefix}_{ts}"
    new_train_tmux = f"det50_{new_run}"
    new_watch_tmux = f"watch_{new_run}"

    print(f"[restart] starting new run {new_run} (eval every epoch) from {ckpt_path}")
    detector_root.mkdir(parents=True, exist_ok=True)
    (detector_root / "latest_session.txt").write_text(new_run + "\n", encoding="utf-8")
    (detector_root / "latest_tmux_session.txt").write_text(new_train_tmux + "\n", encoding="utf-8")

    resume_flags = f"--resume -c {ckpt_path}" if args.resume_training else f"-c {ckpt_path}"
    train_cmd = (
        f"cd {repo_root} && mkdir -p {args.output_dir} && "
        f"env PYTHONPATH=third_party/ByteTrack python third_party/ByteTrack/tools/train.py "
        f"-expn {new_run} -f {args.exp_file} -d 1 -b {args.batch_size} --fp16 "
        f"{resume_flags} "
        f"output_dir {args.output_dir} "
        f"max_epoch {args.max_epoch} "
        f"no_aug_epochs {args.no_aug_epochs} "
        f"eval_interval 1 "
        f"print_interval {args.print_interval} "
        f"warmup_epochs {args.warmup_epochs} "
        f"train_ann {args.train_ann}"
    )
    _start_tmux_session(new_train_tmux, train_cmd)

    (detector_root / "latest_watch_tmux_session.txt").write_text(new_watch_tmux + "\n", encoding="utf-8")
    watch_cmd = (
        f"cd {repo_root} && python scripts/watch_yolox_detector_early_stop.py "
        f"--run-name {new_run} "
        f"--tmux-session {new_train_tmux} "
        f"--metric {args.metric} "
        f"--min-delta {args.min_delta} "
        f"--min-evals {args.min_evals} "
        f"--patience-evals {args.patience_evals} "
        f"--check-interval-sec {args.check_interval_sec} "
        f"--grace-sec {args.grace_sec}"
    )
    _start_tmux_session(new_watch_tmux, watch_cmd)

    print(f"[restart] DONE. train_tmux={new_train_tmux} watch_tmux={new_watch_tmux} run_dir={detector_root / new_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
