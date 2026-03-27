#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


_EPOCH_RE = re.compile(r"epoch_(\d+)")


def _parse_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        return {}
    if len(lines) < 2:
        return {}
    names = lines[0].strip().split()
    vals = lines[1].strip().split()
    out: Dict[str, float] = {}
    for n, v in zip(names, vals):
        try:
            out[n] = float(v)
        except ValueError:
            continue
    return out


def _epoch_from_path(p: Path) -> Optional[int]:
    # .../val/epoch_3/... -> 3
    for part in p.parts[::-1]:
        m = _EPOCH_RE.fullmatch(part)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select best ByteTrack training checkpoint by validation metric.")
    ap.add_argument("--exp-dir", required=True, help="Experiment output dir, e.g. outputs/<EXP_NAME>")
    ap.add_argument("--dataset", default="MOT17", help="Dataset name for summary path (default: MOT17)")
    ap.add_argument("--split", default="train", help="Split name for summary path (default: train)")
    ap.add_argument("--metric", default="HOTA", help="Metric to maximize (default: HOTA)")
    args = ap.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    val_root = exp_dir / "val"
    if not val_root.exists():
        raise SystemExit(f"Missing val dir: {val_root}")

    best: Tuple[float, int, Path] | None = None

    # Accept both tracker/ and tracker_min* variants (submit_bytetrack.py may suffix).
    summary_globs = [
        f"epoch_*/tracker/{args.dataset}-{args.split}/pedestrian_summary.txt",
        f"epoch_*/tracker_min*/{args.dataset}-{args.split}/pedestrian_summary.txt",
    ]
    summaries = []
    for g in summary_globs:
        summaries.extend(val_root.glob(g))
    summaries = sorted(set(summaries))

    if not summaries:
        raise SystemExit(f"No pedestrian_summary.txt found under: {val_root}")

    for s in summaries:
        epoch = _epoch_from_path(s)
        if epoch is None:
            continue
        metrics = _parse_summary(s)
        if args.metric not in metrics:
            continue
        score = float(metrics[args.metric])
        if best is None or score > best[0]:
            best = (score, epoch, s)

    if best is None:
        raise SystemExit(f"Could not find metric '{args.metric}' in any summary under {val_root}")

    score, epoch, summary_path = best
    ckpt = exp_dir / f"checkpoint_epoch_{epoch}.pth"
    if not ckpt.exists():
        # fallback: older naming
        ckpt = exp_dir / f"checkpoint_{epoch}.pth"

    print(f"best_metric={args.metric} best_value={score:.4f} best_epoch={epoch}")
    print(f"summary={summary_path}")
    print(f"checkpoint={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

