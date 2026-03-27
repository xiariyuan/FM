#!/usr/bin/env python3
import argparse
import configparser
import math
import os
import sys
from pathlib import Path


def load_seq_dims(data_root: Path, benchmark: str, split: str, seq_name: str) -> tuple[int, int]:
    seqinfo_path = data_root / benchmark / split / seq_name / "seqinfo.ini"
    cfg = configparser.ConfigParser()
    if not cfg.read(seqinfo_path):
        raise FileNotFoundError(f"Could not read seqinfo: {seqinfo_path}")
    return int(cfg["Sequence"]["imWidth"]), int(cfg["Sequence"]["imHeight"])


def sanitize_file(
    src_path: Path,
    dst_path: Path,
    width: int,
    height: int,
    precision: int,
    drop_raw_negxy: bool,
) -> tuple[int, int]:
    kept = 0
    dropped = 0
    rows = []
    with src_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 10:
                dropped += 1
                continue
            try:
                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
            except ValueError:
                dropped += 1
                continue

            if not (
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(w)
                and math.isfinite(h)
                and math.isfinite(conf)
            ):
                dropped += 1
                continue

            # Some MOTChallenge evaluation servers are stricter than TrackEval and
            # may reject negative confidence values (e.g., -1 after DTI interpolation).
            # For tracking evaluation, the confidence is typically unused, so we clamp
            # it into [0, 1] to be safe.
            if conf < 0.0 or conf > 1.0:
                conf = 1.0

            if drop_raw_negxy and (x < 0.0 or y < 0.0):
                dropped += 1
                continue

            x1 = max(1.0, min(x, float(width)))
            y1 = max(1.0, min(y, float(height)))
            x2 = max(x1, min(x + w, float(width)))
            y2 = max(y1, min(y + h, float(height)))
            new_w = x2 - x1
            new_h = y2 - y1

            if frame <= 0 or track_id <= 0 or new_w <= 0.0 or new_h <= 0.0:
                dropped += 1
                continue

            rows.append((frame, track_id, x1, y1, new_w, new_h, conf))
            kept += 1

    rows.sort(key=lambda item: (item[0], item[1]))
    with dst_path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame, track_id, x, y, w, h, conf in rows:
            handle.write(
                f"{frame:d},{track_id:d},{x:.{precision}f},{y:.{precision}f},{w:.{precision}f},{h:.{precision}f},{conf:.{precision}f},-1,-1,-1\n"
            )
    return kept, dropped


def infer_split(seq_name: str) -> str:
    if seq_name in {"MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"}:
        return "train"
    if seq_name in {"MOT20-04", "MOT20-06", "MOT20-07", "MOT20-08"}:
        return "test"
    if seq_name.startswith("MOT17-"):
        core = seq_name.split(".txt")[0]
        track = core.rsplit("-", 1)[0]
        if track in {"MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"}:
            return "train"
        return "test"
    raise ValueError(f"Cannot infer split for {seq_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--benchmark", required=True, choices=["MOT17", "MOT20"])
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--drop-raw-negxy", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    data_root = Path(args.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for src_path in sorted(input_dir.glob("*.txt")):
        seq_name = src_path.stem
        split = infer_split(seq_name)
        width, height = load_seq_dims(data_root, args.benchmark, split, seq_name)
        dst_path = output_dir / src_path.name
        kept, dropped = sanitize_file(
            src_path,
            dst_path,
            width,
            height,
            args.precision,
            args.drop_raw_negxy,
        )
        summary.append((src_path.name, kept, dropped, width, height))

    for name, kept, dropped, width, height in summary:
        print(f"{name}: kept={kept} dropped={dropped} size={width}x{height}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
