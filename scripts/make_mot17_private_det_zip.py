#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


TEST_SEQS = [1, 3, 6, 7, 8, 12, 14]
TRAIN_SEQS = [2, 4, 5, 9, 10, 11, 13]
DETECTORS = ["DPM", "FRCNN", "SDP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack MOT17 submission zip by copying FRCNN results to DPM/SDP (private-det style)."
    )
    parser.add_argument("--frcnn-dir", required=True, help="Directory containing MOT17-XX-FRCNN.txt files")
    parser.add_argument("--out-zip", required=True, help="Output submission zip path")
    parser.add_argument(
        "--profile",
        default="test",
        choices=["test", "train", "full"],
        help="Which sequences to include: test(21) / train(FRCNN-7) / full(42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing out-zip if it exists",
    )
    return parser.parse_args()


def expected_seq_ids(profile: str) -> list[int]:
    if profile == "test":
        return TEST_SEQS
    if profile == "train":
        return TRAIN_SEQS
    if profile == "full":
        return sorted(set(TEST_SEQS + TRAIN_SEQS))
    raise ValueError(f"Unsupported profile: {profile}")


def main() -> int:
    args = parse_args()
    frcnn_dir = Path(args.frcnn_dir)
    out_zip = Path(args.out_zip)

    if not frcnn_dir.is_dir():
        raise SystemExit(f"--frcnn-dir does not exist or is not a directory: {frcnn_dir}")

    if out_zip.exists():
        if args.overwrite:
            out_zip.unlink()
        else:
            raise SystemExit(f"--out-zip already exists (use --overwrite): {out_zip}")

    seq_ids = expected_seq_ids(args.profile)
    with tempfile.TemporaryDirectory(prefix="mot17_private_det_") as tmp:
        tmp_dir = Path(tmp)
        for seq_id in seq_ids:
            src_name = f"MOT17-{seq_id:02d}-FRCNN.txt"
            src_path = frcnn_dir / src_name
            if not src_path.is_file():
                raise SystemExit(f"Missing FRCNN result file: {src_path}")
            for det in DETECTORS:
                dst_name = f"MOT17-{seq_id:02d}-{det}.txt"
                shutil.copyfile(src_path, tmp_dir / dst_name)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(tmp_dir.glob("MOT17-*.txt")):
                zf.write(path, arcname=path.name)

    print(f"[OK] wrote {out_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

