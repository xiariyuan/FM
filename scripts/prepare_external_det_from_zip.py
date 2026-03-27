#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


SEQ_PATTERNS = [
    r"MOT17-\d{2}-(?:FRCNN|DPM|SDP)",
    r"MOT17-\d{2}",
    r"MOT20-\d{2}",
]

_MOT17_TRAIN_IDS = {"02", "04", "05", "09", "10", "11", "13"}
_MOT17_TEST_IDS = {"01", "03", "06", "07", "08", "12", "14"}
_MOT20_TRAIN_IDS = {"01", "02", "03", "05"}
_MOT20_TEST_IDS = {"04", "06", "07", "08"}


def find_seq_name(path: Path) -> str | None:
    text = str(path)
    for pat in SEQ_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()

    stem = path.stem.upper()
    for pat in SEQ_PATTERNS:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return m.group(0).upper()
    return None


def sequence_matches_split(seq_name: str, dataset: str, split: str) -> bool:
    """
    Keep zip extraction split-consistent.
    Example:
    - MOT20 train should only include 01/02/03/05
    - MOT20 test should only include 04/06/07/08
    """
    split = split.lower()
    if split not in {"train", "test"}:
        return True

    m = re.search(r"(MOT17|MOT20)-(\d{2})", seq_name, flags=re.IGNORECASE)
    if m is None:
        return False
    ds = m.group(1).upper()
    seq_id = m.group(2)
    if ds != dataset.upper():
        return False

    if ds == "MOT17":
        return seq_id in (_MOT17_TRAIN_IDS if split == "train" else _MOT17_TEST_IDS)
    if ds == "MOT20":
        return seq_id in (_MOT20_TRAIN_IDS if split == "train" else _MOT20_TEST_IDS)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to downloaded detector zip")
    ap.add_argument("--detector", required=True, choices=["sw_yolox", "sgt"])
    ap.add_argument("--dataset", required=True, choices=["MOT17", "MOT20"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", default="outputs/external_det")
    ap.add_argument("--mot17-detector-suffix", default="FRCNN", choices=["FRCNN", "DPM", "SDP"])
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_file():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    out_dir = Path(args.out_root) / args.detector / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="fmtrack_det_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        txt_files = list(tmp_path.rglob("*.txt"))
        if len(txt_files) == 0:
            raise RuntimeError(f"No txt files found in zip: {zip_path}")

        for txt in txt_files:
            seq_name = find_seq_name(txt)
            if seq_name is None:
                skipped += 1
                continue
            if not seq_name.startswith(args.dataset):
                skipped += 1
                continue
            if args.dataset == "MOT17" and seq_name.count("-") == 1:
                seq_name = f"{seq_name}-{args.mot17_detector_suffix}"
            if not sequence_matches_split(seq_name=seq_name, dataset=args.dataset, split=args.split):
                skipped += 1
                continue
            dst = out_dir / f"{seq_name}.txt"
            shutil.copyfile(txt, dst)
            copied += 1

    print(f"Prepared detector={args.detector}, dataset={args.dataset}, split={args.split}")
    print(f"Output dir: {out_dir}")
    print(f"Copied files: {copied}")
    print(f"Skipped files: {skipped}")
    if copied == 0:
        raise RuntimeError("No matching sequence txt copied. Please inspect zip structure.")


if __name__ == "__main__":
    main()
