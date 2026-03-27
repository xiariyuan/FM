#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a DanceTrack submission zip.\n\n"
            "DanceTrack official evaluation expects a zip that CONTAINS a top-level folder named `tracker/`.\n"
            "Do NOT zip the txt files directly.\n"
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-sequence MOT-format txt files (e.g. dancetrack0001.txt).",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="DanceTrack dataset root containing split folders (train/val/test). Used to infer expected sequences.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Split name to infer expected sequences from --data-root.",
    )
    parser.add_argument(
        "--out-zip",
        required=True,
        help="Output zip path.",
    )
    parser.add_argument(
        "--tracker-folder",
        default="tracker",
        help="Folder name inside zip (must be 'tracker' for DanceTrack Codalab).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --out-zip if it exists.",
    )
    return parser.parse_args()


def list_sequences(data_root: Path, split: str) -> list[str]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Missing split directory under --data-root: {split_dir}")
    seqs = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    if not seqs:
        raise SystemExit(f"No sequences found in: {split_dir}")
    return seqs


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    data_root = Path(args.data_root)
    out_zip = Path(args.out_zip)
    tracker_folder = str(args.tracker_folder)

    if tracker_folder != "tracker":
        raise SystemExit("--tracker-folder must be exactly 'tracker' for DanceTrack submission.")

    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir does not exist or is not a directory: {results_dir}")

    if out_zip.exists():
        if args.overwrite:
            out_zip.unlink()
        else:
            raise SystemExit(f"--out-zip already exists (use --overwrite): {out_zip}")

    expected_seqs = list_sequences(data_root=data_root, split=args.split)
    expected_files = [f"{seq}.txt" for seq in expected_seqs]

    missing = [name for name in expected_files if not (results_dir / name).is_file()]
    if missing:
        sample = "\n".join(f"  - {name}" for name in missing[:10])
        raise SystemExit(
            f"Missing {len(missing)} expected result files under {results_dir}.\nSample:\n{sample}"
        )

    # Note: results_dir might contain extra txt files (e.g., from a previous run). We ignore them and
    # package only the expected split files for determinism.
    with tempfile.TemporaryDirectory(prefix="dancetrack_submit_") as tmp:
        tmp_dir = Path(tmp)
        tracker_dir = tmp_dir / tracker_folder
        tracker_dir.mkdir(parents=True, exist_ok=True)

        for name in expected_files:
            shutil.copyfile(results_dir / name, tracker_dir / name)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(tracker_dir.glob("*.txt")):
                zf.write(path, arcname=f"{tracker_folder}/{path.name}")

    print(f"[OK] wrote {out_zip}")
    print(f"[OK] packaged {len(expected_files)} sequences from split={args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

