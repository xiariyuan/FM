import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Check dataset folders for frame count consistency and image corruption.")
    parser.add_argument(
        "--root",
        type=str,
        default="datasets",
        help="Root directory that contains dataset folders/sequences.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Image extensions to consider.",
    )
    return parser.parse_args()


def find_sequences(root: Path) -> List[Path]:
    """
    Find sequence folders that contain an img1 directory (MOT-style layout).
    """
    sequences = []
    for img_dir in root.rglob("img1"):
        if img_dir.is_dir():
            sequences.append(img_dir.parent)
    return sequences


def get_images(img_dir: Path, extensions: List[str]) -> List[Path]:
    images = []
    ext_set = {e.lower() for e in extensions}
    for p in sorted(img_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in ext_set:
            images.append(p)
    return images


def highest_annotation_frame(gt_path: Path) -> Tuple[int | None, int]:
    """
    Returns highest frame index and number of annotation lines.
    """
    max_frame = None
    lines = 0
    with gt_path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            lines += 1
            try:
                frame_id = int(float(row[0]))
            except Exception:
                continue
            if max_frame is None or frame_id > max_frame:
                max_frame = frame_id
    return max_frame, lines


def check_corrupted(images: List[Path]) -> List[Path]:
    corrupted = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupted.append(img_path)
    return corrupted


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    sequences = find_sequences(root)
    if not sequences:
        print(f"No sequences with img1 found under {root}")
        return

    total_corrupted = []
    mismatches = []
    missing_gt = []

    for seq in sorted(sequences):
        img_dir = seq / "img1"
        gt_path = seq / "gt" / "gt.txt"
        images = get_images(img_dir, args.extensions)
        img_count = len(images)

        if gt_path.exists():
            max_frame, ann_lines = highest_annotation_frame(gt_path)
        else:
            max_frame, ann_lines = (None, 0)
            missing_gt.append(seq)

        if max_frame is not None and max_frame != img_count:
            mismatches.append((seq, img_count, max_frame, ann_lines))

        corrupted = check_corrupted(images)
        if corrupted:
            total_corrupted.extend(corrupted)

    print("=== Dataset Integrity Report ===")
    print(f"Checked sequences: {len(sequences)}")
    print(f"Sequences missing gt.txt: {len(missing_gt)}")
    if missing_gt:
        for seq in missing_gt:
            print(f"  - {seq}")

    print(f"Frame count mismatches: {len(mismatches)}")
    if mismatches:
        for seq, img_count, max_frame, ann_lines in mismatches:
            print(
                f"  - {seq}: images={img_count}, max_ann_frame={max_frame}, ann_lines={ann_lines}"
            )

    print(f"Corrupted images: {len(total_corrupted)}")
    if total_corrupted:
        for p in total_corrupted:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
