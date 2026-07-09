#!/usr/bin/env python3
"""Pre-compute GMC (Global Motion Compensation) warp matrices for MOT20 sequences.

Uses BoT-SORT's GMC module (sparseOptFlow by default) to compute frame-to-frame
camera motion warps. Saves one 2x3 warp matrix per frame as a .npy file.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BOT = REPO / "external" / "BoT-SORT-main"
sys.path.insert(0, str(BOT))
from tracker.gmc import GMC

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_frames(img_dir: Path) -> list[tuple[int, Path]]:
    frames = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTS:
            try:
                num = int(f.stem.split("img")[-1])
            except (ValueError, IndexError):
                num = int("".join(c for c in f.stem if c.isdigit()) or "0")
            frames.append((num, f))
    frames.sort()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True, help="MOT20 sequence directory with img1/ subfolder")
    parser.add_argument("--out", required=True, help="Output .npy file for warp matrices")
    parser.add_argument("--method", default="sparseOptFlow", choices=["sparseOptFlow", "orb", "sift", "ecc"])
    parser.add_argument("--downscale", type=int, default=2)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    frames = load_frames(img_dir)
    n_frames = len(frames)
    print(f"[gmc] {n_frames} frames in {img_dir}", flush=True)

    gmc = GMC(method=args.method, downscale=args.downscale)
    warps = np.zeros((n_frames, 2, 3), dtype=np.float64)

    import cv2
    t0 = time.time()
    for idx, (fnum, fpath) in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            warps[idx] = np.eye(2, 3)
            continue
        warp = gmc.apply(img)
        warps[idx] = warp.astype(np.float64)
        if idx == 0 or idx == n_frames - 1 or idx % 200 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / max(elapsed, 0.001)
            print(f"[gmc] frame {idx+1}/{n_frames} fps={fps:.1f}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, warps)
    elapsed = time.time() - t0
    print(f"[gmc] saved {out_path} ({n_frames} warps, {elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
