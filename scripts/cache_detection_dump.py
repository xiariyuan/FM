#!/usr/bin/env python3
"""Extract a large detection NPZ into per-array .npy files for mmap loading.

The original MOT20 dumps can exceed 1 GB. Reopening a compressed NPZ for every
experiment repeatedly pays decompression and memory costs. This script performs
that cost once and writes a manifest plus individual .npy arrays. Later tracker
runs can memory-map the arrays and only cast the current frame's feature slice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.input_npz)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"

    if manifest_path.exists() and not args.overwrite:
        print(f"cache already exists: {manifest_path}")
        return

    manifest = {"source": str(src.resolve()), "arrays": {}}
    with np.load(src, allow_pickle=True) as dump:
        for key in dump.files:
            target = out / f"{key}.npy"
            arr = dump[key]
            np.save(target, arr, allow_pickle=bool(arr.dtype == object))
            manifest["arrays"][key] = {
                "file": target.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "bytes": int(arr.nbytes),
            }
            print(f"saved {key}: shape={arr.shape} dtype={arr.dtype} -> {target}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
