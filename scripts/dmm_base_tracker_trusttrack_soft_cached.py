#!/usr/bin/env python3
"""Memory-mapped launcher for TrustTrack-soft.

Set TRUST_DUMP_CACHE_DIR to a directory created by cache_detection_dump.py.
The launcher replaces the NPZ loader with mmap-backed .npy arrays, then runs the
unchanged TrustTrack-soft implementation. Matching, tracking, and metrics are
not altered; this only removes repeated decompression of very large NPZ files.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import dmm_base_tracker_trusttrack_soft as soft


_ORIGINAL_LOAD_DUMP = soft.spda.base.load_dump


def _load_cached_or_original(path: Path):
    cache_value = os.environ.get("TRUST_DUMP_CACHE_DIR", "").strip()
    if not cache_value:
        return _ORIGINAL_LOAD_DUMP(path)

    cache_dir = Path(cache_value)
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"TRUST_DUMP_CACHE_DIR is set, but cache manifest is missing: {manifest}"
        )

    required = [
        "detections",
        "features",
        "frame_offsets",
        "columns",
        "image_files",
        "image_wh",
    ]
    result = {}
    for key in required:
        array_path = cache_dir / f"{key}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"Cached dump array is missing: {array_path}")
        result[key] = np.load(array_path, mmap_mode="r", allow_pickle=True)
    return result


soft.spda.base.load_dump = _load_cached_or_original


if __name__ == "__main__":
    soft.main()
