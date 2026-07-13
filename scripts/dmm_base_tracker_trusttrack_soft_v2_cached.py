#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import dmm_base_tracker_trusttrack_soft_v2 as soft

_ORIGINAL_LOAD_DUMP = soft.spda.base.load_dump

def _load_cached_or_original(path: Path):
    cache_value = os.environ.get('TRUST_DUMP_CACHE_DIR', '').strip()
    if not cache_value:
        return _ORIGINAL_LOAD_DUMP(path)
    cache_dir = Path(cache_value)
    if not (cache_dir / 'manifest.json').exists():
        raise FileNotFoundError(f'missing cache manifest: {cache_dir / "manifest.json"}')
    keys = ['detections','features','frame_offsets','columns','image_files','image_wh']
    out = {}
    for key in keys:
        p = cache_dir / f'{key}.npy'
        if not p.exists():
            raise FileNotFoundError(f'missing cached array: {p}')
        out[key] = np.load(p, mmap_mode='r', allow_pickle=True)
    return out

soft.spda.base.load_dump = _load_cached_or_original

if __name__ == '__main__':
    soft.main()
