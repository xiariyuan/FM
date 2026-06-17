"""ArchiveIndex — Hilbert-sharded storage for removed track identities.

Each removed track becomes an IdentityEntry that holds spatial, temporal,
appearance, and motion state. Entries are partitioned by Hilbert curve bucket
for coarse spatial routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hilbert curve encoding (2D → 1D)
# ---------------------------------------------------------------------------

def _hilbert_encode(x: int, y: int, order: int = 16) -> int:
    """Map 2-D integer coordinates to a Hilbert curve index.

    *order* controls the depth of the curve (2^order cells per axis).
    Returns a non-negative integer that preserves spatial locality.
    """
    x, y = int(x), int(y)
    d = 0
    s = 1 << (order - 1)
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = (1 << order) - 1 - x
                y = (1 << order) - 1 - y
            x, y = y, x
        s >>= 1
    return d


def _hilbert_bucket(x: int, y: int, order: int = 8) -> int:
    """Coarse Hilbert bucket (lower-order curve for sharding)."""
    return _hilbert_encode(x, y, order)


# ---------------------------------------------------------------------------
# IdentityEntry
# ---------------------------------------------------------------------------

@dataclass
class IdentityEntry:
    """A single archived identity ready for re-entry queries."""

    track_id: int
    # Spatial: last known position (centre of bbox)
    exit_x: float
    exit_y: float
    hilbert_key: int  # precomputed from (exit_x, exit_y)
    # Temporal
    exit_frame: int  # frame at which track was removed
    # Appearance
    smooth_feat: Optional[np.ndarray] = None  # (D,)
    feat_history: Optional[List[np.ndarray]] = None  # recent raw features
    prototype_set: Optional[List[np.ndarray]] = None  # top-k stable prototypes for multi-proto matching
    # Motion
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) from Kalman
    mean_state: Optional[np.ndarray] = None  # full Kalman mean
    covariance: Optional[np.ndarray] = None  # full Kalman covariance
    # Score
    exit_score: float = 0.0  # detection score at last update
    # Metadata
    tracklet_len: int = 0

    def gap(self, current_frame: int) -> int:
        return max(1, current_frame - self.exit_frame)


# ---------------------------------------------------------------------------
# ArchiveIndex
# ---------------------------------------------------------------------------

class ArchiveIndex:
    """Hilbert-sharded archive of removed identities.

    Coarse layer: entries are bucketed by Hilbert key so that spatially close
    exits live in the same shard.

    Fine layer: within each shard entries are stored in a list sorted by
    exit_frame (most recent first) for efficient gap-based expiration.
    """

    def __init__(
        self,
        hilbert_order: int = 8,
        max_gap: int = 120,
        max_shard_size: int = 2048,
    ):
        self.hilbert_order = hilbert_order
        self.max_gap = max_gap
        self.max_shard_size = max_shard_size

        # shard_key -> list[IdentityEntry], sorted newest-first
        self._shards: Dict[int, List[IdentityEntry]] = {}
        # fast lookup
        self._id_to_entry: Dict[int, IdentityEntry] = {}
        # neighbour offsets for shard traversal
        self._shard_count = 0

    # -- public API ---------------------------------------------------------

    def insert(self, entry: IdentityEntry) -> None:
        """Add an identity entry to the archive."""
        if entry.track_id in self._id_to_entry:
            self.delete(entry.track_id)
        shard_key = _hilbert_bucket(int(entry.exit_x), int(entry.exit_y), self.hilbert_order)
        entry.hilbert_key = shard_key
        if shard_key not in self._shards:
            self._shards[shard_key] = []
        # insert maintaining newest-first order
        bucket = self._shards[shard_key]
        bucket.append(entry)
        # keep sorted by exit_frame descending
        bucket.sort(key=lambda e: e.exit_frame, reverse=True)
        # enforce per-shard size cap (drop oldest)
        if len(bucket) > self.max_shard_size:
            dropped = bucket[self.max_shard_size:]
            bucket[:] = bucket[: self.max_shard_size]
            for d in dropped:
                self._id_to_entry.pop(d.track_id, None)
        self._id_to_entry[entry.track_id] = entry
        self._shard_count = len(self._shards)

    def delete(self, track_id: int) -> bool:
        """Remove an identity from the archive. Returns True if found."""
        entry = self._id_to_entry.pop(track_id, None)
        if entry is None:
            return False
        bucket = self._shards.get(entry.hilbert_key)
        if bucket is not None:
            self._shards[entry.hilbert_key] = [e for e in bucket if e.track_id != track_id]
            if not self._shards[entry.hilbert_key]:
                del self._shards[entry.hilbert_key]
        self._shard_count = len(self._shards)
        return True

    def expire(self, current_frame: int) -> int:
        """Remove entries whose gap exceeds max_gap. Returns count removed."""
        expired_ids = []
        for entry in self._id_to_entry.values():
            if entry.gap(current_frame) > self.max_gap:
                expired_ids.append(entry.track_id)
        for tid in expired_ids:
            self.delete(tid)
        return len(expired_ids)

    def get(self, track_id: int) -> Optional[IdentityEntry]:
        return self._id_to_entry.get(track_id)

    @property
    def size(self) -> int:
        return len(self._id_to_entry)

    @property
    def shard_count(self) -> int:
        return self._shard_count

    def all_entries(self) -> List[IdentityEntry]:
        return list(self._id_to_entry.values())

    # -- spatial query ------------------------------------------------------

    def spatial_candidates(
        self,
        x: float,
        y: float,
        radius: int = 1,
    ) -> List[IdentityEntry]:
        """Return entries from Hilbert-neighbouring shards.

        Uses a two-phase approach:
        1. Probe the exact shard for (x, y).
        2. Probe neighbouring shards by jittering at cell granularity.
        3. Also check entries that Hilbert-keyed in nearby but non-adjacent
           shards by scanning a bounded pixel radius around (x, y) across
           all shards.

        Parameters
        ----------
        radius : int
            Controls both Hilbert neighbourhood hops and pixel search radius
            multiplier.  pixel_radius = radius * cell_size.
        """
        candidates: List[IdentityEntry] = []
        seen_ids: set = set()

        cell_size = max(1, (1 << 16) >> self.hilbert_order)

        # Phase 1: Hilbert neighbourhood via jitter
        visited_keys = set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = max(0, min((1 << 16) - 1, int(x) + dx * cell_size))
                ny = max(0, min((1 << 16) - 1, int(y) + dy * cell_size))
                key = _hilbert_bucket(nx, ny, self.hilbert_order)
                if key in visited_keys:
                    continue
                visited_keys.add(key)
                for entry in self._shards.get(key, []):
                    if entry.track_id not in seen_ids:
                        candidates.append(entry)
                        seen_ids.add(entry.track_id)

        # Phase 2: pixel-radius fallback — scan all entries whose exit
        # position is within a pixel radius.  This catches cases where
        # Hilbert locality doesn't align with Euclidean locality at the
        # current bucket granularity.
        pixel_radius = radius * cell_size * 0.5  # half cell as pixel radius
        for bucket_entries in self._shards.values():
            for entry in bucket_entries:
                if entry.track_id in seen_ids:
                    continue
                dx_px = abs(entry.exit_x - x)
                dy_px = abs(entry.exit_y - y)
                if dx_px <= pixel_radius and dy_px <= pixel_radius:
                    candidates.append(entry)
                    seen_ids.add(entry.track_id)

        return candidates

    def scan_all(self) -> List[IdentityEntry]:
        """Brute-force: return every entry. For small archives."""
        return self.all_entries()
