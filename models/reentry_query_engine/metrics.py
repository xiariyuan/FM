"""ReentryMetrics — query-level observability for the re-entry engine.

Tracks per-frame and cumulative statistics: candidate set sizes, query
latency, recovery precision/recall, false recovery rate, gap-binned
recovery rates.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FrameMetrics:
    frame_id: int
    detections_count: int = 0
    queries_issued: int = 0
    queries_skipped: int = 0
    candidates_returned: int = 0  # total across all queries
    matches_above_threshold: int = 0
    commits: int = 0
    query_time_ms: float = 0.0


class ReentryMetrics:
    """Accumulates re-entry query metrics over a tracking session."""

    def __init__(self):
        self.enabled = True
        self._frame_metrics: List[FrameMetrics] = []
        self._current: FrameMetrics | None = None

        # Cumulative counters
        self.total_queries = 0
        self.total_skipped = 0
        self.total_candidates = 0
        self.total_matches = 0
        self.total_commits = 0
        self.total_query_time_ms = 0.0

        # Gap-binned recovery rates
        # gap_bin -> {"attempts": N, "commits": N}
        self._gap_bins: Dict[str, Dict[str, int]] = defaultdict(lambda: {"attempts": 0, "commits": 0})

        # Candidate set size histogram
        self._candidate_sizes: List[int] = []

    def begin_frame(self, frame_id: int, detections_count: int = 0) -> None:
        self._current = FrameMetrics(frame_id=frame_id, detections_count=detections_count)

    def record_query(
        self,
        candidates_count: int,
        matches_count: int,
        query_time_ms: float = 0.0,
        skipped: bool = False,
    ) -> None:
        if self._current is None:
            return
        if skipped:
            self._current.queries_skipped += 1
            self.total_skipped += 1
        else:
            self._current.queries_issued += 1
            self._current.candidates_returned += candidates_count
            self._current.matches_above_threshold += matches_count
            self._current.query_time_ms += query_time_ms
            self.total_queries += 1
            self.total_candidates += candidates_count
            self.total_matches += matches_count
            self.total_query_time_ms += query_time_ms
            self._candidate_sizes.append(candidates_count)

    def record_commit(self, gap: int = 0) -> None:
        if self._current is None:
            return
        self._current.commits += 1
        self.total_commits += 1
        bin_key = self._gap_bin_key(gap)
        self._gap_bins[bin_key]["commits"] += 1

    def record_recovery_attempt(self, gap: int = 0) -> None:
        bin_key = self._gap_bin_key(gap)
        self._gap_bins[bin_key]["attempts"] += 1

    def end_frame(self) -> None:
        if self._current is not None:
            self._frame_metrics.append(self._current)
            self._current = None

    # -- reporting ----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        frames = len(self._frame_metrics)
        avg_candidates = (
            self.total_candidates / max(1, self.total_queries)
            if self.total_queries > 0
            else 0.0
        )
        avg_query_ms = (
            self.total_query_time_ms / max(1, self.total_queries)
            if self.total_queries > 0
            else 0.0
        )
        recovery_rate = (
            self.total_commits / max(1, self.total_matches)
            if self.total_matches > 0
            else 0.0
        )

        gap_binned = {}
        for bin_key, counts in sorted(self._gap_bins.items()):
            att = counts["attempts"]
            com = counts["commits"]
            gap_binned[bin_key] = {
                "attempts": att,
                "commits": com,
                "rate": com / max(1, att),
            }

        p50 = p90 = p99 = 0
        if self._candidate_sizes:
            s = sorted(self._candidate_sizes)
            p50 = s[len(s) * 50 // 100]
            p90 = s[min(len(s) * 90 // 100, len(s) - 1)]
            p99 = s[min(len(s) * 99 // 100, len(s) - 1)]

        return {
            "enabled": self.enabled,
            "frames": frames,
            "total_queries": self.total_queries,
            "total_skipped": self.total_skipped,
            "total_commits": self.total_commits,
            "total_matches_above_threshold": self.total_matches,
            "recovery_rate": round(recovery_rate, 4),
            "avg_candidates_per_query": round(avg_candidates, 2),
            "avg_query_time_ms": round(avg_query_ms, 3),
            "total_query_time_ms": round(self.total_query_time_ms, 1),
            "candidate_set_size_p50": p50,
            "candidate_set_size_p90": p90,
            "candidate_set_size_p99": p99,
            "gap_binned": gap_binned,
        }

    def summary_csv_row(self, run_name: str = "") -> Dict[str, str]:
        s = self.summary()
        return {
            "run": run_name,
            "frames": str(s["frames"]),
            "queries": str(s["total_queries"]),
            "skipped": str(s["total_skipped"]),
            "matches": str(s["total_matches_above_threshold"]),
            "commits": str(s["total_commits"]),
            "recovery_rate": str(s["recovery_rate"]),
            "avg_candidates": str(s["avg_candidates_per_query"]),
            "avg_query_ms": str(s["avg_query_time_ms"]),
        }

    @staticmethod
    def _gap_bin_key(gap: int) -> str:
        if gap <= 5:
            return "1-5"
        elif gap <= 10:
            return "6-10"
        elif gap <= 20:
            return "11-20"
        elif gap <= 40:
            return "21-40"
        elif gap <= 80:
            return "41-80"
        else:
            return "81+"
