"""CommitController — two-phase commit for re-entry identity recovery.

Encapsulates the pending → confirm → reactivate protocol already present in
bot_sort.py.  The controller is stateful (holds pending proposals) but does
not own any track objects — it only returns commit decisions.

Phase 1 (propose): a candidate match enters the pending pool.
Phase 2 (confirm): if the same (track_id, det) pair is matched consecutively
                    for *confirm_streak* frames with score >= *confirm_min_sim*,
                    the controller returns a COMMIT decision.
If the pair is not seen again within *confirm_gap* frames, the proposal expires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class CommitDecision(Enum):
    PENDING = "pending"  # first match, need more evidence
    CONFIRMED = "confirmed"  # enough streak → commit
    RESET = "reset"  # gap broke streak
    SKIP = "skip"  # pair below threshold


@dataclass
class PendingProposal:
    track_id: int
    det_index: int
    score: float
    streak: int = 1
    last_frame: int = -1
    origin: str = ""


@dataclass
class CommitStats:
    enabled: bool = True
    frames: int = 0
    proposals: int = 0
    updates: int = 0
    resets: int = 0
    confirmations: int = 0
    commits: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frames": self.frames,
            "proposals": self.proposals,
            "updates": self.updates,
            "resets": self.resets,
            "confirmations": self.confirmations,
            "commits": self.commits,
        }


class CommitController:
    """Two-phase commit protocol for re-entry recovery.

    Parameters
    ----------
    confirm_streak : int
        Number of consecutive matching frames needed to confirm.
    confirm_gap : int
        Max gap between matching frames before the streak resets.
    confirm_min_similarity : float
        Minimum score to count as a match towards the streak.
    """

    def __init__(
        self,
        confirm_streak: int = 2,
        confirm_gap: int = 2,
        confirm_min_similarity: float = 0.65,
    ):
        self.confirm_streak = confirm_streak
        self.confirm_gap = confirm_gap
        self.confirm_min_similarity = confirm_min_similarity

        self._pending: Dict[int, PendingProposal] = {}
        self.stats = CommitStats()

    def propose(
        self,
        track_id: int,
        det_index: int,
        score: float,
        frame_id: int,
        origin: str = "",
    ) -> CommitDecision:
        """Feed a candidate match.  Returns the current commit state.

        Call this each frame that a (track, detection) pair is matched.
        """
        self.stats.frames = frame_id

        key = track_id
        existing = self._pending.get(key)

        if existing is None:
            # First proposal
            self._pending[key] = PendingProposal(
                track_id=track_id,
                det_index=det_index,
                score=score,
                streak=1,
                last_frame=frame_id,
                origin=origin,
            )
            self.stats.proposals += 1
            return CommitDecision.PENDING

        # Check if streak was broken
        if existing.last_frame != frame_id - 1:
            existing.det_index = det_index
            existing.score = score
            existing.streak = 1
            existing.last_frame = frame_id
            existing.origin = origin
            self.stats.resets += 1
            return CommitDecision.RESET

        # Continue streak
        existing.det_index = det_index
        existing.score = score
        existing.streak += 1
        existing.last_frame = frame_id
        existing.origin = origin
        self.stats.updates += 1

        # Check confirmation
        if (
            existing.streak >= self.confirm_streak
            and score >= self.confirm_min_similarity
        ):
            self._pending.pop(key, None)
            self.stats.confirmations += 1
            self.stats.commits += 1
            return CommitDecision.CONFIRMED

        return CommitDecision.PENDING

    def cleanup(
        self,
        current_frame: int,
        active_track_ids: set[int],
    ) -> list[Tuple[int, int]]:
        """Remove stale proposals. Returns list of (track_id, det_index) that expired."""
        expired = []
        stale_keys = []
        for key, proposal in self._pending.items():
            if proposal.track_id not in active_track_ids:
                stale_keys.append(key)
                expired.append((proposal.track_id, proposal.det_index))
                continue
            if current_frame - proposal.last_frame > self.confirm_gap:
                stale_keys.append(key)
                self.stats.resets += 1
        for key in stale_keys:
            self._pending.pop(key, None)
        return expired

    def get_pending(self, track_id: int) -> Optional[PendingProposal]:
        return self._pending.get(track_id)

    def pop_pending(self, track_id: int) -> Optional[PendingProposal]:
        return self._pending.pop(track_id, None)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def reset(self) -> None:
        self._pending.clear()
