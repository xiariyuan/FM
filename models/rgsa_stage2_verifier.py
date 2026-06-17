"""Heuristic verifier for RGSA Stage 2.

Confirms or vetoes the host's local top-1 candidate for deferred detections
using hand-crafted rules on HACA runtime signals. No learned parameters.

Rules are conservative: confirm only when multiple signals agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from models.rgsa_contract import Stage2Output, STAGE1_FEATURE_NAMES, VERIFIER_FEATURE_NAMES


@dataclass
class VerifierRule:
    """A single heuristic confirm/veto rule."""
    name: str
    confirm_condition: str  # human-readable description


# Default rule set: confirm if (s_final >= a AND margin >= b AND entropy <= c)
# These are the most conservative rules — confirm only when HACA is confident.

DEFAULT_RULES = [
    VerifierRule(
        name="high_confidence",
        confirm_condition="s_final >= 0.8 AND margin >= 0.1 AND entropy <= 0.5",
    ),
    VerifierRule(
        name="strong_activation",
        confirm_condition="activation >= 0.5 AND margin >= 0.05 AND bg_prob <= 0.3",
    ),
    VerifierRule(
        name="reliable_history",
        confirm_condition="beta_hist * beta_ood >= 0.3 AND margin >= 0.05",
    ),
]


class HeuristicVerifier:
    """Rule-based verifier for Stage 2.

    Input: per-deferred-detection HACA runtime signals.
    Output: confirm_local or veto_local per detection.

    Rules are OR-combined: if ANY rule confirms, the detection is confirmed.
    Only detections that fail ALL rules are vetoed.
    """

    def __init__(
        self,
        # Rule 1: high_confidence
        rule1_s_final_min: float = 0.8,
        rule1_margin_min: float = 0.1,
        rule1_entropy_max: float = 0.5,
        # Rule 2: strong_activation
        rule2_activation_min: float = 0.5,
        rule2_margin_min: float = 0.05,
        rule2_bg_prob_max: float = 0.3,
        # Rule 3: reliable_history
        rule3_beta_product_min: float = 0.3,
        rule3_margin_min: float = 0.05,
    ):
        self.rule1_s_final_min = rule1_s_final_min
        self.rule1_margin_min = rule1_margin_min
        self.rule1_entropy_max = rule1_entropy_max
        self.rule2_activation_min = rule2_activation_min
        self.rule2_margin_min = rule2_margin_min
        self.rule2_bg_prob_max = rule2_bg_prob_max
        self.rule3_beta_product_min = rule3_beta_product_min
        self.rule3_margin_min = rule3_margin_min

    def _check_rule1(self, signals: Dict[str, float]) -> bool:
        """high_confidence: s_final >= min AND margin >= min AND entropy <= max"""
        return (
            signals.get("s_final", 0.0) >= self.rule1_s_final_min
            and signals.get("margin", 0.0) >= self.rule1_margin_min
            and signals.get("entropy", 1.0) <= self.rule1_entropy_max
        )

    def _check_rule2(self, signals: Dict[str, float]) -> bool:
        """strong_activation: activation >= min AND margin >= min AND bg_prob <= max"""
        return (
            signals.get("activation", 0.0) >= self.rule2_activation_min
            and signals.get("margin", 0.0) >= self.rule2_margin_min
            and signals.get("bg_prob", 1.0) <= self.rule2_bg_prob_max
        )

    def _check_rule3(self, signals: Dict[str, float]) -> bool:
        """reliable_history: beta_hist*beta_ood >= min AND margin >= min"""
        beta_product = signals.get("beta_hist", 0.0) * signals.get("beta_ood", 0.0)
        return (
            beta_product >= self.rule3_beta_product_min
            and signals.get("margin", 0.0) >= self.rule3_margin_min
        )

    def verify_single(self, signals: Dict[str, float]) -> tuple:
        """Verify a single deferred detection.

        Returns:
            (action, signals_dict) where action=0 (confirm) or 1 (veto)
        """
        r1 = self._check_rule1(signals)
        r2 = self._check_rule2(signals)
        r3 = self._check_rule3(signals)

        confirmed = r1 or r2 or r3
        action = 0 if confirmed else 1  # 0=confirm, 1=veto

        verification = {
            "rule1_high_confidence": float(r1),
            "rule2_strong_activation": float(r2),
            "rule3_reliable_history": float(r3),
            "any_confirmed": float(confirmed),
        }
        return action, verification

    def verify_batch(
        self,
        deferred_det_ids: List[int],
        signals_per_det: Dict[int, Dict[str, float]],
        track_ids_per_det: Dict[int, int],
    ) -> Stage2Output:
        """Verify a batch of deferred detections.

        Args:
            deferred_det_ids: detection ids deferred from Stage 1
            signals_per_det: det_id -> {s_final, margin, entropy, ...}
            track_ids_per_det: det_id -> host's top-1 track_id

        Returns:
            Stage2Output with confirmed_matches and vetoed_det_ids
        """
        output = Stage2Output()

        for det_id in deferred_det_ids:
            det_id = int(det_id)
            signals = signals_per_det.get(det_id, {})

            action, verification = self.verify_single(signals)
            output.verification_signals[det_id] = verification

            if action == 0:  # confirm
                track_id = track_ids_per_det.get(det_id, -1)
                if track_id >= 0:
                    output.confirmed_matches[det_id] = track_id
            else:  # veto
                output.vetoed_det_ids.append(det_id)

            output.best_local_scores[det_id] = float(signals.get("s_final", 0.0))

        return output

    def get_thresholds(self) -> Dict[str, float]:
        """Return current threshold settings for logging."""
        return {
            "rule1_s_final_min": self.rule1_s_final_min,
            "rule1_margin_min": self.rule1_margin_min,
            "rule1_entropy_max": self.rule1_entropy_max,
            "rule2_activation_min": self.rule2_activation_min,
            "rule2_margin_min": self.rule2_margin_min,
            "rule2_bg_prob_max": self.rule2_bg_prob_max,
            "rule3_beta_product_min": self.rule3_beta_product_min,
            "rule3_margin_min": self.rule3_margin_min,
        }

    @classmethod
    def from_thresholds(cls, thresholds: Dict[str, float]) -> "HeuristicVerifier":
        """Create from a threshold dict (e.g. from sweep results)."""
        return cls(**thresholds)
