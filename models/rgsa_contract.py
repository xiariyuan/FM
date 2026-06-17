"""RGSA three-stage interface contract.

This module defines the data structures that pass between stages.
It is the SINGLE source of truth for the staged pipeline contract.

Stage 1 (Deferral):  per det-track pair -> accept / defer / reject
Stage 2 (Verifier):  per deferred det -> confirm_local / veto_local
Stage 3 (Retrieval): per unmatched det -> recover / miss

Invariant:
  - A detection is matched at most once per frame.
  - Stage 2/3 never rewrite a Stage 1 accept.
  - Stage 3 recovery has priority over newborn creation.
  - Stage 2 is a VERIFIER, not a rewriter. It only confirms or vetoes
    the host's top-1 local candidate for deferred detections. It does
    NOT do rank switch / candidate rewrite within top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Action label constants
# ---------------------------------------------------------------------------
# Stage 1
ACTION_ACCEPT = 0   # keep host match
ACTION_DEFER = 1    # send downstream
ACTION_REJECT = 2   # discard host match

# Stage 2 (verifier): confirm or veto the local top-1 candidate
ACTION_CONFIRM = 0  # host's local top-1 is good enough
ACTION_VETO = 1     # host's local top-1 is not reliable

# Stage 3
ACTION_RECOVER = 0  # match from archive
ACTION_MISS = 1     # give up, create newborn

STAGE1_ACTION_NAMES = ("accept", "defer", "reject")
STAGE2_ACTION_NAMES = ("confirm_local", "veto_local")
STAGE3_ACTION_NAMES = ("recover", "miss")


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
HACA_PAIR_FEATURE_NAMES = (
    "anchor_sim", "spatial_sim", "motion_sim",
    "temp_sim", "hist_last_sim", "hist_max_sim", "hist_std_sim",
    "gap_log1p", "hist_norm", "stability", "coherence",
    "anchor_z", "anchor_margin", "anchor_rank", "det_score",
)
HACA_PAIR_FEATURE_DIM = len(HACA_PAIR_FEATURE_NAMES)

STAGE1_FEATURE_NAMES = (
    "activation", "margin", "entropy", "bg_prob",
    "beta_hist", "beta_ood", "ood_score",
    "track_gap", "track_age", "history_len", "det_score",
)
STAGE1_FEATURE_DIM = len(STAGE1_FEATURE_NAMES)

# Verifier features: HACA runtime signals available at decision time
VERIFIER_FEATURE_NAMES = (
    "s_final", "margin", "entropy", "activation",
    "bg_prob", "beta_hist", "beta_ood",
    "track_gap", "track_age", "history_len", "det_score",
)
VERIFIER_FEATURE_DIM = len(VERIFIER_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Stage 1 -> Stage 2 payload
# ---------------------------------------------------------------------------
@dataclass
class Stage1Output:
    """Output of Stage 1 deferral head for one frame."""

    accepted_det_ids: List[int] = field(default_factory=list)
    deferred_det_ids: List[int] = field(default_factory=list)
    rejected_det_ids: List[int] = field(default_factory=list)

    accepted_matches: Dict[int, int] = field(default_factory=dict)  # det_id -> track_id
    deferred_topk: Dict[int, List[int]] = field(default_factory=dict)  # det_id -> [track_ids]
    deferred_features: Dict[int, np.ndarray] = field(default_factory=dict)  # det_id -> (k, pair_dim)
    deferred_host_signals: Dict[int, Dict[str, float]] = field(default_factory=dict)
    deferred_cost_bias: Dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 2 -> Stage 3 payload (VERIFIER semantics)
# ---------------------------------------------------------------------------
@dataclass
class Stage2Output:
    """Output of Stage 2 verifier for one frame.

    Stage 2 is a VERIFIER, not a rewriter. It only confirms or vetoes
    the host's local top-1 candidate for deferred detections.
    """

    confirmed_matches: Dict[int, int] = field(default_factory=dict)  # det_id -> track_id
    vetoed_det_ids: List[int] = field(default_factory=list)
    best_local_scores: Dict[int, float] = field(default_factory=dict)
    verification_signals: Dict[int, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 3 output
# ---------------------------------------------------------------------------
@dataclass
class Stage3Output:
    """Output of Stage 3 retrieval late recovery for one frame."""

    recovered_matches: Dict[int, int] = field(default_factory=dict)  # det_id -> archive track_id
    remaining_unmatched_det_ids: List[int] = field(default_factory=list)
    recovery_scores: Dict[int, float] = field(default_factory=dict)
