"""CCRC calibrators: conflict-conditioned reliability calibration.

Three calibration methods, ordered by complexity:
1. Global temperature scaling
2. Feature-binned calibration (per-bucket temperature)
3. Small nonlinear MLP calibrator

All take HACA runtime signals and output p(commit is correct).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Features used for conditioning (from Phase 3.5 stability analysis)
# Top stable features: candidate_count, beta_hist, top1_top2_gap, margin,
#                      ambiguity_score, bg_prob, activation, nearby_track_count
CCRC_FEATURE_NAMES = (
    "s_final", "margin", "activation", "bg_prob",
    "beta_hist", "candidate_count", "top1_top2_gap",
    "nearby_track_count", "det_score",
)
CCRC_FEATURE_DIM = len(CCRC_FEATURE_NAMES)


def _logit(p, eps=1e-6):
    """Convert probability to logit space."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


# ---------------------------------------------------------------------------
# 1. Global Temperature Scaling (logit-space)
# ---------------------------------------------------------------------------
class GlobalTemperatureScaling(nn.Module):
    """Standard temperature scaling in logit space.

    logit(p) -> logit(p) / T + b -> sigmoid
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Calibrate scores in logit space."""
        logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
        return torch.sigmoid(logits / self.temperature.clamp(min=0.01) + self.bias)

    def predict_numpy(self, scores: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = float(self.temperature)
            b = float(self.bias)
            logits = _logit(scores)
            return _sigmoid(logits / max(t, 0.01) + b)


# ---------------------------------------------------------------------------
# 1b. Platt Scaling
# ---------------------------------------------------------------------------
class PlattScaling(nn.Module):
    """Platt scaling: sigmoid(a * logit(p) + b)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
        return torch.sigmoid(self.a * logits + self.b)

    def predict_numpy(self, scores: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = float(self.a)
            b = float(self.b)
            logits = _logit(scores)
            return _sigmoid(a * logits + b)


# ---------------------------------------------------------------------------
# 2. Feature-Binned Calibration
# ---------------------------------------------------------------------------
class FeatureBinnedCalibrator:
    """Per-feature-bucket temperature scaling.

    Fits a separate temperature for each bucket of a conditioning feature.
    This allows the calibration to vary non-monotonically with the feature.
    """

    def __init__(self, feature_name: str, n_buckets: int = 4):
        self.feature_name = feature_name
        self.n_buckets = n_buckets
        self.edges: List[float] = []
        self.temperatures: List[float] = [1.0] * n_buckets
        self.bucket_counts: List[int] = [0] * n_buckets

    def fit(self, scores: np.ndarray, features: np.ndarray, labels: np.ndarray):
        """Fit per-bucket temperatures.

        Args:
            scores: (N,) raw s_final scores
            features: dict-like or (N, D) feature array
            labels: (N,) binary labels (1=correct, 0=incorrect)
        """
        feat_vals = features if features.ndim == 1 else features[:, 0]
        edges = np.percentile(feat_vals, [100 * i / self.n_buckets for i in range(1, self.n_buckets)])
        self.edges = edges.tolist()

        bucket_ids = np.digitize(feat_vals, edges)
        for b in range(self.n_buckets):
            mask = bucket_ids == b
            if mask.sum() < 5:
                self.temperatures[b] = 1.0
                self.bucket_counts[b] = int(mask.sum())
                continue

            b_scores = scores[mask]
            b_labels = labels[mask]

            # Optimize temperature via grid search (more robust than gradient for small data)
            best_t, best_nll = 1.0, float("inf")
            for t in np.arange(0.1, 5.0, 0.1):
                logits = _logit(b_scores)
                calibrated = _sigmoid(logits / t)
                calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
                nll = -np.mean(b_labels * np.log(calibrated) + (1 - b_labels) * np.log(1 - calibrated))
                if nll < best_nll:
                    best_nll = nll
                    best_t = float(t)
            self.temperatures[b] = best_t
            self.bucket_counts[b] = int(mask.sum())

    def predict(self, scores: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calibrate scores conditioned on feature buckets (logit-space)."""
        feat_vals = features if features.ndim == 1 else features[:, 0]
        bucket_ids = np.digitize(feat_vals, np.array(self.edges))
        calibrated = np.zeros_like(scores)
        for b in range(self.n_buckets):
            mask = bucket_ids == b
            if mask.sum() == 0:
                continue
            t = self.temperatures[b]
            logits = _logit(scores[mask])
            calibrated[mask] = _sigmoid(logits / max(t, 0.01))
        return calibrated

    def get_params(self) -> dict:
        return {
            "feature": self.feature_name,
            "n_buckets": self.n_buckets,
            "edges": self.edges,
            "temperatures": self.temperatures,
            "bucket_counts": self.bucket_counts,
        }


# ---------------------------------------------------------------------------
# 3. Small Nonlinear MLP Calibrator
# ---------------------------------------------------------------------------
class MLP_Calibrator(nn.Module):
    """Small nonlinear calibrator: [s_final, features] -> p(correct).

    Uses only top stable features from Phase 3.5 to avoid overfitting.
    Architecture: input_dim -> 32 -> 16 -> 1 (sigmoid out)
    """

    def __init__(self, input_dim: int = CCRC_FEATURE_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, input_dim) -> (N,) calibrated probabilities"""
        return torch.sigmoid(self.net(x).squeeze(-1))

    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return self.forward(t).numpy()

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        ckpt = {"state_dict": self.state_dict(), "metadata": metadata or {}}
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "MLP_Calibrator":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls()
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        return model


# ---------------------------------------------------------------------------
# 4. Multi-feature Binned Calibrator (combo of multiple features)
# ---------------------------------------------------------------------------
class MultiFeatureBinnedCalibrator:
    """Calibration conditioned on multiple features via sequential binning.

    Uses top 3 stable features (candidate_count, margin, top1_top2_gap)
    to create a joint bucket, then fits per-bucket temperature.
    """

    def __init__(self, feature_names: List[str], n_buckets_per_feature: int = 3):
        self.feature_names = feature_names
        self.n_per = n_buckets_per_feature
        self.edges_per_feature: Dict[str, List[float]] = {}
        self.temperatures: Dict[tuple, float] = {}
        self.bucket_counts: Dict[tuple, int] = {}

    def _get_bucket_id(self, feature_values: np.ndarray) -> tuple:
        """Get multi-dimensional bucket id."""
        ids = []
        for i, fname in enumerate(self.feature_names):
            edges = self.edges_per_feature.get(fname, [])
            ids.append(int(np.digitize(feature_values[i], edges)))
        return tuple(ids)

    def fit(self, scores: np.ndarray, features: np.ndarray, feature_names: List[str], labels: np.ndarray):
        """Fit per-joint-bucket temperatures."""
        name_to_idx = {n: i for i, n in enumerate(feature_names)}

        # Compute edges per feature
        for fname in self.feature_names:
            idx = name_to_idx.get(fname)
            if idx is None:
                continue
            feat_vals = features[:, idx]
            edges = np.percentile(feat_vals, [100 * i / self.n_per for i in range(1, self.n_per)])
            self.edges_per_feature[fname] = edges.tolist()

        # Assign bucket ids and fit temperatures
        for n in range(len(scores)):
            fv = np.array([features[n, name_to_idx[fn]] for fn in self.feature_names])
            bid = self._get_bucket_id(fv)
            self.bucket_counts[bid] = self.bucket_counts.get(bid, 0) + 1

        # Fit per bucket
        for bid in self.bucket_counts:
            masks = []
            for i in range(len(scores)):
                fv = np.array([features[i, name_to_idx[fn]] for fn in self.feature_names])
                if self._get_bucket_id(fv) == bid:
                    masks.append(i)
            if len(masks) < 5:
                self.temperatures[bid] = 1.0
                continue
            b_scores = scores[masks]
            b_labels = labels[masks]
            best_t, best_nll = 1.0, float("inf")
            for t in np.arange(0.1, 5.0, 0.1):
                logits = _logit(b_scores)
                calibrated = _sigmoid(logits / t)
                calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
                nll = -np.mean(b_labels * np.log(calibrated) + (1 - b_labels) * np.log(1 - calibrated))
                if nll < best_nll:
                    best_nll = nll
                    best_t = float(t)
            self.temperatures[bid] = best_t

    def predict(self, scores: np.ndarray, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        calibrated = np.ones_like(scores)
        for i in range(len(scores)):
            fv = np.array([features[i, name_to_idx[fn]] for fn in self.feature_names])
            bid = self._get_bucket_id(fv)
            t = self.temperatures.get(bid, 1.0)
            logits = _logit(scores[i:i+1])
            calibrated[i] = float(_sigmoid(logits / max(t, 0.01)))
        return calibrated

    def get_params(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "n_per": self.n_per,
            "edges": {k: v for k, v in self.edges_per_feature.items()},
            "temperatures": {str(k): v for k, v in self.temperatures.items()},
            "n_buckets_filled": len(self.temperatures),
        }
