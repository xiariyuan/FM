import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), 1e-12, None)


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # tanh approximation (same as PyTorch default GELU)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


@dataclass
class LaplaceAlphaRCalibrator:
    """
    Backward-compatible learned LTRA checkpoint.

    Old checkpoints contain only the tiny pairwise alpha/r head:
      - feature_mean / feature_std
      - W1 / b1 / W_alpha / b_alpha / W_r / b_r

    New checkpoints may additionally contain a learnable Laplace pole bank:
      - tau_values
      - track_feature_mean / track_feature_std
      - W_track1 / b_track1 / W_track2 / b_track2
    """

    pair_feature_mean: np.ndarray
    pair_feature_std: np.ndarray
    W1: Optional[np.ndarray] = None
    b1: Optional[np.ndarray] = None
    W_alpha: Optional[np.ndarray] = None
    b_alpha: Optional[np.ndarray] = None
    W_r: Optional[np.ndarray] = None
    b_r: Optional[np.ndarray] = None

    tau_values: Optional[np.ndarray] = None
    track_feature_mean: Optional[np.ndarray] = None
    track_feature_std: Optional[np.ndarray] = None
    W_track1: Optional[np.ndarray] = None
    b_track1: Optional[np.ndarray] = None
    W_track2: Optional[np.ndarray] = None
    b_track2: Optional[np.ndarray] = None

    pair_feature_names: Optional[Tuple[str, ...]] = None
    track_feature_names: Optional[Tuple[str, ...]] = None
    temperature: float = 1.0

    @classmethod
    def from_npz(cls, path: str) -> "LaplaceAlphaRCalibrator":
        if not path:
            raise ValueError("Missing calibrator path")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing calibrator file: {path}")

        data = np.load(path, allow_pickle=True)

        def maybe(name: str):
            return np.asarray(data[name], dtype=np.float32) if name in data.files else None

        pair_feature_mean = (
            np.asarray(data["pair_feature_mean"], dtype=np.float32)
            if "pair_feature_mean" in data.files
            else np.asarray(data["feature_mean"], dtype=np.float32)
        )
        pair_feature_std = (
            np.asarray(data["pair_feature_std"], dtype=np.float32)
            if "pair_feature_std" in data.files
            else np.asarray(data["feature_std"], dtype=np.float32)
        )
        pair_feature_names = (
            tuple(data["pair_feature_names"].tolist())
            if "pair_feature_names" in data.files
            else (tuple(data["feature_names"].tolist()) if "feature_names" in data.files else None)
        )

        temperature = 1.0
        if "temperature" in data.files:
            temperature = float(np.asarray(data["temperature"], dtype=np.float32).reshape(-1)[0])

        return cls(
            pair_feature_mean=pair_feature_mean,
            pair_feature_std=pair_feature_std,
            W1=maybe("W1"),
            b1=maybe("b1"),
            W_alpha=maybe("W_alpha"),
            b_alpha=maybe("b_alpha"),
            W_r=maybe("W_r"),
            b_r=maybe("b_r"),
            tau_values=maybe("tau_values"),
            track_feature_mean=maybe("track_feature_mean"),
            track_feature_std=maybe("track_feature_std"),
            W_track1=maybe("W_track1"),
            b_track1=maybe("b_track1"),
            W_track2=maybe("W_track2"),
            b_track2=maybe("b_track2"),
            pair_feature_names=pair_feature_names,
            track_feature_names=tuple(data["track_feature_names"].tolist()) if "track_feature_names" in data.files else None,
            temperature=temperature,
        )

    @property
    def feature_mean(self) -> np.ndarray:
        return self.pair_feature_mean

    @property
    def feature_std(self) -> np.ndarray:
        return self.pair_feature_std

    @property
    def feature_names(self) -> Optional[Tuple[str, ...]]:
        return self.pair_feature_names

    @property
    def has_pole_bank(self) -> bool:
        return (
            self.tau_values is not None
            and self.track_feature_mean is not None
            and self.track_feature_std is not None
            and self.W_track2 is not None
            and self.b_track2 is not None
        )

    def _normalize_pair(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = np.asarray(self.pair_feature_mean, dtype=np.float32).reshape(1, -1)
        std = np.asarray(self.pair_feature_std, dtype=np.float32).reshape(1, -1)
        std = np.clip(std, 1e-6, None)
        return (x - mean) / std

    def _normalize_track(self, x: np.ndarray) -> np.ndarray:
        if self.track_feature_mean is None or self.track_feature_std is None:
            raise ValueError("Track-feature statistics missing from pole-bank checkpoint")
        x = np.asarray(x, dtype=np.float32)
        mean = np.asarray(self.track_feature_mean, dtype=np.float32).reshape(1, -1)
        std = np.asarray(self.track_feature_std, dtype=np.float32).reshape(1, -1)
        std = np.clip(std, 1e-6, None)
        return (x - mean) / std

    def validate_pair_feature_names(self, runtime_feature_names: Tuple[str, ...]) -> None:
        if self.pair_feature_names is None:
            return
        if tuple(self.pair_feature_names) != tuple(runtime_feature_names):
            raise ValueError(
                f"Pair feature mismatch. runtime={tuple(runtime_feature_names)} "
                f"checkpoint={tuple(self.pair_feature_names)}"
            )

    def validate_track_feature_names(self, runtime_feature_names: Tuple[str, ...]) -> None:
        if self.track_feature_names is None:
            return
        if tuple(self.track_feature_names) != tuple(runtime_feature_names):
            raise ValueError(
                f"Track feature mismatch. runtime={tuple(runtime_feature_names)} "
                f"checkpoint={tuple(self.track_feature_names)}"
            )

    def validate_feature_names(self, runtime_feature_names: Tuple[str, ...]) -> None:
        # Backward-compatible alias for the existing alpha/r runtime.
        self.validate_pair_feature_names(runtime_feature_names)

    def predict_alpha_r(self, pair_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        pair_features: (..., D)
        Returns:
          alpha: (...,)
          r: (...,)
        """
        feats = np.asarray(pair_features, dtype=np.float32)
        original_shape = feats.shape[:-1]
        feats = feats.reshape(-1, feats.shape[-1])
        feats = self._normalize_pair(feats)

        if self.W1 is not None and self.b1 is not None:
            h = _gelu(np.matmul(feats, self.W1) + self.b1.reshape(1, -1))
        else:
            h = feats

        if self.W_alpha is None or self.b_alpha is None or self.W_r is None or self.b_r is None:
            raise ValueError("Calibrator weights missing (W_alpha/b_alpha/W_r/b_r)")

        alpha = _sigmoid(np.matmul(h, self.W_alpha).reshape(-1) + float(np.asarray(self.b_alpha).reshape(-1)[0]))
        r = _sigmoid(np.matmul(h, self.W_r).reshape(-1) + float(np.asarray(self.b_r).reshape(-1)[0]))

        alpha = alpha.reshape(*original_shape).astype(np.float32)
        r = r.reshape(*original_shape).astype(np.float32)
        return alpha, r

    def predict_track_pi(self, track_features: np.ndarray) -> np.ndarray:
        """
        track_features: (..., D_track)
        Returns:
          pi: (..., K)
        """
        if not self.has_pole_bank:
            raise ValueError("Checkpoint does not contain a learnable pole bank")

        feats = np.asarray(track_features, dtype=np.float32)
        original_shape = feats.shape[:-1]
        feats = feats.reshape(-1, feats.shape[-1])
        feats = self._normalize_track(feats)

        if self.W_track1 is not None and self.b_track1 is not None:
            h = _gelu(np.matmul(feats, self.W_track1) + self.b_track1.reshape(1, -1))
        else:
            h = feats

        logits = np.matmul(h, self.W_track2) + self.b_track2.reshape(1, -1)
        pi = _softmax(logits, axis=-1)
        return pi.reshape(*original_shape, pi.shape[-1]).astype(np.float32)
