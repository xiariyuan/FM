from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


def _crop_tlbr(image: np.ndarray, tlbr: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in tlbr.tolist()]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((32, 16, 3), dtype=np.uint8)
    return crop


def _to_gray_resized(image: np.ndarray, tlbr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    crop = _crop_tlbr(image, tlbr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    resized = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def _haar_level(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if image.shape[0] % 2 == 1:
        image = image[:-1, :]
    if image.shape[1] % 2 == 1:
        image = image[:, :-1]
    x00 = image[0::2, 0::2]
    x01 = image[0::2, 1::2]
    x10 = image[1::2, 0::2]
    x11 = image[1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (x00 - x01 + x10 - x11) * 0.5
    hl = (x00 + x01 - x10 - x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def _pack_descriptor(image: np.ndarray) -> np.ndarray:
    pooled = cv2.resize(np.abs(image), (4, 8), interpolation=cv2.INTER_AREA)
    vector = pooled.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    return vector


@dataclass
class BandDescriptor:
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "low": self.low,
            "mid": self.mid,
            "high": self.high,
        }


def extract_band_descriptor(
    image: np.ndarray,
    tlbr: np.ndarray,
    *,
    image_height: int = 128,
    image_width: int = 64,
) -> BandDescriptor:
    gray = _to_gray_resized(image, tlbr, (image_height, image_width))
    ll1, lh1, hl1, hh1 = _haar_level(gray)
    ll2, lh2, hl2, hh2 = _haar_level(ll1)
    low = _pack_descriptor(ll2)
    mid = _pack_descriptor((lh2 + hl2 + hh2) / 3.0)
    high = _pack_descriptor((lh1 + hl1 + hh1) / 3.0)
    return BandDescriptor(low=low, mid=mid, high=high)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
