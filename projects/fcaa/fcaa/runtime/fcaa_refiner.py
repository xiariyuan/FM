from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from projects.fcaa.fcaa.model.freq_dwt import BandDescriptor, cosine_similarity, extract_band_descriptor
from projects.fcaa.fcaa.model.pair_scorer import FCAAPairScorer


@dataclass
class FCAAConfig:
    enabled: bool = False
    scorer_checkpoint: str = ""
    trigger_mode: str = "row_margin"
    trigger_margin: float = 0.05
    lambda_weight: float = 0.3
    top_k: int = 3
    appearance_thresh: float = 0.25
    crop_height: int = 128
    crop_width: int = 64
    device: str = "cpu"


class FCAARefiner:
    def __init__(self, config: FCAAConfig) -> None:
        self.config = config
        self.model: Optional[FCAAPairScorer] = None
        self.device = torch.device(config.device)
        self.mode = "freq"
        if config.enabled and config.scorer_checkpoint:
            payload = torch.load(config.scorer_checkpoint, map_location="cpu")
            input_dim = int(payload.get("input_dim", 4))
            self.mode = str(payload.get("mode", "freq"))
            model = FCAAPairScorer(input_dim=input_dim)
            model.load_state_dict(payload["model_state"])
            model.eval()
            model.to(self.device)
            self.model = model

    def is_active(self) -> bool:
        return bool(self.config.enabled and self.model is not None)

    def _pair_features(self, track: object, det_desc: BandDescriptor, s_reid: float) -> List[float]:
        if self.mode == "control":
            return [float(s_reid)]
        s_low = cosine_similarity(np.asarray(track.fcaa_low), det_desc.low)
        s_mid = cosine_similarity(np.asarray(track.fcaa_mid), det_desc.mid)
        s_high = cosine_similarity(np.asarray(track.fcaa_high), det_desc.high)
        return [float(s_reid), s_low, s_mid, s_high]

    def _predict_probs(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        with torch.no_grad():
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.model(feature_tensor)).cpu().numpy()

    def _refine_row_margin(
        self,
        *,
        track_pool: Sequence[object],
        det_descs: Sequence[BandDescriptor],
        refined: np.ndarray,
        base_emb: np.ndarray,
        base_cost: np.ndarray,
        base_similarity: np.ndarray,
        ious_dists_mask: np.ndarray,
        debug: Dict[str, object],
    ) -> np.ndarray:
        for track_idx, track in enumerate(track_pool):
            valid_det_idx = np.where(~ious_dists_mask[track_idx])[0]
            if valid_det_idx.size < 2:
                continue
            order = valid_det_idx[np.argsort(base_cost[track_idx, valid_det_idx])]
            order = order[: max(1, int(self.config.top_k))]
            if order.size < 2:
                continue
            best = float(base_similarity[track_idx, order[0]])
            second = float(base_similarity[track_idx, order[1]])
            margin = best - second
            if margin >= float(self.config.trigger_margin):
                continue
            if any(getattr(track, f"fcaa_{band_name}", None) is None for band_name in ("low", "mid", "high")):
                continue
            features = [
                self._pair_features(track, det_descs[det_idx], float(1.0 - base_emb[track_idx, det_idx]))
                for det_idx in order.tolist()
            ]
            probs = self._predict_probs(features)
            refined_scores = []
            for local_idx, det_idx in enumerate(order.tolist()):
                s_reid = float(1.0 - base_emb[track_idx, det_idx])
                refined_similarity = (1.0 - float(self.config.lambda_weight)) * s_reid + float(self.config.lambda_weight) * float(probs[local_idx])
                refined_scores.append(float(refined_similarity))
            base_best_det_idx = int(order[0])
            refined_best_local_idx = int(np.argmax(np.asarray(refined_scores, dtype=np.float32)))
            refined_best_det_idx = int(order[refined_best_local_idx])
            changed_winner = bool(base_best_det_idx != refined_best_det_idx)
            debug["triggered_tracks"] = int(debug["triggered_tracks"]) + 1
            debug["refined_pairs"] = int(debug["refined_pairs"]) + int(len(order))
            debug["trigger_groups"] = int(debug["trigger_groups"]) + 1
            if changed_winner:
                debug["changed_groups"] = int(debug["changed_groups"]) + 1
            debug["trigger_rows"].append(
                {
                    "trigger_mode": "row_margin",
                    "track_id": int(getattr(track, "track_id", -1)),
                    "candidate_count": int(len(order)),
                    "margin": margin,
                    "base_best_det_idx": base_best_det_idx,
                    "refined_best_det_idx": refined_best_det_idx,
                    "changed_winner": changed_winner,
                }
            )
            for local_idx, det_idx in enumerate(order.tolist()):
                refined[track_idx, det_idx] = float(np.clip(1.0 - refined_scores[local_idx], 0.0, 1.0))
        return refined

    def _refine_shared_det_top1(
        self,
        *,
        track_pool: Sequence[object],
        det_descs: Sequence[BandDescriptor],
        refined: np.ndarray,
        base_emb: np.ndarray,
        base_cost: np.ndarray,
        ious_dists_mask: np.ndarray,
        debug: Dict[str, object],
        use_margin_gate: bool,
    ) -> np.ndarray:
        shared_det_groups: Dict[int, List[int]] = {}
        for track_idx, track in enumerate(track_pool):
            valid_det_idx = np.where(~ious_dists_mask[track_idx])[0]
            if valid_det_idx.size == 0:
                continue
            if any(getattr(track, f"fcaa_{band_name}", None) is None for band_name in ("low", "mid", "high")):
                continue
            order = valid_det_idx[np.argsort(base_cost[track_idx, valid_det_idx])]
            if order.size == 0:
                continue
            det_idx = int(order[0])
            shared_det_groups.setdefault(det_idx, []).append(int(track_idx))

        for det_idx, contender_track_indices in shared_det_groups.items():
            if len(contender_track_indices) < 2:
                continue
            features: List[List[float]] = []
            base_scores: List[float] = []
            for track_idx in contender_track_indices:
                track = track_pool[track_idx]
                s_reid = float(1.0 - base_emb[track_idx, det_idx])
                features.append(self._pair_features(track, det_descs[det_idx], s_reid))
                base_scores.append(float(1.0 - base_cost[track_idx, det_idx]))
            sorted_base_scores = sorted(base_scores, reverse=True)
            margin = float(sorted_base_scores[0] - sorted_base_scores[1]) if len(sorted_base_scores) >= 2 else 0.0
            if use_margin_gate and margin >= float(self.config.trigger_margin):
                continue
            probs = self._predict_probs(features)
            refined_scores = []
            for local_idx, track_idx in enumerate(contender_track_indices):
                s_reid = float(1.0 - base_emb[track_idx, det_idx])
                refined_similarity = (1.0 - float(self.config.lambda_weight)) * s_reid + float(self.config.lambda_weight) * float(probs[local_idx])
                refined_scores.append(float(refined_similarity))
            base_winner_local_idx = int(np.argmax(np.asarray(base_scores, dtype=np.float32)))
            refined_winner_local_idx = int(np.argmax(np.asarray(refined_scores, dtype=np.float32)))
            base_winner_track_idx = int(contender_track_indices[base_winner_local_idx])
            refined_winner_track_idx = int(contender_track_indices[refined_winner_local_idx])
            changed_winner = bool(base_winner_track_idx != refined_winner_track_idx)
            debug["triggered_tracks"] = int(debug["triggered_tracks"]) + int(len(contender_track_indices))
            debug["refined_pairs"] = int(debug["refined_pairs"]) + int(len(contender_track_indices))
            debug["trigger_groups"] = int(debug["trigger_groups"]) + 1
            if changed_winner:
                debug["changed_groups"] = int(debug["changed_groups"]) + 1
            debug["trigger_rows"].append(
                {
                    "trigger_mode": "shared_det_top1",
                    "det_idx": int(det_idx),
                    "candidate_count": int(len(contender_track_indices)),
                    "track_ids": [int(getattr(track_pool[track_idx], "track_id", -1)) for track_idx in contender_track_indices],
                    "base_winner_track_id": int(getattr(track_pool[base_winner_track_idx], "track_id", -1)),
                    "refined_winner_track_id": int(getattr(track_pool[refined_winner_track_idx], "track_id", -1)),
                    "margin": margin,
                    "changed_winner": changed_winner,
                }
            )
            for local_idx, track_idx in enumerate(contender_track_indices):
                refined[track_idx, det_idx] = float(np.clip(1.0 - refined_scores[local_idx], 0.0, 1.0))
        return refined

    def extract_detection_descriptors(self, image: np.ndarray, detections: Sequence[object]) -> List[BandDescriptor]:
        descriptors: List[BandDescriptor] = []
        for det in detections:
            descriptors.append(
                extract_band_descriptor(
                    image,
                    np.asarray(det.tlbr, dtype=np.float32),
                    image_height=self.config.crop_height,
                    image_width=self.config.crop_width,
                )
            )
        return descriptors

    def update_track_memory(self, track: object, descriptor: BandDescriptor, momentum: float = 0.9) -> None:
        for band_name in ("low", "mid", "high"):
            current = getattr(track, f"fcaa_{band_name}", None)
            new_value = getattr(descriptor, band_name)
            if current is None:
                setattr(track, f"fcaa_{band_name}", np.asarray(new_value, dtype=np.float32))
                continue
            mixed = float(momentum) * np.asarray(current, dtype=np.float32) + (1.0 - float(momentum)) * np.asarray(new_value, dtype=np.float32)
            norm = float(np.linalg.norm(mixed))
            if norm > 1e-8:
                mixed = mixed / norm
            setattr(track, f"fcaa_{band_name}", mixed.astype(np.float32))

    def refine_embedding_cost(
        self,
        *,
        track_pool: Sequence[object],
        detections: Sequence[object],
        emb_dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        ious_dists_mask: np.ndarray,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        debug: Dict[str, object] = {
            "triggered_tracks": 0,
            "refined_pairs": 0,
            "trigger_groups": 0,
            "changed_groups": 0,
            "trigger_rows": [],
        }
        if not self.is_active() or emb_dists.size == 0 or len(track_pool) == 0 or len(detections) == 0:
            return emb_dists, debug

        det_descs = self.extract_detection_descriptors(image, detections)
        refined = emb_dists.copy()
        base_emb = emb_dists.copy()
        clipped_base = base_emb.copy()
        clipped_base[clipped_base > float(self.config.appearance_thresh)] = 1.0
        clipped_base[ious_dists_mask] = 1.0
        base_cost = np.minimum(raw_ious_dists, clipped_base)
        base_similarity = 1.0 - base_cost
        if str(self.config.trigger_mode) in {"shared_det_top1", "shared_det_top1_margin"}:
            refined = self._refine_shared_det_top1(
                track_pool=track_pool,
                det_descs=det_descs,
                refined=refined,
                base_emb=base_emb,
                base_cost=base_cost,
                ious_dists_mask=ious_dists_mask,
                debug=debug,
                use_margin_gate=str(self.config.trigger_mode) == "shared_det_top1_margin",
            )
        else:
            refined = self._refine_row_margin(
                track_pool=track_pool,
                det_descs=det_descs,
                refined=refined,
                base_emb=base_emb,
                base_cost=base_cost,
                base_similarity=base_similarity,
                ious_dists_mask=ious_dists_mask,
                debug=debug,
            )
        return refined, debug
