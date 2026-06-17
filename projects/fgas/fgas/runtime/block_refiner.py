from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from projects.fcaa.fcaa.model.freq_dwt import BandDescriptor, cosine_similarity, extract_band_descriptor
from projects.fgas.fgas.features.acceptance_features import ACCEPTANCE_FEATURE_NAMES, build_acceptance_feature_vector
from projects.fgas.fgas.features.block_gate_features import BLOCK_GATE_FEATURE_NAMES, build_block_gate_feature_vector
from projects.fgas.fgas.features.edge_features import (
    COL_CONTEXT_FEATURE_NAMES,
    DOMAIN_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    ROW_CONTEXT_FEATURE_NAMES,
    build_domain_feature_vector,
    build_col_context_from_rows,
    build_edge_feature_vector,
    build_row_context_from_rows,
)
from projects.fgas.fgas.model.acceptance_gate import FGASAcceptanceGate
from projects.fgas.fgas.model.block_matcher import BlockMatcherOutput, FGASTrueBlockMatcher
from projects.fgas.fgas.model.block_primitive import FGASAmbiguityBlockPrimitive
from projects.fgas.fgas.model.block_resolver import FGASBlockResolver, STAGE_NAME_TO_ID
from projects.fgas.fgas.model.block_resolver_v2 import FGASAssociationResolverV2
from projects.fgas.fgas.model.block_resolver_v3_trackquery import FGASAssociationResolverV3TrackQuery
from projects.fgas.fgas.model.pair_scorer import FGASPairScorer
from projects.fgas.fgas.runtime.block_primitive_runtime import decode_block_matcher_output, decode_block_primitive_output


@dataclass
class FGASConfig:
    enabled: bool = False
    resolver_checkpoint: str = ""
    block_primitive_checkpoint: str = ""
    block_primitive_conf_thresh: float = 0.5
    block_matcher_checkpoint: str = ""
    block_matcher_margin_thresh: float = 0.0
    block_matcher_base_margin_thresh: float = 0.0
    block_matcher_base_logit_scale_override: Optional[float] = None
    block_matcher_force_only: bool = False
    block_matcher_forceonly_keep_base_on_nomatch: bool = False
    block_matcher_skip_single_row_unchanged_takeover: bool = False
    block_matcher_stale_match_bias: float = 0.0
    block_matcher_stale_match_mode: str = "all_edges"
    block_matcher_stale_match_min_time_since_update: int = 0
    block_matcher_stale_match_max_hit_streak: int = 0
    block_matcher_stale_match_min_hits: int = 0
    block_matcher_stale_match_max_component_rows: int = 0
    block_matcher_stale_match_max_component_cols: int = 0
    pair_scorer_checkpoint: str = ""
    block_gate_checkpoint: str = ""
    block_gate_thresh: float = 0.5
    top_k: int = 5
    proximity_thresh: float = 0.9
    appearance_thresh: float = 0.25
    max_rows: int = 3
    max_cols: int = 3
    crop_height: int = 128
    crop_width: int = 64
    blend_weight: float = 0.5
    assignment_mode: str = "blend"
    row_nomatch_weight: float = 0.0
    controller_enable: bool = False
    controller_edge_thresh: float = 0.6
    controller_row_defer_thresh: float = 0.6
    controller_col_newborn_thresh: float = 0.6
    controller_margin_thresh: float = 0.05
    controller_ambiguity_margin: float = 0.05
    controller_mutual_top1_only: bool = False
    controller_require_base_top1: bool = False
    controller_only_changed_blocks: bool = False
    primitive_direct_takeover: bool = False
    soft_apply_only_changed_blocks: bool = False
    soft_apply_only_changed_rows: bool = False
    soft_apply_only_changed_frontier: bool = False
    soft_allow_without_takeover: bool = False
    soft_row_base_margin_thresh: float = 1.0
    soft_changed_row_flip_gap_thresh: float = 0.0
    soft_changed_row_refined_margin_thresh: float = 0.0
    pair_ambiguity_margin: float = 0.05
    acceptance_gate_checkpoint: str = ""
    acceptance_gate_thresh: float = 0.5
    device: str = "cpu"


@dataclass
class FGASControllerActions:
    forced_matches: List[Tuple[int, int]]
    blocked_rows: List[int]
    blocked_cols: List[int]


class FGASBlockRefiner:
    def __init__(self, config: FGASConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model: Optional[FGASBlockResolver] = None
        self.block_primitive: Optional[FGASAmbiguityBlockPrimitive] = None
        self.block_matcher: Optional[FGASTrueBlockMatcher] = None
        self.acceptance_gate: Optional[FGASAcceptanceGate] = None
        self.pair_scorer: Optional[FGASPairScorer] = None
        self.block_gate: Optional[FGASAcceptanceGate] = None
        self.acceptance_gate_feature_names: List[str] = []
        self.block_gate_feature_names: List[str] = []
        self.block_primitive_feature_names: List[str] = []
        self.block_matcher_feature_names: List[str] = []
        self.pair_edge_feature_names: List[str] = []
        self.pair_row_feature_names: List[str] = []
        self.pair_col_feature_names: List[str] = []
        self.pair_domain_feature_names: List[str] = []
        self.block_matcher_margin_thresh: float = float(config.block_matcher_margin_thresh)
        self.block_matcher_base_margin_thresh: float = float(config.block_matcher_base_margin_thresh)
        self.arch = "v1"
        self.feature_names: List[str] = list(EDGE_FEATURE_NAMES)
        self.row_feature_names: List[str] = list(ROW_CONTEXT_FEATURE_NAMES)
        self.col_feature_names: List[str] = list(COL_CONTEXT_FEATURE_NAMES)
        self.stage_name = "primary"
        if config.enabled and config.resolver_checkpoint:
            payload = torch.load(config.resolver_checkpoint, map_location="cpu")
            arch = str(payload.get("arch", "v1"))
            self.arch = arch
            input_dim = int(payload.get("input_dim", len(EDGE_FEATURE_NAMES)))
            hidden_dim = int(payload.get("hidden_dim", 64))
            stage_embed_dim = int(payload.get("stage_embed_dim", 8))
            self.feature_names = list(payload.get("feature_names", self._default_feature_names(input_dim)))
            self.row_feature_names = list(payload.get("row_feature_names", ROW_CONTEXT_FEATURE_NAMES))
            self.col_feature_names = list(payload.get("col_feature_names", COL_CONTEXT_FEATURE_NAMES))
            if arch == "v3_trackquery":
                model = FGASAssociationResolverV3TrackQuery(
                    input_dim=input_dim,
                    row_context_dim=int(payload.get("row_context_dim", len(self.row_feature_names))),
                    col_context_dim=int(payload.get("col_context_dim", len(self.col_feature_names))),
                    hidden_dim=hidden_dim,
                    stage_embed_dim=stage_embed_dim,
                    num_stages=len(STAGE_NAME_TO_ID),
                    num_heads=int(payload.get("num_heads", 4)),
                    num_layers=int(payload.get("num_attn_layers", 2)),
                )
            elif arch == "v2_trackdet":
                model = FGASAssociationResolverV2(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    stage_embed_dim=stage_embed_dim,
                    num_stages=len(STAGE_NAME_TO_ID),
                    num_heads=int(payload.get("num_heads", 4)),
                    num_layers=int(payload.get("num_attn_layers", 2)),
                )
            else:
                model = FGASBlockResolver(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    stage_embed_dim=stage_embed_dim,
                    num_stages=len(STAGE_NAME_TO_ID),
                )
            model.load_state_dict(payload["model_state"])
            model.eval()
            model.to(self.device)
            self.model = model
        if config.block_primitive_checkpoint:
            primitive_payload = torch.load(config.block_primitive_checkpoint, map_location="cpu")
            primitive_model = FGASAmbiguityBlockPrimitive(
                input_dim=int(primitive_payload.get("input_dim", len(EDGE_FEATURE_NAMES))),
                row_context_dim=int(primitive_payload.get("row_context_dim", len(ROW_CONTEXT_FEATURE_NAMES))),
                col_context_dim=int(primitive_payload.get("col_context_dim", len(COL_CONTEXT_FEATURE_NAMES))),
                hidden_dim=int(primitive_payload.get("hidden_dim", 128)),
                stage_embed_dim=int(primitive_payload.get("stage_embed_dim", 16)),
                num_stages=int(primitive_payload.get("num_stages", len(STAGE_NAME_TO_ID))),
                num_heads=int(primitive_payload.get("num_heads", 4)),
                num_layers=int(primitive_payload.get("num_layers", 2)),
                dropout=float(primitive_payload.get("dropout", 0.0)),
            )
            primitive_model.load_state_dict(primitive_payload["model_state"])
            primitive_model.eval()
            primitive_model.to(self.device)
            self.block_primitive = primitive_model
            self.block_primitive_feature_names = list(primitive_payload.get("feature_names", EDGE_FEATURE_NAMES))
            self.feature_names = list(self.block_primitive_feature_names)
            self.row_feature_names = list(primitive_payload.get("row_feature_names", ROW_CONTEXT_FEATURE_NAMES))
            self.col_feature_names = list(primitive_payload.get("col_feature_names", COL_CONTEXT_FEATURE_NAMES))
        if config.block_matcher_checkpoint:
            matcher_payload = torch.load(config.block_matcher_checkpoint, map_location="cpu")
            matcher_base_logit_scale = float(matcher_payload.get("base_logit_scale", 0.35))
            if config.block_matcher_base_logit_scale_override is not None:
                matcher_base_logit_scale = float(config.block_matcher_base_logit_scale_override)
            matcher_feature_names = list(matcher_payload.get("feature_names", EDGE_FEATURE_NAMES))
            matcher_row_feature_names = list(matcher_payload.get("row_feature_names", ROW_CONTEXT_FEATURE_NAMES))
            matcher_col_feature_names = list(matcher_payload.get("col_feature_names", COL_CONTEXT_FEATURE_NAMES))
            edge_aux_feature_names = [str(name) for name in matcher_payload.get("edge_aux_feature_names", [])]
            row_aux_feature_names = [str(name) for name in matcher_payload.get("row_aux_feature_names", [])]
            col_aux_feature_names = [str(name) for name in matcher_payload.get("col_aux_feature_names", [])]
            matcher_model = FGASTrueBlockMatcher(
                input_dim=int(matcher_payload.get("input_dim", len(EDGE_FEATURE_NAMES))),
                row_context_dim=int(matcher_payload.get("row_context_dim", len(matcher_row_feature_names))),
                col_context_dim=int(matcher_payload.get("col_context_dim", len(matcher_col_feature_names))),
                hidden_dim=int(matcher_payload.get("hidden_dim", 128)),
                stage_embed_dim=int(matcher_payload.get("stage_embed_dim", 16)),
                num_stages=int(matcher_payload.get("num_stages", len(STAGE_NAME_TO_ID))),
                num_heads=int(matcher_payload.get("num_heads", 4)),
                num_layers=int(matcher_payload.get("num_layers", 2)),
                dropout=float(matcher_payload.get("dropout", 0.0)),
                use_base_residual=bool(matcher_payload.get("use_base_residual", False)),
                base_score_index=int(matcher_payload.get("base_score_index", -1)),
                base_logit_scale=matcher_base_logit_scale,
                edge_aux_indices=[
                    int(matcher_feature_names.index(name)) for name in edge_aux_feature_names if name in matcher_feature_names
                ],
                row_aux_indices=[
                    int(matcher_row_feature_names.index(name)) for name in row_aux_feature_names if name in matcher_row_feature_names
                ],
                col_aux_indices=[
                    int(matcher_col_feature_names.index(name)) for name in col_aux_feature_names if name in matcher_col_feature_names
                ],
                side_init_scale=float(matcher_payload.get("side_init_scale", 0.1)),
            )
            matcher_model.load_state_dict(matcher_payload["model_state"])
            matcher_model.eval()
            matcher_model.to(self.device)
            self.block_matcher = matcher_model
            self.block_matcher_feature_names = matcher_feature_names
            self.feature_names = list(self.block_matcher_feature_names)
            self.row_feature_names = matcher_row_feature_names
            self.col_feature_names = matcher_col_feature_names
            if self.block_matcher_margin_thresh <= 0.0:
                self.block_matcher_margin_thresh = float(matcher_payload.get("block_margin", 0.0))
        if config.pair_scorer_checkpoint:
            pair_payload = torch.load(config.pair_scorer_checkpoint, map_location="cpu")
            pair_model = FGASPairScorer(
                input_dim=int(pair_payload.get("input_dim", 0)),
                hidden_dim=int(pair_payload.get("hidden_dim", 64)),
                dropout=float(pair_payload.get("dropout", 0.0)),
            )
            pair_model.load_state_dict(pair_payload["model_state"])
            pair_model.eval()
            pair_model.to(self.device)
            self.pair_scorer = pair_model
            self.pair_edge_feature_names = list(pair_payload.get("edge_feature_names", []))
            self.pair_row_feature_names = list(pair_payload.get("row_feature_names", []))
            self.pair_col_feature_names = list(pair_payload.get("col_feature_names", []))
            self.pair_domain_feature_names = list(pair_payload.get("domain_feature_names", DOMAIN_FEATURE_NAMES))
        if config.acceptance_gate_checkpoint:
            gate_payload = torch.load(config.acceptance_gate_checkpoint, map_location="cpu")
            gate_model = FGASAcceptanceGate(
                input_dim=int(gate_payload.get("input_dim", 0)),
                hidden_dim=int(gate_payload.get("hidden_dim", 32)),
                dropout=float(gate_payload.get("dropout", 0.0)),
            )
            gate_model.load_state_dict(gate_payload["model_state"])
            gate_model.eval()
            gate_model.to(self.device)
            self.acceptance_gate = gate_model
            self.acceptance_gate_feature_names = list(gate_payload.get("feature_names", []))
            if float(self.config.acceptance_gate_thresh) <= 0.0:
                learned_gate_thresh = float(
                    gate_payload.get("acceptance_gate_thresh", gate_payload.get("best_threshold", 0.5)) or 0.5
                )
                self.config.acceptance_gate_thresh = learned_gate_thresh
        if config.block_gate_checkpoint:
            block_gate_payload = torch.load(config.block_gate_checkpoint, map_location="cpu")
            block_gate_model = FGASAcceptanceGate(
                input_dim=int(block_gate_payload.get("input_dim", 0)),
                hidden_dim=int(block_gate_payload.get("hidden_dim", 32)),
                dropout=float(block_gate_payload.get("dropout", 0.0)),
            )
            block_gate_model.load_state_dict(block_gate_payload["model_state"])
            block_gate_model.eval()
            block_gate_model.to(self.device)
            self.block_gate = block_gate_model
            self.block_gate_feature_names = list(block_gate_payload.get("feature_names", BLOCK_GATE_FEATURE_NAMES))

    def _default_feature_names(self, input_dim: int) -> List[str]:
        if int(input_dim) == len(EDGE_FEATURE_NAMES):
            return list(EDGE_FEATURE_NAMES)
        if int(input_dim) == len(EDGE_FEATURE_NAMES) - 3:
            return [name for name in EDGE_FEATURE_NAMES if name not in {"s_low", "s_mid", "s_high"}]
        if int(input_dim) == 19:
            return [
                "s_reid",
                "s_low",
                "s_mid",
                "s_high",
                "base_similarity",
                "det_score",
                "track_age",
                "raw_iou_similarity",
                "fused_iou_similarity",
                "track_width",
                "track_height",
                "det_width",
                "det_height",
                "track_aspect",
                "det_aspect",
                "center_dx",
                "center_dy",
                "area_ratio",
                "candidate_rank_norm",
            ]
        if int(input_dim) == 16:
            return [
                "s_reid",
                "base_similarity",
                "det_score",
                "track_age",
                "raw_iou_similarity",
                "fused_iou_similarity",
                "track_width",
                "track_height",
                "det_width",
                "det_height",
                "track_aspect",
                "det_aspect",
                "center_dx",
                "center_dy",
                "area_ratio",
                "candidate_rank_norm",
            ]
        return list(EDGE_FEATURE_NAMES[: int(input_dim)])

    def is_active(self) -> bool:
        return bool(
            self.config.enabled
            and (
                self.model is not None
                or self.block_primitive is not None
                or self.block_matcher is not None
                or self.pair_scorer is not None
            )
        )

    def uses_frequency(self) -> bool:
        runtime_feature_names = list(self.feature_names)
        if self.block_matcher is not None:
            runtime_feature_names = list(self.block_matcher_feature_names or self.feature_names)
        elif self.block_primitive is not None:
            runtime_feature_names = list(self.block_primitive_feature_names or self.feature_names)
        elif self.pair_scorer is not None:
            runtime_feature_names = list(self.pair_edge_feature_names)
        return any(name in {"s_low", "s_mid", "s_high"} for name in runtime_feature_names)

    def has_acceptance_gate(self) -> bool:
        return self.acceptance_gate is not None and len(self.acceptance_gate_feature_names) > 0

    def has_pair_scorer(self) -> bool:
        return self.pair_scorer is not None

    def has_block_matcher(self) -> bool:
        return self.block_matcher is not None

    def has_block_primitive(self) -> bool:
        return self.block_primitive is not None

    def has_block_gate(self) -> bool:
        return self.block_gate is not None and len(self.block_gate_feature_names) > 0

    def extract_detection_descriptors(self, image: np.ndarray, detections: Sequence[object]) -> List[BandDescriptor]:
        return [
            extract_band_descriptor(
                image,
                np.asarray(det.tlbr, dtype=np.float32),
                image_height=self.config.crop_height,
                image_width=self.config.crop_width,
            )
            for det in detections
        ]

    def _connected_components(self, row_candidates: Dict[int, List[int]]) -> List[Tuple[List[int], List[int]]]:
        row_to_cols = {int(k): set(map(int, v)) for k, v in row_candidates.items() if v}
        col_to_rows: Dict[int, Set[int]] = defaultdict(set)
        for row_idx, col_list in row_to_cols.items():
            for col_idx in col_list:
                col_to_rows[int(col_idx)].add(int(row_idx))
        visited_rows: Set[int] = set()
        visited_cols: Set[int] = set()
        components: List[Tuple[List[int], List[int]]] = []
        for start_row in sorted(row_to_cols.keys()):
            if start_row in visited_rows:
                continue
            comp_rows: Set[int] = set()
            comp_cols: Set[int] = set()
            queue: deque[Tuple[str, int]] = deque([("row", int(start_row))])
            while queue:
                kind, idx = queue.popleft()
                if kind == "row":
                    if idx in visited_rows:
                        continue
                    visited_rows.add(idx)
                    comp_rows.add(idx)
                    for col_idx in row_to_cols.get(idx, set()):
                        if col_idx not in visited_cols:
                            queue.append(("col", int(col_idx)))
                else:
                    if idx in visited_cols:
                        continue
                    visited_cols.add(idx)
                    comp_cols.add(idx)
                    for row_idx in col_to_rows.get(idx, set()):
                        if row_idx not in visited_rows:
                            queue.append(("row", int(row_idx)))
            if comp_rows and comp_cols:
                components.append((sorted(comp_rows), sorted(comp_cols)))
        return components

    def _build_feature_vector(
        self,
        *,
        track: object,
        detection: object,
        det_desc: Optional[BandDescriptor],
        s_reid: float,
        base_similarity: float,
        raw_iou_cost: float,
        fused_iou_cost: float,
        candidate_rank: int,
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[float]:
        s_low = 0.0
        s_mid = 0.0
        s_high = 0.0
        if det_desc is not None and getattr(track, "fcaa_low", None) is not None:
            s_low = cosine_similarity(np.asarray(track.fcaa_low), det_desc.low)
            s_mid = cosine_similarity(np.asarray(track.fcaa_mid), det_desc.mid)
            s_high = cosine_similarity(np.asarray(track.fcaa_high), det_desc.high)
        # Keep runtime aligned with blockbank training: "track_age" means staleness
        # (frames since last update), not track persistence length.
        track_age = float(getattr(track, "tracklet_len", 0.0))
        if hasattr(track, "time_since_update") and hasattr(track, "hit_streak"):
            track_age = float(getattr(track, "time_since_update", 0.0))
        all_features = build_edge_feature_vector(
            s_reid=s_reid,
            s_low=s_low,
            s_mid=s_mid,
            s_high=s_high,
            base_similarity=base_similarity,
            det_score=float(getattr(detection, "score", 0.0)),
            track_age=track_age,
            raw_iou_cost=raw_iou_cost,
            fused_iou_cost=fused_iou_cost,
            track_box=[float(v) for v in np.asarray(track.tlbr, dtype=np.float32).tolist()],
            det_box=[float(v) for v in np.asarray(detection.tlbr, dtype=np.float32).tolist()],
            candidate_rank=float(candidate_rank),
            top_k=float(self.config.top_k),
        )
        feature_map = dict(zip(EDGE_FEATURE_NAMES, all_features))
        ordered_feature_names = list(feature_names) if feature_names is not None else list(self.feature_names)
        return [float(feature_map[name]) for name in ordered_feature_names]

    def _build_candidate_payload(
        self,
        *,
        track: object,
        detection: object,
        det_desc: Optional[BandDescriptor],
        s_reid: float,
        base_similarity: float,
        raw_iou_cost: float,
        fused_iou_cost: float,
        candidate_rank: int,
        ambiguous_flag: float,
    ) -> Dict[str, object]:
        track_age = float(getattr(track, "tracklet_len", 0.0))
        if hasattr(track, "time_since_update") and hasattr(track, "hit_streak"):
            track_age = float(getattr(track, "time_since_update", 0.0))
        return {
            "s_reid": float(s_reid),
            "s_low": float(cosine_similarity(np.asarray(track.fcaa_low), det_desc.low)) if det_desc is not None and getattr(track, "fcaa_low", None) is not None else 0.0,
            "s_mid": float(cosine_similarity(np.asarray(track.fcaa_mid), det_desc.mid)) if det_desc is not None and getattr(track, "fcaa_mid", None) is not None else 0.0,
            "s_high": float(cosine_similarity(np.asarray(track.fcaa_high), det_desc.high)) if det_desc is not None and getattr(track, "fcaa_high", None) is not None else 0.0,
            "base_similarity": float(base_similarity),
            "det_score": float(getattr(detection, "score", 0.0)),
            "track_age": track_age,
            "raw_iou_cost": float(raw_iou_cost),
            "fused_iou_cost": float(fused_iou_cost),
            "track_box": [float(v) for v in np.asarray(track.tlbr, dtype=np.float32).tolist()],
            "det_box": [float(v) for v in np.asarray(detection.tlbr, dtype=np.float32).tolist()],
            "candidate_rank": float(candidate_rank),
            "ambiguous_flag": float(ambiguous_flag),
        }

    def _build_pair_feature_vector(
        self,
        *,
        edge_feature_map: Dict[str, float],
        row_feature_map: Dict[str, float],
        col_feature_map: Dict[str, float],
        seq_name: str,
    ) -> List[float]:
        if self.pair_scorer is None:
            return []
        domain_feature_map = {
            name: float(value)
            for name, value in zip(DOMAIN_FEATURE_NAMES, build_domain_feature_vector(seq_name))
        }
        values: List[float] = []
        values.extend(float(edge_feature_map.get(name, 0.0)) for name in self.pair_edge_feature_names)
        values.extend(float(row_feature_map.get(name, 0.0)) for name in self.pair_row_feature_names)
        values.extend(float(col_feature_map.get(name, 0.0)) for name in self.pair_col_feature_names)
        values.extend(float(domain_feature_map.get(name, 0.0)) for name in self.pair_domain_feature_names)
        return values

    def _ordered_row_context(self, rows: Sequence[Dict[str, object]], *, candidate_limit: float) -> List[float]:
        values = build_row_context_from_rows(rows, candidate_limit=candidate_limit)
        feature_map = dict(zip(ROW_CONTEXT_FEATURE_NAMES, values))
        return [float(feature_map.get(name, 0.0)) for name in self.row_feature_names]

    def _ordered_col_context(self, rows: Sequence[Dict[str, object]], *, candidate_limit: float) -> List[float]:
        values = build_col_context_from_rows(rows, candidate_limit=candidate_limit)
        feature_map = dict(zip(COL_CONTEXT_FEATURE_NAMES, values))
        return [float(feature_map.get(name, 0.0)) for name in self.col_feature_names]

    def _empty_actions(self) -> FGASControllerActions:
        return FGASControllerActions(forced_matches=[], blocked_rows=[], blocked_cols=[])

    def _row_top1_and_margin(self, matrix: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row_best = np.full((matrix.shape[0],), fill_value=-1, dtype=int)
        row_margin = np.zeros((matrix.shape[0],), dtype=np.float32)
        for row_idx in range(matrix.shape[0]):
            valid_cols = np.where(valid_mask[row_idx])[0]
            if valid_cols.size == 0:
                continue
            row_scores = np.asarray(matrix[row_idx, valid_cols], dtype=np.float32)
            order = np.argsort(row_scores)[::-1]
            best_local = int(valid_cols[order[0]])
            row_best[row_idx] = best_local
            best_score = float(row_scores[order[0]])
            second_score = float(row_scores[order[1]]) if order.size > 1 else 0.0
            row_margin[row_idx] = float(best_score - second_score)
        return row_best, row_margin

    def _build_frontier_col_mask(
        self,
        *,
        base_best: np.ndarray,
        refined_best: np.ndarray,
        row_changed_mask: np.ndarray,
        col_count: int,
    ) -> np.ndarray:
        frontier_col_mask = np.zeros((int(col_count),), dtype=bool)
        if not np.any(row_changed_mask):
            return frontier_col_mask
        changed_base_cols = base_best[row_changed_mask]
        changed_refined_cols = refined_best[row_changed_mask]
        frontier_cols: List[int] = []
        frontier_cols.extend(int(col_idx) for col_idx in changed_base_cols.tolist() if int(col_idx) >= 0)
        frontier_cols.extend(int(col_idx) for col_idx in changed_refined_cols.tolist() if int(col_idx) >= 0)
        if frontier_cols:
            frontier_col_mask[np.asarray(frontier_cols, dtype=int)] = True
        return frontier_col_mask

    def _solution_best_and_changed_rows(
        self,
        *,
        valid_edge_mask: np.ndarray,
        row_assignment: np.ndarray,
        row_no_match: np.ndarray,
        base_best: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        solution_best = np.full((int(valid_edge_mask.shape[0]),), fill_value=-1, dtype=int)
        for local_r in range(int(valid_edge_mask.shape[0])):
            if not bool(valid_edge_mask[local_r].any()):
                continue
            if bool(row_no_match[local_r]):
                continue
            local_c = int(row_assignment[local_r])
            if local_c < 0 or local_c >= int(valid_edge_mask.shape[1]):
                continue
            if not bool(valid_edge_mask[local_r, local_c]):
                continue
            solution_best[local_r] = local_c
        changed_row_mask = (base_best >= 0) & (solution_best != base_best)
        return solution_best, changed_row_mask

    def _solution_row_margin(
        self,
        *,
        matrix: np.ndarray,
        valid_mask: np.ndarray,
        solution_best: np.ndarray,
        row_no_match: np.ndarray,
        row_no_match_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        row_margin = np.zeros((int(valid_mask.shape[0]),), dtype=np.float32)
        if row_no_match_scores is None:
            row_no_match_scores = np.zeros((int(valid_mask.shape[0]),), dtype=np.float32)
        for row_idx in range(int(valid_mask.shape[0])):
            valid_cols = np.where(valid_mask[row_idx])[0]
            if valid_cols.size == 0:
                continue
            if bool(row_no_match[row_idx]):
                best_edge_score = float(np.max(matrix[row_idx, valid_cols]))
                row_margin[row_idx] = float(row_no_match_scores[row_idx] - best_edge_score)
                continue
            local_c = int(solution_best[row_idx])
            if local_c < 0 or local_c >= int(valid_mask.shape[1]) or not bool(valid_mask[row_idx, local_c]):
                continue
            solution_score = float(matrix[row_idx, local_c])
            alt_scores = [float(matrix[row_idx, alt_c]) for alt_c in valid_cols.tolist() if int(alt_c) != local_c]
            best_alt_score = max(alt_scores) if alt_scores else 0.0
            best_alt_score = max(best_alt_score, float(row_no_match_scores[row_idx]))
            row_margin[row_idx] = float(solution_score - best_alt_score)
        return row_margin

    def _compute_changed_row_state(
        self,
        *,
        valid_edge_mask: np.ndarray,
        base_best: np.ndarray,
        base_row_margin: np.ndarray,
        probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        refined_best, refined_row_margin = self._row_top1_and_margin(probs, valid_edge_mask)
        row_changed_mask = (base_best >= 0) & (refined_best >= 0) & (base_best != refined_best)
        if float(self.config.soft_row_base_margin_thresh) < 1.0:
            row_changed_mask = row_changed_mask & (base_row_margin <= float(self.config.soft_row_base_margin_thresh))
        if float(self.config.soft_changed_row_refined_margin_thresh) > 0.0:
            row_changed_mask = row_changed_mask & (
                refined_row_margin >= float(self.config.soft_changed_row_refined_margin_thresh)
            )
        if float(self.config.soft_changed_row_flip_gap_thresh) > 0.0 and np.any(row_changed_mask):
            gated_changed_mask = row_changed_mask.copy()
            changed_local_rows = np.where(row_changed_mask)[0]
            for local_r in changed_local_rows.tolist():
                base_local_c = int(base_best[local_r])
                refined_local_c = int(refined_best[local_r])
                if base_local_c < 0 or refined_local_c < 0:
                    gated_changed_mask[local_r] = False
                    continue
                flip_gap = float(probs[local_r, refined_local_c] - probs[local_r, base_local_c])
                if flip_gap < float(self.config.soft_changed_row_flip_gap_thresh):
                    gated_changed_mask[local_r] = False
            row_changed_mask = gated_changed_mask
        frontier_col_mask = self._build_frontier_col_mask(
            base_best=base_best,
            refined_best=refined_best,
            row_changed_mask=row_changed_mask,
            col_count=int(valid_edge_mask.shape[1]),
        )
        return refined_best, refined_row_margin, row_changed_mask, frontier_col_mask, bool(np.any(row_changed_mask))

    def _apply_acceptance_gate_mask(
        self,
        *,
        edge_feature_names: Sequence[str],
        row_feature_names: Sequence[str],
        col_feature_names: Sequence[str],
        edge_features: np.ndarray,
        row_features: np.ndarray,
        col_features: np.ndarray,
        valid_mask: np.ndarray,
        probs: np.ndarray,
        row_nomatch_probs: np.ndarray,
        base_best: np.ndarray,
        refined_best: np.ndarray,
        base_row_margin: np.ndarray,
        refined_row_margin: np.ndarray,
        row_changed_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_acceptance_gate() or not np.any(row_changed_mask):
            frontier_col_mask = self._build_frontier_col_mask(
                base_best=base_best,
                refined_best=refined_best,
                row_changed_mask=row_changed_mask,
                col_count=int(valid_mask.shape[1]),
            )
            return row_changed_mask, frontier_col_mask
        gated_mask = row_changed_mask.copy()
        changed_local_rows = np.where(row_changed_mask)[0]
        for local_r in changed_local_rows.tolist():
            acceptance_features = build_acceptance_feature_vector(
                edge_feature_names=edge_feature_names,
                row_feature_names=row_feature_names,
                col_feature_names=col_feature_names,
                edge_features=edge_features,
                row_features=row_features,
                col_features=col_features,
                valid_mask=valid_mask,
                probs=probs,
                row_nomatch_probs=row_nomatch_probs,
                base_best=base_best,
                refined_best=refined_best,
                base_row_margin=base_row_margin,
                refined_row_margin=refined_row_margin,
                row_idx=int(local_r),
            )
            gate_feature_map = dict(zip(self.acceptance_gate_feature_names, acceptance_features))
            gate_vector = [float(gate_feature_map.get(name, 0.0)) for name in self.acceptance_gate_feature_names]
            gate_input = torch.tensor(gate_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                gate_score = float(torch.sigmoid(self.acceptance_gate(gate_input)).item())
            if gate_score < float(self.config.acceptance_gate_thresh):
                gated_mask[local_r] = False
        frontier_col_mask = self._build_frontier_col_mask(
            base_best=base_best,
            refined_best=refined_best,
            row_changed_mask=gated_mask,
            col_count=int(valid_mask.shape[1]),
        )
        return gated_mask, frontier_col_mask

    def _derive_component_controller_actions(
        self,
        *,
        comp_rows: Sequence[int],
        comp_cols: Sequence[int],
        edge_mask: np.ndarray,
        base_similarity: np.ndarray,
        probs: np.ndarray,
        row_nomatch_probs: np.ndarray,
        col_newborn_probs: Optional[np.ndarray],
    ) -> Tuple[FGASControllerActions, Dict[str, int]]:
        margin_thresh = float(self.config.controller_margin_thresh)
        ambiguity_margin = float(self.config.controller_ambiguity_margin)
        edge_thresh = float(self.config.controller_edge_thresh)
        row_defer_thresh = float(self.config.controller_row_defer_thresh)
        col_newborn_thresh = float(self.config.controller_col_newborn_thresh)
        forced_candidates: List[Tuple[int, int, float]] = []
        blocked_rows: Set[int] = set()
        blocked_cols: Set[int] = set()
        row_is_ambiguous: Dict[int, bool] = {}
        row_best_local_col: Dict[int, int] = {}
        col_top1_count: Dict[int, int] = defaultdict(int)
        col_best_local_row: Dict[int, int] = {}
        col_best_score: Dict[int, float] = {}
        col_second_score: Dict[int, float] = {}
        forced_rejected_not_mutual = 0
        forced_rejected_not_base_top1 = 0

        if col_newborn_probs is None:
            col_newborn_probs = np.zeros((len(comp_cols),), dtype=np.float32)

        for local_r, row_idx in enumerate(comp_rows):
            valid_cols = np.where(edge_mask[local_r])[0]
            if valid_cols.size == 0:
                row_is_ambiguous[int(row_idx)] = False
                continue
            base_scores = base_similarity[local_r, valid_cols]
            order = valid_cols[np.argsort(base_scores)[::-1]]
            best_local_c = int(order[0])
            row_best_local_col[int(row_idx)] = best_local_c
            col_top1_count[best_local_c] = int(col_top1_count.get(best_local_c, 0)) + 1
            base_best = float(base_similarity[local_r, best_local_c])
            base_second = float(base_similarity[local_r, order[1]]) if order.size > 1 else 0.0
            row_is_ambiguous[int(row_idx)] = bool(order.size > 1 and (base_best - base_second) < ambiguity_margin)
        for row_idx, best_local_c in row_best_local_col.items():
            if int(col_top1_count.get(best_local_c, 0)) > 1:
                row_is_ambiguous[int(row_idx)] = True
        ambiguous_cols = {int(local_c) for row_idx, local_c in row_best_local_col.items() if row_is_ambiguous.get(int(row_idx), False)}

        for local_c, _col_idx in enumerate(comp_cols):
            valid_rows = np.where(edge_mask[:, local_c])[0]
            if valid_rows.size == 0:
                continue
            local_scores = probs[valid_rows, local_c]
            order = valid_rows[np.argsort(local_scores)[::-1]]
            best_local_r = int(order[0])
            col_best_local_row[local_c] = best_local_r
            col_best_score[local_c] = float(probs[best_local_r, local_c])
            col_second_score[local_c] = float(probs[order[1], local_c]) if order.size > 1 else 0.0

        for local_r, row_idx in enumerate(comp_rows):
            if not row_is_ambiguous.get(int(row_idx), False):
                continue
            valid_cols = np.where(edge_mask[local_r])[0]
            if valid_cols.size == 0:
                continue
            local_scores = probs[local_r, valid_cols]
            order = valid_cols[np.argsort(local_scores)[::-1]]
            best_local_c = int(order[0])
            best_score = float(probs[local_r, best_local_c])
            second_score = float(probs[local_r, order[1]]) if order.size > 1 else 0.0
            row_nomatch = float(row_nomatch_probs[local_r])
            if row_nomatch >= row_defer_thresh and (row_nomatch - best_score) >= margin_thresh:
                blocked_rows.add(int(row_idx))
                continue
            if best_score >= edge_thresh and (best_score - max(second_score, row_nomatch)) >= margin_thresh:
                if bool(self.config.controller_require_base_top1) and row_best_local_col.get(int(row_idx)) != best_local_c:
                    forced_rejected_not_base_top1 += 1
                    continue
                if bool(self.config.controller_mutual_top1_only):
                    if col_best_local_row.get(best_local_c) != local_r:
                        forced_rejected_not_mutual += 1
                        continue
                    col_margin = float(col_best_score.get(best_local_c, 0.0)) - max(
                        float(col_second_score.get(best_local_c, 0.0)),
                        float(col_newborn_probs[best_local_c]),
                    )
                    if col_margin < margin_thresh:
                        forced_rejected_not_mutual += 1
                        continue
                forced_candidates.append((int(row_idx), int(comp_cols[best_local_c]), best_score))

        for local_c, col_idx in enumerate(comp_cols):
            if local_c not in ambiguous_cols:
                continue
            valid_rows = np.where(edge_mask[:, local_c])[0]
            if valid_rows.size == 0:
                continue
            top_match_score = float(probs[valid_rows, local_c].max())
            col_newborn = float(col_newborn_probs[local_c])
            if col_newborn >= col_newborn_thresh and (col_newborn - top_match_score) >= margin_thresh:
                blocked_cols.add(int(col_idx))

        forced_matches: List[Tuple[int, int]] = []
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        for row_idx, col_idx, _score in sorted(forced_candidates, key=lambda item: item[2], reverse=True):
            if row_idx in blocked_rows or row_idx in used_rows or col_idx in used_cols:
                continue
            forced_matches.append((int(row_idx), int(col_idx)))
            used_rows.add(int(row_idx))
            used_cols.add(int(col_idx))

        blocked_rows.difference_update(used_rows)
        blocked_cols.difference_update(used_cols)
        debug = {
            "forced_candidates": int(len(forced_candidates)),
            "forced_matches": int(len(forced_matches)),
            "blocked_rows": int(len(blocked_rows)),
            "blocked_cols": int(len(blocked_cols)),
            "conflicts_dropped": int(max(0, len(forced_candidates) - len(forced_matches))),
            "controller_forced_rejected_not_mutual": int(forced_rejected_not_mutual),
            "controller_forced_rejected_not_base_top1": int(forced_rejected_not_base_top1),
        }
        return (
            FGASControllerActions(
                forced_matches=forced_matches,
                blocked_rows=sorted(blocked_rows),
                blocked_cols=sorted(blocked_cols),
            ),
            debug,
        )

    def _derive_primitive_controller_actions(
        self,
        *,
        comp_rows: Sequence[int],
        comp_cols: Sequence[int],
        valid_edge_mask: np.ndarray,
        edge_probs: np.ndarray,
        row_assignment: np.ndarray,
        row_no_match: np.ndarray,
        col_newborn: np.ndarray,
    ) -> Tuple[FGASControllerActions, Dict[str, int]]:
        forced_candidates: List[Tuple[int, int, float]] = []
        blocked_rows: Set[int] = set()
        blocked_cols: Set[int] = set()
        for local_r, row_idx in enumerate(comp_rows):
            if not bool(valid_edge_mask[local_r].any()):
                continue
            if bool(row_no_match[local_r]):
                blocked_rows.add(int(row_idx))
                continue
            local_c = int(row_assignment[local_r])
            if local_c < 0 or local_c >= len(comp_cols):
                continue
            if not bool(valid_edge_mask[local_r, local_c]):
                continue
            forced_candidates.append((int(row_idx), int(comp_cols[local_c]), float(edge_probs[local_r, local_c])))

        forced_matches: List[Tuple[int, int]] = []
        used_rows: Set[int] = set()
        used_cols: Set[int] = set()
        for row_idx, col_idx, _score in sorted(forced_candidates, key=lambda item: item[2], reverse=True):
            if row_idx in used_rows or col_idx in used_cols:
                continue
            forced_matches.append((int(row_idx), int(col_idx)))
            used_rows.add(int(row_idx))
            used_cols.add(int(col_idx))

        for local_c, col_idx in enumerate(comp_cols):
            if bool(col_newborn[local_c]) and int(col_idx) not in used_cols:
                blocked_cols.add(int(col_idx))

        blocked_rows.difference_update(used_rows)
        blocked_cols.difference_update(used_cols)
        debug = {
            "forced_candidates": int(len(forced_candidates)),
            "forced_matches": int(len(forced_matches)),
            "blocked_rows": int(len(blocked_rows)),
            "blocked_cols": int(len(blocked_cols)),
            "conflicts_dropped": int(max(0, len(forced_candidates) - len(forced_matches))),
        }
        return (
            FGASControllerActions(
                forced_matches=forced_matches,
                blocked_rows=sorted(blocked_rows),
                blocked_cols=sorted(blocked_cols),
            ),
            debug,
        )

    def _matcher_stale_bias_rows(
        self,
        *,
        track_pool: Sequence[object],
        comp_rows: Sequence[int],
        comp_cols: Sequence[int],
        valid_edge_mask: np.ndarray,
    ) -> np.ndarray:
        bias = float(self.config.block_matcher_stale_match_bias)
        if bias <= 0.0:
            return np.zeros((int(len(comp_rows)),), dtype=bool)
        if (
            int(self.config.block_matcher_stale_match_max_component_rows) > 0
            and int(len(comp_rows)) > int(self.config.block_matcher_stale_match_max_component_rows)
        ):
            return np.zeros((int(len(comp_rows)),), dtype=bool)
        if (
            int(self.config.block_matcher_stale_match_max_component_cols) > 0
            and int(len(comp_cols)) > int(self.config.block_matcher_stale_match_max_component_cols)
        ):
            return np.zeros((int(len(comp_rows)),), dtype=bool)

        biased_rows = np.zeros((int(len(comp_rows)),), dtype=bool)
        min_time_since_update = int(self.config.block_matcher_stale_match_min_time_since_update)
        max_hit_streak = int(self.config.block_matcher_stale_match_max_hit_streak)
        min_hits = int(self.config.block_matcher_stale_match_min_hits)
        for local_r, row_idx in enumerate(comp_rows):
            if not bool(valid_edge_mask[local_r].any()):
                continue
            track = track_pool[row_idx]
            time_since_update = int(getattr(track, "time_since_update", -1))
            hit_streak = int(getattr(track, "hit_streak", 0))
            hits = int(getattr(track, "hits", 0))
            if time_since_update < min_time_since_update:
                continue
            if hit_streak > max_hit_streak:
                continue
            if hits < min_hits:
                continue
            biased_rows[local_r] = True
        return biased_rows

    def refine_primary_cost(
        self,
        *,
        track_pool: Sequence[object],
        detections: Sequence[object],
        emb_dists: np.ndarray,
        raw_ious_dists: np.ndarray,
        image: np.ndarray,
        seq_name: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], FGASControllerActions]:
        debug: Dict[str, object] = {
            "trigger_blocks": 0,
            "changed_blocks": 0,
            "block_gate_blocks": 0,
            "block_gate_pass_blocks": 0,
            "block_gate_filtered_blocks": 0,
            "controller_skipped_unchanged_blocks": 0,
            "soft_applied_blocks": 0,
            "rows_touched": 0,
            "edges_touched": 0,
            "row_nomatch_rows": 0,
            "controller_blocks": 0,
            "controller_forced_candidates": 0,
            "controller_forced_matches": 0,
            "controller_blocked_rows": 0,
            "controller_blocked_cols": 0,
            "controller_conflicts_dropped": 0,
            "controller_forced_rejected_not_mutual": 0,
            "controller_forced_rejected_not_base_top1": 0,
            "pair_mode_blocks": 0,
            "pair_mode_rows": 0,
            "pair_mode_edges": 0,
            "primitive_blocks": 0,
            "primitive_takeover_blocks": 0,
            "primitive_fallback_blocks": 0,
            "primitive_direct_takeover_blocks": 0,
            "primitive_soft_fallback_blocks": 0,
            "primitive_takeover_pairs": 0,
            "matcher_blocks": 0,
            "matcher_takeover_blocks": 0,
            "matcher_margin_reject_blocks": 0,
            "matcher_base_margin_reject_blocks": 0,
            "matcher_base_margin_reject_rows": 0,
            "matcher_stale_match_bias_blocks": 0,
            "matcher_stale_match_bias_rows": 0,
            "matcher_forceonly_nomatch_baseprotect_rows": 0,
            "matcher_single_row_unchanged_reject_blocks": 0,
            "matcher_direct_takeover_blocks": 0,
            "matcher_takeover_pairs": 0,
            "acceptance_rows": [],
            "matcher_rows": [],
        }
        empty_actions = self._empty_actions()
        if not self.is_active() or emb_dists.size == 0 or len(track_pool) == 0 or len(detections) == 0:
            return emb_dists, np.zeros_like(emb_dists, dtype=bool), debug, empty_actions

        invalid_mask = raw_ious_dists > float(self.config.proximity_thresh)
        fused_iou_cost = raw_ious_dists.copy()
        if len(detections) > 0:
            fused_iou_cost = (1.0 - (1.0 - raw_ious_dists) * np.asarray([float(getattr(det, "score", 0.0)) for det in detections], dtype=np.float32).reshape(1, -1)).astype(np.float32)
        clipped_emb = emb_dists.copy()
        clipped_emb[clipped_emb > float(self.config.appearance_thresh)] = 1.0
        clipped_emb[invalid_mask] = 1.0
        base_cost = np.minimum(fused_iou_cost, clipped_emb)
        base_cost[invalid_mask] = 1.0
        base_similarity = 1.0 - base_cost
        det_descs = self.extract_detection_descriptors(image, detections) if self.uses_frequency() else [None for _ in detections]

        row_orders: Dict[int, List[int]] = {}
        for row_idx, _track in enumerate(track_pool):
            valid_det_idx = np.where(~invalid_mask[row_idx])[0]
            if valid_det_idx.size == 0:
                continue
            order = valid_det_idx[np.argsort(base_cost[row_idx, valid_det_idx])]
            order = order[: max(1, int(self.config.top_k))]
            if order.size < 2:
                continue
            if self.has_pair_scorer() and not self.has_block_matcher():
                row_scores = np.asarray(base_similarity[row_idx, order], dtype=np.float32)
                margin = float(row_scores[0] - row_scores[1]) if row_scores.size > 1 else 0.0
                if margin > float(self.config.pair_ambiguity_margin):
                    continue
            row_orders[int(row_idx)] = [int(v) for v in order.tolist()]

        components = self._connected_components(row_orders)
        refined = emb_dists.copy()
        refined_mask = np.zeros_like(refined, dtype=bool)
        forced_matches: List[Tuple[int, int]] = []
        blocked_rows: Set[int] = set()
        blocked_cols: Set[int] = set()
        stage_ids = None
        if self.model is not None or self.block_primitive is not None or self.block_matcher is not None:
            stage_ids = torch.tensor([STAGE_NAME_TO_ID[self.stage_name]], dtype=torch.long, device=self.device)

        for comp_rows, comp_cols in components:
            if len(comp_rows) == 1 and len(comp_cols) == 1:
                continue
            comp_rows = comp_rows[: int(self.config.max_rows)]
            comp_cols = comp_cols[: int(self.config.max_cols)]
            runtime_feature_names = list(self.feature_names)
            if self.block_matcher is not None and len(self.block_matcher_feature_names) > 0:
                runtime_feature_names = list(self.block_matcher_feature_names)
            elif self.block_primitive is not None and len(self.block_primitive_feature_names) > 0:
                runtime_feature_names = list(self.block_primitive_feature_names)
            row_map = {row_idx: local_idx for local_idx, row_idx in enumerate(comp_rows)}
            col_map = {col_idx: local_idx for local_idx, col_idx in enumerate(comp_cols)}
            edge_features = np.zeros((1, len(comp_rows), len(comp_cols), len(runtime_feature_names)), dtype=np.float32)
            edge_mask = np.zeros((1, len(comp_rows), len(comp_cols)), dtype=bool)
            row_context = np.zeros((1, len(comp_rows), len(self.row_feature_names)), dtype=np.float32)
            col_context = np.zeros((1, len(comp_cols), len(self.col_feature_names)), dtype=np.float32)
            row_payloads: Dict[int, List[Dict[str, object]]] = {row_idx: [] for row_idx in comp_rows}
            col_payloads: Dict[int, List[Dict[str, object]]] = {col_idx: [] for col_idx in comp_cols}
            candidate_payloads: Dict[Tuple[int, int], Dict[str, object]] = {}
            for row_idx in comp_rows:
                row_local_order = [det_idx for det_idx in row_orders.get(row_idx, []) if det_idx in col_map]
                local_base_scores = [float(base_similarity[row_idx, det_idx]) for det_idx in row_local_order]
                local_margin = 0.0
                if len(local_base_scores) >= 2:
                    ordered_scores = sorted(local_base_scores, reverse=True)
                    local_margin = ordered_scores[0] - ordered_scores[1]
                ambiguous_flag = 1.0 if len(row_local_order) >= 2 and local_margin < 0.05 else 0.0
                for rank, det_idx in enumerate(row_orders.get(row_idx, [])):
                    if det_idx not in col_map:
                        continue
                    payload = self._build_candidate_payload(
                        track=track_pool[row_idx],
                        detection=detections[det_idx],
                        det_desc=det_descs[det_idx],
                        s_reid=float(1.0 - emb_dists[row_idx, det_idx]),
                        base_similarity=float(base_similarity[row_idx, det_idx]),
                        raw_iou_cost=float(raw_ious_dists[row_idx, det_idx]),
                        fused_iou_cost=float(fused_iou_cost[row_idx, det_idx]),
                        candidate_rank=int(rank),
                        ambiguous_flag=ambiguous_flag,
                    )
                    edge_mask[0, row_map[row_idx], col_map[det_idx]] = True
                    edge_features[0, row_map[row_idx], col_map[det_idx]] = np.asarray(
                        self._build_feature_vector(
                            track=track_pool[row_idx],
                            detection=detections[det_idx],
                            det_desc=det_descs[det_idx],
                            s_reid=float(1.0 - emb_dists[row_idx, det_idx]),
                            base_similarity=float(base_similarity[row_idx, det_idx]),
                            raw_iou_cost=float(raw_ious_dists[row_idx, det_idx]),
                            fused_iou_cost=float(fused_iou_cost[row_idx, det_idx]),
                            candidate_rank=int(rank),
                            feature_names=runtime_feature_names,
                        ),
                        dtype=np.float32,
                    )
                    row_payloads[row_idx].append(payload)
                    col_payloads[det_idx].append(payload)
                    candidate_payloads[(int(row_idx), int(det_idx))] = dict(payload)
            for row_idx in comp_rows:
                row_context[0, row_map[row_idx]] = np.asarray(
                    self._ordered_row_context(
                        row_payloads[row_idx],
                        candidate_limit=float(max(len(comp_cols), 1)),
                    ),
                    dtype=np.float32,
                )
            for col_idx in comp_cols:
                col_context[0, col_map[col_idx]] = np.asarray(
                    self._ordered_col_context(
                        col_payloads[col_idx],
                        candidate_limit=float(max(len(comp_rows), 1)),
                    ),
                    dtype=np.float32,
                )
            if not edge_mask.any():
                continue
            comp_base_similarity = base_similarity[np.ix_(comp_rows, comp_cols)]
            valid_edge_mask = edge_mask[0]
            if self.block_matcher is not None:
                with torch.no_grad():
                    matcher_output = self.block_matcher(
                        edge_features=torch.tensor(edge_features, dtype=torch.float32, device=self.device),
                        edge_mask=torch.tensor(edge_mask, dtype=torch.bool, device=self.device),
                        stage_ids=stage_ids,
                        row_context=torch.tensor(row_context, dtype=torch.float32, device=self.device),
                        col_context=torch.tensor(col_context, dtype=torch.float32, device=self.device),
                    )
                stale_bias_rows = self._matcher_stale_bias_rows(
                    track_pool=track_pool,
                    comp_rows=comp_rows,
                    comp_cols=comp_cols,
                    valid_edge_mask=valid_edge_mask,
                )
                if np.any(stale_bias_rows):
                    bias_value = float(self.config.block_matcher_stale_match_bias)
                    stale_match_mode = str(getattr(self.config, "block_matcher_stale_match_mode", "all_edges") or "all_edges")
                    adjusted_edge_logits = matcher_output.edge_logits.clone()
                    adjusted_row_nomatch_logits = matcher_output.row_no_match_logits.clone()
                    row_mask = torch.as_tensor(stale_bias_rows, dtype=torch.bool, device=self.device).unsqueeze(0)
                    if stale_match_mode == "all_edges":
                        adjusted_edge_logits = adjusted_edge_logits + row_mask.unsqueeze(-1).to(adjusted_edge_logits.dtype) * bias_value
                    adjusted_row_nomatch_logits[row_mask] = adjusted_row_nomatch_logits[row_mask] - bias_value
                    matcher_output = BlockMatcherOutput(
                        edge_logits=adjusted_edge_logits,
                        row_no_match_logits=adjusted_row_nomatch_logits,
                        col_newborn_logits=matcher_output.col_newborn_logits,
                    )
                    debug["matcher_stale_match_bias_blocks"] = int(debug["matcher_stale_match_bias_blocks"]) + 1
                    debug["matcher_stale_match_bias_rows"] = int(debug["matcher_stale_match_bias_rows"]) + int(
                        np.count_nonzero(stale_bias_rows)
                    )
                else:
                    stale_bias_rows = np.zeros((int(len(comp_rows)),), dtype=bool)
                matcher_probs = torch.sigmoid(matcher_output.edge_logits).cpu().numpy()[0]
                matcher_row_nomatch_probs = torch.sigmoid(matcher_output.row_no_match_logits).cpu().numpy()[0]
                matcher_decision = decode_block_matcher_output(
                    matcher_output,
                    edge_mask=torch.tensor(edge_mask, dtype=torch.bool, device=self.device),
                    margin_threshold=float(self.block_matcher_margin_thresh),
                )
                matcher_takeover = bool(matcher_decision.takeover.cpu().numpy()[0])
                row_assignment = matcher_decision.row_assignment.cpu().numpy()[0]
                row_no_match = matcher_decision.row_no_match.cpu().numpy()[0]
                col_newborn = matcher_decision.col_newborn.cpu().numpy()[0]
                matcher_assignment_margin = float(matcher_decision.assignment_margin.cpu().numpy()[0])
                matcher_objective = float(matcher_decision.objective.cpu().numpy()[0])
                base_best, base_row_margin = self._row_top1_and_margin(comp_base_similarity, valid_edge_mask)
                solution_best, changed_row_mask = self._solution_best_and_changed_rows(
                    valid_edge_mask=valid_edge_mask,
                    row_assignment=row_assignment,
                    row_no_match=row_no_match,
                    base_best=base_best,
                )
                solution_row_margin = self._solution_row_margin(
                    matrix=matcher_probs,
                    valid_mask=valid_edge_mask,
                    solution_best=solution_best,
                    row_no_match=row_no_match,
                    row_no_match_scores=matcher_row_nomatch_probs,
                )
                stable_changed_row_mask = np.zeros_like(changed_row_mask, dtype=bool)
                if self.block_matcher_base_margin_thresh > 0.0:
                    stable_changed_row_mask = changed_row_mask & (
                        base_row_margin > float(self.block_matcher_base_margin_thresh)
                    )
                if bool(self.config.block_matcher_force_only) and bool(
                    self.config.block_matcher_forceonly_keep_base_on_nomatch
                ):
                    forceonly_nomatch_baseprotect_mask = row_no_match & (base_best >= 0)
                    if np.any(forceonly_nomatch_baseprotect_mask):
                        row_assignment = row_assignment.copy()
                        row_no_match = row_no_match.copy()
                        row_assignment[forceonly_nomatch_baseprotect_mask] = base_best[forceonly_nomatch_baseprotect_mask]
                        row_no_match[forceonly_nomatch_baseprotect_mask] = False
                        solution_best, changed_row_mask = self._solution_best_and_changed_rows(
                            valid_edge_mask=valid_edge_mask,
                            row_assignment=row_assignment,
                            row_no_match=row_no_match,
                            base_best=base_best,
                        )
                        solution_row_margin = self._solution_row_margin(
                            matrix=matcher_probs,
                            valid_mask=valid_edge_mask,
                            solution_best=solution_best,
                            row_no_match=row_no_match,
                            row_no_match_scores=matcher_row_nomatch_probs,
                        )
                        if self.block_matcher_base_margin_thresh > 0.0:
                            stable_changed_row_mask = changed_row_mask & (
                                base_row_margin > float(self.block_matcher_base_margin_thresh)
                            )
                        debug["matcher_forceonly_nomatch_baseprotect_rows"] = int(
                            debug["matcher_forceonly_nomatch_baseprotect_rows"]
                        ) + int(np.count_nonzero(forceonly_nomatch_baseprotect_mask))
                if matcher_takeover:
                    matcher_rows = list(debug.get("matcher_rows", []))
                    changed_row_count = int(np.count_nonzero(changed_row_mask))
                    base_margin_rejected = int(np.any(stable_changed_row_mask))
                    takeover_applied = int(not bool(base_margin_rejected))
                    export_local_rows = np.where(valid_edge_mask.any(axis=1))[0]
                    for local_r in export_local_rows.tolist():
                        base_local_c = int(base_best[local_r])
                        refined_local_c = int(solution_best[local_r])
                        row_no_match_flag = int(bool(row_no_match[local_r]))
                        base_prob = float(matcher_probs[local_r, base_local_c]) if base_local_c >= 0 else 0.0
                        if row_no_match_flag:
                            refined_choice_prob = float(matcher_row_nomatch_probs[local_r])
                            features = [0.0 for _ in ACCEPTANCE_FEATURE_NAMES]
                        else:
                            refined_choice_prob = float(matcher_probs[local_r, refined_local_c]) if refined_local_c >= 0 else 0.0
                            features = build_acceptance_feature_vector(
                                edge_feature_names=runtime_feature_names,
                                row_feature_names=self.row_feature_names,
                                col_feature_names=self.col_feature_names,
                                edge_features=edge_features[0],
                                row_features=row_context[0],
                                col_features=col_context[0],
                                valid_mask=valid_edge_mask,
                                probs=matcher_probs,
                                row_nomatch_probs=matcher_row_nomatch_probs,
                                base_best=base_best,
                                refined_best=solution_best,
                                base_row_margin=base_row_margin,
                                refined_row_margin=solution_row_margin,
                                row_idx=int(local_r),
                            )
                        matcher_rows.append(
                            {
                                "track_index": int(comp_rows[local_r]),
                                "base_best_det_index": int(comp_cols[base_local_c]) if base_local_c >= 0 else -1,
                                "refined_best_det_index": int(comp_cols[refined_local_c]) if refined_local_c >= 0 else -1,
                                "solver_changed_row": int(bool(changed_row_mask[local_r])),
                                "row_no_match": int(row_no_match_flag),
                                "row_nomatch_prob": float(matcher_row_nomatch_probs[local_r]),
                                "base_row_margin": float(base_row_margin[local_r]),
                                "refined_row_margin": float(solution_row_margin[local_r]),
                                "base_prob_under_matcher": float(base_prob),
                                "refined_choice_prob": float(refined_choice_prob),
                                "flip_gap": float(refined_choice_prob - base_prob),
                                "matcher_assignment_margin": float(matcher_assignment_margin),
                                "matcher_objective": float(matcher_objective),
                                "component_row_count": int(len(comp_rows)),
                                "component_col_count": int(len(comp_cols)),
                                "changed_row_count": int(changed_row_count),
                                "base_margin_rejected": int(base_margin_rejected),
                                "takeover_applied": int(takeover_applied),
                                "stale_match_bias_applied": int(bool(stale_bias_rows[local_r])),
                                "feature_names": list(ACCEPTANCE_FEATURE_NAMES),
                                "features": [float(v) for v in features],
                            }
                        )
                    debug["matcher_rows"] = matcher_rows
                debug["trigger_blocks"] = int(debug["trigger_blocks"]) + 1
                debug["rows_touched"] = int(debug["rows_touched"]) + int(len(comp_rows))
                debug["edges_touched"] = int(debug["edges_touched"]) + int(valid_edge_mask.sum())
                debug["matcher_blocks"] = int(debug["matcher_blocks"]) + 1
                if not matcher_takeover:
                    debug["matcher_margin_reject_blocks"] = int(debug["matcher_margin_reject_blocks"]) + 1
                    continue
                if bool(self.config.block_matcher_skip_single_row_unchanged_takeover) and len(comp_rows) == 1 and not np.any(changed_row_mask):
                    debug["matcher_single_row_unchanged_reject_blocks"] = int(debug["matcher_single_row_unchanged_reject_blocks"]) + 1
                    continue
                if self.block_matcher_base_margin_thresh > 0.0:
                    if np.any(stable_changed_row_mask):
                        debug["matcher_base_margin_reject_blocks"] = int(debug["matcher_base_margin_reject_blocks"]) + 1
                        debug["matcher_base_margin_reject_rows"] = int(debug["matcher_base_margin_reject_rows"]) + int(
                            np.count_nonzero(stable_changed_row_mask)
                        )
                        continue
                debug["changed_blocks"] = int(debug["changed_blocks"]) + 1
                debug["matcher_takeover_blocks"] = int(debug["matcher_takeover_blocks"]) + 1
                debug["matcher_direct_takeover_blocks"] = int(debug["matcher_direct_takeover_blocks"]) + 1
                debug["matcher_takeover_pairs"] = int(debug["matcher_takeover_pairs"]) + int(valid_edge_mask.sum())
                debug["row_nomatch_rows"] = int(debug["row_nomatch_rows"]) + int(np.count_nonzero(row_no_match))
                comp_actions, comp_debug = self._derive_primitive_controller_actions(
                    comp_rows=comp_rows,
                    comp_cols=comp_cols,
                    valid_edge_mask=valid_edge_mask,
                    edge_probs=matcher_probs,
                    row_assignment=row_assignment,
                    row_no_match=row_no_match,
                    col_newborn=col_newborn,
                )
                if bool(self.config.block_matcher_force_only):
                    comp_actions = FGASControllerActions(
                        forced_matches=list(comp_actions.forced_matches),
                        blocked_rows=[],
                        blocked_cols=[],
                    )
                    comp_debug = {
                        "forced_candidates": int(comp_debug["forced_candidates"]),
                        "forced_matches": int(comp_debug["forced_matches"]),
                        "blocked_rows": 0,
                        "blocked_cols": 0,
                        "conflicts_dropped": int(comp_debug["conflicts_dropped"]),
                    }
                if comp_actions.forced_matches or comp_actions.blocked_rows or comp_actions.blocked_cols:
                    debug["controller_blocks"] = int(debug["controller_blocks"]) + 1
                forced_matches.extend(comp_actions.forced_matches)
                blocked_rows.update(comp_actions.blocked_rows)
                blocked_cols.update(comp_actions.blocked_cols)
                debug["controller_forced_candidates"] = int(debug["controller_forced_candidates"]) + int(comp_debug["forced_candidates"])
                debug["controller_forced_matches"] = int(debug["controller_forced_matches"]) + int(comp_debug["forced_matches"])
                debug["controller_blocked_rows"] = int(debug["controller_blocked_rows"]) + int(comp_debug["blocked_rows"])
                debug["controller_blocked_cols"] = int(debug["controller_blocked_cols"]) + int(comp_debug["blocked_cols"])
                debug["controller_conflicts_dropped"] = int(debug["controller_conflicts_dropped"]) + int(comp_debug["conflicts_dropped"])
                continue
            if self.block_primitive is not None:
                with torch.no_grad():
                    primitive_output = self.block_primitive(
                        edge_features=torch.tensor(edge_features, dtype=torch.float32, device=self.device),
                        edge_mask=torch.tensor(edge_mask, dtype=torch.bool, device=self.device),
                        stage_ids=stage_ids,
                        row_context=torch.tensor(row_context, dtype=torch.float32, device=self.device),
                        col_context=torch.tensor(col_context, dtype=torch.float32, device=self.device),
                    )
                primitive_probs = torch.sigmoid(primitive_output.edge_logits).cpu().numpy()[0]
                row_nomatch_probs = torch.sigmoid(primitive_output.row_no_match_logits).cpu().numpy()[0]
                col_newborn_probs = torch.sigmoid(primitive_output.col_newborn_logits).cpu().numpy()[0]
                primitive_decision = decode_block_primitive_output(
                    primitive_output,
                    edge_mask=torch.tensor(edge_mask, dtype=torch.bool, device=self.device),
                    confidence_threshold=float(self.config.block_primitive_conf_thresh),
                )
                primitive_takeover = bool(primitive_decision.takeover.cpu().numpy()[0])
                row_assignment = primitive_decision.row_assignment.cpu().numpy()[0]
                row_no_match = primitive_decision.row_no_match.cpu().numpy()[0]
                col_newborn = primitive_decision.col_newborn.cpu().numpy()[0]
                base_best, base_row_margin = self._row_top1_and_margin(comp_base_similarity, valid_edge_mask)
                refined_best, refined_row_margin, row_changed_mask, frontier_col_mask, component_changed = self._compute_changed_row_state(
                    valid_edge_mask=valid_edge_mask,
                    base_best=base_best,
                    base_row_margin=base_row_margin,
                    probs=primitive_probs,
                )
                pre_gate_row_changed_mask = row_changed_mask.copy()
                if np.any(row_no_match):
                    row_changed_mask = row_changed_mask | ((base_best >= 0) & row_no_match)
                    frontier_col_mask = self._build_frontier_col_mask(
                        base_best=base_best,
                        refined_best=refined_best,
                        row_changed_mask=row_changed_mask,
                        col_count=len(comp_cols),
                    )
                    pre_gate_row_changed_mask = row_changed_mask.copy()
                if np.any(pre_gate_row_changed_mask):
                    acceptance_rows = list(debug.get("acceptance_rows", []))
                    changed_local_rows = np.where(pre_gate_row_changed_mask)[0]
                    for local_r in changed_local_rows.tolist():
                        base_local_c = int(base_best[local_r])
                        refined_local_c = int(refined_best[local_r])
                        acceptance_rows.append(
                            {
                                "track_index": int(comp_rows[local_r]),
                                "base_best_det_index": int(comp_cols[base_local_c]) if base_local_c >= 0 else -1,
                                "refined_best_det_index": int(comp_cols[refined_local_c]) if refined_local_c >= 0 else -1,
                                "row_no_match": int(bool(row_no_match[local_r])),
                                "feature_names": list(ACCEPTANCE_FEATURE_NAMES),
                                "features": build_acceptance_feature_vector(
                                    edge_feature_names=runtime_feature_names,
                                    row_feature_names=self.row_feature_names,
                                    col_feature_names=self.col_feature_names,
                                    edge_features=edge_features[0],
                                    row_features=row_context[0],
                                    col_features=col_context[0],
                                    valid_mask=valid_edge_mask,
                                    probs=primitive_probs,
                                    row_nomatch_probs=row_nomatch_probs,
                                    base_best=base_best,
                                    refined_best=refined_best,
                                    base_row_margin=base_row_margin,
                                    refined_row_margin=refined_row_margin,
                                    row_idx=int(local_r),
                                ),
                            }
                        )
                    debug["acceptance_rows"] = acceptance_rows
                if self.has_acceptance_gate() and np.any(row_changed_mask):
                    row_changed_mask, frontier_col_mask = self._apply_acceptance_gate_mask(
                        edge_feature_names=runtime_feature_names,
                        row_feature_names=self.row_feature_names,
                        col_feature_names=self.col_feature_names,
                        edge_features=edge_features[0],
                        row_features=row_context[0],
                        col_features=col_context[0],
                        valid_mask=valid_edge_mask,
                        probs=primitive_probs,
                        row_nomatch_probs=row_nomatch_probs,
                        base_best=base_best,
                        refined_best=refined_best,
                        base_row_margin=base_row_margin,
                        refined_row_margin=refined_row_margin,
                        row_changed_mask=row_changed_mask,
                    )
                component_changed = bool(np.any(row_changed_mask) or np.any(col_newborn))
                debug["trigger_blocks"] = int(debug["trigger_blocks"]) + 1
                debug["rows_touched"] = int(debug["rows_touched"]) + int(len(comp_rows))
                debug["edges_touched"] = int(debug["edges_touched"]) + int(valid_edge_mask.sum())
                debug["primitive_blocks"] = int(debug["primitive_blocks"]) + 1
                if component_changed:
                    debug["changed_blocks"] = int(debug["changed_blocks"]) + 1
                soft_allowed_without_takeover = bool(self.config.soft_allow_without_takeover)
                if not primitive_takeover:
                    debug["primitive_fallback_blocks"] = int(debug["primitive_fallback_blocks"]) + 1
                    if not soft_allowed_without_takeover:
                        continue
                else:
                    debug["primitive_takeover_blocks"] = int(debug["primitive_takeover_blocks"]) + 1
                    debug["primitive_takeover_pairs"] = int(debug["primitive_takeover_pairs"]) + int(valid_edge_mask.sum())
                    debug["row_nomatch_rows"] = int(debug["row_nomatch_rows"]) + int(np.count_nonzero(row_no_match))
                    if bool(self.config.primitive_direct_takeover):
                        comp_actions, comp_debug = self._derive_primitive_controller_actions(
                            comp_rows=comp_rows,
                            comp_cols=comp_cols,
                            valid_edge_mask=valid_edge_mask,
                            edge_probs=primitive_probs,
                            row_assignment=row_assignment,
                            row_no_match=row_no_match,
                            col_newborn=col_newborn,
                        )
                        debug["primitive_direct_takeover_blocks"] = int(debug["primitive_direct_takeover_blocks"]) + 1
                        if comp_actions.forced_matches or comp_actions.blocked_rows or comp_actions.blocked_cols:
                            debug["controller_blocks"] = int(debug["controller_blocks"]) + 1
                        forced_matches.extend(comp_actions.forced_matches)
                        blocked_rows.update(comp_actions.blocked_rows)
                        blocked_cols.update(comp_actions.blocked_cols)
                        debug["controller_forced_candidates"] = int(debug["controller_forced_candidates"]) + int(comp_debug["forced_candidates"])
                        debug["controller_forced_matches"] = int(debug["controller_forced_matches"]) + int(comp_debug["forced_matches"])
                        debug["controller_blocked_rows"] = int(debug["controller_blocked_rows"]) + int(comp_debug["blocked_rows"])
                        debug["controller_blocked_cols"] = int(debug["controller_blocked_cols"]) + int(comp_debug["blocked_cols"])
                        debug["controller_conflicts_dropped"] = int(debug["controller_conflicts_dropped"]) + int(comp_debug["conflicts_dropped"])
                        continue
                apply_soft_on_component = bool(component_changed or not bool(self.config.soft_apply_only_changed_blocks))
                if apply_soft_on_component:
                    debug["soft_applied_blocks"] = int(debug["soft_applied_blocks"]) + 1
                    if not primitive_takeover:
                        debug["primitive_soft_fallback_blocks"] = int(debug["primitive_soft_fallback_blocks"]) + 1
                for row_idx in comp_rows:
                    for det_idx in row_orders.get(row_idx, []):
                        if det_idx not in col_map:
                            continue
                        local_r = row_map[row_idx]
                        local_c = col_map[det_idx]
                        if not bool(valid_edge_mask[local_r, local_c]):
                            continue
                        base_score = float(base_similarity[row_idx, det_idx])
                        pred_score = float(primitive_probs[local_r, local_c])
                        if str(self.config.assignment_mode) == "replace":
                            score = pred_score
                        else:
                            score = (1.0 - float(self.config.blend_weight)) * base_score + float(self.config.blend_weight) * pred_score
                        if float(self.config.row_nomatch_weight) > 0.0:
                            score *= max(0.0, 1.0 - float(self.config.row_nomatch_weight) * float(row_nomatch_probs[local_r]))
                        refined[row_idx, det_idx] = float(np.clip(1.0 - score, 0.0, 1.0))
                        row_soft_allowed = bool(apply_soft_on_component)
                        if bool(self.config.soft_apply_only_changed_frontier):
                            row_soft_allowed = bool(row_changed_mask[local_r] or frontier_col_mask[local_c])
                        elif bool(self.config.soft_apply_only_changed_rows):
                            row_soft_allowed = bool(row_changed_mask[local_r])
                        if row_soft_allowed:
                            refined_mask[row_idx, det_idx] = True
                controller_allowed = bool(primitive_takeover and self.config.controller_enable) and (
                    bool(component_changed) or not bool(self.config.controller_only_changed_blocks)
                )
                if bool(self.config.controller_enable) and not controller_allowed:
                    debug["controller_skipped_unchanged_blocks"] = int(debug["controller_skipped_unchanged_blocks"]) + 1
                if controller_allowed:
                    comp_actions, comp_debug = self._derive_primitive_controller_actions(
                        comp_rows=comp_rows,
                        comp_cols=comp_cols,
                        valid_edge_mask=valid_edge_mask,
                        edge_probs=primitive_probs,
                        row_assignment=row_assignment,
                        row_no_match=row_no_match,
                        col_newborn=col_newborn,
                    )
                    if comp_actions.forced_matches or comp_actions.blocked_rows or comp_actions.blocked_cols:
                        debug["controller_blocks"] = int(debug["controller_blocks"]) + 1
                    forced_matches.extend(comp_actions.forced_matches)
                    blocked_rows.update(comp_actions.blocked_rows)
                    blocked_cols.update(comp_actions.blocked_cols)
                    debug["controller_forced_candidates"] = int(debug["controller_forced_candidates"]) + int(comp_debug["forced_candidates"])
                    debug["controller_forced_matches"] = int(debug["controller_forced_matches"]) + int(comp_debug["forced_matches"])
                    debug["controller_blocked_rows"] = int(debug["controller_blocked_rows"]) + int(comp_debug["blocked_rows"])
                    debug["controller_blocked_cols"] = int(debug["controller_blocked_cols"]) + int(comp_debug["blocked_cols"])
                    debug["controller_conflicts_dropped"] = int(debug["controller_conflicts_dropped"]) + int(comp_debug["conflicts_dropped"])
                continue
            if self.has_pair_scorer():
                pair_row_feature_maps = {
                    int(row_idx): dict(
                        zip(
                            ROW_CONTEXT_FEATURE_NAMES,
                            build_row_context_from_rows(
                                row_payloads[row_idx],
                                candidate_limit=float(max(len(comp_cols), 1)),
                            ),
                        )
                    )
                    for row_idx in comp_rows
                }
                pair_col_feature_maps = {
                    int(col_idx): dict(
                        zip(
                            COL_CONTEXT_FEATURE_NAMES,
                            build_col_context_from_rows(
                                col_payloads[col_idx],
                                candidate_limit=float(max(len(comp_rows), 1)),
                            ),
                        )
                    )
                    for col_idx in comp_cols
                }
                pair_vectors: List[List[float]] = []
                pair_positions: List[Tuple[int, int, int, int]] = []
                for row_idx in comp_rows:
                    for det_idx in row_orders.get(row_idx, []):
                        if det_idx not in col_map:
                            continue
                        payload = candidate_payloads.get((int(row_idx), int(det_idx)))
                        if payload is None:
                            continue
                        canonical_edge_vector = build_edge_feature_vector(
                            s_reid=float(payload.get("s_reid", 0.0)),
                            s_low=float(payload.get("s_low", 0.0)),
                            s_mid=float(payload.get("s_mid", 0.0)),
                            s_high=float(payload.get("s_high", 0.0)),
                            base_similarity=float(payload.get("base_similarity", 0.0)),
                            det_score=float(payload.get("det_score", 0.0)),
                            track_age=float(payload.get("track_age", 0.0)),
                            raw_iou_cost=float(payload.get("raw_iou_cost", 1.0)),
                            fused_iou_cost=float(payload.get("fused_iou_cost", 1.0)),
                            track_box=payload.get("track_box", [0.0, 0.0, 1.0, 1.0]),
                            det_box=payload.get("det_box", [0.0, 0.0, 1.0, 1.0]),
                            candidate_rank=float(payload.get("candidate_rank", 0.0)),
                            top_k=float(self.config.top_k),
                        )
                        edge_feature_map = {
                            name: float(value)
                            for name, value in zip(EDGE_FEATURE_NAMES, canonical_edge_vector)
                        }
                        pair_vectors.append(
                            self._build_pair_feature_vector(
                                edge_feature_map=edge_feature_map,
                                row_feature_map=pair_row_feature_maps.get(int(row_idx), {}),
                                col_feature_map=pair_col_feature_maps.get(int(det_idx), {}),
                                seq_name=seq_name,
                            )
                        )
                        pair_positions.append((int(row_map[row_idx]), int(col_map[det_idx]), int(row_idx), int(det_idx)))
                if not pair_vectors:
                    continue
                with torch.no_grad():
                    pair_input = torch.tensor(np.asarray(pair_vectors, dtype=np.float32), dtype=torch.float32, device=self.device)
                    pair_logits = self.pair_scorer(pair_input)
                pair_probs = torch.sigmoid(pair_logits).cpu().numpy().astype(np.float32)
                probs = np.zeros_like(comp_base_similarity, dtype=np.float32)
                for pair_idx, (local_r, local_c, _row_idx, _det_idx) in enumerate(pair_positions):
                    probs[local_r, local_c] = float(pair_probs[pair_idx])
                row_nomatch_probs = np.zeros((len(comp_rows),), dtype=np.float32)
                col_newborn_probs = None
                base_best, base_row_margin = self._row_top1_and_margin(comp_base_similarity, valid_edge_mask)
                refined_best, refined_row_margin = self._row_top1_and_margin(probs, valid_edge_mask)
                actual_changed_mask = (base_best >= 0) & (refined_best >= 0) & (base_best != refined_best)
                row_changed_mask = np.ones((len(comp_rows),), dtype=bool)
                frontier_col_mask = np.ones((len(comp_cols),), dtype=bool)
                component_changed = bool(np.any(actual_changed_mask))
                block_ambiguous_flag = float(
                    np.any((base_best >= 0) & (base_row_margin < float(self.config.pair_ambiguity_margin)))
                )
                apply_soft_on_component = True
                debug["trigger_blocks"] = int(debug["trigger_blocks"]) + 1
                debug["rows_touched"] = int(debug["rows_touched"]) + int(len(comp_rows))
                debug["edges_touched"] = int(debug["edges_touched"]) + int(edge_mask.sum())
                debug["pair_mode_blocks"] = int(debug["pair_mode_blocks"]) + 1
                debug["pair_mode_rows"] = int(debug["pair_mode_rows"]) + int(len(comp_rows))
                debug["pair_mode_edges"] = int(debug["pair_mode_edges"]) + int(valid_edge_mask.sum())
                if component_changed:
                    debug["changed_blocks"] = int(debug["changed_blocks"]) + 1
                if self.has_block_gate():
                    debug["block_gate_blocks"] = int(debug["block_gate_blocks"]) + 1
                    block_gate_features = build_block_gate_feature_vector(
                        row_feature_names=self.row_feature_names,
                        row_features=row_context[0],
                        valid_mask=valid_edge_mask,
                        base_similarity=comp_base_similarity,
                        probs=probs,
                        base_best=base_best,
                        refined_best=refined_best,
                        base_row_margin=base_row_margin,
                        refined_row_margin=refined_row_margin,
                        seq_name=seq_name,
                        block_ambiguous_flag=block_ambiguous_flag,
                    )
                    block_gate_feature_map = dict(zip(BLOCK_GATE_FEATURE_NAMES, block_gate_features))
                    gate_vector = [
                        float(block_gate_feature_map.get(name, 0.0))
                        for name in self.block_gate_feature_names
                    ]
                    gate_input = torch.tensor(gate_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        gate_score = float(torch.sigmoid(self.block_gate(gate_input)).item())
                    if gate_score < float(self.config.block_gate_thresh):
                        debug["block_gate_filtered_blocks"] = int(debug["block_gate_filtered_blocks"]) + 1
                        continue
                    debug["block_gate_pass_blocks"] = int(debug["block_gate_pass_blocks"]) + 1
                debug["soft_applied_blocks"] = int(debug["soft_applied_blocks"]) + 1
                for row_idx in comp_rows:
                    for det_idx in row_orders.get(row_idx, []):
                        if det_idx not in col_map:
                            continue
                        local_r = row_map[row_idx]
                        local_c = col_map[det_idx]
                        base_score = float(base_similarity[row_idx, det_idx])
                        pred_score = float(probs[local_r, local_c])
                        if str(self.config.assignment_mode) == "replace":
                            score = pred_score
                        else:
                            score = (1.0 - float(self.config.blend_weight)) * base_score + float(self.config.blend_weight) * pred_score
                        refined[row_idx, det_idx] = float(np.clip(1.0 - score, 0.0, 1.0))
                        refined_mask[row_idx, det_idx] = True
                continue

            with torch.no_grad():
                forward_kwargs = {
                    "edge_features": torch.tensor(edge_features, dtype=torch.float32, device=self.device),
                    "edge_mask": torch.tensor(edge_mask, dtype=torch.bool, device=self.device),
                    "stage_ids": stage_ids,
                }
                if self.arch == "v3_trackquery":
                    forward_kwargs["row_context"] = torch.tensor(row_context, dtype=torch.float32, device=self.device)
                    forward_kwargs["col_context"] = torch.tensor(col_context, dtype=torch.float32, device=self.device)
                output = self.model(**forward_kwargs)
            probs = torch.sigmoid(output.edge_logits).cpu().numpy()[0]
            row_nomatch_probs = torch.sigmoid(output.row_no_match_logits).cpu().numpy()[0]
            col_newborn_probs = (
                torch.sigmoid(output.col_newborn_logits).cpu().numpy()[0]
                if output.col_newborn_logits is not None
                else None
            )
            base_best, base_row_margin = self._row_top1_and_margin(comp_base_similarity, valid_edge_mask)
            refined_best, refined_row_margin, row_changed_mask, frontier_col_mask, component_changed = self._compute_changed_row_state(
                valid_edge_mask=valid_edge_mask,
                base_best=base_best,
                base_row_margin=base_row_margin,
                probs=probs,
            )
            apply_soft_on_component = bool(component_changed or not bool(self.config.soft_apply_only_changed_blocks))
            debug["trigger_blocks"] = int(debug["trigger_blocks"]) + 1
            debug["rows_touched"] = int(debug["rows_touched"]) + int(len(comp_rows))
            debug["edges_touched"] = int(debug["edges_touched"]) + int(edge_mask.sum())
            if component_changed:
                debug["changed_blocks"] = int(debug["changed_blocks"]) + 1
            if self.has_acceptance_gate() and np.any(row_changed_mask):
                row_changed_mask, frontier_col_mask = self._apply_acceptance_gate_mask(
                    edge_feature_names=self.feature_names,
                    row_feature_names=self.row_feature_names,
                    col_feature_names=self.col_feature_names,
                    edge_features=edge_features[0],
                    row_features=row_context[0],
                    col_features=col_context[0],
                    valid_mask=valid_edge_mask,
                    probs=probs,
                    row_nomatch_probs=row_nomatch_probs,
                    base_best=base_best,
                    refined_best=refined_best,
                    base_row_margin=base_row_margin,
                    refined_row_margin=refined_row_margin,
                    row_changed_mask=row_changed_mask,
                )
                component_changed = bool(np.any(row_changed_mask))
            if apply_soft_on_component:
                debug["soft_applied_blocks"] = int(debug["soft_applied_blocks"]) + 1
            for row_idx in comp_rows:
                for det_idx in row_orders.get(row_idx, []):
                    if det_idx not in col_map:
                        continue
                    local_r = row_map[row_idx]
                    local_c = col_map[det_idx]
                    base_score = float(base_similarity[row_idx, det_idx])
                    pred_score = float(probs[local_r, local_c])
                    if str(self.config.assignment_mode) == "replace":
                        score = pred_score
                    else:
                        score = (1.0 - float(self.config.blend_weight)) * base_score + float(self.config.blend_weight) * pred_score
                    if float(self.config.row_nomatch_weight) > 0.0:
                        row_nomatch = float(row_nomatch_probs[local_r])
                        if row_nomatch > 0.5:
                            debug["row_nomatch_rows"] = int(debug["row_nomatch_rows"]) + 1
                        score *= max(0.0, 1.0 - float(self.config.row_nomatch_weight) * row_nomatch)
                    refined[row_idx, det_idx] = float(np.clip(1.0 - score, 0.0, 1.0))
                    row_soft_allowed = bool(apply_soft_on_component)
                    if bool(self.config.soft_apply_only_changed_frontier):
                        row_soft_allowed = bool(row_changed_mask[local_r] or frontier_col_mask[local_c])
                    elif bool(self.config.soft_apply_only_changed_rows):
                        row_soft_allowed = bool(row_changed_mask[local_r])
                    if row_soft_allowed:
                        refined_mask[row_idx, det_idx] = True
            controller_allowed = bool(self.config.controller_enable) and (
                bool(component_changed) or not bool(self.config.controller_only_changed_blocks)
            )
            if bool(self.config.controller_enable) and not controller_allowed:
                debug["controller_skipped_unchanged_blocks"] = int(debug["controller_skipped_unchanged_blocks"]) + 1
            if controller_allowed:
                comp_actions, comp_debug = self._derive_component_controller_actions(
                    comp_rows=comp_rows,
                    comp_cols=comp_cols,
                    edge_mask=edge_mask[0],
                    base_similarity=base_similarity[np.ix_(comp_rows, comp_cols)],
                    probs=probs,
                    row_nomatch_probs=row_nomatch_probs,
                    col_newborn_probs=col_newborn_probs,
                )
                if comp_actions.forced_matches or comp_actions.blocked_rows or comp_actions.blocked_cols:
                    debug["controller_blocks"] = int(debug["controller_blocks"]) + 1
                forced_matches.extend(comp_actions.forced_matches)
                blocked_rows.update(comp_actions.blocked_rows)
                blocked_cols.update(comp_actions.blocked_cols)
                debug["controller_forced_candidates"] = int(debug["controller_forced_candidates"]) + int(comp_debug["forced_candidates"])
                debug["controller_forced_matches"] = int(debug["controller_forced_matches"]) + int(comp_debug["forced_matches"])
                debug["controller_blocked_rows"] = int(debug["controller_blocked_rows"]) + int(comp_debug["blocked_rows"])
                debug["controller_blocked_cols"] = int(debug["controller_blocked_cols"]) + int(comp_debug["blocked_cols"])
                debug["controller_conflicts_dropped"] = int(debug["controller_conflicts_dropped"]) + int(comp_debug["conflicts_dropped"])
                debug["controller_forced_rejected_not_mutual"] = int(debug["controller_forced_rejected_not_mutual"]) + int(
                    comp_debug["controller_forced_rejected_not_mutual"]
                )
                debug["controller_forced_rejected_not_base_top1"] = int(debug["controller_forced_rejected_not_base_top1"]) + int(
                    comp_debug["controller_forced_rejected_not_base_top1"]
                )
        forced_row_ids = {int(row_idx) for row_idx, _ in forced_matches}
        forced_col_ids = {int(col_idx) for _, col_idx in forced_matches}
        actions = FGASControllerActions(
            forced_matches=sorted({(int(row_idx), int(col_idx)) for row_idx, col_idx in forced_matches}),
            blocked_rows=sorted(int(row_idx) for row_idx in blocked_rows.difference(forced_row_ids)),
            blocked_cols=sorted(int(col_idx) for col_idx in blocked_cols.difference(forced_col_ids)),
        )
        return refined, refined_mask, debug, actions
