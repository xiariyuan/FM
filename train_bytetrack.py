# Copyright (c) 2024. All Rights Reserved.
"""
使用 ByteTrack/YOLOX 冻结特征训练频域关联模块

这个训练脚本：
1. 使用 ByteTrack (YOLOX) 作为冻结的特征提取器
2. 只训练频域关联模块 (trajectory_modeling + id_decoder)
3. 不使用 DINO 检测器

用法：
    python train_bytetrack.py \
        --config-path configs/bytetrack_fa_mot_mot17.yaml \
        --data-root /path/to/datasets \
        --exp-name bytetrack_fa_mot_mot17
"""

import os
import math
import numpy as np
import hashlib
import traceback
import torch.nn.functional as F
import torch
import einops
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.state import PartialState
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from collections import defaultdict
from typing import List, Tuple, Optional

from runtime_option import runtime_option
from utils.misc import yaml_to_dict, set_seed
from utils.detector_profile import resolve_bytetrack_profile
from utils.mot_detections import load_mot_detections, resolve_external_det_path
from configs.util import load_super_config, update_config
from log.logger import Logger
from data import build_dataset
from data.naive_sampler import NaiveSampler
from data.util import collate_fn
from log.log import TPS, Metrics
from models.misc import save_checkpoint, load_checkpoint, get_model
from utils.nested_tensor import NestedTensor

from models.bytetrack_feature_extractor import (
    ByteTrackFeatureConfig,
    ByteTrackFeatureExtractor,
)
from vnext_laplace_assoc import LaplaceAssociationAdapter, compute_laplace_supervision_loss

# Optional Top-Conference losses (Triplet/TP-Drop/Memory) - lazy use
try:
    from models.motip.topconf_losses import build_triplet_loss, build_memory_bank
    _TOPCONF_AVAILABLE = True
except Exception:
    build_triplet_loss = None
    build_memory_bank = None
    _TOPCONF_AVAILABLE = False

# Optional direct matching supervision (det <-> track)
try:
    from models.motip.matching_losses import build_det_track_match_loss
    _MATCHLOSS_AVAILABLE = True
except Exception:
    build_det_track_match_loss = None
    _MATCHLOSS_AVAILABLE = False

# Make TORCH_HOME portable
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "torch")

def _parse_pedestrian_summary(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = f.readline().strip().split()
            vals = f.readline().strip().split()
        out = {}
        for k, v in zip(names, vals):
            try:
                out[k] = float(v)
            except ValueError:
                continue
        return out
    except Exception:
        return {}


def _find_tracker_root(eval_dir: str, dataset_name: str, split: str, min_track_len: int) -> Optional[str]:
    base_root = os.path.join(eval_dir, "tracker")
    if min_track_len > 1:
        filtered_root = os.path.join(eval_dir, f"tracker_min{min_track_len}")
        if os.path.isdir(os.path.join(filtered_root, f"{dataset_name}-{split}")):
            return filtered_root
    if os.path.isdir(os.path.join(base_root, f"{dataset_name}-{split}")):
        return base_root
    return None


def _compute_tracker_id_stats(tracker_dir: str, val_bases: Optional[List[str]] = None) -> dict:
    stats = {"sequences": {}, "overall": {}}
    total_dets = 0
    total_unique = 0
    if not os.path.isdir(tracker_dir):
        return stats
    for fname in sorted(os.listdir(tracker_dir)):
        if not fname.endswith(".txt"):
            continue
        if fname == "pedestrian_summary.txt":
            continue
        if val_bases:
            if not any(fname.startswith(base) for base in val_bases):
                continue
        seq_path = os.path.join(tracker_dir, fname)
        dets = 0
        id_counts = {}
        try:
            with open(seq_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) < 2:
                        continue
                    try:
                        tid = int(float(parts[1]))
                    except ValueError:
                        continue
                    dets += 1
                    id_counts[tid] = id_counts.get(tid, 0) + 1
        except Exception:
            continue
        unique = len(id_counts)
        avg_len = float(dets) / unique if unique > 0 else 0.0
        reuse_ratio = 1.0 - (float(unique) / dets) if dets > 0 else 0.0
        seq_name = os.path.splitext(fname)[0]
        stats["sequences"][seq_name] = {
            "total_dets": dets,
            "unique_ids": unique,
            "avg_track_len": avg_len,
            "id_reuse_ratio": reuse_ratio,
        }
        total_dets += dets
        total_unique += unique
    overall_avg = float(total_dets) / total_unique if total_unique > 0 else 0.0
    overall_reuse = 1.0 - (float(total_unique) / total_dets) if total_dets > 0 else 0.0
    stats["overall"] = {
        "total_dets": total_dets,
        "unique_ids": total_unique,
        "avg_track_len": overall_avg,
        "id_reuse_ratio": overall_reuse,
    }
    return stats


def _write_validation_diagnostics(eval_dir: str, config: dict, logger: Logger) -> None:
    dataset_name = config.get("INFERENCE_DATASET", "MOT17")
    split = config.get("INFERENCE_SPLIT", "train")
    min_track_len = int(config.get("EVAL_MIN_TRACK_LEN", 1))
    tracker_root = _find_tracker_root(eval_dir, dataset_name, split, min_track_len)
    if tracker_root is None:
        logger.warning(f"[Diag] tracker root not found under {eval_dir}")
        return
    tracker_dir = os.path.join(tracker_root, f"{dataset_name}-{split}")
    val_sequences = config.get("VAL_SEQUENCES", None)
    if isinstance(val_sequences, str):
        val_sequences = [val_sequences]
    stats = _compute_tracker_id_stats(tracker_dir, val_sequences)
    summary_path = os.path.join(tracker_dir, "pedestrian_summary.txt")
    metrics = _parse_pedestrian_summary(summary_path)
    diag_path = os.path.join(eval_dir, "diagnostics.json")
    try:
        import json
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump({"id_stats": stats, "trackeval": metrics}, f, indent=2)
    except Exception as e:
        logger.warning(f"[Diag] Failed to write diagnostics to {diag_path}: {e}")
    overall = stats.get("overall", {})
    idsw = metrics.get("IDSW", None)
    frag = metrics.get("Frag", None)
    logger.info(
        "[Diag] ID reuse={:.4f} avg_len={:.2f} unique_ids={} total_dets={} IDSW={} Frag={}".format(
            float(overall.get("id_reuse_ratio", 0.0)),
            float(overall.get("avg_track_len", 0.0)),
            int(overall.get("unique_ids", 0)),
            int(overall.get("total_dets", 0)),
            "NA" if idsw is None else int(idsw),
            "NA" if frag is None else int(frag),
        )
    )
    # Also print key TrackEval metrics for quick per-epoch comparison (especially HOTA/AssA/IDSW/Frag).
    if metrics:
        def _fmt_metric(name: str, default: str = "NA") -> str:
            v = metrics.get(name, None)
            if v is None:
                return default
            # TrackEval reports most percentages already in [0, 100]
            if name in ("IDSW", "Frag", "CLR_FP", "CLR_FN", "MT", "ML", "PT"):
                try:
                    return str(int(v))
                except Exception:
                    return default
            try:
                return f"{float(v):.2f}"
            except Exception:
                return default

        logger.info(
            "[Val] HOTA={} DetA={} AssA={} IDF1={} MOTA={} IDSW={} Frag={}".format(
                _fmt_metric("HOTA"),
                _fmt_metric("DetA"),
                _fmt_metric("AssA"),
                _fmt_metric("IDF1"),
                _fmt_metric("MOTA"),
                _fmt_metric("IDSW"),
                _fmt_metric("Frag"),
            )
        )


def _run_validation_inprocess(
    *,
    accelerator: Accelerator,
    model,
    config: dict,
    eval_dir: str,
    logger: Logger,
) -> bool:
    """
    Run validation inside the training process to avoid GPU OOM from spawning a second
    Python process that loads another copy of YOLOX + tracking modules.
    """
    try:
        import gc
        from PIL import Image
        from models.runtime_tracker_bytetrack import RuntimeTrackerByteTrack
        from submit_bytetrack import write_results, run_trackeval
        from models.public_reid import build_public_reid_encoder

        device = accelerator.device
        use_fp16 = bool(config.get("BYTETRACK_FP16", True)) and device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        # Best-effort: clear transient training caches before long inference.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Unwrap model (Accelerate/DDP) to access submodules directly.
        model_wo_ddp = accelerator.unwrap_model(model) if hasattr(accelerator, "unwrap_model") else get_model(model)
        feature_extractor = getattr(model_wo_ddp, "feature_extractor", None)
        trajectory_modeling = getattr(model_wo_ddp, "trajectory_modeling", None)
        id_decoder = getattr(model_wo_ddp, "id_decoder", None)
        if feature_extractor is None or trajectory_modeling is None or id_decoder is None:
            logger.warning("[Val] In-process validation skipped: model missing required submodules.")
            return False

        feature_extractor.eval()
        trajectory_modeling.eval()
        id_decoder.eval()

        # Optional: build (and cache) external ReID encoder for association.
        reid_encoder = None
        if bool(config.get("ASSOC_USE_REID", False)):
            # Cache per-process to avoid re-building every epoch.
            cache_key = "_cached_reid_encoder"
            if not hasattr(_run_validation_inprocess, cache_key):
                reid_cfg = {
                    "PUBLIC_REID_BACKBONE": config.get("ASSOC_REID_BACKBONE", config.get("PUBLIC_REID_BACKBONE", "torchreid:osnet_x1_0")),
                    "PUBLIC_REID_WEIGHTS": config.get("ASSOC_REID_WEIGHTS", config.get("PUBLIC_REID_WEIGHTS", None)),
                    "PUBLIC_REID_PRETRAINED": config.get("ASSOC_REID_PRETRAINED", config.get("PUBLIC_REID_PRETRAINED", True)),
                    "PUBLIC_REID_INPUT_H": config.get("ASSOC_REID_INPUT_H", config.get("PUBLIC_REID_INPUT_H", 256)),
                    "PUBLIC_REID_INPUT_W": config.get("ASSOC_REID_INPUT_W", config.get("PUBLIC_REID_INPUT_W", 128)),
                    "PUBLIC_REID_BATCH_SIZE": config.get("ASSOC_REID_BATCH_SIZE", config.get("PUBLIC_REID_BATCH_SIZE", 64)),
                    "PUBLIC_REID_L2_NORM": config.get("ASSOC_REID_L2_NORM", config.get("PUBLIC_REID_L2_NORM", True)),
                    "PUBLIC_REID_PROJ_SEED": config.get("ASSOC_REID_PROJ_SEED", config.get("PUBLIC_REID_PROJ_SEED", 12345)),
                    "PUBLIC_REID_BOX_EXPAND": float(config.get("ASSOC_REID_BOX_EXPAND", config.get("PUBLIC_REID_BOX_EXPAND", 1.0))),
                }
                reid_dim = int(config.get("ASSOC_REID_DIM", 512))
                reid_dtype = torch.float16 if device.type == "cuda" else torch.float32
                try:
                    setattr(
                        _run_validation_inprocess,
                        cache_key,
                        build_public_reid_encoder(
                            config=reid_cfg,
                            device=device,
                            feature_dim=reid_dim,
                            dtype=reid_dtype,
                        ),
                    )
                    logger.info(f"[Val] Built ReID encoder: {reid_cfg.get('PUBLIC_REID_BACKBONE')} (dim={reid_dim})")
                except Exception as exc:
                    logger.warning(f"[Val] Failed to build ReID encoder; disabled. Error: {exc}")
                    setattr(_run_validation_inprocess, cache_key, None)
            reid_encoder = getattr(_run_validation_inprocess, cache_key, None)

        # Dataset / sequences
        data_root = config.get("DATA_ROOT", None)
        dataset_name = config.get("INFERENCE_DATASET", "MOT17")
        split = config.get("INFERENCE_SPLIT", "train")
        if not data_root:
            logger.warning("[Val] Missing DATA_ROOT in config; skip validation.")
            return False
        dataset_dir = os.path.join(data_root, dataset_name, split)
        if not os.path.isdir(dataset_dir):
            logger.warning(f"[Val] Dataset dir not found: {dataset_dir}")
            return False

        sequences = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        detector_filter = config.get("DETECTOR_FILTER", None)
        if detector_filter:
            tokens = list(detector_filter) if not isinstance(detector_filter, (list, tuple)) else list(detector_filter)
            filtered = [d for d in sequences if any(t in d for t in tokens)]
            if filtered:
                sequences = filtered
        if config.get("EVAL_ONLY_VAL", False):
            val_sequences = config.get("VAL_SEQUENCES", None)
            if isinstance(val_sequences, str):
                val_sequences = [val_sequences]
            if val_sequences:
                sequences = [d for d in sequences if any(d.startswith(base) for base in val_sequences)]

        if len(sequences) == 0:
            logger.warning(f"[Val] No sequences found under {dataset_dir} after filtering.")
            return False

        # Inference settings
        det_source = str(config.get("BYTETRACK_DET_SOURCE", "model")).lower()
        if det_source in ("txt", "public"):
            det_source = "external"
        if det_source not in ("model", "external"):
            det_source = "model"

        os.makedirs(eval_dir, exist_ok=True)
        out_tracker_dir = os.path.join(eval_dir, "tracker", f"{dataset_name}-{split}")
        os.makedirs(out_tracker_dir, exist_ok=True)

        for seq_name in sequences:
            seq_dir = os.path.join(dataset_dir, seq_name)
            img_dir = os.path.join(seq_dir, "img1")
            if not os.path.isdir(img_dir):
                logger.warning(f"[Val] Missing img dir: {img_dir}")
                continue
            images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
            if not images:
                logger.warning(f"[Val] No images found in {img_dir}")
                continue

            # Resolve sequence HW
            first_img_path = os.path.join(img_dir, images[0])
            try:
                with Image.open(first_img_path) as im:
                    seq_w, seq_h = im.size
            except Exception:
                logger.warning(f"[Val] Failed to read first image for {seq_name}: {first_img_path}")
                continue

            # Per-detector overrides (DPM/FRCNN/SDP) to mirror submit_bytetrack behavior.
            miss_tolerance = int(config.get("MISS_TOLERANCE", 30))
            det_thresh = float(config.get("DET_THRESH", 0.25))
            newborn_thresh = float(config.get("NEWBORN_THRESH", 0.25))
            det_max_per_frame = int(config.get("DET_MAX_PER_FRAME", 0))
            id_thresh = float(config.get("ID_THRESH", 0.08))
            assoc_iou_gate = float(config.get("ASSOC_IOU_GATE", 0.0))
            assoc_mode = str(config.get("ASSOC_MODE", "logit")).lower()
            assoc_id_weight = float(config.get("ASSOC_ID_WEIGHT", 1.0))
            assoc_iou_weight = float(config.get("ASSOC_IOU_WEIGHT", 0.0))
            assoc_feat_weight = float(config.get("ASSOC_FEAT_WEIGHT", 1.0))

            seq_upper = str(seq_name).upper()
            det_suffix = None
            if seq_upper.endswith("DPM"):
                det_suffix = "DPM"
            elif seq_upper.endswith("FRCNN"):
                det_suffix = "FRCNN"
            elif seq_upper.endswith("SDP"):
                det_suffix = "SDP"
            if det_suffix is not None:
                miss_tolerance = int(config.get(f"MISS_TOLERANCE_{det_suffix}", miss_tolerance))
                det_thresh = float(config.get(f"DET_THRESH_{det_suffix}", det_thresh))
                newborn_thresh = float(config.get(f"NEWBORN_THRESH_{det_suffix}", newborn_thresh))
                det_max_per_frame = int(config.get(f"DET_MAX_PER_FRAME_{det_suffix}", det_max_per_frame))
                id_thresh = float(config.get(f"ID_THRESH_{det_suffix}", id_thresh))
                assoc_iou_gate = float(config.get(f"ASSOC_IOU_GATE_{det_suffix}", assoc_iou_gate))
                assoc_mode = str(config.get(f"ASSOC_MODE_{det_suffix}", assoc_mode)).lower()
                assoc_id_weight = float(config.get(f"ASSOC_ID_WEIGHT_{det_suffix}", assoc_id_weight))
                assoc_iou_weight = float(config.get(f"ASSOC_IOU_WEIGHT_{det_suffix}", assoc_iou_weight))
                assoc_feat_weight = float(config.get(f"ASSOC_FEAT_WEIGHT_{det_suffix}", assoc_feat_weight))

            external_detections = None
            if det_source == "external":
                det_path = resolve_external_det_path(
                    config=config,
                    dataset_name=dataset_name,
                    split=split,
                    seq_name=seq_name,
                )
                if not os.path.isfile(det_path):
                    raise FileNotFoundError(f"[Val] External detection file not found for {seq_name}: {det_path}")
                external_detections = load_mot_detections(det_path)

            seq_key = f"{dataset_name}/{split}/{seq_name}"
            tracker = RuntimeTrackerByteTrack(
                trajectory_modeling=trajectory_modeling,
                id_decoder=id_decoder,
                feature_extractor=feature_extractor,
                sequence_hw=(seq_h, seq_w),
                miss_tolerance=miss_tolerance,
                det_thresh=det_thresh,
                det_max_per_frame=det_max_per_frame,
                newborn_thresh=newborn_thresh,
                id_thresh=id_thresh,
                area_thresh=int(config.get("AREA_THRESH", 0)),
                num_id_vocabulary=int(config.get("NUM_ID_VOCABULARY", 500)),
                feature_dim=int(config.get("FEATURE_DIM", 256)),
                track_window=int(config.get("INFERENCE_TRACK_WINDOW", config.get("MAX_SEQ_LEN", 30))),
                matching_method=str(config.get("ASSOC_MATCHING", "hungarian")),
                assoc_iou_gate=assoc_iou_gate,
                assoc_id_weight=assoc_id_weight,
                assoc_iou_weight=assoc_iou_weight,
                assoc_logit_temp=float(config.get("ASSOC_LOGIT_TEMP", 1.0)),
                assoc_use_det_score=bool(config.get("ASSOC_USE_DET_SCORE", False)),
                assoc_mode=assoc_mode,
                assoc_feat_weight=assoc_feat_weight,
                assoc_feat_agg=str(config.get("ASSOC_FEAT_AGG", "last")),
                assoc_feat_k=int(config.get("ASSOC_FEAT_K", 5)),
                assoc_feat_tau=float(config.get("ASSOC_FEAT_TAU", 1.0)),
                assoc_feat_source=str(config.get("ASSOC_FEAT_SOURCE", "yolox")),
                assoc_feat_score_mode=str(config.get("ASSOC_FEAT_SCORE_MODE", "raw")),
                assoc_freq_gate=bool(config.get("ASSOC_FREQ_GATE", False)),
                assoc_freq_gate_min=float(config.get("ASSOC_FREQ_GATE_MIN", 0.2)),
                assoc_freq_gate_max=float(config.get("ASSOC_FREQ_GATE_MAX", 1.0)),
                assoc_use_laplace=bool(config.get("ASSOC_USE_LAPLACE", False)),
                assoc_laplace_weight=float(config.get("ASSOC_LAPLACE_WEIGHT", 0.35)),
                assoc_laplace_decay_scales=config.get("ASSOC_LAPLACE_DECAY_SCALES", [1.0, 2.0, 4.0]),
                assoc_laplace_hidden_dim=int(config.get("ASSOC_LAPLACE_HIDDEN_DIM", 16)),
                assoc_use_mtcr=bool(config.get("ASSOC_USE_MTCR", False)),
                assoc_mtcr_checkpoint=str(config.get("ASSOC_MTCR_CHECKPOINT", "")),
                assoc_mtcr_hist_hidden=int(config.get("ASSOC_MTCR_HIST_HIDDEN", 16)),
                assoc_mtcr_comp_hidden=int(config.get("ASSOC_MTCR_COMP_HIDDEN", 64)),
                assoc_mtcr_topk=int(config.get("ASSOC_MTCR_TOPK", 3)),
                assoc_mtcr_margin_threshold=float(config.get("ASSOC_MTCR_MARGIN_THRESHOLD", 0.10)),
                assoc_mtcr_margin_temperature=float(config.get("ASSOC_MTCR_MARGIN_TEMPERATURE", 0.03)),
                assoc_mtcr_delta_scale=float(config.get("ASSOC_MTCR_DELTA_SCALE", 1.0)),
                assoc_mtcr_min_history=int(config.get("ASSOC_MTCR_MIN_HISTORY", 3)),
                assoc_mtcr_decay_scales=config.get("ASSOC_MTCR_DECAY_SCALES", [1.0, 2.0, 4.0]),
                assoc_use_runtime_replay=bool(config.get("ASSOC_USE_RUNTIME_REPLAY", False)),
                assoc_runtime_replay_checkpoint=str(config.get("ASSOC_RUNTIME_REPLAY_CHECKPOINT", "")),
                assoc_runtime_replay_hard_margin_gate=bool(config.get("ASSOC_RUNTIME_REPLAY_HARD_MARGIN_GATE", False)),
                assoc_runtime_replay_margin_threshold=config.get("ASSOC_RUNTIME_REPLAY_MARGIN_THRESHOLD", None),
                assoc_use_competition=bool(config.get("ASSOC_USE_COMPETITION", False)),
                assoc_competition_checkpoint=str(config.get("ASSOC_COMPETITION_CHECKPOINT", "")),
                assoc_competition_topk=int(config.get("ASSOC_COMPETITION_TOPK", 3)),
                assoc_competition_delta_scale=float(config.get("ASSOC_COMPETITION_DELTA_SCALE", 0.05)),
                assoc_competition_mode=str(config.get("ASSOC_COMPETITION_MODE", "rerank_only")),
                assoc_competition_hard_action=bool(config.get("ASSOC_COMPETITION_HARD_ACTION", True)),
                assoc_competition_margin_threshold=config.get("ASSOC_COMPETITION_MARGIN_THRESHOLD", None),
                assoc_use_competition_oracle=bool(config.get("ASSOC_USE_COMPETITION_ORACLE", False)),
                assoc_competition_oracle_csv=str(config.get("ASSOC_COMPETITION_ORACLE_CSV", "")),
                assoc_use_local_conflict_graph=bool(config.get("ASSOC_USE_LOCAL_CONFLICT_GRAPH", False)),
                assoc_local_conflict_graph_mode=str(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_MODE", "disabled")),
                assoc_use_local_conflict_graph_oracle=bool(config.get("ASSOC_USE_LOCAL_CONFLICT_GRAPH_ORACLE", False)),
                assoc_local_conflict_graph_oracle_jsonl=str(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_ORACLE_JSONL", "")),
                assoc_local_conflict_graph_checkpoint=str(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_CHECKPOINT", "")),
                assoc_local_conflict_graph_topk=int(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_TOPK", 8)),
                assoc_local_conflict_graph_min_detections=int(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_MIN_DETECTIONS", 2)),
                assoc_local_conflict_graph_min_committed_matches=int(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_MIN_COMMITTED_MATCHES", 1)
                ),
                assoc_local_conflict_graph_max_detections=int(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_MAX_DETECTIONS", 8)
                ),
                assoc_local_conflict_graph_max_tracks=int(config.get("ASSOC_LOCAL_CONFLICT_GRAPH_MAX_TRACKS", 32)),
                assoc_local_conflict_graph_cluster_gate_thresh=float(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_THRESH", 0.5)
                ),
                assoc_local_conflict_graph_cluster_gate_temp=float(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_TEMP", 1.0)
                ),
                assoc_local_conflict_graph_cluster_gate_bias=float(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_BIAS", 0.0)
                ),
                assoc_local_conflict_graph_host_variant=str(
                    config.get("ASSOC_LOCAL_CONFLICT_GRAPH_HOST_VARIANT", "")
                ),
                assoc_bbox_dist_weight=float(config.get("ASSOC_BBOX_DIST_WEIGHT", 0.0)),
                assoc_bbox_dist_tau=float(config.get("ASSOC_BBOX_DIST_TAU", 1.0)),
                assoc_bbox_dist_use_cal_factor=bool(config.get("ASSOC_BBOX_DIST_USE_CAL_FACTOR", False)),
                assoc_two_stage=bool(config.get("ASSOC_TWO_STAGE", False)),
                assoc_stage2_iou_gate=config.get("ASSOC_STAGE2_IOU_GATE", None),
                assoc_stage2_id_thresh=config.get("ASSOC_STAGE2_ID_THRESH", None),
                assoc_stage2_bbox_weight=config.get("ASSOC_STAGE2_BBOX_WEIGHT", None),
                reid_encoder=reid_encoder,
                use_kalman=bool(config.get("ASSOC_USE_KALMAN", True)),
                dtype=dtype,
                id_label_strategy=str(config.get("ID_LABEL_STRATEGY", "random")),
                sequence_name=seq_key,
                use_confidence_calibration=bool(config.get("USE_CONFIDENCE_CALIBRATION", False)),
                calibration_strength=float(config.get("CALIBRATION_STRENGTH", 0.5)),
                min_confidence=float(config.get("MIN_CONFIDENCE", 0.1)),
                use_tta=bool(config.get("USE_TTA", False)),
                tta_scales=config.get("TTA_SCALES", [0.8, 1.0, 1.2]),
                tta_flip=bool(config.get("TTA_FLIP", True)),
                tta_fusion=str(config.get("TTA_FUSION", "average")),
                use_memory_bank=False,  # attach trained MB below to avoid random-weight inference.
                memory_lambda=float(config.get("MEMORY_LAMBDA", 0.9)),
                memory_update_threshold=float(config.get("MEMORY_UPDATE_THRESHOLD", 0.5)),
                det_source=det_source,
                external_detections=external_detections,
            )

            # Attach trained MemoryBank weights if present; otherwise disable it.
            if bool(config.get("USE_MEMORY_BANK", False)) and hasattr(model_wo_ddp, "memory_bank"):
                try:
                    tracker.memory_bank = getattr(model_wo_ddp, "memory_bank")
                    if tracker.memory_bank is not None:
                        tracker.memory_bank.eval()
                except Exception:
                    tracker.memory_bank = None

            results = []
            for img_name in images:
                img_path = os.path.join(img_dir, img_name)
                frame_id = int(os.path.splitext(img_name)[0])
                track_results = tracker.update(img_path)
                for i in range(len(track_results["id"])):
                    track_id = int(track_results["id"][i].item())
                    if track_id <= 0:
                        continue
                    bbox = track_results["bbox"][i]
                    x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                    conf = float(track_results["score"][i].item())
                    results.append((frame_id, track_id, x, y, w, h, conf))

            output_path = os.path.join(out_tracker_dir, f"{seq_name}.txt")
            write_results(results, output_path)

        if config.get("RUN_TRACKEVAL", True):
            run_trackeval(config, eval_dir)
        return True
    except Exception as e:
        logger.warning(f"[Val] In-process validation failed: {e}")
        if bool(config.get("LOG_EXCEPTION_TRACE", False)):
            logger.warning(traceback.format_exc())
        return False


def build_tracking_modules(config: dict):
    """
    只构建跟踪相关的模块（不包括 DETR 检测器）

    返回：
        trajectory_modeling: 轨迹建模模块
        id_decoder: ID 解码器模块
        id_criterion: ID 损失函数
    """
    from models.motip.id_criterion import build as build_id_criterion

    feature_dim = config.get("FEATURE_DIM", 256)
    num_bands = int(config.get("NUM_FREQ_BANDS", config.get("NUM_BANDS", 4)))

    # 构建频域轨迹建模模块
    if config.get("USE_FREQ_AWARE", True):
        from models.motip.freq_aware_trajectory_modeling import FrequencyAwareTrajectoryModeling
        trajectory_modeling = FrequencyAwareTrajectoryModeling(
            detr_dim=feature_dim,
            feature_dim=feature_dim,
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 2),
            num_bands=num_bands,
            freq_kernel_size=config.get("FREQ_KERNEL_SIZE", 7),
            use_fixed_laplacian=config.get("USE_FIXED_LAPLACIAN", False),
            freq_ortho_metric=config.get("FREQ_ORTHO_METRIC", "dot"),
            use_multiscale_freq=config.get("USE_MULTISCALE_FREQ", False),
            num_freq_scales=config.get("NUM_FREQ_SCALES", 3),
            num_temporal_layers=config.get("NUM_FREQ_TEMPORAL_LAYERS", config.get("NUM_TEMPORAL_LAYERS", 2)),
            temporal_num_heads=config.get("FREQ_TEMPORAL_HEADS", config.get("TEMPORAL_NUM_HEADS", 8)),
            # MAX_SEQ_LEN controls temporal modeling window / positional encoding length.
            # REL_PE_LENGTH is for ID-decoder relative position embedding and should NOT drive FTT length.
            max_seq_len=config.get("MAX_SEQ_LEN", config.get("REL_PE_LENGTH", 30)),
            use_mamba_for_lowfreq=config.get("USE_MAMBA_FOR_LOWFREQ", True),
            use_global_mamba=config.get("USE_GLOBAL_MAMBA", True),
            band_window_sizes=config.get("BAND_WINDOW_SIZES", None),
            use_spatial_freq_interaction=config.get("USE_SPATIAL_FREQ_INTERACTION", False),
            sfi_hidden_ratio=float(config.get("SFI_HIDDEN_RATIO", 2.0)),
            sfi_alpha_init=float(config.get("SFI_ALPHA_INIT", 0.1)),
            dropout=config.get("DROPOUT", 0.1),
            use_occlusion_recovery=config.get("USE_OCCLUSION_RECOVERY", False),
            occlusion_recovery_ratio=config.get("OCCLUSION_RECOVERY_RATIO", 0.3),
            use_adaptive_bands=config.get("USE_ADAPTIVE_BANDS", False),
            min_bands=config.get("MIN_BANDS", 2),
            max_bands=config.get("MAX_BANDS", 8),
            soft_band_temp=config.get("SOFT_BAND_TEMP", 1.0),
            lfd_feature_ortho_weight=float(config.get("LFD_FEATURE_ORTHO_WEIGHT", 0.1)),
        )
    else:
        from models.motip.trajectory_modeling import TrajectoryModeling
        trajectory_modeling = TrajectoryModeling(
            detr_dim=feature_dim,
            feature_dim=feature_dim,
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 2),
            use_freq_adapter=bool(config.get("USE_FREQ_ADAPTER", False)),
        )

    # 构建 ID 解码器
    if config.get("USE_FREQ_DECODER_V2", True) and config.get("USE_FREQ_AWARE", True):
        from models.motip.freq_aware_id_decoder_v2 import FrequencyAwareIDDecoderV2
        use_freq_guided_assoc = config.get(
            "USE_FREQ_GUIDED_ASSOC",
            config.get("USE_FREQ_GUIDED_ASSOCIATION", True),
        )
        id_decoder = FrequencyAwareIDDecoderV2(
            feature_dim=feature_dim,
            id_dim=config.get("ID_DIM", feature_dim),
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 2),
            num_layers=config.get("NUM_ID_DECODER_LAYERS", 6),
            head_dim=config.get("HEAD_DIM", 32),
            num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 500),
            rel_pe_length=config.get("REL_PE_LENGTH", 30),
            use_aux_loss=config.get("USE_AUX_LOSS", True),
            use_shared_aux_head=config.get("USE_SHARED_AUX_HEAD", True),
            num_bands=num_bands,
            use_freq_guided_association=use_freq_guided_assoc,
            use_learnable_fusion=config.get("USE_LEARNABLE_FUSION", True),
            freq_loss_weight=config.get("FREQ_LOSS_WEIGHT", 1.0),
            fusion_loss_weight=config.get("FUSION_LOSS_WEIGHT", 1.0),
            use_mamba_self_attn=config.get("USE_MAMBA_IN_ID_DECODER", True),
            label_smoothing=float(config.get("LABEL_SMOOTHING", 0.0)),
            use_confidence_calibration=config.get("USE_CONFIDENCE_CALIBRATION", False),
            calibration_strength=config.get("CALIBRATION_STRENGTH", 0.5),
            min_confidence=config.get("MIN_CONFIDENCE", 0.1),
            use_newborn_head=config.get("USE_NEWBORN_HEAD", False),
            newborn_head_dim=config.get("NEWBORN_HEAD_DIM", 128),
        )
    else:
        from models.motip.id_decoder import IDDecoder
        id_decoder = IDDecoder(
            feature_dim=feature_dim,
            id_dim=config.get("ID_DIM", feature_dim),
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 2),
            num_layers=config.get("NUM_ID_DECODER_LAYERS", 6),
            head_dim=config.get("HEAD_DIM", 32),
            num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 500),
            rel_pe_length=config.get("REL_PE_LENGTH", 30),
            use_aux_loss=config.get("USE_AUX_LOSS", True),
            use_shared_aux_head=config.get("USE_SHARED_AUX_HEAD", True),
        )

    # 构建 ID 损失函数
    id_criterion = build_id_criterion(config)

    # Optional: Triplet loss (Top-Conference strategy)
    triplet_criterion = None
    if config.get("USE_TRIPLET_LOSS", False):
        if _TOPCONF_AVAILABLE:
            triplet_criterion = build_triplet_loss(config)
        else:
            print("[WARNING] USE_TRIPLET_LOSS=True but topconf_losses is unavailable; TripletLoss is disabled.")

    # Optional: det<->track matching supervision (InfoNCE-style)
    det_track_match_criterion = None
    if config.get("USE_DET_TRACK_MATCH_LOSS", False):
        if _MATCHLOSS_AVAILABLE:
            det_track_match_criterion = build_det_track_match_loss(config)
        else:
            print(
                "[WARNING] USE_DET_TRACK_MATCH_LOSS=True but matching_losses is unavailable; "
                "DetTrackMatchLoss is disabled."
            )

    return trajectory_modeling, id_decoder, id_criterion, triplet_criterion, det_track_match_criterion


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def _cxcywh_to_xyxy_pixel(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = (cx - 0.5 * w) * img_w
    y1 = (cy - 0.5 * h) * img_h
    x2 = (cx + 0.5 * w) * img_w
    y2 = (cy + 0.5 * h) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def _apply_box_noise_cxcywh(
    boxes: torch.Tensor,
    noise_std: float,
    prob: float = 1.0,
    min_size: float = 1e-3,
    max_size: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply Gaussian noise to normalized cxcywh boxes.
    """
    if boxes.numel() == 0 or noise_std <= 0:
        return boxes
    if prob <= 0:
        return boxes
    noise = torch.randn_like(boxes) * noise_std
    if mask is None:
        use = torch.rand((boxes.shape[0],), device=boxes.device) < prob
    else:
        use = mask & (torch.rand((boxes.shape[0],), device=boxes.device) < prob)
    if not use.any():
        return boxes
    boxes_noisy = boxes.clone()
    boxes_noisy[use] = boxes_noisy[use] + noise[use]
    boxes_noisy[:, 0:2] = boxes_noisy[:, 0:2].clamp(0.0, 1.0)
    boxes_noisy[:, 2:4] = boxes_noisy[:, 2:4].clamp(min_size, max_size)
    return boxes_noisy


_linear_sum_assignment = None


def _get_linear_sum_assignment():
    global _linear_sum_assignment
    if _linear_sum_assignment is None:
        try:
            from scipy.optimize import linear_sum_assignment
            _linear_sum_assignment = linear_sum_assignment
        except Exception:
            _linear_sum_assignment = None
    return _linear_sum_assignment


def _det_cache_path(image_path: str, cfg: ByteTrackFeatureConfig, cache_dir: str, version: str) -> str:
    key_str = f"{image_path}|{cfg.exp_file}|{cfg.ckpt}|{cfg.conf_thre}|{cfg.nms_thre}|{cfg.test_size}|{cfg.roi_size}|{version}"
    key = hashlib.md5(key_str.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.npz")


def _load_det_cache(cache_path: str, image_path: str, validate_mtime: bool = True):
    if not os.path.exists(cache_path):
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if validate_mtime and os.path.exists(image_path):
            mtime = os.path.getmtime(image_path)
            size = os.path.getsize(image_path)
            if float(data.get("mtime", -1)) != float(mtime) or int(data.get("size", -1)) != int(size):
                return None
        dets = data["dets"]  # (N, 5) xywh + conf
        feats = data["features"]  # (N, C)
        return dets, feats
    except Exception:
        return None


def _save_det_cache(cache_path: str, image_path: str, dets: np.ndarray, features: np.ndarray, compress: bool = True):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mtime = os.path.getmtime(image_path) if os.path.exists(image_path) else -1
    size = os.path.getsize(image_path) if os.path.exists(image_path) else -1
    if compress:
        np.savez_compressed(cache_path, dets=dets, features=features, mtime=mtime, size=size)
    else:
        np.savez(cache_path, dets=dets, features=features, mtime=mtime, size=size)


def match_detections_to_gt(
    det_boxes_xywh: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    img_w: int,
    img_h: int,
    iou_thresh: float,
    method: str = "greedy",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if det_boxes_xywh.numel() == 0 or gt_boxes_cxcywh.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=det_boxes_xywh.device),
            torch.zeros((0,), dtype=torch.long, device=det_boxes_xywh.device),
        )
    det_xyxy = _xywh_to_xyxy(det_boxes_xywh)
    gt_xyxy = _cxcywh_to_xyxy_pixel(gt_boxes_cxcywh, img_w, img_h)
    iou = _box_iou(det_xyxy, gt_xyxy)
    method = str(method).lower()
    det_indices = []
    gt_indices = []
    if method == "hungarian":
        lsa = _get_linear_sum_assignment()
        if lsa is not None:
            cost = (-iou).detach().cpu().numpy()
            rows, cols = lsa(cost)
            for r, c in zip(rows, cols):
                if iou[r, c] >= iou_thresh:
                    det_indices.append(int(r))
                    gt_indices.append(int(c))
        else:
            method = "greedy"

    if method == "greedy":
        iou_work = iou.clone()
        while True:
            max_val, max_idx = torch.max(iou_work.view(-1), dim=0)
            if max_val.item() < iou_thresh:
                break
            d = int(max_idx.item() // iou_work.shape[1])
            g = int(max_idx.item() % iou_work.shape[1])
            det_indices.append(d)
            gt_indices.append(g)
            iou_work[d, :] = -1
            iou_work[:, g] = -1
            if iou_work.numel() == 0:
                break
    if len(det_indices) == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=det_boxes_xywh.device),
            torch.zeros((0,), dtype=torch.long, device=det_boxes_xywh.device),
        )
    return (
        torch.tensor(det_indices, dtype=torch.long, device=det_boxes_xywh.device),
        torch.tensor(gt_indices, dtype=torch.long, device=det_boxes_xywh.device),
    )


class ByteTrackTrainingModel(torch.nn.Module):
    """
    封装 ByteTrack 特征提取器和频域关联模块的训练模型
    """
    def __init__(
        self,
        feature_extractor: ByteTrackFeatureExtractor,
        trajectory_modeling: torch.nn.Module,
        id_decoder: torch.nn.Module,
        laplace_assoc: Optional[torch.nn.Module] = None,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.laplace_assoc = laplace_assoc
        self.feature_dim = feature_dim

        # 冻结特征提取器的 YOLOX 部分
        for param in self.feature_extractor.yolox_model.parameters():
            param.requires_grad = False

    def forward(self, seq_info: dict, part: str, **kwargs):
        """
        分部分前向传播
        """
        if part == "trajectory_modeling":
            return self.trajectory_modeling(seq_info)
        elif part == "id_decoder":
            return self.id_decoder(
                seq_info,
                use_decoder_checkpoint=kwargs.get("use_decoder_checkpoint", False),
            )
        else:
            raise ValueError(f"Unknown part: {part}")


def prepare_for_motip_bytetrack(
    features: torch.Tensor,
    boxes: torch.Tensor,
    annotations: list,
    feature_dim: int,
    device: torch.device,
):
    """
    准备 MOTIP 训练所需的数据结构（使用 ByteTrack 特征）

    参数：
        features: (B*T, N, feature_dim) - ByteTrack 提取的特征
        boxes: (B*T, N, 4) - 归一化的 cxcywh 框
        annotations: 标注信息
        feature_dim: 特征维度
        device: 设备

    返回：
        seq_info: dict - MOTIP 训练所需的数据结构
    """
    _B, _T = len(annotations), len(annotations[0])
    _G, _, _N = annotations[0][0]["trajectory_id_labels"].shape

    # 初始化张量
    trajectory_id_labels = -torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=device)
    trajectory_times = -torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=device)
    trajectory_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=device)
    trajectory_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=device)
    trajectory_features = torch.zeros((_B, _G, _T, _N, feature_dim), dtype=torch.float32, device=device)

    unknown_id_labels = -torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=device)
    unknown_times = -torch.ones((_B, _G, _T, _N), dtype=torch.int64, device=device)
    unknown_masks = torch.ones((_B, _G, _T, _N), dtype=torch.bool, device=device)
    unknown_boxes = torch.zeros((_B, _G, _T, _N, 4), dtype=torch.float32, device=device)
    unknown_features = torch.zeros((_B, _G, _T, _N, feature_dim), dtype=torch.float32, device=device)

    for b in range(_B):
        for t in range(_T):
            flatten_idx = b * _T + t

            # 从 ByteTrack 特征中获取
            frame_features = features[flatten_idx]  # (N, feature_dim)
            frame_boxes = boxes[flatten_idx]  # (N, 4)
            _num_dets = frame_features.shape[0]

            for group in range(_G):
                _curr_traj_ann_idxs = annotations[b][t]["trajectory_ann_idxs"][group, 0, :]
                _curr_unk_ann_idxs = annotations[b][t]["unknown_ann_idxs"][group, 0, :]
                _curr_traj_masks = annotations[b][t]["trajectory_id_masks"][group, 0, :]
                _curr_unk_masks = annotations[b][t]["unknown_id_masks"][group, 0, :]

                # 填充标签和时间
                trajectory_id_labels[b, group, t] = annotations[b][t]["trajectory_id_labels"][group, 0, :]
                unknown_id_labels[b, group, t] = annotations[b][t]["unknown_id_labels"][group, 0, :]
                trajectory_times[b, group, t] = annotations[b][t]["trajectory_times"][group, 0, :]
                unknown_times[b, group, t] = annotations[b][t]["unknown_times"][group, 0, :]
                trajectory_masks[b, group, t] = _curr_traj_masks
                unknown_masks[b, group, t] = _curr_unk_masks

                # 填充特征和框
                _valid_traj = ~_curr_traj_masks
                _traj_idxs = _curr_traj_ann_idxs[_valid_traj]
                _traj_in_bounds = _traj_idxs < _num_dets
                if _traj_in_bounds.any():
                    _valid_traj_positions = _valid_traj.nonzero(as_tuple=True)[0][_traj_in_bounds]
                    trajectory_features[b, group, t, _valid_traj_positions] = frame_features[_traj_idxs[_traj_in_bounds]]
                    trajectory_boxes[b, group, t, _valid_traj_positions] = frame_boxes[_traj_idxs[_traj_in_bounds]]

                _valid_unk = ~_curr_unk_masks
                _unk_idxs = _curr_unk_ann_idxs[_valid_unk]
                _unk_in_bounds = _unk_idxs < _num_dets
                if _unk_in_bounds.any():
                    _valid_unk_positions = _valid_unk.nonzero(as_tuple=True)[0][_unk_in_bounds]
                    unknown_features[b, group, t, _valid_unk_positions] = frame_features[_unk_idxs[_unk_in_bounds]]
                    unknown_boxes[b, group, t, _valid_unk_positions] = frame_boxes[_unk_idxs[_unk_in_bounds]]

    return {
        "trajectory_id_labels": trajectory_id_labels,
        "trajectory_times": trajectory_times,
        "trajectory_masks": trajectory_masks,
        "trajectory_boxes": trajectory_boxes,
        "trajectory_features": trajectory_features,
        "unknown_id_labels": unknown_id_labels,
        "unknown_times": unknown_times,
        "unknown_masks": unknown_masks,
        "unknown_boxes": unknown_boxes,
        "unknown_features": unknown_features,
    }


def train_engine(config: dict):
    """主训练循环"""

    assert "EXP_NAME" in config and config["EXP_NAME"] is not None, "Please set the experiment name."

    train_dataset_name = None
    if isinstance(config.get("DATASETS", None), (list, tuple)) and len(config["DATASETS"]) > 0:
        train_dataset_name = config["DATASETS"][0]
    config, selected_profile = resolve_bytetrack_profile(
        config=config,
        dataset_name=train_dataset_name,
        explicit_profile=config.get("BYTETRACK_PROFILE_TRAIN", None),
    )

    det_source_train = str(config.get("BYTETRACK_DET_SOURCE", "model")).lower()
    if det_source_train in ("txt", "public"):
        det_source_train = "external"
    if det_source_train not in ("model", "external"):
        det_source_train = "model"
    external_det_cfg = config if det_source_train == "external" else None

    outputs_dir = config["OUTPUTS_DIR"] if config["OUTPUTS_DIR"] is not None \
        else os.path.join("./outputs/", config["EXP_NAME"])

    # 初始化 Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    state = PartialState()

    set_seed(config["SEED"])
    torch.multiprocessing.set_sharing_strategy('file_system')

    # 初始化 Logger
    logger = Logger(
        logdir=os.path.join(outputs_dir, "train"),
        use_wandb=config.get("USE_WANDB", False),
        config=config,
        exp_owner=config.get("EXP_OWNER", ""),
        exp_project=config.get("EXP_PROJECT", ""),
        exp_group=config.get("EXP_GROUP", ""),
        exp_name=config["EXP_NAME"],
    )
    logger.info(f"ByteTrack Feature Training initialized at {logger.logdir}")
    if selected_profile is not None:
        logger.info(f"Using detector profile: {selected_profile}")
    logger.info(f"[DET] det_source={det_source_train}")
    logger.config(config=config)

    # 保存配置
    import yaml
    if accelerator.is_main_process:
        config_save_path = os.path.join(outputs_dir, "config_effective.yaml")
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved effective config to: {config_save_path}")
    accelerator.wait_for_everyone()

    # 构建数据集
    train_dataset = build_dataset(config=config, is_validation=False)
    logger.dataset(train_dataset)

    # 构建采样器和数据加载器
    if "DATASET_WEIGHTS" in config:
        data_weights = defaultdict(lambda: defaultdict())
        for _ in range(len(config["DATASET_WEIGHTS"])):
            data_weights[config["DATASETS"][_]][config["DATASET_SPLITS"][_]] = config["DATASET_WEIGHTS"][_]
        data_weights = dict(data_weights)
    else:
        data_weights = None

    train_sampler = NaiveSampler(
        data_source=train_dataset,
        sample_steps=config["SAMPLE_STEPS"],
        sample_lengths=config["SAMPLE_LENGTHS"],
        sample_intervals=config["SAMPLE_INTERVALS"],
        length_per_iteration=config["LENGTH_PER_ITERATION"],
        data_weights=data_weights,
        min_legal_ratio=config.get("MIN_LEGAL_RATIO", 1.0),
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS"],
        prefetch_factor=config["PREFETCH_FACTOR"] if config["NUM_WORKERS"] > 0 else None,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 初始化训练状态
    train_states = {
        "start_epoch": 0,
        "global_step": 0
    }

    # 构建 ByteTrack 特征提取器
    bytetrack_cfg = ByteTrackFeatureConfig(
        exp_file=config["BYTETRACK_EXP_FILE"],
        ckpt=config["BYTETRACK_CKPT"],
        fp16=config.get("BYTETRACK_FP16", True),
        test_size=config.get("BYTETRACK_TEST_SIZE", None),
        conf_thre=config.get("BYTETRACK_CONF_THRE", 0.01),
        nms_thre=config.get("BYTETRACK_NMS_THRE", 0.7),
        feature_dim=config.get("FEATURE_DIM", 256),
        canonical_scale=config.get("CANONICAL_SCALE", 224.0),
        canonical_level=config.get("CANONICAL_LEVEL", 4),
        roi_multi_level=config.get("ROI_MULTI_LEVEL", False),
        roi_level_fusion=config.get("ROI_LEVEL_FUSION", "scale"),
    )

    device = accelerator.device
    feature_extractor = ByteTrackFeatureExtractor(bytetrack_cfg, device)
    logger.success(f"Loaded ByteTrack feature extractor from {config['BYTETRACK_CKPT']}")

    # 构建跟踪模块
    trajectory_modeling, id_decoder, id_criterion, triplet_criterion, det_track_match_criterion = build_tracking_modules(config)
    laplace_assoc = None
    if bool(config.get("ASSOC_USE_LAPLACE", False)):
        laplace_assoc = LaplaceAssociationAdapter(
            decay_scales=config.get("ASSOC_LAPLACE_DECAY_SCALES", [1.0, 2.0, 4.0]),
            hidden_dim=int(config.get("ASSOC_LAPLACE_HIDDEN_DIM", 16)),
            blend=float(config.get("ASSOC_LAPLACE_WEIGHT", 0.35)),
        )

    # 封装为训练模型
    model = ByteTrackTrainingModel(
        feature_extractor=feature_extractor,
        trajectory_modeling=trajectory_modeling,
        id_decoder=id_decoder,
        laplace_assoc=laplace_assoc,
        feature_dim=config.get("FEATURE_DIM", 256),
    )
    model = model.to(device)

    # Optional memory bank (training)
    memory_bank = None
    if config.get("USE_MEMORY_BANK", False):
        if _TOPCONF_AVAILABLE and build_memory_bank is not None:
            memory_bank = build_memory_bank(config).to(device)
            logger.info("✓ MemoryBank enabled for training")
        else:
            logger.warning("USE_MEMORY_BANK=True but topconf_losses unavailable; MemoryBank disabled.")
    if memory_bank is not None:
        # register as a submodule so its params are optimized/synchronized
        model.memory_bank = memory_bank

    # 验证 YOLOX 确实被冻结（调试用）
    yolox_params = [p for p in model.feature_extractor.yolox_model.parameters()]
    yolox_trainable = [p for p in yolox_params if p.requires_grad]
    if len(yolox_trainable) > 0:
        logger.warning(f"WARNING: YOLOX has {len(yolox_trainable)} trainable parameters!")
    else:
        logger.info(f"✓ YOLOX is frozen ({len(yolox_params)} parameters)")

    # 可选：冻结特征投影层（严格冻结检测特征分布）
    if config.get("FREEZE_FEATURE_PROJ", False):
        for param in model.feature_extractor.feature_proj.parameters():
            param.requires_grad = False
        logger.info("✓ Feature projection is frozen (FREEZE_FEATURE_PROJ=True)")

    # 构建优化器（只优化可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        params=trainable_params,
        lr=config["LR"],
        weight_decay=config["WEIGHT_DECAY"],
    )

    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config["SCHEDULER_MILESTONES"],
        gamma=config["SCHEDULER_GAMMA"],
    )

    def _load_checkpoint_filtered(
        *,
        model,
        path: str,
        skip_prefixes: tuple[str, ...],
        states: Optional[dict] = None,
        optimizer=None,
        scheduler=None,
    ):
        """
        Load a checkpoint but skip some key prefixes.

        This is useful for detector-aligned fine-tuning:
        - We want to resume trajectory_modeling/id_decoder/feature_proj
        - But keep the detector (YOLOX) weights from BYTETRACK_CKPT
        """
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        model_state_dict = model.state_dict()
        loadable_state = {}
        skipped = 0
        shape_mismatch = 0
        missing_in_model = 0
        for k, v in state_dict.items():
            if skip_prefixes and any(k.startswith(p) for p in skip_prefixes):
                skipped += 1
                continue
            if k not in model_state_dict:
                missing_in_model += 1
                continue
            if model_state_dict[k].shape != v.shape:
                shape_mismatch += 1
                continue
            loadable_state[k] = v

        incompatible = model.load_state_dict(loadable_state, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])

        logger.info(
            f"[Resume] Loaded {len(loadable_state)} keys from {path} "
            f"(skipped={skipped}, shape_mismatch={shape_mismatch}, missing_in_model={missing_in_model}, "
            f"missing_in_ckpt={len(missing)}, unexpected={len(unexpected)})"
        )
        if len(missing) > 0:
            logger.info(f"[Resume] Missing keys (sample): {missing[:5]}")
        if len(unexpected) > 0:
            logger.info(f"[Resume] Unexpected keys (sample): {unexpected[:5]}")

        if states is not None and isinstance(ckpt, dict) and isinstance(ckpt.get("states", None), dict):
            states.update(ckpt["states"])

        if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer", None) is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler", None) is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        return

    # 恢复训练
    if config.get("RESUME_MODEL") is not None:
        resume_model_path = str(config["RESUME_MODEL"])
        resume_optimizer = optimizer if config.get("RESUME_OPTIMIZER", False) else None
        resume_scheduler = scheduler if config.get("RESUME_SCHEDULER", False) else None
        resume_only_weights = bool(config.get("RESUME_ONLY_WEIGHTS", False))
        resume_skip_yolox = bool(config.get("RESUME_SKIP_YOLOX", False))

        states_arg = None if resume_only_weights else train_states

        if resume_skip_yolox:
            _load_checkpoint_filtered(
                model=model,
                path=resume_model_path,
                skip_prefixes=("feature_extractor.yolox_model.",),
                states=states_arg,
                optimizer=resume_optimizer,
                scheduler=resume_scheduler,
            )
        else:
            load_checkpoint(
                model=model,
                path=resume_model_path,
                optimizer=resume_optimizer,
                scheduler=resume_scheduler,
                states=states_arg,
            )

        if resume_only_weights:
            train_states["start_epoch"] = 0
            train_states["global_step"] = 0
            logger.info("[Resume] RESUME_ONLY_WEIGHTS=True -> reset start_epoch/global_step to 0.")

        logger.success(f"Resumed from {resume_model_path}")

    # 使用 Accelerator 准备
    train_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, model, optimizer,
    )

    feature_dim = config.get("FEATURE_DIM", 256)

    # 训练循环
    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        logger.info(f"Start training epoch {epoch}")
        epoch_start_timestamp = TPS.timestamp()
        train_sampler.prepare_for_epoch(epoch=epoch)

        # LFD importance temperature schedule
        if config.get("USE_IMPORTANCE_TAU_SCHEDULE", True):
            tau_start = float(config.get("IMPORTANCE_TAU_START", 2.0))
            tau_end = float(config.get("IMPORTANCE_TAU_END", 1.0))
            denom = max(config["EPOCHS"] - 1, 1)
            prog = float(epoch) / float(denom)
            importance_tau = tau_start + (tau_end - tau_start) * prog
        else:
            importance_tau = float(config.get("IMPORTANCE_TAU", 1.0))

        importance_tau_set_failed = False
        def _set_importance_tau(m):
            nonlocal importance_tau_set_failed
            if hasattr(m, "importance_tau"):
                try:
                    m.importance_tau.fill_(float(importance_tau))
                except Exception as e:
                    importance_tau_set_failed = True
                    if bool(config.get("DEBUG_RAISE_EXCEPTIONS", False)):
                        raise

        model.apply(_set_importance_tau)
        if importance_tau_set_failed and accelerator.is_main_process:
            logger.warning("[LFD] Failed to set importance_tau on some modules; check module buffers/dtypes.")
        if accelerator.is_main_process:
            logger.info(f"[LFD] importance_tau={importance_tau:.4f}")

        # DET train prob schedule
        det_train_prob = float(config.get("DET_TRAIN_PROB", 0.5))
        if config.get("DET_TRAIN_PROB_SCHEDULE", False):
            start_p = float(config.get("DET_TRAIN_PROB_START", det_train_prob))
            end_p = float(config.get("DET_TRAIN_PROB_END", det_train_prob))
            denom = max(config["EPOCHS"] - 1, 1)
            prog = float(epoch) / float(denom)
            det_train_prob = start_p + (end_p - start_p) * prog

        # Optional train box source schedule
        train_box_source = config.get("TRAIN_BOX_SOURCE", "gt")
        schedule = config.get("TRAIN_BOX_SOURCE_SCHEDULE", None)
        if isinstance(schedule, list):
            for item in schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    train_box_source = item.get("source", train_box_source)
                    det_train_prob = float(item.get("prob", det_train_prob))
                    break

        if accelerator.is_main_process:
            logger.info(f"[DET] train_box_source={train_box_source} det_prob={det_train_prob:.3f}")


        # Loss weight schedules
        freq_ortho_loss_weight = float(config.get("FREQ_ORTHO_LOSS_WEIGHT", 0.3))
        ortho_schedule = config.get("FREQ_ORTHO_LOSS_WEIGHT_SCHEDULE", None)
        if isinstance(ortho_schedule, list):
            for item in ortho_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    freq_ortho_loss_weight = float(item.get("weight", freq_ortho_loss_weight))
                    break

        freq_consistency_loss_weight = float(config.get("FREQ_CONSISTENCY_LOSS_WEIGHT", 0.1))
        consist_schedule = config.get("FREQ_CONSISTENCY_LOSS_WEIGHT_SCHEDULE", None)
        if isinstance(consist_schedule, list):
            for item in consist_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    freq_consistency_loss_weight = float(item.get("weight", freq_consistency_loss_weight))
                    break

        freq_energy_loss_weight = float(config.get("FREQ_ENERGY_BALANCE_WEIGHT", 0.0))
        energy_schedule = config.get("FREQ_ENERGY_BALANCE_WEIGHT_SCHEDULE", None)
        if isinstance(energy_schedule, list):
            for item in energy_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    freq_energy_loss_weight = float(item.get("weight", freq_energy_loss_weight))
                    break

        newborn_penalty_weight = float(config.get("NEWBORN_PENALTY_WEIGHT", 0.0))
        newborn_schedule = config.get("NEWBORN_PENALTY_WEIGHT_SCHEDULE", None)
        if isinstance(newborn_schedule, list):
            for item in newborn_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    newborn_penalty_weight = float(item.get("weight", newborn_penalty_weight))
                    break

        triplet_loss_weight = float(config.get("TRIPLET_LOSS_WEIGHT", 0.0))
        triplet_schedule = config.get("TRIPLET_LOSS_WEIGHT_SCHEDULE", None)
        if isinstance(triplet_schedule, list):
            for item in triplet_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    triplet_loss_weight = float(item.get("weight", triplet_loss_weight))
                    break

        det_track_match_weight = float(config.get("DET_TRACK_MATCH_LOSS_WEIGHT", 0.0))
        det_track_match_schedule = config.get("DET_TRACK_MATCH_LOSS_WEIGHT_SCHEDULE", None)
        if isinstance(det_track_match_schedule, list):
            for item in det_track_match_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    det_track_match_weight = float(item.get("weight", det_track_match_weight))
                    break

        laplace_loss_weight = float(config.get("LAPLACE_LOSS_WEIGHT", 0.0))
        laplace_schedule = config.get("LAPLACE_LOSS_WEIGHT_SCHEDULE", None)
        if isinstance(laplace_schedule, list):
            for item in laplace_schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start", 0))
                end = int(item.get("end", config["EPOCHS"] - 1))
                if start <= epoch <= end:
                    laplace_loss_weight = float(item.get("weight", laplace_loss_weight))
                    break

        if accelerator.is_main_process:
            logger.info(
                f"[LOSS] freq_ortho_weight={freq_ortho_loss_weight:.2f} "
                f"freq_consist_weight={freq_consistency_loss_weight:.2f} "
                f"freq_energy_weight={freq_energy_loss_weight:.2f} "
                f"newborn_weight={newborn_penalty_weight:.2f} "
                f"triplet_weight={triplet_loss_weight:.2f} "
                f"det_track_match_weight={det_track_match_weight:.2f} "
                f"laplace_weight={laplace_loss_weight:.2f}"
            )

        # 训练一个 epoch
        train_metrics = train_one_epoch(
            accelerator=accelerator,
            logger=logger,
            states=train_states,
            epoch=epoch,
            dataloader=train_dataloader,
            model=model,
            id_criterion=id_criterion,
            optimizer=optimizer,
            feature_dim=feature_dim,
            normalize_id_loss_by_weight_sum=bool(config.get("ID_LOSS_NORMALIZE_BY_WEIGHT_SUM", True)),
            lr_warmup_epochs=config["LR_WARMUP_EPOCHS"],
            lr_warmup_tgt_lr=config["LR"],
            use_decoder_checkpoint=config.get("USE_DECODER_CHECKPOINT", False),
            accumulate_steps=config.get("ACCUMULATE_STEPS", 1),
            max_clip_norm=config.get("MAX_CLIP_NORM", 0.1),
            freq_ortho_loss_weight=freq_ortho_loss_weight,
            freq_consistency_loss_weight=freq_consistency_loss_weight,
            freq_energy_loss_weight=freq_energy_loss_weight,
            triplet_criterion=triplet_criterion,
            triplet_loss_weight=triplet_loss_weight,
            det_track_match_criterion=det_track_match_criterion,
            det_track_match_loss_weight=det_track_match_weight,
            laplace_loss_weight=laplace_loss_weight,
            laplace_loss_temperature=float(config.get("LAPLACE_LOSS_TEMPERATURE", 1.0)),
            laplace_background_weight=float(config.get("LAPLACE_BACKGROUND_WEIGHT", 0.25)),
            laplace_bbox_tau=float(config.get("LAPLACE_BBOX_TAU", 1.0)),
            laplace_min_history=int(config.get("LAPLACE_MIN_HISTORY", 1)),
            use_tp_drop_fp_insert=config.get("USE_TP_DROP_FP_INSERT", False),
            tp_drop_ratio=float(config.get("TP_DROP_RATIO", 0.1)),
            fp_insert_ratio=float(config.get("FP_INSERT_RATIO", 0.3)),
            memory_bank=memory_bank,
            use_memory_bank=bool(config.get("USE_MEMORY_BANK", False)),
            train_box_source=train_box_source,
            det_train_start_epoch=config.get("DET_TRAIN_START_EPOCH", 0),
            det_train_prob=det_train_prob,
            det_match_iou_thresh=config.get("DET_MATCH_IOU_THRESH", 0.5),
            det_matching=config.get("DET_MATCHING", "greedy"),
            det_fallback_to_gt=config.get("DET_FALLBACK_TO_GT_ON_EMPTY", True),
            det_max_per_frame=config.get("DET_MAX_PER_FRAME", 0),
            det_fill_unmatched_with_gt=config.get("DET_FILL_UNMATCHED_WITH_GT", True),
            gt_box_noise_scale=config.get("GT_BOX_NOISE_SCALE", 0.0),
            det_cache_use=config.get("DET_CACHE_USE", False),
            det_cache_write=config.get("DET_CACHE_WRITE", False),
            det_cache_dir=config.get("DET_CACHE_DIR", "outputs/det_cache"),
            det_cache_compress=config.get("DET_CACHE_COMPRESS", True),
            det_cache_version=str(config.get("DET_CACHE_VERSION", "v1")),
            det_cache_validate_mtime=config.get("DET_CACHE_VALIDATE_MTIME", True),
            use_aug_images=config.get("BYTETRACK_USE_AUG_IMAGES", False),
            assert_bbox_normalized=config.get("ASSERT_BBOX_NORMALIZED", False),
            bbox_norm_eps=float(config.get("BBOX_NORM_EPS", 1e-3)),
            newborn_penalty_weight=newborn_penalty_weight,
            newborn_penalty_margin=config.get("NEWBORN_PENALTY_MARGIN", 0.0),
            newborn_penalty_warmup_epochs=config.get("NEWBORN_PENALTY_WARMUP_EPOCHS", 0),
            num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 500),
            log_exception_trace=config.get("LOG_EXCEPTION_TRACE", False),
            debug_raise_exceptions=config.get("DEBUG_RAISE_EXCEPTIONS", False),
            det_unknown_supervision=config.get("DET_UNKNOWN_SUPERVISION", False),
            det_source=det_source_train,
            external_det_cfg=external_det_cfg,
            logging_interval=int(config.get("LOGGING_INTERVAL", 20)),
            log_freq_stats=bool(config.get("LOG_FREQ_STATS", False)),
            log_occlusion_stats=bool(config.get("LOG_OCCLUSION_STATS", False)),
            log_calib_stats=bool(config.get("LOG_CALIB_STATS", False)),
        )

        # 记录学习率
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        train_metrics["lr"].update(lr)
        train_metrics["lr"].sync()

        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.metrics(
            log=f"[Finish epoch: {epoch}] [Time: {time_per_epoch}] ",
            metrics=train_metrics,
            fmt="{global_average:.4f}",
            statistic="global_average",
            global_step=train_states["global_step"],
            prefix="epoch",
            x_axis_step=epoch,
            x_axis_name="epoch",
        )

        # 保存检查点
        if (epoch + 1) % config.get("SAVE_CHECKPOINT_PER_EPOCH", 5) == 0:
            save_checkpoint(
                model=model,
                path=os.path.join(outputs_dir, f"checkpoint_{epoch}.pth"),
                states=train_states,
                optimizer=optimizer,
                scheduler=scheduler,
                only_detr=False,
            )
            logger.success(f"Saved checkpoint at epoch {epoch}")

        # 每个 epoch 评估（可选）
        if accelerator.is_main_process and config.get("EVAL_EVERY_EPOCH", False):
            try:
                eval_dir = os.path.join(outputs_dir, "val", f"epoch_{epoch}")
                os.makedirs(eval_dir, exist_ok=True)
                # 保存一个用于评估的 checkpoint
                eval_ckpt = os.path.join(outputs_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(
                    model=model,
                    path=eval_ckpt,
                    states=train_states,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    only_detr=False,
                )
                logger.info(f"Saved eval checkpoint at epoch {epoch}")

                eval_log = os.path.join(eval_dir, "eval.log")
                logger.info(f"Running validation for epoch {epoch} (in-process)...")
                ok = _run_validation_inprocess(
                    accelerator=accelerator,
                    model=model,
                    config=config,
                    eval_dir=eval_dir,
                    logger=logger,
                )
                # Keep a minimal marker log file for tooling compatibility.
                try:
                    with open(eval_log, "w", encoding="utf-8") as f:
                        f.write("ok\n" if ok else "failed\n")
                except Exception:
                    pass
                logger.info(f"Validation finished for epoch {epoch}. ok={ok} log={eval_log}")
                try:
                    _write_validation_diagnostics(eval_dir, config, logger)
                except Exception as e:
                    logger.warning(f"[Diag] failed at epoch {epoch}: {e}")
            except Exception as e:
                logger.warning(f"Validation failed at epoch {epoch}: {e}")

        scheduler.step()
        logger.success(f"Finish training epoch {epoch}")

    # 保存最终模型
    save_checkpoint(
        model=model,
        path=os.path.join(outputs_dir, "checkpoint_final.pth"),
        states=train_states,
        optimizer=optimizer,
        scheduler=scheduler,
        only_detr=False,
    )
    logger.success("Training completed!")


def train_one_epoch(
    accelerator: Accelerator,
    logger: Logger,
    states: dict,
    epoch: int,
    dataloader: DataLoader,
    model,
    id_criterion,
    optimizer,
    feature_dim: int,
    lr_warmup_epochs: int,
    lr_warmup_tgt_lr: float,
    normalize_id_loss_by_weight_sum: bool = True,
    use_decoder_checkpoint: bool = False,
    accumulate_steps: int = 1,
    max_clip_norm: float = 0.1,
    freq_ortho_loss_weight: float = 1.0,
    freq_consistency_loss_weight: float = 0.1,
    freq_energy_loss_weight: float = 0.0,
    triplet_criterion=None,
    triplet_loss_weight: float = 0.0,
    det_track_match_criterion=None,
    det_track_match_loss_weight: float = 0.0,
    laplace_loss_weight: float = 0.0,
    laplace_loss_temperature: float = 1.0,
    laplace_background_weight: float = 0.25,
    laplace_bbox_tau: float = 1.0,
    laplace_min_history: int = 1,
    use_tp_drop_fp_insert: bool = False,
    tp_drop_ratio: float = 0.1,
    fp_insert_ratio: float = 0.3,
    memory_bank=None,
    use_memory_bank: bool = False,
    train_box_source: str = "gt",
    det_train_start_epoch: int = 0,
    det_train_prob: float = 0.5,
    det_match_iou_thresh: float = 0.5,
    det_matching: str = "greedy",
    det_fallback_to_gt: bool = True,
    det_max_per_frame: int = 0,
    det_fill_unmatched_with_gt: bool = True,
    gt_box_noise_scale: float = 0.0,
    det_cache_use: bool = False,
    det_cache_write: bool = False,
    det_cache_dir: str = "outputs/det_cache",
    det_cache_compress: bool = True,
    det_cache_version: str = "v1",
    det_cache_validate_mtime: bool = True,
    use_aug_images: bool = False,
    assert_bbox_normalized: bool = False,
    bbox_norm_eps: float = 1e-3,
    newborn_penalty_weight: float = 0.0,
    newborn_penalty_margin: float = 0.0,
    newborn_penalty_warmup_epochs: int = 0,
    num_id_vocabulary: int = 500,
    log_exception_trace: bool = False,
    debug_raise_exceptions: bool = False,
    det_unknown_supervision: bool = False,
    det_source: str = "model",
    external_det_cfg: Optional[dict] = None,
    logging_interval: int = 20,
    log_freq_stats: bool = False,
    log_occlusion_stats: bool = False,
    log_calib_stats: bool = False,
):
    """训练一个 epoch"""

    model.train()
    # 但保持特征提取器为 eval 模式
    model_without_ddp = get_model(model)
    model_without_ddp.feature_extractor.eval()

    tps = TPS()
    metrics = Metrics()
    optimizer.zero_grad(set_to_none=True)
    step_timestamp = tps.timestamp()
    device = accelerator.device
    memory_bank_error_logged = False
    tp_fp_aug_error_logged = False

    if use_aug_images and (det_cache_use or det_cache_write):
        if accelerator.is_main_process:
            logger.warning("DET cache disabled because BYTETRACK_USE_AUG_IMAGES=True (augmented images are not cacheable).")
        det_cache_use = False
        det_cache_write = False

    det_source = str(det_source).lower()
    if det_source in ("txt", "public"):
        det_source = "external"
    if det_source not in ("model", "external"):
        det_source = "model"

    if det_source == "external":
        if external_det_cfg is None:
            raise ValueError("det_source='external' but external_det_cfg is None")
        if use_aug_images:
            raise ValueError(
                "Training with external detections does not support BYTETRACK_USE_AUG_IMAGES=True, because external "
                "detection boxes are in original image coordinates. Set BYTETRACK_USE_AUG_IMAGES=False."
            )
        if det_cache_use or det_cache_write:
            if accelerator.is_main_process:
                logger.warning("DET cache disabled for det_source='external' (cache currently supports model detections only).")
            det_cache_use = False
            det_cache_write = False

    external_det_path_cache: dict[tuple[str, str, str], str] = {}
    external_det_file_cache: dict[str, dict] = {}

    for step, samples in enumerate(dataloader):
        images, annotations, metas = samples["images"], samples["annotations"], samples["metas"]

        # 获取图像路径（用于特征提取）
        image_paths = []
        for meta_batch in metas:
            for meta in meta_batch:
                image_paths.append(meta.get("image_path", ""))

        _B, _T = len(annotations), len(annotations[0])

        # 从标注中获取 GT 框或检测框并提取特征
        all_features = []
        all_boxes = []
        all_logits = []
        detr_indices = []
        det_unknown_cache = {}
        det_unknown_max = 0

        train_box_source = str(train_box_source).lower()
        use_det_epoch = train_box_source in ("det", "mix") and epoch >= int(det_train_start_epoch)

        for b in range(_B):
            for t in range(_T):
                ann = annotations[b][t]
                gt_boxes_cxcywh = ann["bbox"]  # (N, 4) 归一化的 cxcywh
                if assert_bbox_normalized:
                    if torch.is_tensor(gt_boxes_cxcywh):
                        _gt = gt_boxes_cxcywh
                    else:
                        _gt = torch.tensor(gt_boxes_cxcywh, dtype=torch.float32)
                    if _gt.numel() > 0:
                        min_v = float(_gt.min().item())
                        max_v = float(_gt.max().item())
                        if min_v < -bbox_norm_eps or max_v > 1.0 + bbox_norm_eps:
                            raise ValueError(
                                f"GT bbox not normalized: min={min_v:.4f}, max={max_v:.4f}. "
                                "Check bbox format (expected normalized cxcywh) or disable ASSERT_BBOX_NORMALIZED."
                            )
                        w = _gt[:, 2]
                        h = _gt[:, 3]
                        if (w < -bbox_norm_eps).any() or (h < -bbox_norm_eps).any():
                            raise ValueError("GT bbox has negative width/height. Check data pipeline.")

                # 获取图像尺寸（避免使用硬编码默认值）
                meta = metas[b][t]
                img_h = meta.get("height", None)
                img_w = meta.get("width", None)
                if (img_h is None or img_w is None) and use_aug_images:
                    try:
                        img_h = int(images.tensors[b, t].shape[-2])
                        img_w = int(images.tensors[b, t].shape[-1])
                    except Exception as e:
                        if debug_raise_exceptions:
                            raise
                if (img_h is None or img_w is None) and meta.get("image_path", ""):
                    try:
                        import cv2
                        _img = cv2.imread(meta["image_path"])
                        if _img is not None:
                            img_h, img_w = _img.shape[:2]
                    except Exception as e:
                        if debug_raise_exceptions:
                            raise
                if img_h is None or img_w is None:
                    raise ValueError("Image size missing in meta; cannot convert boxes to pixel coords.")

                image_path = meta.get("image_path", "")
                image_tensor = None
                if use_aug_images:
                    try:
                        img_tensor = images.tensors[b, t]
                        # crop to actual size to avoid padded regions
                        img_tensor = img_tensor[:, :img_h, :img_w]
                        image_tensor = img_tensor
                    except Exception as e:
                        if debug_raise_exceptions:
                            raise
                        image_tensor = None
                use_det = False
                if use_det_epoch:
                    if train_box_source == "det":
                        use_det = True
                    elif train_box_source == "mix":
                        use_det = np.random.rand() < float(det_train_prob)
                if use_det and not image_path:
                    use_det = False

                if use_det and (image_tensor is not None or image_path):
                    dets = None
                    det_features = None
                    if det_source == "external":
                        try:
                            dataset_name = str(meta.get("dataset", "") or "")
                            split_name = str(meta.get("split", "") or "")
                            seq_name = str(meta.get("sequence", "") or "")
                            frame_idx = int(meta.get("frame_idx", 0))
                            frame_id = int(frame_idx) + 1  # MOT format is 1-indexed
                            if not dataset_name or not split_name or not seq_name:
                                raise KeyError(
                                    "Missing meta fields for external detections (expected: dataset/split/sequence)."
                                )

                            seq_key = (dataset_name, split_name, seq_name)
                            det_path = external_det_path_cache.get(seq_key)
                            if det_path is None:
                                det_path = resolve_external_det_path(
                                    config=external_det_cfg,
                                    dataset_name=dataset_name,
                                    split=split_name,
                                    seq_name=seq_name,
                                )
                                external_det_path_cache[seq_key] = det_path

                            if not os.path.isfile(det_path):
                                raise FileNotFoundError(f"External det file not found: {det_path}")

                            if det_path not in external_det_file_cache:
                                external_det_file_cache[det_path] = load_mot_detections(det_path)
                                if accelerator.is_main_process:
                                    logger.info(f"[DET][external] loaded: {det_path}")
                            det_map = external_det_file_cache[det_path]

                            frame_dets = det_map.get(int(frame_id), [])
                            min_conf = float(getattr(model_without_ddp.feature_extractor.cfg, "conf_thre", 0.0))
                            dets = []
                            boxes_xywh = []
                            for det in frame_dets:
                                if len(det) < 5:
                                    continue
                                x, y, w, h, conf = det[:5]
                                if float(conf) < float(min_conf):
                                    continue
                                if float(w) <= 1.0 or float(h) <= 1.0:
                                    continue
                                dets.append((float(x), float(y), float(w), float(h), float(conf)))
                                boxes_xywh.append((float(x), float(y), float(w), float(h)))

                            if len(boxes_xywh) == 0:
                                det_features = torch.zeros((0, feature_dim), device=device, dtype=torch.float32)
                            else:
                                det_features = model_without_ddp.feature_extractor.extract_features_from_boxes(
                                    image_path=image_path,
                                    boxes_xywh=boxes_xywh,
                                ).to(dtype=torch.float32)
                        except Exception as e:
                            # Numerical issues should stop immediately (continuing will silently corrupt training).
                            if isinstance(e, FloatingPointError):
                                raise
                            logger.warning(f"Failed to load/extract external detections: {e}")
                            if log_exception_trace:
                                logger.warning(traceback.format_exc())
                            if debug_raise_exceptions:
                                raise
                            dets, det_features = [], torch.zeros((0, feature_dim), device=device, dtype=torch.float32)
                    else:
                        cache_hit = False
                        if det_cache_use:
                            cache_path = _det_cache_path(
                                image_path,
                                model_without_ddp.feature_extractor.cfg,
                                det_cache_dir,
                                det_cache_version,
                            )
                            cached = _load_det_cache(cache_path, image_path, validate_mtime=det_cache_validate_mtime)
                            if cached is not None:
                                dets_np, feats_np = cached
                                dets = [tuple(d.tolist()) for d in dets_np]
                                det_features = torch.tensor(feats_np, device=device)
                                cache_hit = True
                        if not cache_hit:
                            try:
                                if image_tensor is not None:
                                    dets, det_features = model_without_ddp.feature_extractor.detect_with_features_tensor(
                                        image_tensor
                                    )
                                else:
                                    dets, det_features = model_without_ddp.feature_extractor.detect_with_features(
                                        image_path
                                    )
                            except Exception as e:
                                if isinstance(e, FloatingPointError):
                                    raise
                                logger.warning(f"Failed to detect with ByteTrack: {e}")
                                if log_exception_trace:
                                    logger.warning(traceback.format_exc())
                                if debug_raise_exceptions:
                                    raise
                                dets, det_features = [], torch.zeros((0, feature_dim), device=device)
                            if det_cache_write:
                                try:
                                    cache_path = _det_cache_path(
                                        image_path,
                                        model_without_ddp.feature_extractor.cfg,
                                        det_cache_dir,
                                        det_cache_version,
                                    )
                                    dets_np = np.array(dets, dtype=np.float32).reshape(-1, 5)
                                    feats_np = det_features.detach().cpu().numpy()
                                    _save_det_cache(
                                        cache_path,
                                        image_path,
                                        dets_np,
                                        feats_np,
                                        compress=det_cache_compress,
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to write det cache: {e}")
                                    if log_exception_trace:
                                        logger.warning(traceback.format_exc())
                                    if debug_raise_exceptions:
                                        raise
                    if det_max_per_frame and len(dets) > int(det_max_per_frame):
                        confs = np.array([d[4] for d in dets], dtype=np.float32)
                        topk_idx = np.argsort(-confs)[: int(det_max_per_frame)]
                        dets = [dets[i] for i in topk_idx]
                        det_features = det_features[topk_idx]

                    if len(dets) == 0 and det_fallback_to_gt:
                        use_det = False
                    else:
                        if len(dets) == 0:
                            det_boxes_xywh = torch.zeros((0, 4), dtype=torch.float32, device=device)
                            det_scores = torch.zeros((0, 1), dtype=torch.float32, device=device)
                            det_features = torch.zeros((0, feature_dim), dtype=torch.float32, device=device)
                        else:
                            det_boxes_xywh = torch.tensor(
                                [[d[0], d[1], d[2], d[3]] for d in dets],
                                dtype=torch.float32,
                                device=device,
                            )
                            det_scores = torch.tensor(
                                [d[4] for d in dets],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(-1)

                        det_boxes_cxcywh = det_boxes_xywh.clone()
                        if det_boxes_cxcywh.numel() > 0:
                            det_boxes_cxcywh[:, 0] = det_boxes_xywh[:, 0] + det_boxes_xywh[:, 2] / 2.0
                            det_boxes_cxcywh[:, 1] = det_boxes_xywh[:, 1] + det_boxes_xywh[:, 3] / 2.0
                            det_boxes_cxcywh[:, 2] = det_boxes_xywh[:, 2]
                            det_boxes_cxcywh[:, 3] = det_boxes_xywh[:, 3]
                            det_boxes_cxcywh[:, 0] /= img_w
                            det_boxes_cxcywh[:, 1] /= img_h
                            det_boxes_cxcywh[:, 2] /= img_w
                            det_boxes_cxcywh[:, 3] /= img_h

                        gt_boxes_tensor = gt_boxes_cxcywh.to(device) if hasattr(gt_boxes_cxcywh, "to") else torch.tensor(gt_boxes_cxcywh, device=device)
                        if gt_boxes_tensor.numel() == 0:
                            gt_boxes_tensor = gt_boxes_tensor.reshape(0, 4)

                        n_gt = gt_boxes_tensor.shape[0]
                        aligned_features = torch.zeros((n_gt, feature_dim), device=device, dtype=torch.float32)
                        aligned_boxes = gt_boxes_tensor.clone()
                        aligned_logits = torch.zeros((n_gt, 1), device=device, dtype=torch.float32)

                        if n_gt > 0 and det_boxes_xywh.numel() > 0:
                            det_idx, gt_idx = match_detections_to_gt(
                                det_boxes_xywh=det_boxes_xywh,
                                gt_boxes_cxcywh=gt_boxes_tensor,
                                img_w=img_w,
                                img_h=img_h,
                                iou_thresh=float(det_match_iou_thresh),
                                method=det_matching,
                            )
                            if det_idx.numel() == 0 and det_fallback_to_gt:
                                use_det = False
                            else:
                                if det_idx.numel() > 0:
                                    aligned_features[gt_idx] = det_features[det_idx]
                                    aligned_boxes[gt_idx] = det_boxes_cxcywh[det_idx]
                                    aligned_logits[gt_idx] = det_scores[det_idx]
                        else:
                            det_idx = torch.zeros((0,), dtype=torch.long, device=device)
                            gt_idx = torch.zeros((0,), dtype=torch.long, device=device)

                        if use_det:
                            if det_fill_unmatched_with_gt and n_gt > 0:
                                unmatched = aligned_logits.squeeze(-1) <= 0
                                if unmatched.any():
                                    gt_boxes_xywh = []
                                    for cx, cy, w, h in gt_boxes_tensor:
                                        x = (cx - w / 2) * img_w
                                        y = (cy - h / 2) * img_h
                                        pw = w * img_w
                                        ph = h * img_h
                                        x = max(0.0, min(float(x), img_w - 1.0))
                                        y = max(0.0, min(float(y), img_h - 1.0))
                                        if x + pw >= img_w:
                                            pw = max(1.0, img_w - x - 1.0)
                                        if y + ph >= img_h:
                                            ph = max(1.0, img_h - y - 1.0)
                                        gt_boxes_xywh.append((x, y, pw, ph))
                                    try:
                                        if image_tensor is not None:
                                            gt_features = model_without_ddp.feature_extractor.extract_features_from_boxes_tensor(
                                                image_tensor, gt_boxes_xywh
                                            )
                                        else:
                                            gt_features = model_without_ddp.feature_extractor.extract_features_from_boxes(
                                                image_path, gt_boxes_xywh
                                            )
                                        aligned_features[unmatched] = gt_features[unmatched]
                                        aligned_logits[unmatched] = 1.0
                                    except Exception as e:
                                        logger.warning(f"Failed to extract GT features for unmatched: {e}")
                                        if log_exception_trace:
                                            logger.warning(traceback.format_exc())
                                        if debug_raise_exceptions:
                                            raise

                            # Optional: add noise to det-derived boxes for robustness
                            det_box_noise_std = 0.0
                            det_box_noise_prob = 0.0
                            det_box_noise_clip = 1.0
                            if det_box_noise_std > 0.0 and det_box_noise_prob > 0.0:
                                valid_det = aligned_logits.squeeze(-1) > 0
                                aligned_boxes = _apply_box_noise_cxcywh(
                                    aligned_boxes,
                                    noise_std=det_box_noise_std,
                                    prob=det_box_noise_prob,
                                    max_size=det_box_noise_clip,
                                    mask=valid_det,
                                )

                            all_features.append(aligned_features)
                            all_boxes.append(aligned_boxes)
                            all_logits.append(aligned_logits)
                            n = n_gt
                            indices = (torch.arange(n, device=device), torch.arange(n, device=device))
                            detr_indices.append(indices)

                            # Build det-based unknown supervision for this frame if enabled
                            if det_unknown_supervision:
                                unk_features = det_features
                                unk_boxes = det_boxes_cxcywh
                                if det_box_noise_std > 0.0 and det_box_noise_prob > 0.0:
                                    unk_boxes = _apply_box_noise_cxcywh(
                                        unk_boxes,
                                        noise_std=det_box_noise_std,
                                        prob=det_box_noise_prob,
                                        max_size=det_box_noise_clip,
                                        mask=None,
                                    )
                                # Map dets -> GT labels (per group) using annotations
                                _G = ann["trajectory_id_labels"].shape[0]
                                labels_g = torch.full((_G, det_features.shape[0]), num_id_vocabulary,
                                                      dtype=torch.long, device=device)
                                if det_idx.numel() > 0:
                                    det_to_gt = {int(d.item()): int(g.item()) for d, g in zip(det_idx, gt_idx)}
                                    for g in range(_G):
                                        traj_ann = ann["trajectory_ann_idxs"][g, 0]
                                        traj_labels = ann["trajectory_id_labels"][g, 0]
                                        traj_masks = ann["trajectory_id_masks"][g, 0]
                                        mapping = {}
                                        for pos in range(traj_ann.shape[0]):
                                            if traj_masks[pos] or traj_ann[pos] < 0:
                                                continue
                                            mapping[int(traj_ann[pos].item())] = int(traj_labels[pos].item())
                                        for d_i, gt_i in det_to_gt.items():
                                            labels_g[g, d_i] = mapping.get(gt_i, num_id_vocabulary)
                                det_unknown_cache[(b, t)] = (unk_features, unk_boxes, labels_g)
                                det_unknown_max = max(det_unknown_max, int(unk_features.shape[0]))

                if not use_det:
                    # 转换为像素坐标的 xywh
                    gt_boxes_xywh = []
                    gt_box_noise = float(gt_box_noise_scale)
                    for cx, cy, w, h in gt_boxes_cxcywh:
                        x = (cx - w / 2) * img_w
                        y = (cy - h / 2) * img_h
                        pw = w * img_w
                        ph = h * img_h
                        if gt_box_noise > 0:
                            dx = np.random.randn() * gt_box_noise * pw
                            dy = np.random.randn() * gt_box_noise * ph
                            dw = np.random.randn() * gt_box_noise * pw
                            dh = np.random.randn() * gt_box_noise * ph
                            x += dx
                            y += dy
                            pw = max(1.0, pw + dw)
                            ph = max(1.0, ph + dh)
                        # clamp to image bounds
                        x = max(0.0, min(x, img_w - 1.0))
                        y = max(0.0, min(y, img_h - 1.0))
                        if x + pw >= img_w:
                            pw = max(1.0, img_w - x - 1.0)
                        if y + ph >= img_h:
                            ph = max(1.0, img_h - y - 1.0)
                        gt_boxes_xywh.append((x, y, pw, ph))

                    if len(gt_boxes_xywh) > 0 and (image_tensor is not None or image_path):
                        try:
                            if image_tensor is not None:
                                features = model_without_ddp.feature_extractor.extract_features_from_boxes_tensor(
                                    image_tensor, gt_boxes_xywh
                                )
                            else:
                                features = model_without_ddp.feature_extractor.extract_features_from_boxes(
                                    image_path, gt_boxes_xywh
                                )
                        except Exception as e:
                            logger.warning(f"Failed to extract features: {e}")
                            if log_exception_trace:
                                logger.warning(traceback.format_exc())
                            if debug_raise_exceptions:
                                raise
                            features = torch.zeros((len(gt_boxes_xywh), feature_dim), device=device)
                    else:
                        features = torch.zeros((len(gt_boxes_xywh), feature_dim), device=device)

                    all_features.append(features)
                    gt_boxes_tensor = gt_boxes_cxcywh.to(device) if hasattr(gt_boxes_cxcywh, 'to') else torch.tensor(gt_boxes_cxcywh, device=device)
                    if gt_boxes_tensor.numel() == 0:
                        gt_boxes_tensor = gt_boxes_tensor.reshape(0, 4)
                    all_boxes.append(gt_boxes_tensor)
                    all_logits.append(torch.ones((gt_boxes_tensor.shape[0], 1), dtype=torch.float32, device=device))

                    n = gt_boxes_tensor.shape[0]
                    indices = (torch.arange(n, device=device), torch.arange(n, device=device))
                    detr_indices.append(indices)

                    # Use GT as unknown supervision if enabled (labels mapped via ann_idxs)
                    if det_unknown_supervision:
                        unk_features = features
                        unk_boxes = gt_boxes_tensor
                        _G = ann["trajectory_id_labels"].shape[0]
                        labels_g = torch.full((_G, unk_features.shape[0]), num_id_vocabulary,
                                              dtype=torch.long, device=device)
                        # direct match: gt index == annotation index
                        for g in range(_G):
                            traj_ann = ann["trajectory_ann_idxs"][g, 0]
                            traj_labels = ann["trajectory_id_labels"][g, 0]
                            traj_masks = ann["trajectory_id_masks"][g, 0]
                            mapping = {}
                            for pos in range(traj_ann.shape[0]):
                                if traj_masks[pos] or traj_ann[pos] < 0:
                                    continue
                                mapping[int(traj_ann[pos].item())] = int(traj_labels[pos].item())
                            for gt_i in range(unk_features.shape[0]):
                                labels_g[g, gt_i] = mapping.get(gt_i, num_id_vocabulary)
                        det_unknown_cache[(b, t)] = (unk_features, unk_boxes, labels_g)
                        det_unknown_max = max(det_unknown_max, int(unk_features.shape[0]))

        # Padding 到相同大小
        max_n = max(f.shape[0] for f in all_features)
        max_n = max(max_n, 1)

        padded_features = []
        padded_boxes = []
        padded_logits = []
        for feat, box in zip(all_features, all_boxes):
            n = feat.shape[0]
            if n < max_n:
                pad_feat = torch.zeros((max_n - n, feature_dim), device=device)
                feat = torch.cat([feat, pad_feat], dim=0)
                pad_box = torch.zeros((max_n - n, 4), device=device)
                box = torch.cat([box, pad_box], dim=0)
            padded_features.append(feat)
            padded_boxes.append(box)
        for logit in all_logits:
            n = logit.shape[0]
            if n < max_n:
                pad_logit = torch.zeros((max_n - n, 1), device=device)
                logit = torch.cat([logit, pad_logit], dim=0)
            padded_logits.append(logit)

        features_tensor = torch.stack(padded_features, dim=0)  # (B*T, max_N, feature_dim)
        boxes_tensor = torch.stack(padded_boxes, dim=0)  # (B*T, max_N, 4)
        logits_tensor = torch.stack(padded_logits, dim=0)  # (B*T, max_N, 1)

        # Numerical safety: detector/ROI features should never contain NaN/Inf.
        # If this triggers, fix the upstream feature extraction (e.g., invalid RoIs) instead of "training through" it.
        if not torch.isfinite(features_tensor).all():
            bad_frames = (~torch.isfinite(features_tensor)).any(dim=-1).any(dim=-1)
            bad_idx = bad_frames.nonzero(as_tuple=True)[0].detach().cpu().tolist()
            bad_desc = []
            for flat in bad_idx[:5]:
                try:
                    b = int(flat) // int(_T)
                    t = int(flat) % int(_T)
                    meta = metas[b][t]
                    bad_desc.append(
                        f"(b={b},t={t}) {meta.get('dataset','')}/{meta.get('sequence','')} "
                        f"frame_idx={meta.get('frame_idx','')} path={meta.get('image_path','')}"
                    )
                except Exception:
                    continue
            raise FloatingPointError(
                f"[train_bytetrack] Non-finite features_tensor detected at epoch={epoch} step={step}. "
                f"bad_frames(B*T indices)={bad_idx[:20]} sample={bad_desc}"
            )
        if not torch.isfinite(boxes_tensor).all():
            bad_frames = (~torch.isfinite(boxes_tensor)).any(dim=-1).any(dim=-1)
            bad_idx = bad_frames.nonzero(as_tuple=True)[0].detach().cpu().tolist()
            bad_desc = []
            for flat in bad_idx[:5]:
                try:
                    b = int(flat) // int(_T)
                    t = int(flat) % int(_T)
                    meta = metas[b][t]
                    bad_desc.append(
                        f"(b={b},t={t}) {meta.get('dataset','')}/{meta.get('sequence','')} "
                        f"frame_idx={meta.get('frame_idx','')} path={meta.get('image_path','')}"
                    )
                except Exception:
                    continue
            raise FloatingPointError(
                f"[train_bytetrack] Non-finite boxes_tensor detected at epoch={epoch} step={step}. "
                f"bad_frames(B*T indices)={bad_idx[:20]} sample={bad_desc}"
            )

        # 学习率 warmup
        if epoch < lr_warmup_epochs:
            lr_warmup(
                optimizer=optimizer,
                epoch=epoch, curr_iter=step, tgt_lr=lr_warmup_tgt_lr,
                warmup_epochs=lr_warmup_epochs, num_iter_per_epoch=len(dataloader),
            )

        # 构造伪 detr_outputs
        detr_outputs = {
            "outputs": features_tensor,
            "pred_boxes": boxes_tensor,
            "pred_logits": logits_tensor,
        }

        # 使用原始的 prepare_for_motip 逻辑
        from train import prepare_for_motip
        seq_info = prepare_for_motip(
            detr_outputs=detr_outputs,
            annotations=annotations,
            detr_indices=detr_indices,
        )

        # Override unknown with det-based supervision to align with inference
        if det_unknown_supervision and det_unknown_max > 0:
            _G = annotations[0][0]["trajectory_id_labels"].shape[0]
            unk_feat = torch.zeros((_B, _G, _T, det_unknown_max, feature_dim),
                                   device=device, dtype=features_tensor.dtype)
            unk_boxes = torch.zeros((_B, _G, _T, det_unknown_max, 4),
                                    device=device, dtype=boxes_tensor.dtype)
            unk_labels = torch.full((_B, _G, _T, det_unknown_max), -1,
                                    device=device, dtype=torch.long)
            unk_masks = torch.ones((_B, _G, _T, det_unknown_max), device=device, dtype=torch.bool)
            unk_times = torch.zeros((_B, _G, _T, det_unknown_max), device=device, dtype=torch.long)
            for (b, t), (f, bx, lg) in det_unknown_cache.items():
                n = f.shape[0]
                if n == 0:
                    continue
                for g in range(_G):
                    unk_feat[b, g, t, :n] = f
                    unk_boxes[b, g, t, :n] = bx
                    unk_labels[b, g, t, :n] = lg[g]
                    unk_masks[b, g, t, :n] = False
                    unk_times[b, g, t, :n] = t
            seq_info["unknown_features"] = unk_feat
            seq_info["unknown_boxes"] = unk_boxes
            seq_info["unknown_id_labels"] = unk_labels
            seq_info["unknown_masks"] = unk_masks
            seq_info["unknown_times"] = unk_times

        # Memory bank update on trajectory features (training)
        if use_memory_bank and memory_bank is not None:
            try:
                traj_feats = seq_info["trajectory_features"]   # (B,G,T,N,C)
                traj_masks = seq_info["trajectory_masks"]      # (B,G,T,N)
                Bm, Gm, Tm, Nm, Cm = traj_feats.shape

                # Initialize memory with first frame (flatten B,G,N)
                long_mem = traj_feats[:, :, 0].reshape(-1, Cm)
                last_feat = long_mem.clone()

                # Avoid in-place writes into traj_feats to keep autograd stable.
                query_feats_per_t = []
                for t in range(Tm):
                    cur = traj_feats[:, :, t].reshape(-1, Cm)
                    mask_t = traj_masks[:, :, t].reshape(-1)
                    cur_in = cur
                    if mask_t.any():
                        cur_in = cur_in.clone()
                        cur_in[mask_t] = 0
                    scores_t = (~mask_t).float()
                    try:
                        long_mem, query_feat = memory_bank.update(
                            current_features=cur_in,
                            long_memory=long_mem,
                            last_features=last_feat,
                            scores=scores_t,
                        )
                    except FloatingPointError as e:
                        # Add sequence/frame context to make debugging actionable.
                        meta_desc = []
                        try:
                            for bb in range(min(int(Bm), 3)):
                                m = metas[bb][t]
                                meta_desc.append(
                                    f"(b={bb},t={t}) {m.get('dataset','')}/{m.get('sequence','')} "
                                    f"frame_idx={m.get('frame_idx','')} path={m.get('image_path','')}"
                                )
                        except Exception:
                            pass
                        raise FloatingPointError(
                            f"[train_bytetrack] MemoryBank failed at epoch={epoch} step={step} t={t}. "
                            f"sample={meta_desc}"
                        ) from e
                    query_feats_per_t.append(query_feat.reshape(Bm, Gm, Nm, Cm))
                    last_feat = cur_in

                seq_info["trajectory_features"] = torch.stack(query_feats_per_t, dim=2)
            except Exception as e:
                # Numerical issues should stop the run immediately to avoid silently corrupting weights.
                if isinstance(e, FloatingPointError):
                    raise
                if debug_raise_exceptions:
                    raise
                if accelerator.is_main_process and not memory_bank_error_logged:
                    logger.warning(f"[MemoryBank] update failed; disabling for this epoch. Error: {e}")
                    if log_exception_trace:
                        logger.warning(traceback.format_exc())
                    memory_bank_error_logged = True

        # TP Drop / FP Insert augmentation (feature-level)
        if use_tp_drop_fp_insert:
            try:
                unk_feat = seq_info["unknown_features"]
                unk_boxes = seq_info["unknown_boxes"]
                unk_labels = seq_info["unknown_id_labels"]
                unk_masks = seq_info["unknown_masks"]
                _B, _G, _T, _N, _C = unk_feat.shape
                newborn_idx = int(num_id_vocabulary)

                for b in range(_B):
                    for g in range(_G):
                        for t in range(_T):
                            masks = unk_masks[b, g, t]
                            labels = unk_labels[b, g, t]
                            # Valid positives only (TPs)
                            valid = (~masks) & (labels >= 0) & (labels != newborn_idx)
                            valid_idx = valid.nonzero(as_tuple=True)[0]

                            # TP Drop: mask out some true positives
                            if tp_drop_ratio > 0.0 and valid_idx.numel() > 0:
                                drop_mask = (torch.rand(valid_idx.shape[0], device=device) < tp_drop_ratio)
                                drop_idx = valid_idx[drop_mask]
                                if drop_idx.numel() > 0:
                                    masks[drop_idx] = True
                                    labels[drop_idx] = -1
                                    unk_feat[b, g, t, drop_idx] = 0
                                    unk_boxes[b, g, t, drop_idx] = 0

                            # Recompute valid and available slots after drop
                            masks = unk_masks[b, g, t]
                            labels = unk_labels[b, g, t]
                            valid_after = (~masks) & (labels >= 0) & (labels != newborn_idx)
                            avail_idx = masks.nonzero(as_tuple=True)[0]
                            fp_src_idx = ((~masks) & (labels == newborn_idx)).nonzero(as_tuple=True)[0]

                            # FP Insert: fill some empty slots with noisy features/boxes
                            if fp_insert_ratio > 0.0 and avail_idx.numel() > 0 and fp_src_idx.numel() > 0:
                                num_fp = int(fp_src_idx.numel() * fp_insert_ratio)
                                num_fp = min(num_fp, int(avail_idx.numel()))
                                if num_fp > 0:
                                    perm_slots = torch.randperm(avail_idx.numel(), device=device)[:num_fp]
                                    perm_src = torch.randperm(fp_src_idx.numel(), device=device)[:num_fp]
                                    fp_slots = avail_idx[perm_slots]
                                    src_idx = fp_src_idx[perm_src]

                                    src_feat = unk_feat[b, g, t, src_idx]
                                    src_box = unk_boxes[b, g, t, src_idx]

                                    # Use real unmatched detections as FPs (copy as-is)
                                    fp_feat = src_feat
                                    fp_box = src_box

                                    unk_feat[b, g, t, fp_slots] = fp_feat
                                    unk_boxes[b, g, t, fp_slots] = fp_box
                                    unk_labels[b, g, t, fp_slots] = newborn_idx
                                    unk_masks[b, g, t, fp_slots] = False

                seq_info["unknown_features"] = unk_feat
                seq_info["unknown_boxes"] = unk_boxes
                seq_info["unknown_id_labels"] = unk_labels
                seq_info["unknown_masks"] = unk_masks
            except Exception as e:
                if debug_raise_exceptions:
                    raise
                if accelerator.is_main_process and not tp_fp_aug_error_logged:
                    logger.warning(f"[TP/FP Aug] failed; disabling augmentation. Error: {e}")
                    if log_exception_trace:
                        logger.warning(traceback.format_exc())
                    tp_fp_aug_error_logged = True

        # 前向传播
        try:
            with accelerator.autocast():
                seq_info = model(seq_info=seq_info, part="trajectory_modeling")
        except FloatingPointError as e:
            meta_desc = []
            try:
                for bb in range(min(int(_B), 3)):
                    for tt in range(min(int(_T), 3)):
                        m = metas[bb][tt]
                        meta_desc.append(
                            f"(b={bb},t={tt}) {m.get('dataset','')}/{m.get('sequence','')} "
                            f"frame_idx={m.get('frame_idx','')} path={m.get('image_path','')}"
                        )
            except Exception:
                pass
            raise FloatingPointError(
                f"[train_bytetrack] Non-finite detected inside model(part='trajectory_modeling') "
                f"at epoch={epoch} step={step}. sample={meta_desc}"
            ) from e

        # Det <-> Track matching supervision (InfoNCE-style, optional)
        # IMPORTANT: compute this loss on the *post-trajectory_modeling* features so it can
        # actually supervise the learned temporal representations (not only raw detector features).
        det_track_match_loss = torch.tensor(0.0, device=device)
        det_track_match_loss_weight = float(det_track_match_loss_weight) if det_track_match_loss_weight is not None else 0.0
        if det_track_match_criterion is not None and det_track_match_loss_weight > 0.0:
            try:
                det_track_match_loss = det_track_match_criterion(
                    unknown_features=seq_info["unknown_features"],
                    unknown_id_labels=seq_info["unknown_id_labels"],
                    unknown_masks=seq_info["unknown_masks"],
                    trajectory_features=seq_info["trajectory_features"],
                    trajectory_id_labels=seq_info["trajectory_id_labels"],
                    trajectory_masks=seq_info["trajectory_masks"],
                    newborn_label=int(num_id_vocabulary),
                )
            except Exception as e:
                if debug_raise_exceptions:
                    raise
                det_track_match_loss = torch.tensor(0.0, device=device)
                if accelerator.is_main_process:
                    logger.warning(f"[DetTrackMatchLoss] failed; disabled for this step. Error: {e}")
                    if log_exception_trace:
                        logger.warning(traceback.format_exc())

        laplace_loss = torch.tensor(0.0, device=device)
        laplace_matched_rows = torch.tensor(0.0, device=device)
        laplace_background_rows = torch.tensor(0.0, device=device)
        if laplace_loss_weight > 0.0:
            try:
                train_model = accelerator.unwrap_model(model)
                laplace_assoc = getattr(train_model, "laplace_assoc", None)
                if laplace_assoc is not None:
                    with accelerator.autocast():
                        laplace_out = compute_laplace_supervision_loss(
                            seq_info=seq_info,
                            laplace_assoc=laplace_assoc,
                            num_id_vocabulary=num_id_vocabulary,
                            temperature=laplace_loss_temperature,
                            background_weight=laplace_background_weight,
                            bbox_tau=laplace_bbox_tau,
                            min_history=laplace_min_history,
                        )
                    laplace_loss = laplace_out["loss"].to(device=device)
                    laplace_matched_rows = laplace_out["matched_rows"].to(device=device)
                    laplace_background_rows = laplace_out["background_rows"].to(device=device)
            except Exception as e:
                if debug_raise_exceptions:
                    raise
                laplace_loss = torch.tensor(0.0, device=device)
                if accelerator.is_main_process:
                    logger.warning(f"[LaplaceLoss] failed; disabled for this step. Error: {e}")
                    if log_exception_trace:
                        logger.warning(traceback.format_exc())

        with accelerator.autocast():
            id_decoder_output = model(
                seq_info=seq_info,
                part="id_decoder",
                use_decoder_checkpoint=use_decoder_checkpoint,
            )

        # 解包输出
        if isinstance(id_decoder_output, tuple) and len(id_decoder_output) == 4:
            id_logits, id_gts, id_masks, freq_extra_losses = id_decoder_output
        else:
            id_logits, id_gts, id_masks = id_decoder_output
            freq_extra_losses = None

        # 计算 ID 损失（支持 decoder 提供的分支/层权重）
        loss_weights = None
        if freq_extra_losses is not None and isinstance(freq_extra_losses, dict):
            loss_weights = freq_extra_losses.get("loss_weights", None)

        if (
            isinstance(loss_weights, (list, tuple))
            and len(loss_weights) > 0
            and id_logits is not None
            and id_masks is not None
            and id_logits.shape[0] % len(loss_weights) == 0
        ):
            k = len(loss_weights)
            chunk = id_logits.shape[0] // k
            loss_sum = 0.0
            weight_sum = 0.0
            for i, w in enumerate(loss_weights):
                if w is None:
                    continue
                w_val = float(w)
                if w_val == 0.0:
                    continue
                logits_i = id_logits[i * chunk:(i + 1) * chunk]
                labels_i = id_gts[i * chunk:(i + 1) * chunk] if id_gts is not None else None
                masks_i = id_masks[i * chunk:(i + 1) * chunk]
                loss_i = id_criterion(id_logits=logits_i, id_labels=labels_i, id_masks=masks_i)
                loss_sum = loss_sum + w_val * loss_i
                weight_sum += w_val
            if weight_sum > 0:
                # Treat weights as a convex-combination by default so enabling aux/freq/fusion branches
                # does not inflate the overall ID loss scale.
                #
                # If you want "additive" supervision (each branch adds extra gradient), disable this with:
                #   ID_LOSS_NORMALIZE_BY_WEIGHT_SUM: false
                if normalize_id_loss_by_weight_sum:
                    id_loss = loss_sum / weight_sum
                else:
                    id_loss = loss_sum
            else:
                id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)
        else:
            id_loss = id_criterion(id_logits=id_logits, id_labels=id_gts, id_masks=id_masks)

        # Triplet loss (optional)
        triplet_loss = torch.tensor(0.0, device=device)
        triplet_src_decoder = 0.0
        triplet_w = float(triplet_loss_weight) if triplet_loss_weight is not None else 0.0
        if triplet_criterion is not None and triplet_w > 0.0:
            triplet_feats = None
            triplet_labels = None
            triplet_masks = None

            if freq_extra_losses is not None and isinstance(freq_extra_losses, dict):
                triplet_feats = freq_extra_losses.get("triplet_embeddings", None)
                triplet_labels = freq_extra_losses.get("triplet_labels", None)
                triplet_masks = freq_extra_losses.get("triplet_masks", None)
                if triplet_feats is not None and triplet_labels is not None:
                    triplet_src_decoder = 1.0

            if triplet_feats is None or triplet_labels is None:
                triplet_feats = seq_info.get("trajectory_features", None)
                triplet_labels = seq_info.get("trajectory_id_labels", None)
                triplet_masks = seq_info.get("trajectory_masks", None)

            if triplet_feats is not None and triplet_labels is not None:
                triplet_loss = triplet_criterion(
                    embeddings=triplet_feats,
                    labels=triplet_labels,
                    masks=triplet_masks,
                )

        # 频率损失
        freq_losses = seq_info.get("freq_losses", {})
        freq_ortho_loss = freq_losses.get("ortho_loss", 0.0)
        if not torch.is_tensor(freq_ortho_loss):
            freq_ortho_loss = torch.tensor(freq_ortho_loss, device=device)
        freq_ortho_loss = torch.clamp(freq_ortho_loss, min=0.0, max=10.0)

        freq_energy_loss = freq_losses.get("energy_balance_loss", 0.0)
        if not torch.is_tensor(freq_energy_loss):
            freq_energy_loss = torch.tensor(freq_energy_loss, device=device)
        freq_energy_loss = torch.clamp(freq_energy_loss, min=0.0, max=10.0)

        freq_consistency_loss = 0.0
        if freq_extra_losses is not None and isinstance(freq_extra_losses, dict):
            freq_consistency_loss = freq_extra_losses.get("consistency_loss", 0.0)
        if not torch.is_tensor(freq_consistency_loss):
            freq_consistency_loss = torch.tensor(freq_consistency_loss, device=device)
        freq_consistency_loss = torch.clamp(freq_consistency_loss, min=0.0, max=10.0)

        # 新生惩罚：仅对“已匹配到已有GT”的样本抑制 newborn logit
        newborn_penalty = torch.tensor(0.0, device=device)
        newborn_weight = float(newborn_penalty_weight)
        newborn_margin = float(newborn_penalty_margin)
        newborn_warmup = int(newborn_penalty_warmup_epochs)
        if newborn_weight > 0.0 and id_logits is not None and id_gts is not None and id_masks is not None:
            _logits = id_logits
            _labels = id_gts
            _masks = id_masks
            if _logits.ndim == 4:
                _logits = _logits.unsqueeze(2)
            if _labels.ndim == 3:
                _labels = _labels.unsqueeze(2)
            if _masks.ndim == 3:
                _masks = _masks.unsqueeze(2)
            # 与 IDCriterion 保持一致：跳过 t=0
            _logits = _logits[:, :, 1:, :, :]
            _labels = _labels[:, :, 1:, :]
            _masks = _masks[:, :, 1:, :]

            valid = (~_masks) & (_labels >= 0)
            if valid.any():
                newborn_idx = int(num_id_vocabulary)
                # Ensure newborn index within range
                if newborn_idx >= _logits.shape[-1]:
                    newborn_idx = _logits.shape[-1] - 1
                target_existing = valid & (_labels != newborn_idx)
                if target_existing.any():
                    newborn_logit = _logits[..., newborn_idx]
                    if newborn_idx == _logits.shape[-1] - 1:
                        other_max = _logits[..., :-1].max(dim=-1).values
                    else:
                        other_logits = torch.cat(
                            [_logits[..., :newborn_idx], _logits[..., newborn_idx + 1:]], dim=-1
                        )
                        other_max = other_logits.max(dim=-1).values
                    penalty = F.softplus(newborn_logit - other_max + newborn_margin)
                    denom = target_existing.float().sum().clamp_min(1.0)
                    newborn_penalty = (penalty * target_existing.float()).sum() / denom
            if newborn_warmup > 0:
                warm_ratio = min(1.0, float(epoch + 1) / float(newborn_warmup))
                newborn_weight = newborn_weight * warm_ratio

        # 总损失
        loss = id_loss * id_criterion.weight \
               + freq_ortho_loss_weight * freq_ortho_loss \
               + freq_energy_loss_weight * freq_energy_loss \
               + freq_consistency_loss_weight * freq_consistency_loss \
               + newborn_weight * newborn_penalty \
               + det_track_match_loss_weight * det_track_match_loss \
               + laplace_loss_weight * laplace_loss \
               + triplet_w * triplet_loss

        # Fail fast on NaN/Inf: continuing will corrupt weights and silently waste GPU time.
        if not torch.isfinite(loss).item():
            if accelerator.is_main_process:
                logger.warning(
                    f"[Numerics] Non-finite total loss detected; aborting this run. "
                    f"epoch={epoch} step={step} "
                    f"loss={loss} id_loss={id_loss} "
                    f"freq_ortho={freq_ortho_loss} freq_energy={freq_energy_loss} "
                    f"freq_consist={freq_consistency_loss} newborn_penalty={newborn_penalty} "
                    f"det_track_match={det_track_match_loss} laplace_loss={laplace_loss} "
                    f"triplet_loss={triplet_loss}"
                )
            raise FloatingPointError("[train_bytetrack] Non-finite loss (NaN/Inf).")

        # 记录损失
        metrics.update(name="loss", value=loss.item())
        metrics.update(name="id_loss", value=id_loss.item())
        if det_track_match_loss_weight > 0.0:
            metrics.update(name="det_track_match_loss", value=det_track_match_loss.item())
        if laplace_loss_weight > 0.0:
            metrics.update(name="laplace_loss", value=laplace_loss.item())
            metrics.update(name="laplace_rows_pos", value=float(laplace_matched_rows.item()))
            metrics.update(name="laplace_rows_bg", value=float(laplace_background_rows.item()))
        if triplet_w > 0.0:
            metrics.update(name="triplet_loss", value=triplet_loss.item())
            metrics.update(name="triplet_src_decoder", value=triplet_src_decoder)
        metrics.update(name="freq_ortho_loss", value=freq_ortho_loss.item())
        if log_freq_stats and freq_energy_loss_weight > 0.0:
            metrics.update(name="freq_energy_loss", value=freq_energy_loss.item())
        metrics.update(name="freq_consistency_loss", value=freq_consistency_loss.item())
        if newborn_weight > 0.0:
            metrics.update(name="newborn_penalty", value=newborn_penalty.item())

        # Optional diagnostics for frequency modules
        if log_freq_stats:
            try:
                freq_info = seq_info.get("freq_info", {})
                decomp_info = freq_info.get("decomposition_info", {})
                band_features = decomp_info.get("band_features", {})
                importance = band_features.get("importance", None)
                if torch.is_tensor(importance):
                    # mean importance per band
                    dims = tuple(range(importance.dim() - 1))
                    mean_imp = importance.mean(dim=dims)
                    metrics.update(name="freq_imp_max", value=float(mean_imp.max().item()))
                    # entropy
                    p = importance.clamp(min=1e-6)
                    ent = -(p * p.log()).sum(dim=-1).mean()
                    metrics.update(name="freq_imp_entropy", value=float(ent.item()))
                band_energy = decomp_info.get("band_energy", None)
                if torch.is_tensor(band_energy):
                    metrics.update(name="freq_energy_max", value=float(band_energy.max().item()))
            except Exception:
                if debug_raise_exceptions:
                    raise

        if log_occlusion_stats:
            try:
                occ_info = seq_info.get("occlusion_info", None)
                if isinstance(occ_info, dict):
                    occ_scores = occ_info.get("occlusion_scores", None)
                    if torch.is_tensor(occ_scores):
                        metrics.update(name="occ_score", value=float(occ_scores.mean().item()))
                occ_traj = seq_info.get("occlusion_info_traj", None)
                if isinstance(occ_traj, dict):
                    occ_scores = occ_traj.get("occlusion_scores", None)
                    if torch.is_tensor(occ_scores):
                        metrics.update(name="occ_score_traj", value=float(occ_scores.mean().item()))
            except Exception:
                if debug_raise_exceptions:
                    raise

        if log_calib_stats:
            try:
                if freq_extra_losses is not None and isinstance(freq_extra_losses, dict):
                    cal = freq_extra_losses.get("calibration_factor", None)
                    if torch.is_tensor(cal):
                        metrics.update(name="calib_factor", value=float(cal.mean().item()))
            except Exception:
                if debug_raise_exceptions:
                    raise

        # 反向传播
        loss /= accumulate_steps
        accelerator.backward(loss)

        if (step + 1) % accumulate_steps == 0:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_clip_norm)
            try:
                grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
            except Exception:
                grad_norm_val = float("nan")

            # If grads contain NaN/Inf, clip_grad_norm_ will typically return NaN.
            # Stepping the optimizer in this state will permanently corrupt weights (common cause of later NaN forward).
            if not math.isfinite(grad_norm_val):
                if accelerator.is_main_process:
                    logger.warning(
                        f"[Numerics] Non-finite grad_norm detected; skipping optimizer.step(). "
                        f"epoch={epoch} step={step} grad_norm={grad_norm} loss={loss}"
                    )
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # 日志
        tps.update(tps=tps.timestamp() - step_timestamp)
        step_timestamp = tps.timestamp()

        if step % logging_interval == 0:
            _lr = optimizer.state_dict()["param_groups"][-1]["lr"]
            metrics["lr"].clear()
            metrics.update(name="lr", value=_lr)
            metrics.sync()
            eta = tps.eta(total_steps=len(dataloader), current_steps=step)
            logger.metrics(
                log=f"[Epoch: {epoch}] [{step}/{len(dataloader)}] "
                    f"[tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] ",
                metrics=metrics,
                global_step=states["global_step"],
            )

        states["global_step"] += 1

    states["start_epoch"] += 1
    return metrics


def lr_warmup(optimizer, epoch: int, curr_iter: int, tgt_lr: float, warmup_epochs: int, num_iter_per_epoch: int):
    """学习率 warmup"""
    total_warmup_iters = warmup_epochs * num_iter_per_epoch
    current_lr_ratio = (epoch * num_iter_per_epoch + curr_iter + 1) / total_warmup_iters
    current_lr = tgt_lr * current_lr_ratio
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = current_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = current_lr


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # 获取运行时选项
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # 加载超级配置
    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg.get("SUPER_CONFIG_PATH"))

    # 合并命令行参数
    cfg = update_config(config=cfg, option=opt)
    cfg["CONFIG_PATH"] = opt.config_path

    # 训练
    train_engine(config=cfg)
