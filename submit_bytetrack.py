# Copyright (c) 2024. All Rights Reserved.
"""
使用 ByteTrack 特征进行推理和评估

用法：
    python submit_bytetrack.py \
        --config-path configs/bytetrack_fa_mot_mot17.yaml \
        --data-root /path/to/datasets \
        --inference-model outputs/bytetrack_fa_mot_mot17/checkpoint_final.pth \
        --inference-dataset MOT17 \
        --inference-split train \
        --inference-mode evaluate
"""

import os
import argparse
import shutil
import json
import torch
import numpy as np
from tqdm import tqdm

from utils.misc import yaml_to_dict
from utils.detector_profile import resolve_bytetrack_profile
from utils.mot_detections import load_mot_detections, resolve_external_det_path
from configs.util import load_super_config, update_config
from models.bytetrack_feature_extractor import (
    ByteTrackFeatureConfig,
    ByteTrackFeatureExtractor,
)
from models.runtime_tracker_bytetrack import RuntimeTrackerByteTrack
from models.public_reid import build_public_reid_encoder


def _has_any_gt_files(dataset_dir: str, sequences: list[str]) -> bool:
    """
    TrackEval requires per-sequence GT files (e.g. gt/gt.txt). MOT test splits typically ship without GT,
    so we must skip evaluation in that case.
    """
    for seq_name in sequences:
        gt_path = os.path.join(dataset_dir, seq_name, "gt", "gt.txt")
        if os.path.isfile(gt_path):
            return True
    return False


def _runtime_dump_seq_csv_path(dump_root: str, seq_key: str) -> str:
    dump_root = str(dump_root or "")
    if not dump_root:
        return ""
    if dump_root.lower().endswith(".csv"):
        return dump_root
    return os.path.join(dump_root, f"{seq_key}.csv")


def _runtime_dump_seq_tensor_dir(dump_root: str, seq_key: str) -> str:
    dump_root = str(dump_root or "")
    if not dump_root or dump_root.lower().endswith(".csv"):
        return ""
    return os.path.join(dump_root, "tensor_shards", seq_key)


def _path_has_files(path_value: str) -> bool:
    if not path_value or not os.path.isdir(path_value):
        return False
    try:
        with os.scandir(path_value) as it:
            for _ in it:
                return True
    except Exception:
        return False
    return False


def _cleanup_partial_runtime_dump(seq_dump_csv: str, seq_tensor_dir: str) -> None:
    if seq_dump_csv and os.path.isfile(seq_dump_csv):
        os.remove(seq_dump_csv)
    if seq_tensor_dir and os.path.isdir(seq_tensor_dir):
        shutil.rmtree(seq_tensor_dir)


def _combine_local_conflict_graph_diagnostics(sequence_stats: dict[str, dict]) -> dict:
    if not sequence_stats:
        return {}
    combined = {
        "graph_modes": sorted({str((stats or {}).get("graph_mode", "")) for stats in sequence_stats.values()}),
        "sequence_count": len(sequence_stats),
        "frames_seen": 0,
        "frames_with_eligible_clusters": 0,
        "frames_with_replaced_clusters": 0,
        "eligible_clusters": 0,
        "replaced_clusters": 0,
        "resolved_dets": 0,
        "matched_dets": 0,
        "null_dets": 0,
        "deferred_dets": 0,
        "blocked_tracks": 0,
        "gate_pass_clusters": 0,
        "gate_filtered_clusters": 0,
        "trigger_filtered_clusters": 0,
        "skipped_large_clusters": 0,
    }
    for stats in sequence_stats.values():
        for key in (
            "frames_seen",
            "frames_with_eligible_clusters",
            "frames_with_replaced_clusters",
            "eligible_clusters",
            "replaced_clusters",
            "resolved_dets",
            "matched_dets",
            "null_dets",
            "deferred_dets",
            "blocked_tracks",
            "gate_pass_clusters",
            "gate_filtered_clusters",
            "trigger_filtered_clusters",
            "skipped_large_clusters",
        ):
            combined[key] += int((stats or {}).get(key, 0) or 0)
    return combined


def build_tracking_modules_for_inference(config: dict, device: torch.device):
    """构建推理用的跟踪模块"""
    feature_dim = config.get("FEATURE_DIM", 256)
    num_bands = int(config.get("NUM_FREQ_BANDS", config.get("NUM_BANDS", 4)))
    assoc_mode = str(config.get("ASSOC_MODE", "logit")).lower()

    # Fast path: feature-only association doesn't use the learned trajectory_modeling / id_decoder at all.
    # Building the full decoder stack can require optional deps (e.g., mamba-ssm) and wastes init time.
    if assoc_mode == "feature":
        class _NoOpIDDecoder(torch.nn.Module):
            def __init__(self, num_id_vocabulary: int):
                super().__init__()
                self.num_id_vocabulary = int(num_id_vocabulary)

            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "ID decoder was not built because ASSOC_MODE=feature. "
                    "Switch ASSOC_MODE to 'logit' or 'hybrid' to enable it."
                )

        trajectory_modeling = torch.nn.Identity().to(device)
        id_decoder = _NoOpIDDecoder(num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 500)).to(device)
        return trajectory_modeling, id_decoder

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

    trajectory_modeling = trajectory_modeling.to(device)
    id_decoder = id_decoder.to(device)

    return trajectory_modeling, id_decoder


def load_checkpoint(checkpoint_path: str, trajectory_modeling, id_decoder, feature_extractor):
    """加载检查点

    Returns:
        dict with optional loaded submodule state_dicts (for inference-only modules).
        Currently contains:
            - memory_bank_state: state_dict for RuntimeTrackerByteTrack.memory_bank (without prefix)
            - laplace_assoc_state: state_dict for RuntimeTrackerByteTrack.laplace_assoc (without prefix)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 提取各模块的 state_dict
    trajectory_state = {}
    id_decoder_state = {}
    feature_proj_state = {}
    memory_bank_state = {}
    laplace_assoc_state = {}

    for key, value in state_dict.items():
        if key.startswith("trajectory_modeling."):
            new_key = key.replace("trajectory_modeling.", "")
            trajectory_state[new_key] = value
        elif key.startswith("id_decoder."):
            new_key = key.replace("id_decoder.", "")
            id_decoder_state[new_key] = value
        elif key.startswith("feature_extractor.feature_proj."):
            new_key = key.replace("feature_extractor.feature_proj.", "")
            feature_proj_state[new_key] = value
        elif key.startswith("memory_bank."):
            # MemoryBank is created inside RuntimeTrackerByteTrack at inference.
            # During training we register it as `model.memory_bank`, so it is saved in the checkpoint.
            # If we don't load it here, inference will silently use a random MemoryBank.
            new_key = key.replace("memory_bank.", "")
            memory_bank_state[new_key] = value
        elif key.startswith("laplace_assoc."):
            new_key = key.replace("laplace_assoc.", "")
            laplace_assoc_state[new_key] = value

    # 加载权重
    if trajectory_state:
        incompatible = trajectory_modeling.load_state_dict(trajectory_state, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Loaded trajectory_modeling with {len(trajectory_state)} keys "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if len(missing) > 0:
            print(f"  trajectory_modeling missing keys (sample): {missing[:3]}")
        if len(unexpected) > 0:
            print(f"  trajectory_modeling unexpected keys (sample): {unexpected[:3]}")

    if id_decoder_state:
        incompatible = id_decoder.load_state_dict(id_decoder_state, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Loaded id_decoder with {len(id_decoder_state)} keys "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if len(missing) > 0:
            print(f"  id_decoder missing keys (sample): {missing[:3]}")
        if len(unexpected) > 0:
            print(f"  id_decoder unexpected keys (sample): {unexpected[:3]}")

    if feature_proj_state:
        incompatible = feature_extractor.feature_proj.load_state_dict(feature_proj_state, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(
            f"Loaded feature_proj with {len(feature_proj_state)} keys "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if len(missing) > 0:
            print(f"  feature_proj missing keys (sample): {missing[:3]}")
        if len(unexpected) > 0:
            print(f"  feature_proj unexpected keys (sample): {unexpected[:3]}")

    if memory_bank_state:
        print(f"Found memory_bank in checkpoint with {len(memory_bank_state)} keys")
    if laplace_assoc_state:
        print(f"Found laplace_assoc in checkpoint with {len(laplace_assoc_state)} keys")

    return {
        "memory_bank_state": memory_bank_state,
        "laplace_assoc_state": laplace_assoc_state,
    }


def write_results(results: list, output_path: str):
    """写入 MOT 格式的跟踪结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for frame_id, track_id, x, y, w, h, conf in results:
            f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n")


def run_evaluation(
    config: dict,
    checkpoint_path: str,
    output_dir: str,
    detector_profile: str | None = None,
):
    """运行评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, selected_profile = resolve_bytetrack_profile(
        config=config,
        dataset_name=config.get("INFERENCE_DATASET", None),
        explicit_profile=detector_profile,
    )
    if selected_profile is not None:
        print(f"Using detector profile: {selected_profile}")
    print(
        "ByteTrack detector config: "
        f"exp={config.get('BYTETRACK_EXP_FILE')} | "
        f"ckpt={config.get('BYTETRACK_CKPT')} | "
        f"conf_thre={config.get('BYTETRACK_CONF_THRE', 0.01)} | "
        f"nms_thre={config.get('BYTETRACK_NMS_THRE', 0.7)} | "
        f"fp16={config.get('BYTETRACK_FP16', True)}"
    )
    det_source = str(config.get("BYTETRACK_DET_SOURCE", "model")).lower()
    if det_source in ("txt", "public"):
        det_source = "external"
    if det_source not in ("model", "external"):
        raise ValueError(f"Unsupported BYTETRACK_DET_SOURCE={det_source}")

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
    feature_extractor = ByteTrackFeatureExtractor(bytetrack_cfg, device)
    print(f"Loaded ByteTrack feature extractor")

    # Optional: external ReID encoder for association (inference-time only).
    # Uses the same crop->embed interface as public det pipeline, but can be applied to private det as well.
    reid_encoder = None
    if bool(config.get("ASSOC_USE_REID", False)):
        reid_cfg = {
            "PUBLIC_REID_BACKBONE": config.get(
                "ASSOC_REID_BACKBONE",
                config.get("PUBLIC_REID_BACKBONE", "torchreid:osnet_x1_0"),
            ),
            "PUBLIC_REID_WEIGHTS": config.get(
                "ASSOC_REID_WEIGHTS",
                config.get("PUBLIC_REID_WEIGHTS", None),
            ),
            "PUBLIC_REID_PRETRAINED": config.get(
                "ASSOC_REID_PRETRAINED",
                config.get("PUBLIC_REID_PRETRAINED", True),
            ),
            "PUBLIC_REID_INPUT_H": config.get(
                "ASSOC_REID_INPUT_H",
                config.get("PUBLIC_REID_INPUT_H", 256),
            ),
            "PUBLIC_REID_INPUT_W": config.get(
                "ASSOC_REID_INPUT_W",
                config.get("PUBLIC_REID_INPUT_W", 128),
            ),
            "PUBLIC_REID_BATCH_SIZE": config.get(
                "ASSOC_REID_BATCH_SIZE",
                config.get("PUBLIC_REID_BATCH_SIZE", 64),
            ),
            "PUBLIC_REID_L2_NORM": config.get(
                "ASSOC_REID_L2_NORM",
                config.get("PUBLIC_REID_L2_NORM", True),
            ),
            "PUBLIC_REID_PROJ_SEED": config.get(
                "ASSOC_REID_PROJ_SEED",
                config.get("PUBLIC_REID_PROJ_SEED", 12345),
            ),
            "PUBLIC_REID_BOX_EXPAND": float(config.get(
                "ASSOC_REID_BOX_EXPAND",
                config.get("PUBLIC_REID_BOX_EXPAND", 1.0),
            )),
        }
        reid_dim = int(config.get("ASSOC_REID_DIM", config.get("FEATURE_DIM", 256)))
        reid_dtype = torch.float16 if device.type == "cuda" else torch.float32
        try:
            reid_encoder = build_public_reid_encoder(
                config=reid_cfg,
                device=device,
                feature_dim=reid_dim,
                dtype=reid_dtype,
            )
            print(
                f"Built ReID encoder: {reid_cfg.get('PUBLIC_REID_BACKBONE')} "
                f"(dim={reid_dim}, dtype={reid_dtype})"
            )
        except Exception as exc:
            print(f"Warning: Failed to build ReID encoder; disabled. Error: {exc}")
            reid_encoder = None

    # 构建跟踪模块
    trajectory_modeling, id_decoder = build_tracking_modules_for_inference(config, device)
    print("Built tracking modules")

    # 加载检查点
    if checkpoint_path and os.path.exists(checkpoint_path):
        loaded = load_checkpoint(checkpoint_path, trajectory_modeling, id_decoder, feature_extractor)
        memory_bank_state = loaded.get("memory_bank_state", {}) if isinstance(loaded, dict) else {}
        laplace_assoc_state = loaded.get("laplace_assoc_state", {}) if isinstance(loaded, dict) else {}
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        if str(config.get("ASSOC_MODE", "logit")).lower() == "feature":
            print("No checkpoint loaded; ASSOC_MODE=feature so learned trajectory/ID modules are bypassed.")
        else:
            print("Warning: No checkpoint loaded, using random weights")
        memory_bank_state = {}
        laplace_assoc_state = {}

    # 获取数据集信息
    data_root = config["DATA_ROOT"]
    dataset_name = config["INFERENCE_DATASET"]
    split = config["INFERENCE_SPLIT"]

    # 获取序列列表
    dataset_dir = os.path.join(data_root, dataset_name, split)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    sequences = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    # Filter by detector version if specified (e.g., FRCNN only)
    detector_filter = config.get("DETECTOR_FILTER", None)
    if detector_filter:
        detector_filter = list(detector_filter) if not isinstance(detector_filter, (list, tuple)) else detector_filter
        filtered_sequences = [d for d in sequences if any(token in d for token in detector_filter)]
        if len(filtered_sequences) == 0 and len(sequences) > 0:
            print(
                f"Warning: DETECTOR_FILTER={detector_filter} filtered out all sequences for "
                f"{dataset_name}/{split}; fallback to unfiltered sequences."
            )
        else:
            sequences = filtered_sequences
    # Filter to validation sequences if requested
    if config.get("EVAL_ONLY_VAL", False):
        val_sequences = config.get("VAL_SEQUENCES", None)
        if isinstance(val_sequences, str):
            val_sequences = [val_sequences]
        if val_sequences:
            sequences = [d for d in sequences if any(d.startswith(base) for base in val_sequences)]
    print(f"Found {len(sequences)} sequences in {dataset_dir}")

    # 处理每个序列
    local_conflict_graph_seq_stats = {}
    for seq_name in tqdm(sequences, desc="Processing sequences"):
        seq_dir = os.path.join(dataset_dir, seq_name)
        img_dir = os.path.join(seq_dir, "img1")

        if not os.path.exists(img_dir):
            print(f"Warning: Image directory not found: {img_dir}")
            continue

        # 获取图像列表
        images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        if len(images) == 0:
            print(f"Warning: No images found in {img_dir}")
            continue

        # 获取序列尺寸
        import cv2
        first_img = cv2.imread(os.path.join(img_dir, images[0]))
        if first_img is None:
            print(f"Warning: Cannot read first image in {seq_name}")
            continue
        seq_h, seq_w = first_img.shape[:2]

        # Use the same sequence key format as training (GenerateIDLabels): "{dataset}/{split}/{seq}".
        # This only matters when ID_LABEL_STRATEGY != "random".
        seq_key = f"{dataset_name}/{split}/{seq_name}"
        output_path = os.path.join(output_dir, "tracker", dataset_name + "-" + split, seq_name + ".txt")
        dump_root = str(config.get("ASSOC_RUNTIME_DUMP_PATH", "") or "")
        seq_dump_csv = _runtime_dump_seq_csv_path(dump_root, seq_key)
        seq_tensor_dir = _runtime_dump_seq_tensor_dir(dump_root, seq_key)
        result_ready = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
        dump_ready = True
        if dump_root:
            dump_ready = os.path.isfile(seq_dump_csv) and os.path.getsize(seq_dump_csv) > 0
        if result_ready and dump_ready:
            print(f"Skipping completed sequence {seq_name} (existing tracker result + runtime dump found)")
            continue
        if dump_root:
            partial_dump = (
                (os.path.isfile(seq_dump_csv) and os.path.getsize(seq_dump_csv) > 0)
                or _path_has_files(seq_tensor_dir)
            )
            if result_ready and not dump_ready:
                print(
                    f"Re-running {seq_name}: tracker result exists but runtime dump is incomplete. "
                    "Removing stale per-sequence artifacts first."
                )
                os.remove(output_path)
                _cleanup_partial_runtime_dump(seq_dump_csv, seq_tensor_dir)
            elif (not result_ready) and partial_dump:
                print(f"Cleaning partial runtime dump for {seq_name} before resume")
                _cleanup_partial_runtime_dump(seq_dump_csv, seq_tensor_dir)

        external_detections = None
        if det_source == "external":
            det_path = resolve_external_det_path(
                config=config,
                dataset_name=dataset_name,
                split=split,
                seq_name=seq_name,
            )
            if not os.path.isfile(det_path):
                raise FileNotFoundError(
                    f"External detection file not found for sequence '{seq_name}': {det_path}"
                )
            external_detections = load_mot_detections(det_path)
            total_dets = sum(len(v) for v in external_detections.values())
            print(f"Loaded external detections for {seq_name}: {total_dets} boxes from {det_path}")

        # 创建跟踪器

        # Public detections ship with three detector variants (DPM/FRCNN/SDP) whose score scales differ.
        # Allow per-detector inference thresholds to avoid globally over-filtering (especially for DPM).
        miss_tolerance = config.get("MISS_TOLERANCE", 30)
        det_thresh = config.get("DET_THRESH", 0.3)
        newborn_thresh = config.get("NEWBORN_THRESH", 0.5)
        det_max_per_frame = config.get("DET_MAX_PER_FRAME", 0)
        id_thresh = config.get("ID_THRESH", 0.1)
        assoc_iou_gate = config.get("ASSOC_IOU_GATE", 0.0)
        assoc_mode = config.get("ASSOC_MODE", "logit")
        assoc_id_weight = config.get("ASSOC_ID_WEIGHT", 1.0)
        assoc_iou_weight = config.get("ASSOC_IOU_WEIGHT", 0.0)
        assoc_feat_weight = config.get("ASSOC_FEAT_WEIGHT", 1.0)
        seq_upper = str(seq_name).upper()
        det_suffix = None
        if seq_upper.endswith("DPM"):
            det_suffix = "DPM"
        elif seq_upper.endswith("FRCNN"):
            det_suffix = "FRCNN"
        elif seq_upper.endswith("SDP"):
            det_suffix = "SDP"
        if det_suffix is not None:
            miss_tolerance = config.get(f"MISS_TOLERANCE_{det_suffix}", miss_tolerance)
            det_thresh = config.get(f"DET_THRESH_{det_suffix}", det_thresh)
            newborn_thresh = config.get(f"NEWBORN_THRESH_{det_suffix}", newborn_thresh)
            det_max_per_frame = config.get(f"DET_MAX_PER_FRAME_{det_suffix}", det_max_per_frame)
            id_thresh = config.get(f"ID_THRESH_{det_suffix}", id_thresh)
            assoc_iou_gate = config.get(f"ASSOC_IOU_GATE_{det_suffix}", assoc_iou_gate)
            assoc_mode = config.get(f"ASSOC_MODE_{det_suffix}", assoc_mode)
            assoc_id_weight = config.get(f"ASSOC_ID_WEIGHT_{det_suffix}", assoc_id_weight)
            assoc_iou_weight = config.get(f"ASSOC_IOU_WEIGHT_{det_suffix}", assoc_iou_weight)
            assoc_feat_weight = config.get(f"ASSOC_FEAT_WEIGHT_{det_suffix}", assoc_feat_weight)

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
            area_thresh=config.get("AREA_THRESH", 0),
            num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 500),
            feature_dim=config.get("FEATURE_DIM", 256),
            track_window=config.get("INFERENCE_TRACK_WINDOW", config.get("MAX_SEQ_LEN", 30)),
            matching_method=config.get("ASSOC_MATCHING", "hungarian"),
            assoc_iou_gate=assoc_iou_gate,
            assoc_id_weight=assoc_id_weight,
            assoc_iou_weight=assoc_iou_weight,
            assoc_logit_temp=config.get("ASSOC_LOGIT_TEMP", 1.0),
            assoc_use_det_score=config.get("ASSOC_USE_DET_SCORE", False),
            assoc_mode=assoc_mode,
            assoc_feat_weight=assoc_feat_weight,
            assoc_feat_agg=config.get("ASSOC_FEAT_AGG", "last"),
            assoc_feat_k=config.get("ASSOC_FEAT_K", 5),
            assoc_feat_tau=config.get("ASSOC_FEAT_TAU", 1.0),
            assoc_feat_source=config.get("ASSOC_FEAT_SOURCE", "yolox"),
            assoc_feat_score_mode=config.get("ASSOC_FEAT_SCORE_MODE", "raw"),
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
            use_kalman=config.get("ASSOC_USE_KALMAN", True),
            id_label_strategy=config.get("ID_LABEL_STRATEGY", "random"),
            sequence_name=seq_key,
            use_confidence_calibration=config.get("USE_CONFIDENCE_CALIBRATION", False),
            calibration_strength=config.get("CALIBRATION_STRENGTH", 0.5),
            min_confidence=config.get("MIN_CONFIDENCE", 0.1),
            use_tta=config.get("USE_TTA", False),
            tta_scales=config.get("TTA_SCALES", [0.8, 1.0, 1.2]),
            tta_flip=config.get("TTA_FLIP", True),
            tta_fusion=config.get("TTA_FUSION", "average"),
            use_memory_bank=config.get("USE_MEMORY_BANK", False),
            memory_lambda=config.get("MEMORY_LAMBDA", 0.9),
            memory_update_threshold=config.get("MEMORY_UPDATE_THRESHOLD", 0.5),
            det_source=det_source,
            external_detections=external_detections,
            assoc_runtime_dump_path=str(config.get("ASSOC_RUNTIME_DUMP_PATH", "")),
            assoc_runtime_dump_topk=int(config.get("ASSOC_RUNTIME_DUMP_TOPK", 8)),
            assoc_runtime_dump_min_score=float(config.get("ASSOC_RUNTIME_DUMP_MIN_SCORE", 0.0)),
            assoc_runtime_dump_save_tensors=bool(config.get("ASSOC_RUNTIME_DUMP_SAVE_TENSORS", False)),
            assoc_runtime_dump_npz_every_n_groups=int(config.get("ASSOC_RUNTIME_DUMP_NPZ_EVERY_N_GROUPS", 2048)),
        )
        # Load trained MemoryBank weights (if enabled).
        if getattr(tracker, "memory_bank", None) is not None:
            if memory_bank_state:
                try:
                    tracker.memory_bank.load_state_dict(memory_bank_state, strict=True)
                    tracker.memory_bank.eval()
                    print(f"Loaded memory_bank into runtime tracker ({len(memory_bank_state)} keys)")
                except Exception as e:
                    tracker.memory_bank = None
                    print(f"Warning: Failed to load memory_bank state_dict; disabled. Error: {e}")
            else:
                # Safety: never run inference with a randomly initialized MemoryBank.
                # If the checkpoint doesn't have memory_bank weights, disable it to keep results reliable.
                tracker.memory_bank = None
                print(
                    "Warning: USE_MEMORY_BANK=True but checkpoint has no memory_bank weights; "
                    "disabled MemoryBank to avoid random-weight contamination."
                )

        if getattr(tracker, "laplace_assoc", None) is not None:
            if laplace_assoc_state:
                try:
                    tracker.laplace_assoc.load_state_dict(laplace_assoc_state, strict=True)
                    tracker.laplace_assoc.eval()
                    print(f"Loaded laplace_assoc into runtime tracker ({len(laplace_assoc_state)} keys)")
                except Exception as e:
                    print(f"Warning: Failed to load laplace_assoc state_dict; keeping default init. Error: {e}")
            else:
                print("Warning: checkpoint has no laplace_assoc weights; using default Laplace init.")

        # 处理每帧
        results = []
        for img_name in tqdm(images, desc=f"Processing {seq_name}", leave=False):
            img_path = os.path.join(img_dir, img_name)
            frame_id = int(os.path.splitext(img_name)[0])

            # 跟踪
            track_results = tracker.update(img_path)

            # 收集结果
            for i in range(len(track_results["id"])):
                track_id = track_results["id"][i].item()
                if track_id <= 0:
                    continue
                bbox = track_results["bbox"][i]
                x, y, w, h = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()
                conf = track_results["score"][i].item()
                results.append((frame_id, track_id, x, y, w, h, conf))

        # Track interpolation (optional post-processing)
        if config.get("USE_TRACK_INTERPOLATION", False):
            try:
                from models.motip.advanced_strategies import TrackInterpolation
                interp = TrackInterpolation(
                    max_gap=config.get("INTERPOLATION_MAX_GAP", 10),
                    min_track_length=config.get("INTERPOLATION_MIN_LENGTH", 3),
                    interpolation_method=config.get("INTERPOLATION_METHOD", "linear"),
                )

                # Build per-track dicts
                track_boxes = {}
                track_scores = {}
                for (frame_id, track_id, x, y, w, h, conf) in results:
                    track_boxes.setdefault(track_id, {})[frame_id] = np.array([x, y, w, h], dtype=np.float32)
                    track_scores.setdefault(track_id, {})[frame_id] = float(conf)

                processed_boxes = interp.process_tracks(track_boxes)

                # Rebuild results with interpolated frames
                new_results = []
                for track_id, frames in processed_boxes.items():
                    orig_scores = track_scores.get(track_id, {})
                    for frame_id, box in frames.items():
                        if frame_id in orig_scores:
                            conf = orig_scores[frame_id]
                        else:
                            # Conservative confidence for interpolated frames
                            # Use nearest known neighbors if available
                            prev_frames = [f for f in orig_scores.keys() if f < frame_id]
                            next_frames = [f for f in orig_scores.keys() if f > frame_id]
                            if prev_frames and next_frames:
                                conf = min(orig_scores[max(prev_frames)], orig_scores[min(next_frames)])
                            elif prev_frames:
                                conf = orig_scores[max(prev_frames)]
                            elif next_frames:
                                conf = orig_scores[min(next_frames)]
                            else:
                                conf = 1.0
                        x, y, w, h = box.tolist()
                        new_results.append((frame_id, track_id, x, y, w, h, conf))
                results = new_results
            except Exception:
                pass

        # 写入结果
        write_results(results, output_path)
        seq_lcg_diag = {}
        if hasattr(tracker, "get_local_conflict_graph_diagnostics"):
            try:
                seq_lcg_diag = tracker.get_local_conflict_graph_diagnostics()
            except Exception:
                seq_lcg_diag = {}
        if seq_lcg_diag and str(seq_lcg_diag.get("graph_mode", "disabled")) != "disabled":
            local_conflict_graph_seq_stats[seq_name] = seq_lcg_diag
        # Explicitly reset at sequence end so pending runtime dump buffers are flushed.
        tracker.reset()
        print(f"Written {len(results)} tracks for {seq_name}")

    print(f"Results saved to {output_dir}")
    if local_conflict_graph_seq_stats:
        diagnostics_path = os.path.join(output_dir, "local_conflict_graph_diagnostics.json")
        payload = {
            "sequence_stats": local_conflict_graph_seq_stats,
            "combined": _combine_local_conflict_graph_diagnostics(local_conflict_graph_seq_stats),
        }
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    # 运行 TrackEval
    if config.get("RUN_TRACKEVAL", True):
        if _has_any_gt_files(dataset_dir, sequences):
            run_trackeval(config, output_dir)
        else:
            print(
                f"Warning: No GT files found under {dataset_dir} (split={split}); "
                "skipping TrackEval evaluation."
            )


def run_trackeval(config: dict, output_dir: str):
    """运行 TrackEval 评估"""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TrackEval"))
        import trackeval
    except ImportError:
        print("Warning: TrackEval not found, skipping evaluation")
        return

    dataset_name = config["INFERENCE_DATASET"]
    split = config["INFERENCE_SPLIT"]

    # Optional: filter short tracks for evaluation to avoid Identity OOM with many 1-frame IDs.
    min_track_len = int(config.get("EVAL_MIN_TRACK_LEN", 1))
    trackers_root = os.path.join(output_dir, "tracker")
    if min_track_len > 1 and os.path.isdir(trackers_root):
        filtered_root = os.path.join(output_dir, f"tracker_min{min_track_len}")
        os.makedirs(filtered_root, exist_ok=True)
        for tracker_name in os.listdir(trackers_root):
            src_dir = os.path.join(trackers_root, tracker_name)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(filtered_root, tracker_name)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in os.listdir(src_dir):
                if not fname.endswith(".txt"):
                    continue
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dst_dir, fname)
                # Skip if already filtered
                if os.path.isfile(dst_path):
                    continue
                track_counts = {}
                lines = []
                with open(src_path, "r") as f:
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
                        track_counts[tid] = track_counts.get(tid, 0) + 1
                        lines.append((tid, line))
                keep_ids = {tid for tid, c in track_counts.items() if c >= min_track_len}
                with open(dst_path, "w") as f:
                    for tid, line in lines:
                        if tid in keep_ids:
                            f.write(line + "\n")
        trackers_root = filtered_root

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_CONFIG"] = False
    eval_config["TIME_PROGRESS"] = True
    eval_config["OUTPUT_SUMMARY"] = True
    eval_config["OUTPUT_DETAILED"] = True
    eval_config["PLOT_CURVES"] = False
    eval_config["NUM_PARALLEL_CORES"] = 1

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = os.path.join(config["DATA_ROOT"], dataset_name, split)
    dataset_config["TRACKERS_FOLDER"] = trackers_root
    dataset_config["TRACKERS_TO_EVAL"] = [dataset_name + "-" + split]
    dataset_config["SPLIT_TO_EVAL"] = split
    dataset_config["BENCHMARK"] = dataset_name
    dataset_config["SKIP_SPLIT_FOL"] = True
    dataset_config["TRACKER_SUB_FOLDER"] = ""

    # Provide explicit seq info to avoid requiring seqmap files.
    seq_info = {}
    gt_folder = dataset_config["GT_FOLDER"]
    if os.path.isdir(gt_folder):
        sequences = sorted([d for d in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, d))])
        detector_filter = config.get("DETECTOR_FILTER", None)
        if detector_filter:
            detector_filter = list(detector_filter) if not isinstance(detector_filter, (list, tuple)) else detector_filter
            filtered_sequences = [d for d in sequences if any(token in d for token in detector_filter)]
            if len(filtered_sequences) == 0 and len(sequences) > 0:
                print(
                    f"Warning: DETECTOR_FILTER={detector_filter} filtered out all GT sequences for "
                    f"{dataset_name}/{split}; fallback to unfiltered sequences."
                )
            else:
                sequences = filtered_sequences
        if config.get("EVAL_ONLY_VAL", False):
            val_sequences = config.get("VAL_SEQUENCES", None)
            if isinstance(val_sequences, str):
                val_sequences = [val_sequences]
            if val_sequences:
                sequences = [d for d in sequences if any(d.startswith(base) for base in val_sequences)]
        for seq in sequences:
            ini_file = os.path.join(gt_folder, seq, "seqinfo.ini")
            if not os.path.isfile(ini_file):
                continue
            try:
                import configparser
                ini_data = configparser.ConfigParser()
                ini_data.read(ini_file)
                seq_len = int(ini_data["Sequence"]["seqLength"])
                seq_info[seq] = seq_len
            except Exception:
                continue
    if seq_info:
        dataset_config["SEQ_INFO"] = seq_info

    metrics_list_cfg = config.get("EVAL_METRICS", ["HOTA", "CLEAR", "Identity"])
    if isinstance(metrics_list_cfg, str):
        metrics_list_cfg = [metrics_list_cfg]
    metrics_config = {"METRICS": list(metrics_list_cfg), "THRESHOLD": 0.5}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    evaluator.evaluate(dataset_list, metrics_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--inference-model", type=str, default=None)
    parser.add_argument("--inference-dataset", type=str, default=None)
    parser.add_argument("--inference-split", type=str, default=None)
    parser.add_argument("--inference-mode", type=str, default="evaluate", choices=["evaluate", "submit"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--det-thresh", type=float, default=None)
    parser.add_argument("--newborn-thresh", type=float, default=None)
    parser.add_argument("--id-thresh", type=float, default=None)
    parser.add_argument("--assoc-iou-gate", type=float, default=None)
    parser.add_argument("--assoc-mode", type=str, default=None, choices=["logit", "hybrid", "feature"])
    parser.add_argument("--assoc-id-weight", type=float, default=None)
    parser.add_argument("--assoc-iou-weight", type=float, default=None)
    parser.add_argument("--assoc-feat-weight", type=float, default=None)
    parser.add_argument("--assoc-feat-source", type=str, default=None, choices=["yolox", "reid"])
    parser.add_argument("--assoc-feat-score-mode", type=str, default=None, choices=["raw", "softmax"])
    parser.add_argument("--assoc-feat-tau", type=float, default=None)
    parser.add_argument("--assoc-feat-agg", type=str, default=None, choices=["last", "mean"])
    parser.add_argument("--assoc-feat-k", type=int, default=None)
    parser.add_argument("--assoc-use-det-score", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-use-runtime-replay", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-runtime-replay-checkpoint", type=str, default=None)
    parser.add_argument("--assoc-runtime-replay-hard-margin-gate", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-runtime-replay-margin-threshold", type=float, default=None)
    parser.add_argument("--assoc-use-competition", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-competition-checkpoint", type=str, default=None)
    parser.add_argument("--assoc-competition-topk", type=int, default=None)
    parser.add_argument("--assoc-competition-delta-scale", type=float, default=None)
    parser.add_argument("--assoc-competition-mode", type=str, default=None, choices=["noop", "rerank_only", "rerank_minimal", "oracle_rerank"])
    parser.add_argument("--assoc-competition-hard-action", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-competition-margin-threshold", type=float, default=None)
    parser.add_argument("--assoc-use-competition-oracle", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-competition-oracle-csv", type=str, default=None)
    parser.add_argument("--assoc-use-local-conflict-graph", type=int, default=None, choices=[0, 1])
    parser.add_argument(
        "--assoc-local-conflict-graph-mode",
        type=str,
        default=None,
        choices=["disabled", "oracle_full", "oracle_commit_matches", "learned_commit"],
    )
    parser.add_argument("--assoc-use-local-conflict-graph-oracle", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-local-conflict-graph-oracle-jsonl", type=str, default=None)
    parser.add_argument("--assoc-local-conflict-graph-checkpoint", type=str, default=None)
    parser.add_argument("--assoc-local-conflict-graph-topk", type=int, default=None)
    parser.add_argument("--assoc-local-conflict-graph-min-detections", type=int, default=None)
    parser.add_argument("--assoc-local-conflict-graph-min-committed-matches", type=int, default=None)
    parser.add_argument("--assoc-local-conflict-graph-max-detections", type=int, default=None)
    parser.add_argument("--assoc-local-conflict-graph-max-tracks", type=int, default=None)
    parser.add_argument("--assoc-local-conflict-graph-cluster-gate-thresh", type=float, default=None)
    parser.add_argument("--assoc-local-conflict-graph-cluster-gate-temp", type=float, default=None)
    parser.add_argument("--assoc-local-conflict-graph-cluster-gate-bias", type=float, default=None)
    parser.add_argument("--assoc-local-conflict-graph-host-variant", type=str, default=None)
    # Geometry / two-stage matching (opt-in)
    parser.add_argument("--assoc-bbox-dist-weight", type=float, default=None)
    parser.add_argument("--assoc-bbox-dist-tau", type=float, default=None)
    parser.add_argument("--assoc-bbox-dist-use-cal-factor", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-two-stage", type=int, default=None, choices=[0, 1])
    parser.add_argument("--assoc-stage2-iou-gate", type=float, default=None)
    parser.add_argument("--assoc-stage2-id-thresh", type=float, default=None)
    parser.add_argument("--assoc-stage2-bbox-weight", type=float, default=None)
    parser.add_argument("--assoc-reid-box-expand", type=float, default=None)
    parser.add_argument("--det-max-per-frame", type=int, default=None)
    parser.add_argument("--miss-tolerance", type=int, default=None)
    parser.add_argument("--detector-profile", type=str, default=None)
    parser.add_argument("--det-source", type=str, default=None, choices=["model", "external", "txt", "public"])
    parser.add_argument("--external-det-root", type=str, default=None)
    parser.add_argument("--external-det-pattern", type=str, default=None)
    parser.add_argument(
        "--detector-filter",
        type=str,
        default=None,
        help="Comma-separated tokens to filter sequences (e.g., 'FRCNN' or 'DPM'). Matches if token is contained in seq name.",
    )
    parser.add_argument("--eval-only-val", action="store_true")
    parser.add_argument("--val-sequences", type=str, default=None)
    args = parser.parse_args()

    # 加载配置
    config = yaml_to_dict(args.config_path)
    if config.get("SUPER_CONFIG_PATH"):
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])

    # 覆盖配置
    if args.data_root:
        config["DATA_ROOT"] = args.data_root
    if args.inference_dataset:
        config["INFERENCE_DATASET"] = args.inference_dataset
    if args.inference_split:
        config["INFERENCE_SPLIT"] = args.inference_split
    if args.det_thresh is not None:
        config["DET_THRESH"] = float(args.det_thresh)
    if args.newborn_thresh is not None:
        config["NEWBORN_THRESH"] = float(args.newborn_thresh)
    if args.id_thresh is not None:
        config["ID_THRESH"] = float(args.id_thresh)
    if args.assoc_iou_gate is not None:
        config["ASSOC_IOU_GATE"] = float(args.assoc_iou_gate)
    if args.assoc_mode is not None:
        config["ASSOC_MODE"] = str(args.assoc_mode)
    if args.assoc_id_weight is not None:
        config["ASSOC_ID_WEIGHT"] = float(args.assoc_id_weight)
    if args.assoc_iou_weight is not None:
        config["ASSOC_IOU_WEIGHT"] = float(args.assoc_iou_weight)
    if args.assoc_feat_weight is not None:
        config["ASSOC_FEAT_WEIGHT"] = float(args.assoc_feat_weight)
    if args.assoc_feat_source is not None:
        config["ASSOC_FEAT_SOURCE"] = str(args.assoc_feat_source)
    if args.assoc_feat_score_mode is not None:
        config["ASSOC_FEAT_SCORE_MODE"] = str(args.assoc_feat_score_mode)
    if args.assoc_feat_tau is not None:
        config["ASSOC_FEAT_TAU"] = float(args.assoc_feat_tau)
    if args.assoc_feat_agg is not None:
        config["ASSOC_FEAT_AGG"] = str(args.assoc_feat_agg)
    if args.assoc_feat_k is not None:
        config["ASSOC_FEAT_K"] = int(args.assoc_feat_k)
    if args.assoc_use_det_score is not None:
        config["ASSOC_USE_DET_SCORE"] = bool(int(args.assoc_use_det_score))
    if args.assoc_use_runtime_replay is not None:
        config["ASSOC_USE_RUNTIME_REPLAY"] = bool(int(args.assoc_use_runtime_replay))
    if args.assoc_runtime_replay_checkpoint is not None:
        config["ASSOC_RUNTIME_REPLAY_CHECKPOINT"] = str(args.assoc_runtime_replay_checkpoint)
    if args.assoc_runtime_replay_hard_margin_gate is not None:
        config["ASSOC_RUNTIME_REPLAY_HARD_MARGIN_GATE"] = bool(int(args.assoc_runtime_replay_hard_margin_gate))
    if args.assoc_runtime_replay_margin_threshold is not None:
        config["ASSOC_RUNTIME_REPLAY_MARGIN_THRESHOLD"] = float(args.assoc_runtime_replay_margin_threshold)
    if args.assoc_use_competition is not None:
        config["ASSOC_USE_COMPETITION"] = bool(int(args.assoc_use_competition))
    if args.assoc_competition_checkpoint is not None:
        config["ASSOC_COMPETITION_CHECKPOINT"] = str(args.assoc_competition_checkpoint)
    if args.assoc_competition_topk is not None:
        config["ASSOC_COMPETITION_TOPK"] = int(args.assoc_competition_topk)
    if args.assoc_competition_delta_scale is not None:
        config["ASSOC_COMPETITION_DELTA_SCALE"] = float(args.assoc_competition_delta_scale)
    if args.assoc_competition_mode is not None:
        config["ASSOC_COMPETITION_MODE"] = str(args.assoc_competition_mode)
    if args.assoc_competition_hard_action is not None:
        config["ASSOC_COMPETITION_HARD_ACTION"] = bool(int(args.assoc_competition_hard_action))
    if args.assoc_competition_margin_threshold is not None:
        config["ASSOC_COMPETITION_MARGIN_THRESHOLD"] = float(args.assoc_competition_margin_threshold)
    if args.assoc_use_competition_oracle is not None:
        config["ASSOC_USE_COMPETITION_ORACLE"] = bool(int(args.assoc_use_competition_oracle))
    if args.assoc_competition_oracle_csv is not None:
        config["ASSOC_COMPETITION_ORACLE_CSV"] = str(args.assoc_competition_oracle_csv)
    if args.assoc_use_local_conflict_graph is not None:
        config["ASSOC_USE_LOCAL_CONFLICT_GRAPH"] = bool(int(args.assoc_use_local_conflict_graph))
    if args.assoc_local_conflict_graph_mode is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_MODE"] = str(args.assoc_local_conflict_graph_mode)
    if args.assoc_use_local_conflict_graph_oracle is not None:
        config["ASSOC_USE_LOCAL_CONFLICT_GRAPH_ORACLE"] = bool(int(args.assoc_use_local_conflict_graph_oracle))
    if args.assoc_local_conflict_graph_oracle_jsonl is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_ORACLE_JSONL"] = str(args.assoc_local_conflict_graph_oracle_jsonl)
    if args.assoc_local_conflict_graph_checkpoint is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_CHECKPOINT"] = str(args.assoc_local_conflict_graph_checkpoint)
    if args.assoc_local_conflict_graph_topk is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_TOPK"] = int(args.assoc_local_conflict_graph_topk)
    if args.assoc_local_conflict_graph_min_detections is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_MIN_DETECTIONS"] = int(args.assoc_local_conflict_graph_min_detections)
    if args.assoc_local_conflict_graph_min_committed_matches is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_MIN_COMMITTED_MATCHES"] = int(
            args.assoc_local_conflict_graph_min_committed_matches
        )
    if args.assoc_local_conflict_graph_max_detections is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_MAX_DETECTIONS"] = int(args.assoc_local_conflict_graph_max_detections)
    if args.assoc_local_conflict_graph_max_tracks is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_MAX_TRACKS"] = int(args.assoc_local_conflict_graph_max_tracks)
    if args.assoc_local_conflict_graph_cluster_gate_thresh is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_THRESH"] = float(
            args.assoc_local_conflict_graph_cluster_gate_thresh
        )
    if args.assoc_local_conflict_graph_cluster_gate_temp is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_TEMP"] = float(
            args.assoc_local_conflict_graph_cluster_gate_temp
        )
    if args.assoc_local_conflict_graph_cluster_gate_bias is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_CLUSTER_GATE_BIAS"] = float(
            args.assoc_local_conflict_graph_cluster_gate_bias
        )
    if args.assoc_local_conflict_graph_host_variant is not None:
        config["ASSOC_LOCAL_CONFLICT_GRAPH_HOST_VARIANT"] = str(args.assoc_local_conflict_graph_host_variant)
    if args.assoc_bbox_dist_weight is not None:
        config["ASSOC_BBOX_DIST_WEIGHT"] = float(args.assoc_bbox_dist_weight)
    if args.assoc_bbox_dist_tau is not None:
        config["ASSOC_BBOX_DIST_TAU"] = float(args.assoc_bbox_dist_tau)
    if args.assoc_bbox_dist_use_cal_factor is not None:
        config["ASSOC_BBOX_DIST_USE_CAL_FACTOR"] = bool(int(args.assoc_bbox_dist_use_cal_factor))
    if args.assoc_two_stage is not None:
        config["ASSOC_TWO_STAGE"] = bool(int(args.assoc_two_stage))
    if args.assoc_stage2_iou_gate is not None:
        config["ASSOC_STAGE2_IOU_GATE"] = float(args.assoc_stage2_iou_gate)
    if args.assoc_stage2_id_thresh is not None:
        config["ASSOC_STAGE2_ID_THRESH"] = float(args.assoc_stage2_id_thresh)
    if args.assoc_stage2_bbox_weight is not None:
        config["ASSOC_STAGE2_BBOX_WEIGHT"] = float(args.assoc_stage2_bbox_weight)
    if args.assoc_reid_box_expand is not None:
        config["ASSOC_REID_BOX_EXPAND"] = float(args.assoc_reid_box_expand)
    if args.det_max_per_frame is not None:
        config["DET_MAX_PER_FRAME"] = int(args.det_max_per_frame)
    if args.miss_tolerance is not None:
        config["MISS_TOLERANCE"] = int(args.miss_tolerance)
    if args.det_source is not None:
        config["BYTETRACK_DET_SOURCE"] = str(args.det_source)
    if args.external_det_root is not None:
        config["EXTERNAL_DET_ROOT"] = str(args.external_det_root)
    if args.external_det_pattern is not None:
        config["EXTERNAL_DET_PATTERN"] = str(args.external_det_pattern)
    if args.detector_filter:
        tokens = [s.strip() for s in str(args.detector_filter).split(",") if s.strip()]
        if tokens:
            config["DETECTOR_FILTER"] = tokens
    if args.eval_only_val:
        config["EVAL_ONLY_VAL"] = True
    if args.val_sequences:
        seqs = [s.strip() for s in args.val_sequences.split(",") if s.strip()]
        if seqs:
            config["VAL_SEQUENCES"] = seqs

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("outputs", config.get("EXP_NAME", "bytetrack_eval"), "inference")

    # 运行评估
    run_evaluation(
        config=config,
        checkpoint_path=args.inference_model,
        output_dir=output_dir,
        detector_profile=args.detector_profile,
    )


if __name__ == "__main__":
    main()
