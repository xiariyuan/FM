#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit and evaluate with PUBLIC DETECTIONS for MOT17/MOT20.

Usage:
    python submit_public.py \
        --config-path configs/r50_dino_fa_mot_v2_mot17.yaml \
        --inference-model ./outputs/fa_mot_v2_resume/checkpoint_20.pth \
        --inference-mode evaluate \
        --inference-dataset MOT17 \
        --inference-split train \
        --outputs-dir ./outputs/public_eval \
        --data-root /gemini/code/datasets

Key difference from submit_and_evaluate.py:
    - Uses PUBLIC detections from det/det.txt instead of DINO detections
    - Matches DINO features with public detection boxes via IoU
    - Outputs use public detection box coordinates
"""

import os
import time
import torch
import subprocess
from torch.utils.data import DataLoader
from accelerate import Accelerator, PartialState

from configs.util import yaml_to_dict, load_super_config, update_config
from runtime_option import runtime_option
from log.logger import Logger
from models.motip import build as build_model
from models.misc import load_checkpoint
from data.mot17 import MOT17
from data.mot20 import MOT20
from data.seq_dataset import SeqDataset
from models.runtime_tracker_public import RuntimeTrackerPublic, load_public_detections
from models.public_reid import build_public_reid_encoder
from models.bytetrack_detector import ByteTrackDetector, ByteTrackDetConfig


# Dataset classes registry
dataset_classes = {
    "MOT17": MOT17,
    "MOT20": MOT20,
}


def submit_and_evaluate_public(config: dict):
    """Main function for public detection evaluation"""
    accelerator = Accelerator()
    state = PartialState()

    # Setup output directory first (needed for Logger)
    outputs_dir = config["OUTPUTS_DIR"]
    os.makedirs(outputs_dir, exist_ok=True)

    # Init Logger
    logger = Logger(
        logdir=str(outputs_dir),
        use_wandb=False,
        config=config,
    )

    # Build model
    model, _ = build_model(config=config)
    
    # Load checkpoint
    checkpoint_path = config["INFERENCE_MODEL"]
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    device = state.device
    model = model.to(device)
    model.eval()

    # Set dtype
    dtype_str = config.get("INFERENCE_DTYPE", "FP32")
    if dtype_str == "FP32":
        dtype = torch.float32
    elif dtype_str == "FP16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown dtype '{dtype_str}'.")

    # ---------------------------------------------------------------------
    # P1-b: Optional public det + external ReID embeddings (crop -> encoder)
    # ---------------------------------------------------------------------
    use_public_reid = bool(config.get("USE_PUBLIC_REID", False))
    public_reid_encoder = None
    if use_public_reid:
        public_reid_encoder = build_public_reid_encoder(
            config=config,
            device=device,
            feature_dim=int(config.get("FEATURE_DIM", 256)),
            dtype=dtype,
        )

    # Get dataset info
    dataset_name = config["INFERENCE_DATASET"]
    data_split = config["INFERENCE_SPLIT"]
    data_root = config["DATA_ROOT"]
    is_evaluate = config["INFERENCE_MODE"] == "evaluate"
    
    # Get inference parameters
    image_max_shorter = config.get("INFERENCE_MAX_SHORTER", 800)
    image_max_longer = config.get("INFERENCE_MAX_LONGER", 1536)
    size_divisibility = config.get("SIZE_DIVISIBILITY", 0)
    det_thresh = config["DET_THRESH"]
    newborn_thresh = config["NEWBORN_THRESH"]
    id_thresh = config["ID_THRESH"]
    area_thresh = config.get("AREA_THRESH", 0)
    miss_tolerance = config["MISS_TOLERANCE"]
    min_track_len = config.get("MIN_TRACK_LEN", 0)
    iou_thresh = config.get("IOU_THRESH", 0.5)
    use_sigmoid = config.get("USE_FOCAL_LOSS", False)
    assignment_protocol = config.get("ASSIGNMENT_PROTOCOL", "object-priority")
    inference_only_detr = config.get("INFERENCE_ONLY_DETR", False) or config.get("ONLY_DETR", False)
    det_source_cfg = str(config.get("DET_SOURCE", "public")).lower()
    cache_private_det = bool(config.get("CACHE_PRIVATE_DET", False))
    cache_private_det_dir = str(config.get("CACHE_PRIVATE_DET_DIR", "outputs/private_det_cache"))
    
    # Build dataset
    if dataset_name not in dataset_classes:
        if dataset_name.lower() == "mot20" and "MOT17" in dataset_classes:
            dataset_classes[dataset_name] = dataset_classes["MOT17"]
        else:
            raise KeyError(f"Dataset {dataset_name} is not registered.")

    # Get VAL_SEQUENCES for filtering (only evaluate on validation sequences)
    val_sequences = config.get("VAL_SEQUENCES", None)
    if val_sequences is not None and len(val_sequences) > 0:
        logger.info(f"Filtering to VAL_SEQUENCES: {val_sequences}")
        sequence_include = val_sequences
    else:
        sequence_include = None

    inference_dataset = dataset_classes[dataset_name](
        data_root=data_root,
        split=data_split,
        load_annotation=False,
        sequence_include=sequence_include,
    )
    
    # Get sequences
    sequence_names = sorted(list(inference_dataset.sequence_infos.keys()))
    if len(sequence_names) == 0:
        raise ValueError(
            f"No sequences found for dataset '{dataset_name}' split '{data_split}' at '{inference_dataset.data_dir}'. "
            f"Please check data_root/split path and dataset name."
        )
    logger.info(f"Found {len(sequence_names)} sequences")
    
    # Filter sequences for multi-GPU (same logic as original)
    if len(sequence_names) <= state.process_index:
        logger.info(f"Number of sequences is smaller than the number of processes, "
                   f"a fake sequence will be processed on process {state.process_index}.")
        sequence_names = [sequence_names[0]]
        is_fake = True
    else:
        sequence_names = [s for i, s in enumerate(sequence_names) if i % state.num_processes == state.process_index]
        is_fake = False
    
    # Process each sequence
    bytetrack_detector = None
    bytetrack_cfg = None
    if det_source_cfg == "bytetrack":
        test_size_cfg = config.get("BYTETRACK_TEST_SIZE", None)
        test_size = tuple(test_size_cfg) if test_size_cfg is not None else None
        bytetrack_cfg = ByteTrackDetConfig(
            exp_file=str(config["BYTETRACK_EXP_FILE"]),
            ckpt=str(config["BYTETRACK_CKPT"]),
            fp16=bool(config.get("BYTETRACK_FP16", True)),
            test_size=test_size,
            conf_thre=float(config.get("BYTETRACK_CONF_THRE", 0.01)),
            nms_thre=float(config.get("BYTETRACK_NMS_THRE", 0.7)),
            class_agnostic_nms=bool(config.get("BYTETRACK_CLASS_AGNOSTIC_NMS", True)),
        )

    for sequence_name in sequence_names:
        logger.info(f"Processing sequence: {sequence_name}")
        
        # Build sequence dataset and loader
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
        sequence_hw = sequence_dataset.seq_hw()
        
        det_source = det_source_cfg
        public_detections = None
        private_det_cache = None
        cache_det_path = None

        if det_source_cfg == "bytetrack" and cache_private_det:
            cache_det_path = os.path.join(
                cache_private_det_dir, dataset_name, data_split, sequence_name, "det", "det.txt"
            )
            if os.path.exists(cache_det_path):
                logger.info(f"Loading cached private detections from {cache_det_path}")
                public_detections = load_public_detections(cache_det_path)
                det_source = "public"
            else:
                private_det_cache = {}

        if det_source == "bytetrack":
            if bytetrack_cfg is None:
                raise RuntimeError("DET_SOURCE=bytetrack but ByteTrack config is missing")
            if bytetrack_detector is None:
                bytetrack_detector = ByteTrackDetector(cfg=bytetrack_cfg, device=device)
                logger.info(
                    f"Using ByteTrack detector: exp={bytetrack_cfg.exp_file}, ckpt={bytetrack_cfg.ckpt}"
                )
        else:
            if public_detections is None:
                if dataset_name == "MOT17":
                    det_path = os.path.join(data_root, "MOT17", data_split, sequence_name, "det", "det.txt")
                elif dataset_name == "MOT20":
                    det_path = os.path.join(data_root, "MOT20", data_split, sequence_name, "det", "det.txt")
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")

                logger.info(f"Loading public detections from {det_path}")
                public_detections = load_public_detections(det_path)
        
        # Build runtime tracker with public detections
        runtime_tracker = RuntimeTrackerPublic(
            model=model,
            sequence_hw=sequence_hw,
            public_detections=public_detections,
            detector=bytetrack_detector,
            det_source=det_source,
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            iou_thresh=iou_thresh,
            public_reid_encoder=public_reid_encoder,
            use_public_reid=use_public_reid,
            only_detr=inference_only_detr,
            dtype=dtype,
        )
        
        if is_fake:
            logger.info(f"Fake submitting sequence {sequence_name} with {len(sequence_loader)} frames.")
        else:
            logger.info(f"Submitting sequence {sequence_name} with {len(sequence_loader)} frames.")
        
        # Process frames
        sequence_results = []
        for t, (image, image_path) in enumerate(sequence_loader):
            if t == 10:
                begin_time = time.time()
            image.tensors = image.tensors.to(device)
            image.mask = image.mask.to(device)
            runtime_tracker.update(image=image, image_path=image_path)
            results = runtime_tracker.get_track_results()
            sequence_results.append(results)
            if private_det_cache is not None:
                private_det_cache[t + 1] = list(runtime_tracker.last_dets)
        
        if len(sequence_loader) > 10:
            fps = (len(sequence_loader) - 10) / (time.time() - begin_time)
        else:
            fps = 0.0
        
        # Write results (same format as original)
        tracker_name = "tracker_default"

        if private_det_cache is not None and cache_det_path is not None and not is_fake:
            os.makedirs(os.path.dirname(cache_det_path), exist_ok=True)
            with open(cache_det_path, "w") as f:
                for frame_id in sorted(private_det_cache.keys()):
                    for x, y, w, h, conf in private_det_cache[frame_id]:
                        f.write(
                            f"{frame_id},-1,"
                            f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                            f"{conf:.6f},-1,-1,-1\n"
                        )
            logger.info(f"Saved private detections to {cache_det_path}")
        
        # Count track lengths
        track_counts = {}
        for t in range(len(sequence_results)):
            for obj_id in sequence_results[t]["id"]:
                _oid = obj_id.item()
                track_counts[_oid] = track_counts.get(_oid, 0) + 1
        
        # Build results with filtering
        sequence_tracker_results = []
        id_remap = {}
        for t in range(len(sequence_results)):
            for obj_id, score, category, bbox in zip(
                    sequence_results[t]["id"],
                    sequence_results[t]["score"],
                    sequence_results[t]["category"],
                    sequence_results[t]["bbox"],
            ):
                _oid = obj_id.item()
                # Filter short tracks (修正: 使用 < 而非 <=，这样长度刚好等于阈值的轨迹不会被过滤)
                if min_track_len > 0 and track_counts[_oid] < min_track_len:
                    continue
                mapped_id = id_remap.setdefault(_oid, len(id_remap) + 1)
                sequence_tracker_results.append(
                    f"{t + 1},{mapped_id},"
                    f"{bbox[0].item():.2f},{bbox[1].item():.2f},{bbox[2].item():.2f},{bbox[3].item():.2f},"
                    f"1,-1,-1,-1\n"
                )
        
        if not is_fake:
            # Save results
            tracker_seq_dir = os.path.join(outputs_dir, "tracker", tracker_name, "data")
            os.makedirs(tracker_seq_dir, exist_ok=True)
            result_path = os.path.join(tracker_seq_dir, f"{sequence_name}.txt")
            with open(result_path, "w") as f:
                f.writelines(sequence_tracker_results)
            logger.success(f"Submit sequence {sequence_name} done, FPS: {fps:.2f}. Saved to {result_path}.")
    
    # Run evaluation if on train split
    if is_evaluate and not is_fake:
        logger.info("Running TrackEval evaluation...")
        run_trackeval(config, outputs_dir, inference_dataset.sequence_infos.keys(), logger)
    
    # Create submission zip if in submit mode
    if not is_evaluate and not is_fake:
        create_submission_zip(outputs_dir, sequence_names, logger)
    
    logger.success("Done!")


def run_trackeval(config: dict, outputs_dir: str, sequence_names, logger: Logger):
    """Run TrackEval evaluation"""
    dataset_name = config["INFERENCE_DATASET"]
    data_split = config["INFERENCE_SPLIT"]
    data_root = config["DATA_ROOT"]

    # Get repo root directory (where submit_public.py is located)
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Create seqmap
    seqmap_path = os.path.join(outputs_dir, "seqmap.txt")
    with open(seqmap_path, 'w') as f:
        f.write("name\n")
        for seq_name in sequence_names:
            f.write(f"{seq_name}\n")

    # TrackEval command - 使用本地 TrackEval 脚本而非 python -m trackeval
    trackeval_script = os.path.join(repo_root, "TrackEval", "scripts", "run_mot_challenge.py")
    trackeval_cmd = [
        "python", trackeval_script,
        "--BENCHMARK", dataset_name,
        "--SPLIT_TO_EVAL", data_split,
        "--TRACKERS_TO_EVAL", "tracker_default",
        "--TRACKER_SUB_FOLDER", "data",
        "--GT_FOLDER", os.path.join(data_root, dataset_name, data_split),
        "--TRACKERS_FOLDER", os.path.join(outputs_dir, "tracker"),
        "--SEQMAP_FILE", seqmap_path,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
    ]

    logger.info(f"Running: {' '.join(trackeval_cmd)}")

    try:
        # 使用 repo_root 作为 cwd，确保能找到 TrackEval 模块
        result = subprocess.run(trackeval_cmd, capture_output=True, text=True, cwd=repo_root)
        print(result.stdout)
        if result.returncode != 0:
            logger.warning(f"TrackEval error: {result.stderr}")
    except Exception as e:
        logger.warning(f"Failed to run TrackEval: {e}")


def create_submission_zip(outputs_dir: str, sequence_names, logger: Logger):
    """Create submission zip file"""
    import zipfile

    zip_path = os.path.join(outputs_dir, "submission.zip")
    tracker_data_dir = os.path.join(outputs_dir, "tracker", "tracker_default", "data")

    # 先检查所有期望的文件是否存在
    missing_files = []
    existing_files = []
    for seq_name in sequence_names:
        txt_path = os.path.join(tracker_data_dir, f"{seq_name}.txt")
        if os.path.exists(txt_path):
            existing_files.append((txt_path, seq_name))
        else:
            missing_files.append(txt_path)

    # 如果有缺失文件，报错而非静默跳过
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} result files:")
        for f in missing_files[:10]:  # 最多显示10个
            logger.warning(f"  - {f}")
        if len(missing_files) > 10:
            logger.warning(f"  ... and {len(missing_files) - 10} more")
        raise FileNotFoundError(f"Missing {len(missing_files)} result files. Cannot create submission zip.")

    # 打包文件
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for txt_path, seq_name in existing_files:
            zf.write(txt_path, f"{seq_name}.txt")

    logger.success(f"Created submission zip: {zip_path} ({len(existing_files)} files)")


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Get runtime option
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)
    
    # Load super config
    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])
    
    # Update config with runtime options
    cfg = update_config(config=cfg, option=opt)
    
    # Add IOU_THRESH to config if not present
    if "IOU_THRESH" not in cfg:
        cfg["IOU_THRESH"] = 0.5
    
    # Run evaluation
    submit_and_evaluate_public(config=cfg)
