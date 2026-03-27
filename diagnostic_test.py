#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FM-Track Diagnostic Test Script

按照诊断文档的快速定位流程进行测试：
1. Step 1: 确认 checkpoint 完整加载
2. Step 2: 排除"过滤导致的假象" (NEWBORN_THRESH=-1, DET_THRESH=0.1)
3. Step 3: 只评估检测 (ONLY_DETR)
4. Step 4: 逐步回加复杂模块与增强

Usage:
    python diagnostic_test.py --step 1  # 测试 checkpoint 加载
    python diagnostic_test.py --step 2  # 关闭 newborn 过滤测试
    python diagnostic_test.py --step 3  # 仅 DETR 检测测试
    python diagnostic_test.py --step all  # 运行所有步骤
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from models.motip import build as build_motip
from models.misc import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="FM-Track Diagnostic Test")
    parser.add_argument("--step", type=str, default="1",
                       choices=["1", "2", "3", "4", "all"],
                       help="Which diagnostic step to run")
    parser.add_argument("--config", type=str,
                       default="./outputs/fa_mot_v2_resume/train/config.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint", type=str,
                       default="./outputs/fa_mot_v2_resume/checkpoint_30.pth",
                       help="Path to checkpoint file")
    parser.add_argument("--dataset", type=str, default="MOT17",
                       help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    return parser.parse_args()


def step1_check_checkpoint_loading(config: dict, checkpoint_path: str):
    """
    Step 1: 确认 checkpoint 完整加载
    使用 check_load_strictness 打印 missing/unexpected，
    重点看 detr.* 关键层是否缺失
    """
    print("\n" + "="*60)
    print("Step 1: 检查 Checkpoint 加载完整性")
    print("="*60)

    # Build model
    print("\n[1.1] Building model...")
    model, _ = build_motip(config=config)

    # Get model state dict info
    model_state = model.state_dict()
    print(f"[1.2] Model has {len(model_state)} parameters")

    # Load checkpoint and check
    print(f"\n[1.3] Loading checkpoint: {checkpoint_path}")
    print("-" * 40)

    # The load_checkpoint function now prints detailed statistics
    load_checkpoint(model, path=checkpoint_path, strict_check=True)

    print("\n[1.4] Checkpoint loading completed!")
    print("-" * 40)

    # Additional check: verify DETR layers
    print("\n[1.5] Verifying DETR layers are loaded...")
    detr_keys = [k for k in model_state.keys() if k.startswith("detr.")]
    print(f"Total DETR keys in model: {len(detr_keys)}")

    # Sample some DETR weights to verify they're not random
    sample_keys = [
        "detr.backbone.0.body.layer1.0.conv1.weight",
        "detr.transformer.encoder.layers.0.self_attn.in_proj_weight",
    ]
    for key in sample_keys:
        if key in model_state:
            weight = model_state[key]
            print(f"  {key}: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")

    return model


def step2_test_without_filtering(config: dict, model, checkpoint_path: str):
    """
    Step 2: 排除"过滤导致的假象"
    设置 NEWBORN_THRESH=-1, ID_THRESH=0, DET_THRESH=0.1
    """
    print("\n" + "="*60)
    print("Step 2: 测试关闭 Newborn 过滤的效果")
    print("="*60)

    # Modify config for diagnostic
    diag_config = config.copy()
    diag_config["NEWBORN_THRESH"] = -1.0  # 关闭 newborn 过滤
    diag_config["ID_THRESH"] = 0.0
    diag_config["DET_THRESH"] = 0.1  # 降低检测阈值
    diag_config["INFERENCE_MODE"] = "evaluate"
    diag_config["INFERENCE_MODEL"] = checkpoint_path
    diag_config["OUTPUTS_DIR"] = "./outputs/diagnostic_step2"

    print("\n诊断配置:")
    print(f"  NEWBORN_THRESH: {diag_config['NEWBORN_THRESH']} (原值: {config.get('NEWBORN_THRESH', 0.6)})")
    print(f"  ID_THRESH: {diag_config['ID_THRESH']} (原值: {config.get('ID_THRESH', 0.2)})")
    print(f"  DET_THRESH: {diag_config['DET_THRESH']} (原值: {config.get('DET_THRESH', 0.3)})")

    print("\n[提示] 若关闭过滤后框数量明显增加，说明问题主要来自 ID/unknown 二次过滤")
    print("[提示] 运行完整评估请执行:")
    print(f"  python submit_and_evaluate.py --config {checkpoint_path.replace('.pth', '_diag.yaml')}")

    # Save diagnostic config
    diag_config_path = "./outputs/diagnostic_step2_config.yaml"
    os.makedirs("./outputs", exist_ok=True)
    import yaml
    with open(diag_config_path, "w") as f:
        yaml.dump(diag_config, f)
    print(f"\n诊断配置已保存到: {diag_config_path}")

    return diag_config


def step3_test_detr_only(config: dict, checkpoint_path: str):
    """
    Step 3: 只评估检测（不做 ID/轨迹）
    设置 ONLY_DETR=True
    """
    print("\n" + "="*60)
    print("Step 3: 仅测试 DETR 检测器 (禁用 ID/轨迹模块)")
    print("="*60)

    # Modify config for DETR-only evaluation
    diag_config = config.copy()
    diag_config["INFERENCE_ONLY_DETR"] = True
    diag_config["ONLY_DETR"] = True
    diag_config["NEWBORN_THRESH"] = -1.0  # 关闭 newborn 过滤
    diag_config["DET_THRESH"] = 0.1
    diag_config["INFERENCE_MODE"] = "evaluate"
    diag_config["INFERENCE_MODEL"] = checkpoint_path
    diag_config["OUTPUTS_DIR"] = "./outputs/diagnostic_step3"

    print("\n诊断配置:")
    print(f"  ONLY_DETR: True (仅使用 DETR 检测，跳过 ID 预测)")
    print(f"  NEWBORN_THRESH: {diag_config['NEWBORN_THRESH']}")
    print(f"  DET_THRESH: {diag_config['DET_THRESH']}")

    print("\n[提示] 此测试确认 DETR 检测器本身是否能产生合理的框")
    print("[提示] 如果 DETR-only 检测效果正常，则问题在 ID/关联模块")

    # Save diagnostic config
    diag_config_path = "./outputs/diagnostic_step3_config.yaml"
    import yaml
    with open(diag_config_path, "w") as f:
        yaml.dump(diag_config, f)
    print(f"\n诊断配置已保存到: {diag_config_path}")

    return diag_config


def step4_ablation_augmentations(config: dict):
    """
    Step 4: 逐步回加复杂模块与增强
    创建一系列消融实验配置
    """
    print("\n" + "="*60)
    print("Step 4: 消融实验配置 - 逐步测试增强策略")
    print("="*60)

    ablations = {
        "E1_no_newborn_filter": {
            "NEWBORN_THRESH": -1.0,
            "ID_THRESH": 0.0,
            "DET_THRESH": 0.1,
            "desc": "关闭 newborn 过滤"
        },
        "E2_no_trajectory_switch": {
            "AUG_TRAJECTORY_SWITCH_PROB": 0.0,
            "desc": "关闭轨迹 switch 增强"
        },
        "E3_no_trajectory_occlusion": {
            "AUG_TRAJECTORY_OCCLUSION_PROB": 0.0,
            "desc": "关闭轨迹 occlusion 增强"
        },
        "E4_no_bbox_noise": {
            "BBOX_AUG_PROB": 0.0,
            "BBOX_POSITION_NOISE": 0.0,
            "BBOX_SIZE_NOISE": 0.0,
            "BBOX_DROP_PROB": 0.0,
            "desc": "关闭 bbox noise 增强"
        },
        "E5_light_augmentation": {
            "AUG_TRAJECTORY_SWITCH_PROB": 0.1,
            "AUG_TRAJECTORY_OCCLUSION_PROB": 0.1,
            "BBOX_AUG_PROB": 0.0,
            "desc": "轻量级增强配置"
        },
    }

    print("\n建议的消融实验:")
    print("-" * 50)

    import yaml
    os.makedirs("./outputs/ablation_configs", exist_ok=True)

    for exp_name, exp_changes in ablations.items():
        desc = exp_changes.pop("desc")
        print(f"\n{exp_name}: {desc}")

        # Create ablation config
        abl_config = config.copy()
        for k, v in exp_changes.items():
            old_val = abl_config.get(k, "N/A")
            abl_config[k] = v
            print(f"  {k}: {old_val} -> {v}")

        # Save config
        config_path = f"./outputs/ablation_configs/{exp_name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(abl_config, f)
        print(f"  配置已保存到: {config_path}")

        # Restore desc for next iteration
        exp_changes["desc"] = desc

    print("\n" + "-" * 50)
    print("运行消融实验示例:")
    print("  python submit_and_evaluate.py --config ./outputs/ablation_configs/E1_no_newborn_filter.yaml")


def run_quick_inference_test(config: dict, model, num_frames: int = 5):
    """
    快速推理测试：在少量帧上运行推理，收集诊断统计
    """
    print("\n" + "="*60)
    print("快速推理测试 (诊断模式)")
    print("="*60)

    from data.joint_dataset import dataset_classes
    from data.seq_dataset import SeqDataset
    from models.runtime_tracker import RuntimeTracker
    from torch.utils.data import DataLoader

    dataset_name = config.get("INFERENCE_DATASET", "MOT17")
    data_split = config.get("INFERENCE_SPLIT", "train")
    data_root = config["DATA_ROOT"]

    print(f"\n数据集: {dataset_name}/{data_split}")
    print(f"数据根目录: {data_root}")

    # Check if dataset exists
    if dataset_name not in dataset_classes:
        print(f"[ERROR] Dataset {dataset_name} not found!")
        return

    try:
        inference_dataset = dataset_classes[dataset_name](
            data_root=data_root,
            split=data_split,
            load_annotation=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    # Get first sequence
    sequence_names = list(inference_dataset.sequence_infos.keys())
    if not sequence_names:
        print("[ERROR] No sequences found!")
        return

    sequence_name = sequence_names[0]
    print(f"\n测试序列: {sequence_name}")

    # Create sequence dataset
    seq_info = inference_dataset.sequence_infos[sequence_name]
    image_paths = inference_dataset.image_paths[sequence_name]

    sequence_dataset = SeqDataset(
        seq_info=seq_info,
        image_paths=image_paths,
        max_shorter=800,
        max_longer=config.get("INFERENCE_MAX_LONGER", 1440),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        dtype=torch.float32,
    )

    sequence_loader = DataLoader(
        dataset=sequence_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    # Create RuntimeTracker with diagnostics enabled
    runtime_tracker = RuntimeTracker(
        model=model,
        sequence_hw=sequence_dataset.seq_hw(),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=config.get("MISS_TOLERANCE", 30),
        det_thresh=config.get("DET_THRESH", 0.3),
        newborn_thresh=config.get("NEWBORN_THRESH", 0.6),
        id_thresh=config.get("ID_THRESH", 0.2),
        area_thresh=config.get("AREA_THRESH", 0),
        only_detr=config.get("ONLY_DETR", False),
        dtype=torch.float32,
        num_classes=config.get("NUM_CLASSES", 1),
        target_class_idx=0,  # person class
    )

    # Enable diagnostics
    runtime_tracker.enable_diagnostics = True

    print(f"\n运行推理 ({min(num_frames, len(sequence_loader))} 帧)...")
    print("-" * 40)

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        for t, (image, image_path) in enumerate(sequence_loader):
            if t >= num_frames:
                break

            image.tensors = image.tensors.to(device)
            image.mask = image.mask.to(device)

            runtime_tracker.update(image=image)
            results = runtime_tracker.get_track_results()

            n_dets = len(results.get("id", []))
            print(f"  Frame {t+1}: {n_dets} detections")

    # Print diagnostic summary
    runtime_tracker.print_diagnostics()


def main():
    args = parse_args()

    print("="*60)
    print("FM-Track 诊断测试工具")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"测试步骤: {args.step}")

    # Load config
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        return

    config = yaml_to_dict(args.config)

    # Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    # Run diagnostic steps
    model = None

    if args.step in ["1", "all"]:
        model = step1_check_checkpoint_loading(config, args.checkpoint)

    if args.step in ["2", "all"]:
        if model is None:
            model, _ = build_motip(config=config)
            load_checkpoint(model, path=args.checkpoint, strict_check=False)
        step2_test_without_filtering(config, model, args.checkpoint)

    if args.step in ["3", "all"]:
        step3_test_detr_only(config, args.checkpoint)

    if args.step in ["4", "all"]:
        step4_ablation_augmentations(config)

    # Run quick inference test if model is loaded
    if model is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        run_quick_inference_test(config, model, num_frames=10)

    print("\n" + "="*60)
    print("诊断测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
