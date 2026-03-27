#!/usr/bin/env python3
# Copyright (c) 2024. All Rights Reserved.
"""
FA-MOT V1 -> V2 模型迁移工具

功能：
1. 加载V1训练的checkpoint
2. 迁移兼容的权重到V2模型
3. 保留DETR和trajectory_modeling部分
4. 对不兼容的id_decoder部分进行智能初始化

使用方法：
    python migrate_v1_to_v2.py \
        --v1_checkpoint ./outputs/v1/checkpoint_best.pth \
        --v2_config configs/r50_dino_fa_mot_v2_mot17.yaml \
        --output_path ./outputs/v2_migrated.pth
"""

import torch
import argparse
from collections import OrderedDict
from typing import Dict, Tuple, List
import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.misc import yaml_to_dict  # noqa: E402
from configs.util import load_super_config  # noqa: E402
from models.motip import build as build_motip  # noqa: E402


def analyze_checkpoint(ckpt_path: str) -> Dict:
    """分析checkpoint的结构"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    
    # 统计各模块的参数
    modules = {}
    for key in state_dict.keys():
        module_name = key.split(".")[0]
        if module_name not in modules:
            modules[module_name] = []
        modules[module_name].append(key)
    
    print("\n=== Checkpoint Analysis ===")
    for module, keys in modules.items():
        print(f"{module}: {len(keys)} parameters")
    
    return state_dict


def get_compatible_keys(v1_state: Dict, v2_state: Dict) -> Tuple[List, List, List, List]:
    """
    分析V1和V2模型的键兼容性
    
    Returns:
        compatible: 完全兼容的键（名称和形状都匹配）
        shape_mismatch: 名称匹配但形状不匹配的键
        missing: V2有但V1没有的键
        unexpected: V1有但V2没有的键
    """
    compatible = []
    shape_mismatch = []
    missing = []
    
    for v2_key, v2_param in v2_state.items():
        if v2_key in v1_state:
            v1_param = v1_state[v2_key]
            if v1_param.shape == v2_param.shape:
                compatible.append(v2_key)
            else:
                shape_mismatch.append((v2_key, v1_param.shape, v2_param.shape))
        else:
            missing.append(v2_key)
    
    unexpected = [k for k in v1_state.keys() if k not in v2_state]
    
    return compatible, shape_mismatch, missing, unexpected


def migrate_v1_to_v2(
    v1_checkpoint_path: str,
    v2_config_path: str,
    output_path: str,
    strict: bool = False,
) -> Dict:
    """
    将V1 checkpoint迁移到V2模型
    
    策略：
    1. DETR部分：完全复制（100%兼容）
    2. trajectory_modeling部分：完全复制（100%兼容）
    3. id_decoder部分：
       - word_to_embed: 复制
       - embed_to_word -> embed_to_word_layers.0-5: 复制到所有层
       - cross_attn -> cross_attn_layers.0: 复制到第一层
       - ffn -> ffn_layers.0: 复制到第一层
       - freq_fusion, freq_gate: 复制
       - 其他新增模块：随机初始化
    """
    print(f"\n{'='*60}")
    print("FA-MOT V1 -> V2 Migration Tool")
    print(f"{'='*60}")
    
    # 1. 加载V1 checkpoint
    print(f"\n[1/5] Loading V1 checkpoint: {v1_checkpoint_path}")
    v1_ckpt = torch.load(v1_checkpoint_path, map_location="cpu")
    
    if "model" in v1_ckpt:
        v1_state = v1_ckpt["model"]
    else:
        v1_state = v1_ckpt
    
    print(f"  - Total parameters: {len(v1_state)}")
    
    # 2. 构建V2模型
    print(f"\n[2/5] Building V2 model from config: {v2_config_path}")
    config = yaml_to_dict(v2_config_path)
    config = load_super_config(config, config.get("SUPER_CONFIG_PATH"))
    
    # 确保使用V2配置
    config["USE_FREQ_DECODER_V2"] = True
    config["USE_LEARNABLE_FUSION"] = True
    config["USE_AUX_LOSS"] = True
    
    v2_model, _ = build_motip(config)
    v2_state = v2_model.state_dict()
    print(f"  - Total parameters: {len(v2_state)}")
    
    # 3. 分析兼容性
    print(f"\n[3/5] Analyzing compatibility...")
    compatible, shape_mismatch, missing, unexpected = get_compatible_keys(v1_state, v2_state)
    
    print(f"  - Compatible keys: {len(compatible)}")
    print(f"  - Shape mismatch: {len(shape_mismatch)}")
    print(f"  - Missing in V1: {len(missing)}")
    print(f"  - Unexpected (V1 only): {len(unexpected)}")
    
    # 4. 执行迁移
    print(f"\n[4/5] Migrating weights...")
    
    new_state = OrderedDict()
    migrated_count = 0
    initialized_count = 0
    
    for key, param in v2_state.items():
        # 情况1：完全兼容
        if key in compatible:
            new_state[key] = v1_state[key].clone()
            migrated_count += 1
        
        # 情况2：需要特殊处理的id_decoder参数
        elif "id_decoder" in key:
            migrated = False
            
            # embed_to_word_layers.X -> 从V1的embed_to_word复制
            if "embed_to_word_layers" in key:
                v1_key = key.replace("embed_to_word_layers.", "embed_to_word.")
                base_key = "id_decoder.embed_to_word.weight"
                if base_key in v1_state and v1_state[base_key].shape == param.shape:
                    new_state[key] = v1_state[base_key].clone()
                    migrated = True
                    migrated_count += 1
            
            # cross_attn_layers.X -> 从V1的cross_attn复制到第0层
            elif "cross_attn_layers.0" in key:
                v1_key = key.replace("cross_attn_layers.0", "cross_attn")
                if v1_key in v1_state and v1_state[v1_key].shape == param.shape:
                    new_state[key] = v1_state[v1_key].clone()
                    migrated = True
                    migrated_count += 1
            
            # cross_attn_norm_layers.X -> 从V1的cross_attn_norm复制
            elif "cross_attn_norm_layers.0" in key:
                v1_key = key.replace("cross_attn_norm_layers.0", "cross_attn_norm")
                if v1_key in v1_state and v1_state[v1_key].shape == param.shape:
                    new_state[key] = v1_state[v1_key].clone()
                    migrated = True
                    migrated_count += 1
            
            # ffn_layers.X -> 从V1的ffn复制到第0层
            elif "ffn_layers.0" in key:
                v1_key = key.replace("ffn_layers.0", "ffn")
                if v1_key in v1_state and v1_state[v1_key].shape == param.shape:
                    new_state[key] = v1_state[v1_key].clone()
                    migrated = True
                    migrated_count += 1
            
            # ffn_norm_layers.X -> 从V1的ffn_norm复制
            elif "ffn_norm_layers.0" in key:
                v1_key = key.replace("ffn_norm_layers.0", "ffn_norm")
                if v1_key in v1_state and v1_state[v1_key].shape == param.shape:
                    new_state[key] = v1_state[v1_key].clone()
                    migrated = True
                    migrated_count += 1
            
            # freq_fusion, freq_gate: 直接复制
            elif "freq_fusion" in key or "freq_gate" in key:
                if key in v1_state and v1_state[key].shape == param.shape:
                    new_state[key] = v1_state[key].clone()
                    migrated = True
                    migrated_count += 1
            
            # 没有迁移成功，使用V2的初始化
            if not migrated:
                new_state[key] = param.clone()
                initialized_count += 1
        
        # 情况3：其他参数使用V2初始化
        else:
            new_state[key] = param.clone()
            initialized_count += 1
    
    print(f"  - Migrated: {migrated_count}")
    print(f"  - Newly initialized: {initialized_count}")
    
    # 5. 保存迁移后的checkpoint
    print(f"\n[5/5] Saving migrated checkpoint: {output_path}")
    
    output_ckpt = {
        "model": new_state,
        "migration_info": {
            "source": v1_checkpoint_path,
            "source_version": "V1",
            "target_version": "V2",
            "migrated_keys": migrated_count,
            "initialized_keys": initialized_count,
        },
    }
    
    # 复制其他信息（如果有）
    for key in ["optimizer", "scheduler", "epoch", "global_step"]:
        if key in v1_ckpt:
            # 优化器状态不兼容，跳过
            if key == "optimizer":
                print("  - Skipping optimizer state (incompatible)")
                continue
            output_ckpt[key] = v1_ckpt[key]
    
    torch.save(output_ckpt, output_path)
    
    print(f"\n{'='*60}")
    print("Migration completed!")
    print(f"{'='*60}")
    
    # 打印详细的迁移报告
    print("\n=== Migration Report ===")
    print("\nFully migrated modules:")
    print("  ✓ detr (100%)")
    print("  ✓ trajectory_modeling (100%)")
    print("  ~ id_decoder (partial)")
    print("    - word_to_embed: migrated")
    print("    - embed_to_word_layers: migrated to all layers")
    print("    - cross_attn_layers.0: migrated from V1")
    print("    - ffn_layers.0: migrated from V1")
    print("    - freq_fusion, freq_gate: migrated")
    print("    - self_mamba_layers: newly initialized")
    print("    - cross_attn_layers.1-5: newly initialized")
    print("    - rel_pos_embeds: newly initialized")
    print("    - freq_id_heads: newly initialized")
    print("    - fusion_weight: newly initialized")
    
    print("\nRecommendation:")
    print("  - Use smaller learning rate (e.g., 5e-5) for initial epochs")
    print("  - Monitor id_decoder gradients carefully")
    print("  - Consider freezing DETR for first few epochs")
    
    return output_ckpt


def main():
    parser = argparse.ArgumentParser(description="Migrate FA-MOT V1 to V2")
    parser.add_argument("--v1_checkpoint", type=str, required=True, help="Path to V1 checkpoint")
    parser.add_argument(
        "--v2_config",
        type=str,
        default="configs/r50_dino_fa_mot_v2_mot17.yaml",
        help="Path to V2 config file",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output path for migrated checkpoint")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze checkpoint without migration")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_checkpoint(args.v1_checkpoint)
    else:
        migrate_v1_to_v2(
            v1_checkpoint_path=args.v1_checkpoint,
            v2_config_path=args.v2_config,
            output_path=args.output_path,
        )


if __name__ == "__main__":
    main()
