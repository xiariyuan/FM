#!/bin/bash
# FA-MOT 消融实验运行脚本
#
# 使用方法:
#   bash run_ablations.sh [experiment_id]
#
# 实验ID:
#   full     - 完整模型
#   no_lfd   - 去掉可学习频率分解
#   no_ftt   - 去掉频率-时序Transformer
#   no_fga   - 去掉频率引导关联
#   no_ortho - 去掉正交性损失
#   no_consist - 去掉一致性损失
#   bands_2  - 2个频带
#   bands_3  - 3个频带
#   bands_5  - 5个频带
#   v1_decoder - 使用V1简化解码器
#   fixed_fusion - 固定融合权重

set -e

# 基础配置
BASE_CONFIG="configs/r50_dino_fa_mot_v2_mot17.yaml"
OUTPUT_BASE="./outputs/ablations"
DATA_ROOT="/path/to/datasets"  # 修改为你的数据路径

# 解析参数
EXPERIMENT=${1:-"full"}

echo "========================================"
echo "Running ablation experiment: $EXPERIMENT"
echo "========================================"

case $EXPERIMENT in
    "full")
        # 完整模型
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/full \
            --exp-name ablation_full \
            --data-root $DATA_ROOT
        ;;
    
    "no_lfd")
        # 去掉可学习频率分解，使用固定Laplacian
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/no_lfd \
            --exp-name ablation_no_lfd \
            --data-root $DATA_ROOT \
            --use-fixed-laplacian True
        ;;
    
    "no_ftt")
        # 去掉频率-时序Transformer
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/no_ftt \
            --exp-name ablation_no_ftt \
            --data-root $DATA_ROOT \
            --num-freq-temporal-layers 0
        ;;
    
    "no_fga")
        # 去掉频率引导关联
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/no_fga \
            --exp-name ablation_no_fga \
            --data-root $DATA_ROOT \
            --use-freq-guided-assoc False
        ;;
    
    "no_ortho")
        # 去掉正交性损失
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/no_ortho \
            --exp-name ablation_no_ortho \
            --data-root $DATA_ROOT \
            --freq-ortho-loss-weight 0.0
        ;;
    
    "no_consist")
        # 去掉一致性损失
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/no_consist \
            --exp-name ablation_no_consist \
            --data-root $DATA_ROOT \
            --freq-consistency-loss-weight 0.0
        ;;
    
    "bands_2")
        # 2个频带
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/bands_2 \
            --exp-name ablation_bands_2 \
            --data-root $DATA_ROOT \
            --num-freq-bands 2
        ;;
    
    "bands_3")
        # 3个频带
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/bands_3 \
            --exp-name ablation_bands_3 \
            --data-root $DATA_ROOT \
            --num-freq-bands 3
        ;;
    
    "bands_5")
        # 5个频带
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/bands_5 \
            --exp-name ablation_bands_5 \
            --data-root $DATA_ROOT \
            --num-freq-bands 5
        ;;
    
    "v1_decoder")
        # V1简化解码器
        python train.py \
            --config-path configs/r50_dino_fa_mot_mot17.yaml \
            --outputs-dir ${OUTPUT_BASE}/v1_decoder \
            --exp-name ablation_v1_decoder \
            --data-root $DATA_ROOT
        ;;
    
    "fixed_fusion")
        # 固定融合权重
        python train.py \
            --config-path $BASE_CONFIG \
            --outputs-dir ${OUTPUT_BASE}/fixed_fusion \
            --exp-name ablation_fixed_fusion \
            --data-root $DATA_ROOT \
            --use-learnable-fusion False
        ;;
    
    "all")
        # 运行所有消融实验
        for exp in full no_lfd no_ftt no_fga no_ortho no_consist bands_2 bands_3 bands_5 v1_decoder fixed_fusion; do
            echo "Running $exp..."
            bash $0 $exp
        done
        ;;
    
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: full, no_lfd, no_ftt, no_fga, no_ortho, no_consist, bands_2, bands_3, bands_5, v1_decoder, fixed_fusion, all"
        exit 1
        ;;
esac

echo "========================================"
echo "Experiment $EXPERIMENT completed!"
echo "========================================"
