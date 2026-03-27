#!/bin/bash
# Phase 0: 评估可信性验证
# 用途：确保评估链路没有问题（防止train 97%的幻觉）

set -e

echo "======================================"
echo "Phase 0: Sanity Check"
echo "======================================"

# A. 空输出测试
echo "[Test A] 空输出评估（期望指标接近0）"
# TODO: 生成空的tracker输出文件
# python scripts/generate_empty_results.py
# python evaluate.py --empty-test

# B. 极严阈值测试
echo "[Test B] 极严阈值评估（期望召回大幅下降）"
# TODO: 临时修改阈值到0.99
# DET_THRESH=0.99 NEWBORN_THRESH=0.99 python evaluate.py

# C. 打乱ID测试
echo "[Test C] 打乱ID评估（期望IDF1大幅下降）"
# TODO: 打乱所有track_id
# python scripts/shuffle_track_ids.py
# python evaluate.py --shuffled

echo "Phase 0完成。如果三项都符合预期，进入Phase 1"
