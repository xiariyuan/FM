# Round1 准备工作总结

## 已完成的改进

### 1. Config调整（综合GPT建议）

**文件**: `configs/round1_minimal_change.yaml`

**改动**：
```yaml
# 轨迹增强（综合建议）
AUG_TRAJECTORY_OCCLUSION_PROB: 0.1    # 原0.5 -> 0.1（保留少量遮挡学习能力）
AUG_TRAJECTORY_SWITCH_PROB: 0.0       # 原0.5 -> 0.0（完全关闭ID切换干扰）

# BBox噪声增强
BBOX_POSITION_NOISE: 0.0              # 原0.02 -> 0.0
BBOX_SIZE_NOISE: 0.0                  # 原0.03 -> 0.0
BBOX_DROP_PROB: 0.0                   # 原0.02 -> 0.0
BBOX_AUG_PROB: 0.0                    # 原0.5 -> 0.0

# 验证集
VAL_SEQUENCES: [MOT17-02, MOT17-10]   # 6个序列（3个检测器版本）
```

**理由**：
- `OCCLUSION_PROB=0.1`而非0.0：模型需要学习一定遮挡处理能力
- `SWITCH_PROB=0.0`：这是ID关联最大的干扰源，优先关闭
- BBox噪声全关：先锁定增强问题的根源

### 2. 训练/验证序列日志（GPT要求）

**修改文件**：
- `data/joint_dataset.py`: 添加`get_loaded_sequences()`方法
- `train.py`: 添加训练集序列硬打印
- `submit_and_evaluate.py`: 添加验证集序列硬打印

**效果**：
训练时会打印：
```
[TRAIN] MOT17/train: 15 sequences loaded: ['MOT17-04-DPM', ...]
```

验证时会打印：
```
[EVAL] Evaluating on 6 sequences: ['MOT17-02-DPM', 'MOT17-02-FRCNN', ...]
```

**理由**：确保VAL_SEQUENCES真正从训练集排除，避免语义歧义

### 3. 诊断统计（GPT要求）

**已有实现**: `models/runtime_tracker.py`的`diagnostic_stats`

**统计指标**：
- 总帧数
- 原始检测数（DETR输出）
- Score过滤后检测数
- Newborn过滤后检测数
- Unknown/newborn比例
- 各阶段保留率

**理由**：快速定位问题（检测召回 vs ID过滤 vs gating阈值）

### 4. 验证脚本

**文件**: `test_val_split.py`

**功能**：
- 验证训练集不含VAL_SEQUENCES
- 验证验证集只含VAL_SEQUENCES
- 打印序列统计

**验证结果**：
```
✅ PASS: No VAL sequences in training set (15 seqs)
✅ PASS: All validation sequences match VAL_SEQUENCES (6 seqs)
```

## 验证完成的事项

- ✅ Checkpoint加载正常（1086/1086 keys）
- ✅ DETR检测模块正常（DetA=77.48%）
- ✅ 完整模型正常（MOTA=95.88%，与之前结果一致）
- ✅ 无缓存问题（DET_THRESH=0.99测试通过）
- ✅ MOT20评估正常（MOTA=24.45%，确认泛化问题）
- ✅ VAL_SEQUENCES正确分离（训练15序列，验证6序列）
- ✅ 日志系统就绪（会打印实际序列和诊断统计）

## 未解决的问题（需要训练来验证）

问题：**严重过拟合**
- MOT17 Train: MOTA=95.88%
- MOT17 Test: MOTA=52.79%
- MOT20 Train: MOTA=24.45%

假设：**增强过强导致监督噪声**
- 原config的TRAJECTORY_*_PROB=0.5和BBOX噪声可能过强
- 导致ID学习不稳定，在clean test set上崩溃

Round1目标：验证假设
- 如果验证集MOTA稳定且接近训练集 → 假设成立
- 如果仍然差距大 → 需要考虑其他因素（LR/dropout/架构等）

## 下一步（需要在其他机器上执行）

1. **启动训练**：
   ```bash
   cd /gemini/code/FMtrack-main/FM-Track
   nohup python train.py --config-path configs/round1_minimal_change.yaml \
     > logs/round1_minimal_change.log 2>&1 &
   echo $! > logs/round1_pid.txt
   ```

2. **监控日志**：
   ```bash
   tail -f logs/round1_minimal_change.log
   ```

3. **观察指标**（每epoch评估）：
   - MOTA/HOTA/IDF1是否比之前稳定
   - 验证集与训练集差距是否缩小
   - DetA vs AssA比例（定位检测or关联问题）

4. **根据结果调整**：
   - 如果改善 → Round2逐步加回增强
   - 如果不变 → 考虑其他因素（LR/dropout/WD/架构）

## 训练配置

保持原始设置（最小改动原则）：
- Epochs: 40
- LR: 1e-4 (backbone 0.1x)
- Weight Decay: 0.001
- Dropout: 0.1（DETR和Freq模块）
- Batch Size: 1
- 从DINO预训练开始（不继承过拟合权重）

## 参考文档

- 启动指南: `START_TRAINING.md`
- Config文件: `configs/round1_minimal_change.yaml`
- 验证脚本: `test_val_split.py`
