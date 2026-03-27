# Round1 训练启动指南

## ✅ 验证完成

所有预训练验证已通过：
- ✅ VAL_SEQUENCES正确分离：训练集15个序列，验证集6个序列
- ✅ **硬断言已添加（GPT强烈推荐）**：
  - 训练启动时自动检测VAL泄漏
  - 评估时自动检测非VAL序列泄漏
- ✅ Config已按综合建议调整：
  - `AUG_TRAJECTORY_OCCLUSION_PROB`: 0.1 (保留少量遮挡学习)
  - `AUG_TRAJECTORY_SWITCH_PROB`: 0.0 (关闭ID切换干扰)
  - `BBOX_*`: 全部设为0.0 (关闭bbox噪声)
- ✅ 日志系统已添加：会打印实际训练/验证序列
- ✅ Diagnostic stats已就绪：会记录检测数量、过滤率等

## 训练启动命令（在80G显卡机器上执行）

```bash
cd /gemini/code/FMtrack-main/FM-Track
mkdir -p logs

# 后台训练
nohup python train.py --config-path configs/round1_minimal_change.yaml \
  > logs/round1_minimal_change.log 2>&1 &

# 保存PID
echo $! > logs/round1_pid.txt

# 查看实时日志
tail -f logs/round1_minimal_change.log
```

## 🔍 关键日志检查（GPT要求的3行必看）

训练启动后，**立即检查日志**，必须看到以下3行：

### 1. 训练集序列确认（开始时）
```
[TRAIN] MOT17/train: 15 sequences loaded: ['MOT17-04-DPM', 'MOT17-04-FRCNN', ...]
✅ Confirmed: VAL_SEQUENCES are excluded from TRAIN dataset.
```
**如果没看到 ✅ 或者看到 ❌ → 立即停止训练！**

### 2. 验证集序列确认（每个epoch）
```
[EVAL] Evaluating on 6 sequences: ['MOT17-02-DPM', 'MOT17-02-FRCNN', 'MOT17-02-SDP', 'MOT17-10-DPM', 'MOT17-10-FRCNN', 'MOT17-10-SDP']
✅ Confirmed: Eval sequences are restricted to VAL_SEQUENCES=['MOT17-02', 'MOT17-10']
```
**如果没看到 ✅ 或者看到 ❌ → 立即停止训练！**

### 3. 验证指标（每个epoch）
```
HOTA: 0.xxxx, MOTA: 0.xxxx, IDF1: 0.xxxx
```

## 快速检查命令

```bash
# 查看训练启动日志（包含断言结果）
head -100 logs/round1_minimal_change.log | grep -E "TRAIN|Confirmed|VAL"

# 查看最新的验证日志
tail -200 logs/round1_minimal_change.log | grep -E "EVAL|Confirmed|MOTA|HOTA|IDF1"

# 查看进程状态
ps aux | grep $(cat logs/round1_pid.txt)

# GPU使用情况
nvidia-smi
```

### 2. 检查进程
```bash
# 查看PID
cat logs/round1_pid.txt

# 检查进程是否运行
ps aux | grep $(cat logs/round1_pid.txt)

# GPU使用情况
nvidia-smi
```

### 3. 停止训练（如需要）
```bash
kill $(cat logs/round1_pid.txt)
```

## 预期输出

### 训练开始时会看到：
```
Train/Val Split enabled. VAL_SEQUENCES: ['MOT17-02', 'MOT17-10']
Training will exclude these sequences.
[TRAIN] MOT17/train: 15 sequences loaded: ['MOT17-04-DPM', 'MOT17-04-FRCNN', ...]
```

### 验证时会看到：
```
Evaluating on VAL_SEQUENCES: ['MOT17-02', 'MOT17-10']
[EVAL] Evaluating on 6 sequences: ['MOT17-02-DPM', 'MOT17-02-FRCNN', ...]
```

### 验证指标示例：
```
HOTA: 0.xxxx
MOTA: 0.xxxx
IDF1: 0.xxxx
```

## 关键指标观察

根据GPT建议，重点观察：
1. **验证集MOTA**：是否比之前95.88%更稳定（过拟合程度降低）
2. **DetA vs AssA**：定位检测问题还是关联问题
3. **Per-frame检测数量**：是否正常（目标~22 det/frame）
4. **Newborn过滤率**：是否合理（目标~0.5%）

## 训练配置摘要

- **Epochs**: 40
- **LR**: 1e-4 (backbone 0.1x)
- **Weight Decay**: 0.001
- **Dropout**: 0.1
- **Batch Size**: 1
- **预训练**: DINO checkpoint0089.pth
- **从头开始训练**（不继承过拟合模型权重）

## 预期训练时间

- 单卡A100: 约12-16小时 (40 epochs)
- 验证评估: 每epoch约15分钟

## 如果需要调整

如果Round1效果良好，可以逐步添加增强：
1. Round2: 恢复`AUG_TRAJECTORY_OCCLUSION_PROB`到0.2-0.3
2. Round3: 小心添加bbox噪声（0.01/0.015）
3. 找到sweet spot

## 文件位置

- **Config**: `configs/round1_minimal_change.yaml`
- **输出目录**: `./outputs/round1_minimal_change/`
- **Checkpoints**: `./outputs/round1_minimal_change/checkpoints/`
- **评估结果**: `./outputs/round1_minimal_change/train/eval_during_train/`
