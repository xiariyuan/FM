# FM-Track 实验记录（持续更新）

> 记录日期：2026-02-10  
> 项目路径：`/gemini/code/FMtrack-main/FM-Track`  
> 目标：MOT 顶会论文（方法有效性 + 可复现性）

---

## 1. 实验背景与统一口径

- 主训练框架：`train_bytetrack.py`
- 主要配置族：`configs/bytetrack_fa_mot_mot17_v5_seqid*.yaml`
- 检测器策略：以 ByteTrack/YOLOX 为主，后续加入外部检测结果（`sw_yolox` / `sgt`）
- 关键提醒：
  - `MOT17 train 全7序列` 与 `MOT17 04/05 子集` 结果**不可直接横比**
  - `MOT20-01/02` 是“跨域/外部验证”，不等于官方 test leaderboard
  - `quick sweep` 子集结果仅用于方向判断，不作为最终主结果

---

## 2. 关键实验阶段（按时间/逻辑）

## 2.1 v5_seqid 主线训练与复评

- 配置：`configs/bytetrack_fa_mot_mot17_v5_seqid.yaml`
- 关键结果（统一复评，MOT17 train 全7）：
  - `HOTA 66.703 / IDF1 73.915 / MOTA 78.924 / AssA 63.229 / DetA 71.097 / IDSW 1077`
  - 结果文件：`outputs/submit_mot17_train_e24_best/tracker/MOT17-train/pedestrian_summary.txt`
- 子集验证（04/05，consistent identity）最佳：
  - `epoch 19: HOTA 76.315 / IDF1 85.182 / MOTA 77.977 / AssA 79.086 / IDSW 134`
  - 结果文件：`outputs/bytetrack_fa_mot_mot17_v5_seqid/reval_consistent_identity/reval_results.csv`

## 2.2 MOT20-01/02 跨域验证

- v5_seqid 最佳：
  - `epoch 22: HOTA 51.033 / IDF1 61.395 / MOTA 69.275 / AssA 45.612 / DetA 57.334 / IDSW 757`
  - 文件：`outputs/bytetrack_fa_mot_mot17_v5_seqid/reval_mot20_01_02/reval_results.csv`
- finetune_v2 最佳：
  - `epoch 25: HOTA 51.020 / IDF1 61.316 / MOTA 69.278 / AssA 45.579 / DetA 57.347 / IDSW 760`
  - 文件：`outputs/bytetrack_fa_mot_mot17_v5_seqid_finetune_v2/reval_mot20_01_02/reval_results.csv`

## 2.3 关联阈值扫参与推理参数实验

- MOT17 full7 sweep 最佳（用于可比口径）：
  - `id_thresh=0.003, iou_gate=0.3, det_max=50`
  - `HOTA 66.777 / IDF1 73.828 / MOTA 78.942 / AssA 63.270 / IDSW 1074`
  - 文件：`outputs/sweep_assoc_mot17_e24_full7/sweep_assoc_results.csv`
- MOT17 quick sweep（小子集，仅方向判断）：
  - `HOTA 77.863 / IDF1 86.902 / AssA 81.133`
  - 文件：`outputs/sweep_assoc_mot17e22_quick/sweep_assoc_results.csv`

## 2.4 外部检测器接入实验

- SW-YOLOX + MOT20 sweep 最佳：
  - `run=id0.01_iou0.3_det80`
  - `HOTA 69.809 / IDF1 76.617 / MOTA 91.678 / AssA 60.606 / DetA 80.499 / IDSW 403`
  - 文件：`outputs/bytetrack_fa_mot_mot17_v6_shortft/sweep_mot20_sw_small/sweep_assoc_results.csv`
- SGT（MOT20）表现异常低：
  - `HOTA 3.2827 / IDF1 1.6095 / MOTA -35.128`
  - 文件：`outputs/reval_sgt_mot20_e22/tracker/MOT20-train/pedestrian_summary.txt`

---

## 3. 训练问题诊断与修复记录（高价值）

## 3.1 发现问题：Triplet Loss 近“死值”

- 现象：
  - 先长期接近 `0.3003`（margin hinge 饱和）
  - 改 `TRIPLET_MARGIN=0` 后接近 `0.6933`（softplus(0)附近）
- 初步结论：Triplet 使用的输入特征不合适（`trajectory_features` 被时序平滑，难形成有效 hard contrast）

## 3.2 第一阶段修复（仅超参）

- 新配置：`configs/bytetrack_fa_mot_mot17_v5_seqid_assocft22_v2.yaml`
- 关键改动：
  - `LR=5e-5`
  - `TRIPLET_MARGIN=0.0`
  - `TRIPLET_LOSS_WEIGHT=0.5`
  - 维持 `freq_ortho=0.5`, `freq_consistency=0.2`, `det_mix_prob=0.65`

## 3.3 第二阶段修复（代码级，核心）

- 修复目标：Triplet 改用 `id_decoder` 的 unknown-query embedding（与 ID 分类空间一致）
- 代码修改：
  - `models/motip/freq_aware_id_decoder_v2.py`
    - 在训练返回 `extra_info` 中新增：
      - `triplet_embeddings`
      - `triplet_labels`
      - `triplet_masks`
  - `train_bytetrack.py`
    - Triplet 优先使用 decoder 输出（若无再回退旧逻辑）
    - 增加日志指标 `triplet_src_decoder`
- 生效证据（当前训练）：
  - `triplet_src_decoder = 1.0000`（证明已走 decoder 特征）
  - triplet 出现下降：`0.9852 -> 0.9250 -> 0.8941 -> 0.8512 -> 0.8025 -> 0.7780 -> 0.7611 ...`
  - 日志文件：`outputs/bytetrack_fa_mot_mot17_v5_seqid_assocft22_v2/train.log`

---

## 4. 当前损失贡献快照（修复后）

以最新一条日志估算（epoch 23 过程）：

- `id_loss` 贡献约 `64.06%`
- `triplet_loss` 贡献约 `27.26%`
- `freq_ortho` 贡献约 `6.23%`
- `freq_consistency` 贡献约 `1.74%`
- `newborn_penalty` 贡献约 `0.71%`

结论：Triplet 已成为有效的第二主损失；频域辅助与 newborn 仍偏弱，但当前阶段建议先保持单变量验证，不同时改太多。

---

## 5. 已确认的“硬问题 / 风险点”

- `train_bytetrack.py` 中存在硬编码项：`+ 0.0 * freq_energy_loss`  
  - 当前配置 `FREQ_ENERGY_BALANCE_WEIGHT=0.0`，所以行为上等价关闭；
  - 建议后续做代码清理：改为读取配置权重（默认仍可设 0.0），避免“看似可配、实际硬关”。

---

## 6. 论文可写性状态（截至 2026-02-10）

- 可写为：**关键修复消融（Triplet from ineffective to effective）**
- 暂不宜写成最终主结论：需补齐同预算对照（至少到 epoch25/30）
- 推荐对照组（同训练预算）：
  1. 旧 triplet（trajectory + margin=0.3）
  2. 仅 margin 修复（trajectory + margin=0）
  3. 当前修复（decoder-embedding triplet）

主汇报指标：`HOTA / AssA / IDF1 / IDSW / MOTA`

---

## 7. 下一步执行建议（当前版本）

- 继续当前修复版训练到 `epoch 25`
- 若 `AssA/HOTA/IDF1` 无提升，再做“单变量”下一步（例如频域损失重配，或位置关联损失）
- 不建议立即从 `epoch 0` 全部重训；优先保留 `epoch22` 作为稳定恢复点做验证

---

## 8. 相关关键路径索引

- 当前训练日志：`outputs/bytetrack_fa_mot_mot17_v5_seqid_assocft22_v2/train.log`
- 当前配置：`configs/bytetrack_fa_mot_mot17_v5_seqid_assocft22_v2.yaml`
- 主线配置：`configs/bytetrack_fa_mot_mot17_v5_seqid.yaml`
- 核心训练代码：`train_bytetrack.py`
- 核心 decoder 代码：`models/motip/freq_aware_id_decoder_v2.py`
- 统一复评结果（MOT17 full7）：`outputs/submit_mot17_train_e24_best/tracker/MOT17-train/pedestrian_summary.txt`

