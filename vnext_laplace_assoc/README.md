# VNext Laplace Association

这个目录是从零重新思考后的新主线原型。

## Baseline

这里的 baseline 不是你当前仓库里已经堆过很多模块的版本，而是一个更干净、可复现、适合论文讲故事的基线：

- `Detector`：强检测器，独立训练并尽量固定
- `Tracker`：在线 tracking-by-detection
- `Motion`：Kalman 预测与几何一致性
- `Spatial Appearance`：标准外观分支，用于空间域身份相似度
- `Matching`：Hungarian 或 greedy 的 track-centric 关联

这个 baseline 的核心是不包含旧的频域分支，不包含 Mamba，不包含额外的复杂记忆模块。它只回答一个问题：

> 如果没有 Laplace / frequency 可靠性分支，仅靠 motion + spatial appearance，能做到什么程度？

## New Idea

新主线不是“再加一个频域特征”，而是：

> 用 Laplace-inspired 的时间衰减轨迹原型，去估计当前 appearance 关联到底值不值得信。

模块拆分如下：

- `spatial branch`
  - 当前检测与轨迹外观相似度
- `laplace branch`
  - 对轨迹历史特征做多尺度指数衰减聚合，形成 Laplace temporal signatures
- `reliability head`
  - 根据 spatial / laplace / motion 三类分数，动态决定本次关联更该信哪一类 cue

## Why This Baseline Is Better

这样定义后，论文主线会更清楚：

- `Base`：motion + spatial
- `Base + Laplace`
- `Base + ReID`
- `Full`

而不是在旧系统上不断叠已有模块，导致审稿人很难判断到底是谁起作用。

## Current Status

当前目录已经实现了：

- Laplace temporal signature 构建
- Laplace reliability association head
- 与现有 `RuntimeTrackerByteTrack` 的最小侵入式接线

默认情况下这些开关是关闭的，不会影响现有实验。

## First Recommended Experiment

优先跑：

- baseline: `v14_ctrl_base_reid_da`
- baseline + laplace: `v15_laplace_reid_da`

先在 `MOT17-02/13` proxy 上看：

- `HOTA`
- `AssA`
- `IDF1`
- `IDSW`
- `Frag`

如果 `AssA↑ / IDSW↓ / Frag↓`，这条主线就成立。
