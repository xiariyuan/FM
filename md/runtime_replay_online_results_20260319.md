# Runtime Replay Online Results 2026-03-19

## 1. 关键说明

- 当前验证跑的是 `external/sw_yolox` 检测线。
- `RuntimeTrackerByteTrack` 只是跟踪宿主类名，不代表回到了旧的 `bytetrack_x_mot17.pth.tar` 检测器主线。
- Runtime replay learned 插件已经真正接入线上推理，不再只是离线 proxy loss 或离线 rerank 实验。

## 2. 今天完成的代码改动

- 文件: `models/runtime_tracker_bytetrack.py`
- 改动:
  - 保持 runtime replay 插件接在线上真实关联路径。
  - 将 `_refine_assoc_scores_with_runtime_replay(...)` 从“每个 detection 单独前向一次”改为“整帧 batched 前向一次”。
  - 保持原有 score 逻辑不变，只优化推理接法和速度。

## 3. 单序列真实线上验证

### MOT17-02-FRCNN, external/sw_yolox

- Base 输出:
  - `outputs/runtime_replay_smoke_sw_yolox_base_MOT17-02-FRCNN_batch_20260319`
- Runtime replay 输出:
  - `outputs/runtime_replay_smoke_sw_yolox_rr_MOT17-02-FRCNN_batch_20260319`

### 指标

- Base:
  - HOTA `66.260`
  - AssA `58.910`
  - IDF1 `74.544`
  - IDSW `117`
  - Elapsed `91.668s`
- Runtime replay:
  - HOTA `67.186`
  - AssA `60.408`
  - IDF1 `76.106`
  - IDSW `109`
  - Elapsed `118.787s`

### Delta

- HOTA `+0.926`
- AssA `+1.498`
- IDF1 `+1.562`
- IDSW `-8`

## 4. Proxy 组合验证

### MOT17-02 + MOT17-13, external/sw_yolox

- Base 输出:
  - `outputs/runtime_replay_proxy0213_sw_yolox_base_20260319`
- 旧 replay 输出:
  - `outputs/runtime_replay_proxy0213_sw_yolox_rr_20260319`
- batched replay 输出:
  - `outputs/runtime_replay_proxy0213_sw_yolox_rr_batch_20260319`

### Base vs Replay 指标

- Base:
  - HOTA `67.536`
  - AssA `60.813`
  - IDF1 `76.744`
  - IDSW `181`
- Replay:
  - HOTA `68.096`
  - AssA `61.729`
  - IDF1 `77.699`
  - IDSW `173`

### Delta

- HOTA `+0.560`
- AssA `+0.916`
- IDF1 `+0.955`
- IDSW `-8`

## 5. Full7 真实线上验证

### external/sw_yolox full7 FRCNN

- 现成 Base:
  - `outputs/external_det_eval_queue/mot17_external_sw_yolox_base_full7_20260316_174526`
- 现成 Heuristic:
  - `outputs/external_det_eval_queue/mot17_external_sw_yolox_heuristic_full7_20260316_175126`
- 新 Runtime replay:
  - `outputs/external_det_eval_queue/mot17_external_sw_yolox_runtime_replay_full7_20260319_batch`

### Base vs Replay

- Base:
  - HOTA `78.577`
  - AssA `76.074`
  - IDF1 `86.230`
  - IDSW `456`
  - MOTA `92.655`
  - DetA `81.591`
- Replay:
  - HOTA `78.681`
  - AssA `76.240`
  - IDF1 `86.460`
  - IDSW `450`
  - MOTA `92.643`
  - DetA `81.627`

### Delta vs Base

- HOTA `+0.104`
- AssA `+0.166`
- IDF1 `+0.230`
- IDSW `-6`
- MOTA `-0.012`
- DetA `+0.036`

### Delta vs Heuristic

- HOTA `+0.067`
- AssA `-0.013`
- IDF1 `+0.163`
- IDSW `-5`
- MOTA `+0.003`
- DetA `+0.145`

### Replay 运行时长

- `417.94s` from `run_manifest.json`

## 6. Full7 按序列观察

- 明显正向:
  - `MOT17-02-FRCNN`
- 基本持平:
  - `MOT17-04-FRCNN`
  - `MOT17-05-FRCNN`
  - `MOT17-13-FRCNN`
- 轻微回撤:
  - `MOT17-09-FRCNN`
  - `MOT17-11-FRCNN`
- 小幅正向:
  - `MOT17-10-FRCNN`

## 7. 当前结论

- Runtime replay learned 插件已经被证实能在线上真实 ByteTrack host 中起作用，不再是“离线训练好看、线上无效”。
- 在 hard case 和 proxy 组合上收益明显。
- 在 full7 完整线上验证上，收益仍为正，但幅度较小，当前结论应表述为:
  - “稳定小增益”
  - 不是“大幅提升”
- 现阶段最准确的说法是:
  - learned runtime replay 插件有效
  - 但要把它打磨成强论文主结论，还需要继续增强 full7 稳定收益，尤其要处理 `09/11` 一类回撤序列

## 8. 下一步建议

- 先做 hard-sequence diagnosis:
  - 比较 `09/11` 与 `02/13` 的候选集统计、margin 分布、top-k recoverability、null/background 激活情况
- 再做更强证据链:
  - frozen host 上的 simple non-neural rerank 对照
  - ambiguity-stratified gain 统计
  - easy/hard split 的 no-op 安全性分析
