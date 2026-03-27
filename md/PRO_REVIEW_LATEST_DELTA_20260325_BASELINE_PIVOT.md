# Pro Review Latest Delta (2026-03-25, Baseline Pivot)

这份补充 delta 只记录一个新收束点:

- `v2` 作为 local-conflict operator，已经在当前内部宿主线上拿到真实正号；
- 但当前内部宿主线 `base_reid_da` 不是严格论文意义上的 ByteTrack baseline；
- 因此下一步不再是泛泛地问“模块方向是否继续”，而是固定为:
  - 顶会论文里，应该优先复现哪些 baseline；
  - 哪些 baseline 适合作为插件式 operator 的主宿主；
  - 哪些 baseline 只能做 transfer / appendix / negative evidence。

## 1. 当前已经可以视为稳定事实的结果

### 1.1 stable `v2` 在内部宿主线有效

运行根目录:

- `/gemini/code/FMtrack-main/FM-Track/outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`

相对 enlarged-data `v1`:

- `proxy0213`
  - `v1`: `53.046 / 44.414 / 59.627 / 73.461 / 796`
  - `v2 stable`: `53.118 / 44.577 / 58.730 / 73.437 / 811`
- `full md2/mm2`
  - `v1`: `62.691 / 59.151 / 71.814 / 75.996 / 1525`
  - `v2 stable`: `63.257 / 60.191 / 72.128 / 76.055 / 1481`

当前可接受的最简判断:

- `v2` 不是幻觉；
- 它在当前内部宿主线上已经证明自己能带来真实 operator 增益；
- 但这还不等于“论文级最终有效”。

## 2. 为什么现在必须 pivot 到 baseline selection

### 2.1 `base_reid_da` 不是干净的论文 baseline

证据路径:

- `outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/config_effective.yaml`

其中可见:

- `ASSOC_USE_REID: true`
- `ASSOC_FEAT_SOURCE: reid`
- `ASSOC_MODE: hybrid`

这说明它不是“官方 ByteTrack 原味 baseline”，而是 FM-Track 内部改造过的一条 host line。

### 2.2 因此当前主问题已经换了

当前不该再问:

- `v2 要不要继续`
- `还要不要再发明更强模块`

当前真正该问的是:

- 论文主表到底该基于哪些 baseline；
- 哪个 baseline 是 primary carrier；
- 哪个 baseline 适合做 transfer evidence；
- 哪些 baseline 虽然仓库里有，但不适合当前 operator 语义。

## 3. 当前仓库里已知存在的 baseline family

### 3.1 ByteTrack 官方代码

路径:

- `third_party/ByteTrack/README.md`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/exps/example/mot/yolox_x_ablation.py`
- `third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py`

这是当前最像“严格论文复现”的来源。

### 3.2 FM-Track 内部 ByteTrack-style / external-det lines

路径:

- `scripts/run_bytetrack_external_ctrl.sh`
- `scripts/run_bytetrack_external_batch.sh`
- `configs/profiles/mot17_external_sw_yolox_base_full7.json`
- `configs/profiles/mot17_external_sgt_base_full7.json`

这些更像系统恢复线或内部对照线，不应自动当作论文主 baseline。

### 3.3 其他 repo 内已存在的 baseline family

路径举例:

- BoT-SORT:
  - `scripts/run_botsort_*`
- StrongSORT:
  - `scripts/run_strongsort_*`
- MOTIP:
  - `configs/r50_deformable_detr_motip_*`

但这些 family 是否适合当前 operator 语义，需要重新明确，不应默认都能作为同等级主 baseline。

## 4. 当前 operator 契约已经固定

请不要再把 baseline 选择和模块语义混在一起。

当前 operator 语义固定为:

- cluster-level local conflict operator
- conservative partial commit
- defer to host
- primary-only
- pre-Hungarian
- frozen host / plug-in style injection

这意味着:

- 一些 tracking-by-detection baseline 可能天然适合；
- 一些 end-to-end / query-based tracker 可能不适合当前注入语义；
- baseline selection 必须考虑“论文公平性”与“工程注入边界”。

## 5. 现在最值钱的 Pro 问题

下一次向 Pro，不应再问“模块要不要继续”。

应该只问:

1. 顶会论文里，应该优先基于哪些 baseline 做严格复现；
2. 当前 operator 最适合挂在哪个 primary baseline 上做主结果；
3. 哪个 baseline 适合做 secondary transfer evidence；
4. 哪些 baseline 虽然仓库里有，但不适合作为这篇论文的主线 carrier。

## 6. 我们当前自己的管理级判断

当前本地判断已经收紧为:

- `v2` 在内部宿主线上有效；
- 但还不能直接宣称“ByteTrack + v2 在论文基线上有效”；
- 现在最先要解决的是 baseline cleanliness，而不是继续在内部宿主线上打补丁。
