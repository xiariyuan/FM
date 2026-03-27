# Pro Review Request: For A Top-Tier MOT Paper, Which Baselines Should We Strictly Reproduce And Augment?

## Zero-Context Opening

你现在面对的是一个没有共享历史上下文的项目审查问题，请只依赖我附带的文件和下面这段自洽摘要来判断，不要脑补我们之前已经做过哪些决定。

当前项目状态可以压缩成 5 句话:

1. 我们的方法不是重写整个 tracker，而是一个 `frozen strong host` 上的 `plugin-style local association operator`。
2. 它的在线语义已经固定为:
   - cluster-level
   - conservative partial commit
   - defer to host
   - primary-only
   - pre-Hungarian
3. 当前最强版本 `v2 = HostConditionedLocalConflictSetPredictor` 已在一条内部宿主线 `base_reid_da` 上拿到真实正号，因此“模块方向是否成立”这个问题已经基本回答为成立。
4. 但 `base_reid_da` 不是严格论文 baseline，因为它混入了 `reid/hybrid` 语义，所以它不能直接作为最终 paper 主表的 carrier。
5. 因此这次要你解决的唯一问题，不是继续设计模块，而是为顶会论文固定:
   - 应该优先复现哪些 baseline family
   - 哪个 baseline family 最适合作为 primary carrier
   - 哪些适合做 transfer evidence
   - 哪些只该保留为内部线或排除

如果你只看这一页，也请至少按上面 5 条理解问题，不要把问题改写成“默认先做 ByteTrack”，也不要把问题改写成“重新发明更强模块”。

请把下面这些文件当作本次问题的权威上下文:

- `md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260324.md`
- `md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md`
- `md/PRO_REVIEW_INTERACTION_LOG.md`

如果你打开代码包，请优先核对这些文件:

- `third_party/ByteTrack/README.md`
- `third_party/ByteTrack/tools/track.py`
- `scripts/run_bytetrack_external_ctrl.sh`
- `scripts/run_bytetrack_external_batch.sh`
- `scripts/make_mot17_full_submission_bytetrack.py`
- `configs/profiles/mot17_external_sw_yolox_base_full7.json`
- `configs/profiles/mot17_external_sgt_base_full7.json`
- `configs/bytetrack_fa_mot_mot17.yaml`
- `scripts/run_botsort_*.sh`
- `scripts/run_strongsort_*.sh`
- `configs/r50_deformable_detr_motip_*.yaml`

## 这次不要重开的结论

这些结论已经固定，请不要回头重开:

- row-local rerank 已死
- full cluster replacement 已死
- 当前 operator 语义固定为:
  - cluster-level
  - conservative partial commit
  - defer to host
  - primary-only
  - pre-Hungarian
- 当前 stronger module 主线固定为:
  - `HostConditionedLocalConflictSetPredictor`

## 当前最新事实

### 1. `v2` 在内部宿主线上已经有效

运行根目录:

- `outputs/local_conflict_set_predictor_large_base_stable_20260325_023500`

相对 enlarged-data `v1`:

- `proxy0213`
  - `v1`: `53.046 / 44.414 / 59.627 / 73.461 / 796`
  - `v2 stable`: `53.118 / 44.577 / 58.730 / 73.437 / 811`
- `full md2/mm2`
  - `v1`: `62.691 / 59.151 / 71.814 / 75.996 / 1525`
  - `v2 stable`: `63.257 / 60.191 / 72.128 / 76.055 / 1481`

所以:

- 我们不再问“这条模块线要不要继续”
- 当前 answer 已经是继续

### 2. 但当前内部宿主线不是严格论文 baseline

关键证据:

- `outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/config_effective.yaml`

其中明确有:

- `ASSOC_USE_REID: true`
- `ASSOC_FEAT_SOURCE: reid`
- `ASSOC_MODE: hybrid`

因此它不是干净的官方 ByteTrack baseline。

### 3. 当前真正未决的是 baseline strategy

我们的目标已经收紧为:

- 严格论文 baseline 复现
- 在同一 baseline 上插入我们的 operator
- 做 paired gain
- 形成顶会 paper 的主表与迁移证据

## 我现在真正要你回答的唯一问题

请只回答这个问题:

> 对于一篇顶会 MOT 论文，如果我们的方法是一个“frozen strong host 上的 plugin-style local association operator”，那么我们应该优先基于哪些 baseline family 做严格复现与模块插入提升？

## 这次只允许你在 baseline strategy 上做决策

不要把回答重新转回:

- 继续 redesign operator
- 继续问 stronger host migration
- 继续问 tiny proxy 实验
- 继续问 loss / gate / feature 小修

这次只回答 baseline strategy。

## 你必须在下面这些 baseline family 里明确取舍

请至少讨论并排序这些 family:

1. `official ByteTrack`
   - 即 `third_party/ByteTrack` 这条最接近原论文实现的线
2. `FM-Track internal ByteTrack-style hosts`
   - 如 `base_reid_da` / `external-det` / `system-rescue` 这些内部演化线
3. `BoT-SORT`
4. `StrongSORT`
5. `MOTIP`
6. 你认为明确不适合当前 operator 注入语义的其他 family

## 这次回答的硬要求

### 1. 必须给一个明确的 baseline hierarchy

请明确给出:

- `primary paper baseline`
- `secondary transfer baseline`
- `internal ablation-only baseline`
- `currently exclude`

不要只说“都可以试试”。

### 2. 必须考虑当前 operator 的注入边界

当前方法不是 whole-tracker rewrite，而是:

- frozen host
- plugin-style operator
- pre-Hungarian / primary association injection

所以请明确说明:

- 哪些 baseline 适合这种注入方式
- 哪些 baseline 不适合

### 3. 必须回答“官方 ByteTrack 是否应排第一”

这是现在最关键的管理问题。

请直接回答:

- 是不是必须把 `official ByteTrack` 放在第一主线，先做严格复现，再做 `ByteTrack + operator`
- 还是可以把别的 host 放在主线，把 ByteTrack 只当补充

### 4. 必须给文件级落点

请尽量结合当前仓库已有入口，说明:

- 哪条路径最像 strict reproduction
- 哪条路径只适合内部控制，不适合论文主表
- 哪些 runner / config 应该直接停用为论文主线

### 5. 必须给唯一 first-priority experiment

而且它必须是 baseline-related，而不是 module-related。

例如只能是这种类型:

- strict official ByteTrack reproduction on MOT17 val
- strict ByteTrack host-only vs ByteTrack + operator paired eval

不能再回到:

- redesign module
- host migration
- gate tuning

## 我当前自己的判断，你可以直接反驳

我当前判断是:

- `v2` 已经证明自己在内部宿主线上有效
- 现在最值钱的问题已经不是“模块真假”
- 而是“论文应该挂在哪个 baseline family 上”
- 我不想预设一定是 `official ByteTrack`
- 我真正想要的是: 让你在 `official ByteTrack / BoT-SORT / StrongSORT / MOTIP / 其他可行 family` 中，选出最适合当前 plugin-style operator 的 primary carrier
- `base_reid_da` 更适合保留为 internal evidence / ablation line

如果你同意，请收紧成一个可执行的 baseline hierarchy。
如果你不同意，请明确说为什么 primary carrier 应该是别的 family，而不是 `official ByteTrack`。

## 希望的回答格式

请直接按下面结构回答:

1. `管理级决策`
2. `为什么当前问题已经收敛为 baseline selection`
3. `baseline hierarchy`
4. `为什么选这些，不选另外那些`
5. `文件级与 runner 级落点`
6. `唯一 first-priority experiment`
7. `当前不要做什么`

我要的是一份可以直接据此锁论文主线的回答，不要高层空话。
