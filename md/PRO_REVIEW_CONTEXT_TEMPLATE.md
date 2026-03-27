# Pro Review Context Template

每次向外部审阅模型提问时，优先使用这份模板，避免上下文丢失导致重复讨论旧路线。

## 使用原则

- 不要只发“最新问题”，必须把历史、已死路线、当前主线、最新证据一起发。
- 不要让对方重新开放式发散，先明确“哪些路线已经停掉”。
- 不要只问“方向是什么”，默认要求对方同时产出“代码级实现设计文档”。
- 问题尽量收窄到一个决策点，例如：
  - online integration 怎么接
  - go / kill 怎么判
  - 离线强线上弱时先查什么
  - 下一阶段 continuity 应不应该上
- 每次发送前，把方括号占位符替换成当前事实。

## 长期固定硬要求

以后给 Pro 的提示词，默认必须要求对方输出两层内容：

1. 管理级决策
2. 代码级实现设计文档

第二层不能停留在概念层，必须落到：

- 要改哪些文件
- 每个文件当前在做什么
- 哪个函数 / 类 / 配置段要动
- 修改前的大致行号范围，或者明确代码锚点
- 修改后逻辑是什么
- 新增什么文件 / 类 / 函数 / 配置
- 哪些旧路径保留作 ablation，哪些正式停用
- 第一批实验如何接入新实现

建议把下面这段固定附在每次提示词末尾：

```markdown
## 额外硬要求：请直接给代码级重设计文档

这次请不要只给方向判断，我还需要你直接对当前代码做“可执行级别的重设计”。

你的回答必须包含两部分：

### Part A. 管理级决策

你必须明确给出：
- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

### Part B. 代码级实现设计文档

请你基于我提供的代码包，直接写一份可执行的 `MD` 设计文档。
这份文档不能停留在概念层，必须具体到代码改动层。

你的实现设计文档，至少必须包含以下内容：

1. 总体模块设计
- 新主模块叫什么
- 它替代 / 升级的是哪条旧线
- 在线注入点在哪
- 输入是什么
- 输出是什么
- 与 host 的接口是什么

2. 文件级改动清单
对于每一个需要修改或新增的文件，你都必须写清楚：
- 文件路径
- 为什么改这个文件
- 当前这个文件里相关逻辑是什么
- 需要修改的函数 / 类 / 配置段是什么
- 修改前大致在哪几行到哪几行
- 如果无法保证精确行号，也必须给明确代码锚点：
  - 函数名
  - 类名
  - 关键变量名
  - 可 grep 的关键字符串
- 修改后应该变成什么
- 是“局部改写”还是“新增模块后接入”
- 这部分改动的风险点是什么

3. 请按下面这种格式逐项写

### Change 1
- File:
- Current code responsibility:
- Current anchor / lines:
- Problem in current implementation:
- New implementation:
- Exact modification plan:
- Whether old path stays for ablation:
- Dependencies:

### Change 2
- File:
- Current code responsibility:
- Current anchor / lines:
- Problem in current implementation:
- New implementation:
- Exact modification plan:
- Whether old path stays for ablation:
- Dependencies:

依次列完。

4. 新增文件
如果你建议新增文件，也必须写清楚：
- 新文件路径
- 这个文件负责什么
- 为什么不能只在旧文件里硬改
- 这个文件的核心类 / 函数列表
- 与现有哪些文件交互

5. 配置与脚本
你必须明确指出：
- 哪些 config 文件要新增参数
- 哪些 shell runner 要新增
- 哪些训练 / 推理 / oracle / eval 脚本要改
- 结构化记录怎么记：
  - `result.csv`
  - `summary.csv`
  - `experiment_registry.csv`
  - bundle

6. 最小可落地版本
请你明确给出：
- 第一版最小实现是什么
- 它应该先跑什么实验
- 这个最小版本不做什么
- 哪些复杂部分先延后

7. 不允许只给高层建议
以下形式都不算合格回答：
- 只说“建议引入图模块”
- 只说“可以做 joint assignment”
- 只说“建议改成 transformer / graph network”
- 只给方向，不给文件级落点
- 只给伪代码，不给代码入口

我需要的是：
- 能直接指导工程修改的文档
- 最好是我拿到文档后，就能按文件逐项实现

8. 输出格式
请你最后直接输出一份完整的 Markdown 文档正文。
文档标题请写成类似：

`IMPLEMENTATION_REDESIGN_PLAN.md`

文档中必须有：
- 总体结论
- 文件级改动表
- 逐文件详细修改说明
- 第一批实验执行顺序
- 风险与回退策略
```

## 模板正文

```markdown
请你把自己当成独立研究顾问，只审下面这条已经收敛下来的 MOT 主线。不要重新发散到旧方向，也不要泛泛给建议。请基于我给出的完整历史、当前证据和代码包，回答最后的具体问题。

你的回答必须包含两部分：

Part A. 管理级决策
- `GO / NARROW GO / KILL`
- 为什么
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要再做

Part B. 代码级实现设计文档
- 请直接写成一份完整 Markdown 文档正文
- 必须落到文件级改动
- 必须说明修改文件、函数锚点 / 行号范围、修改前逻辑、修改后逻辑、配置与脚本接入方式
- 不允许只给概念性方向

## 1. 项目背景

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前宿主是：[填写当前固定 host，例如 ByteTrack 风格 host / base_reid_da]
当前研究目标不是直接改 detector，而是在固定宿主上研究一个可插拔的 learned association module。

当前我认定的主线是：
- 宿主：[`填写宿主名`]
- 主角方法：[`填写主方法名`]
- 当前阶段目标：[`例如 keep / rerank / null 的局部冲突控制`]
- 下一阶段目标：[`例如 short-gap continuity / tracklet continuity`]

注意：
宿主不是主角方法本身，宿主只是承载平台。

## 2. 已经做过的路线，以及当前结论

### 路线 A：[填写旧路线名]

核心思路：
[简要描述]

当前结论：
- [结论 1]
- [结论 2]
- [为什么停掉]

### 路线 B：[填写旧路线名]

核心思路：
[简要描述]

当前结论：
- [结论 1]
- [结论 2]
- [为什么降级或停止]

### 路线 C：[填写旧路线名]

核心思路：
[简要描述]

当前结论：
- [结论 1]
- [结论 2]
- [为什么停止]

因此当前明确结论是：
- [哪些旧路线停掉]
- [哪些路线降级为二线]
- [当前唯一主线是什么]

## 3. 当前 host / baseline 对照证据

这是当前关键对照。

### baseline A: [名称]
- HOTA = [数值]
- AssA = [数值]
- IDF1 = [数值]
- MOTA = [数值]
- IDSW = [数值]

### baseline B: [名称]
- HOTA = [数值]
- AssA = [数值]
- IDF1 = [数值]
- MOTA = [数值]
- IDSW = [数值]

### baseline C: [名称]
- HOTA = [数值]
- AssA = [数值]
- IDF1 = [数值]
- MOTA = [数值]
- IDSW = [数值]

结论：
- [谁优于谁]
- [为什么当前固定这个宿主]
- [为什么另一个分支被判负信号]

## 4. 当前主线的核心动机

我现在认为真正的问题不是：
- [例如不是特征还不够 fancy]

而是：
- [例如 runtime association 里的局部竞争关系没有被直接建模]

因此我把主线切成：
- 固定 [`宿主名`] 作为 host
- 从真实 runtime dump 中构建 [`ambiguity/conflict groups`]
- 训练 [`方法名`] 来输出 [`动作空间`]
- 如果第一阶段在线成立，再扩到 [`下一阶段`]

## 5. 当前主线已经跑通的事实

### 5.1 数据 / runtime pipeline

- [host dump 是否成功]
- [replay label 是否成功]
- [competition case / continuity case 是否成功]

如果有统计，请直接给：
- groups = [数值]
- positive_groups = [数值]
- ambiguous_groups = [数值]
- recoverable_groups = [数值]
- recoverable_rate_among_positive = [数值]
- bridge_rate_among_positive = [数值]

这说明：
- [当前 hard groups 是否够多]
- [当前更该先做 rerank 还是 continuity]

### 5.2 当前训练状态

如果之前有 bug，要说明清楚：
- [例如之前 NaN 的根因]
- [现在怎么修掉的]

### 5.3 最新正式结果

当前最新结果：
- run = [`填写 run 名称或输出目录`]
- best_epoch = [数值]
- [关键指标 1] = [数值]
- [关键指标 2] = [数值]
- [关键指标 3] = [数值]
- status = [ok / failed / running]

这说明：
- [当前至少学到了什么]
- [但还缺哪一步证据]

## 6. 当前最需要你审的具体问题

现在请你不要再回头讨论：
- [不要再重复的旧路线 1]
- [不要再重复的旧路线 2]
- [不要再重复的旧路线 3]

我接下来的动作是：
- [例如把 stage1 best checkpoint 接回 host 做 proxy0213 在线评测]

请你只回答下面这些问题：
1. [问题 1]
2. [问题 2]
3. [问题 3]
4. [问题 4]

## 7. 我希望你的输出方式

请直接给我：
- go / kill 风险判断
- 推荐设计
- 必做 ablation 清单
- 最需要警惕的 failure modes
- 一份可执行的代码级实现设计文档
  - 要改哪些文件
  - 每个文件改什么
  - 当前锚点 / 行号范围
  - 修改后逻辑
  - 新增哪些文件 / 配置 / runner / 记录链

不要泛泛谈方向，不要重复旧历史，不要只说“先做实验看看”。
```

## 推荐提问类型

### 类型 A：在线接回审阅

适用场景：
- 离线训练已完成
- 下一步是把 learned module 接回 host

最后 4 问建议：
1. 最稳的注入方式是什么
2. 模块输出应该怎样作用到 host 匹配流程
3. 第一轮 online ablation 顺序是什么
4. 离线强线上弱时最先查哪 3 个 failure point

### 类型 B：结果出来后的 go / kill 审阅

适用场景：
- 已经拿到 proxy 或 full7 在线结果

最后 4 问建议：
1. 这是否足够支持继续作为主线
2. 增益是否真实，最可能来自哪里
3. 应该继续扩大验证还是先 redesign
4. 如果只保留一个主线和一个备线，应该怎么选

### 类型 C：失败后的定点诊断

适用场景：
- 离线强但线上无效
- 或者线上回撤

最后 4 问建议：
1. 最可能的闭环错位点是什么
2. 应先改 action design、gating 还是 selection criterion
3. 哪个 ablation 最有信息增益
4. 应该继续修还是直接 kill

### 类型 D：模块级重设计审阅

适用场景：
- 已经确认旧 decision unit 该停
- 已经需要 Pro 直接给出新模块设计和代码落点

最后 4 问建议：
1. 新 decision unit 应该是什么
2. 是否该沿当前实现继续，还是直接重设计更强模块
3. 第一版最小可落地实现是什么
4. 请直接给出 `IMPLEMENTATION_REDESIGN_PLAN.md`，写清文件级改动
