# Pro Review Long-Term Hard Template

这是一份长期固定模板。

用途：

- 每次向 Pro / 外部大模型提审时使用
- 不再只问“方向是否正确”
- 强制对方同时输出“管理级决策 + 文件级代码重设计文档”

适用场景：

- 主线切换
- go / kill 节点
- oracle upper bound 之后的 redesign
- learned module 重新设计
- 线上负结果后的结构性改版

## 长期固定原则

- 不要只发最新问题，必须附历史、已停路线、当前主线、最新硬证据。
- 不要只问方向，必须要求对方直接设计代码级修改方案。
- 不要接受泛泛建议，必须要求对方落到文件级改动。
- 不要接受“可以试试 A/B/C”的松散回答，必须要求给优先级和唯一 first-priority experiment。
- 不要接受只讲模块名不讲落点的回答，必须要求函数锚点 / 行号范围 / 配置接入点 / runner 接入点。

## 固定发送模板

```markdown
请你把自己当成独立研究顾问 + 技术设计 reviewer。

这次我不要你只给方向判断，也不要你泛泛讲文献和可能性。
我需要你基于我提供的完整历史、当前证据和代码包，直接给：

1. 管理级决策
2. 代码级重设计文档

请注意：
你不需要被我当前已有模块形式绑住。
如果你认为当前实现虽然方向对，但模块设计太弱，请你直接提出更强的新模块 / 新 decision unit / 新训练目标 / 新在线注入方式。

## Part A. 管理级决策

你必须明确给出：

- `GO / NARROW GO / KILL`
- 为什么
- 是否建议“沿当前实现继续”还是“直接重设计更强模块”
- 下一步唯一 first-priority experiment 是什么
- 哪些事情现在不要做

## Part B. 代码级实现设计文档

请你直接输出一份完整 Markdown 文档正文，标题类似：

`IMPLEMENTATION_REDESIGN_PLAN.md`

这份文档不能停留在概念层，必须具体到代码改动层。

你输出的实现设计文档，至少必须包含以下内容：

### 1. 总体模块设计

- 新主模块叫什么
- 它替代 / 升级的是哪条旧线
- 主 decision unit 是什么
- 输入是什么
- 输出是什么
- one-to-one 约束怎么处理
- 在线注入点在哪
- 与 host 的接口是什么

### 2. 文件级改动清单

对于每一个需要修改或新增的文件，你都必须写清楚：

- 文件路径
- 为什么改这个文件
- 当前这个文件的职责是什么
- 当前需要修改的函数 / 类 / 配置段是什么
- 修改前大致在哪几行到哪几行
- 如果无法保证精确行号，也必须给明确锚点：
  - 函数名
  - 类名
  - 关键变量名
  - 可 grep 的关键字符串
- 当前实现的问题是什么
- 修改后应该变成什么
- 是局部改写还是新增模块后接入
- 这部分改动的风险点是什么

请按下面格式逐项写：

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

### 3. 新增文件

如果你建议新增文件，也必须写清楚：

- 新文件路径
- 这个文件负责什么
- 为什么不能只在旧文件里硬改
- 这个文件的核心类 / 函数列表
- 与现有哪些文件交互

### 4. 配置与脚本

你必须明确指出：

- 哪些 config 文件要新增参数
- 哪些 shell runner 要新增
- 哪些训练 / 推理 / oracle / eval 脚本要改
- 结构化记录怎么记：
  - `result.csv`
  - `summary.csv`
  - `experiment_registry.csv`
  - bundle

### 5. 最小可落地版本

请你明确给出：

- 第一版最小实现是什么
- 它应该先跑什么实验
- 这个最小版本不做什么
- 哪些复杂部分先延后

### 6. 不允许的回答方式

以下形式都不算合格回答：

- 只说“建议引入图模块”
- 只说“可以做 joint assignment”
- 只说“建议改成 transformer / graph network”
- 只给方向，不给文件级落点
- 只给伪代码，不给代码入口

我需要的是：

- 能直接指导工程修改的文档
- 最好我拿到文档后，就能按文件逐项实现

## 你回答时请严格遵守这个输出顺序

1. `GO / NARROW GO / KILL`
2. 为什么
3. `沿当前实现继续` 还是 `直接重设计更强模块`
4. 如果重设计，你给出的唯一主方案是什么
5. 下一步唯一 first-priority experiment 是什么
6. 哪些事情现在不要做
7. `IMPLEMENTATION_REDESIGN_PLAN.md` 正文
```

## 使用建议

- 如果本次重点是“方向裁决 + 模块重设计”，优先使用这份模板。
- 如果只是普通结果审阅，可以继续用 `PRO_REVIEW_CONTEXT_TEMPLATE.md`，但也建议把这里的硬要求块一并附上。
- 如果已经拿到 oracle upper bound，下一次优先要求对方直接输出文件级改版方案，不要只给研究方向点评。
