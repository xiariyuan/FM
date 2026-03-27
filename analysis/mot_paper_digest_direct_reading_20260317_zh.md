# 多目标跟踪相关论文中文直读版导读

日期：2026-03-17

这份文档的目标不是“列论文目录”，而是让你**不需要先去啃原文**，也能快速知道这些论文到底在做什么、为什么重要、和我们现在的方向有什么关系。

我把每篇论文都压成下面这几个问题来讲：

- 这篇论文到底想解决什么问题？
- 它的方法核心是什么？
- 它和传统方法相比，真正新的点在哪？
- 它的优点是什么？
- 它的局限是什么？
- 它对我们当前项目的启发是什么？

建议你把这份文档当成“预读笔记”。
如果某篇看起来和我们特别相关，再回头去读原文。

---

## 先给结论：这些论文可以分成 4 类

### 第一类：处理 confusion / ambiguity 的论文

代表：

- DeconfuseTrack
- TrackTrack
- Modelling Ambiguous Assignments for Multi-Person Tracking in Crowds

这类论文的共同点是：
它们都认为 MOT 的难点不是“平均意义上的匹配”，而是**遮挡、相似外观、候选竞争、歧义场景**。

### 第二类：learnable association 论文

代表：

- MOTIP
- LA-MOTR
- Learning a Neural Solver
- GMTracker
- FAMNet

这类论文在做的事情是：
不满足于手工设计 association rules，而是试图**把 association 本身做成一个可学习问题**。

### 第三类：proposal / candidate 级建模论文

代表：

- LPC-MOT
- TrackFlow

这类论文的重点不再是“单个 pair 打分”，而是去建模**更高层次的候选结构或匹配分布**。

### 第四类：强宿主 / 外观 / 记忆相关论文

代表：

- Deep OC-SORT
- GeneralTrack
- MeMOT
- QDTrack

这类论文的重要提醒是：
很多 MOT 提升并不来自再造一个关联头，而是来自**更强的宿主、外观特征和长时记忆**。

---

# 第一部分：最值得你优先看懂的论文

这部分是和我们当前方向最接近、也最值得优先看懂的。

---

## 1. DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking

- 会议：CVPR 2024
- 链接：
  https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf

### 一句话讲清楚

这篇论文的核心观点是：
**MOT 的很多错误，不是一般意义上的匹配不准，而是 confusion 造成的。**
所以你不能把所有关联错误都看成一种错误。

### 它到底在解决什么问题

在拥挤场景、遮挡场景、多人相似外观场景中，tracker 经常会发生 ID switch。
传统 tracker 通常把这个问题简单归结为“匹配分数不够准”，然后继续调 motion、appearance、IoU 权重。

这篇论文认为问题没这么简单：

- 有些错误是因为目标之间太像
- 有些错误是因为周围 distractor 太多
- 有些错误是因为 NMS 或 detection 本身造成了干扰

也就是说，真正的问题是 confusion，而不是一个统一的 score 不够强。

### 方法核心

它不是粗暴地换一个更深的关联头，而是围绕 confusion 做更有针对性的设计。
从论文公开描述来看，它关注的是：

- 混淆源头识别
- 减少干扰目标影响
- 对易混淆目标的特殊处理

它本质上是在告诉你：
**hard cases 要单独对待。**

### 它和传统方法比，新在哪

新意不在“更复杂的分数网络”，而在于：
它把 confusion 当成 MOT 的核心矛盾之一来正面处理。

### 优点

- 论文叙事很自然，问题定义清晰
- reviewer 很容易接受“confusion 是 MOT 核心难点”这件事
- 比“我们换了一个更好的融合头”有更强的问题驱动

### 局限

- 它不是 runtime replay 训练对象
- 它也不是我们现在这种 frozen host 上的安全型 candidate-group rerank
- 如果你只学它的“处理 confusion”口号，而不补训练对象，还是容易回到 old learned 的老路

### 对我们的启发

这篇论文对我们最重要的启发只有一句话：

> 不要再把所有关联错误当成统一的 pair-wise calibration 问题。

这正是我们从 old MTCR / Laplace learned 路线切开的原因。

### 对我们论文的危险点

如果我们论文只写：
“我们处理 ambiguity / confusion”
那 reviewer 直接拿这篇来压我们。

所以我们必须更具体地说：

- 我们用的是 **runtime replay candidate groups**
- 我们做的是 **ambiguity-triggered safe reranking**
- 我们强调的是 **frozen host plugin**

---

## 2. Focusing on Tracks for Online Multi-Object Tracking (TrackTrack)

- 会议：CVPR 2025
- 链接：
  https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf

### 一句话讲清楚

这篇论文在说：
**传统 online MOT 太习惯“以检测为中心做一次全局匹配”，但真正稳定的 online tracking 应该更多地“以轨迹为中心”来思考。**

### 它到底在解决什么问题

经典 tracking-by-detection 里面，很多方法都默认：

- 当前帧有 detections
- 过去有 tracks
- 构造一个 cost matrix
- Hungarian 一匹配

这套逻辑很强，但也有天然问题：

- 遮挡时容易乱
- 多个 plausible candidate 时，全局匹配未必最稳
- 新旧轨迹生命周期管理容易互相干扰

### 方法核心

TrackTrack 的重要思想是：

- 不要只从“当前 detection 该匹配谁”的角度想问题
- 要从“已有 track 应该如何延续”这个角度想问题

这其实是在改 online association 的决策视角。

### 它和传统方法比，新在哪

不是简单调 cost，也不是换 ReID，而是重想 association process。

### 优点

- 对 online tracking 的理解更深
- 不只是调一个头，而是动到了“决策方式”
- 容易在 hard cases 上解释为什么会更稳

### 局限

- 这条路线更像“重做关联流程”
- 和我们现在“冻结强 host，加一个安全插件”的路线不完全一样
- 它不是 runtime replay 学出来的

### 对我们的启发

这篇论文提醒我们：

> association 的关键不一定是“分数更准”，而可能是“决策对象和决策顺序更合理”。

我们现在做 runtime replay，本质上也在修正这件事：
不再学单个 pair，而是学一个 detection-centered candidate group。

### 对我们论文的危险点

如果 reviewer 觉得我们的方法只是“小修 cost”，而 TrackTrack 是在真正处理 online association 的决策逻辑，那我们会显得弱。

因此我们要强调：
我们的贡献不是普通 cost tweak，而是**runtime object 对齐 + safe reranking**。

---

## 3. Towards Generalizable Multi-Object Tracking (GeneralTrack)

- 会议：CVPR 2024
- 链接：
  https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.pdf

### 一句话讲清楚

这篇论文的重点不是“把某个 benchmark 冲高”，而是：
**为什么现在很多 MOT 方法在一个设置里好看，一换场景就不稳。**

### 它到底在解决什么问题

很多 tracker 表面上是在做 motion + appearance + matching，
但实际上非常依赖：

- 特定数据分布
- 特定场景密度
- 特定遮挡模式
- 特定 motion / appearance 的手工平衡

结果就是：
同一套方法在一个 split 上好，在另一个 split 上掉。

### 方法核心

GeneralTrack 想做的是更 generalizable 的关系建模，
不再那么依赖人工设定 motion 和 appearance 的平衡方式。

### 它和传统方法比，新在哪

它把“泛化差”当成 MOT 的核心问题来讲，而不是只盯最终 leaderboard。

### 优点

- 对我们现在的处境很有解释力
- 非常适合用来理解为什么 random split 正、cross-seq 不稳
- 论文问题意识强

### 局限

- 它不是 plug-in style 方法
- 和我们当前 runtime replay 路线不是一模一样
- 它更像一个 broader tracking framework 方向

### 对我们的启发

这篇对我们最关键的意义是：

> 只要跨序列不稳，我们就不能过早宣布 learned module 有效。

这正是为什么我现在一直在补：

- leave-one-sequence-out
- full7 replay
- stronger host 上的 replay baseline

### 对我们论文的危险点

如果我们的 learned module 最后只能在 random split 正，LOSO 不稳，
这篇会让 reviewer 很自然地质疑我们的泛化性。

---

## 4. Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification

- 会议：ICIP 2023
- 链接：
  https://arxiv.org/abs/2302.11813

### 一句话讲清楚

这篇论文是在说：
**不要机械地把 appearance 作为一个固定权重项加进来，而应该根据情况自适应地使用外观信息。**

### 它到底在解决什么问题

纯 motion tracker 在遮挡和交叉场景里容易崩。
但直接把 appearance 强塞进来又会遇到：

- 低质量外观特征误导匹配
- 相似行人之间 appearance cue 也不可靠

### 方法核心

它的核心是：
基于 OC-SORT 这种强 motion host，设计 adaptive ReID integration。

换句话说，它不是“另起炉灶做新 tracker”，而是：

- host 保持强
- association 时更聪明地使用 appearance

### 它和传统方法比，新在哪

重点不是加 ReID，而是加**自适应的 ReID 使用策略**。

### 优点

- 非常符合强宿主 + 插件增强的论文风格
- 工程上可落地
- 容易做干净的 on/off 比较

### 局限

- 本质上还是在 adaptive cue integration
- 不是 runtime replay candidate-group 学习
- 不是 ambiguity-triggered reranking

### 对我们的启发

它给我们的最大启发是：

> 论文贡献不一定非要是“重做整个 tracker”，一个强 host 上的精确插件也能讲成论文。

这对我们现在的路线非常重要。

### 对我们论文的危险点

如果我们最后的方法看起来只是在“adaptive association”层面小修，
而没有把 runtime replay 这个训练对象讲清楚，
那 reviewer 会觉得我们只是另一版 Deep OC-SORT 风格方法。

---

## 5. Learning a Proposal Classifier for Multiple Object Tracking (LPC-MOT)

- 会议：CVPR 2021
- 链接：
  https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Learning_a_Proposal_Classifier_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf

### 一句话讲清楚

这篇论文的核心是：
**不要只给单个匹配对打分，而要给更高层次的候选 proposal 打分。**

### 它到底在解决什么问题

传统 tracker 过于依赖局部 pair-wise 亲和度。
但真正的轨迹形成并不是由一个 pair 决定的，而是一个更大结构。

### 方法核心

它把 MOT 重新表述成：

- proposal generation
- proposal scoring
- trajectory inference

也就是说，先产生有意义的候选，再学习哪个 proposal 更靠谱。

### 它和传统方法比，新在哪

它跳出了“pair affinity 越准越好”的思路。

### 优点

- 结构层级比 pair-wise 更高
- 直觉上更符合 tracking 任务本质
- 论文叙事比 calibrator 强得多

### 局限

- proposal 级别通常更复杂
- 系统实现成本高
- 和 frozen host plug-in 的简单性相比，工程侵入更大

### 对我们的启发

LPC-MOT 对我们最大的提醒是：

> 训练对象不应该停留在单个 pair。

我们现在做 detection-centered candidate group，其实和 LPC-MOT 在哲学上是一致的：
都在往“比 pair 更高一级的决策对象”走。

### 对我们论文的危险点

如果我们只说“我们利用 candidate context”，这不够。
因为 LPC-MOT 已经说明 proposal / higher-order object 是合理方向。

我们必须讲清楚我们的具体不同点：

- runtime replay
- strong host plugin
- ambiguity-triggered safe rerank

---

# 第二部分：learnable association 的大路线论文

这部分论文决定了我们不能再随便把论文写成“我们提出 learnable association”。

---

## 6. Multiple Object Tracking as ID Prediction (MOTIP)

- 会议：CVPR 2025
- 链接：
  https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf

### 一句话讲清楚

MOTIP 的核心是：
**把 MOT 关联问题直接写成 ID prediction。**

### 它到底在解决什么问题

传统 tracking-by-detection 最大的问题之一是：
association 还是靠 heuristic 或半启发式策略。

MOTIP 认为：
与其手工设计 matching logic，不如直接让模型预测每个 detection 属于哪个 identity。

### 方法核心

把当前 detections 放进已有轨迹上下文中，
将关联问题变成一个上下文条件下的 ID prediction 任务。

### 它和传统方法比，新在哪

不是“更好的 affinity score”，而是直接换了问题表述。

### 优点

- 新颖度高
- 叙事强
- 学术上更像“重新定义 association”

### 局限

- 路线更大、更重
- 离我们的 current frozen host plugin 路线更远
- 工程投入和风险都更高

### 对我们的启发

MOTIP 告诉我们：

> 如果你要讲“learnable association”，你面对的竞争对手已经不是小 MLP，而是这类重新定义任务的问题表述。

所以我们的论文绝不能再说成泛泛的 learnable association。

---

## 7. LA-MOTR: End-to-End Multi-Object Tracking by Learnable Association

- 会议：ICCV 2025
- 链接：
  https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.pdf

### 一句话讲清楚

这篇论文就是在正面宣称：
**association 本身应该被 end-to-end 地学出来。**

### 它到底在解决什么问题

它想彻底摆脱 heuristic data association。

### 方法核心

在 end-to-end MOTR 风格框架里，把 learnable association 做成主模块。

### 它和传统方法比，新在哪

不是在 host 上打补丁，而是把 association 当作系统主体来学。

### 优点

- 新颖度高
- “学 association” 这个口号在这里更成立

### 局限

- 和 tracking-by-detection 强宿主插件路线差异很大
- 不是我们当前资源和代码基础最适合走的路

### 对我们的启发

这篇论文的作用更多是“边界提醒”：

> 如果我们的贡献只写成“learnable association”，就会显得弱，而且容易被这类工作覆盖。

---

## 8. Learning a Neural Solver for Multiple Object Tracking

- 会议：CVPR 2020
- 链接：
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Braso_Learning_a_Neural_Solver_for_Multiple_Object_Tracking_CVPR_2020_paper.pdf

### 一句话讲清楚

这篇论文是在做：
**用神经网络来学图上的关联决策，而不是只靠局部匹配分数。**

### 它到底在解决什么问题

pair-wise affinity 不够表达全局结构关系。

### 方法核心

把检测和候选连接建图，
然后用图神经网络和 message passing 去做更全局的关联推理。

### 优点

- 很适合解释“为什么只看 pair 不够”
- 理论上比局部打分更强

### 局限

- 更偏 graph/global solver
- 不像我们这种强宿主插件那么轻量

### 对我们的启发

它支持我们的一个根本判断：

> 旧 learned 模块失败，不是 hidden dim 太小，而是它学的对象太弱。

---

## 9. Learnable Graph Matching (GMTracker)

- 会议：CVPR 2021
- 链接：
  https://openaccess.thecvf.com/content/CVPR2021/papers/He_Learnable_Graph_Matching_Incorporating_Graph_Partitioning_With_Deep_Feature_Learning_CVPR_2021_paper.pdf

### 一句话讲清楚

GMTracker 在做的是：
**让图匹配和特征学习更一致，而不是特征学一套、图优化再来一套。**

### 它到底在解决什么问题

很多 MOT 方法的问题是：

- 特征学习和最终匹配优化是分裂的
- 缺少 context

### 方法核心

把 tracklet 和 detection 看作图结构中的元素，
通过 learnable graph matching 做更一致的关联。

### 对我们的启发

它告诉我们：

> 如果训练对象和最终推理对象不一致，学习出来的东西往往不稳。

这和我们现在弃用 GT pseudo-group 主线完全一致。

---

## 10. FAMNet

- 会议：ICCV 2019
- 链接：
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_FAMNet_Joint_Learning_of_Feature_Affinity_and_Multi-Dimensional_Assignment_for_ICCV_2019_paper.pdf

### 一句话讲清楚

FAMNet 试图联合学习：

- feature
- affinity
- assignment

而不是把它们完全割裂开。

### 它的重要性

它更多是历史脉络型论文：
说明“association should be learned”这条线并不是新鲜事。

### 对我们的启发

它的意义主要是提醒我们：

> 不能把“我们用了 learnable association”本身当创新点。

---

# 第三部分：proposal / distribution / uncertainty 视角

---

## 11. TrackFlow: Multi-Object Tracking with Normalizing Flows

- 会议：ICCV 2023
- 链接：
  https://openaccess.thecvf.com/content/ICCV2023/papers/Mancusi_TrackFlow_Multi-Object_tracking_with_Normalizing_Flows_ICCV_2023_paper.pdf

### 一句话讲清楚

TrackFlow 的核心是：
**不要只做 heuristic score fusion，而是直接去学“正确关联”的概率分布。**

### 它到底在解决什么问题

传统 MOT 经常手工平衡多个 cue：

- IoU
- motion
- appearance
- 3D 信息

但这种平衡很脆弱。

### 方法核心

通过 normalizing flows 建模关联分布，让模型更灵活地表示“什么样的候选更像正确匹配”。

### 对我们的启发

它说明：

> association 学习可以不只是输出一个 score，也可以从 distribution 层面建模。

但对我们当前阶段来说，它更多是思想启发，不是最适合照搬的工程路线。

---

# 第四部分：host / memory / appearance 真正强在哪里

这一部分非常关键，因为它解释了：
为什么单独一个 association 小模块经常救不活系统。

---

## 12. MeMOT: Multi-Object Tracking With Memory

- 会议：CVPR 2022
- 链接：
  https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.pdf

### 一句话讲清楚

MeMOT 认为：
**长时记忆对 tracking 很重要，单帧或短历史信息不够。**

### 它到底在解决什么问题

目标被遮挡、离开、再回来时，
普通短期关联经常会断。

### 方法核心

通过 memory 结构维护更长期、更稳定的目标信息，
同时做 detection 和 association。

### 对我们的启发

如果以后我们要把 history encoder 重新引回来，
MeMOT 是最值得参考的论文之一。

但它也提醒我们：

> 历史建模要建立在正确的训练对象上。

否则 memory 很容易学到的只是干净 proxy 上的规律。

---

## 13. Quasi-Dense Similarity Learning for Multiple Object Tracking (QDTrack)

- 会议：CVPR 2021
- 链接：
  https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Quasi-Dense_Similarity_Learning_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf

### 一句话讲清楚

QDTrack 的核心结论其实很朴素：
**很多 tracking 提升，本质上来自更强的 similarity / ReID 学习，而不是更花的关联规则。**

### 它到底在解决什么问题

如果 appearance embedding 不够强，
后面再怎么做 association 也很难彻底解决问题。

### 方法核心

用 quasi-dense supervision 学习更强的目标相似度表示。

### 对我们的启发

这篇论文是我们必须长期记着的“边界提醒”：

> 不要把所有问题都甩给 association 模块。

如果后面 runtime replay reranker 证据补齐后仍然不够强，
下一步就很可能是更强 ReID，而不是继续折腾一个更黑的 association head。

---

# 第五部分：和“只在 ambiguity 组上动”最像的论文

---

## 14. Modelling Ambiguous Assignments for Multi-Person Tracking in Crowds

- 会议：WACV 2022 Workshop
- 链接：
  https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Stadler_Modelling_Ambiguous_Assignments_for_Multi-Person_Tracking_in_Crowds_WACVW_2022_paper.pdf

### 一句话讲清楚

这篇论文最重要的思想是：
**不是所有 assignment 都应该用同一种处理逻辑。**

### 它到底在解决什么问题

在人群场景里，很多 assignment 非常接近、非常模糊。
如果还强行用统一 Hungarian 逻辑，很容易出错。

### 方法核心

先识别 ambiguous assignments，
再把这些高风险情况单独处理。

### 对我们的启发

这篇几乎就是在支持我们当前的一个关键判断：

> ambiguity-only activation 不是可选项，而是必须项。

---

# 第六部分：新近值得关注但不必现在深挖的

---

## 15. UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking

- 状态：2026 近作
- OpenReview：
  https://openreview.net/forum?id=XpddZpGck9
- ArXiv：
  https://arxiv.org/abs/2602.05037

### 一句话讲清楚

UniTrack 关注的是：
**用图结构相关的可微目标，增强 MOT 的训练信号，并让方法更可插拔。**

### 为什么值得看

它和我们一样，带有一些“模块化 / plug-and-play”的味道。

### 为什么现在不用深挖

它更偏训练目标 / graph loss，
和我们当前最核心的问题“runtime replay rerank 是否站得住”不是同一个关键矛盾。

所以它现在是“关注即可”，不是“必须立刻照着改”。

---

# 最后一部分：读完这份导读，你应该得到什么结论

如果你不看原文，只看这一份，也应该能先建立下面这几个判断：

## 1. 我们原来的 old learned / Laplace 主线为什么不行

因为它本质上还是：

- 学 pair-wise calibration
- 在 GT-clean proxy 上训练
- 最终却想解决 runtime candidate competition

训练对象和推理对象没有对齐。

## 2. 我们现在的方向为什么比以前更对

因为它终于开始对齐真正的问题对象：

- 真实 tracker runtime candidate groups
- ambiguity groups
- safe plugin on strong host

## 3. 我们不能把创新点写成什么

不能写成：

- “我们提出了 learnable association”
- “我们提出了 ambiguity-aware tracking”
- “我们提出了 Laplace/frequency 模块”

这些说法都太泛，都会被现有文献压住。

## 4. 我们还能写成什么

最合理、也最有生存空间的写法是：

> 我们在强 MOT host 上，从真实 runtime 导出 detection-centered candidate groups，
> 训练一个只在 ambiguity 场景触发的安全型 reranking 插件，
> 用于修正 hard associations，同时保持 easy cases 稳定。

## 5. 哪几篇论文是必须重点对照的

如果时间很有限，你只重点看下面 6 篇即可：

1. DeconfuseTrack
2. TrackTrack
3. GeneralTrack
4. Deep OC-SORT
5. LPC-MOT
6. MOTIP

这 6 篇基本就能决定 reviewer 会怎么看我们。

---

# 给你的最终建议

如果你现在时间真的很少，你就按这个顺序使用这份文档：

1. 先把“第一部分”和“最后一部分”读完
2. 再挑 `DeconfuseTrack / TrackTrack / GeneralTrack / Deep OC-SORT / LPC-MOT / MOTIP` 这 6 篇看原文
3. 其它论文先只保留这份导读的理解，不着急逐篇细啃

这会比你从零开始读 15 篇原文更高效。

