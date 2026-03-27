# MOT Runtime Replay / 关联重排序相关工作中文阅读清单

日期：2026-03-17

这份笔记是围绕我们当前已经收敛下来的论文方向整理的：

- 强 tracking-by-detection 宿主
- 关联阶段插件，而不是重做整个 tracker
- 面向 ambiguity / confusion 的安全型重排序
- 训练对象是 runtime candidate groups，而不是 GT-clean 的 proxy pair

我把论文分成 3 组：

1. 和我们当前方向最接近的论文
2. learnable association / proposal scoring / relation modeling 相关论文
3. 宿主强度、记忆建模、外观表征相关支撑论文

每篇论文都尽量回答 6 个问题：

- 会议和年份
- 官方链接
- 核心思想是什么
- 它真正解决的“决策对象”是什么
- 它和我们工作的重合点在哪里
- 读的时候最该关注什么

阅读时最重要的提醒：

- 不要只看“它用了什么模块”。
- 要看“它到底在学习什么对象”。
- 还要看“它把模块插在 tracker 的哪个位置”。

对我们现在最关键的区分是：

- 它学的是 pair-wise score calibration
- 还是 candidate / proposal scoring
- 还是 ambiguity / confusion 处理
- 还是 end-to-end association
- 还是更强的 host / ReID / memory

---

## 一、和我们当前方向最接近的论文

这组论文和我们现在的方向最接近，因为它们都涉及下面这些关键词中的至少一个：

- confusion-aware
- ambiguity-aware
- association refinement
- strong host plug-in
- online hard-case handling

### 1. DeconfuseTrack: Dealing with Confusion for Multi-Object Tracking

- 会议：CVPR 2024
- 官方论文链接：
  https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_DeconfuseTrack_Dealing_with_Confusion_for_Multi-Object_Tracking_CVPR_2024_paper.pdf
- 为什么重要：
  这是和我们当前“ambiguity / confusion handling”方向最接近的论文之一。它不是泛泛而谈“学一个更好的关联分数”，而是明确把 confusion 当成问题中心。
- 核心思想：
  MOT 中很多错误并不是普通关联误差，而是 confusion 引起的。论文把混淆来源拆开建模，并通过 confusion-aware 的设计去缓解，而不是把所有情况都交给一个统一打分器处理。
- 它真正处理的决策对象：
  跟踪过程中的混淆型关联问题。
- 和我们工作的重合点：
  很高。它和我们都在强调：hard cases 不应该和 easy cases 用完全一样的规则处理。
- 和我们工作的关键差异：
  它不是建立在 runtime replay candidate groups 训练对象上的；我们最强的区分点是“从真实运行时导出候选组再训练”。
- 阅读重点：
  重点看它如何定义 confusion，如何证明 confusion-specific 的处理是必要的。
- 对我们论文的提醒：
  如果我们只说“我们处理 ambiguity / confusion”，reviewer 很容易说这篇已经做过。因此我们必须强调：
  **runtime replay + safety-aware reranking + frozen host plugin**

### 2. Focusing on Tracks for Online Multi-Object Tracking (TrackTrack)

- 会议：CVPR 2025
- 官方论文链接：
  https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf
- 为什么重要：
  这篇论文直接质疑传统 online MOT 里默认的“全局 cost matrix + Hungarian”逻辑，强调从 track 的角度重新看 online association。
- 核心思想：
  不是一股脑把所有 detection-track 对塞进一个统一匹配矩阵里，而是更关注 track 的连续性、可靠性和 hard association 的本质。
- 它真正处理的决策对象：
  online association 的决策流程本身。
- 和我们工作的重合点：
  都认为困难关联不该和普通关联一视同仁，也都认为强 host 里的 hard cases 才是真正值得动手的地方。
- 和我们工作的关键差异：
  TrackTrack 更像是重设计在线关联逻辑；我们的方向更窄，是在冻结强 host 的前提下做一个安全型重排序插件。
- 阅读重点：
  看它为什么认为传统 frame-centric matching 不够，为什么要转向 track-centered 思路。
- 对我们论文的提醒：
  如果我们只说“我们改善 hard online association”，它会成为非常自然的对照工作。

### 3. Modelling Ambiguous Assignments for Multi-Person Tracking in Crowds

- 会议：WACV 2022 Workshop
- 官方论文链接：
  https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Stadler_Modelling_Ambiguous_Assignments_for_Multi-Person_Tracking_in_Crowds_WACVW_2022_paper.pdf
- 为什么重要：
  这篇论文和我们“只在 ambiguity groups 上触发模块”的思想很接近。
- 核心思想：
  把 ambiguous assignments 单独识别出来，和普通清晰匹配区分开处理，而不是统一地跑一个标准匹配算法。
- 它真正处理的决策对象：
  ambiguous assignment set。
- 和我们工作的重合点：
  都认为 ambiguity 本身就是一个值得单独建模的问题。
- 和我们工作的关键差异：
  它不是 learned runtime replay reranker，也不是基于真实运行时 candidate group 训练。
- 阅读重点：
  看它怎么定义 ambiguous assignment，以及它怎么把问题拆开。
- 对我们论文的提醒：
  如果我们只说“我们是 ambiguity-aware”，这个点不够新，必须再往上加：
  **runtime replay + safe reranking**

### 4. Deep OC-SORT: Multi-Pedestrian Tracking by Adaptive Re-Identification

- 会议：ICIP 2023
- 官方页面：
  https://resourcecenter.ieee.org/conferences/icip-2023/spsicip23vid0663
- ArXiv：
  https://arxiv.org/abs/2302.11813
- 为什么重要：
  虽然不是 CVPR/ICCV/ECCV 主会，但它是“强 host + association refinement”这条工程与论文风格的一个非常重要参考。
- 核心思想：
  在强 tracker 的基础上，自适应地使用 appearance cues，而不是用固定规则融合 appearance 和 motion。
- 它真正处理的决策对象：
  association-time 的 cue integration。
- 和我们工作的重合点：
  非常高。都属于“不要重做整个 tracker，而是在强 host 上做 association refinement”。
- 和我们工作的关键差异：
  它更偏 adaptive appearance integration；我们现在更偏 runtime replay candidate-group reranking。
- 阅读重点：
  看它如何把一个 plug-in 型贡献讲得足够像论文贡献，而不是普通工程调参。
- 对我们论文的提醒：
  reviewer 可能会问我们是不是只是 another adaptive association plugin。
  所以我们必须说明：
  **我们的真正创新不在于“又加了一个关联头”，而在于 runtime replay 训练对象和 ambiguity-triggered 安全重排。**

### 5. Towards Generalizable Multi-Object Tracking (GeneralTrack)

- 会议：CVPR 2024
- 官方页面：
  https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html
- PDF：
  https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.pdf
- 为什么重要：
  这篇论文对我们现在尤其关键，因为我们已经看到：random split 结果是正的，但跨序列结果不稳定。这个问题和它讨论的 generalization 直接相关。
- 核心思想：
  传统 MOT 容易过拟合到特定场景下的 motion/appearance 平衡方式，因此需要更 generalizable 的关系建模。
- 它真正处理的决策对象：
  instance-wise relation，而不是手工 balancing。
- 和我们工作的重合点：
  很高。我们现在最大的一个风险就是：学到的不是稳定规律，而是局部场景偏好。
- 和我们工作的关键差异：
  GeneralTrack 更像一个更一般化的跟踪框架方向；我们是一个更窄、更模块化的 runtime replay rerank 方向。
- 阅读重点：
  重点看它如何分析“为什么已有 MOT 方法泛化差”。
- 对我们论文的提醒：
  如果我们最后只能在 random split 上好看，跨序列和跨 host 都不稳，这篇论文会非常不利于我们。

---

## 二、Learnable Association / Proposal Scoring / Relation Modeling 相关论文

这一组论文很重要，因为它们决定了“learnable association”这个大方向已经拥挤到什么程度。

如果我们把论文写成“我们提出了 learnable association”，这些论文会直接成为 reviewer 的比较对象。

### 1. Multiple Object Tracking as ID Prediction (MOTIP)

- 会议：CVPR 2025
- 官方论文链接：
  https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf
- 为什么重要：
  这是近两年 learned association 方向里非常强的一篇。
- 核心思想：
  把 MOT 关联直接重写成 ID prediction 问题，即在已有轨迹上下文中预测当前检测对应的 identity。
- 它真正处理的决策对象：
  detection 的 ID prediction。
- 和我们工作的重合点：
  都认为传统 heuristic association 已经不够了。
- 和我们工作的关键差异：
  MOTIP 更激进，更像“association 主体完全 learnable”；我们现在更适合讲成“冻结强 host 上的安全型插件”。
- 阅读重点：
  看它如何把 association 本身写成一个完整学习问题，而不是 cue fusion 小修小补。
- 对我们论文的提醒：
  如果我们把自己说成“learnable association”，很容易被 MOTIP 直接压住。因此必须收窄为：
  **runtime replay trained safe reranking plugin**

### 2. LA-MOTR: End-to-End Multi-Object Tracking by Learnable Association

- 会议：ICCV 2025
- 官方页面：
  https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.html
- PDF：
  https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV_2025_paper.pdf
- 为什么重要：
  这篇论文的标题本身就在提醒：learnable association 已经不是一个空白地带了。
- 核心思想：
  在 end-to-end MOT 框架中，把 association 设计成 learnable component，而不是传统 heuristic matching。
- 它真正处理的决策对象：
  end-to-end learnable association。
- 和我们工作的重合点：
  都在反对 purely heuristic association。
- 和我们工作的关键差异：
  它是完整 end-to-end tracker 框架；我们是强 host 上的可插拔安全模块。
- 阅读重点：
  看它如何论证 heuristic association 的上限，以及 learnable association 的优势。
- 对我们论文的提醒：
  我们不能把 novelty 写成笼统的“learnable association”，必须更具体。

### 3. Learning a Proposal Classifier for Multiple Object Tracking (LPC-MOT)

- 会议：CVPR 2021
- 官方页面：
  https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Learning_a_Proposal_Classifier_for_Multiple_Object_Tracking_CVPR_2021_paper.html
- PDF：
  https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Learning_a_Proposal_Classifier_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf
- 为什么重要：
  这是 proposal-level 学习方向里和我们最相关的一篇之一。
- 核心思想：
  MOT 不应该只依赖 local pairwise affinity，而应该通过生成候选 proposal，再对 proposal 进行学习式打分。
- 它真正处理的决策对象：
  candidate trajectory proposal。
- 和我们工作的重合点：
  很强。它和我们都在往“不要只学 pair-wise 小分数”这个方向走。
- 和我们工作的关键差异：
  LPC-MOT 是 proposal scoring；我们是 detection-centered candidate-group reranking。
- 阅读重点：
  看它为什么认为 proposal-level object 比 pair-wise affinity 更合理。
- 对我们论文的提醒：
  如果我们只说“我们利用 higher-order context”，reviewer 可能会把我们当成 proposal scoring 的窄化版本。

### 4. Learning a Neural Solver for Multiple Object Tracking

- 会议：CVPR 2020
- 官方论文链接：
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Braso_Learning_a_Neural_Solver_for_Multiple_Object_Tracking_CVPR_2020_paper.pdf
- 为什么重要：
  这是 graph-based learnable tracking 的代表论文之一。
- 核心思想：
  用神经网络学习图上的边决策和全局推理，而不是只看局部 pairwise similarity。
- 它真正处理的决策对象：
  detection graph 上的 edge decisions。
- 和我们工作的重合点：
  都不满足于简单 pair-wise 关联。
- 和我们工作的关键差异：
  它是更完整、更大范围的 graph solver；我们则更实用、更受约束，是 host-compatible reranker。
- 阅读重点：
  重点不是网络结构，而是它如何论证“全局结构推理是必要的”。

### 5. Learnable Graph Matching: Incorporating Graph Partitioning With Deep Feature Learning for Multiple Object Tracking (GMTracker)

- 会议：CVPR 2021
- 官方论文链接：
  https://openaccess.thecvf.com/content/CVPR2021/papers/He_Learnable_Graph_Matching_Incorporating_Graph_Partitioning_With_Deep_Feature_Learning_CVPR_2021_paper.pdf
- ArXiv：
  https://arxiv.org/abs/2103.16178
- 为什么重要：
  这篇论文明确强调：context 和 optimization consistency 很重要。
- 核心思想：
  把 tracklet-detection association 写成图匹配问题，并通过可微分方式让特征学习和图优化更一致。
- 它真正处理的决策对象：
  tracklet graph 和 detection graph 的 matching。
- 和我们工作的重合点：
  都认为 pair-wise 层面的建模能力不够。
- 和我们工作的关键差异：
  GMTracker 更图优化导向；我们更 runtime replay / plugin 导向。
- 阅读重点：
  看它如何解释 feature learning 和 graph optimization 脱节的问题。

### 6. TrackFlow: Multi-Object Tracking with Normalizing Flows

- 会议：ICCV 2023
- 官方页面：
  https://openaccess.thecvf.com/content/ICCV2023/html/Mancusi_TrackFlow_Multi-Object_tracking_with_Normalizing_Flows_ICCV_2023_paper.html
- PDF：
  https://openaccess.thecvf.com/content/ICCV2023/papers/Mancusi_TrackFlow_Multi-Object_tracking_with_Normalizing_Flows_ICCV_2023_paper.pdf
- 为什么重要：
  这篇论文给 learned association 提供了一个更 probabilistic 的建模视角。
- 核心思想：
  用 normalizing flow 去建模正确关联的分布，而不是用手工方式融合多个 cue。
- 它真正处理的决策对象：
  association likelihood / matching distribution。
- 和我们工作的重合点：
  都想摆脱 ad hoc 的 cue fusion。
- 和我们工作的关键差异：
  它是 learned probabilistic association；我们当前是更保守的 runtime candidate-group reranking。
- 阅读重点：
  看它如何证明 learned probabilistic modeling 比 heuristic fusion 更合理。

### 7. FAMNet: Joint Learning of Feature, Affinity and Multi-dimensional Assignment

- 会议：ICCV 2019
- 官方论文链接：
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_FAMNet_Joint_Learning_of_Feature_Affinity_and_Multi-Dimensional_Assignment_for_Online_Multiple_Object_Tracking_ICCV_2019_paper.pdf
- 为什么重要：
  这篇算是“feature / affinity / assignment 联合学习”路线里的代表之一。
- 核心思想：
  特征学习、亲和度估计和 assignment 本身不应该完全拆开做。
- 它真正处理的决策对象：
  联合 feature-affinity-assignment pipeline。
- 和我们工作的重合点：
  是 learnable association 这条线的历史背景。
- 和我们工作的关键差异：
  我们现在更像一个实用、窄范围、插件式的方法，而不是联合重做整个 tracking pipeline。
- 阅读重点：
  主要把它当作历史脉络，不是最接近的对照。

---

## 三、宿主强度、记忆建模、外观表征相关支撑论文

这组论文虽然和 runtime replay reranking 不是最直接重合，但非常重要，因为它们解释了：

- 很多 MOT 提升其实来自更强 detector / ReID / host
- 长时历史建模为什么会重要
- 为什么单纯调 association 头经常不够

### 1. MeMOT: Multi-Object Tracking With Memory

- 会议：CVPR 2022
- 官方页面：
  https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html
- PDF：
  https://openaccess.thecvf.com/content/CVPR2022/papers/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.pdf
- ArXiv：
  https://arxiv.org/abs/2203.16761
- 为什么重要：
  如果我们后面重新把 history encoder 做强，这篇是必须读的。
- 核心思想：
  用显式 memory 提升长时关联能力，让 identity representation 更稳定。
- 它真正处理的决策对象：
  memory-enhanced long-term association。
- 和我们工作的重合点：
  高。它支持“历史建模确实可能提升关联”的这个方向。
- 和我们工作的关键差异：
  它不是 frozen-host plug-in，也不是 runtime replay reranker。
- 阅读重点：
  看它如何组织 long-term memory，以及它如何把 memory 和当前关联任务对接。

### 2. Quasi-Dense Similarity Learning for Multiple Object Tracking (QDTrack)

- 会议：CVPR 2021
- 官方页面：
  https://openaccess.thecvf.com/content/CVPR2021/html/Pang_Quasi-Dense_Similarity_Learning_for_Multiple_Object_Tracking_CVPR_2021_paper.html
- PDF：
  https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Quasi-Dense_Similarity_Learning_for_Multiple_Object_Tracking_CVPR_2021_paper.pdf
- 为什么重要：
  这是对我们非常重要的一篇“提醒型论文”。它说明很多 MOT 增益其实先来自更强的相似度学习 / appearance supervision，而不是后端加一个小 calibrator。
- 核心思想：
  用 quasi-dense 的方式学习更丰富的目标相似度，而不是只靠稀疏配对监督。
- 它真正处理的决策对象：
  appearance similarity learning。
- 和我们工作的重合点：
  它不是 reranking，但它直接决定了 host 的 candidate quality 和 ReID 区分能力。
- 和我们工作的关键差异：
  QDTrack 是在学更强的 matching feature；我们是在学更好的关联决策对象。
- 阅读重点：
  把这篇当成一个提醒：
  不是所有提升都应该从 association head 里找。

### 3. Deep OC-SORT 也应从 host 强度角度再读一遍

- 为什么这里再次提它：
  它不仅是“最接近的 refinement paper”，也是一个很好的“强 host 论文写法”例子。
- 该关注什么：
  看它如何证明自己是在强 host 上提供额外价值，而不是靠整条系统堆料。

---

## 四、近期观察论文

这类论文值得关注，但引用时要注意正式发表状态和说法。

### 1. UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking

- 状态：
  2026 年较新的工作；有 OpenReview 和 arXiv，可作为近邻方向参考，但正式引用表述前要确认其最终发表状态
- OpenReview：
  https://openreview.net/forum?id=XpddZpGck9
- ArXiv：
  https://arxiv.org/abs/2602.05037
- 为什么重要：
  它提出了 graph-theoretic、plug-and-play 的训练目标，并声称能在多个 host 上工作。
- 核心思想：
  直接优化和 MOT 图结构相关的目标，而不是只调局部关联分数。
- 和我们工作的关系：
  概念上接近“模块化 + 跨宿主”，但它主打的是训练目标，而不是 runtime candidate-group reranking。
- 阅读重点：
  看它是怎么把 modular method 讲成论文贡献的。

---

## 五、推荐阅读顺序

如果你想按最适合我们当前项目的顺序来读，建议是：

1. DeconfuseTrack
2. TrackTrack
3. GeneralTrack
4. LPC-MOT
5. MOTIP
6. Deep OC-SORT
7. MeMOT
8. QDTrack
9. GMTracker
10. Learning a Neural Solver
11. LA-MOTR
12. TrackFlow
13. FAMNet
14. Modelling Ambiguous Assignments in Crowds
15. UniTrack

推荐这样排的原因：

- 前三篇先把问题定义看清楚：
  confusion、hard cases、generalization
- 中间几篇看 learnable association 这一大方向已经卷到了哪里
- 再往后看 memory 和 appearance，理解宿主为什么重要
- 最后补图建模、概率建模、end-to-end 大方向

---

## 六、这些论文对我们当前论文方向意味着什么

读完这些论文之后，我们现在最安全、也最合理的定位不是：

- “我们提出了一个更好的 pair-wise association score”
- “我们提出了一个 Laplace / frequency 模块”
- “我们能处理 ambiguity”

因为这些说法都不够稳，也很容易被已有工作覆盖。

现在最该坚持的定位应该是：

- 强 host
- 真实 runtime candidate groups
- ambiguity-triggered activation
- bounded / safety-aware reranking
- frozen host plugin on/off 实验

更准确地说：

> 我们不是在重新发明 learnable association，而是在强 MOT host 上，用 runtime replay 训练一个只对歧义候选组生效的安全型重排序插件。

这才是我们还有希望立住的空间。

