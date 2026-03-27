# Send To Pro: Future Experiment Decision Review

这是一份用于审阅未来实验排序、go/kill 节点和主线延续性的发送版提示词。

```markdown
请你把自己当成独立研究顾问。不要重新开放式发散到旧方向，也不要泛泛谈可能性。我现在需要你基于完整上下文，直接审阅“未来实验决策应该如何排、哪些该继续、哪些该停”。

## 1. 当前项目已经收敛后的主线

我当前做的是 tracking-by-detection 范式下的 MOT 研究。
当前固定宿主是 `base_reid_da`，但它不是主角方法，只是当前最稳定的 host baseline。

当前真正的主线是：
- 主角方法：`competition-aware association controller`
- 第一阶段：在真实 runtime ambiguity/conflict groups 上输出 `keep / rerank / null`
- 第二阶段：如果第一阶段在线成立，再补 `short-gap continuity / tracklet-level continuity`

## 2. 已明确停掉或降级的路线

### 已停主线
- Laplace / MTCR / HACA / pairwise residual safe plugin
- 当前 frequency-aware / spatial-freq interaction 主线

### 已降级为二线
- runtime replay safe plugin

因此，请你不要再建议我回到这些路线当主线，除非你能基于下面事实明确指出它们为什么应该被复活。

## 3. 当前关键对照证据

这是 MOT17 proxy0213（MOT17-02 / 13）上的干净控制链：

### `base_spatial` 最好点
- HOTA = 52.233
- AssA = 43.089
- IDF1 = 58.557
- MOTA = 72.382
- IDSW = 1148

### `base_reid_da` 最好点
- HOTA = 52.704
- AssA = 43.997
- IDF1 = 58.608
- MOTA = 73.361
- IDSW = 834

### `full_reid_da` epoch0
- HOTA = 51.733
- AssA = 42.365
- IDF1 = 57.969
- MOTA = 73.471
- IDSW = 791

结论：
- `base_reid_da` 相对 `base_spatial` 有真实净提升
- `full_reid_da` 并没有继续提升关联质量，因此 frequency family 当前已停主线

## 4. 当前新主线的数据与训练已经跑通

### 4.1 真实 runtime conflict-set 已生成成功

proxy0213 competition summary:
- groups = 29319
- positive_groups = 19756
- ambiguous_groups = 19756
- recoverable_groups = 4751
- recoverable_rate_among_positive = 0.24048
- bridge_rate_among_positive = 0.03801

这说明：
- hard groups 数量是够的
- 可恢复重排组约占 positive groups 的 24%
- continuity bridge 占比很小，所以第一阶段先做 rerank / conflict decision 是合理的

### 4.2 stage1 controller 离线训练已修复并可学习

之前 NaN 的原因是：
- 一部分 group 的 `valid_mask` 全 false
- `MultiheadAttention` 在全 mask 行上产出 NaN
- 训练脚本没有强制把非有限值标失败

现在这个问题已经修复：
- all-invalid rows 已做数值稳定处理
- finite checks 已加入训练脚本
- 非有限 loss 会直接回写 failed

### 4.3 最新正式 stage1 结果

当前 stage1 正式结果：
- best_epoch = 10
- val_action_acc = 0.618739
- val_rerank_candidate_acc = 0.915464
- val_rerank_action_acc = 0.657732
- status = ok

这说明：
- 离线冲突控制器已经学到有效 rerank signal
- 但现在还没有 online integration 结果

## 5. 当前最关键的下一步

我接下来最直接的动作是：
- 把 stage1 best checkpoint 接回 `base_reid_da` host
- 先做 proxy0213 在线闭环

之后我可能会进入这些分支：
- 分支 A：如果 proxy0213 在线为正，做安全性 / gating / trigger ablation
- 分支 B：如果 proxy0213 在线依旧为正，再扩 full7
- 分支 C：如果 proxy0213 离线强但线上不涨，做 integration / gating / objective mismatch 诊断
- 分支 D：如果第一阶段在线成立，再补 continuity / tracklet-level continuity

## 6. 请你只回答未来实验决策问题

请你直接回答：

1. 以当前证据看，这条主线现在是否值得继续推进？请给明确判断，不要模糊。
2. 我接下来的实验顺序应该怎么排？请按优先级给出 1, 2, 3, 4。
3. 哪个实验是当前信息增益最高、最能决定 go / kill 的？
4. 哪些实验现在不值得做，应该暂缓或直接砍掉？
5. 如果 proxy0213 在线不涨，你建议我优先：
   - 修 online integration
   - 修 gating / intervention design
   - 改 loss / selection
   - 还是直接切到 continuity / tracklet level
   请明确排序。
6. 如果 proxy0213 在线为正，你建议是先做 full7，还是先做 safety ablation？为什么？
7. 以你判断，当前这条线未来最可能变成：
   - 一个能成立的主线
   - 一个二线插件
   - 或者一个最终应被 kill 的过渡方向
   请直接给判断，并说明依据。

## 7. 我希望你的输出方式

请直接给我：
- go / kill 判断
- 未来实验决策顺序
- 最该做和最不该做的实验
- 最值得警惕的失败模式

不要泛泛谈方向，不要重复旧历史，不要只说“先跑看看”。
```

