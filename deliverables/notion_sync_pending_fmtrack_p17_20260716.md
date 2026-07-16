# Pending Notion Sync｜AssocRiskBench P17

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Reason pending: 当前工具会话未暴露可写 Notion connector。未声称任何 Notion 写入成功。

Source of truth:

- `deliverables/assocriskbench_p17_mot17_train7_domain_expansion_20260716.md`
- `deliverables/assocriskbench_p17_mot17_train7_domain_audit_20260716.json`

## Pending section｜P17 MOT17 七域反事实扩展

- 使用无 GT 轨迹拓扑候选枚举与冻结 directional planner，构建 MOT17 七序列独立反事实 teacher bank。
- 候选 priority 与 30-frame per-track canonicalization 均只使用 tracker geometry；GT 仅在候选集合冻结后生成 dense local teacher。
- 共审计 27,336 个 ordered directions，其中 11,705 个 executable；冻结 705 个 teacher events，生成 85,291 条 changed-row labels。
- Teacher 分布：104 positive、571 negative、30 zero；正例率跨序列从 2.56% 到 26.83%。
- 13-target multitask 将 MOT17 七域 sequence-LOSO positive AUC 从 0.524990 提升到 0.702051，但 28 个 temporal-block top-one 中仍有 18 个负例，worst sequence utility 为 −0.642163。
- 在 P15 LOSO 中加入七域 teacher 后，multitask pooled AUC 从 0.558145 提升到 0.598399，但 local worst sequence 为 −0.334845，HOTA worst sequence 为 −0.056277，仍有 2 个 catastrophic windows。
- 两个固定模型族均未通过所有 MOT17/P15 domain 的 worst-sequence nonnegative 约束。
- 决策：保留 P17 teacher bank；拒绝 naive/multitask pooling 直接部署；P15 policy 保持 `no_op`；不创建 locked manifest。
- P15 新 locked labels：0；新 locked TrackEval：0；156 条 remaining locked rows 保持未读。
- 六条 P17 正式链均完成逐文件 SHA256 复现。

## Next method

进入 domain-conditioned / invariant-mechanism utility modeling，并使用 nested leave-one-sequence-out 与 worst-domain lower-tail objective。任何后续 P15 manifest 仍必须满足全部域覆盖和 worst-sequence utility 非负。
