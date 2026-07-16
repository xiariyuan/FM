# Pending Notion Sync｜AssocRiskBench P18–P19

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Reason pending: 当前会话未暴露可写 Notion connector。未声称任何 Notion 写入成功。

Source of truth:

- `deliverables/assocriskbench_p18_p19_calibration_fullbank_closure_20260716.md`
- `deliverables/assocriskbench_p19_fullbank_audit_20260716.json`

## Pending section｜P18–P19 域校准闭环与 Full Executable Bank

- P18 在 705-event canonical bank 上执行严格 nested sequence calibration：三类 base view 完全 inner-OOF，阈值仅从六条源序列选择，并要求源序列全覆盖、总 utility 为正、最差源序列非负。
- 7 个 outer folds 的 inner authorization 全部为 0，最终 selected blocks 为 0；canonical scalar calibration family 正式关闭。
- P19 保留七条 MOT17 序列全部 11,705 个 executable events，不做 geometry priority、spacing 或数量上限；完整候选集合冻结后才读取 GT 生成 event-level teacher。
- Full bank 包含 1,050 positive、10,147 negative、508 zero events，代表 1,786,622 条 changed rows。
- P17 canonical bank 仅覆盖 104 / 1,050 positive events，正例召回为 9.9048%，说明 canonicalization 丢失约 90.1% 正例。
- 新 metadata reconstruction 模式生成 11,705×373 actual-anchor motion table；367 个 label-free features，最大缺失率 1.0423%，无重复键或禁用字段。
- P16 与 P17 原 changed-row motion 模式全部字节回归一致。
- Full-bank sequence-LOSO：geometry positive AUC 0.845246；geometry utility Spearman 0.673480；motion multitask 28-block top-one utility 首次转正至 +0.202141，但 worst sequence 仍为 −0.769903。
- Full-bank oracle utility sum 为 +7.372292，worst sequence 为 +0.286058；候选空间足够，剩余问题是极端高分伪阳性。
- 最佳 retrospective motion threshold 在七域覆盖下 utility 为 +0.995300，但 worst sequence 仍为 −0.643856；所有 scalar gate eligibility 均为 0。
- 决策：保留 full teacher 与 full motion bank；不推广任何固定模型；关闭 scalar calibration / scalar threshold family；P15 保持 no-op；156 条 locked rows 保持未读。
- 四条正式链均完成逐文件 SHA256 复现。

## Next method

进入 set-valued selective prediction、support/OOD tail certificate 与 conformal risk-set authorization。禁止继续对 frozen P15 做 scalar threshold 或 manifest sweep。
