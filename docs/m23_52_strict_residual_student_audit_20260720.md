# M23-52：严格 residual transaction student 审计

日期：2026-07-20

## 目标与协议

M23-52 只在 M23-51 teacher 容量接近 80 后启动，且只运行关键 outer-held M05 与 M02。两折均满足：

- baseline 为字节冻结的严格、可部署 M23-39 tracker 与 selected transactions；
- LABEL_PATHS 全部指向 M23-51 exact residual labels；
- 仅打开三条 outer-training 序列的 labels；
- held 序列仅从 GT-free prediction bank 与 frozen M23-39 transaction state 生成 residual candidates；
- held label 文件在 tracker 冻结前未读取，`held_label_file_read=false`；
- no-op / top-1 / top-2 由 nested inner official TrackEval 决定；
- held TrackEval 只在 `outer_tracker_frozen.json` 落盘后运行。

脚本：`scripts/m23_research/m23_52_action_type_conformal_residual_student.py`

## v1：absolute-HOTA conformal LCB

实验：`m23_52_action_type_conformal_m05_v1`

- 分头：drop / replacement HGB + linear heads；
- sequence/action-type rank features；
- source-domain logistic density ratio；
- 80% leave-one-inner-sequence one-sided exact-HOTA conformal LCB；
- 三条 inner 序列全部 abstain，严格冻结 no-op；
- held-M05 tracker 与 M23-39 字节一致；
- 官方 HOTA：**79.732850**；
- 相对 M23-39：0；相对 M23-46 M05：-0.037477。

结论：absolute-HOTA 非同质尺度导致 conformal correction 过大，该版本关闭。

## v2：local expert rank-conformal LCB

固定方法：

- 每个 source sequence、每个 action type 建独立局部专家；
- replacement 使用 5 个近邻，drop 使用 3 个近邻；
- target 为各序列、各 action type 内的 exact-delta rank；
- logistic source/target discriminator 估计 GT-free density ratio并加权专家；
- 80% leave-one-inner-sequence conformal rank lower bound；
- 仅 no-op / top-1 / top-2，不扫描阈值。

### held-M05

实验：`m23_52_local_rank_conformal_m05_v2`

训练 labels 仅 M01/M02/M03。inner exact TrackEval：

| Policy | M01 ΔHOTA | M02 ΔHOTA | M03 ΔHOTA | Worst | Mean |
|---|---:|---:|---:|---:|---:|
| top-1 | +0.058639 | +0.019920 | +0.003115 | +0.003115 | +0.027225 |
| top-2 | +0.058639 | +0.039527 | +0.003115 | +0.003115 | +0.033760 |

严格冻结 top-2，held labels 未读。选择两条 GT-free replacement：

- `source_index=167781`，LCB 0.156836，positive probability 0.663594，density support 0.918111；
- `source_index=223011`，LCB 0.129336，positive probability 0.514886，density support 0.883102。

冻结后官方 TrackEval：

- HOTA：**79.724750**
- DetA：81.954850
- AssA：77.597123
- IDSW：477
- 相对 M23-39：**-0.008100 HOTA**
- 相对 M23-46 M05：**-0.045577 HOTA**

冻结后 teacher 诊断才读取 M05 labels：两条动作的 exact 单动作增益分别为 -0.008214 与 +0.000125。inner 三折一致正增益仍未迁移到 M05。

### held-M02

实验：`m23_52_local_rank_conformal_m02_v1`

训练 labels 仅 M01/M03/M05。inner exact TrackEval：

- top-1：三个 inner fold 全负，worst -0.008216；
- top-2：仅 M01 为正，worst -0.013366；
- 因此严格冻结 no-op。

冻结后官方 TrackEval：

- HOTA：**73.098150**
- DetA：80.584820
- AssA：66.407293
- IDSW：325
- tracker 与 M23-39 byte-exact；无增益。

## 决策

M05 未超过 M23-46 的 79.770327，M02 未超过 73.098150。两个关键 fold 均无明显增益，因此按照预注册规则：

- **不扩展 M23-52 到 M01/M03 outer folds**；
- M23-52 严格协议有效，但方法路线关闭；
- 当前严格最佳仍为 M23-46 COMBINED HOTA **79.123193**；
- M23-51 teacher-only combined **79.719687** 仍仅代表 nondeployable capacity。

当前证据表明，残余动作的可部署瓶颈不是简单 top-k、概率阈值或 ordinary pooled classification，而是只有四个序列时，局部 residual action 的跨域符号稳定性不足。下一阶段不应继续扫描 M23-52 的 k、coverage 或阈值。
