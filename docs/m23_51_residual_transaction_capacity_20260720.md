# M23-51：严格 M23-39 基线上的 residual transaction 容量审计

日期：2026-07-20

## 结论

M23-51 以字节校验后的严格、可部署 M23-39 tracker / selected transactions / applied graph 为冻结基线。候选 shortlist 完全由原 GT-free transaction prediction bank 构造；shortlist 冻结后才读取当前已授权序列 GT，生成 exact HOTA teacher labels。所有 M23-51 与 interaction oracle 结果均为 `teacher_only=true`、`deployable=false`。

四个 residual label 任务均完整成功，候选表中未出现 `same_gt`、`modal_gt`、`purity`、`label_confidence`、`actual_assa`：

| Fold | 成功动作 | 正/负/零 | Baseline HOTA | 最佳单动作 HOTA | 单动作增益 |
|---|---:|---:|---:|---:|---:|
| M01 | 137/137 | 37/98/2 | 78.805125 | 79.291773 | +0.486648 |
| M02 | 153/153 | 44/99/10 | 73.098153 | 73.467559 | +0.369406 |
| M03 | 135/135 | 23/110/2 | 80.603278 | 80.668545 | +0.065267 |
| M05 | 159/159 | 34/112/13 | 79.732847 | 79.908365 | +0.175518 |

Interaction oracle 最佳结果：

| Fold | 最佳策略 | HOTA | 相对 baseline | Drop | Replace |
|---|---|---:|---:|---:|---:|
| M01 | combo_t0p025 | 80.307120 | +1.501995 | 5 | 9 |
| M02 | combo_t0p05 | 74.504799 | +1.406646 | 5 | 9 |
| M03 | combo_t0p0 | 81.057537 | +0.454259 | 3 | 16 |
| M05 | combo_t0p005 | 80.229372 | +0.496525 | 4 | 12 |

四个最佳 tracker 拼接后仅运行一次官方 TrackEval：

- COMBINED HOTA：**79.719687**
- DetA：**81.566010**
- AssA：**77.965885**
- IDSW：**1010**
- 相对严格最佳 M23-46（79.123193）：**+0.596494**
- 距离 80：**0.280313**

该结果证明：围绕真实严格 M23-39 基线的 residual transaction 动作族具有明显交互容量，但其当前 teacher 上限仍未超过 80。由于 79.719687 已接近 80，按预注册判断进入严格 M23-52 student；先只测试 held-M05 与 held-M02。M23-51 oracle 不得提交为可部署成绩。

## 协议审计

- outer-held GT 未参与冻结 baseline、candidate construction 或 student 选择。
- M23-51 labels 只能用于 outer-training 或已经暴露的诊断序列。
- exact HOTA GT teacher 在 shortlist 冻结后才执行。
- 每个 action 均获得 exact HOTA label，所有四折无失败 action。
- 官方 combined 只评测一次。
- 当前严格最佳仍为 M23-46：79.123193。
- M23-43 的 80.029637 仍是 nondeployable oracle；M23-47/48/49/50 路线保持关闭。
