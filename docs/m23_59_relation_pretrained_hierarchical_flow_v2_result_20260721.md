# M23-59 v2 Relation-Pretrained Hierarchical Flow — Final Result (2026-07-21)

## 1. Final decision

M23-59 v2 is **closed under preregistered branch C**. All four strict sequence-LOSO outer representation gates failed, so every held sequence froze the preregistered **P0 no-op policy**. No learned inner tracker was permitted, no new outer TrackEval was permitted, and the deployable result is byte-exact M23-46.

- Selected policies: `MOT20-01=P0`, `MOT20-02=P0`, `MOT20-03=P0`, `MOT20-05=P0`.
- Final deployable metrics: **HOTA 79.123193, DetA 81.543470, AssA 76.825150, IDSW 996**.
- Strict improvement over M23-46: **0.000000 HOTA**; M23-46 remains the internal-data strict deployable best.
- Margin to HOTA 80: **-0.876807**.
- External supervision: **yes** (MOT17 physical-sequence pretraining only).
- MOT20 test submission: **no**.

## 2. Validity and frozen implementation

M23-59 v1 is wholly invalidated because the deterministic CUDA configuration was applied incorrectly; no v1 scientific artifact was reused. M23-59 v2 was executed only after the implementation, preregistration, external dataset manifest, GT-free MOT20 observables, and external checkpoint were frozen and hash-verified.

- Git HEAD / origin-main at implementation freeze: `a04dfa0012114c1663ea1cb543158bc4d1d975a5`.
- Frozen script SHA-256: `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d`.
- Preregistration document SHA-256: `b78ffcf40397c8c9dcd1493afc4eb2519667e876186ed3d331e882ef975d4f46`.
- Preregistered protocol JSON SHA-256: `452bbd4305525eff635daa2bb0cfefd9dec70075e6672104254eed44b82b0822`.
- External dataset manifest SHA-256: `3c555e708a7341f6747a4a24ab9200bb9bf848d37b847dde6bc646b44e28e406`.
- Frozen external checkpoint SHA-256: `141c91a93ae58164ed5399cf1b1407e6fcd3dfec52b6950f94afa65251e1da42`.
- Parameter count: `881,124` (cap `5,000,000`).
- Determinism: hard deterministic algorithms, TF32 disabled, cuDNN benchmark disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before importing torch.

## 3. External pretraining evidence (not deployable MOT20 metrics)

The frozen external model was selected from seeds `2359001, 2359002, 2359003` using the fixed MOT17 validation composite. Selected seed `2359003`, epoch `12`:

- Checkpoint-selection composite: `0.871551282`.
- Conditional boundary PR-AUC: `0.935765546`.
- Conditional boundary precision at actual count: `0.864163614`.
- Conditional boundary recall at 95% precision: `0.717643468`.
- Node impurity PR-AUC: `0.878073035`.
- Outgoing successor pairwise R@1: `0.778217409`.
- Incoming predecessor pairwise R@1: `0.805705956`.
- Paired replacement R@1: `0.986816406`.
- Catastrophic false-link rate: `0.173789907`.

External data accounting:

- Feature extraction: `95,872` eligible rows over `5,316` frames; audit errors: `0`.
- Training examples: `10,210` nodes, `13,297` relation triplets, `5,104` paired replacements.
- Validation examples: `3,286` nodes, `4,802` relation triplets, `2,048` paired replacements.
- Train/validation physical-sequence disjointness: `true`.

These metrics validate the frozen external representation only; they are not MOT20 deployable tracking results.

## 4. MOT20 freeze and label ordering

All four GT-free observable tensors and their M23-57 fixed-membership manifests were SHA-verified. Protocol events show all observable freezes completed before the first MOT20 label unlock. Held-outer GT and teacher actions were not read before outer policy freeze.

## 5. Strict outer representation gates and policies

Frozen gate thresholds were: mean boundary PR-AUC ≥ 0.283; mean precision at actual boundary count ≥ 0.35; every-fold precision ≥ 0.20; mean recall at 95% precision ≥ 0.05; every-fold pure-node false-split rate ≤ 0.002. ROC-AUC could not satisfy the gate.

| Outer | Gate | Mean boundary PR-AUC | Mean precision@actual | Mean recall@95P | Max pure-node false-split | Mean node PR-AUC | Successor R@1 | Catastrophic false-link | Policy | Inner TrackEval | Fold HOTA | Fold AssA | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT20-01 | FAIL | 0.069699 | 0.097192 | 0.000000 | 0.000000 | 0.171317 | 0.019649 | 0.981724 | P0 | 0 | 78.805125 | 76.043165 | 46 |
| MOT20-02 | FAIL | 0.060923 | 0.077449 | 0.000000 | 0.000000 | 0.134667 | 0.018226 | 0.982859 | P0 | 0 | 73.098150 | 66.407293 | 325 |
| MOT20-03 | FAIL | 0.061426 | 0.070851 | 0.000000 | 0.000000 | 0.196421 | 0.021721 | 0.979722 | P0 | 0 | 80.603280 | 80.068130 | 146 |
| MOT20-05 | FAIL | 0.057894 | 0.065735 | 0.000000 | 0.000000 | 0.184383 | 0.023363 | 0.978139 | P0 | 0 | 79.770327 | 77.685820 | 479 |

Every outer failed multiple representation criteria. Therefore P1/P2 were not evaluated, no inner exact gate was opened, and each outer froze P0 with the byte-exact M23-46 tracker.

## 6. Exact TrackEval accounting

- Inner TrackEval runs: **0**.
- Held-outer TrackEval runs: **0**.
- New combined TrackEval runs: **0**.

Because all four frozen trackers are byte-exact M23-46, the preregistered all-P0 branch reuses the already official M23-46 per-fold and combined metrics rather than rerunning TrackEval.

## 7. Deployable MOT20 result

| Scope | HOTA | DetA | AssA | IDSW |
|---|---:|---:|---:|---:|
| MOT20-01 | 78.805125 | 81.837410 | 76.043165 | 46 |
| MOT20-02 | 73.098150 | 80.584820 | 66.407293 | 325 |
| MOT20-03 | 80.603280 | 81.174386 | 80.068130 | 146 |
| MOT20-05 | 79.770327 | 81.954810 | 77.685820 | 479 |
| COMBINED | **79.123193** | **81.543470** | **76.825150** | **996** |

Delta versus M23-46 is exactly zero for all reported deployable metrics because the selected output is byte-exact M23-46.

## 8. Teacher-only capacity reference

M23-57 v2 reports teacher-capacity HOTA **81.148046**, which is 2.024853 above the deployable M23-46/M23-59 result. This remains a non-deployable teacher-only reference and must not be reported as an M23-59 deployable gain.

## 9. Resource record

- External pretraining runtime: `588.984` seconds; peak GPU memory `807.784` MiB; peak RSS `1874.629` MiB.
- External feature extraction runtime: `373.212` seconds; max peak GPU memory `426.382` MiB; max peak RSS `1761.266` MiB.
- Nested MOT20 representation runs: maximum recorded peak GPU memory `2144.947` MiB; maximum recorded peak RSS `6881.539` MiB.

## 10. Scientific conclusion

The external pretrained representation transfers strongly on the fixed MOT17 validation construction, but under strict sequence-LOSO MOT20 evaluation it does not meet the preregistered boundary-separability gate. The dominant failure is very low conditional boundary precision/PR-AUC and near-total catastrophic top-1 cross-link error. Consequently, deploying learned segmentation or flow would be scientifically unsupported. The correct frozen conclusion is P0 for all outers, no strict gain, and no escalation to M23-54, M23-58, or MOT20 test submission.

## 11. Executed commands

```bash
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py run-outer-representation --outer MOT20-02
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py freeze-outer-policy --outer MOT20-02
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py run-outer-representation --outer MOT20-03
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py freeze-outer-policy --outer MOT20-03
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py run-outer-representation --outer MOT20-05
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py freeze-outer-policy --outer MOT20-05
python -u scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py summarize
```

`run-inner-exact` was not executed for any outer because every representation gate failed.

## 12. Primary artifacts

- Queue record: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/summary.csv`.
- Protocol events: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/protocol_events.jsonl`.
- Final machine-readable summary: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/final_summary.json`.
- Strict outer evaluation: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/strict_outer_evaluation/report.json`.
- Outer policy manifests: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/outer_policies/<SEQ>/outer_policy_manifest.json`.
- Representation gates: `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/<SEQ>/representation_gate.json`.
- Central registry: `outputs/experiment_registry.csv`.

## 13. Closure invariants

- Four outer policy manifests exist and select P0.
- Four frozen tracker SHA-256 values match their policy manifests and are marked byte-exact M23-46.
- No inner exact gate was opened.
- No new TrackEval was run.
- `strict_sequence_loso=true`, `deployable=true`, `external_supervision=true`.
- `m23_54_started=false`, `m23_58_started=false`, `mot20_test_submission=false`.
- No v1 checkpoint, label, model, policy, or evaluation artifact was reused.
