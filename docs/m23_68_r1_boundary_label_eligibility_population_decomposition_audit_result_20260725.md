# M23-68-R1 Boundary Label-Eligibility and Population Decomposition Audit Repair — Result

Status: completed / closed
Decision: COMPLETED_POST_HOC_DIAGNOSTIC
Overall primary classification: cross_protocol_label_eligibility_mismatch

## Measurement integrity

- measurement_integrity_decision: FAIL_MIXED_LABEL_ELIGIBILITY_DEFINITIONS
- mixed_label_eligibility_definitions: true
- M23-67-R2 population failure stability: UNSTABLE_FAIL
- original mixed ratio: 6.58133383327135
- reconstructed mixed ratio: 6.58133383327135
- native/native raw ratio: 1.79534817682825
- strict/strict raw ratio: 1.31501776658343
- strict/strict unique ratio: 1.24596759623115

The frozen M23-63 boundary tensor follows its native matched-endpoint definition exactly. The later M23-66/M23-67 audit excludes ambiguity/tie endpoints. M23-67-R2 compared a native-label train numerator to a strict-label audit denominator; M23-68 reports this as a cross-protocol estimand mismatch rather than an M23-63 tensor implementation defect.

## Harmonized population decomposition

- scientific component: no_material_population_component
- sequence composition material: false
- strict unique train/validation risk ratio: 1.24596759623115
- absolute rate difference: 0.000323677064894598
- Fisher exact p: 0.349847908066786
- train sequences in common direction: 4
- optimizer-adapter sampling material: false
- optimizer/full risk ratio: 1.00002710173993
- excluded train-window fraction: 0.00211939244083363
- duplicate weighting material: false

## Score capacity

- strict validation macro PR-AUC: 0.00510155677616649
- strict validation macro precision@actual: 0.0294117647058824
- strict validation macro recall@95P: 0
- strict validation reference passed: false
- native validation macro PR-AUC: 0.00824687700846272
- native validation macro precision@actual: 0.0072463768115942
- native validation macro recall@95P: 0
- native validation sensitivity passed: false

## Scope

No training, optimizer step, checkpoint output/modification, new model inference, tracker, TrackEval, HOTA, raw GT, MOT20 test, teacher, held-outer, threshold search, calibration, score reversal, policy, M23-54, or M23-58 run occurred. HOTA is null. No next policy is authorized.

Notion writeback was not executed.
