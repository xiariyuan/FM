# AssocRiskBench P15 — Directional Sequence Abstention V1

Date: 2026-07-16

## Goal

Add a sequence-level fail-closed decision before directional handoff candidate execution. The gate must be learned and audited only from frozen ranks 21–100, and must not consume any additional locked rank1–20 utility or TrackEval labels.

## Why a direct sequence classifier was rejected

MOT20 train contains only four sequences. Training a conventional sequence classifier on four samples would be statistically indefensible and would make the result highly sensitive to sequence identity. Instead, each held-out sequence's ranks 21–100 were divided into four contiguous rank windows:

- ranks 21–40
- ranks 41–60
- ranks 61–80
- ranks 81–100

Each window contains 20 base proposals and 40 directional candidates, matching the size of a locked top-20 deployment unit. Model, feature variant, candidate policy, and cap were still selected through nested leave-one-sequence-out calibration on the other three sequences.

The formal audit reuses the exact random-seed schedule of the frozen locked directional v2 trainer:

- inner models: `23000 + 1000 * outer_index + 100 * variant_index + inner_index`
- outer model: `29000 + outer_index`

An earlier pre-commit diagnostic run used different audit-only seeds and was discarded before Git commit. It is not part of the formal evidence.

## Preregistered abstention family

The audit evaluated a fixed, interpretable family before reading the outer OOF utilities:

- base policy
- any positive regression estimate
- positive median estimate
- positive expected estimate
- positive expected and median consensus
- positive q25 estimate
- strict classifier confidence
- hybrid regression/classifier safety
- unique active signal
- at most two active signals
- selection-score gap at least 0.10
- sparse signal plus any positive regression estimate

A gate was eligible only when it selected at least four pseudo-deployment windows, achieved at least 75% positive precision, had positive total utility, had nonnegative utility on every sequence, and negative mass no greater than 25% of positive mass.

## Train-only OOF result

There were 16 pseudo-deployment units. The exact frozen-v2 model family fired four times:

| Sequence | Window | Candidate | Direction | Delta HOTA |
|---|---:|---:|---|---:|
| MOT20-02 | 61–80 | rank 73 | v_to_u | -0.009540 |
| MOT20-03 | 21–40 | rank 27 | v_to_u | +0.000055 |
| MOT20-03 | 81–100 | rank 94 | u_to_v | -0.000100 |
| MOT20-05 | 81–100 | rank 96 | v_to_u | -0.002264 |

Three of the four selected candidates were negative despite positive median utility predictions and high positive-class probabilities. The only positive candidate had a negligible gain. The base policy therefore had:

- selected windows: 4 / 16
- positive windows: 1
- negative windows: 3
- positive precision: 25%
- positive mass: +0.000055 HOTA
- negative mass: -0.011904 HOTA
- total utility: -0.011849 HOTA
- worst-sequence utility: -0.009540 HOTA

No preregistered abstention gate was eligible. The q25-positive rule avoided all negative selections, but selected zero windows and therefore provided no evidence of useful coverage.

## Frozen decision

The train-only audit selected `no_op`. A fail-closed freezer was applied to the already-frozen locked predictions:

- previous locked selections: 4
- selections after sequence abstention: 0
- suppressed sequences: MOT20-01, MOT20-02, MOT20-03, MOT20-05
- additional locked labels consumed: 0
- remaining unread locked directional labels: 156

Frozen locked prediction SHA-256:

`e7f3a4fb9ffe4a67572f69ab25a04e3b70a9d17adcbbbf2c1ce853f283ebd8c8`

## Reproducibility verification

The complete normalized audit was executed twice from the same frozen inputs. The following files were byte-identical across both independent runs:

- `report.json`
- `abstention_gate_summary.csv`
- `fold_policy_selection.csv`
- `window_oof_units.csv`
- `inner_calibration.csv`

Formal audit report SHA-256:

`433fc5d99136b80025819ccf7489748245354679d1f80cf4318f3dd8d3d7e7e9`

Formal window OOF unit SHA-256:

`16c20ebbbc5d0fdabfa7b9ffd7a6da02a491a9dea6d8a58835d50ade11d3c085`

## Interpretation

The failure is not fixed by adding another probability threshold. The two false positives already have apparently favorable classifier confidence and median predictions. The lower-quantile estimate is conservative enough to abstain, but has zero coverage. This indicates that the current directional utility model lacks calibrated lower-tail discrimination under sequence shift.

The directional handoff executor remains mechanically valid, and the earlier locked audit showed that some directional edits can be positive. However, the current candidate utility model cannot safely decide when to execute them. The family must not consume more locked labels until candidate-level modeling is redesigned.

## Next research direction

1. Replace independent point prediction with a window-conditioned risk objective that directly optimizes top-one utility and false-positive cost.
2. Generate outer-OOF candidate predictions for all training events and study positive-event recall, rank calibration, and lower-tail residuals by sequence/window.
3. Evaluate distribution-free abstention based on nonconformity of the selected candidate relative to training-window residuals, rather than raw model probabilities.
4. Keep the remaining 156 locked directional labels unread until a new model and manifest are frozen.

## Reproducibility paths

- Audit script: `scripts/audit_directional_sequence_abstention_windows.py`
- Fail-closed freezer: `scripts/freeze_directional_sequence_abstention.py`
- Train-only audit: `outputs/assocriskbench_p15_20260716/directional_sequence_abstention_window_oof_v3/`
- Independent reproduction: `outputs/assocriskbench_p15_20260716/directional_sequence_abstention_window_oof_v3_repro/`
- Frozen locked no-op: `outputs/assocriskbench_p15_20260716/locked_directional_sequence_abstention_v3/`
- MCP experiment: `assocriskbench_p15_directional_sequence_abstention_v1_20260716`
