# Current Mainline: SPOT-Track

**Date:** 2026-06-26  
**Status:** Oracle Gate CLOSED → SPOT_MAINLINE

## Authority Order

When reading this repository, trust these files in this order:

1. `outputs/oracle_gate/decision.md` — current canonical decision
2. `outputs/oracle_gate/summary.csv` — structured status
3. `docs/oracle_gate_recap_2026-06-25.md` — detailed recap
4. `docs/current_mainline_2026-06-26.md` — this file
5. Older README / experiment_index only as **historical context**

## Current Decision

| Item | Status |
|------|--------|
| protocol_lock | completed |
| gt_alignment | completed_verified |
| oracle_0a | completed_positive |
| oracle_0c_inline_gt | completed_partial_trusted |
| oracle_0e | closed |
| final_decision | SPOT_MAINLINE |
| runtime_patch_allowed | 1 |

## What This Means

SPOT-Track = State-Protected Online Tracking under Ambiguous Association

The oracle gate experiments have confirmed:

- **Oracle 0A** (State Protection): 7.29% IDSW reduction on MOT20-05. Above 3% threshold.
- **Oracle 0C** (Cost Reranking): 43.28% of wrong selections fixable by reranking. Moderate signal, partial but trusted.
- **Oracle 0E** (Joint Decision): CLOSED with SPOT_MAINLINE. Runtime patches allowed.

## Current Novelty

- **Main novelty:** P4 ADG-freeze / State Protection
- **Support module:** PCC (strong support, 43.28% fixable)
- **P5 delayed commitment:** SKIP (not required)

## What NOT to Read as Current Mainline

The following files describe the **old mainline** (2026-03) and should be treated as historical context only:

- `README.md` — still describes `official_bytetrack / post-host one-edit`
- `docs/github_reader_guide.md` — still describes `official_bytetrack post-host one-edit`
- `docs/experiment_index.md` — still describes `official_bytetrack post-host one-edit`

These are NOT the current mainline. The current mainline is SPOT-Track.

## What to Read for SPOT-Track

### Oracle Evidence

- `scripts/spot_oracle/run_oracle_state_protection.py`
- `scripts/spot_oracle/run_oracle_cost_rerank_inline.py`
- `scripts/spot_oracle/run_joint_oracle.py`
- `scripts/spot_p0/build_gt_alignment.py`
- `scripts/spot_common/mot_format.py`

### Runtime Implementation (next step)

- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`

### Decision Documents

- `outputs/oracle_gate/decision.md`
- `outputs/oracle_gate/summary.csv`
- `docs/oracle_gate_recap_2026-06-25.md`

## Next Steps

1. Implement minimal P4 ADG-freeze runtime patch (appearance/history freeze only, no KF change)
2. Run smoke: spot_enable=0 must equal baseline
3. Run MOT20-05 paired eval
4. Only if positive: MOT20 half-val, MOT17 transfer
5. Only after P4 positive: consider PCC as support module
6. P5 remains SKIP

## What NOT to Do

- Do not go back to `official_bytetrack / post-host one-edit`
- Do not implement P5 delayed commitment
- Do not train large models
- Do not change detector / ReID / TrackEval
- Do not open new brainstorming lines
