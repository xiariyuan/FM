# Current Mainline: SPOT-Track

**Date:** 2026-06-26 (corrected)  
**Status:** Oracle Gate PROVISIONAL → SPOT_PROVISIONAL

## ⚠️ IMPORTANT CORRECTION

Previous version incorrectly stated `runtime_patch_allowed=1`. Oracle ceiling ≠ runtime gain.
Runtime patches require real paired eval to unlock. See `outputs/oracle_gate/decision.md`.

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
| oracle_0a | completed_positive (oracle ceiling, not runtime gain) |
| oracle_0c_inline_gt | completed_partial_trusted |
| oracle_0e | provisional |
| final_decision | SPOT_PROVISIONAL |
| runtime_patch_allowed | 0 (requires real paired eval) |

## What This Means

SPOT-Track = State-Protected Online Tracking under Ambiguous Association

The oracle gate experiments have confirmed:

- **Oracle 0A** (State Protection): 7.29% oracle recoverable rate on MOT20-05. This is an ORACLE CEILING, NOT a runtime improvement. Indicates headroom exists.
- **Oracle 0C** (Cost Reranking): 43.28% of wrong selections fixable by reranking. Moderate signal, partial but trusted.
- **Oracle 0E** (Joint Decision): PROVISIONAL. Runtime patches NOT yet allowed. Requires real paired eval.

## Current Novelty

- **Candidate novelty:** P4 ADG-freeze / State Protection (needs paired eval confirmation)
- **Support module:** PCC (strong support, 43.28% fixable)
- **P5 delayed commitment:** SKIP (not required)

## What NOT to Read as Current Mainline

The following files describe the **old mainline** (2026-03) and should be treated as historical context only:

- `README.md` — updated to SPOT-Track
- `docs/github_reader_guide.md` — updated to SPOT-Track
- `docs/experiment_index.md` — updated to SPOT-Track

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
3. Run MOT20-05 paired eval (baseline vs SPOT)
4. Only if paired eval positive: unlock runtime_patch_allowed=1
5. Only then: expand to DanceTrack/SportsMOT
6. Only after P4 positive: consider PCC as support module
7. P5 remains SKIP

## What NOT to Do

- Do NOT use oracle ceiling as go/kill criterion for runtime patches
- Do NOT unlock runtime_patch_allowed until paired eval confirms positive
- Do NOT go back to `official_bytetrack / post-host one-edit`
- Do NOT implement P5 delayed commitment
- Do NOT train large models
- Do NOT change detector / ReID / TrackEval
- Do NOT open new brainstorming lines
