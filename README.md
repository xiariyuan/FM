# FM-Track Research Repository

This repository is the working research repository for SPOT-Track: State-Protected Online Tracking under Ambiguous Association.

If you are a human reviewer or a GitHub-connected GPT, **start from the current mainline docs below**, not the old training entrypoints.

## Start Here (2026-06-26)

**Current canonical status:**

1. `docs/current_mainline_2026-06-26.md` — current mainline overview
2. `outputs/oracle_gate/decision.md` — current decision
3. `outputs/oracle_gate/summary.csv` — structured status
4. `docs/oracle_gate_recap_2026-06-25.md` — detailed recap

**Current code entrypoints:**

- `scripts/spot_oracle/` — oracle experiments
- `scripts/spot_p0/` — GT alignment
- `scripts/spot_common/` — shared utilities
- `external/BoT-SORT-main/tracker/bot_sort.py` — runtime tracker
- `external/BoT-SORT-main/tools/track.py` — runtime entry

## Current Research Status

As of `2026-06-26`, the repository-wide state is:

- **Current mainline:** SPOT-Track (State-Protected Online Tracking)
- **Oracle Gate:** CLOSED → SPOT_MAINLINE
- **Runtime patch allowed:** YES
- **Main novelty:** P4 ADG-freeze / State Protection
- **Support module:** PCC (strong support)
- **P5 delayed commitment:** SKIP

Oracle evidence:
- Oracle 0A: 7.29% IDSW reduction on MOT20-05 (positive)
- Oracle 0C: 43.28% fixable by reranking (moderate, partial but trusted)
- Oracle 0E: CLOSED with SPOT_MAINLINE

## Historical Context (2026-03)

The following files describe the **old mainline** (2026-03) and should be treated as historical context only:

- `docs/github_reader_guide.md` — old `official_bytetrack / post-host one-edit`
- `docs/experiment_index.md` — old experiment history
- `outputs/experiment_registry.csv` — old experiment records

These are NOT the current mainline. The current mainline is SPOT-Track.

## Important Code Paths

There are two different kinds of code in this repository:

1. **Current mainline:** SPOT-Track oracle and runtime code
2. **Historical:** old `official_bytetrack / post-host one-edit` lines

If you want the **current mainline**, start here:

- `scripts/spot_oracle/run_oracle_state_protection.py`
- `scripts/spot_oracle/run_oracle_cost_rerank_inline.py`
- `scripts/spot_oracle/run_joint_oracle.py`
- `scripts/spot_p0/build_gt_alignment.py`
- `scripts/spot_common/mot_format.py`
- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`

If you want the **historical official ByteTrack lines**, start here:

- `docs/github_reader_guide.md`
- `docs/experiment_index.md`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_shared_detection_pair_core.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`

## Experiment Records

This repository intentionally stores lightweight structured experiment evidence in git.

Typical tracked artifacts include:

- `summary.csv`
- `result.csv`
- `metrics.jsonl`
- `report.md`
- `outputs/experiment_registry.csv`

Heavy artifacts are intentionally excluded from git:

- checkpoints
- dataset dumps
- raw logs
- large runtime shards
- packaged archives

This is deliberate. The goal is to keep the repository readable by both humans and GitHub-connected GPT systems while still preserving experiment traceability.

## Environment and Reproduction Notes

This repository includes multiple tracker families and historical branches, so there is no single one-line reproduction command that explains the whole project.

For current SPOT-Track work, the most useful reproduction-related files are:

- `scripts/spot_oracle/`
- `scripts/spot_p0/`
- `scripts/spot_common/`
- `external/BoT-SORT-main/tracker/bot_sort.py`
- `external/BoT-SORT-main/tools/track.py`

Weights and datasets are expected to exist locally and are **not** fully vendored in git.

## Legacy / Broad Training Entry Points

Older broad entrypoints remain in the repo for historical completeness:

- `train.py`
- `train_bytetrack.py`
- `submit_bytetrack.py`
- `submit_public.py`

They should not be treated as the main explanation of the current research state.
