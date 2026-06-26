# SPOT Implementation Plan

**Date:** 2026-06-26  
**Status:** Oracle Gate PROVISIONAL → SPOT_PROVISIONAL  
**Current rule:** `runtime_patch_allowed=0` until real paired eval is positive.

## 0. Current State

Authoritative files currently agree on the corrected state:

- `outputs/oracle_gate/decision.md` — canonical decision: `SPOT_PROVISIONAL`, `runtime_patch_allowed=0`.
- `outputs/oracle_gate/summary.csv` — structured status: provisional, paired eval required.
- `docs/current_mainline_2026-06-26.md` — authority order and corrected mainline summary.
- `docs/oracle_gate_recap_2026-06-25.md` — recap: oracle ceiling is not runtime gain.

Documentation correction status:

- `docs/experiment_index.md` — corrected in commit `277884a068eb2dcfc8022185353f7213bbf1e06c`.
- `docs/github_reader_guide.md` — corrected in commit `277884a068eb2dcfc8022185353f7213bbf1e06c`.
- Remaining documentation work is consistency/navigation only: old ByteTrack/post-host content must be clearly marked historical.

## 1. Non-Negotiable Constraints

Do not unlock or claim runtime gain from oracle numbers.

- Oracle 0A `7.29%` = oracle recoverable ceiling, not runtime IDSW reduction.
- Oracle 0C `43.28%` = partial inline-GT fixable rate, not runtime gain.
- Real paired eval is the only go/kill criterion for `runtime_patch_allowed=1`.

SPOT v0 scope is deliberately narrow:

- freeze appearance/history only via existing `update_features(mode="freeze")` path.
- no KF change.
- no Hungarian change.
- no lifecycle change.
- no detector / ReID / TrackEval change.
- no PCC runtime integration.
- no P5 delayed commitment.
- no training or learned ADG.

## 2. P0 Documentation Cleanup

Status: mostly complete.

Remaining checks:

1. Grep cold-start docs for stale phrases:
   - `CLOSED → SPOT_MAINLINE`
   - `Runtime patch allowed: YES`
   - `7.29% IDSW reduction`
2. Keep current SPOT reading order first.
3. Mark ByteTrack/post-host sections as historical context, not current runtime guidance.

## 3. P1 Paired-Eval Harness Strategy

Do not start with a complex dual-tracker shared-detection harness unless necessary.

First implementation should be conservative:

1. Fix detector/checkpoint/thresholds/seeds/config in `run_manifest.json`.
2. Run baseline once without any SPOT flags.
3. Run observe-only once with `--spot-enable` and without `--spot-freeze-app`.
4. Compare result files and metrics for exact parity.
5. Run `--spot-enable --spot-freeze-app` only after observe-only parity passes.

The older ByteTrack paired-eval scripts remain useful references, not the first required implementation target:

- `scripts/run_official_bytetrack_shared_detection_pair_core.py`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/eval_botsort_halfval_trackeval.py`

Target output layout:

```text
outputs/spot_runtime/<run_name>/
├── 00_baseline/
│   ├── track_results/*.txt
│   └── summary.csv
├── 01_spot_observe/
│   ├── track_results/*.txt
│   ├── summary.csv
│   └── spot_debug/
├── 02_spot_freeze_app/
│   ├── track_results/*.txt
│   ├── summary.csv
│   └── spot_debug/
├── observe_parity_report.json
├── freeze_diff_report.json
├── parity_report.md
└── run_manifest.json
```

## 4. P2 Empty SPOT Switch Patch

First code commit should add a no-op SPOT switch only.

Add arguments to `external/BoT-SORT-main/tools/track.py`:

```python
--spot-enable
--spot-debug-dir
--spot-margin-thresh
--spot-freeze-app
```

Initialize fields in `external/BoT-SORT-main/tracker/bot_sort.py`, but do not change tracker behavior yet.

Hard requirement:

- baseline and observe-only (`--spot-enable` without `--spot-freeze-app`) must be identical.
- If the observe-only switch changes results, stop and debug before interpreting freeze logic.

## 5. P3 Minimal `freeze_app_history_only` Runtime Patch

SPOT v0 trigger is margin-only:

```text
row_margin = second_best_cost_for_track - chosen_cost
col_margin = second_best_cost_for_detection - chosen_cost
cost_margin = min(row_margin, col_margin)
spot_triggered = cost_margin < spot_margin_thresh
```

Entropy and density triggers are not part of v0. Keep them out until a separate paired-eval-backed design justifies them.

Do not use detection confidence as local density.

- `det.score` is not local density.
- SPOT v0 should not enable a density trigger.
- If density is added later, compute it from neighborhood/cost competition, e.g. nearby detection count, IoU-neighbor count, or row/column cost competition.

Cost margin should consider ambiguity carefully:

- At minimum, compute row-side margin for the matched track against detections.
- Prefer also checking column-side margin for the matched detection against tracks.
- Use the more ambiguous side, e.g. `min(row_margin, col_margin)`, if both are available.

Integration point:

- after association returns `matches` and while the cost matrix is still available or recoverable.
- before `track.update(det, frame_id)` / `track.re_activate(...)`.
- set on the matched detection/track object:

```python
det.tcgau_update_mode = "freeze"
det.tcgau_append_history = False
det.tcgau_alpha_override = None
```

Initialize SPOT stats explicitly, e.g.:

```python
self.spot_stats = {
    "enabled": bool(self.spot_enable),
    "freeze_count": 0,
    "matched_pairs": 0,
}
```

## 6. Required `spot_debug.csv`

Each matched pair should record at least:

```text
seq_name
frame_id
track_id
det_id
det_score
cost_top1
cost_top2
cost_margin
row_margin
col_margin
spot_triggered
spot_reason
spot_action
update_mode
append_history
track_age
lost_age
```

No debug output means no interpretable paired eval.

## 7. Parity Test

Before any positive claim:

1. Run baseline.
2. Run `spot_enable=0`.
3. Compare result txt files exactly.
4. Compare `summary.csv` exactly for HOTA, AssA, IDF1, MOTA, IDSW.
5. Record command/config/commit in `run_manifest.json`.

If parity fails, stop.

## 8. MOT20-05 Paired Eval

Only after parity passes:

1. Run baseline.
2. Run `--spot-enable --spot-freeze-app`.
3. Produce:
   - `summary.csv`
   - `result.csv`
   - `spot_debug.csv`
   - `report.md`
   - `run_manifest.json`

GO conditions:

- IDSW decreases.
- HOTA does not decrease.
- AssA / IDF1 do not decrease, preferably increase.
- MOTA does not materially decrease.
- freeze rate is reasonable.
- debug shows triggers concentrated on ambiguous cases.

NO-GO conditions:

- `spot_enable=0` parity fails.
- IDSW does not decrease or increases.
- HOTA decreases.
- AssA / IDF1 decrease materially.
- freeze rate is too high.
- debug shows easy cases being frozen.

## 9. Deferred Work

Do not do these before P4 paired eval is positive:

- PCC runtime module.
- P5 delayed commitment.
- learned ADG / large model training.
- DanceTrack / SportsMOT runtime expansion.

0B evidence-latency semantics still need correction before using 0B to make a strong P5 decision.
