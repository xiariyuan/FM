# FM-Track M23-3-1 sync payload

Status: completed, deployable-form oracle ceiling failed, fail-closed

## Protocol

Starting from the exact raw MOT20 baseline, only same-frame/same-ID box and score replacement was allowed. Row count, tracker IDs, and tail columns were frozen. No additions, deletions, suppressor routing, spawn, or identity relabeling were permitted. Within each frame/track group the train-GT oracle selected the maximum strictly positive IoU improvement.

## Results

Baseline HOTA 77.699020. M23-selected replacement oracle 77.987070 (+0.288050). Dense replacement oracle 78.001330 (+0.302310). All four sequences improve, but the preregistered 78.2 selected and 78.5 dense gates both fail.

There are 174,497 dense positive groups, but 164,268 (94.14%) improve target IoU by at most 0.05. Dense replacement reduces FP by 448, FN by 478, and IDSW by 2. This confirms that same-ID box refinement is not the main source of the M23-1 ceiling; the missing value is observation recovery and identity reattachment.

## Decision

Close the direct learned segment-replacement graph. Appearance remains auxiliary only. Next stage is M23-4-0: a fixed 30-frame inter-segment gap-bridge addition oracle from the raw baseline, inheriting an existing tracker ID only when absent in the frame, combined with dense replacement; no new ID, arbitrary spawn, or identity relabeling.

## Audit

The first complete run exposed only a combined-summary precision issue: TrackEval summary values are rounded to three decimals although all per-sequence detailed metrics and tracker hashes were exact. v2 changed only the combined metric source to the detailed COMBINED row; candidates, trackers, gates, and actions did not change.

Formal/reproduction compact outputs 6/6, trackers 12/12, and TrackEval files 6/6 byte-identical. Unified audit 90/90 passed. P15 remains no-op with zero locked-label reads, zero locked TrackEval calls, and 156 rows untouched.

Notion write has not been performed because no writable Notion connector is available in this session.
