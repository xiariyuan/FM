# M23-68 Boundary Label-Eligibility and Population Decomposition Audit — Fail-closed Result

Status: failed / closed

Decision: FAIL_IMPLEMENTATION

The frozen implementation incorrectly required the categorical join-count mapping to have length one. Both splits had exact one-to-one joins, but pandas value_counts retained left_only and right_only keys with zero counts. All row-sequence, tensor-index, R62 feature, mask, padding, and observation-count checks passed. The implementation failed after freeze, so M23-68 was not modified or resumed and no scientific decomposition was claimed.

No training, optimizer step, checkpoint output/modification, model inference, tracker, TrackEval, HOTA, raw GT, MOT20 test, teacher, held-outer, threshold search, calibration, score reversal, or policy run occurred. HOTA is null and no next policy is authorized.

A successor repair must use a new M23-68-R1 preregistration, script SHA, and run root. Notion writeback was not executed.
