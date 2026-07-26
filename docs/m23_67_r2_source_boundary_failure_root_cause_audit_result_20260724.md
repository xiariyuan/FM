# M23-67-R2 — Process Identity Repair and Deterministic Reproduction — Result

Final status: **completed**  
Decision: **FAIL_BOUNDARY_POPULATION_SHIFT**  
Measurement integrity: **PASS_BOUNDARY_IMPLEMENTATION**  
Boundary label/mapping: **PASS**  
Training population alignment: **FAIL**  
Score collapse status: **PASS**  
Source capacity status: **FAIL**  
Scientific primary failure: **boundary_population_mismatch**  
Overall primary classification: **FAIL_BOUNDARY_POPULATION_SHIFT**

## Mapping and label semantics

Every M23-66 validation boundary score was joined by explicit semantic row_index to frozen observable rows and R63 supervision. The score_to_source_row tables matched observable rows exactly. Endpoints remained on the same source track with forward time. line_index was not substituted for row_index. Only matched endpoints entered binary metrics; unknown, distractor, ambiguous, or otherwise non-matched endpoints were excluded and never became negatives. Repeated observations of one physical transition had consistent labels.

The primary corrected physical-transition score remained sigmoid(arithmetic mean finite boundary_logit). Arithmetic mean probability was reported only as sensitivity; no result-based aggregation selection occurred.

## Training versus full-audit population

Train positive rate: `0.007685874970475771`. Validation positive rate: `0.0042809941100416605`. Full MOT17-11/13 audit positive rate: `0.0011678293739819908`.
Train-to-audit positive-rate ratio: `6.581333833271345`. Validation-to-audit ratio: `3.665770193332779`. Train-to-audit unique-transition ratio: `4.633676938260774`.
The preregistered severe population mismatch decision was `True`.

## Score properties

Score orientation anomaly: `False`. Reversed score used: `false`.
Score collapse: `False`. Corrected pooled source ROC-AUC: `0.7083722228322537`; logit standard deviation: `3.7564428706653428`; exact tie rate: `0.0003289834411668302`; distinct finite score count: `18232`.

## Corrected source capacity

MOT17-11/13 macro PR-AUC: `0.005101556776166486`; macro precision@actual: `0.029411764705882353`; macro recall@95P: `0.0`; minimum sequence precision@actual: `0.0`.
The unchanged reference was PR-AUC 0.283, precision@actual 0.35, recall@95P 0.05, and every-sequence precision@actual 0.20.

## Scope and authorization

No training, optimizer step, checkpoint output or modification, tracker, TrackEval, HOTA, raw MOT17/MOT20 GT read, MOT20 test read/submission, teacher, held outer, M23-54, M23-58, threshold search, calibration, temperature scaling, score reversal repair, or policy run occurred. M23-68 was not started and is not authorized by M23-67-R2.

- `post_hoc_diagnostic_only=true`
- `uses_frozen_gt_derived_label_sidecars=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- `training_runs=0`
- `tracker_outputs=0`
- `trackeval_runs=0`
- `hota_evaluations=0`
- `hota=null`
- `next_policy_authorized=false`

未执行Notion写回。
