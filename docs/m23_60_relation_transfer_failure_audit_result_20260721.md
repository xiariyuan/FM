# M23-60 Relation Transfer Failure Audit — Result (2026-07-21)

## Scope and validity

This is an independent **post-hoc diagnostic** of frozen M23-59 v2 artifacts.

- `uses_mot20_gt=true`
- `post_hoc_diagnostic_only=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- M23-59 modified: **no**
- training runs: **0**
- TrackEval runs: **0**
- tracker outputs: **0**
- MOT20 test submission: **no**

## Primary diagnosis

**implementation_or_semantic_mismatch**

The preregistered priority rule selected this category. Threshold evidence is stored in `final_diagnosis.json`; secondary candidate, domain-shift, and ranking evidence does not override the unique primary label.

## Semantic audit

Critical semantic checks passed: **false**.

Critical failures: `['feature_143_physical_semantics_match']`.

The key mismatch is feature index 143: MOT17 uses clipped nearest-neighbor distance, while MOT20 overwrites the same column with a binary GT-free appearance-mapping indicator. Raw successor traces for one MOT17 relation and one MOT20 relation preserve source/destination IDs, frames, candidate index and label through sorting, padding, batching and candidate reversal.

## Candidate oracle

| Sequence | Successor queries | Candidate present | Oracle upper bound | Coverage@1 | Coverage@256 |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 710 | 708 | 0.997183 | 0.670423 | 0.997183 |
| MOT20-02 | 5729 | 5622 | 0.981323 | 0.620702 | 0.979403 |
| MOT20-03 | 10122 | 10114 | 0.999210 | 0.812784 | 0.999012 |
| MOT20-05 | 21353 | 21292 | 0.997143 | 0.725940 | 0.996909 |

Pooled MOT20 candidate coverage@256 / oracle upper bound: **0.994092**.

## Largest MOT17→MOT20 row-feature shifts

| Domain | Feature | KS | Wasserstein | PSI |
|---|---|---:|---:|---:|
| MOT20-03 | geometry_15_nearest_neighbor_distance_or_mapped_indicator | 0.995305 | 0.934652 | 11.556918 |
| MOT20-05 | geometry_15_nearest_neighbor_distance_or_mapped_indicator | 0.993890 | 0.933252 | 11.523338 |
| MOT20-01 | geometry_15_nearest_neighbor_distance_or_mapped_indicator | 0.984978 | 0.924444 | 11.393567 |
| MOT20-02 | geometry_15_nearest_neighbor_distance_or_mapped_indicator | 0.974542 | 0.914153 | 11.301290 |
| MOT20-01 | geometry_06_visibility | 0.705884 | 0.319809 | 8.898696 |
| MOT20-02 | geometry_06_visibility | 0.705884 | 0.319809 | 8.898696 |
| MOT20-03 | geometry_06_visibility | 0.705884 | 0.319809 | 8.898696 |
| MOT20-05 | geometry_06_visibility | 0.705884 | 0.319809 | 8.898696 |
| MOT20-01 | geometry_14_crowd_density_over_100_clipped | 0.672451 | 0.216980 | 7.969427 |
| MOT20-01 | geometry_01_center_y_norm | 0.642306 | 0.297905 | 6.850998 |
| MOT20-02 | geometry_14_crowd_density_over_100_clipped | 0.560329 | 0.181922 | 5.147435 |
| MOT20-03 | geometry_03_box_height_norm | 0.567658 | 0.094341 | 4.873941 |
| MOT20-05 | geometry_14_crowd_density_over_100_clipped | 0.581643 | 0.156669 | 4.255928 |
| MOT20-03 | geometry_04_log_aspect | 0.693289 | 0.306193 | 3.503338 |
| MOT20-05 | geometry_03_box_height_norm | 0.341406 | 0.073636 | 3.037852 |
| MOT20-01 | geometry_04_log_aspect | 0.549837 | 0.204828 | 2.914401 |
| MOT20-05 | geometry_02_box_width_norm | 0.483237 | 0.019518 | 2.358326 |
| MOT20-05 | geometry_04_log_aspect | 0.608992 | 0.259455 | 2.348078 |
| MOT20-03 | geometry_02_box_width_norm | 0.376442 | 0.018450 | 2.330548 |
| MOT20-03 | geometry_08_velocity_y_height_frame | 0.617070 | 0.009720 | 2.243573 |

## Frozen model and fixed baselines

| Domain | Frozen model/fold | Method | R@1 | MRR | R@256 |
|---|---|---|---:|---:|---:|
| MOT17-validation | external_frozen | frozen_model | 0.778217 | 0.889109 | 1.000000 |
| MOT17-validation | external_frozen | frozen_model | 0.805706 | 0.902853 | 1.000000 |
| MOT17-validation | external_frozen | frozen_model | 0.986816 | 0.993408 | 1.000000 |
| MOT20-01 | none | original_candidate_order | 0.622881 | 0.723199 | 1.000000 |
| MOT20-01 | none | minimum_gap | 0.543785 | 0.709374 | 1.000000 |
| MOT20-01 | none | minimum_displacement | 0.860169 | 0.902606 | 1.000000 |
| MOT20-01 | none | minimum_velocity_residual | 0.864407 | 0.909632 | 1.000000 |
| MOT20-01 | none | maximum_IoU | 0.872881 | 0.906381 | 1.000000 |
| MOT20-01 | none | maximum_detector_score | 0.021186 | 0.071661 | 0.981638 |
| MOT20-01 | none | fixed_seed_random | 0.011299 | 0.050429 | 0.964689 |
| MOT20-01 | none | oracle | 1.000000 | 1.000000 | 1.000000 |
| MOT20-01 | outer_MOT20-02_valid_MOT20-01 | frozen_model | 0.024011 | 0.073846 | 1.000000 |
| MOT20-01 | outer_MOT20-03_valid_MOT20-01 | frozen_model | 0.024011 | 0.073846 | 1.000000 |
| MOT20-01 | outer_MOT20-05_valid_MOT20-01 | frozen_model | 0.024011 | 0.073846 | 1.000000 |
| MOT20-02 | none | original_candidate_order | 0.506403 | 0.618745 | 1.000000 |
| MOT20-02 | none | minimum_gap | 0.817681 | 0.881668 | 1.000000 |
| MOT20-02 | none | minimum_displacement | 0.839203 | 0.887742 | 1.000000 |
| MOT20-02 | none | minimum_velocity_residual | 0.846674 | 0.887212 | 1.000000 |
| MOT20-02 | none | maximum_IoU | 0.854322 | 0.894248 | 1.000000 |
| MOT20-02 | none | maximum_detector_score | 0.004803 | 0.032104 | 0.984525 |
| MOT20-02 | none | fixed_seed_random | 0.005692 | 0.033812 | 0.978833 |
| MOT20-02 | none | oracle | 1.000000 | 1.000000 | 1.000000 |
| MOT20-02 | outer_MOT20-01_valid_MOT20-02 | frozen_model | 0.028282 | 0.065758 | 0.999466 |
| MOT20-02 | outer_MOT20-03_valid_MOT20-02 | frozen_model | 0.028282 | 0.065758 | 0.999466 |
| MOT20-02 | outer_MOT20-05_valid_MOT20-02 | frozen_model | 0.028282 | 0.065758 | 0.999466 |
| MOT20-03 | none | original_candidate_order | 0.607475 | 0.735028 | 1.000000 |
| MOT20-03 | none | minimum_gap | 0.868697 | 0.919561 | 1.000000 |
| MOT20-03 | none | minimum_displacement | 0.957584 | 0.971808 | 1.000000 |
| MOT20-03 | none | minimum_velocity_residual | 0.956891 | 0.970873 | 1.000000 |
| MOT20-03 | none | maximum_IoU | 0.963417 | 0.975177 | 1.000000 |
| MOT20-03 | none | maximum_detector_score | 0.008206 | 0.042930 | 0.990706 |
| MOT20-03 | none | fixed_seed_random | 0.005932 | 0.034348 | 0.988432 |
| MOT20-03 | none | oracle | 1.000000 | 1.000000 | 1.000000 |
| MOT20-03 | outer_MOT20-01_valid_MOT20-03 | frozen_model | 0.017797 | 0.055215 | 0.999209 |
| MOT20-03 | outer_MOT20-02_valid_MOT20-03 | frozen_model | 0.017797 | 0.055215 | 0.999209 |
| MOT20-03 | outer_MOT20-05_valid_MOT20-03 | frozen_model | 0.017797 | 0.055215 | 0.999209 |
| MOT20-05 | none | original_candidate_order | 0.488728 | 0.637327 | 1.000000 |
| MOT20-05 | none | minimum_gap | 0.844308 | 0.905133 | 1.000000 |
| MOT20-05 | none | minimum_displacement | 0.943876 | 0.962561 | 1.000000 |
| MOT20-05 | none | minimum_velocity_residual | 0.942561 | 0.961356 | 1.000000 |
| MOT20-05 | none | maximum_IoU | 0.952189 | 0.966851 | 0.999953 |
| MOT20-05 | none | maximum_detector_score | 0.006200 | 0.035581 | 0.978865 |
| MOT20-05 | none | fixed_seed_random | 0.005213 | 0.031884 | 0.979899 |
| MOT20-05 | none | oracle | 1.000000 | 1.000000 | 1.000000 |
| MOT20-05 | outer_MOT20-01_valid_MOT20-05 | frozen_model | 0.012869 | 0.041775 | 0.996008 |
| MOT20-05 | outer_MOT20-02_valid_MOT20-05 | frozen_model | 0.012869 | 0.041775 | 0.996008 |
| MOT20-05 | outer_MOT20-03_valid_MOT20-05 | frozen_model | 0.012869 | 0.041775 | 0.996008 |

## Error waterfall

```json
{
  "all_successor_query_instances": 113742,
  "fixed_geometry_baseline_recoverable": 106830,
  "frozen_model_top1_correct": 1890,
  "frozen_model_top1_wrong": 111318,
  "gt_successor_in_candidate_set": 113208,
  "gt_successor_not_in_candidate_set": 534,
  "out_of_support_explainable": 18387,
  "score_direction_anomaly_explainable": 0
}
```

## Allowed next research direction

Create a new preregistered M23-59 v3 semantic-alignment validation that restores one physical meaning per feature index (especially geometry index 15), regenerates all affected MOT20 observables from raw frozen inputs, and reruns from an empty versioned root; do not tune candidates, thresholds, policy risks, or HOTA.

This direction requires a new versioned protocol and cannot reuse M23-59 gate results as a tuned decision rule.

## Structured records

- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/audit_manifest.json`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/semantic_validation.json`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/candidate_oracle.json`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/feature_shift.csv`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/ranking_diagnostics.csv`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/error_waterfall.json`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/final_diagnosis.json`
- `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/summary.csv`
- `outputs/experiment_registry.csv`
