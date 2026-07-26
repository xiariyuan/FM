# M23-63 result — supervision join and example construction audit

## Decision
`FAIL_EXAMPLE_VALIDATION`; status=`closed`. This Stage-A experiment performed no training, optimizer step, checkpoint generation, TrackEval, tracker output, MOT20 GT/test read, or M23-54/M23-58 start.

## Frozen inputs
- Contract hash: `90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5`
- prereg SHA: `b1a47ea82e32c3e2df7ae9d409c2db767baeaf935251b428add09de3923b25fd`
- implementation script SHA: `b4257f25f844ac1d1aec979388b097f83b1fd5016a3483f7c71077f2fbb94cbb`
- input manifest SHA: `9b3bd49b12a9ee230ac4332246331bc134f5563fdddca74f0c8c924a32787f53`
- topology manifest SHA: `82974588e0b6ba5c1c8478f0a2f49da8c7925dece5e9abec23e44595990cd2a7`
- candidate pool SHA: `689846d9b311436ba6f88b5d5b09b7caf4ee1e15f5153f5b1e80d8ed8833b2c4`
- labels SHA: `1cccd06c0a6fdbbe3ed1a3d9baca0d23529bff173fb364853e186d4634956c04`
- train examples SHA: `c98e686962d301c883fe54bb48399160802c895acf99d02ee57a10ee1ae076cc`
- validation examples SHA: `b6eccb7145546116b468c3b7b35c3d452fc4ce065f6af96302cb2611d34ff18c`

## Commands
```bash
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py init
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py prepare
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py join
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py build-examples
python -u scripts/m23_research/m23_63_v3_supervision_join_example_audit.py validate-close
```

## Join statistics
| sequence | source rows | matched | unknown | distractor removed | unmatched GT | median IoU | ambiguity | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-02 | 16892 | 16033 | 803 | 56 | 2548 | 0.889680 | 3039 | 0 |
| MOT17-04 | 47292 | 47033 | 212 | 47 | 524 | 0.936387 | 5024 | 0 |
| MOT17-05 | 6652 | 6268 | 379 | 5 | 649 | 0.874046 | 321 | 2 |
| MOT17-09 | 4968 | 4871 | 97 | 0 | 454 | 0.904746 | 912 | 0 |
| MOT17-10 | 11651 | 11191 | 460 | 0 | 1648 | 0.838377 | 875 | 0 |
| MOT17-11 | 9276 | 8952 | 324 | 0 | 484 | 0.909966 | 146 | 0 |
| MOT17-13 | 11116 | 10794 | 322 | 0 | 848 | 0.858361 | 977 | 0 |

## Objective support
Train: `{"boundary_ignored": 7547, "boundary_negative": 155447, "boundary_positive": 1204, "ignored_node": 12, "impure_node": 535, "node_examples": 5662, "paired_examples": 2, "paired_replacement": 2, "pure_node": 5115, "relation_examples": 10240, "successor_ignored": 2117, "successor_negative": 7635, "successor_positive": 488}`

Validation: `{"boundary_ignored": 2344, "boundary_negative": 34656, "boundary_positive": 149, "ignored_node": 5, "impure_node": 102, "node_examples": 1281, "paired_examples": 3, "paired_replacement": 3, "pure_node": 1174, "relation_examples": 4096, "successor_ignored": 554, "successor_negative": 3365, "successor_positive": 177}`

Support checks train: `{"boundary_negative": true, "boundary_positive": true, "impure_node": true, "paired_replacement": false, "pure_node": true, "successor_negative": true, "successor_positive": true}`

Support checks validation: `{"boundary_negative": true, "boundary_positive": true, "impure_node": true, "paired_replacement": true, "pure_node": true, "successor_negative": true, "successor_positive": true}`

## Candidate compatibility
`{"all": {"candidate_recall": 0.9984152139461173, "paired_cross_edge_realizability": 0.13379739350284722, "paired_cross_edge_realizable": 143679, "paired_positive_opportunities": 1073855, "true_successor_candidate_missing": 5, "true_successor_candidate_present": 3150}, "train": {"candidate_recall": 0.9980798771121352, "paired_cross_edge_realizability": 0.12386642494029243, "paired_cross_edge_realizable": 128311, "paired_positive_opportunities": 1035882, "true_successor_candidate_missing": 5, "true_successor_candidate_present": 2599}, "validation": {"candidate_recall": 1.0, "paired_cross_edge_realizability": 0.40470860874832115, "paired_cross_edge_realizable": 15368, "paired_positive_opportunities": 37973, "true_successor_candidate_missing": 0, "true_successor_candidate_present": 551}}`

GT successor candidates that are absent remain `candidate_missing`; no edge was added. Paired examples require both cross edges in the pre-label frozen candidate pool.

## Provenance risk
The relation supervision/validation split is sequence-disjoint, but it uses a historically selected frozen source host. MOT17-11/13 are not claimed unseen for source-host selection. M23-62 feature extraction provenance is recorded separately.

## Gates
- `m23_62_unchanged`: `true`
- `contract_hash_exact`: `true`
- `v2_checkpoint_incompatible_and_not_loaded`: `true`
- `prereg_script_topology_candidates_before_label_unlock`: `true`
- `assignment_one_to_one`: `true`
- `distractor_rule_frozen`: `true`
- `source_row_order_traceable`: `true`
- `feature_labels_physically_separate`: `true`
- `unknown_as_negative_zero`: `true`
- `gt_derived_input_zero`: `true`
- `candidate_topology_unchanged_after_unlock`: `true`
- `split_physical_rows_candidates_examples_disjoint`: `true`
- `all_arrays_finite`: `true`
- `all_masks_prefix_valid`: `true`
- `stable_id_index_mapping`: `true`
- `train_minimum_support`: `false`
- `validation_minimum_support`: `true`
- `candidate_compatibility_reported`: `true`
- `manual_traces_pass`: `true`
- `mot20_gt_reads_zero`: `true`
- `mot20_test_reads_submissions_zero`: `true`
- `teacher_held_outer_reads_zero`: `true`
- `training_runs_zero`: `true`
- `optimizer_steps_zero`: `true`
- `checkpoint_outputs_zero`: `true`
- `trackeval_runs_zero`: `true`
- `tracker_outputs_zero`: `true`
- `m23_54_m23_58_starts_zero`: `true`
- `no_active_relevant_processes`: `true`

## Structured artifacts
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/candidate_compatibility.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/candidate_pool_manifest.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/example_manifest_train.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/example_manifest_validation.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/example_validation.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/final_summary.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/implementation_manifest.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/input_manifest.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/join_manifest.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/join_statistics.csv`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/leakage_validation.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/m23_62_reverification.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/protocol_events.jsonl`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/provenance_disclosure.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/row_supervision.parquet`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/source_topology_manifest.json`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/source_topology_summary.csv`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/summary.csv`
- `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/track_purity.csv`
