# M23-64 result — frozen-pool pair reconstruction and from-scratch relation training

## Final decision
`PASS_V3_FROM_SCRATCH_RELATION_TRAINING`; status=`closed`; training_runs=3; TrackEval=0; tracker_outputs=0; HOTA is intentionally empty.

## Commands
```bash
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py init
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py run-all
```

## Frozen SHA
- script: `ed7242cae656f0b31e7cdbce8ac35a5cfb303a70992d657e39fc52e0c19d19fb`
- prereg: `2b640e0ea77d6c6fa1ba81bdc3596abc8ab4b3e3c8ec23049162e6c4b280932d`
- input manifest: `f18dd0e2d6c860afbce5b5e20be72f1b84db4d8f06f313cc38d5a5fb98326e3f`
- R63 candidate pool: `689846d9b311436ba6f88b5d5b09b7caf4ee1e15f5153f5b1e80d8ed8833b2c4`
- R63 paired pool: `6d3f1401ac42104ef83718d037f6442af0d6c605fb080d9e584a3acd34af0d79`
- R63 supervision: `1cccd06c0a6fdbbe3ed1a3d9baca0d23529bff173fb364853e186d4634956c04`

## Pair diagnostic
| sequence | frozen pool | valid corrected | M23-63 selected valid | valid not selected |
|---|---:|---:|---:|---:|
| MOT17-02 | 20000 | 42 | 1 | 41 |
| MOT17-04 | 20000 | 17 | 0 | 17 |
| MOT17-05 | 20000 | 17 | 0 | 17 |
| MOT17-09 | 20000 | 55 | 0 | 55 |
| MOT17-10 | 20000 | 51 | 1 | 50 |
| MOT17-11 | 20000 | 63 | 3 | 60 |
| MOT17-13 | 20000 | 36 | 0 | 36 |

Prior diagnostic exact replication: `True`. Candidate topology was not modified.

## Stage A gates
```json
{
  "all_arrays_finite": true,
  "all_pair_ids_from_frozen_pool": true,
  "all_r62_r63_sha_match": true,
  "all_tensor_provenance_valid": true,
  "all_train_sequences_nonzero": true,
  "all_validation_sequences_nonzero": true,
  "candidate_pool_unmodified": true,
  "contract_hash_match": true,
  "m23_63_closure_valid": true,
  "m23_63_only_paired_train_support_failed": true,
  "masks_prefix_valid": true,
  "no_checkpoint_loaded": true,
  "no_edge_added": true,
  "no_topology_changed": true,
  "no_unknown_as_negative": true,
  "node_relation_byte_exact": true,
  "pair_diagnostic_exact": true,
  "paired_pool_unmodified": true,
  "scope_guard": true,
  "split_isolation": true,
  "supervision_sidecar_unmodified": true,
  "train_paired_support_at_least_5": true,
  "train_valid_pair_182": true,
  "training_not_started": true,
  "validation_paired_support_at_least_1": true,
  "validation_valid_pair_99": true,
  "validator_no_tb_reads": true
}
```

## Training
| seed | epochs | best epoch | best composite | checkpoint SHA |
|---:|---:|---:|---:|---|
| 2359001 | 30 | 1 | 0.37057869629423396 | 15ccca9fdb811082c8eb327b5c01f5272af64314a7824a8ad02b6cf7ba225f4c |
| 2359002 | 30 | 2 | 0.3686652129550528 | 5d01bf68c117ba6432b1d203034f7d53d189ecf068e66be4052333af21e25860 |
| 2359003 | 30 | 19 | 0.37629228967045475 | dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329 |

Selected checkpoint: `{"checkpoint_sha256": "dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329", "composite": 0.37629228967045475, "epoch": 19, "seed": 2359003}`

MOT17 validation: `{"validation_ABA_exact_two_boundary_recall": 0.0, "validation_boundary_pr_auc": 0.07761571889450053, "validation_boundary_precision_at_actual": 0.04697986577181208, "validation_boundary_recall_at_95_precision": 0.020134228187919462, "validation_catastrophic_false_link_rate": 0.1185682326621924, "validation_composite": 0.37629228967045475, "validation_incoming_R1": 0.8333333333333334, "validation_node_pr_auc": 0.1635277465575899, "validation_outgoing_R1": 0.8620689655172413, "validation_paired_replacement_R1": 1.0, "validation_risk_pr_auc": 0.9109707608565373}`

## Scope
`{"held_outer_reads": 0, "hota_evaluations": 0, "m23_54_starts": 0, "m23_58_starts": 0, "mot20_gt_reads": 0, "mot20_test_reads": 0, "mot20_test_submissions": 0, "teacher_reads": 0, "tracker_outputs": 0, "trackeval_runs": 0, "training_runs": 3, "v2_checkpoint_loads": 0, "warm_starts": 0}`

No MOT20 GT, teacher, held-outer or test input was read. No tracker, TrackEval, HOTA, v2 checkpoint load, warm start, M23-54 or M23-58 was performed.
