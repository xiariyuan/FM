# M23-64 result — frozen-pool pair reconstruction and from-scratch relation training

## Final decision
`FAIL_TRAINING_INCOMPLETE`; status=`closed`; training_runs=0; TrackEval=0; tracker_outputs=0; HOTA is intentionally empty.

## Commands
```bash
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py init
python -u scripts/m23_research/m23_64_v3_pair_reconstruction_training.py run-all
```

## Frozen SHA
- script: `a3bed845b1f01ea5484c255b4bc22fac8b2230cd1c6835a1f9c3ef95dec36087`
- prereg: `42168e10c797ab29b3d049acaff990118d1d74c5bc387338fcbb02eec02c784f`
- input manifest: `934aa905a4c275d9e1e953f6a08c69b5a91fe5ad8e838c150b991d0a3a7aac4a`
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


Selected checkpoint: `{}`

MOT17 validation: `{}`

## Scope
`{"held_outer_reads": 0, "hota_evaluations": 0, "m23_54_starts": 0, "m23_58_starts": 0, "mot20_gt_reads": 0, "mot20_test_reads": 0, "mot20_test_submissions": 0, "teacher_reads": 0, "tracker_outputs": 0, "trackeval_runs": 0, "training_runs": 0, "v2_checkpoint_loads": 0, "warm_starts": 0}`

No MOT20 GT, teacher, held-outer or test input was read. No tracker, TrackEval, HOTA, v2 checkpoint load, warm start, M23-54 or M23-58 was performed.
