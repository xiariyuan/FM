# M23-59 v3 Semantic Alignment Validation — Preregistration (M23-61, 2026-07-22)

## Scope

This is a fail-closed semantic contract and lineage validation. It is not a normal model experiment.
No model training, tracker generation, TrackEval, threshold search, policy change, MOT20 test read/submission, or M23-54/M23-58 execution is allowed before every front-end gate passes.
M23-59 v2 and M23-60 are immutable inputs.

## Canonical index convention

- tensor width: 144
- appearance: global zero-based 0..127
- geometry: global zero-based 128..143, local zero-based 0..15
- `feature_143` means global zero-based column 143, geometry local index 15, one-based display column 144
- its preregistered canonical meaning is nearest same-frame row-center distance after x/width and y/height normalization, clipped to [0,1], singleton sentinel 1.0
- this meaning is selected from the v2 preregistration and generic generator source, never from MOT20 metrics

## Frozen front-end gates

1. canonical definition must be unique for 144/144 features;
2. semantic/formula parity and GT-free provenance must pass for 144/144 features on both source and target generation paths;
3. no GT, teacher action, identity label/mapping, held-outer label, or outer-conditioned normalizer may enter feature generation;
4. MOT17 physical videos must be disjoint across train/validation; only canonical FRCNN may be admitted; exact-image duplicates across splits are forbidden;
5. score/candidate mapping, stable tie-break and mask/index invariants must pass;
6. old checkpoint compatibility requires byte-identical training-side 144-D inputs and unchanged normalization;
7. counterfactual B must improve both R@1 and MRR versus A on at least 3 of 4 fixed outers, with no new semantic, lineage, score/index or mask failure;
8. failure of any prior gate blocks B/C replay, training and strict outer evaluation.

## Frozen counterfactual conditions

- A_original_v2: historical v2 observable/checkpoint reference only.
- B_canonical_feature_143: replace only global column 143 with the canonical nearest-neighbor formula; old checkpoint permitted only after compatibility gate.
- C_neutralized_feature_143: replace global column 143 by the MOT17-train-only fixed median; no MOT20 statistics or labels.
- K=256 ranking, K=32 flow, gap buckets, architecture, loss, seed, epochs, risk/UCB, representation gate and P0/P1/P2 remain unchanged.
- diagnostics only; no tracker or TrackEval.

## Validator reconciliation rule

Expected-negative invariants are first converted into positive predicates (`actual == expected`). Raw expected false values must never be passed directly to `all(checks.values())`.

## Frozen inputs

| Path | SHA-256 | Bytes |
|---|---|---:|
| `AGENTS.md` | `d31b652a13d99e366e30cb33788e458bb85cfd763e860c78f93059dff137e97b` | 1016 |
| `docs/m23_59_relation_pretrained_hierarchical_flow_v2_result_20260721.md` | `f3a3bd116d1ec3eb3dae951bdc7b4be8995d3d185ff23722f1d6727543a1e5de` | 9195 |
| `docs/m23_60_relation_transfer_failure_audit_result_20260721.md` | `5dcd71f3b229dcb655c0d2ea0d54a3f24f2ad890d72a911ad6be11b3205801a6` | 8891 |
| `docs/m23_59_relation_pretrained_hierarchical_flow_prereg_v2_20260720.md` | `b78ffcf40397c8c9dcd1493afc4eb2519667e876186ed3d331e882ef975d4f46` | 18591 |
| `docs/m23_59_v1_invalidated_determinism_20260720.md` | `b50e8c263151d5cb4a4fa6020c00d2abc9e1f5196b14b7672d7c50224dfd949a` | 1846 |
| `scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py` | `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d` | 131516 |
| `scripts/m23_research/m23_60_relation_transfer_failure_audit.py` | `6fee73ecdc9ae2a8cd327179434ce2c10c5682bf83695994971d2d822ffc538b` | 74258 |
| `scripts/m23_research/m23_57_intra_node_change_point_capacity.py` | `3314e8e900553efd77d06ef5f8ebc944fdfffbf01bf6136d7b10052d69361501` | 56835 |
| `scripts/m23_research/m23_10_build_micrograph.py` | `d01fddb99c445c8236117b859b9c1322b69402a0f07522aebe0216f0a8dae35a` | 11626 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/final_summary.json` | `513ef98340ab5992ff92dba21b1ae5f4577346fb796dd2fb20807d158fcff246` | 10072 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/closure_validation.json` | `5e631110ec7d4bf32c8ca6d969d597dfa7c3dff92ce47c294f3c6bc93dc5c352` | 3805 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/strict_outer_evaluation/report.json` | `68ec8d4b37d34c6c5c077db7758c56aa6d7babf27d55a2c9c3b0926bc150c325` | 1310 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/protocol_events.jsonl` | `3e886fde26f4f475d015c2874da2097fe16ae4eeb9cfcd080425daff3221de90` | 10741 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/implementation_manifest.json` | `5da5a3d45ef3366e2e9cb61a3635a5d6dbd5583df07031507985173b265c9dff` | 3952 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_dataset_manifest.json` | `3c555e708a7341f6747a4a24ab9200bb9bf848d37b847dde6bc646b44e28e406` | 966076 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_pretraining/frozen_checkpoint_manifest.json` | `2a75c8b163ce5549cd513f78964d649773f92d78bbe4652646745981e22e5205` | 9719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_pretraining/relation_pretrained_frozen.pt` | `141c91a93ae58164ed5399cf1b1407e6fcd3dfec52b6950f94afa65251e1da42` | 3535028 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/audit_manifest.json` | `94e5d705845cc6e067aceea0a611d4ad24275d7108858e841bd25e918eb29435` | 45504 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/semantic_validation.json` | `a6d6b6ae17463d6d95f461634a907b2699a837c05a62c775ef1cd2e3083a51a1` | 12414 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/candidate_oracle.json` | `a8b8652164683cc4dc6032bc75a84174b0a3aab675d6b7e5a871affffbf69588` | 13692 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/ranking_diagnostics.csv` | `3e2deefa7c4f666515fd1a110f86a53f8d8fd9ff9127d0ff5a5f0a44c2cfbd28` | 32163 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/error_waterfall.json` | `91779d8068646da6a5746ba1d249d9e5853285b38a1ddd2cfb3522ab301939f8` | 339 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/final_diagnosis.json` | `78098c78e50bb4110cbe3e0e995e7ab43f990ff4cb8acde90ec2cf9ab65cc7b0` | 1576 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/completion_validation.json` | `5e69da70f948dee03e29d9edd7a4c0290b3df907206ba3cc5f1618644ccf7d19` | 3093 |
| `outputs/mot20_m23_20260718/m23_60_relation_transfer_failure_audit/independent_closure_validation_v2.json` | `3c8ce90e5c16b799dd2189f575fd46d9c857e6e34ee75096dd5eb659996f4426` | 35122 |
| `scripts/m23_research/test_m23_61_validator_reconciliation.py` | `028e008f0e89e7b232a83abb47a9437b84a908692771e2934eafe3e6cefd4469` | 1543 |
