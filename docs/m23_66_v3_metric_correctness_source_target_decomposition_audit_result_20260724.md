# M23-66 — M23-59 v3 Metric Correctness and Source-to-Target Decomposition Audit — Result

Final status: **completed**  
Decision: **COMPLETED_POST_HOC_DIAGNOSTIC**  
Measurement integrity: **FAIL_METRIC_DEFINITION_BUG**  
M23-65 gate stability: **STABLE_FAIL**  
Scientific primary failure: **source_boundary_capacity_failure**  
Overall primary classification: **source_boundary_capacity_failure**

## Scope

This is an independent post-hoc diagnostic using frozen GT-derived label sidecars. It is not deployable, not a strict result, and does not authorize a next policy. No training, tracker, TrackEval, HOTA, MOT20 test read, or raw GT read occurred.

## Executed commands

- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py init`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py verify-inputs`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py freeze-source-scores`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py freeze-canonical-queries`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py reproduce-legacy`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py audit-corrected-metrics`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py diagnose`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py validate`
- `python -B -u scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py summarize`

## Input and historical provenance

All specified top-level and nested frozen scientific artifacts matched: `True`.
M23-65 recorded script SHA: `9f690c7f667ce45d7c8bc1d1a67b68190bfde7c2923904abf5bab40a0d5710f5`.
M23-65 current script SHA: `719284faf5702ecb1c2601e1704762cbe71f8e75dd7dec943562dbe052a2b7be`.
Byte-exact historical source found: `False`; reproduction status: `unavailable`.
Legacy frozen-artifact behavioral reproduction: `True`.

### Input SHA verification table

| Input | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---|---|
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/feature_contract_v3_1.json` | `4c243b411cdfe02bad65d9add856db5702c757707675ff326c47a87f2d555e99` | `4c243b411cdfe02bad65d9add856db5702c757707675ff326c47a87f2d555e99` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/source_topology_manifest.json` | `82974588e0b6ba5c1c8478f0a2f49da8c7925dece5e9abec23e44595990cd2a7` | `82974588e0b6ba5c1c8478f0a2f49da8c7925dece5e9abec23e44595990cd2a7` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/source_chunks.parquet` | `fac94cc2833f55b8f906e88d930d31054a4fd107b3392605bf4a4c2a9d3e4b6f` | `fac94cc2833f55b8f906e88d930d31054a4fd107b3392605bf4a4c2a9d3e4b6f` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/candidate_pool.parquet` | `689846d9b311436ba6f88b5d5b09b7caf4ee1e15f5153f5b1e80d8ed8833b2c4` | `689846d9b311436ba6f88b5d5b09b7caf4ee1e15f5153f5b1e80d8ed8833b2c4` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/paired_candidate_pool.parquet` | `6d3f1401ac42104ef83718d037f6442af0d6c605fb080d9e584a3acd34af0d79` | `6d3f1401ac42104ef83718d037f6442af0d6c605fb080d9e584a3acd34af0d79` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_supervision_join/row_supervision.parquet` | `1cccd06c0a6fdbbe3ed1a3d9baca0d23529bff173fb364853e186d4634956c04` | `1cccd06c0a6fdbbe3ed1a3d9baca0d23529bff173fb364853e186d4634956c04` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1/frozen_checkpoint/relation_v3_frozen.pt` | `dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329` | `dc24211ff8c050f03920711aadac009d4feac26568ff356f8d6bd29dabf7d329` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1/external_validation_metrics.json` | `5aa240738c5abf1076cfe1c1ea2a83ed1e2d90ad3e3e75e09d2130736c311c45` | `5aa240738c5abf1076cfe1c1ea2a83ed1e2d90ad3e3e75e09d2130736c311c45` | `True` |
| `scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py` | `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d` | `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d` | `True` |
| `scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py` | `ed7242cae656f0b31e7cdbce8ac35a5cfb303a70992d657e39fc52e0c19d19fb` | `ed7242cae656f0b31e7cdbce8ac35a5cfb303a70992d657e39fc52e0c19d19fb` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/topology_manifest.json` | `2f5d3a20d882cc4d630ef89e2abb3692e2acf3dbb3f2cb7ce0126cdfc5019445` | `2f5d3a20d882cc4d630ef89e2abb3692e2acf3dbb3f2cb7ce0126cdfc5019445` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/score_freeze_manifest.json` | `e43484f7e40e9ef3660bb96e64d442b56b5ed17e946380f3c76f10f09927b3a5` | `e43484f7e40e9ef3660bb96e64d442b56b5ed17e946380f3c76f10f09927b3a5` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/label_join_manifest.json` | `b65978f906438b9d21bc143eb959897cef83b2006cf810f788455feca2a67fed` | `b65978f906438b9d21bc143eb959897cef83b2006cf810f788455feca2a67fed` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/representation_metrics.json` | `622eed49f8e6cb58ab3e838ffc234c4e33e276c89ac43de277a39deafb2fbf3e` | `622eed49f8e6cb58ab3e838ffc234c4e33e276c89ac43de277a39deafb2fbf3e` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/representation_metrics.csv` | `e56185ccb21bbd8235cf7d42d59abc56323e055adc2b152710000a8520560e81` | `e56185ccb21bbd8235cf7d42d59abc56323e055adc2b152710000a8520560e81` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/representation_gate.json` | `47e128cd29297df3e8ab95798a14e7b5228436d4e0f4a34ac7d0d174dc989f37` | `47e128cd29297df3e8ab95798a14e7b5228436d4e0f4a34ac7d0d174dc989f37` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/input_manifest.json` | `58e2deca17e76aa643b161232225d3be733dad4c59d429921583b2736d4f2bd4` | `58e2deca17e76aa643b161232225d3be733dad4c59d429921583b2736d4f2bd4` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/topology/candidate_pool.parquet` | `32ee357eefeabb67586528f1fe7a3e396a5e9e9f448d5a364f516fa1663d8c71` | `32ee357eefeabb67586528f1fe7a3e396a5e9e9f448d5a364f516fa1663d8c71` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/topology/chunks.parquet` | `61370e3fbf780f9c2ef7501b138eae0f7e2c227cc98434fc59bb5d40f0ad439e` | `61370e3fbf780f9c2ef7501b138eae0f7e2c227cc98434fc59bb5d40f0ad439e` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/topology/paired_candidate_pool.parquet` | `ed106a7d9da069183e9d633e24f7ab09f9619d176eba2e76c3c2eefd51255c2d` | `ed106a7d9da069183e9d633e24f7ab09f9619d176eba2e76c3c2eefd51255c2d` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/topology/windows.parquet` | `14b0ead7220fbf1480cf8efaccfbc488f45c0b013e05f7620fdc94d7fe448584` | `14b0ead7220fbf1480cf8efaccfbc488f45c0b013e05f7620fdc94d7fe448584` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-01/rows.parquet` | `07e587ab05a328412110336fde6fbb1b16e8602eb9295b274b5a52f286ab5b93` | `07e587ab05a328412110336fde6fbb1b16e8602eb9295b274b5a52f286ab5b93` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-01/row_features.f16.npy` | `eeca8d8d75d0fcf66366a988812035a5b2e68e5d1437eda2798f0902d3960cd3` | `eeca8d8d75d0fcf66366a988812035a5b2e68e5d1437eda2798f0902d3960cd3` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/topology/candidate_pool.parquet` | `3d95657b935b500b72ce421a704ae878e893f3e849e6123907429e00d0526c2d` | `3d95657b935b500b72ce421a704ae878e893f3e849e6123907429e00d0526c2d` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/topology/chunks.parquet` | `83226e25378c2577442a857dae1b5854c9369759f8ccf172718366f8de014e97` | `83226e25378c2577442a857dae1b5854c9369759f8ccf172718366f8de014e97` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/topology/paired_candidate_pool.parquet` | `5683dc868f31b8e6c62f2e9635e572490335396b75908521c881dcb0079e89c0` | `5683dc868f31b8e6c62f2e9635e572490335396b75908521c881dcb0079e89c0` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/topology/windows.parquet` | `aa49fc3ad15fbd8124b10ec78bab0cc7dff404ac56837d35602de8eaa12358de` | `aa49fc3ad15fbd8124b10ec78bab0cc7dff404ac56837d35602de8eaa12358de` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-02/rows.parquet` | `7eea65077413bf726ce67265b3cd61ea59a23cb6203c4b9a0420fa29e897657f` | `7eea65077413bf726ce67265b3cd61ea59a23cb6203c4b9a0420fa29e897657f` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-02/row_features.f16.npy` | `21cab93d9721388866bbf84842d2dce4e79836739e3d4f5397f15528103a25d9` | `21cab93d9721388866bbf84842d2dce4e79836739e3d4f5397f15528103a25d9` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/topology/candidate_pool.parquet` | `724d668dcdf3468b0aaada3e82b75b8293796b044723523f1bfdf1d47ac3a351` | `724d668dcdf3468b0aaada3e82b75b8293796b044723523f1bfdf1d47ac3a351` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/topology/chunks.parquet` | `466c7de6ffb6501d14012b50535af21060384ee8625a482202230a62240333bd` | `466c7de6ffb6501d14012b50535af21060384ee8625a482202230a62240333bd` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/topology/paired_candidate_pool.parquet` | `d86e20656fff4829f359ae556e2d3f7d71645e9e44fbaac2921ade09b25e8d0a` | `d86e20656fff4829f359ae556e2d3f7d71645e9e44fbaac2921ade09b25e8d0a` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/topology/windows.parquet` | `031a89cbf74fd995be8886ae4e725e396629233f4744ee1b7ce6bfa61a3e7a87` | `031a89cbf74fd995be8886ae4e725e396629233f4744ee1b7ce6bfa61a3e7a87` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-03/rows.parquet` | `3bbee2c8412cf633619c32cfb91b548eb6200e724b9d0a6c4c0027354eabc2c1` | `3bbee2c8412cf633619c32cfb91b548eb6200e724b9d0a6c4c0027354eabc2c1` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-03/row_features.f16.npy` | `f706905b691e1d96fe5dde798b0b39f3e02de518fd042b5a5b5ab9d63a22b6dc` | `f706905b691e1d96fe5dde798b0b39f3e02de518fd042b5a5b5ab9d63a22b6dc` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/topology/candidate_pool.parquet` | `ee727d319cc7ead00aeef9e08981820384926445175f73e1819427a7c3f12045` | `ee727d319cc7ead00aeef9e08981820384926445175f73e1819427a7c3f12045` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/topology/chunks.parquet` | `2998124cfbb4f716deab224cfc43abc67264ed1a054ef1996f4b483a328097c6` | `2998124cfbb4f716deab224cfc43abc67264ed1a054ef1996f4b483a328097c6` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/topology/paired_candidate_pool.parquet` | `e43a4d34f198ab6f17e3c8037fcbd5251e39ab82156b924747c28ba3df23347f` | `e43a4d34f198ab6f17e3c8037fcbd5251e39ab82156b924747c28ba3df23347f` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/topology/windows.parquet` | `f51f1d0e5fa4a49491fb06398f14e80ecfa091800f225de585dbac2510c3c125` | `f51f1d0e5fa4a49491fb06398f14e80ecfa091800f225de585dbac2510c3c125` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-05/rows.parquet` | `14a1a506dd90bd295de32859e45528a5a7a2550e7bac118412f2b8c9154ad069` | `14a1a506dd90bd295de32859e45528a5a7a2550e7bac118412f2b8c9154ad069` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_gtfree_source_regeneration/observables/MOT20/MOT20-05/row_features.f16.npy` | `8d73160a635fc901f84a8915c15016b1b7b37db7c8eb907fc4924a5322c18709` | `8d73160a635fc901f84a8915c15016b1b7b37db7c8eb907fc4924a5322c18709` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/boundary_scores.parquet` | `990ecd3e3a1a41abff8313ee6243792a5c0a59c7b13608981079e3adfd91fe72` | `990ecd3e3a1a41abff8313ee6243792a5c0a59c7b13608981079e3adfd91fe72` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/node_scores.parquet` | `b6ce80bb9681c714c2eda9f3d7971d53c70caa099e9c3090894ccd77296ee545` | `b6ce80bb9681c714c2eda9f3d7971d53c70caa099e9c3090894ccd77296ee545` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/pair_scores.parquet` | `4028b293ce69d6a3c8be42b8678f54dab6d75178d4d92d03c4d1befda6ef553e` | `4028b293ce69d6a3c8be42b8678f54dab6d75178d4d92d03c4d1befda6ef553e` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/relation_scores.parquet` | `b62720f62aed689277a2bef412977030b32997c8b404b94f209c46a6559ca948` | `b62720f62aed689277a2bef412977030b32997c8b404b94f209c46a6559ca948` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/score_manifest.json` | `8a208540f97a199dfd4de2cd22ac8375384441e8e8d7607e8cec0a07b8a37a76` | `8a208540f97a199dfd4de2cd22ac8375384441e8e8d7607e8cec0a07b8a37a76` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/scores/score_to_source_row.parquet` | `bc10fa5bdb39cb5cd0e6f752ae5493fbc3d653b94c6d09b42b01a16ec53e3b21` | `bc10fa5bdb39cb5cd0e6f752ae5493fbc3d653b94c6d09b42b01a16ec53e3b21` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/boundary_scores.parquet` | `2595abce91c675ac6297b39fbfdb108dd3361403dd6456723fc298768251a1ec` | `2595abce91c675ac6297b39fbfdb108dd3361403dd6456723fc298768251a1ec` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/node_scores.parquet` | `57e36fb87e170aa9cb33d88ff4703062051a1eb927d31376b49bc8d0352c61a3` | `57e36fb87e170aa9cb33d88ff4703062051a1eb927d31376b49bc8d0352c61a3` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/pair_scores.parquet` | `19b210cf324c54f8eac9c8af030e850f34f55bbea87ddb04d066dd3c8f747ba4` | `19b210cf324c54f8eac9c8af030e850f34f55bbea87ddb04d066dd3c8f747ba4` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/relation_scores.parquet` | `ac9f3e930e83e6461d3e70f1117affeaf066980c6d5eb4c0d5ac58c1c883d51b` | `ac9f3e930e83e6461d3e70f1117affeaf066980c6d5eb4c0d5ac58c1c883d51b` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/score_manifest.json` | `39f69a3aff3a42a2c72441a633b26e423e0b7d7217f954a0b26b0cebd2902077` | `39f69a3aff3a42a2c72441a633b26e423e0b7d7217f954a0b26b0cebd2902077` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/scores/score_to_source_row.parquet` | `48b8f0b1dda99e0fb3be124dcb41abc4b9a36fe6d783e9bb2c6550bc9528f1c8` | `48b8f0b1dda99e0fb3be124dcb41abc4b9a36fe6d783e9bb2c6550bc9528f1c8` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/boundary_scores.parquet` | `547bad2791e2664343f1a64cca7ed775e9dee6a18602f85c90a0dbbbbde680ee` | `547bad2791e2664343f1a64cca7ed775e9dee6a18602f85c90a0dbbbbde680ee` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/node_scores.parquet` | `c7e3459bdfcf5abefd99f139488fbcdea40ecbc9b7cdda213c912cd52f54b970` | `c7e3459bdfcf5abefd99f139488fbcdea40ecbc9b7cdda213c912cd52f54b970` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/pair_scores.parquet` | `d92ec4fe763d5b36c82621843cb6b770ec575331fbdaccc65a7f64dff08c9f5d` | `d92ec4fe763d5b36c82621843cb6b770ec575331fbdaccc65a7f64dff08c9f5d` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/relation_scores.parquet` | `1415d3a2898935f10b3ecbdf7c686f0f5500f5cd843686defbe601ef4e43d6a6` | `1415d3a2898935f10b3ecbdf7c686f0f5500f5cd843686defbe601ef4e43d6a6` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/score_manifest.json` | `4d5e28fb0b0d5cb2122d88d70311e470c3f9afcb97fca0b0d4d971725cf04b3c` | `4d5e28fb0b0d5cb2122d88d70311e470c3f9afcb97fca0b0d4d971725cf04b3c` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/scores/score_to_source_row.parquet` | `a67153bbfaf11549122539ca58962f727636f857edd46cea8cd403aa6a077ff0` | `a67153bbfaf11549122539ca58962f727636f857edd46cea8cd403aa6a077ff0` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/boundary_scores.parquet` | `f6097c4c3d7719c2f3480f2d99c83368c64530bace39bac274d07d7b9ce3063e` | `f6097c4c3d7719c2f3480f2d99c83368c64530bace39bac274d07d7b9ce3063e` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/node_scores.parquet` | `fe885a3eacf3fffbb93875e0de36ef6d26eabd123b64fba4217678c0eec23b7b` | `fe885a3eacf3fffbb93875e0de36ef6d26eabd123b64fba4217678c0eec23b7b` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/pair_scores.parquet` | `fad5e1eb57db139dfb45b25186bdc4991d6643de6cf195657abcaf49df0a53d9` | `fad5e1eb57db139dfb45b25186bdc4991d6643de6cf195657abcaf49df0a53d9` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/relation_scores.parquet` | `d60289e441bd3e7287d7843bbf62f5208b500e5dc94cec959b52b08747b24561` | `d60289e441bd3e7287d7843bbf62f5208b500e5dc94cec959b52b08747b24561` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/score_manifest.json` | `b6e57f209f91155278b0bfe28d838ee8ab7a6a59f5da93ba258991968d16b85b` | `b6e57f209f91155278b0bfe28d838ee8ab7a6a59f5da93ba258991968d16b85b` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/scores/score_to_source_row.parquet` | `6af814c5cd939f7679bfff3013d9d710e3451884c975bba525f5e1644d754239` | `6af814c5cd939f7679bfff3013d9d710e3451884c975bba525f5e1644d754239` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/labels/row_labels.parquet` | `535eaf28bc6104749c8b5738e9266fc143e4c32f3276c836dc31dca376dc55b5` | `535eaf28bc6104749c8b5738e9266fc143e4c32f3276c836dc31dca376dc55b5` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/labels/track_purity.parquet` | `b2875d45802294e6a6d553adf9dda5b5fd718e1a0121a8bc0730def59ef7f0c2` | `b2875d45802294e6a6d553adf9dda5b5fd718e1a0121a8bc0730def59ef7f0c2` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-01/labels/join_trace.json` | `4b88051a0be7680e614d66a258e2c4caac0d9e17fb5b12a0b2e818e037970bd3` | `4b88051a0be7680e614d66a258e2c4caac0d9e17fb5b12a0b2e818e037970bd3` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/labels/row_labels.parquet` | `7d3e96aa6cf1133eea42430ff4ad453f2c50cf2829bb0539592f0300d19f43a3` | `7d3e96aa6cf1133eea42430ff4ad453f2c50cf2829bb0539592f0300d19f43a3` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/labels/track_purity.parquet` | `200a7f27f22d8fe195a1c9407701a24c4d6abc7fb266405e9bf1cfeb231c650e` | `200a7f27f22d8fe195a1c9407701a24c4d6abc7fb266405e9bf1cfeb231c650e` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-02/labels/join_trace.json` | `865a756d51bfafb725fcdf01582de1e5c1028b848f687aaf7980386fce984052` | `865a756d51bfafb725fcdf01582de1e5c1028b848f687aaf7980386fce984052` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/labels/row_labels.parquet` | `19f86d225110693c23a7bc0858278b2a41fd5592c637a56c478526422eef8e58` | `19f86d225110693c23a7bc0858278b2a41fd5592c637a56c478526422eef8e58` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/labels/track_purity.parquet` | `bc875b09a051285d49cef4f52000b4b34509289e686121587e03294ad41524c6` | `bc875b09a051285d49cef4f52000b4b34509289e686121587e03294ad41524c6` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-03/labels/join_trace.json` | `72906ca9d78016bfe9a25313f34621a4022cefa9cfb7e2d6dbe6fc823d47e8ce` | `72906ca9d78016bfe9a25313f34621a4022cefa9cfb7e2d6dbe6fc823d47e8ce` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/labels/row_labels.parquet` | `67adb1e93f840b35ae624951f48d1cf07634aaa43df3e5ed80a5d1bf7925d411` | `67adb1e93f840b35ae624951f48d1cf07634aaa43df3e5ed80a5d1bf7925d411` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/labels/track_purity.parquet` | `0dfa385000d67ce38472c4ff3bb6fc4276eaccc23f8fdfb3d68e6bf39a16e9fa` | `0dfa385000d67ce38472c4ff3bb6fc4276eaccc23f8fdfb3d68e6bf39a16e9fa` | `True` |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_mot20_representation_gate/MOT20-05/labels/join_trace.json` | `68e6ff6764ca7fa1234411ce64009f192316ffdceee1f95e22e029ac8af01005` | `68e6ff6764ca7fa1234411ce64009f192316ffdceee1f95e22e029ac8af01005` | `True` |

## Boundary observation versus unique-transition audit

| Sequence | Raw observations | Matched observations | Unique transitions | Duplicates | Legacy AP | Corrected AP | Corrected P@actual | Corrected R@95P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-11 | 16351 | 15850 | 8844 | 7006 | 0.005879477 | 0.006735045 | 0.000000000 | 0.000000000 |
| MOT17-13 | 19547 | 18955 | 10631 | 8324 | 0.009258379 | 0.009758709 | 0.014492754 | 0.000000000 |
| MOT20-01 | 35646 | 35137 | 18976 | 16161 | 0.009428417 | 0.009974367 | 0.044444444 | 0.000000000 |
| MOT20-02 | 284620 | 279844 | 147608 | 132236 | 0.008273990 | 0.008569603 | 0.013906448 | 0.000000000 |
| MOT20-03 | 575968 | 569218 | 300734 | 268484 | 0.001060942 | 0.001095810 | 0.000000000 | 0.000000000 |
| MOT20-05 | 1205949 | 1185646 | 624074 | 561572 | 0.001388842 | 0.001457855 | 0.000000000 | 0.000000000 |

## Canonical query and candidate coverage

Canonical outgoing and incoming query counts are in `canonical_query_manifest.json`. Candidate-missing queries remain in all-query denominators with zero rank contribution.

| Domain | Direction | Pooled canonical coverage | Pooled any-valid coverage |
|---|---|---:|---:|
| source | outgoing | 1.000000000 | 1.000000000 |
| source | incoming | 1.000000000 | 1.000000000 |
| target | outgoing | 0.998775033 | 0.999259323 |
| target | incoming | 0.998775522 | 0.999117237 |

## Corrected relation ranking

| Domain | Direction | Canonical R@1/R@3/MRR | Any-valid R@1/R@3/MRR |
|---|---|---|---|
| source | outgoing | 0.459165/0.631579/0.574311 | 0.618875/0.751361/0.703821 |
| source | incoming | 0.441818/0.616364/0.556964 | 0.578182/0.720000/0.669672 |
| target | outgoing | 0.369911/0.544284/0.490179 | 0.539242/0.686266/0.638159 |
| target | incoming | 0.379190/0.550189/0.495774 | 0.556853/0.693055/0.648303 |

## Paired metric correction

The primary paired metric is strict `original_margin > 0` on legal pairs. The old threshold field is reported only as `legacy_misnamed_paired_replacement_R@1`.

| Sequence | Valid pairs | Original-over-cross accuracy | Threshold accuracy@0.5 | PR-AUC |
|---|---:|---:|---:|---:|
| MOT17-11 | 63 | 1.0 | 0.5118876655497048 | 0.3521935038837419 |
| MOT17-13 | 36 | 1.0 | 0.5044723969252272 | 0.26428989938923786 |
| MOT20-01 | 28 | 1.0 | 0.50665 | 0.25004329040606327 |
| MOT20-02 | 18 | 1.0 | 0.49244844612256755 | 0.1736452597016784 |
| MOT20-03 | 47 | 1.0 | 0.48372303970973596 | 0.37884389165864013 |
| MOT20-05 | 11 | 1.0 | 0.527819344306897 | 0.3913289561775322 |

## Source-to-target decomposition

Boundary AP is interpreted together with base rate, AP/base-rate lift, and ROC-AUC. Relation retention, fixed baselines, and the preregistered gap/crowd-density/purity/appearance-mapped/frame-density/presence/saturation/tie/multiplicity strata are stored in `source_target_comparison.json` and `.csv`.

## Final declarations

- `post_hoc_diagnostic_only=true`
- `uses_frozen_gt_derived_label_sidecars=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- `hota=null`
- `training_runs=0`
- `tracker_outputs=0`
- `trackeval_runs=0`
- `hota_evaluations=0`
- `mot20_test_reads=0`
- `mot20_test_submissions=0`
- `next_policy_authorized=false`

未执行Notion写回。

Registry row: `3974`. Run root: `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit`. Structured artifact hashes are in `artifact_sha256_manifest.json`.

## Structured records and SHA-256

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/boundary_duplicate_audit.csv` | 1109 | `67c595328475081a5a5fcf55f33932fcc5f3c4d144d9f15f4d64ab5bfc2812ab` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/boundary_metrics.csv` | 18280 | `9652efb50d7781d5ca0b92affa9844eee2549c98d3c7508aaf4eb077befe7d6c` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/boundary_metrics.json` | 67123 | `3e44cf2d5c340a314ca3328b23b6a3f8f3ec7970b4d85d3ff34f614577cb5fe2` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/candidate_coverage.csv` | 6736 | `3c60213d59407714d00433e392a112f442a1e7159c941e4e1c9eee8795062a8d` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/candidate_coverage.json` | 42229 | `e68fb0342bbe66b3b7de580ffd45512fab513541148e1d26f2736ebe19ce637e` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/canonical_queries.parquet` | 6416861 | `7247195d5e228721c8992a1f2e28189aaba18465b492838611abc01dc4f37e23` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/canonical_query_manifest.json` | 6084 | `0c542af68626fa04bc40547cf88362f497959c9c52a2069f234bea13e346a660` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/closure_validation.json` | 1045 | `f11e9a312e46ea11745071377127dfdf8c5d271cfc7dc69c2e35d759c8e275f2` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/corrected_ranking_metrics.csv` | 80764 | `3bcf3fa6dc19865f9d4a0f0f40f9e6a7036cfca30ceb1780cad3381697ddf5fa` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/corrected_ranking_metrics.json` | 124202 | `6bb368ad5c42c1ab50525bb1b8fa096c8327ce93d0cbd44040cad2f6e8e34444` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/decomposition_strata_manifest.json` | 1396 | `daac37eab7afb3ea9517ce84f9c38d49266483eb98ac077755a83f4bef961ff7` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/final_diagnosis.json` | 2714 | `9bd7dbe1ccf2dc71c31df8e45644c942110c34625e9c0c213e33b2c09de52d1c` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/final_summary.json` | 6835 | `813596aca4363fa4f9dd2803f445dd1a3da97f9e2e035d1b7c3a9fb2dd320b9d` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/gap_convention_reconciliation.json` | 570 | `aa3af29817ccbb3ff348c6abadc7005ccad62fc41e312815af5cfa96a380f814` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/implementation_manifest.json` | 3382 | `fa0beb6b83922640f5ed759c7e55095ca9bb994ccc4b353ec8a61848e3d4dacc` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/independent_closure_validation.json` | 582 | `d873270719ef671c196250e8e8bfc9fd2c6be44fb8468e203c9b3a0e7c123c1d` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/input_manifest.json` | 29451 | `f7e8b00c021a90c6cb0233ed4dd40166de2ffab22d2d6e0b8cf0b35492960fa4` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/legacy_metric_reproduction.json` | 12034 | `2a66b2a16ad2a5998ebd6b04f321244d0d390de84824a65257cc29ea29656163` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/m23_65_historical_implementation_provenance.json` | 1397 | `00be368ce1e65e3222c3c6da1c9069aa266084796190c3b8d636e73c6d3f3d9a` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/metric_definition_validation.json` | 4216 | `d35ac02f57fdc4879bf1b4f9aacf1623a884b7909bb5160c995f83bde48f7a94` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/paired_metric_audit.csv` | 2664 | `dac89f36d8595686ca4d8cf39ba96b7e9b3cc428774b67addfe2c2e2e371157d` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/paired_metric_audit.json` | 7066 | `993e1bcd1f21dda48e0b1f0092bf75e88aebebd0a8308310b5e642685db88fc2` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/paired_metric_records.parquet` | 4588028 | `78bf41b337050faf7ae3e8ec3d1fe2aefe401ad9dbac098a253c87a6967ff1a4` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/preregistration.json` | 1799 | `67462280434ca6534872ca008927f67d96ee9785a65e91e9640b82da39c0f8ec` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/process_gpu_validation.json` | 1059 | `ed9afa74fa0fcf947e79ba5c40b3224556345fc0c86336a7ed5e2e92fc26d160` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/protocol_events.jsonl` | 18957 | `60ceb98543456f692baec63d6efa2fed0edca00a940d75226dd12482c491b8e4` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/relation_query_diagnostics.parquet` | 8094461 | `cac24af536a33cb0e0c8e3a856c43d03863b39c3feb4433c272c5eb280c3d26c` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/scope_validation.json` | 1000 | `9ba8743546cd600ec9ee212f3ff1875121c62e1f75021d8c9a74508dcc283178` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_score_manifest.json` | 3291 | `d73dd3b7397ee235601e93a9bf72ce2cf366c9801a8f0ef52bd11a61b7dd2cb2` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/boundary_scores.parquet` | 440703 | `627a96da12d9505933919cc2cbf5b75906a5631a6850023c53aa746cbcca5c8e` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/node_scores.parquet` | 87347 | `e918e33ca384e147c40de4670af968a00a14dbc3aa8adc224bd7bc5126e223db` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/pair_scores.parquet` | 1361235 | `8ca214fedbb64d24cfcb5ff6bd99ed1b8829d4fb8253da0b512b934f21718869` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/relation_scores.parquet` | 2547389 | `86319e922ad43c326781b51377221e985faca6e8bf1aa0f7008771c7f618c10e` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/score_manifest.json` | 1015 | `293550712c9f16017446d625690ce9f66b7a0002c7683b044c156e24524c9d2f` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-11/score_to_source_row.parquet` | 352494 | `00bc31610e28d3d1b5c2fa2526f97320e804a8bedd3a71b1bd325104acd656e5` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/boundary_scores.parquet` | 532066 | `6dace492d8ae621018554372c5d3e0b17a869ab1341d0359fec936329104a052` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/node_scores.parquet` | 103638 | `69d0a0300dc6c7ccb5744de094dc6dcf2e5ea7855f36eb395b542b2d1edabf20` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/pair_scores.parquet` | 1343085 | `40e4754377392bd3259ddb093214d2189379483cfc6afa6da62d34b45cb408db` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/relation_scores.parquet` | 3949953 | `59d4bc1bfe4de17e7a764bf352fba91cc7326df262fc8aabdc095af0d16393b7` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/score_manifest.json` | 1015 | `0bc7302e47a58a7aa1fc5350bc90f776efdaf30892cfad1cc658cd6f3f1a7f57` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_scores/MOT17-13/score_to_source_row.parquet` | 398177 | `bb3e3af0c9fcff3ae40e1cd85857cf6be4e8a65c6d4748678fa743d8ef6feafa` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_target_comparison.csv` | 60260 | `ac4ae88dcda9b948d7842946fc979ac1aea9302685a5a4238102c9b80b2d0b80` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/source_target_comparison.json` | 173748 | `810200e35320adc9923a9d9acc4c91686986333e2ef413c577bb2db3621be6e8` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/summary.csv` | 2150 | `becf8a2b7dc7f1035ea4b7851c933f660c780fb9a71d72247fb225d7142a5122` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/trusted_chunk_inventory.parquet` | 1292195 | `38af0cec33aed32a3825b6982b46d18980bc09dd373908ff10f64f2c98d30014` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/valid_targets.parquet` | 9090443 | `a99e1bf574fb93fdee23723cfcbd57aa4313fb1ecb7237a6526e9f81159c94e0` |
| `outputs/mot20_m23_20260718/m23_66_v3_metric_correctness_source_target_decomposition_audit/validation_report.json` | 405 | `7fc9fd36eea38642aec80bd5f28de007d763c8ac62d68561c1cabce62d788c51` |
| `scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py` | 150309 | `e5194d155fd971c017f6cc3f3b0c8eb919214e31caf9e83f25b99532f9936072` |
| `scripts/m23_research/test_m23_66_metric_definitions.py` | 1478 | `7f758ffb390e828204b21590425841960263e838bce767e90fe3d9c8a9e10582` |
| `docs/m23_66_v3_metric_correctness_source_target_decomposition_audit_prereg_20260724.md` | 13840 | `c75b81624c2f31fdad4ab719cc365d60d92a892a58329a746e9b1995d3b847e0` |
