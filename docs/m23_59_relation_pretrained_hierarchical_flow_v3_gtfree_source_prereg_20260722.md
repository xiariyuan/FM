# M23-59 v3 GT-Free Source-Row Regeneration — M23-62 Preregistration (2026-07-22)

## Scope

M23-62 is a source-row and observable regeneration stage, not a model experiment. It may read only frozen tracker rows, train images, seqinfo, frozen detector/ReID artifacts and phase0 dumps. It must not read MOT17 or MOT20 GT, teacher actions, identity labels, held-outer labels, MOT20 test data, or TrackEval outputs for selection.

No relation training, counterfactual model replay, tracker generation, TrackEval, strict outer gate, M23-54, M23-58 or test submission is authorized in this experiment.

## Frozen source-row construction

- MOT17 source rows: the pre-existing `full7_best_raw` raw BoT-SORT FRCNN outputs for all seven physical train videos. Its run manifest fixes image-only inference, YOLOX/ByteTrack checkpoint, SBS_S50 ReID and raw association; evaluation is outside tracker generation.
- MOT20 source rows: the existing M23-59 source tracker rows for MOT20-01/02/03/05.
- Row order is source tracker file order. Temporal context is grouped only by source `track_id` and sorted by `(frame,line_index)`.
- MOT17 physical train/validation split remains 02/04/05/09/10 versus 11/13. Detector variants are not duplicated; only FRCNN physical videos are admitted.

## Canonical 144-D contract

Contract version: `m23_59_v3_gtfree_source_contract_3.1.0`  
Contract hash: `90cfab7d3a1fc87cc46b26441d3d883b9fc72f27d8852d1e2074b00e628428f5`

- columns 0..127: same-frame phase0 YOLOX/ReID mapping by stable Hungarian IoU, threshold 0.5, fixed MOT20 SBS_S50 2048-D embedding, seed-2310 128-D projection and L2 normalization; unmatched rows use the all-zero vector;
- global column 134 / geometry local 6: fixed value `1.0`, explicitly named visibility-unavailable sentinel and never interpreted as measured visibility;
- global column 143 / geometry local 15 / one-based 144: nearest same-frame normalized source-row center distance clipped to `[0,1]`, singleton sentinel `1.0`;
- geometry local 14 counts same-frame source rows in both domains;
- no data-conditioned normalization or label-conditioned statistic is permitted.

## Execution order and pass rule

1. freeze preregistration, implementation and input SHA;
2. validate source tracker/image lineage without labels;
3. generate MOT17 phase0 detector/ReID dumps from FRCNN images only using the exact critical settings of frozen MOT20 phase0;
4. regenerate all seven MOT17 and four MOT20 observables from source rows and phase0 dumps;
5. validate 144/144 formula, semantic and GT-free provenance, full geometry float16 round-trip, feature-143 round-trip, mapping/index stability, source split and all SHA;
6. only a complete 144/144 pass may authorize a later fresh experiment to open MOT17 supervision labels and retrain. M23-59 v2 checkpoint reuse is prohibited because training-side inputs changed.

Any ambiguity, GT/teacher/held-outer lineage, phase0 setting mismatch, score/index instability, missing sequence or numerical mismatch causes fail-closed termination. This experiment never unlocks labels itself.

## Frozen inputs

| Path | SHA-256 | Bytes |
|---|---|---:|
| `AGENTS.md` | `d31b652a13d99e366e30cb33788e458bb85cfd763e860c78f93059dff137e97b` | 1016 |
| `scripts/m23_research/m23_62_gtfree_source_regeneration.py` | `ba213b205b27b718fbda553926093d4f48814990cdac961f36dc72e116381cb4` | 74731 |
| `scripts/m23_research/test_m23_62_gtfree_feature_contract.py` | `e0d9b4d15bc5cf55ebda0d14156376cdebf2b4668413fd52a98f56087e635d0d` | 5571 |
| `scripts/dump_yolox_reid_phase0.py` | `a2b5d7620148a94b07dbe71f9e21f8a8f1eccc72e9d2ef9b2257710142af942d` | 22298 |
| `scripts/m23_research/m23_10_build_micrograph.py` | `d01fddb99c445c8236117b859b9c1322b69402a0f07522aebe0216f0a8dae35a` | 11626 |
| `scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py` | `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d` | 131516 |
| `docs/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment_result_20260722.md` | `3555d0efcb28e35753455389c9f6a31c1a721368531aa0fac869e4e19943c016` | 2024 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment/final_summary.json` | `05e07b247729e1d4717063662f8ff953631b27749d1de3edc799f05488004840` | 1532 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment/feature_contract_v3.json` | `79e7493c36e8de6ceae189129cc5da134b9c1c9f45c6041a1a08f6d29210dca0` | 217768 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_semantic_alignment/independent_closure_validation.json` | `2906bdb264a2241d7faaeb95dab3d8622d351d70df814e71f9089a0beb9eade2` | 4109 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_dataset_manifest.json` | `3c555e708a7341f6747a4a24ab9200bb9bf848d37b847dde6bc646b44e28e406` | 966076 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/run_manifest.json` | `83da8a7eb60dbb759657ca6a68fd89d7814ae1bb23024d81f2c47e87118c66e6` | 9816 |
| `external/BoT-SORT-main/tools/track.py` | `6d6c0b8b6bb4e3dc27d98d323df75d8517540feac886f6b2bb297238ede02d11` | 108114 |
| `external/BoT-SORT-main/pretrained/bytetrack_x_mot20.pth.tar` | `021d7bc47fe20ae690007454cd2192df6237c2f4446ae92737a79523de89de64` | 792835731 |
| `external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth` | `d3a39a1ab54a63ac6724f2864a36485b0b5762fefced98941fca2a0784180656` | 315810563 |
| `external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml` | `900ab06e87e407b30b1a5ee6167d2de543c9039d1a6d64254f4a62f150276065` | 160 |
| `external/BoT-SORT-main/yolox/exps/example/mot/yolox_x_mix_mot20_ch.py` | `7b679ffa5c6b7c3d7b76ba76033900753ac0b046f785a1dd7a0bd90f932ad747` | 4498 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-02-FRCNN.txt` | `9adf706f9b8c804938b77f3c9594be070b270da8f6d2ab3399f3693f1109eb52` | 806558 |
| `datasets/MOT17/train/MOT17-02-FRCNN/seqinfo.ini` | `669c5135aa1b7f1726ee50ab6cec86780fc448dadaf93ea9c4fb6e2a0d4256aa` | 108 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-04-FRCNN.txt` | `688df2e05e85e3a13d78be5f18ae3ea0ea8d0d55330d76098affe89ebbf9d738` | 2260674 |
| `datasets/MOT17/train/MOT17-04-FRCNN/seqinfo.ini` | `f2d5bbe2c4d2a884422571987bc5363b61bb7e2c6748d1cff444e55b4886b888` | 109 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-05-FRCNN.txt` | `69ade59692894dd006680a9e203708c8ff00969d6db1aa7cd13d794e6ea0d020` | 316608 |
| `datasets/MOT17/train/MOT17-05-FRCNN/seqinfo.ini` | `2f49de730bcdfcd684069aed291627d2e5f13f8524e5163129e4bac88af815d0` | 106 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-09-FRCNN.txt` | `d8459c9c65831065595622e2e48715ea15a5f1d0f6bcc29824328b68ec99c008` | 242687 |
| `datasets/MOT17/train/MOT17-09-FRCNN/seqinfo.ini` | `96cec357e5661afad673cc2259310d2c0fb93854ffe18b765fd973c38f3957d9` | 108 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-10-FRCNN.txt` | `18f8680fbc026f580bbff47224b244fc0b60d50e01e5c568dd061904922f130e` | 555041 |
| `datasets/MOT17/train/MOT17-10-FRCNN/seqinfo.ini` | `7af062d088f3b7a88522f8095246eb01a6113d89823a96b7099a09d54421ecbd` | 108 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-11-FRCNN.txt` | `d21e6c1af7f0511e81d53bfb675bb35ab7d9833c45bd0c8402dda53257221a52` | 449701 |
| `datasets/MOT17/train/MOT17-11-FRCNN/seqinfo.ini` | `5e376f2489be7c5c5e64456a3842e9f62aa79ce7455903820c9bd218ff0c806f` | 108 |
| `external/BoT-SORT-main/YOLOX_outputs/full7_best_raw/track_results/MOT17-13-FRCNN.txt` | `af441d49fff067b22ce6902433355e8cf5a4cdce77141d9445e1c783493b8f1f` | 530426 |
| `datasets/MOT17/train/MOT17-13-FRCNN/seqinfo.ini` | `a7a0f607a31e3f311be6e9836214e23d4642c088db56126f51f150f13fe7a3af` | 108 |
| `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results/MOT20-01.txt` | `9ed62ce3a684dacd8492b4a15f294c5b5ec859979eafe524337843f85f4b11d2` | 935053 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_mot20_01/MOT20-01/dump_yolox_reid.npz` | `b2b7e6d2cf5797f77b1f74058022db6cfcb9a1111067e41440cb4c37ed89e734` | 88885512 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_mot20_01/MOT20-01/manifest.json` | `df010d6c8e6f6cd801d907cb8a6c778d87b1b80370ecc0215aa4198c8c1d100d` | 2056 |
| `datasets/MOT20/train/MOT20-01/seqinfo.ini` | `5850098d845d1a92c4b8649ee7ab4e2b7ef2b0a2104eb855cdef9a8e93da0a5e` | 102 |
| `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results/MOT20-02.txt` | `f7aa8b679cfb32dfd87f739994f80e88198ac3ebb00d05482c976d80f20cd94e` | 7534749 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_mot20_02/MOT20-02/dump_yolox_reid.npz` | `bf1899970e300e8bbc0a32cf0bba1abd4d2db1e5e23667f5f87ebb8620d73341` | 663943488 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_mot20_02/MOT20-02/manifest.json` | `ba5f304ec7174f657ff0990119f3e6d1d1d1ae412bfaec76c2d5baa3802e4680` | 2061 |
| `datasets/MOT20/train/MOT20-02/seqinfo.ini` | `e94c47840038e71e84613eeda0a9635728de1e488b56c5b05b16f68ff7305a44` | 103 |
| `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results/MOT20-03.txt` | `f318b80d2ae5c29d0e4fcd9e9472d69b4f46801547b342dc5677ebf8739480f8` | 14854705 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_yolox_reid_20260706_103713/MOT20-03/dump_yolox_reid.npz` | `0c154dbff546bd65f3bd535b2ff92c1ccee6a4b1d123fbd51cd49159c1e7838e` | 1377566968 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_yolox_reid_20260706_103713/MOT20-03/manifest.json` | `97e54636171c926921c8966223eca722f9026b6ace905d5fcb9c44b15fc6c49f` | 2273 |
| `datasets/MOT20/train/MOT20-03/seqinfo.ini` | `fb62095bce77a2e422e258d76266d767c7ea04816e8462f631b9b65ec2a64b5e` | 102 |
| `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results/MOT20-05.txt` | `244242dc002e601921a3ba65671ac71602773b2d26f84e386afaa9fdc1280e28` | 31779550 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_yolox_reid_20260706_103713/MOT20-05/dump_yolox_reid.npz` | `842db332441b15f8dd8863ceff16941756bdfa317c147d8db451f68b493b1afc` | 2794904040 |
| `/gemini/code/FMtrack-main/FM-Track/outputs/dmm_phase0_yolox_reid_20260706_103713/MOT20-05/manifest.json` | `be4fb869965d8692f54e81baebed7dbbbfcad665916efa5d29a4239029d278b5` | 2274 |
| `datasets/MOT20/train/MOT20-05/seqinfo.ini` | `1834b94212c439addf4662acba986e80f63f1b2a22f4f92ef7181ee2be47088e` | 103 |
