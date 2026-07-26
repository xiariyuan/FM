# M23-60 Relation Transfer Failure Audit — Preregistration (2026-07-21)

## Status and scope

This document is frozen before any M23-60 aggregate diagnostic result is calculated. M23-60 is an independent, post-hoc audit of frozen M23-59 v2 artifacts.

- `uses_mot20_gt=true`
- `post_hoc_diagnostic_only=true`
- `not_deployable=true`
- `not_a_strict_result=true`
- no training, tracker generation, TrackEval, threshold search, policy change, test submission, M23-54, or M23-58
- M23-59 files are read-only; M23-59 v1 artifacts are prohibited

## Fixed feature schema

Input order is exactly `appearance_000..appearance_127` followed by these geometry columns:

0. `center_x_norm`
1. `center_y_norm`
2. `box_width_norm`
3. `box_height_norm`
4. `log_aspect`
5. `log_area_fraction`
6. `visibility`
7. `velocity_x_height_frame`
8. `velocity_y_height_frame`
9. `log_width_change_per_frame`
10. `log_height_change_per_frame`
11. `frame_delta_over_30_clipped`
12. `velocity_x_residual`
13. `velocity_y_residual`
14. `crowd_density_over_100_clipped`
15. `nearest_neighbor_distance_or_mapped_indicator`

A semantic mismatch is critical when the same feature index has a different physical meaning across MOT17 and MOT20, when source/destination or predecessor/successor orientation changes, or when label/index mapping changes through sorting, reversal, padding, batching, or model input.

## Fixed candidate oracle

- MOT17 validation uses the frozen synthetic candidate sets: outgoing `[true successor, frozen outgoing hard negative]`, incoming `[true predecessor, frozen incoming hard negative]`, and paired replacement `[true joint assignment, crossed joint assignment]`.
- MOT20 uses frozen M23-57 candidate graph nodes/edges, teacher edge utilities, and successor-events tables.
- Outgoing coverage uses `out_rank`; incoming coverage uses `in_rank`; mutual coverage uses `max_rank`.
- Report coverage at 1, 3, 5, 32 and 256. `K=256` is ranking K; `K=32` is flow K.
- Correct-candidate absence (`candidate_present=0`) is always separated from candidate-present ranking error.
- No-successor queries are candidate-graph source nodes absent from the successor-event source set.
- Paired replacement uses deterministic adjacent distinct-identity true-successor events in the same fixed gap bucket; both crossed edges must exist.
- Oracle theoretical upper bound over all successor queries is candidate-present coverage; conditional oracle among candidate-present queries is 1.0.

## Fixed diagnostics

Ranking: AUROC, PR-AUC, MRR, R@1/R@3/R@5/R@32/R@256, score quantiles, top-1 margin, rank histogram, 10-bin ECE, constant prediction rate, tie rate, candidate counts, and fixed-seed random expectation. `-score` and candidate-order reversal are implementation diagnostics only.

Baselines on the identical candidate set: original outgoing rank, minimum gap, minimum endpoint displacement, minimum motion/velocity residual, maximum endpoint IoU, maximum destination detector score, fixed-seed random, and oracle.

Feature shift: count, missing/non-finite, mean/std/min/max, p01/p05/p25/p50/p75/p95/p99, hard clipping, outside-MOT17-train min/max and p01/p99 rates, KS, Wasserstein, PSI, and positive/negative conditional relation distributions. Deterministic sampling cap is 200,000 rows per domain with seed 2360001.

## Frozen failure classification

Priority is fixed and cannot be changed after results:

1. `implementation_or_semantic_mismatch` when any critical semantic check fails: frozen SHA/config/state mismatch; predecessor/successor or source/target inversion; xywh/xyxy or frame/gap mismatch; feature-index physical meaning mismatch; non-prefix mask/padding corruption; trace label/index instability; non-finite model input; or external score orientation where `-score` R@1 exceeds normal score R@1 by at least 0.10.
2. Otherwise `mixed_candidate_and_transfer_failure` when pooled MOT20 successor coverage@256 is below 0.80 **and** candidate-present frozen-model R@1 is more than 0.05 below the best fixed non-oracle baseline.
3. Otherwise `candidate_graph_bottleneck` when pooled MOT20 successor coverage@256 is below 0.80.
4. Otherwise `observable_domain_shift` when at least 25% of 144 row features have KS >=0.25 or PSI >=0.25 in at least three MOT20 sequences, and candidate-present frozen-model R@1 is at least 0.20 below MOT17 validation R@1.
5. Otherwise `learned_relation_transfer_failure` when candidate-present frozen-model R@1 is more than 0.05 below the best fixed non-oracle baseline or at least 0.20 below MOT17 validation R@1.
6. If none of 1–5 fires, the primary category remains `learned_relation_transfer_failure`, with an explicit weak-evidence flag; no new category may be invented.

The primary category is unique. Secondary evidence is reported but cannot override the priority order.

## Output schemas

- `audit_manifest.json`: immutable preregistration/input SHA record and flags.
- `semantic_validation.json`: check name, critical flag, pass, evidence, two trace examples.
- `candidate_oracle.json`: per-domain query counts, coverage, rank/count distributions, exclusion waterfall, paired coverage, oracle bound.
- `feature_shift.csv`: `kind,feature,domain,condition,count,missing_nonfinite,mean,std,min,max,p01,p05,p25,p50,p75,p95,p99,clip_rate,outside_train_minmax_rate,outside_train_p01_p99_rate,ks,wasserstein,psi`.
- `ranking_diagnostics.csv`: one row per domain/model/method/direction with ranking/calibration statistics.
- `error_waterfall.json`: fixed query waterfall.
- `final_diagnosis.json`: unique primary classification, threshold evidence, flags, allowed next direction.
- `summary.csv`: queue-level structured status.

## Frozen inputs

| Input | SHA-256 | Bytes |
|---|---|---:|
| `AGENTS.md` | `d31b652a13d99e366e30cb33788e458bb85cfd763e860c78f93059dff137e97b` | 1016 |
| `docs/m23_59_relation_pretrained_hierarchical_flow_prereg_v2_20260720.md` | `b78ffcf40397c8c9dcd1493afc4eb2519667e876186ed3d331e882ef975d4f46` | 18591 |
| `docs/m23_59_v1_invalidated_determinism_20260720.md` | `b50e8c263151d5cb4a4fa6020c00d2abc9e1f5196b14b7672d7c50224dfd949a` | 1846 |
| `docs/m23_59_relation_pretrained_hierarchical_flow_v2_result_20260721.md` | `f3a3bd116d1ec3eb3dae951bdc7b4be8995d3d185ff23722f1d6727543a1e5de` | 9195 |
| `scripts/m23_research/m23_59_relation_pretrained_hierarchical_flow_v2.py` | `50e12cdc6259903f9a7a76adfd3329622899c0c727f1466c2399034c49f6451d` | 131516 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/implementation_manifest.json` | `5da5a3d45ef3366e2e9cb61a3635a5d6dbd5583df07031507985173b265c9dff` | 3952 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/preregistered_protocol.json` | `452bbd4305525eff635daa2bb0cfefd9dec70075e6672104254eed44b82b0822` | 12491 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_dataset_manifest.json` | `3c555e708a7341f6747a4a24ab9200bb9bf848d37b847dde6bc646b44e28e406` | 966076 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_pretraining/frozen_checkpoint_manifest.json` | `2a75c8b163ce5549cd513f78964d649773f92d78bbe4652646745981e22e5205` | 9719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_pretraining/relation_pretrained_frozen.pt` | `141c91a93ae58164ed5399cf1b1407e6fcd3dfec52b6950f94afa65251e1da42` | 3535028 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/manifest.json` | `375c425d685e510a5db953cad9da863d3c3d6419028cab072c7fbbf81125f6ab` | 12167 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/manifest.json` | `81aa51560c43325f7d87e086969f2f8e95a666afe4cef8afdd03ec28bc9cf056` | 11412 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/final_summary.json` | `513ef98340ab5992ff92dba21b1ae5f4577346fb796dd2fb20807d158fcff246` | 10072 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/closure_validation.json` | `5e631110ec7d4bf32c8ca6d969d597dfa7c3dff92ce47c294f3c6bc93dc5c352` | 3805 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/strict_outer_evaluation/report.json` | `68ec8d4b37d34c6c5c077db7758c56aa6d7babf27d55a2c9c3b0926bc150c325` | 1310 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/summary.csv` | `4b1c1e943ac6e99627410da66809354554526b8557d392f23cbea9a3fc73aa8b` | 4647 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/protocol_events.jsonl` | `3e886fde26f4f475d015c2874da2097fe16ae4eeb9cfcd080425daff3221de90` | 10741 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/aba_event.npy` | `0abdfd3c2f0327b5b849aa00e9d0cc0258356a7d72996838822360cc459ad1ca` | 10338 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/boundary_label.npy` | `0c4ae7db19ff29238a73ea3d1a1ff6dd4dedd57c51e387fb251e5e2d3a03d12f` | 296218 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/node_group.npy` | `77bad50da09f3abdbff1e18d3fddcb6cb08749a0fd16bbaa040d3a04b4a0b356` | 20548 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/node_label.npy` | `9ab86f9ec00d3e94020186ac9f0959a4f024d94bcf9d4fa5af8cb03135d34d4a` | 10338 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/node_mask.npy` | `2cd5c8071b20df3c3be78201c93fa6b2d286eeea84eb70fc45852817a424cf74` | 306428 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/node_type.npy` | `11aea64754cb5790c17c01bb29138640dcbcc2f45aceaaa17469dfd7637f076e` | 10338 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/node_x.npy` | `4389f3ee60b17fab27bdf6bde7ca36a780fc6eb819f0fd6984ab59ac187bc138` | 88214528 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_bucket.npy` | `39216eab592b534e32a7fe5129b8f5da2f63b9f9d42763adaa13f8b37aeb7960` | 5232 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_group.npy` | `3b1b63ff3505773f851d9ce1a8482fe6fac0e6f5c54a52423142260119675f03` | 10336 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_pos1.npy` | `73ab4b20d2f0fd65c958548d828db50ae9fe8fdb0c251f9b4f4c1e410a31d830` | 11759744 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_pos1_mask.npy` | `2412e843ca79e38276b12c044fcb392ecc9b3408897175eab4024a55beb8f5fd` | 40960 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_pos2.npy` | `c68045e820de78e611d7661ec10baeaddf808e5e2da696230c7aabf5c3ea3947` | 11759744 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_pos2_mask.npy` | `2412e843ca79e38276b12c044fcb392ecc9b3408897175eab4024a55beb8f5fd` | 40960 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_src1.npy` | `1ab1d3a410ad12163a8770c14797193179afd33a28174d60ea0c1e275004997b` | 11759744 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_src1_mask.npy` | `2412e843ca79e38276b12c044fcb392ecc9b3408897175eab4024a55beb8f5fd` | 40960 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_src2.npy` | `b558ad7fca9918c7f0d2f13dcacb2a3f574e92cc298dd211c985937bef772248` | 11759744 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/pair_src2_mask.npy` | `2412e843ca79e38276b12c044fcb392ecc9b3408897175eab4024a55beb8f5fd` | 40960 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_bucket.npy` | `d7b79a238236433db99e43c7ddcc68d4d99504847ae9f3c62d85c87bc9831f9c` | 13425 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_group.npy` | `0f37686f5bda68cf15b43ddd8e24223b3587ee64c60ebecc39202fd6e05fbebd` | 26722 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_in_neg.npy` | `b18b89e46728c1e22f749b79cd8d8036d4ac2e9dd1bcc5e30268b48731d47365` | 30636416 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_in_neg_mask.npy` | `8b27ceefc0893ef020522121931132a46ba760b9bd5caa4f4f567b343ac9e77f` | 106504 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_out_neg.npy` | `aeaa49b9099c7186a457864c2cde31fec22a3288332dc3a554b9deb7e6f20e17` | 30636416 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_out_neg_mask.npy` | `8b27ceefc0893ef020522121931132a46ba760b9bd5caa4f4f567b343ac9e77f` | 106504 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_pos.npy` | `8dfd1373b2a4b6513e2f63ec0cb5130545efb2d9998f8dc940679bf844b5c07c` | 30636416 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_pos_mask.npy` | `8b27ceefc0893ef020522121931132a46ba760b9bd5caa4f4f567b343ac9e77f` | 106504 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_src.npy` | `715401547830a0d9d525dfa7529940f7321fde39f6053bef7a9077dd4a374631` | 30636416 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/train/rel_src_mask.npy` | `8b27ceefc0893ef020522121931132a46ba760b9bd5caa4f4f567b343ac9e77f` | 106504 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/aba_event.npy` | `3889284f5cb6ce21b58cb3e71dd8b2965c87d46a49c6fc321eb465fc9fd87e7b` | 3414 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/boundary_label.npy` | `3884e3941c091dd8a8e2fb5e6ff5f402bf17bf3347910f5dcbb9939e6510bbf2` | 95422 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/node_group.npy` | `ee651a6e077b8cf96bf97751039b7e600beb06cc673de0b6a570ac6ca64cdb3a` | 6700 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/node_label.npy` | `0b980c61bad85431b0ff5ce8d1ca9a57b905775c205229e3e00effd89e6303f4` | 3414 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/node_mask.npy` | `6995f8442782f5b40bf3d796fcb345b0d3082663d85e4c9e9057d6963aa7b809` | 98708 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/node_type.npy` | `cf051c074db949876d8114c390233d7283f8ddeb0ae555e630738b3dff9fe475` | 3414 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/node_x.npy` | `79d2e8f303b02f752a72ec99df56ccdac215a3014c6b8bcc7697ebcf8c5a11b3` | 28391168 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_bucket.npy` | `7c05693ed17e0e0f426c15032bc4749de4b5fe449e0ca96b79dcff6564d511fb` | 2176 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_group.npy` | `6c951648c843087e91c188d7b7ae2428aa9e3b2b6f78bdaa1edb7f1dd3eb99c9` | 4224 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_pos1.npy` | `24908ec2ca8e0100418f1f2ba604d32d65da1291732d90b3533c2c2fc953fc1a` | 4718720 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_pos1_mask.npy` | `f21e5bc2a6d5a4a69ba6b7367ae5b1ca8df3ae8310173e4963b4713e0dcd3494` | 16512 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_pos2.npy` | `617d52a8baab72cc8037a3eb137f13d435ff4908dc6aeba5155d78bade390f20` | 4718720 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_pos2_mask.npy` | `f21e5bc2a6d5a4a69ba6b7367ae5b1ca8df3ae8310173e4963b4713e0dcd3494` | 16512 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_src1.npy` | `124f31ea5233409841fb0882cb31d8f8b3e269a3e9f2d47d5b8f675b125bb128` | 4718720 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_src1_mask.npy` | `f21e5bc2a6d5a4a69ba6b7367ae5b1ca8df3ae8310173e4963b4713e0dcd3494` | 16512 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_src2.npy` | `890e6607970e7237eb4feb236c5de998f3ce1c857357f7e54299881b01f2d69d` | 4718720 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/pair_src2_mask.npy` | `f21e5bc2a6d5a4a69ba6b7367ae5b1ca8df3ae8310173e4963b4713e0dcd3494` | 16512 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_bucket.npy` | `7cb313bf1ac6a598056495be1cf87497243088408336debcf03383ba622bb7fd` | 4930 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_group.npy` | `4b61e22a4e4fe345b4d0b3264ec3329522e0b9adbe94a3316fdd62a256ffbc5d` | 9732 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_in_neg.npy` | `a6bb921ef01d969f0e576cc461efcec114124a995f03838e23fc7e32b35684ac` | 11063936 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_in_neg_mask.npy` | `2fa74963c0f7f750e0c38a14d367072a16d767cd2767cb189367cb721bbd7e1a` | 38544 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_out_neg.npy` | `417b185cba563dae87cbed0af42e4807ee2158d5586027ffbedd5f9f73f57e5c` | 11063936 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_out_neg_mask.npy` | `2fa74963c0f7f750e0c38a14d367072a16d767cd2767cb189367cb721bbd7e1a` | 38544 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_pos.npy` | `517a0e4633174cc20a2ab86a3792909a6eddebe0bfd20ad75d329b87f622c2c4` | 11063936 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_pos_mask.npy` | `2fa74963c0f7f750e0c38a14d367072a16d767cd2767cb189367cb721bbd7e1a` | 38544 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_src.npy` | `435138b288ecfc3629084de32e69a350c51a1c532d8489236b39173c7f63d169` | 11063936 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_examples/validation/rel_src_mask.npy` | `2fa74963c0f7f750e0c38a14d367072a16d767cd2767cb189367cb721bbd7e1a` | 38544 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-02/manifest.json` | `e01cbdefe5127ed4d277a88a468c687a33e1c3ba7be840c4d3dfce814ec16035` | 2456 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-02/eligible_gt_rows.parquet` | `da7227f25ecdfed2c20675aaea4c09d7a8045e7ddcb6f1e51a3005b4f373658b` | 179183 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-02/appearance_128.f16.npy` | `b3f5bf5d7724e2faebaa63e98a865a59667a957a22141a30fd02751279617da3` | 2814336 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-02/geometry_16.f16.npy` | `189b8b18d4af23b14f1c801c82723d08c0bf10eab4c2148686283c4bf133f08b` | 351904 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-04/manifest.json` | `a9270e83437afb74ff4ed2e6b8fdba703c350dbfdc6980b2216cfca40a3d729f` | 2454 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-04/eligible_gt_rows.parquet` | `2340d1ef65b20beec7a7322c40fce21a3d1bc489cdfb30590f5ccf8286de1dfb` | 784384 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-04/appearance_128.f16.npy` | `e98dae123d0740befefc83e0daaf5b5319eedb7f4c59c91f13379b824cdec41b` | 11589248 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-04/geometry_16.f16.npy` | `d678d674752e67808a3545da55cd194ff702fc3608f9b680a06c44211421d2a6` | 1448768 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-05/manifest.json` | `35dffce00d05dc846a3a6967b4083681471f04a1e96f9dc019a84c2ef67f3e20` | 2456 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-05/eligible_gt_rows.parquet` | `b440d7b0b5c23eed442ca2d0c9f04cf7182223408e6cdc964283a9179963c191` | 105621 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-05/appearance_128.f16.npy` | `9108bc273a5dc19e59fe500e7f38903539e199c97ebd824bccd41c2b18e39c10` | 1349504 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-05/geometry_16.f16.npy` | `a3f02d2661fbf39f1b256719c7b2ca7197ca03710713dd47ba55096c6f4a6ad2` | 168800 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-09/manifest.json` | `8883263b03336c4c71c7ca8bc5108a336c8e94a96e3c7724830e7e89f7cd90e8` | 2453 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-09/eligible_gt_rows.parquet` | `ae543c83ab9e452a7fc4bc053a3d03d8251f4ab2690e1ab09a33d28c2c062142` | 91413 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-09/appearance_128.f16.npy` | `b8b33069f4f8f4986052786fbbef944d0ce1f35d7ebfcbcc152feaa18d130c59` | 1041280 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-09/geometry_16.f16.npy` | `99223fea11177b715b8179197abb874175387a325d51be10b2f5231d1debb4ea` | 130272 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-10/manifest.json` | `abd354687f247c7bf2673d17764e87cebc0cc2e1017f4705cca40d4eb76addfe` | 2454 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-10/eligible_gt_rows.parquet` | `e6db5d911f7b8056f4aef4c3c2134cd4dc565c7d7ed223440e06f428203bc5c7` | 191242 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-10/appearance_128.f16.npy` | `d9123e7ce520215709e9c658af6a608a4d998522be6d9c8724db3ca9fea0fc48` | 2846080 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-10/geometry_16.f16.npy` | `96353851c1594c28af53e1a934a9e3bfd295281caf178c38141710f5a1833bba` | 355872 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-11/manifest.json` | `b9ba529163b6f9cdfdb316e5db98c1bd87e3348d3512f844128023453fa491c6` | 2456 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-11/eligible_gt_rows.parquet` | `445a2c01ef5cf7c3cd25ed5d21d4da0e53de6822b1ec105ecd76e2aa9273ea20` | 167040 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-11/appearance_128.f16.npy` | `beda75b00cfa2f55bb0dd487b1b8e881201089f88b480ee96756ebf735456a27` | 2069888 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-11/geometry_16.f16.npy` | `42ec00ebf8015d01822f34ad121149994681e23f428f962ff02e55ad7df60ef9` | 258848 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-13/manifest.json` | `ce278ba59c0177965e16d9377dd43a688253b4d0e248dd396391d042c4558c64` | 2456 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-13/eligible_gt_rows.parquet` | `1472d88ee743c17e2c67bb58741a3f1ca194b8ed584558871d7857886493cc52` | 209363 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-13/appearance_128.f16.npy` | `4cf5399aff02ab766c335f59d0a424d86744a4b196a8e0e65b5ca60af740a84c` | 2833792 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/external_features/MOT17-13/geometry_16.f16.npy` | `790822c7058b6861f5d7c496570d2733ae6a78da511ccd3829aa3fc000865c00` | 354336 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-01/manifest.json` | `f23f971de067708b265852db7e12032afa8ca5cc9b0a45a3a5677153380e4f85` | 1848 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-01/rows.parquet` | `f9e617f5921423a719ec2a448098ef648f467d90de2c9a1c32d6909046d9c2ba` | 696831 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-01/row_features.f16.npy` | `5848494936624779651eb10816cec01359ea20073977de97bc20098d935680fd` | 5579264 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_labels/MOT20-01/manifest.json` | `ccbca4e3eda0029f6e5b36360b882c9444114578c1f0120aa3fedf743c841d20` | 2913 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/boundary_universe/MOT20-01/chunk_membership.parquet` | `c4517d5dd0ca053a31cda9b1077c1dabe7b69a6c72e13d43bf429989c9c7a307` | 251692 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-01/frozen_candidate_graph/nodes.parquet` | `1f5691318fbda1b6e190a9b3ce8495738d22b0304c10122e17f0d1126092db8a` | 103388 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-01/frozen_candidate_graph/edges.parquet` | `a5dfd6c54d4adb08fb21243b4e0ff839746c766b110028eb61d2ed7a5bd6ff2e` | 9169264 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-01/teacher_identity_flow/teacher_edge_utilities.parquet` | `7656dddb14e2a812bead17325837d46e407141c57eee351335a484c3aadf364a` | 9886850 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-01/postfreeze_audit/successor_events.parquet` | `acb1604a8281b9044fcc367a933d3707b917acb3cc2874ae05662ca8105247f0` | 32615 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-01/model.pt` | `3091c200929579d3d0b4d656ac857bdf243559ece2d52b83e4cee1f3b98b0c8e` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-01/model_frozen_before_validation_labels.json` | `870b1bb74db959b337fc9567d37e79a6e78bf7a3bfb8aef5a17f041c06d51b88` | 2730 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-01/model.pt` | `6125db2f478c26c5e652c8459bbf8186c5cf1b84a5c5282fb0cd0c5f37345efa` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-01/model_frozen_before_validation_labels.json` | `fbd47e9d0c665b940e6e7aa4d02b14b603828fb44987b35294771b2df24b707d` | 2719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-01/model.pt` | `cdfb0a0c9b742fb1eff78b8638d78a7470cb1f3b6562fdf7fc7b6ca427551381` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-01/model_frozen_before_validation_labels.json` | `6245377421a7c1fb49547ea244615b381b84d136c52ecff92019a8052485788e` | 2729 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-02/manifest.json` | `758004c02cdfbd3233292a533630927dc467a15839e8bb6819afa577e7613d5d` | 1852 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-02/rows.parquet` | `2fd80427f56eca946d5de4b63d5fd85958e9b0bbc20cfdd0b7d185dd7e899ab4` | 4601315 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-02/row_features.f16.npy` | `4f3de50070734734722971e65ddc05c71514799a7f0063a984f7ad9504831642` | 43531904 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_labels/MOT20-02/manifest.json` | `34b88fbb06acce07ab6813370d528c37dd8ce3db5254e568e077283e4a1a8658` | 2925 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/boundary_universe/MOT20-02/chunk_membership.parquet` | `092c644377139b8383a29adabd7694e077ca51004b6a2b6fe23fefd6345f6e2b` | 1931211 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-02/frozen_candidate_graph/nodes.parquet` | `f3fd53b8c462a5ac5f33d8b36f41fe869e94e9a359309ea568a5428ea3335421` | 664083 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-02/frozen_candidate_graph/edges.parquet` | `bcfd578d6ab312748f6faeeef07687444df28669a49c5f40040f9066a47e31d5` | 79428043 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-02/teacher_identity_flow/teacher_edge_utilities.parquet` | `31d555577ab7f7290edffca48bfbe67aab2a9370a63f9edcb486bfada39a4780` | 88588910 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-02/postfreeze_audit/successor_events.parquet` | `432ec12d3fb91a8ded86492aaa32e2b8361c28aef1f952174e10dd3d7b326448` | 179591 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-02/model.pt` | `f31e7db5d0109ccefb13f22667d02ace331f20b066d9086c4614d373e9804e7b` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-02/model_frozen_before_validation_labels.json` | `66e0df488f220fed6e94f99381f6f605170fc28c14d464bcfb93c4c83ccb3b12` | 2720 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-02/model.pt` | `2a363acdde4125f1840d31cbb6a68effaa50521c2742d58e8a37bc8d0e3ded89` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-02/model_frozen_before_validation_labels.json` | `58fa93b70f73bba705c30b8b0ea9ce4ef1c0152f769840798a3c184defa3c751` | 2719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-02/model.pt` | `7e0f049208d5837c67d2b0566018969ae87b7c959c8dadd03f783dc03b18f97f` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-02/model_frozen_before_validation_labels.json` | `69fb072621fe0475920690bc5eecf2d749d22ddef4077b3d95444c15fb8ad46d` | 2729 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-03/manifest.json` | `ade59359e1ed2a5bed0ccd5c3e2db5c214cba6bcfcfdf409d8be358eb0ab363c` | 1853 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-03/rows.parquet` | `aea6a87942090789129ab6e91215d61870655fc3053a8e3f12d216b45b3ea9ff` | 7797701 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-03/row_features.f16.npy` | `71aca8a35c1d06a0c003096a42e5e7edd67b4c13c8344f455d530bafad353c88` | 88109984 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_labels/MOT20-03/manifest.json` | `ba8b63a70597662922b620cc44db02c4c9052a89dfdac8a3f966b8b12af61295` | 2930 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/boundary_universe/MOT20-03/chunk_membership.parquet` | `b1569d8534522b27e3c3876fd1a967124f131f16eb59c7481ff1007272cc9198` | 3459225 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-03/frozen_candidate_graph/nodes.parquet` | `ec2e0ba75b96a4e33c169a98a4f0c90612e973e1dc659d0830939e886bb42609` | 1117373 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-03/frozen_candidate_graph/edges.parquet` | `29dd40629e9bd5934e867f25df1d5abbabb22c020e8254a0790cb8ef11a22c7f` | 134697564 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-03/teacher_identity_flow/teacher_edge_utilities.parquet` | `9e02d3567ea68edead85a19d217215af4f336c282298d7fcf6d747bf3fad0e54` | 147175758 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-03/postfreeze_audit/successor_events.parquet` | `d61956989d8462affe4c01fb6639829dd89c541f0f19a684c4122cfbf8a92a42` | 243522 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-03/model.pt` | `51c1c6b8b37d378fb0b039a3ac9eb2b01481a52e6b75c17d1f1b35ceed7ec708` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-03/model_frozen_before_validation_labels.json` | `329ec03ecd42de0f22497dd957ab5e1d91c6f963e3e1aeb9d4819e449cc9f3eb` | 2719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-03/model.pt` | `611037f724feab93015d8448a5fdf4ec73adb92ef5bfd452a0da1a75e73aabb0` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-03/model_frozen_before_validation_labels.json` | `8f2da41dffc8135937a2f924be4b4ab456382ed7a20021a79e20ef98117cf20a` | 2719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-03/model.pt` | `ce6590acb901def3791c488763609a7ca34b3c201983652df47fa9b93e1813aa` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-05/inner_valid_MOT20-03/model_frozen_before_validation_labels.json` | `3f2538bb56b14d4132f578532d3faf4c22fbb2c4380862db116fed6776110e33` | 2729 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-05/manifest.json` | `0784a0eff440d1d12756f45440c018be8725f6981dd66dcd84f8df2e921cfa6e` | 1851 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-05/rows.parquet` | `631387d1c58e61abb4a0f7abe659057f7c08206f65ad0b55eadf33419e10fafa` | 16756159 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_observable/MOT20-05/row_features.f16.npy` | `0eaf8263ea3cd7a6943dd97671321fed699e0ddb6dca1a0bd85d9189ed38aa46` | 183913472 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/mot20_labels/MOT20-05/manifest.json` | `360bf59652f243de9944ab8742b76f7bf649788f1903d2c5da7af15519852529` | 2931 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/boundary_universe/MOT20-05/chunk_membership.parquet` | `cad73b818e59a3cf4a0a7725d14deff84effaf276b173fb4435ee4fb016fecdb` | 6959313 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-05/frozen_candidate_graph/nodes.parquet` | `2e9a7726e61f577d4fdd2cb712f87835cb747e85d1b5fb567068e42391eca093` | 2303622 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-05/frozen_candidate_graph/edges.parquet` | `a714f891b92c27f25bbf6fae09dbab7ba72ff4116fcc5954d55b04159833d6a1` | 308003247 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-05/teacher_identity_flow/teacher_edge_utilities.parquet` | `1cbfee6e0e968939d7a988bb78c767c9fae6b7c926b7b14a8e6ec4edf5071203` | 338470988 |
| `outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2/capacity/MOT20-05/postfreeze_audit/successor_events.parquet` | `41c83895ae871ba17027ec65d67cff40c5340379f3c55cd9996f3ad82a49f513` | 486242 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-05/model.pt` | `5b54514006d5a0f70a62f9e54364e62f4a54ac8e0f124960e7f6cbc6f12a0282` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-01/inner_valid_MOT20-05/model_frozen_before_validation_labels.json` | `c87042172b9c22188a74a79b03728780933b77d0f5d60b8c3cecf2c32c8b7145` | 2729 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-05/model.pt` | `b676f714738ff455f3fd39dd667f9c3a2516cecfa48965d647c82050f1014c2c` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-02/inner_valid_MOT20-05/model_frozen_before_validation_labels.json` | `502a425b702bb9acbd8c951ae817f995cc959c5d057fe1e39a8fc55378f3657d` | 2719 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-05/model.pt` | `6c3462e6c1c0d034466a1e38e35133300e6c44e7a691022239e422acb46624e0` | 3534225 |
| `outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v2/nested_loso/MOT20-03/inner_valid_MOT20-05/model_frozen_before_validation_labels.json` | `abec0347d3198dae691e14fbed04ccb0a5c9078bf82e95bed618f878929ab598` | 2729 |
