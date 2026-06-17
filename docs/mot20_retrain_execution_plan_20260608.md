# MOT20 Retrain Execution Plan

## Why Current MOT20 Fails
- HACA v3 association head trained on MOT17 → does not generalize to MOT20 crowd patterns
- Stage1 deferral head trained on MOT17 pair distribution → MOT20 reject rate 54% vs MOT17 28%
- TCGAU freeze-only is domain-agnostic and NOT the cause of failure
- Detector/ReID are MOT20-native — failure is purely in the learned association/calibration heads

## What Must Be Retrained
1. HACA v1→v2→v3 on MOT20 pseudo-track data
2. RGSA oracle dump from HACA v3 MOT20
3. Stage1 deferral head on MOT20 labels

## What Transfers Without Retraining
- TCGAU freeze-only rule (no learned parameters, domain-agnostic)
- Evaluation infrastructure (TrackEval, eval scripts)
- BoT-SORT core tracker framework

## Existing MOT20 Assets
- MOT20 dataset: MOT20-{01,02,03,05} train, MOT20-{04,06,07,08} test
- GT splits: train_half / val_half for each train sequence
- Detector/ReID pretrained models
- All training/evaluation Python scripts

## Missing Assets
- GT pseudo-track NPZ shards for MOT20
- HACA v1/v2/v3 checkpoints for MOT20
- RGSA oracle dump pairbank for MOT20
- Stage1 labels for MOT20
- Stage1 trained checkpoint for MOT20

## Execution Order
1. `bash scripts/run_haca_v3_mot20_train.sh` (~2h)
   - Builds GT pseudo-track shards for train={01,02,03} and held-out val={05}
   - Trains HACA v1→v2→v3
   - Output: MOT20 HACA v3 checkpoint

2. Verify MOT20 HACA baseline on val_half
   - Should get reasonable HOTA (likely 60-68 range for MOT20)

3. `bash scripts/run_rgsa_stage1_mot20_train.sh` (~1h)
   - Runs oracle dump on MOT20-{01,02,03,05}
   - Builds labels
   - Trains Stage1 with train={01,02,03}, val={05}
   - Output: MOT20 Stage1 checkpoint

4. `bash scripts/run_stage1_freezeonly_mot20_eval.sh` (~30min)
   - Runs baseline / Stage1 / Stage1+freeze variants
   - Runs TrackEval automatically
   - Output: summary.csv with HOTA/IDSW per variant

## Key Metrics to Watch
- MOT20 HACA baseline HOTA (expect ~65-68)
- Stage1+freeze should beat Stage1 only (by ~0.3-0.7 HOTA)
- Stage1+freeze should NOT be worse than baseline
- freeze rate should be ~0.2-0.5% (similar to MOT17)

## Estimated Total Time
~4 hours on current hardware (12GB GPU, 16GB RAM cgroup)

## Risk
- MOT20 has only 4 train sequences → may underfit
- RAM limit: MOT20-04 is large, may need to build shards in smaller batches
- Sequence names must use plain `MOT20-01` style; trailing `-` names will break label/oracle paths
