# BoT-SORT-LTRA Mainline

This document is the execution anchor for the new paper mainline:

- carrier tracker: `BoT-SORT`
- only novelty: `LTRA` (Laplace-Guided Temporal Reliability Association)
- ReID: standard component
- detector: fixed standard component

## What is no longer mainline

The following directions are no longer part of the primary paper story:

- `FM-Track`
- frequency decomposition
- frequency transformer
- Mamba integration
- SFI line
- trainable Laplace gate line

Those code paths remain in the repository for reference only.

## Stage 1 goals

The first stage is a strict same-base comparison:

1. `BoT-SORT`
2. `BoT-SORT + LTRA`

Run on:

- `MOT17` validation half
- `MOT20` validation half

Keep fixed:

- detector checkpoint
- ReID weights
- baseline tracker hyperparameters
- matching thresholds

Only toggle:

- `--laplace-assoc`

## Metrics that matter most

Report these first:

1. `HOTA`
2. `AssA`
3. `IDF1`
4. `IDSW`
5. `Frag`

`MOTA` and `DetA` are supporting metrics, not the headline metrics.

## Lock-line thresholds

The mainline is worth locking if the same-base BoT-SORT comparison shows roughly:

- `HOTA +0.8` or better
- `AssA +1.5` or better
- `IDSW` relative reduction around `8%` or better

and these gains are not confined to only one easy sequence.

## Recommended commands

### 1. Run MOT17 validation-half base vs LTRA

```bash
bash scripts/run_botsort_ltra_stage1.sh MOT17
```

### 2. Run MOT20 validation-half base vs LTRA

```bash
bash scripts/run_botsort_ltra_stage1.sh MOT20
```

### 3. Run both

```bash
bash scripts/run_botsort_ltra_stage1.sh all
```

### 4. Launch in background

```bash
bash scripts/launch_botsort_ltra_stage1_bg.sh MOT17
bash scripts/launch_botsort_ltra_stage1_bg.sh MOT20
```

## Output locations

- raw BoT-SORT outputs:
  - `external/BoT-SORT-main/YOLOX_outputs/laplace_mot17_val_base`
  - `external/BoT-SORT-main/YOLOX_outputs/laplace_mot17_val_laplace`
  - `external/BoT-SORT-main/YOLOX_outputs/laplace_mot20_val_base`
  - `external/BoT-SORT-main/YOLOX_outputs/laplace_mot20_val_laplace`

- TrackEval summaries:
  - `outputs/botsort_ltra_stage1/MOT17/summary.csv`
  - `outputs/botsort_ltra_stage1/MOT20/summary.csv`

## Next steps after Stage 1

If Stage 1 is positive:

1. run the minimal ablation ladder
2. run StrongSORT transfer
3. lock one LTRA setting
4. submit full MOT17/MOT20 benchmark runs
