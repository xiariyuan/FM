# FM-Track (FA-MOT) - Reproducible Training & Evaluation

This repository contains the FM-Track codebase for frequency-aware multi-object tracking.
Below is a minimal, reviewer-friendly quickstart to reproduce training and evaluation.

## 1) Environment

Python 3.9+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- If you use the non-frequency-aware path, `mamba_ssm` is optional.
- For the frequency-aware decoder, `mamba_ssm` is required.

## 2) Data Preparation

Prepare datasets following the dataset structure used in this repo. Typical MOT datasets:
- MOT17 / MOT20 (train/val/test splits)
- CrowdHuman / additional detection data (optional)

Please ensure the dataset root paths are consistent with your config YAML.

## 3) Training (Standard / DINO-based)

```bash
python train.py --config_path configs/r50_dino_fa_mot_v2_mot17.yaml
```

Key configs live under `configs/`. Adjust:
- `DATA_ROOT`, `DATASETS`, `SPLITS`
- `BATCH_SIZE`, `MAX_EPOCH`, `LR`
- Frequency-aware switches: `USE_FREQ_AWARE`, `USE_FREQ_DECODER_V2`

## 4) Training (ByteTrack Feature Extractor)

```bash
python train_bytetrack.py --config_path configs/bytetrack_fa_mot_mot17.yaml
```

Make sure ByteTrack is available under `third_party/ByteTrack` and weights are set:
- `BYTETRACK_EXP_FILE`
- `BYTETRACK_CKPT`

## 5) Inference / Submission

```bash
python submit_bytetrack.py --config_path configs/bytetrack_fa_mot_mot17.yaml
```

Other submission scripts:
- `submit_public.py`
- `submit_and_evaluate.py`

## 6) Evaluation (TrackEval)

TrackEval is included under `TrackEval/`. Follow its official instructions
to evaluate your submission files.

## 7) Experiment Evidence

If you want the shortest path through the diagnosis and experiment history, start with:

- `docs/experiment_index.md`
- `outputs/experiment_registry.csv`

This repository intentionally keeps lightweight structured experiment records in git
while excluding heavy artifacts such as checkpoints, shard dumps, and raw dataset
outputs.

Useful helper scripts:

- `scripts/git_stage_experiment_records.sh`
- `scripts/git_sync_experiment_records.sh`

---

If you run into issues, check:
- `log/` for runtime logs
- `outputs/` for checkpoints and submission outputs
