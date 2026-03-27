# Reproducibility (Auto-Generated)

## Best Checkpoint Selection
- Metric: HOTA
- Best experiment: bytetrack_fa_mot_mot17_v2
- Best epoch: 26.0
- Best HOTA: 35.0670
- Summary file: /gemini/code/FMtrack-main/FM-Track/outputs/bytetrack_fa_mot_mot17_v2/val/epoch_26/tracker/MOT17-train/pedestrian_summary.txt

## Config
- Source: /gemini/code/FMtrack-main/FM-Track/outputs/bytetrack_fa_mot_mot17_v2/config_effective.yaml
- Saved as: /gemini/code/FMtrack-main/FM-Track/configs/bytetrack_fa_mot_mot17_best.yaml

## Dataset & Split
- DATA_ROOT: /gemini/code/datasets
- DATASETS: ['MOT17']
- DATASET_SPLITS: ['train']
- DETECTOR_FILTER: ['FRCNN']
- VAL_SEQUENCES: ['MOT17-04', 'MOT17-05']

## Model & Training
- SEED: 42
- FEATURE_DIM: 256
- NUM_BANDS: 4
- MAX_SEQ_LEN: 36
- EPOCHS: 50
- BATCH_SIZE: 2
- LR: 0.0001
- WEIGHT_DECAY: 0.0001
- USE_FREQ_AWARE: True
- USE_FREQ_DECODER_V2: True

## Commands
- Train:
  ```bash
  cd /gemini/code/FMtrack-main/FM-Track
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   /root/miniconda3/bin/python -u train_bytetrack.py     --config-path /gemini/code/FMtrack-main/FM-Track/configs/bytetrack_fa_mot_mot17_best.yaml
  ```
- Evaluate:
  ```bash
  cd /gemini/code/FMtrack-main/FM-Track
  /root/miniconda3/bin/python -u submit_bytetrack.py     --config-path /gemini/code/FMtrack-main/FM-Track/configs/bytetrack_fa_mot_mot17_best.yaml
  ```
