#!/bin/bash
# ============================================================================
# FA-MOT V3 Training Script
#
# Usage:
#   bash scripts/train_v3.sh [EPOCHS] [DATA_ROOT] [OUTPUT_DIR]
#
# Examples:
#   bash scripts/train_v3.sh                        # Default: 20 epochs
#   bash scripts/train_v3.sh 2                      # Smoke test: 2 epochs
#   bash scripts/train_v3.sh 40 /path/to/datasets   # Full training
# ============================================================================

set -e

# Default parameters
EPOCHS=${1:-20}
DATA_ROOT=${2:-/gemini/code/datasets}
OUTPUT_DIR=${3:-./outputs/fa_mot_v3_robust}

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "FA-MOT V3 Training"
echo "=============================================="
echo "Epochs:     $EPOCHS"
echo "Data Root:  $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Check if weight file exists
WEIGHT_PATH="./weight/checkpoint0089.pth"
if [ ! -f "$WEIGHT_PATH" ]; then
    echo "[WARNING] Pretrained weight not found at $WEIGHT_PATH"
    echo "Please download and place the weight file, or modify DETR_PRETRAIN in config."
fi

# Start training
echo "Starting training..."
python train.py \
    --config-path configs/r50_dino_fa_mot_v3_robust.yaml \
    --outputs-dir $OUTPUT_DIR \
    --data-root $DATA_ROOT \
    --epochs $EPOCHS

echo "Training completed! Results saved to $OUTPUT_DIR"
