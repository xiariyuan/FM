#!/bin/bash
# ============================================================================
# FA-MOT V3 Evaluation Script (Public Detection)
#
# Usage:
#   bash scripts/evaluate_v3.sh <CHECKPOINT> [SPLIT] [DATASET] [DATA_ROOT]
#
# Examples:
#   bash scripts/evaluate_v3.sh ./outputs/fa_mot_v3_robust/checkpoint_20.pth
#   bash scripts/evaluate_v3.sh ./outputs/fa_mot_v3_robust/checkpoint_20.pth val
#   bash scripts/evaluate_v3.sh ./outputs/fa_mot_v3_robust/checkpoint_20.pth train MOT17
# ============================================================================

set -e

# Parameters
CHECKPOINT=${1:?"Error: Checkpoint path required. Usage: $0 <checkpoint>"}
SPLIT=${2:-val}
DATASET=${3:-MOT17}
DATA_ROOT=${4:-/gemini/code/datasets}

# Output directory based on checkpoint
CKPT_NAME=$(basename $CHECKPOINT .pth)
OUTPUT_DIR="./outputs/eval_public_${DATASET}_${SPLIT}_${CKPT_NAME}"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "FA-MOT V3 Public Detection Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Split:      $SPLIT"
echo "Data Root:  $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Run evaluation with public detections
python submit_public.py \
    --config-path configs/r50_dino_fa_mot_v3_robust.yaml \
    --inference-model $CHECKPOINT \
    --inference-mode evaluate \
    --inference-dataset $DATASET \
    --inference-split $SPLIT \
    --outputs-dir $OUTPUT_DIR \
    --data-root $DATA_ROOT

echo "Evaluation completed! Results saved to $OUTPUT_DIR"

# Run TrackEval if on train/val split
if [ "$SPLIT" != "test" ]; then
    echo "Running TrackEval..."
    python TrackEval/scripts/run_mot_challenge.py \
        --BENCHMARK $DATASET \
        --SPLIT_TO_EVAL $SPLIT \
        --TRACKERS_TO_EVAL tracker_default \
        --TRACKER_SUB_FOLDER data \
        --GT_FOLDER $DATA_ROOT/$DATASET/$SPLIT \
        --TRACKERS_FOLDER $OUTPUT_DIR/tracker \
        --SEQMAP_FILE $OUTPUT_DIR/seqmap.txt \
        --METRICS HOTA CLEAR Identity \
        --USE_PARALLEL False \
        --SKIP_SPLIT_FOL True
fi
