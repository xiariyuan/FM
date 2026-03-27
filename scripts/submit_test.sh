#!/bin/bash
# ============================================================================
# FA-MOT V3 Test Submission Script
#
# Generates submission.zip for MOTChallenge/Codabench
#
# Usage:
#   bash scripts/submit_test.sh <CHECKPOINT> [DATASET] [DATA_ROOT]
#
# Examples:
#   bash scripts/submit_test.sh ./outputs/fa_mot_v3_robust/checkpoint_best.pth
#   bash scripts/submit_test.sh ./outputs/fa_mot_v3_robust/checkpoint_best.pth MOT17
#   bash scripts/submit_test.sh ./outputs/fa_mot_v3_robust/checkpoint_best.pth MOT20
# ============================================================================

set -e

# Parameters
CHECKPOINT=${1:?"Error: Checkpoint path required. Usage: $0 <checkpoint>"}
DATASET=${2:-MOT17}
DATA_ROOT=${3:-/gemini/code/datasets}

# Output directory
CKPT_NAME=$(basename $CHECKPOINT .pth)
OUTPUT_DIR="./outputs/submit_${DATASET}_test_${CKPT_NAME}"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "FA-MOT V3 Test Submission Generation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET"
echo "Data Root:  $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Run submission generation with public detections
python submit_public.py \
    --config-path configs/r50_dino_fa_mot_v3_robust.yaml \
    --inference-model $CHECKPOINT \
    --inference-mode submit \
    --inference-dataset $DATASET \
    --inference-split test \
    --outputs-dir $OUTPUT_DIR \
    --data-root $DATA_ROOT

# Verify output files
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

# Count output files
TRACKER_DIR="$OUTPUT_DIR/tracker/tracker_default/data"
if [ -d "$TRACKER_DIR" ]; then
    FILE_COUNT=$(ls -1 $TRACKER_DIR/*.txt 2>/dev/null | wc -l)
    echo "Generated files: $FILE_COUNT"

    # Expected counts
    if [ "$DATASET" == "MOT17" ]; then
        EXPECTED=21
    elif [ "$DATASET" == "MOT20" ]; then
        EXPECTED=12
    else
        EXPECTED="unknown"
    fi
    echo "Expected files:  $EXPECTED"

    if [ "$FILE_COUNT" -eq "$EXPECTED" ]; then
        echo "[OK] File count matches expected."
    else
        echo "[WARNING] File count mismatch!"
    fi
else
    echo "[ERROR] Tracker output directory not found!"
    exit 1
fi

# Check zip file
ZIP_PATH="$OUTPUT_DIR/submission.zip"
if [ -f "$ZIP_PATH" ]; then
    echo ""
    echo "Submission zip: $ZIP_PATH"
    echo "Zip contents:"
    unzip -l $ZIP_PATH | head -20
else
    echo "[WARNING] submission.zip not found!"
fi

echo ""
echo "=============================================="
echo "Submission generation completed!"
echo "Upload $ZIP_PATH to MOTChallenge/Codabench"
echo "=============================================="
