#!/bin/bash
# Run BoT-SORT + ReentryQueryEngine on DanceTrack val, one sequence at a time.
# Avoids OOM by killing the Python process between sequences.

set -euo pipefail

DATA_ROOT="/gemini/code/datasets/DanceTrack/extracted"
BOT_ROOT="/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main"
EXP_NAME="reentry_engine_only_dancetrack_val_20260529"
OUT_DIR="${BOT_ROOT}/YOLOX_outputs/${EXP_NAME}/track_results"
SPLIT="val"

mkdir -p "$OUT_DIR"

cd "$BOT_ROOT"

SEQUENCES=(dancetrack0004 dancetrack0005 dancetrack0007 dancetrack0010 dancetrack0014
           dancetrack0018 dancetrack0019 dancetrack0025 dancetrack0026 dancetrack0030
           dancetrack0034 dancetrack0035 dancetrack0041 dancetrack0043 dancetrack0047
           dancetrack0058 dancetrack0063 dancetrack0065 dancetrack0073 dancetrack0077
           dancetrack0079 dancetrack0081 dancetrack0090 dancetrack0094 dancetrack0097)

TOTAL=${#SEQUENCES[@]}
COUNT=0

for SEQ in "${SEQUENCES[@]}"; do
    COUNT=$((COUNT + 1))
    RESULT_FILE="${OUT_DIR}/${SEQ}.txt"
    if [ -f "$RESULT_FILE" ] && [ -s "$RESULT_FILE" ]; then
        echo "[${COUNT}/${TOTAL}] ${SEQ} — already done, skipping"
        continue
    fi

    echo "[${COUNT}/${TOTAL}] ${SEQ} — starting tracking..."

    # Compute seq_id from directory name
    SEQ_ID=$(echo "$SEQ" | sed 's/dancetrack//;s/^0*//' | tr -d '\n')

    python tools/track.py "$DATA_ROOT" \
        --benchmark DanceTrack \
        --eval "$SPLIT" \
        --seq-ids "$SEQ_ID" \
        -f ./yolox/exps/example/mot/yolox_x_mix_det.py \
        -c ./pretrained/bytetrack_x_mot17.pth.tar \
        --with-reid \
        --fast-reid-config fast_reid/configs/MOT17/sbs_S50.yml \
        --fast-reid-weights pretrained/mot17_sbs_S50.pth \
        --cmc-method none \
        --experiment-name "$EXP_NAME" \
        --track_high_thresh 0.6 \
        --track_low_thresh 0.1 \
        --new_track_thresh 0.7 \
        --track_buffer 30 \
        --match_thresh 0.8 \
        --proximity_thresh 0.5 \
        --appearance_thresh 0.25 \
        --reentry-memory-max-gap 90 \
        --reentry-memory-max-size 256 \
        --reentry-memory-min-similarity 0.6 \
        --reentry-memory-confirm-streak 2 \
        --reentry-memory-confirm-gap 2 \
        --reentry-memory-confirm-min-similarity 0.68 \
        --reentry-memory-min-det-score 0.1 \
        --reentry-memory-appearance-weight 0.55 \
        --reentry-memory-iou-weight 0.25 \
        --reentry-memory-score-weight 0.1 \
        --reentry-memory-gap-weight 0.1 \
        --reentry-engine-enable \
        --reentry-engine-hilbert-order 8 \
        --reentry-engine-bf-threshold 50 \
        --reentry-engine-spatial-radius 2 \
        --reentry-engine-max-spatial-radius 4 \
        2>&1 | tail -5

    echo "[${COUNT}/${TOTAL}] ${SEQ} — done"
    sleep 2  # let GPU memory settle
done

echo "=== All ${TOTAL} sequences complete ==="
echo "Results in: $OUT_DIR"
ls -la "$OUT_DIR"/*.txt 2>/dev/null | wc -l
