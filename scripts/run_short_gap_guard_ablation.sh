#!/bin/bash
# Run short-gap guard ablation on DanceTrack val.
# Three variants: gap<=20, gap<=30, gap<=30 with stricter confirm.
# Each variant runs per-sequence to avoid OOM.

set -euo pipefail

DATA_ROOT="/gemini/code/datasets/DanceTrack/extracted"
BOT_ROOT="/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main"
SPLIT="val"

cd "$BOT_ROOT"

SEQUENCES=(dancetrack0004 dancetrack0005 dancetrack0007 dancetrack0010 dancetrack0014
           dancetrack0018 dancetrack0019 dancetrack0025 dancetrack0026 dancetrack0030
           dancetrack0034 dancetrack0035 dancetrack0041 dancetrack0043 dancetrack0047
           dancetrack0058 dancetrack0063 dancetrack0065 dancetrack0073 dancetrack0077
           dancetrack0079 dancetrack0081 dancetrack0090 dancetrack0094 dancetrack0097)

# Base engine args (same as engine_full but with short_gap_threshold)
BASE_ARGS=(
    --reentry-memory-max-gap 60
    --reentry-memory-max-size 256
    --reentry-memory-min-similarity 0.60
    --reentry-memory-confirm-streak 2
    --reentry-memory-confirm-gap 2
    --reentry-memory-confirm-min-similarity 0.65
    --reentry-memory-min-det-score 0.10
    --reentry-memory-appearance-weight 0.55
    --reentry-memory-iou-weight 0.25
    --reentry-memory-score-weight 0.10
    --reentry-memory-gap-weight 0.10
    --reentry-engine-hilbert-order 8
    --reentry-engine-bf-threshold 50
    --reentry-engine-spatial-radius 2
    --reentry-engine-max-spatial-radius 4
)

declare -A VARIANTS
VARIANTS[short_gap_20]="--reentry-engine-short-gap-threshold 20"
VARIANTS[short_gap_30]="--reentry-engine-short-gap-threshold 30"

run_variant() {
    local variant_name="$1"
    shift
    local extra_args=("$@")
    local exp_name="engine_${variant_name}_dancetrack_val_20260529"
    local out_dir="${BOT_ROOT}/YOLOX_outputs/${exp_name}/track_results"

    mkdir -p "$out_dir"

    echo "=== Variant: ${variant_name} (${exp_name}) ==="

    local total=${#SEQUENCES[@]}
    local count=0
    local skipped=0

    for seq in "${SEQUENCES[@]}"; do
        count=$((count + 1))
        local result_file="${out_dir}/${seq}.txt"
        if [ -f "$result_file" ] && [ -s "$result_file" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        local seq_id=$(echo "$seq" | sed 's/dancetrack//;s/^0*//' | tr -d '\n')

        echo "  [${count}/${total}] ${seq}..."

        python tools/track.py "$DATA_ROOT" \
            --benchmark DanceTrack \
            --eval "$SPLIT" \
            --seq-ids "$seq_id" \
            -f ./yolox/exps/example/mot/yolox_x_mix_det.py \
            -c ./pretrained/bytetrack_x_mot17.pth.tar \
            --with-reid \
            --fast-reid-config fast_reid/configs/MOT17/sbs_S50.yml \
            --fast-reid-weights pretrained/mot17_sbs_S50.pth \
            --cmc-method none \
            --experiment-name "$exp_name" \
            --track_high_thresh 0.6 \
            --track_low_thresh 0.1 \
            --new_track_thresh 0.7 \
            --track_buffer 30 \
            --match_thresh 0.8 \
            --proximity_thresh 0.5 \
            --appearance_thresh 0.25 \
            --reentry-memory-enable \
            --reentry-engine-enable \
            "${BASE_ARGS[@]}" \
            "${extra_args[@]}" \
            2>&1 | tail -3

        sleep 2
    done
    echo "  Done: ${count} sequences (${skipped} skipped)"
    echo "  Results: $out_dir"
    echo ""
}

# Run both variants
for variant in "${!VARIANTS[@]}"; do
    run_variant "$variant" ${VARIANTS[$variant]}
done

echo "=== All short-gap guard variants complete ==="
