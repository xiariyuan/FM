#!/usr/bin/env bash
set -euo pipefail
cd /gemini/code/FMtrack-main/FM-Track
OUT_ROOT="outputs/detector_rebuild"
RUN_DIR="$OUT_ROOT/mixed_human_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
ln -sfn "$(basename "$RUN_DIR")" "$OUT_ROOT/mixed_human_quick_latest"
echo "RUN_DIR=$RUN_DIR" | tee "$RUN_DIR/status.txt"
echo "START=$(date -Iseconds)" | tee -a "$RUN_DIR/status.txt"
echo $$ > "$RUN_DIR/PID"
cd external/BoT-SORT-main
export MIX_DET_INPUT_H=${MIX_DET_INPUT_H:-640}
export MIX_DET_INPUT_W=${MIX_DET_INPUT_W:-1152}
export MIX_DET_TEST_H=${MIX_DET_TEST_H:-640}
export MIX_DET_TEST_W=${MIX_DET_TEST_W:-1152}
export MIX_DET_MAX_EPOCH=${MIX_DET_MAX_EPOCH:-2}
export MIX_DET_EVAL_INTERVAL=${MIX_DET_EVAL_INTERVAL:-1}
export MIX_DET_WORKERS=${MIX_DET_WORKERS:-2}
export MIX_DET_LR_PER_IMG=${MIX_DET_LR_PER_IMG:-0.00025}
export MIX_DET_SMOKE_TRAIN_IMAGES=${MIX_DET_SMOKE_TRAIN_IMAGES:-3000}
export MIX_DET_SMOKE_VAL_IMAGES=${MIX_DET_SMOKE_VAL_IMAGES:-1200}
PYTHONPATH=. PYTHONPATH=. python -u yolox/train.py \
  -f yolox/exps/example/mot/yolox_x_mixed_human_quick.py \
  -d 1 -b 1 \
  -c pretrained/bytetrack_x_mot20.pth.tar \
  --experiment-name MIXED_HUMAN_QUICK \
  --fp16 2>&1 | tee "../../$RUN_DIR/launcher.log"
