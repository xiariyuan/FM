#!/usr/bin/env bash
# Run SPOT soft appearance-update sweep on MOT20-05 with BoT-SORT ReID.
# Reuses existing baseline metrics from the previous SPOT_REID_BASELINE_MOT20_05 run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOT_ROOT="${REPO_ROOT}/external/BoT-SORT-main"
DATA_ROOT="${DATA_ROOT:-/gemini/code/datasets/MOT20}"
GT_ROOT="${GT_ROOT:-/gemini/code/datasets/MOT20/train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_RUN_ID="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/spot_runtime/mot20_05_reid_soft_${DEFAULT_RUN_ID}}"
MARGIN_THRESH="${SPOT_MARGIN_THRESH:-0.05}"
REID_CFG="${REID_CFG:-fast_reid/configs/MOT20/sbs_S50.yml}"
REID_WTS="${REID_WTS:-pretrained/mot20_sbs_S50.pth}"
BASE_EXP="${BASE_EXP:-SPOT_REID_BASELINE_MOT20_05}"

mkdir -p "${OUT_ROOT}"
cd "${REPO_ROOT}"
{
  echo "START $(date -Iseconds)"
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "BOT_ROOT=${BOT_ROOT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "GT_ROOT=${GT_ROOT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "GIT_SHA=$(git rev-parse HEAD)"
  echo "REID_CFG=${REID_CFG}"
  echo "REID_WTS=${REID_WTS}"
  echo "SPOT_MARGIN_THRESH=${MARGIN_THRESH}"
  echo "BASE_EXP=${BASE_EXP}"
} | tee "${OUT_ROOT}/status.txt"

cd "${BOT_ROOT}"
export PYTHONPATH="${BOT_ROOT}:${PYTHONPATH:-}"
COMMON_ARGS=(
  "${DATA_ROOT}"
  --benchmark MOT20
  --eval train
  --seq-ids 5
  --default-parameters
  --with-reid
  --fast-reid-config "${REID_CFG}"
  --fast-reid-weights "${REID_WTS}"
  --device gpu
)

run_soft() {
  local name="$1"
  local alpha="$2"
  local min_age="$3"
  local max_score="$4"
  local exp="SPOT_REID_${name}_MOT20_05"
  local debug_dir="${OUT_ROOT}/${name}/spot_debug"
  echo "RUN ${name} alpha=${alpha} min_age=${min_age} max_score=${max_score} $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"
  "${PYTHON_BIN}" -u tools/track.py \
    "${COMMON_ARGS[@]}" \
    --experiment-name "${exp}" \
    --spot-enable \
    --spot-margin-thresh "${MARGIN_THRESH}" \
    --spot-soft-app-alpha "${alpha}" \
    --spot-soft-min-track-age "${min_age}" \
    --spot-soft-max-det-score "${max_score}" \
    --spot-debug-dir "${debug_dir}" \
    > "${OUT_ROOT}/${name}.log" 2>&1
  echo "TRACK_DONE ${name} $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"

  cd "${REPO_ROOT}"
  local eval_out="${OUT_ROOT}/eval_${name}_mot20_05"
  rm -rf "${eval_out}" "${eval_out}.log"
  "${PYTHON_BIN}" scripts/eval_motstyle_trackeval.py \
    --benchmark-name MOT20 \
    --split-to-eval train \
    --gt-root "${GT_ROOT}" \
    --results-dir "${BOT_ROOT}/YOLOX_outputs/${exp}/track_results" \
    --tracker-name "${exp}" \
    --work-dir "${eval_out}" \
    --keep-workdir \
    --seqs MOT20-05 \
    > "${eval_out}.log" 2>&1
  echo "EVAL_DONE ${name} $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"
  cd "${BOT_ROOT}"
}

run_soft SOFT099 0.99 0 1.01
run_soft SOFT097 0.97 0 1.01
run_soft GATED099 0.99 30 0.75

cd "${REPO_ROOT}"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import csv, os
run = Path(os.environ.get('OUT_ROOT', ''))
if not str(run):
    raise SystemExit('OUT_ROOT env missing')
base = Path('outputs/spot_runtime/mot20_05_reid_c31c732/eval_baseline_mot20_05/eval/SPOT_REID_BASELINE_MOT20_05/pedestrian_summary.txt')
items = {
    'baseline': base,
    'SOFT099': run/'eval_SOFT099_mot20_05/eval/SPOT_REID_SOFT099_MOT20_05/pedestrian_summary.txt',
    'SOFT097': run/'eval_SOFT097_mot20_05/eval/SPOT_REID_SOFT097_MOT20_05/pedestrian_summary.txt',
    'GATED099': run/'eval_GATED099_mot20_05/eval/SPOT_REID_GATED099_MOT20_05/pedestrian_summary.txt',
}
rows = {}
for k,p in items.items():
    if not p.exists():
        continue
    lines = p.read_text().strip().splitlines()
    hdr = lines[0].split(); vals = lines[1].split()
    rows[k] = {h: vals[i] for i,h in enumerate(hdr)}
metrics = ['HOTA','DetA','AssA','MOTA','IDF1','IDSW','Frag','CLR_FP','CLR_FN','Dets']
out = run/'soft_metrics_delta.csv'
with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['metric'] + list(rows.keys()) + ['best_delta_vs_baseline'])
    for m in metrics:
        vals = [rows[k].get(m,'') for k in rows]
        deltas = []
        if 'baseline' in rows:
            b = float(rows['baseline'][m])
            for k in rows:
                if k == 'baseline':
                    continue
                deltas.append(float(rows[k][m]) - b)
        w.writerow([m] + vals + ([max(deltas) if deltas else '']))
print(out)
PY

echo "DONE $(date -Iseconds)" | tee -a "${OUT_ROOT}/status.txt"
touch "${OUT_ROOT}/DONE"
