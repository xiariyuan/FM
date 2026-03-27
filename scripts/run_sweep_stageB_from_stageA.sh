#!/bin/bash
set -euo pipefail

# Run association sweep Stage B (IOU gate × miss tolerance) after Stage A (id_thresh × feat_tau) finishes.
#
# Usage:
#   bash scripts/run_sweep_stageB_from_stageA.sh \
#     <stageA_out_root> <expected_runs> <config_path> <checkpoint> <data_root> <val_sequences_csv>
#
# Example:
#   bash scripts/run_sweep_stageB_from_stageA.sh \
#     outputs/sweep_proxy021013_idtau_stageA_xxx 12 \
#     configs/bytetrack_fa_mot_mot17_proxy_alltrainFRCNN_v10e1_privdet_reid512_feature_mean_detscore1.yaml \
#     outputs/.../checkpoint_epoch_1.pth /gemini/code/datasets \
#     MOT17-02-FRCNN,MOT17-10-FRCNN,MOT17-13-FRCNN

STAGEA_DIR="${1:?stageA_out_root required}"
EXPECTED_RUNS="${2:?expected_runs required}"
CONFIG_PATH="${3:?config_path required}"
CHECKPOINT="${4:?checkpoint required}"
DATA_ROOT="${5:?data_root required}"
VAL_SEQS="${6:?val_sequences_csv required}"

STAGEA_CSV="${STAGEA_DIR}/sweep_assoc_results.csv"

echo "[stageB] Waiting for Stage A to finish..."
while true; do
  if [[ -f "${STAGEA_CSV}" ]]; then
    # Header line + N results
    LINES="$(wc -l < "${STAGEA_CSV}")"
    DONE="$((LINES - 1))"
    if [[ "${DONE}" -ge "${EXPECTED_RUNS}" ]]; then
      break
    fi
    echo "[stageB] Stage A progress: ${DONE}/${EXPECTED_RUNS} runs finished"
  else
    echo "[stageB] Stage A csv not found yet: ${STAGEA_CSV}"
  fi
  sleep 60
done

echo "[stageB] Stage A complete. Selecting best (id_thresh, assoc_feat_tau) by (HOTA, AssA, -IDSW)..."
read -r BEST_ID BEST_TAU <<<"$(
python - <<PY
import csv
path = "${STAGEA_CSV}"
best = None
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ok = str(r.get("ok", "")).lower() in ("true", "1", "yes", "y")
        if not ok:
            continue
        try:
            hota = float(r.get("HOTA", "nan"))
            assa = float(r.get("AssA", "nan"))
            idsw = float(r.get("IDSW", "1e18"))
            id_thr = float(r.get("id_thresh", "nan"))
            ftau = float(r.get("assoc_feat_tau", "nan"))
        except Exception:
            continue
        score = (hota, assa, -idsw)
        if best is None or score > best[0]:
            best = (score, id_thr, ftau)
if best is None:
    raise SystemExit("No successful runs found in Stage A")
print(f"{best[1]:.6g} {best[2]:.6g}")
PY
)"
echo "[stageB] BEST_ID=${BEST_ID} BEST_TAU=${BEST_TAU}"

STAGEB_DIR="outputs/sweep_stageB_iou_miss_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${STAGEB_DIR}"
echo "[stageB] Running Stage B -> ${STAGEB_DIR}"

python -u scripts/sweep_assoc_params.py \
  --config-path "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --data-root "${DATA_ROOT}" \
  --dataset MOT17 \
  --split train \
  --out-root "${STAGEB_DIR}" \
  --eval-only-val \
  --val-sequences "${VAL_SEQS}" \
  --detector-filter FRCNN \
  --id-thresh-list "${BEST_ID}" \
  --assoc-feat-tau-list "${BEST_TAU}" \
  --assoc-iou-gate-list 0.2,0.25,0.3,0.35 \
  --det-max-per-frame-list 200 \
  --miss-tolerance-list 30,50 \
  --keep-going |& tee "${STAGEB_DIR}/sweep.log"

echo "[stageB] Done. Best row (HOTA,AssA,DetA,-IDSW):"
python - <<PY
import csv
path = "${STAGEB_DIR}/sweep_assoc_results.csv"
best = None
with open(path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        ok = str(r.get("ok", "")).lower() in ("true", "1", "yes", "y")
        if not ok:
            continue
        try:
            hota = float(r.get("HOTA", "nan"))
            assa = float(r.get("AssA", "nan"))
            deta = float(r.get("DetA", "nan"))
            idsw = float(r.get("IDSW", "1e18"))
        except Exception:
            continue
        score = (hota, assa, deta, -idsw)
        if best is None or score > best[0]:
            best = (score, r)
if best is None:
    raise SystemExit("No ok rows in Stage B")
r = best[1]
keys = ["HOTA","DetA","AssA","IDF1","IDSW","id_thresh","assoc_feat_tau","assoc_iou_gate","miss_tolerance","det_thresh","newborn_thresh","det_max_per_frame"]
print("BEST_STAGEB")
for k in keys:
    if k in r and r[k] != "":
        print(f"{k}={r[k]}")
PY

