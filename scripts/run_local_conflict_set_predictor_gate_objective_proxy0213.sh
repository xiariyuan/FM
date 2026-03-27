#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v18_local_conflict_set_predictor_val0213.yaml}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
DATA_JSONL="${DATA_JSONL:-${REPO_ROOT}/outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/cluster_set_predictor_data/cluster_examples.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-${REPO_ROOT}/outputs/local_conflict_set_predictor_large_base_stable_20260325_023500/source_manifest.csv}"
HOST_VARIANT="${HOST_VARIANT:-base_reid_da}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_set_predictor_gate_objective_proxy_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-MOT17-04,MOT17-05,MOT17-09,MOT17-10,MOT17-11}"
VAL_SEQUENCES="${VAL_SEQUENCES:-MOT17-02,MOT17-13}"
MIN_VAL_EXAMPLES="${MIN_VAL_EXAMPLES:-64}"
TOPK="${TOPK:-8}"
MIN_DETECTIONS="${MIN_DETECTIONS:-2}"
MIN_COMMITTED_MATCHES="${MIN_COMMITTED_MATCHES:-2}"
MAX_DETECTIONS="${MAX_DETECTIONS:-8}"
MAX_TRACKS="${MAX_TRACKS:-32}"
CLUSTER_GATE_THRESH="${CLUSTER_GATE_THRESH:-0.5}"
CLUSTER_GATE_CALIBRATION="${CLUSTER_GATE_CALIBRATION:-temp_bias}"
CLUSTER_GATE_SELECT_METRIC="${CLUSTER_GATE_SELECT_METRIC:-f0.5}"
CLUSTER_GATE_BETA="${CLUSTER_GATE_BETA:-0.5}"
CLUSTER_GATE_FP_WEIGHT="${CLUSTER_GATE_FP_WEIGHT:-2.0}"
CLUSTER_GATE_SEARCH_MIN="${CLUSTER_GATE_SEARCH_MIN:-0.50}"
CLUSTER_GATE_SEARCH_MAX="${CLUSTER_GATE_SEARCH_MAX:-0.95}"
CLUSTER_GATE_SEARCH_STEPS="${CLUSTER_GATE_SEARCH_STEPS:-19}"
CLUSTER_GATE_LOSS_MODE="${CLUSTER_GATE_LOSS_MODE:-weighted_bce}"
CLUSTER_GATE_POSITIVE_WEIGHT="${CLUSTER_GATE_POSITIVE_WEIGHT:-1.0}"
CLUSTER_GATE_NEGATIVE_WEIGHT="${CLUSTER_GATE_NEGATIVE_WEIGHT:-8.0}"
MODEL_SELECTION_METRIC="${MODEL_SELECTION_METRIC:-hybrid_gate_f0_5_loss}"
DATASET_TAG="${DATASET_TAG:-local_conflict_set_predictor_gate_objective_proxy0213}"
FEATURE_VERSION="${FEATURE_VERSION:-v2_hostnorm_geom}"

PROXY_BASELINE_HOTA="${PROXY_BASELINE_HOTA:-53.118}"
PROXY_BASELINE_ASSA="${PROXY_BASELINE_ASSA:-44.577}"
PROXY_BASELINE_IDF1="${PROXY_BASELINE_IDF1:-58.730}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi
if [[ ! -f "${DATA_JSONL}" ]]; then
  echo "Missing data jsonl: ${DATA_JSONL}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  echo "Missing source manifest: ${SOURCE_MANIFEST}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

STAGE1_OUT="${OUT_DIR}/01_stage1"
PROXY_OUT="${OUT_DIR}/02_proxy_eval"
FULL_OUT="${OUT_DIR}/03_full_eval_md2_mm2"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
PIPELINE_LOG="${OUT_DIR}/pipeline.log"

mkdir -p "${STAGE1_OUT}"

update_batch_row() {
  "${PYTHON_BIN}" - "${RESULT_CSV}" "${SUMMARY_CSV}" "${OUT_DIR}" "${DATA_JSONL}" "${SOURCE_MANIFEST}" "${BASE_CONFIG}" "${BASE_CKPT}" "${HOST_VARIANT}" "${STAGE1_OUT}" "${PROXY_OUT}" "${FULL_OUT}" "${TOPK}" "${MIN_DETECTIONS}" "${MIN_COMMITTED_MATCHES}" "${MAX_DETECTIONS}" "${MAX_TRACKS}" "${CLUSTER_GATE_CALIBRATION}" "${CLUSTER_GATE_SELECT_METRIC}" "${CLUSTER_GATE_BETA}" "${CLUSTER_GATE_FP_WEIGHT}" "${CLUSTER_GATE_LOSS_MODE}" "${CLUSTER_GATE_POSITIVE_WEIGHT}" "${CLUSTER_GATE_NEGATIVE_WEIGHT}" "${MODEL_SELECTION_METRIC}" "$1" "$2" "$3" "$4" <<'PY'
import csv
import sys
from pathlib import Path

(
    result_csv,
    summary_csv,
    out_dir,
    data_jsonl,
    source_manifest,
    base_config,
    base_ckpt,
    host_variant,
    stage1_out,
    proxy_out,
    full_out,
    topk,
    min_detections,
    min_committed_matches,
    max_detections,
    max_tracks,
    cluster_gate_calibration,
    cluster_gate_select_metric,
    cluster_gate_beta,
    cluster_gate_fp_weight,
    cluster_gate_loss_mode,
    cluster_gate_positive_weight,
    cluster_gate_negative_weight,
    model_selection_metric,
    current_stage,
    status,
    error,
    decision,
) = sys.argv[1:]

result_csv = Path(result_csv)
summary_csv = Path(summary_csv)
row = {}
if summary_csv.is_file():
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), {})

row.update(
    {
        "exp_name": "base_host_v2_2_gate_objective_proxy0213",
        "module_family": "set_predictor_v2",
        "out_dir": out_dir,
        "data_jsonl": data_jsonl,
        "source_manifest": source_manifest,
        "host_variant": host_variant,
        "host_config_path": base_config,
        "host_checkpoint": base_ckpt,
        "graph_topk": topk,
        "graph_min_detections": min_detections,
        "graph_min_committed_matches": min_committed_matches,
        "graph_max_detections": max_detections,
        "graph_max_tracks": max_tracks,
        "graph_cluster_gate_thresh": row.get("graph_cluster_gate_thresh", "0.5"),
        "graph_cluster_gate_temp": row.get("graph_cluster_gate_temp", "1.0"),
        "graph_cluster_gate_bias": row.get("graph_cluster_gate_bias", "0.0"),
        "cluster_gate_calibration": cluster_gate_calibration,
        "cluster_gate_select_metric": cluster_gate_select_metric,
        "cluster_gate_beta": cluster_gate_beta,
        "cluster_gate_fp_weight": cluster_gate_fp_weight,
        "cluster_gate_loss_mode": cluster_gate_loss_mode,
        "cluster_gate_positive_weight": cluster_gate_positive_weight,
        "cluster_gate_negative_weight": cluster_gate_negative_weight,
        "model_selection_metric": model_selection_metric,
        "checkpoint": row.get("checkpoint", ""),
        "current_stage": current_stage,
        "status": status,
        "error": error,
        "train_examples": row.get("train_examples", ""),
        "val_examples": row.get("val_examples", ""),
        "best_epoch": row.get("best_epoch", ""),
        "val_cluster_gate_thresh_calibrated": row.get("val_cluster_gate_thresh_calibrated", ""),
        "val_cluster_gate_temp": row.get("val_cluster_gate_temp", ""),
        "val_cluster_gate_bias": row.get("val_cluster_gate_bias", ""),
        "val_cluster_gate_precision_cal": row.get("val_cluster_gate_precision_cal", ""),
        "val_cluster_gate_recall_cal": row.get("val_cluster_gate_recall_cal", ""),
        "val_cluster_gate_f0_5": row.get("val_cluster_gate_f0_5", ""),
        "val_cluster_gate_utility_cal": row.get("val_cluster_gate_utility_cal", ""),
        "val_cluster_gate_coverage_cal": row.get("val_cluster_gate_coverage_cal", ""),
        "proxy_out_dir": proxy_out if Path(proxy_out).exists() else row.get("proxy_out_dir", ""),
        "proxy_HOTA": row.get("proxy_HOTA", ""),
        "proxy_AssA": row.get("proxy_AssA", ""),
        "proxy_IDF1": row.get("proxy_IDF1", ""),
        "proxy_MOTA": row.get("proxy_MOTA", ""),
        "proxy_IDSW": row.get("proxy_IDSW", ""),
        "proxy_gate_filtered_clusters": row.get("proxy_gate_filtered_clusters", ""),
        "proxy_replaced_clusters": row.get("proxy_replaced_clusters", ""),
        "full_eval_out_dir": full_out if Path(full_out).exists() else row.get("full_eval_out_dir", ""),
        "full_HOTA": row.get("full_HOTA", ""),
        "full_AssA": row.get("full_AssA", ""),
        "full_IDF1": row.get("full_IDF1", ""),
        "full_MOTA": row.get("full_MOTA", ""),
        "full_IDSW": row.get("full_IDSW", ""),
        "decision": decision,
    }
)

fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
}

merge_stage1_into_batch() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${STAGE1_OUT}/summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

batch_csv = Path(sys.argv[1])
stage1_csv = Path(sys.argv[2])
with batch_csv.open("r", encoding="utf-8", newline="") as f:
    batch_row = next(csv.DictReader(f), {})
with stage1_csv.open("r", encoding="utf-8", newline="") as f:
    stage1_row = next(csv.DictReader(f), {})

batch_row.update(
    {
        "checkpoint": stage1_row.get("checkpoint", batch_row.get("checkpoint", "")),
        "train_examples": stage1_row.get("train_examples", ""),
        "val_examples": stage1_row.get("val_examples", ""),
        "best_epoch": stage1_row.get("best_epoch", ""),
        "graph_cluster_gate_thresh": stage1_row.get(
            "val_cluster_gate_thresh_calibrated",
            batch_row.get("graph_cluster_gate_thresh", ""),
        ),
        "graph_cluster_gate_temp": stage1_row.get(
            "val_cluster_gate_temp",
            batch_row.get("graph_cluster_gate_temp", ""),
        ),
        "graph_cluster_gate_bias": stage1_row.get(
            "val_cluster_gate_bias",
            batch_row.get("graph_cluster_gate_bias", ""),
        ),
        "val_cluster_gate_thresh_calibrated": stage1_row.get("val_cluster_gate_thresh_calibrated", ""),
        "val_cluster_gate_temp": stage1_row.get("val_cluster_gate_temp", ""),
        "val_cluster_gate_bias": stage1_row.get("val_cluster_gate_bias", ""),
        "val_cluster_gate_precision_cal": stage1_row.get("val_cluster_gate_precision_cal", ""),
        "val_cluster_gate_recall_cal": stage1_row.get("val_cluster_gate_recall_cal", ""),
        "val_cluster_gate_f0_5": stage1_row.get("val_cluster_gate_f0_5", ""),
        "val_cluster_gate_utility_cal": stage1_row.get("val_cluster_gate_utility_cal", ""),
        "val_cluster_gate_coverage_cal": stage1_row.get("val_cluster_gate_coverage_cal", ""),
    }
)

fieldnames = list(batch_row.keys())
for path in (batch_csv, batch_csv.with_name("result.csv")):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: batch_row.get(key, "") for key in fieldnames})
PY
}

merge_proxy_into_batch() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${PROXY_OUT}/summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

batch_csv = Path(sys.argv[1])
proxy_csv = Path(sys.argv[2])
with batch_csv.open("r", encoding="utf-8", newline="") as f:
    batch_row = next(csv.DictReader(f), {})
with proxy_csv.open("r", encoding="utf-8", newline="") as f:
    proxy_row = next(csv.DictReader(f), {})

batch_row.update(
    {
        "proxy_HOTA": proxy_row.get("HOTA", ""),
        "proxy_AssA": proxy_row.get("AssA", ""),
        "proxy_IDF1": proxy_row.get("IDF1", ""),
        "proxy_MOTA": proxy_row.get("MOTA", ""),
        "proxy_IDSW": proxy_row.get("IDSW", ""),
        "proxy_gate_filtered_clusters": proxy_row.get("gate_filtered_clusters", ""),
        "proxy_replaced_clusters": proxy_row.get("replaced_clusters", ""),
    }
)

fieldnames = list(batch_row.keys())
for path in (batch_csv, batch_csv.with_name("result.csv")):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: batch_row.get(key, "") for key in fieldnames})
PY
}

merge_full_into_batch() {
  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${FULL_OUT}/summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

batch_csv = Path(sys.argv[1])
full_csv = Path(sys.argv[2])
with batch_csv.open("r", encoding="utf-8", newline="") as f:
    batch_row = next(csv.DictReader(f), {})
with full_csv.open("r", encoding="utf-8", newline="") as f:
    full_row = next(csv.DictReader(f), {})

batch_row.update(
    {
        "full_HOTA": full_row.get("HOTA", ""),
        "full_AssA": full_row.get("AssA", ""),
        "full_IDF1": full_row.get("IDF1", ""),
        "full_MOTA": full_row.get("MOTA", ""),
        "full_IDSW": full_row.get("IDSW", ""),
    }
)

fieldnames = list(batch_row.keys())
for path in (batch_csv, batch_csv.with_name("result.csv")):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: batch_row.get(key, "") for key in fieldnames})
PY
}

read_gate_triplet() {
  "${PYTHON_BIN}" - "${STAGE1_OUT}/summary.csv" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
with summary_csv.open("r", encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f), {})
print(row.get("val_cluster_gate_temp", "1.0"))
print(row.get("val_cluster_gate_bias", "0.0"))
print(row.get("val_cluster_gate_thresh_calibrated", row.get("cluster_gate_thresh", "0.5")))
PY
}

proxy_should_run_full() {
  "${PYTHON_BIN}" - "${PROXY_OUT}/summary.csv" "${PROXY_BASELINE_HOTA}" "${PROXY_BASELINE_ASSA}" "${PROXY_BASELINE_IDF1}" <<'PY'
import csv
import sys
from pathlib import Path

proxy_csv = Path(sys.argv[1])
baseline_hota = float(sys.argv[2])
baseline_assa = float(sys.argv[3])
baseline_idf1 = float(sys.argv[4])

with proxy_csv.open("r", encoding="utf-8", newline="") as f:
    row = next(csv.DictReader(f), {})

try:
    hota = float(row.get("HOTA", "nan"))
    assa = float(row.get("AssA", "nan"))
    idf1 = float(row.get("IDF1", "nan"))
    gate_filtered = float(row.get("gate_filtered_clusters", "0") or 0)
except Exception:
    print("no")
    raise SystemExit(0)

passes = hota >= baseline_hota and assa >= baseline_assa and idf1 > baseline_idf1 and gate_filtered > 0.0
print("yes" if passes else "no")
PY
}

update_batch_row "01_stage1" "running" "" "pending"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_local_conflict_set_predictor.py" \
  --data-jsonl "${DATA_JSONL}" \
  --out-dir "${STAGE1_OUT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-heads "${NUM_HEADS}" \
  --num-conflict-blocks "${NUM_BLOCKS}" \
  --train-sequences "${TRAIN_SEQUENCES}" \
  --val-sequences "${VAL_SEQUENCES}" \
  --strict-sequence-split \
  --min-val-examples "${MIN_VAL_EXAMPLES}" \
  --dataset-tag "${DATASET_TAG}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --feature-version "${FEATURE_VERSION}" \
  --cluster-gate-thresh "${CLUSTER_GATE_THRESH}" \
  --cluster-gate-calibration "${CLUSTER_GATE_CALIBRATION}" \
  --cluster-gate-select-metric "${CLUSTER_GATE_SELECT_METRIC}" \
  --cluster-gate-beta "${CLUSTER_GATE_BETA}" \
  --cluster-gate-fp-weight "${CLUSTER_GATE_FP_WEIGHT}" \
  --cluster-gate-search-min "${CLUSTER_GATE_SEARCH_MIN}" \
  --cluster-gate-search-max "${CLUSTER_GATE_SEARCH_MAX}" \
  --cluster-gate-search-steps "${CLUSTER_GATE_SEARCH_STEPS}" \
  --cluster-gate-loss-mode "${CLUSTER_GATE_LOSS_MODE}" \
  --cluster-gate-positive-weight "${CLUSTER_GATE_POSITIVE_WEIGHT}" \
  --cluster-gate-negative-weight "${CLUSTER_GATE_NEGATIVE_WEIGHT}" \
  --model-selection-metric "${MODEL_SELECTION_METRIC}" >"${STAGE1_OUT}/run.log" 2>&1; then
  update_batch_row "01_stage1" "failed" "stage1_failed" "stop"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
    --kind train \
    --status failed \
    --script "scripts/run_local_conflict_set_predictor_gate_objective_proxy0213.sh" \
    --dataset "MOT17" \
    --split "val0213_proxy" \
    --tracker-family "ByteTrack" \
    --variant "base_host_v2_2_gate_objective_proxy0213" \
    --tag "local_conflict_set_predictor_gate_objective_v22" \
    --run-root "${OUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${STAGE1_OUT}/best.pt" \
    --log-path "${PIPELINE_LOG}" \
    --notes "gate-objective proxy batch failed at stage1"
  exit 1
fi

merge_stage1_into_batch
mapfile -t GATE_VALUES < <(read_gate_triplet)
GATE_TEMP="${GATE_VALUES[0]:-1.0}"
GATE_BIAS="${GATE_VALUES[1]:-0.0}"
GATE_THRESH="${GATE_VALUES[2]:-0.5}"

update_batch_row "02_proxy_eval" "running" "" "pending"
merge_stage1_into_batch

if ! BASE_CONFIG="${BASE_CONFIG}" BASE_CKPT="${BASE_CKPT}" HOST_VARIANT="${HOST_VARIANT}" \
  bash "${REPO_ROOT}/scripts/run_local_conflict_graph_set_predictor_proxy0213.sh" \
  "${PROXY_OUT}" \
  "${STAGE1_OUT}/best.pt" \
  "${TOPK}" \
  "${MIN_DETECTIONS}" \
  "${MIN_COMMITTED_MATCHES}" \
  "${MAX_DETECTIONS}" \
  "${MAX_TRACKS}" \
  "${GATE_THRESH}" \
  "${GATE_TEMP}" \
  "${GATE_BIAS}"; then
  merge_proxy_into_batch || true
  update_batch_row "02_proxy_eval" "failed" "proxy_failed" "stop"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
    --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
    --kind eval \
    --status failed \
    --script "scripts/run_local_conflict_set_predictor_gate_objective_proxy0213.sh" \
    --dataset "MOT17" \
    --split "val0213_proxy" \
    --tracker-family "ByteTrack" \
    --variant "base_host_v2_2_gate_objective_proxy0213" \
    --tag "local_conflict_set_predictor_gate_objective_v22" \
    --run-root "${OUT_DIR}" \
    --summary-csv "${SUMMARY_CSV}" \
    --checkpoint "${STAGE1_OUT}/best.pt" \
    --log-path "${PIPELINE_LOG}" \
    --notes "gate-objective proxy batch failed at proxy eval"
  exit 1
fi

merge_proxy_into_batch
DECISION="stop_after_proxy"
if [[ "$(proxy_should_run_full)" == "yes" ]]; then
  DECISION="run_full"
  update_batch_row "03_full_eval_md2_mm2" "running" "" "${DECISION}"
  merge_proxy_into_batch
  if BASE_CONFIG="${BASE_CONFIG}" BASE_CKPT="${BASE_CKPT}" HOST_VARIANT="${HOST_VARIANT}" \
    bash "${REPO_ROOT}/scripts/run_local_conflict_graph_set_predictor_generic.sh" \
    "${FULL_OUT}" \
    "local_conflict_graph_set_predictor_gate_objective_full_frcnn_md${MIN_DETECTIONS}_mm${MIN_COMMITTED_MATCHES}" \
    "${STAGE1_OUT}/best.pt" \
    "${TOPK}" \
    "${MIN_DETECTIONS}" \
    "${MIN_COMMITTED_MATCHES}" \
    "${MAX_DETECTIONS}" \
    "${MAX_TRACKS}" \
    "${GATE_THRESH}" \
    "${GATE_TEMP}" \
    "${GATE_BIAS}" \
    "FRCNN" \
    "" \
    "gate-objective set predictor evaluation on MOT17 full public FRCNN"; then
    merge_full_into_batch
    update_batch_row "done" "ok" "" "${DECISION}"
  else
    merge_full_into_batch || true
    update_batch_row "03_full_eval_md2_mm2" "failed" "full_eval_failed" "${DECISION}"
    exit 1
  fi
else
  update_batch_row "done_proxy_only" "ok" "" "${DECISION}"
  merge_proxy_into_batch
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind eval \
  --status success \
  --script "scripts/run_local_conflict_set_predictor_gate_objective_proxy0213.sh" \
  --dataset "MOT17" \
  --split "val0213_proxy" \
  --tracker-family "ByteTrack" \
  --variant "base_host_v2_2_gate_objective_proxy0213" \
  --tag "local_conflict_set_predictor_gate_objective_v22" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${STAGE1_OUT}/best.pt" \
  --log-path "${PIPELINE_LOG}" \
  --notes "gate-objective base-host proxy rerun with conditional full follow-up" \
  --extra \
    "module_family=set_predictor_v2" \
    "host_variant=${HOST_VARIANT}" \
    "graph_topk=${TOPK}" \
    "graph_min_detections=${MIN_DETECTIONS}" \
    "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" \
    "graph_max_detections=${MAX_DETECTIONS}" \
    "graph_max_tracks=${MAX_TRACKS}" \
    "cluster_gate_calibration=${CLUSTER_GATE_CALIBRATION}" \
    "cluster_gate_select_metric=${CLUSTER_GATE_SELECT_METRIC}" \
    "cluster_gate_loss_mode=${CLUSTER_GATE_LOSS_MODE}" \
    "cluster_gate_positive_weight=${CLUSTER_GATE_POSITIVE_WEIGHT}" \
    "cluster_gate_negative_weight=${CLUSTER_GATE_NEGATIVE_WEIGHT}" \
    "model_selection_metric=${MODEL_SELECTION_METRIC}"

echo "[local-conflict-set-predictor-gate-objective-proxy0213] status=ok out_dir=${OUT_DIR}"
