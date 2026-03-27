#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/experiments/bytetrack_fa_mot_mot17_v17_local_conflict_commit_val0213.yaml}"
BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/checkpoint_epoch_0.pth}"
HOST_VARIANT="${HOST_VARIANT:-base_reid_da}"
DETECTOR_FILTER_CSV="${DETECTOR_FILTER_CSV:-FRCNN}"
DATASET_TAG="${DATASET_TAG:-local_conflict_commit_large_base}"
FEATURE_VERSION="${FEATURE_VERSION:-v1_raw}"

OUT_DIR="${1:-${REPO_ROOT}/outputs/local_conflict_commit_large_base_$(date +%Y%m%d_%H%M%S)}"
TRAIN_SEQUENCES_CSV="${2:-MOT17-04,MOT17-05,MOT17-09,MOT17-10,MOT17-11}"
VAL_SEQUENCES_CSV="${3:-MOT17-02,MOT17-13}"
TOPK="${4:-8}"
HORIZON="${5:-16}"
EPOCHS="${6:-10}"
HIDDEN_DIM="${7:-128}"
BATCH_SIZE="${8:-8}"
MIN_VAL_EXAMPLES="${9:-64}"
MIN_DETECTIONS="${10:-2}"
MIN_COMMITTED_MATCHES="${11:-2}"
MAX_DETECTIONS="${12:-8}"
MAX_TRACKS="${13:-32}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base config: ${BASE_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "Missing eval config: ${EVAL_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

DUMP_OUT="${OUT_DIR}/00_base_dump"
PROFILE_PATH="${OUT_DIR}/dump_profile.json"
RESULT_CSV="${OUT_DIR}/result.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
LOG_PATH="${OUT_DIR}/pipeline.log"
MANIFEST_PATH="${OUT_DIR}/run_manifest.json"
DUMP_ROOT="${DUMP_OUT}/runtime_dump"
LABELED_CSV="${OUT_DIR}/labeled_replay_top${TOPK}.csv"
GROUP_JSONL="${OUT_DIR}/labeled_replay_top${TOPK}.groups.jsonl"
RECOVERABILITY_JSON="${OUT_DIR}/labeled_replay_top${TOPK}.recoverability.json"
COMP_DIR="${OUT_DIR}/competition_cases"
SOURCE_MANIFEST="${OUT_DIR}/source_manifest.csv"
CLUSTER_OUT="${OUT_DIR}/cluster_commit_data"
STAGE1_OUT="${OUT_DIR}/01_stage1"
PROXY_OUT="${OUT_DIR}/02_proxy_eval"
FULL_OUT="${OUT_DIR}/03_full_eval_md2_mm2"

CURRENT_STAGE="init"

update_queue_row() {
  "${PYTHON_BIN}" - \
    "${RESULT_CSV}" \
    "${SUMMARY_CSV}" \
    "${OUT_DIR}" \
    "${BASE_CONFIG}" \
    "${EVAL_CONFIG}" \
    "${BASE_CKPT}" \
    "${HOST_VARIANT}" \
    "${DETECTOR_FILTER_CSV}" \
    "${TRAIN_SEQUENCES_CSV}" \
    "${VAL_SEQUENCES_CSV}" \
    "${TOPK}" \
    "${HORIZON}" \
    "${EPOCHS}" \
    "${HIDDEN_DIM}" \
    "${BATCH_SIZE}" \
    "${MIN_VAL_EXAMPLES}" \
    "${MIN_DETECTIONS}" \
    "${MIN_COMMITTED_MATCHES}" \
    "${MAX_DETECTIONS}" \
    "${MAX_TRACKS}" \
    "${DATASET_TAG}" \
    "${FEATURE_VERSION}" \
    "${SOURCE_MANIFEST}" \
    "${CLUSTER_OUT}/cluster_examples.jsonl" \
    "${STAGE1_OUT}/best.pt" \
    "${PROXY_OUT}" \
    "${FULL_OUT}" \
    "$1" \
    "$2" \
    "$3" <<'PY'
import csv
import sys
from pathlib import Path

(
    result_csv,
    summary_csv,
    out_dir,
    base_config,
    eval_config,
    base_ckpt,
    host_variant,
    detector_filter,
    train_sequences,
    val_sequences,
    topk,
    horizon,
    epochs,
    hidden_dim,
    batch_size,
    min_val_examples,
    min_detections,
    min_committed_matches,
    max_detections,
    max_tracks,
    dataset_tag,
    feature_version,
    source_manifest,
    data_jsonl,
    checkpoint,
    proxy_out_dir,
    full_out_dir,
    current_stage,
    status,
    error,
) = sys.argv[1:]

result_csv = Path(result_csv)
summary_csv = Path(summary_csv)

row = {}
if summary_csv.is_file():
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, {})

row.update(
    {
        "exp_name": "local_conflict_commit_stage1_large_base",
        "out_dir": out_dir,
        "base_config": base_config,
        "eval_config": eval_config,
        "checkpoint": checkpoint if Path(checkpoint).is_file() else row.get("checkpoint", ""),
        "host_variant": host_variant,
        "host_checkpoint": base_ckpt,
        "detector_filter": detector_filter,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "topk": topk,
        "horizon": horizon,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "min_val_examples": min_val_examples,
        "graph_min_detections": min_detections,
        "graph_min_committed_matches": min_committed_matches,
        "graph_max_detections": max_detections,
        "graph_max_tracks": max_tracks,
        "dataset_tag": dataset_tag,
        "feature_version": feature_version,
        "source_manifest": source_manifest if Path(source_manifest).is_file() else row.get("source_manifest", ""),
        "data_jsonl": data_jsonl if Path(data_jsonl).is_file() else row.get("data_jsonl", ""),
        "proxy_out_dir": proxy_out_dir if Path(proxy_out_dir).exists() else row.get("proxy_out_dir", ""),
        "full_eval_out_dir": full_out_dir if Path(full_out_dir).exists() else row.get("full_eval_out_dir", ""),
        "current_stage": current_stage,
        "status": status,
        "error": error,
    }
)

fieldnames = list(row.keys()) if row else []
if not fieldnames:
    fieldnames = list(row.keys())
for path in (result_csv, summary_csv):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})
PY
}

finalize_queue_row() {
  "${PYTHON_BIN}" - \
    "${RESULT_CSV}" \
    "${SUMMARY_CSV}" \
    "${OUT_DIR}" \
    "${STAGE1_OUT}/summary.csv" \
    "${PROXY_OUT}/summary.csv" \
    "${FULL_OUT}/summary.csv" \
    "${CLUSTER_OUT}/summary.json" \
    "$1" \
    "$2" <<'PY'
import csv
import json
import sys
from pathlib import Path

result_csv = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
out_dir = sys.argv[3]
stage1_summary = Path(sys.argv[4])
proxy_summary = Path(sys.argv[5])
full_summary = Path(sys.argv[6])
cluster_summary_json = Path(sys.argv[7])
status = sys.argv[8]
current_stage = sys.argv[9]


def load_single_row_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, {})


row = load_single_row_csv(summary_csv)
cluster_doc = {}
if cluster_summary_json.is_file():
    with cluster_summary_json.open("r", encoding="utf-8") as f:
        cluster_doc = json.load(f)
stage1_row = load_single_row_csv(stage1_summary)
proxy_row = load_single_row_csv(proxy_summary)
full_row = load_single_row_csv(full_summary)

row.update(
    {
        "out_dir": out_dir,
        "current_stage": current_stage,
        "status": "ok" if status == "success" else "failed",
        "checkpoint": stage1_row.get("checkpoint", row.get("checkpoint", "")),
        "data_jsonl": stage1_row.get("data_jsonl", row.get("data_jsonl", "")),
        "train_examples": stage1_row.get("train_examples", ""),
        "val_examples": stage1_row.get("val_examples", ""),
        "train_host_variants": stage1_row.get("train_host_variants", ""),
        "val_host_variants": stage1_row.get("val_host_variants", ""),
        "split_mode": stage1_row.get("split_mode", ""),
        "cluster_examples": cluster_doc.get("eligible_clusters", ""),
        "cluster_trigger_pass_clusters": cluster_doc.get("trigger_pass_clusters", ""),
        "cluster_skipped_large_clusters": cluster_doc.get("skipped_large_clusters", ""),
        "proxy_HOTA": proxy_row.get("HOTA", ""),
        "proxy_AssA": proxy_row.get("AssA", ""),
        "proxy_IDF1": proxy_row.get("IDF1", ""),
        "proxy_MOTA": proxy_row.get("MOTA", ""),
        "proxy_IDSW": proxy_row.get("IDSW", ""),
        "full_md2_mm2_HOTA": full_row.get("HOTA", ""),
        "full_md2_mm2_AssA": full_row.get("AssA", ""),
        "full_md2_mm2_IDF1": full_row.get("IDF1", ""),
        "full_md2_mm2_MOTA": full_row.get("MOTA", ""),
        "full_md2_mm2_IDSW": full_row.get("IDSW", ""),
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

update_queue_row "${CURRENT_STAGE}" "running" ""

STATUS="failed"
if {
  CURRENT_STAGE="00_prepare_dump_profile"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" - "${PROFILE_PATH}" "${BASE_CONFIG}" "${BASE_CKPT}" "${DUMP_ROOT}" "${TOPK}" "${DETECTOR_FILTER_CSV}" <<'PY'
import json
import sys
from pathlib import Path

profile_path = Path(sys.argv[1]).resolve()
config_path = sys.argv[2]
checkpoint = str(Path(sys.argv[3]).resolve())
dump_root = str(Path(sys.argv[4]).resolve())
topk = int(sys.argv[5])
detector_filter = [token.strip() for token in str(sys.argv[6] or "").split(",") if token.strip()]

doc = {
    "description": "Large-base FRCNN runtime dump for local-conflict commit stage1 training.",
    "manifest": {
        "line": "local_conflict_commit_mainline",
        "protocol_tier": "large_base_dump",
        "host_variant": "base_reid_da",
        "eval_scope": "mot17_train_frcnn_full",
        "inference_model": checkpoint,
        "dump_root": dump_root,
        "topk": topk,
        "detector_filter": detector_filter,
    },
    "settings": {
        "config_path": config_path,
        "inference_dataset": "MOT17",
        "inference_split": "train",
        "inference_model": checkpoint,
        "config_overrides": {
            "EXP_NAME": "local_conflict_commit_large_base_dump",
            "EVAL_ONLY_VAL": False,
            "RUN_TRACKEVAL": False,
            "DETECTOR_FILTER": detector_filter,
            "ASSOC_USE_LOCAL_CONFLICT_GRAPH": False,
            "ASSOC_RUNTIME_DUMP_PATH": dump_root,
            "ASSOC_RUNTIME_DUMP_TOPK": topk,
            "ASSOC_RUNTIME_DUMP_MIN_SCORE": 0.0,
            "ASSOC_RUNTIME_DUMP_SAVE_TENSORS": True,
            "ASSOC_RUNTIME_DUMP_NPZ_EVERY_N_GROUPS": 2048,
        },
    },
}
profile_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY

  CURRENT_STAGE="01_run_base_dump"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_bytetrack_profile.py" \
    --exp-profile "${PROFILE_PATH}" \
    --out-dir "${DUMP_OUT}"

  CURRENT_STAGE="02_build_replay_labels"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_runtime_assoc_replay_labels.py" \
    --dump-root "${DUMP_ROOT}" \
    --dataset MOT17 \
    --data-root /gemini/code/datasets \
    --split train \
    --split-part full \
    --out-csv "${LABELED_CSV}" \
    --out-group-jsonl "${GROUP_JSONL}" \
    --out-recoverability-json "${RECOVERABILITY_JSON}" \
    --topk "${TOPK}" \
    --rank-score-col refined_score \
    --ambiguity-margin 0.10

  CURRENT_STAGE="03_build_cases"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_competition_assoc_cases.py" \
    --group-jsonl "${GROUP_JSONL}" \
    --out-dir "${COMP_DIR}" \
    --horizon "${HORIZON}"

  CURRENT_STAGE="04_build_manifest_and_dataset"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_local_conflict_commit_dataset_manifest.py" \
    --out-csv "${SOURCE_MANIFEST}" \
    --group-jsonl "${GROUP_JSONL}" \
    --cases-csv "${COMP_DIR}/competition_cases.csv" \
    --host-variant "${HOST_VARIANT}" \
    --source-tag "base_frcnn_full" \
    --split-tag auto \
    --dataset-tag "${DATASET_TAG}" \
    --feature-version "${FEATURE_VERSION}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_local_conflict_commit_dataset.py" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --out-dir "${CLUSTER_OUT}" \
    --topk "${TOPK}" \
    --min-detections "${MIN_DETECTIONS}" \
    --min-committed-matches "${MIN_COMMITTED_MATCHES}" \
    --max-detections "${MAX_DETECTIONS}" \
    --max-tracks "${MAX_TRACKS}" \
    --train-sequences "${TRAIN_SEQUENCES_CSV}" \
    --val-sequences "${VAL_SEQUENCES_CSV}" \
    --strict-sequence-split \
    --feature-version "${FEATURE_VERSION}" \
    --dataset-tag "${DATASET_TAG}"

  CURRENT_STAGE="05_train_stage1"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_local_conflict_commit_stage1.py" \
    --data-jsonl "${CLUSTER_OUT}/cluster_examples.jsonl" \
    --out-dir "${STAGE1_OUT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --train-sequences "${TRAIN_SEQUENCES_CSV}" \
    --val-sequences "${VAL_SEQUENCES_CSV}" \
    --strict-sequence-split \
    --min-val-examples "${MIN_VAL_EXAMPLES}" \
    --dataset-tag "${DATASET_TAG}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --feature-version "${FEATURE_VERSION}"

  CURRENT_STAGE="06_proxy_eval"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  BASE_CONFIG="${EVAL_CONFIG}" BASE_CKPT="${BASE_CKPT}" HOST_VARIANT="${HOST_VARIANT}" \
    bash "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_proxy0213.sh" \
    "${PROXY_OUT}" \
    "${STAGE1_OUT}/best.pt" \
    "${TOPK}" \
    "${MIN_DETECTIONS}" \
    "${MIN_COMMITTED_MATCHES}" \
    "${MAX_DETECTIONS}" \
    "${MAX_TRACKS}"

  CURRENT_STAGE="07_full_eval_md2_mm2"
  update_queue_row "${CURRENT_STAGE}" "running" ""
  BASE_CONFIG="${EVAL_CONFIG}" BASE_CKPT="${BASE_CKPT}" HOST_VARIANT="${HOST_VARIANT}" \
    bash "${REPO_ROOT}/scripts/run_local_conflict_graph_learned_commit_generic.sh" \
    "${FULL_OUT}" \
    "local_conflict_graph_learned_commit_largebase_full_frcnn_md${MIN_DETECTIONS}_mm${MIN_COMMITTED_MATCHES}" \
    "${STAGE1_OUT}/best.pt" \
    "${TOPK}" \
    "${MIN_DETECTIONS}" \
    "${MIN_COMMITTED_MATCHES}" \
    "${MAX_DETECTIONS}" \
    "${MAX_TRACKS}" \
    "${DETECTOR_FILTER_CSV}" \
    "" \
    "large-data learned commit evaluation on MOT17 full public ${DETECTOR_FILTER_CSV}"

} >"${LOG_PATH}" 2>&1; then
  STATUS="success"
else
  STATUS="failed"
fi

if [[ "${STATUS}" == "success" ]]; then
  CURRENT_STAGE="done"
  finalize_queue_row "${STATUS}" "${CURRENT_STAGE}"
else
  update_queue_row "${CURRENT_STAGE}" "failed" "stage_failed:${CURRENT_STAGE}"
  finalize_queue_row "${STATUS}" "${CURRENT_STAGE}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/append_experiment_record.py" \
  --csv "${REPO_ROOT}/outputs/experiment_registry.csv" \
  --kind train \
  --status "$( [[ "${STATUS}" == "success" ]] && echo success || echo failed )" \
  --script "scripts/run_local_conflict_commit_stage1_large_base.sh" \
  --dataset "MOT17" \
  --split "train_frcnn_large_base" \
  --tracker-family "ByteTrack" \
  --variant "local_conflict_commit_stage1_large_base" \
  --tag "local_conflict_commit_mainline" \
  --run-root "${OUT_DIR}" \
  --summary-csv "${SUMMARY_CSV}" \
  --checkpoint "${STAGE1_OUT}/best.pt" \
  --log-path "${LOG_PATH}" \
  --notes "large-data base host retrain for local conflict commit" \
  --extra "host_variant=${HOST_VARIANT}" "detector_filter=${DETECTOR_FILTER_CSV}" "train_sequences=${TRAIN_SEQUENCES_CSV}" "val_sequences=${VAL_SEQUENCES_CSV}" "graph_topk=${TOPK}" "graph_min_detections=${MIN_DETECTIONS}" "graph_min_committed_matches=${MIN_COMMITTED_MATCHES}" "graph_max_detections=${MAX_DETECTIONS}" "graph_max_tracks=${MAX_TRACKS}" "dataset_tag=${DATASET_TAG}" "feature_version=${FEATURE_VERSION}"

if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/post_experiment_pro_bundle.py" \
  --run-root "${OUT_DIR}" \
  --tag "local_conflict_commit_large_base_bundle" \
  --label "local_conflict_commit_stage1_large_base" \
  --status "$( [[ "${STATUS}" == "success" ]] && echo ok || echo failed )"; then
  echo "[local-conflict-commit-stage1-large-base] warning: failed to build Pro review bundle for ${OUT_DIR}" >&2
fi

echo "[local-conflict-commit-stage1-large-base] status=${STATUS} out_dir=${OUT_DIR}"
