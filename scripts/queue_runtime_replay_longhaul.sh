#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FMtrack-main/FM-Track}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATASET="${DATASET:-MOT17}"
SPLIT="${SPLIT:-train}"
EXPECTED_SEQS="${EXPECTED_SEQS:-7}"
POLL_SECS="${POLL_SECS:-120}"

HEUR_RUN_DIR="${HEUR_RUN_DIR:-${REPO_ROOT}/outputs/runtime_assoc_dump_run_sw_yolox_heuristic_full7_fullcand_20260317}"
HEUR_DUMP_ROOT="${HEUR_DUMP_ROOT:-${REPO_ROOT}/outputs/runtime_assoc_dump_sw_yolox_heuristic_full7_fullcand_20260317}"
HEUR_REPLAY_ROOT="${HEUR_REPLAY_ROOT:-${HEUR_DUMP_ROOT}}"

BASE_RUN_DIR="${BASE_RUN_DIR:-${REPO_ROOT}/outputs/runtime_assoc_dump_run_sw_yolox_base_full7_fullcand_20260317}"
BASE_DUMP_ROOT="${BASE_DUMP_ROOT:-${REPO_ROOT}/outputs/runtime_assoc_dump_sw_yolox_base_full7_fullcand_20260317}"
BASE_REPLAY_ROOT="${BASE_REPLAY_ROOT:-${BASE_DUMP_ROOT}}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/outputs/runtime_replay_longhaul_20260317}"
LOG_PATH="${LOG_PATH:-${LOG_ROOT}/queue.log}"

mkdir -p "${LOG_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_PATH}"
}

count_seq_files() {
  local root="$1"
  local pattern="$2"
  find "${root}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
}

wait_for_existing_dump() {
  local run_dir="$1"
  local dump_root="$2"
  local expected="$3"
  local config_path="${run_dir}/resolved_config.yaml"
  local tracker_dir="${run_dir}/tracker/${DATASET}-${SPLIT}"
  local csv_dir="${dump_root}/${DATASET}/${SPLIT}"

  log "waiting for dump completion: run_dir=${run_dir}"
  while true; do
    local tracker_count
    local csv_count
    local running
    tracker_count="$(count_seq_files "${tracker_dir}" '*.txt')"
    csv_count="$(count_seq_files "${csv_dir}" '*.csv')"
    if pgrep -af "submit_bytetrack.py --config-path ${config_path}" >/dev/null 2>&1; then
      running=1
    else
      running=0
    fi
    log "dump status tracker_txt=${tracker_count}/${expected} dump_csv=${csv_count}/${expected} running=${running}"
    if [[ "${running}" == "0" ]]; then
      if [[ "${tracker_count}" -ge "${expected}" && "${csv_count}" -ge "${expected}" ]]; then
        log "dump completed successfully: ${run_dir}"
        return 0
      fi
      log "dump terminated before all sequences were written: ${run_dir}"
      return 1
    fi
    sleep "${POLL_SECS}"
  done
}

run_labeling() {
  local dump_root="$1"
  local replay_root="$2"
  mkdir -p "${replay_root}"
  local out_csv="${replay_root}/labeled_replay_allcand.csv"
  local out_summary="${replay_root}/labeled_replay_allcand.summary.json"
  local out_group="${replay_root}/labeled_replay_allcand.groups.jsonl"
  local out_recover="${replay_root}/labeled_replay_allcand.recoverability.json"

  if [[ -f "${out_csv}" && -f "${out_recover}" ]]; then
    log "skip labeling; outputs already exist at ${replay_root}"
    return 0
  fi

  log "label runtime replay groups: dump_root=${dump_root}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_runtime_assoc_replay_labels.py" \
    --dump-root "${dump_root}" \
    --dataset "${DATASET}" \
    --data-root /gemini/code/datasets \
    --split "${SPLIT}" \
    --split-part full \
    --out-csv "${out_csv}" \
    --summary-json "${out_summary}" \
    --out-group-jsonl "${out_group}" \
    --out-recoverability-json "${out_recover}" \
    --topk 0 \
    --rank-score-col refined_score \
    --ambiguity-margin 0.10 | tee -a "${LOG_PATH}"
}

run_baseline() {
  local input_csv="$1"
  local out_dir="$2"
  shift 2
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/metrics.json" ]]; then
    log "skip baseline; metrics already exist at ${out_dir}"
    return 0
  fi
  log "run baseline -> ${out_dir}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/train_runtime_rerank_baseline.py" \
    --input-csv "${input_csv}" \
    --out-dir "${out_dir}" \
    "$@" | tee -a "${LOG_PATH}"
}

get_seqs_csv() {
  local input_csv="$1"
  "${PYTHON_BIN}" - <<'PY' "${input_csv}"
import csv
import sys
from collections import OrderedDict

seqs = OrderedDict()
with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = str(row.get("seq", "")).strip()
        if seq:
            seqs.setdefault(seq, None)
print("\n".join(seqs.keys()))
PY
}

run_random_and_seed_sweeps() {
  local replay_root="$1"
  local tag="$2"
  local input_csv="${replay_root}/labeled_replay_allcand.csv"

  run_baseline "${input_csv}" "${replay_root}/baseline_logistic_ambonly_seed42_${tag}" \
    --model logistic --rank-score-col refined_score --valid-only --apply-on ambiguous --seed 42

  for seed in 42 43 44 45 46; do
    run_baseline "${input_csv}" "${replay_root}/baseline_gbdt_ambonly_seed${seed}_${tag}" \
      --model gbdt --rank-score-col refined_score --valid-only --apply-on ambiguous --seed "${seed}"
  done

  for seed in 42 43 44; do
    run_baseline "${input_csv}" "${replay_root}/baseline_gbdt_all_seed${seed}_${tag}" \
      --model gbdt --rank-score-col refined_score --valid-only --apply-on all --seed "${seed}"
  done

  for seed in 42 43 44; do
    run_baseline "${input_csv}" "${replay_root}/baseline_gbdt_ambonly_baseanchor_seed${seed}_${tag}" \
      --model gbdt --rank-score-col base_score --valid-only --apply-on ambiguous --seed "${seed}"
  done
}

run_leave_one_seq_out() {
  local replay_root="$1"
  local tag="$2"
  local input_csv="${replay_root}/labeled_replay_allcand.csv"
  mapfile -t seqs < <(get_seqs_csv "${input_csv}")
  for val_seq in "${seqs[@]}"; do
    if [[ -z "${val_seq}" ]]; then
      continue
    fi
    local train_csv
    train_csv="$("${PYTHON_BIN}" - <<'PY' "${val_seq}" "${input_csv}"
import csv
import sys
val_seq = sys.argv[1]
input_csv = sys.argv[2]
seqs = []
seen = set()
with open(input_csv, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        seq = str(row.get("seq", "")).strip()
        if seq and seq not in seen:
            seen.add(seq)
            seqs.append(seq)
train = [seq for seq in seqs if seq != val_seq]
print(",".join(train))
PY
)"
    run_baseline "${input_csv}" "${replay_root}/baseline_gbdt_loso_${val_seq}_${tag}" \
      --model gbdt --rank-score-col refined_score --valid-only --apply-on ambiguous \
      --train-seqs "${train_csv}" --val-seqs "${val_seq}" --seed 42
  done
}

write_summary() {
  local replay_root="$1"
  local tag="$2"
  local summary_path="${replay_root}/baseline_sweep_summary_${tag}.md"
  "${PYTHON_BIN}" - <<'PY' "${replay_root}" "${summary_path}" "${tag}"
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
tag = sys.argv[3]
rows = []
for path in sorted(root.rglob("metrics.json")):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    val = data.get("val_eval", {})
    rows.append(
        {
            "name": path.parent.name,
            "model": data.get("model", ""),
            "apply_on": data.get("apply_on", ""),
            "rank_score_col": data.get("rank_score_col", ""),
            "seed": data.get("seed", ""),
            "top1_gain": val.get("top1_gain", 0.0),
            "amb_top1_gain": val.get("amb_top1_gain", 0.0),
            "easy_top1_gain": val.get("easy_top1_gain", 0.0),
            "base_top1": val.get("base_top1", 0.0),
            "final_top1": val.get("final_top1", 0.0),
        }
    )

rows.sort(key=lambda x: (x["top1_gain"], x["amb_top1_gain"]), reverse=True)
lines = [f"# Runtime Replay Baseline Sweep Summary ({tag})", ""]
lines.append("| name | model | apply_on | rank_score_col | seed | top1_gain | amb_top1_gain | easy_top1_gain |")
lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
for row in rows:
    lines.append(
        f"| {row['name']} | {row['model']} | {row['apply_on']} | {row['rank_score_col']} | "
        f"{row['seed']} | {row['top1_gain']:.6f} | {row['amb_top1_gain']:.6f} | {row['easy_top1_gain']:.6f} |"
    )
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path)
PY
  log "wrote summary: ${summary_path}"
}

launch_base_dump_if_needed() {
  if [[ -d "${BASE_DUMP_ROOT}/${DATASET}/${SPLIT}" ]] && [[ "$(count_seq_files "${BASE_DUMP_ROOT}/${DATASET}/${SPLIT}" '*.csv')" -ge "${EXPECTED_SEQS}" ]]; then
    log "base full7 dump already present; skip launch"
    return 0
  fi
  log "launch base full7 full-candidate dump"
  DUMP_TOPK=0 /bin/bash "${REPO_ROOT}/scripts/run_bytetrack_assoc_dump_external.sh" \
    sw_yolox base full7 "${BASE_DUMP_ROOT}" "${BASE_RUN_DIR}" | tee -a "${LOG_PATH}"
}

main() {
  log "=== runtime replay longhaul queue started ==="

  wait_for_existing_dump "${HEUR_RUN_DIR}" "${HEUR_DUMP_ROOT}" "${EXPECTED_SEQS}"
  run_labeling "${HEUR_DUMP_ROOT}" "${HEUR_REPLAY_ROOT}"
  run_random_and_seed_sweeps "${HEUR_REPLAY_ROOT}" "heurfull7"
  run_leave_one_seq_out "${HEUR_REPLAY_ROOT}" "heurfull7"
  write_summary "${HEUR_REPLAY_ROOT}" "heurfull7"

  launch_base_dump_if_needed
  run_labeling "${BASE_DUMP_ROOT}" "${BASE_REPLAY_ROOT}"
  run_random_and_seed_sweeps "${BASE_REPLAY_ROOT}" "basefull7"
  run_leave_one_seq_out "${BASE_REPLAY_ROOT}" "basefull7"
  write_summary "${BASE_REPLAY_ROOT}" "basefull7"

  log "=== runtime replay longhaul queue finished ==="
}

main "$@"
