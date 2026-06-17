#!/usr/bin/env bash
set -euo pipefail

unset ORION_TASK_IDLE_TIME || true

REPO_ROOT="/gemini/code/FMtrack-main/FM-Track"
cd "$REPO_ROOT"

QUEUE_DIR="outputs/dance_followup_training_queue_20260406_1"
QUEUE_SUMMARY="$QUEUE_DIR/summary.csv"
QUEUE_LOG="$QUEUE_DIR/queue.log"
REGISTRY_CSV="outputs/experiment_registry.csv"

EXPORT_DIR="outputs/deep_ocsort_local_contention_export_dance_best_20260406_1"
EXPORT_SUMMARY="$EXPORT_DIR/summary.csv"
EXPORT_JSONL="$EXPORT_DIR/local_contention_units.jsonl"

MOT17_JSONL="outputs/deep_ocsort_local_contention_export_mot17_best_20260406_1/local_contention_units.jsonl"
MOT20_JSONL="outputs/deep_ocsort_local_contention_export_mot20_best_20260406_1/local_contention_units.jsonl"

TRAIN1_DIR="outputs/local_contention_ranker_mot17_dance_train_mot20_val_20260406_1"
TRAIN1_LOG="$TRAIN1_DIR/launcher.log"

TRAIN2_DIR="outputs/local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1"
TRAIN2_LOG="$TRAIN2_DIR/launcher.log"

MOT17_ALL="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-10-FRCNN,MOT17-11-FRCNN,MOT17-13-FRCNN"
MOT20_ALL="MOT20-01,MOT20-02,MOT20-03,MOT20-05"
DANCE_ALL="dancetrack0004,dancetrack0005,dancetrack0007,dancetrack0010,dancetrack0014,dancetrack0018,dancetrack0019,dancetrack0025,dancetrack0026,dancetrack0030,dancetrack0034,dancetrack0035,dancetrack0041,dancetrack0043,dancetrack0047,dancetrack0058,dancetrack0063,dancetrack0065,dancetrack0073,dancetrack0077,dancetrack0079,dancetrack0081,dancetrack0090,dancetrack0094,dancetrack0097"

TRAIN2_MOT17_TRAIN="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-05-FRCNN,MOT17-09-FRCNN,MOT17-13-FRCNN"
TRAIN2_MOT17_VAL="MOT17-10-FRCNN,MOT17-11-FRCNN"
TRAIN2_MOT20_TRAIN="MOT20-01,MOT20-02"
TRAIN2_MOT20_VAL="MOT20-03,MOT20-05"
TRAIN2_DANCE_TRAIN="dancetrack0004,dancetrack0005,dancetrack0007,dancetrack0010,dancetrack0014,dancetrack0018,dancetrack0019,dancetrack0025,dancetrack0026,dancetrack0030,dancetrack0034,dancetrack0035,dancetrack0041,dancetrack0043,dancetrack0047,dancetrack0058,dancetrack0063,dancetrack0065,dancetrack0073,dancetrack0077"
TRAIN2_DANCE_VAL="dancetrack0079,dancetrack0081,dancetrack0090,dancetrack0094,dancetrack0097"

mkdir -p "$QUEUE_DIR" "$TRAIN1_DIR" "$TRAIN2_DIR"

timestamp_now() {
  date --iso-8601=seconds
}

write_initial_queue_summary() {
  python - <<'PY'
import csv
from pathlib import Path

path = Path("outputs/dance_followup_training_queue_20260406_1/summary.csv")
path.parent.mkdir(parents=True, exist_ok=True)
rows = [
    {
        "step": "wait_export",
        "name": "deep_ocsort_local_contention_export_dance_best_20260406_1",
        "status": "running",
        "out_dir": str((Path("outputs/deep_ocsort_local_contention_export_dance_best_20260406_1")).resolve()),
        "summary_csv": str(path.resolve()),
        "log_path": str((Path("outputs/dance_followup_training_queue_20260406_1/queue.log")).resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": "wait DanceTrack full export before training follow-up",
    },
    {
        "step": "train_mot17_dance_val_mot20",
        "name": "local_contention_ranker_mot17_dance_train_mot20_val_20260406_1",
        "status": "pending",
        "out_dir": str((Path("outputs/local_contention_ranker_mot17_dance_train_mot20_val_20260406_1")).resolve()),
        "summary_csv": str(path.resolve()),
        "log_path": str((Path("outputs/local_contention_ranker_mot17_dance_train_mot20_val_20260406_1/launcher.log")).resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": "train on MOT17 and DanceTrack, validate on MOT20",
    },
    {
        "step": "train_threeway_seqholdout",
        "name": "local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1",
        "status": "pending",
        "out_dir": str((Path("outputs/local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1")).resolve()),
        "summary_csv": str(path.resolve()),
        "log_path": str((Path("outputs/local_contention_ranker_mot17_mot20_dance_seqholdout_20260406_1/launcher.log")).resolve()),
        "started_at": "",
        "finished_at": "",
        "notes": "sequence holdout training across MOT17, MOT20, and DanceTrack",
    },
]
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["step", "name", "status", "out_dir", "summary_csv", "log_path", "started_at", "finished_at", "notes"],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY
}

update_queue_row() {
  local step="$1"
  local status="$2"
  local started_at="$3"
  local finished_at="$4"
  local notes="$5"
  python - "$step" "$status" "$started_at" "$finished_at" "$notes" <<'PY'
import csv
import sys
from pathlib import Path

step, status, started_at, finished_at, notes = sys.argv[1:]
path = Path("outputs/dance_followup_training_queue_20260406_1/summary.csv")
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
for row in rows:
    if row["step"] == step:
        row["status"] = status
        if started_at:
            row["started_at"] = started_at
        if finished_at:
            row["finished_at"] = finished_at
        if notes:
            row["notes"] = notes
        break
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
PY
}

append_queue_registry() {
  local status="$1"
  local notes="$2"
  python scripts/append_experiment_record.py \
    --csv "$REGISTRY_CSV" \
    --kind other \
    --status "$status" \
    --script scripts/watch_dance_full_then_train.sh \
    --dataset MOT17+MOT20+DanceTrack \
    --split mixed \
    --tracker-family local_contention_ranker_followup_queue \
    --variant dance_followup_training_queue_20260406_1 \
    --tag local_contention_followup_queue \
    --run-root "$QUEUE_DIR" \
    --summary-csv "$QUEUE_SUMMARY" \
    --log-path "$QUEUE_LOG" \
    --notes "$notes" \
    >/dev/null 2>&1 || true
}

read_export_status() {
  python - <<'PY'
import csv
import sys
from pathlib import Path

path = Path("outputs/deep_ocsort_local_contention_export_dance_best_20260406_1/summary.csv")
if not path.is_file():
    print("missing")
    sys.exit(0)
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
status = {row["step"]: row["status"] for row in rows}
if any(value == "failed" for value in status.values()):
    print("failed")
elif status.get("compare") == "success":
    print("success")
else:
    print("running")
PY
}

read_train_status() {
  local summary_path="$1"
  python - "$summary_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("missing")
    sys.exit(0)
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
if not rows:
    print("missing")
    sys.exit(0)
print(rows[0].get("status", "missing"))
PY
}

run_training() {
  local step_name="$1"
  local out_dir="$2"
  local log_path="$3"
  shift 3

  local current_status
  current_status="$(read_train_status "$out_dir/summary.csv")"
  if [[ "$current_status" == "success" ]]; then
    update_queue_row "$step_name" "success" "" "$(timestamp_now)" "reuse existing success run in $out_dir"
    echo "[$(timestamp_now)] reuse $step_name from $out_dir" >> "$QUEUE_LOG"
    return 0
  fi
  if [[ "$current_status" == "running" ]]; then
    update_queue_row "$step_name" "running" "$(timestamp_now)" "" "existing training summary already marked running in $out_dir"
    echo "[$(timestamp_now)] found pre-existing running $step_name in $out_dir" >> "$QUEUE_LOG"
    return 1
  fi

  update_queue_row "$step_name" "running" "$(timestamp_now)" "" "launch training in $out_dir"
  echo "[$(timestamp_now)] launch $step_name" >> "$QUEUE_LOG"
  echo "[$(timestamp_now)] command: python scripts/train_local_contention_ranker.py $*" >> "$log_path"
  python scripts/train_local_contention_ranker.py "$@" >> "$log_path" 2>&1
  local rc=$?
  local final_status
  final_status="$(read_train_status "$out_dir/summary.csv")"
  if [[ "$rc" -eq 0 && "$final_status" == "success" ]]; then
    update_queue_row "$step_name" "success" "" "$(timestamp_now)" "training finished successfully in $out_dir"
    echo "[$(timestamp_now)] success $step_name" >> "$QUEUE_LOG"
    return 0
  fi

  update_queue_row "$step_name" "failed" "" "$(timestamp_now)" "training failed in $out_dir rc=$rc summary_status=$final_status"
  echo "[$(timestamp_now)] failed $step_name rc=$rc summary_status=$final_status" >> "$QUEUE_LOG"
  return 1
}

write_initial_queue_summary
update_queue_row "wait_export" "running" "$(timestamp_now)" "" "waiting for DanceTrack full export to reach compare=success"
append_queue_registry "running" "queue started: wait DanceTrack export then launch two follow-up trainings"

{
  echo "[started_at] $(timestamp_now)"
  echo "[mode] wait DanceTrack export, then train MOT17+DanceTrack->MOT20 and three-way sequence-holdout"
} >> "$QUEUE_LOG"

while true; do
  export_status="$(read_export_status)"
  if [[ "$export_status" == "success" ]]; then
    break
  fi
  if [[ "$export_status" == "failed" ]]; then
    update_queue_row "wait_export" "failed" "" "$(timestamp_now)" "DanceTrack export summary contains failed status"
    update_queue_row "train_mot17_dance_val_mot20" "cancelled" "" "$(timestamp_now)" "cancelled after DanceTrack export failure"
    update_queue_row "train_threeway_seqholdout" "cancelled" "" "$(timestamp_now)" "cancelled after DanceTrack export failure"
    append_queue_registry "failed" "queue failed because DanceTrack export failed"
    echo "[$(timestamp_now)] export failed, stop queue" >> "$QUEUE_LOG"
    exit 1
  fi
  sleep 120
done

if [[ ! -s "$EXPORT_JSONL" ]]; then
  update_queue_row "wait_export" "failed" "" "$(timestamp_now)" "DanceTrack export finished but local_contention_units.jsonl is missing or empty"
  update_queue_row "train_mot17_dance_val_mot20" "cancelled" "" "$(timestamp_now)" "cancelled because DanceTrack JSONL is unavailable"
  update_queue_row "train_threeway_seqholdout" "cancelled" "" "$(timestamp_now)" "cancelled because DanceTrack JSONL is unavailable"
  append_queue_registry "failed" "queue failed because DanceTrack local contention export JSONL is missing"
  echo "[$(timestamp_now)] missing DanceTrack JSONL, stop queue" >> "$QUEUE_LOG"
  exit 1
fi

update_queue_row "wait_export" "success" "" "$(timestamp_now)" "DanceTrack full export completed successfully"
echo "[$(timestamp_now)] DanceTrack export success" >> "$QUEUE_LOG"

run_training \
  "train_mot17_dance_val_mot20" \
  "$TRAIN1_DIR" \
  "$TRAIN1_LOG" \
  --jsonl "$MOT17_JSONL" "$MOT20_JSONL" "$EXPORT_JSONL" \
  --out-dir "$TRAIN1_DIR" \
  --epochs 12 \
  --batch-size 512 \
  --hidden-dim 64 \
  --lr 0.001 \
  --sampler-mode balanced \
  --selection-metric average_precision \
  --positive-weight-power 0.5 \
  --train-sequences "$MOT17_ALL,$DANCE_ALL" \
  --val-sequences "$MOT20_ALL" \
  --dataset-tag "MOT17+MOT20+DanceTrack" \
  --split-label mixed || {
    update_queue_row "train_threeway_seqholdout" "cancelled" "" "$(timestamp_now)" "cancelled after previous training failed"
    append_queue_registry "failed" "queue failed at train_mot17_dance_val_mot20"
    exit 1
  }

run_training \
  "train_threeway_seqholdout" \
  "$TRAIN2_DIR" \
  "$TRAIN2_LOG" \
  --jsonl "$MOT17_JSONL" "$MOT20_JSONL" "$EXPORT_JSONL" \
  --out-dir "$TRAIN2_DIR" \
  --epochs 12 \
  --batch-size 512 \
  --hidden-dim 64 \
  --lr 0.001 \
  --sampler-mode balanced \
  --selection-metric average_precision \
  --positive-weight-power 0.5 \
  --train-sequences "$TRAIN2_MOT17_TRAIN,$TRAIN2_MOT20_TRAIN,$TRAIN2_DANCE_TRAIN" \
  --val-sequences "$TRAIN2_MOT17_VAL,$TRAIN2_MOT20_VAL,$TRAIN2_DANCE_VAL" \
  --dataset-tag "MOT17+MOT20+DanceTrack" \
  --split-label mixed || {
    append_queue_registry "failed" "queue failed at train_threeway_seqholdout"
    exit 1
  }

append_queue_registry "success" "queue completed: DanceTrack export follow-up trainings finished"
echo "[$(timestamp_now)] queue finished successfully" >> "$QUEUE_LOG"
