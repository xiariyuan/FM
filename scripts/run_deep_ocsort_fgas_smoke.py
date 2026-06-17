#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path("/gemini/code/FMtrack-main/FM-Track")
DEEP_ROOT = REPO_ROOT / "external" / "Deep-OC-SORT-main"
REGISTRY_CSV = REPO_ROOT / "outputs" / "experiment_registry.csv"

QUEUE_FIELDS = [
    "step",
    "name",
    "status",
    "out_dir",
    "summary_csv",
    "log_path",
    "started_at",
    "finished_at",
    "notes",
]

METRIC_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
    "summary_txt",
    "detailed_csv",
    "tracker_dir",
]

DELTA_FIELDS = [
    "name",
    "seq",
    "delta_HOTA",
    "delta_AssA",
    "delta_IDF1",
    "delta_MOTA",
    "delta_IDs",
    "delta_Frag",
]

PER_SEQUENCE_FIELDS = [
    "name",
    "seq",
    "HOTA",
    "AssA",
    "IDF1",
    "MOTA",
    "IDs",
    "Frag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded Deep-OC-SORT raw vs FGAS paired eval on one or more MOT17 half-val sequences.")
    parser.add_argument("--seq-name", default="MOT17-05-FRCNN")
    parser.add_argument(
        "--seq-names",
        nargs="+",
        default=None,
        help="optional explicit sequence list; overrides --seq-name",
    )
    parser.add_argument("--out-root", default="")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "fgas_block_resolver_v3_nofreq_20260331_1" / "best.pt"),
    )
    parser.add_argument("--block-primitive-checkpoint", default="", help="optional FGAS block-level ambiguity primitive checkpoint")
    parser.add_argument("--fgas-block-primitive-conf-thresh", type=float, default=0.5, help="minimum block confidence required for primitive takeover")
    parser.add_argument("--fgas-primitive-direct-takeover", action="store_true", help="let FGAS primitive takeover blocks emit direct local assignment actions instead of soft score blending")
    parser.add_argument("--block-matcher-checkpoint", default="", help="optional FGAS true block matcher checkpoint")
    parser.add_argument("--fgas-block-matcher-margin-thresh", type=float, default=0.0, help="minimum joint-assignment margin required for matcher takeover; <=0 uses the checkpoint margin")
    parser.add_argument("--fgas-block-matcher-base-margin-thresh", type=float, default=0.0, help="maximum baseline row-margin allowed on matcher-changed rows; <=0 disables this gate")
    parser.add_argument("--fgas-block-matcher-base-logit-scale-override", type=float, default=None, help="optional runtime override for matcher base-residual logit scale; only affects checkpoints trained with residual logits")
    parser.add_argument("--fgas-block-matcher-force-only", action="store_true", help="matcher takeover emits forced matches only and ignores row/col blocking")
    parser.add_argument("--fgas-block-matcher-forceonly-keep-base-on-nomatch", action="store_true", help="for matcher force-only mode, convert solver row-no-match decisions back to the baseline match whenever a baseline winner exists")
    parser.add_argument("--fgas-block-matcher-skip-single-row-unchanged-takeover", action="store_true", help="reject matcher takeover on single-row blocks when the solver keeps the baseline row winner unchanged")
    parser.add_argument("--fgas-block-matcher-stale-match-bias", type=float, default=0.0, help="optional logit bias that favors assigning detections over row-no-match for stale established tracks in small matcher blocks")
    parser.add_argument("--fgas-block-matcher-stale-match-mode", choices=["all_edges", "row_nomatch"], default="all_edges", help="how stale-track matcher bias is applied: all_edges boosts every valid edge on the row, row_nomatch only lowers the row-no-match prior")
    parser.add_argument("--fgas-block-matcher-stale-match-min-time-since-update", type=int, default=0, help="minimum time_since_update required before applying the stale-track matcher bias")
    parser.add_argument("--fgas-block-matcher-stale-match-max-hit-streak", type=int, default=0, help="maximum hit_streak allowed before applying the stale-track matcher bias")
    parser.add_argument("--fgas-block-matcher-stale-match-min-hits", type=int, default=0, help="minimum lifetime hits required before applying the stale-track matcher bias")
    parser.add_argument("--fgas-block-matcher-stale-match-max-component-rows", type=int, default=0, help="optional maximum block row count allowed for the stale-track matcher bias; <=0 disables the size filter")
    parser.add_argument("--fgas-block-matcher-stale-match-max-component-cols", type=int, default=0, help="optional maximum block col count allowed for the stale-track matcher bias; <=0 disables the size filter")
    parser.add_argument("--fgas-block-matcher-skip-raw-matched-unchanged-external-competition", action="store_true", help="for matcher force-only mode, reject unchanged takeover rows when a track already has a raw match and the baseline-best detection is owned by another raw track")
    parser.add_argument("--fgas-block-matcher-skip-raw-matched-unchanged-external-competition-max-assignment-margin", type=float, default=0.0, help="optional maximum matcher assignment margin allowed for the raw-matched unchanged external-competition gate; <=0 disables the extra margin filter")
    parser.add_argument("--fgas-block-matcher-skip-raw-matched-unchanged-external-competition-owner-margin-thresh", type=float, default=0.0, help="optional minimum raw-owner local score margin required between the stolen baseline-best detection and the target track's raw detection; <=0 disables this extra filter")
    parser.add_argument("--fgas-block-matcher-skip-raw-matched-unchanged-target-margin-thresh", type=float, default=0.0, help="optional minimum target-track base-best vs raw-det score margin required to allow an unchanged raw-matched takeover; <=0 disables this extra protection")
    parser.add_argument("--fgas-matcher-acceptance-gate-checkpoint", default="", help="optional learned gate checkpoint for raw-matched unchanged matcher changed-match rows")
    parser.add_argument("--fgas-matcher-acceptance-gate-thresh", type=float, default=0.5, help="minimum matcher gate probability required to keep a raw-matched unchanged matcher changed-match row")
    parser.add_argument("--fgas-matcher-acceptance-gate-min-component-rows", type=int, default=0, help="optional minimum component row count required before applying the raw-matched unchanged matcher acceptance gate; <=0 disables the shape filter")
    parser.add_argument("--fgas-matcher-acceptance-gate-min-component-cols", type=int, default=0, help="optional minimum component col count required before applying the raw-matched unchanged matcher acceptance gate; <=0 disables the shape filter")
    parser.add_argument("--fgas-matcher-changed-acceptance-gate-checkpoint", default="", help="optional learned gate checkpoint for raw-matched solver-changed matcher rows")
    parser.add_argument("--fgas-matcher-changed-acceptance-gate-thresh", type=float, default=0.5, help="minimum matcher gate probability required to keep a raw-matched solver-changed matcher row")
    parser.add_argument("--fgas-block-matcher-skip-single-row-unchanged-external-competition", action="store_true", help="for matcher force-only mode, reject single-row unchanged takeover when the track already has a raw match elsewhere and the baseline-best detection is owned by another raw track")
    parser.add_argument("--fgas-block-matcher-skip-single-row-unchanged-external-competition-max-assignment-margin", type=float, default=0.0, help="optional maximum matcher assignment margin allowed for the external-competition single-row unchanged gate; <=0 disables the extra margin filter")
    parser.add_argument("--fgas-block-matcher-skip-single-row-unchanged-external-competition-owner-margin-thresh", type=float, default=0.0, help="optional minimum raw-owner local score margin required between the stolen baseline-best detection and the target track's raw detection; <=0 disables this extra filter")
    parser.add_argument("--fgas-block-matcher-skip-multirow-owned-competition", action="store_true", help="for matcher force-only mode, reject takeover rows when a raw-matched track tries to steal another raw-owned baseline-best detection inside a multi-row block with at least 2 rows and 3 cols")
    parser.add_argument("--fgas-block-matcher-skip-multirow-owned-competition-owner-margin-thresh", type=float, default=0.0, help="minimum raw-owner base-best vs target-raw score margin required for the multi-row owned-competition gate; <=0 disables the extra owner-margin filter")
    parser.add_argument("--fgas-block-matcher-skip-temporal-inconsistent-owner-swap", action="store_true", help="for matcher force-only mode, reject owner-swap rows when the target track is still temporally active but the raw owner track is already stale")
    parser.add_argument("--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-target-max-time-since-update", type=int, default=1, help="maximum target-track time_since_update allowed for the temporal owner-swap reject rule")
    parser.add_argument("--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-target-min-hit-streak", type=int, default=2, help="minimum target-track hit_streak required for the temporal owner-swap reject rule")
    parser.add_argument("--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-owner-min-time-since-update", type=int, default=2, help="minimum raw-owner track time_since_update required for the temporal owner-swap reject rule")
    parser.add_argument("--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-owner-max-hit-streak", type=int, default=0, help="maximum raw-owner track hit_streak allowed for the temporal owner-swap reject rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active", action="store_true", help="protect raw-matched active anchors on ownerless multi-row changed matcher rows")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-max-time-since-update", type=int, default=1, help="maximum target-track time_since_update allowed for the active ownerless raw-matched protect rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-min-hit-streak", type=int, default=8, help="minimum target-track hit_streak required for the active ownerless raw-matched protect rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-min-hits", type=int, default=20, help="minimum target-track hits required for the active ownerless raw-matched protect rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-min-component-rows", type=int, default=2, help="minimum component row count required for the active ownerless raw-matched protect rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-min-component-cols", type=int, default=3, help="minimum component col count required for the active ownerless raw-matched protect rule")
    parser.add_argument("--fgas-block-matcher-protect-ownerless-rawmatched-active-max-assignment-margin", type=float, default=0.0, help="optional maximum matcher assignment margin allowed for the active ownerless raw-matched protect rule; <=0 disables this extra filter")
    parser.add_argument("--fgas-lifecycle-reclaim-enable", action="store_true", help="enable the post-association lifecycle reclaim stage for strong stale tracks on the FGAS arm")
    parser.add_argument("--fgas-lifecycle-reclaim-min-time-since-update", type=int, default=2, help="minimum time_since_update required before a stale track is eligible for lifecycle reclaim")
    parser.add_argument("--fgas-lifecycle-reclaim-max-time-since-update", type=int, default=8, help="maximum time_since_update allowed for lifecycle reclaim")
    parser.add_argument("--fgas-lifecycle-reclaim-min-hits", type=int, default=15, help="minimum lifetime hits required before a stale track can reclaim an unmatched detection")
    parser.add_argument("--fgas-lifecycle-reclaim-min-box-iou", type=float, default=0.65, help="minimum max(last-observation IoU, predicted-box IoU) required for lifecycle reclaim")
    parser.add_argument("--fgas-lifecycle-reclaim-min-box-area-ratio", type=float, default=0.5, help="minimum det/track box-area ratio allowed for lifecycle reclaim")
    parser.add_argument("--fgas-lifecycle-reclaim-max-box-area-ratio", type=float, default=2.0, help="maximum det/track box-area ratio allowed for lifecycle reclaim")
    parser.add_argument("--fgas-lifecycle-reclaim-min-emb-similarity", type=float, default=0.0, help="optional minimum flattened cosine similarity required for lifecycle reclaim; <=0 disables the embedding gate")
    parser.add_argument("--fgas-block-matcher-skip-ownerless-rawmatched-single-row-changed", action="store_true", help="for matcher force-only mode, reject ownerless raw-matched changed rows on 1x2 blocks when the baseline row margin is already above a protection threshold")
    parser.add_argument("--fgas-block-matcher-skip-ownerless-rawmatched-single-row-base-margin-thresh", type=float, default=0.0, help="minimum baseline row margin required to reject ownerless raw-matched changed 1x2 rows; <=0 disables the margin gate")
    parser.add_argument("--fgas-block-matcher-skip-ownerless-rawmatched-multirow-changed", action="store_true", help="for matcher force-only mode, reject ownerless raw-matched changed rows on multi-row blocks when matcher confidence is already too strong")
    parser.add_argument("--fgas-block-matcher-skip-ownerless-rawmatched-multirow-assignment-margin-thresh", type=float, default=0.0, help="minimum matcher assignment margin required to reject ownerless raw-matched changed multi-row rows; <=0 disables the margin gate")
    parser.add_argument("--pair-scorer-checkpoint", default="", help="optional detector-aware FGAS pair scorer checkpoint")
    parser.add_argument("--fgas-block-gate-checkpoint", default="", help="optional trained FGAS block gate checkpoint for pair-scorer block filtering")
    parser.add_argument("--fgas-block-gate-thresh", type=float, default=0.5, help="minimum block-gate probability required to allow a pair-scorer block intervention")
    parser.add_argument("--fgas-assignment-mode", default="blend", choices=["blend", "replace"])
    parser.add_argument("--fgas-blend-weight", type=float, default=0.5)
    parser.add_argument("--fgas-pair-ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--fgas-soft-enable", action="store_true", help="enable FGAS soft score refinement during first association")
    parser.add_argument("--fgas-soft-lambda", type=float, default=0.5, help="blend weight for FGAS soft refinement")
    parser.add_argument("--fgas-soft-allow-fallback", action="store_true", help="allow FGAS soft refinement even when the primitive falls back instead of taking over the block")
    parser.add_argument("--fgas-soft-only-changed-blocks", action="store_true", help="apply FGAS soft refinement only on blocks whose FGAS top-1 differs from the baseline")
    parser.add_argument("--fgas-soft-only-changed-rows", action="store_true", help="apply FGAS soft refinement only on rows whose FGAS top-1 differs from the baseline")
    parser.add_argument("--fgas-soft-only-changed-frontier", action="store_true", help="apply FGAS soft refinement only on changed rows and edges incident to their competing columns")
    parser.add_argument("--fgas-soft-row-base-margin-thresh", type=float, default=1.0, help="maximum baseline row top1-second margin allowed for FGAS soft changed-row application")
    parser.add_argument("--fgas-soft-changed-row-flip-gap-thresh", type=float, default=0.0, help="minimum FGAS probability advantage of refined winner over baseline winner for changed-row soft application")
    parser.add_argument("--fgas-soft-changed-row-refined-margin-thresh", type=float, default=0.0, help="minimum FGAS refined top1-second margin for changed-row soft application")
    parser.add_argument("--fgas-acceptance-gate-checkpoint", default="", help="optional trained acceptance gate checkpoint for FGAS changed-row filtering")
    parser.add_argument("--fgas-acceptance-gate-thresh", type=float, default=0.5, help="acceptance threshold for the FGAS changed-row gate")
    parser.add_argument("--fgas-matcher-case-export-jsonl", default="", help="optional JSONL path for exporting matcher changed-row runtime cases")
    parser.add_argument("--disable-controller", action="store_true", help="disable FGAS hard controller actions for the FGAS arm")
    parser.add_argument(
        "--allow-controller-with-force-only",
        action="store_true",
        help="also pass FGAS controller flags when matcher force-only mode is enabled; useful for reproducing older runtime families",
    )
    parser.add_argument("--fgas-controller-only-changed-blocks", action="store_true", help="limit FGAS controller actions to blocks where FGAS changes the block-level top-1")
    parser.add_argument("--fgas-controller-edge-thresh", type=float, default=0.7)
    parser.add_argument("--fgas-controller-row-defer-thresh", type=float, default=0.7)
    parser.add_argument("--fgas-controller-col-newborn-thresh", type=float, default=0.7)
    parser.add_argument("--fgas-controller-margin-thresh", type=float, default=0.1)
    parser.add_argument("--fgas-controller-ambiguity-margin", type=float, default=0.04)
    parser.add_argument("--compare-only", action="store_true", help="skip tracking/eval and only backfill compare artifacts from an existing run")
    parser.add_argument("--registry-csv", default=str(REGISTRY_CSV))
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def update_row(rows: List[Dict[str, object]], step: str, **updates: object) -> None:
    for row in rows:
        if str(row["step"]) == step:
            row.update(updates)
            return
    raise KeyError(f"Missing queue step: {step}")


def run_step(cmd: List[str], log_path: Path, *, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[started_at] {now_iso()}\n")
        handle.write(f"[cwd] {cwd}\n")
        handle.write("[cmd] " + " ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[finished_at] {now_iso()}\n")
        handle.write(f"[return_code] {process.returncode}\n")
    return int(process.returncode)


def append_registry(summary_csv: Path, run_root: Path, status: str, notes: str, registry_csv: str) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "append_experiment_record.py"),
        "--csv",
        str(registry_csv),
        "--kind",
        "eval",
        "--status",
        status,
        "--script",
        "scripts/run_deep_ocsort_fgas_smoke.py",
        "--dataset",
        "MOT17",
        "--split",
        "val_half",
        "--tracker-family",
        "deep_ocsort_fgas",
        "--variant",
        run_root.name,
        "--tag",
        "deep_ocsort_fgas_smoke",
        "--run-root",
        str(run_root),
        "--summary-csv",
        str(summary_csv),
        "--notes",
        notes,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def mark_running_rows_failed(rows: List[Dict[str, object]], summary_csv: Path, reason: str) -> None:
    finished_at = now_iso()
    changed = False
    for row in rows:
        if str(row.get("status", "")) == "running":
            row["status"] = "failed"
            row["finished_at"] = finished_at
            row["notes"] = f"{row.get('notes', '')} | failed: {reason}".strip()
            changed = True
    if changed:
        write_rows(summary_csv, QUEUE_FIELDS, rows)


def parse_summary_txt(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=" ")
        rows = []
        for row in reader:
            filtered = [token for token in row if token != ""]
            if filtered:
                rows.append(filtered)
    if len(rows) < 2:
        raise RuntimeError(f"Unexpected TrackEval summary format: {path}")
    fields = rows[0]
    values = rows[1]
    data: Dict[str, float] = {}
    for key, value in zip(fields, values):
        try:
            data[key] = float(value)
        except ValueError:
            continue
    return data


def parse_step_log_metadata(log_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not log_path.is_file():
        return meta
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("[") or "] " not in line:
                continue
            key, value = line.split("] ", 1)
            meta[key[1:]] = value
    return meta


def resolve_existing_dir(path: Path) -> Path:
    if path.is_dir():
        return path.resolve()
    if path.is_absolute():
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            rel = None
        if rel is not None:
            alt = (DEEP_ROOT / rel).resolve()
            if alt.is_dir():
                return alt
    else:
        alt = (DEEP_ROOT / path).resolve()
        if alt.is_dir():
            return alt
    raise FileNotFoundError(f"Missing directory: {path}")


def resolve_existing_file(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    try:
        alt = resolve_existing_dir(path.parent) / path.name
    except FileNotFoundError:
        alt = None
    if alt is not None and alt.is_file():
        return alt.resolve()
    raise FileNotFoundError(f"Missing file: {path}")


def repair_step_from_artifacts(
    rows: List[Dict[str, object]],
    summary_csv: Path,
    *,
    step: str,
    out_dir: Path,
    log_path: Path,
    notes: str,
) -> None:
    current_rows = {str(row["step"]): row for row in rows}
    row = current_rows.get(step)
    if row is None:
        return
    if str(row.get("status", "")) == "success":
        return
    meta = parse_step_log_metadata(log_path)
    if meta.get("return_code") != "0":
        return
    if not out_dir.is_dir():
        return
    row["status"] = "success"
    row["out_dir"] = str(out_dir)
    row["summary_csv"] = str(summary_csv)
    row["log_path"] = str(log_path)
    row["finished_at"] = meta.get("finished_at", row.get("finished_at", ""))
    if not row.get("started_at"):
        row["started_at"] = meta.get("started_at", "")
    row["notes"] = notes
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def resolve_seq_names(args: argparse.Namespace) -> List[str]:
    if args.seq_names:
        return [str(seq) for seq in args.seq_names]
    return [str(args.seq_name)]


def seq_note(seq_names: List[str]) -> str:
    return "|".join(seq_names)


def load_per_sequence_metrics(detailed_csv: Path, label: str) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    with detailed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = str(row.get("seq", ""))
            if not seq or seq == "COMBINED":
                continue
            rows.append(
                {
                    "name": label,
                    "seq": seq,
                    "HOTA": float(row["HOTA___AUC"]) * 100.0,
                    "AssA": float(row["AssA___AUC"]) * 100.0,
                    "IDF1": float(row["IDF1"]) * 100.0,
                    "MOTA": float(row["MOTA"]) * 100.0,
                    "IDs": int(round(float(row["IDs"]))),
                    "Frag": int(round(float(row["Frag"]))),
                }
            )
    return rows


def ensure_success(step: str, return_code: int, rows: List[Dict[str, object]], summary_csv: Path, out_dir: Path, log_path: Path, notes: str) -> None:
    finished_at = now_iso()
    status = "success" if return_code == 0 else "failed"
    update_row(
        rows,
        step,
        status=status,
        finished_at=finished_at,
        out_dir=str(out_dir),
        summary_csv=str(summary_csv),
        log_path=str(log_path),
        notes=notes,
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    if return_code != 0:
        raise RuntimeError(f"Step failed: {step}")


def backfill_compare(
    *,
    rows: List[Dict[str, object]],
    summary_csv: Path,
    logs_dir: Path,
    run_root: Path,
    seq_label: str,
    raw_track_out: Path,
    raw_eval_out: Path,
    fgas_track_out: Path,
    fgas_eval_out: Path,
    metrics_compare_csv: Path,
    metrics_delta_csv: Path,
    per_sequence_csv: Path,
) -> None:
    update_row(rows, "compare", status="running", started_at=now_iso())
    write_rows(summary_csv, QUEUE_FIELDS, rows)

    raw_track_dir = resolve_existing_dir(raw_track_out)
    fgas_track_dir = resolve_existing_dir(fgas_track_out)
    raw_eval_dir = resolve_existing_dir(raw_eval_out)
    fgas_eval_dir = resolve_existing_dir(fgas_eval_out)

    raw_summary_txt = resolve_existing_file(raw_eval_dir / "pedestrian_summary.txt")
    raw_detailed_csv = resolve_existing_file(raw_eval_dir / "pedestrian_detailed.csv")
    fgas_summary_txt = resolve_existing_file(fgas_eval_dir / "pedestrian_summary.txt")
    fgas_detailed_csv = resolve_existing_file(fgas_eval_dir / "pedestrian_detailed.csv")
    raw_metrics = parse_summary_txt(raw_summary_txt)
    fgas_metrics = parse_summary_txt(fgas_summary_txt)
    per_sequence_rows = load_per_sequence_metrics(raw_detailed_csv, "raw") + load_per_sequence_metrics(fgas_detailed_csv, "fgas")

    compare_rows = [
        {
            "name": "raw",
            "seq": seq_label,
            "HOTA": raw_metrics.get("HOTA", ""),
            "AssA": raw_metrics.get("AssA", ""),
            "IDF1": raw_metrics.get("IDF1", ""),
            "MOTA": raw_metrics.get("MOTA", ""),
            "IDs": raw_metrics.get("IDs", ""),
            "Frag": raw_metrics.get("Frag", ""),
            "summary_txt": str(raw_summary_txt),
            "detailed_csv": str(raw_detailed_csv),
            "tracker_dir": str(raw_track_dir),
        },
        {
            "name": "fgas",
            "seq": seq_label,
            "HOTA": fgas_metrics.get("HOTA", ""),
            "AssA": fgas_metrics.get("AssA", ""),
            "IDF1": fgas_metrics.get("IDF1", ""),
            "MOTA": fgas_metrics.get("MOTA", ""),
            "IDs": fgas_metrics.get("IDs", ""),
            "Frag": fgas_metrics.get("Frag", ""),
            "summary_txt": str(fgas_summary_txt),
            "detailed_csv": str(fgas_detailed_csv),
            "tracker_dir": str(fgas_track_dir),
        },
    ]
    delta_rows = [
        {
            "name": "fgas_minus_raw",
            "seq": seq_label,
            "delta_HOTA": float(fgas_metrics.get("HOTA", 0.0)) - float(raw_metrics.get("HOTA", 0.0)),
            "delta_AssA": float(fgas_metrics.get("AssA", 0.0)) - float(raw_metrics.get("AssA", 0.0)),
            "delta_IDF1": float(fgas_metrics.get("IDF1", 0.0)) - float(raw_metrics.get("IDF1", 0.0)),
            "delta_MOTA": float(fgas_metrics.get("MOTA", 0.0)) - float(raw_metrics.get("MOTA", 0.0)),
            "delta_IDs": float(fgas_metrics.get("IDs", 0.0)) - float(raw_metrics.get("IDs", 0.0)),
            "delta_Frag": float(fgas_metrics.get("Frag", 0.0)) - float(raw_metrics.get("Frag", 0.0)),
        }
    ]
    write_rows(metrics_compare_csv, METRIC_FIELDS, compare_rows)
    write_rows(metrics_delta_csv, DELTA_FIELDS, delta_rows)
    write_rows(per_sequence_csv, PER_SEQUENCE_FIELDS, per_sequence_rows)

    compare_log = logs_dir / "compare.log"
    compare_log.write_text(
        f"raw_summary={raw_summary_txt}\nfgas_summary={fgas_summary_txt}\nmetrics_compare={metrics_compare_csv}\nmetrics_delta={metrics_delta_csv}\nper_sequence_metrics={per_sequence_csv}\n",
        encoding="utf-8",
    )
    update_row(
        rows,
        "compare",
        status="success",
        finished_at=now_iso(),
        out_dir=str(run_root),
        summary_csv=str(summary_csv),
        log_path=str(compare_log),
        notes=f"compare complete for {seq_label}",
    )
    write_rows(summary_csv, QUEUE_FIELDS, rows)


def main() -> None:
    args = parse_args()
    seq_names = resolve_seq_names(args)
    seq_label = seq_note(seq_names)
    run_root = (Path(args.out_root) if args.out_root else REPO_ROOT / "outputs" / f"deep_ocsort_fgas_smoke_{timestamp_tag()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    trackers_root = (run_root / "results" / "trackers").resolve()
    metrics_compare_csv = run_root / "metrics_compare.csv"
    metrics_delta_csv = run_root / "metrics_delta.csv"
    per_sequence_csv = run_root / "per_sequence_metrics.csv"
    summary_csv = run_root / "summary.csv"

    raw_exp = f"{run_root.name}_raw"
    fgas_exp = f"{run_root.name}_fgas"

    raw_track_out = trackers_root / "MOT17-val" / raw_exp
    raw_eval_out = trackers_root / "MOT17-val" / (raw_exp + "_post")
    fgas_track_out = trackers_root / "MOT17-val" / fgas_exp
    fgas_eval_out = trackers_root / "MOT17-val" / (fgas_exp + "_post")

    if args.compare_only:
        rows = read_rows(summary_csv)
        if not rows:
            raise FileNotFoundError(f"Missing summary.csv for compare-only run: {summary_csv}")
        try:
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="raw_track",
                out_dir=raw_track_out,
                log_path=logs_dir / "raw_track.log",
                notes=f"raw tracking complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="raw_eval",
                out_dir=raw_eval_out,
                log_path=logs_dir / "raw_eval.log",
                notes=f"raw eval complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="fgas_track",
                out_dir=fgas_track_out,
                log_path=logs_dir / "fgas_track.log",
                notes=f"fgas tracking complete for {seq_label}",
            )
            repair_step_from_artifacts(
                rows,
                summary_csv,
                step="fgas_eval",
                out_dir=fgas_eval_out,
                log_path=logs_dir / "fgas_eval.log",
                notes=f"fgas eval complete for {seq_label}",
            )
            backfill_compare(
                rows=rows,
                summary_csv=summary_csv,
                logs_dir=logs_dir,
                run_root=run_root,
                seq_label=seq_label,
                raw_track_out=raw_track_out,
                raw_eval_out=raw_eval_out,
                fgas_track_out=fgas_track_out,
                fgas_eval_out=fgas_eval_out,
                metrics_compare_csv=metrics_compare_csv,
                metrics_delta_csv=metrics_delta_csv,
                per_sequence_csv=per_sequence_csv,
            )
            append_registry(summary_csv, run_root, "success", f"compare-only backfill completed on {seq_label}", args.registry_csv)
            return
        except Exception as exc:
            mark_running_rows_failed(rows, summary_csv, str(exc))
            append_registry(summary_csv, run_root, "failed", f"compare-only backfill failed on {seq_label}: {exc}", args.registry_csv)
            raise

    rows: List[Dict[str, object]] = [
        {
            "step": "raw_track",
            "name": raw_exp,
            "status": "running",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_track.log"),
            "started_at": now_iso(),
            "finished_at": "",
            "notes": f"Deep-OC-SORT raw tracking on {seq_label}",
        },
        {
            "step": "raw_eval",
            "name": raw_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "raw_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"TrackEval for {raw_exp}",
        },
        {
            "step": "fgas_track",
            "name": fgas_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "fgas_track.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Deep-OC-SORT + FGAS tracking on {seq_label}",
        },
        {
            "step": "fgas_eval",
            "name": fgas_exp,
            "status": "pending",
            "out_dir": "",
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "fgas_eval.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"TrackEval for {fgas_exp}",
        },
        {
            "step": "compare",
            "name": run_root.name,
            "status": "pending",
            "out_dir": str(run_root),
            "summary_csv": str(summary_csv),
            "log_path": str(logs_dir / "compare.log"),
            "started_at": "",
            "finished_at": "",
            "notes": f"Compare raw vs FGAS on {seq_label}",
        },
    ]
    write_rows(summary_csv, QUEUE_FIELDS, rows)
    append_registry(summary_csv, run_root, "running", f"started paired eval on {seq_label}", args.registry_csv)
    try:
        raw_track_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            "mot17",
            "--result_folder",
            str(trackers_root),
            "--exp_name",
            raw_exp,
            "--seq-filter",
            *seq_names,
            "--post",
            "--grid_off",
            "--new_kf_off",
            "--w_assoc_emb",
            "0.75",
            "--aw_param",
            "0.5",
        ]
        raw_track_log = logs_dir / "raw_track.log"
        return_code = run_step(raw_track_cmd, raw_track_log, cwd=DEEP_ROOT)
        ensure_success("raw_track", return_code, rows, summary_csv, raw_track_out, raw_track_log, f"raw tracking complete for {seq_label}")

        update_row(rows, "raw_eval", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        raw_eval_cmd = [
            sys.executable,
            "external/TrackEval/scripts/run_mot_challenge.py",
            "--BENCHMARK",
            "MOT17",
            "--SPLIT_TO_EVAL",
            "val",
            "--GT_FOLDER",
            str(DEEP_ROOT / "results" / "gt"),
            "--TRACKERS_FOLDER",
            str(trackers_root),
            "--TRACKERS_TO_EVAL",
            raw_exp + "_post",
            "--SEQ_INFO",
            *seq_names,
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
            "--USE_PARALLEL",
            "False",
            "--PRINT_ONLY_COMBINED",
            "True",
        ]
        raw_eval_log = logs_dir / "raw_eval.log"
        return_code = run_step(raw_eval_cmd, raw_eval_log, cwd=DEEP_ROOT)
        ensure_success("raw_eval", return_code, rows, summary_csv, raw_eval_out, raw_eval_log, f"raw eval complete for {seq_label}")

        update_row(rows, "fgas_track", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        fgas_track_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            "mot17",
            "--result_folder",
            str(trackers_root),
            "--exp_name",
            fgas_exp,
            "--seq-filter",
            *seq_names,
            "--post",
            "--grid_off",
            "--new_kf_off",
            "--w_assoc_emb",
            "0.75",
            "--aw_param",
            "0.5",
            "--fgas-enable",
            "--fgas-topk",
            "5",
            "--fgas-max-rows",
            "3",
            "--fgas-max-cols",
            "3",
            "--fgas-blend-weight",
            str(args.fgas_blend_weight),
            "--fgas-assignment-mode",
            str(args.fgas_assignment_mode),
            "--fgas-row-nomatch-weight",
            "0.0",
            "--fgas-pair-ambiguity-margin",
            str(args.fgas_pair_ambiguity_margin),
        ]
        if args.checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-resolver-checkpoint",
                    str(args.checkpoint),
                ]
            )
        if args.block_primitive_checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-block-primitive-checkpoint",
                    str(args.block_primitive_checkpoint),
                    "--fgas-block-primitive-conf-thresh",
                    str(args.fgas_block_primitive_conf_thresh),
                ]
            )
        if args.fgas_primitive_direct_takeover:
            fgas_track_cmd.append("--fgas-primitive-direct-takeover")
        if args.block_matcher_checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-block-matcher-checkpoint",
                    str(args.block_matcher_checkpoint),
                    "--fgas-block-matcher-margin-thresh",
                    str(args.fgas_block_matcher_margin_thresh),
                    "--fgas-block-matcher-base-margin-thresh",
                    str(args.fgas_block_matcher_base_margin_thresh),
                ]
            )
            if args.fgas_block_matcher_base_logit_scale_override is not None:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-base-logit-scale-override",
                        str(args.fgas_block_matcher_base_logit_scale_override),
                    ]
                )
            if args.fgas_block_matcher_force_only:
                fgas_track_cmd.append("--fgas-block-matcher-force-only")
            if args.fgas_block_matcher_forceonly_keep_base_on_nomatch:
                fgas_track_cmd.append("--fgas-block-matcher-forceonly-keep-base-on-nomatch")
            if args.fgas_block_matcher_skip_single_row_unchanged_takeover:
                fgas_track_cmd.append("--fgas-block-matcher-skip-single-row-unchanged-takeover")
            if args.fgas_block_matcher_stale_match_bias > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-stale-match-bias",
                        str(args.fgas_block_matcher_stale_match_bias),
                        "--fgas-block-matcher-stale-match-mode",
                        str(args.fgas_block_matcher_stale_match_mode),
                        "--fgas-block-matcher-stale-match-min-time-since-update",
                        str(args.fgas_block_matcher_stale_match_min_time_since_update),
                        "--fgas-block-matcher-stale-match-max-hit-streak",
                        str(args.fgas_block_matcher_stale_match_max_hit_streak),
                        "--fgas-block-matcher-stale-match-min-hits",
                        str(args.fgas_block_matcher_stale_match_min_hits),
                    ]
                )
                if args.fgas_block_matcher_stale_match_max_component_rows > 0:
                    fgas_track_cmd.extend(
                        [
                            "--fgas-block-matcher-stale-match-max-component-rows",
                            str(args.fgas_block_matcher_stale_match_max_component_rows),
                        ]
                    )
                if args.fgas_block_matcher_stale_match_max_component_cols > 0:
                    fgas_track_cmd.extend(
                        [
                            "--fgas-block-matcher-stale-match-max-component-cols",
                            str(args.fgas_block_matcher_stale_match_max_component_cols),
                        ]
                    )
            if args.fgas_block_matcher_skip_raw_matched_unchanged_external_competition:
                fgas_track_cmd.append("--fgas-block-matcher-skip-raw-matched-unchanged-external-competition")
            if args.fgas_block_matcher_skip_raw_matched_unchanged_external_competition_max_assignment_margin > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-raw-matched-unchanged-external-competition-max-assignment-margin",
                        str(args.fgas_block_matcher_skip_raw_matched_unchanged_external_competition_max_assignment_margin),
                    ]
                )
            if args.fgas_block_matcher_skip_raw_matched_unchanged_external_competition_owner_margin_thresh > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-raw-matched-unchanged-external-competition-owner-margin-thresh",
                        str(args.fgas_block_matcher_skip_raw_matched_unchanged_external_competition_owner_margin_thresh),
                    ]
                )
            if args.fgas_block_matcher_skip_raw_matched_unchanged_target_margin_thresh > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-raw-matched-unchanged-target-margin-thresh",
                        str(args.fgas_block_matcher_skip_raw_matched_unchanged_target_margin_thresh),
                    ]
                )
            if args.fgas_matcher_acceptance_gate_checkpoint:
                fgas_track_cmd.extend(
                    [
                        "--fgas-matcher-acceptance-gate-checkpoint",
                        str(args.fgas_matcher_acceptance_gate_checkpoint),
                        "--fgas-matcher-acceptance-gate-thresh",
                        str(args.fgas_matcher_acceptance_gate_thresh),
                    ]
                )
                if args.fgas_matcher_acceptance_gate_min_component_rows > 0:
                    fgas_track_cmd.extend(
                        [
                            "--fgas-matcher-acceptance-gate-min-component-rows",
                            str(args.fgas_matcher_acceptance_gate_min_component_rows),
                        ]
                    )
                if args.fgas_matcher_acceptance_gate_min_component_cols > 0:
                    fgas_track_cmd.extend(
                        [
                            "--fgas-matcher-acceptance-gate-min-component-cols",
                            str(args.fgas_matcher_acceptance_gate_min_component_cols),
                        ]
                    )
            if args.fgas_matcher_changed_acceptance_gate_checkpoint:
                fgas_track_cmd.extend(
                    [
                        "--fgas-matcher-changed-acceptance-gate-checkpoint",
                        str(args.fgas_matcher_changed_acceptance_gate_checkpoint),
                        "--fgas-matcher-changed-acceptance-gate-thresh",
                        str(args.fgas_matcher_changed_acceptance_gate_thresh),
                    ]
                )
            if args.fgas_block_matcher_skip_single_row_unchanged_external_competition:
                fgas_track_cmd.append("--fgas-block-matcher-skip-single-row-unchanged-external-competition")
            if args.fgas_block_matcher_skip_single_row_unchanged_external_competition_max_assignment_margin > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-single-row-unchanged-external-competition-max-assignment-margin",
                        str(args.fgas_block_matcher_skip_single_row_unchanged_external_competition_max_assignment_margin),
                    ]
                )
            if args.fgas_block_matcher_skip_single_row_unchanged_external_competition_owner_margin_thresh > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-single-row-unchanged-external-competition-owner-margin-thresh",
                        str(args.fgas_block_matcher_skip_single_row_unchanged_external_competition_owner_margin_thresh),
                    ]
                )
            if args.fgas_block_matcher_skip_multirow_owned_competition:
                fgas_track_cmd.append("--fgas-block-matcher-skip-multirow-owned-competition")
            if args.fgas_block_matcher_skip_multirow_owned_competition_owner_margin_thresh > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-multirow-owned-competition-owner-margin-thresh",
                        str(args.fgas_block_matcher_skip_multirow_owned_competition_owner_margin_thresh),
                    ]
                )
            if args.fgas_block_matcher_skip_temporal_inconsistent_owner_swap:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-temporal-inconsistent-owner-swap",
                        "--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-target-max-time-since-update",
                        str(args.fgas_block_matcher_skip_temporal_inconsistent_owner_swap_target_max_time_since_update),
                        "--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-target-min-hit-streak",
                        str(args.fgas_block_matcher_skip_temporal_inconsistent_owner_swap_target_min_hit_streak),
                        "--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-owner-min-time-since-update",
                        str(args.fgas_block_matcher_skip_temporal_inconsistent_owner_swap_owner_min_time_since_update),
                        "--fgas-block-matcher-skip-temporal-inconsistent-owner-swap-owner-max-hit-streak",
                        str(args.fgas_block_matcher_skip_temporal_inconsistent_owner_swap_owner_max_hit_streak),
                    ]
                )
            if args.fgas_block_matcher_protect_ownerless_rawmatched_active:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active",
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active-max-time-since-update",
                        str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_max_time_since_update),
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active-min-hit-streak",
                        str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_min_hit_streak),
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active-min-hits",
                        str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_min_hits),
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active-min-component-rows",
                        str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_min_component_rows),
                        "--fgas-block-matcher-protect-ownerless-rawmatched-active-min-component-cols",
                        str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_min_component_cols),
                    ]
                )
                if args.fgas_block_matcher_protect_ownerless_rawmatched_active_max_assignment_margin > 0.0:
                    fgas_track_cmd.extend(
                        [
                            "--fgas-block-matcher-protect-ownerless-rawmatched-active-max-assignment-margin",
                            str(args.fgas_block_matcher_protect_ownerless_rawmatched_active_max_assignment_margin),
                        ]
                    )
            if args.fgas_block_matcher_skip_ownerless_rawmatched_single_row_changed:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-ownerless-rawmatched-single-row-changed",
                        "--fgas-block-matcher-skip-ownerless-rawmatched-single-row-base-margin-thresh",
                        str(args.fgas_block_matcher_skip_ownerless_rawmatched_single_row_base_margin_thresh),
                    ]
                )
            if args.fgas_block_matcher_skip_ownerless_rawmatched_multirow_changed:
                fgas_track_cmd.extend(
                    [
                        "--fgas-block-matcher-skip-ownerless-rawmatched-multirow-changed",
                        "--fgas-block-matcher-skip-ownerless-rawmatched-multirow-assignment-margin-thresh",
                        str(args.fgas_block_matcher_skip_ownerless_rawmatched_multirow_assignment_margin_thresh),
                    ]
                )
        if args.fgas_lifecycle_reclaim_enable:
            fgas_track_cmd.extend(
                [
                    "--fgas-lifecycle-reclaim-enable",
                    "--fgas-lifecycle-reclaim-min-time-since-update",
                    str(args.fgas_lifecycle_reclaim_min_time_since_update),
                    "--fgas-lifecycle-reclaim-max-time-since-update",
                    str(args.fgas_lifecycle_reclaim_max_time_since_update),
                    "--fgas-lifecycle-reclaim-min-hits",
                    str(args.fgas_lifecycle_reclaim_min_hits),
                    "--fgas-lifecycle-reclaim-min-box-iou",
                    str(args.fgas_lifecycle_reclaim_min_box_iou),
                    "--fgas-lifecycle-reclaim-min-box-area-ratio",
                    str(args.fgas_lifecycle_reclaim_min_box_area_ratio),
                    "--fgas-lifecycle-reclaim-max-box-area-ratio",
                    str(args.fgas_lifecycle_reclaim_max_box_area_ratio),
                ]
            )
            if args.fgas_lifecycle_reclaim_min_emb_similarity > 0.0:
                fgas_track_cmd.extend(
                    [
                        "--fgas-lifecycle-reclaim-min-emb-similarity",
                        str(args.fgas_lifecycle_reclaim_min_emb_similarity),
                    ]
                )
        if args.pair_scorer_checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-pair-scorer-checkpoint",
                    str(args.pair_scorer_checkpoint),
                ]
            )
        if args.fgas_block_gate_checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-block-gate-checkpoint",
                    str(args.fgas_block_gate_checkpoint),
                    "--fgas-block-gate-thresh",
                    str(args.fgas_block_gate_thresh),
                ]
            )
        if args.fgas_soft_enable:
            fgas_track_cmd.extend(
                [
                    "--fgas-soft-enable",
                    "--fgas-soft-lambda",
                    str(args.fgas_soft_lambda),
                    "--fgas-soft-row-base-margin-thresh",
                    str(args.fgas_soft_row_base_margin_thresh),
                    "--fgas-soft-changed-row-flip-gap-thresh",
                    str(args.fgas_soft_changed_row_flip_gap_thresh),
                    "--fgas-soft-changed-row-refined-margin-thresh",
                    str(args.fgas_soft_changed_row_refined_margin_thresh),
                ]
            )
        if args.fgas_soft_allow_fallback:
            fgas_track_cmd.append("--fgas-soft-allow-fallback")
        if args.fgas_acceptance_gate_checkpoint:
            fgas_track_cmd.extend(
                [
                    "--fgas-acceptance-gate-checkpoint",
                    str(args.fgas_acceptance_gate_checkpoint),
                    "--fgas-acceptance-gate-thresh",
                    str(args.fgas_acceptance_gate_thresh),
                ]
            )
        if args.fgas_matcher_case_export_jsonl:
            matcher_case_export_path = Path(str(args.fgas_matcher_case_export_jsonl))
            if not matcher_case_export_path.is_absolute():
                matcher_case_export_path = (REPO_ROOT / matcher_case_export_path).resolve()
            else:
                matcher_case_export_path = matcher_case_export_path.resolve()
            fgas_track_cmd.extend(
                [
                    "--fgas-matcher-case-export-jsonl",
                    str(matcher_case_export_path),
                ]
            )
        if args.fgas_soft_only_changed_blocks:
            fgas_track_cmd.append("--fgas-soft-only-changed-blocks")
        if args.fgas_soft_only_changed_rows:
            fgas_track_cmd.append("--fgas-soft-only-changed-rows")
        if args.fgas_soft_only_changed_frontier:
            fgas_track_cmd.append("--fgas-soft-only-changed-frontier")
        if not args.disable_controller and (
            not args.fgas_block_matcher_force_only or args.allow_controller_with_force_only
        ):
            fgas_track_cmd.extend(
                [
                    "--fgas-controller-enable",
                    "--fgas-controller-edge-thresh",
                    str(args.fgas_controller_edge_thresh),
                    "--fgas-controller-row-defer-thresh",
                    str(args.fgas_controller_row_defer_thresh),
                    "--fgas-controller-col-newborn-thresh",
                    str(args.fgas_controller_col_newborn_thresh),
                    "--fgas-controller-margin-thresh",
                    str(args.fgas_controller_margin_thresh),
                    "--fgas-controller-ambiguity-margin",
                    str(args.fgas_controller_ambiguity_margin),
                ]
            )
            if args.fgas_controller_only_changed_blocks:
                fgas_track_cmd.append("--fgas-controller-only-changed-blocks")
        fgas_track_log = logs_dir / "fgas_track.log"
        return_code = run_step(fgas_track_cmd, fgas_track_log, cwd=DEEP_ROOT)
        ensure_success("fgas_track", return_code, rows, summary_csv, fgas_track_out, fgas_track_log, f"fgas tracking complete for {seq_label}")

        update_row(rows, "fgas_eval", status="running", started_at=now_iso())
        write_rows(summary_csv, QUEUE_FIELDS, rows)
        fgas_eval_cmd = [
            sys.executable,
            "external/TrackEval/scripts/run_mot_challenge.py",
            "--BENCHMARK",
            "MOT17",
            "--SPLIT_TO_EVAL",
            "val",
            "--GT_FOLDER",
            str(DEEP_ROOT / "results" / "gt"),
            "--TRACKERS_FOLDER",
            str(trackers_root),
            "--TRACKERS_TO_EVAL",
            fgas_exp + "_post",
            "--SEQ_INFO",
            *seq_names,
            "--METRICS",
            "HOTA",
            "CLEAR",
            "Identity",
            "--USE_PARALLEL",
            "False",
            "--PRINT_ONLY_COMBINED",
            "True",
        ]
        fgas_eval_log = logs_dir / "fgas_eval.log"
        return_code = run_step(fgas_eval_cmd, fgas_eval_log, cwd=DEEP_ROOT)
        ensure_success("fgas_eval", return_code, rows, summary_csv, fgas_eval_out, fgas_eval_log, f"fgas eval complete for {seq_label}")

        backfill_compare(
            rows=rows,
            summary_csv=summary_csv,
            logs_dir=logs_dir,
            run_root=run_root,
            seq_label=seq_label,
            raw_track_out=raw_track_out,
            raw_eval_out=raw_eval_out,
            fgas_track_out=fgas_track_out,
            fgas_eval_out=fgas_eval_out,
            metrics_compare_csv=metrics_compare_csv,
            metrics_delta_csv=metrics_delta_csv,
            per_sequence_csv=per_sequence_csv,
        )
        append_registry(summary_csv, run_root, "success", f"completed paired eval on {seq_label}", args.registry_csv)
    except Exception as exc:
        mark_running_rows_failed(rows, summary_csv, str(exc))
        append_registry(summary_csv, run_root, "failed", f"paired eval failed on {seq_label}: {exc}", args.registry_csv)
        raise


if __name__ == "__main__":
    main()
