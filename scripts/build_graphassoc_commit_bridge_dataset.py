#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build bridge-supervision local-conflict set-predictor data from graph-assoc candidate/event dumps."
    )
    ap.add_argument("--source-manifest", required=True, help="CSV manifest with rows_jsonl and event_jsonl entries.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--min-detections", type=int, default=2)
    ap.add_argument("--min-committed-matches", type=int, default=1)
    ap.add_argument("--max-detections", type=int, default=8)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--train-sequences", default="")
    ap.add_argument("--val-sequences", default="")
    ap.add_argument("--strict-sequence-split", action="store_true")
    ap.add_argument("--dataset-tag", default="graphassoc_bridge_commit")
    ap.add_argument("--feature-version", default="graphassoc_bridge_v1")
    return ap.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _parse_csv_tokens(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def _sequence_aliases(seq: str) -> set[str]:
    raw = str(seq or "").strip()
    if not raw:
        return set()
    name = Path(raw).name
    aliases = {raw, name}
    if name.count("-") >= 2:
        aliases.add(name.rsplit("-", 1)[0])
    return {token for token in aliases if token}


def _sequence_matches(seq: str, tokens: list[str]) -> bool:
    if not tokens:
        return False
    aliases = _sequence_aliases(seq)
    for token in tokens:
        token = str(token or "").strip()
        if not token:
            continue
        for alias in aliases:
            if alias == token or alias.startswith(token):
                return True
    return False


def _determine_split_tag(
    *,
    seq: str,
    explicit_split_tag: str,
    train_tokens: list[str],
    val_tokens: list[str],
    strict_sequence_split: bool,
) -> str:
    explicit = str(explicit_split_tag or "").strip().lower()
    if explicit in {"train", "val", "unused"}:
        return explicit
    if _sequence_matches(seq, val_tokens):
        return "val"
    if train_tokens:
        if _sequence_matches(seq, train_tokens):
            return "train"
        return "unused" if strict_sequence_split else "train"
    if val_tokens:
        return "train"
    return "all"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"Empty source manifest: {path}")
    sources: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        rows_jsonl_raw = str(row.get("rows_jsonl", "")).strip()
        events_jsonl_raw = str(row.get("events_jsonl", "")).strip()
        if not rows_jsonl_raw or not events_jsonl_raw:
            raise ValueError(f"Manifest row {idx} is missing rows_jsonl/events_jsonl")
        rows_jsonl = Path(rows_jsonl_raw)
        if not rows_jsonl.is_absolute():
            rows_jsonl = (path.parent / rows_jsonl).resolve()
        events_jsonl = Path(events_jsonl_raw)
        if not events_jsonl.is_absolute():
            events_jsonl = (path.parent / events_jsonl).resolve()
        sources.append(
            {
                "rows_jsonl": str(rows_jsonl),
                "events_jsonl": str(events_jsonl),
                "source_tag": str(row.get("source_tag", "")).strip() or f"source_{idx:03d}",
                "host_variant": str(row.get("host_variant", "")).strip() or "unknown",
                "split_tag": str(row.get("split_tag", "")).strip(),
                "dataset_tag": str(row.get("dataset_tag", "")).strip(),
                "feature_version": str(row.get("feature_version", "")).strip(),
            }
        )
    return sources


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _merge_rows(group_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for row in group_rows:
        det_index = _safe_int(row.get("det_index", -1), -1)
        track_rank = _safe_int(row.get("track_rank", -1), -1)
        if det_index < 0 or track_rank < 0:
            continue
        merged[(det_index, track_rank)] = dict(row)
    for row in event_rows:
        det_index = _safe_int(row.get("det_index", -1), -1)
        track_rank = _safe_int(row.get("track_rank", -1), -1)
        if det_index < 0 or track_rank < 0:
            continue
        merged.setdefault((det_index, track_rank), dict(row))
    rows = list(merged.values())
    rows.sort(key=lambda row: (_safe_int(row.get("det_index", -1), -1), _safe_int(row.get("track_rank", -1), -1)))
    return rows


def _read_by_group(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        group_id = str(row.get("group_id", "")).strip()
        if not group_id:
            continue
        groups[group_id].append(row)
    return groups


def _group_key_from_row(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("seq", "")).strip(),
        _safe_int(row.get("frame", 0), 0),
        _safe_int(row.get("det_index", -1), -1),
    )


def _build_group_record(
    *,
    seq: str,
    frame: int,
    det_index: int,
    rows: list[dict[str, Any]],
    source_tag: str,
    host_variant: str,
    split_tag: str,
    feature_version: str,
) -> dict[str, Any] | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: _safe_int(row.get("track_rank", 0), 0))
    first = rows[0]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidates.append(
            {
                "track_rank": _safe_int(row.get("track_rank", 0), 0),
                "track_id": _safe_int(row.get("track_id", -1), -1),
                "label": _safe_int(row.get("label", 0), 0),
                "valid_train_row": _safe_int(row.get("valid_train_row", 1), 1),
                "base_score": _safe_float(row.get("base_score", 0.0), 0.0),
                "refined_score": _safe_float(row.get("refined_score", 0.0), 0.0),
                "motion_score": _safe_float(row.get("motion_score", 0.0), 0.0),
                "track_gap": _safe_int(row.get("track_gap", 0), 0),
                "track_hist_len": _safe_int(row.get("track_hist_len", 0), 0),
                "det_score": _safe_float(row.get("det_score", 0.0), 0.0),
                "det_cx": _safe_float(row.get("det_cx", 0.0), 0.0),
                "det_cy": _safe_float(row.get("det_cy", 0.0), 0.0),
                "det_w": _safe_float(row.get("det_w", 0.0), 0.0),
                "det_h": _safe_float(row.get("det_h", 0.0), 0.0),
                "track_cx": _safe_float(row.get("track_cx", 0.0), 0.0),
                "track_cy": _safe_float(row.get("track_cy", 0.0), 0.0),
                "track_w": _safe_float(row.get("track_w", 0.0), 0.0),
                "track_h": _safe_float(row.get("track_h", 0.0), 0.0),
            }
        )

    positive_rows = [idx for idx, row in enumerate(rows) if _safe_int(row.get("label", 0), 0) == 1 and _safe_int(row.get("valid_train_row", 1), 1) == 1]
    if not positive_rows:
        return None

    rank_vals = np.asarray([_safe_float(row.get("refined_score", 0.0), 0.0) for row in rows], dtype=np.float32)
    rank_top1_idx = int(np.argmax(rank_vals)) if rank_vals.size > 0 else -1
    positive_rank = min((_safe_int(rows[idx].get("track_rank", 0), 0) for idx in positive_rows), default=-1)
    positive_in_topk = 1 if positive_rank > 0 and positive_rank <= len(rows) else 0
    rank_top1_correct = 1 if rank_top1_idx in positive_rows else 0
    rank_margin = 0.0
    if rank_vals.size > 1:
        top2 = np.sort(rank_vals)[::-1][:2]
        rank_margin = float(top2[0] - top2[1])
    group_is_background = 0
    group_is_ambiguous = 1 if (rank_top1_correct == 0 or rank_margin < 0.10) else 0
    group_is_recoverable = 1 if (rank_top1_correct == 0 and positive_in_topk == 1) else 0
    det_gt_id = _safe_int(first.get("det_gt_id", -1), -1)
    det_ignore = _safe_int(first.get("det_ignore", 0), 0)
    group_has_positive = 1

    return {
        "group_id": str(first.get("group_id", f"{seq}:{frame}:{det_index}")),
        "seq": seq,
        "frame": int(frame),
        "det_index": int(det_index),
        "assoc_mode": str(first.get("assoc_mode", "")),
        "group_size": int(len(rows)),
        "candidate_count_total": int(len(rows)),
        "det_gt_id": int(det_gt_id),
        "det_ignore": int(det_ignore),
        "group_has_positive": int(group_has_positive),
        "group_is_background": int(group_is_background),
        "group_is_ambiguous": int(group_is_ambiguous),
        "group_is_recoverable": int(group_is_recoverable),
        "positive_rank": int(positive_rank),
        "positive_in_topk": int(positive_in_topk),
        "rank_score_col": "refined_score",
        "rank_margin": float(rank_margin),
        "rank_entropy": float(-np.sum(np.clip((rank_vals / max(float(rank_vals.sum()), 1e-8)) if rank_vals.size > 0 else np.asarray([]), 1e-8, 1.0) * np.log(np.clip((rank_vals / max(float(rank_vals.sum()), 1e-8)) if rank_vals.size > 0 else np.asarray([]), 1e-8, 1.0)))) if rank_vals.size > 0 else 0.0,
        "rank_top1_correct": int(rank_top1_correct),
        "base_margin": float(_safe_float(first.get("base_margin", 0.0), 0.0)),
        "refined_margin": float(_safe_float(first.get("refined_margin", 0.0), 0.0)),
        "candidates": candidates,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _load_manifest(Path(args.source_manifest).resolve())
    train_tokens = _parse_csv_tokens(args.train_sequences)
    val_tokens = _parse_csv_tokens(args.val_sequences)

    rows_out: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    summary = Counter()
    seq_counter: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    split_counter: dict[str, Counter[str]] = defaultdict(Counter)
    seen_sequences: set[str] = set()
    seen_sources: set[str] = set()

    for source in sources:
        rows_jsonl = Path(source["rows_jsonl"]).resolve()
        events_jsonl = Path(source["events_jsonl"]).resolve()
        if not rows_jsonl.is_file():
            raise FileNotFoundError(f"Missing rows jsonl: {rows_jsonl}")
        if not events_jsonl.is_file():
            raise FileNotFoundError(f"Missing events jsonl: {events_jsonl}")

        rows_by_group = _read_by_group(rows_jsonl)
        events_by_group = _read_by_group(events_jsonl)
        source_tag = str(source["source_tag"])
        host_variant = str(source["host_variant"])
        feature_version = str(source["feature_version"] or args.feature_version)
        seen_sources.add(source_tag)
        summary["sources"] += 1

        all_group_ids = sorted(set(rows_by_group.keys()) | set(events_by_group.keys()))
        for group_id in all_group_ids:
            rows = rows_by_group.get(group_id, [])
            events = events_by_group.get(group_id, [])
            if not rows and not events:
                continue
            merged = _merge_rows(rows, events)
            if not merged:
                continue
            seq = str(merged[0].get("seq", "")).strip()
            frame = _safe_int(merged[0].get("frame_id", merged[0].get("frame", 0)), 0)
            det_index = _safe_int(merged[0].get("det_index", -1), -1)
            split_tag = _determine_split_tag(
                seq=seq,
                explicit_split_tag=str(source.get("split_tag", "")),
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                strict_sequence_split=bool(args.strict_sequence_split),
            )
            if split_tag == "unused":
                summary["unused_groups"] += 1
                continue
            record = _build_group_record(
                seq=seq,
                frame=frame,
                det_index=det_index,
                rows=merged,
                source_tag=source_tag,
                host_variant=host_variant,
                split_tag=split_tag,
                feature_version=feature_version,
            )
            if record is None:
                summary["skipped_no_positive"] += 1
                continue
            group_records.append(record)
            seen_sequences.add(seq)
            summary["groups"] += 1
            summary["rows"] += int(len(record["candidates"]))
            summary["positive_rows"] += int(sum(int(c["label"]) for c in record["candidates"]))
            summary["positive_groups"] += 1
            summary["background_groups"] += int(record["group_is_background"])
            summary["ambiguous_groups"] += int(record["group_is_ambiguous"])
            summary["recoverable_groups"] += int(record["group_is_recoverable"])
            summary["positive_in_topk_groups"] += int(record["positive_in_topk"])
            summary["rank_top1_correct_groups"] += int(record["rank_top1_correct"])
            split_counter[split_tag]["groups"] += 1
            split_counter[split_tag]["rows"] += int(len(record["candidates"]))
            split_counter[split_tag]["positive_groups"] += 1
            split_counter[split_tag]["background_groups"] += int(record["group_is_background"])
            split_counter[split_tag]["ambiguous_groups"] += int(record["group_is_ambiguous"])
            split_counter[split_tag]["recoverable_groups"] += int(record["group_is_recoverable"])
            split_counter[split_tag]["positive_in_topk_groups"] += int(record["positive_in_topk"])
            split_counter[split_tag]["rank_top1_correct_groups"] += int(record["rank_top1_correct"])
            seq_counter[(seq, split_tag)]["groups"] += 1
            seq_counter[(seq, split_tag)]["rows"] += int(len(record["candidates"]))
            seq_counter[(seq, split_tag)]["positive_groups"] += 1
            seq_counter[(seq, split_tag)]["background_groups"] += int(record["group_is_background"])
            seq_counter[(seq, split_tag)]["ambiguous_groups"] += int(record["group_is_ambiguous"])
            seq_counter[(seq, split_tag)]["recoverable_groups"] += int(record["group_is_recoverable"])
            seq_counter[(seq, split_tag)]["positive_in_topk_groups"] += int(record["positive_in_topk"])
            seq_counter[(seq, split_tag)]["rank_top1_correct_groups"] += int(record["rank_top1_correct"])

            for cand in record["candidates"]:
                row_out = {
                    "seq": seq,
                    "frame": int(frame),
                    "assoc_mode": str(record["assoc_mode"]),
                    "group_id": str(record["group_id"]),
                    "det_index": int(det_index),
                    "track_rank": int(cand["track_rank"]),
                    "track_id": int(cand["track_id"]),
                    "is_selected": 1 if int(cand["label"]) == 1 else 0,
                    "det_score": float(cand["det_score"]),
                    "base_score": float(cand["base_score"]),
                    "refined_score": float(cand["refined_score"]),
                    "motion_score": float(cand["motion_score"]),
                    "track_gap": int(cand["track_gap"]),
                    "track_hist_len": int(cand["track_hist_len"]),
                    "det_gt_id": int(record["det_gt_id"]),
                    "track_gt_id": -1,
                    "det_ignore": int(record["det_ignore"]),
                    "track_ignore": 0,
                    "label": int(cand["label"]),
                    "valid_train_row": int(cand["valid_train_row"]),
                    "group_has_positive": int(record["group_has_positive"]),
                    "group_size": int(record["group_size"]),
                    "candidate_count_total": int(record["candidate_count_total"]),
                    "base_margin": float(record["base_margin"]),
                    "refined_margin": float(record["refined_margin"]),
                    "rank_score_col": str(record["rank_score_col"]),
                    "rank_margin": float(record["rank_margin"]),
                    "rank_entropy": float(record["rank_entropy"]),
                    "rank_top1_correct": int(record["rank_top1_correct"]),
                    "positive_in_topk": int(record["positive_in_topk"]),
                    "positive_rank": int(record["positive_rank"]),
                    "group_is_ambiguous": int(record["group_is_ambiguous"]),
                    "group_is_background": int(record["group_is_background"]),
                    "group_is_recoverable": int(record["group_is_recoverable"]),
                    "det_cx": float(cand["det_cx"]),
                    "det_cy": float(cand["det_cy"]),
                    "det_w": float(cand["det_w"]),
                    "det_h": float(cand["det_h"]),
                    "track_cx": float(cand["track_cx"]),
                    "track_cy": float(cand["track_cy"]),
                    "track_w": float(cand["track_w"]),
                    "track_h": float(cand["track_h"]),
                    "source_csv": str(rows_jsonl),
                    "source_event_jsonl": str(events_jsonl),
                }
                rows_out.append(row_out)

    rows_csv = out_dir / "bridge_rows.csv"
    group_jsonl = out_dir / "bridge_groups.jsonl"
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    fieldnames = list(rows_out[0].keys()) if rows_out else []
    if rows_out:
        with rows_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
    with group_jsonl.open("w", encoding="utf-8") as f:
        for record in group_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_row = {
        "dataset_tag": str(args.dataset_tag),
        "feature_version": str(args.feature_version),
        "source_manifest": str(Path(args.source_manifest).resolve()),
        "rows_csv": str(rows_csv),
        "group_jsonl": str(group_jsonl),
        "sources": int(summary.get("sources", 0)),
        "groups": int(summary.get("groups", 0)),
        "rows": int(summary.get("rows", 0)),
        "positive_rows": int(summary.get("positive_rows", 0)),
        "positive_groups": int(summary.get("positive_groups", 0)),
        "background_groups": int(summary.get("background_groups", 0)),
        "ambiguous_groups": int(summary.get("ambiguous_groups", 0)),
        "recoverable_groups": int(summary.get("recoverable_groups", 0)),
        "positive_in_topk_groups": int(summary.get("positive_in_topk_groups", 0)),
        "rank_top1_correct_groups": int(summary.get("rank_top1_correct_groups", 0)),
        "skipped_no_positive": int(summary.get("skipped_no_positive", 0)),
        "unused_groups": int(summary.get("unused_groups", 0)),
        "train_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "train")),
        "val_sequences": ",".join(sorted(seq for (seq, split_tag) in seq_counter.keys() if split_tag == "val")),
        "all_sequences": ",".join(sorted(seen_sequences)),
        "all_sources": ",".join(sorted(seen_sources)),
        "status": "ok",
    }

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    summary_json.write_text(json.dumps(summary_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_row, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
