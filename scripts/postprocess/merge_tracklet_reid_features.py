#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    args = ap.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {k: [] for k in ["start", "end", "global_mean", "high_score"]}
    seq_values = []
    track_ids = []
    index_rows = []
    summaries = []

    for seq in args.seqs:
        d = input_root / seq
        npz_path = d / "tracklet_reid_features.npz"
        idx_path = d / "tracklet_reid_index.csv"
        summary_path = d / "feature_extract_summary.json"
        if not npz_path.exists() or not idx_path.exists():
            raise FileNotFoundError(f"missing features for {seq}: {d}")
        data = np.load(npz_path)
        n = len(data["track_id"])
        for key in arrays:
            arrays[key].append(data[key].astype(np.float32))
        seq_values.extend([seq] * n)
        track_ids.extend([int(x) for x in data["track_id"]])
        with idx_path.open("r", encoding="utf-8") as f:
            index_rows.extend(list(csv.DictReader(f)))
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    merged = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}
    np.savez_compressed(
        out_dir / "tracklet_reid_features.npz",
        **merged,
        seq=np.asarray(seq_values),
        track_id=np.asarray(track_ids, dtype=np.int64),
    )
    with (out_dir / "tracklet_reid_index.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(index_rows[0].keys()) if index_rows else ["seq", "track_id"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(index_rows)

    summary = {
        "seqs": args.seqs,
        "tracks": len(index_rows),
        "feature_dim": int(merged["start"].shape[1]) if len(index_rows) else 0,
        "missing_tracks": sum(int(s.get("missing_tracks", 0)) for s in summaries),
        "features_extracted": sum(int(s.get("features_extracted", 0)) for s in summaries),
        "total_unique_samples_planned": sum(int(s.get("total_unique_samples_planned", 0)) for s in summaries),
        "output_npz": str(out_dir / "tracklet_reid_features.npz"),
        "output_index": str(out_dir / "tracklet_reid_index.csv"),
        "norms": {},
    }
    for key, arr in merged.items():
        norms = np.linalg.norm(arr, axis=1)
        summary["norms"][key] = {
            "mean_norm": float(norms.mean()) if len(norms) else 0.0,
            "min_norm": float(norms.min()) if len(norms) else 0.0,
            "max_norm": float(norms.max()) if len(norms) else 0.0,
        }
    (out_dir / "feature_merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Merged Tracklet ReID Features", "", "| metric | value |", "|---|---:|"]
    for key, value in summary.items():
        if key != "norms":
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Norms", "| feature | mean_norm | min_norm | max_norm |", "|---|---:|---:|---:|"]
    for key, stats in summary["norms"].items():
        lines.append(f"| {key} | {stats['mean_norm']:.6f} | {stats['min_norm']:.6f} | {stats['max_norm']:.6f} |")
    (out_dir / "feature_merge_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
