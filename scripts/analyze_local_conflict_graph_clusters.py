#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.local_conflict_graph_common import build_group_components_from_group_rows


FRAME_CLUSTER_FIELDNAMES = [
    "cluster_id",
    "seq",
    "frame",
    "num_detections",
    "num_tracks",
    "num_edges",
    "groups",
    "positive_groups",
    "recoverable_groups",
    "bridge_groups",
    "background_groups",
    "ambiguous_groups",
    "action_keep",
    "action_rerank",
    "action_null",
    "multi_detection",
    "multi_track",
    "group_ids_json",
    "track_ids_json",
]

OVERLAP_CLUSTER_FIELDNAMES = [
    "cluster_id",
    "flag",
    "seq",
    "frame",
    "num_detections",
    "num_tracks",
    "multi_detection",
    "group_ids_json",
    "track_ids_json",
]

SEQ_CLUSTER_FIELDNAMES = [
    "seq",
    "clusters",
    "groups",
    "recoverable_groups",
    "bridge_groups",
    "multi_detection_clusters",
    "multi_track_clusters",
    "groups_per_cluster",
    "recoverable_rate_per_group",
    "bridge_rate_per_group",
    "recoverable_groups_per_cluster",
    "bridge_groups_per_cluster",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze frame-local conflict graph clusters from competition cases.")
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--group-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _load_case_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row["group_id"]): dict(row) for row in reader}


def _load_groups(path: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
    return groups


def _component_clusters(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame_buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        frame_buckets[(str(group["seq"]), _safe_int(group["frame"]))].append(group)

    clusters: list[dict[str, Any]] = []
    for (seq, frame), frame_groups in sorted(frame_buckets.items()):
        group_by_det = {_safe_int(group.get("det_index", -1), -1): group for group in frame_groups}
        edge_count = 0
        topk = max((len(list(group.get("candidates", []))) for group in frame_groups), default=1)
        for group in frame_groups:
            for cand in group.get("candidates", []):
                if _safe_int(cand.get("track_id", -1), -1) >= 0:
                    edge_count += 1
        cluster_idx = 0
        for component in build_group_components_from_group_rows(frame_groups, topk=topk):
            det_rows = [int(x) for x in component.get("det_rows", [])]
            det_group_ids = sorted(
                str(group_by_det[det_row]["group_id"])
                for det_row in det_rows
                if det_row in group_by_det
            )
            trk_ids = [int(x) for x in component.get("track_ids", [])]
            local_edges = 0
            trk_set = set(trk_ids)
            for det_row in det_rows:
                group = group_by_det.get(det_row)
                if group is None:
                    continue
                for cand in group.get("candidates", []):
                    if _safe_int(cand.get("track_id", -1), -1) in trk_set:
                        local_edges += 1
            clusters.append(
                {
                    "cluster_id": f"{seq}:{frame}:{cluster_idx}",
                    "seq": seq,
                    "frame": frame,
                    "group_ids": det_group_ids,
                    "track_ids": trk_ids,
                    "num_detections": len(det_group_ids),
                    "num_tracks": len(trk_ids),
                    "num_edges": int(local_edges),
                    "frame_total_groups": len(frame_groups),
                    "frame_total_edges": edge_count,
                }
            )
            cluster_idx += 1
    return clusters


def _subset_overlap_clusters(
    *,
    groups: list[dict[str, Any]],
    case_rows: dict[str, dict[str, str]],
    flag_name: str,
    predicate,
) -> list[dict[str, Any]]:
    frame_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    tracks_by_gid: dict[str, set[int]] = {}
    for group in groups:
        gid = str(group["group_id"])
        frame_groups[(str(group["seq"]), _safe_int(group["frame"]))].append(gid)
        tracks_by_gid[gid] = {int(c["track_id"]) for c in group.get("candidates", []) if _safe_int(c.get("track_id", -1), -1) >= 0}

    out: list[dict[str, Any]] = []
    cluster_idx = 0
    for (seq, frame), gids in sorted(frame_groups.items()):
        subset = [gid for gid in gids if gid in case_rows and predicate(case_rows[gid])]
        if not subset:
            continue
        adj = {gid: set() for gid in subset}
        for i, a in enumerate(subset):
            ta = tracks_by_gid.get(a, set())
            for b in subset[i + 1 :]:
                if ta & tracks_by_gid.get(b, set()):
                    adj[a].add(b)
                    adj[b].add(a)
        visited: set[str] = set()
        for gid in subset:
            if gid in visited:
                continue
            stack = [gid]
            visited.add(gid)
            comp: list[str] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in adj[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            track_ids = sorted(set().union(*(tracks_by_gid.get(x, set()) for x in comp)))
            out.append(
                {
                    "cluster_id": f"{flag_name}:{seq}:{frame}:{cluster_idx}",
                    "flag": flag_name,
                    "seq": seq,
                    "frame": frame,
                    "num_detections": len(comp),
                    "num_tracks": len(track_ids),
                    "multi_detection": int(len(comp) >= 2),
                    "group_ids_json": json.dumps(sorted(comp)),
                    "track_ids_json": json.dumps(track_ids),
                }
            )
            cluster_idx += 1
    return out


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_rows = _load_case_rows(Path(args.cases_csv).resolve())
    groups = _load_groups(Path(args.group_jsonl).resolve())
    clusters = _component_clusters(groups)
    recoverable_overlap_clusters = _subset_overlap_clusters(
        groups=groups,
        case_rows=case_rows,
        flag_name="recoverable",
        predicate=lambda row: _safe_int(row.get("group_is_recoverable", 0)) > 0,
    )
    bridge_overlap_clusters = _subset_overlap_clusters(
        groups=groups,
        case_rows=case_rows,
        flag_name="bridge",
        predicate=lambda row: _safe_int(row.get("continuity_bridge", 0)) > 0,
    )

    cluster_rows: list[dict[str, Any]] = []
    seq_counter: dict[str, Counter[str]] = defaultdict(Counter)
    recoverable_cluster_rows: list[dict[str, Any]] = []
    bridge_cluster_rows: list[dict[str, Any]] = []
    recoverable_groups_in_multi_det = 0
    bridge_groups_in_multi_det = 0
    total_recoverable_groups = 0
    total_bridge_groups = 0

    for cluster in clusters:
        group_case_rows = [case_rows[g] for g in cluster["group_ids"] if g in case_rows]
        counts = Counter()
        for row in group_case_rows:
            counts["groups"] += 1
            counts["positive_groups"] += int(_safe_int(row.get("group_has_positive", 0)) > 0)
            counts["recoverable_groups"] += int(_safe_int(row.get("group_is_recoverable", 0)) > 0)
            counts["bridge_groups"] += int(_safe_int(row.get("continuity_bridge", 0)) > 0)
            counts["background_groups"] += int(_safe_int(row.get("group_is_background", 0)) > 0)
            counts["ambiguous_groups"] += int(_safe_int(row.get("group_is_ambiguous", 0)) > 0)
            counts["action_keep"] += int(_safe_int(row.get("action_target_id", 0)) == 0)
            counts["action_rerank"] += int(_safe_int(row.get("action_target_id", 0)) == 1)
            counts["action_null"] += int(_safe_int(row.get("action_target_id", 0)) == 2)

        multi_detection = int(cluster["num_detections"] >= 2)
        multi_track = int(cluster["num_tracks"] >= 2)
        row = {
            "cluster_id": cluster["cluster_id"],
            "seq": cluster["seq"],
            "frame": cluster["frame"],
            "num_detections": cluster["num_detections"],
            "num_tracks": cluster["num_tracks"],
            "num_edges": cluster["num_edges"],
            "groups": counts["groups"],
            "positive_groups": counts["positive_groups"],
            "recoverable_groups": counts["recoverable_groups"],
            "bridge_groups": counts["bridge_groups"],
            "background_groups": counts["background_groups"],
            "ambiguous_groups": counts["ambiguous_groups"],
            "action_keep": counts["action_keep"],
            "action_rerank": counts["action_rerank"],
            "action_null": counts["action_null"],
            "multi_detection": multi_detection,
            "multi_track": multi_track,
            "group_ids_json": json.dumps(cluster["group_ids"]),
            "track_ids_json": json.dumps(cluster["track_ids"]),
        }
        cluster_rows.append(row)
        seq_counter[cluster["seq"]].update(
            {
                "clusters": 1,
                "groups": counts["groups"],
                "recoverable_groups": counts["recoverable_groups"],
                "bridge_groups": counts["bridge_groups"],
                "multi_detection_clusters": multi_detection,
                "multi_track_clusters": multi_track,
            }
        )

        total_recoverable_groups += counts["recoverable_groups"]
        total_bridge_groups += counts["bridge_groups"]
        if multi_detection:
            recoverable_groups_in_multi_det += counts["recoverable_groups"]
            bridge_groups_in_multi_det += counts["bridge_groups"]
        if counts["recoverable_groups"] > 0:
            recoverable_cluster_rows.append(row)
        if counts["bridge_groups"] > 0:
            bridge_cluster_rows.append(row)

    cluster_csv = out_dir / "cluster_summary.csv"
    _write_csv_rows(cluster_csv, FRAME_CLUSTER_FIELDNAMES, cluster_rows)

    cluster_jsonl = out_dir / "cluster_members.jsonl"
    with cluster_jsonl.open("w", encoding="utf-8") as f:
        for row in cluster_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for name, rows in (
        ("recoverable_overlap_clusters.csv", recoverable_overlap_clusters),
        ("bridge_overlap_clusters.csv", bridge_overlap_clusters),
    ):
        _write_csv_rows(out_dir / name, OVERLAP_CLUSTER_FIELDNAMES, rows)

    seq_rows: list[dict[str, Any]] = []
    for seq, counter in sorted(seq_counter.items()):
        clusters_n = max(int(counter["clusters"]), 1)
        groups_n = max(int(counter["groups"]), 1)
        recoverable_n = max(int(counter["recoverable_groups"]), 1)
        bridge_n = max(int(counter["bridge_groups"]), 1)
        seq_rows.append(
            {
                "seq": seq,
                "clusters": int(counter["clusters"]),
                "groups": int(counter["groups"]),
                "recoverable_groups": int(counter["recoverable_groups"]),
                "bridge_groups": int(counter["bridge_groups"]),
                "multi_detection_clusters": int(counter["multi_detection_clusters"]),
                "multi_track_clusters": int(counter["multi_track_clusters"]),
                "groups_per_cluster": float(counter["groups"]) / float(clusters_n),
                "recoverable_rate_per_group": float(counter["recoverable_groups"]) / float(groups_n),
                "bridge_rate_per_group": float(counter["bridge_groups"]) / float(groups_n),
                "recoverable_groups_per_cluster": float(counter["recoverable_groups"]) / float(clusters_n),
                "bridge_groups_per_cluster": float(counter["bridge_groups"]) / float(clusters_n),
            }
        )
    seq_csv = out_dir / "sequence_cluster_summary.csv"
    _write_csv_rows(seq_csv, SEQ_CLUSTER_FIELDNAMES, seq_rows)

    recoverable_cluster_count = len(recoverable_cluster_rows)
    bridge_cluster_count = len(bridge_cluster_rows)
    recoverable_avg_dets = (
        sum(float(r["num_detections"]) for r in recoverable_cluster_rows) / float(max(recoverable_cluster_count, 1))
    )
    recoverable_avg_tracks = (
        sum(float(r["num_tracks"]) for r in recoverable_cluster_rows) / float(max(recoverable_cluster_count, 1))
    )
    bridge_avg_dets = (
        sum(float(r["num_detections"]) for r in bridge_cluster_rows) / float(max(bridge_cluster_count, 1))
    )
    bridge_avg_tracks = (
        sum(float(r["num_tracks"]) for r in bridge_cluster_rows) / float(max(bridge_cluster_count, 1))
    )
    recoverable_multi_det_clusters = sum(int(r["multi_detection"]) for r in recoverable_cluster_rows)
    bridge_multi_det_clusters = sum(int(r["multi_detection"]) for r in bridge_cluster_rows)
    recoverable_overlap_cluster_count = len(recoverable_overlap_clusters)
    bridge_overlap_cluster_count = len(bridge_overlap_clusters)
    recoverable_overlap_avg_dets = (
        sum(float(r["num_detections"]) for r in recoverable_overlap_clusters) / float(max(recoverable_overlap_cluster_count, 1))
    )
    recoverable_overlap_avg_tracks = (
        sum(float(r["num_tracks"]) for r in recoverable_overlap_clusters) / float(max(recoverable_overlap_cluster_count, 1))
    )
    bridge_overlap_avg_dets = (
        sum(float(r["num_detections"]) for r in bridge_overlap_clusters) / float(max(bridge_overlap_cluster_count, 1))
    )
    bridge_overlap_avg_tracks = (
        sum(float(r["num_tracks"]) for r in bridge_overlap_clusters) / float(max(bridge_overlap_cluster_count, 1))
    )
    recoverable_overlap_groups_in_multi = sum(
        json.loads(r["group_ids_json"]).__len__() for r in recoverable_overlap_clusters if int(r["multi_detection"]) > 0
    )
    bridge_overlap_groups_in_multi = sum(
        json.loads(r["group_ids_json"]).__len__() for r in bridge_overlap_clusters if int(r["multi_detection"]) > 0
    )
    recoverable_overlap_multi_det_share = float(
        sum(int(r["multi_detection"]) for r in recoverable_overlap_clusters)
    ) / float(max(recoverable_overlap_cluster_count, 1))
    bridge_overlap_multi_det_share = float(
        sum(int(r["multi_detection"]) for r in bridge_overlap_clusters)
    ) / float(max(bridge_overlap_cluster_count, 1))

    summary = {
        "cases_csv": str(Path(args.cases_csv).resolve()),
        "group_jsonl": str(Path(args.group_jsonl).resolve()),
        "clusters": len(cluster_rows),
        "frame_bipartite_clusters": len(cluster_rows),
        "recoverable_clusters": recoverable_cluster_count,
        "bridge_clusters": bridge_cluster_count,
        "multi_detection_clusters": sum(int(r["multi_detection"]) for r in cluster_rows),
        "multi_track_clusters": sum(int(r["multi_track"]) for r in cluster_rows),
        "recoverable_groups_total": total_recoverable_groups,
        "recoverable_groups_in_multi_detection_clusters": recoverable_groups_in_multi_det,
        "recoverable_groups_multi_detection_share": float(recoverable_groups_in_multi_det) / float(max(total_recoverable_groups, 1)),
        "recoverable_cluster_avg_detections": recoverable_avg_dets,
        "recoverable_cluster_avg_tracks": recoverable_avg_tracks,
        "recoverable_multi_detection_cluster_share": float(recoverable_multi_det_clusters) / float(max(recoverable_cluster_count, 1)),
        "bridge_groups_total": total_bridge_groups,
        "bridge_groups_in_multi_detection_clusters": bridge_groups_in_multi_det,
        "bridge_groups_multi_detection_share": float(bridge_groups_in_multi_det) / float(max(total_bridge_groups, 1)),
        "bridge_cluster_avg_detections": bridge_avg_dets,
        "bridge_cluster_avg_tracks": bridge_avg_tracks,
        "bridge_multi_detection_cluster_share": float(bridge_multi_det_clusters) / float(max(bridge_cluster_count, 1)),
        "recoverable_overlap_clusters": recoverable_overlap_cluster_count,
        "recoverable_overlap_groups_in_multi_detection_clusters": recoverable_overlap_groups_in_multi,
        "recoverable_overlap_groups_multi_detection_share": float(recoverable_overlap_groups_in_multi) / float(max(total_recoverable_groups, 1)),
        "recoverable_overlap_cluster_avg_detections": recoverable_overlap_avg_dets,
        "recoverable_overlap_cluster_avg_tracks": recoverable_overlap_avg_tracks,
        "recoverable_overlap_multi_detection_cluster_share": recoverable_overlap_multi_det_share,
        "bridge_overlap_clusters": bridge_overlap_cluster_count,
        "bridge_overlap_groups_in_multi_detection_clusters": bridge_overlap_groups_in_multi,
        "bridge_overlap_groups_multi_detection_share": float(bridge_overlap_groups_in_multi) / float(max(total_bridge_groups, 1)),
        "bridge_overlap_cluster_avg_detections": bridge_overlap_avg_dets,
        "bridge_overlap_cluster_avg_tracks": bridge_overlap_avg_tracks,
        "bridge_overlap_multi_detection_cluster_share": bridge_overlap_multi_det_share,
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
