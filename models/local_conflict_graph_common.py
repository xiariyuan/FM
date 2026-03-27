from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_linear_sum_assignment = None


def _get_linear_sum_assignment():
    global _linear_sum_assignment
    if _linear_sum_assignment is None:
        try:
            from scipy.optimize import linear_sum_assignment

            _linear_sum_assignment = linear_sum_assignment
        except Exception:
            _linear_sum_assignment = None
    return _linear_sum_assignment


def build_topk_bipartite_components(
    score_mat: torch.Tensor,
    *,
    topk: int,
    min_edge_score: float = 0.0,
    det_rows: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    if score_mat.ndim != 2 or score_mat.numel() == 0:
        return []

    num_dets, num_tracks = score_mat.shape
    if num_dets == 0 or num_tracks == 0:
        return []

    candidate_det_rows = list(range(num_dets)) if det_rows is None else [int(x) for x in det_rows]
    if not candidate_det_rows:
        return []

    k = int(max(min(int(topk), num_tracks), 1))
    top_vals, top_idx = torch.topk(score_mat.index_select(0, torch.as_tensor(candidate_det_rows, device=score_mat.device)), k=k, dim=1, sorted=True)

    det_to_tracks: Dict[int, List[int]] = {}
    track_to_dets: Dict[int, set[int]] = defaultdict(set)
    for local_row, det_row in enumerate(candidate_det_rows):
        cols: List[int] = []
        for score_val, col in zip(top_vals[local_row].tolist(), top_idx[local_row].tolist()):
            if float(score_val) <= float(min_edge_score):
                continue
            cols.append(int(col))
        cols = sorted(set(cols))
        if not cols:
            continue
        det_to_tracks[int(det_row)] = cols
        for col in cols:
            track_to_dets[int(col)].add(int(det_row))

    if not det_to_tracks:
        return []

    components: List[Dict[str, Any]] = []
    visited_dets: set[int] = set()
    visited_tracks: set[int] = set()
    for start_det in sorted(det_to_tracks.keys()):
        if start_det in visited_dets:
            continue
        det_stack = [int(start_det)]
        cluster_dets: set[int] = set()
        cluster_tracks: set[int] = set()
        while det_stack:
            det_row = int(det_stack.pop())
            if det_row in visited_dets:
                continue
            visited_dets.add(det_row)
            cluster_dets.add(det_row)
            for col in det_to_tracks.get(det_row, []):
                if col not in cluster_tracks:
                    cluster_tracks.add(col)
                if col in visited_tracks:
                    continue
                visited_tracks.add(col)
                for nxt_det in track_to_dets.get(col, set()):
                    if nxt_det not in visited_dets:
                        det_stack.append(int(nxt_det))
        components.append(
            {
                "det_rows": sorted(cluster_dets),
                "track_cols": sorted(cluster_tracks),
                "num_detections": int(len(cluster_dets)),
                "num_tracks": int(len(cluster_tracks)),
            }
        )
    return components


def solve_assignment_with_private_null(
    score_sub: torch.Tensor,
    *,
    feasible_mask: Optional[torch.Tensor] = None,
    null_scores: Optional[torch.Tensor] = None,
    use_hungarian: bool = True,
) -> List[Dict[str, Optional[int]]]:
    if score_sub.ndim != 2:
        raise ValueError(f"score_sub must be [N, M], got {tuple(score_sub.shape)}")
    num_dets, num_tracks = score_sub.shape
    if num_dets == 0:
        return []

    if feasible_mask is None:
        feasible_mask = torch.ones_like(score_sub, dtype=torch.bool)
    else:
        feasible_mask = feasible_mask.to(dtype=torch.bool, device=score_sub.device)
    if null_scores is None:
        null_scores = torch.zeros((num_dets,), device=score_sub.device, dtype=score_sub.dtype)
    else:
        null_scores = null_scores.to(device=score_sub.device, dtype=score_sub.dtype).view(num_dets)

    assign_scores = torch.full(
        (num_dets, num_tracks + num_dets),
        -1e6,
        device=score_sub.device,
        dtype=score_sub.dtype,
    )
    if num_tracks > 0:
        assign_scores[:, :num_tracks] = torch.where(feasible_mask, score_sub, torch.full_like(score_sub, -1e6))
    row_index = torch.arange(num_dets, device=score_sub.device)
    assign_scores[row_index, num_tracks + row_index] = null_scores

    assignments: List[Dict[str, Optional[int]]] = []
    lsa = _get_linear_sum_assignment() if use_hungarian else None
    if lsa is None:
        used_tracks: set[int] = set()
        edge_choices: List[tuple[float, int, int]] = []
        if num_tracks > 0:
            for det_idx in range(num_dets):
                for track_idx in feasible_mask[det_idx].nonzero(as_tuple=True)[0].tolist():
                    edge_choices.append((float(score_sub[det_idx, track_idx].item()), int(det_idx), int(track_idx)))
        edge_choices.sort(reverse=True)
        chosen_track_by_det: Dict[int, Optional[int]] = {}
        for _, det_idx, track_idx in edge_choices:
            if det_idx in chosen_track_by_det or track_idx in used_tracks:
                continue
            chosen_track_by_det[det_idx] = int(track_idx)
            used_tracks.add(int(track_idx))
        for det_idx in range(num_dets):
            assignments.append(
                {
                    "det_local_idx": int(det_idx),
                    "track_local_idx": chosen_track_by_det.get(int(det_idx), None),
                }
            )
        return assignments

    cost = (-assign_scores).detach().cpu().numpy()
    rows, cols = lsa(cost)
    chosen_cols: Dict[int, int] = {int(r): int(c) for r, c in zip(rows, cols)}
    for det_idx in range(num_dets):
        local_assign_idx = chosen_cols.get(int(det_idx), num_tracks + det_idx)
        if local_assign_idx >= num_tracks:
            assignments.append({"det_local_idx": int(det_idx), "track_local_idx": None})
            continue
        if num_tracks == 0 or not bool(feasible_mask[det_idx, local_assign_idx].item()):
            assignments.append({"det_local_idx": int(det_idx), "track_local_idx": None})
            continue
        assignments.append({"det_local_idx": int(det_idx), "track_local_idx": int(local_assign_idx)})
    return assignments


def summarize_component_sizes(components: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    comps = list(components)
    if not comps:
        return {
            "clusters": 0.0,
            "avg_detections": 0.0,
            "avg_tracks": 0.0,
        }
    dets = sum(float(comp.get("num_detections", 0.0)) for comp in comps)
    tracks = sum(float(comp.get("num_tracks", 0.0)) for comp in comps)
    n = float(len(comps))
    return {
        "clusters": n,
        "avg_detections": dets / n,
        "avg_tracks": tracks / n,
    }


def _extract_group_bipartite_maps(
    frame_groups: Sequence[Dict[str, Any]],
    *,
    topk: int,
) -> Tuple[Dict[int, List[int]], Dict[int, set[int]], Dict[int, Dict[str, Any]], Dict[int, Dict[int, int]]]:
    det_to_tracks: Dict[int, List[int]] = {}
    track_to_dets: Dict[int, set[int]] = defaultdict(set)
    group_by_det: Dict[int, Dict[str, Any]] = {}
    det_track_rank: Dict[int, Dict[int, int]] = {}
    for group in frame_groups:
        try:
            det_row = int(group.get("det_index", -1))
        except Exception:
            det_row = -1
        if det_row < 0:
            continue
        candidates = sorted(
            list(group.get("candidates", [])),
            key=lambda row: int(row.get("track_rank", 0) or 0),
        )
        det_tracks: List[int] = []
        track_rank_by_id: Dict[int, int] = {}
        for cand in candidates[: max(int(topk), 1)]:
            try:
                valid = int(cand.get("valid_train_row", 1))
            except Exception:
                valid = 1
            try:
                track_id = int(cand.get("track_id", -1))
            except Exception:
                track_id = -1
            if valid <= 0 or track_id < 0:
                continue
            det_tracks.append(track_id)
            track_rank_by_id[int(track_id)] = int(cand.get("track_rank", len(det_tracks)) or len(det_tracks))
        det_tracks = sorted(set(det_tracks))
        if not det_tracks:
            continue
        det_to_tracks[det_row] = det_tracks
        det_track_rank[det_row] = track_rank_by_id
        group_by_det[det_row] = dict(group)
        for track_id in det_tracks:
            track_to_dets[int(track_id)].add(int(det_row))
    return det_to_tracks, track_to_dets, group_by_det, det_track_rank


def _build_components_from_det_track_maps(
    det_to_tracks: Dict[int, List[int]],
    track_to_dets: Dict[int, set[int]],
    *,
    track_key: str,
) -> List[Dict[str, Any]]:
    if not det_to_tracks:
        return []

    components: List[Dict[str, Any]] = []
    visited_dets: set[int] = set()
    visited_tracks: set[int] = set()
    for start_det in sorted(det_to_tracks.keys()):
        if start_det in visited_dets:
            continue
        stack = [int(start_det)]
        cluster_dets: set[int] = set()
        cluster_tracks: set[int] = set()
        while stack:
            det_row = int(stack.pop())
            if det_row in visited_dets:
                continue
            visited_dets.add(det_row)
            cluster_dets.add(det_row)
            for track_id in det_to_tracks.get(det_row, []):
                cluster_tracks.add(int(track_id))
                if track_id in visited_tracks:
                    continue
                visited_tracks.add(track_id)
                for nxt_det in track_to_dets.get(track_id, set()):
                    if nxt_det not in visited_dets:
                        stack.append(int(nxt_det))
        components.append(
            {
                "det_rows": sorted(cluster_dets),
                track_key: sorted(cluster_tracks),
                "num_detections": int(len(cluster_dets)),
                "num_tracks": int(len(cluster_tracks)),
            }
        )
    return components


def _component_track_values(component: Dict[str, Any]) -> List[int]:
    values = component.get("track_ids", component.get("track_cols", []))
    return [int(x) for x in values]


def _seed_priority_key(group: Dict[str, Any], det_row: int) -> Tuple[float, ...]:
    rank_margin = float(group.get("rank_margin", group.get("refined_margin", 0.0)) or 0.0)
    rank_entropy = float(group.get("rank_entropy", 0.0) or 0.0)
    positive_rank = int(group.get("positive_rank", 999) or 999)
    return (
        -float(int(group.get("group_is_recoverable", 0) or 0)),
        -float(int(group.get("group_is_ambiguous", 0) or 0)),
        rank_margin,
        -rank_entropy,
        float(positive_rank if positive_rank > 0 else 999),
        float(det_row),
    )


def _det_priority_key(group: Dict[str, Any], det_row: int, seed_det_row: int) -> Tuple[float, ...]:
    seed_bonus = 0.0 if int(det_row) == int(seed_det_row) else 1.0
    base_key = _seed_priority_key(group, det_row)
    return (seed_bonus, *base_key)


def _track_priority_key(
    track_id: int,
    *,
    det_rows: Sequence[int],
    det_to_tracks: Dict[int, List[int]],
    det_track_rank: Dict[int, Dict[int, int]],
    seed_track_ids: set[int],
) -> Tuple[float, ...]:
    shared_count = 0
    best_rank = 10**9
    for det_row in det_rows:
        if int(track_id) in det_to_tracks.get(int(det_row), []):
            shared_count += 1
            best_rank = min(best_rank, int(det_track_rank.get(int(det_row), {}).get(int(track_id), 10**9)))
    seed_bonus = 0.0 if int(track_id) in seed_track_ids else 1.0
    return (
        seed_bonus,
        -float(shared_count),
        float(best_rank),
        float(track_id),
    )


def mine_centered_subcomponents_from_group_rows(
    frame_groups: Sequence[Dict[str, Any]],
    *,
    component: Dict[str, Any],
    topk: int,
    min_detections: int,
    max_detections: Optional[int] = None,
    max_tracks: Optional[int] = None,
    max_subcomponents: int = 0,
) -> List[Dict[str, Any]]:
    if max_subcomponents <= 0:
        return []
    det_to_tracks, track_to_dets, group_by_det, det_track_rank = _extract_group_bipartite_maps(frame_groups, topk=topk)
    component_dets = [int(x) for x in component.get("det_rows", []) if int(x) in det_to_tracks]
    component_tracks = set(
        int(x)
        for x in _component_track_values(component)
        if int(x) in track_to_dets
    )
    if not component_dets or not component_tracks:
        return []

    max_d = int(max_detections) if max_detections is not None and int(max_detections) > 0 else None
    max_t = int(max_tracks) if max_tracks is not None and int(max_tracks) > 0 else None
    ordered_seeds = sorted(component_dets, key=lambda det_row: _seed_priority_key(group_by_det.get(det_row, {}), det_row))

    subcomponents: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
    for seed_det_row in ordered_seeds[: max(int(max_subcomponents), 0)]:
        seed_track_ids = set(
            track_id
            for track_id in det_to_tracks.get(int(seed_det_row), [])
            if int(track_id) in component_tracks
        )
        if not seed_track_ids:
            continue

        selected_tracks = set(seed_track_ids)
        selected_dets = [int(seed_det_row)]
        prev_key: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
        for _ in range(4):
            neighbor_dets: set[int] = {int(seed_det_row)}
            for track_id in selected_tracks:
                neighbor_dets.update(
                    int(det_row)
                    for det_row in track_to_dets.get(int(track_id), set())
                    if int(det_row) in component_dets
                )
            ordered_dets = sorted(
                neighbor_dets,
                key=lambda det_row: _det_priority_key(group_by_det.get(int(det_row), {}), int(det_row), int(seed_det_row)),
            )
            if max_d is not None:
                ordered_dets = ordered_dets[:max_d]

            candidate_tracks: set[int] = set()
            for det_row in ordered_dets:
                candidate_tracks.update(
                    int(track_id)
                    for track_id in det_to_tracks.get(int(det_row), [])
                    if int(track_id) in component_tracks
                )
            ordered_tracks = sorted(
                candidate_tracks,
                key=lambda track_id: _track_priority_key(
                    int(track_id),
                    det_rows=ordered_dets,
                    det_to_tracks=det_to_tracks,
                    det_track_rank=det_track_rank,
                    seed_track_ids=seed_track_ids,
                ),
            )
            if max_t is not None:
                ordered_tracks = ordered_tracks[:max_t]
            selected_tracks = set(int(track_id) for track_id in ordered_tracks)
            if not selected_tracks:
                break

            pruned_dets = [
                int(det_row)
                for det_row in ordered_dets
                if any(int(track_id) in selected_tracks for track_id in det_to_tracks.get(int(det_row), []))
            ]
            if max_d is not None:
                pruned_dets = pruned_dets[:max_d]
            selected_dets = pruned_dets
            current_key = (tuple(sorted(selected_dets)), tuple(sorted(selected_tracks)))
            if current_key == prev_key:
                break
            prev_key = current_key

        if len(selected_dets) < int(min_detections):
            continue
        key = (tuple(sorted(selected_dets)), tuple(sorted(selected_tracks)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        subcomponents.append(
            {
                "det_rows": sorted(selected_dets),
                "track_ids": sorted(selected_tracks),
                "num_detections": int(len(selected_dets)),
                "num_tracks": int(len(selected_tracks)),
                "mined_from_large_component": 1,
                "seed_det_rows": [int(seed_det_row)],
                "source_component_num_detections": int(component.get("num_detections", len(component_dets))),
                "source_component_num_tracks": int(component.get("num_tracks", len(component_tracks))),
                "component_id_suffix": f"subdet{int(seed_det_row)}",
            }
        )
    return subcomponents


def build_group_components_from_group_rows(
    frame_groups: Sequence[Dict[str, Any]],
    *,
    topk: int,
) -> List[Dict[str, Any]]:
    if not frame_groups:
        return []
    det_to_tracks, track_to_dets, _, _ = _extract_group_bipartite_maps(frame_groups, topk=topk)
    return _build_components_from_det_track_maps(det_to_tracks, track_to_dets, track_key="track_ids")


def filter_local_conflict_clusters_by_size(
    components: Sequence[Dict[str, Any]],
    *,
    min_detections: int,
    max_detections: Optional[int] = None,
    max_tracks: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    filtered: List[Dict[str, Any]] = []
    skipped_large = 0
    max_d = int(max_detections) if max_detections is not None and int(max_detections) > 0 else None
    max_t = int(max_tracks) if max_tracks is not None and int(max_tracks) > 0 else None
    for comp in components:
        num_d = int(comp.get("num_detections", len(comp.get("det_rows", [])) or 0))
        num_t = int(comp.get("num_tracks", len(comp.get("track_cols", comp.get("track_ids", []))) or 0))
        if num_d < int(min_detections):
            continue
        if (max_d is not None and num_d > max_d) or (max_t is not None and num_t > max_t):
            skipped_large += 1
            continue
        filtered.append(dict(comp))
    return filtered, int(skipped_large)


def solve_assignment_with_private_defer(
    score_sub: torch.Tensor,
    *,
    feasible_mask: Optional[torch.Tensor] = None,
    defer_scores: Optional[torch.Tensor] = None,
    use_hungarian: bool = True,
) -> List[Dict[str, Optional[int]]]:
    return solve_assignment_with_private_null(
        score_sub=score_sub,
        feasible_mask=feasible_mask,
        null_scores=defer_scores,
        use_hungarian=use_hungarian,
    )


def compute_component_degree_features(
    num_detections: int,
    num_tracks: int,
    edge_det_index: Sequence[int],
    edge_track_index: Sequence[int],
) -> Dict[str, torch.Tensor]:
    row_degree = torch.zeros((int(num_detections),), dtype=torch.float32)
    col_degree = torch.zeros((int(num_tracks),), dtype=torch.float32)
    for det_idx, track_idx in zip(edge_det_index, edge_track_index):
        det_i = int(det_idx)
        trk_j = int(track_idx)
        if 0 <= det_i < int(num_detections):
            row_degree[det_i] += 1.0
        if 0 <= trk_j < int(num_tracks):
            col_degree[trk_j] += 1.0
    return {
        "row_degree": row_degree,
        "col_degree": col_degree,
    }
