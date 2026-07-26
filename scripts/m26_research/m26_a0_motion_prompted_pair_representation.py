from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import dmm_base_tracker as base  # noqa: E402
sys.path.insert(0, str(REPO / "external/BoT-SORT-main"))
from fast_reid.fast_reid_interfece import setup_cfg  # noqa: E402
from fast_reid.fastreid.modeling.meta_arch import build_model  # noqa: E402
from fast_reid.fastreid.utils.checkpoint import Checkpointer  # noqa: E402

SEQ = "MOT20-01"
DUMP = REPO / "outputs/alink_train_inputs/phase0_root/MOT20-01/dump_yolox_reid.npz"
IMAGE_ROOT = REPO / "datasets/MOT20/train/MOT20-01/img1"
GT = REPO / "datasets/MOT20/train/MOT20-01/gt/gt.txt"
ROOT = REPO / "outputs/mot20_m26_20260726/m26_a0_motion_prompted_pair_representation_m01_v1"
BASELINE_REFERENCE = REPO / ".mcp_tmp/m24_online_b0_parity/run1/track_results/MOT20-01.txt"
WEIGHTS = REPO / "external/BoT-SORT-main/pretrained/mot20_sbs_S50.pth"
CONFIG = REPO / "external/BoT-SORT-main/fast_reid/configs/MOT20/sbs_S50.yml"
EVENT_BUDGET = 64
TOPK = 8
MIN_AGE = 16
TRACK_NMS = 12
MAX_PER_FRAME = 4
FEATURE_H, FEATURE_W = 24, 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_class():
    for cls in base.DMMTrack.mro():
        if hasattr(cls, "_count"):
            return cls
    raise RuntimeError("track counter missing")


def implementation_globals() -> dict:
    return base.DMMBaseTracker.update.__globals__


def load_dump_arrays():
    dump = base.load_dump(DUMP)
    detections = np.asarray(dump["detections"], dtype=np.float32)
    features = np.asarray(dump["features"])
    offsets = np.asarray(dump["frame_offsets"], dtype=np.int64)
    columns = [str(value) for value in dump["columns"].tolist()]
    column = {name: index for index, name in enumerate(columns)}
    return detections, features, offsets, column


def frame_data(detections, features, offsets, column, frame: int):
    start, end = int(offsets[frame - 1]), int(offsets[frame])
    rows = detections[start:end]
    if len(rows):
        boxes = rows[:, [column["x1"], column["y1"], column["x2"], column["y2"]]].astype(np.float32)
        scores = rows[:, column["score"]].astype(np.float32)
        feats = np.asarray(features[start:end], dtype=np.float32)
        gids = rows[:, column["global_det_idx"]].astype(np.int64)
    else:
        dim = int(features.shape[1]) if features.ndim == 2 else 0
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        feats = np.zeros((0, dim), dtype=np.float32)
        gids = np.zeros((0,), dtype=np.int64)
    return boxes, scores, feats, gids


def output_rows(tracks, frame: int, cfg):
    rows = []
    for track in tracks:
        x, y, width, height = [float(value) for value in track.tlwh]
        if width * height < float(cfg.min_box_area) or width <= 0 or height <= 0:
            continue
        if width / max(height, 1e-12) > float(cfg.aspect_ratio_thresh):
            continue
        rows.append((frame, int(track.track_id), x, y, width, height, float(track.score)))
    return rows


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_l = np.maximum(left[:, 2] - left[:, 0], 0) * np.maximum(left[:, 3] - left[:, 1], 0)
    area_r = np.maximum(right[:, 2] - right[:, 0], 0) * np.maximum(right[:, 3] - right[:, 1], 0)
    return intersection / np.maximum(area_l[:, None] + area_r[None, :] - intersection, 1e-12)


def intersection_over_det(box: np.ndarray, others: np.ndarray) -> float:
    if len(others) == 0:
        return 0.0
    x1 = np.maximum(float(box[0]), others[:, 0])
    y1 = np.maximum(float(box[1]), others[:, 1])
    x2 = np.minimum(float(box[2]), others[:, 2])
    y2 = np.minimum(float(box[3]), others[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area = max((float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])), 1e-12)
    return float(np.max(intersection / area))


def center_inside_count(box: np.ndarray, track_boxes: np.ndarray) -> int:
    centers = 0.5 * (track_boxes[:, :2] + track_boxes[:, 2:])
    inside = (
        (centers[:, 0] >= box[0]) & (centers[:, 0] <= box[2])
        & (centers[:, 1] >= box[1]) & (centers[:, 1] <= box[3])
    )
    return int(inside.sum())


def box_iou_single(a: np.ndarray, b: np.ndarray) -> float:
    return float(iou_matrix(np.asarray([a]), np.asarray([b]))[0, 0])


def percentile(values: pd.Series, high_risk: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    fill = float(numeric.median()) if numeric.notna().any() else 0.0
    rank = numeric.fillna(fill).rank(method="average", pct=True)
    return rank if high_risk else 1.0 - rank


def freeze_pairs() -> None:
    out = ROOT / "frozen_pairs"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    detections, features, offsets, column = load_dump_arrays()
    frames = len(offsets) - 1
    cfg = base.TrackerConfig(dmm_v3_enable=False)
    count_class()._count = 0
    tracker = base.DMMBaseTracker(cfg)
    globals_ = implementation_globals()
    original_cost = globals_["assoc_cost"]
    original_assignment = globals_["matching"].linear_assignment
    event_records: list[dict] = []
    candidate_records: list[dict] = []
    event_arrays: dict[int, dict] = {}
    baseline_rows = []
    raw_event_index = 0

    for frame in range(1, frames + 1):
        boxes, scores, feats, gids = frame_data(detections, features, offsets, column, frame)
        calls = []

        def cost_hook(tracks, dets, tracker_cfg):
            cost, debug = original_cost(tracks, dets, tracker_cfg)
            calls.append({
                "cost": np.asarray(cost, dtype=np.float32).copy(),
                "debug": {key: np.asarray(value).copy() for key, value in debug.items()},
                "track_ids": np.asarray([int(track.track_id) for track in tracks], dtype=np.int64),
                "track_ages": np.asarray([int(track.tracklet_len) for track in tracks], dtype=np.int32),
                "track_boxes": np.asarray([track.tlbr for track in tracks], dtype=np.float32),
                "track_features": np.asarray([
                    np.zeros(2048, dtype=np.float32) if track.smooth_feat is None else np.asarray(track.smooth_feat, dtype=np.float32)
                    for track in tracks
                ], dtype=np.float32),
                "det_ids": np.asarray([int(det.det_global_idx) for det in dets], dtype=np.int64),
                "det_boxes": np.asarray([det.tlbr for det in dets], dtype=np.float32),
                "det_scores": np.asarray([float(det.score) for det in dets], dtype=np.float32),
                "det_features": np.asarray([
                    np.zeros(2048, dtype=np.float32) if det.curr_feat is None else np.asarray(det.curr_feat, dtype=np.float32)
                    for det in dets
                ], dtype=np.float32),
            })
            return cost, debug

        def assignment_hook(cost, thresh):
            result = original_assignment(cost, thresh=thresh)
            if calls and "matches" not in calls[-1] and np.asarray(cost).shape == calls[-1]["cost"].shape:
                calls[-1]["matches"] = np.asarray(result[0], dtype=np.int64).copy()
                calls[-1]["threshold"] = float(thresh)
            return result

        globals_["assoc_cost"] = cost_hook
        globals_["matching"].linear_assignment = assignment_hook
        try:
            tracks = tracker.update(boxes, scores, feats, gids)
        finally:
            globals_["assoc_cost"] = original_cost
            globals_["matching"].linear_assignment = original_assignment
        baseline_rows.extend(output_rows(tracks, frame, cfg))
        if not calls or frame >= frames:
            continue
        call = calls[0]
        cost = call["cost"]
        if not cost.size:
            continue
        raw_iou = call["debug"].get("raw_iou", np.ones_like(cost))
        emb = call["debug"].get("emb", np.ones_like(cost))
        matches = call.get("matches", np.empty((0, 2), dtype=np.int64))
        for track_row, chosen_col in matches:
            i, j = int(track_row), int(chosen_col)
            if i >= len(call["track_ids"]) or j >= len(call["det_ids"]):
                continue
            if int(call["track_ages"][i]) < MIN_AGE or not np.any(call["track_features"][i]):
                continue
            row_order = np.argsort(cost[i], kind="stable")[: min(TOPK, cost.shape[1])]
            if j not in row_order:
                continue
            chosen = float(cost[i, j])
            alternatives = np.delete(cost[i], j)
            columns = np.delete(cost[:, j], i)
            row_margin = (float(np.min(alternatives)) - chosen) if len(alternatives) else 1.0
            col_margin = (float(np.min(columns)) - chosen) if len(columns) else 1.0
            det_box = call["det_boxes"][j]
            other_dets = np.delete(call["det_boxes"], j, axis=0)
            overlap = intersection_over_det(det_box, other_dets)
            occupancy = center_inside_count(det_box, call["track_boxes"])
            box_iou = box_iou_single(call["track_boxes"][i], det_box)
            event_records.append({
                "raw_event_index": raw_event_index,
                "seq": SEQ,
                "frame": frame,
                "track_id": int(call["track_ids"][i]),
                "track_age": int(call["track_ages"][i]),
                "chosen_det_global_idx": int(call["det_ids"][j]),
                "chosen_cost": chosen,
                "row_margin": row_margin,
                "col_margin": col_margin,
                "pair_margin": min(row_margin, col_margin),
                "chosen_detection_overlap": overlap,
                "track_centers_in_chosen_detection": occupancy,
                "predicted_detection_iou": box_iou,
                "candidate_count": len(row_order),
                "gt_opened": False,
            })
            event_arrays[raw_event_index] = {
                "track_box": call["track_boxes"][i].copy(),
                "track_feature": call["track_features"][i].copy(),
                "all_track_boxes": call["track_boxes"].copy(),
                "all_track_ids": call["track_ids"].copy(),
            }
            for rank, col_index in enumerate(row_order, start=1):
                c = int(col_index)
                candidate_records.append({
                    "raw_event_index": raw_event_index,
                    "rank_original": rank,
                    "det_global_idx": int(call["det_ids"][c]),
                    "det_score": float(call["det_scores"][c]),
                    "det_x1": float(call["det_boxes"][c, 0]),
                    "det_y1": float(call["det_boxes"][c, 1]),
                    "det_x2": float(call["det_boxes"][c, 2]),
                    "det_y2": float(call["det_boxes"][c, 3]),
                    "global_similarity": float(1.0 - 2.0 * emb[i, c]) if float(emb[i, c]) < 1.0 else -1.0,
                    "embedding_cost": float(emb[i, c]),
                    "raw_iou_cost": float(raw_iou[i, c]),
                    "original_final_cost": float(cost[i, c]),
                    "chosen_by_b0": int(c == j),
                    "gt_opened": False,
                })
            raw_event_index += 1

    baseline_path = out / "baseline_online.txt"
    base.write_mot_results(baseline_path, baseline_rows)
    if baseline_path.read_bytes() != BASELINE_REFERENCE.read_bytes():
        raise RuntimeError("B0 replay mismatch")
    events = pd.DataFrame(event_records)
    pairs = pd.DataFrame(candidate_records)
    if events.empty:
        raise RuntimeError("no events")
    events["risk_low_margin"] = percentile(events.pair_margin, False)
    events["risk_detection_overlap"] = percentile(events.chosen_detection_overlap, True)
    events["risk_multi_track"] = percentile(events.track_centers_in_chosen_detection, True)
    events["risk_box_mismatch"] = percentile(events.predicted_detection_iou, False)
    channels = {
        "low_margin": events.sort_values(["risk_low_margin", "frame"], ascending=[False, True]).index.tolist(),
        "detection_overlap": events.sort_values(["risk_detection_overlap", "frame"], ascending=[False, True]).index.tolist(),
        "multi_track": events.sort_values(["risk_multi_track", "frame"], ascending=[False, True]).index.tolist(),
        "box_mismatch": events.sort_values(["risk_box_mismatch", "frame"], ascending=[False, True]).index.tolist(),
    }
    cursor = {name: 0 for name in channels}
    selected = []
    selected_set = set()
    selection = {}
    per_frame = defaultdict(int)
    active = list(channels)
    while active and len(selected) < EVENT_BUDGET:
        progressed = False
        for name in list(active):
            order = channels[name]
            chosen = None
            while cursor[name] < len(order):
                idx = int(order[cursor[name]])
                cursor[name] += 1
                if idx in selected_set:
                    continue
                row = events.loc[idx]
                if per_frame[int(row.frame)] >= MAX_PER_FRAME:
                    continue
                if any(int(events.loc[other].track_id) == int(row.track_id) and abs(int(events.loc[other].frame) - int(row.frame)) <= TRACK_NMS for other in selected):
                    continue
                chosen = idx
                break
            if chosen is None:
                active.remove(name)
                continue
            selected.append(chosen)
            selected_set.add(chosen)
            selection[chosen] = (name, cursor[name])
            per_frame[int(events.loc[chosen].frame)] += 1
            progressed = True
            if len(selected) >= EVENT_BUDGET:
                break
        if not progressed:
            break
    frozen = events.loc[selected].copy()
    frozen["selection_channel"] = [selection[int(index)][0] for index in frozen.index]
    frozen["selection_rank"] = [selection[int(index)][1] for index in frozen.index]
    frozen.sort_values(["frame", "track_id"], inplace=True)
    frozen.reset_index(drop=True, inplace=True)
    frozen.insert(0, "event_index", np.arange(len(frozen), dtype=int))
    raw_to_event = dict(zip(frozen.raw_event_index.astype(int), frozen.event_index.astype(int)))
    frozen_pairs = pairs[pairs.raw_event_index.isin(raw_to_event)].copy()
    frozen_pairs["event_index"] = frozen_pairs.raw_event_index.map(raw_to_event).astype(int)
    frozen_pairs.sort_values(["event_index", "rank_original"], inplace=True)
    frozen_pairs.reset_index(drop=True, inplace=True)
    max_tracks = max(len(event_arrays[int(raw)]["all_track_ids"]) for raw in frozen.raw_event_index)
    track_features = np.zeros((len(frozen), 2048), dtype=np.float32)
    track_boxes = np.zeros((len(frozen), 4), dtype=np.float32)
    all_boxes = np.zeros((len(frozen), max_tracks, 4), dtype=np.float32)
    all_ids = np.full((len(frozen), max_tracks), -1, dtype=np.int64)
    all_mask = np.zeros((len(frozen), max_tracks), dtype=np.uint8)
    for row in frozen.itertuples(index=False):
        data = event_arrays[int(row.raw_event_index)]
        index = int(row.event_index)
        count = len(data["all_track_ids"])
        track_features[index] = data["track_feature"]
        track_boxes[index] = data["track_box"]
        all_boxes[index, :count] = data["all_track_boxes"]
        all_ids[index, :count] = data["all_track_ids"]
        all_mask[index, :count] = 1
    frozen.to_parquet(out / "events.parquet", index=False)
    frozen_pairs.to_parquet(out / "pairs.parquet", index=False)
    np.savez_compressed(out / "event_state.npz", track_features=track_features, track_boxes=track_boxes, all_track_boxes=all_boxes, all_track_ids=all_ids, all_track_mask=all_mask)
    manifest = {
        "experiment_id": "M26-A0",
        "stage": "pairs_frozen",
        "gt_opened": False,
        "mot20_test_reads": 0,
        "raw_events": len(events),
        "frozen_events": len(frozen),
        "frozen_pairs": len(frozen_pairs),
        "candidate_topk": TOPK,
        "baseline_sha256": sha256(baseline_path),
        "events_sha256": sha256(out / "events.parquet"),
        "pairs_sha256": sha256(out / "pairs.parquet"),
        "state_sha256": sha256(out / "event_state.npz"),
        "input_sha256": sha256(DUMP),
        "script_sha256": sha256(Path(__file__)),
    }
    json_write(out / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(frozen[["event_index", "frame", "track_id", "selection_channel", "pair_margin", "chosen_detection_overlap", "track_centers_in_chosen_detection", "predicted_detection_iou"]].to_string(index=False))


def build_reid_model():
    cfg = setup_cfg(str(CONFIG), ["MODEL.WEIGHTS", str(WEIGHTS)])
    model = build_model(cfg)
    Checkpointer(model).load(str(WEIGHTS))
    model = model.eval().cuda()
    return cfg, model


def crop_tensor(image: np.ndarray, box: np.ndarray, size) -> torch.Tensor:
    height, width = image.shape[:2]
    tlbr = np.asarray(box, dtype=np.int64).copy()
    tlbr[0] = max(0, tlbr[0]); tlbr[1] = max(0, tlbr[1])
    tlbr[2] = min(width - 1, tlbr[2]); tlbr[3] = min(height - 1, tlbr[3])
    patch = image[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2], ::-1]
    if patch.size == 0:
        patch = np.zeros((int(size[0]), int(size[1]), 3), dtype=np.uint8)
    else:
        patch = cv2.resize(patch, tuple(size[::-1]), interpolation=cv2.INTER_LINEAR)
    return torch.as_tensor(patch.astype(np.float32).transpose(2, 0, 1))


def gaussian_for_box(box: np.ndarray, detection: np.ndarray, h: int, w: int) -> np.ndarray:
    dx1, dy1, dx2, dy2 = [float(value) for value in detection]
    dw, dh = max(dx2 - dx1, 1.0), max(dy2 - dy1, 1.0)
    cx = (0.5 * (float(box[0]) + float(box[2])) - dx1) / dw
    cy = (0.5 * (float(box[1]) + float(box[3])) - dy1) / dh
    bw = max((float(box[2]) - float(box[0])) / dw, 0.05)
    bh = max((float(box[3]) - float(box[1])) / dh, 0.05)
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w
    ys = (np.arange(h, dtype=np.float32) + 0.5) / h
    xx, yy = np.meshgrid(xs, ys)
    sx, sy = max(0.35 * bw, 0.10), max(0.35 * bh, 0.10)
    return np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)).astype(np.float32)


def weighted_gem(feature_map: torch.Tensor, weights: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    value = feature_map.float().clamp_min(1e-6).pow(float(p.detach().float().cpu()))
    weight = weights[:, None, :, :].float()
    pooled = (value * weight).sum(dim=(2, 3), keepdim=True) / weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return pooled.clamp_min(1e-12).pow(1.0 / float(p.detach().float().cpu()))


def extract_prompt_features() -> None:
    pair_dir = ROOT / "frozen_pairs"
    out = ROOT / "frozen_prompt_features"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    events = pd.read_parquet(pair_dir / "events.parquet")
    pairs = pd.read_parquet(pair_dir / "pairs.parquet")
    if "frame" not in pairs.columns:
        pairs = pairs.merge(
            events[["event_index", "frame"]],
            on="event_index",
            how="left",
            validate="many_to_one",
        )
    state = np.load(pair_dir / "event_state.npz")
    detections, cached_features, offsets, column = load_dump_arrays()
    cfg, model = build_reid_model()
    unique = pairs[["frame", "det_global_idx", "det_x1", "det_y1", "det_x2", "det_y2"]].drop_duplicates(["frame", "det_global_idx"]).sort_values(["frame", "det_global_idx"]).reset_index(drop=True)
    key_to_index = {(int(row.frame), int(row.det_global_idx)): index for index, row in unique.iterrows()}
    maps = np.zeros((len(unique), 2048, FEATURE_H, FEATURE_W), dtype=np.float16)
    global_features = np.zeros((len(unique), 2048), dtype=np.float32)
    batch_tensors = []
    batch_indices = []

    def flush():
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).cuda()
        with torch.no_grad():
            normalized = (x - model.pixel_mean) / model.pixel_std
            fmap = model.backbone(normalized)
            raw_global = model.heads.bottleneck(model.heads.pool_layer(fmap))[..., 0, 0]
            bad = (~torch.isfinite(fmap).flatten(1).all(1)) | (~torch.isfinite(raw_global).all(1))
            if bad.any():
                for local in torch.nonzero(bad, as_tuple=False).flatten().tolist():
                    single = x[local:local + 1]
                    single_map = model.backbone((single - model.pixel_mean) / model.pixel_std)
                    single_global = model.heads.bottleneck(model.heads.pool_layer(single_map))[..., 0, 0]
                    if not torch.isfinite(single_map).all() or not torch.isfinite(single_global).all():
                        raise RuntimeError(f"non-finite FastReID output after single-crop retry: batch_index={local}")
                    fmap[local] = single_map[0]
                    raw_global[local] = single_global[0]
            global_feat = torch.nn.functional.normalize(raw_global.float(), dim=1)
        for local, index in enumerate(batch_indices):
            maps[index] = fmap[local].cpu().numpy().astype(np.float16)
            global_features[index] = global_feat[local].cpu().numpy()
        batch_tensors.clear(); batch_indices.clear()

    current_frame = None
    image = None
    for index, row in unique.iterrows():
        frame = int(row.frame)
        if current_frame != frame:
            image = cv2.imread(str(IMAGE_ROOT / f"{frame:06d}.jpg"))
            if image is None:
                raise FileNotFoundError(frame)
            current_frame = frame
        box = np.asarray([row.det_x1, row.det_y1, row.det_x2, row.det_y2], dtype=np.float32)
        batch_tensors.append(crop_tensor(image, box, cfg.INPUT.SIZE_TEST))
        batch_indices.append(index)
        if len(batch_tensors) >= 8:
            flush()
    flush()
    p = model.heads.pool_layer.p
    records = []
    reproduction = []
    for row in pairs.itertuples(index=False):
        event_index = int(row.event_index)
        map_index = key_to_index[(int(row.frame), int(row.det_global_idx))]
        fmap = torch.from_numpy(maps[map_index]).cuda()[None]
        detection = np.asarray([row.det_x1, row.det_y1, row.det_x2, row.det_y2], dtype=np.float32)
        positive = gaussian_for_box(state["track_boxes"][event_index], detection, FEATURE_H, FEATURE_W)
        negatives = []
        for box, track_id, valid in zip(state["all_track_boxes"][event_index], state["all_track_ids"][event_index], state["all_track_mask"][event_index]):
            if not valid or int(track_id) == int(events.loc[event_index, "track_id"]):
                continue
            center = 0.5 * (box[:2] + box[2:])
            if detection[0] <= center[0] <= detection[2] and detection[1] <= center[1] <= detection[3]:
                negatives.append(gaussian_for_box(box, detection, FEATURE_H, FEATURE_W))
        negative = np.maximum.reduce(negatives) if negatives else np.zeros((FEATURE_H, FEATURE_W), dtype=np.float32)
        weight = positive * (1.0 - 0.9 * negative) + 0.02
        weight_t = torch.from_numpy(weight).cuda()[None]
        with torch.no_grad():
            pooled = weighted_gem(fmap, weight_t, p).to(dtype=next(model.heads.bottleneck.parameters()).dtype)
            prompted = model.heads.bottleneck(pooled)[..., 0, 0]
            prompted = torch.nn.functional.normalize(prompted.float(), dim=1)[0].cpu().numpy()
        track_feature = state["track_features"][event_index].astype(np.float32)
        track_feature /= max(float(np.linalg.norm(track_feature)), 1e-12)
        prompt_similarity = float(track_feature @ prompted)
        prompt_embedding_cost = (1.0 - prompt_similarity) / 2.0
        gated_embedding_cost = 1.0 if prompt_embedding_cost > 0.25 else prompt_embedding_cost
        if float(row.raw_iou_cost) > 0.5:
            gated_embedding_cost = 1.0
        prompt_final_cost = min(float(row.raw_iou_cost), float(gated_embedding_cost))
        cached = np.asarray(cached_features[int(row.det_global_idx)], dtype=np.float32)
        cached /= max(float(np.linalg.norm(cached)), 1e-12)
        reproduction_cos = float(cached @ global_features[map_index])
        reproduction.append(reproduction_cos)
        records.append({
            "event_index": event_index,
            "frame": int(row.frame),
            "det_global_idx": int(row.det_global_idx),
            "rank_original": int(row.rank_original),
            "original_final_cost": float(row.original_final_cost),
            "global_similarity": float(row.global_similarity),
            "raw_iou_cost": float(row.raw_iou_cost),
            "prompt_similarity": prompt_similarity,
            "prompt_embedding_cost": prompt_embedding_cost,
            "prompt_final_cost": prompt_final_cost,
            "positive_mass": float(positive.mean()),
            "negative_mass": float(negative.mean()),
            "prompt_mass": float(weight.mean()),
            "global_reproduction_cosine": reproduction_cos,
            "gt_opened": False,
        })
    frame = pd.DataFrame(records)
    frame["rank_prompt"] = frame.groupby("event_index", sort=False).prompt_final_cost.rank(method="first", ascending=True).astype(int)
    path = out / "prompt_pair_scores.parquet"
    frame.to_parquet(path, index=False)
    integrity = {
        "experiment_id": "M26-A0",
        "stage": "prompt_features_frozen",
        "gt_opened": False,
        "mot20_test_reads": 0,
        "events_sha256": sha256(pair_dir / "events.parquet"),
        "pairs_sha256": sha256(pair_dir / "pairs.parquet"),
        "state_sha256": sha256(pair_dir / "event_state.npz"),
        "prompt_scores_sha256": sha256(path),
        "unique_detection_crops": len(unique),
        "pair_scores": len(frame),
        "median_global_reproduction_cosine": float(np.median(reproduction)),
        "p05_global_reproduction_cosine": float(np.quantile(reproduction, 0.05)),
        "minimum_global_reproduction_cosine": float(np.min(reproduction)),
        "script_sha256": sha256(Path(__file__)),
        "weights_sha256": sha256(WEIGHTS),
    }
    json_write(out / "freeze_manifest.json", integrity)
    print(json.dumps(integrity, indent=2, sort_keys=True))
    print(frame.groupby("event_index").agg(original_top=("rank_original", "min"), prompt_top=("rank_prompt", "min"), negative_mass=("negative_mass", "max"), reproduction=("global_reproduction_cosine", "min")).head(20).to_string())


def parse_mot_rows(path: Path):
    by_frame = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0])); identity = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            by_frame[frame].append((identity, np.asarray([x, y, x + width, y + height], dtype=np.float32)))
    return by_frame


def parse_gt():
    by_frame = defaultdict(list)
    with GT.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split(",")
            if len(fields) < 8:
                continue
            frame, identity = int(float(fields[0])), int(float(fields[1]))
            conf, cls = int(float(fields[6])), int(float(fields[7]))
            if conf != 1 or cls != 1:
                continue
            x, y, width, height = map(float, fields[2:6])
            by_frame[frame].append((identity, np.asarray([x, y, x + width, y + height], dtype=np.float32)))
    return by_frame


def frame_match_ids(boxes: list[np.ndarray], gt_rows: list[tuple[int, np.ndarray]], threshold: float = 0.5):
    labels = np.full(len(boxes), -1, dtype=np.int64)
    if not boxes or not gt_rows:
        return labels
    gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=np.float32)
    similarity = iou_matrix(np.asarray(boxes, dtype=np.float32), gt_boxes)
    left, right = linear_sum_assignment(-similarity)
    for i, j in zip(left, right):
        if similarity[i, j] >= threshold:
            labels[int(i)] = int(gt_rows[int(j)][0])
    return labels


def label_after_freeze() -> None:
    pair_dir = ROOT / "frozen_pairs"
    prompt_dir = ROOT / "frozen_prompt_features"
    events = pd.read_parquet(pair_dir / "events.parquet")
    pairs = pd.read_parquet(pair_dir / "pairs.parquet")
    if "frame" not in pairs.columns:
        pairs = pairs.merge(
            events[["event_index", "frame"]],
            on="event_index",
            how="left",
            validate="many_to_one",
        )
    scores = pd.read_parquet(prompt_dir / "prompt_pair_scores.parquet")
    merged = pairs.merge(
        scores,
        on=["event_index", "frame", "det_global_idx", "rank_original"],
        how="left",
        suffixes=("", "_prompt"),
        validate="one_to_one",
    )
    tracker = parse_mot_rows(pair_dir / "baseline_online.txt")
    gt = parse_gt()
    track_identity = {}
    for event in events.itertuples(index=False):
        votes = []
        for frame in range(max(1, int(event.frame) - 8), int(event.frame)):
            rows = tracker.get(frame, [])
            boxes = [row[1] for row in rows]
            labels = frame_match_ids(boxes, gt.get(frame, []))
            for row, label in zip(rows, labels):
                if int(row[0]) == int(event.track_id) and int(label) > 0:
                    votes.append(int(label))
        track_identity[int(event.event_index)] = Counter(votes).most_common(1)[0][0] if votes else -1
    candidate_labels = {}
    detections, _features, offsets, column = load_dump_arrays()
    for frame in sorted(merged.frame.astype(int).unique()):
        start, end = int(offsets[frame - 1]), int(offsets[frame])
        rows = detections[start:end]
        boxes = [
            np.asarray(
                [row[column["x1"]], row[column["y1"]], row[column["x2"]], row[column["y2"]]],
                dtype=np.float32,
            )
            for row in rows
        ]
        labels = frame_match_ids(boxes, gt.get(int(frame), []))
        for row, label in zip(rows, labels):
            candidate_labels[(int(frame), int(row[column["global_det_idx"]]))] = int(label)
    merged["track_gt_id"] = merged.event_index.map(track_identity).astype(int)
    merged["candidate_gt_id"] = [candidate_labels.get((int(frame), int(gid)), -1) for frame, gid in zip(merged.frame, merged.det_global_idx)]
    merged["correct_candidate"] = ((merged.track_gt_id > 0) & (merged.track_gt_id == merged.candidate_gt_id)).astype(int)
    metrics = []
    fixes = breaks = present = global_correct = prompt_correct = union_correct = 0
    for event_index, group in merged.groupby("event_index", sort=True):
        correct = group[group.correct_candidate == 1]
        if correct.empty:
            continue
        present += 1
        global_top = group.sort_values(["original_final_cost", "det_global_idx"]).iloc[0]
        prompt_top = group.sort_values(["prompt_final_cost", "det_global_idx"]).iloc[0]
        g = int(global_top.correct_candidate)
        p = int(prompt_top.correct_candidate)
        global_correct += g; prompt_correct += p; union_correct += int(g or p)
        fixes += int((not g) and p); breaks += int(g and (not p))
        correct_row = correct.sort_values("rank_original").iloc[0]
        metrics.append({
            "event_index": int(event_index),
            "track_gt_id": int(correct_row.track_gt_id),
            "correct_det_global_idx": int(correct_row.det_global_idx),
            "global_correct": g,
            "prompt_correct": p,
            "global_rank_correct": int(group.sort_values(["original_final_cost", "det_global_idx"]).reset_index(drop=True).query("correct_candidate == 1").index[0]) + 1,
            "prompt_rank_correct": int(group.sort_values(["prompt_final_cost", "det_global_idx"]).reset_index(drop=True).query("correct_candidate == 1").index[0]) + 1,
        })
    integrity = json.loads((prompt_dir / "freeze_manifest.json").read_text())
    global_accuracy = global_correct / max(present, 1)
    prompt_accuracy = prompt_correct / max(present, 1)
    union_accuracy = union_correct / max(present, 1)
    top1_gain = prompt_accuracy - global_accuracy
    union_gain = union_accuracy - global_accuracy
    integrity_pass = integrity["median_global_reproduction_cosine"] >= 0.999 and integrity["p05_global_reproduction_cosine"] >= 0.995
    gate_pass = bool(top1_gain >= 0.08 and fixes >= 6 and fixes - breaks >= 4 and union_gain >= 0.10 and integrity_pass)
    decision = "PASS_M26_A0_AUTHORIZE_ONLINE_HOTA" if gate_pass else "FAIL_M26_A0_CLOSE_MOTION_PROMPTED_REID"
    report = {
        "experiment_id": "M26-A0",
        "status": "completed",
        "decision": decision,
        "gt_opened_after_prompt_feature_freeze": True,
        "deployable": False,
        "mot20_test_reads": 0,
        "test_submission": False,
        "frozen_events": len(events),
        "events_with_correct_candidate_in_top8": present,
        "global_top1_accuracy": global_accuracy,
        "prompt_top1_accuracy": prompt_accuracy,
        "top1_gain": top1_gain,
        "fixes": fixes,
        "breaks": breaks,
        "net_fixes": fixes - breaks,
        "oracle_union_top1_accuracy": union_accuracy,
        "oracle_union_gain": union_gain,
        "integrity": integrity,
        "gate_pass": gate_pass,
        "conclusion": "Pair-conditioned motion prompts contain sufficient complementary association evidence." if gate_pass else "The fixed motion prompt does not provide sufficient correct-association ranking gain on the M01 gate.",
    }
    merged.to_parquet(ROOT / "labeled_pair_scores.parquet", index=False)
    pd.DataFrame(metrics).to_csv(ROOT / "event_rank_metrics.csv", index=False)
    json_write(ROOT / "report.json", report)
    pd.DataFrame([{
        "experiment_id": "M26-A0", "present_events": present, "global_top1": global_accuracy,
        "prompt_top1": prompt_accuracy, "top1_gain": top1_gain, "fixes": fixes, "breaks": breaks,
        "net_fixes": fixes - breaks, "union_gain": union_gain, "gate_pass": int(gate_pass), "decision": decision,
    }]).to_csv(ROOT / "summary.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(pd.DataFrame(metrics).sort_values(["global_correct", "prompt_correct", "prompt_rank_correct"], ascending=[True, False, True]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze-pairs", "extract-prompt", "label"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "freeze-pairs":
        freeze_pairs()
    elif args.stage == "extract-prompt":
        extract_prompt_features()
    else:
        label_after_freeze()


if __name__ == "__main__":
    main()
