#!/usr/bin/env python3
"""M23-59 relation-pretrained hierarchical segmentation and flow feasibility.

Scientific scope
----------------
* New relation supervision comes only from MOT17 train, grouped by the seven
  physical videos rather than the 21 detector variants.
* The pre-existing M23-46 appearance extractor is frozen and inherited.  This
  script never trains a generic identity classifier or ReID reranker.
* MOT20 is evaluated by strict nested sequence LOSO.  An outer-held sequence is
  never used for training, normalization, calibration, checkpoint selection,
  policy selection, or teacher-action construction.
* Representation gates precede learned trackers.  A failed gate freezes P0.
* Detection rows, boxes and scores are immutable.  P1/P2, when admitted, use the
  unchanged M23-55 candidate generator and one-to-one time-forward path cover.

The implementation is intentionally one fixed small architecture.  There is no
hidden-size, layer-count, model-family, candidate-K, gap-quota, risk-level, edit-
budget, threshold-grid, or conformal-coverage scan.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import resource
import shutil
import struct
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.stats import beta
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v1")
EXTERNAL_MANIFEST = ROOT / "external_dataset_manifest.json"
PREREG_DOC = Path("docs/m23_59_relation_pretrained_hierarchical_flow_prereg_20260720.md")
PREREG_JSON = ROOT / "preregistered_protocol.json"
IMPLEMENTATION_MANIFEST = ROOT / "implementation_manifest.json"
EVENTS = ROOT / "protocol_events.jsonl"
M57_ROOT = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
M46_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
SOURCE_PARENT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
MOT17_ROOT = Path("datasets/MOT17/train")
MOT20_SEQS = ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
MOT17_PHYSICAL = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
MOT17_TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10"]
MOT17_VALIDATION = ["MOT17-11", "MOT17-13"]
MOT17_CANONICAL_VARIANT = "FRCNN"

APP_DIM = 128
GEOM_DIM = 16
GEOM_PROJ_DIM = 32
HIDDEN = 128
GRU_LAYERS = 2
MAX_NODE_ROWS = 30
SEG_ROWS = 8
PROJECTION_SEED = 2310
PARAMETER_CAP = 5_000_000
EXTERNAL_SEEDS = [2359001, 2359002, 2359003]
EXTERNAL_EPOCHS = 30
MOT20_FINETUNE_EPOCHS = 10
EXTERNAL_LR = 3e-4
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_NODES = 256
BATCH_RELATIONS = 256
PURE_PER_SEQ = 2048
AB_PER_SEQ = 2048
ABA_PER_SEQ = 1024
RELATION_PER_BUCKET_PER_SEQ = 1024
SWAP_PER_SEQ = 1024
RELATION_TRAIN_CAP_PER_MOT20_SEQ = 65536
PURE_FALSE_SPLIT_MAX = 0.002
BOUNDARY_PR_GATE = 0.283
BOUNDARY_PRECISION_ACTUAL_GATE = 0.35
BOUNDARY_RECALL95_GATE = 0.05
BOUNDARY_EACH_PRECISION_ACTUAL_GATE = 0.20
INNER_EACH_DELTA_GATE = 0.05
INNER_MEAN_DELTA_GATE = 0.20
P1_RISK = 0.02
P2_RISK = 0.05
RISK_CONFIDENCE = 0.95
GAP_BUCKETS = [("1-30", 1, 30), ("31-90", 31, 90), ("91-180", 91, 180), ("181-600", 181, 600)]

FORBIDDEN_TOKENS = (
    "same_gt", "modal_gt", "purity", "label_confidence", "actual_assa",
    "delta_hota", "matched_gt", "held_fold", "outer_teacher_action",
)

LOSS_WEIGHTS = {
    "node_focal": 1.0,
    "conditional_boundary_focal": 1.0,
    "boundary_count_consistency": 0.2,
    "outgoing_listwise": 0.5,
    "incoming_listwise": 0.5,
    "paired_replacement": 0.5,
    "catastrophic_risk": 1.5,
    "sequence_group_dro": 0.5,
    "edit_sparsity_source_anchor": 0.02,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj: Any, *, refuse_existing: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_existing and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def event(name: str, **payload: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {"event": name, "time": time.time(), **payload}
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def projection_matrix() -> np.ndarray:
    rng = np.random.default_rng(PROJECTION_SEED)
    return rng.normal(size=(2048, APP_DIM)).astype(np.float32) / math.sqrt(APP_DIM)


def project_embeddings(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = x @ projection_matrix()
    y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return y.astype(np.float32)


def audit_columns(columns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        low = str(col).lower()
        if any(token in low for token in FORBIDDEN_TOKENS):
            out.append(str(col))
    return sorted(out)


def detector_payload(path: Path) -> list[tuple[str, ...]]:
    payload = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split(",")
        if len(p) >= 6:
            payload.append((p[0], p[2], p[3], p[4], p[5], *p[6:]))
    return payload


def import_fastreid():
    bot = REPO / "external/BoT-SORT-main"
    if str(bot) not in sys.path:
        sys.path.insert(0, str(bot))
    from fast_reid.fast_reid_interfece import FastReIDInterface
    return FastReIDInterface


def feature_extractor_paths() -> tuple[Path, Path, Path]:
    bot = REPO / "external/BoT-SORT-main"
    return (
        bot / "fast_reid/configs/MOT20/sbs_S50.yml",
        bot / "pretrained/mot20_sbs_S50.pth",
        bot / "fast_reid/fast_reid_interfece.py",
    )


def init_frozen_encoder(batch_size: int = 256):
    cfg, weights, _ = feature_extractor_paths()
    FastReIDInterface = import_fastreid()
    return FastReIDInterface(str(cfg), str(weights), "gpu" if torch.cuda.is_available() else "cpu", batch_size=batch_size)


def read_mot_gt(path: Path) -> pd.DataFrame:
    names = ["frame", "identity", "x", "y", "w", "h", "mark", "class_id", "visibility"]
    frame = pd.read_csv(path, header=None, names=names)
    frame = frame[(frame.mark == 1) & (frame.class_id == 1) & (frame.visibility >= 0.1)].copy()
    frame = frame[(frame.w > 1.0) & (frame.h > 1.0)].copy()
    frame["frame"] = frame.frame.astype(int)
    frame["identity"] = frame.identity.astype(int)
    frame["x1"] = frame.x.astype(float)
    frame["y1"] = frame.y.astype(float)
    frame["x2"] = frame.x + frame.w
    frame["y2"] = frame.y + frame.h
    frame.sort_values(["frame", "identity"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame[["row_id", "frame", "identity", "x1", "y1", "x2", "y2", "visibility"]]


def geometry_features(meta: pd.DataFrame, width: int, height: int) -> np.ndarray:
    n = len(meta)
    out = np.zeros((n, GEOM_DIM), np.float32)
    x1 = meta.x1.to_numpy(np.float32); y1 = meta.y1.to_numpy(np.float32)
    x2 = meta.x2.to_numpy(np.float32); y2 = meta.y2.to_numpy(np.float32)
    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    w = np.maximum(x2 - x1, 1.0); h = np.maximum(y2 - y1, 1.0)
    out[:, 0] = cx / max(width, 1); out[:, 1] = cy / max(height, 1)
    out[:, 2] = w / max(width, 1); out[:, 3] = h / max(height, 1)
    out[:, 4] = np.log(w / h); out[:, 5] = np.log(np.maximum(w * h / max(width * height, 1), 1e-8))
    out[:, 6] = meta.visibility.to_numpy(np.float32) if "visibility" in meta else 1.0
    by_id = defaultdict(list)
    for i, ident in enumerate(meta.identity.to_numpy(int) if "identity" in meta else meta.track_id.to_numpy(int)):
        by_id[int(ident)].append(i)
    frames = meta.frame.to_numpy(np.float32)
    for ids in by_id.values():
        ids = sorted(ids, key=lambda i: (frames[i], i))
        for k, i in enumerate(ids):
            if k:
                j = ids[k - 1]; dt = max(frames[i] - frames[j], 1.0)
                out[i, 7] = (cx[i] - cx[j]) / max(h[i] * dt, 1.0)
                out[i, 8] = (cy[i] - cy[j]) / max(h[i] * dt, 1.0)
                out[i, 9] = math.log(max(w[i], 1.0) / max(w[j], 1.0)) / dt
                out[i, 10] = math.log(max(h[i], 1.0) / max(h[j], 1.0)) / dt
                out[i, 11] = min(dt / 30.0, 20.0)
            if k >= 2:
                j = ids[k - 1]; q = ids[k - 2]
                out[i, 12] = out[i, 7] - out[j, 7]
                out[i, 13] = out[i, 8] - out[j, 8]
    by_frame = defaultdict(list)
    for i, f in enumerate(meta.frame.to_numpy(int)):
        by_frame[int(f)].append(i)
    for ids in by_frame.values():
        centers = np.stack([cx[ids] / max(width, 1), cy[ids] / max(height, 1)], axis=1)
        if len(ids) > 1:
            dist = np.sqrt(((centers[:, None] - centers[None, :]) ** 2).sum(axis=2))
            dist += np.eye(len(ids), dtype=np.float32) * 1e6
            near = dist.min(axis=1)
        else:
            near = np.ones(1, np.float32)
        out[np.asarray(ids), 14] = np.minimum(len(ids) / 100.0, 5.0)
        out[np.asarray(ids), 15] = np.minimum(near, 1.0)
    return out


def read_seqinfo(path: Path) -> dict[str, int]:
    result = {"width": 0, "height": 0, "length": 0, "fps": 30}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip(); value = value.strip()
        if key == "imWidth": result["width"] = int(value)
        elif key == "imHeight": result["height"] = int(value)
        elif key == "seqLength": result["length"] = int(value)
        elif key == "frameRate": result["fps"] = int(value)
    return result


class HierarchicalRelationEncoder(nn.Module):
    """Fixed sub-5M relation-specific temporal encoder with exactly four heads."""

    def __init__(self) -> None:
        super().__init__()
        self.appearance_projection = nn.Sequential(nn.Linear(APP_DIM, APP_DIM), nn.LayerNorm(APP_DIM), nn.GELU())
        self.geometry_projection = nn.Sequential(nn.Linear(GEOM_DIM, GEOM_PROJ_DIM), nn.LayerNorm(GEOM_PROJ_DIM), nn.GELU())
        self.gru = nn.GRU(APP_DIM + GEOM_PROJ_DIM, HIDDEN, num_layers=GRU_LAYERS,
                          batch_first=True, bidirectional=True, dropout=0.1)
        enc = 2 * HIDDEN
        self.node_impurity_head = nn.Sequential(nn.Linear(2 * enc, 64), nn.GELU(), nn.Linear(64, 1))
        pair_dim = 3 * enc
        self.conditional_boundary_head = nn.Sequential(nn.Linear(pair_dim, 64), nn.GELU(), nn.Linear(64, 1))
        relation_dim = 8 * enc
        self.cross_relation_head = nn.Sequential(nn.Linear(relation_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.catastrophic_risk_head = nn.Sequential(nn.Linear(relation_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        app = self.appearance_projection(x[..., :APP_DIM])
        geo = self.geometry_projection(x[..., APP_DIM:])
        h, _ = self.gru(torch.cat([app, geo], dim=-1))
        return h * mask.unsqueeze(-1)

    @staticmethod
    def pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = h.sum(dim=1) / denom
        masked = h.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        mx = masked.max(dim=1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=-1)

    def node_and_boundary(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x, mask)
        node = self.node_impurity_head(self.pool(h, mask)).squeeze(-1)
        left = h[:, :-1]; right = h[:, 1:]
        pair = torch.cat([left, right, torch.abs(left - right)], dim=-1)
        boundary = self.conditional_boundary_head(pair).squeeze(-1)
        valid = mask[:, :-1] * mask[:, 1:]
        return node, boundary, valid

    def relation(self, src_x: torch.Tensor, src_m: torch.Tensor, dst_x: torch.Tensor, dst_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src = self.pool(self.encode(src_x, src_m), src_m)
        dst = self.pool(self.encode(dst_x, dst_m), dst_m)
        pair = torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1)
        score = self.cross_relation_head(pair).squeeze(-1)
        risk = self.catastrophic_risk_head(pair).squeeze(-1)
        return score, risk


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def focal_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None, gamma: float = 2.0) -> torch.Tensor:
    if valid is not None:
        keep = valid > 0
        logits = logits[keep]; target = target[keep]
    if logits.numel() == 0:
        return logits.sum() * 0.0
    target = target.float()
    pos = target.sum(); neg = target.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 1000.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, int); score = np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def precision_at_actual(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    k = int(y.sum())
    if k <= 0:
        return 0.0
    order = np.lexsort((np.arange(len(score)), -score))[:k]
    return float(y[order].mean())


def recall_at_precision(y: np.ndarray, score: np.ndarray, required: float) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    if y.sum() <= 0 or len(np.unique(y)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y, score)
    valid = recall[precision >= required]
    return float(valid.max()) if len(valid) else 0.0


def threshold_at_precision(y: np.ndarray, score: np.ndarray, required: float) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    if y.sum() <= 0:
        return 1.000001
    order = np.lexsort((np.arange(len(score)), -score))
    tp = 0
    best = None
    for rank, idx in enumerate(order, start=1):
        tp += int(y[idx] > 0)
        precision = tp / rank
        if precision >= required:
            best = float(score[idx])
    return 1.000001 if best is None else best


def clopper_pearson_upper(errors: int, total: int, confidence: float = RISK_CONFIDENCE) -> float:
    if total <= 0:
        return 1.0
    if errors >= total:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, total - errors))


def summarize_binary(y: np.ndarray, score: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, int); score = np.asarray(score, float)
    result = {
        "rows": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else 0.0,
        "pr_auc": safe_pr_auc(y, score), "roc_auc": safe_roc_auc(y, score),
        "precision_at_actual_count": precision_at_actual(y, score),
        "recall_at_90_precision": recall_at_precision(y, score, 0.90),
        "recall_at_95_precision": recall_at_precision(y, score, 0.95),
        "recall_at_99_precision": recall_at_precision(y, score, 0.99),
    }
    if threshold is not None:
        pred = score >= threshold
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        result.update({
            "threshold": float(threshold), "predicted": int(pred.sum()), "true_positive": tp,
            "false_positive": fp, "false_negative": fn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        })
    return result


def verify_freeze() -> dict[str, Any]:
    if not EXTERNAL_MANIFEST.exists() or not PREREG_DOC.exists() or not PREREG_JSON.exists() or not IMPLEMENTATION_MANIFEST.exists():
        raise FileNotFoundError("M23-59 implementation/preregistration freeze is incomplete")
    impl = json.loads(IMPLEMENTATION_MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "script": sha256_file(Path(__file__)) == impl["script_sha256"],
        "prereg_doc": sha256_file(PREREG_DOC) == impl["preregistration_document_sha256"],
        "prereg_json": sha256_file(PREREG_JSON) == impl["preregistration_json_sha256"],
        "external_manifest": sha256_file(EXTERNAL_MANIFEST) == impl["external_dataset_manifest_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen implementation mismatch: {checks}")
    return impl


def extract_external(seq: str) -> dict[str, Any]:
    verify_freeze()
    if seq not in MOT17_PHYSICAL:
        raise ValueError(seq)
    out = ROOT / "external_features" / seq
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed external feature fold {seq}")
    started = time.time()
    sid = seq[-2:]
    seq_dir = MOT17_ROOT / f"MOT17-{sid}-{MOT17_CANONICAL_VARIANT}"
    meta = read_mot_gt(seq_dir / "gt/gt.txt")
    info = read_seqinfo(seq_dir / "seqinfo.ini")
    encoder = init_frozen_encoder(256)
    feats = np.zeros((len(meta), APP_DIM), np.float32)
    by_frame = meta.groupby("frame", sort=True).indices
    failed = 0
    for frame_no, ids0 in by_frame.items():
        ids = np.asarray(ids0, np.int64)
        image_path = seq_dir / "img1" / f"{int(frame_no):06d}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            failed += 1
            continue
        boxes = meta.loc[ids, ["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        raw = encoder.inference(image, boxes)
        if len(raw) != len(ids):
            raise RuntimeError(f"{seq} frame {frame_no}: FastReID row mismatch")
        feats[ids] = project_embeddings(raw)
    if failed:
        raise RuntimeError(f"{seq}: {failed} image frames failed")
    geom = geometry_features(meta, info["width"], info["height"])
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "eligible_gt_rows.parquet"; feat_path = out / "appearance_128.f16.npy"; geom_path = out / "geometry_16.f16.npy"
    meta.to_parquet(meta_path, index=False)
    np.save(feat_path, feats.astype(np.float16)); np.save(geom_path, geom.astype(np.float16))
    manifest = {
        "experiment": "M23-59 external frozen feature extraction", "seq": seq,
        "physical_sequence": seq, "canonical_variant": MOT17_CANONICAL_VARIANT,
        "external_supervision": True, "labels_opened_after_freeze": True,
        "eligibility": "mark=1,class=1,visibility>=0.1,valid box",
        "rows": len(meta), "frames": int(meta.frame.nunique()), "identities": int(meta.identity.nunique()),
        "feature_dim": APP_DIM, "geometry_dim": GEOM_DIM, "failed_frames": failed,
        "artifacts": {
            "metadata": str(meta_path), "metadata_sha256": sha256_file(meta_path),
            "appearance": str(feat_path), "appearance_sha256": sha256_file(feat_path),
            "geometry": str(geom_path), "geometry_sha256": sha256_file(geom_path),
        },
        "feature_extractor": json.loads(EXTERNAL_MANIFEST.read_text(encoding="utf-8"))["inherited_frozen_feature_extractor"],
        "runtime_seconds": time.time() - started, "peak_rss_mb": peak_rss_mb(), "peak_gpu_memory_mb": peak_gpu_mb(),
    }
    json_write(out / "manifest.json", manifest, refuse_existing=True)
    event("external_features_completed", seq=seq, manifest_sha256=sha256_file(out / "manifest.json"))
    return manifest


def load_external_sequence(seq: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = ROOT / "external_features" / seq
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing external feature manifest for {seq}")
    meta = pd.read_parquet(root / "eligible_gt_rows.parquet")
    app = np.asarray(np.load(root / "appearance_128.f16.npy"), np.float32)
    geom = np.asarray(np.load(root / "geometry_16.f16.npy"), np.float32)
    if len(meta) != len(app) or len(meta) != len(geom):
        raise RuntimeError(f"{seq}: external feature length mismatch")
    return meta, app, geom


@dataclass(frozen=True)
class Window:
    seq: str
    identity: int
    indices: tuple[int, ...]
    first_frame: int
    last_frame: int
    center_frame: float
    prototype: tuple[float, ...]
    endpoint: tuple[float, ...]


def continuous_runs(indices: Sequence[int], frames: np.ndarray, max_step: int = 2) -> list[list[int]]:
    if not indices:
        return []
    runs: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        idx = int(idx)
        if int(frames[idx]) - int(frames[runs[-1][-1]]) <= max_step:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def build_windows(seq: str, meta: pd.DataFrame, app: np.ndarray, geom: np.ndarray,
                  length: int = MAX_NODE_ROWS, stride: int = 15) -> list[Window]:
    frames = meta.frame.to_numpy(int)
    identities = meta.identity.to_numpy(int)
    windows: list[Window] = []
    for ident in sorted(set(map(int, identities))):
        ids = np.flatnonzero(identities == ident).tolist()
        ids.sort(key=lambda i: (frames[i], i))
        for run in continuous_runs(ids, frames):
            if len(run) < length:
                continue
            starts = list(range(0, len(run) - length + 1, stride))
            if starts[-1] != len(run) - length:
                starts.append(len(run) - length)
            for start in starts:
                q = tuple(map(int, run[start:start + length]))
                proto = app[np.asarray(q)].mean(axis=0)
                proto /= max(float(np.linalg.norm(proto)), 1e-12)
                end = np.concatenate([geom[q[length // 2], :6], geom[q[-1], 7:14]]).astype(np.float32)
                windows.append(Window(
                    seq=seq, identity=ident, indices=q, first_frame=int(frames[q[0]]),
                    last_frame=int(frames[q[-1]]), center_frame=float(0.5 * (frames[q[0]] + frames[q[-1]])),
                    prototype=tuple(map(float, proto)), endpoint=tuple(map(float, end)),
                ))
    windows.sort(key=lambda w: (w.seq, w.identity, w.first_frame, w.indices[0]))
    return windows


def window_hardness(a: Window, b: Window) -> float:
    if a.identity == b.identity:
        return -1e9
    dt = abs(a.center_frame - b.center_frame)
    if dt > 30:
        return -1e9
    pa = np.asarray(a.prototype, np.float32); pb = np.asarray(b.prototype, np.float32)
    ea = np.asarray(a.endpoint, np.float32); eb = np.asarray(b.endpoint, np.float32)
    app = float(pa @ pb)
    geom = float(np.linalg.norm(ea[:4] - eb[:4]))
    scale = float(abs(ea[4] - eb[4]) + abs(ea[5] - eb[5]))
    if geom > 3.0 or scale > 1.4:
        return -1e9
    return 0.55 * app + 0.30 * math.exp(-geom) + 0.15 * math.exp(-dt / 30.0)


def hard_partner(index: int, windows: Sequence[Window]) -> int | None:
    a = windows[index]
    candidates: list[tuple[float, int]] = []
    for j, b in enumerate(windows):
        score = window_hardness(a, b)
        if score > -1e8:
            candidates.append((score, j))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], windows[x[1]].identity, windows[x[1]].first_frame, x[1]))
    return int(candidates[0][1])


def pad_sequence(indices: Sequence[int], app: np.ndarray, geom: np.ndarray,
                 length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((length, APP_DIM + GEOM_DIM), np.float32)
    mask = np.zeros(length, np.float32)
    ids = np.asarray(list(indices)[:length], np.int64)
    if len(ids):
        x[:len(ids), :APP_DIM] = app[ids]
        x[:len(ids), APP_DIM:] = geom[ids]
        mask[:len(ids)] = 1.0
    return x, mask


def save_array_set(root: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, value in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, value)
        artifacts[name] = {"path": str(path), "shape": list(value.shape), "dtype": str(value.dtype),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return artifacts


def build_external_examples(split: str) -> dict[str, Any]:
    verify_freeze()
    if split not in {"train", "validation"}:
        raise ValueError(split)
    sequences = MOT17_TRAIN if split == "train" else MOT17_VALIDATION
    out = ROOT / "external_examples" / split
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed external examples {split}")
    started = time.time()
    node_x: list[np.ndarray] = []; node_m: list[np.ndarray] = []
    node_y: list[int] = []; boundary_y: list[np.ndarray] = []; node_group: list[int] = []
    node_type: list[int] = []; aba_event: list[int] = []
    rel_src: list[np.ndarray] = []; rel_src_m: list[np.ndarray] = []
    rel_pos: list[np.ndarray] = []; rel_pos_m: list[np.ndarray] = []
    rel_neg: list[np.ndarray] = []; rel_neg_m: list[np.ndarray] = []
    rel_group: list[int] = []; rel_bucket: list[int] = []; rel_swap: list[int] = []
    seq_reports = {}
    for group, seq in enumerate(sequences):
        meta, app, geom = load_external_sequence(seq)
        windows = build_windows(seq, meta, app, geom)
        if not windows:
            raise RuntimeError(f"{seq}: no length-30 continuous windows")
        pure_ids = np.linspace(0, len(windows) - 1, min(PURE_PER_SEQ, len(windows)), dtype=int)
        pure_ids = np.unique(pure_ids)
        partner_cache: dict[int, int | None] = {}
        counts = defaultdict(int)
        for wi in pure_ids:
            w = windows[int(wi)]
            x, m = pad_sequence(w.indices, app, geom, MAX_NODE_ROWS)
            by = np.zeros(MAX_NODE_ROWS - 1, np.int8)
            node_x.append(x); node_m.append(m); node_y.append(0); boundary_y.append(by)
            node_group.append(group); node_type.append(0); aba_event.append(0); counts["pure"] += 1
        candidate_ids = np.linspace(0, len(windows) - 1, min(max(AB_PER_SEQ, ABA_PER_SEQ), len(windows)), dtype=int)
        candidate_ids = np.unique(candidate_ids)
        for wi in candidate_ids:
            partner_cache[int(wi)] = hard_partner(int(wi), windows)
        for wi in candidate_ids[:AB_PER_SEQ]:
            pj = partner_cache[int(wi)]
            if pj is None:
                continue
            a = windows[int(wi)]; b = windows[pj]
            ids_a = list(a.indices[:15]); ids_b = list(b.indices[-15:])
            x1, m1 = pad_sequence(ids_a, app, geom, 15); x2, m2 = pad_sequence(ids_b, app, geom, 15)
            x = np.concatenate([x1, x2], axis=0); m = np.concatenate([m1, m2], axis=0)
            by = np.zeros(29, np.int8); by[14] = 1
            node_x.append(x); node_m.append(m); node_y.append(1); boundary_y.append(by)
            node_group.append(group); node_type.append(1); aba_event.append(0); counts["A_to_B"] += 1
        for wi in candidate_ids[:ABA_PER_SEQ]:
            pj = partner_cache[int(wi)]
            if pj is None:
                continue
            a = windows[int(wi)]; b = windows[pj]
            ids = list(a.indices[:10]) + list(b.indices[10:20]) + list(a.indices[20:30])
            x, m = pad_sequence(ids, app, geom, MAX_NODE_ROWS)
            by = np.zeros(29, np.int8); by[9] = 1; by[19] = 1
            node_x.append(x); node_m.append(m); node_y.append(1); boundary_y.append(by)
            node_group.append(group); node_type.append(2); aba_event.append(1); counts["A_to_B_to_A"] += 1

        frames = meta.frame.to_numpy(int); identities = meta.identity.to_numpy(int)
        segment_bank: list[tuple[int, tuple[int, ...], int, np.ndarray, np.ndarray]] = []
        for ident in sorted(set(map(int, identities))):
            ids = np.flatnonzero(identities == ident).tolist(); ids.sort(key=lambda i: (frames[i], i))
            if len(ids) < SEG_ROWS:
                continue
            starts = list(range(0, len(ids) - SEG_ROWS + 1, SEG_ROWS))
            if starts and starts[-1] != len(ids) - SEG_ROWS:
                starts.append(len(ids) - SEG_ROWS)
            for st in starts:
                q = tuple(map(int, ids[st:st + SEG_ROWS]))
                proto = app[np.asarray(q)].mean(axis=0); proto /= max(float(np.linalg.norm(proto)), 1e-12)
                endpoint = geom[q[-1], :6].copy()
                segment_bank.append((ident, q, int(frames[q[0]]), proto, endpoint))
        segment_bank.sort(key=lambda z: (z[0], z[2], z[1][0]))
        by_identity: dict[int, list[int]] = defaultdict(list)
        for i, item in enumerate(segment_bank): by_identity[item[0]].append(i)
        for bucket_index, (bucket_name, lo, hi) in enumerate(GAP_BUCKETS):
            candidates: list[tuple[int, int]] = []
            for ident, ids in sorted(by_identity.items()):
                for ii in range(len(ids)):
                    aidx = ids[ii]; aq = segment_bank[aidx][1]; aend = int(frames[aq[-1]])
                    for jj in range(ii + 1, len(ids)):
                        bidx = ids[jj]; bstart = segment_bank[bidx][2]
                        gap = bstart - aend - 1
                        if gap < lo: continue
                        if gap > hi: break
                        candidates.append((aidx, bidx)); break
            if len(candidates) > RELATION_PER_BUCKET_PER_SEQ:
                keep = np.linspace(0, len(candidates) - 1, RELATION_PER_BUCKET_PER_SEQ, dtype=int)
                candidates = [candidates[int(i)] for i in np.unique(keep)]
            for aidx, bidx in candidates:
                a = segment_bank[aidx]; b = segment_bank[bidx]
                hard: list[tuple[float, int]] = []
                for ni, n in enumerate(segment_bank):
                    if n[0] == a[0] or abs(n[2] - b[2]) > 30:
                        continue
                    geom_dist = float(np.linalg.norm(b[4][:4] - n[4][:4]))
                    if geom_dist > 3.0: continue
                    score = 0.7 * float(a[3] @ n[3]) + 0.3 * math.exp(-geom_dist)
                    hard.append((score, ni))
                if not hard:
                    continue
                hard.sort(key=lambda z: (-z[0], segment_bank[z[1]][0], segment_bank[z[1]][2], z[1]))
                n = segment_bank[hard[0][1]]
                sx, sm = pad_sequence(a[1], app, geom, SEG_ROWS)
                px, pm = pad_sequence(b[1], app, geom, SEG_ROWS)
                nx, nm = pad_sequence(n[1], app, geom, SEG_ROWS)
                rel_src.append(sx); rel_src_m.append(sm); rel_pos.append(px); rel_pos_m.append(pm)
                rel_neg.append(nx); rel_neg_m.append(nm); rel_group.append(group); rel_bucket.append(bucket_index)
                rel_swap.append(0); counts[f"gap_{bucket_name}"] += 1
        swap_candidates = [(i, partner_cache[i]) for i in candidate_ids[:SWAP_PER_SEQ] if partner_cache.get(int(i)) is not None]
        for wi, pj in swap_candidates:
            a = windows[int(wi)]; b = windows[int(pj)]
            sx, sm = pad_sequence(a.indices[-SEG_ROWS:], app, geom, SEG_ROWS)
            px, pm = pad_sequence(a.indices[:SEG_ROWS], app, geom, SEG_ROWS)
            nx, nm = pad_sequence(b.indices[:SEG_ROWS], app, geom, SEG_ROWS)
            rel_src.append(sx); rel_src_m.append(sm); rel_pos.append(px); rel_pos_m.append(pm)
            rel_neg.append(nx); rel_neg_m.append(nm); rel_group.append(group); rel_bucket.append(-1)
            rel_swap.append(1); counts["paired_swap"] += 1
        seq_reports[seq] = {"windows": len(windows), "segments": len(segment_bank), **dict(counts)}

    arrays = {
        "node_x": np.asarray(node_x, np.float16), "node_mask": np.asarray(node_m, np.uint8),
        "node_label": np.asarray(node_y, np.uint8), "boundary_label": np.asarray(boundary_y, np.int8),
        "node_group": np.asarray(node_group, np.int16), "node_type": np.asarray(node_type, np.int8),
        "aba_event": np.asarray(aba_event, np.int8),
        "rel_src": np.asarray(rel_src, np.float16), "rel_src_mask": np.asarray(rel_src_m, np.uint8),
        "rel_pos": np.asarray(rel_pos, np.float16), "rel_pos_mask": np.asarray(rel_pos_m, np.uint8),
        "rel_neg": np.asarray(rel_neg, np.float16), "rel_neg_mask": np.asarray(rel_neg_m, np.uint8),
        "rel_group": np.asarray(rel_group, np.int16), "rel_bucket": np.asarray(rel_bucket, np.int8),
        "rel_swap": np.asarray(rel_swap, np.int8),
    }
    artifacts = save_array_set(out, arrays)
    report = {
        "experiment": "M23-59 relation-specific hard corruption examples", "split": split,
        "physical_sequences": sequences, "external_supervision": True,
        "node_examples": len(node_x), "relation_triplets": len(rel_src),
        "synthetic_protocol": {
            "node_rows": MAX_NODE_ROWS, "pure_per_sequence_cap": PURE_PER_SEQ,
            "A_to_B_per_sequence_cap": AB_PER_SEQ, "A_to_B_to_A_per_sequence_cap": ABA_PER_SEQ,
            "segment_rows": SEG_ROWS, "relation_per_gap_bucket_per_sequence_cap": RELATION_PER_BUCKET_PER_SEQ,
            "swap_per_sequence_cap": SWAP_PER_SEQ, "gap_buckets": GAP_BUCKETS,
            "hard_negative": "same physical scene, |time offset|<=30, different identity, geometry distance<=3, ranked by 0.7 appearance + 0.3 geometry",
        },
        "sequence_reports": seq_reports, "artifacts": artifacts,
        "runtime_seconds": time.time() - started, "peak_rss_mb": peak_rss_mb(),
    }
    json_write(out / "manifest.json", report, refuse_existing=True)
    event("external_examples_completed", split=split, manifest_sha256=sha256_file(out / "manifest.json"))
    return report


def load_array_set(root: Path) -> dict[str, np.ndarray]:
    return {p.stem: np.load(p, mmap_mode="r") for p in root.glob("*.npy")}


def group_weighted(losses: torch.Tensor, groups: torch.Tensor, q: torch.Tensor, eta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for group in range(len(q)):
        mask = groups == group
        values.append(losses[mask].mean() if mask.any() else losses.mean() * 0.0)
    lv = torch.stack(values)
    with torch.no_grad():
        q = q * torch.exp(eta * lv.detach().clamp(max=20.0))
        q = q / q.sum().clamp_min(1e-12)
    return (q * lv).sum(), q


def sample_indices(n: int, batch: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    order = rng.permutation(n)
    for start in range(0, n, batch):
        yield order[start:start + batch]


def tensor(a: np.ndarray, dev: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=dev, dtype=dtype)


def validation_metrics(model: HierarchicalRelationEncoder, data: dict[str, np.ndarray], dev: torch.device) -> dict[str, Any]:
    model.eval()
    node_scores = []; node_y = []; boundary_scores = []; boundary_y = []; aba_hit = []
    with torch.no_grad():
        for ids in [np.arange(i, min(i + 1024, len(data["node_label"]))) for i in range(0, len(data["node_label"]), 1024)]:
            x = tensor(data["node_x"][ids], dev); m = tensor(data["node_mask"][ids], dev)
            nl, bl, valid = model.node_and_boundary(x, m)
            node_scores.append(torch.sigmoid(nl).cpu().numpy()); node_y.append(np.asarray(data["node_label"][ids]))
            by = np.asarray(data["boundary_label"][ids]); v = valid.cpu().numpy() > 0
            conditional = (np.asarray(data["node_label"][ids])[:, None] > 0) & v
            boundary_scores.append(torch.sigmoid(bl).cpu().numpy()[conditional]); boundary_y.append(by[conditional])
            probs = torch.sigmoid(bl).cpu().numpy()
            for local, idx in enumerate(ids):
                if int(data["aba_event"][idx]) > 0:
                    top2 = np.argsort(-probs[local])[:2]
                    aba_hit.append(int(set(top2.tolist()) == {9, 19}))
        pos_scores = []; neg_scores = []; neg_risk = []; pos_risk = []
        for ids in [np.arange(i, min(i + 1024, len(data["rel_group"]))) for i in range(0, len(data["rel_group"]), 1024)]:
            sx = tensor(data["rel_src"][ids], dev); sm = tensor(data["rel_src_mask"][ids], dev)
            px = tensor(data["rel_pos"][ids], dev); pm = tensor(data["rel_pos_mask"][ids], dev)
            nx = tensor(data["rel_neg"][ids], dev); nm = tensor(data["rel_neg_mask"][ids], dev)
            ps, pr = model.relation(sx, sm, px, pm); ns, nr = model.relation(sx, sm, nx, nm)
            pos_scores.append(ps.cpu().numpy()); neg_scores.append(ns.cpu().numpy())
            pos_risk.append(torch.sigmoid(pr).cpu().numpy()); neg_risk.append(torch.sigmoid(nr).cpu().numpy())
    ny = np.concatenate(node_y); ns = np.concatenate(node_scores)
    by = np.concatenate(boundary_y) if boundary_y else np.zeros(0, int)
    bs = np.concatenate(boundary_scores) if boundary_scores else np.zeros(0, float)
    ps = np.concatenate(pos_scores); ng = np.concatenate(neg_scores)
    pr = np.concatenate(pos_risk); nr = np.concatenate(neg_risk)
    relation_r1 = float((ps > ng).mean()) if len(ps) else 0.0
    risk_y = np.concatenate([np.zeros(len(pr), int), np.ones(len(nr), int)])
    risk_s = np.concatenate([pr, nr])
    catastrophic_false_link = float((ng >= ps).mean()) if len(ps) else 1.0
    node_report = summarize_binary(ny, ns)
    boundary_report = summarize_binary(by, bs)
    risk_report = summarize_binary(risk_y, risk_s)
    composite = (
        0.30 * float(node_report["pr_auc"] or 0.0)
        + 0.35 * float(boundary_report["pr_auc"] or 0.0)
        + 0.25 * relation_r1
        + 0.10 * (1.0 - catastrophic_false_link)
    )
    return {
        "node": node_report, "conditional_boundary": boundary_report,
        "A_to_B_to_A_exact_two_boundary_recall": float(np.mean(aba_hit)) if aba_hit else 0.0,
        "relation_R_at_1_pairwise": relation_r1, "catastrophic_false_link_rate": catastrophic_false_link,
        "risk": risk_report, "checkpoint_selection_composite": composite,
    }


def train_external_seed(seed: int, train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> dict[str, Any]:
    set_determinism(seed)
    dev = device()
    model = HierarchicalRelationEncoder().to(dev)
    params = parameter_count(model)
    if params > PARAMETER_CAP:
        raise RuntimeError(f"parameter cap exceeded: {params}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=EXTERNAL_LR, weight_decay=WEIGHT_DECAY)
    q_node = torch.ones(len(MOT17_TRAIN), device=dev) / len(MOT17_TRAIN)
    q_rel = torch.ones(len(MOT17_TRAIN), device=dev) / len(MOT17_TRAIN)
    rng = np.random.default_rng(seed)
    seed_root = ROOT / "external_pretraining" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    best = None; history = []
    for epoch in range(EXTERNAL_EPOCHS):
        model.train(); epoch_losses = []
        node_batches = list(sample_indices(len(train["node_label"]), BATCH_NODES, rng))
        rel_batches = list(sample_indices(len(train["rel_group"]), BATCH_RELATIONS, rng))
        steps = max(len(node_batches), len(rel_batches))
        for step in range(steps):
            ni = node_batches[step % len(node_batches)]; ri = rel_batches[step % len(rel_batches)]
            x = tensor(train["node_x"][ni], dev); m = tensor(train["node_mask"][ni], dev)
            ny = tensor(train["node_label"][ni], dev); by = tensor(train["boundary_label"][ni], dev)
            ng = tensor(train["node_group"][ni], dev, torch.long)
            nl, bl, bv = model.node_and_boundary(x, m)
            node_per = F.binary_cross_entropy_with_logits(nl, ny, reduction="none")
            node_loss, q_node = group_weighted(node_per, ng, q_node)
            valid_boundary = (bv > 0) & (ny[:, None] > 0)
            boundary_loss = focal_loss(bl, by, valid_boundary)
            count_loss = ((torch.sigmoid(bl) * bv).sum(dim=1) - by.clamp_min(0).sum(dim=1)).abs().mean()
            sparsity = (torch.sigmoid(bl) * bv).mean()

            sx = tensor(train["rel_src"][ri], dev); sm = tensor(train["rel_src_mask"][ri], dev)
            px = tensor(train["rel_pos"][ri], dev); pm = tensor(train["rel_pos_mask"][ri], dev)
            nx = tensor(train["rel_neg"][ri], dev); nm = tensor(train["rel_neg_mask"][ri], dev)
            rg = tensor(train["rel_group"][ri], dev, torch.long)
            swap = tensor(train["rel_swap"][ri], dev)
            ps, pr = model.relation(sx, sm, px, pm); ns, nr = model.relation(sx, sm, nx, nm)
            rank_per = F.softplus(-(ps - ns))
            relation_loss, q_rel = group_weighted(rank_per, rg, q_rel)
            paired = (rank_per * (1.0 + swap)).mean()
            risk = 0.5 * (F.binary_cross_entropy_with_logits(pr, torch.zeros_like(pr)) + F.binary_cross_entropy_with_logits(nr, torch.ones_like(nr)))
            loss = (
                LOSS_WEIGHTS["node_focal"] * node_loss
                + LOSS_WEIGHTS["conditional_boundary_focal"] * boundary_loss
                + LOSS_WEIGHTS["boundary_count_consistency"] * count_loss
                + (LOSS_WEIGHTS["outgoing_listwise"] + LOSS_WEIGHTS["incoming_listwise"]) * relation_loss
                + LOSS_WEIGHTS["paired_replacement"] * paired
                + LOSS_WEIGHTS["catastrophic_risk"] * risk
                + LOSS_WEIGHTS["edit_sparsity_source_anchor"] * sparsity
            )
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        metrics = validation_metrics(model, val, dev)
        row = {"epoch": epoch + 1, "mean_train_loss": float(np.mean(epoch_losses)), **metrics}
        history.append(row)
        candidate = (float(metrics["checkpoint_selection_composite"]), -(epoch + 1))
        if best is None or candidate > best[0]:
            ckpt = seed_root / "best.pt"
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch + 1,
                        "parameter_count": params, "validation": metrics}, ckpt)
            best = (candidate, ckpt, row)
    history_path = seed_root / "history.json"
    json_write(history_path, history)
    assert best is not None
    report = {
        "seed": seed, "epochs": EXTERNAL_EPOCHS, "parameter_count": params,
        "best_epoch": best[2]["epoch"], "validation": best[2],
        "checkpoint": str(best[1]), "checkpoint_sha256": sha256_file(best[1]),
        "history": str(history_path), "history_sha256": sha256_file(history_path),
    }
    json_write(seed_root / "report.json", report)
    return report


def train_external() -> dict[str, Any]:
    verify_freeze()
    out = ROOT / "external_pretraining"
    if (out / "frozen_checkpoint_manifest.json").exists():
        raise FileExistsError("refusing to overwrite completed external pretraining")
    started = time.time()
    train = load_array_set(ROOT / "external_examples/train")
    val = load_array_set(ROOT / "external_examples/validation")
    reports = [train_external_seed(seed, train, val) for seed in EXTERNAL_SEEDS]
    reports.sort(key=lambda r: (-float(r["validation"]["checkpoint_selection_composite"]), int(r["seed"])))
    selected = reports[0]
    source = Path(selected["checkpoint"]); frozen = out / "relation_pretrained_frozen.pt"
    shutil.copyfile(source, frozen)
    checkpoint = torch.load(frozen, map_location="cpu")
    params = int(checkpoint["parameter_count"])
    if params > PARAMETER_CAP:
        raise RuntimeError("frozen checkpoint exceeds parameter cap")
    manifest = {
        "experiment": "M23-59 frozen external relation checkpoint", "external_supervision": True,
        "checkpoint_selection": "maximum fixed MOT17 validation composite; tie lower seed then earlier epoch",
        "training_physical_sequences": MOT17_TRAIN, "validation_physical_sequences": MOT17_VALIDATION,
        "seeds": EXTERNAL_SEEDS, "epochs": EXTERNAL_EPOCHS, "optimizer": "AdamW",
        "learning_rate": EXTERNAL_LR, "weight_decay": WEIGHT_DECAY,
        "selected_seed": selected["seed"], "selected_epoch": selected["best_epoch"],
        "parameter_count": params, "parameter_cap": PARAMETER_CAP,
        "checkpoint": str(frozen), "checkpoint_sha256": sha256_file(frozen),
        "seed_reports": reports,
        "train_examples_manifest_sha256": sha256_file(ROOT / "external_examples/train/manifest.json"),
        "validation_examples_manifest_sha256": sha256_file(ROOT / "external_examples/validation/manifest.json"),
        "runtime_seconds": time.time() - started, "peak_rss_mb": peak_rss_mb(), "peak_gpu_memory_mb": peak_gpu_mb(),
    }
    json_write(out / "frozen_checkpoint_manifest.json", manifest, refuse_existing=True)
    event("external_checkpoint_frozen", checkpoint_sha256=manifest["checkpoint_sha256"], selected_seed=selected["seed"])
    return manifest


def load_frozen_external_model(dev: torch.device) -> tuple[HierarchicalRelationEncoder, dict[str, Any]]:
    manifest_path = ROOT / "external_pretraining/frozen_checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("external checkpoint not frozen")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["checkpoint"])
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("external checkpoint SHA mismatch")
    state = torch.load(checkpoint_path, map_location="cpu")
    model = HierarchicalRelationEncoder().to(dev)
    model.load_state_dict(state["model"])
    if parameter_count(model) != int(manifest["parameter_count"]):
        raise RuntimeError("parameter count mismatch")
    return model, manifest


def mot20_source_tracker(seq: str) -> Path:
    cached = M46_CACHE / seq / "track_results" / f"{seq}.txt"
    if cached.exists():
        return cached
    return SOURCE_PARENT / f"{seq}.txt"


def freeze_mot20_observable(seq: str) -> dict[str, Any]:
    verify_freeze()
    if seq not in MOT20_SEQS:
        raise ValueError(seq)
    out = ROOT / "mot20_observable" / seq
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite frozen MOT20 observable {seq}")
    started = time.time()
    m57 = load_module(f"m23_59_m57_obs_{seq[-2:]}", "scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
    (
        _, _, _, source_path, baseline_path, source_rows, fixed_nodes, chunks,
        parent_ids, crowd, mapped, _, row_embeddings,
    ) = m57.prepare_observable_rows(seq)
    # m57 row embeddings are the exact seed-2310 128-D appearance representation.
    app = np.zeros((len(source_rows), APP_DIM), np.float32)
    for i in range(len(source_rows)):
        if bool(mapped[i]):
            app[i] = row_embeddings[i]
    rows_frame = pd.DataFrame({
        "row_index": np.arange(len(source_rows), dtype=np.int64),
        "line_index": [int(r.get("line", i)) for i, r in enumerate(source_rows)],
        "frame": [int(r["frame"]) for r in source_rows],
        "track_id": [int(r["track_id"]) for r in source_rows],
        "x1": [float(r["x1"]) for r in source_rows], "y1": [float(r["y1"]) for r in source_rows],
        "x2": [float(r["x2"]) for r in source_rows], "y2": [float(r["y2"]) for r in source_rows],
        "visibility": np.ones(len(source_rows), np.float32),
    })
    info = read_seqinfo(Path("datasets/MOT20/train") / seq / "seqinfo.ini")
    geom_input = rows_frame.rename(columns={"track_id": "identity"})
    geom = geometry_features(geom_input, info["width"], info["height"])
    # Preserve M23-57 crowd and mapping context without changing dimensions.
    geom[:, 14] = np.minimum(np.asarray(crowd, np.float32) / 100.0, 5.0)
    geom[:, 15] = np.asarray(mapped, np.float32)
    features = np.concatenate([app, geom], axis=1)
    if not np.isfinite(features).all():
        raise RuntimeError(f"{seq}: non-finite MOT20 observable tensor")
    forbidden = audit_columns(rows_frame.columns)
    if forbidden:
        raise RuntimeError(f"{seq}: forbidden observable columns {forbidden}")
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.parquet"; feature_path = out / "row_features.f16.npy"
    rows_frame.to_parquet(rows_path, index=False); np.save(feature_path, features.astype(np.float16))
    membership_path = M57_ROOT / "boundary_universe" / seq / "chunk_membership.parquet"
    boundary_manifest = M57_ROOT / "boundary_universe" / seq / "freeze_manifest.json"
    manifest = {
        "experiment": "M23-59 MOT20 GT-free hierarchical observable freeze", "seq": seq,
        "gt_opened": False, "outer_teacher_action_read": False, "external_supervision": True,
        "source_tracker": str(source_path), "source_tracker_sha256": sha256_file(source_path),
        "m23_46_baseline": str(baseline_path), "m23_46_baseline_sha256": sha256_file(baseline_path),
        "rows": len(source_rows), "fixed_nodes": len(fixed_nodes), "feature_dim": APP_DIM + GEOM_DIM,
        "mapped_rows": int(np.asarray(mapped, bool).sum()), "forbidden_columns": forbidden,
        "artifacts": {
            "rows": str(rows_path), "rows_sha256": sha256_file(rows_path),
            "row_features": str(feature_path), "row_features_sha256": sha256_file(feature_path),
            "fixed_membership": str(membership_path), "fixed_membership_sha256": sha256_file(membership_path),
            "m23_57_boundary_manifest": str(boundary_manifest), "m23_57_boundary_manifest_sha256": sha256_file(boundary_manifest),
        },
        "runtime_seconds": time.time() - started, "peak_rss_mb": peak_rss_mb(), "peak_gpu_memory_mb": peak_gpu_mb(),
    }
    json_write(out / "manifest.json", manifest, refuse_existing=True)
    event("mot20_observable_frozen", seq=seq, manifest_sha256=sha256_file(out / "manifest.json"))
    return manifest


def verify_all_mot20_observable_frozen() -> None:
    missing = [seq for seq in MOT20_SEQS if not (ROOT / "mot20_observable" / seq / "manifest.json").exists()]
    if missing:
        raise RuntimeError(f"all four GT-free observables must freeze before labels: {missing}")
    events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    freezes = [i for i, row in enumerate(events) if row["event"] == "mot20_observable_frozen"]
    labels = [i for i, row in enumerate(events) if row["event"] == "mot20_labels_unlocked"]
    if len(freezes) < 4:
        raise RuntimeError("fewer than four observable freeze events")
    if labels and max(sorted(freezes)[:4]) >= min(labels):
        raise RuntimeError("MOT20 labels opened before all observable freezes")


def unlock_mot20_labels(seq: str) -> dict[str, Any]:
    verify_freeze(); verify_all_mot20_observable_frozen()
    if seq not in MOT20_SEQS:
        raise ValueError(seq)
    out = ROOT / "mot20_labels" / seq
    if (out / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite labels {seq}")
    membership = pd.read_parquet(M57_ROOT / "boundary_universe" / seq / "chunk_membership.parquet")
    labels_path = M57_ROOT / "teacher_labels" / seq / "boundary_labels.parquet"
    labels = pd.read_parquet(labels_path)
    n = len(membership)
    node_label = np.zeros(n, np.uint8)
    boundary = np.full((n, MAX_NODE_ROWS - 1), -1, np.int8)
    capacity = np.zeros((n, MAX_NODE_ROWS - 1), np.int8)
    aba = np.zeros(n, np.uint8)
    for chunk_id, group in labels.groupby("fixed_chunk_id", sort=True):
        chunk_id = int(chunk_id)
        node_label[chunk_id] = int((group.node_identity_count.max() > 1) or (group.capacity_split.sum() > 0))
        positives = 0
        for row in group.itertuples():
            pos = int(row.position) - 1
            if 0 <= pos < MAX_NODE_ROWS - 1:
                if int(row.ambiguous_or_unsupported) == 0:
                    boundary[chunk_id, pos] = int(row.audit_label)
                capacity[chunk_id, pos] = int(row.capacity_split)
                positives += int(row.capacity_split)
        aba[chunk_id] = int(positives >= 2)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {
        "node_label": node_label, "boundary_label": boundary,
        "capacity_split": capacity, "aba_event": aba,
    }
    artifacts = save_array_set(out, arrays)
    relation_files = {
        "nodes": M57_ROOT / "capacity" / seq / "frozen_candidate_graph/nodes.parquet",
        "edges": M57_ROOT / "capacity" / seq / "frozen_candidate_graph/edges.parquet",
        "teacher_edges": M57_ROOT / "capacity" / seq / "teacher_identity_flow/teacher_edge_utilities.parquet",
        "successor_events": M57_ROOT / "capacity" / seq / "postfreeze_audit/successor_events.parquet",
    }
    report = {
        "experiment": "M23-59 MOT20 teacher labels for nested training only", "seq": seq,
        "teacher_only": True, "outer_deployment_input": False,
        "node_impure": int(node_label.sum()), "boundary_positive": int((boundary > 0).sum()),
        "capacity_change_points": int(capacity.sum()), "A_to_B_to_A_nodes": int(aba.sum()),
        "artifacts": artifacts,
        "source_teacher_labels": str(labels_path), "source_teacher_labels_sha256": sha256_file(labels_path),
        "relation_sources": {k: {"path": str(p), "sha256": sha256_file(p)} for k, p in relation_files.items()},
    }
    json_write(out / "manifest.json", report, refuse_existing=True)
    event("mot20_labels_unlocked", seq=seq, manifest_sha256=sha256_file(out / "manifest.json"))
    return report


def mot20_fixed_node_arrays(seq: str, with_labels: bool) -> dict[str, np.ndarray]:
    features = np.asarray(np.load(ROOT / "mot20_observable" / seq / "row_features.f16.npy", mmap_mode="r"), np.float32)
    membership = pd.read_parquet(M57_ROOT / "boundary_universe" / seq / "chunk_membership.parquet")
    x = np.zeros((len(membership), MAX_NODE_ROWS, APP_DIM + GEOM_DIM), np.float16)
    mask = np.zeros((len(membership), MAX_NODE_ROWS), np.uint8)
    for row in membership.itertuples():
        ids = np.asarray(row.row_indices, np.int64)[:MAX_NODE_ROWS]
        cid = int(row.fixed_chunk_id)
        x[cid, :len(ids)] = features[ids].astype(np.float16)
        mask[cid, :len(ids)] = 1
    result: dict[str, np.ndarray] = {"node_x": x, "node_mask": mask}
    if with_labels:
        label_root = ROOT / "mot20_labels" / seq
        for name in ["node_label", "boundary_label", "capacity_split", "aba_event"]:
            result[name] = np.load(label_root / f"{name}.npy", mmap_mode="r")
    return result


def capacity_node_arrays(seq: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = pd.read_parquet(ROOT / "mot20_observable" / seq / "rows.parquet")
    features = np.asarray(np.load(ROOT / "mot20_observable" / seq / "row_features.f16.npy", mmap_mode="r"), np.float32)
    nodes = pd.read_parquet(M57_ROOT / "capacity" / seq / "frozen_candidate_graph/nodes.parquet")
    x = np.zeros((len(nodes), MAX_NODE_ROWS, APP_DIM + GEOM_DIM), np.float16)
    mask = np.zeros((len(nodes), MAX_NODE_ROWS), np.uint8)
    by_track: dict[int, np.ndarray] = {}
    for tid, group in rows.groupby("track_id", sort=False):
        by_track[int(tid)] = group.row_index.to_numpy(np.int64)
    frames = rows.frame.to_numpy(int)
    for node in nodes.itertuples():
        ids = by_track.get(int(node.parent_tracker_id), np.zeros(0, np.int64))
        ids = ids[(frames[ids] >= int(node.first_frame)) & (frames[ids] <= int(node.last_frame))]
        ids = ids[:MAX_NODE_ROWS]
        cid = int(node.chunk_id)
        if len(ids):
            x[cid, :len(ids)] = features[ids].astype(np.float16); mask[cid, :len(ids)] = 1
    return nodes, x, mask


def mot20_relation_triplets(seq: str, cap: int = RELATION_TRAIN_CAP_PER_MOT20_SEQ) -> dict[str, np.ndarray]:
    nodes, x, mask = capacity_node_arrays(seq)
    teacher = pd.read_parquet(M57_ROOT / "capacity" / seq / "teacher_identity_flow/teacher_edge_utilities.parquet")
    rows = []
    for src, group in teacher.groupby("src_chunk", sort=True):
        positive = group[(group.teacher_same_identity_forward > 0) & (group.teacher_identity_order_delta > 0)].copy()
        if positive.empty:
            continue
        positive.sort_values(["teacher_identity_order_delta", "dst_chunk"], kind="mergesort", inplace=True)
        pos = positive.iloc[0]
        negative = group[group.teacher_same_identity_forward <= 0].copy()
        if negative.empty:
            continue
        negative.sort_values(["appearance_cos", "motion_error_min", "dst_chunk"], ascending=[False, True, True], kind="mergesort", inplace=True)
        for neg in negative.head(4).itertuples():
            rows.append((int(src), int(pos.dst_chunk), int(neg.dst_chunk), int(pos.gap)))
    if len(rows) > cap:
        ids = np.linspace(0, len(rows) - 1, cap, dtype=int); rows = [rows[int(i)] for i in np.unique(ids)]
    src = np.asarray([r[0] for r in rows], np.int64); pos = np.asarray([r[1] for r in rows], np.int64); neg = np.asarray([r[2] for r in rows], np.int64)
    return {
        "rel_src": x[src], "rel_src_mask": mask[src], "rel_pos": x[pos], "rel_pos_mask": mask[pos],
        "rel_neg": x[neg], "rel_neg_mask": mask[neg], "rel_gap": np.asarray([r[3] for r in rows], np.int16),
    }


def fit_inner_model(outer: str, validation_seq: str) -> tuple[Path, dict[str, Any]]:
    training = sorted(set(MOT20_SEQS) - {outer, validation_seq})
    if len(training) != 2:
        raise RuntimeError("inner fold must train on exactly two sequences")
    root = ROOT / "nested_loso" / outer / f"inner_valid_{validation_seq}"
    frozen_manifest = root / "model_frozen_before_validation_labels.json"
    if frozen_manifest.exists():
        data = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        return Path(data["model"]), data
    seed = 2359100 + MOT20_SEQS.index(outer) * 10 + MOT20_SEQS.index(validation_seq)
    set_determinism(seed); dev = device()
    model, external_manifest = load_frozen_external_model(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    node_data = []
    for group, seq in enumerate(training):
        d = mot20_fixed_node_arrays(seq, True); d["group"] = np.full(len(d["node_label"]), group, np.int16); node_data.append(d)
    node = {name: np.concatenate([d[name] for d in node_data]) for name in ["node_x", "node_mask", "node_label", "boundary_label", "aba_event", "group"]}
    rel_data = []
    for group, seq in enumerate(training):
        d = mot20_relation_triplets(seq); d["group"] = np.full(len(d["rel_gap"]), group, np.int16); rel_data.append(d)
    rel = {name: np.concatenate([d[name] for d in rel_data]) for name in ["rel_src", "rel_src_mask", "rel_pos", "rel_pos_mask", "rel_neg", "rel_neg_mask", "rel_gap", "group"]}
    qn = torch.ones(2, device=dev) / 2; qr = torch.ones(2, device=dev) / 2
    rng = np.random.default_rng(seed); history = []
    for epoch in range(MOT20_FINETUNE_EPOCHS):
        model.train(); losses = []
        nb = list(sample_indices(len(node["node_label"]), BATCH_NODES, rng)); rb = list(sample_indices(len(rel["group"]), BATCH_RELATIONS, rng))
        for step in range(max(len(nb), len(rb))):
            ni = nb[step % len(nb)]; ri = rb[step % len(rb)]
            x = tensor(node["node_x"][ni], dev); m = tensor(node["node_mask"][ni], dev)
            y = tensor(node["node_label"][ni], dev); by = tensor(node["boundary_label"][ni], dev)
            g = tensor(node["group"][ni], dev, torch.long)
            nl, bl, valid = model.node_and_boundary(x, m)
            nloss, qn = group_weighted(F.binary_cross_entropy_with_logits(nl, y, reduction="none"), g, qn)
            bvalid = (valid > 0) & (by >= 0) & (y[:, None] > 0)
            bloss = focal_loss(bl, by.clamp_min(0), bvalid)
            count = ((torch.sigmoid(bl) * valid).sum(1) - by.clamp_min(0).sum(1)).abs().mean()
            sx = tensor(rel["rel_src"][ri], dev); sm = tensor(rel["rel_src_mask"][ri], dev)
            px = tensor(rel["rel_pos"][ri], dev); pm = tensor(rel["rel_pos_mask"][ri], dev)
            nx = tensor(rel["rel_neg"][ri], dev); nm = tensor(rel["rel_neg_mask"][ri], dev)
            rg = tensor(rel["group"][ri], dev, torch.long)
            ps, pr = model.relation(sx, sm, px, pm); ns, nr = model.relation(sx, sm, nx, nm)
            rank_per = F.softplus(-(ps - ns)); rloss, qr = group_weighted(rank_per, rg, qr)
            risk = 0.5 * (F.binary_cross_entropy_with_logits(pr, torch.zeros_like(pr)) + F.binary_cross_entropy_with_logits(nr, torch.ones_like(nr)))
            sparsity = (torch.sigmoid(bl) * valid).mean()
            total = nloss + bloss + 0.2 * count + rloss + 0.5 * rank_per.mean() + 1.5 * risk + 0.02 * sparsity
            optimizer.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(total.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    # Calibration uses only the two inner-training sequences.
    model.eval(); train_node_y = []; train_node_s = []; train_bound_y = []; train_bound_s = []
    with torch.no_grad():
        for d in node_data:
            for start in range(0, len(d["node_label"]), 1024):
                ids = np.arange(start, min(start + 1024, len(d["node_label"])))
                nl, bl, valid = model.node_and_boundary(tensor(d["node_x"][ids], dev), tensor(d["node_mask"][ids], dev))
                ny = np.asarray(d["node_label"][ids]); by = np.asarray(d["boundary_label"][ids])
                train_node_y.append(ny); train_node_s.append(torch.sigmoid(nl).cpu().numpy())
                keep = (ny[:, None] > 0) & (by >= 0) & (valid.cpu().numpy() > 0)
                train_bound_y.append(by[keep]); train_bound_s.append(torch.sigmoid(bl).cpu().numpy()[keep])
    node_y = np.concatenate(train_node_y); node_s = np.concatenate(train_node_s)
    bound_y = np.concatenate(train_bound_y); bound_s = np.concatenate(train_bound_s)
    node_threshold = threshold_at_precision(node_y, node_s, 0.95)
    boundary_threshold = threshold_at_precision(bound_y, bound_s, 0.90)
    root.mkdir(parents=True, exist_ok=True); model_path = root / "model.pt"
    torch.save({"model": model.state_dict(), "seed": seed, "training_sequences": training,
                "outer": outer, "validation_sequence": validation_seq}, model_path)
    history_path = root / "training_history.json"; json_write(history_path, history)
    manifest = {
        "experiment": "M23-59 inner model frozen before validation labels", "outer": outer,
        "validation_sequence": validation_seq, "training_sequences": training,
        "validation_labels_opened": False, "outer_gt_read": False, "outer_teacher_action_read": False,
        "external_checkpoint_sha256": external_manifest["checkpoint_sha256"],
        "model": str(model_path), "model_sha256": sha256_file(model_path),
        "identity_scaler": {"type": "none/fixed raw normalized appearance and predeclared geometry", "sha256": hashlib.sha256(b"M23-59-identity-scaler-v1").hexdigest()},
        "node_threshold": node_threshold, "boundary_threshold": boundary_threshold,
        "seed": seed, "epochs": MOT20_FINETUNE_EPOCHS, "learning_rate": FINETUNE_LR,
        "history": str(history_path), "history_sha256": sha256_file(history_path),
    }
    json_write(frozen_manifest, manifest, refuse_existing=True)
    event("inner_model_frozen_before_validation_labels", outer=outer, validation=validation_seq,
          manifest_sha256=sha256_file(frozen_manifest))
    return model_path, manifest


def load_inner_model(path: Path, dev: torch.device) -> HierarchicalRelationEncoder:
    state = torch.load(path, map_location="cpu")
    model = HierarchicalRelationEncoder().to(dev)
    model.load_state_dict(state["model"]); model.eval()
    return model


def score_fixed_nodes(model: HierarchicalRelationEncoder, data: dict[str, np.ndarray], dev: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = np.zeros(len(data["node_mask"]), np.float32)
    boundary = np.zeros((len(data["node_mask"]), MAX_NODE_ROWS - 1), np.float32)
    valid = np.zeros_like(boundary, np.uint8)
    with torch.no_grad():
        for start in range(0, len(node), 1024):
            ids = np.arange(start, min(start + 1024, len(node)))
            nl, bl, bv = model.node_and_boundary(tensor(data["node_x"][ids], dev), tensor(data["node_mask"][ids], dev))
            node[ids] = torch.sigmoid(nl).cpu().numpy()
            boundary[ids] = torch.sigmoid(bl).cpu().numpy()
            valid[ids] = (bv.cpu().numpy() > 0).astype(np.uint8)
    return node, boundary, valid


def boundary_offset_report(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    offsets = []; exact = 0; events = 0
    for y, s in zip(labels, scores):
        true = np.flatnonzero(y > 0)
        if not len(true):
            continue
        k = len(true); pred = np.argsort(-s)[:k]
        for t in true:
            nearest = int(pred[np.argmin(np.abs(pred - t))])
            off = nearest - int(t); offsets.append(off); exact += int(off == 0); events += 1
    if not offsets:
        return {"events": 0, "exact_boundary_rate": 0.0, "mean_signed_offset_rows": None,
                "median_absolute_offset_rows": None, "p90_absolute_offset_rows": None}
    arr = np.asarray(offsets, float)
    return {"events": events, "exact_boundary_rate": exact / events,
            "mean_signed_offset_rows": float(arr.mean()),
            "median_absolute_offset_rows": float(np.median(np.abs(arr))),
            "p90_absolute_offset_rows": float(np.quantile(np.abs(arr), 0.90))}


def pooled_capacity_nodes(model: HierarchicalRelationEncoder, seq: str, dev: torch.device) -> tuple[pd.DataFrame, np.ndarray]:
    nodes, x, mask = capacity_node_arrays(seq)
    pooled = np.zeros((len(nodes), 4 * HIDDEN), np.float32)
    with torch.no_grad():
        for start in range(0, len(nodes), 1024):
            ids = np.arange(start, min(start + 1024, len(nodes)))
            tx = tensor(x[ids], dev); tm = tensor(mask[ids], dev)
            pooled[ids] = model.pool(model.encode(tx, tm), tm).cpu().numpy()
    return nodes, pooled


def relation_from_pooled(model: HierarchicalRelationEncoder, src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1)
    return model.cross_relation_head(pair).squeeze(-1), model.catastrophic_risk_head(pair).squeeze(-1)


def score_capacity_edges(model: HierarchicalRelationEncoder, seq: str, dev: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, pooled = pooled_capacity_nodes(model, seq, dev)
    edge_path = M57_ROOT / "capacity" / seq / "teacher_identity_flow/teacher_edge_utilities.parquet"
    edges = pd.read_parquet(edge_path)
    score = np.zeros(len(edges), np.float32); risk = np.zeros(len(edges), np.float32)
    src_all = edges.src_chunk.to_numpy(np.int64); dst_all = edges.dst_chunk.to_numpy(np.int64)
    with torch.no_grad():
        for start in range(0, len(edges), 16384):
            ids = np.arange(start, min(start + 16384, len(edges)))
            src = tensor(pooled[src_all[ids]], dev); dst = tensor(pooled[dst_all[ids]], dev)
            s, r = relation_from_pooled(model, src, dst)
            score[ids] = s.cpu().numpy(); risk[ids] = torch.sigmoid(r).cpu().numpy()
    result = edges[["src_chunk", "dst_chunk", "parent_edge", "same_source", "gap", "appearance_cos",
                    "teacher_same_identity_forward", "teacher_identity_order_delta"]].copy()
    result["relation_score"] = score; result["catastrophic_risk"] = risk
    events = pd.read_parquet(M57_ROOT / "capacity" / seq / "postfreeze_audit/successor_events.parquet")
    return result, events


def relation_metrics(scored: pd.DataFrame, events: pd.DataFrame) -> dict[str, Any]:
    reciprocal = []; at1 = []; at8 = []; at32 = []; correct_cross = 0; selected_cross = 0
    catastrophic = 0; selected = 0
    event_lookup = {(int(r.src_chunk), int(r.dst_chunk)): r for r in events.itertuples() if int(r.candidate_present) > 0}
    for src, group in scored.groupby("src_chunk", sort=True):
        group = group.sort_values(["relation_score", "dst_chunk"], ascending=[False, True], kind="mergesort")
        correct = [i for i, row in enumerate(group.itertuples(), start=1)
                   if (int(row.src_chunk), int(row.dst_chunk)) in event_lookup]
        if correct:
            rank = min(correct); reciprocal.append(1.0 / rank); at1.append(rank <= 1); at8.append(rank <= 8); at32.append(rank <= 32)
        top = group.iloc[0]
        selected += 1
        is_correct = (int(top.src_chunk), int(top.dst_chunk)) in event_lookup
        catastrophic += int(not is_correct)
        expected = event_lookup.get((int(top.src_chunk), int(top.dst_chunk)))
        if int(top.parent_edge) == 0:
            selected_cross += 1; correct_cross += int(expected is not None)
    return {
        "successor_events_with_candidate": len(reciprocal),
        "successor_MRR": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "successor_R_at_1": float(np.mean(at1)) if at1 else 0.0,
        "successor_R_at_8": float(np.mean(at8)) if at8 else 0.0,
        "successor_R_at_32": float(np.mean(at32)) if at32 else 0.0,
        "repair_precision_top_cross": correct_cross / max(selected_cross, 1),
        "catastrophic_false_link_rate_top1": catastrophic / max(selected, 1),
        "top1_sources": selected,
    }


def evaluate_inner_validation(outer: str, validation_seq: str, model_path: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / "nested_loso" / outer / f"inner_valid_{validation_seq}"
    report_path = root / "representation_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    # The model/scaler/threshold manifest already exists before this read.
    labels_manifest = ROOT / "mot20_labels" / validation_seq / "manifest.json"
    if not labels_manifest.exists():
        raise FileNotFoundError(f"validation labels not unlocked for {validation_seq}")
    dev = device(); model = load_inner_model(model_path, dev)
    data = mot20_fixed_node_arrays(validation_seq, True)
    node_score, boundary_score, valid = score_fixed_nodes(model, data, dev)
    node_y = np.asarray(data["node_label"], int)
    boundary_y = np.asarray(data["boundary_label"], int)
    node_report = summarize_binary(node_y, node_score, float(frozen["node_threshold"]))
    conditional = (node_y[:, None] > 0) & (boundary_y >= 0) & (valid > 0)
    by = boundary_y[conditional]; bs = boundary_score[conditional]
    boundary_report = summarize_binary(by, bs, float(frozen["boundary_threshold"]))
    predicted_cut = (node_score[:, None] >= float(frozen["node_threshold"])) & (boundary_score >= float(frozen["boundary_threshold"])) & (valid > 0)
    pure = node_y == 0
    pure_false = float(predicted_cut[pure].any(axis=1).mean()) if pure.any() else 0.0
    aba_hits = []
    for i in np.flatnonzero(np.asarray(data["aba_event"], int) > 0):
        true = set(np.flatnonzero(boundary_y[i] > 0).tolist())
        top = set(np.argsort(-boundary_score[i])[:len(true)].tolist())
        aba_hits.append(int(true.issubset(top)))
    scored, events = score_capacity_edges(model, validation_seq, dev)
    rel_report = relation_metrics(scored, events)
    report = {
        "experiment": "M23-59 strict inner representation validation", "outer": outer,
        "validation_sequence": validation_seq, "training_sequences": frozen["training_sequences"],
        "validation_labels_opened_after_model_freeze": True,
        "outer_gt_read": False, "outer_teacher_action_read": False,
        "node_impurity": node_report, "conditional_change_point": boundary_report,
        "pure_node_false_split_rate": pure_false,
        "boundary_offset": boundary_offset_report(boundary_y[node_y > 0], boundary_score[node_y > 0]),
        "A_to_B_to_A_event_recall": float(np.mean(aba_hits)) if aba_hits else 0.0,
        "cross_relation": rel_report,
        "model_sha256": sha256_file(model_path),
        "validation_label_manifest_sha256": sha256_file(labels_manifest),
        "runtime_resources": {"peak_rss_mb": peak_rss_mb(), "peak_gpu_memory_mb": peak_gpu_mb()},
    }
    json_write(report_path, report, refuse_existing=True)
    event("inner_representation_completed", outer=outer, validation=validation_seq,
          report_sha256=sha256_file(report_path))
    return report


def run_outer_representation_gate(outer: str) -> dict[str, Any]:
    verify_freeze(); verify_all_mot20_observable_frozen()
    if outer not in MOT20_SEQS:
        raise ValueError(outer)
    root = ROOT / "nested_loso" / outer
    summary_path = root / "representation_gate.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    training_sequences = sorted(set(MOT20_SEQS) - {outer})
    folds = []
    for validation in training_sequences:
        model_path, frozen = fit_inner_model(outer, validation)
        folds.append(evaluate_inner_validation(outer, validation, model_path, frozen))
    boundary_pr = [float(f["conditional_change_point"]["pr_auc"] or 0.0) for f in folds]
    precision_actual = [float(f["conditional_change_point"]["precision_at_actual_count"]) for f in folds]
    recall95 = [float(f["conditional_change_point"]["recall_at_95_precision"]) for f in folds]
    false_split = [float(f["pure_node_false_split_rate"]) for f in folds]
    checks = {
        "mean_boundary_pr_auc_at_least_0p283": float(np.mean(boundary_pr)) >= BOUNDARY_PR_GATE,
        "mean_precision_at_actual_at_least_0p35": float(np.mean(precision_actual)) >= BOUNDARY_PRECISION_ACTUAL_GATE,
        "mean_recall_at_95_precision_at_least_0p05": float(np.mean(recall95)) >= BOUNDARY_RECALL95_GATE,
        "each_precision_at_actual_at_least_0p20": min(precision_actual) >= BOUNDARY_EACH_PRECISION_ACTUAL_GATE,
        "each_pure_node_false_split_at_most_0p002": max(false_split) <= PURE_FALSE_SPLIT_MAX,
    }
    passed = all(checks.values())
    summary = {
        "experiment": "M23-59 outer representation gate", "outer": outer,
        "outer_gt_read": False, "outer_teacher_action_read": False,
        "external_checkpoint_sha256": json.loads((ROOT / "external_pretraining/frozen_checkpoint_manifest.json").read_text())["checkpoint_sha256"],
        "inner_validation_sequences": training_sequences,
        "aggregate": {
            "mean_boundary_pr_auc": float(np.mean(boundary_pr)),
            "mean_precision_at_actual_boundary_count": float(np.mean(precision_actual)),
            "mean_recall_at_95_precision": float(np.mean(recall95)),
            "max_pure_node_false_split_rate": float(max(false_split)),
            "min_fold_precision_at_actual": float(min(precision_actual)),
            "mean_node_impurity_pr_auc": float(np.mean([float(f["node_impurity"]["pr_auc"] or 0.0) for f in folds])),
            "mean_successor_MRR": float(np.mean([f["cross_relation"]["successor_MRR"] for f in folds])),
            "mean_successor_R_at_1": float(np.mean([f["cross_relation"]["successor_R_at_1"] for f in folds])),
            "mean_catastrophic_false_link_rate": float(np.mean([f["cross_relation"]["catastrophic_false_link_rate_top1"] for f in folds])),
        },
        "fixed_gate": checks, "passed": passed,
        "folds": {f["validation_sequence"]: f for f in folds},
        "next_action": "run P1/P2 inner exact trackers" if passed else "freeze P0 without learned inner tracker",
    }
    root.mkdir(parents=True, exist_ok=True); json_write(summary_path, summary, refuse_existing=True)
    event("outer_representation_gate_completed", outer=outer, passed=passed, report_sha256=sha256_file(summary_path))
    if not passed:
        freeze_outer_p0_manifest(outer, summary)
    return summary


def freeze_outer_p0_manifest(outer: str, representation: dict[str, Any]) -> dict[str, Any]:
    out = ROOT / "outer_policies" / outer
    path = out / "outer_policy_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    external = json.loads((ROOT / "external_pretraining/frozen_checkpoint_manifest.json").read_text(encoding="utf-8"))
    tracker = mot20_source_tracker(outer)
    candidate_manifest = M57_ROOT / "boundary_universe" / outer / "freeze_manifest.json"
    manifest = {
        "experiment": "M23-59 strict outer policy", "outer": outer,
        "selected_policy": "P0", "reason": "representation_gate_failed",
        "external_supervision": True, "strict_sequence_loso": True, "deployable": True,
        "external_checkpoint_sha256": external["checkpoint_sha256"],
        "fine_tuned_model_sha256": None,
        "scaler_sha256": hashlib.sha256(b"M23-59-identity-scaler-v1").hexdigest(),
        "risk_calibration_sha256": None,
        "boundary_policy": "no-op", "flow_policy": "byte-exact M23-46",
        "candidate_graph_sha256": sha256_file(candidate_manifest),
        "outer_gt_read": False, "outer_teacher_action_read": False,
        "tracker": {"path": str(tracker), "sha256": sha256_file(tracker), "byte_exact_m23_46": True},
        "representation_gate_sha256": sha256_file(ROOT / "nested_loso" / outer / "representation_gate.json"),
        "representation_gate": representation["aggregate"],
        "P1": {"risk_ucb_limit": P1_RISK, "evaluated": False},
        "P2": {"risk_ucb_limit": P2_RISK, "evaluated": False},
        "inner_trackeval_runs": 0,
    }
    out.mkdir(parents=True, exist_ok=True); json_write(path, manifest, refuse_existing=True)
    event("outer_policy_frozen", outer=outer, policy="P0", manifest_sha256=sha256_file(path))
    return manifest


def boundary_risk_scores(model: HierarchicalRelationEncoder, node_x: np.ndarray, node_mask: np.ndarray,
                         node_score: np.ndarray, node_threshold: float, dev: torch.device) -> np.ndarray:
    risks = np.ones((len(node_x), MAX_NODE_ROWS - 1), np.float32)
    candidates = []
    for i in np.flatnonzero(node_score >= node_threshold):
        length = int(np.asarray(node_mask[i]).sum())
        for pos in range(1, length):
            candidates.append((int(i), pos))
    with torch.no_grad():
        for start in range(0, len(candidates), 1024):
            batch = candidates[start:start + 1024]
            lx = []; lm = []; rx = []; rm = []
            for i, pos in batch:
                left = node_x[i, max(0, pos - SEG_ROWS):pos]
                right = node_x[i, pos:min(int(node_mask[i].sum()), pos + SEG_ROWS)]
                a = np.zeros((SEG_ROWS, APP_DIM + GEOM_DIM), np.float32); am = np.zeros(SEG_ROWS, np.float32)
                b = np.zeros_like(a); bm = np.zeros_like(am)
                a[-len(left):] = left; am[-len(left):] = 1
                b[:len(right)] = right; bm[:len(right)] = 1
                lx.append(a); lm.append(am); rx.append(b); rm.append(bm)
            _, r = model.relation(tensor(np.asarray(lx), dev), tensor(np.asarray(lm), dev),
                                  tensor(np.asarray(rx), dev), tensor(np.asarray(rm), dev))
            for (i, pos), value in zip(batch, torch.sigmoid(r).cpu().numpy()):
                risks[i, pos - 1] = float(value)
    return risks


def path_cover_from_utility(nodes: pd.DataFrame, edges: pd.DataFrame, utility: np.ndarray) -> pd.DataFrame:
    positive = edges[np.asarray(utility) > 0].copy()
    if positive.empty:
        return positive
    positive["learned_utility"] = np.asarray(utility)[np.asarray(utility) > 0]
    src = positive.src_chunk.to_numpy(int); dst = positive.dst_chunk.to_numpy(int)
    val = positive.learned_utility.to_numpy(float); offset = float(val.max()) + 1.0
    rows_idx = np.concatenate([src, np.arange(len(nodes), dtype=int)])
    cols_idx = np.concatenate([dst, len(nodes) + np.arange(len(nodes), dtype=int)])
    costs = np.concatenate([offset - val, np.full(len(nodes), offset, float)])
    matrix = coo_matrix((costs, (rows_idx, cols_idx)), shape=(len(nodes), 2 * len(nodes))).tocsr()
    mr, mc = min_weight_full_bipartite_matching(matrix)
    real = mc < len(nodes); pairs = set(zip(mr[real].tolist(), mc[real].tolist()))
    selected = positive[[((int(a), int(b)) in pairs) for a, b in zip(positive.src_chunk, positive.dst_chunk)]].copy()
    if selected.src_chunk.duplicated().any() or selected.dst_chunk.duplicated().any():
        raise RuntimeError("learned path cover is not one-to-one")
    first = nodes.first_frame.to_numpy(int); last = nodes.last_frame.to_numpy(int)
    if len(selected) and np.any(first[selected.dst_chunk.to_numpy(int)] <= last[selected.src_chunk.to_numpy(int)]):
        raise RuntimeError("learned path cover is not time-forward")
    return selected


def build_learned_tracker(seq: str, model_path: Path, thresholds: dict[str, float], policy: str,
                          output_root: Path, *, teacher_audit: bool) -> dict[str, Any]:
    if policy not in {"P1", "P2"}:
        raise ValueError(policy)
    if (output_root / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite learned tracker {output_root}")
    risk_limit = P1_RISK if policy == "P1" else P2_RISK
    dev = device(); model = load_inner_model(model_path, dev)
    fixed = mot20_fixed_node_arrays(seq, with_labels=teacher_audit)
    node_score, boundary_score, valid = score_fixed_nodes(model, fixed, dev)
    boundary_risk = boundary_risk_scores(model, fixed["node_x"], fixed["node_mask"], node_score,
                                         float(thresholds["node_threshold"]), dev)
    cuts: dict[int, list[int]] = {}
    for cid in range(len(node_score)):
        if node_score[cid] < float(thresholds["node_threshold"]):
            continue
        positions = []
        for pos in range(1, MAX_NODE_ROWS):
            if valid[cid, pos - 1] and boundary_score[cid, pos - 1] >= float(thresholds["boundary_threshold"]) and boundary_risk[cid, pos - 1] <= risk_limit:
                positions.append(pos)
        if positions:
            cuts[cid] = positions
    m57 = load_module(f"m23_59_m57_infer_{seq[-2:]}_{policy}", "scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
    m53 = load_module(f"m23_59_m53_infer_{seq[-2:]}_{policy}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    m53b = load_module(f"m23_59_m53b_infer_{seq[-2:]}_{policy}", "scripts/m23_research/m23_53b_build_adaptive_micrograph.py")
    m55 = load_module(f"m23_59_m55_infer_{seq[-2:]}_{policy}", "scripts/m23_research/m23_55_stratified_gap_candidate_expansion.py")
    m10 = load_module(f"m23_59_m10_infer_{seq[-2:]}_{policy}", "scripts/m23_research/m23_10_build_micrograph.py")
    (
        _, _, _, source_path, baseline_path, source_rows, fixed_nodes, chunks,
        parent_ids, crowd, mapped, _, row_embeddings,
    ) = m57.prepare_observable_rows(seq)
    nodes, prototypes = m53b.build_adaptive_nodes(
        source_rows=source_rows, fixed_nodes=fixed_nodes, chunk_rows=chunks,
        selected_boundaries=cuts, row_embeddings=row_embeddings, mapped=mapped,
        parent_ids=parent_ids, crowd_density=crowd,
    )
    graph_root = output_root / "inference_graph"; seq_graph = graph_root / seq; seq_graph.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(seq_graph / "microtracklets.parquet", index=False)
    np.save(seq_graph / "prototypes.f16.npy", prototypes.astype(np.float16))
    legacy, _ = m53b.build_candidate_edges(nodes=nodes, prototypes=prototypes, max_gap=600, appearance_bank_k=32, motion_bank_k=8)
    legacy = m57.add_legacy_columns(legacy)
    descriptor, descriptor_report = m55.build_descriptors(seq, nodes, SOURCE_PARENT, graph_root, m53, m10)
    outgoing, out_report = m55.generate_direction_pool(nodes, descriptor, "out", 256, 256, dev)
    incoming, in_report = m55.generate_direction_pool(nodes, descriptor, "in", 256, 256, dev)
    out_flow = outgoing[outgoing["rank"] <= 32][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "out_rank"})
    in_flow = incoming[incoming["rank"] <= 32][["src_chunk", "dst_chunk", "gap_bucket", "rank"]].rename(columns={"rank": "in_rank"})
    pairs = out_flow.merge(in_flow, on=["src_chunk", "dst_chunk", "gap_bucket"], how="outer")
    pairs["out_rank"] = pairs.out_rank.fillna(np.inf); pairs["in_rank"] = pairs.in_rank.fillna(np.inf)
    pairs.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True)
    expanded = m55.vectorized_edge_features(nodes, pairs, descriptor)
    edges = pd.concat([legacy, expanded], ignore_index=True, sort=False)
    edges.sort_values(["parent_edge", "src_chunk", "dst_chunk"], ascending=[False, True, True], kind="mergesort", inplace=True)
    edges.drop_duplicates(["src_chunk", "dst_chunk"], keep="first", inplace=True); edges.reset_index(drop=True, inplace=True)
    if audit_columns(edges.columns):
        raise RuntimeError("forbidden field in learned inference graph")
    # Build new-node tensors from the already frozen row feature tensor.
    row_table = pd.read_parquet(ROOT / "mot20_observable" / seq / "rows.parquet")
    row_feat = np.asarray(np.load(ROOT / "mot20_observable" / seq / "row_features.f16.npy", mmap_mode="r"), np.float32)
    by_track = {int(t): g.row_index.to_numpy(np.int64) for t, g in row_table.groupby("track_id", sort=False)}
    frame_array = row_table.frame.to_numpy(int)
    nx = np.zeros((len(nodes), MAX_NODE_ROWS, APP_DIM + GEOM_DIM), np.float16); nm = np.zeros((len(nodes), MAX_NODE_ROWS), np.uint8)
    for n in nodes.itertuples():
        ids = by_track.get(int(n.parent_tracker_id), np.zeros(0, np.int64))
        ids = ids[(frame_array[ids] >= int(n.first_frame)) & (frame_array[ids] <= int(n.last_frame))][:MAX_NODE_ROWS]
        nx[int(n.chunk_id), :len(ids)] = row_feat[ids].astype(np.float16); nm[int(n.chunk_id), :len(ids)] = 1
    pooled = np.zeros((len(nodes), 4 * HIDDEN), np.float32)
    with torch.no_grad():
        for start in range(0, len(nodes), 1024):
            ids = np.arange(start, min(start + 1024, len(nodes)))
            tx = tensor(nx[ids], dev); tm = tensor(nm[ids], dev)
            pooled[ids] = model.pool(model.encode(tx, tm), tm).cpu().numpy()
    edge_score = np.zeros(len(edges), np.float32); edge_risk = np.ones(len(edges), np.float32)
    src = edges.src_chunk.to_numpy(np.int64); dst = edges.dst_chunk.to_numpy(np.int64)
    with torch.no_grad():
        for start in range(0, len(edges), 16384):
            ids = np.arange(start, min(start + 16384, len(edges)))
            s, r = relation_from_pooled(model, tensor(pooled[src[ids]], dev), tensor(pooled[dst[ids]], dev))
            edge_score[ids] = s.cpu().numpy(); edge_risk[ids] = torch.sigmoid(r).cpu().numpy()
    parent = edges.parent_edge.to_numpy(float)
    utility = edge_score + 0.25 * parent - 0.25 * (1.0 - parent)
    utility[edge_risk > risk_limit] = -np.inf
    selected = path_cover_from_utility(nodes, edges, utility)
    edges["relation_score"] = edge_score; edges["catastrophic_risk"] = edge_risk; edges["learned_utility"] = utility
    graph_dir = output_root / "frozen_inference_graph"; graph_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = graph_dir / "nodes.parquet"; edges_path = graph_dir / "edges.parquet"; selected_path = graph_dir / "selected_edges.parquet"
    nodes.to_parquet(nodes_path, index=False); edges.to_parquet(edges_path, index=False); selected.to_parquet(selected_path, index=False)
    tracker = output_root / "track_results" / f"{seq}.txt"
    tracker_report = m53.write_tracker(seq, source_path, nodes, selected, tracker)
    payload_ok = detector_payload(source_path) == detector_payload(tracker)
    if not payload_ok:
        raise RuntimeError("detection payload changed")
    audit = None
    if teacher_audit:
        cap = np.asarray(fixed["capacity_split"], int)
        false_cuts = sum(int(cap[cid, pos - 1] <= 0) for cid, positions in cuts.items() for pos in positions)
        cut_count = sum(len(v) for v in cuts.values())
        # Teacher utility audit is opened only for an inner-validation sequence.
        freeze_manifest = {
            "frozen_artifacts": {"nodes": str(nodes_path), "edges": str(edges_path)},
            "nodes_sha256": sha256_file(nodes_path), "edges_sha256": sha256_file(edges_path),
        }
        audit_root = output_root / "teacher_audit"
        m53.build_teacher_utilities(seq=seq, source_parent_root=SOURCE_PARENT, output_root=audit_root, freeze_manifest=freeze_manifest)
        teacher_edges = pd.read_parquet(audit_root / "teacher_identity_flow/teacher_edge_utilities.parquet")
        chosen = selected.merge(teacher_edges[["src_chunk", "dst_chunk", "teacher_same_identity_forward"]], on=["src_chunk", "dst_chunk"], how="left")
        wrong_links = int((chosen.teacher_same_identity_forward.fillna(0) <= 0).sum())
        link_count = len(chosen); errors = false_cuts + wrong_links; actions = cut_count + link_count
        audit = {"false_cuts": false_cuts, "selected_cuts": cut_count, "wrong_links": wrong_links,
                 "selected_links": link_count, "catastrophic_errors": errors, "identity_actions": actions,
                 "catastrophic_risk_ucb": clopper_pearson_upper(errors, actions)}
    report = {
        "experiment": "M23-59 hierarchical learned tracker", "seq": seq, "policy": policy,
        "external_supervision": True, "risk_limit": risk_limit,
        "selected_cuts": sum(len(v) for v in cuts.values()), "nodes_after_split": len(nodes),
        "candidate_edges": len(edges), "selected_links": len(selected),
        "descriptor_report": descriptor_report, "ranking_reports": {"outgoing": out_report, "incoming": in_report},
        "integrity": {"detection_payload_unchanged": payload_ok, "one_to_one": not selected.src_chunk.duplicated().any() and not selected.dst_chunk.duplicated().any(), "time_forward": True},
        "tracker": {**tracker_report, "path": str(tracker), "sha256": sha256_file(tracker)},
        "graph": {"nodes_sha256": sha256_file(nodes_path), "edges_sha256": sha256_file(edges_path), "selected_sha256": sha256_file(selected_path)},
        "teacher_audit": audit,
        "resources": {"peak_rss_mb": peak_rss_mb(), "peak_gpu_memory_mb": peak_gpu_mb()},
    }
    json_write(output_root / "report.json", report, refuse_existing=True)
    return report


def run_inner_exact_gate(outer: str) -> dict[str, Any]:
    rep = json.loads((ROOT / "nested_loso" / outer / "representation_gate.json").read_text(encoding="utf-8"))
    if not rep["passed"]:
        raise RuntimeError(f"{outer}: representation gate failed; learned inner tracker prohibited")
    result_path = ROOT / "nested_loso" / outer / "inner_exact_gate.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    baseline = json.loads(Path("outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/report.json").read_text(encoding="utf-8"))["folds"]
    policies = {"P1": [], "P2": []}; trackeval_runs = 0
    for validation in sorted(set(MOT20_SEQS) - {outer}):
        inner_root = ROOT / "nested_loso" / outer / f"inner_valid_{validation}"
        frozen = json.loads((inner_root / "model_frozen_before_validation_labels.json").read_text(encoding="utf-8"))
        for policy in ["P1", "P2"]:
            run_root = inner_root / f"tracker_{policy}"
            report = build_learned_tracker(validation, Path(frozen["model"]), frozen, policy, run_root, teacher_audit=True)
            m53 = load_module(f"m23_59_m53_eval_{outer[-2:]}_{validation[-2:]}_{policy}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
            official = m53.run_official_trackeval(seq=validation, output_root=run_root, tracker_name=f"m23_59_{outer[-2:]}_{validation[-2:]}_{policy}")
            trackeval_runs += 1
            delta = float(official["HOTA"]) - float(baseline[validation]["HOTA"])
            risk_pass = bool(report["teacher_audit"]["catastrophic_risk_ucb"] <= (P1_RISK if policy == "P1" else P2_RISK))
            policies[policy].append({"validation_sequence": validation, "official_trackeval": official,
                                     "delta_HOTA": delta, "risk_pass": risk_pass,
                                     "tracker_report_sha256": sha256_file(run_root / "report.json")})
    eligible = {}
    for policy, folds in policies.items():
        deltas = [f["delta_HOTA"] for f in folds]
        eligible[policy] = {
            "each_delta_at_least_0p05": min(deltas) >= INNER_EACH_DELTA_GATE,
            "mean_delta_at_least_0p20": float(np.mean(deltas)) >= INNER_MEAN_DELTA_GATE,
            "all_risk_pass": all(f["risk_pass"] for f in folds),
            "worst_delta_HOTA": float(min(deltas)), "mean_delta_HOTA": float(np.mean(deltas)),
            "identity_edits": int(sum(json.loads((ROOT / "nested_loso" / outer / f"inner_valid_{f['validation_sequence']}" / f"tracker_{policy}/report.json").read_text())["selected_cuts"] for f in folds)),
        }
        eligible[policy]["eligible"] = eligible[policy]["each_delta_at_least_0p05"] and eligible[policy]["mean_delta_at_least_0p20"] and eligible[policy]["all_risk_pass"]
    candidates = [p for p in ["P1", "P2"] if eligible[p]["eligible"]]
    if candidates:
        candidates.sort(key=lambda p: (-eligible[p]["worst_delta_HOTA"], -eligible[p]["mean_delta_HOTA"], eligible[p]["identity_edits"], 0 if p == "P1" else 1))
        selected = candidates[0]
    else:
        selected = "P0"
    summary = {"experiment": "M23-59 inner exact TrackEval gate", "outer": outer,
               "representation_gate_passed": True, "policies": policies, "eligibility": eligible,
               "selected_policy": selected, "trackeval_runs": trackeval_runs,
               "tie_break": ["max worst-fold delta HOTA", "max mean delta HOTA", "fewest identity edits", "prefer P1"]}
    json_write(result_path, summary, refuse_existing=True)
    event("inner_exact_gate_completed", outer=outer, selected_policy=selected, report_sha256=sha256_file(result_path))
    return summary


def fit_outer_final_model(outer: str) -> dict[str, Any]:
    training = sorted(set(MOT20_SEQS) - {outer})
    root = ROOT / "outer_models" / outer
    manifest_path = root / "model_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    exact = json.loads((ROOT / "nested_loso" / outer / "inner_exact_gate.json").read_text(encoding="utf-8"))
    policy = exact["selected_policy"]
    if policy == "P0":
        raise RuntimeError("P0 does not train an outer model")
    seed = 2359200 + MOT20_SEQS.index(outer)
    set_determinism(seed); dev = device(); model, ext = load_frozen_external_model(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    nodes = []
    for group, seq in enumerate(training):
        d = mot20_fixed_node_arrays(seq, True); d["group"] = np.full(len(d["node_label"]), group, np.int16); nodes.append(d)
    node = {name: np.concatenate([d[name] for d in nodes]) for name in ["node_x", "node_mask", "node_label", "boundary_label", "group"]}
    relations = []
    for group, seq in enumerate(training):
        d = mot20_relation_triplets(seq); d["group"] = np.full(len(d["rel_gap"]), group, np.int16); relations.append(d)
    rel = {name: np.concatenate([d[name] for d in relations]) for name in ["rel_src", "rel_src_mask", "rel_pos", "rel_pos_mask", "rel_neg", "rel_neg_mask", "group"]}
    qn = torch.ones(3, device=dev) / 3; qr = torch.ones(3, device=dev) / 3
    rng = np.random.default_rng(seed); history = []
    for epoch in range(MOT20_FINETUNE_EPOCHS):
        model.train(); losses = []
        nb = list(sample_indices(len(node["node_label"]), BATCH_NODES, rng)); rb = list(sample_indices(len(rel["group"]), BATCH_RELATIONS, rng))
        for step in range(max(len(nb), len(rb))):
            ni = nb[step % len(nb)]; ri = rb[step % len(rb)]
            x = tensor(node["node_x"][ni], dev); m = tensor(node["node_mask"][ni], dev)
            y = tensor(node["node_label"][ni], dev); by = tensor(node["boundary_label"][ni], dev); g = tensor(node["group"][ni], dev, torch.long)
            nl, bl, valid = model.node_and_boundary(x, m)
            nloss, qn = group_weighted(F.binary_cross_entropy_with_logits(nl, y, reduction="none"), g, qn)
            bvalid = (valid > 0) & (by >= 0) & (y[:, None] > 0)
            bloss = focal_loss(bl, by.clamp_min(0), bvalid)
            count = ((torch.sigmoid(bl) * valid).sum(1) - by.clamp_min(0).sum(1)).abs().mean()
            sx = tensor(rel["rel_src"][ri], dev); sm = tensor(rel["rel_src_mask"][ri], dev)
            px = tensor(rel["rel_pos"][ri], dev); pm = tensor(rel["rel_pos_mask"][ri], dev)
            nx = tensor(rel["rel_neg"][ri], dev); nm = tensor(rel["rel_neg_mask"][ri], dev); rg = tensor(rel["group"][ri], dev, torch.long)
            ps, pr = model.relation(sx, sm, px, pm); ns, nr = model.relation(sx, sm, nx, nm)
            rank = F.softplus(-(ps - ns)); rloss, qr = group_weighted(rank, rg, qr)
            risk = 0.5 * (F.binary_cross_entropy_with_logits(pr, torch.zeros_like(pr)) + F.binary_cross_entropy_with_logits(nr, torch.ones_like(nr)))
            total = nloss + bloss + 0.2 * count + rloss + 0.5 * rank.mean() + 1.5 * risk + 0.02 * (torch.sigmoid(bl) * valid).mean()
            optimizer.zero_grad(set_to_none=True); total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(total.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_loss": float(np.mean(losses))})
    threshold_rows = [json.loads((ROOT / "nested_loso" / outer / f"inner_valid_{v}/model_frozen_before_validation_labels.json").read_text()) for v in training]
    node_threshold = float(np.median([r["node_threshold"] for r in threshold_rows]))
    boundary_threshold = float(np.median([r["boundary_threshold"] for r in threshold_rows]))
    root.mkdir(parents=True, exist_ok=True); model_path = root / "model.pt"; history_path = root / "history.json"
    torch.save({"model": model.state_dict(), "seed": seed, "outer": outer, "training_sequences": training}, model_path)
    json_write(history_path, history)
    manifest = {
        "experiment": "M23-59 final outer-training model", "outer": outer, "training_sequences": training,
        "outer_gt_read": False, "outer_teacher_action_read": False, "external_supervision": True,
        "selected_policy": policy, "external_checkpoint_sha256": ext["checkpoint_sha256"],
        "model": str(model_path), "model_sha256": sha256_file(model_path),
        "scaler_sha256": hashlib.sha256(b"M23-59-identity-scaler-v1").hexdigest(),
        "risk_calibration_sha256": hashlib.sha256(f"M23-59-fixed-risk-{policy}".encode()).hexdigest(),
        "node_threshold": node_threshold, "boundary_threshold": boundary_threshold,
        "risk_limit": P1_RISK if policy == "P1" else P2_RISK,
        "threshold_selection": "median of three nested inner-training-only frozen thresholds",
        "epochs": MOT20_FINETUNE_EPOCHS, "seed": seed,
        "history": str(history_path), "history_sha256": sha256_file(history_path),
    }
    json_write(manifest_path, manifest, refuse_existing=True)
    event("outer_model_frozen", outer=outer, model_sha256=manifest["model_sha256"])
    return manifest


def freeze_outer_policy(outer: str) -> dict[str, Any]:
    rep = json.loads((ROOT / "nested_loso" / outer / "representation_gate.json").read_text(encoding="utf-8"))
    if not rep["passed"]:
        return freeze_outer_p0_manifest(outer, rep)
    exact = run_inner_exact_gate(outer)
    if exact["selected_policy"] == "P0":
        # Representation is separable but exact HOTA/risk gate rejects edits.
        out = ROOT / "outer_policies" / outer; path = out / "outer_policy_manifest.json"
        if path.exists(): return json.loads(path.read_text())
        ext = json.loads((ROOT / "external_pretraining/frozen_checkpoint_manifest.json").read_text())
        tracker = mot20_source_tracker(outer)
        manifest = {
            "experiment": "M23-59 strict outer policy", "outer": outer, "selected_policy": "P0",
            "reason": "inner_exact_HOTA_or_risk_gate_failed", "external_supervision": True,
            "strict_sequence_loso": True, "deployable": True,
            "external_checkpoint_sha256": ext["checkpoint_sha256"], "fine_tuned_model_sha256": None,
            "scaler_sha256": hashlib.sha256(b"M23-59-identity-scaler-v1").hexdigest(), "risk_calibration_sha256": None,
            "boundary_policy": "no-op", "flow_policy": "byte-exact M23-46",
            "candidate_graph_sha256": sha256_file(M57_ROOT / "boundary_universe" / outer / "freeze_manifest.json"),
            "outer_gt_read": False, "outer_teacher_action_read": False,
            "tracker": {"path": str(tracker), "sha256": sha256_file(tracker), "byte_exact_m23_46": True},
            "representation_gate_sha256": sha256_file(ROOT / "nested_loso" / outer / "representation_gate.json"),
            "inner_exact_gate_sha256": sha256_file(ROOT / "nested_loso" / outer / "inner_exact_gate.json"),
            "inner_trackeval_runs": int(exact["trackeval_runs"]),
        }
        out.mkdir(parents=True, exist_ok=True); json_write(path, manifest, refuse_existing=True)
        event("outer_policy_frozen", outer=outer, policy="P0", manifest_sha256=sha256_file(path)); return manifest
    model = fit_outer_final_model(outer)
    out = ROOT / "outer_policies" / outer; path = out / "outer_policy_manifest.json"
    if path.exists(): return json.loads(path.read_text())
    manifest = {
        "experiment": "M23-59 strict outer policy", "outer": outer,
        "selected_policy": exact["selected_policy"], "reason": "representation_and_inner_exact_gates_passed",
        "external_supervision": True, "strict_sequence_loso": True, "deployable": True,
        "external_checkpoint_sha256": model["external_checkpoint_sha256"],
        "fine_tuned_model": model["model"], "fine_tuned_model_sha256": model["model_sha256"],
        "scaler_sha256": model["scaler_sha256"], "risk_calibration_sha256": model["risk_calibration_sha256"],
        "boundary_policy": {"node_threshold": model["node_threshold"], "boundary_threshold": model["boundary_threshold"], "multiple_cuts": True},
        "flow_policy": {"candidate_generator": "fixed M23-55 K256/K32", "utility": "relation logit +0.25 parent -0.25 cross; risk-filtered", "one_to_one": True},
        "candidate_graph_sha256": sha256_file(M57_ROOT / "boundary_universe" / outer / "freeze_manifest.json"),
        "outer_gt_read": False, "outer_teacher_action_read": False,
        "representation_gate_sha256": sha256_file(ROOT / "nested_loso" / outer / "representation_gate.json"),
        "inner_exact_gate_sha256": sha256_file(ROOT / "nested_loso" / outer / "inner_exact_gate.json"),
        "inner_trackeval_runs": int(exact["trackeval_runs"]),
    }
    out.mkdir(parents=True, exist_ok=True); json_write(path, manifest, refuse_existing=True)
    event("outer_policy_frozen", outer=outer, policy=manifest["selected_policy"], manifest_sha256=sha256_file(path)); return manifest


def verify_all_outer_policies_frozen() -> dict[str, dict[str, Any]]:
    manifests = {}
    for seq in MOT20_SEQS:
        path = ROOT / "outer_policies" / seq / "outer_policy_manifest.json"
        if not path.exists(): raise RuntimeError(f"outer policy not frozen: {seq}")
        manifests[seq] = json.loads(path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len([r for r in events if r["event"] == "outer_policy_frozen"]) < 4:
        raise RuntimeError("fewer than four outer policy freeze events")
    return manifests


def run_combined_trackeval(results_dir: Path, output_root: Path, tracker_name: str) -> dict[str, Any]:
    import csv as _csv
    import subprocess as _subprocess
    work = output_root / "official_eval"
    command = [sys.executable, "scripts/eval_motstyle_trackeval.py", "--benchmark-name", "MOT20",
               "--split-to-eval", "train", "--gt-root", "datasets/MOT20/train",
               "--results-dir", str(results_dir), "--tracker-name", tracker_name,
               "--work-dir", str(work), "--keep-workdir", "--seqs", *MOT20_SEQS]
    completed = _subprocess.run(command, cwd=REPO, text=True, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT)
    (output_root / "official_trackeval.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode: raise RuntimeError(completed.stdout[-8000:])
    detailed = work / "eval" / tracker_name / "pedestrian_detailed.csv"
    rows = list(_csv.DictReader(detailed.open(encoding="utf-8")))
    row = next(item for item in rows if item["seq"] == "COMBINED")
    return {"HOTA": 100.0 * float(row["HOTA___AUC"]), "DetA": 100.0 * float(row["DetA___AUC"]),
            "AssA": 100.0 * float(row["AssA___AUC"]), "IDSW": int(float(row["IDSW"]))}


def evaluate_outer_policies() -> dict[str, Any]:
    manifests = verify_all_outer_policies_frozen()
    out = ROOT / "strict_outer_evaluation"; report_path = out / "report.json"
    if report_path.exists(): return json.loads(report_path.read_text(encoding="utf-8"))
    learned = [s for s, m in manifests.items() if m["selected_policy"] != "P0"]
    baseline_report = json.loads(Path("outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/report.json").read_text())
    if not learned:
        hashes = {s: sha256_file(mot20_source_tracker(s)) for s in MOT20_SEQS}
        result = {
            "experiment": "M23-59 strict outer evaluation", "all_outer_policies_P0": True,
            "outer_trackeval_runs": 0, "combined_trackeval_runs": 0,
            "official_trackeval": baseline_report["metrics"], "folds": baseline_report["folds"],
            "byte_exact_tracker_sha256": hashes, "strict_sequence_loso": True,
            "deployable": True, "external_supervision": True,
            "decision": "C_representation_or_inner_exact_gate_failed; M23-59 closed at P0",
        }
        out.mkdir(parents=True, exist_ok=True); json_write(report_path, result, refuse_existing=True)
        event("strict_outer_evaluation_completed", all_p0=True, outer_runs=0, combined_runs=0)
        return result
    results_dir = out / "track_results"; results_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = {}; outer_runs = 0
    m53 = load_module("m23_59_m53_outer_eval", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    for seq in MOT20_SEQS:
        manifest = manifests[seq]
        if manifest["selected_policy"] == "P0":
            shutil.copyfile(mot20_source_tracker(seq), results_dir / f"{seq}.txt")
        else:
            run_root = out / "folds" / seq
            thresholds = manifest["boundary_policy"]
            tracker_report = build_learned_tracker(seq, Path(manifest["fine_tuned_model"]), thresholds,
                                                   manifest["selected_policy"], run_root, teacher_audit=False)
            shutil.copyfile(Path(tracker_report["tracker"]["path"]), results_dir / f"{seq}.txt")
        # Once at most per held fold, after all policy manifests froze.
        fold_root = out / "official_folds" / seq; fold_root.mkdir(parents=True, exist_ok=True)
        (fold_root / "track_results").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(results_dir / f"{seq}.txt", fold_root / "track_results" / f"{seq}.txt")
        fold_metrics[seq] = m53.run_official_trackeval(seq=seq, output_root=fold_root, tracker_name=f"m23_59_outer_{seq[-2:]}")
        outer_runs += 1
    combined = run_combined_trackeval(results_dir, out / "combined", "m23_59_combined")
    result = {
        "experiment": "M23-59 strict outer evaluation", "all_outer_policies_P0": False,
        "learned_outer_sequences": learned, "outer_trackeval_runs": outer_runs, "combined_trackeval_runs": 1,
        "official_trackeval": combined, "folds": fold_metrics, "strict_sequence_loso": True,
        "deployable": True, "external_supervision": True,
        "decision": "A_above_80" if combined["HOTA"] > 80.0 else "B_strict_gain_not_above_80" if combined["HOTA"] > baseline_report["metrics"]["HOTA"] else "C_no_strict_gain",
    }
    json_write(report_path, result, refuse_existing=True)
    event("strict_outer_evaluation_completed", all_p0=False, outer_runs=outer_runs, combined_runs=1,
          HOTA=combined["HOTA"]); return result


def summarize() -> dict[str, Any]:
    evaluation = evaluate_outer_policies()
    output = {
        "experiment": "M23-59 relation-pretrained hierarchical identity segmentation and flow feasibility",
        "status": "closed", "external_supervision": True,
        "strict_sequence_loso": True, "deployable": True,
        "internal_data_strict_best": {"experiment": "M23-46", "HOTA": 79.123193, "DetA": 81.543470, "AssA": 76.825150, "IDSW": 996},
        "teacher_capacity_reference": {"experiment": "M23-57 v2", "HOTA": 81.148046, "teacher_only": True, "deployable": False},
        "external_dataset_manifest_sha256": sha256_file(EXTERNAL_MANIFEST),
        "preregistration_sha256": sha256_file(PREREG_JSON),
        "implementation_manifest_sha256": sha256_file(IMPLEMENTATION_MANIFEST),
        "external_checkpoint_manifest_sha256": sha256_file(ROOT / "external_pretraining/frozen_checkpoint_manifest.json"),
        "outer_policies": {s: json.loads((ROOT / "outer_policies" / s / "outer_policy_manifest.json").read_text()) for s in MOT20_SEQS},
        "evaluation": evaluation,
        "m23_54_started": False, "m23_58_started": False, "mot20_test_submission": False,
    }
    path = ROOT / "final_summary.json"; json_write(path, output)
    print(json.dumps(output, indent=2, ensure_ascii=False)); return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("extract-external"); p.add_argument("--seq", required=True, choices=MOT17_PHYSICAL)
    p = sub.add_parser("build-external-examples"); p.add_argument("--split", required=True, choices=["train", "validation"])
    sub.add_parser("train-external")
    p = sub.add_parser("freeze-mot20-observable"); p.add_argument("--seq", required=True, choices=MOT20_SEQS)
    p = sub.add_parser("unlock-mot20-labels"); p.add_argument("--seq", required=True, choices=MOT20_SEQS)
    p = sub.add_parser("run-outer-representation"); p.add_argument("--outer", required=True, choices=MOT20_SEQS)
    p = sub.add_parser("run-inner-exact"); p.add_argument("--outer", required=True, choices=MOT20_SEQS)
    p = sub.add_parser("freeze-outer-policy"); p.add_argument("--outer", required=True, choices=MOT20_SEQS)
    sub.add_parser("evaluate-outers"); sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "extract-external": result = extract_external(args.seq)
    elif args.command == "build-external-examples": result = build_external_examples(args.split)
    elif args.command == "train-external": result = train_external()
    elif args.command == "freeze-mot20-observable": result = freeze_mot20_observable(args.seq)
    elif args.command == "unlock-mot20-labels": result = unlock_mot20_labels(args.seq)
    elif args.command == "run-outer-representation": result = run_outer_representation_gate(args.outer)
    elif args.command == "run-inner-exact": result = run_inner_exact_gate(args.outer)
    elif args.command == "freeze-outer-policy": result = freeze_outer_policy(args.outer)
    elif args.command == "evaluate-outers": result = evaluate_outer_policies()
    elif args.command == "summarize": result = summarize()
    else: raise AssertionError(args.command)
    if args.command != "summarize": print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
