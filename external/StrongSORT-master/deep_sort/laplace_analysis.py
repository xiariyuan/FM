import csv
import os

import numpy as np


def _tlbr_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _track_history_len(track):
    history = list(getattr(track, "laplace_history", []))
    if len(history) > 0:
        return len(history)
    feats = list(getattr(track, "features", []))
    if len(feats) > 0:
        return len(feats)
    if getattr(track, "smooth_feat", None) is not None:
        return 1
    if getattr(track, "curr_feat", None) is not None:
        return 1
    return 0


class LaplaceAnalysisWriter:
    FIELDNAMES = [
        "seq",
        "frame",
        "assoc_stage",
        "track_id",
        "det_index",
        "gap",
        "history_len",
        "chosen",
        "is_true_match",
        "track_gt_id",
        "det_gt_id",
        "pair_rel",
        "learned_alpha",
        "learned_r",
        "appearance_sim",
        "fused_sim",
        "motion_sim",
        "spatial_sim",
        "laplace_sim",
        "agreement",
        "stability",
        "coherence",
        "det_score",
        "prod_sim",
        "amb_spa",
        "amb_lap",
        "amb_mot",
    ]

    def __init__(self, seq_name, img_dir, out_dir, iou_thresh=0.5):
        self.seq_name = seq_name
        self.iou_thresh = float(iou_thresh)
        self.gt_by_frame = self._load_gt(img_dir)
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{seq_name}_pairs.csv")
        self._fp = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fp, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def close(self):
        if getattr(self, "_fp", None) is not None:
            self._fp.close()
            self._fp = None

    def _load_gt(self, img_dir):
        seq_dir = os.path.dirname(img_dir)
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            return {}

        gt_by_frame = {}
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cols = line.split(",")
                if len(cols) < 8:
                    continue
                frame_id = int(float(cols[0]))
                gt_id = int(float(cols[1]))
                x = float(cols[2])
                y = float(cols[3])
                w = float(cols[4])
                h = float(cols[5])
                mark = int(float(cols[6]))
                cls = int(float(cols[7]))
                if mark != 1 or cls != 1:
                    continue
                tlbr = np.asarray([x, y, x + w, y + h], dtype=np.float32)
                gt_by_frame.setdefault(frame_id, []).append((gt_id, tlbr))
        return gt_by_frame

    def assign_detection_gt(self, det, frame_id):
        gt_items = self.gt_by_frame.get(int(frame_id), [])
        if not gt_items:
            det.analysis_gt_id = -1
            return -1
        det_box = np.asarray(det.to_tlbr(), dtype=np.float32)
        best_iou = 0.0
        best_gt = -1
        for gt_id, gt_box in gt_items:
            iou = _tlbr_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_id
        det.analysis_gt_id = best_gt if best_iou >= self.iou_thresh else -1
        return det.analysis_gt_id

    def assign_track_gt(self, track, frame_id):
        gt_items = self.gt_by_frame.get(int(frame_id), [])
        if not gt_items:
            return -1
        track_box = np.asarray(track.to_tlbr(), dtype=np.float32)
        best_iou = 0.0
        best_gt = -1
        for gt_id, gt_box in gt_items:
            iou = _tlbr_iou(track_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_id
        return best_gt if best_iou >= self.iou_thresh else -1

    def log_first_assoc(self, frame_id, tracks, detections, debug, chosen_pairs, valid_mask, assoc_stage="primary"):
        if debug is None:
            return
        pair_rel = debug["pair_rel"]
        learned_alpha = debug.get("learned_alpha", None)
        learned_r = debug.get("learned_r", None)
        appearance_sim = debug["appearance_sim"]
        fused_sim = debug.get("fused_sim", None)
        motion_sim = debug["motion_sim"]
        spatial_sim = debug["spatial_sim"]
        laplace_sim = debug["laplace_sim"]
        agreement = debug["agreement"]
        stability = debug["stability"]
        coherence = debug["coherence"]
        prod_sim = debug.get("prod_sim", None)
        amb_spa = debug.get("amb_spa", None)
        amb_lap = debug.get("amb_lap", None)
        amb_mot = debug.get("amb_mot", None)

        for row, track in enumerate(tracks):
            track_gt_id = int(self.assign_track_gt(track, frame_id))
            gap = int(max(0, int(getattr(track, "time_since_update", 0))))
            history_len = int(_track_history_len(track))
            for col, det in enumerate(detections):
                if valid_mask is not None and not bool(valid_mask[row, col]):
                    continue
                det_gt_id = int(getattr(det, "analysis_gt_id", -1) or -1)
                chosen = 1 if (row, col) in chosen_pairs else 0
                is_true = 1 if track_gt_id > 0 and det_gt_id > 0 and track_gt_id == det_gt_id else 0
                self._writer.writerow(
                    {
                        "seq": self.seq_name,
                        "frame": int(frame_id),
                        "assoc_stage": assoc_stage,
                        "track_id": int(getattr(track, "track_id", -1)),
                        "det_index": int(col),
                        "gap": gap,
                        "history_len": history_len,
                        "chosen": chosen,
                        "is_true_match": is_true,
                        "track_gt_id": track_gt_id,
                        "det_gt_id": det_gt_id,
                        "pair_rel": float(pair_rel[row, col]),
                        "learned_alpha": float(learned_alpha[row, col]) if learned_alpha is not None else float("nan"),
                        "learned_r": float(learned_r[row, col]) if learned_r is not None else float("nan"),
                        "appearance_sim": float(appearance_sim[row, col]),
                        "fused_sim": float(fused_sim[row, col]) if fused_sim is not None else float("nan"),
                        "motion_sim": float(motion_sim[row, col]),
                        "spatial_sim": float(spatial_sim[row, col]),
                        "laplace_sim": float(laplace_sim[row, col]),
                        "agreement": float(agreement[row, col]),
                        "stability": float(stability[row]),
                        "coherence": float(coherence[row]),
                        "det_score": float(getattr(det, "confidence", 0.0)),
                        "prod_sim": float(prod_sim[row, col]) if prod_sim is not None else float("nan"),
                        "amb_spa": float(amb_spa[row, col]) if amb_spa is not None else float("nan"),
                        "amb_lap": float(amb_lap[row, col]) if amb_lap is not None else float("nan"),
                        "amb_mot": float(amb_mot[row, col]) if amb_mot is not None else float("nan"),
                    }
                )
        self._fp.flush()
