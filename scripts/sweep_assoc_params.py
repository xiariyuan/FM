#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep FM-Track association parameters (ID_THRESH / ASSOC_IOU_GATE / DET_MAX_PER_FRAME).

Usage:
  python scripts/sweep_assoc_params.py \
    --config-path configs/bytetrack_fa_mot_mot17_v5_seqid.yaml \
    --checkpoint outputs/bytetrack_fa_mot_mot17_v5_seqid/checkpoint_epoch_10.pth \
    --data-root /gemini/code/datasets \
    --split train \
    --out-root outputs/sweep_assoc

Optional lists (comma-separated):
  --id-thresh-list 0.002,0.005,0.01
  --assoc-iou-gate-list 0.2,0.3,0.4
  --det-max-per-frame-list 40,50,60
  --miss-tolerance-list 20,30,50
  --det-thresh-list -0.4,-0.2,0.0,0.1
  --newborn-thresh-list -0.4,-0.2,0.0,0.1
  --assoc-mode-list logit,hybrid,feature
  --assoc-id-weight-list 0.0,1.0
  --assoc-iou-weight-list 0.0,1.0
  --assoc-feat-weight-list 0.0,1.0
  --assoc-feat-agg-list last,mean
  --assoc-feat-k-list 1,5,10
  --assoc-use-det-score-list 0,1
  --assoc-reid-box-expand-list 1.0,1.05,1.1
  --assoc-bbox-dist-weight-list 0.0,0.25,0.5
  --assoc-bbox-dist-tau-list 0.5,1.0,1.5
  --assoc-bbox-dist-use-cal-factor-list 0,1
  --assoc-two-stage-list 0,1
  --assoc-stage2-id-thresh-list 0.25,0.3,0.35
  --assoc-stage2-iou-gate-list 0.0,0.1
  --assoc-stage2-bbox-weight-list 0.5,1.0

For public detections, it's common to tie newborn_thresh to det_thresh (so any kept detection can "be born"):
  --tie-newborn-to-det
"""

from __future__ import annotations
import argparse
import csv
import itertools
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any


def _parse_list(raw: str, cast) -> List:
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def _parse_pedestrian_summary(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        names = f.readline().strip().split()
        vals = f.readline().strip().split()
    out = {}
    for k, v in zip(names, vals):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _find_summary_path(out_dir: Path, dataset: str, split: str) -> Path:
    # Prefer tracker_min* if present
    candidates = []
    for p in out_dir.glob("tracker_min*/{}-{}/pedestrian_summary.txt".format(dataset, split)):
        candidates.append(p)
    if candidates:
        return sorted(candidates)[0]
    return out_dir / "tracker" / f"{dataset}-{split}" / "pedestrian_summary.txt"


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="MOT17")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--detector-profile", default=None)
    ap.add_argument("--det-source", default=None)
    ap.add_argument("--external-det-root", default=None)
    ap.add_argument("--external-det-pattern", default=None)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--id-thresh-list", default=None)
    ap.add_argument("--assoc-iou-gate-list", default=None)
    ap.add_argument("--det-max-per-frame-list", default=None)
    ap.add_argument("--miss-tolerance-list", default=None)
    ap.add_argument("--det-thresh-list", default=None)
    ap.add_argument("--newborn-thresh-list", default=None)
    ap.add_argument("--assoc-mode-list", default=None)
    ap.add_argument("--assoc-id-weight-list", default=None)
    ap.add_argument("--assoc-iou-weight-list", default=None)
    ap.add_argument("--assoc-feat-weight-list", default=None)
    ap.add_argument("--assoc-feat-score-mode-list", default=None, help="Comma-separated: raw,softmax")
    ap.add_argument("--assoc-feat-tau-list", default=None, help="Comma-separated float list for ASSOC_FEAT_TAU")
    ap.add_argument("--assoc-feat-agg-list", default=None, help="Comma-separated: last,mean")
    ap.add_argument("--assoc-feat-k-list", default=None, help="Comma-separated int list for ASSOC_FEAT_K")
    ap.add_argument("--assoc-use-det-score-list", default=None, help="Comma-separated int list: 0,1")
    ap.add_argument("--assoc-reid-box-expand-list", default=None, help="Comma-separated float list for ASSOC_REID_BOX_EXPAND")
    ap.add_argument("--assoc-bbox-dist-weight-list", default=None, help="Comma-separated float list for ASSOC_BBOX_DIST_WEIGHT")
    ap.add_argument("--assoc-bbox-dist-tau-list", default=None, help="Comma-separated float list for ASSOC_BBOX_DIST_TAU")
    ap.add_argument("--assoc-bbox-dist-use-cal-factor-list", default=None, help="Comma-separated int list: 0,1")
    ap.add_argument("--assoc-two-stage-list", default=None, help="Comma-separated int list: 0,1")
    ap.add_argument("--assoc-stage2-id-thresh-list", default=None, help="Comma-separated float list for ASSOC_STAGE2_ID_THRESH")
    ap.add_argument("--assoc-stage2-iou-gate-list", default=None, help="Comma-separated float list for ASSOC_STAGE2_IOU_GATE")
    ap.add_argument("--assoc-stage2-bbox-weight-list", default=None, help="Comma-separated float list for ASSOC_STAGE2_BBOX_WEIGHT")
    ap.add_argument("--tie-newborn-to-det", action="store_true")
    ap.add_argument("--eval-only-val", action="store_true")
    ap.add_argument("--val-sequences", default=None)
    ap.add_argument("--detector-filter", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    id_list = [0.005, 0.01, 0.02]
    if args.id_thresh_list:
        id_list = _parse_list(args.id_thresh_list, float)
    iou_list = [0.2, 0.3, 0.4]
    if args.assoc_iou_gate_list:
        iou_list = _parse_list(args.assoc_iou_gate_list, float)
    det_list = [40, 50, 60]
    if args.det_max_per_frame_list:
        det_list = _parse_list(args.det_max_per_frame_list, int)

    miss_tol_list = [None]
    if args.miss_tolerance_list:
        miss_tol_list = _parse_list(args.miss_tolerance_list, int)

    det_thr_list = [None]
    if args.det_thresh_list:
        det_thr_list = _parse_list(args.det_thresh_list, float)

    newborn_thr_list = [None]
    if args.newborn_thresh_list:
        newborn_thr_list = _parse_list(args.newborn_thresh_list, float)

    assoc_mode_list = [None]
    if args.assoc_mode_list:
        assoc_mode_list = _parse_list(args.assoc_mode_list, str)

    assoc_id_w_list = [None]
    if args.assoc_id_weight_list:
        assoc_id_w_list = _parse_list(args.assoc_id_weight_list, float)

    assoc_iou_w_list = [None]
    if args.assoc_iou_weight_list:
        assoc_iou_w_list = _parse_list(args.assoc_iou_weight_list, float)

    assoc_feat_w_list = [None]
    if args.assoc_feat_weight_list:
        assoc_feat_w_list = _parse_list(args.assoc_feat_weight_list, float)

    assoc_feat_score_mode_list = [None]
    if args.assoc_feat_score_mode_list:
        assoc_feat_score_mode_list = _parse_list(args.assoc_feat_score_mode_list, str)

    assoc_feat_tau_list = [None]
    if args.assoc_feat_tau_list:
        assoc_feat_tau_list = _parse_list(args.assoc_feat_tau_list, float)

    assoc_feat_agg_list = [None]
    if args.assoc_feat_agg_list:
        assoc_feat_agg_list = _parse_list(args.assoc_feat_agg_list, str)

    assoc_feat_k_list = [None]
    if args.assoc_feat_k_list:
        assoc_feat_k_list = _parse_list(args.assoc_feat_k_list, int)

    assoc_use_det_score_list = [None]
    if args.assoc_use_det_score_list:
        assoc_use_det_score_list = _parse_list(args.assoc_use_det_score_list, int)

    assoc_reid_box_expand_list = [None]
    if args.assoc_reid_box_expand_list:
        assoc_reid_box_expand_list = _parse_list(args.assoc_reid_box_expand_list, float)

    assoc_bbox_dist_w_list = [None]
    if args.assoc_bbox_dist_weight_list:
        assoc_bbox_dist_w_list = _parse_list(args.assoc_bbox_dist_weight_list, float)

    assoc_bbox_dist_tau_list = [None]
    if args.assoc_bbox_dist_tau_list:
        assoc_bbox_dist_tau_list = _parse_list(args.assoc_bbox_dist_tau_list, float)

    assoc_bbox_dist_use_cf_list = [None]
    if args.assoc_bbox_dist_use_cal_factor_list:
        assoc_bbox_dist_use_cf_list = _parse_list(args.assoc_bbox_dist_use_cal_factor_list, int)

    assoc_two_stage_list = [None]
    if args.assoc_two_stage_list:
        assoc_two_stage_list = _parse_list(args.assoc_two_stage_list, int)

    assoc_stage2_id_thr_list = [None]
    if args.assoc_stage2_id_thresh_list:
        assoc_stage2_id_thr_list = _parse_list(args.assoc_stage2_id_thresh_list, float)

    assoc_stage2_iou_gate_list = [None]
    if args.assoc_stage2_iou_gate_list:
        assoc_stage2_iou_gate_list = _parse_list(args.assoc_stage2_iou_gate_list, float)

    assoc_stage2_bbox_w_list = [None]
    if args.assoc_stage2_bbox_weight_list:
        assoc_stage2_bbox_w_list = _parse_list(args.assoc_stage2_bbox_weight_list, float)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    newborn_count = 1 if args.tie_newborn_to_det else len(newborn_thr_list)
    total_runs = (
        len(det_thr_list)
        * newborn_count
        * len(id_list)
        * len(iou_list)
        * len(det_list)
        * len(miss_tol_list)
        * len(assoc_mode_list)
        * len(assoc_id_w_list)
        * len(assoc_iou_w_list)
        * len(assoc_feat_w_list)
        * len(assoc_feat_score_mode_list)
        * len(assoc_feat_tau_list)
        * len(assoc_feat_agg_list)
        * len(assoc_feat_k_list)
        * len(assoc_use_det_score_list)
        * len(assoc_reid_box_expand_list)
        * len(assoc_bbox_dist_w_list)
        * len(assoc_bbox_dist_tau_list)
        * len(assoc_bbox_dist_use_cf_list)
        * len(assoc_two_stage_list)
        * len(assoc_stage2_id_thr_list)
        * len(assoc_stage2_iou_gate_list)
        * len(assoc_stage2_bbox_w_list)
    )
    run_idx = 0

    for det_thr in det_thr_list:
        _newborn_iter = [det_thr] if args.tie_newborn_to_det else newborn_thr_list
        for newborn_thr in _newborn_iter:
            for (
                id_thr,
                iou_gate,
                det_max,
                miss_tol,
                assoc_mode,
                assoc_id_w,
                assoc_iou_w,
                assoc_feat_w,
                assoc_feat_score_mode,
                assoc_feat_tau,
                assoc_feat_agg,
                assoc_feat_k,
                assoc_use_det_score,
                assoc_reid_box_expand,
                assoc_bbox_dist_w,
                assoc_bbox_dist_tau,
                assoc_bbox_dist_use_cf,
                assoc_two_stage,
                assoc_stage2_id_thr,
                assoc_stage2_iou_gate,
                assoc_stage2_bbox_w,
            ) in itertools.product(
                id_list,
                iou_list,
                det_list,
                miss_tol_list,
                assoc_mode_list,
                assoc_id_w_list,
                assoc_iou_w_list,
                assoc_feat_w_list,
                assoc_feat_score_mode_list,
                assoc_feat_tau_list,
                assoc_feat_agg_list,
                assoc_feat_k_list,
                assoc_use_det_score_list,
                assoc_reid_box_expand_list,
                assoc_bbox_dist_w_list,
                assoc_bbox_dist_tau_list,
                assoc_bbox_dist_use_cf_list,
                assoc_two_stage_list,
                assoc_stage2_id_thr_list,
                assoc_stage2_iou_gate_list,
                assoc_stage2_bbox_w_list,
            ):
                run_idx += 1
                name_parts = []
                if det_thr is not None:
                    name_parts.append(f"detthr{det_thr}")
                if newborn_thr is not None:
                    name_parts.append(f"new{newborn_thr}")
                name_parts.append(f"id{id_thr}_iou{iou_gate}_det{det_max}")
                if miss_tol is not None:
                    name_parts.append(f"miss{miss_tol}")
                if assoc_mode is not None:
                    name_parts.append(f"mode{assoc_mode}")
                if assoc_id_w is not None:
                    name_parts.append(f"idw{assoc_id_w}")
                if assoc_iou_w is not None:
                    name_parts.append(f"iouw{assoc_iou_w}")
                if assoc_feat_w is not None:
                    name_parts.append(f"featw{assoc_feat_w}")
                if assoc_feat_score_mode is not None:
                    name_parts.append(f"fscore{assoc_feat_score_mode}")
                if assoc_feat_tau is not None:
                    name_parts.append(f"ftau{assoc_feat_tau}")
                if assoc_feat_agg is not None:
                    name_parts.append(f"fagg{assoc_feat_agg}")
                if assoc_feat_k is not None:
                    name_parts.append(f"fk{assoc_feat_k}")
                if assoc_use_det_score is not None:
                    name_parts.append(f"detscore{int(assoc_use_det_score)}")
                if assoc_reid_box_expand is not None:
                    name_parts.append(f"reidbox{assoc_reid_box_expand}")
                if assoc_bbox_dist_w is not None:
                    name_parts.append(f"bboxw{assoc_bbox_dist_w}")
                if assoc_bbox_dist_tau is not None:
                    name_parts.append(f"bboxtau{assoc_bbox_dist_tau}")
                if assoc_bbox_dist_use_cf is not None:
                    name_parts.append(f"bboxcf{int(assoc_bbox_dist_use_cf)}")
                if assoc_two_stage is not None:
                    name_parts.append(f"stage2{int(assoc_two_stage)}")
                if assoc_stage2_id_thr is not None:
                    name_parts.append(f"s2thr{assoc_stage2_id_thr}")
                if assoc_stage2_iou_gate is not None:
                    name_parts.append(f"s2iou{assoc_stage2_iou_gate}")
                if assoc_stage2_bbox_w is not None:
                    name_parts.append(f"s2bboxw{assoc_stage2_bbox_w}")
                run_name = "_".join(name_parts)
                out_dir = out_root / run_name
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "/root/miniconda3/bin/python", "-u", "submit_bytetrack.py",
                    "--config-path", args.config_path,
                    "--inference-model", args.checkpoint,
                    "--inference-dataset", args.dataset,
                    "--inference-split", args.split,
                    "--data-root", args.data_root,
                    "--output-dir", str(out_dir),
                    "--id-thresh", str(id_thr),
                    "--assoc-iou-gate", str(iou_gate),
                    "--det-max-per-frame", str(det_max),
                ]
                if miss_tol is not None:
                    cmd += ["--miss-tolerance", str(miss_tol)]
                if det_thr is not None:
                    cmd += ["--det-thresh", str(det_thr)]
                if newborn_thr is not None:
                    cmd += ["--newborn-thresh", str(newborn_thr)]
                if assoc_mode is not None:
                    cmd += ["--assoc-mode", str(assoc_mode)]
                if assoc_id_w is not None:
                    cmd += ["--assoc-id-weight", str(assoc_id_w)]
                if assoc_iou_w is not None:
                    cmd += ["--assoc-iou-weight", str(assoc_iou_w)]
                if assoc_feat_w is not None:
                    cmd += ["--assoc-feat-weight", str(assoc_feat_w)]
                if assoc_feat_score_mode is not None:
                    cmd += ["--assoc-feat-score-mode", str(assoc_feat_score_mode)]
                if assoc_feat_tau is not None:
                    cmd += ["--assoc-feat-tau", str(assoc_feat_tau)]
                if assoc_feat_agg is not None:
                    cmd += ["--assoc-feat-agg", str(assoc_feat_agg)]
                if assoc_feat_k is not None:
                    cmd += ["--assoc-feat-k", str(int(assoc_feat_k))]
                if assoc_use_det_score is not None:
                    cmd += ["--assoc-use-det-score", str(int(assoc_use_det_score))]
                if assoc_reid_box_expand is not None:
                    cmd += ["--assoc-reid-box-expand", str(float(assoc_reid_box_expand))]
                if assoc_bbox_dist_w is not None:
                    cmd += ["--assoc-bbox-dist-weight", str(float(assoc_bbox_dist_w))]
                if assoc_bbox_dist_tau is not None:
                    cmd += ["--assoc-bbox-dist-tau", str(float(assoc_bbox_dist_tau))]
                if assoc_bbox_dist_use_cf is not None:
                    cmd += ["--assoc-bbox-dist-use-cal-factor", str(int(assoc_bbox_dist_use_cf))]
                if assoc_two_stage is not None:
                    cmd += ["--assoc-two-stage", str(int(assoc_two_stage))]
                if assoc_stage2_id_thr is not None:
                    cmd += ["--assoc-stage2-id-thresh", str(float(assoc_stage2_id_thr))]
                if assoc_stage2_iou_gate is not None:
                    cmd += ["--assoc-stage2-iou-gate", str(float(assoc_stage2_iou_gate))]
                if assoc_stage2_bbox_w is not None:
                    cmd += ["--assoc-stage2-bbox-weight", str(float(assoc_stage2_bbox_w))]
                if args.detector_filter:
                    cmd += ["--detector-filter", args.detector_filter]
                if args.eval_only_val:
                    cmd += ["--eval-only-val"]
                if args.val_sequences:
                    cmd += ["--val-sequences", args.val_sequences]
                if args.detector_profile:
                    cmd += ["--detector-profile", args.detector_profile]
                if args.det_source:
                    cmd += ["--det-source", args.det_source]
                if args.external_det_root:
                    cmd += ["--external-det-root", args.external_det_root]
                if args.external_det_pattern:
                    cmd += ["--external-det-pattern", args.external_det_pattern]

                print(f"\n[{run_idx}/{total_runs}] {run_name}")
                print("CMD:", " ".join(cmd))

                t0 = time.time()
                ok = True
                err = ""
                if not args.dry_run:
                    run_log = out_dir / "run.log"
                    try:
                        with run_log.open("w", encoding="utf-8") as f:
                            f.write("CMD: " + " ".join(cmd) + "\n\n")
                            f.flush()
                            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
                    except subprocess.CalledProcessError as e:
                        ok = False
                        err = f"CalledProcessError: {e}"
                    except Exception as e:
                        ok = False
                        err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0

                metrics = {}
                metrics_path = _find_summary_path(out_dir, args.dataset, args.split)
                if ok and (not args.dry_run):
                    try:
                        if not metrics_path.exists():
                            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
                        metrics = _parse_pedestrian_summary(metrics_path)
                    except Exception as e:
                        ok = False
                        err = f"MetricsParseError: {e}"

                rec: Dict[str, Any] = {
                    "run": run_name,
                    "ok": ok,
                    "seconds": round(dt, 2),
                    "det_thresh": det_thr,
                    "newborn_thresh": newborn_thr,
                    "id_thresh": id_thr,
                    "assoc_iou_gate": iou_gate,
                    "det_max_per_frame": det_max,
                    "miss_tolerance": miss_tol,
                    "assoc_mode": assoc_mode,
                    "assoc_id_weight": assoc_id_w,
                    "assoc_iou_weight": assoc_iou_w,
                    "assoc_feat_weight": assoc_feat_w,
                    "assoc_feat_score_mode": assoc_feat_score_mode,
                    "assoc_feat_tau": assoc_feat_tau,
                    "assoc_feat_agg": assoc_feat_agg,
                    "assoc_feat_k": assoc_feat_k,
                    "assoc_use_det_score": assoc_use_det_score,
                    "assoc_reid_box_expand": assoc_reid_box_expand,
                    "assoc_bbox_dist_weight": assoc_bbox_dist_w,
                    "assoc_bbox_dist_tau": assoc_bbox_dist_tau,
                    "assoc_bbox_dist_use_cal_factor": assoc_bbox_dist_use_cf,
                    "assoc_two_stage": assoc_two_stage,
                    "assoc_stage2_id_thresh": assoc_stage2_id_thr,
                    "assoc_stage2_iou_gate": assoc_stage2_iou_gate,
                    "assoc_stage2_bbox_weight": assoc_stage2_bbox_w,
                    "metrics_path": str(metrics_path),
                    "error": err,
                }
                for k in [
                    "HOTA",
                    "MOTA",
                    "IDF1",
                    "AssA",
                    "DetA",
                    "DetRe",
                    "DetPr",
                    "AssRe",
                    "AssPr",
                    "IDSW",
                    "Frag",
                ]:
                    if k in metrics:
                        rec[k] = metrics[k]
                records.append(rec)
                _write_csv(records, out_root / "sweep_assoc_results.csv")

                if ok:
                    hota = metrics.get("HOTA", None)
                    deta = metrics.get("DetA", None)
                    assa = metrics.get("AssA", None)
                    idsw = metrics.get("IDSW", None)
                    msg = []
                    if hota is not None:
                        msg.append(f"HOTA={hota:.3f}")
                    if deta is not None:
                        msg.append(f"DetA={deta:.3f}")
                    if assa is not None:
                        msg.append(f"AssA={assa:.3f}")
                    if idsw is not None:
                        msg.append(f"IDSW={int(idsw)}")
                    if msg:
                        print("Result:", ", ".join(msg))

                if (not ok) and (not args.keep_going) and (not args.dry_run):
                    print(f"Run failed, stopping. Error: {err}")
                    return

    _write_csv(records, out_root / "sweep_assoc_results.csv")


if __name__ == "__main__":
    main()
