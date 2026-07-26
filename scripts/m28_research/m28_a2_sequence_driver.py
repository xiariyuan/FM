from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/mot20_m28_20260726/m28_a2_multisequence_capacity"
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")


def load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seq_root(seq: str) -> Path:
    return ROOT / seq


def dump_path(seq: str) -> Path:
    return REPO / "outputs/alink_train_inputs/phase0_root" / seq / "dump_yolox_reid.npz"


def baseline_reference(seq: str) -> Path:
    return seq_root(seq) / "reference" / "track_results" / f"{seq}.txt"


def make_reference(seq: str) -> None:
    out = seq_root(seq) / "reference"
    if out.exists():
        raise FileExistsError(out)
    path = out / "track_results" / f"{seq}.txt"
    path.parent.mkdir(parents=True)
    command = [
        sys.executable, "scripts/dmm_base_tracker.py", "--dump-npz", str(dump_path(seq)),
        "--seq", seq, "--out", str(path), "--summary-json", str(out / "summary.json"),
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stdout[-8000:])
    print((out / "summary.json").read_text())


def freeze_runtime(seq: str) -> None:
    module = load(f"m27_a2_{seq[-2:]}", "scripts/m27_research/m27_a0_exact_idsw_source_attribution.py")
    module.SEQ = seq
    module.DUMP = dump_path(seq)
    module.REFERENCE = baseline_reference(seq)
    module.ROOT = seq_root(seq)
    module.freeze_runtime()


def freeze_candidates(seq: str) -> None:
    module = load(f"m28_a2_{seq[-2:]}", "scripts/m28_research/m28_a0_deferred_identity_inheritance.py")
    module.SEQ = seq
    module.ROOT = seq_root(seq)
    module.M27 = seq_root(seq) / "frozen_runtime"
    module.BASELINE = module.M27 / "baseline_online.txt"
    module.UPDATES = module.M27 / "association_updates.parquet"
    module.DUMP = dump_path(seq)
    module.GT = REPO / "datasets/MOT20/train" / seq / "gt/gt.txt"
    module.freeze_candidates()


def dominant_identity(matches: pd.DataFrame):
    counts = matches.groupby(["tracker_id", "gt_id"]).size().reset_index(name="count")
    totals = counts.groupby("tracker_id").count.transform("sum") if False else None
    total_map = counts.groupby("tracker_id")["count"].sum().to_dict()
    counts["purity"] = [float(row.count) / max(int(total_map[int(row.tracker_id)]), 1) for row in counts.itertuples(index=False)]
    best = counts.sort_values(["tracker_id", "count", "gt_id"], ascending=[True, False, True]).groupby("tracker_id", as_index=False).first()
    return dict(zip(best.tracker_id.astype(int), best.gt_id.astype(int))), dict(zip(best.tracker_id.astype(int), best.purity.astype(float)))


def teacher(seq: str) -> None:
    root = seq_root(seq)
    out = root / "teacher_capacity"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    candidates = pd.read_parquet(root / "frozen_candidates/candidates.parquet")
    m27 = load(f"m27_teacher_{seq[-2:]}", "scripts/m27_research/m27_a0_exact_idsw_source_attribution.py")
    m37 = load(f"m37_teacher_{seq[-2:]}", "scripts/m23_research/m23_37_fast_exact_hota_teacher.py")
    m28 = load(f"m28_teacher_{seq[-2:]}", "scripts/m28_research/m28_a0_deferred_identity_inheritance.py")
    baseline = root / "frozen_runtime/baseline_online.txt"
    prepared = m37.PreparedExactHOTA(seq, baseline, out / "exact_cache")
    tracker_map, _ = m27.original_tracker_mapping(prepared)
    switches, matches = m27.reconstruct_clear_idsw(prepared.data, tracker_map)
    dmap, purity = dominant_identity(matches)
    candidates["young_gt"] = candidates.young_track_id.map(dmap).fillna(-1).astype(int)
    candidates["old_gt"] = candidates.old_track_id.map(dmap).fillna(-2).astype(int)
    candidates["young_purity"] = candidates.young_track_id.map(purity).fillna(0.0)
    candidates["old_purity"] = candidates.old_track_id.map(purity).fillna(0.0)
    candidates["same_dominant_gt"] = ((candidates.young_gt > 0) & (candidates.young_gt == candidates.old_gt)).astype(int)
    screened = candidates[candidates.same_dominant_gt == 1].copy()
    baseline_ids = prepared.parent_row_ids.copy()
    parent_frames = np.asarray([int(float(fields[0])) for fields in prepared.parent_rows], dtype=np.int32)
    baseline_metrics = prepared.evaluate_row_ids_incremental(baseline_ids)
    labels = []
    for ordinal, candidate in enumerate(screened.itertuples(index=False), start=1):
        ids = baseline_ids.copy()
        mask = (ids == int(candidate.young_track_id)) & (parent_frames >= int(candidate.frame))
        ids[mask] = int(candidate.old_track_id)
        valid = bool(mask.any())
        if valid:
            for frame in np.unique(parent_frames[mask]):
                frame_ids = ids[parent_frames == frame]
                if len(frame_ids) != len(np.unique(frame_ids)):
                    valid = False; break
        if not valid:
            labels.append({**candidate._asdict(), "status": "invalid", "modified_rows": int(mask.sum()), "delta_HOTA": math.nan})
            continue
        metrics = prepared.evaluate_row_ids_incremental(ids)
        labels.append({
            **candidate._asdict(), "status": "success", "modified_rows": int(mask.sum()),
            "HOTA": float(metrics["HOTA"]), "DetA": float(metrics["DetA"]), "AssA": float(metrics["AssA"]),
            "delta_HOTA": float(metrics["HOTA"] - baseline_metrics["HOTA"]),
            "delta_DetA": float(metrics["DetA"] - baseline_metrics["DetA"]),
            "delta_AssA": float(metrics["AssA"] - baseline_metrics["AssA"]),
        })
        if ordinal % 100 == 0:
            print(seq, "exact", ordinal, "/", len(screened), flush=True)
    label_frame = pd.DataFrame(labels)
    label_frame.to_parquet(out / "identity_consistent_exact_labels.parquet", index=False)
    success = label_frame[label_frame.status == "success"].sort_values(["delta_HOTA", "candidate_score"], ascending=[False, False])
    current = baseline_ids.copy(); current_metrics = dict(baseline_metrics)
    used_young = set(); used_old = set(); selected = []
    for candidate in success[success.delta_HOTA > 0].itertuples(index=False):
        young, old = int(candidate.young_track_id), int(candidate.old_track_id)
        if young in used_young or old in used_old:
            continue
        proposal = current.copy(); mask = (proposal == young) & (parent_frames >= int(candidate.frame))
        if not mask.any():
            continue
        proposal[mask] = old
        valid = True
        for frame in np.unique(parent_frames[mask]):
            frame_ids = proposal[parent_frames == frame]
            if len(frame_ids) != len(np.unique(frame_ids)):
                valid = False; break
        if not valid:
            continue
        metrics = prepared.evaluate_row_ids_incremental(proposal)
        gain = float(metrics["HOTA"] - current_metrics["HOTA"])
        if gain <= 0:
            continue
        current = proposal; current_metrics = metrics
        used_young.add(young); used_old.add(old)
        selected.append({
            "event_index": int(candidate.event_index), "frame": int(candidate.frame),
            "young_track_id": young, "old_track_id": old,
            "individual_delta_HOTA": float(candidate.delta_HOTA), "step_delta_HOTA": gain,
            "candidate_rank": int(candidate.candidate_rank), "young_purity": float(candidate.young_purity),
            "old_purity": float(candidate.old_purity), "modified_rows": int(mask.sum()),
        })
    selected_frame = pd.DataFrame(selected)
    selected_frame.to_csv(out / "selected_actions.csv", index=False)
    tracker_path = out / "track_results" / f"{seq}.txt"
    m28.write_tracker(tracker_path, prepared, current)
    m28.SEQ = seq
    official = m28.official_eval(tracker_path.parent, f"m28_a2_{seq}_teacher", out / "official_eval")
    # Attribute exact CLEAR switches to frozen runtime stages.
    updates = pd.read_parquet(root / "frozen_runtime/association_updates.parquet")
    grouped = {(int(f), int(t)): g for (f, t), g in updates.groupby(["frame", "track_id"], sort=False)}
    unconfirmed_switches = []
    for switch in switches.itertuples(index=False):
        g = grouped.get((int(switch.frame), int(switch.tracker_id)), pd.DataFrame())
        if len(g) and ((g.method.astype(str) == "update") & (g.stage.astype(str) == "unconfirmed")).any():
            unconfirmed_switches.append(switch)
    covered = 0
    for switch in unconfirmed_switches:
        if ((candidates.frame == int(switch.frame)) & (candidates.young_track_id == int(switch.tracker_id)) & (candidates.old_track_id == int(switch.previous_tracker_id))).any():
            covered += 1
    positive = int((success.delta_HOTA > 0).sum())
    report = {
        "experiment_id": "M28-A2", "seq": seq, "status": "completed", "teacher_only": True, "deployable": False,
        "gt_opened_after_candidate_freeze": True, "mot20_test_reads": 0, "test_submission": False,
        "baseline_metrics": {key: float(baseline_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "candidate_actions": len(candidates), "identity_consistent_actions": len(screened),
        "positive_identity_consistent_actions": positive, "selected_actions": len(selected),
        "combined_metrics": {key: float(current_metrics[key]) for key in ("HOTA", "DetA", "AssA")},
        "combined_delta_HOTA": float(current_metrics["HOTA"] - baseline_metrics["HOTA"]),
        "official_trackeval": official, "official_clear_idsw_baseline": int(len(switches)),
        "unconfirmed_source_idsw": len(unconfirmed_switches), "correct_predecessor_top8_coverage": covered,
        "correct_predecessor_top8_rate": covered / max(len(unconfirmed_switches), 1),
        "candidate_manifest": json.loads((root / "frozen_candidates/freeze_manifest.json").read_text()),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    pd.DataFrame([{
        "seq": seq, "baseline_HOTA": baseline_metrics["HOTA"], "candidate_actions": len(candidates),
        "identity_consistent_actions": len(screened), "positive_actions": positive, "selected_actions": len(selected),
        "teacher_HOTA": current_metrics["HOTA"], "delta_HOTA": report["combined_delta_HOTA"],
        "official_IDSW": official["IDSW"], "unconfirmed_IDSW": len(unconfirmed_switches),
        "predecessor_coverage": covered, "predecessor_rate": report["correct_predecessor_top8_rate"],
    }]).to_csv(root / "summary.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nTOP\n", success.head(30).to_string(index=False))
    print("\nSELECTED\n", selected_frame.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("reference", "freeze-runtime", "freeze-candidates", "teacher"))
    parser.add_argument("--seq", required=True, choices=SEQUENCES)
    args = parser.parse_args()
    seq_root(args.seq).mkdir(parents=True, exist_ok=True)
    if args.stage == "reference": make_reference(args.seq)
    elif args.stage == "freeze-runtime": freeze_runtime(args.seq)
    elif args.stage == "freeze-candidates": freeze_candidates(args.seq)
    else: teacher(args.seq)


if __name__ == "__main__":
    main()
