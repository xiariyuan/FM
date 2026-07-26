from __future__ import annotations

"""M23-70B0: GT-free uniform K=3/D8 causal lattice capacity audit.

Nodes and candidate edges are generated without GT. GT is opened only after
node/edge hashes are frozen, solely to measure maximum-weight path-cover
capacity. This experiment trains no model and never reads MOT20 test.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("outputs/mot20_m23_20260718/m23_70b0_uniform_causal_lattice_capacity_v1")
SEQUENCES = ("MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
PREREG_SHA256 = "a09b277609e27f972cc199f9dfa2a2d59869253e97b16dd2410b85233be678fa"
M46_REPORT = Path("outputs/mot20_m23_20260718/m23_46_pure_hgb_topk_combined_oof_v1/report.json")
M57_ROOT = Path("outputs/mot20_m23_20260718/m23_57_intra_node_change_point_capacity_v2")
SOURCE_PARENT = Path("outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/track_results")
BASELINE_CACHE = Path("outputs/mot20_m23_20260718/m23_47_m23_46_deployable_baseline_cache_v1")
SEGMENT_ROWS = 4
DELAY = 8
K = 3
ALTERNATIVES = 2
GO_HOTA = 80.80


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def event(root: Path, name: str, **payload: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "protocol_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), "event": name, **payload}, sort_keys=True) + "\n")


def load_module(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def verify_prereg(root: Path) -> dict[str, Any]:
    path = root / "preregistered_protocol.json"
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha(path)
    if actual != PREREG_SHA256:
        raise RuntimeError(f"preregistration hash mismatch: {actual} != {PREREG_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixed = payload["fixed"]
    if fixed["segment_rows"] != SEGMENT_ROWS or fixed["delay_frames"] != DELAY:
        raise RuntimeError("frozen protocol constants do not match implementation")
    if fixed["branch_count"] != K or fixed["alternatives_per_source"] != ALTERNATIVES:
        raise RuntimeError("frozen branch budget mismatch")
    return payload


def freeze_inputs(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    output = root / "input_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    required = [
        M46_REPORT,
        M57_ROOT / "capacity_combined/report.json",
        Path("scripts/m23_research/m23_70a_causal_branch_capacity.py"),
        Path("scripts/m23_research/m23_57_intra_node_change_point_capacity.py"),
        Path("scripts/m23_research/m23_53_global_identity_flow_capacity.py"),
        Path("scripts/m23_research/m23_53b_build_adaptive_micrograph.py"),
        Path("scripts/m23_research/m23_45_domain_robust_source_cut_student.py"),
    ]
    for seq in SEQUENCES:
        required += [
            SOURCE_PARENT / f"{seq}.txt",
            BASELINE_CACHE / seq / "track_results" / f"{seq}.txt",
            M57_ROOT / f"boundary_universe/{seq}/freeze_manifest.json",
            M57_ROOT / f"boundary_universe/{seq}/chunk_membership.parquet",
        ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    baseline = json.loads(M46_REPORT.read_text(encoding="utf-8"))
    payload = {
        "experiment_id": "M23-70B0",
        "created_at": now(),
        "git_head": git_head(),
        "m23_46": baseline["metrics"],
        "artifacts": {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in required},
        "training_runs": 0,
        "mot20_test_reads": 0,
    }
    write_json(output, payload)
    event(root, "inputs_frozen", manifest_sha256=sha(output))
    return payload


def freeze_implementation(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    if not (root / "input_manifest.json").exists():
        raise RuntimeError("freeze inputs first")
    output = root / "implementation_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    script = Path("scripts/m23_research/m23_70b0_uniform_causal_lattice_capacity.py")
    test = Path("scripts/m23_research/test_m23_70b0_uniform_causal_lattice_capacity.py")
    payload = {
        "experiment_id": "M23-70B0",
        "created_at": now(),
        "git_head": git_head(),
        "preregistration_sha256": PREREG_SHA256,
        "input_manifest_sha256": sha(root / "input_manifest.json"),
        "script_sha256": sha(script),
        "test_sha256": sha(test),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "execution_device": "cpu",
            "cuda_probe_executed": False,
            "gpu": None,
            "cpu_count": os.cpu_count(),
        },
        "fixed": {"segment_rows": SEGMENT_ROWS, "delay_frames": DELAY, "branch_count": K},
        "training_runs": 0,
    }
    write_json(output, payload)
    event(root, "implementation_frozen", manifest_sha256=sha(output))
    return payload


def uniform_cuts(chunk_rows: dict[int, list[int]]) -> tuple[dict[int, list[int]], dict[str, Any]]:
    cuts: dict[int, list[int]] = {}
    total = 0
    for chunk_id, rows in sorted(chunk_rows.items()):
        positions = list(range(SEGMENT_ROWS, len(rows), SEGMENT_ROWS))
        if positions:
            cuts[int(chunk_id)] = positions
            total += len(positions)
    return cuts, {
        "split_generation": "uniform_observation_stride",
        "segment_rows": SEGMENT_ROWS,
        "chunks_with_cut": len(cuts),
        "candidate_cuts": total,
        "gt_used": False,
    }


def run_sequence(seq: str, root: Path, skip_trackeval: bool = False) -> dict[str, Any]:
    verify_prereg(root)
    if not (root / "input_manifest.json").exists() or not (root / "implementation_manifest.json").exists():
        raise RuntimeError("input and implementation freezes are required")
    output = root / "capacity" / seq
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.time()
    m70a = load_module(f"m70b0_m70a_{seq[-2:]}", "scripts/m23_research/m23_70a_causal_branch_capacity.py")
    m57 = load_module(f"m70b0_m57_{seq[-2:]}", "scripts/m23_research/m23_57_intra_node_change_point_capacity.py")
    m53 = load_module(f"m70b0_m53_{seq[-2:]}", "scripts/m23_research/m23_53_global_identity_flow_capacity.py")
    (
        _m10, _m53_from_57, m53b, source_path, baseline_path, source_rows,
        fixed_nodes, fixed_chunk_rows, parent_ids, crowd, mapped, _match_iou, row_embeddings,
    ) = m57.prepare_observable_rows(seq)

    cuts, split_report = uniform_cuts(fixed_chunk_rows)
    nodes, _ = m53b.build_adaptive_nodes(
        source_rows=source_rows,
        fixed_nodes=fixed_nodes,
        chunk_rows=fixed_chunk_rows,
        selected_boundaries=cuts,
        row_embeddings=row_embeddings,
        mapped=mapped,
        parent_ids=parent_ids,
        crowd_density=crowd,
    )
    rows_by_node = m70a.adaptive_row_map(fixed_nodes, fixed_chunk_rows, cuts, nodes)
    edges, branches, candidate_report = m70a.causal_candidate_edges(
        nodes, rows_by_node, source_rows, row_embeddings, mapped
    )

    frozen = output / "frozen_candidate_graph"
    frozen.mkdir(parents=True, exist_ok=True)
    nodes_path = frozen / "nodes.parquet"
    edges_path = frozen / "edges.parquet"
    branch_path = frozen / "causal_branch_table.parquet"
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    branches.to_parquet(branch_path, index=False)

    empty = pd.DataFrame(columns=["src_chunk", "dst_chunk"])
    reconstructed = output / "baseline_reconstruction/track_results" / f"{seq}.txt"
    baseline_report = m53.write_tracker(seq, source_path, nodes, empty, reconstructed, preserve_parent_ids=True)
    baseline_exact = reconstructed.read_bytes() == baseline_path.read_bytes()
    if not baseline_exact:
        raise RuntimeError(f"{seq}: parent-ID reconstruction is not byte-exact M23-46")

    graph_manifest = {
        "experiment_id": "M23-70B0",
        "seq": seq,
        "teacher_only": True,
        "deployable": False,
        "gt_free_node_and_candidate_generation": True,
        "candidate_graph_frozen_before_path_cover_teacher": True,
        "input_manifest_sha256": sha(root / "input_manifest.json"),
        "implementation_manifest_sha256": sha(root / "implementation_manifest.json"),
        "split_report": split_report,
        "candidate_report": candidate_report,
        "baseline_reconstruction": {**baseline_report, "byte_exact": baseline_exact},
        "frozen_artifacts": {
            "nodes": str(nodes_path), "nodes_sha256": sha(nodes_path), "node_rows": len(nodes),
            "edges": str(edges_path), "edges_sha256": sha(edges_path), "edge_rows": len(edges),
            "branch_table": str(branch_path), "branch_table_sha256": sha(branch_path),
        },
        "protocol": {
            "segment_rows": SEGMENT_ROWS,
            "branch_count": K,
            "alternatives_per_source": ALTERNATIVES,
            "delay_frames": DELAY,
            "complete_future_destination_descriptor_used": False,
            "teacher_objective": "M23-53 maximum-weight path cover over frozen GT-free lattice",
        },
    }
    manifest_path = frozen / "freeze_manifest.json"
    write_json(manifest_path, graph_manifest)
    event(root, "gt_free_candidate_graph_frozen", seq=seq, manifest_sha256=sha(manifest_path))

    nodes, selected, teacher_report = m53.build_teacher_utilities(
        seq=seq, source_parent_root=SOURCE_PARENT, output_root=output, freeze_manifest=graph_manifest
    )
    tracker = output / "track_results" / f"{seq}.txt"
    tracker_report = m53.write_tracker(seq, source_path, nodes, selected, tracker)
    payload_unchanged = m57.detector_payload(source_path) == m57.detector_payload(tracker)
    if not payload_unchanged:
        raise RuntimeError(f"{seq}: detector payload changed")
    official = None if skip_trackeval else m53.run_official_trackeval(
        seq=seq, output_root=output, tracker_name=f"m23_70b0_{seq[-2:]}"
    )
    report = {
        "experiment_id": "M23-70B0",
        "seq": seq,
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "gt_free_node_and_candidate_generation": True,
        "split_report": split_report,
        "candidate_report": candidate_report,
        "teacher": teacher_report,
        "integrity": {
            "baseline_m23_46_byte_exact": baseline_exact,
            "detection_payload_unchanged": payload_unchanged,
            "one_to_one": bool(teacher_report["one_to_one"]),
            "time_forward": bool(teacher_report["time_forward"]),
            "acyclic": bool(teacher_report["acyclic"]),
            "complete_future_destination_descriptor_used": False,
            "gt_free_node_and_candidate_generation": True,
        },
        "official_trackeval": official,
        "tracker": {**tracker_report, "sha256": sha(tracker)},
        "artifacts": {"candidate_manifest_sha256": sha(manifest_path), "tracker_sha256": sha(tracker)},
        "runtime_seconds": time.time() - started,
        "peak_rss_mb": peak_rss_mb(),
        "peak_gpu_memory_mb": 0.0,
    }
    write_json(report_path, report)
    event(root, "sequence_completed", seq=seq, HOTA=None if official is None else official["HOTA"])
    return report


def combine(root: Path) -> dict[str, Any]:
    verify_prereg(root)
    combined_root = root / "capacity_combined"
    report_path = combined_root / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    baseline = json.loads(M46_REPORT.read_text(encoding="utf-8"))
    track_results = combined_root / "track_results"
    track_results.mkdir(parents=True, exist_ok=True)
    fold_reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for seq in SEQUENCES:
        path = root / "capacity" / seq / "report.json"
        if not path.exists():
            raise FileNotFoundError(path)
        fold_reports[seq] = json.loads(path.read_text(encoding="utf-8"))
        src = root / "capacity" / seq / "track_results" / f"{seq}.txt"
        dst = track_results / f"{seq}.txt"
        shutil.copy2(src, dst)
        hashes[f"{seq}.txt"] = sha(dst)
    evaluator = load_module("m70b0_combined_eval", "scripts/m23_research/m23_45_domain_robust_source_cut_student.py")
    metrics = evaluator.evaluate_detailed(track_results, combined_root / "official_eval", "m23_70b0_uniform_lattice_capacity", SEQUENCES)
    non_degradation = {seq: metrics[seq]["HOTA"] >= baseline["folds"][seq]["HOTA"] for seq in SEQUENCES}
    integrity = {
        seq: all([
            fold_reports[seq]["integrity"]["baseline_m23_46_byte_exact"],
            fold_reports[seq]["integrity"]["detection_payload_unchanged"],
            fold_reports[seq]["integrity"]["one_to_one"],
            fold_reports[seq]["integrity"]["time_forward"],
            fold_reports[seq]["integrity"]["acyclic"],
            fold_reports[seq]["integrity"]["gt_free_node_and_candidate_generation"],
            not fold_reports[seq]["integrity"]["complete_future_destination_descriptor_used"],
        ]) for seq in SEQUENCES
    }
    combined = metrics["COMBINED"]
    passed = bool(combined["HOTA"] >= GO_HOTA and all(non_degradation.values()) and all(integrity.values()))
    decision = "PASS_GT_FREE_LATTICE_CAPACITY_AUTHORIZE_M23_70B1" if passed else "FAIL_GT_FREE_LATTICE_CAPACITY_DO_NOT_TRAIN"
    delta = {
        scope: {
            key: metrics[scope][key] - (baseline["metrics"][key] if scope == "COMBINED" else baseline["folds"][scope][key])
            for key in ("HOTA", "DetA", "AssA", "IDSW")
        } for scope in (*SEQUENCES, "COMBINED")
    }
    payload = {
        "experiment_id": "M23-70B0",
        "status": "completed",
        "teacher_only": True,
        "deployable": False,
        "gt_free_node_and_candidate_generation": True,
        "official_trackeval": metrics,
        "baseline_m23_46": baseline,
        "delta_vs_m23_46": delta,
        "capacity_gate": {
            "threshold_HOTA": GO_HOTA,
            "combined_hota_pass": combined["HOTA"] >= GO_HOTA,
            "per_sequence_non_degradation": non_degradation,
            "integrity": integrity,
            "pass": passed,
            "margin": combined["HOTA"] - GO_HOTA,
        },
        "decision": decision,
        "m23_70b1_authorized": passed,
        "training_runs": 0,
        "mot20_test_reads": 0,
        "test_submission": False,
        "trackeval_runs": 5,
        "tracker_sha256": hashes,
    }
    write_json(report_path, payload)
    rows = [{
        "experiment_id": "M23-70B0", "scope": scope,
        **{key: metrics[scope][key] for key in ("HOTA", "DetA", "AssA", "IDSW")},
        "delta_hota_vs_m23_46": delta[scope]["HOTA"],
        "teacher_only": 1, "deployable": 0, "gt_free_lattice": 1,
        "gate_pass": int(passed), "decision": decision,
    } for scope in (*SEQUENCES, "COMBINED")]
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    write_json(root / "final_summary.json", {
        "experiment_id": "M23-70B0", "status": "completed", "decision": decision,
        "hota": combined["HOTA"], "teacher_only": True, "deployable": False,
        "gt_free_node_and_candidate_generation": True, "m23_70b1_authorized": passed,
        "training_runs": 0, "trackeval_runs": 5, "mot20_test_reads": 0,
        "test_submission": False, "capacity_gate": payload["capacity_gate"],
    })
    lines = [
        "# M23-70B0 GT-Free Uniform Causal Lattice Capacity — Result", "",
        f"Decision: **{decision}**", "",
        "Node and candidate generation are GT-free; edge selection remains teacher-only capacity.", "",
        "| Scope | HOTA | DetA | AssA | IDSW | ΔHOTA vs M23-46 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in (*SEQUENCES, "COMBINED"):
        lines.append(f"| {scope} | {metrics[scope]['HOTA']:.6f} | {metrics[scope]['DetA']:.6f} | {metrics[scope]['AssA']:.6f} | {int(metrics[scope]['IDSW'])} | {delta[scope]['HOTA']:+.6f} |")
    lines += ["", f"- Uniform segment rows: `{SEGMENT_ROWS}`.", f"- K/D: `{K}/{DELAY}`.", f"- Gate: `{GO_HOTA:.2f}` HOTA.", f"- Combined HOTA: `{combined['HOTA']:.6f}`.", f"- M23-70B1 authorized: `{str(passed).lower()}`.", "- MOT20 test submission: `false`."]
    doc = Path("docs/m23_70b0_uniform_causal_lattice_capacity_result_20260725.md")
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    event(root, "combined_completed", HOTA=combined["HOTA"], decision=decision, result_doc_sha256=sha(doc))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-inputs")
    sub.add_parser("freeze-implementation")
    fold = sub.add_parser("run-sequence")
    fold.add_argument("--seq", required=True, choices=SEQUENCES)
    fold.add_argument("--skip-trackeval", action="store_true")
    sub.add_parser("combine")
    args = parser.parse_args()
    if args.command == "freeze-inputs":
        result = freeze_inputs(args.root)
    elif args.command == "freeze-implementation":
        result = freeze_implementation(args.root)
    elif args.command == "run-sequence":
        result = run_sequence(args.seq, args.root, args.skip_trackeval)
    elif args.command == "combine":
        result = combine(args.root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
