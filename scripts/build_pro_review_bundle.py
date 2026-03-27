#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path


ROOT_FILES = (
    "AGENTS.md",
    "README.md",
    "HACA_v3_ATCR_codex_spec.md",
    "IMPLEMENT_HONEST_TOPK_FIRST.md",
    "RUNTIME_REPLAY_REFACTOR_PLAN.md",
    "codex_execution_plan_20260316.md",
    "requirements.txt",
    "requirements_dino.txt",
    "runtime_option.py",
    "train.py",
    "train_bytetrack.py",
    "submit_and_evaluate.py",
    "submit_bytetrack.py",
    "submit_public.py",
    "create_seqinfo.py",
    "migrate_v1_to_v2.py",
    "official_bytetrack_redesign_decision_20260325.md",
)

CODE_DIRS = (
    "models",
    "scripts",
    "configs",
    "md",
    "docs",
    "analysis",
)

PROMPT_FILES = (
    "md/PRO_REVIEW_CONTEXT_TEMPLATE.md",
    "md/PRO_REVIEW_CURRENT_MAINLINE_20260322.md",
    "md/PRO_REVIEW_CANONICAL_CONTEXT_20260324.md",
    "md/PRO_REVIEW_LATEST_DELTA_20260324.md",
    "md/PRO_REVIEW_LATEST_DELTA_20260325_BASELINE_PIVOT.md",
    "md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_BYTETRACK_STRICT_NEGATIVE.md",
    "md/PRO_REVIEW_LATEST_DELTA_20260325_OFFICIAL_DELTA_UTILITY_NOOP.md",
    "md/PRO_REVIEW_EXPERIMENT_CHAIN_INDEX_20260325.md",
    "md/PRO_REVIEW_INTERACTION_LOG.md",
    "md/PRO_REVIEW_INTERACTION_ENTRY_TEMPLATE.md",
    "md/PRO_REVIEW_REPLY_AFTER_LEARNED_COMMIT_QUEUE_20260324.md",
    "md/PRO_REVIEW_REPLY_AFTER_V15_NEGATIVE_20260324.md",
    "md/PRO_REVIEW_SEND_WITH_CONTEXT_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_NOW.md",
    "md/PRO_REVIEW_SEND_TO_PRO_FUTURE_DECISION.md",
    "md/PRO_REVIEW_SEND_TO_PRO_LOCAL_CONFLICT_GRAPH_REDESIGN_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_AFTER_LEARNED_COMMIT_QUEUE_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_AFTER_V15_HOST_MIGRATION_NEGATIVE_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_DECISION_AND_OPEN_REDESIGN_AFTER_V15_NEGATIVE_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260324.md",
    "md/PRO_REVIEW_SEND_TO_PRO_AFTER_STABLE_V2_BASE_RESULTS_20260325.md",
    "md/PRO_REVIEW_SEND_TO_PRO_BASELINE_SELECTION_AND_PAPER_PROTOCOL_20260325.md",
    "md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_BYTETRACK_STRICT_NEGATIVE_20260325.md",
    "md/PRO_REVIEW_SEND_TO_PRO_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md",
    "md/PRO_REVIEW_UPLOAD_LIST_AFTER_OFFICIAL_BYTETRACK_STRICT_NEGATIVE_20260325.md",
    "md/PRO_REVIEW_UPLOAD_LIST_AFTER_OFFICIAL_DELTA_UTILITY_NOOP_20260325.md",
    "md/PRO_REVIEW_REPLY_STRONGER_V2_WHILE_LARGEDATA_RUNNING_20260325.md",
)

THIRD_PARTY_COPY_MAP = {
    "third_party/ByteTrack/README.md": "code/third_party/ByteTrack/README.md",
    "third_party/ByteTrack/tools/track.py": "code/third_party/ByteTrack/tools/track.py",
    "third_party/ByteTrack/yolox/tracker/byte_tracker.py": "code/third_party/ByteTrack/yolox/tracker/byte_tracker.py",
    "third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py": "code/third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py",
    "third_party/ByteTrack/yolox/evaluators/mot_evaluator.py": "code/third_party/ByteTrack/yolox/evaluators/mot_evaluator.py",
    "third_party/ByteTrack/exps/example/mot/yolox_x_ablation.py": "code/third_party/ByteTrack/exps/example/mot/yolox_x_ablation.py",
    "third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py": "code/third_party/ByteTrack/exps/example/mot/yolox_x_mix_det.py",
    "third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py": "code/third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py",
    "third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_trainhalf_dump.py": "code/third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_trainhalf_dump.py",
}

CURRENT_MAINLINE_COPY_MAP = {
    "outputs/experiment_registry.csv": "evidence/current_mainline/experiment_registry.csv",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/competition_cases": "evidence/current_mainline/competition_cases",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.groups.jsonl": "evidence/current_mainline/labeled_replay_top8.groups.jsonl",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/labeled_replay_top8.recoverability.json": "evidence/current_mainline/labeled_replay_top8.recoverability.json",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/profile.json": "evidence/current_mainline/profile.json",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/resolved_config.yaml": "evidence/current_mainline/resolved_config.yaml",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/result.csv": "evidence/current_mainline/competition_assoc_base_reid_da_proxy0213_result.csv",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/summary.csv": "evidence/current_mainline/competition_assoc_base_reid_da_proxy0213_summary.csv",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/run_manifest.json": "evidence/current_mainline/run_manifest.json",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/pipeline.log": "evidence/current_mainline/pipeline.log",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/run.log": "evidence/current_mainline/run.log",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/tracker/MOT17-train/pedestrian_summary.txt": "evidence/current_mainline/tracker/pedestrian_summary.txt",
    "outputs/competition_assoc_base_reid_da_proxy0213_hybriddumpfix/tracker/MOT17-train/pedestrian_detailed.csv": "evidence/current_mainline/tracker/pedestrian_detailed.csv",
    "outputs/competition_assoc_stage1_fix1_full12/result.csv": "evidence/current_mainline/competition_assoc_stage1_result.csv",
    "outputs/competition_assoc_stage1_fix1_full12/summary.csv": "evidence/current_mainline/competition_assoc_stage1_summary.csv",
    "outputs/competition_assoc_stage1_fix1_full12/metrics.jsonl": "evidence/current_mainline/metrics.jsonl",
    "outputs/competition_assoc_stage1_fix1_full12/run.log": "evidence/current_mainline/competition_assoc_stage1_run.log",
    "outputs/competition_assoc_stage1_fix1_full12/best.pt": "evidence/current_mainline/competition_assoc_stage1_best.pt",
}

CONTROL_CHAIN_COPY_MAP = {
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_spatial_val0213/result.csv": "evidence/control_chain/base_spatial/result.csv",
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_spatial_val0213/val": "evidence/control_chain/base_spatial/val",
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/result.csv": "evidence/control_chain/base_reid_da/result.csv",
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_base_reid_da_val0213/val": "evidence/control_chain/base_reid_da/val",
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_full_reid_da_val0213/result.csv": "evidence/control_chain/full_reid_da/result.csv",
    "outputs/paper_ctrl_mot17_val0213/bytetrack_fa_mot_mot17_v14_ctrl_full_reid_da_val0213/val/epoch_0": "evidence/control_chain/full_reid_da/val/epoch_0",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a compact Pro review bundle for the current FM-Track mainline.")
    ap.add_argument("--tag", default="pro_future_decision_bundle")
    ap.add_argument("--timestamp", default="")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--extra-evidence", nargs="*", default=[])
    ap.add_argument("--latest-run-root", default="")
    ap.add_argument("--latest-run-label", default="latest_run")
    ap.add_argument("--skip-archives", action="store_true")
    return ap.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def _copy_if_exists(
    *,
    src: Path,
    dst: Path,
    copied: list[str],
    missing: list[str],
    label: str,
) -> None:
    if src.exists():
        _copy_path(src, dst)
        copied.append(label)
    else:
        missing.append(label)


def _remove_code_artifacts(code_root: Path) -> None:
    for path in list(code_root.rglob("*")):
        if not path.exists():
            continue
        if path.is_dir() and path.name in {"__pycache__", ".ipynb_checkpoints", "build"}:
            shutil.rmtree(path)
            continue
        if path.is_file() and path.suffix in {".pyc", ".pyo", ".so", ".o"}:
            path.unlink()
            continue
        if path.is_file() and path.name in {".ninja_deps", ".ninja_log"}:
            path.unlink()


def _write_bundle_readme(bundle_dir: Path) -> None:
    text = """# Pro Review Bundle

This bundle is prepared for external review of the current FM-Track local-conflict operator mainline.

## Included

- `code/`
  - first-party source snapshot
  - relevant official ByteTrack anchor files used by the strict paper-baseline path
- `evidence/current_mainline/`
  - historical internal-host current-mainline evidence
- `evidence/control_chain/`
  - internal host control evidence
- `evidence/extras/`
  - explicitly requested run roots and structured results
- `evidence/latest_run/`
  - the current focal run and its structured artifacts
- `prompt/`
  - ready-to-send prompts for Pro review

## Intentionally Excluded

- heavyweight checkpoints not needed for review
- runtime tensor shard `.npz` dumps
- large runtime dump CSVs
- historical bulk outputs unrelated to the current mainline
- external vendor code directories
- datasets, caches, and build artifacts

## Current Problem Shape

- paper carrier is now `official ByteTrack`
- historical row-local controller line is dead
- full cluster replacement line is dead
- current method family is `local conflict set predictor`
- current key question is no longer baseline selection
- current key question is how to redesign the operator after strict official-ByteTrack negative evidence
"""
    (bundle_dir / "README_BUNDLE.md").write_text(text, encoding="utf-8")


def _write_manifest(bundle_dir: Path, manifest: dict) -> None:
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_latest_run_artifacts(
    *,
    repo_root: Path,
    bundle_dir: Path,
    latest_run_root: Path,
    latest_run_label: str,
    copied: list[str],
    missing: list[str],
) -> dict:
    run_root = latest_run_root.resolve()
    label = latest_run_label.strip() or "latest_run"
    latest_dir = bundle_dir / "evidence" / "latest_run"
    latest_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "label": label,
        "run_root": str(run_root),
        "copied_entries": [],
        "missing_entries": [],
    }
    if not run_root.exists():
        missing.append(f"{label}:{run_root}")
        manifest["missing_entries"] = [str(run_root)]
        return manifest

    direct_names = (
        "result.csv",
        "summary.csv",
        "metrics.csv",
        "metrics.jsonl",
        "run.log",
        "pipeline.log",
        "profile.json",
        "resolved_config.yaml",
        "resolved_config.yml",
        "resolved_config.json",
        "run_manifest.json",
        "diagnostics.json",
        "pro_review_bundle.json",
        "best.pt",
    )
    for name in direct_names:
        src = run_root / name
        dst = latest_dir / name
        tag = f"{label}:{name}"
        if src.exists():
            _copy_path(src, dst)
            copied.append(tag)
            manifest["copied_entries"].append(name)
        else:
            manifest["missing_entries"].append(name)

    for src in sorted(run_root.glob("*.metrics.jsonl")):
        rel = src.name
        _copy_path(src, latest_dir / rel)
        copied.append(f"{label}:{rel}")
        manifest["copied_entries"].append(rel)

    structured_subpaths = (
        ("competition_cases/summary.json", "competition_cases/summary.json"),
        ("competition_cases/sequence_summary.csv", "competition_cases/sequence_summary.csv"),
        ("competition_cases/competition_cases.csv", "competition_cases/competition_cases.csv"),
        ("cluster_commit_data/summary.json", "cluster_commit_data/summary.json"),
        ("cluster_commit_data/cluster_summary.csv", "cluster_commit_data/cluster_summary.csv"),
        ("cluster_commit_data/sequence_cluster_summary.csv", "cluster_commit_data/sequence_cluster_summary.csv"),
        ("cluster_commit_data/cluster_examples.sample.jsonl", "cluster_commit_data/cluster_examples.sample.jsonl"),
        ("cluster_set_predictor_data/summary.json", "cluster_set_predictor_data/summary.json"),
        ("cluster_set_predictor_data/cluster_summary.csv", "cluster_set_predictor_data/cluster_summary.csv"),
        ("cluster_set_predictor_data/sequence_cluster_summary.csv", "cluster_set_predictor_data/sequence_cluster_summary.csv"),
        ("cluster_set_predictor_data/cluster_examples.sample.jsonl", "cluster_set_predictor_data/cluster_examples.sample.jsonl"),
        ("local_conflict_graph_diagnostics.json", "local_conflict_graph_diagnostics.json"),
        ("tracker/MOT17-train/pedestrian_summary.txt", "tracker/MOT17-train/pedestrian_summary.txt"),
        ("tracker/MOT17-train/pedestrian_detailed.csv", "tracker/MOT17-train/pedestrian_detailed.csv"),
    )
    for rel_src, rel_dst in structured_subpaths:
        src = run_root / rel_src
        tag = f"{label}:{rel_src}"
        if src.exists():
            _copy_path(src, latest_dir / rel_dst)
            copied.append(tag)
            manifest["copied_entries"].append(rel_src)

    for src in sorted(run_root.glob("val/epoch_*/diagnostics.json")):
        rel = src.relative_to(run_root)
        _copy_path(src, latest_dir / rel)
        copied.append(f"{label}:{rel.as_posix()}")
        manifest["copied_entries"].append(rel.as_posix())
    for src in sorted(run_root.glob("val/epoch_*/eval.log")):
        rel = src.relative_to(run_root)
        _copy_path(src, latest_dir / rel)
        copied.append(f"{label}:{rel.as_posix()}")
        manifest["copied_entries"].append(rel.as_posix())

    if run_root.is_relative_to(repo_root):
        manifest["run_root_relative"] = str(run_root.relative_to(repo_root))
    return manifest


def _archive_bundle(bundle_dir: Path) -> tuple[Path, Path]:
    tar_path = bundle_dir.with_suffix(".tar.gz")
    zip_path = bundle_dir.with_suffix(".zip")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(bundle_dir, arcname=bundle_dir.name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_dir.rglob("*"):
            zf.write(path, arcname=str(path.relative_to(bundle_dir.parent)))
    return tar_path, zip_path


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    out_root = (repo_root / args.out_root).resolve()
    timestamp = args.timestamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{args.tag}_{timestamp}"
    bundle_dir = out_root / bundle_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)

    missing: list[str] = []
    copied: list[str] = []

    (bundle_dir / "code").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "evidence" / "current_mainline").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "evidence" / "control_chain").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "evidence" / "extras").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "evidence" / "latest_run").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "prompt").mkdir(parents=True, exist_ok=True)

    for rel in ROOT_FILES:
        src = repo_root / rel
        if src.exists():
            _copy_path(src, bundle_dir / "code" / rel)
            copied.append(rel)
        else:
            missing.append(rel)

    for rel in CODE_DIRS:
        src = repo_root / rel
        if src.exists():
            _copy_path(src, bundle_dir / "code" / rel)
            copied.append(rel)
        else:
            missing.append(rel)
    _remove_code_artifacts(bundle_dir / "code")

    for rel in PROMPT_FILES:
        src = repo_root / rel
        if src.exists():
            _copy_path(src, bundle_dir / "prompt" / src.name)
            copied.append(rel)
        else:
            missing.append(rel)

    for rel_src, rel_dst in THIRD_PARTY_COPY_MAP.items():
        src = repo_root / rel_src
        dst = bundle_dir / rel_dst
        if src.exists():
            _copy_path(src, dst)
            copied.append(rel_src)
        else:
            missing.append(rel_src)

    for rel_src, rel_dst in CURRENT_MAINLINE_COPY_MAP.items():
        src = repo_root / rel_src
        dst = bundle_dir / rel_dst
        if src.exists():
            _copy_path(src, dst)
            copied.append(rel_src)
        else:
            missing.append(rel_src)

    for rel_src, rel_dst in CONTROL_CHAIN_COPY_MAP.items():
        src = repo_root / rel_src
        dst = bundle_dir / rel_dst
        if src.exists():
            _copy_path(src, dst)
            copied.append(rel_src)
        else:
            missing.append(rel_src)

    for extra in args.extra_evidence:
        src = (repo_root / extra).resolve()
        dst = bundle_dir / "evidence" / "extras" / src.name
        if src.exists():
            _copy_path(src, dst)
            copied.append(str(src.relative_to(repo_root)))
        else:
            missing.append(extra)

    latest_run_manifest = None
    if args.latest_run_root.strip():
        latest_run_manifest = _copy_latest_run_artifacts(
            repo_root=repo_root,
            bundle_dir=bundle_dir,
            latest_run_root=(repo_root / args.latest_run_root).resolve()
            if not Path(args.latest_run_root).is_absolute()
            else Path(args.latest_run_root).resolve(),
            latest_run_label=args.latest_run_label,
            copied=copied,
            missing=missing,
        )

    _write_bundle_readme(bundle_dir)

    manifest = {
        "bundle_name": bundle_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "copied_entries": copied,
        "missing_entries": missing,
        "extra_evidence": list(args.extra_evidence),
        "latest_run": latest_run_manifest,
    }

    tar_path = None
    zip_path = None
    if not args.skip_archives:
        tar_path, zip_path = _archive_bundle(bundle_dir)
        manifest["tar_gz"] = str(tar_path)
        manifest["zip"] = str(zip_path)

    _write_manifest(bundle_dir, manifest)
    latest_txt = out_root / "latest_pro_review_bundle.txt"
    latest_txt.write_text(str(bundle_dir) + "\n", encoding="utf-8")

    summary = {
        "bundle_dir": str(bundle_dir),
        "tar_gz": str(tar_path) if tar_path else "",
        "zip": str(zip_path) if zip_path else "",
        "latest_run_root": str(latest_run_manifest.get("run_root", "")) if latest_run_manifest else "",
        "missing_count": len(missing),
        "missing_entries": missing,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
