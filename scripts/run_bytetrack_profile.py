#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.util import load_super_config
from utils.misc import yaml_to_dict


PROFILE_META_KEYS = {"description", "settings", "manifest"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_profile_root() -> Path:
    return repo_root() / "configs" / "profiles"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_profile_path(profile_name: str, profile_root: str | None) -> Path:
    candidate = Path(profile_name)
    if candidate.is_file():
        return candidate.resolve()

    root = Path(profile_root) if profile_root else default_profile_root()
    for name in (f"{profile_name}.json", profile_name):
        path = root / name
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Unable to resolve profile '{profile_name}' under {root}")


def compute_file_md5(path_value: str | os.PathLike[str] | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_file():
        return ""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": "", "dirty": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=str(root),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        pass
    return info


def load_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    settings = payload.get("settings", payload)
    manifest = payload.get("manifest", {})
    if not isinstance(settings, dict):
        raise ValueError(f"Profile settings must be a dict: {path}")
    if not isinstance(manifest, dict):
        raise ValueError(f"Profile manifest must be a dict: {path}")
    return {
        "raw": payload,
        "settings": settings,
        "manifest": manifest,
        "description": payload.get("description", ""),
    }


def load_resolved_config(config_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    config = yaml_to_dict(str(config_path))
    if config.get("SUPER_CONFIG_PATH"):
        config = load_super_config(config, config["SUPER_CONFIG_PATH"])
    for key, value in overrides.items():
        config[str(key)] = value
    return config


def write_yaml(doc: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=False)


def write_json(doc: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_summary_txt(path: Path) -> dict[str, Any]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return {}
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values = lines[1].split()
    if len(headers) != len(values):
        return {"raw_header": lines[0], "raw_values": lines[1]}
    summary: dict[str, Any] = {}
    for key, raw in zip(headers, values):
        try:
            summary[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            summary[key] = float(raw)
            continue
        except ValueError:
            pass
        summary[key] = raw
    return summary


def detect_outputs(out_dir: Path) -> dict[str, Any]:
    summary_files = sorted(out_dir.glob("tracker/**/pedestrian_summary.txt"))
    detailed_files = sorted(out_dir.glob("tracker/**/pedestrian_detailed.csv"))
    diagnostics_path = out_dir / "local_conflict_graph_diagnostics.json"
    diagnostics_payload = {}
    if diagnostics_path.is_file():
        try:
            diagnostics_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            diagnostics_payload = {}
    latest_zip = out_dir / "latest_zip.txt"
    latest_zip_path = ""
    zip_md5 = ""
    if latest_zip.is_file():
        try:
            latest_zip_path = latest_zip.read_text(encoding="utf-8").strip()
        except Exception:
            latest_zip_path = ""
    if latest_zip_path:
        zip_md5 = compute_file_md5(latest_zip_path)
    return {
        "summary_path": str(summary_files[0]) if summary_files else "",
        "summary_metrics": parse_summary_txt(summary_files[0]) if summary_files else {},
        "detailed_path": str(detailed_files[0]) if detailed_files else "",
        "local_conflict_graph_diagnostics_path": str(diagnostics_path) if diagnostics_path.is_file() else "",
        "local_conflict_graph_diagnostics": diagnostics_payload,
        "latest_zip_txt": str(latest_zip) if latest_zip.is_file() else "",
        "zip_path": latest_zip_path,
        "zip_md5": zip_md5,
    }


def build_manifest(
    profile_name: str,
    profile_path: Path,
    profile_doc: dict[str, Any],
    resolved_config_path: Path,
    resolved_config: dict[str, Any],
    cmd: list[str],
    out_dir: Path,
    run_log: Path,
) -> dict[str, Any]:
    settings = profile_doc["settings"]
    config_path = Path(settings["config_path"]).resolve()
    inference_model = str(settings.get("inference_model", "") or "")
    external_det_root = str(resolved_config.get("EXTERNAL_DET_ROOT", "") or "")
    return {
        "created_at_utc": utc_now(),
        "runner": "scripts/run_bytetrack_profile.py",
        "repo_root": str(repo_root()),
        "git": git_metadata(repo_root()),
        "profile": {
            "name": profile_name,
            "path": str(profile_path),
            "description": profile_doc["description"],
            "manifest": profile_doc["manifest"],
        },
        "inputs": {
            "config_path": str(config_path),
            "config_md5": compute_file_md5(config_path),
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_md5": compute_file_md5(resolved_config_path),
            "inference_model": inference_model,
            "inference_model_md5": compute_file_md5(inference_model),
            "external_det_root": external_det_root,
        },
        "runtime": {
            "output_dir": str(out_dir),
            "run_log": str(run_log),
            "command": cmd,
        },
        "resolved_config": resolved_config,
        "status": "initialized",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run submit_bytetrack.py from a JSON runtime profile and save a manifest.")
    ap.add_argument("--exp-profile", required=True, help="Named JSON profile under configs/profiles or an explicit json path")
    ap.add_argument("--profile-root", default="", help="Optional profile root directory")
    ap.add_argument("--out-dir", default="", help="Optional output directory")
    ap.add_argument("--dry-run", action="store_true", help="Only materialize resolved config + manifest without launching")
    args = ap.parse_args()

    profile_path = resolve_profile_path(args.exp_profile, args.profile_root or None)
    profile_doc = load_profile(profile_path)
    settings = dict(profile_doc["settings"])

    config_path_raw = settings.get("config_path", "")
    if not config_path_raw:
        raise SystemExit(f"Profile '{args.exp_profile}' is missing settings.config_path")
    config_path = (repo_root() / config_path_raw).resolve() if not Path(config_path_raw).is_absolute() else Path(config_path_raw)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config_overrides = settings.get("config_overrides", {}) or {}
    if not isinstance(config_overrides, dict):
        raise ValueError("settings.config_overrides must be a dict")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_root() / "outputs" / f"{Path(args.exp_profile).stem}_{timestamp()}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    manifest_path = out_dir / "run_manifest.json"
    resolved_config_path = out_dir / "resolved_config.yaml"

    resolved_config = load_resolved_config(config_path, config_overrides)
    write_yaml(resolved_config, resolved_config_path)

    cmd = [
        sys.executable,
        "-u",
        str(repo_root() / "submit_bytetrack.py"),
        "--config-path",
        str(resolved_config_path),
        "--output-dir",
        str(out_dir),
    ]

    if settings.get("data_root"):
        cmd.extend(["--data-root", str(settings["data_root"])])
    if settings.get("inference_dataset"):
        cmd.extend(["--inference-dataset", str(settings["inference_dataset"])])
    if settings.get("inference_split"):
        cmd.extend(["--inference-split", str(settings["inference_split"])])
    if settings.get("inference_mode"):
        cmd.extend(["--inference-mode", str(settings["inference_mode"])])
    if settings.get("inference_model"):
        cmd.extend(["--inference-model", str(settings["inference_model"])])

    manifest = build_manifest(
        profile_name=args.exp_profile,
        profile_path=profile_path,
        profile_doc=profile_doc,
        resolved_config_path=resolved_config_path,
        resolved_config=resolved_config,
        cmd=cmd,
        out_dir=out_dir,
        run_log=run_log,
    )
    if args.dry_run:
        manifest["status"] = "dry_run"
        write_json(manifest, manifest_path)
        print(f"[DRY-RUN] profile={args.exp_profile}")
        print(f"[DRY-RUN] resolved_config={resolved_config_path}")
        print(f"[DRY-RUN] manifest={manifest_path}")
        print(f"[DRY-RUN] command={' '.join(cmd)}")
        return 0

    manifest["status"] = "running"
    manifest["started_at_utc"] = utc_now()
    write_json(manifest, manifest_path)

    with run_log.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(repo_root()), stdout=f, stderr=subprocess.STDOUT)

    manifest["finished_at_utc"] = utc_now()
    manifest["returncode"] = proc.returncode
    manifest["status"] = "succeeded" if proc.returncode == 0 else "failed"
    manifest["outputs"] = detect_outputs(out_dir)
    write_json(manifest, manifest_path)

    print(f"[INFO] manifest={manifest_path}")
    print(f"[INFO] run_log={run_log}")
    if manifest["outputs"].get("summary_path"):
        print(f"[INFO] summary={manifest['outputs']['summary_path']}")
    if manifest["outputs"].get("zip_path"):
        print(f"[INFO] zip={manifest['outputs']['zip_path']}")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
