#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shutil
import json
import yaml

import pandas as pd
import matplotlib.pyplot as plt


def parse_summary(path: Path):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    headers = re.split(r"\s+", lines[0].strip())
    values = re.split(r"\s+", lines[1].strip())
    if len(headers) != len(values):
        # try to recover by trimming
        n = min(len(headers), len(values))
        headers = headers[:n]
        values = values[:n]
    data = {}
    for k, v in zip(headers, values):
        try:
            data[k] = float(v)
        except Exception:
            data[k] = v
    return data


def find_exp_and_epoch(path: Path):
    parts = list(path.parts)
    exp = None
    epoch = None
    dataset = None

    # experiment name: outputs/<exp>/...
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            exp = parts[idx + 1]

    # epoch from path: epoch_XX
    for p in parts:
        m = re.match(r"epoch_(\d+)", p)
        if m:
            epoch = int(m.group(1))
            break

    # dataset name heuristic
    for p in parts:
        if "MOT" in p or "DanceTrack" in p or "MOTS" in p:
            dataset = p
            break

    return exp, epoch, dataset


def find_config_for_exp(exp_dir: Path):
    candidates = [
        exp_dir / "config_effective.yaml",
        exp_dir / "config.yaml",
        exp_dir / "train" / "config_effective.yaml",
        exp_dir / "train" / "config.yaml",
        exp_dir / "train" / "config.yml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-dir", default="analysis/ablation")
    parser.add_argument("--metric", default="HOTA")
    parser.add_argument("--exp-include", default=None, help="substring to filter exp names")
    parser.add_argument("--dataset-include", default=None, help="substring to filter dataset names")
    parser.add_argument("--best-config-out", default="configs/bytetrack_fa_mot_mot17_best.yaml")
    parser.add_argument("--repro-out", default="docs/REPRODUCIBILITY.md")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_files = list(outputs_dir.rglob("pedestrian_summary.txt"))
    rows = []
    for f in summary_files:
        data = parse_summary(f)
        if data is None:
            continue
        exp, epoch, dataset = find_exp_and_epoch(f)
        if args.exp_include and exp and args.exp_include not in exp:
            continue
        if args.dataset_include and dataset and args.dataset_include not in dataset:
            continue
        row = {
            "exp": exp or "unknown",
            "epoch": epoch,
            "dataset": dataset or "unknown",
            "path": str(f),
        }
        row.update(data)
        rows.append(row)

    if not rows:
        raise SystemExit("No pedestrian_summary.txt found with valid metrics.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_all.csv", index=False)

    # Best per exp by metric
    metric = args.metric
    if metric not in df.columns:
        raise SystemExit(f"Metric {metric} not found in summary files.")

    # drop rows with nan metric
    df_metric = df.dropna(subset=[metric])
    best_rows = (
        df_metric.sort_values(metric, ascending=False)
        .groupby("exp", as_index=False)
        .first()
    )

    best_rows.to_csv(out_dir / "ablation_table.csv", index=False)
    best_rows.to_markdown(out_dir / "ablation_table.md", index=False)

    # Plots: best bar chart
    plt.figure(figsize=(10, max(3, 0.3 * len(best_rows))))
    best_rows = best_rows.sort_values(metric, ascending=True)
    plt.barh(best_rows["exp"].astype(str), best_rows[metric].astype(float))
    plt.title(f"Best {metric} per experiment")
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(out_dir / f"best_{metric.lower()}_bar.png", dpi=150)
    plt.close()

    # Curves per exp for metric (if epochs exist)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    for exp_name, g in df_metric.groupby("exp"):
        if g["epoch"].notna().sum() == 0:
            continue
        g2 = g.dropna(subset=["epoch"]).sort_values("epoch")
        if g2.empty:
            continue
        plt.figure(figsize=(8, 4))
        plt.plot(g2["epoch"], g2[metric], marker="o")
        plt.title(f"{exp_name} {metric} vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(curves_dir / f"{exp_name}_{metric.lower()}.png", dpi=150)
        plt.close()

    # Determine overall best
    best_overall = df_metric.sort_values(metric, ascending=False).iloc[0]
    best_exp = best_overall["exp"]
    best_epoch = best_overall.get("epoch")
    best_path = Path(str(best_overall["path"]))
    exp_dir = outputs_dir / best_exp

    # Copy best config
    best_config = find_config_for_exp(exp_dir)
    best_config_out = Path(args.best_config_out)
    if best_config is not None:
        best_config_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_config, best_config_out)

    # Write a small summary json
    summary = {
        "metric": metric,
        "best_exp": str(best_exp),
        "best_epoch": int(best_epoch) if pd.notna(best_epoch) else None,
        "best_value": float(best_overall[metric]),
        "best_summary_path": str(best_path),
        "best_config_src": str(best_config) if best_config is not None else None,
        "best_config_out": str(best_config_out) if best_config is not None else None,
    }
    (out_dir / "best_summary.json").write_text(json.dumps(summary, indent=2))

    # Reproducibility doc (auto-generated)
    repro_out = Path(args.repro_out)
    repro_out.parent.mkdir(parents=True, exist_ok=True)
    cfg_data = {}
    try:
        if best_config_out.exists():
            cfg_data = yaml.safe_load(best_config_out.read_text()) or {}
        elif best_config is not None:
            cfg_data = yaml.safe_load(Path(best_config).read_text()) or {}
    except Exception:
        cfg_data = {}

    repro_text = """# Reproducibility (Auto-Generated)

## Best Checkpoint Selection
- Metric: {metric}
- Best experiment: {best_exp}
- Best epoch: {best_epoch}
- Best {metric}: {best_value:.4f}
- Summary file: {best_summary_path}

## Config
- Source: {best_config_src}
- Saved as: {best_config_out}

## Dataset & Split
- DATA_ROOT: {data_root}
- DATASETS: {datasets}
- DATASET_SPLITS: {splits}
- DETECTOR_FILTER: {det_filter}
- VAL_SEQUENCES: {val_seq}

## Model & Training
- SEED: {seed}
- FEATURE_DIM: {feat_dim}
- NUM_BANDS: {num_bands}
- MAX_SEQ_LEN: {max_seq}
- EPOCHS: {epochs}
- BATCH_SIZE: {batch_size}
- LR: {lr}
- WEIGHT_DECAY: {wd}
- USE_FREQ_AWARE: {use_freq}
- USE_FREQ_DECODER_V2: {use_decoder}

## Commands
- Train:
  ```bash
  cd /gemini/code/FMtrack-main/FM-Track
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/miniconda3/bin/python -u train_bytetrack.py \
    --config-path {best_config_out}
  ```
- Evaluate:
  ```bash
  cd /gemini/code/FMtrack-main/FM-Track
  /root/miniconda3/bin/python -u submit_bytetrack.py \
    --config-path {best_config_out}
  ```
""".format(
        metric=metric,
        best_exp=best_exp,
        best_epoch=best_epoch,
        best_value=float(best_overall[metric]),
        best_summary_path=str(best_path),
        best_config_src=str(best_config) if best_config is not None else "N/A",
        best_config_out=str(best_config_out),
        data_root=cfg_data.get("DATA_ROOT", "/gemini/code/datasets"),
        datasets=cfg_data.get("DATASETS", "N/A"),
        splits=cfg_data.get("DATASET_SPLITS", "N/A"),
        det_filter=cfg_data.get("DETECTOR_FILTER", "N/A"),
        val_seq=cfg_data.get("VAL_SEQUENCES", "N/A"),
        seed=cfg_data.get("SEED", "N/A"),
        feat_dim=cfg_data.get("FEATURE_DIM", "N/A"),
        num_bands=cfg_data.get("NUM_BANDS", "N/A"),
        max_seq=cfg_data.get("MAX_SEQ_LEN", "N/A"),
        epochs=cfg_data.get("EPOCHS", "N/A"),
        batch_size=cfg_data.get("BATCH_SIZE", "N/A"),
        lr=cfg_data.get("LR", "N/A"),
        wd=cfg_data.get("WEIGHT_DECAY", "N/A"),
        use_freq=cfg_data.get("USE_FREQ_AWARE", "N/A"),
        use_decoder=cfg_data.get("USE_FREQ_DECODER_V2", "N/A"),
    )
    repro_out.write_text(repro_text)


if __name__ == "__main__":
    main()
