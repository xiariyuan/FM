from __future__ import annotations

# Research artifact for the MOT20 M23 pair-state representation audit.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECTION_DIM = 32
PROJECTION_SEED = 23021
EPOCHS = 10
BATCH_SIZE = 8192


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalized_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def fixed_projection(input_dim: int) -> np.ndarray:
    generator = np.random.default_rng(PROJECTION_SEED)
    matrix = generator.standard_normal((input_dim, PROJECTION_DIM))
    projection, _ = np.linalg.qr(matrix)
    return projection[:, :PROJECTION_DIM].astype(np.float32)


def track_part_vectors(meta: pd.DataFrame, prototypes: np.ndarray):
    prefix = np.zeros_like(prototypes, dtype=np.float32)
    suffix = np.zeros_like(prototypes, dtype=np.float32)
    track = np.zeros_like(prototypes, dtype=np.float32)
    ordered = meta.sort_values(["source_track_id", "source_ordinal"])
    for _, part in ordered.groupby("source_track_id", sort=False):
        ids = part.chunk_id.to_numpy(int)
        weights = part.rows.to_numpy(np.float32)
        weighted = prototypes[ids] * weights[:, None]
        prefix_values = np.cumsum(weighted, axis=0)
        suffix_values = np.cumsum(weighted[::-1], axis=0)[::-1]
        prefix[ids] = prefix_values
        suffix[ids] = suffix_values
        track[ids] = prefix_values[-1]
    return normalized_rows(prefix), normalized_rows(suffix), normalized_rows(track)


def add_pair_state_features(
    seq: str,
    frame: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    root = Path("outputs/mot20_m23_20260718/micrograph_chunk30_v1") / seq
    meta = pd.read_parquet(root / "microtracklets.parquet")
    prototypes = normalized_rows(
        np.load(root / "prototypes.f16.npy").astype(np.float32)
    )
    prefix, suffix, track = track_part_vectors(meta, prototypes)
    projection = fixed_projection(prototypes.shape[1])
    src = frame.src_chunk.to_numpy(int)
    dst = frame.dst_chunk.to_numpy(int)

    src_segment = prefix[src] @ projection
    dst_segment = suffix[dst] @ projection
    src_track = track[src] @ projection
    dst_track = track[dst] @ projection
    blocks = {
        "segment_absdiff": np.abs(src_segment - dst_segment),
        "segment_product": src_segment * dst_segment,
        "track_absdiff": np.abs(src_track - dst_track),
        "track_product": src_track * dst_track,
    }
    added = []
    matrices = []
    for prefix_name, values in blocks.items():
        names = [f"pair_state_{prefix_name}_{index:02d}" for index in range(PROJECTION_DIM)]
        added.extend(names)
        matrices.append(pd.DataFrame(values.astype(np.float32), index=frame.index, columns=names))
    output = pd.concat([frame, *matrices], axis=1)
    return output, [*features, *added]


class PairStateNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
        )
        self.output = nn.Linear(96, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.backbone(values))


class PairStatePredictor:
    def __init__(self, network: PairStateNetwork, device: torch.device):
        self.network = network
        self.device = device

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        array = values.to_numpy(dtype=np.float32, copy=True)
        outputs = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(array), 32768):
                batch = torch.from_numpy(array[start : start + 32768]).to(self.device)
                prediction = self.network(batch)[:, 0]
                outputs.append(prediction.cpu().numpy())
        return np.concatenate(outputs).astype(np.float64)


def fit_pair_state_model(frame: pd.DataFrame, features: list[str], seed: int):
    base = sys.modules["m23_17_pair_state_base"]
    target, weights, metadata = base.target_and_weights(frame)
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    signs = (frame[base.TARGET].to_numpy(float) > 0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(signs),
        torch.from_numpy(weights.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    network = PairStateNetwork(len(features)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1.5e-3, weight_decay=2e-4)
    final_losses = {}
    for epoch in range(EPOCHS):
        network.train()
        loss_sums = {"total": 0.0, "regression": 0.0, "classification": 0.0, "ranking": 0.0}
        batches = 0
        for batch_values, batch_target, batch_sign, batch_weight in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)
            batch_sign = batch_sign.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            outputs = network(batch_values)
            utility = outputs[:, 0]
            logit = outputs[:, 1]
            regression = F.smooth_l1_loss(
                utility, batch_target, reduction="none", beta=0.35
            )
            classification = F.binary_cross_entropy_with_logits(
                logit, batch_sign, reduction="none"
            )
            supervised = ((regression + 0.25 * classification) * batch_weight).sum()
            supervised = supervised / batch_weight.sum().clamp_min(1e-6)
            positives = utility[batch_sign > 0.5]
            negatives = utility[batch_sign <= 0.5]
            pairs = min(len(positives), len(negatives), 1024)
            ranking = (
                F.softplus(-(positives[:pairs] - negatives[:pairs])).mean()
                if pairs
                else utility.new_zeros(())
            )
            loss = supervised + 0.15 * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            loss_sums["total"] += float(loss.detach().cpu())
            loss_sums["regression"] += float(regression.mean().detach().cpu())
            loss_sums["classification"] += float(classification.mean().detach().cpu())
            loss_sums["ranking"] += float(ranking.detach().cpu())
            batches += 1
        final_losses = {name: value / max(batches, 1) for name, value in loss_sums.items()}
    metadata.update(
        {
            "model": "two-head pair-state MLP with utility, sign, and minibatch ranking losses",
            "device": str(device),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "projection_dim": PROJECTION_DIM,
            "projection_seed": PROJECTION_SEED,
            "final_epoch_losses": final_losses,
        }
    )
    return PairStatePredictor(network, device), metadata


def main() -> None:
    base = load_module(
        "m23_17_pair_state_base",
        Path("scripts/m23_research/m23_17_sequence_normalized_utility.py"),
    )
    base.OUT = Path(
        "outputs/mot20_m23_20260718/m23_21_pair_state_metric_utility_v1"
    )
    base.NAME = "pair_state_metric_normalized_utility_ood_policy_v1"
    base.SCORE = "pred_pair_state_normalized_utility"
    base.TRAINING_GT_USE = (
        "sequence-normalized transaction utility and sign targets on fit sequences only"
    )
    base.FEATURE_TRANSFORM_DESCRIPTION = (
        "M23-17 within-sequence percentile features plus a fixed-seed 32D projection "
        "of 128D GT-free prefix/suffix and whole-track representations, encoded as "
        "partner-aware absolute differences and elementwise products"
    )
    base.TARGET_TRANSFORM_DESCRIPTION = (
        "M23-17 sequence-normalized asinh utility with an auxiliary positive-sign "
        "classification loss and minibatch positive-vs-negative ranking loss"
    )
    base.STATUS_DESCRIPTION = (
        "nested pair-state representation audit on reused development sequences; "
        "fixed-parent provenance remains exploratory"
    )
    base.augment_model_features = add_pair_state_features
    base.fit_model = fit_pair_state_model
    base.main()


if __name__ == "__main__":
    main()
