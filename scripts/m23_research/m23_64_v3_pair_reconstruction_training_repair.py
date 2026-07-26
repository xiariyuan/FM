#!/usr/bin/env python3
"""M23-64-R1 repair wrapper.

The closed M23-64 run exposed a validation-only bug: v2's conditional-boundary
labels contain -1 for unknown/padded positions, but the frozen v2 metric helper
passed those values directly to sklearn's binary average-precision routine.
This wrapper keeps R62/R63 and the closed M23-64 output immutable, and applies
one deterministic adapter: unknown labels are excluded from binary summaries;
they are never converted to negatives.  Model, loss, optimizer, sampling,
checkpoint selection, and scope rules remain those of the frozen v2 source.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".").resolve()
BASE_SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training.py"
SCRIPT = ROOT / "scripts/m23_research/m23_64_v3_pair_reconstruction_training_repair.py"
ORIGINAL_R64 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training"
REPAIR_R64 = ROOT / "outputs/mot20_m23_20260718/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1"
REPAIR_PREREG = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_prereg_20260723.md"
REPAIR_RESULT = ROOT / "docs/m23_59_relation_pretrained_hierarchical_flow_v3_pair_reconstruction_training_repair1_result_20260723.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


spec = importlib.util.spec_from_file_location("m23_64_base_for_repair", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Redirect every mutable output/document/registry reference to the repair run.
base.EXP_ID = "M23-64-R1"
base.TITLE = "M23-64-R1 validation-shape repair and from-scratch relation training"
base.R64 = REPAIR_R64
base.SCRIPT = SCRIPT
base.PREREG = REPAIR_PREREG
base.RESULT = REPAIR_RESULT


_base_input_paths = base.input_paths


def repair_input_paths() -> list[Path]:
    paths = list(_base_input_paths())
    paths.extend(
        [
            BASE_SCRIPT,
            ORIGINAL_R64 / "final_summary.json",
            ORIGINAL_R64 / "closure_validation.json",
            ORIGINAL_R64 / "stage_a_gate.json",
            ORIGINAL_R64 / "examples_train.npz",
            ORIGINAL_R64 / "examples_validation.npz",
        ]
    )
    # Preserve deterministic order while avoiding duplicate paths.
    return list(dict.fromkeys(paths))


base.input_paths = repair_input_paths


_base_prereg_text = base.prereg_text


def repair_prereg_text() -> str:
    text = _base_prereg_text().replace(
        "# M23-64 preregistration",
        "# M23-64-R1 repair preregistration",
        1,
    )
    return (
        text
        + "\n\n## Repair rationale and frozen boundary\n"
        "The original M23-64 run is immutable and closed as `FAIL_TRAINING_INCOMPLETE`. "
        "Its read-only traceback is frozen: `v2.validation_metrics` passed conditional-boundary "
        "labels containing `-1` unknown values into `summarize_binary`, and sklearn then raised "
        "`ValueError: Expected 2D array, got 1D array instead`. The repair changes no R62/R63 "
        "artifact and does not overwrite the original R64.\n\n"
        "The sole repair is a validation adapter around the unchanged v2 metric definition: "
        "before binary summaries, retain only labels exactly in `{0,1}` and finite scores. "
        "Unknown labels are excluded, never mapped to negative. Model architecture, loss, "
        "optimizer, seeds, epochs, batches, sampling, gap buckets, composite checkpoint rule, "
        "and all forbidden-scope guards are unchanged.\n\n"
        f"Original M23-64 script SHA256: `{sha256_file(BASE_SCRIPT)}`.\n"
        f"Repair wrapper SHA256 (after freeze): `{sha256_file(SCRIPT)}`.\n"
    )


base.prereg_text = repair_prereg_text


_base_load_v2 = base.load_v2


def repaired_load_v2():
    v2 = _base_load_v2()
    if not getattr(v2, "_m23_64_r1_unknown_mask_adapter", False):
        original_summarize_binary = v2.summarize_binary

        def summarize_binary_known_only(
            y: np.ndarray, score: np.ndarray, threshold: float | None = None
        ) -> dict[str, Any]:
            y_arr = np.asarray(y)
            score_arr = np.asarray(score, float)
            keep = np.isfinite(score_arr) & np.isin(y_arr, [0, 1])
            return original_summarize_binary(
                y_arr[keep].astype(int), score_arr[keep], threshold
            )

        v2.summarize_binary = summarize_binary_known_only
        v2._m23_64_r1_unknown_mask_adapter = True
    return v2


base.load_v2 = repaired_load_v2


_base_training_config = base.training_config


def repair_training_config() -> dict[str, Any]:
    cfg = _base_training_config()
    cfg["repair_adapter"] = {
        "name": "known_binary_label_mask",
        "rule": "exclude labels not exactly in {0,1} and nonfinite scores before frozen v2 binary summaries",
        "unknown_as_negative": False,
        "source_script": str(SCRIPT.relative_to(ROOT)),
        "source_sha256": sha256_file(SCRIPT),
        "original_m23_64_decision": "FAIL_TRAINING_INCOMPLETE",
        "original_failure": "v2 validation conditional-boundary unknown -1 labels reached sklearn average_precision_score",
    }
    base.json_write(base.R64 / "training_config.json", cfg)
    return cfg


base.training_config = repair_training_config


if __name__ == "__main__":
    base.main()
