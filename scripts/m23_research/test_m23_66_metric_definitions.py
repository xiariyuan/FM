"""Synthetic metric-definition fixtures for M23-66.

No experiment input artifact, checkpoint, GT file, or model inference is used.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_66_v3_metric_correctness_source_target_decomposition_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m23_66_metric_fixture", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M23-66 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_metric_definitions():
    checks = load_module().synthetic_self_test()
    assert checks["canonical_missing_all_query_zero"]
    assert checks["multiple_positive_any_valid_differs"]
    assert checks["candidate_pool_no_positive_not_skipped"]
    assert checks["incoming_latest_predecessor"]
    assert checks["score_tie_candidate_id_break"]
    assert checks["duplicate_boundary_unique_differs"]
    assert checks["duplicate_boundary_mean_logit_sigmoid"]
    assert checks["unknown_label_excluded_contract"]
    assert checks["paired_fields_separate"]
    assert checks["invalid_pair_excluded_primary"]


if __name__ == "__main__":
    test_synthetic_metric_definitions()
    print("M23-66 synthetic metric fixtures: PASS")
