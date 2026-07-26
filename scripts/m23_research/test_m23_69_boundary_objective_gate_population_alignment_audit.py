from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_69_boundary_objective_gate_population_alignment_audit.py"
SPEC = importlib.util.spec_from_file_location("m23_69_audit", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M23-69 audit")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_synthetic_contract() -> None:
    checks = MODULE.self_test()
    assert checks and all(checks.values()), checks


def test_frozen_source_contract() -> None:
    checks = MODULE.source_contract_checks()
    assert checks and all(checks.values()), {key: value for key, value in checks.items() if not value}


def test_direct_population_counts() -> None:
    train = MODULE.direct_array_population("train")
    validation = MODULE.direct_array_population("validation")
    assert train["windows"] == 5662
    assert train["boundary_focal_rows"] == 13254
    assert train["boundary_focal_positives"] == 1203
    assert validation["windows"] == 1281
    assert validation["boundary_focal_rows"] == 2534
    assert validation["boundary_focal_positives"] == 149
    assert validation["native_known_observations"] == 34805


def test_target_node_label_semantics() -> None:
    node = pd.DataFrame(
        [
            {
                "sequence": "S",
                "window_id": "pure",
                "node_logit": -1.0,
                "node_probability": float(MODULE.sigmoid(-1.0)),
                "row_indices": "[0,1,2,3,4]",
            },
            {
                "sequence": "S",
                "window_id": "impure",
                "node_logit": 1.0,
                "node_probability": float(MODULE.sigmoid(1.0)),
                "row_indices": "[0,1,2,3,5]",
            },
            {
                "sequence": "S",
                "window_id": "unknown",
                "node_logit": 0.0,
                "node_probability": 0.5,
                "row_indices": "[0,1,2,3,6]",
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "row_index": list(range(7)),
            "supervision_status": ["matched"] * 6 + ["unknown"],
            "gt_identity_key": ["A", "A", "A", "A", "A", "B", ""],
            "distractor_removed": [False] * 7,
            "ambiguity_flag": [False] * 7,
            "tie_flag": [False] * 7,
        }
    )
    result = MODULE.target_node_labels(node, labels).set_index("window_id")
    assert int(result.loc["pure", "node_label"]) == 1
    assert int(result.loc["impure", "node_label"]) == 0
    assert int(result.loc["unknown", "node_label"]) == -1


def test_metric_and_ranking_contract() -> None:
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
    metrics = MODULE.binary_metrics(labels, scores, np.asarray(["a", "b", "c", "d"]))
    assert metrics["rows"] == 4
    assert metrics["positives"] == 2
    assert metrics["precision_at_actual"] == 0.5
    candidates = [
        {"seed": 2, "epoch": 1, "score": 0.5},
        {"seed": 1, "epoch": 2, "score": 0.5},
    ]
    assert MODULE.rank_candidates(candidates, "score")[0]["seed"] == 1


def test_predecessor_and_process_contract() -> None:
    predecessor = MODULE.predecessor_check()
    assert predecessor["all_passed"], {key: value for key, value in predecessor["checks"].items() if not value}
    process_checks = MODULE.process_module().process_classifier_self_test()
    assert process_checks and all(process_checks.values()), process_checks


def main() -> None:
    tests = [
        test_synthetic_contract,
        test_frozen_source_contract,
        test_direct_population_counts,
        test_target_node_label_semantics,
        test_metric_and_ranking_contract,
        test_predecessor_and_process_contract,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("M23-69 pre-freeze production-path tests: PASS")


if __name__ == "__main__":
    main()
