"""Pre-freeze production-path tests for M23-67-R2."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_67_r2_source_boundary_failure_root_cause_audit.py"

spec = importlib.util.spec_from_file_location("m23_67_r2_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
if spec.loader is None:
    raise RuntimeError("missing R2 loader")
spec.loader.exec_module(module)


def expect_raises(expected: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def test_equal_and_different_row_line_values() -> None:
    for equal_row_line in (True, False):
        scores, rows, labels = module.synthetic_join_fixture(equal_row_line=equal_row_line)
        joined, evidence = module.semantic_boundary_join(scores, rows, labels, join_key="row_index")
        assert evidence["semantic_join_passed"] is True
        assert evidence["physical_row_reorder_invariant"] is True
        assert joined.loc[0, "src_observable_row_index"] == joined.loc[0, "src_row_index"]
        assert joined.loc[0, "src_label_row_index"] == joined.loc[0, "src_row_index"]


def test_line_index_and_invalid_row_index_rejected() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=False)
    expect_raises(ValueError, module.semantic_boundary_join, scores, rows, labels, join_key="line_index")
    missing = scores.copy()
    missing.loc[0, "src_row_index"] = 999
    expect_raises(KeyError, module.semantic_boundary_join, missing, rows, labels, join_key="row_index")
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    expect_raises(ValueError, module.semantic_boundary_join, scores, duplicate, labels, join_key="row_index")


def test_physical_shuffle_invariant() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=True)
    baseline, _ = module.semantic_boundary_join(scores, rows, labels, join_key="row_index")
    shuffled, evidence = module.semantic_boundary_join(
        scores,
        rows.sample(frac=1.0, random_state=672).reset_index(drop=True),
        labels.sample(frac=1.0, random_state=267).reset_index(drop=True),
        join_key="row_index",
    )
    columns = [
        "src_row_index",
        "dst_row_index",
        "src_observable_row_index",
        "dst_observable_row_index",
        "src_label_row_index",
        "dst_label_row_index",
        "label",
    ]
    assert evidence["physical_row_reorder_invariant"] is True
    pd.testing.assert_frame_equal(baseline[columns], shuffled[columns])


def test_frozen_scientific_checks_and_hashes() -> None:
    checks = module.synthetic_self_test()
    assert checks and all(checks.values())
    hashes = module.scientific_function_hashes()
    assert set(hashes) == set(module.SCIENTIFIC_FUNCTION_NAMES)
    assert all(len(value) == 64 for value in hashes.values())
    assert all(getattr(module.legacy, name).__module__ == "m23_67_r1_frozen_base" for name in module.SCIENTIFIC_FUNCTION_NAMES)


def test_process_classifier_contract_and_live_fixture() -> None:
    checks = module.process_classifier_self_test()
    assert checks and all(checks.values())
    snapshot = module.process_gpu_snapshot(exclude_self=True)
    assert snapshot["blocking_processes"] == []
    assert snapshot["relevant_processes"] == []
    assert snapshot["gpu"]["compute_processes"] == []
    assert snapshot["gpu"]["memory_used_mib"] in (None, 0)


def test_reproduction_comparator() -> None:
    reference = module._normalize_json({"experiment_id": "R1", "value": 1.0, "nested": [1, {"x": 2.0}]})
    candidate = module._normalize_json({"experiment_id": "R2", "value": 1.0 + 5e-13, "nested": [1, {"x": 2.0}]})
    assert module._compare_nested(reference, candidate) == []
    mismatch = module._normalize_json({"experiment_id": "R2", "value": 1.0 + 5e-10, "nested": [1, {"x": 2.0}]})
    assert module._compare_nested(reference, mismatch)


def test_fail_close_end_to_end() -> None:
    checks = module.fail_close_end_to_end_self_test()
    assert checks and all(checks.values())


def main() -> None:
    tests = [
        test_equal_and_different_row_line_values,
        test_line_index_and_invalid_row_index_rejected,
        test_physical_shuffle_invariant,
        test_frozen_scientific_checks_and_hashes,
        test_process_classifier_contract_and_live_fixture,
        test_reproduction_comparator,
        test_fail_close_end_to_end,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("M23-67-R2 pre-freeze production-path tests: PASS")


if __name__ == "__main__":
    main()
