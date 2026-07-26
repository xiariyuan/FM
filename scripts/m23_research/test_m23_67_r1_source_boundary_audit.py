"""Pre-freeze tests for M23-67-R1 using the production join and fail-close paths."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_67_r1_source_boundary_failure_root_cause_audit.py"

spec = importlib.util.spec_from_file_location("m23_67_r1_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


def expect_raises(expected, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except expected:
        return
    raise AssertionError(f"expected {expected.__name__}")


def test_equal_row_line_values_pass() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=True)
    joined, evidence = module.semantic_boundary_join(scores, rows, labels, join_key="row_index")
    assert evidence["semantic_join_passed"] is True
    assert evidence["observable_row_line_equal_fraction"] == 1.0
    assert joined.loc[0, "src_observable_row_index"] == joined.loc[0, "src_row_index"]
    assert joined.loc[0, "src_label_row_index"] == joined.loc[0, "src_row_index"]


def test_different_row_line_values_pass() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=False)
    joined, evidence = module.semantic_boundary_join(scores, rows, labels, join_key="row_index")
    assert evidence["semantic_join_passed"] is True
    assert evidence["observable_row_line_equal_fraction"] == 0.0
    assert joined.loc[0, "label"] == 1


def test_line_index_misjoin_fails() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=False)
    expect_raises(ValueError, module.semantic_boundary_join, scores, rows, labels, join_key="line_index")


def test_physical_shuffle_does_not_change_row_index_join() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=True)
    baseline, _ = module.semantic_boundary_join(scores, rows, labels, join_key="row_index")
    shuffled, evidence = module.semantic_boundary_join(
        scores,
        rows.sample(frac=1.0, random_state=67).reset_index(drop=True),
        labels.sample(frac=1.0, random_state=671).reset_index(drop=True),
        join_key="row_index",
    )
    columns = [
        "src_row_index", "dst_row_index", "src_observable_row_index", "dst_observable_row_index",
        "src_label_row_index", "dst_label_row_index", "src_frame", "dst_frame", "label",
    ]
    assert evidence["physical_row_reorder_invariant"] is True
    pd.testing.assert_frame_equal(baseline[columns], shuffled[columns])


def test_missing_or_duplicate_row_index_fails() -> None:
    scores, rows, labels = module.synthetic_join_fixture(equal_row_line=True)
    missing = scores.copy()
    missing.loc[0, "src_row_index"] = 999
    expect_raises(KeyError, module.semantic_boundary_join, missing, rows, labels, join_key="row_index")
    duplicate = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    expect_raises(ValueError, module.semantic_boundary_join, scores, duplicate, labels, join_key="row_index")
    expect_raises(ValueError, module.semantic_boundary_join, scores, rows.drop(columns=["row_index"]), labels, join_key="row_index")


def test_unknown_ambiguity_repeat_aggregation_and_fixed_rules() -> None:
    checks = module.synthetic_self_test()
    required = {
        "unknown_endpoint_excluded", "ambiguity_endpoint_excluded", "repeated_label_consistent",
        "physical_transition_key", "primary_aggregation_sigmoid_mean_logit",
        "mean_probability_sensitivity_present", "gap_boundaries", "crowd_boundaries",
        "appearance_boundaries", "collapse_rule", "decision_priority",
    }
    assert required.issubset(checks)
    assert all(checks[name] for name in required)


def test_fail_close_end_to_end() -> None:
    checks = module.fail_close_end_to_end_self_test()
    assert checks
    assert all(checks.values())


def main() -> None:
    tests = [
        test_equal_row_line_values_pass,
        test_different_row_line_values_pass,
        test_line_index_misjoin_fails,
        test_physical_shuffle_does_not_change_row_index_join,
        test_missing_or_duplicate_row_index_fails,
        test_unknown_ambiguity_repeat_aggregation_and_fixed_rules,
        test_fail_close_end_to_end,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("M23-67-R1 pre-freeze production-path tests: PASS")


if __name__ == "__main__":
    main()
