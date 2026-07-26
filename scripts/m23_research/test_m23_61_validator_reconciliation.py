#!/usr/bin/env python3
"""Regression test for expected-negative invariant reconciliation."""
from __future__ import annotations
import json


def reconcile(actual: dict[str, bool], expected: dict[str, bool]) -> dict:
    predicates = {key: actual.get(key) == value for key, value in expected.items()}
    return {"predicates": predicates, "passed": all(predicates.values())}


def main() -> None:
    historical = {
        "json_parse": True,
        "csv_parse": True,
        "input_sha_verified": True,
        "summary_no_running": True,
        "registry_no_stale_running": True,
        "registry_completed_row": True,
        "M23_59_modified": False,
        "training_runs_zero": True,
        "trackeval_runs_zero": True,
        "tracker_outputs_zero": True,
        "uses_mot20_gt": True,
        "post_hoc_diagnostic_only": True,
        "not_deployable": True,
        "not_a_strict_result": True,
    }
    expected = {key: True for key in historical}
    expected["M23_59_modified"] = False
    fixed = reconcile(historical, expected)
    assert fixed["passed"], fixed
    bad = dict(historical); bad["M23_59_modified"] = True
    rejected = reconcile(bad, expected)
    assert not rejected["passed"], rejected
    legacy = all(historical.values())
    assert legacy is False
    print(json.dumps({
        "passed": True,
        "legacy_all_values": legacy,
        "fixed_reconciliation": fixed,
        "mutation_detected": not rejected["passed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
