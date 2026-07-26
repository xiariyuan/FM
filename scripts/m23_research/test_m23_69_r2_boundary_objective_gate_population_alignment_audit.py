from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_69_r2_boundary_objective_gate_population_alignment_audit.py"
SPECIFICATION = importlib.util.spec_from_file_location("m23_69_r2_audit", SCRIPT)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError("cannot load M23-69-R2 audit")
MODULE = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = MODULE
SPECIFICATION.loader.exec_module(MODULE)


def test_inherited_scientific_contract() -> None:
    checks = MODULE.base.self_test()
    assert checks and all(checks.values()), checks
    source_checks = MODULE.base.source_contract_checks()
    assert source_checks and all(source_checks.values()), source_checks


def test_failed_predecessors_closed() -> None:
    m69 = MODULE.failed_predecessor_check(MODULE.M69, "M23-69", partial_objective_expected=False)
    r1 = MODULE.failed_predecessor_check(MODULE.R1, "M23-69-R1", partial_objective_expected=True)
    assert m69["all_passed"], {key: value for key, value in m69["checks"].items() if not value}
    assert r1["all_passed"], {key: value for key, value in r1["checks"].items() if not value}


def test_numerical_tolerance_scope() -> None:
    assert MODULE.tolerant_abs(5e-5) == 0.0
    assert MODULE.tolerant_abs(2e-4) == 2e-4

    def check_inside() -> bool:
        return MODULE.base.abs(5e-5) <= 1e-12

    assert MODULE.call_with_historical_tolerance(check_inside) is True
    assert not hasattr(MODULE.base, "abs")


def test_real_historical_reproduction_repair() -> None:
    frame = MODULE.base.source_population_frame()
    unique = MODULE.base.aggregate_unique(frame)
    result = MODULE.selected_source_metrics(frame, unique)
    assert result["historical_conditional_reproduction_passed"] is True
    numeric = result["historical_reproduction_numerics"]
    assert numeric["pr_auc_absolute_difference"] < MODULE.HISTORICAL_REPRODUCTION_TOLERANCE
    assert numeric["pr_auc_absolute_difference"] > 0
    assert numeric["metric_values_unmodified"] is True


def test_registry_and_preflight() -> None:
    assert MODULE.current_registry_state_valid()
    predecessor = MODULE.predecessor_check()
    assert predecessor["all_passed"], {
        key: value for key, value in predecessor["checks"].items() if not value
    }
    fields, _ = MODULE.base.registry_rows()
    assert "hota" not in fields
    assert "result" not in fields
    preflight = MODULE.base.command_preflight()
    assert preflight["all_passed"], preflight


def test_input_extension() -> None:
    paths = {path.resolve() for path in MODULE.collect_input_paths()}
    assert MODULE.BASE_SCRIPT.resolve() in paths
    assert MODULE.R1_SCRIPT.resolve() in paths
    assert MODULE.R1_RECONCILER.resolve() in paths
    assert (MODULE.M69 / "artifact_sha256_manifest.json").resolve() in paths
    assert (MODULE.R1 / "artifact_sha256_manifest.json").resolve() in paths


def main() -> None:
    tests = [
        test_inherited_scientific_contract,
        test_failed_predecessors_closed,
        test_numerical_tolerance_scope,
        test_real_historical_reproduction_repair,
        test_registry_and_preflight,
        test_input_extension,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("M23-69-R2 pre-freeze production-path tests: PASS")


if __name__ == "__main__":
    main()
