from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/m23_research/m23_69_r1_boundary_objective_gate_population_alignment_audit.py"
SPECIFICATION = importlib.util.spec_from_file_location("m23_69_r1_audit", SCRIPT)
if SPECIFICATION is None or SPECIFICATION.loader is None:
    raise RuntimeError("cannot load M23-69-R1 audit")
MODULE = importlib.util.module_from_spec(SPECIFICATION)
sys.modules[SPECIFICATION.name] = MODULE
SPECIFICATION.loader.exec_module(MODULE)


def test_inherited_scientific_contract() -> None:
    checks = MODULE.base.self_test()
    assert checks and all(checks.values()), checks
    source_checks = MODULE.base.source_contract_checks()
    assert source_checks and all(source_checks.values()), source_checks


def test_predecessor_failure_closure() -> None:
    report = MODULE.predecessor_m23_69_check()
    assert report["all_passed"], {key: value for key, value in report["checks"].items() if not value}


def test_registry_lifecycle_predicate() -> None:
    assert MODULE.current_registry_state_valid()
    report = MODULE.predecessor_check()
    assert report["all_passed"], {key: value for key, value in report["checks"].items() if not value}
    assert "m23_69_registry_unused" not in report["checks"]
    assert report["checks"]["current_r1_registry_state_valid"] is True


def test_input_extension_and_preflight() -> None:
    paths = MODULE.collect_input_paths()
    resolved = {path.resolve() for path in paths}
    assert MODULE.BASE_SCRIPT.resolve() in resolved
    assert MODULE.RECONCILER.resolve() in resolved
    assert (MODULE.PREDECESSOR / "closure_validation.json").resolve() in resolved
    assert (MODULE.PREDECESSOR / "artifact_sha256_manifest.json").resolve() in resolved
    preflight = MODULE.base.command_preflight()
    assert preflight["all_passed"], preflight


def test_registry_close_schema_contract() -> None:
    fields, _ = MODULE.base.registry_rows()
    assert "status" in fields
    assert "current_stage" in fields
    assert "decision" in fields
    assert "hota" not in fields
    assert "result" not in fields


def main() -> None:
    tests = [
        test_inherited_scientific_contract,
        test_predecessor_failure_closure,
        test_registry_lifecycle_predicate,
        test_input_extension_and_preflight,
        test_registry_close_schema_contract,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("M23-69-R1 pre-freeze production-path tests: PASS")


if __name__ == "__main__":
    main()
