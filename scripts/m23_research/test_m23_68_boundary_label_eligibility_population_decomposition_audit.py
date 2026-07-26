from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("m23_68_boundary_label_eligibility_population_decomposition_audit.py")
SPEC = importlib.util.spec_from_file_location("m23_68_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def endpoint(status: str = "matched", identity: str = "a", ambiguity: bool = False, tie: bool = False):
    return {
        "supervision_status": status,
        "gt_identity_key": identity,
        "distractor_removed": status == "distractor_removed",
        "ambiguity_flag": ambiguity,
        "tie_flag": tie,
    }


def main() -> None:
    checks = MODULE.self_test()
    assert all(checks.values())
    assert MODULE.endpoint_labels(endpoint(), endpoint())[:2] == (0, 0)
    assert MODULE.endpoint_labels(endpoint(), endpoint(identity="b"))[:2] == (1, 1)
    assert MODULE.endpoint_labels(endpoint(), endpoint(identity="b", ambiguity=True))[:2] == (1, -1)
    assert MODULE.endpoint_labels(endpoint(), endpoint(identity="b", tie=True))[:2] == (1, -1)
    assert MODULE.endpoint_labels(endpoint(), endpoint(status="unknown", identity="b"))[:2] == (-1, -1)
    assert MODULE.material_ratio_shift(0.003, 0.001, 0.0005)
    assert not MODULE.material_ratio_shift(0.0011, 0.001, 0.0005)
    assert MODULE.classify_components(True, False, False) == "sequence_composition_shift"
    assert MODULE.classify_components(False, True, False) == "example_sampling_shift"
    assert MODULE.classify_components(False, False, True) == "observation_weighting_shift"
    assert MODULE.classify_components(True, True, False) == "multiple_population_components"
    assert MODULE.classify_components(False, False, False) == "no_material_population_component"
    assert MODULE.classify_components(False, False, False, False) == "inconclusive_population_decomposition"
    print("M23-68 synthetic tests passed")


if __name__ == "__main__":
    main()
