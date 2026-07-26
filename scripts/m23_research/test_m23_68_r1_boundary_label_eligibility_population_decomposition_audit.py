from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("m23_68_r1_boundary_label_eligibility_population_decomposition_audit.py")
SPEC = importlib.util.spec_from_file_location("m23_68_r1_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def item(both: int, left: int, right: int, windows: int, examples: int):
    return {
        "join_counts": {"both": both, "left_only": left, "right_only": right},
        "source_window_count": windows,
        "node_example_count": examples,
    }


def main() -> None:
    checks = MODULE.self_test()
    assert all(checks.values())
    assert MODULE.join_counts_exact(item(5662, 0, 0, 5662, 5662))
    assert MODULE.join_counts_exact(
        {"join_counts": {"both": 1281}, "source_window_count": 1281, "node_example_count": 1281}
    )
    assert not MODULE.join_counts_exact(item(5661, 1, 0, 5662, 5661))
    assert not MODULE.join_counts_exact(item(1280, 0, 1, 1280, 1281))
    assert not MODULE.join_counts_exact(item(1280, 0, 0, 1281, 1281))
    predecessor = MODULE.predecessor_m23_68_check()
    assert predecessor["all_passed"], predecessor["checks"]
    print("M23-68-R1 synthetic and predecessor tests passed")


if __name__ == "__main__":
    main()
