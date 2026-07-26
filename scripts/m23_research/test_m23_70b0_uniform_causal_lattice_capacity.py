from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

PATH = Path(__file__).with_name("m23_70b0_uniform_causal_lattice_capacity.py")
spec = importlib.util.spec_from_file_location("m23_70b0_tested", PATH)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_uniform_cuts() -> None:
    cuts, report = m.uniform_cuts({0: list(range(3)), 1: list(range(4)), 2: list(range(9))})
    assert cuts == {2: [4, 8]}
    assert report["candidate_cuts"] == 2
    assert report["gt_used"] is False


def test_frozen_constants() -> None:
    assert m.SEGMENT_ROWS == 4
    assert m.DELAY == 8
    assert m.K == 3
    assert m.ALTERNATIVES == 2


def main() -> None:
    test_uniform_cuts()
    test_frozen_constants()
    print("PASS test_uniform_cuts")
    print("PASS test_frozen_constants")


if __name__ == "__main__":
    main()
