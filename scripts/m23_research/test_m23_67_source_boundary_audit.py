from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
path = Path(__file__).with_name("m23_67_source_boundary_failure_root_cause_audit.py")
spec = importlib.util.spec_from_file_location("m23_67_impl", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
checks = module.synthetic_self_test()
assert checks and all(checks.values())
print("M23-67 synthetic boundary audit fixtures: PASS")
