#!/usr/bin/env python3
"""Compatibility wrapper for the original dmm_base_tracker implementation.

The original source file was accidentally overwritten during patching.  The
stable compiled Python 3.11 bytecode was backed up at:
outputs/recovery_backup/dmm_base_tracker.cpython-311.pyc

This wrapper executes that bytecode with __file__ set to this source path so the
original path logic (e.g. external/BoT-SORT-main) remains valid.
"""
from __future__ import annotations

import marshal
import sys
import types
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO = _THIS_FILE.parents[1]
_IMPL_PYC = _REPO / "outputs" / "recovery_backup" / "dmm_base_tracker.cpython-311.pyc"
if not _IMPL_PYC.exists():
    raise FileNotFoundError(f"Cannot find original compiled tracker bytecode backup: {_IMPL_PYC}")

# Ensure BoT-SORT tracker package is importable before executing the bytecode.
_BOT_ROOT = _REPO / "external" / "BoT-SORT-main"
if str(_BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOT_ROOT))

with _IMPL_PYC.open("rb") as _f:
    _f.read(16)  # pyc header: magic, flags/timestamp/hash, size/hash
    _code = marshal.load(_f)

_impl = types.ModuleType("_dmm_base_tracker_impl")
_impl.__file__ = str(_THIS_FILE)
_impl.__package__ = ""
_impl.__name__ = "_dmm_base_tracker_impl"
sys.modules.setdefault("_dmm_base_tracker_impl", _impl)
exec(_code, _impl.__dict__)

for _name, _value in vars(_impl).items():
    if _name.startswith("__") and _name not in {"__doc__", "__all__"}:
        continue
    globals()[_name] = _value

if __name__ == "__main__":
    _impl.main()
