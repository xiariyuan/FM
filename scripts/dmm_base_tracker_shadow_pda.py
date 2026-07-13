#!/usr/bin/env python3
from __future__ import annotations
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
_PYC = _ROOT / 'outputs' / 'recovery_backup' / 'dmm_base_tracker_shadow_pda.contact_identity.pyc'
_mod = SourcelessFileLoader('_shadow_pda_contact_identity_recovered', str(_PYC)).load_module()
for _name, _value in _mod.__dict__.items():
    if not (_name.startswith('__') and _name not in {'__doc__', '__all__'}):
        globals()[_name] = _value
if __name__ == '__main__':
    _mod.main()
