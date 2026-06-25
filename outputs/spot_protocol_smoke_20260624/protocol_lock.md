# SPOT-Track Phase 0 Protocol Lock

## Objective
- Lock the first SPOT execution round to protocol scaffolding, GT alignment, and oracle analysis only.

## Allowed Work
- Add new files under `scripts/spot_common/`, `scripts/spot_protocol/`, `scripts/spot_p0/`, `scripts/spot_oracle/`
- Generate structured outputs under a dedicated `outputs/` run root
- Run smoke analysis on synthetic or tiny local fixtures

## Forbidden Work
- Do not modify `external/BoT-SORT-main/tracker/*.py`
- Do not modify `external/BoT-SORT-main/tools/track.py`
- Do not train ADG
- Do not implement delayed commitment runtime behavior

## Required Go/No-Go Order
1. Protocol lock
2. GT alignment
3. Oracle 0A and Oracle 0C
4. Oracle 0D and Oracle 0B
5. Oracle 0E joint decision

## Current Decision
- Runtime tracker patches are blocked until oracle evidence is written and reviewed.

## Notes
- scaffold smoke
