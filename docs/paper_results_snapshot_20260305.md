# Results Snapshot (2026-03-05)

This file captures known key numbers for drafting and decision tracking.

## 1) MOT17 Historical Submission Trend (COMBINED)

From local csv summaries:

- `mot17_metrics_tabbed_v3.csv`
  - HOTA: `58.449`
  - MOTA: `68.109`
  - IDF1: `72.217`
  - DetA: `59.318`
  - AssA: `57.915`
  - IDSW: `4293`
  - Frag: `5970`

- `mot17_metrics_tabbed_v4.csv`
  - HOTA: `58.345`
  - MOTA: `67.897`
  - IDF1: `71.842`
  - DetA: `59.436`
  - AssA: `57.600`
  - IDSW: `5421`
  - Frag: `7641`

Observation:
- v4 did not beat v3 on association stability (`IDSW/Frag` worsened).

## 2) ReID MOT20-only Training

Run:
- `outputs/reid_mot20_only_osnetain_20260304_203842/metrics.csv`

Best validation:
- Rank-1: `51.40`
- mAP: `29.86`
- best weights:
  - `outputs/reid_mot20_only_osnetain_20260304_203842/reid_best.pth`

## 3) MOT20 Association Probe

v13 baseline (proxy MOT20-05, epoch0):
- HOTA: `69.38`
- DetA: `71.61`
- AssA: `67.26`
- IDF1: `81.47`
- IDSW: `766`
- Frag: `11393`

v14-SFI probe (proxy MOT20-05, epoch0):
- HOTA: `68.65`
- DetA: `71.44`
- AssA: `66.02`
- IDF1: `80.55`
- IDSW: `834`
- Frag: `11396`

Decision:
- v14-SFI early probe underperformed; branch stopped for cost control.
- Switched to controlled run: `v13 + MOT20-only ReID best`.

## 4) Current Active Controlled Run

Config:
- `configs/experiments/bytetrack_fa_mot_mot20_v13_assoc_only_val05_reidmot20.yaml`

Output dir:
- `outputs/bytetrack_fa_mot_mot20_v13_assoc_only_val05_reidmot20_20260305_121613`

Log:
- `outputs/ops_logs/mot20_assoc_only_v13_reidmot20_20260305_121613.log`

Purpose:
- isolate ReID contribution without structural changes.

