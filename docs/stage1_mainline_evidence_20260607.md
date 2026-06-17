# Stage1 Mainline Evidence Snapshot (2026-06-07)

This note fixes the current experiment state into a small set of citable claims.
It is intended to stop the project from drifting back to superseded lines.

## Confirmed Positive Mainline

Source: [outputs/rgsa_main_table_template.csv](/gemini/code/FMtrack-main/FM-Track/outputs/rgsa_main_table_template.csv)

- Baseline `BoT-SORT + HACA v3` on `MOT17-{10,11,13} val_half`:
  - `HOTA=73.825`, `AssA=72.740`, `IDF1=86.753`, `MOTA=88.695`, `IDSW=41`
- `Stage1 soft deferral` best operating point:
  - `lambda=0.15`: `HOTA=74.104`, `AssA=73.062`, `IDF1=86.565`, `MOTA=88.695`, `IDSW=39`
  - `lambda=0.18`: identical result

Confirmed claim:

- The only stable positive method result in the current line is **Stage1 soft deferral** on the strong `BoT-SORT + HACA v3` carrier, with `HOTA +0.279` and `IDSW -2`.

## Confirmed Failure Boundary

Source: [outputs/rgsa_mot20_transfer_20260606/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/rgsa_mot20_transfer_20260606/summary.csv)

- `MOT20-02` baseline:
  - `HOTA=66.850`, `AssA=57.348`, `IDF1=73.588`, `MOTA=90.277`, `IDSW=296`
- `Stage1 soft deferral` transfer:
  - `lambda=0.15`: `HOTA=62.601`, `IDSW=433`
  - `lambda=0.18`: `HOTA=63.487`, `IDSW=460`

Confirmed claim:

- The current Stage1 soft deferral design **does not generalize to MOT20** and is a negative transfer result.

## CCRC Diagnostic Result

Source: [outputs/ccrc_calibrators_v2_20260606/summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/ccrc_calibrators_v2_20260606/summary.csv)

On `MOT17 test` calibration metrics:

- Uncalibrated:
  - `ECE=0.697103`, `Brier=0.661586`, `NLL=2.314405`
- Platt scaling:
  - `ECE=0.073118`, `Brier=0.177608`, `NLL=0.539694`
- MLP:
  - `ECE=0.102254`, `Brier=0.195269`, `NLL=0.580674`

Confirmed claim:

- Association confidence is severely miscalibrated before calibration.
- Standard logit-space `Platt` scaling is the best calibration model in the current CCRC line.

## CCRC Runtime Failure

Source group: [outputs/ccrc_tau_sweep_mot17_20260607](/gemini/code/FMtrack-main/FM-Track/outputs/ccrc_tau_sweep_mot17_20260607)

Runtime result summary from the executed sweep:

- Baseline:
  - `HOTA=73.825`, `AssA=72.740`, `IDF1=84.775`, `MOTA=88.695`, `IDSW=41`
- `tau=0.1/0.2/0.3/0.4/0.5`:
  - identical result for all values
  - `HOTA=73.257`, `AssA=71.944`, `IDF1=84.233`, `MOTA=87.920`, `IDSW=71`

Confirmed claim:

- CCRC runtime abstention is a **negative utility** result in the current line.
- The collapse of all tau values indicates value-range collapse and, more importantly, a semantic mismatch between the calibration label and the runtime decision object.

## Current Project Decision

The project should currently treat the following as fixed:

1. `Stage1 soft deferral` is the only positive mainline.
2. `CCRC runtime` is a staged failure and should not be the active mainline.
3. The paper story must be narrowed to:
   - **Selective conservative association**
   - **High-risk local ambiguity**
   - not density generalization
   - not calibration-driven runtime utility

## Recommended Immediate Next Step

Use the Stage1 positive line as the active mainline and build:

- per-sequence analysis
- lambda sensitivity analysis
- trigger taxonomy and failure mode analysis
- MOT20 negative transfer boundary analysis

Treat weak-carrier probing as a separate decision experiment, not as the current mainline.
