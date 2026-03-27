# Paper Experiment Checklist

Last updated: 2026-03-05

Use this as execution checklist. Mark each item only when output files are archived.

## A. Main Results (must)

- [ ] MOT17 final submission metrics (COMBINED + per-seq) archived.
- [ ] MOT20 final submission metrics (COMBINED + per-seq) archived.
- [ ] One unified comparison table vs baselines/SOTA generated.

Expected output files:
- `docs/tables/table_main_mot17_mot20.csv`
- `docs/tables/table_main_mot17_mot20.md`

## B. Attribution (must)

2x2 matrix:
- [ ] Base
- [ ] Base + ReID
- [ ] Base + Assoc strategy
- [ ] Base + ReID + Assoc strategy

Each row must include:
- HOTA, AssA, IDF1, IDSW, Frag

Expected output files:
- `docs/tables/table_ablation_2x2.csv`
- `docs/tables/table_ablation_2x2.md`

## C. Hyperparameter Evidence (must)

- [ ] Stage-A sweep summary (top-k + search space).
- [ ] Stage-B sweep summary (top-k + local refinement).
- [ ] Final parameter rationale documented.

Expected output files:
- `docs/tables/table_sweep_stageA.csv`
- `docs/tables/table_sweep_stageB.csv`

## D. Efficiency (must)

- [ ] Inference FPS measured on target GPU.
- [ ] Peak memory measured.
- [ ] Params/FLOPs measured.

Expected output files:
- `docs/tables/table_efficiency.csv`

## E. Robustness (recommended)

- [ ] Re-run key configuration with >=2 seeds.
- [ ] Report mean±std.

Expected output files:
- `docs/tables/table_robustness_seed.csv`

## F. Qualitative Analysis (recommended)

- [ ] 3 success cases (occlusion, crossing, re-entry).
- [ ] 3 failure cases with diagnosis.

Expected output files:
- `docs/figures/qualitative_success_*.png`
- `docs/figures/qualitative_failure_*.png`
- `docs/figures/qualitative_notes.md`

## G. Reproducibility (must)

- [ ] Final config paths frozen.
- [ ] Final checkpoints archived.
- [ ] Final command lines archived.

Expected output files:
- `docs/repro/final_configs.md`
- `docs/repro/final_commands.md`
- `docs/repro/final_checkpoints.md`

