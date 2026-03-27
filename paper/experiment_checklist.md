# Experiment Checklist (LTRA Mainline)

Status legend: `TODO`, `RUNNING`, `DONE`, `SKIP`

## Mainline Decision
- DONE Pivot paper mainline to `BoT-SORT + LTRA`
- DONE Remove `FM-Track / frequency / Mamba / SFI` from paper main story
- DONE Keep ReID as standard component only

## Stage 1: Same-Base Validation
- TODO BoT-SORT val (MOT17) baseline
- TODO BoT-SORT val (MOT17) + LTRA
- TODO BoT-SORT val (MOT20) baseline
- TODO BoT-SORT val (MOT20) + LTRA
- TODO Collect HOTA / AssA / IDF1 / IDSW / Frag

## Stage 2: Minimal Ablation Ladder
- TODO Baseline
- TODO + mean-history prototype
- TODO + single-scale exponential prototype
- TODO + multi-scale signature, no reliability fusion
- TODO + full LTRA

## Stage 3: Plug-in Transfer
- TODO StrongSORT val baseline
- TODO StrongSORT val + LTRA

## Stage 4: Full Benchmark
- TODO MOT17 full benchmark with locked LTRA setting
- TODO MOT20 full benchmark with locked LTRA setting

## Stage 5: Paper Figures and Analysis
- TODO Sequence-level crowded-scene breakdown
- TODO Reliability bucket analysis
- TODO Qualitative occlusion recovery figure
- TODO Runtime overhead table

## Hard Rules
- STOP trainable Laplace for the main paper
- STOP old FM-Track frequency-line development
- Keep detector / ReID / tracker settings fixed in same-base comparisons
- Report HOTA / AssA / IDF1 / IDSW / Frag as primary evidence
