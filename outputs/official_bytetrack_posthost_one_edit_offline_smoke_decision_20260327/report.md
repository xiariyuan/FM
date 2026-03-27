## Official ByteTrack Post-Host One-Edit Offline Smoke Decision

Date: 2026-03-27

### Scope

This note summarizes the first offline learned smoke runs after the post-host oracle ceiling confirmed executable headroom on the canonical `official_bytetrack` carrier.

Runs:

- dataset build:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv)
- base learned scorer:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv)
- swap-focused learned scorer:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv)

### Dataset result

The post-host action dataset is healthy and no longer sparse-collapsed:

- `clusters=2657`
- `train_clusters=1982`
- `val_clusters=675`
- `target_nonkeep_clusters=1530`
- `target_defer_clusters=1472`
- `target_swap_clusters=58`
- `target_keep_clusters=1127`

Interpretation:

- the post-host contract produces abundant supervision
- the dominant learned target is `defer`
- `swap` exists but is rare
- `add` is absent in this official trainhalf slice

### Base scorer result

Best run:

- `best_epoch=20`
- `val_top1_acc=0.4059`
- `val_nonkeep_f0_5=0.3123`

Ad hoc action-type read on the val split:

- exact candidate top1: `0.4059`
- action-type accuracy: `0.9733`
- keep-vs-edit accuracy: `0.9985`
- swap action recall: `0.0000`

Confusion:

- `defer -> defer`: `565`
- `keep -> keep`: `92`
- `keep -> defer`: `1`
- `swap -> defer`: `17`

Interpretation:

- the model already separates `keep` vs `edit` almost perfectly
- it also classifies the dominant `defer` action type correctly
- but it collapses every rare `swap` into `defer`

So the base scorer is **not** a no-op learner. It learns the coarse gate, but fails on rare edit type coverage.

### Swap-focused scorer result

Best run:

- `best_epoch=19`
- `val_top1_acc=0.2919`
- `val_nonkeep_f0_5=0.1923`
- `val_action_type_acc=0.5585`
- `val_keep_vs_edit_acc=0.9304`
- `val_swap_action_recall=0.8824`

Confusion:

- `defer -> defer`: `270`
- `defer -> swap`: `250`
- `defer -> keep`: `45`
- `swap -> swap`: `15`
- `swap -> defer`: `1`
- `swap -> keep`: `1`
- `keep -> keep`: `92`
- `keep -> defer`: `1`

Interpretation:

- aggressive swap oversampling recovers rare `swap` coverage
- but it damages defer precision and exact candidate ranking badly
- a single flat candidate softmax is therefore unstable across the dominant `defer` mass and the rare `swap` tail

### Decision

Do **not** connect the current flat one-stage learned scorer to the online tracker yet.

What is validated:

- the post-host contract is learnable at the coarse decision level
- a learned model can already separate `keep` vs `edit`
- the remaining difficulty is not target sparsity anymore

What is not solved:

- exact candidate selection among many defer candidates
- stable recovery of rare `swap` actions without damaging defer precision

### Next implication

The next learned family should be **hierarchical**, not flat:

1. Stage A: `keep` vs `edit` gate
2. Stage B: `defer` vs `swap` action-type selector for edit-positive clusters
3. Stage C: within-action candidate ranker

This is a better fit than forcing all candidates into one softmax head.
