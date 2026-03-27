## Official ByteTrack Post-Host One-Edit Hierarchical Offline Smoke Decision

Date: 2026-03-28

### Scope

This note summarizes the first hierarchical offline learned smoke after the flat post-host one-edit scorer family showed a structural tradeoff between dominant `defer` behavior and the rare `swap` tail.

Runs:

- dataset:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_dataset_20260327_234041/summary.csv)
- flat base scorer:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_scorer_smoke_20260327_234041/summary.csv)
- flat swap-focused scorer:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_scorer_swapfocus_20260327_234816/summary.csv)
- hierarchical scorer:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_smoke_20260328_000238/summary.csv)

### Result snapshot

Validation comparison:

- flat base scorer:
  - `val_exact_top1_acc = 0.4059`
  - `swap_action_recall = 0.0000`
- flat swap-focused scorer:
  - `val_exact_top1_acc = 0.2919`
  - `val_action_type_acc = 0.5585`
  - `val_keep_vs_edit_acc = 0.9304`
  - `val_swap_action_recall = 0.8824`
- hierarchical scorer:
  - `val_exact_top1_acc = 0.3363`
  - `val_action_type_acc = 0.7363`
  - `val_keep_vs_edit_acc = 1.0000`
  - `val_swap_action_recall = 0.7647`

### Interpretation

The hierarchical decomposition is the first learned post-host family that clears the flat-family tradeoff cleanly enough to justify moving forward.

What improved:

- It preserves the coarse `keep` vs `edit` decision perfectly on the validation split.
- It keeps most of the rare `swap` coverage instead of collapsing `swap -> defer`.
- It materially improves overall action-type accuracy relative to the swap-focused flat scorer.
- It also recovers much of the exact-candidate accuracy that the swap-focused flat scorer gave up.

Why this matters:

- The flat base scorer was only strong because it effectively learned `keep` and dominant `defer`, but it was unusable for real corrective behavior because `swap_action_recall = 0`.
- The flat swap-focused scorer recovered `swap`, but at the cost of a large drop in defer precision and overall action classification stability.
- The hierarchical scorer is therefore the first offline learned result that is plausibly compatible with a conservative online post-host operator.

### Remaining weakness

This is still not a finished online model.

The remaining bottleneck is not `keep` vs `edit` anymore. It is candidate-level edit resolution:

- exact candidate ranking is still below the flat base scorer
- residual mistakes are concentrated in `defer` clusters that the model sometimes overcalls as `swap`

So the next step should not be another flat-scorer sweep. It should be a conservative online integration smoke of the hierarchical family, ideally with explicit diagnostics for:

- `predicted_keep_clusters`
- `predicted_defer_clusters`
- `predicted_swap_clusters`
- executed `posthost_swap_clusters`
- executed `posthost_defer_clusters`
- per-slice behavior on `MOT17-02 / 05 / 10 / 13`

### Decision

Do **not** return to the flat scorer family.

Promote the hierarchical post-host one-edit family to the current learned mainline for the `official_bytetrack` carrier under the changed post-host contract.

The next experiment should be a conservative online smoke integration of this hierarchical learner, not another offline architecture sweep and not a return to the stopped pre-Hungarian line.
