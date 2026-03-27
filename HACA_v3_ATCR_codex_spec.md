# HACA-v3 / ATCR implementation spec

## Hard judgment

Continue association-only for one more serious round. Do not go back to expanding the old tiny alpha/r calibrator. The remaining gap is not capacity; it is missing candidate-set competition.

## Verified evidence from the uploaded bundles

- Old learned path in `external/BoT-SORT-main/tracker/laplace_calibrator.py` is a pair-wise alpha/r head over 15 hand-crafted pair features.
- Old learned features in `external/BoT-SORT-main/tracker/laplace_assoc.py` are:
  `spatial_sim, laplace_sim, motion_sim, absdiff, min_sim, prod_sim, agreement, stability, coherence, det_score, gap_log1p, hist_norm, amb_spa, amb_lap, amb_mot`.
- Old learned therefore does reweighting of existing scores, not candidate competition.
- HACA-v1 full underperformed: MOT17 same-base HOTA 78.391.
- HACA-v1 no-set improved to 78.654.
- HACA-v1 no-background improved to 78.676.
- HACA-v2 train config in the bundle is effectively: no-set backbone + background on + hist gate on + OOD gate on + shift fallback loss 0.15.
- HACA-v2 results:
  - MOT17 same-base: 78.715 / 77.489 / 86.455 / 180
  - MOT20 zero-shot: 77.840 / 75.105 / 89.522 / 781
  - StrongSORT zero-shot: 69.674 / 73.648 / 82.865 / 200
- Conclusion: HACA-v2 fixed transfer stability, but same-base learned remains too conservative and only ties heuristic.

## Why HACA-v1 full set failed

Do not interpret the failure as “candidate-set competition is wrong”.

The current so-called set branch is only:

`pair_embed -> concat(mean_embed, max_embed) -> linear -> GELU`

This is always-on group context injection, not real rival interaction. The likely problems are:

1. competition information was injected at the wrong place;
2. the set branch was always-on and perturbed easy groups;
3. background/null prediction was entangled with group context;
4. the training loss never directly forced top1-vs-top2 competition learning.

## Recommended single direction

Implement **HACA-v3 = ATCR (Ambiguity-Triggered Top-K Competitive Residual)**.

Core idea:

- keep the current HACA-v2 base as the safe backbone;
- add a lightweight competition head only for detection-centered ambiguous groups;
- only re-rank top-K rivals;
- do not let the competition head handle background/null;
- preserve hist gate, OOD gate, and shift-fallback behavior from HACA-v2.

## Runtime design

### Base score from HACA-v2

For each pair `(track i, det j)`, first run current HACA-v2 and obtain:

- pair token `x_ij` (same 15-D token already used by HACA)
- pre-background pair score `s0_ij`
- `beta_hist_ij`
- `beta_ood_ij`
- detection-level background gate `b_j`

Use logit form:

`z0_ij = logit(clamp(s0_ij, 1e-4, 1-1e-4))`

### Detection-centered candidate set

For each detection `j`, use the existing valid/gated candidate set `C_j` already formed in `haca_assoc.py`.

Take only top-K by `z0`:

`T_j = TopK_K(C_j, z0)`

Recommended default: `K = 3`.

Training-time rule: if GT positive is outside top-K, force-include it by replacing the last slot.

### Ambiguity trigger

Define top1-top2 probability margin on the HACA-v2 pre-background scores:

`m_j = s0_(1,j) - s0_(2,j)`

Define entropy over top-K logits:

`p_ij = softmax(z0_ij, i in T_j)`

`e_j = -sum_i p_ij log p_ij`

Define trust:

`u_j = mean_i( beta_hist_ij * beta_ood_ij ), i in T_j`

Activation gate:

`a_j = (1 - b_j) * u_j * sigmoid((tau_m - m_j) / t_m)`

Where:

- `tau_m` is a stored training-set ambiguity threshold (recommend 35th percentile of positive-group margins)
- `t_m` is a small temperature, default `0.03`

Effect:

- easy groups => low activation
- OOD / weak-history groups => low activation
- background groups => low activation

### Lightweight rival interaction

For each candidate `i in T_j`, compare it only to the other rivals `k != i` in `T_j`.

Use duel token:

`d_ikj = phi([x_ij, x_kj, x_ij - x_kj, z0_ij - z0_kj, m_j, e_j])`

Attention over rivals:

`w_ikj = softmax_k(v^T d_ikj)`

Competition context:

`c_ij = sum_k w_ikj * d_ikj`

Residual:

`r_ij = h([x_ij, c_ij, z0_ij, m_j, e_j])`

### Zero-sum group residual

Center the residual inside the top-K set:

`rbar_ij = r_ij - mean_k(r_kj)`

Apply only to top-K:

`z_ij = z0_ij + a_j * Delta_c * tanh(rbar_ij),  if i in T_j`

`z_ij = z0_ij,  otherwise`

Then:

`score_comp_ij = sigmoid(z_ij)`

Final score keeps the HACA-v2 background gate:

`score_final_ij = (1 - b_j) * score_comp_ij`

### Why this form

- bounded residual keeps stability;
- zero-sum residual makes the head do re-ranking, not global score drift;
- ambiguity trigger prevents damage on easy groups;
- keeping `b_j` in the base path avoids repeating the HACA-v1 full background entanglement failure.

## What to modify

### Extend runtime, do not replace it

Modify these files:

- `external/BoT-SORT-main/tracker/haca_assoc.py`
- `external/StrongSORT-master/deep_sort/haca_assoc.py`

Do not create a separate runtime path unless necessary. Extend the current checkpoint versioning.

### New checkpoint version

Extend the checkpoint loader from `haca_v1 / haca_v2` to also support `haca_v3`.

Add weights/statistics for:

- `comp_topk`
- `comp_margin_threshold`
- `comp_margin_temperature`
- `comp_delta_scale`
- `W_duel1`, `b_duel1`
- `W_duel2`, `b_duel2`
- `W_attn`, `b_attn`
- `W_comp1`, `b_comp1`
- `W_comp2`, `b_comp2`

Keep all HACA-v2 tensors intact in the same checkpoint.

### Runtime insertion point

BoT-SORT:
- inside `external/BoT-SORT-main/tracker/haca_assoc.py`
- within the existing detection-centered loop `rows = np.where(valid_mask[:, det_idx])[0]`
- after current HACA-v2 base outputs are computed
- before returning the final cost matrix to `bot_sort.py`
- still only for **primary association**

StrongSORT:
- inside `external/StrongSORT-master/deep_sort/haca_assoc.py`
- same detection-centered point
- keep the call site in `deep_sort/tracker.py::gated_metric`
- still only for primary association

## Training design

### New script

Create a new script:

- `scripts/train_haca_v3_from_gt_tracks.py`

Do not overload the existing v1/v2 trainer too much.

### Initialization

Inputs:

- existing GT pseudo-track group NPZs
- existing HACA-v2 checkpoint

Procedure:

1. load HACA-v2 base weights;
2. freeze all HACA-v2 parameters;
3. randomly initialize the ATCR competition head;
4. train only the competition head.

Optional later ablation, not default:
- unfreeze `pair_fc` only for a very short low-LR finetune.

### Ambiguous group mining

For each group, compute base HACA-v2 scores first.

Mark a positive group as ambiguous if:

- `m_j < tau_m_train`, or
- the base top1 is wrong.

Sampling ratio recommendation:

- 50% ambiguous positive groups
- 25% ordinary positive groups
- 25% background/easy groups

### Losses

Use top-K logits after the competition residual.

1. Group CE:

`L_list = CE(z_j over T_j, GT index)`

2. Hard duel margin, only for ambiguous positives:

`L_duel = max(0, gamma - z_pos + max(z_neg))`

3. Safe fallback for easy/background groups:

`L_safe = mean((z - z0)^2 over T_j)`

Final loss:

`L = L_list + lambda_duel * L_duel + lambda_safe * L_safe`

Recommended defaults:

- `gamma = 0.20`
- `lambda_duel = 0.5`
- `lambda_safe = 0.2`

Optional only if unstable:
- small L1 on `a_j * tanh(rbar)`

### Optimizer

- AdamW
- LR `1e-4`
- weight decay `1e-4`
- epochs `10-15`
- batch groups `256`
- grad clip `5.0`

## Debug and logging

Add the following debug fields in both runtime files:

- `haca_comp_active`
- `haca_comp_margin`
- `haca_comp_entropy`
- `haca_comp_residual`
- `haca_comp_rank_before`
- `haca_comp_rank_after`
- `haca_comp_topk`
- `haca_comp_swapped`

These should go into the same analysis logs used today.

## New scripts

Create:

- `scripts/train_haca_v3_mot17.sh`
- `scripts/run_botsort_haca_v3_eval.sh`
- `scripts/run_botsort_haca_v3_mot20_eval.sh`
- `scripts/run_strongsort_haca_v3_mot17_val.sh`

Mirror the current v2 wrappers.

## Minimal experiment plan

1. **BoT-SORT MOT17 same-base**
   - compare heuristic vs HACA-v2 vs HACA-v3
   - success: HOTA `+0.2` or AssA `+0.3` over heuristic without worse stability

2. **No ambiguity trigger ablation**
   - competition head always-on
   - expected worse than full HACA-v3

3. **No rival interaction ablation**
   - keep top-K and trigger, but remove pairwise rival interaction
   - expected worse than full HACA-v3

4. **BoT-SORT MOT20 zero-shot**
   - success: no worse than HACA-v2; ideal small HOTA/AssA gain

5. **StrongSORT zero-shot**
   - success: remains near HACA-v2 / heuristic and does not re-collapse

## Success / failure criteria

Continue if:

- MOT17 same-base clearly beats heuristic or HACA-v2
- StrongSORT stays stable
- top-K competition only activates on genuinely hard groups

Stop association-only if:

- HACA-v3 still cannot beat heuristic by more than noise-level margin
- or StrongSORT stability degrades again

If that happens, the next main line should be stronger temporal/uncertainty-aware ReID retraining, not more associator toggles.

## What not to do

- do not go back to making the old alpha/r MLP wider/deeper;
- do not re-introduce an always-on full set encoder over all candidates;
- do not let the competition head own the background/null decision;
- do not remove HACA-v2 hist/OOD/fallback safety for the first HACA-v3 implementation;
- do not expand beyond primary association in the first version.

## One-line summary for Codex

Implement `HACA-v3 / ATCR` as a frozen-HACA-v2, ambiguity-triggered, top-K, zero-sum competitive residual head that only re-ranks ambiguous detection-centered candidate groups and preserves the current HACA-v2 safety path.
