# Identity Transaction Fusion — 2026-07-14

## Scope

This note records the deployable MOT20-train fusion of two offline identity edits:

1. risk-aware fragmentation merge;
2. AssA-aware suffix identity swap.

No MOT20 test GT is used. GT-derived quantities are used only for train-side oracle analysis and OOF label generation.

## Implementation bug and correction

The first `eval_assa_swap_merge_fusion.py` implementation applied the final swap permutation to every frame in the sequence. That operation is only a global bijective tracker-ID rename, so HOTA, AssA and IDF1 are invariant. The identical metrics across all swap policies exposed the bug.

The corrected implementation:

- iterates frames in chronological order;
- activates each transposition only at its event frame;
- affects that frame and the suffix only;
- composes multiple transpositions on current labels;
- rejects any transaction that creates duplicate IDs in a future frame;
- raises an error if an accepted event changes no output row;
- verifies that the reconstructed merge-only output is byte-identical to the stored best merge result.

## Inputs

- A43 source:
  `outputs/spot_runtime_gate_20260628/A43_sequence_failure_recovery/A43_01_gap_fill_sensitivity/all_gap90_a4_s100/track_results`
- Best merge links:
  `outputs/assocriskbench_p15_20260714/fused_a42_adaptive_risk_eval/l1_r2_q75_adaptive_priority/selected_links.csv`
- OOF AssA-aware scores:
  `outputs/assocriskbench_p15_20260714/assa_aware_swap_ranker/oof_assa_swap_scores.csv`
- OOF cutoff diagnostics:
  `outputs/assocriskbench_p15_20260714/oof_assa_swap_commit_audit/heldout_diagnostics.csv`

## Policies

### Conservative

- horizon: permanent suffix
- score: `assa_swap_perm_risk_et_l0p25`
- spacing: 30
- cutoff aggregation: max
- selected events: 4

### Aggressive

- horizon: permanent suffix
- score: `assa_swap_perm_risk_et_l1p0`
- spacing: 30
- cutoff aggregation: max
- selected events: 15

## TrackEval results

| Variant | COMBINED HOTA | Mean HOTA | AssA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|
| A43 | 78.713113 | 77.559758 | 76.026934 | 90.790451 | 922 |
| Best merge | 78.763497 | 77.614254 | 76.123375 | 90.875278 | 901 |
| Conservative swap only | 78.724450 | 77.568637 | 76.047540 | 90.797831 | 922 |
| Conservative merge + swap | 78.774820 | 77.623129 | 76.143986 | 90.882658 | 901 |
| Aggressive swap only | 78.763470 | 77.573663 | 76.121900 | 90.827530 | 920 |
| **Aggressive merge + swap** | **78.813840** | **77.628159** | **76.218560** | **90.912357** | **899** |

The corrected best result improves:

- COMBINED HOTA by **+0.100727** over A43;
- COMBINED HOTA by **+0.050343** over the best merge-only result;
- mean HOTA by **+0.068401** over A43;
- mean HOTA by **+0.013905** over merge-only.

## Per-sequence delta of the best fusion versus merge-only

| Sequence | HOTA delta | AssA delta | IDSW delta |
|---|---:|---:|---:|
| MOT20-01 | -0.029077 | -0.016320 | 0 |
| MOT20-02 | -0.004090 | -0.001605 | +4 |
| MOT20-03 | -0.001206 | -0.004570 | -1 |
| MOT20-05 | +0.089994 | +0.169140 | -5 |

The gain is dominated by MOT20-05. This prevents a claim that the current policy is uniformly better on every train sequence.

## Operation-order audit

For both conservative and aggressive policies, the outputs of:

- `merge_then_swap`, and
- `swap_then_merge`

are byte-identical for all four sequences. No suffix transaction was rejected for a collision. Most selected swap IDs are disjoint from merge IDs; only IDs 331, 665 and 806 overlap, without changing the final composition in this static merge graph.

## Current conclusion

The experiment supports the core Identity Transaction hypothesis: fragmentation merges and suffix permutations provide complementary, additive association gains. However, the current result is still below the hard 80-HOTA target, and aggressive permanent swaps slightly hurt M01–M03 while substantially helping M05.

Sequence-specific manual policy selection is prohibited because it would use train GT outcomes as a deployment rule.

## Next deployable candidates

Uniform OOF policies to evaluate next:

1. h30 `assa_swap_h30_risk_hgb_l0p5`, q75, 5 events, 5/5 positive proxy;
2. h60 `assa_swap_h60_risk_et_l1p0`, q75, 6 events;
3. permanent `assa_swap_perm_risk_et_l0p5`, q75, 15 events, but with a large worst-event risk;
4. multi-horizon consensus requiring positive evidence from h30, h60 and permanent rankers.
