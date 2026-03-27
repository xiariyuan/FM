# Runtime Replay Stage-0 Findings (2026-03-17)

## 1. Strong-host audit: GT fallback/fill risk

The external-det inference path was traced through:

- [submit_bytetrack.py](/gemini/code/FMtrack-main/FM-Track/submit_bytetrack.py)
- [utils/mot_detections.py](/gemini/code/FMtrack-main/FM-Track/utils/mot_detections.py)
- [models/runtime_tracker_bytetrack.py](/gemini/code/FMtrack-main/FM-Track/models/runtime_tracker_bytetrack.py)

Current finding:

- `submit_bytetrack.py` resolves `BYTETRACK_DET_SOURCE=external`, loads detections from external MOT-format txt files, and passes them into `RuntimeTrackerByteTrack` as `external_detections`.
- `RuntimeTrackerByteTrack.update()` reads detections from `self.external_detections` when `det_source == "external"`.
- No active usage of `DET_FALLBACK_TO_GT_ON_EMPTY` or `DET_FILL_UNMATCHED_WITH_GT` was found in the current external-det inference path.

Interpretation:

- these two flags are present in resolved configs / training configs,
- but they do not currently appear to be consumed by the external-det runtime inference chain.

This is a positive audit result, but it should still be cited carefully as:

- "not found in current inference path,"
- not "globally impossible anywhere in the repo."

## 2. Minimal replay diagnostics implemented

The following minimal changes were added:

- [runtime_tracker_bytetrack.py](/gemini/code/FMtrack-main/FM-Track/models/runtime_tracker_bytetrack.py)
  - runtime dump now supports `ASSOC_RUNTIME_DUMP_TOPK=0` meaning dump the full candidate set
  - dump rows now include `group_id` and `candidate_count_total`
- [build_runtime_assoc_replay_labels.py](/gemini/code/FMtrack-main/FM-Track/scripts/build_runtime_assoc_replay_labels.py)
  - now outputs group-aware replay labels
  - now computes recoverability diagnostics
  - supports `rank_score_col`
  - can write row CSV, group JSONL, and recoverability JSON
- [train_runtime_rerank_baseline.py](/gemini/code/FMtrack-main/FM-Track/scripts/train_runtime_rerank_baseline.py)
  - simple rerank baseline with `logistic` or `gbdt`
  - evaluates top-1 correction on grouped runtime replay rows
  - supports `apply_on=all|ambiguous`

## 3. Proxy0213 recoverability result

Input:

- dump root: `outputs/runtime_assoc_dump_sw_yolox_heuristic_proxy0213_20260317`
- rank score column: `refined_score`
- top-k diagnostic cutoff: `8`

Recoverability summary:

- `groups = 27792`
- `positive_groups = 22015`
- `background_groups = 5777`
- `ambiguous_groups = 2468`
- `recoverable_groups = 336`
- `positive_in_topk_rate = 1.0`
- `rank_top1_acc_positive = 0.9847`
- `rank_top1_acc_ambiguous = 0.8639`
- `recoverable_rate_among_positive = 0.0153`
- `recoverable_rate_among_ambiguous = 0.1361`

Interpretation:

- the correct continuation is almost always still present in the dumped candidate set;
- but only a small fraction of positive groups are truly recoverable ranking mistakes;
- the real signal is concentrated in ambiguity groups, not in the whole dataset.

## 4. Simple baseline results

### A. Logistic, apply on all groups

Validation:

- `base_top1 = 0.9856`
- `final_top1 = 0.9823`
- `top1_gain = -0.0032`
- `amb_top1_gain = +0.0188`
- `easy_top1_gain = -0.0059`

Interpretation:

- naive always-on reranking is wrong;
- ambiguity groups benefit,
- easy groups are damaged.

### B. Logistic, apply on ambiguous groups only

Validation:

- `base_top1 = 0.9856`
- `final_top1 = 0.9876`
- `top1_gain = +0.0021`
- `amb_top1_gain = +0.0188`
- `easy_top1_gain = 0.0`

Interpretation:

- ambiguity-only triggering is already enough to flip the sign positive;
- safety gating is not optional.

### C. GBDT, apply on ambiguous groups only, random group split

Validation:

- `base_top1 = 0.9856`
- `final_top1 = 0.9945`
- `top1_gain = +0.0089`
- `amb_top1_gain = +0.0816`
- `easy_top1_gain = 0.0`

Interpretation:

- simple non-neural reranking has strong signal on the runtime replay object;
- this is exactly the control baseline that must be beaten before claiming a deep reranker is necessary.

### D. GBDT, apply on ambiguous groups only, seq split `02 -> 13`

Validation:

- `base_top1 = 0.9811`
- `final_top1 = 0.9831`
- `top1_gain = +0.0020`
- `amb_top1_gain = +0.0226`
- `easy_top1_gain = 0.0`

### E. GBDT, apply on ambiguous groups only, seq split `13 -> 02`

Validation:

- `base_top1 = 0.9863`
- `final_top1 = 0.9924`
- `top1_gain = +0.0061`
- `amb_top1_gain = +0.0502`
- `easy_top1_gain = 0.0`

Interpretation:

- the strong random-split result was not purely a leakage artifact;
- under seq-level holdout, the sign stays positive;
- the magnitude drops, but the runtime replay direction still looks viable.

## 5. Immediate conclusion

Current evidence supports:

1. `GT pseudo-group -> learned rerank` was the wrong training object.
2. Runtime replay is worth continuing.
3. Ambiguity-only activation is necessary.
4. A simple GBDT reranker is already competitive and must become a first-class baseline.
5. The next decisive experiment should use a fuller candidate dump than the current top-8 proxy dump.

