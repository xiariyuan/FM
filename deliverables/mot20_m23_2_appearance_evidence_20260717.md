# M23-2 MOT20 Appearance Evidence Audit

Date: 2026-07-17
Base commit: `899040ee2429ee226473811c42f2638be9838a07`
Unified audit: `c5854ab48db62974eb271ecb8681c43290c50a1993deedd8d4356649210c4f88`

## 1. Executive conclusion

M23-2 tested whether dense FastReID evidence can convert the M23-1 pre-NMS oracle ceiling into a deployable observation selector or a direct identity router.

The answer is fail-closed for both direct components:

1. **Suppressed-observation authenticity selector: rejected.** Under fixed four-fold MOT20 sequence-LOSO, adding appearance decreases AP from `0.102922` to `0.096596` and does not improve fixed top-10% precision or recall.
2. **Direct appearance identity router: rejected after strong-baseline correction.** The raw appearance margin reaches `61.84%` accuracy and strongly beats the weak geometry-IoU rule, but the construction-aware candidate-only prior reaches `88.91%`. Appearance is therefore `27.07%` worse than the stronger baseline.
3. **Appearance feature bank: retained only as auxiliary evidence.** It may be used inside a future global tracklet graph or a separately calibrated selective-override head, but not as a direct selector or router.

Final policy:

- `appearance_authenticity_selector_retained = false`
- `appearance_identity_router_retained = false`
- `appearance_feature_bank_retained_as_auxiliary_evidence = true`
- `deployment_allowed = false`
- `locked_manifest_created = false`
- P15 remains `no_op`.

## 2. Frozen evidence pipeline

### 2.1 GT-free appearance manifest

- Frozen M23-1 candidates: **459,899**
- Unique candidate crops: **459,891**
- Candidate-to-baseline-track coverage at IoU≥0.5: **424,553 / 92.31%**
- Suppressor-to-baseline-track coverage at IoU≥0.5: **425,412 / 92.50%**
- Invalid crops: **0**
- Suppressor mapping failures: **0**
- GT reads: **0**
- TrackEval calls: **0**

### 2.2 Deterministic FastReID extraction

The original FastReID interface and the optimized complete-frame engine were verified byte-for-byte on a complete-frame smoke test before full extraction.

- Crops: **459,891**
- Frames: **8,931**
- Complete-frame shards: **12**
- Raw float16 bytes: **1,883,713,536**
- Non-finite values: **0**
- Sharded-content Merkle SHA: `b9b1daae691c7aa9594c3db6456c21592709cb7e2f4d49a91e6cdc9601be0ca3`

The 1.88 GB raw embedding cache is intentionally excluded from Git. Its shard plan, frame catalog, model asset hashes, byte counts, and Merkle hash are retained.

### 2.3 Compact deployable feature bank

- Candidates: **459,899**
- Features: **35**
- Non-finite values: **0**
- Different observable candidate/suppressor tracks: **2,857**
- Track prototypes: at most **16** uniformly sampled observable suppressor embeddings per track
- Feature matrix SHA: `baf5b01387084211a3246fbe8f4f096dc6d21f5f3028509733be5bd508a9dd0e`

The 35 features include detection/NMS geometry, candidate-suppressor cosine, candidate and suppressor track-prototype similarities, prototype coherence, observable support counts, and box-shape differences.

## 3. Pre-training integrity corrections

### 3.1 Candidate key reconstruction

The first compact builder used NumPy integer fancy indexing:

```python
keys_view = keys[rows]
keys_view["global_pre_idx"] = global_pre_idx
```

Because integer indexing returns a copy, assignments did not update the original key table. The preflight correctly rejected the result before any model training.

The repaired implementation writes directly into each original field. Independent reconstruction verified:

- Candidate rows: **459,899**
- Unique candidate keys: **459,899**
- Duplicate keys: **0**
- Feature matrix changed: **no**
- Corrected key SHA: `4a3e444c1548b31125176aa68ece3fc9b8261857e00c214bcacfa9ce92cb2a56`
- Formal/reproduction: **4/4 files byte-identical**

### 3.2 Selector target correction

Before prediction, a second protocol inconsistency was found. The replace/add oracle contained 187,596 positives, making 40% recall impossible under a fixed 10% selection budget. The selector target was therefore fixed to the additive oracle, which represents genuine recovery beyond the baseline:

- Additive positives: **17,715**
- Replace/add positives retained only for identity attribution: **187,596**
- Missing frozen-budget keys: **0**
- Duplicate label keys: **0**

The final preregistration was written and hash-bound before predictions:

`2639a91c6965d656ac0e149b6ab1d274df031abc440cab19f20952c216046f1d`

## 4. Preregistered sequence-LOSO authenticity selector

Fixed protocol:

- Four outer folds: MOT20-01, 02, 03, 05
- HistGradientBoosting parameters fixed before prediction
- Equal total training mass per source sequence and class
- Geometry-only versus all 35 features
- Fixed top 10% per outer sequence
- No parameter sweep
- No threshold sweep
- No TrackEval

### 4.1 Combined results

| Model | AUC | AP | Top-10% precision | Top-10% recall |
|---|---:|---:|---:|---:|
| Geometry | 0.6914 | 0.1029 | 0.1374 | 0.3568 |
| Appearance | 0.6777 | 0.0966 | 0.1372 | 0.3561 |

Appearance deltas:

- AP: **-0.006326**
- Top-10% precision: **-0.000283**
- Worst-sequence AP gain: **-0.003314**
- Worst-sequence precision gain: **-0.023207**
- Worst-sequence recall: **0.1459**

MOT20-02 is the clearest failure: appearance top-10% precision falls from 0.1448 to 0.1216 and recall from 0.1737 to 0.1459.

**Decision: authenticity selector rejected.**

## 5. Identity attribution and disclosed strong-baseline correction

The preregistered identity audit evaluated 857 unambiguous different-track events. It initially found:

- Appearance margin: **61.84%**
- Geometry-IoU rule: **19.72%**
- Apparent gain: **+42.12 percentage points**

Formal and independent reproduction were byte-identical. However, after predictions were inspected, a stronger construction-aware baseline was identified: always choose `candidate_track_id`.

This is explicitly recorded as a **post-hoc strong-baseline correction**. The original preregistered outputs were not modified.

### 5.1 Corrected baseline table

| Rule | Accuracy |
|---|---:|
| Candidate-only | 0.8891 |
| Appearance margin | 0.6184 |
| Geometry-IoU | 0.1972 |
| Suppressor-only | 0.1109 |

Appearance relative to:

- Geometry: **+0.4212**
- Candidate-only: **-0.2707**

Candidate-only is at least as accurate as appearance on all four sequences. All 857 events have both candidate and suppressor prototype support, so the result is not caused by missing appearance prototypes. No fixed absolute-margin bin beats candidate-only.

**Authoritative corrected decision: direct identity router rejected.**

## 6. Scientific interpretation

M23-2 separates two questions that are often conflated:

1. **Does appearance identify a plausible identity relation better than a weak geometry heuristic?** Yes.
2. **Does raw appearance provide enough incremental information to outperform the structural prior induced by the candidate construction?** No.

The candidate pool is strongly asymmetric: in 762/857 unambiguous conflicts, the candidate-side track is correct. A raw cosine-margin router discards this prior and over-corrects. The appropriate architecture is therefore not an appearance-only router.

The paper-level conclusion is:

> Dense appearance evidence is informative relative to local geometry, but it neither improves domain-generalized suppressed-observation authenticity ranking nor surpasses the construction-aware candidate-track prior. Appearance should enter a counterfactual global graph as auxiliary evidence for rare selective overrides, rather than act as a direct per-frame selector or router.

## 7. Next stage: M23-3 candidate-default global graph

The next experiment should freeze the following design before evaluation:

1. Use the M23-1 budgeted pre-NMS set as sparse observation nodes.
2. Use `candidate_track_id` as the default identity edge.
3. Represent suppressor assignment as an explicit minority override edge rather than an equal alternative.
4. Aggregate temporal path consistency, tracklet continuity, motion, appearance, and namespace-change evidence at tracklet level.
5. Generate K-best global graph solutions.
6. Accept a graph solution only through baseline-relative solution-level risk control; otherwise preserve the baseline.
7. Evaluate with four cross-fitted MOT20 outputs and require nonnegative per-sequence score deltas before any deployment claim.

No further local appearance threshold, margin-bin, or selector model sweep is justified by M23-2.

## 8. Reproducibility and locked-state audit

- Unified checks: **29/29 passed**
- Key repair reproduction: **4/4 byte-identical**
- Sequence-LOSO reproduction: **8/8 byte-identical**
- Strong-baseline correction reproduction: **6/6 byte-identical**
- P15 policy: `no_op`
- Locked-label reads: **0**
- Locked TrackEval calls: **0**
- Remaining locked rows untouched: **156**
- Deployment: **false**
- Locked manifest: **false**

Large raw embedding shards, the 64 MB feature matrix, the 20 MB key table, OOF binary predictions, and reproduction directories are excluded from Git. Formal scripts, preregistration, reports, manifests, metrics, source hashes, and content hashes are committed.
