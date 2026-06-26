# SPOT Implementation Plan

**Date:** 2026-06-26
**Status:** Oracle Gate PROVISIONAL → SPOT_PROVISIONAL
**Goal:** Fix documentation and prepare for minimal P4 ADG-freeze runtime patch

## Current State Summary

The repository has been correctly updated to `SPOT_PROVISIONAL` with `runtime_patch_allowed=0` in the core decision files, but two "cold-start reader guide" documents still contain outdated information that could mislead future reviewers.

### Correct (Authoritative) Files

- `outputs/oracle_gate/decision.md` ✅ - PROVISIONAL, allowed=0, oracle ceiling correctly labeled
- `outputs/oracle_gate/summary.csv` ✅ - provisional, SPOT_PROVISIONAL, runtime_patch_allowed=0
- `docs/current_mainline_2026-06-26.md` ✅ - CORRECTION warning, authority order defined
- `docs/oracle_gate_recap_2026-06-25.md` ✅ - CORRECTION warning, oracle ceiling ≠ runtime gain
- `scripts/spot_oracle/run_joint_oracle.py` ✅ - all branches runtime_patch_allowed=0
- `scripts/spot_oracle/run_oracle_state_protection.py` ✅ - "NOT a runtime improvement" labeled

### Incorrect (Outdated) Files

- `docs/experiment_index.md` ❌ - Lines 46-47 still show "SPOT_MAINLINE / Runtime patch allowed: YES"
- `docs/github_reader_guide.md` ❌ - Lines 20-22 show "7.29% IDSW reduction" (should be oracle ceiling), lines 29-30 show "CLOSED → SPOT_MAINLINE / Runtime patch allowed: YES"

## Implementation Tasks

### Task 1: Fix Outdated Documentation (P0 - Immediate)

**Goal:** Ensure all documentation reflects the correct `SPOT_PROVISIONAL` state.

**Files to modify:**
1. `docs/experiment_index.md`
2. `docs/github_reader_guide.md`

**Changes required:**

#### 1.1 Fix `docs/experiment_index.md`

**Line 46-47 (Section 2):**
```markdown
# BEFORE (INCORRECT):
- Oracle Gate: CLOSED → SPOT_MAINLINE
- Runtime patch allowed: YES

# AFTER (CORRECT):
- Oracle Gate: PROVISIONAL → SPOT_PROVISIONAL
- Runtime patch allowed: NO
```

**Line 20-22 (Section 1):**
```markdown
# BEFORE (INCORRECT):
The oracle gate has confirmed:
- Oracle 0A: 7.29% IDSW reduction (positive)
- Oracle 0C: 43.28% fixable (moderate)
- Oracle 0E: CLOSED → SPOT_MAINLINE

# AFTER (CORRECT):
The oracle gate has confirmed:
- Oracle 0A: 7.29% oracle recoverable ceiling (NOT runtime gain)
- Oracle 0C: 43.28% fixable (moderate, partial inline GT)
- Oracle 0E: PROVISIONAL → SPOT_PROVISIONAL (requires paired eval)
```

#### 1.2 Fix `docs/github_reader_guide.md`

**Line 20-22 (Section 1):**
```markdown
# BEFORE (INCORRECT):
The oracle gate has confirmed:
- Oracle 0A: 7.29% IDSW reduction (positive)
- Oracle 0C: 43.28% fixable (moderate)
- Oracle 0E: CLOSED → SPOT_MAINLINE

# AFTER (CORRECT):
The oracle gate has confirmed:
- Oracle 0A: 7.29% oracle recoverable ceiling (NOT runtime gain)
- Oracle 0C: 43.28% fixable (moderate, partial inline GT)
- Oracle 0E: PROVISIONAL → SPOT_PROVISIONAL (requires paired eval)
```

**Line 29-30 (Section 2):**
```markdown
# BEFORE (INCORRECT):
- **Oracle Gate:** CLOSED → SPOT_MAINLINE
- **Runtime patch allowed:** YES

# AFTER (CORRECT):
- **Oracle Gate:** PROVISIONAL → SPOT_PROVISIONAL
- **Runtime patch allowed:** NO
```

**Verification:** After changes, ensure all instances of:
- "SPOT_MAINLINE" → "SPOT_PROVISIONAL" (except in historical context sections)
- "Runtime patch allowed: YES" → "Runtime patch allowed: NO"
- "7.29% IDSW reduction" → "7.29% oracle recoverable ceiling (NOT runtime gain)"

---

### Task 2: Design Paired-Eval Harness (P1 - Infrastructure)

**Goal:** Create a paired-evaluation harness for BoT-SORT/SPOT that can run baseline vs variant with shared detection.

**Approach:** Adapt the existing ByteTrack paired-eval infrastructure.

**Reference files to study:**
- `scripts/run_official_bytetrack_shared_detection_pair_core.py` - core engine
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py` - orchestration
- `scripts/eval_botsort_halfval_trackeval.py` - BoT-SORT evaluation

**New files to create:**
1. `scripts/run_botsort_spot_shared_detection_pair_core.py` - core harness
2. `scripts/run_botsort_spot_paired_eval.py` - orchestration

**Key design decisions:**

1. **Detection Sharing:**
   - Load BoT-SORT detector once
   - Clone outputs for baseline and SPOT trackers
   - Maintain separate ID counters and tracker states

**Output structure:**
```
outputs/spot_runtime/<run_name>/
├── 00_baseline/
│   ├── track_results/*.txt
│   ├── summary.csv
│   └── trackeval/
├── 01_spot_freeze_app/
│   ├── track_results/*.txt
│   ├── spot_debug.csv
│   ├── summary.csv
│   └── trackeval/
├── result.csv (delta metrics)
├── report.md
└── run_manifest.json
```

**Hard requirements:**
- `spot_enable=0` must produce bit-level identical results to baseline
- All metrics (HOTA, AssA, IDF1, MOTA, IDSW) must match exactly
- run_manifest.json must record all parameters

---

### Task 3: Design Minimal P4 Runtime Patch (P2 - Core Implementation)

**Goal:** Implement minimal SPOT freeze_app that only freezes appearance/history.

**Files to modify:**
1. `external/BoT-SORT-main/tools/track.py` - add SPOT arguments
2. `external/BoT-SORT-main/tracker/bot_sort.py` - add SPOT decision module

**New SPOT arguments (track.py):**
```python
# Add after existing TCGAU/TOS arguments (around line 430)
parser.add_argument("--spot-enable", action="store_true", help="Enable SPOT freeze mechanism")
parser.add_argument("--spot-debug-dir", type=str, default="", help="Directory for SPOT debug output")
parser.add_argument("--spot-margin-thresh", type=float, default=0.05, help="Cost margin threshold for triggering freeze")
parser.add_argument("--spot-entropy-thresh", type=float, default=0.5, help="Cost entropy threshold for triggering freeze")
parser.add_argument("--spot-density-thresh", type=float, default=0.7, help="Local density threshold for triggering freeze")
parser.add_argument("--spot-action", type=str, default="freeze_app", help="SPOT action: freeze_app")
```

**SPOT initialization (bot_sort.py, after TOS init around line 694):**
```python
self.spot_enable = bool(getattr(args, "spot_enable", False))
self.spot_debug_dir = str(getattr(args, "spot_debug_dir", ""))
self.spot_margin_thresh = float(getattr(args, "spot_margin_thresh", 0.05))
self.spot_entropy_thresh = float(getattr(args, "spot_entropy_thresh", 0.5))
self.spot_density_thresh = float(getattr(args, "spot_density_thresh", 0.7))
self.spot_action = str(getattr(args, "spot_action", "freeze_app"))
self.spot_debug_rows = []
```

**SPOT decision module (bot_sort.py, new function):**
```python
def _compute_spot_decision(self, track, det, cost_matrix, itracked, idet):
    """
    Compute SPOT freeze decision based on ambiguity triggers.

    Returns: dict with keys:
        - triggered: bool
        - reason: str
        - cost_margin: float
        - cost_entropy: float
        - local_density: float
    """
    if not self.spot_enable:
        return {"triggered": False, "reason": "spot_disabled"}

    # Get top-2 costs for this track-det pair
    costs = cost_matrix[itracked, :]
    sorted_costs = np.sort(costs)
    cost_top1 = sorted_costs[0]
    cost_top2 = sorted_costs[1] if len(sorted_costs) > 1 else float('inf')

    # Compute cost margin
    cost_margin = cost_top2 - cost_top1

    # Compute cost entropy (simplified)
    # For first version, use simple binary entropy
    if cost_top2 < float('inf'):
        p1 = np.exp(-cost_top1) / (np.exp(-cost_top1) + np.exp(-cost_top2))
        p2 = 1.0 - p1
        entropy = -p1 * np.log2(p1 + 1e-10) - p2 * np.log2(p2 + 1e-10)
    else:
        entropy = 0.0

    # Compute local density (simplified)
    # For first version, use detection score as proxy
    local_density = det.score

    # Decision logic
    triggered = False
    reason = ""

    if cost_margin < self.spot_margin_thresh:
        triggered = True
        reason = "low_margin"
    elif entropy > self.spot_entropy_thresh:
        triggered = True
        reason = "high_entropy"
    elif local_density >= self.spot_density_thresh:
        triggered = True
        reason = "high_density"

    return {
        "triggered": triggered,
        "reason": reason,
        "cost_margin": float(cost_margin),
        "cost_entropy": float(entropy),
        "local_density": float(local_density),
    }
```

**SPOT integration point (bot_sort.py, in association loop after TOS, around line 1721):**
```python
# SPOT decision (after TOS, before track.update)
if self.spot_enable:
    spot_decision = self._compute_spot_decision(track, det, cost_matrix, itracked, idet)
    det.spot_triggered = spot_decision["triggered"]
    det.spot_reason = spot_decision["reason"]

    # If triggered and action is freeze_app, set freeze mode
    if spot_decision["triggered"] and self.spot_action == "freeze_app":
        det.tcgau_update_mode = "freeze"
        det.tcgau_append_history = False
        det.tcgau_alpha_override = None
        self.spot_stats["freeze_count"] += 1
```

**SPOT debug output (bot_sort.py, after processing each frame):**
```python
if self.spot_enable and self.spot_debug_dir:
    # Write spot_debug.csv at end of sequence
    # Fields: seq_name, frame_id, track_id, det_id, det_score,
    #         cost_top1, cost_top2, cost_margin, spot_triggered,
    #         spot_action, update_mode, append_history, track_age, lost_age, reason
    pass
```

**Key constraints:**
- Only freeze appearance/history (mode="freeze")
- Do NOT modify KF
- Do NOT modify Hungarian
- Do NOT modify lifecycle
- Do NOT modify detector
- Do NOT modify ReID
- Do NOT add PCC
- Do NOT add P5

---

### Task 4: Implement Parity Test (P3 - Validation)

**Goal:** Ensure `spot_enable=0` produces identical results to baseline.

**Test procedure:**
1. Run baseline on MOT20-05
2. Run SPOT with `spot_enable=0` on MOT20-05
3. Compare:
   - result.txt files (bit-level)
   - summary.csv metrics (HOTA, AssA, IDF1, MOTA, IDSW)
   - run_manifest.json parameters

**Expected result:**
- All metrics must match exactly
- Any difference indicates a bug in SPOT code path
- Parity must pass before proceeding to paired eval

---

### Task 5: Implement Oracle ADG-Freeze Paired Eval (P4 - Optional Validation)

**Goal:** Run oracle-freeze paired eval to establish theoretical ceiling.

**Approach:**
1. Use GT alignment data to identify association errors
2. In bot_sort.py, add oracle-freeze mode that freezes on known error matches
3. Run baseline vs oracle-freeze paired eval on MOT20-05

**Decision criteria:**
- If oracle-freeze is negative → P4 direction is likely not viable
- If oracle-freeze is positive → P4 has theoretical headroom

**Note:** This is optional but recommended before implementing full SPOT trigger.

---

### Task 6: Run MOT20-05 Paired Eval (P5 - Critical Validation)

**Goal:** Validate minimal P4 freeze_app on real paired evaluation.

**Test procedure:**
1. Run baseline on MOT20-05
2. Run SPOT with `spot_enable=1` and default thresholds on MOT20-05
3. Compute delta metrics

**GO conditions:**
- IDSW decreases
- HOTA does not decrease
- AssA/IDF1 do not decrease (preferably increase)
- MOTA does not significantly decrease
- freeze_rate is reasonable (not too high)
- spot_debug.csv shows triggers concentrated on ambiguous cases

**NO-GO conditions:**
- IDSW does not decrease or increases
- HOTA decreases
- AssA/IDF1 decrease significantly
- freeze_rate is too high
- spot_enable=0 parity fails
- spot_debug.csv shows many easy cases being frozen

**Outcome:**
- If GO → Can consider unlocking `runtime_patch_allowed=1` and expanding to MOT20 half-val
- If NO-GO → Must adjust trigger conditions or abandon P4

---

## Execution Order

1. **Task 1:** Fix outdated documentation (immediate, low risk)
2. **Task 2:** Design paired-eval harness (infrastructure, no runtime changes)
3. **Task 3:** Design minimal P4 runtime patch (core implementation)
4. **Task 4:** Implement parity test (validation)
5. **Task 5:** (Optional) Oracle ADG-freeze paired eval (validation)
6. **Task 6:** Run MOT20-05 paired eval (critical validation)

---

## What NOT to Do

- Do NOT implement PCC until P4 is validated
- Do NOT implement P5 delayed commitment
- Do NOT train large models
- Do NOT modify detector / ReID / TrackEval
- Do NOT modify KF or Hungarian
- Do NOT open new brainstorming lines
- Do NOT use oracle ceiling as go/kill criterion
- Do NOT unlock `runtime_patch_allowed=1` until paired eval confirms positive

---

## Success Criteria

1. All documentation consistently reflects `SPOT_PROVISIONAL` / `runtime_patch_allowed=0`
2. Paired-eval harness exists and can run baseline vs SPOT
3. `spot_enable=0` parity test passes exactly
4. (Optional) Oracle ADG-freeze shows positive headroom
5. MOT20-05 paired eval shows:
   - IDSW decrease
   - HOTA non-decrease
   - AssA/IDF1 non-decrease
   - Reasonable freeze_rate
6. Only after Task 6 passes: consider unlocking `runtime_patch_allowed=1`

---

## Notes

- This plan follows the consensus from four AI reviews
- Oracle ceiling (7.29%) is NOT runtime gain
- 0C fixable (43.28%) is partial inline GT, NOT runtime gain
- Real paired eval is the only valid go/kill criterion
- P4 freeze_app is the only allowed runtime patch
- PCC and P5 remain blocked until P4 is validated
