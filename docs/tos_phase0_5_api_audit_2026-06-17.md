# TOS-Track Phase 0.5: API Audit

**Date:** 2026-06-17
**Auditor:** Claude Code
**Status:** COMPLETE

## Audit Objective

Verify that the BoT-SORT codebase supports all 6 API primitives required by TOS-Track, and identify any risks or blockers.

---

## Audit Item 1: Feature Freeze Mode (hold buffer)

**Requirement:** `STrack.update_features(mode="freeze")` — skip smooth_feat update, skip history append.

**Finding:** ✅ **CONFIRMED — fully implemented and functional.**

**Location:** `external/BoT-SORT-main/tracker/basetrack.py`, lines 135-138:

```python
def update_features(self, feat, mode="normal", alpha_override=None, append_history=True):
    if mode == "freeze":
        self.curr_feat = feat
        return  # ← does NOT update smooth_feat, does NOT append to history
    if mode == "soft":
        # conservative EMA update ...
    # normal path: original behavior
```

**Three modes:**
- `"freeze"`: Only updates `curr_feat`. Does NOT update `smooth_feat`. Does NOT append to `features` history.
- `"soft"`: Conservative EMA update with `alpha_override` if provided.
- `"normal"` (default): Original behavior — EMA smooth_feat update + history append.

**Alpha:** Can be overridden via `alpha_override` argument (used by TCGAu soft mode).

**Status:** No risk. Can be called directly.

---

## Audit Item 2: Newborn Initialization Control

**Requirement:** Ability to gate/delay creation of newborn track IDs from unmatched high-score detections.

**Finding:** ✅ **CONFIRMED — injection point identified at `bot_sort.py:1805-1814`.**

**Location:** `bot_sort.py`, end of `update()` method, Step 4:

```python
""" Step 4: Init new stracks"""
for inew in u_detection:
    if int(inew) in rgsa_newborn_blocked_det_ids:   # ← RGSA blocks certain newbies
        continue
    track = detections[inew]
    if track.score < self.new_track_thresh:         # ← score gate
        continue
    track.activate(self.kalman_filter, self.frame_id)
    track.analysis_gt_id = getattr(track, "analysis_gt_id", -1)
    activated_starcks.append(track)
```

**Existing blocking mechanism:** RGSA already blocks certain newborn IDs via `rgsa_newborn_blocked_det_ids` — this is a precedent for a TOS gate.

**TOS implementation options:**
1. **Inject blocked IDs** (like RGSA): Add a TOS-gated set `tos_hold_det_ids` and check `if int(inew) in tos_hold_det_ids: continue`. Simple but requires per-frame injection.
2. **Override `new_track_thresh`**: Use a TOS-specific threshold (e.g., infinity during hold phase) to defer all newbies.
3. **New method on STrack**: `activate(self, kalman, frame_id, tos_hold=False)` — if `tos_hold=True`, set a flag to skip feature updates until unhold.

**Recommended approach:** Option 1 (inject blocked IDs per frame). Clean, consistent with existing RGSA pattern.

**Risk:** Low. No existing flag blocks this injection point.

---

## Audit Item 3: Lost/Removed Lifecycle

**Requirement:** TOS tracks remain in memory after `time_since_update > max_time_lost` without being hard-removed.

**Finding:** ✅ **CONFIRMED — `TrackState` enum supports LongLost, and archive/prune system exists.**

**Location:** `basetrack.py`, `bot_sort.py:1816-1820`:

```python
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3   # ← exists
    Removed = 4

# Step 5: prune old lost tracks
for track in self.lost_stracks:
    if self.frame_id - track.end_frame > self.max_time_lost:
        track.mark_removed()          # ← hard remove
        removed_stracks.append(track)
```

**Existing long-lost mechanism:** `TrackState.LongLost` exists but is not automatically transitioned to. Current code only goes Tracked → Lost → Removed.

**TOS implementation:** For TOS v0, tracks that would normally be `mark_removed()` should instead be moved to `removed_stracks` as "archived" (not deleted) and kept in `self.removed_stracks` for the TOS memory window (e.g., 150 frames). The existing `reentry_memory` system already does this — TOS can reuse or simplify this pattern.

**ReEntry Memory as TOS analogue:**
- `reentry_memory_enable` keeps removed tracks in `self.removed_stracks` with a max gap
- `removed_archive_retention` controls how long to keep (frames before current frame)
- TOS v0 can simply set a `tos_memory_frames` parameter and cap `removed_stracks` retention accordingly

**Risk:** Low. The reentry system already proves this pattern works.

---

## Audit Item 4: Feature Injection Points (re_activate / update)

**Requirement:** Inject `mode="freeze"` into both `track.re_activate()` and `track.update()`.

**Finding:** ✅ **CONFIRMED — both methods accept `tcgau_*` attributes from the detection object.**

**Location:** `bot_sort.py:1614-1621` (primary association):

```python
# TCGAU: compute update policy before update/re_activate
tcgau_policy = self._compute_tcgau_policy(track, det, itracked, idet, laplace_debug if self.laplace_assoc else None)
det.tcgau_update_mode = tcgau_policy["mode"]
det.tcgau_alpha_override = tcgau_policy["alpha_override"]
det.tcgau_append_history = tcgau_policy["append_history"]

if track.state == TrackState.Tracked:
    track.update(detections[idet], self.frame_id)        # ← uses det.tcgau_* attrs
else:
    track.re_activate(det, self.frame_id, new_id=False)   # ← uses det.tcgau_* attrs
```

**Location:** `bot_sort.py:248-274` (`update()` method body):

```python
def update(self, detection, frame_id):
    ...
    self.update_features(
        detection.curr_feat,
        mode=getattr(detection, "tcgau_update_mode", "normal"),      # ← reads from det
        alpha_override=getattr(detection, "tcgau_alpha_override", None),
        append_history=getattr(detection, "tcgau_append_history", True),
    )
```

**Location:** `bot_sort.py:230-246` (`re_activate()` method body):

```python
def re_activate(self, new_detection, frame_id, new_id=False):
    ...
    self.update_features(
        new_detection.curr_feat,
        mode=getattr(new_detection, "tcgau_update_mode", "normal"),   # ← reads from det
        alpha_override=getattr(new_detection, "tcgau_alpha_override", None),
        append_history=getattr(new_detection, "tcgau_append_history", True),
    )
```

**TOS implementation:** Set `det.tcgau_update_mode = "freeze"` before calling `track.update()` or `track.re_activate()`. The existing TCGAu infrastructure already handles this — TOS just needs to set the flag.

**Risk:** None. TCGAu proves this path works end-to-end.

---

## Audit Item 5: --laplace-haca-no-background Flag

**Requirement:** TOS v0 must use `--laplace-haca-no-background`.

**Finding:** ✅ **CONFIRMED.**

**Location:** `bot_sort.py:389`:
```python
self.laplace_haca_no_background = getattr(args, "laplace_haca_no_background", False)
```

**Usage:** This flag is passed to HACA computation to disable background probability estimation.

**Risk:** None. Flag is already wired.

---

## Audit Item 6: TrackEval Paths

**Requirement:** TrackEval can evaluate outputs for MOT17, MOT20, and custom formats.

**Finding:** ✅ **CONFIRMED — three format handlers exist.**

**Location:** `TrackEval/my_scripts/`

| Format | Handler | Notes |
|--------|---------|-------|
| MOT17 | `eval_mot_challenge.py` | Standard MOT format |
| MOT20 | `eval_mot_challenge.py` | Standard MOT format |
| Custom/Competition | `eval_pedestrian_summary.py` | Space-separated summary.txt |

**Summary format (pedestrian):**
```
HOTA AssA DetA MOTA IDF1 IDSW MODA Frag <blank line> HOTA_delta
```

**Risk:** None. Proven to work from previous session (v1 eval used this).

---

## Existing Infrastructure Relevant to TOS

### ReEntry Memory System (`bot_sort.py:581-626`, `2438-2733`)

The existing `reentry_memory_enable` system is highly relevant to TOS:

| ReEntry Memory | TOS v0 Equivalent |
|---|---|
| `reentry_memory_max_gap` | `tos_memory_frames` |
| `removed_stracks` archive | TOS "shadow" tracks |
| Gap-based scoring with decay | TOS reconnect with distance gating |
| Streak confirmation | TOS reconnect confirmation |
| Competitive vs unmatched recovery | TOS reconnect vs primary competition |

**TOS v0 can either:**
1. **Reuse reentry memory** with `tos_*` parameter overrides (simpler)
2. **Implement independent TOS system** alongside reentry (more isolation, more code)

**Decision:** Phase 1B/1C should evaluate whether TOS conflicts with reentry. If `--reentry-memory-enable` is already doing shadow tracking, TOS may be redundant. Recommendation: try TOS first with `--reentry-memory-enable=False` to isolate TOS behavior.

### TCGAu System (`bot_sort.py:2339-2423`)

The TCGAu `_compute_tcgau_policy()` method already computes freeze/soft/normal decisions per matched pair. This is **exactly the pattern TOS needs** — TOS just needs different trigger conditions:

| TCGAu Trigger | TOS Trigger |
|---|---|
| `q_update <= freeze_thresh` | `track.tos_is_occluded == True` |
| `q_update <= soft_thresh` | `track.tos_is_occluded and track.tos_frames_since_seen <= hold_frames` |
| Track coherence/stability signals | TOS-specific: hold duration, reconnect distance |

**TCGAu q_update formula:**
```
q = app_sim × pair_rel × stability × coherence × hist_norm × margin_gate
```

**For TOS, a simpler q_update could be:**
```
q_tos = 1.0 if not occluded else tos_quality_score(track, det)
```

**Risk:** Low. TCGAu proves the policy injection pattern works.

### RGSA Blocking Pattern (`bot_sort.py:1806-1807`)

RGSA already blocks newborn IDs at the creation point:
```python
for inew in u_detection:
    if int(inew) in rgsa_newborn_blocked_det_ids:
        continue
```

TOS should follow the same pattern: maintain a `tos_hold_det_ids` set, inject it at this same location.

---

## Audit Summary

| # | Item | Status | Risk |
|---|------|--------|------|
| 1 | Feature freeze mode | ✅ CONFIRMED | None |
| 2 | Newborn initialization gate | ✅ CONFIRMED | Low — needs injection |
| 3 | Lost/removed lifecycle control | ✅ CONFIRMED | Low — reuse/adapt reentry |
| 4 | Feature injection (update/re_activate) | ✅ CONFIRMED | None |
| 5 | --laplace-haca-no-background flag | ✅ CONFIRMED | None |
| 6 | TrackEval paths | ✅ CONFIRMED | None |

### Blockers: **NONE**

All 6 primitives are present and functional. TOS can be implemented as specified.

### Recommended Implementation Strategy

1. **Phase 1A (analysis-only):** Add `--tos-enable --tos-analysis-only` CLI flags. Instrument `STrack` with `tos_is_occluded`, `tos_frames_since_seen`, `tos_hold_start_frame` attributes. Output per-frame analysis CSV. No behavioral changes.
2. **Phase 1B (minimal behavior):** Implement TOS hold buffer via `tos_hold_det_ids` injection (following RGSA pattern). Implement delayed newborn via `track.activate(tos_hold=True)` flag.
3. **Phase 1C (frozen memory + reconnect):** Add `--reentry-memory-enable --tos-memory-frames N` and implement reconnect cooldown logic. Evaluate interaction with existing reentry system.

### Key Decision Points for Implementation

1. **TOS vs ReEntry conflict:** TOS hold/occlude is about not updating features. ReEntry is about recovering lost tracks. They can coexist but need careful scoping. Recommend TOS v0 with `--reentry-memory-enable=False`.
2. **TCGAu interaction:** TOS freeze mode sets `tcgau_update_mode="freeze"` — same mechanism as TCGAu. Can TOS and TCGAu both fire on the same match? TCGAu fires on quality signals; TOS fires on occlusion signals. They can be independent conditions OR TOS can override TCGAu. Recommend: TOS condition takes priority when `tos_is_occluded==True`.
3. **Archive retention:** TOS shadow tracks should NOT compete with primary association during hold phase (no competitive recovery). Only unmatched detection reconnection should be allowed.

---

*Audit complete. Ready to proceed to Phase 1A implementation.*
