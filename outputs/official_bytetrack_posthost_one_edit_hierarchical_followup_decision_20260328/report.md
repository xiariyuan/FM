## Official ByteTrack Post-Host One-Edit Hierarchical Follow-Up Decision

Date: 2026-03-28

### Scope

This note summarizes the first online follow-up cycle after the hierarchical post-host one-edit family became the learned mainline on the `official_bytetrack` carrier.

Runs covered:

- runtime-aligned offline learner:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_runtime_safe_zero_20260328_013610/summary.csv)
- first conservative online non-no-op:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_runtime_safe_zero_halfval_smoke_20260328_013900/summary.csv)
  - [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_runtime_safe_zero_halfval_smoke_20260328_013900/result.csv)
- swap-priority offline learner:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_20260328_101340/summary.csv)
- swap-priority online smoke at `keep_thresh=0.988`:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_20260328_102900/summary.csv)
  - [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_20260328_102900/result.csv)
- swap-priority online smoke at `keep_thresh=0.97`:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_keep097_20260328_110500/summary.csv)
  - [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_swap_priority_halfval_smoke_keep097_20260328_110500/result.csv)
- balanced-gate-swap offline learner:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_20260328_131500/summary.csv)
- balanced-gate-swap online smoke at `keep_thresh=0.988`:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep0988_20260328_131900/summary.csv)
  - [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep0988_20260328_131900/result.csv)
- balanced-gate-swap online smoke at `keep_thresh=0.97`:
  - [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep097_20260328_142200/summary.csv)
  - [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_hierarchical_balanced_gate_swap_halfval_smoke_keep097_20260328_142200/result.csv)

### Result snapshot

The online learned line is no longer blocked by wiring or full execution failure. It can now reach three distinct regimes:

- `runtime_safe_zero`:
  - first non-zero online execution
  - only `defer` edits
  - paired delta still tiny: `HOTA +0.017 / AssA +0.038 / IDF1 +0.060 / MOTA -0.012 / IDSW +3`
- `swap_priority` at `keep_thresh=0.988`:
  - one online `swap`
  - still slightly negative overall: `HOTA -0.003 / AssA -0.001 / IDF1 -0.002 / MOTA -0.009 / IDSW +2`
- `swap_priority` at `keep_thresh=0.97`:
  - two online `swap`s
  - clearly worse: `HOTA -0.058 / AssA -0.111 / IDF1 -0.110 / MOTA -0.013 / IDSW +3`
- `balanced_gate_swap` at `keep_thresh=0.988`:
  - exact online no-op
  - paired delta exactly zero
- `balanced_gate_swap` at `keep_thresh=0.97`:
  - one online `swap`
  - still negative: `HOTA -0.056 / AssA -0.110 / IDF1 -0.109 / MOTA -0.004 / IDSW +1`

The sequence-level behavior is even more important than the global deltas:

- `MOT17-05` remained untouched in all online follow-up runs.
- `MOT17-13` remained untouched in all online follow-up runs.
- `MOT17-02` was touched by the earlier `swap_priority` checkpoint.
- `MOT17-10` was finally touched after lowering the keep threshold, but the touched run was still globally negative.

So the latest cycle did not reveal a useful threshold window on the hard official slices. It revealed a tradeoff:

- tighter gate: exact no-op
- looser gate: one or two executed `swap`s, but negative paired outcome

### Interpretation

The current bottleneck is no longer "can the model execute anything online?" and it is not "does the dataset contain any `swap` positives on the hard slices?".

Those questions now have answers:

- execution-level no longer blocks the line
- hard slices do have supervised `swap` positives in the current dataset construction

What remains unsolved is candidate quality under the frozen runtime contract.

More specifically:

- the model can be made conservative enough to become an exact no-op
- or slightly more open so that it executes a very small number of `swap`s
- but the executed `swap`s are still not reliably profitable

That means the highest-value missing piece is not another broad threshold sweep. It is a better edit utility / ranking objective that prefers truly safe, high-value candidate edits.

### Decision

Do **not** keep expanding the runtime threshold sweep line.

The current follow-up evidence is enough to say:

- a narrow threshold check was worth doing
- but threshold-only rescue is now close to exhausted

The next redesign, if this learned line continues, should move away from gate calibration as the main lever and toward:

- candidate utility target redesign
- safer candidate ranking
- or an explicit stop decision for this learned family under the current post-host one-edit contract

Operationally:

- keep the recorded follow-up runs as evidence
- stop treating runtime threshold sweeps as the mainline research step
- if external advice is requested, ask about `candidate utility / ranking objective` redesign rather than about another threshold grid
