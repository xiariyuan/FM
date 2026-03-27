## Official ByteTrack Post-Host One-Edit Oracle Decision

Date: 2026-03-27

### Scope

This note summarizes the first strict paired half-val oracle ceiling run after stopping the learned pre-Hungarian line on the canonical `official_bytetrack` carrier.

Run:

- [summary.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/summary.csv)
- [result.csv](/gemini/code/FMtrack-main/FM-Track/outputs/official_bytetrack_posthost_one_edit_oracle_halfval_20260327_215036/result.csv)

The contract for this run is:

- same official ByteTrack detector / checkpoint / split / evaluator
- intervention moved to post-host first-stage matching
- at most one local edit per cluster
- allowed actions: swap one pair or defer one host pair

### Global paired result

Host-only:

- `HOTA=77.928`
- `AssA=77.095`
- `IDF1=86.513`
- `MOTA=90.143`
- `IDSW=180`

Post-host one-edit oracle:

- `HOTA=79.740`
- `AssA=80.946`
- `IDF1=89.797`
- `MOTA=89.519`
- `IDSW=208`

Paired delta:

- `delta_HOTA=+1.812`
- `delta_AssA=+3.851`
- `delta_IDF1=+3.284`
- `delta_MOTA=-0.624`
- `delta_IDSW=+28`

Conclusion: the post-host contract has real executable headroom for association-quality gains, but the current unconstrained oracle also introduces switch pressure and recall-style cost.

### Execution-level diagnostics

Plugin diagnostics aggregated across half-val:

- `eligible_clusters=7005`
- `replaced_clusters=522`
- `delta_commit_pairs=115`
- `delta_drop_pairs=522`
- `deferred_dets=426`
- `posthost_selected_clusters=522`
- `posthost_swap_clusters=115`
- `posthost_add_clusters=0`
- `posthost_defer_clusters=407`

Interpretation:

- this line is no longer an execution-level no-op
- most profitable edits are not additive bridge commits
- the dominant action mass is host-pair removal / defer, not add
- this directly supports the contract-change diagnosis: the official ByteTrack defect is more edit/defer-shaped than pre-Hungarian bridge-shaped

### Slice read

Hard slices:

- `MOT17-05-FRCNN`: `HOTA +4.636`, `AssA +9.063`, `IDF1 +7.226`, `IDSW +0`
- `MOT17-10-FRCNN`: `HOTA +1.732`, `AssA +3.306`, `IDF1 +3.929`, `IDSW +4`
- `MOT17-13-FRCNN`: `HOTA +2.599`, `AssA +6.300`, `IDF1 +4.935`, `IDSW +4`

Official-favorable slices:

- `MOT17-02-FRCNN`: `HOTA +1.940`, `AssA +4.597`, `IDF1 +3.696`, but `MOTA -2.156`, `IDSW +25`
- `MOT17-04-FRCNN`: `HOTA +0.566`, `AssA +1.088`, `IDF1 +1.149`, `IDSW -5`
- `MOT17-11-FRCNN`: `HOTA +5.062`, `AssA +9.759`, `IDF1 +7.255`, `IDSW -3`

Conclusion:

- the oracle does improve the intended hard slices
- but it is not yet sufficiently risk-controlled on all official-favorable slices, especially `MOT17-02-FRCNN`

### Decision

The contract-change direction is **validated as having executable headroom**.

What is validated:

- stopping the learned pre-Hungarian bridge line was correct
- moving the operator to post-host one-edit exposes real, executable local correction opportunities
- those opportunities are mostly swap / defer shaped, not add-only bridge shaped

What is not yet validated:

- a safe learned operator under this new contract
- a globally switch-safe policy for official-favorable slices

### Next implication

If this direction is continued, the next learned family should be a post-host action scorer with explicit switch-risk control, not another pre-Hungarian bridge predictor.
