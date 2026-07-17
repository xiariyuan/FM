# M23-6-0 MOT20 Long-Horizon Tracklet Identity-Chain Oracle Audit

Date: 2026-07-18
Repository: `FM-Track`
Scope: MOT20 train sequences `01/02/03/05`
Status: **completed, reproduced, fail-closed**

## Research question

M23-5 closed frame-local NMS-conflict actions. M23-6-0 asks whether longer-horizon tracklet identity chains provide enough oracle ceiling to justify graph training.

The exact M23-4 all-additions tracker is held fixed. Every variant preserves all 1,104,528 source rows, boxes, scores, and tail fields; only identity labels can change.

## Frozen action spaces

1. **Pure segment chain** — purity-1.0 segments only, selected per GT by weighted interval scheduling.
2. **Modal segment chain** — all positive-modal segments, selected per GT by weighted interval scheduling; every row in a selected segment receives the chain identity.
3. **Modal-core chain** — a row receives the chain identity only when its frame-level Hungarian match equals the segment modal GT.
4. **Full matched identity ceiling** — every Hungarian-matched row receives its matched GT chain identity.

The interval solver maximizes modal matched frames, then segment rows, then uses fewer segments. Synthetic chain IDs are sequence-separated and strictly below `2^24`.

## Structural inventory

| Item | Count |
|---|---:|
| Source rows | 1,104,528 |
| Total contiguous segments | 4,347 |
| Positive-modal segments | 3,914 |
| Purity-1.0 segments | 3,013 |
| Modal selected segments | 3,639 |
| Modal relink edges | 1,495 |
| Modal match recall | 98.456% |
| Same-GT duplicate frames | 7,036 |
| Modal-core row actions | 1,069,789 |
| Full identity row actions | 1,086,684 |

Pure chains cover only 70.371% of modal matches and are therefore a coverage diagnostic, not a sufficient representation.

## Combined TrackEval results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All-additions context | 78.139540 | 81.445354 | 75.017410 | 89.449685 | 94.007566 | 1,104 | 18,270 | 48,617 |
| Pure segment chain | 78.647774 | 81.589556 | 75.858580 | 90.213549 | 93.992582 | 1,274 | 18,270 | 48,617 |
| Modal segment chain | 80.994950 | 81.619530 | 80.412970 | 94.540621 | 94.021227 | 949 | 18,270 | 48,617 |
| Modal-core chain | 82.298195 | 81.839350 | 82.796050 | 95.752744 | 93.976277 | 2,623 | 17,688 | 48,035 |
| Full matched identity ceiling | 83.086020 | 81.858000 | 84.363250 | 97.073761 | 94.225613 | 4 | 17,583 | 47,930 |


## Preregistered gate decision

| Gate | Requirement | Result | Decision |
|---|---|---:|---|
| Modal segment HOTA | >=80.00 | 80.994950 | pass |
| Modal segment IDSW | <=552 | 949 | **fail** |
| Modal-core HOTA | >=82.00 | 82.298195 | pass |
| Modal-core IDSW | <=276 | 2623 | **fail** |
| Full identity HOTA | >=82.50 | 83.086020 | pass |

All four sequences improve under modal, modal-core, and full identity. Nevertheless, the long-horizon chain action space fails because both deployable-form identity-safety gates fail.

## Per-sequence HOTA

| Sequence | Source | Modal segment | Delta | Modal core | Delta | Full identity | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 77.581310 | 79.880910 | +2.299600 | 81.047820 | +3.466510 | 82.720876 | +5.139566 |
| MOT20-02 | 69.171137 | 77.474900 | +8.303763 | 80.060790 | +10.889653 | 82.295525 | +13.124388 |
| MOT20-03 | 80.449086 | 81.537200 | +1.088114 | 82.144374 | +1.695288 | 82.613860 | +2.164774 |
| MOT20-05 | 79.006390 | 81.556160 | +2.549770 | 82.904630 | +3.898240 | 83.485174 | +4.478784 |


## Boundary mechanism forensic

The modal-core representation is accurate at the row level but temporally unstable:

- 3,914 positive segments contain 5,572 modal runs;
- this creates 5,245 chain/original-ID boundary toggles;
- 1,658 internal gaps separate modal runs;
- 1,307 gaps contain another GT and must remain hard boundaries;
- only 351 gaps are unmatched-only holes;
- filling these holes adds 927 rows without crossing an other-GT boundary;
- an other-GT-split support-span representation retains 99.954% of modal matches.

This explains why modal-core reaches HOTA 82.298195 while producing 2,623 ID switches. The issue is not insufficient identity evidence; it is a representation that alternates between chain and source IDs at short support gaps.

## Decision

**Do not train the current segment-chain graph. Close this representation, not the identity direction.**

The identity target is technically feasible: the full matched identity ceiling reaches **83.086020 HOTA** with only **4 ID switches**, exceeding the 82.5 target.

The next justified experiment is M23-7-0: a purified support-span chain oracle with:

- other-GT rows as hard boundaries;
- unmatched-only holes filled inside support spans;
- per-GT span-level weighted interval scheduling;
- no threshold or gap sweep.

P15 remains no-op. Locked-label reads: 0. Locked TrackEval calls: 0. All 156 remaining locked rows remain unread.

## Reproducibility

- compact files: 6/6 byte-identical;
- tracker files: 20/20 byte-identical;
- TrackEval summary/detailed files: 10/10 byte-identical;
- unified audit: 110/110 checks passed.

## Hashes

- formal script: `6ca22bf2328ce6fd6e2b932299499ac1730b414e56cbd8bc94786437d759c649`
- preregistration: `71746df260de9123ee33b8c3e34b1ad1e179f9b12cd7275d0c319aa397b2657a`
- formal report: `18e38562f26f1cfe0e758f3bdb6d856eabd0f6a2154e0630a3acfc500b884824`
- formal manifest: `94bc44098966ec6e04cd67d09f45b0a15d9cdbe255ea79c2df6f5cc41febb449`
- unified audit script: `b6543b171a35257631a9d11abddbadf04125f46ef24a7d84206d32cd2509213f`
- unified audit JSON: `bc3270b1afde18f88a6366aaed4c41a0b41d56ed1c3fc8dcb63cc2775e6f8dc2`
