# M23-5-0 MOT20 Mechanism-Decomposed NMS-Conflict Oracle Audit

Date: 2026-07-17
Repository: `FM-Track`
Scope: MOT20 train sequences `01/02/03/05`
Status: **completed, reproduced, fail-closed**

## Research question

M23-4 showed that 2,996 of 3,825 internal-gap candidates overlap an existing context observation at IoU >= 0.5. A naive interpretation is that these candidates should replace or relabel the host observation. Structural review rejected that assumption: most host observations represent a different pedestrian in the crowded scene.

M23-5-0 therefore preregistered a mechanism-decomposed oracle:

- **other target**: retain the valid host and add the suppressed candidate under the inherited existing ID;
- **same target**: migrate the host to the inherited ID and use whichever source/candidate box has higher target IoU;
- **unmatched host**: replace the host with the candidate;
- no new identity namespace;
- no model, threshold sweep, parameter sweep, or locked evaluation.

## Frozen action inventory

| Mechanism | Selected actions | Fixed action |
|---|---:|---|
| Different valid pedestrian | 2,603 | Dual retention: keep host, add candidate |
| Same GT pedestrian | 384 | Migrate/merge; select higher-target-IoU box |
| Host unmatched to valid GT | 7 | Replace host with candidate |
| Source-row collisions | 2 | Fixed preregistered tie-break |
| Total unique actions | 2,994 | — |

For the 384 same-target actions, the candidate box is better in 172 cases; the source box is better or tied in 212. This confirms that a fixed candidate-box replacement would not be a valid ceiling.

## Combined TrackEval results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense replacement context | 78.001330 | 81.267375 | 74.915320 | 89.336889 | 93.688250 | 1,220 | 18,118 | 52,276 |
| Safe gap context | 78.035430 | 81.312310 | 74.939580 | 89.363428 | 93.757789 | 1,228 | 18,134 | 51,463 |
| All additions reference | 78.139540 | 81.445354 | 75.017410 | 89.449685 | 94.007566 | 1,104 | 18,270 | 48,617 |
| Distinct-person dual retention | 78.096470 | 81.393456 | 74.982340 | 89.417545 | 93.920576 | 1,198 | 18,109 | 49,671 |
| Mechanism-decomposed oracle | 78.102845 | 81.399244 | 74.989170 | 89.421169 | 93.925511 | 1,167 | 18,097 | 49,658 |
| Safe gap + mechanism oracle | 78.137070 | 81.444120 | 75.013727 | 89.447558 | 93.996284 | 1,163 | 18,112 | 48,844 |


## Preregistered gate decision

Primary gates:

- mechanism HOTA >= 78.20;
- safe + mechanism HOTA >= 78.25;
- all four sequence-level HOTA deltas nonnegative;
- IDSW nonincreasing against the corresponding context.

Results:

- mechanism HOTA: **78.102845**, gain over dense **+0.101515**;
- mechanism IDSW change: **-53**;
- safe + mechanism HOTA: **78.137070**, gain over safe **+0.101640**;
- safe + mechanism IDSW change: **-65**.

Both identity-safety conditions pass and every sequence is directionally positive, but both HOTA gates fail. Deployment remains false.

## Per-sequence mechanism deltas

| Sequence | Dense | Mechanism | Delta | Safe | Safe + mechanism | Delta |
|---|---:|---:|---:|---:|---:|---:|
| MOT20-01 | 77.311534 | 77.529550 | +0.218016 | 77.362096 | 77.582645 | +0.220549 |
| MOT20-02 | 68.966925 | 69.153520 | +0.186595 | 68.977240 | 69.163930 | +0.186690 |
| MOT20-03 | 80.338050 | 80.395740 | +0.057690 | 80.387240 | 80.444956 | +0.057716 |
| MOT20-05 | 78.871155 | 78.974280 | +0.103125 | 78.902140 | 79.005400 | +0.103260 |


## Mechanism interpretation

- Dual retention alone contributes **+0.095140 HOTA**.
- Same-target migration and unmatched replacement add only **+0.006375 HOTA** beyond dual retention.
- The complete all-additions reference still exceeds safe + mechanism by **+0.002470 HOTA** and has **59 fewer ID switches**.

The last result is important: frame-local migration can reduce individual row ambiguity but damage longer temporal identity continuity. The remaining bottleneck is not a per-frame NMS action classifier. It requires episode- or tracklet-level identity reasoning.

## Decision

**Close the local NMS-conflict action space. Do not train a local mechanism selector.**

The next justified audit is M23-6-0: a longer-horizon cross-tracklet identity graph oracle with segment/episode actions, explicit temporal non-overlap, and identity-chain utility. This stage should determine whether tracklet-level relinking has enough ceiling before any graph-model training.

P15 remains no-op. Locked-label reads: 0. Locked TrackEval calls: 0. All 156 remaining locked rows remain unread.

## Reproducibility

- compact files: 6/6 byte-identical;
- tracker files: 24/24 byte-identical;
- TrackEval summary/detailed files: 12/12 byte-identical;
- unified audit: 109/109 checks passed.

## Hashes

- formal script: `f14a2275e7501b1701d850571c76eba9a7e2220ad2ff4d810c1e7c1309449d14`
- preregistration: `7f631f1490f56fc8ad6be779ae8902bd914c8c8a161eab5982c7db7409ef48af`
- formal report: `a5aeefc673ba1e189741e44b1888c3ca1217110e434fe72546c53baa83050082`
- formal manifest: `06ef361a5599529347836e5cb1787150ceb6c6be875be936b8f5da078d05a16e`
- unified audit script: `2e6dad658bdeb8a1cef5512eee10df07d6a3cc17d96df9c036e5da9c7324b168`
- unified audit JSON: `9ac6265d86421c9f7f05a1cd14c26511083a8f942e02eb902a61818e288c52e8`
