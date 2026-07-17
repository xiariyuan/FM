# M23-4-0 MOT20 Existing-ID Internal Gap-Bridge Oracle Audit

Date: 2026-07-17
Repository: `FM-Track`
Scope: MOT20 train sequences `01/02/03/05`
Status: **completed, reproduced, fail-closed**

## Research question

After M23-3-1 showed that same-ID box replacement has only a +0.30231 HOTA oracle ceiling, M23-4-0 tests whether missing observations inside short gaps of an existing baseline identity can provide a useful deployable-form ceiling.

The action space is deliberately narrow:

- start from the exact M23-3-1 dense replacement context;
- use consecutive segments of the same baseline tracker ID;
- require the two segments to have the same positive modal MOT20-train GT support;
- fixed gap length 1–30 frames;
- add a candidate only when the inherited ID is absent in that frame;
- no new identity namespace, no identity relabeling, no suppressor routing;
- one fixed global frame-level Hungarian match at IoU >= 0.5;
- no parameter, threshold, or gap sweep.

## Frozen plans

| Plan | Added rows | Definition |
|---|---:|---|
| All additions | 3,825 | Every GT-matched internal-gap candidate |
| Safe additions | 829 | Candidate has maximum IoU < 0.5 against every dense-context row in the frame |

Of 3,825 matched candidates, 2,996 overlap an existing context row at IoU >= 0.5. Thus 78.33% of the evidence is not a missing-detection problem; it is an identity-conflict/reassignment problem.

## TrackEval results

| Variant | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline raw | 77.699020 | 80.907240 | 74.672127 | 89.308469 | 93.606460 | 1,222 | 18,566 | 52,754 |
| Dense replacement context | 78.001330 | 81.267375 | 74.915320 | 89.336889 | 93.688250 | 1,220 | 18,118 | 52,276 |
| Safe gap bridge | 78.035430 | 81.312310 | 74.939580 | 89.363428 | 93.757789 | 1,228 | 18,134 | 51,463 |
| All gap additions, diagnostic | 78.139540 | 81.445354 | 75.017410 | 89.449685 | 94.007566 | 1,104 | 18,270 | 48,617 |

The preregistered acceptance condition was safe HOTA >= 78.10 with nonnegative sequence-level HOTA deltas. Sequence safety passed, but the combined ceiling failed:

- safe gain over dense context: **+0.034100 HOTA**;
- target gain required: +0.098670 HOTA;
- shortfall: **0.064570 HOTA**.

Safe additions reduce FN by 813 but increase FP by 16 and IDSW by 8. Therefore they predominantly improve detection coverage and do not repair the identity graph.

The all-additions diagnostic gains +0.138210 HOTA and reduces IDSW by 116, but it includes 2,996 candidates that overlap existing observations. It is not an acceptable additive policy; its value is diagnostic evidence that reassignment/relinking is the missing action.

## Per-sequence safe deltas

| Sequence | Dense context | Safe bridge | Delta |
|---|---:|---:|---:|
| MOT20-01 | 77.311534 | 77.362096 | +0.050562 |
| MOT20-02 | 68.966925 | 68.977240 | +0.010315 |
| MOT20-03 | 80.338050 | 80.387240 | +0.049190 |
| MOT20-05 | 78.871155 | 78.902140 | +0.030985 |


All four deltas are nonnegative; the worst is MOT20-02 at +0.010315.

## Decision

**Close same-ID internal gap filling. Do not train a same-ID gap selector.**

The next justified experiment is M23-5-0: a cross-ID tracklet relinking/reassignment action-space oracle over the 2,996 overlapping conflict events. It must replace or relabel conflicting context evidence rather than add duplicate observations.

Deployment remains false. No locked manifest was created. P15 remains no-op and all 156 remaining locked rows remain unread.

## Reproducibility

- formal/reproduction compact files: 7/7 byte-identical;
- tracker files: 16/16 byte-identical;
- TrackEval summary/detailed files: 8/8 byte-identical;
- unified audit: 64/64 checks passed.

## Hashes

- formal script: `6a3fe917c9ec874e005e82d0c5d56b894a98539bdcbc549dad1a095cca568190`
- preregistration: `15f439a76cbb802779183e5e20cdab03b06ddefd4f5c923594069e8675b5295f`
- formal report: `4408d6928fb35025a5d6b79a89c61ff6789cbc25ebabd84a194228cfb08567f2`
- formal manifest: `09dcd13bc9b3afefc3283c09bddff0fcfa8813774f97a57d95e52d3776f3f290`
- unified audit script: `3774e411ce81b1785d87285d798367a2f502866539a454f059ddd87d3337e7ed`
- unified audit JSON: `c87832e2d227039068e129a1ca98c9e3a03cde82dc335edbce802948a242a62d`
