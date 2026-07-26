# M29-A0 M23-46 Identity-Episode Attribution — Result

Decision: **FAIL_M29_A0_EPISODE_BIRTH_NOT_DOMINANT**

## Integrity

- The frozen four-sequence M23-46 tracker was audited without modifying tracker output.
- Exact CLEAR reconstruction recovered all `996` official M23-46 identity switches.
- The frozen event table contains `996` unique `(sequence, frame, GT identity, tracker identity)` rows and no duplicate event keys.
- All four tracker SHA-256 values match the files used by the audit.
- MOT20 test reads and submissions: `0`.
- This is a MOT20-train post-hoc diagnostic; it is not deployable and does not report a new HOTA result.

## Attribution

| Sequence | Reconstructed IDSW | Episode age 1–3 | Rate | New full track | Re-entry episode |
|---|---:|---:|---:|---:|---:|
| MOT20-01 | 46 | 22 | 0.478261 | 16 | 6 |
| MOT20-02 | 325 | 130 | 0.400000 | 106 | 25 |
| MOT20-03 | 146 | 48 | 0.328767 | 43 | 5 |
| MOT20-05 | 479 | 211 | 0.440501 | 182 | 32 |
| **COMBINED** | **996** | **411** | **0.412651** | **347** | **68** |

Combined episode-age categories:

- observations 1–3: `411` (`41.2651%`);
- observations 4–8: `39` (`3.9157%`);
- observation 9 or later: `546` (`54.8193%`).

## Gate

The preregistered authorization gate required both:

- combined episode-age-1–3 IDSW share at least `0.45`;
- every sequence episode-age-1–3 share at least `0.25`.

The per-sequence condition passed, but the combined share was `0.4126506024`, short of the fixed threshold by `0.0373493976`. The gate therefore failed and no deferred episode-identity capacity experiment is authorized from M29-A0.

## Interpretation

Early visibility-episode identity allocation is a substantial source of M23-46 ID switches, but it is not the dominant combined mechanism under the preregistered definition. Most switches (`546/996`) occur at episode age 9 or later. The current episode-birth hypothesis is therefore closed rather than promoted to a tracker intervention or student-training stage.

The historical strict deployable M23-46 result remains HOTA `79.123193`, DetA `81.543470`, AssA `76.825150`, and IDSW `996`. M29-A0 changes none of those values.

## Artifacts

- Script SHA-256: `00ea3035b4cd7487b917048728621cc64d6219a1be3f91c963fddd8135696268`.
- Preregistration SHA-256: `b9d2a828cf6006b20f3bea8e064eb6a4fbeac6cbaf207f38cced1982b9be47c2`.
- Report SHA-256: `2f7a2f21e72efbbde01b4778509ea84dfbd94c1ae65ba79050a951d00089e62f`.
- Per-sequence summary SHA-256: `2ba37a4ef39f1d7b89a1f6403a26c424667d5bf26e2a3840c391805e12482aef`.
- Frozen event-table SHA-256: `8ac1c8a33b04133847529f142b5f39be9d5b106a7fd0b11a2edcb9d84dd34f22`.
- Run root: `outputs/mot20_m29_20260726/m29_a0_m23_46_identity_episode_attribution`.
