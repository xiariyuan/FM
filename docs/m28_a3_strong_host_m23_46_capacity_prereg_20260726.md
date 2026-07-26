# M28-A3 Strong-Host M23-46 Deferred Identity Capacity — Preregistration

- Host: strict M23-46 MOT20-01 tracker, byte frozen.
- Event: second valid ReID observation of each new M23-46 ID.
- Candidate: prior non-overlapping ID ending before birth, gap 1–120, at least four rows and two valid ReID rows.
- Candidate top-k: 8 using fixed M28 appearance-motion-scale score.
- Candidate table is frozen before GT.
- Teacher opens GT only after freeze, screens dominant-identity-consistent actions, and evaluates exact HOTA.
- Action: assign the complete young ID to the old identity; equivalent online behavior uses a two-observation output buffer.
- M01 GO gate: at least five positive actions, combined delta HOTA at least +0.30, and IDSW no greater than M23-46 M01 baseline 46.
- Teacher-only, non-deployable, MOT20 test reads/submissions: 0.
