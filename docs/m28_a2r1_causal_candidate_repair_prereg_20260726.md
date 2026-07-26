# M28-A2-R1 Causal Candidate Repair — Preregistration

Independent repair audit. Existing M28-A0/A2 artifacts are immutable.

Candidate generation may read only tracker rows and ReID update features whose frame is no later than the decision frame; old-identity evidence must be strictly earlier. It may not use full-sequence last-frame, future overlap, future removal, GT, or teacher labels. Top-k=8 and max gap=120 remain fixed.

After candidate parquet and hashes are frozen, GT may be opened for exact-HOTA labels. Actions that cause any future duplicate frame/identity are invalid teacher actions.

M01 pass gate: at least 8 positive actions, best single delta at least +0.10 HOTA, combined exact delta at least +0.50 HOTA, and official IDSW nonincrease. Passing authorizes the same repair on all four sequences; failing closes M28 before student training.
