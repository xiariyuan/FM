# M28-B0 Deferred Identity Inheritance on M23-46 — Preregistration

Host: the frozen strict sequence-LOSO M23-46 tracker. Each final tracker ID is treated as an anonymous geometric track at birth. The decision event is its second observation; only the first two observations may describe the young track. An old identity candidate may use only rows and ReID features strictly before the young track's first observation. Top-k=8, maximum gap=120, and the M28 appearance-motion-scale score are fixed.

After candidate parquet and hashes are frozen, GT may screen dominant-identity-consistent actions and exact HOTA may label them. Any action producing a future duplicate frame/identity is invalid. The action assigns the old external identity to all rows of the young track, which is implementable with bounded output buffering from birth.

MOT20-01 kill gate: at least 5 positive valid actions, best single delta at least +0.05 HOTA, combined delta at least +0.25 HOTA, and official IDSW nonincrease. Passing authorizes the same protocol on all four sequences. Final all-four authorization requires teacher combined HOTA at least 80.80 before student training.
