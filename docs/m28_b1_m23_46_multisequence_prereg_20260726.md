# M28-B1 Deferred Identity Inheritance on M23-46 — Four-Sequence Preregistration

The M23-46 strict sequence-LOSO tracker is frozen as host. The M28-B0 MOT20-01 protocol is applied without changes to MOT20-02, MOT20-03, and MOT20-05: a final tracker ID is anonymous at its first observation; the decision occurs at its second observation; the young prototype uses at most those two observations; every old candidate uses only rows and ReID features strictly before the young birth. Top-k=8 and maximum gap=120 are fixed.

GT may be opened only after each candidate parquet and manifest is frozen. Dominant-identity-consistent actions are evaluated with exact HOTA; actions causing a future duplicate frame/identity are invalid. The external identity action rewrites the young ID from birth and is implementable with bounded output buffering.

Authorization gate before student training:
- every sequence has positive HOTA delta;
- every sequence has at least three positive valid actions;
- official four-sequence teacher combined HOTA is at least 80.80;
- combined IDSW does not increase;
- candidate generation has zero future-row reads on every sequence.

If combined teacher HOTA is below 80.80, no student is trained on this host.
