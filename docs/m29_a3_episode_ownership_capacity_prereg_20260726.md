# M29-A3 Episode Ownership Capacity — M01 Kill Gate

Baseline C0 is the frozen M28-B1 teacher on the strict M23-46 M01 geometry. Candidate generation reads only M23-46 boxes/IDs and phase-cache ReID features available by each fixed three-observation decision frame. GT is forbidden before candidate parquet and manifest freeze.

C1 re-certifies mature visibility episodes (previous history >=9 rows) by assigning the current contiguous episode to one of eight past identity candidates. C2 proposes reciprocal ownership swaps between two mature overlapping visibility episodes, using only past anchors, the first three current observations, spatial proximity, and prior causal close-frame count. A single C1/C2 output transform overlaps historical interval/transaction actions; no novelty claim is allowed unless the composed two-stage gate passes.

Fixed M01 gate:
- C1 exact HOTA gain over C0 >= +0.15;
- C2 incremental exact HOTA gain over C1 >= +0.15;
- final exact/official HOTA >= 80.0;
- official IDSW <= C0 IDSW 42;
- geometry unchanged, no duplicate frame/identity, future feature reads = 0.

Failure closes episode ownership without four-sequence expansion or student training.
