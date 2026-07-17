# FM-Track M23-4-0 Notion Sync Pending

Target page: `39dabff3-387a-8125-99da-caec2ec8f7ec`
Status: **pending — no writable Notion connector available in this session**

## Result

M23-4-0 audited a fixed 30-frame existing-ID internal gap bridge from the exact M23-3-1 dense replacement context.

- 1,365 eligible same-support segment pairs;
- 6,381 visible internal-gap GT frames;
- 3,825 GT-matched candidates;
- 2,996 overlap existing context observations at IoU >= 0.5;
- 829 context-exclusive safe additions.

TrackEval:

- dense context HOTA: 78.001330;
- safe bridge HOTA: 78.035430, delta +0.034100;
- preregistered safe gate: 78.10 — failed;
- all-additions diagnostic HOTA: 78.139540, but this plan contains overlapping duplicate/conflict evidence and is not deployable.

Decision: close same-ID gap filling and do not train a same-ID gap selector. Continue with M23-5-0 cross-ID tracklet relinking/reassignment oracle on the 2,996 overlapping conflict events.

Reproduction: compact 7/7, trackers 16/16, TrackEval 8/8 byte-identical; unified audit 64/64 passed.

P15 remains no-op; locked label reads 0, locked TrackEval calls 0, 156 rows untouched.

Hashes:

- report `4408d6928fb35025a5d6b79a89c61ff6789cbc25ebabd84a194228cfb08567f2`
- manifest `09dcd13bc9b3afefc3283c09bddff0fcfa8813774f97a57d95e52d3776f3f290`
- audit `c87832e2d227039068e129a1ca98c9e3a03cde82dc335edbce802948a242a62d`
