# FM-Track M23-6-0 Notion Sync Pending

Target page: `39dabff3-387a-8125-99da-caec2ec8f7ec`
Status: **pending — no writable Notion connector available in this session**

## Result

M23-6-0 audited long-horizon identity chains from the exact all-additions context while preserving all rows and boxes.

- source HOTA 78.139540, IDSW 1104;
- pure segment chain HOTA 78.647774, IDSW 1274;
- modal segment chain HOTA 80.994950, IDSW 949;
- modal-core chain HOTA 82.298195, IDSW 2623;
- full matched identity ceiling HOTA 83.086020, IDSW 4.

Preregistered modal and modal-core HOTA thresholds pass, but their IDSW safety thresholds fail. The overall chain ceiling is therefore fail-closed and the current graph representation must not be trained.

The identity direction remains feasible because the full identity-only ceiling exceeds the 82.5 target.

Boundary forensic:

- 5,572 modal runs across 3,914 positive segments;
- 5,245 chain/source boundary toggles;
- 1,307 internal gaps contain another GT and require hard boundaries;
- 351 unmatched-only gaps can safely fill 927 rows;
- purified support spans retain 99.954% modal matches.

Next: M23-7-0 other-GT hard boundaries + unmatched-hole fill + span-level WIS relinking.

Reproduction: compact 6/6, trackers 20/20, TrackEval 10/10 byte-identical; unified audit 110/110 passed.

P15 remains no-op; locked-label reads 0, locked TrackEval calls 0, 156 rows untouched.

Hashes:

- report `18e38562f26f1cfe0e758f3bdb6d856eabd0f6a2154e0630a3acfc500b884824`
- manifest `94bc44098966ec6e04cd67d09f45b0a15d9cdbe255ea79c2df6f5cc41febb449`
- unified audit `bc3270b1afde18f88a6366aaed4c41a0b41d56ed1c3fc8dcb63cc2775e6f8dc2`
