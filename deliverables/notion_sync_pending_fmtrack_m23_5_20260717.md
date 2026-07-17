# FM-Track M23-5-0 Notion Sync Pending

Target page: `39dabff3-387a-8125-99da-caec2ec8f7ec`
Status: **pending — no writable Notion connector available in this session**

## Result

M23-5-0 audited 2,994 fixed NMS-conflict actions after mechanism decomposition:

- 2,603 different-person dual-retention actions;
- 384 same-target migration/merge actions;
- 7 unmatched-host replacements;
- 2 source-row collisions resolved by a preregistered tie-break.

TrackEval:

- dense context HOTA 78.001330;
- dual retention HOTA 78.096470;
- mechanism oracle HOTA 78.102845, delta +0.101515, IDSW -53;
- safe context HOTA 78.035430;
- safe + mechanism HOTA 78.137070, delta +0.101640, IDSW -65;
- all-additions reference HOTA 78.139540, IDSW 1104.

Preregistered gates 78.20 and 78.25 both failed, although all four sequences improved and IDSW decreased. Local NMS-conflict actions are therefore closed; do not train a local mechanism selector.

Next stage: M23-6-0 longer-horizon cross-tracklet identity graph oracle using segment/episode actions rather than frame-local edits.

Reproduction: compact 6/6, trackers 24/24, TrackEval 12/12 byte-identical; unified audit 109/109 passed.

P15 remains no-op; locked-label reads 0, locked TrackEval calls 0, 156 rows untouched.

Hashes:

- report `a5aeefc673ba1e189741e44b1888c3ca1217110e434fe72546c53baa83050082`
- manifest `06ef361a5599529347836e5cb1787150ceb6c6be875be936b8f5da078d05a16e`
- unified audit `9ac6265d86421c9f7f05a1cd14c26511083a8f942e02eb902a61818e288c52e8`
