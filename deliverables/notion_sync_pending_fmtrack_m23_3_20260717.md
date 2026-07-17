# FM-Track M23-3-0 sync payload

Status: completed, oracle action-space audit, fail-closed

## Result

M23-3-0 audited a candidate-default graph action space under the fixed M23-1 post-NMS oracle context. Of 187,596 selected pre-NMS suppressed events, 170,809 (91.05%) are candidate-default, only 123 (0.066%) need suppressor override, and 16,664 (8.88%) require spawn/re-linking.

TrackEval: post-NMS context 83.676 HOTA; candidate-default safe 84.226; candidate+suppressor safe 84.227; 30-frame episode spawn 83.597 with 17,190 IDSW; oracle-linked spawn 84.523 and exactly reproduces M23-1. Candidate+suppressor is positive on all four sequences, worst HOTA 83.893.

## Decision

Retain candidate-default action space. Do not build a separate learned suppressor branch because it adds only 0.001 rounded HOTA. Reject direct new-ID spawn because it causes severe association fragmentation. Appearance remains auxiliary evidence only.

Next: M23-3-1 sequence-LOSO segment-conditioned observation replacement graph with baseline no-op, one observation per segment/frame, candidate inheritance only, spawn disabled, and suppressor diagnostic only.

## Audit

The first preregistered run failed before spawn metrics because IDs near 1.5e9 collided numerically in TrackEval. v2 changed only the synthetic namespace to exactly representable IDs below 2^24; all gates and action definitions stayed fixed. Formal/reproduction compact files 8/8, trackers 20/20, TrackEval files 10/10 byte-identical; unified audit 60/60 passed.

P15 remains no-op: zero locked-label reads, zero locked TrackEval calls, 156 rows untouched.

Notion write has not been performed because no writable Notion connector is available in this session.
