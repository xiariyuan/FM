# FM-Track P22 Notion Sync Pending

Target page: `39dabff3-387a-8125-99da-caec2ec8f7ec`

Status: pending because no writable Notion connector is available in the current session.

## P22 headline

P22 froze the P20 generic rank-conformal set and the P21 sequence-OOF mechanism scores. A fixed rescue budget of 12 candidates per temporal block was inherited unchanged from the preregistered P21 efficiency limit; no budget, threshold, model, or alpha sweep was performed.

The budgeted mechanism union recovered the sole P20 miss:

- positive-block coverage: 23/23
- mean rescue additions: 12.0
- maximum combined set size: 99
- newly covered block: MOT17-11-FRCNN block 3
- recovered event: canonical 532, same-identity merge, utility +0.183036
- compact candidate set retained: true

The set is not deployment-authorized:

- rescue positives: 39
- rescue negatives: 293
- block-level 90% upper miss bound: 0.095264
- sequence-cluster 90% upper miss bound: 0.280314
- cluster certificate passed: false
- deployment allowed: false

The difference between the block and sequence bounds demonstrates that the four blocks within each sequence cannot be treated as independent safety samples.

## Appearance preparation

A deterministic sparse boundary ReID manifest was created without running feature extraction:

- events: 11,705
- event-window rows: 163,958
- unique crops: 60,708
- unique image frames: 3,284
- roles: donor history, receiver history, receiver future
- 1–5 rows per event role
- duplicate crop keys: 0
- duplicate event-role positions: 0

The MOT17 FastReID config and weights are available and hashed, but the host inference service returned VM service unavailable before model inference. No appearance embeddings were generated or claimed.

## Decision

- Retain P22 as the compact candidate retrieval set.
- Do not deploy or create a locked manifest.
- Keep P15 no-op and leave all 156 remaining locked rows unread.
- P23 should run sparse appearance extraction when inference service is available, then prune/rerank only the 336 P22 rescue candidates under strict sequence-LOSO.

## Reproducibility

Both formal chains reproduce byte-identically.

- budget report SHA256: `bf1acddfca1003682e76c752e4dd57a61b46e45a396cb05e8662d5358260f4cf`
- sparse manifest report SHA256: `dc3df76f527d454dfc7a8a99e384e1ddd3b47d9fe5ffdd090fc8a681535befa9`
- unified audit SHA256: `d72bd85983ec91274061ec2f0814d46d6e5e74afbb955f14525d55479b1f9617`
