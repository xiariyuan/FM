# FM-Track P20 Notion Sync Pending

Target page ID: `39dabff3-387a-8125-99da-caec2ec8f7ec`

## Title

P20 — Nested conformal candidate sets and mechanism-specific failure discovery

## Summary

P20 replaced unsafe scalar authorization with strict nested sequence-LOSO set-valued retrieval on the full 11,705-event MOT17 teacher bank.

The rank-conformal set uses the union of three fixed P19 views and calibrates its radius from completely inner-OOF source blocks at alpha 0.10.

Results:

- positive-available blocks: 23;
- covered blocks: 22;
- conditional coverage: 95.6522%;
- mean set size: 46.5714;
- maximum set size: 87;
- oracle set utility sum: +6.314904;
- oracle worst sequence: +0.286058.

A positive-manifold conformal set covered 23/23 blocks but required a mean of 310.5714 candidates and was rejected. A block-level OOD certificate also failed to identify the sole rank-set miss.

The missed MOT17-11 block contains three positive events:

- two ephemeral-anchor receiver-split events;
- one same-identity merge event.

The receiver-split mechanism uses a short or unmatched donor label as a fresh namespace to split a receiver identity change. It is not a donor-to-future motion transfer event, explaining the failure of the current motion and nearest-positive support views.

Decision:

- retain the rank-conformal set as a retrieval layer;
- do not deploy a transaction policy;
- reject the broad positive-support set and current OOD certificate;
- do not enlarge the generic rank radius;
- next build receiver change-point and ephemeral-anchor split features as a mechanism-specific rescue head.

Locked status:

- P15 locked labels read: 0;
- P15 locked TrackEval calls: 0;
- global TrackEval calls: 0;
- remaining unread locked rows: 156;
- P15 policy: no-op;
- locked manifest: not created.

Formal report:

`deliverables/assocriskbench_p20_nested_conformal_sets_20260716.md`

Unified audit:

`deliverables/assocriskbench_p20_audit_20260716.json`

Notion write was not performed because no writable Notion connector was available in this session.
