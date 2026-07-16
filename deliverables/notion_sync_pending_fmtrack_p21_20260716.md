# Notion Sync Pending — FM-Track P21

Target page: `39dabff3-387a-8125-99da-caec2ec8f7ec`

## Stage

AssocRiskBench P21 — mechanism-aware conformal rescue sets.

## Summary

P21 froze the P20 rank-conformal candidate set and added two sequence-disjoint mechanism heads:

- receiver-split / ephemeral-anchor namespace split;
- same-identity merge.

A new 75-feature deployable bank was built from tracker rows and executable-event metadata only. The full nested model uses 130 deployable features and no GT, utility, TrackEval, or locked columns as inputs.

## Main result

- P20 positive-block coverage: 22/23.
- P21 combined coverage: 23/23.
- Newly covered block: MOT17-11-FRCNN temporal block 3.
- Mean rescue additions: 98.857143.
- Mean combined set size: 145.428571.
- Maximum combined set size: 218.
- Mechanism rescue retained: false.
- Deployment allowed: false.

## Statistical conclusion

At alpha 0.10, at least 19 positive calibration blocks are required for the finite-sample higher conformal quantile not to equal the maximum. P21 has only 14–17 receiver-split blocks and 8–11 same-identity-merge blocks per outer fold.

All 14 outer × mechanism calibrations therefore use the maximum source-block rank. The median radius is 13.5 times the median positive rank for both mechanisms. This converts useful mechanism rankings into an excessively broad set.

## Decision

Reject the P21 block-rank conformal rescue set despite achieving 23/23 coverage. Keep:

- the 75-feature deployable mechanism bank;
- the mechanism definitions;
- the sequence-disjoint mechanism ranking evidence;
- the frozen P20 rank-conformal base set.

P22 should replace block-rank calibration with cluster-aware event-level risk control or structured mechanism-conditioned pairwise certificates. Do not enlarge the generic P20 radius and do not return to scalar utility-gate tuning.

## Integrity

- feature formal/repro: byte-identical;
- audit formal/repro: byte-identical;
- feature report SHA256: `4e1a52057a51307fc3c0206f252548485cd3c6cea97153d9c2969f72ff9306bb`;
- audit report SHA256: `26cc7d7d0cc8b2ae93fed3da6b12c61e8d447ff508c494154f263df61c81378a`;
- unified audit SHA256: `5b3ab03ef0357b99a8ef19dccee27fa80a8a36de3accb2219cbc2a11211e23a2`;
- locked reads/calls: 0;
- remaining locked rows: 156;
- P15 policy: no-op.

This file remains pending because no writable Notion connector was available in the session.
