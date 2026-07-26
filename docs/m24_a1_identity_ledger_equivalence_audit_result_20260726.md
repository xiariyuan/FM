# M24-A1 Identity Ledger / State Transplant Equivalence Audit

Decision: **CLOSE_M24_CORA_CURRENT_FORM**

The strict deployable best remains M23-46 at HOTA `79.123193`. No training, MOT20 test read, or test submission occurred.

## Equivalence findings

| Proposed action | Existing action family | Equivalence | Existing evidence |
|---|---|---|---|
| Swap two identity roots from time `t` onward | M23-47 reciprocal suffix swap | Exact at output-partition level | Combined HOTA `79.159210`, only `+0.036017` over M23-46 |
| Exchange bounded middle identity segments | M23-48 four-boundary interval swap | Exact at output-partition level | Zero positive actions on expanded M02 gate |
| Directed donor/receiver transfer with conflict deletion | M23-51 residual transaction replacement/drop | Same graph transaction family | Teacher HOTA `79.719687`, still `0.280313` below 80; strict M23-52 failed |
| Force alternate Hungarian assignment and roll out cloned state | No exact prior equivalent | Mechanically new | Best true-online C2 gain only `+0.004000` HOTA on M01 |
| Restore or hold appearance identity root while motion continues | No exact prior equivalent | Distinct state repair | Exploratory M01: `1/16` positive; best `+0.139000` on a weaker online host |

## Conclusion

The proposed identity ledger does not rescue the current CORA formulation. Once observable tracker outputs are considered, root swaps and interval transplants collapse to action families already tested and closed in M23-47, M23-48, and M23-51.

True tracker-state cloning and branch-conditioned rollout were successfully implemented. They are useful for rejecting harmful alternatives: C2 beat paired static C0 in `12/16` M01 events. However, the best C2 action improved the online baseline by only `+0.004000` HOTA, so it functions as a safety verifier rather than a corrective mechanism.

Appearance-root repair is genuinely different, but current evidence is far below the level needed to justify a CCF-A mainline. It must not be expanded by threshold or model tuning.

## Research decision

- Do not train a CORA value/risk model.
- Do not run the current action space on all four sequences.
- Do not rename suffix/interval transactions as an identity ledger contribution.
- Reset the research problem. A future CCF-A idea must introduce new identity evidence or supervision rather than another permutation of existing track IDs and states.
