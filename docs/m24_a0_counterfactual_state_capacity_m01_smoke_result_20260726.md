# M24-A0 Counterfactual State-Causal Capacity — MOT20-01 Smoke Result

Decision: **FAIL_M24_A0_CORRECTIVE_CAPACITY_GATE**

This is a mechanism smoke test, not a strict four-sequence deployable result. The strict deployable best remains M23-46 at HOTA `79.123193`.

## Verified mechanism

- `DMMBaseTracker` can be deep-cloned deterministically when its global ID counter is snapshotted.
- Identical clones produce identical outputs, Kalman states, appearance histories, and lifecycle states.
- A forced alternate first-stage Hungarian assignment creates genuine branch-local state divergence and changes future candidate/output trajectories.

## MOT20-01 results

| Stage | Baseline HOTA | Best HOTA | Delta |
|---|---:|---:|---:|
| Carrier C0 static | 78.805125 | 78.998303 | +0.193179 |
| Carrier C1 state + fixed candidates | 78.805125 | 78.930360 | +0.125235 |
| Carrier C2 regeneration | 78.805125 | 78.823793 | +0.018668 |
| True-online C0 static suffix | 76.737000 | 77.103000 | +0.366000 |
| True-online C2 rollout | 76.737000 | 76.741000 | +0.004000 |
| Memory hold, exploratory | 76.737000 | 76.876000 | +0.139000 |
| Defer/reseed, exploratory | 76.737000 | 76.755000 | +0.018000 |

True C2 rollout beat its paired static C0 alternative on `12/16` events, with mean paired gain `+0.305500` HOTA. However, this mainly reflects rejection or self-correction of harmful static swaps. Its best absolute gain over B0 was only `+0.004000` HOTA.

The current local pair-swap/defer action space therefore lacks corrective capacity. Running all four sequences or training a value model is not authorized.

P2 and P3 were explicitly exploratory and were implemented only after P1 GT evaluation; they must not be presented as preregistered evidence.

## Next decision

Before any further training, compare a proposed identity-root ledger/state-transplant action with the existing M23-47 suffix-swap lineage. If the new action is only a relabelled suffix swap, close M24 rather than repackaging prior postprocessing.
