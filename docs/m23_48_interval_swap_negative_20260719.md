# M23-48 four-boundary interval-swap audit

Date: 2026-07-19

## Action

M23-48 exchanges two time-aligned middle intervals from two different current chains. Each action removes four source boundaries and adds four GT-free synthesized, time-forward cross edges. Every candidate must remain one-to-one and acyclic.

Candidate construction uses only chain structure, projected appearance prototypes, motion, timing, rank, consistency, match-IoU diagnostics that do not use GT, and interval size/alignment. Exact HOTA teacher is opened only after shortlist freeze. All results are `teacher_only=true` and `deployable=false`.

## Smoke

MOT20-02 smoke froze 8 GT-free candidates. All 8 reconstructed successfully and passed one-to-one/acyclic checks. All 8 decreased HOTA; best delta was -0.026566.

## Expanded M02 capacity

The expanded bank used up to 512 boundaries and 512 interval descriptors. Structural constraints yielded 54 feasible candidates.

- Successful exact labels: 54/54
- Positive actions: 0
- Negative actions: 54
- Best single-action delta: -0.013894 HOTA
- Final policy: no-op
- Baseline/final HOTA: 73.098153

## Decision

M23-48 is closed. It is mechanically valid but has no positive capacity on the primary M02 gate, so it must not be expanded to M05 or four folds. Further work should return to the transaction action family whose existing exact teacher bank contains many large positive actions, and use the low-capacity pure-HGB top-k nested-LOSO strategy that succeeded for M23-46 source cuts.
