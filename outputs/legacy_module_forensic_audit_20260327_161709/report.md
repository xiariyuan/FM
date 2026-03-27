# Legacy Module Forensic Audit

Date: 2026-03-27T16:17:11

## Scope

This audit revisits two older idea families that were implemented before the current diagnosis-driven official-ByteTrack work:

- `frequency family`
- `laplace family`

The goal is not to rerun them. The goal is to answer a narrower question:

- were they truly "not validated",
- or were they run already but never investigated deeply enough to understand why they failed or where they actually worked?

## Frequency Family

### Main findings

1. The original `v13_tf_plus_lowfreq_mamba` line was not merely "non-positive". It was optimization-unstable.
   - Main evidence: `/gemini/code/FMtrack-main/FM-Track/outputs/bytetrack_fa_mot_mot17_v13_tf_only_val0213/train/log.txt`
   - `nan_lines=6`
   - epoch 0 validation finishes, but the log reports `tracker root not found`, so the run never produces a clean usable validation artifact.
2. There were at least `5` follow-up `smoke_fixnan` rescue attempts.
3. The strongest rescue attempt that actually validates, `v13_tf_only_val0213_smoke_fixnan_v4`, is numerically stable but behaviorally collapsed.
   - best HOTA = `24.79`
   - best AssA = `9.55`
   - best IDF1 = `18.72`
   - best MOTA = `11.46`
   - best IDSW = `19693`
4. There is one conservative `SFI-lite` branch on MOT20 val05 that is stable:
   - HOTA = `68.65`
   - AssA = `66.02`
   - IDF1 = `80.55`
   - IDSW = `834`
   - but this is off the canonical MOT17 carrier and was never audited in a strict paired setting.

### Diagnosis

The frequency family failed in two stages:

- first on optimization stability,
- then on identity semantics after numerical stabilization.

So the missing work was not "we forgot to validate it". The missing work was that we never explicitly closed the loop and wrote down that the family moved from `NaN instability` to `stable but identity-collapsed`, which is a much stronger and more useful conclusion.

## Laplace Family

### Main findings

1. The fixed `v15` Laplace branch had a real positive regime on `proxy0213`.
   - base vs laplace evidence:
     - base: `/gemini/code/FMtrack-main/FM-Track/outputs/v15_laplace_proxy0213/base/tracker/MOT17-train/pedestrian_summary.txt`
     - laplace: `/gemini/code/FMtrack-main/FM-Track/outputs/v15_laplace_proxy0213/laplace/tracker/MOT17-train/pedestrian_summary.txt`
   - delta HOTA = `1.947`
   - delta AssA = `2.886`
   - delta IDF1 = `2.436`
   - delta MOTA = `0.264`
   - delta IDSW = `-56`
2. Pair-log evidence shows the Laplace trust signal is useful but slice-dependent.
   - `MOT17-02-FRCNN`: high-reliability bucket chosen-true-rate = `0.928846`
   - `MOT17-13-FRCNN`: high-reliability bucket chosen-true-rate = `0.452381`
3. The trainable `v16` Laplace gate did not explode, but it regressed steadily.
   - best epoch = `0` with HOTA `53.40`, AssA `45.16`, IDF1 `59.29`
   - latest recorded epoch = `2` with HOTA `51.97`, AssA `42.81`, IDF1 `57.76`, IDSW `744`

### Diagnosis

The Laplace family should not be labeled "invalid".

The stronger conclusion is:

- the heuristic/fixed branch worked on a real proxy slice,
- but the learned/trainable gate objective degraded that branch,
- and the hard slice calibration breaks badly on `MOT17-13-FRCNN`.

So the missing work was not proving that Laplace never worked. The missing work was proving exactly where it worked, and why the learned version made it worse.

## Final Verdict

1. `frequency family`
   - already ran
   - not merely unvalidated
   - failed first by instability, then by semantic collapse
2. `laplace family`
   - already ran
   - had a genuine positive regime on proxy0213
   - but was never carried into a clean canonical-carrier diagnosis package
   - and its learned gate version regressed

## What This Changes

The current project should not treat both older families as the same kind of "dead end".

- `frequency`: evidence now says the family needs a stability-and-semantics redesign before it deserves new budget.
- `laplace`: evidence now says the core intuition had value, but the learned calibration/supervision was the weak point.

This matters for the next redesign decision: if we reuse anything from legacy lines, Laplace-style `reliability over temporal history` is the more credible seed than the old heavy frequency stack.
