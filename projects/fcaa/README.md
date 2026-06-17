## FCAA

`FCAA` stands for `Frequency-Conditioned Appearance Association`.

This project is an isolated research line built on top of a fixed BoT-SORT-style
carrier. It does not revive the old heavy trajectory-frequency family.

Current scope:

- build an offline pair-bank from GT-aligned pseudo tracks
- train a minimal pair-level scorer
- compare `same-gate control` vs `same-gate frequency`
- keep the online hook selective and optional

Current mainline status:

- the original `row_margin` grouping was tested and found too sparse for useful offline learning on hard MOT17 slices
- the active offline grouping is now `shared_det_top1`
- the runtime hook supports both `row_margin` and `shared_det_top1`
- the current requirement before online claims is still the same: prove stable offline signal first

The first milestone is intentionally narrow:

1. export a MOT17 pair-bank
2. train a control scorer using `s_reid`
3. train an FCAA scorer using `[s_reid, s_low, s_mid, s_high]`
4. prove signal offline before spending budget on online runs

This package lives under `projects/fcaa/` to keep the new line separated from
the previous frequency-family code paths.
