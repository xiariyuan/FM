# Cross-Host Carrier Uniqueness Audit

## High-Level Decision

- `paper canonical carrier`: `official_bytetrack`
- `test-oriented carrier`: `botsort_base`
- `specialist carrier`: `strongsort_base`

## Overall Best Aggregate Carrier

- `HOTA`: `botsort_base`
- `AssA`: `official_bytetrack`
- `IDF1`: `official_bytetrack`
- `MOTA`: `botsort_base`
- `IDSW`: `official_bytetrack`

## Sequence-Level Winners

- `MOT17-02-FRCNN`: HOTA `official_bytetrack` (+gap 0.029), IDF1 `official_bytetrack` (+gap 0.062), IDSW `official_bytetrack` (margin 17)
- `MOT17-04-FRCNN`: HOTA `botsort_base` (+gap 0.002), IDF1 `official_bytetrack` (+gap 0.001), IDSW `official_bytetrack` (margin 1)
- `MOT17-05-FRCNN`: HOTA `botsort_base` (+gap 0.033), IDF1 `botsort_base` (+gap 0.017), IDSW `strongsort_base` (margin 13)
- `MOT17-09-FRCNN`: HOTA `strongsort_base` (+gap 0.019), IDF1 `strongsort_base` (+gap 0.083), IDSW `strongsort_base` (margin 9)
- `MOT17-10-FRCNN`: HOTA `botsort_base` (+gap 0.013), IDF1 `botsort_base` (+gap 0.020), IDSW `strongsort_base` (margin 6)
- `MOT17-11-FRCNN`: HOTA `botsort_base` (+gap 0.002), IDF1 `official_bytetrack` (+gap 0.016), IDSW `botsort_base` (margin 2)
- `MOT17-13-FRCNN`: HOTA `botsort_base` (+gap 0.075), IDF1 `botsort_base` (+gap 0.081), IDSW `botsort_base` (margin 0)

## Carrier Roles

- `canonical_anchor` -> `official_bytetrack` on ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-11-FRCNN"]: Only clean carrier that uniquely holds MOT17-02 and part of MOT17-04 on identity quality / low switches, so it remains the canonical paper baseline.
- `test_oriented_transfer` -> `botsort_base` on ["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"]: Best clean transfer carrier on the hardest official failure slices, especially MOT17-05/10/13 where it beats official ByteTrack on both HOTA and IDF1.
- `specialist_counterexample` -> `strongsort_base` on ["MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN"]: Sequence-specific specialist, especially MOT17-09 and low-switch behavior on MOT17-05/10, but not strong enough overall to be the main carrier.
- `official_assoc_error_reference` -> `official_bytetrack` on ["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"]: These are the official ByteTrack association-error-heavy sequences any learned module must improve without breaking easy slices.

## Metric Win Counts

- `HOTA`: botsort_base=5, official_bytetrack=1, strongsort_base=1
- `AssA`: botsort_base=3, official_bytetrack=2, strongsort_base=2
- `IDF1`: botsort_base=3, official_bytetrack=3, strongsort_base=1
- `MOTA`: botsort_base=7, official_bytetrack=0, strongsort_base=0
- `IDSW`: strongsort_base=3, botsort_base=2, official_bytetrack=2
