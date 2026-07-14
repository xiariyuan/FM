# Identity Family Graph Coverage Audit — 2026-07-14

## Motivation

Current Identity Transaction System reaches 78.81384 HOTA, while GT family oracle reaches about 80.20. This audit asks whether the gap comes from ranking or representation.

## Result

Across 660 fragmented identity families:

- total identity debt rows: 69,702
- current pair candidate graph can recover only 16,235 rows (23.29%)
- utility-positive edges cover 14,203 rows (20.38%)
- deployed selected links recover 1,294 rows (1.86%)

Bottleneck counts:

- track segmentation required: 620/660 families
- selection budget: 26
- candidate generation: 13

## Conclusion

The main bottleneck is not edge ranking. Whole-track merge cannot represent identities that switch inside a track. The required abstraction is segment-level identity reconstruction.

Example:

```
track 39: person A -> person B -> person A
```

A single track ID cannot be globally merged safely. The next method should split tracks at identity change points and optimize segment graph edits.

## Next experiment

Build segment graph:

node: temporal track segment

edge:
- local ReID continuity
- motion continuity
- debt reduction
- AssA utility

Optimize segment family reconstruction instead of whole-track merge.
