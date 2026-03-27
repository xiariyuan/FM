# Cross-Host Baseline Defect Audit

This report diagnoses the main defect profile of each clean carrier baseline rather than only ranking them by aggregate score.

## official_bytetrack

- `paper_role`: `canonical`
- `main_defect_type`: crowded local-association failure and large-component coverage gap
- `what_it_solves`: ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-11-FRCNN"]
- `worst_sequences`: ["MOT17-02-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]
- `mechanism_hypothesis`: First-stage local association is stable on easy slices, but degrades on MOT17-05/10/13 where recoverable conflicts and skipped-large components concentrate.
- `defect_evidence`: {"largest_HOTA_gap_vs_best": ["MOT17-13-FRCNN", "MOT17-09-FRCNN", "MOT17-05-FRCNN"], "official_top_assoc_error_sequences": ["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"], "common_hard_overlap": ["MOT17-05-FRCNN", "MOT17-10-FRCNN"]}

## botsort_base

- `paper_role`: `test_oriented_transfer`
- `main_defect_type`: identity-switch instability on conservative official-favorable slices
- `what_it_solves`: ["MOT17-05-FRCNN", "MOT17-10-FRCNN", "MOT17-13-FRCNN"]
- `worst_sequences`: ["MOT17-02-FRCNN", "MOT17-10-FRCNN", "MOT17-09-FRCNN"]
- `mechanism_hypothesis`: BoT-SORT is stronger on crowded recovery slices, but likely pays for heavier appearance/motion fusion with extra switches on MOT17-02 and mild identity regressions on MOT17-11/04.
- `defect_evidence`: {"largest_negative_HOTA_vs_official": ["MOT17-02-FRCNN", "MOT17-11-FRCNN", "MOT17-04-FRCNN"], "largest_negative_IDF1_vs_official": ["MOT17-02-FRCNN", "MOT17-11-FRCNN", "MOT17-04-FRCNN"], "highest_IDSW_sequences": ["MOT17-02-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]}

## strongsort_base

- `paper_role`: `specialist_only`
- `main_defect_type`: global detection/coverage deficit despite some identity-stability strengths
- `what_it_solves`: ["MOT17-09-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]
- `worst_sequences`: ["MOT17-02-FRCNN", "MOT17-05-FRCNN", "MOT17-10-FRCNN"]
- `mechanism_hypothesis`: StrongSORT behaves like a specialist on a few ID-heavy slices, but its main bottleneck is not local association ranking; it is broader coverage/recall loss that drags HOTA/MOTA down on most sequences.
- `defect_evidence`: {"largest_HOTA_gaps_vs_best": ["MOT17-13-FRCNN", "MOT17-04-FRCNN", "MOT17-02-FRCNN"], "lowest_DetA_sequences": ["MOT17-02-FRCNN", "MOT17-10-FRCNN", "MOT17-05-FRCNN"], "overall_metric_gap_vs_official": {"DetA": -12.855000000000004, "MOTA": -13.037999999999997}}

