# FM-Track Research Repository

This repository is not just a generic training codebase. It is the working research repository for a sequence of multi-object tracking experiments, diagnosis reports, and redesign decisions around local intervention operators on top of standard MOT trackers.

If you are a human reviewer or a GitHub-connected GPT, do **not** start from the old generic training entrypoints alone. Start from the project navigation docs below.

## Start Here

Primary reading entrypoints:

- `docs/github_reader_guide.md`
- `docs/experiment_index.md`
- `outputs/experiment_registry.csv`

These three files are the intended shortest path for reconstructing:

- which baseline families were audited
- which learned directions were stopped
- which carrier is canonical
- what the current active line is
- where the relevant code and structured experiment records live

## Current Research Status

As of `2026-03-28`, the repository-wide state is:

- canonical paper carrier: `official_bytetrack`
- transfer carrier: `botsort_base`
- specialist reference carrier: `strongsort_base`
- strongest internal positive line: `base_reid_da + set_predictor_v2`
- learned `pre-Hungarian` official ByteTrack line: stopped
- current active official ByteTrack line: `post-host one-edit`
- current best learned family on that changed contract: `hierarchical post-host one-edit scorer`

Important nuance:

- the old `set_predictor_v2` direction was not globally disproven
- it did produce real positives on an internal host
- but under the frozen official ByteTrack `pre-Hungarian` contract it failed and was stop-gated
- after changing the contract to a `post-host one-edit` intervention, executable oracle headroom became positive
- the latest offline learned smoke says a hierarchical family is the first plausible learned continuation on that new contract

## Reader Guide

If you need full context without prior conversation history:

1. Read `docs/github_reader_guide.md`
2. Read `docs/experiment_index.md`
3. Follow the linked `report.md`, `summary.csv`, and `result.csv` files in `outputs/`

## Important Code Paths

There are two different kinds of code in this repository:

1. historical / archived research lines
2. current mainline code paths

If you want the **current mainline** for the official ByteTrack direction, start here:

- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/run_official_bytetrack_shared_detection_pair_core.py`
- `third_party/ByteTrack/yolox/tracker/byte_tracker_local_conflict.py`
- `third_party/ByteTrack/tools/track.py`
- `third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py`
- `scripts/build_posthost_one_edit_dataset.py`
- `scripts/train_posthost_one_edit_hierarchical.py`
- `models/posthost_one_edit_hierarchical.py`

If you want the **stopped historical official ByteTrack pre-Hungarian line**, start here:

- `scripts/build_local_conflict_set_predictor_dataset.py`
- `scripts/train_local_conflict_set_predictor.py`
- `scripts/run_official_bytetrack_local_conflict_stage1_trainhalf.py`
- `models/local_conflict_set_predictor.py`

## Experiment Records

This repository intentionally stores lightweight structured experiment evidence in git.

Typical tracked artifacts include:

- `summary.csv`
- `result.csv`
- `metrics.jsonl`
- `report.md`
- `outputs/experiment_registry.csv`

Heavy artifacts are intentionally excluded from git:

- checkpoints
- dataset dumps
- raw logs
- large runtime shards
- packaged archives

This is deliberate. The goal is to keep the repository readable by both humans and GitHub-connected GPT systems while still preserving experiment traceability.

## Environment and Reproduction Notes

This repository includes multiple tracker families and historical branches, so there is no single one-line reproduction command that explains the whole project.

For current official ByteTrack work, the most useful reproduction-related files are:

- `third_party/ByteTrack/exps/example/mot/yolox_x_mix_det_valhalf.py`
- `scripts/run_official_bytetrack_local_conflict_halfval_pair.py`
- `scripts/build_posthost_one_edit_dataset.py`
- `scripts/train_posthost_one_edit_hierarchical.py`

Weights and datasets are expected to exist locally and are **not** fully vendored in git.

## Legacy / Broad Training Entry Points

Older broad entrypoints remain in the repo for historical completeness:

- `train.py`
- `train_bytetrack.py`
- `submit_bytetrack.py`
- `submit_public.py`

They should not be treated as the main explanation of the current research state.
