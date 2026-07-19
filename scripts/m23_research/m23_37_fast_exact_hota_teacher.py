#!/usr/bin/env python3
from __future__ import annotations

"""Fast exact HOTA evaluator for ID-only microtracklet transaction changes.

This module uses the official TrackEval MOT20 preprocessing and HOTA metric, but
keeps all preprocessed boxes, IoUs and GT detections in memory. Candidate
trackers only replace tracker IDs; therefore expensive text parsing, box IoU
calculation and subprocess startup are avoided after preparation.

GT use is teacher-only. Under strict sequence LOSO this evaluator may only be
run on outer-training or already-exposed diagnostic sequences. It must never be
used to select a policy on an unseen outer-held sequence.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

REPO = Path(__file__).resolve().parents[2]
TRACKEVAL_ROOT = REPO / "TrackEval"
if str(TRACKEVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKEVAL_ROOT))

from trackeval.datasets import MotChallenge2DBox  # noqa: E402
from trackeval.metrics import HOTA  # noqa: E402


def load_module(name: str, relative_path: str):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mean_metrics(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {
        "HOTA": 100.0 * float(np.mean(result["HOTA"])),
        "DetA": 100.0 * float(np.mean(result["DetA"])),
        "AssA": 100.0 * float(np.mean(result["AssA"])),
        "DetRe": 100.0 * float(np.mean(result["DetRe"])),
        "DetPr": 100.0 * float(np.mean(result["DetPr"])),
        "AssRe": 100.0 * float(np.mean(result["AssRe"])),
        "AssPr": 100.0 * float(np.mean(result["AssPr"])),
        "LocA": 100.0 * float(np.mean(result["LocA"])),
    }


class PreparedExactHOTA:
    """Official TrackEval preprocessing cached for repeated tracker-ID changes."""

    def __init__(
        self,
        seq: str,
        parent_tracker: Path,
        cache_root: Path,
        gt_root: Path = Path("datasets/MOT20/train"),
    ) -> None:
        self.seq = seq
        self.parent_tracker = parent_tracker
        self.cache_root = cache_root
        self.gt_root = gt_root
        self.metric = HOTA()

        tracker_name = "prepared_parent"
        tracker_data = cache_root / "trackers" / tracker_name / "data"
        tracker_data.mkdir(parents=True, exist_ok=True)
        cached_tracker = tracker_data / f"{seq}.txt"
        shutil.copy2(parent_tracker, cached_tracker)

        config = {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(cache_root / "trackers"),
            "OUTPUT_FOLDER": str(cache_root / "unused_output"),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT20",
            "SPLIT_TO_EVAL": "train",
            "INPUT_AS_ZIP": False,
            "PRINT_CONFIG": False,
            "DO_PREPROC": True,
            "TRACKER_SUB_FOLDER": "data",
            "OUTPUT_SUB_FOLDER": "",
            "SEQMAP_FOLDER": None,
            "SEQMAP_FILE": None,
            "SEQ_INFO": {seq: None},
            "SKIP_SPLIT_FOL": True,
        }
        dataset = MotChallenge2DBox(config)
        raw = dataset.get_raw_seq_data(tracker_name, seq)
        self.data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        self.raw_tracker_dets = raw["tracker_dets"]
        self.processed_keep_indices = self._derive_processed_keep_indices(
            raw["tracker_dets"], self.data["tracker_dets"]
        )
        self.parent_rows, self.frame_row_indices = self._read_parent_rows(parent_tracker)
        self.parent_row_ids = np.asarray(
            [int(float(fields[1])) for fields in self.parent_rows], dtype=np.int64
        )
        self._validate_frame_alignment()
        self.processed_parent_row_indices = [
            np.asarray(self.frame_row_indices.get(frame, []), dtype=np.int64)[keep]
            for frame, keep in enumerate(self.processed_keep_indices, start=1)
        ]
        self.frame_flat_slices: List[slice] = []
        flat_rows: List[np.ndarray] = []
        cursor = 0
        for row_indices in self.processed_parent_row_indices:
            self.frame_flat_slices.append(slice(cursor, cursor + len(row_indices)))
            flat_rows.append(row_indices)
            cursor += len(row_indices)
        self.processed_parent_rows_flat = (
            np.concatenate(flat_rows) if flat_rows else np.empty(0, dtype=np.int64)
        )
        self._prepare_incremental_cache()

    @staticmethod
    def _read_parent_rows(path: Path):
        rows: List[List[str]] = []
        frame_rows: Dict[int, List[int]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(",")
                if len(fields) < 6:
                    continue
                row_index = len(rows)
                frame = int(float(fields[0]))
                rows.append(fields)
                frame_rows.setdefault(frame, []).append(row_index)
        return rows, frame_rows

    @staticmethod
    def _derive_processed_keep_indices(
        raw_dets: Sequence[np.ndarray], processed_dets: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        output: List[np.ndarray] = []
        for frame_index, (raw_frame, processed_frame) in enumerate(
            zip(raw_dets, processed_dets), start=1
        ):
            keep: List[int] = []
            cursor = 0
            for target in processed_frame:
                while cursor < len(raw_frame) and not np.allclose(
                    raw_frame[cursor], target, rtol=0.0, atol=1e-4
                ):
                    cursor += 1
                if cursor >= len(raw_frame):
                    raise RuntimeError(
                        f"frame {frame_index}: processed detection is not an ordered subset"
                    )
                keep.append(cursor)
                cursor += 1
            output.append(np.asarray(keep, dtype=np.int64))
        return output

    def _validate_frame_alignment(self) -> None:
        for frame in range(1, int(self.data["num_timesteps"]) + 1):
            row_indices = self.frame_row_indices.get(frame, [])
            if len(row_indices) != len(self.raw_tracker_dets[frame - 1]):
                raise RuntimeError(
                    f"{self.seq} frame {frame}: parent/raw row mismatch "
                    f"{len(row_indices)} != {len(self.raw_tracker_dets[frame - 1])}"
                )
            for local_index, row_index in enumerate(row_indices):
                fields = self.parent_rows[row_index]
                xywh = np.asarray(
                    [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])],
                    dtype=float,
                )
                if not np.allclose(
                    xywh,
                    self.raw_tracker_dets[frame - 1][local_index],
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise RuntimeError(
                        f"{self.seq} frame {frame}: tracker row ordering mismatch"
                    )

    @property
    def num_parent_rows(self) -> int:
        return len(self.parent_rows)

    def ids_from_tracker_file(self, tracker_file: Path) -> np.ndarray:
        """Return tracker IDs aligned to the parent rows by frame and exact box."""
        if tracker_file.resolve() == self.parent_tracker.resolve():
            return self.parent_row_ids.copy()
        ids = np.empty(self.num_parent_rows, dtype=np.int64)
        candidate_by_frame: Dict[int, List[tuple[np.ndarray, int]]] = {}
        with tracker_file.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split(",")
                if len(fields) < 6:
                    continue
                frame = int(float(fields[0]))
                box = np.asarray(
                    [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])],
                    dtype=float,
                )
                candidate_by_frame.setdefault(frame, []).append(
                    (box, int(float(fields[1])))
                )
        for frame, row_indices in self.frame_row_indices.items():
            candidates = candidate_by_frame.get(frame, [])
            if len(candidates) != len(row_indices):
                raise RuntimeError(
                    f"{self.seq} frame {frame}: candidate row count mismatch"
                )
            unused = set(range(len(candidates)))
            for row_index in row_indices:
                fields = self.parent_rows[row_index]
                parent_box = np.asarray(
                    [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])],
                    dtype=float,
                )
                matches = [
                    index
                    for index in unused
                    if np.allclose(
                        candidates[index][0], parent_box, rtol=0.0, atol=1e-4
                    )
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"{self.seq} frame {frame}: ambiguous box alignment {len(matches)}"
                    )
                match = matches[0]
                unused.remove(match)
                ids[row_index] = candidates[match][1]
        return ids

    def evaluate_row_ids(self, row_ids: np.ndarray) -> Dict[str, float]:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.shape != (self.num_parent_rows,):
            raise ValueError(
                f"expected {self.num_parent_rows} row IDs, got {row_ids.shape}"
            )
        candidate_ids: List[np.ndarray] = []
        all_ids: List[np.ndarray] = []
        for frame in range(1, int(self.data["num_timesteps"]) + 1):
            row_indices = np.asarray(self.frame_row_indices.get(frame, []), dtype=np.int64)
            raw_ids = row_ids[row_indices] if len(row_indices) else np.empty(0, np.int64)
            keep = self.processed_keep_indices[frame - 1]
            kept_ids = raw_ids[keep]
            if len(kept_ids) != len(np.unique(kept_ids)):
                raise RuntimeError(f"{self.seq} frame {frame}: duplicate candidate IDs")
            candidate_ids.append(kept_ids)
            if len(kept_ids):
                all_ids.append(kept_ids)
        unique_ids = (
            np.unique(np.concatenate(all_ids)) if all_ids else np.empty(0, np.int64)
        )
        remapped = [
            np.searchsorted(unique_ids, ids).astype(np.int32) if len(ids) else ids.astype(np.int32)
            for ids in candidate_ids
        ]
        candidate_data = dict(self.data)
        candidate_data["tracker_ids"] = remapped
        candidate_data["num_tracker_ids"] = int(len(unique_ids))
        result = self.metric.eval_sequence(candidate_data)
        return mean_metrics(result)

    @staticmethod
    def _empty_raw_result() -> Dict[str, np.ndarray | float]:
        array_count = 19
        result: Dict[str, np.ndarray | float] = {}
        for field in HOTA().float_array_fields + HOTA().integer_array_fields:
            result[field] = np.zeros(array_count, dtype=np.float32)
        for field in HOTA().float_fields:
            result[field] = 0.0
        return result

    @staticmethod
    def _similarity_iou_contribution(similarity: np.ndarray) -> np.ndarray:
        if similarity.size == 0:
            return np.zeros_like(similarity)
        denominator = (
            similarity.sum(axis=0)[np.newaxis, :]
            + similarity.sum(axis=1)[:, np.newaxis]
            - similarity
        )
        output = np.zeros_like(similarity)
        mask = denominator > np.finfo(float).eps
        output[mask] = similarity[mask] / denominator[mask]
        return output

    def _prepare_incremental_cache(self) -> None:
        """Build exact baseline assignments and row-additive global-alignment data."""
        start = time.perf_counter()
        self.gt_id_count = np.zeros((int(self.data["num_gt_ids"]), 1), dtype=float)
        sparse_rows: List[np.ndarray] = []
        sparse_cols: List[np.ndarray] = []
        sparse_values: List[np.ndarray] = []
        flat_tracker_offset = 0
        for gt_ids, similarity in zip(
            self.data["gt_ids"], self.data["similarity_scores"]
        ):
            if len(gt_ids):
                self.gt_id_count[gt_ids] += 1
            contribution = self._similarity_iou_contribution(similarity)
            if contribution.size:
                gt_local, tracker_local = np.nonzero(contribution)
                if len(gt_local):
                    sparse_rows.append(flat_tracker_offset + tracker_local.astype(np.int64))
                    sparse_cols.append(gt_ids[gt_local].astype(np.int64))
                    sparse_values.append(contribution[gt_local, tracker_local].astype(float))
            flat_tracker_offset += similarity.shape[1]
        if flat_tracker_offset != len(self.processed_parent_rows_flat):
            raise RuntimeError(
                f"processed tracker row count mismatch: {flat_tracker_offset} != "
                f"{len(self.processed_parent_rows_flat)}"
            )
        if sparse_rows:
            row_index = np.concatenate(sparse_rows)
            col_index = np.concatenate(sparse_cols)
            values = np.concatenate(sparse_values)
        else:
            row_index = np.empty(0, dtype=np.int64)
            col_index = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=float)
        self.row_gt_contribution = csr_matrix(
            (values, (row_index, col_index)),
            shape=(len(self.processed_parent_rows_flat), int(self.data["num_gt_ids"])),
        )

        self.baseline_ids_by_frame = [
            np.asarray(ids, dtype=np.int32).copy() for ids in self.data["tracker_ids"]
        ]
        self.baseline_ids_flat = (
            np.concatenate(self.baseline_ids_by_frame)
            if self.baseline_ids_by_frame
            else np.empty(0, dtype=np.int32)
        )
        self.baseline_tracker_count = np.bincount(
            self.baseline_ids_flat,
            minlength=int(self.data["num_tracker_ids"]),
        ).astype(float)
        self.baseline_potential = np.zeros(
            (int(self.data["num_gt_ids"]), int(self.data["num_tracker_ids"])),
            dtype=float,
        )
        for tracker_id in range(int(self.data["num_tracker_ids"])):
            member_rows = np.flatnonzero(self.baseline_ids_flat == tracker_id)
            if len(member_rows):
                self.baseline_potential[:, tracker_id] = np.asarray(
                    self.row_gt_contribution[member_rows].sum(axis=0)
                ).ravel()
        self.baseline_global_alignment = self.baseline_potential / np.maximum(
            np.finfo(float).eps,
            self.gt_id_count
            + self.baseline_tracker_count[np.newaxis, :]
            - self.baseline_potential,
        )
        (
            self.baseline_raw_result,
            self.baseline_matches_counts,
            self.baseline_assignments,
        ) = self._evaluate_all_frames_and_cache_assignments(
            self.baseline_ids_by_frame,
            self.baseline_global_alignment,
        )
        self.incremental_prepare_seconds = time.perf_counter() - start

    def _evaluate_all_frames_and_cache_assignments(
        self,
        tracker_ids_by_frame: Sequence[np.ndarray],
        global_alignment: np.ndarray,
    ):
        result = self._empty_raw_result()
        matches_counts = [
            np.zeros(
                (int(self.data["num_gt_ids"]), global_alignment.shape[1]),
                dtype=float,
            )
            for _ in self.metric.array_labels
        ]
        assignments: List[tuple[np.ndarray, np.ndarray]] = []
        for gt_ids, tracker_ids, similarity in zip(
            self.data["gt_ids"], tracker_ids_by_frame, self.data["similarity_scores"]
        ):
            if len(gt_ids) == 0 or len(tracker_ids) == 0:
                match_rows = np.empty(0, dtype=np.int64)
                match_cols = np.empty(0, dtype=np.int64)
            else:
                score = (
                    global_alignment[
                        gt_ids[:, np.newaxis], tracker_ids[np.newaxis, :]
                    ]
                    * similarity
                )
                match_rows, match_cols = linear_sum_assignment(-score)
            assignments.append(
                (
                    np.asarray(match_rows, dtype=np.int64),
                    np.asarray(match_cols, dtype=np.int64),
                )
            )
            self._add_frame_statistics(
                result,
                matches_counts,
                gt_ids,
                tracker_ids,
                similarity,
                match_rows,
                match_cols,
                sign=1.0,
            )
        return result, matches_counts, assignments

    def _add_frame_statistics(
        self,
        result,
        matches_counts,
        gt_ids: np.ndarray,
        tracker_ids: np.ndarray,
        similarity: np.ndarray,
        match_rows: np.ndarray,
        match_cols: np.ndarray,
        sign: float,
        copied_tracker_columns: np.ndarray | None = None,
    ) -> None:
        for alpha_index, alpha in enumerate(self.metric.array_labels):
            if len(match_rows):
                matched_mask = (
                    similarity[match_rows, match_cols]
                    >= alpha - np.finfo(float).eps
                )
                alpha_rows = match_rows[matched_mask]
                alpha_cols = match_cols[matched_mask]
            else:
                alpha_rows = np.empty(0, dtype=np.int64)
                alpha_cols = np.empty(0, dtype=np.int64)
            match_count = len(alpha_rows)
            result["HOTA_TP"][alpha_index] += sign * match_count
            result["HOTA_FN"][alpha_index] += sign * (len(gt_ids) - match_count)
            result["HOTA_FP"][alpha_index] += sign * (
                len(tracker_ids) - match_count
            )
            if match_count:
                result["LocA"][alpha_index] += sign * float(
                    similarity[alpha_rows, alpha_cols].sum()
                )
                matched_tracker_ids = tracker_ids[alpha_cols]
                matched_gt_ids = gt_ids[alpha_rows]
                if copied_tracker_columns is not None:
                    keep = copied_tracker_columns[matched_tracker_ids] >= 0
                    matched_gt_ids = matched_gt_ids[keep]
                    matched_tracker_ids = copied_tracker_columns[
                        matched_tracker_ids[keep]
                    ]
                if len(matched_gt_ids):
                    np.add.at(
                        matches_counts[alpha_index],
                        (matched_gt_ids, matched_tracker_ids),
                        sign,
                    )

    def _finalize_raw_result(
        self,
        result,
        matches_counts: Sequence[np.ndarray],
        tracker_count: np.ndarray,
    ) -> Dict[str, float]:
        for alpha_index in range(len(self.metric.array_labels)):
            match_count = matches_counts[alpha_index]
            association = match_count / np.maximum(
                1.0,
                self.gt_id_count + tracker_count[np.newaxis, :] - match_count,
            )
            result["AssA"][alpha_index] = np.sum(
                match_count * association
            ) / np.maximum(1.0, result["HOTA_TP"][alpha_index])
            association_recall = match_count / np.maximum(1.0, self.gt_id_count)
            result["AssRe"][alpha_index] = np.sum(
                match_count * association_recall
            ) / np.maximum(1.0, result["HOTA_TP"][alpha_index])
            association_precision = match_count / np.maximum(
                1.0, tracker_count[np.newaxis, :]
            )
            result["AssPr"][alpha_index] = np.sum(
                match_count * association_precision
            ) / np.maximum(1.0, result["HOTA_TP"][alpha_index])
        result["LocA"] = np.maximum(1e-10, result["LocA"]) / np.maximum(
            1e-10, result["HOTA_TP"]
        )
        result = self.metric._compute_final_fields(result)
        return mean_metrics(result)

    def _candidate_processed_ids(self, row_ids: np.ndarray):
        candidate_ids_by_frame: List[np.ndarray] = []
        all_ids: List[np.ndarray] = []
        for row_indices in self.processed_parent_row_indices:
            ids = row_ids[row_indices] if len(row_indices) else np.empty(0, np.int64)
            if len(ids) != len(np.unique(ids)):
                raise RuntimeError(f"{self.seq}: duplicate candidate IDs after preprocessing")
            candidate_ids_by_frame.append(np.asarray(ids, dtype=np.int64))
            if len(ids):
                all_ids.append(np.asarray(ids, dtype=np.int64))
        unique_ids = (
            np.unique(np.concatenate(all_ids)) if all_ids else np.empty(0, np.int64)
        )
        remapped = [
            np.searchsorted(unique_ids, ids).astype(np.int32)
            if len(ids)
            else ids.astype(np.int32)
            for ids in candidate_ids_by_frame
        ]
        flat = np.concatenate(remapped) if remapped else np.empty(0, np.int32)
        return remapped, flat, unique_ids

    def evaluate_row_ids_incremental(self, row_ids: np.ndarray) -> Dict[str, float]:
        """Evaluate exactly, recomputing Hungarian only on changed-partition frames."""
        start = time.perf_counter()
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.shape != (self.num_parent_rows,):
            raise ValueError(
                f"expected {self.num_parent_rows} row IDs, got {row_ids.shape}"
            )
        candidate_ids_by_frame, candidate_ids_flat, unique_ids = (
            self._candidate_processed_ids(row_ids)
        )
        candidate_count = np.bincount(
            candidate_ids_flat, minlength=len(unique_ids)
        ).astype(float)

        # Map candidate partitions which are exactly identical to a baseline partition.
        unchanged_candidate_to_baseline = np.full(len(unique_ids), -1, dtype=np.int32)
        baseline_sizes = np.bincount(
            self.baseline_ids_flat, minlength=int(self.data["num_tracker_ids"])
        )
        for candidate_id in range(len(unique_ids)):
            member_rows = np.flatnonzero(candidate_ids_flat == candidate_id)
            baseline_members = np.unique(self.baseline_ids_flat[member_rows])
            if (
                len(baseline_members) == 1
                and len(member_rows) == baseline_sizes[int(baseline_members[0])]
            ):
                unchanged_candidate_to_baseline[candidate_id] = int(
                    baseline_members[0]
                )
        unchanged_baseline_to_candidate = np.full(
            int(self.data["num_tracker_ids"]), -1, dtype=np.int32
        )
        for candidate_id, baseline_id in enumerate(
            unchanged_candidate_to_baseline
        ):
            if baseline_id >= 0:
                unchanged_baseline_to_candidate[baseline_id] = candidate_id

        candidate_potential = np.zeros(
            (int(self.data["num_gt_ids"]), len(unique_ids)), dtype=float
        )
        for candidate_id in range(len(unique_ids)):
            baseline_id = int(unchanged_candidate_to_baseline[candidate_id])
            if baseline_id >= 0:
                candidate_potential[:, candidate_id] = self.baseline_potential[
                    :, baseline_id
                ]
            else:
                member_rows = np.flatnonzero(candidate_ids_flat == candidate_id)
                candidate_potential[:, candidate_id] = np.asarray(
                    self.row_gt_contribution[member_rows].sum(axis=0)
                ).ravel()
        global_alignment = candidate_potential / np.maximum(
            np.finfo(float).eps,
            self.gt_id_count
            + candidate_count[np.newaxis, :]
            - candidate_potential,
        )

        changed_row_mask = (
            unchanged_candidate_to_baseline[candidate_ids_flat]
            != self.baseline_ids_flat
        )
        affected_frames = np.asarray(
            [
                bool(changed_row_mask[frame_slice].any())
                for frame_slice in self.frame_flat_slices
            ],
            dtype=bool,
        )

        result = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.baseline_raw_result.items()
        }
        candidate_matches_counts = [
            np.zeros(
                (int(self.data["num_gt_ids"]), len(unique_ids)), dtype=float
            )
            for _ in self.metric.array_labels
        ]
        for candidate_id, baseline_id in enumerate(
            unchanged_candidate_to_baseline
        ):
            if baseline_id >= 0:
                for alpha_index in range(len(self.metric.array_labels)):
                    candidate_matches_counts[alpha_index][:, candidate_id] = (
                        self.baseline_matches_counts[alpha_index][:, baseline_id]
                    )

        recomputed_frames = 0
        affected_processed_detections = 0
        for frame_index, is_affected in enumerate(affected_frames):
            if not is_affected:
                continue
            recomputed_frames += 1
            affected_processed_detections += len(candidate_ids_by_frame[frame_index])
            gt_ids = self.data["gt_ids"][frame_index]
            similarity = self.data["similarity_scores"][frame_index]
            baseline_tracker_ids = self.baseline_ids_by_frame[frame_index]
            baseline_rows, baseline_cols = self.baseline_assignments[frame_index]
            self._add_frame_statistics(
                result,
                candidate_matches_counts,
                gt_ids,
                baseline_tracker_ids,
                similarity,
                baseline_rows,
                baseline_cols,
                sign=-1.0,
                copied_tracker_columns=unchanged_baseline_to_candidate,
            )

            candidate_tracker_ids = candidate_ids_by_frame[frame_index]
            if len(gt_ids) == 0 or len(candidate_tracker_ids) == 0:
                candidate_rows = np.empty(0, dtype=np.int64)
                candidate_cols = np.empty(0, dtype=np.int64)
            else:
                score = (
                    global_alignment[
                        gt_ids[:, np.newaxis],
                        candidate_tracker_ids[np.newaxis, :],
                    ]
                    * similarity
                )
                candidate_rows, candidate_cols = linear_sum_assignment(-score)
            self._add_frame_statistics(
                result,
                candidate_matches_counts,
                gt_ids,
                candidate_tracker_ids,
                similarity,
                candidate_rows,
                candidate_cols,
                sign=1.0,
            )

        metrics = self._finalize_raw_result(
            result, candidate_matches_counts, candidate_count
        )
        metrics.update(
            {
                "incremental_seconds": time.perf_counter() - start,
                "affected_frames": int(recomputed_frames),
                "total_frames": int(self.data["num_timesteps"]),
                "affected_processed_detections": int(
                    affected_processed_detections
                ),
                "changed_processed_detections": int(changed_row_mask.sum()),
                "unchanged_partitions": int(
                    (unchanged_candidate_to_baseline >= 0).sum()
                ),
                "candidate_partitions": int(len(unique_ids)),
            }
        )
        return metrics

    def evaluate_tracker_file(self, tracker_file: Path) -> Dict[str, float]:
        return self.evaluate_row_ids(self.ids_from_tracker_file(tracker_file))

    def row_ids_from_selected_graph(
        self,
        meta,
        selected_edges,
        evaluator_module,
    ) -> np.ndarray:
        rows = evaluator_module.read_parent(self.parent_tracker)
        if len(rows) != self.num_parent_rows:
            raise RuntimeError("parent row count changed")
        line_to_chunk = evaluator_module.line_chunks(rows, meta)
        assignment = evaluator_module.chains(selected_edges, len(meta))
        roots = np.asarray(
            [
                int(assignment[int(line_to_chunk[row_index])])
                for row_index in range(len(rows))
            ],
            dtype=np.int64,
        )
        return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True)
    parser.add_argument("--parent-tracker", required=True)
    parser.add_argument("--candidate-tracker", action="append", default=[])
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    prepared = PreparedExactHOTA(
        args.seq,
        Path(args.parent_tracker),
        Path(args.cache_root),
    )
    prepare_seconds = time.perf_counter() - start
    rows = []
    for candidate_raw in args.candidate_tracker:
        candidate = Path(candidate_raw)
        eval_start = time.perf_counter()
        metrics = prepared.evaluate_tracker_file(candidate)
        rows.append(
            {
                "tracker": str(candidate),
                **metrics,
                "evaluation_seconds": time.perf_counter() - eval_start,
            }
        )
    report = {
        "status": "completed",
        "teacher_only": True,
        "seq": args.seq,
        "parent_tracker": args.parent_tracker,
        "prepare_seconds": prepare_seconds,
        "num_parent_rows": prepared.num_parent_rows,
        "candidates": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
