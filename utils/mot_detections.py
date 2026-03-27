from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def load_mot_detections(det_path: str) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """
    Load MOT-format detections / tracking results.

    Expected CSV format per line:
      frame, id, x, y, w, h, score, ...

    Returns:
      dict[frame_id] -> list[(x, y, w, h, score)]

    Notes:
    - `frame_id` is expected to be 1-indexed (standard MOT format).
    - Extra trailing columns (e.g., world coordinates) are ignored.
    """
    detections: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    with open(det_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                frame_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
            except ValueError:
                continue
            detections.setdefault(frame_id, []).append((x, y, w, h, conf))
    return detections


def resolve_external_det_path(config: Dict[str, Any], dataset_name: str, split: str, seq_name: str) -> str:
    """
    Resolve external detection file path for one sequence.

    Priority:
      1) EXTERNAL_DET_FILE_BY_SEQ[seq_name]
      2) EXTERNAL_DET_ROOT + EXTERNAL_DET_PATTERN
      3) fallback to DATA_ROOT/{dataset}/{split}/{seq}/det/det.txt
    """
    det_by_seq = config.get("EXTERNAL_DET_FILE_BY_SEQ", None)
    if isinstance(det_by_seq, dict):
        if seq_name in det_by_seq and det_by_seq[seq_name]:
            return str(det_by_seq[seq_name])
        seq_upper = str(seq_name).upper()
        for key, value in det_by_seq.items():
            if str(key).upper() == seq_upper and value:
                return str(value)

    det_root = config.get("EXTERNAL_DET_ROOT", None)
    det_pattern = config.get(
        "EXTERNAL_DET_PATTERN",
        "{root}/{dataset}/{split}/{seq}/det/det.txt",
    )
    if det_root is not None and str(det_root).strip():
        try:
            return str(det_pattern).format(
                root=str(det_root),
                data_root=str(config["DATA_ROOT"]),
                dataset=str(dataset_name),
                split=str(split),
                seq=str(seq_name),
            )
        except Exception as exc:
            raise ValueError(f"Invalid EXTERNAL_DET_PATTERN='{det_pattern}': {exc}") from exc

    return os.path.join(
        str(config["DATA_ROOT"]),
        str(dataset_name),
        str(split),
        str(seq_name),
        "det",
        "det.txt",
    )

