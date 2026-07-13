#!/usr/bin/env python3
"""
D-FINE -> MOT20/BoT-SORT adapter plan.
Stage A: run D-FINE torch inference over MOT20 frames and export detections.
Stage B: feed detections into tracking/eval adapter.
This script is intentionally a scaffold until D-FINE source/weights are available locally.
"""
from pathlib import Path
import json

ROOT = Path('/gemini/code/FMtrack-main/FM-Track')
MOT20_05 = Path('/gemini/code/datasets/MOT20/train/MOT20-05/img1')
OUT = ROOT / 'outputs/detector_plugins/dfine_mot20_05'
OUT.mkdir(parents=True, exist_ok=True)
frames = sorted(MOT20_05.glob('*.jpg'))
plan = {
    'detector': 'D-FINE',
    'sequence': 'MOT20-05',
    'frames': len(frames),
    'image_dir': str(MOT20_05),
    'target_outputs': {
        'raw_jsonl': str(OUT / 'dfine_raw_detections.jsonl'),
        'mot_txt': str(OUT / 'dfine_mot20_05_det.txt'),
        'coco_json': str(OUT / 'dfine_mot20_05_coco.json'),
    },
    'person_class_policy': 'COCO person class only; keep all scores initially, sweep threshold after inference',
    'tracking_policy': 'first compare detector-only FP/FN, then feed detections into BoT-SORT if adapter path is ready',
}
(OUT / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
print(json.dumps(plan, indent=2, ensure_ascii=False))
