# Optional evaluator import guard for detector-only YOLOX training in this fork.
from .coco_evaluator import COCOEvaluator
try:
    from .mot_evaluator import MOTEvaluator
except ModuleNotFoundError:
    MOTEvaluator = None
try:
    from .voc_evaluator import VOCEvaluator
except ModuleNotFoundError:
    VOCEvaluator = None
