import importlib.util
from pathlib import Path


def _load_bot_sort_module():
    module_path = Path("/gemini/code/FMtrack-main/FM-Track/external/BoT-SORT-main/tracker/bot_sort.py")
    spec = importlib.util.spec_from_file_location("bot_sort_runtime_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_low_score_detections_default_to_no_features():
    module = _load_bot_sort_module()
    det = module.STrack(module.STrack.tlbr_to_tlwh([10.0, 20.0, 30.0, 60.0]), 0.2)
    assert det.curr_feat is None
    assert det.smooth_feat is None


def test_low_score_recovery_can_attach_features_on_demand():
    module = _load_bot_sort_module()
    det = module.STrack(module.STrack.tlbr_to_tlwh([10.0, 20.0, 30.0, 60.0]), 0.2)
    feat = module.np.array([1.0, 2.0, 3.0], dtype=module.np.float32)
    det.update_features(feat.copy())
    assert det.curr_feat is not None
    assert det.smooth_feat is not None
