import sys, os
sys.path.insert(0, "/gemini/code/FMtrack-main/FM-Track")
os.chdir("/gemini/code/FMtrack-main/FM-Track")

from utils.misc import yaml_to_dict
from configs.util import load_super_config
from submit_and_evaluate import submit_and_evaluate_one_model
from models.motip import build as build_motip
from models.misc import load_checkpoint
from accelerate import Accelerator, PartialState
from utils.log import Logger
from utils.detector_profile import resolve_bytetrack_profile
import torch

epochs = [int(e) for e in sys.argv[1:]] if len(sys.argv) > 1 else [3, 4, 5]

cfg = yaml_to_dict("configs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix.yaml")
cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

# Switch to MOT20 detector profile
cfg, profile_name = resolve_bytetrack_profile(cfg, dataset_name="MOT20")
print(f"Using detector profile: {profile_name}")

accelerator = Accelerator()
state = PartialState()

for epoch in epochs:
    ckpt = f"outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/checkpoint_epoch_{epoch}.pth"
    out_dir = f"outputs/bytetrack_fa_mot_mot17_v9_fromscratch_unfreezeproj_longcurr_p0fix/val_mot20/epoch_{epoch}"
    
    summary = os.path.join(out_dir, "tracker", "MOT20-train", "pedestrian_summary.txt")
    if os.path.exists(summary):
        print(f"Epoch {epoch} already done, skipping")
        continue
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== MOT20 val epoch {epoch} ===")
    
    model, _ = build_motip(cfg)
    load_checkpoint(model, path=ckpt)
    model = accelerator.prepare(model)
    
    logger = Logger(logdir=out_dir, use_wandb=False, config=cfg)
    
    metrics = submit_and_evaluate_one_model(
        is_evaluate=True,
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=cfg["DATA_ROOT"],
        dataset="MOT20",
        data_split="train",
        outputs_dir=out_dir,
        image_max_longer=cfg["INFERENCE_MAX_LONGER"],
        size_divisibility=cfg.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=cfg["MISS_TOLERANCE"],
        det_thresh=cfg["DET_THRESH"],
        newborn_thresh=cfg["NEWBORN_THRESH"],
        id_thresh=cfg["ID_THRESH"],
        area_thresh=cfg.get("AREA_THRESH", 0),
        inference_only_detr=cfg["INFERENCE_ONLY_DETR"] if cfg["INFERENCE_ONLY_DETR"] is not None else cfg["ONLY_DETR"],
        sequence_include=["MOT20-01", "MOT20-02"],
        assert_eval_only_val=False,
    )
    if metrics is not None:
        metrics.sync()
        print(f"Epoch {epoch} done")
    
    del model
    torch.cuda.empty_cache()

print("All done")
