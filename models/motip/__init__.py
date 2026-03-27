# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.dino.dino import build_dino
from models.motip.freq_aware_trajectory_modeling import (
    build_frequency_aware_modules,
    FrequencyAwareIDDecoder,
)
from models.motip.freq_aware_id_decoder_v2 import FrequencyAwareIDDecoderV2
from util.slconfig import SLConfig


def build(config: dict):
    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            detr_args = Args()
            detr_args.backbone = config["BACKBONE"]
            detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
            detr_args.dilation = config["DILATION"]
            detr_args.num_classes = config["NUM_CLASSES"]
            detr_args.device = config["DEVICE"]
            detr_args.num_queries = config["DETR_NUM_QUERIES"]
            detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
            detr_args.aux_loss = config["DETR_AUX_LOSS"]
            detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
            detr_args.two_stage = config["DETR_TWO_STAGE"]
            detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
            detr_args.masks = config["DETR_MASKS"]
            detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
            detr_args.nheads = config["DETR_NUM_HEADS"]
            detr_args.enc_layers = config["DETR_ENC_LAYERS"]
            detr_args.dec_layers = config["DETR_DEC_LAYERS"]
            detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
            detr_args.dropout = config["DETR_DROPOUT"]
            detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
            detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
            detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
            detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
            detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
            detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
            detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
            detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
            detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case "dino":
            # Load the base DINO config and override it with MOTIP configs.
            dino_config_path = config.get("DINO_CONFIG_PATH", "config/DINO/DINO_5scale.py")
            detr_args = SLConfig.fromfile(dino_config_path)
            detr_args.device = config["DEVICE"]
            detr_args.num_classes = config["NUM_CLASSES"]
            detr_args.num_queries = config["DETR_NUM_QUERIES"]
            detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
            detr_args.aux_loss = config["DETR_AUX_LOSS"]
            detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
            detr_args.masks = config["DETR_MASKS"]
            detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
            detr_args.nheads = config["DETR_NUM_HEADS"]
            detr_args.enc_layers = config["DETR_ENC_LAYERS"]
            detr_args.dec_layers = config["DETR_DEC_LAYERS"]
            detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
            detr_args.dropout = config["DETR_DROPOUT"]
            detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
            detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
            detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
            detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
            detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
            detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
            detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
            detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
            detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]
            detr_args.two_stage_type = config.get("DINO_TWO_STAGE_TYPE", getattr(detr_args, "two_stage_type", "standard"))
            detr_args.num_select = config.get("DINO_NUM_SELECT", getattr(detr_args, "num_select", 300))
            detr_args.nms_iou_threshold = config.get(
                "DINO_NMS_IOU_THRESHOLD", getattr(detr_args, "nms_iou_threshold", -1)
            )
            detr_args.dec_pred_class_embed_share = config.get(
                "DINO_DEC_PRED_CLASS_EMBED_SHARE", getattr(detr_args, "dec_pred_class_embed_share", True)
            )
            detr_args.dec_pred_bbox_embed_share = config.get(
                "DINO_DEC_PRED_BBOX_EMBED_SHARE", getattr(detr_args, "dec_pred_bbox_embed_share", True)
            )
            detr_args.two_stage_bbox_embed_share = config.get(
                "DINO_TWO_STAGE_BBOX_EMBED_SHARE", getattr(detr_args, "two_stage_bbox_embed_share", False)
            )
            detr_args.two_stage_class_embed_share = config.get(
                "DINO_TWO_STAGE_CLASS_EMBED_SHARE", getattr(detr_args, "two_stage_class_embed_share", False)
            )
            detr_args.decoder_sa_type = config.get("DINO_DECODER_SA_TYPE", getattr(detr_args, "decoder_sa_type", "sa"))
            detr_args.num_patterns = config.get("DINO_NUM_PATTERNS", getattr(detr_args, "num_patterns", 0))
            detr_args.use_dn = config.get("DINO_USE_DN", getattr(detr_args, "use_dn", False))
            detr_args.dn_number = config.get("DINO_DN_NUMBER", getattr(detr_args, "dn_number", 0))
            detr_args.dn_box_noise_scale = config.get(
                "DINO_BOX_NOISE_SCALE", getattr(detr_args, "dn_box_noise_scale", 0.4)
            )
            detr_args.dn_label_noise_ratio = config.get(
                "DINO_LABEL_NOISE_RATIO", getattr(detr_args, "dn_label_noise_ratio", 0.5)
            )
            detr_args.dn_labelbook_size = config.get(
                "DINO_LABELBOOK_SIZE", getattr(detr_args, "dn_labelbook_size", config["NUM_CLASSES"])
            )
            detr, detr_criterion, _ = build_dino(detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build each component:
    # 1. trajectory modeling + ID decoder:
    if config["ONLY_DETR"] is False and config.get("USE_FREQ_AWARE", False):
        _trajectory_modeling = build_frequency_aware_modules(config=config)

        use_freq_guided_assoc = config.get(
            "USE_FREQ_GUIDED_ASSOC",
            config.get("USE_FREQ_GUIDED_ASSOCIATION", True),
        )
        
        use_v2_decoder = config.get("USE_FREQ_DECODER_V2", False)
        if use_v2_decoder:
            _id_decoder = FrequencyAwareIDDecoderV2(
                feature_dim=config["FEATURE_DIM"],
                id_dim=config["ID_DIM"],
                ffn_dim_ratio=config["FFN_DIM_RATIO"],
                num_layers=config["NUM_ID_DECODER_LAYERS"],
                head_dim=config["HEAD_DIM"],
                num_id_vocabulary=config["NUM_ID_VOCABULARY"],
                rel_pe_length=config["REL_PE_LENGTH"],
                use_aux_loss=config["USE_AUX_LOSS"],
                use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
                use_freq_guided_association=use_freq_guided_assoc,
                num_bands=config.get("NUM_FREQ_BANDS", config.get("NUM_BANDS", 4)),
                use_learnable_fusion=config.get("USE_LEARNABLE_FUSION", True),
                freq_loss_weight=config.get("FREQ_LOSS_WEIGHT", 1.0),
                fusion_loss_weight=config.get("FUSION_LOSS_WEIGHT", 1.0),
                use_mamba_self_attn=config.get("USE_MAMBA_IN_ID_DECODER", True),
            )
        else:
            _id_decoder = FrequencyAwareIDDecoder(
                feature_dim=config["FEATURE_DIM"],
                id_dim=config["ID_DIM"],
                ffn_dim_ratio=config["FFN_DIM_RATIO"],
                num_layers=config["NUM_ID_DECODER_LAYERS"],
                head_dim=config["HEAD_DIM"],
                num_id_vocabulary=config["NUM_ID_VOCABULARY"],
                rel_pe_length=config["REL_PE_LENGTH"],
                use_aux_loss=config["USE_AUX_LOSS"],
                use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
                use_freq_guided_association=use_freq_guided_assoc,
                num_bands=config.get("NUM_FREQ_BANDS", config.get("NUM_BANDS", 4)),
            )
    elif config["ONLY_DETR"] is False:
        try:
            from models.motip.trajectory_modeling import TrajectoryModeling
            from models.motip.id_decoder import IDDecoder
        except Exception as e:
            raise ImportError(
                "Failed to import TrajectoryModeling/IDDecoder. "
                "If you are running the frequency-aware version, set USE_FREQ_AWARE: True. "
                "Otherwise please install optional deps (e.g., mamba_ssm)."
            ) from e

        _trajectory_modeling = TrajectoryModeling(
            detr_dim=config["DETR_HIDDEN_DIM"],
            ffn_dim_ratio=config["FFN_DIM_RATIO"],
            feature_dim=config["FEATURE_DIM"],
            use_freq_adapter=config.get("USE_FREQ_ADAPTER", False),
        )
        _id_decoder = IDDecoder(
            feature_dim=config["FEATURE_DIM"],
            id_dim=config["ID_DIM"],
            ffn_dim_ratio=config["FFN_DIM_RATIO"],
            num_layers=config["NUM_ID_DECODER_LAYERS"],
            head_dim=config["HEAD_DIM"],
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            rel_pe_length=config["REL_PE_LENGTH"],
            use_aux_loss=config["USE_AUX_LOSS"],
            use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
        )
    else:
        _trajectory_modeling = None
        _id_decoder = None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
        num_id_vocabulary=config.get("NUM_ID_VOCABULARY", None),
    )

    return motip_model, detr_criterion
