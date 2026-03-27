# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
            num_id_vocabulary: int | None = None,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder

        if self.id_decoder is not None:
            self.num_id_vocabulary = int(self.id_decoder.num_id_vocabulary)
            if num_id_vocabulary is not None and int(num_id_vocabulary) != self.num_id_vocabulary:
                import warnings
                warnings.warn(
                    f"[MOTIP] num_id_vocabulary mismatch: config={int(num_id_vocabulary)} "
                    f"but decoder={self.num_id_vocabulary}. Using decoder value."
                )
        else:
            # `only_detr=True` (or no ID decoder) should not guess a vocabulary size.
            # Keep it explicit so downstream code doesn't silently use an incorrect value.
            self.num_id_vocabulary = 0 if num_id_vocabulary is None else int(num_id_vocabulary)

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    targets = kwargs.get("targets")
                    if self.detr_framework == "dino" and targets is not None:
                        return self.detr(samples=frames, targets=targets)
                    else:
                        return self.detr(samples=frames)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
