from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List

import torch

@dataclass
class ImageModelOutput:
    cad_patch_embeddings_g :torch.Tensor  # concat (current_x, differ_to_now) [b,512]
    cad_patch_embeddings_l  :torch.Tensor  # concat (current_x, differ_to_now)  [b,512, 16, 16]
    diff_all_l  :torch.Tensor  # concat (differ_to_now, differ_to_before)  [b,512, 16, 16]
    diff_all_g  :torch.Tensor # concat (differ_to_now, differ_to_before)  [b,512]
    previous_embedding_l :torch.Tensor  # previous_x [b,256,16,16]
    previous_embedding_g :torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    x3: torch.Tensor
    logits_cp_t1: torch.Tensor
    logits_cp_t2: torch.Tensor
    logits_d_t1: torch.Tensor
    logits_d_t2: torch.Tensor
    logits_cad_t1: torch.Tensor
    logits_cad_t2: torch.Tensor
    logits_c_t3: torch.Tensor
    logits_diff_t3: torch.Tensor
    logits_cad_t3: torch.Tensor

@unique
class ImageEncoderType(str, Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET18_MULTI_IMAGE = "resnet18_multi_image"
    RESNET50_MULTI_IMAGE = "resnet50_multi_image"

    @classmethod
    def get_members(cls, multi_image_encoders_only: bool) -> List[ImageEncoderType]:
        if multi_image_encoders_only:
            return [cls.RESNET18_MULTI_IMAGE, cls.RESNET50_MULTI_IMAGE]
        else:
            return [member for member in cls]

@unique
class ImageEncoderWeightTypes(str, Enum):
    RANDOM = "random"
    IMAGENET = "imagenet"
