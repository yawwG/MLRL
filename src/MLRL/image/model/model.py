from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import get_encoder_from_type, get_encoder_output_dim, MultiImageEncoder
from .modules import MLP, MultiTaskModel
from .types import ImageModelOutput

class BaseImageModel(nn.Module, ABC):
    """Abstract class for image models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> ImageModelOutput:
        raise NotImplementedError

    @abstractmethod
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        raise NotImplementedError


class ImageModel(BaseImageModel):
    """Image encoder module"""

    def __init__(
        self,
        cfg,
        img_encoder_type: str,
        joint_feature_size: int,
        freeze_encoder: bool = False,
        pretrained_model_path: Optional[Union[str, Path]] = None,

        **downstream_classifier_kwargs: Any,
    ):
        super().__init__()

        # Initiate encoder, projector, and classifier
        self.encoder = get_encoder_from_type(img_encoder_type, cfg)
        self.feature_size = get_encoder_output_dim(self.encoder, device=get_module_device(self.encoder))
        self.projector = MLP(
            input_dim=self.feature_size,
            output_dim=joint_feature_size,
            hidden_dim=joint_feature_size,
            use_1x1_convs=True,
        )
        self.projector2 = MLP(
            input_dim= int(self.feature_size / 2),
            output_dim=joint_feature_size,
            hidden_dim=joint_feature_size,
            use_1x1_convs=True,
        )
        self.downstream_classifier_kwargs = downstream_classifier_kwargs
        self.classifier = self.create_downstream_classifier() if downstream_classifier_kwargs else None

        # Initialise the mode of modules
        self.freeze_encoder = freeze_encoder
        # self.train()

    def forward(self, x: torch.Tensor) -> ImageModelOutput:  # type: ignore[override]
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        return self.forward_post_encoder(patch_x, pooled_x)

    def forward_post_encoder(self, patch_x: torch.Tensor, pooled_x: torch.Tensor,  differ_to_before: torch.Tensor, differ_to_before_g: torch.Tensor, patch_x_previous: torch.Tensor, avg_pooled_emb_previous: torch.Tensor,  x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, logits_c_t1: torch.Tensor, logits_c_t2: torch.Tensor, logits_diff_t1: torch.Tensor, logits_diff_t2: torch.Tensor, logits_cad_t1: torch.Tensor, logits_cad_t2: torch.Tensor, logits_c_t3: torch.Tensor, logits_diff_t3: torch.Tensor, logits_cad_t3: torch.Tensor) -> ImageModelOutput:

        return ImageModelOutput(
            cad_patch_embeddings_g=pooled_x, # concat (current_x, differ_to_now) [b,512]
            cad_patch_embeddings_l=patch_x, # concat (current_x, differ_to_now)  [b,512, 16, 16]
            diff_all_l = differ_to_before,  # concat (differ_to_now, differ_to_before)  [b,512, 16, 16]
            diff_all_g =differ_to_before_g,  # concat (differ_to_now, differ_to_before)  [b,512]
            previous_embedding_l = patch_x_previous, #previous_x [b,256,16,16]
            previous_embedding_g = avg_pooled_emb_previous,  #previous_x [b,256]
            x1 = x1,
            x2 = x2,
            x3 = x3,
            logits_cp_t1 = logits_c_t1,
            logits_cp_t2 = logits_c_t2,
            logits_d_t1 = logits_diff_t1,
            logits_d_t2 = logits_diff_t2,
            logits_cad_t1 = logits_cad_t1,
            logits_cad_t2 = logits_cad_t2,
            logits_c_t3= logits_c_t3,
            logits_diff_t3= logits_diff_t3,
            logits_cad_t3= logits_cad_t3,
        )

    def create_downstream_classifier(self, **kwargs: Any) -> MultiTaskModel:
        """Create the classification module for the downstream task."""
        downstream_classifier_kwargs = kwargs if kwargs else self.downstream_classifier_kwargs
        return MultiTaskModel(self.feature_size, **downstream_classifier_kwargs)

    @torch.no_grad()
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Get patch-wise projected embeddings from the CNN model.

        :param input_img: input tensor image [B, C, H, W].
        :param normalize: If ``True``, the embeddings are L2-normalized.
        :returns projected_embeddings: tensor of embeddings in shape [batch, n_patches_h, n_patches_w, feature_size].
        """
        # assert not self.training, "This function is only implemented for evaluation mode"
        outputs = self.forward(input_img)
        projected_embeddings = outputs.projected_patch_embeddings.detach()  # type: ignore
        if normalize:
            projected_embeddings = F.normalize(projected_embeddings, dim=1)
        projected_embeddings = projected_embeddings.permute([0, 2, 3, 1])  # B D H W -> B H W D (D: Features)
        return projected_embeddings


class MultiImageModel(ImageModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.encoder, MultiImageEncoder), "MultiImageModel only supports MultiImageEncoder"

    def forward(  # type: ignore[override]
        self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None
    ) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x, diff_all, diff_all_g, patch_x_previous, avg_pooled_emb_previous, x1, x2, x3, logits_c_t1, logits_c_t2, logits_diff_t1, logits_diff_t2, logits_cad_t1, logits_cad_t2 = self.encoder(
                current_image=current_image, previous_image=previous_image, return_patch_embeddings=True
            )
        return self.forward_post_encoder(patch_x, pooled_x, diff_all, diff_all_g,  patch_x_previous, avg_pooled_emb_previous,  x1, x2, x3, logits_c_t1, logits_c_t2, logits_diff_t1, logits_diff_t2, logits_cad_t1, logits_cad_t2 )
