from __future__ import annotations
from typing import Any, Generator, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .types import ImageEncoderType
DEFAULT_DILATION_VALUES_FOR_RESNET = (False, False, True)
ImageEncoderOutputType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

class ImageEncoder(nn.Module):
    """Image encoder trunk module for the ``ImageModel`` class.

    :param img_encoder_type : Type of image encoder model to use, either ``"resnet18_multi_image"`` or
                              ``"resnet50_multi_image"``.
    """

    def __init__(self, img_encoder_type: str, cfg):
        super().__init__()
        self.img_encoder_type = img_encoder_type
        self.cfg = cfg
        self.encoder = self._create_encoder()
        if self.cfg.ablation == 'cls_baseline':
            self.classifier_pcr = nn.Linear(2048, 2)
        if 'cls' in self.cfg.ablation:
            # self.fc_img_c = nn.Linear(384, 2)
            self.fc_img_c = nn.Sequential(
                nn.Linear(384, int(384/2)),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(int(384/2), 2),
            )  # output layer

            # self.fc_img_p = nn.Linear(384, 2)
            self.fc_img_p = nn.Sequential(
                nn.Linear(384, int(384 / 2)),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(int(384 / 2), 2),
            )  # output layer

            self.fc_tem_v = nn.Linear(384, 2)

            # self.fc_report_c = nn.Linear(384, 2)
            self.fc_report_c = nn.Sequential(
                nn.Linear(384, int(384 / 2)),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(int(384 / 2), 2),
            )  # output layer

            # self.fc_report_p = nn.Linear(384, 2)
            self.fc_report_p = nn.Sequential(
                nn.Linear(384, int(384 / 2)),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(int(384 / 2), 2),
            )  # output layer


            self.fc_lva = nn.Linear(16, 2)
            self.fc_lta = nn.Linear(16, 2)

            # self.fc_Img_MT_b = nn.Linear(384 * 2, 2)
            self.fc_Img_MT_b =  nn.Sequential(
                nn.Linear(384*2, 384),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(384, 2),
            )  # output layer

            # self.fc_Img_MT_p = nn.Linear(384 * 2 + 16, 2) #simirelated
            self.fc_Img_MT_p = nn.Sequential(
                nn.Linear(384 * 2 + 16, 384+8),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(384+8, 2),
            )  # output layer

            # self.fc_Img_MT_p = nn.Linear(384 * 3, 2) #tem
            self.fc_ImgText_ST = nn.Linear(384 * 2, 2)
            self.fc_ImgText_MT_b = nn.Linear(384 * 4, 2)

            if 'ImgText_MT_pre_cls_proposed_tem' == cfg.ablation:
                self.fc_ImgText_MT_p = nn.Linear(384 * 6, 2)
            else:
                if 'ImgText_MT_pre_cls_proposed_simibce'== cfg.ablation:
                    # self.fc_ImgText_MT_p = nn.Linear(384 * 4, 2)
                    # self.fc_ImgText_MT_p = nn.Linear(384 * 4 + 2, 2)
                    # self.fc_ImgText_MT_p = nn.Linear(384 * 4 + 16*2, 2)
                    self.fc_ImgText_MT_p = nn.Sequential(
                        nn.Linear(384 * 4 + 16*2, 384*2+16),
                        nn.Dropout(),
                        nn.ReLU(),
                        nn.Linear(384*2+16, 2),
                    )  # output layer
                else:
                    if 'ImgText_MT_pre_cls_proposed_simibceispy2' == cfg.ablation:
                        self.fc_ImgText_MT_p = nn.Sequential(
                            nn.Linear(384 * 4 + 16 * 1, 384 * 2 + 8),
                            nn.Dropout(),
                            nn.ReLU(),
                            nn.Linear(384 * 2 + 8, 2),
                        )  # output layer

                    else:
                        self.fc_ImgText_MT_p = nn.Linear(384 * 4 + 16 * 2, 2)
            # self.fc_ImgText_MT_p = nn.Linear(384 * 4 + 16 * 1, 2)
        self.classifier = nn.Linear(2048, 3)
        self.classifier2 = nn.Linear(2048, 6)
    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        if self.img_encoder_type in [ImageEncoderType.RESNET18, ImageEncoderType.RESNET18_MULTI_IMAGE]:
            encoder_class = resnet18
        elif self.img_encoder_type in [ImageEncoderType.RESNET50, ImageEncoderType.RESNET50_MULTI_IMAGE]:
            encoder_class = resnet50
        else:
            supported = ImageEncoderType.get_members(multi_image_encoders_only=False)
            raise NotImplementedError(f"Image encoder type \"{self.img_encoder_type}\" must be in {supported}")

        encoder = encoder_class(pretrained=False, **kwargs)

        return encoder

    def forward(self, current_image: torch.Tensor, return_patch_embeddings: bool = False) -> ImageEncoderOutputType:

        x1, x2, x3, patch_emb = self.encoder(current_image)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_emb, (1, 1)), 1)

        logits = self.classifier(avg_pooled_emb)
        logits_task2 = self.classifier2(avg_pooled_emb)
        try:
            logits_pcr = self.classifier_pcr(avg_pooled_emb)
        except:
            logits_pcr = 0
        if return_patch_embeddings:
            # return patch_emb, avg_pooled_emb, x1, x2, x3
            return patch_emb, logits, x1, x2, x3, logits_task2, logits_pcr

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        """Workaround for enabling dilated convolutions after model initialization.

        :param replace_stride_with_dilation: Replace the 2x2 standard convolution stride with a dilated convolution
                                             in each layer in the last three blocks of ResNet architecture.
        """
        if self.img_encoder_type == ImageEncoderType.RESNET18:
            # resnet18 uses BasicBlock implementation, which does not support dilated convolutions.
            raise NotImplementedError("resnet18 does not support dilated convolutions")

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = DEFAULT_DILATION_VALUES_FOR_RESNET

        device = next(self.encoder.parameters()).device
        new_encoder = self._create_encoder(replace_stride_with_dilation=replace_stride_with_dilation).to(device)

        # if self.encoder.training:
        #     new_encoder.train()
        # else:
        #     new_encoder.eval()

        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder


class MultiImageEncoder(ImageEncoder):
    """Multi-image encoder trunk module for the ``ImageModel`` class.
    It can be used to encode multiple images into combined latent representation.
    Currently it only supports two input images but can be extended to support more in future.

    :param img_encoder_type: Type of image encoder model to use: either ``"resnet18"`` or ``"resnet50"``.
    """

    def __init__(self, img_encoder_type: str, cfg):
        super().__init__(img_encoder_type, cfg)

        output_dim = 384  # The aggregate feature dim of the encoder is `2 * output_dim` i.e. [f_static, f_diff]
        grid_shape = (16, 16)  # Spatial dimensions of patch grid. (14,14)

        backbone_output_feature_dim = get_encoder_output_dim(self.encoder, device=get_module_device(self))
        self.cfg = cfg
        self.backbone_to_vit = nn.Conv2d(
            in_channels=backbone_output_feature_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.vit_pooler = VisionTransformerPooler(input_dim=output_dim, grid_shape=grid_shape) #*2 for tem
        self.vit_fc_task1 = nn.Linear(384, 3)
        self.vit_fc_task2 = nn.Linear(384, 6)

        self.cad_fc_task1 = nn.Linear(768, 3)
        self.cad_fc_task2 = nn.Linear(768, 6)

        if self.cfg.ablation == 'cls_baseline':
            self.vit_fc_task3 = nn.Linear(384, 2)
            self.cad_fc_task3 = nn.Linear(768, 2)
        # Missing image embedding
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, output_dim, 1, 1))
        trunc_normal_(self.missing_previous_emb, std=0.02)

    def forward(  # type: ignore[override]
        self,
        current_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False,
    ) -> ImageEncoderOutputType:
        batch_size = current_image.shape[0]

        if previous_image is not None:
            assert current_image.shape == previous_image.shape
            x = torch.cat([current_image, previous_image], dim=0)
            all = super().forward(x, return_patch_embeddings=True) #patch_emb, avg_pooled_emb, x1, x2, x3
            x = all[0]
            logits_c_t1 = all[1]
            logits_c_t2 = all[5]
            logits_c_t3 = all[6]
            x = self.backbone_to_vit(x)
            patch_x, patch_x_previous = x[:batch_size], x[batch_size:]
            diff_x, diff_all = self.vit_pooler(current_image=patch_x, previous_image=patch_x_previous)#diff_x: diff_to_now[b,0:c,0:256], diff_all: diff_to_before[b,0:c,256:-1]
            logits_diff_t1 = self.vit_fc_task1(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(diff_x, (1, 1)), 1))
            logits_diff_t2 = self.vit_fc_task2(torch.flatten(torch.nn.functional.adaptive_avg_pool2d(diff_x, (1, 1)), 1))
            try:
                logits_diff_t3 = self.vit_fc_task3(
                    torch.flatten(torch.nn.functional.adaptive_avg_pool2d(diff_x, (1, 1)), 1))
            except:
                logits_diff_t3 = 0
            diff_all_g = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(diff_all, (1, 1)), 1)
            avg_pooled_emb_previous = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_x_previous, (1, 1)), 1)
        else:
            x = super().forward(current_image, return_patch_embeddings=True)[0] #patch_emb, avg_pooled_emb, x1, x2, x3
            patch_x = self.backbone_to_vit(x)
            B, _, W, H = patch_x.shape
            diff_x = self.missing_previous_emb.repeat(B, 1, W, H)

        patch_fused = torch.cat([patch_x, diff_x], dim=1)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_fused, (1, 1)), 1)
        logits_cad_t1 = self.cad_fc_task1(avg_pooled_emb)
        logits_cad_t2 = self.cad_fc_task2(avg_pooled_emb)
        try:
            logits_cad_t3 = self.cad_fc_task3(avg_pooled_emb)
        except:
            logits_cad_t3 = 0
        if return_patch_embeddings:
            if previous_image is not None:
                return patch_fused, avg_pooled_emb, diff_all, diff_all_g, patch_x_previous, avg_pooled_emb_previous, all[2],all[3],all[4], logits_c_t1, logits_c_t2, logits_diff_t1, logits_diff_t2, logits_cad_t1, logits_cad_t2, logits_c_t3, logits_diff_t3, logits_cad_t3
            else:
                return patch_fused, avg_pooled_emb

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        raise NotImplementedError


@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module, device: torch.device) -> int:
    """Calculate the output dimension of an encoder by making a single forward pass.

    :param module: Encoder module.
    :param device: Compute device to use.
    """
    # Target device
    assert isinstance(device, torch.device)

    x = torch.rand((1, 3, 448, 448)).to(device)

    # Extract the number of output feature dimensions
    # with restore_training_mode(module):
        # module.eval()
    representations = module(x)
    try:
        return representations[3].shape[1]
    except:
        return representations.shape[1]

def get_encoder_from_type(img_encoder_type: str, cfg) -> ImageEncoder:
    """Returns the encoder class for the given encoder type.

    :param img_encoder_type: Encoder type. {RESNET18, RESNET50, RESNET18_MULTI_IMAGE, RESNET50_MULTI_IMAGE}
    """
    if img_encoder_type in ImageEncoderType.get_members(multi_image_encoders_only=True):
        return MultiImageEncoder(img_encoder_type=img_encoder_type, cfg =cfg)
    else:
        return ImageEncoder(img_encoder_type=img_encoder_type, cfg =cfg)
