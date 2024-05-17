# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
    EfficientNet_B5_Weights,
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
)
from .efficientnet import (
    film_efficientnet_b0,
    film_efficientnet_b3,
    film_efficientnet_b5,
)
from .resnet import film_resnet18, film_resnet34, film_resnet50

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython

# from .vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as Vim_tiny
# from .vim.models_mamba import vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as Vim_tiny_ft
# from .vim.models_mamba import vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as Vim_small
# from .vim.models_mamba import vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as Vim_small_ft

e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        name,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # The IntermediateLayerGetter logic below removes the last few layers in the CNNs (e.g., average pooling and classification layer)
        # and returns a dictionary-style model. For example, for the EfficientNet, it returns an ordered dictionary where the key is '0'
        # and the value is the model portion preceding the final average pooling and classification layers.
        if "resnet" in name:
            if return_interm_layers:
                return_layers = {
                    "layer1": "0",
                    "layer2": "1",
                    "layer3": "2",
                    "layer4": "3",
                }
            else:
                return_layers = {"layer4": "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        elif 'efficientnet' in name:  # efficientnet
            return_layers = {"features": "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        elif 'Vim' in name:
            # return_layers = {"norm_f": "0"}
            self.body = backbone #IntermediateLayerGetter(backbone, return_layers=return_layers)
            
        self.num_channels = num_channels

class Backbone(BackboneBase):
    """Image encoder backbone."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        # Load pretrained weights.
        if name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            num_channels = 512
        elif name == "resnet34":
            weights = ResNet34_Weights.DEFAULT
            num_channels = 512
        elif name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            num_channels = 2048
        elif name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            num_channels = 1280
        elif name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT
            num_channels = 1536
            
        # elif name == "Vim_tiny":
        #     weights = '/home/boxjod/RLBench/Mamba/Vim/ckpt/vim_t_midclstok_76p1acc.pth'
        #     num_channels = 192
        # elif name == "Vim_tiny_ft":
        #     weights = '/home/boxjod/RLBench/Mamba/Vim/ckpt/vim_t_midclstok_ft_78p3acc.pth'
        #     num_channels = 192
        # elif name == "Vim_small":
        #     weights = '/home/boxjod/RLBench/Mamba/Vim/ckpt/vim_s_midclstok_80p5acc.pth'
        #     num_channels = 384
        # elif name == "Vim_small_ft":
        #     weights = '/home/boxjod/RLBench/Mamba/Vim/ckpt/vim_s_midclstok_ft_81p6acc.pth'
        #     num_channels = 384
        else:
            raise ValueError
        
        # Initialize pretrained model.
        if "resnet" in name:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                weights=weights,
                norm_layer=FrozenBatchNorm2d,
            )  # pretrained
        elif 'efficientnet' in name:  # efficientnet
            backbone = getattr(torchvision.models, name)(
                weights=weights, norm_layer=FrozenBatchNorm2d
            )  # pretrained
        # elif 'Vim' in name:
        #     backbone = eval(name)(pretrained=True,ckpt_path=weights, drop_block_rate=None, img_size=224)
            
        super().__init__(
            name, backbone, train_backbone, num_channels, return_interm_layers
        )
        # Get image preprocessing function.
        # self.preprocess = (
        #     weights.transforms(antialias=False)
        # )  # Use this to preprocess images the same way as the pretrained model (e.g., ResNet-18).

    def forward(self, tensor):
        # tensor = self.preprocess(tensor) # 第一版的ACT没有做 preprocess,做了效果很不好
        xs = self.body(tensor)
        return xs


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        # xs的形状大小是啥
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x) 
            # position encoding
            pos.append(self[1](x).to(x.dtype)) # 里面根本没有用到x的值
            
        return out, pos


class FilMedBackbone(torch.nn.Module):
    """FiLMed image encoder backbone."""

    def __init__(self, name: str):
        super().__init__()
        # Load pretrained weights.
        if name == "resnet18film":
            weights = ResNet18_Weights.DEFAULT
            self.num_channels = 512
            self.backbone = film_resnet18(weights=weights, use_film=True)
        elif name == "resnet34film":
            weights = ResNet34_Weights.DEFAULT
            self.num_channels = 512
            self.backbone = film_resnet34(weights=weights, use_film=True)
        elif name == "resnet50film":
            weights = ResNet50_Weights.DEFAULT
            self.num_channels = 2048
            self.backbone = film_resnet50(weights=weights, use_film=True)
        elif name == "efficientnet_b0film":
            weights = EfficientNet_B0_Weights.DEFAULT
            self.num_channels = 1280
            self.backbone = film_efficientnet_b0(weights=weights, use_film=True)
        elif name == "efficientnet_b3film":
            weights = EfficientNet_B3_Weights.DEFAULT
            self.num_channels = 1536
            self.backbone = film_efficientnet_b3(weights=weights, use_film=True)
        elif name == "efficientnet_b5film":
            weights = EfficientNet_B5_Weights.DEFAULT
            self.num_channels = 2048
            self.backbone = film_efficientnet_b5(weights=weights, use_film=True)
        else:
            raise ValueError
        # Remove final average pooling and classification layers.
        if "resnet" in name:
            self.backbone.avgpool = nn.Sequential()  # remove average pool layer
            self.backbone.fc = nn.Sequential()  # remove classification layer
        else:  # efficientnet
            self.backbone.avgpool = nn.Sequential()  # remove average pool layer
            self.backbone.classifier = nn.Sequential()  # remove classification layer
        # Get image preprocessing function.
        self.preprocess = (weights.transforms(antialias=False))  # Use this to preprocess images the same way as the pretrained model (e.g., ResNet-18).
        
        # self.preprocess = transforms.Compose([ # Use this if you don't want to resize images to 224x224.
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

    def forward(self, img_obs, language_embed):
        # img_obs shape: (batch_size, 3, H, W)
        img_obs = img_obs.float()  # cast to float type
        img_obs = self.preprocess(img_obs)
        out = self.backbone(img_obs, language_embed)  # shape (B, C_final * H_final * W_final) or (B, C_final, H_final, W_final)
        
        # If needed, unflatten output tensor from (B, C_final * H_final * W_final) to (B, C_final, H_final, W_final)
        if len(out.shape) == 2:
            H_final = W_final = int(math.sqrt(out.shape[-1] // self.num_channels))
            out = torch.unflatten(out, -1, (self.num_channels, H_final, W_final))
        return out


class FiLMedJoiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, img_obs, language_embed):
        """
        Args:
            img_obs: torch.Tensor of shape (batch_size, C, H, W)
            language_embed: torch.Tensor of shape (batch_size, language_embed_size)

        Returns:
            out: 1-length list of torch.Tensor of shape (batch_size, C_final, H_final, W_final).
            pos: 1-length list of torch.Tensor of shape (1, hidden_dim, H_final, W_final).
        """
        # self[0]: backbone, self[1]: position_embedding
        out = [self[0](img_obs, language_embed)]
        pos = [self[1](out[0]).to(out[0].dtype)]
        return out, pos


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    position_embedding = build_position_encoding(args)
    if 'Vim' in args.backbone:
        # backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        # model = Joiner(backbone, position_embedding)
        x = 1
    else:
        
        if "film" in args.backbone:
            # print("Using FiLMed backbone.")
            backbone = FilMedBackbone(args.backbone)
            model = FiLMedJoiner(backbone, position_embedding)
        else:
            backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
            model = Joiner(backbone, position_embedding)
            
    
    model.num_channels = backbone.num_channels
    return model
