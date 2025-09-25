# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask2former.modeling.pixel_decoder.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

import logging
from functools import partial
from typing import Dict, List
from einops import rearrange

from s3.utils.misc import NestedTensor, is_main_process

import torch.utils.checkpoint as cp
from timm.models.layers import DropPath

from .backbones import get_models
from .position_encoding import build_position_encoding
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import CNNBlockBase, Conv2d, get_norm

_logger = logging.getLogger(__name__)





def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`ViTdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        vit_module,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
        freeze_backbone=False,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        in_feature = 'last_feat'
        input_shapes = {'last_feat': ShapeSpec(channels=vit_module.embed_dim, height=None, width=None, stride=16)}
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]

        dim = input_shapes[in_feature].channels

        out_channels = vit_module.embed_dim

        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
# TODO out_channels's dim
            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.vit_module = vit_module
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"res{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["res{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

        if freeze_backbone:
            for p in self.vit_module.parameters():
                p.requires_grad_(False)
        self.freeze_backbone = freeze_backbone

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        _, _, h, w = x.shape
        if not self.freeze_backbone:
            x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        else:
            with torch.no_grad():
                x, H, W = self.vit_module.prepare_tokens_with_masks(x, masks=None, return_HW=True)
        bs, n, dim = x.shape
        cls = x[:, :1, :]
        x = x[:, 1:, :]
        if self.freeze_backbone:
            self.vit_module.eval()
            for idx, blk in enumerate(self.vit_module.blocks):
                x = blk(x)
        else:
            for idx, blk in enumerate(self.vit_module.blocks):
                x = blk(x)
        bottom_up_features = x
        features = x
        b, s, e = features.shape
        # features = features.reshape(b, int(math.sqrt(s)), int(math.sqrt(s)), e).permute(0, 3, 1, 2)
        features = features.reshape(b, h // 16, w // 16, e).permute(0, 3, 1, 2)

        results = []
        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return [results[0], results[1], results[2]]



def get_fpn_args(name='ViTl', backbone_weight=None,
                     freeze_backbone=False, with_cp=False, scale_factors=None):
    if freeze_backbone:
        assert backbone_weight is not None
    ViT_backbone = get_models(name, backbone_weight)
    bg_configs = ViT_backbone.configs_dict
    if name == 'vitl':
        fpn_args = {
            'vit_module': ViT_backbone,
            'scale_factors': scale_factors,
        }
    elif name == 'vitb':
        fpn_args = {
            'vit_module': ViT_backbone,
            'scale_factors': scale_factors,
        }
    else:
        raise NotImplementedError
    fpn_args.update({
        'freeze_backbone': freeze_backbone,
    })
    return fpn_args


class DinoV2FPN(SimpleFeaturePyramid, Backbone):
    def __init__(self, cfg):

        name = cfg.pixel_encoder.type
        backbone_weight = cfg.pixel_encoder.vit_weights
        freeze_backbone = cfg.pixel_encoder.freeze_vit
        with_cp = cfg.pixel_encoder.with_cp
        scale_factors=[4.0, 2.0, 1.0]
        fpn_args = get_fpn_args(
            name=name,
            backbone_weight=backbone_weight,
            freeze_backbone=freeze_backbone,
            with_cp=with_cp,
            scale_factors = scale_factors,
        )
        
        super().__init__(**fpn_args
            ),

        self._out_features = ['res2', 'res3', 'res4']

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": 256,
            "res3": 256,
            "res4": 256,
            "res5": 256,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"DinoV2 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x)
        return y[2], y[1], y[0]

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32







def main():
    # backbone = ViT_base()
    # x = torch.randn(32, 3, 224, 224)
    # out = backbone(x)
    # print(out.shape)
    backbone = build_base_fpn_dinov2()
    x = torch.randn(1, 3, 1024, 1024)
    #out = backbone(x)
    
    model = torch.load("/home/deshui/pro/rvos/s3/checkpoints/dinov2_ViTb14_reg4_pretrain.pth")
    backbone.ViT.load_state_dict(model) 
    out = backbone(x)
    print(out['res2'].size(), out['res3'].size(), out['res4'].size(), out.keys())
    # model = torch.hub.load(repo_or_dir="/home/yangzhen/.cache/torch/hub/facebookresearch_dinov2_main", model='dinov2_ViTs14', source='local')
    # print(out['res2'][0][0][0][0])
    

    

if __name__ == '__main__':
    main()