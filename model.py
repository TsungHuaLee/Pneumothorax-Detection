# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules

class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        
    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        x = self.conv(x)
        return x
    
class FAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(pool_size, pool_size), stride=(pool_size, pool_size)),
            modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

class FAMModule(nn.Module):
    def __init__(self, in_channels, out_channel, sizes=(1, 2, 4, 8), use_bathcnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            FAMBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

        self.conv = modules.Conv2dReLU(in_channels // len(sizes), out_channel, (3, 3), 1)
        

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x_add = torch.add(xs[0], xs[1])
        x_add = torch.add(x_add, xs[2])
        x_add = torch.add(x_add, xs[3])
        
        conv_out = self.conv(x_add)
        return x

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        
    def forward(self, x, skip=None, ggf=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip + ggf
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )
    
class FPNDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [3, 2, 1, 0]
        ])

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        self.GGM = PSPModule(encoder_channels[0], pyramid_channels, sizes=(1,2,3,6))
        self.f4 = FAMModule(pyramid_channels, pyramid_channels)
        self.f3 = FAMModule(pyramid_channels, pyramid_channels)
        self.f2 = FAMModule(pyramid_channels, pyramid_channels)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]

        GGM = self.GGM(c5)

        p5 = self.p5(c5)

        GGF4 =  F.interpolate(GGM, scale_factor=2, mode="bilinear", align_corners=True)
        p4 = self.p4(p5, c4, GGF4)
        f4 = self.f4(p4)

        GGF3 =  F.interpolate(GGF4, scale_factor=2, mode="bilinear", align_corners=True)
        p3 = self.p3(f4, c3, GGF3)
        f3 = self.f3(p3)

        GGF2 =  F.interpolate(GGF3, scale_factor=2, mode="bilinear", align_corners=True)
        p2 = self.p2(p3, c2, GGF2)
        f2 = self.f3(p2)
        
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, f4, f3, f2])]

        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x

from typing import Optional, Union
from segmentation_models_pytorch.base.heads import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.encoders import get_encoder
import torch

class FPN(SegmentationModel):
    def __init__(self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_pyramid_channels: int = 256,
        decoder_segmentation_channels: int = 128,
        decoder_merge_policy: str = "add",
        decoder_dropout: float = 0.2,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        aux_params: Optional[dict] = None
    ):
        super(FPN, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "fpn-{}".format(encoder_name)
        self.initialize()
