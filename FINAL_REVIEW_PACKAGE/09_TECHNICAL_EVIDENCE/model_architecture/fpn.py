import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels: int, skip_channels: int):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy: str = "add"):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(f"`merge_policy` must be one of: ['add', 'cat'], got {policy}")
        self.policy = policy

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        encoder_depth: int = 5,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        dropout: float = 0.2,
        merge_policy: str = "add",
    ):
        super().__init__()
        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        encoder_channels = encoder_channels[::-1]

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

    def forward(self, *features) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        c2, c3, c4, c5 = features[-4:]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x, feature_pyramid


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, upsampling: int = 1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling_layer, nn.Identity())


class StandaloneResNet34Encoder(nn.Module):
    """
    Pure PyTorch 5-stage ResNet-34 encoder with single-channel grayscale input support.
    """
    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet34(weights=weights)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 1:
            with torch.no_grad():
                self.conv1.weight.copy_(base_resnet.conv1.weight.sum(dim=1, keepdim=True))

        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1  # 64
        self.layer2 = base_resnet.layer2  # 128
        self.layer3 = base_resnet.layer3  # 256
        self.layer4 = base_resnet.layer4  # 512
        self.out_channels = [in_channels, 64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c0, c1, c2, c3, c4, c5]


class Net(nn.Module):
    """
    Official MLUA Network Architecture:
    ResNet-34 Encoder + Top-Down Feature Pyramid Network + 4 Auxiliary Segmentation Heads + 1 Fused Head.
    """
    def __init__(
        self,
        in_c: int = 1,
        out_c: int = 1,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
    ):
        super().__init__()
        pretrained = bool(encoder_weights and encoder_weights.lower() != "none")
        self.encoder = StandaloneResNet34Encoder(in_channels=in_c, pretrained=pretrained)
        
        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels[1:],
            encoder_depth=encoder_depth,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            dropout=0.2,
            merge_policy="add",
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=out_c,
            kernel_size=1,
            upsampling=4,
        )
        self.aux_segmentation_head_list = nn.ModuleList([
            SegmentationHead(
                in_channels=segmentation_channels,
                out_channels=out_c,
                kernel_size=1,
                upsampling=4,
            ) for _ in range(encoder_depth - 1)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = self.encoder(x)
        decoder_output, feature_pyramid = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        aux_masks_list = [aux_head(f_p) for aux_head, f_p in zip(self.aux_segmentation_head_list, feature_pyramid)]
        return masks, aux_masks_list
