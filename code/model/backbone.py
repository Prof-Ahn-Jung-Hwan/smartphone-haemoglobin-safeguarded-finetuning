import torch
import torch.nn as nn
from torchvision.models import (
    convnext_base,
    convnext_large,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnext50_32x4d,
    resnext101_64x4d,
    vit_b_16,
    # mobile backbones
    efficientnet_b0,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
)

# Backbones and their feature dimension d. Torchvision mobile backbones use `.features` (B, C, H, W);
# timm backbones use forward_features. ResNet/ResNeXt (not used in the study) enlarge the last feature map by
# setting the layer-4 stride to 1 unless light_backbone is set.

# torchvision mobile backbones: name -> feature dimension
_MOBILE_OUT_DIM = {
    "efficientnet_b0": 1280,
    "mobilenet_v2": 1280,
    "mobilenet_v3_large": 960,
    "mobilenet_v3_small": 576,
    "shufflenet_v2_x1_0": 1024,
}


# timm hybrid convolution–transformer backbones: name -> feature dimension (all return (B, C, H, W))
_TIMM_BACKBONES = {
    "mobilevit_xxs":     320,
    "mobilevit_xs":      384,
    "mobilevit_s":       640,
    "mobilevitv2_050":   256,
    "mobilevitv2_075":   384,
    "edgenext_xx_small": 168,
    "edgenext_x_small":  192,
    "edgenext_small":    304,
    "fastvit_t8":            768,
    "efficientformerv2_s0":  176,
}


def is_spatial_backbone(name):
    """True if the backbone returns a spatial feature map (B, C, H, W)."""
    return (name.startswith("resnet") or name.startswith("resnext")
            or name.startswith("convnext") or name in _MOBILE_OUT_DIM
            or name in _TIMM_BACKBONES)


def create_backbone(args):
    # timm backbones: features only (num_classes=0)
    if args.backbone in _TIMM_BACKBONES:
        import timm
        m = timm.create_model(args.backbone, pretrained=True, num_classes=0)
        out_dim = _TIMM_BACKBONES[args.backbone]

        class _TimmFeat(nn.Module):
            """Feature extractor returning (B, C, H, W)."""
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.forward_features(x)

        return _TimmFeat(m), out_dim

    if args.backbone.startswith("dino"):
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        out_dim = 768
        return backbone, out_dim

    try:
        backbone = eval(args.backbone)(weights="IMAGENET1K_V1")
    except:
        raise NotImplementedError(f"Backbone {args.backbone} is not supported!")

    if args.backbone.startswith("resnet") or args.backbone.startswith("resnext"):
        out_dim = backbone.inplanes
        if not getattr(args, "light_backbone", False):
            # set the stride of the strided convolution in the first layer-4 block to 1 (14 x 14 feature map)
            blk = backbone.layer4[0]
            if getattr(blk, "conv2", None) is not None and blk.conv2.stride != (1, 1):
                blk.conv2.stride = (1, 1)
            elif getattr(blk, "conv1", None) is not None:
                blk.conv1.stride = (1, 1)
            if getattr(blk, "downsample", None) is not None:
                blk.downsample[0].stride = (1, 1)
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif args.backbone in _MOBILE_OUT_DIM:
        out_dim = _MOBILE_OUT_DIM[args.backbone]
        if args.backbone.startswith("efficientnet") or args.backbone.startswith("mobilenet"):
            # .features returns (B, C, H, W)
            backbone = backbone.features
        else:   # shufflenet_v2 has no .features: chain its stages
            backbone = nn.Sequential(backbone.conv1, backbone.maxpool, backbone.stage2,
                                     backbone.stage3, backbone.stage4, backbone.conv5)
    elif args.backbone.startswith("convnext"):
        out_dim = backbone.classifier[0].normalized_shape[0]
        modules = list(backbone.children())[:-2]
        backbone = nn.Sequential(*modules)
    elif args.backbone.startswith("vit"):
        backbone.heads = nn.Identity()
        out_dim = backbone.hidden_dim
    else:
        raise NotImplementedError(f"Backbone {args.backbone} is not supported!")

    return backbone, out_dim


class Bottleneck(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = self.channel_attn(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x
