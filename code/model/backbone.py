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
    # ── 2026-08-19 파일럿 추가: 모바일 배치용 백본 ──
    efficientnet_b0,
    mobilenet_v2,          # 2026-08-24 추가 — 선행 연구 근거 (Xu 2024 MAE 1.19, Chen 2022)
    mobilenet_v3_large,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
)

# ===========================================================================
# 2026-08-19 백본 용량 파일럿을 위한 확장
# ---------------------------------------------------------------------------
# 원본(`../model/backbone.py`)과 다른 점은 두 가지뿐이다.
#
#   1) 모바일 백본 4종 추가 (EfficientNet-B0, MobileNetV3 Large/Small, ShuffleNetV2)
#   2) `light_backbone` 플래그 추가
#        ResNet/ResNeXt 계열은 원본이 layer4 의 stride 를 1 로 바꿔
#        마지막 특징맵을 7x7 -> 14x14 로 키운다 (공간 해상도 4배).
#        이 플래그를 켜면 그 처리를 건너뛰어 표준 7x7 을 쓴다.
#
#        ★ 기본값 False → **기존 설정은 원본과 완전히 동일하게 동작한다.**
#
# 각 백본이 내놓는 특징 차원 d (VE 3d^2 · 어텐션 4d^2 이므로 비용이 d^2 에 비례한다)
#   ResNeXt50 2048 / EfficientNet-B0 1280 / ShuffleNetV2 1024
#   MobileNetV3-Large 960 / MobileNetV3-Small 576 / ResNet18 512
# ===========================================================================

# 백본 이름 -> (특징 추출부를 얻는 함수, 특징 차원)
_MOBILE_OUT_DIM = {
    "efficientnet_b0": 1280,
    # ★ 2026-08-24 추가. 이 과제의 선행 연구가 실제로 쓴 백본이라 반드시 넣는다.
    #   Xu et al. Actor-Critic RL(MobileNetV2) MAE 1.19  ← 선행 최고 기록
    #   Chen et al. 2022 (외부 공개자료 제공 논문) Table 3 후보에도 들어 있다
    "mobilenet_v2": 1280,
    "mobilenet_v3_large": 960,
    "mobilenet_v3_small": 576,
    "shufflenet_v2_x1_0": 1024,
}


# ── 2026-08-22 추가: timm 모바일 ViT 계열 ──────────────────────────────────
# 왜 넣는가
#   VE/OR 머리 비용은 특징차원 d 에 비례한다(인코더 3d² + 어텐션 4d²).
#   MobileViT·EdgeNeXt 는 d 가 168~384 로 작아 **머리가 싸게 붙는다.**
#     mobilevit_xxs   d=320  0.95M -> VE+OR 1.72M   8ms
#     mobilenet_v3_small d=576  0.93M -> VE+OR 3.43M   8ms   ← 백본은 같은 크기인데 머리가 2배
#   전부 (B,C,H,W) 공간 특징맵을 내놓는 것을 확인했으므로 어텐션 경로가 그대로 작동한다.
#
# ⚠️ EdgeViT 은 timm 에 사전학습 가중치가 없다(0개). 같은 계열인 EdgeNeXt 로 대체한다.
#    논문에는 'EdgeNeXt' 로 정확히 쓸 것.
#
# ⚠️ ViT 계열은 자료를 많이 먹는다. 환자 730명·사진 1,460장에서 CNN 보다 나쁠 수 있고
#    그것은 아키텍처가 아니라 자료 크기 탓이다. 나쁘게 나와도 그대로 보고한다.
_TIMM_BACKBONES = {
    "mobilevit_xxs":     320,
    "mobilevit_xs":      384,
    "mobilevit_s":       640,
    "mobilevitv2_050":   256,
    "mobilevitv2_075":   384,
    "edgenext_xx_small": 168,
    "edgenext_x_small":  192,
    "edgenext_small":    304,
    # ── 2026-09-13 추가: 후보 10종 중 2023년 세대 CNN-ViT 두 종 (주 문서 §2-1 Tier 2) ──
    #   둘 다 forward_features 가 (B,C,7,7) 공간 특징맵을 내놓는 것을 실측 확인했다.
    #   가중치 : timm/fastvit_t8.apple_dist_in1k · timm/efficientformerv2_s0.snap_dist_in1k (hf download)
    "fastvit_t8":            768,
    "efficientformerv2_s0":  176,
}


def is_spatial_backbone(name):
    """공간 특징맵 (B,C,H,W) 을 내놓는 백본인가.

    HbNet 의 CBAM·어텐션 경로는 (B,C,H,W) 출력을 전제로 한다.
    ViT·DINO 처럼 이미 평탄화된 출력을 내는 백본은 그 경로를 쓸 수 없다.
    ★ timm 모바일 ViT 계열은 2026-08-22 에 전부 (B,C,H,W) 임을 실측 확인했다.
    """
    return (name.startswith("resnet") or name.startswith("resnext")
            or name.startswith("convnext") or name in _MOBILE_OUT_DIM
            or name in _TIMM_BACKBONES)


def create_backbone(args):
    # ★ timm 백본은 torchvision 보다 먼저 본다 (이름이 겹치지 않는다).
    #   forward_features 만 쓰므로 분류 머리는 만들지 않는다(num_classes=0).
    if args.backbone in _TIMM_BACKBONES:
        import timm
        m = timm.create_model(args.backbone, pretrained=True, num_classes=0)
        out_dim = _TIMM_BACKBONES[args.backbone]

        class _TimmFeat(nn.Module):
            """(B,C,H,W) 를 내놓는 특징 추출부만 감싼다. HbNet 이 기대하는 형태다."""
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
        # light_backbone=True 이면 stride 제거를 건너뛴다 (특징맵 14x14 -> 7x7).
        # 파라미터 수는 변하지 않고 연산량과 어텐션 토큰 수만 줄어든다.
        if not getattr(args, "light_backbone", False):
            # ★ 원본은 conv2 의 stride 만 1 로 바꾼다. ResNeXt50·ResNet50 처럼
            #   Bottleneck 블록을 쓰는 백본은 stride 가 conv2 에 있어 맞다.
            #   그러나 ResNet18·34 는 BasicBlock 이라 stride 가 conv1 에 있고,
            #   원본대로 하면 본선(7x7)과 지름길(14x14)의 크기가 어긋나 순전파가 깨진다.
            #   (원본에서 resnet18 은 import 만 되어 있고 실제로 쓰인 적이 없어 드러나지 않았다)
            #   여기서는 stride 를 실제로 갖고 있는 conv 를 찾아서 바꾼다.
            #   → Bottleneck 계열의 동작은 원본과 완전히 동일하다.
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
            # torchvision 의 이 계열은 .features 가 (B,C,H,W) 를 내놓는다
            backbone = backbone.features
        else:   # shufflenet_v2 — features 속성이 없어 단계를 직접 잇는다
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
