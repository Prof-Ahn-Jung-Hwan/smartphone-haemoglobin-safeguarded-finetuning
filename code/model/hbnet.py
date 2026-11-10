import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from model.backbone import Bottleneck, create_backbone, is_spatial_backbone
from model.encoder import VariantionalEncoder
from model.head import Head


class HbNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hybrid_mode = args.hybrid_mode
        self.backbone_name = args.backbone
        self.ordinal_bins = args.ordinal_bins
        self.use_encoder = args.use_encoder
        self.use_ordinal_regression = args.use_ordinal_regression

        self.backbone, self.out_dim = create_backbone(args)

        # 2026-08-19 파일럿: 백본 이름을 나열하는 대신 is_spatial_backbone() 로 판정한다.
        # 원본은 resnet/resnext/convnext 만 나열했으므로 모바일 백본에는
        # CBAM·어텐션 경로가 아예 만들어지지 않는다. 판정 결과는 기존 백본에 대해 동일하다.
        #
        # ★ 2026-08-24 추가 — force_attn.  **기본 동작은 그대로다** (기본값 False).
        #   원래 구조에서는 어텐션 풀링이 VE/OR 머리에만 붙는다. 그래서 "MSE 대 VE+OR"
        #   비교가 실은 (GAP + Linear + MSE) 대 (어텐션풀링 + Bottleneck + VE + OR) 로
        #   네 가지가 한꺼번에 바뀌는 비교였다. npj 리뷰어 1 이 요구한
        #   "same backbone and feature pathway with a standard continuous regression head"
        #   를 만들려면 어텐션은 켜고 VE/OR 만 끈 칸이 있어야 한다. 그 칸을 여는 스위치다.
        #   그리고 반대 방향 스위치도 둔다 — no_attn (2026-08-24).
        #   1c 결과 : 어텐션은 MSE 머리에서 성능에 기여가 없었다(+0.010, 6/6 유의하지 않음).
        #   그런데 추가 파라미터의 55~58% 를 쓴다(4d²). 휴대폰이 전제인 논문에서는
        #   빼는 것이 이득일 수 있으므로, VE/OR 을 켠 채로 어텐션만 끄는 칸이 필요하다.
        self.force_attn = bool(getattr(args, "force_attn", False))
        self.no_attn = bool(getattr(args, "no_attn", False))
        if (not self.no_attn) \
                and (self.use_encoder or self.use_ordinal_regression or self.force_attn) \
                and is_spatial_backbone(self.backbone_name):
            self.bottleneck = Bottleneck(self.out_dim)
            self.att_norm = nn.LayerNorm(self.out_dim)
            # 헤드 수는 out_dim 을 나눌 수 있어야 한다 (2048/64=32, 576/64=9, 512/64=8 ... 모두 정수)
            n_heads = max(1, self.out_dim // 64)
            while self.out_dim % n_heads:
                n_heads -= 1
            self.attn = nn.MultiheadAttention(self.out_dim, n_heads, batch_first=True)

        if self.hybrid_mode:
            self.hybrid_proj = nn.Sequential(
                nn.Linear(len(args.hybrid_attrs), args.hybrid_proj_dim),
                nn.ReLU(),
                nn.Linear(args.hybrid_proj_dim, args.hybrid_proj_dim),
                nn.ReLU(),
                nn.Linear(args.hybrid_proj_dim, args.hybrid_proj_dim),
            )
            self.hybrid_emb = nn.Linear(args.hybrid_proj_dim + self.out_dim, self.out_dim)

        # ★ 09-25 기전 실험 스위치 (기본값 전부 False — 기존 동작 그대로)
        self.use_norm_only = bool(getattr(args, "use_norm_only", False))   # F : mse + LayerNorm
        self.no_prenorm    = bool(getattr(args, "no_prenorm", False))      # VE−LN : 인코더 앞 정규화만 제거
        if self.use_norm_only and not args.use_encoder:
            self.norm = nn.LayerNorm(self.out_dim)

        if args.use_encoder:
            self.norm = nn.Identity() if self.no_prenorm else nn.LayerNorm(self.out_dim)
            self.encoder = VariantionalEncoder(self.out_dim,
                                               no_sampling=bool(getattr(args, "no_sampling", False)))
            z_prior = torch.empty(self.ordinal_bins - 1, self.out_dim)
            nn.init.orthogonal_(z_prior, gain=1.3)
            self.register_buffer("z_prior", z_prior)

        self.dropout = nn.Dropout(0.15)
        if args.use_ordinal_regression:
            # ★ 2026-08-25 — ArcFace 각도여백 m 을 인자로 뺀다. 기본 0.5 = 기존 동작 그대로.
            #   m=0 이면 cos_m=1, sin_m=0 이라 phi==cosine 이 되어 여백이 완전히 사라지고
            #   '코사인 정규화 + 스케일 s' 만 남는다 (head.py 수정 불필요).
            self.arc_m = float(getattr(args, "arc_m", 0.5))
            self.head = Head(self.out_dim, (self.ordinal_bins - 1) * 2, m=self.arc_m)
        else:
            self.head = nn.Linear(self.out_dim, 1)

    def forward(self, img, hybrid_anno=None, label=None):
        if self.hybrid_mode:
            assert hybrid_anno is not None

        if self.use_ordinal_regression and self.training:
            assert label is not None

        z = self.backbone(img)
        if (not self.no_attn) \
                and (self.use_encoder or self.use_ordinal_regression or self.force_attn) \
                and is_spatial_backbone(self.backbone_name):
            z = self.bottleneck(z)
            z = rearrange(z, "b c h w -> b (h w) c")
            z = self.att_norm(z)
            t = torch.cat([z, z.mean(1, keepdims=True)], 1)
            t = self.attn(t, t, t, need_weights=False)[0]
            z = t[:, -1]
        else:
            if z.ndim == 4:
                z = F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)
            elif z.ndim == 2:
                """Do nothing if the output is already flattened"""
            else:
                raise ValueError(f"Unsupported backbone output shape: {z.shape}")

        if self.hybrid_mode:
            z = self.hybrid_emb(torch.cat([z, self.hybrid_proj(hybrid_anno)], dim=-1))

        if self.use_encoder:
            z = self.norm(z)
            z = self.encoder(z)
        elif self.use_norm_only:                   # ★ F 팔 — 인코더 없이 정규화만
            z = self.norm(z)
        out_latent = z
        z = self.dropout(z)

        if self.use_ordinal_regression:
            logits = self.head(z, label)
            logits = rearrange(logits, "B (K P) -> B K P", P=2)
            ord_k = F.relu(logits.softmax(-1).argmax(-1).sum(-1, keepdim=True) - 1).int()
        else:
            logits = self.head(z)
            ord_k = None

        return {
            "z": out_latent,
            "logits": logits,
            "ord_k": ord_k,
            "z_prior": self.z_prior if self.use_encoder else None,
        }
