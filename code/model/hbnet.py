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

        # Attention pooling over the spatial feature map (optional). The study used global average pooling
        # (no_attn: true in every configuration).
        self.force_attn = bool(getattr(args, "force_attn", False))
        self.no_attn = bool(getattr(args, "no_attn", False))
        if (not self.no_attn) \
                and (self.use_encoder or self.use_ordinal_regression or self.force_attn) \
                and is_spatial_backbone(self.backbone_name):
            self.bottleneck = Bottleneck(self.out_dim)
            self.att_norm = nn.LayerNorm(self.out_dim)
            # number of heads must divide out_dim
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

        # Safeguards: S1 = LayerNorm of the final features; S2 = adaptation layer (two linear layers, the encoder with
        # no_sampling). S1 only: use_norm_only; S2 only: use_encoder with no_prenorm; S1+S2: use_encoder.
        self.use_norm_only = bool(getattr(args, "use_norm_only", False))   # S1 without S2
        self.no_prenorm    = bool(getattr(args, "no_prenorm", False))      # S2 without S1
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
            # ordinal head (ArcFace margin m; not used in the study)
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
        elif self.use_norm_only:                   # S1 only
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
