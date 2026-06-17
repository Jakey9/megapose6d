"""Minimal DINOv2 Vision Transformer implementation (Python 3.9 compatible).

Self-contained ViT that loads official DINOv2 pretrained weights directly
from Facebook AI URLs, avoiding the torch.hub dependency on the upstream
dinov2 repo which requires Python 3.10+.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


DINOV2_WEIGHTS = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}

DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 14},
    "dinov2_vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 14},
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 518, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, N, D]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DinoVisionTransformer(nn.Module):
    """Minimal Vision Transformer matching DINOv2 architecture."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate positional embeddings for arbitrary input resolutions."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patches == N and h == w:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        sqrt_N = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features returning dict with normalized patch and CLS tokens."""
        B, _, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        x = self.patch_embed(x)  # [B, N, D]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, D]

        x = x + self.interpolate_pos_embed(x, h, w)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_token_out = x[:, 0]
        patch_tokens = x[:, 1:]

        # Normalize (matching DINOv2 forward_features output format)
        cls_norm = F.normalize(cls_token_out, dim=-1)
        patches_norm = F.normalize(patch_tokens, dim=-1)

        return {
            "x_norm_clstoken": cls_norm,
            "x_norm_patchtokens": patches_norm,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_features(x)
        return out["x_norm_clstoken"]


def load_dinov2(model_name: str = "dinov2_vits14", pretrained: bool = True) -> DinoVisionTransformer:
    """Load a DINOv2 model with pretrained weights.

    Args:
        model_name: One of "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14".
        pretrained: If True, load official Facebook AI pretrained weights.

    Returns:
        DinoVisionTransformer model.
    """
    if model_name not in DINOV2_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(DINOV2_CONFIGS.keys())}"
        )

    cfg = DINOV2_CONFIGS[model_name]
    model = DinoVisionTransformer(
        img_size=518,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )

    if pretrained:
        url = DINOV2_WEIGHTS[model_name]
        state_dict = load_state_dict_from_url(url, map_location="cpu")
        # Filter out unexpected keys (register tokens, etc. from newer checkpoints)
        model_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in model_keys}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            print(f"[DINOv2] Missing keys (expected for minimal impl): {missing}")

    return model
