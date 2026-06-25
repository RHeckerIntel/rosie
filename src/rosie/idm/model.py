"""
IDM model: SigLIP-2 (frozen) → SA backbone → flow-matching DiT action head.

Architecture matches the DreamGen paper:
  - SigLIP-2 large patch16-256 encodes both frames independently
  - 4-layer self-attention backbone fuses patch tokens from both frames
  - 8-layer DiT with cross-attention + AdaLN predicts action sequences via flow matching
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoModel

SIGLIP_MODEL = "google/siglip2-large-patch16-256"
SIGLIP_DIM   = 1024
SIGLIP_SIZE  = 256


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / max(half - 1, 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t[:, None] * self.freqs[None]
        return self.proj(torch.cat([emb.sin(), emb.cos()], dim=-1))


class SABlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class DiTBlock(nn.Module):
    """Self-attn + cross-attn to visual context + AdaLN from timestep embedding."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn1 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn2 = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        s1, b1, s2, b2, s3, b3 = [a.unsqueeze(1) for a in self.adaLN(t_emb).chunk(6, dim=-1)]
        h = self.norm1(x) * (1 + s1) + b1
        x = x + self.attn1(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        x = x + self.attn2(h, ctx, ctx, need_weights=False)[0]
        h = self.norm3(x) * (1 + s3) + b3
        return x + self.ff(h)


class IDM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_horizon: int = 16,
        num_cameras: int = 1,
        hidden: int = SIGLIP_DIM,
        backbone_layers: int = 4,
        dit_layers: int = 8,
        heads: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon
        self.num_cameras    = num_cameras

        self.siglip   = AutoModel.from_pretrained(SIGLIP_MODEL, torch_dtype=torch.bfloat16)
        self.siglip.requires_grad_(False)

        self.backbone = nn.ModuleList([SABlock(hidden, heads, dropout) for _ in range(backbone_layers)])

        self.t_emb      = TimestepEmbedding(hidden)
        self.action_in  = nn.Linear(action_dim, hidden)
        self.action_out = nn.Linear(hidden, action_dim)
        self.dit        = nn.ModuleList([DiTBlock(hidden, heads, dropout) for _ in range(dit_layers)])

    def _preprocess(self, frame: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] in [0, 1] → resized + normalized for SigLIP-2."""
        if frame.shape[-2:] != (SIGLIP_SIZE, SIGLIP_SIZE):
            frame = TF.resize(frame, [SIGLIP_SIZE, SIGLIP_SIZE], antialias=True)
        return (frame * 2.0 - 1.0).to(dtype=self.siglip.dtype)

    def _encode_one(self, frame: torch.Tensor) -> torch.Tensor:
        dev = next(self.siglip.parameters()).device
        return self.siglip.vision_model(
            pixel_values=self._preprocess(frame).to(dev)
        ).last_hidden_state

    def siglip_encode(
        self,
        frames_t:  list[torch.Tensor],
        frames_tH: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run SigLIP only. Returns raw tokens [B, 2*num_cams*seq, hidden].

        This is the expensive frozen step — cache its output to skip it every epoch.
        Token order: [cam0_t, cam0_tH, cam1_t, cam1_tH, ...]
        """
        def _pad(lst):
            while len(lst) < self.num_cameras:
                lst = lst + [lst[-1]]
            return lst[:self.num_cameras]

        frames_t  = _pad(list(frames_t))
        frames_tH = _pad(list(frames_tH))

        tokens = []
        with torch.no_grad():
            for ft, ftH in zip(frames_t, frames_tH):
                tokens.append(self._encode_one(ft))
                tokens.append(self._encode_one(ftH))
        return torch.cat(tokens, dim=1)

    def _backbone_forward(self, raw: torch.Tensor) -> torch.Tensor:
        ctx = raw.float()
        for block in self.backbone:
            ctx = block(ctx)
        return ctx

    def encode_frames(
        self,
        frames_t:  list[torch.Tensor],
        frames_tH: list[torch.Tensor],
    ) -> torch.Tensor:
        """SigLIP → backbone. Returns fused context [B, seq, hidden]."""
        return self._backbone_forward(self.siglip_encode(frames_t, frames_tH))

    def encode_from_cache(self, cached_tokens: torch.Tensor) -> torch.Tensor:
        """Backbone only — skips SigLIP. Pass output of siglip_encode."""
        return self._backbone_forward(cached_tokens)

    def _dit_forward(self, ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, device = actions.shape[0], actions.device
        noise  = torch.randn_like(actions)
        t      = torch.rand(B, device=device)
        noisy  = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        target = noise - actions
        t_emb  = self.t_emb(t)
        x = self.action_in(noisy)
        for block in self.dit:
            x = block(x, ctx, t_emb)
        return F.mse_loss(self.action_out(x), target)

    def forward(
        self,
        frames_t:  list[torch.Tensor],
        frames_tH: list[torch.Tensor],
        actions:   torch.Tensor,
        *,
        cached_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow matching training loss (MSE on velocity field)."""
        ctx = (self.encode_from_cache(cached_tokens)
               if cached_tokens is not None
               else self.encode_frames(frames_t, frames_tH))
        return self._dit_forward(ctx, actions)

    @torch.no_grad()
    def get_actions(
        self,
        frames_t:  list[torch.Tensor],
        frames_tH: list[torch.Tensor],
        steps: int = 16,
    ) -> torch.Tensor:
        """Euler ODE integration from noise to actions. Returns [B, H, action_dim]."""
        B      = frames_t[0].shape[0]
        device = frames_t[0].device
        ctx = self.encode_frames(frames_t, frames_tH)
        z   = torch.randn(B, self.action_horizon, self.action_dim, device=device)
        dt  = 1.0 / steps
        for i in range(steps, 0, -1):
            t_emb = self.t_emb(torch.full((B,), i / steps, device=device))
            x = self.action_in(z)
            for block in self.dit:
                x = block(x, ctx, t_emb)
            z = z - dt * self.action_out(x)
        return z
