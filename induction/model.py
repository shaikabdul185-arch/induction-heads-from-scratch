"""A 2-layer attention-only transformer written in plain PyTorch.

Architecture (matches TransformerLens' `attn_only=True` HookedTransformer):

    x = W_E[tokens] + W_pos[positions]
    for each layer:  x = x + Attn(LN?(x))        # no MLPs
    logits = Unembed(LN?(x))

LayerNorm is optional (off by default).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        inner = cfg.n_heads * cfg.d_head
        self.q = nn.Linear(cfg.d_model, inner)
        self.k = nn.Linear(cfg.d_model, inner)
        self.v = nn.Linear(cfg.d_model, inner)
        self.out = nn.Linear(inner, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output [B, T, d_model], attention pattern [B, H, T_dst, T_src])."""
        B, T, _ = x.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d_head]

        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        pattern = scores.softmax(dim=-1)
        z = (pattern @ v).transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out(z), pattern


class AttnOnlyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.layers = nn.ModuleList(CausalSelfAttention(cfg) for _ in range(cfg.n_layers))
        if cfg.use_layernorm:
            self.lns = nn.ModuleList(nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers))
            self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.named_parameters():
            if name.startswith(("lns.", "ln_final.")):
                continue  # LayerNorm keeps weight=1, bias=0
            if name.endswith("bias"):
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, mean=0.0, std=self.cfg.init_std)

    def forward(
        self, tokens: torch.Tensor, return_patterns: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        x = self.embed(tokens) + self.pos_embed(pos)
        patterns = []
        for i, attn in enumerate(self.layers):
            h = self.lns[i](x) if self.cfg.use_layernorm else x
            out, pattern = attn(h)
            x = x + out
            patterns.append(pattern)
        if self.cfg.use_layernorm:
            x = self.ln_final(x)
        logits = self.unembed(x)
        return (logits, patterns) if return_patterns else logits


def per_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of predicting tokens[:, t+1] from logits[:, t]. Shape [B, T-1]."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    ).view(tokens.shape[0], -1)
