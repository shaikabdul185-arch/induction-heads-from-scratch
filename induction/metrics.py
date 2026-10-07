"""Per-head attention scores."""

from __future__ import annotations

import torch


def induction_mask(tokens: torch.Tensor) -> torch.Tensor:
    """mask[b, t, s] is True iff source s is the token right after an earlier occurrence
    of the destination token, i.e. tokens[b, s-1] == tokens[b, t] with s-1 < t.

    tokens: [B, T] -> mask: [B, T_dst, T_src]
    """
    B, T = tokens.shape
    prev = torch.zeros_like(tokens)
    prev[:, 1:] = tokens[:, :-1]
    same = tokens[:, :, None] == prev[:, None, :]  # [B, dst, src]: tokens[dst] == tokens[src-1]
    s = torch.arange(T)
    valid_src = (s[None, :] >= 1) & (s[None, :] - 1 < s[:, None])  # s >= 1 and s-1 < t
    return same & valid_src[None]


def induction_score(pattern: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Average attention each head puts on "the token right after the previous occurrence
    of the current token", over all destination positions where such a token exists.

    pattern: [B, H, T_dst, T_src] attention probabilities
    tokens:  [B, T]
    returns: [H] scores in [0, 1]
    """
    mask = induction_mask(tokens).to(pattern.dtype)  # [B, T, T]
    per_dst = (pattern * mask[:, None]).sum(-1)  # [B, H, T]
    has_target = mask.sum(-1) > 0  # [B, T]
    weight = has_target[:, None].to(pattern.dtype)
    return (per_dst * weight).sum((0, 2)) / weight.sum((0, 2)).clamp_min(1)


def prev_token_score(pattern: torch.Tensor) -> torch.Tensor:
    """Average attention from position t to t-1 (t >= 1). Returns [H]."""
    return pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean((0, 2))
