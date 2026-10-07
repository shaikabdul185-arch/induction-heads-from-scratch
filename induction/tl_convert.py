"""Load AttnOnlyTransformer weights into a TransformerLens HookedTransformer."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .config import Config
from .model import AttnOnlyTransformer


def tl_config(cfg: Config) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        n_ctx=cfg.n_ctx,
        d_vocab=cfg.vocab_size,
        attn_only=True,
        act_fn=None,
        normalization_type="LN" if cfg.use_layernorm else None,
        positional_embedding_type="standard",
        default_prepend_bos=False,
        device="cpu",
        seed=cfg.seed,
    )


def to_tl_state_dict(model: AttnOnlyTransformer) -> dict[str, torch.Tensor]:
    """Rearrange nn.Linear / nn.Embedding weights into TransformerLens' per-head layout."""
    cfg = model.cfg
    H, dh, D = cfg.n_heads, cfg.d_head, cfg.d_model
    sd: dict[str, torch.Tensor] = {
        "embed.W_E": model.embed.weight,
        "pos_embed.W_pos": model.pos_embed.weight,
        "unembed.W_U": model.unembed.weight.T,
        "unembed.b_U": model.unembed.bias,
    }
    for i, attn in enumerate(model.layers):
        p = f"blocks.{i}.attn."
        for name, lin in (("Q", attn.q), ("K", attn.k), ("V", attn.v)):
            # nn.Linear weight is [H*dh, D]; TL wants W_[H, D, dh] and b_[H, dh]
            sd[p + f"W_{name}"] = lin.weight.view(H, dh, D).transpose(1, 2)
            sd[p + f"b_{name}"] = lin.bias.view(H, dh)
        # out weight is [D, H*dh]; TL wants W_O[H, dh, D]
        sd[p + "W_O"] = attn.out.weight.T.reshape(H, dh, D)
        sd[p + "b_O"] = attn.out.bias
        if cfg.use_layernorm:
            sd[f"blocks.{i}.ln1.w"] = model.lns[i].weight
            sd[f"blocks.{i}.ln1.b"] = model.lns[i].bias
    if cfg.use_layernorm:
        sd["ln_final.w"] = model.ln_final.weight
        sd["ln_final.b"] = model.ln_final.bias
    return {k: v.detach().clone().contiguous() for k, v in sd.items()}


def to_hooked_transformer(model: AttnOnlyTransformer) -> HookedTransformer:
    tl_model = HookedTransformer(tl_config(model.cfg))
    full_sd = tl_model.state_dict()  # includes non-parameter buffers (masks, IGNORE)
    ours = to_tl_state_dict(model)
    missing = {k for k in full_sd if k not in ours and "mask" not in k and "IGNORE" not in k}
    if missing:
        raise KeyError(f"No source weights for TransformerLens keys: {sorted(missing)}")
    full_sd.update(ours)
    tl_model.load_state_dict(full_sd, strict=True)
    tl_model.eval()
    return tl_model
