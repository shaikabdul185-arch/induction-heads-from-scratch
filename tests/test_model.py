import torch

from induction.config import Config
from induction.data import half_masks, repeated_tokens
from induction.model import AttnOnlyTransformer, per_token_loss


def small_cfg(**kw) -> Config:
    return Config(vocab_size=20, min_half_len=4, max_half_len=8, d_model=16, n_heads=2, d_head=8, **kw)


def test_data_is_repeated():
    x = repeated_tokens(3, 5, 20, generator=torch.Generator().manual_seed(0))
    assert x.shape == (3, 10)
    assert torch.equal(x[:, :5], x[:, 5:])
    assert x.min() >= 0 and x.max() < 20


def test_half_masks():
    first, repeated = half_masks(4)  # targets are positions 1..7
    assert first.tolist() == [True, True, True, False, False, False, False]
    assert repeated.tolist() == [False, False, False, False, True, True, True]


def test_forward_shapes():
    for ln in (False, True):
        cfg = small_cfg(use_layernorm=ln)
        model = AttnOnlyTransformer(cfg)
        tokens = repeated_tokens(2, 6, cfg.vocab_size)
        logits, patterns = model(tokens, return_patterns=True)
        assert logits.shape == (2, 12, cfg.vocab_size)
        assert len(patterns) == cfg.n_layers
        for p in patterns:
            assert p.shape == (2, cfg.n_heads, 12, 12)
            torch.testing.assert_close(p.sum(-1), torch.ones(2, cfg.n_heads, 12))
            assert torch.all(p.triu(diagonal=1) == 0)  # causal
        assert per_token_loss(logits, tokens).shape == (2, 11)


def test_no_mlp_and_param_count():
    cfg = small_cfg()
    model = AttnOnlyTransformer(cfg)
    names = [n for n, _ in model.named_parameters()]
    assert not any("mlp" in n for n in names)
    D, V, C, I = cfg.d_model, cfg.vocab_size, cfg.n_ctx, cfg.n_heads * cfg.d_head
    attn = 3 * (D * I + I) + (I * D + D)
    expected = V * D + C * D + cfg.n_layers * attn + (D * V + V)
    assert sum(p.numel() for p in model.parameters()) == expected
