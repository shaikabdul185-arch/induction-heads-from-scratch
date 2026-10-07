import pytest
import torch

from induction.config import Config
from induction.data import repeated_tokens
from induction.model import AttnOnlyTransformer
from induction.tl_convert import to_hooked_transformer


def randomized_model(use_layernorm: bool) -> AttnOnlyTransformer:
    cfg = Config(vocab_size=20, min_half_len=4, max_half_len=8, d_model=16, n_heads=2, d_head=8,
                 use_layernorm=use_layernorm)
    torch.manual_seed(0)
    model = AttnOnlyTransformer(cfg)
    # Large random weights *and* biases so a layout mistake can't hide behind a near-zero init.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.5)
    return model.eval()


@pytest.mark.parametrize("use_layernorm", [False, True])
def test_transformerlens_matches_pytorch(use_layernorm):
    model = randomized_model(use_layernorm)
    tl_model = to_hooked_transformer(model)
    assert tl_model.cfg.attn_only and tl_model.cfg.n_layers == 2
    tokens = repeated_tokens(4, 8, model.cfg.vocab_size, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ours, patterns = model(tokens, return_patterns=True)
        theirs, cache = tl_model.run_with_cache(tokens)
    torch.testing.assert_close(theirs, ours, atol=1e-5, rtol=1e-5)
    for l in range(model.cfg.n_layers):
        torch.testing.assert_close(cache["pattern", l], patterns[l], atol=1e-6, rtol=1e-5)
