import torch

from induction.metrics import induction_mask, induction_score, prev_token_score


def test_induction_mask_on_repeated_sequence():
    tokens = torch.tensor([[5, 6, 7, 5, 6, 7]])
    mask = induction_mask(tokens)[0]
    # dst 3 (token 5) -> src 1 (token after the earlier 5); dst 4 -> src 2; dst 5 -> src 3
    expected = torch.zeros(6, 6, dtype=torch.bool)
    expected[3, 1] = expected[4, 2] = expected[5, 3] = True
    assert torch.equal(mask, expected)


def test_induction_mask_handles_duplicates():
    tokens = torch.tensor([[1, 2, 1, 3, 1]])
    mask = induction_mask(tokens)[0]
    assert mask[2].nonzero().flatten().tolist() == [1]  # after the first 1
    assert mask[4].nonzero().flatten().tolist() == [1, 3]  # after both earlier 1s


def _one_hot_pattern(T, src_of_dst):
    p = torch.zeros(T, T)
    for dst, src in enumerate(src_of_dst):
        p[dst, src] = 1.0
    return p


def test_perfect_and_zero_induction_heads():
    L = 4
    tokens = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    T = 2 * L
    perfect = _one_hot_pattern(T, [0, 0, 1, 2, 1, 2, 3, 4])  # second half: dst t -> t-L+1
    self_attn = torch.eye(T)
    pattern = torch.stack([perfect, self_attn])[None]  # [1, H=2, T, T]
    score = induction_score(pattern, tokens)
    torch.testing.assert_close(score, torch.tensor([1.0, 0.0]))


def test_partial_induction_score():
    tokens = torch.tensor([[0, 1, 0, 1]])  # induction dsts: 2 -> 1, 3 -> 2
    pattern = torch.zeros(1, 1, 4, 4)
    pattern[0, 0, :, 0] = 1.0
    pattern[0, 0, 2] = torch.tensor([0.5, 0.5, 0.0, 0.0])
    pattern[0, 0, 3] = torch.tensor([0.75, 0.0, 0.25, 0.0])
    torch.testing.assert_close(induction_score(pattern, tokens), torch.tensor([0.375]))


def test_prev_token_score():
    T = 5
    prev = torch.zeros(T, T)
    prev[0, 0] = 1
    prev[torch.arange(1, T), torch.arange(T - 1)] = 1
    pattern = torch.stack([prev, torch.eye(T)])[None]
    torch.testing.assert_close(prev_token_score(pattern), torch.tensor([1.0, 0.0]))
