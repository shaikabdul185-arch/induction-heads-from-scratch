"""Synthetic repeated-random-token sequences.

Each sequence is `half_len` uniformly random tokens followed by an exact copy:

    [x_1, ..., x_L, x_1, ..., x_L]

Tokens in the first half are unpredictable (loss can't beat log(vocab_size)), so the only
way to reduce loss is to copy from context in the second half.
"""

from __future__ import annotations

import torch


def repeated_tokens(
    batch_size: int, half_len: int, vocab_size: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    first = torch.randint(0, vocab_size, (batch_size, half_len), generator=generator)
    return torch.cat([first, first], dim=1)


def half_masks(half_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks over the T-1 next-token *targets* (target index j = position j+1).

    - first half: targets at positions 1..L-1 (random tokens, unpredictable)
    - repeated half: targets at positions L+1..2L-1 (predictable by induction:
      the current token x_i has appeared before, and the answer is the token after it)

    The target at position L (= x_1, predicted from x_L) is in neither: x_L has not
    appeared before, so nothing in context says what comes next.
    """
    target_pos = torch.arange(1, 2 * half_len)
    first = target_pos < half_len
    repeated = target_pos > half_len
    return first, repeated
