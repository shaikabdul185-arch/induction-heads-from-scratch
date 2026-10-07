"""Experiment configuration: every hyperparameter lives here and in configs/*.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # Reproducibility
    seed: int = 0

    # Data: each sequence is L random tokens followed by the same L tokens again.
    # L is drawn per batch from [min_half_len, max_half_len]; a fixed L would let the
    # model solve the task positionally ("attend L-1 positions back") instead of by induction.
    vocab_size: int = 64
    min_half_len: int = 8
    max_half_len: int = 32

    # Model: 2-layer attention-only transformer
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    use_layernorm: bool = False
    init_std: float = 0.02

    # Optimisation
    batch_size: int = 64
    n_steps: int = 4000
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.99
    grad_clip: float = 1.0

    # Logging / checkpointing
    log_every: int = 10
    eval_batch_size: int = 256
    eval_seed: int = 12345
    eval_half_len: int = 32
    ckpt_every: int = 100
    out_dir: str = "runs/default"

    @property
    def n_ctx(self) -> int:
        return 2 * self.max_half_len

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})
