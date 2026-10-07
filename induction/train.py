"""Train the 2-layer attention-only transformer on repeated random tokens.

    python -m induction.train --config configs/default.yaml

Writes to cfg.out_dir:
    config.yaml           the exact config used
    metrics.csv           logged every `log_every` steps
    checkpoints/step_*.pt model weights every `ckpt_every` steps (plus step 0 and the last step)
    transition.json       transition step detected from metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import Config
from .data import half_masks, repeated_tokens
from .metrics import induction_score, prev_token_score
from .model import AttnOnlyTransformer, per_token_loss
from .transition import detect_transition


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_columns(cfg: Config) -> list[str]:
    cols = ["step", "train_loss", "eval_loss", "first_half_loss", "repeated_half_loss"]
    for kind in ("induction", "prev_token"):
        cols += [f"{kind}_L{l}H{h}" for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    return cols


def make_eval_tokens(cfg: Config) -> torch.Tensor:
    """Fixed held-out batch of repeated sequences (half length cfg.eval_half_len)."""
    return repeated_tokens(
        cfg.eval_batch_size, cfg.eval_half_len, cfg.vocab_size,
        generator=torch.Generator().manual_seed(cfg.eval_seed),
    )


@torch.no_grad()
def evaluate(model: AttnOnlyTransformer, tokens: torch.Tensor) -> dict[str, float]:
    cfg = model.cfg
    model.eval()
    logits, patterns = model(tokens, return_patterns=True)
    model.train()
    loss = per_token_loss(logits, tokens)  # [B, T-1]
    first, repeated = half_masks(tokens.shape[1] // 2)
    row = {
        "eval_loss": loss.mean().item(),
        "first_half_loss": loss[:, first].mean().item(),
        "repeated_half_loss": loss[:, repeated].mean().item(),
    }
    for l, pattern in enumerate(patterns):
        ind = induction_score(pattern, tokens)
        prev = prev_token_score(pattern)
        for h in range(cfg.n_heads):
            row[f"induction_L{l}H{h}"] = ind[h].item()
            row[f"prev_token_L{l}H{h}"] = prev[h].item()
    return row


def save_checkpoint(model: AttnOnlyTransformer, step: int, ckpt_dir: Path) -> None:
    torch.save(
        {"step": step, "config": model.cfg.to_dict(), "model": model.state_dict()},
        ckpt_dir / f"step_{step:06d}.pt",
    )


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    out = Path(cfg.out_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    model = AttnOnlyTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    data_gen = torch.Generator().manual_seed(cfg.seed)
    eval_tokens = make_eval_tokens(cfg)

    cols = metric_columns(cfg)
    f = open(out / "metrics.csv", "w", newline="")
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()

    def log(step: int, train_loss: float) -> dict:
        row = {"step": step, "train_loss": train_loss, **evaluate(model, eval_tokens)}
        writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
        f.flush()
        return row

    row = log(0, float("nan"))
    save_checkpoint(model, 0, ckpt_dir)
    t0 = time.time()
    for step in range(1, cfg.n_steps + 1):
        half_len = int(torch.randint(cfg.min_half_len, cfg.max_half_len + 1, (1,), generator=data_gen))
        tokens = repeated_tokens(cfg.batch_size, half_len, cfg.vocab_size, generator=data_gen)
        loss = per_token_loss(model(tokens), tokens).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.n_steps:
            row = log(step, loss.item())
            if step % (cfg.log_every * 25) == 0 or step == cfg.n_steps:
                ind = max(v for k, v in row.items() if k.startswith("induction_L1"))
                print(
                    f"step {step:6d} | train {row['train_loss']:.3f} | "
                    f"first-half {row['first_half_loss']:.3f} | "
                    f"repeated-half {row['repeated_half_loss']:.3f} | "
                    f"max L1 induction {ind:.3f} | {time.time() - t0:.0f}s"
                )
        if step % cfg.ckpt_every == 0 or step == cfg.n_steps:
            save_checkpoint(model, step, ckpt_dir)
    f.close()

    result = detect_transition(out / "metrics.csv", cfg)
    with open(out / "transition.json", "w") as fj:
        json.dump(result, fj, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(Config.from_yaml(args.config))


if __name__ == "__main__":
    main()
