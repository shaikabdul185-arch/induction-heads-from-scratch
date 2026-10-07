"""TransformerLens analysis of a finished run; writes all figures to figures/.

    python -m induction.analyze --run runs/default

Steps:
  1. Load every checkpoint into a TransformerLens HookedTransformer (attn-only, 2 layers)
     and check that its logits match the plain-PyTorch model.
  2. Attention heatmaps for every head on a repeated sequence, before and after the
     transition (patterns read from the HookedTransformer's activation cache).
  3. Induction score per head over training (dense log + TransformerLens checkpoints).
  4. Loss curves with the detected transition step marked.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import Config  # noqa: E402
from .data import half_masks, repeated_tokens  # noqa: E402
from .metrics import induction_score  # noqa: E402
from .model import AttnOnlyTransformer, per_token_loss  # noqa: E402
from .tl_convert import to_hooked_transformer  # noqa: E402
from .train import make_eval_tokens  # noqa: E402
from .transition import detect_transition, load_metrics  # noqa: E402

# Validated categorical slots (blue, orange, aqua, yellow) and a one-hue blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TRANSITION = "#52514e"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "font.size": 10,
    "lines.linewidth": 2,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def load_checkpoint(path: Path) -> AttnOnlyTransformer:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = AttnOnlyTransformer(Config.from_dict(ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def checkpoint_paths(run: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in sorted((run / "checkpoints").glob("step_*.pt"))}


def mark_transition(ax, tr: dict) -> None:
    rep = tr["repeated_half_loss"]["crossing_steps"]
    if rep["10pct"] is not None and rep["90pct"] is not None:
        ax.axvspan(rep["10pct"], rep["90pct"], color=GRID, alpha=0.6, lw=0, zorder=0)
    if tr["transition_step"] is not None:
        ax.axvline(tr["transition_step"], color=TRANSITION, ls="--", lw=1.2, zorder=1)


def plot_losses(m: dict, cfg: Config, tr: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    mark_transition(ax, tr)
    steps = m["step"]
    ax.plot(steps[1:], m["train_loss"][1:], color=SERIES[0], alpha=0.35, lw=1.2)
    curves = [
        ("train loss (all positions)", m["eval_loss"], SERIES[0]),
        ("first half (random tokens)", m["first_half_loss"], SERIES[1]),
        ("repeated half (copyable)", m["repeated_half_loss"], SERIES[2]),
    ]
    for label, y, c in curves:
        ax.plot(steps, y, color=c, label=label)
    ax.axhline(math.log(cfg.vocab_size), color=INK_2, lw=1, ls=":")
    ax.text(steps[-1], math.log(cfg.vocab_size) + 0.06, f"log V = {math.log(cfg.vocab_size):.2f}",
            ha="right", va="bottom", color=INK_2, fontsize=9)
    if tr["transition_step"] is not None:
        ax.text(tr["transition_step"], 0.3, f"  transition\n  step {tr['transition_step']}",
                color=INK, fontsize=9, va="bottom")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_ylim(0, None)
    ax.set_title("Loss on the repeated half collapses at the transition", loc="left")
    ax.legend(loc="lower left")
    fig.savefig(out)
    plt.close(fig)


def plot_induction_scores(m: dict, cfg: Config, tr: dict, tl_scores: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, cfg.n_layers, figsize=(10, 3.8), sharey=True)
    for l, ax in enumerate(np.atleast_1d(axes)):
        mark_transition(ax, tr)
        for h in range(cfg.n_heads):
            c = SERIES[h % len(SERIES)]
            ax.plot(m["step"], m[f"induction_L{l}H{h}"], color=c, label=f"head {h}")
            ax.scatter(tl_scores["steps"], tl_scores["scores"][:, l, h], s=12, color=c,
                       edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        ax.set_title(f"Layer {l}", loc="left")
        ax.set_xlabel("training step")
    axes[0].set_ylabel("induction score")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(loc="upper left", title="lines: training log\ndots: TransformerLens",
                    title_fontsize=8)
    fig.suptitle("Induction score per head over training", x=0.06, y=1.03, ha="left", color=INK)
    fig.savefig(out)
    plt.close(fig)


def plot_attention(tl_model, tokens: torch.Tensor, cfg: Config, step: int, title: str, out: Path):
    _, cache = tl_model.run_with_cache(tokens)
    L = tokens.shape[1] // 2
    fig, axes = plt.subplots(cfg.n_layers, cfg.n_heads, figsize=(3 * cfg.n_heads, 3.1 * cfg.n_layers))
    scores = []
    for l in range(cfg.n_layers):
        pattern = cache["pattern", l]  # [1, H, dst, src]
        ind = induction_score(pattern, tokens)
        for h in range(cfg.n_heads):
            ax = axes[l, h]
            ax.imshow(pattern[0, h].numpy(), cmap=BLUES, vmin=0, vmax=1, interpolation="nearest")
            ax.axhline(L - 0.5, color=SERIES[1], lw=0.6, alpha=0.6)
            ax.axvline(L - 0.5, color=SERIES[1], lw=0.6, alpha=0.6)
            ax.set_title(f"L{l}H{h}   induction {ind[h]:.2f}", fontsize=9, loc="left")
            ax.set_xticks([0, L, 2 * L - 1])
            ax.set_yticks([0, L, 2 * L - 1])
            ax.grid(False)
            ax.tick_params(labelsize=7)
            if h == 0:
                ax.set_ylabel("destination position")
            if l == cfg.n_layers - 1:
                ax.set_xlabel("source position")
            scores.append(ind[h].item())
    fig.suptitle(f"{title} (step {step}): attention on a repeated sequence, L = {L}", x=0.02,
                 y=1.01, ha="left", color=INK)
    fig.savefig(out)
    plt.close(fig)


@torch.no_grad()
def length_generalization(model: AttnOnlyTransformer, cfg: Config, n: int = 256) -> list[dict]:
    """Repeated-half loss and best induction score at several repeat lengths L.

    A real induction head works for any L; a positional shortcut only works for the L
    it was trained on.
    """
    rows = []
    gen = torch.Generator().manual_seed(cfg.eval_seed + 1)
    for L in sorted({8, 16, 24, cfg.max_half_len}):
        tokens = repeated_tokens(n, L, cfg.vocab_size, generator=gen)
        logits, patterns = model(tokens, return_patterns=True)
        _, repeated = half_masks(L)
        rows.append({
            "half_len": L,
            "repeated_half_loss": round(per_token_loss(logits, tokens)[:, repeated].mean().item(), 4),
            "max_induction_score": round(max(induction_score(p, tokens).max().item() for p in patterns), 4),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="runs/default")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--results", default="results",
                        help="small copies of the run's logs are saved to <results>/<run name>/")
    args = parser.parse_args()
    run, fig_dir = Path(args.run), Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(args.results) / run.name
    res_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_yaml(run / "config.yaml")
    m = load_metrics(run / "metrics.csv")
    tr = detect_transition(run / "metrics.csv", cfg)
    print(json.dumps(tr, indent=2))

    eval_tokens = make_eval_tokens(cfg)

    # 1 + 3: every checkpoint through TransformerLens; check logits match, collect scores
    ckpts = checkpoint_paths(run)
    tl_steps, tl_scores, max_diff = [], [], 0.0
    with torch.no_grad():
        for step, path in ckpts.items():
            model = load_checkpoint(path)
            tl_model = to_hooked_transformer(model)
            ours = model(eval_tokens)
            theirs, cache = tl_model.run_with_cache(eval_tokens)
            max_diff = max(max_diff, (ours - theirs).abs().max().item())
            tl_steps.append(step)
            tl_scores.append([induction_score(cache["pattern", l], eval_tokens).numpy()
                              for l in range(cfg.n_layers)])
    print(f"TransformerLens vs PyTorch: max |logit diff| over {len(ckpts)} checkpoints = {max_diff:.2e}")
    assert max_diff < 1e-4, "TransformerLens model does not match the PyTorch model"
    tl = {"steps": np.array(tl_steps), "scores": np.array(tl_scores)}

    plot_losses(m, cfg, tr, fig_dir / "loss_curves.png")
    plot_induction_scores(m, cfg, tr, tl, fig_dir / "induction_scores.png")

    # 2: before = last checkpoint before the loss starts dropping; after = final checkpoint
    onset = tr["repeated_half_loss"]["crossing_steps"]["10pct"] or max(ckpts)
    before = max(s for s in ckpts if s < onset)
    after = max(ckpts)
    seq = eval_tokens[:1]
    with torch.no_grad():
        for step, title, name in ((before, "Before the transition", "attention_before.png"),
                                  (after, "After the transition", "attention_after.png")):
            tl_model = to_hooked_transformer(load_checkpoint(ckpts[step]))
            plot_attention(tl_model, seq, cfg, step, title, fig_dir / name)

    gen = length_generalization(load_checkpoint(ckpts[after]), cfg)
    print("final model at other repeat lengths:", json.dumps(gen, indent=2))

    summary = {"transition": tr, "attention_before_step": before, "attention_after_step": after,
               "tl_max_abs_logit_diff": max_diff, "length_generalization": gen}
    for name in ("config.yaml", "metrics.csv"):
        shutil.copy(run / name, res_dir / name)
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote figures to {fig_dir}/ and logs to {res_dir}/ (before={before}, after={after})")


if __name__ == "__main__":
    main()
