"""Detect the induction-head phase transition from a metrics.csv log.

The transition is where two things happen together:
  1. repeated-half loss drops sharply from the no-context baseline log(vocab_size)
     toward its final value, and
  2. the most induction-like head's score jumps from ~0 toward its final value.

For each curve we lightly smooth it and report the first logged step at which it has
covered 10%, 50% and 90% of the way from its starting level to its final level. The
"transition step" is the 50% crossing of the repeated-half loss.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .config import Config


def load_metrics(path: str | Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; edges use the available values."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x), kernel, mode="same")
    return num / den


def crossing_step(steps: np.ndarray, y: np.ndarray, threshold: float, falling: bool) -> int | None:
    hit = y <= threshold if falling else y >= threshold
    idx = np.flatnonzero(hit)
    return int(steps[idx[0]]) if idx.size else None


def _progress_steps(steps, y, start, end) -> dict[str, int | None]:
    falling = end < start
    return {
        f"{int(p * 100)}pct": crossing_step(steps, y, start + p * (end - start), falling)
        for p in (0.1, 0.5, 0.9)
    }


def detect_transition(path: str | Path, cfg: Config, smooth_window: int = 5) -> dict:
    m = load_metrics(path)
    steps = m["step"]
    n_tail = max(3, len(steps) // 20)

    # 1. repeated-half loss: from log(V) (best possible without using context) to final
    rep = smooth(m["repeated_half_loss"], smooth_window)
    rep_start = math.log(cfg.vocab_size)
    rep_end = float(m["repeated_half_loss"][-n_tail:].mean())
    rep_cross = _progress_steps(steps, rep, rep_start, rep_end)
    drop_rate = np.gradient(rep, steps)
    steepest = int(steps[int(np.argmin(drop_rate))])

    # 2. induction score of the head that ends up most induction-like
    ind_cols = [k for k in m if k.startswith("induction_")]
    best = max(ind_cols, key=lambda k: m[k][-n_tail:].mean())
    ind = smooth(m[best], smooth_window)
    ind_start = float(m[best][:3].mean())
    ind_end = float(m[best][-n_tail:].mean())
    ind_cross = _progress_steps(steps, ind, ind_start, ind_end)

    # Both signals must move: the loss must fall by more than half, and the best head's
    # induction score must rise by at least 0.3. A loss drop without an induction head
    # means the model found some other (e.g. positional) solution.
    loss_dropped = (rep_start - rep_end) > 0.5 * rep_start and rep_cross["50pct"] is not None
    head_formed = (ind_end - ind_start) > 0.3 and ind_cross["50pct"] is not None
    found = loss_dropped and head_formed
    return {
        "transition_found": bool(found),
        "loss_dropped": bool(loss_dropped),
        "induction_head_formed": bool(head_formed),
        "transition_step": rep_cross["50pct"] if found else None,
        "repeated_half_loss": {
            "start (log V)": round(rep_start, 4),
            "final": round(rep_end, 4),
            "crossing_steps": rep_cross,
            "steepest_drop_step": steepest,
        },
        "induction_score": {
            "head": best.removeprefix("induction_"),
            "start": round(ind_start, 4),
            "final": round(ind_end, 4),
            "crossing_steps": ind_cross,
        },
        "first_half_loss_final": round(float(m["first_half_loss"][-n_tail:].mean()), 4),
        "final_induction_scores": {
            k.removeprefix("induction_"): round(float(m[k][-n_tail:].mean()), 4) for k in ind_cols
        },
    }
