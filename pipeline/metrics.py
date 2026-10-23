"""Error measures and the day-block bootstrap. All errors are in units of AC rating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import BOOTSTRAP_REPS, QUANTILES, SEED

BLOCK_DAYS = 7


def nmae(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(np.abs(f - y)))


def nrmse(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sqrt(np.mean((f - y) ** 2)))


def bias(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.mean(f - y))


def skill(y: np.ndarray, f: np.ndarray, ref: np.ndarray) -> float:
    """1 - RMSE(f) / RMSE(ref). Positive means better than the reference."""
    return 1.0 - nrmse(y, f) / nrmse(y, ref)


def pinball(y: np.ndarray, q: np.ndarray, levels: tuple[float, ...] = QUANTILES) -> float:
    """Mean pinball loss over the quantile levels (columns of q)."""
    losses = []
    for j, a in enumerate(levels):
        d = y - q[:, j]
        losses.append(np.mean(np.maximum(a * d, (a - 1) * d)))
    return float(np.mean(losses))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def day_blocks(days: pd.Series) -> np.ndarray:
    """Block id for each row: consecutive runs of BLOCK_DAYS calendar days."""
    d0 = days.min()
    return ((days - d0).dt.days // BLOCK_DAYS).to_numpy()


def bootstrap_skill(y: np.ndarray, f: np.ndarray, ref: np.ndarray, days: pd.Series,
                    reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> tuple[float, float]:
    """95% interval for the skill score, resampling whole 7-day blocks with replacement."""
    blocks = day_blocks(days)
    ids, inv = np.unique(blocks, return_inverse=True)
    se_f = np.bincount(inv, weights=(f - y) ** 2, minlength=len(ids))
    se_r = np.bincount(inv, weights=(ref - y) ** 2, minlength=len(ids))
    n = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(reps, len(ids)))
    nn = n[draws].sum(axis=1)
    s = 1.0 - np.sqrt(se_f[draws].sum(axis=1) / nn) / np.sqrt(se_r[draws].sum(axis=1) / nn)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, days: pd.Series,
                   reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> tuple[float, float]:
    """95% interval for RMSE(a) - RMSE(b), same block resampling."""
    blocks = day_blocks(days)
    ids, inv = np.unique(blocks, return_inverse=True)
    se_a = np.bincount(inv, weights=(a - y) ** 2, minlength=len(ids))
    se_b = np.bincount(inv, weights=(b - y) ** 2, minlength=len(ids))
    n = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(reps, len(ids)))
    nn = n[draws].sum(axis=1)
    d = np.sqrt(se_a[draws].sum(axis=1) / nn) - np.sqrt(se_b[draws].sum(axis=1) / nn)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)
