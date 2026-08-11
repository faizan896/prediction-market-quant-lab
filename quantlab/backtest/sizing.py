"""Position sizing.

Binary contracts make Kelly exact rather than an approximation, which is one of
the genuinely nice things about this asset class.

Buy YES at price b with true probability p:
    win  +(1 - b) per contract, with probability p
    lose -b       per contract, with probability 1 - p

Kelly fraction of bankroll:
    f* = (p - b) / (1 - b)

Note f* depends only on the edge and the price. At b = 0.05 a 2-point edge is
f* = 0.021; at b = 0.95 the same 2-point edge is f* = 0.40. That asymmetry is
why unsized longshot strategies blow up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def kelly_fraction(p: np.ndarray, price: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
    """Exact Kelly for a binary contract, in YES terms.

    side=+1 buys YES at `price`; side=-1 buys NO at `1 - price`.
    Returns a non-negative fraction of bankroll (0 when there is no edge).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    b = np.clip(np.asarray(price, dtype=float), 1e-6, 1 - 1e-6)
    if side is None:
        side = np.where(p >= b, 1.0, -1.0)
    side = np.asarray(side, dtype=float)

    # For the NO leg, flip probability and price.
    p_eff = np.where(side > 0, p, 1 - p)
    b_eff = np.where(side > 0, b, 1 - b)
    f = (p_eff - b_eff) / (1 - b_eff)
    return np.clip(f, 0.0, 1.0)


@dataclass(slots=True)
class SizingConfig:
    """Kelly is the growth-optimal bet under a *correct* model. Your model is
    not correct, so never run full Kelly. Quarter-Kelly with a hard cap is the
    practical default; it gives ~87% of the growth at ~44% of the variance.
    """

    method: str = "kelly"  # "kelly" | "fixed_fraction" | "fixed_notional"
    kelly_multiplier: float = 0.25
    max_fraction: float = 0.05  # hard cap: never risk >5% of bankroll on one bet
    fixed_fraction: float = 0.02
    fixed_notional: float = 1_000.0
    max_notional: float | None = None
    # Liquidity guard: never take more than this share of resting depth.
    max_liquidity_share: float = 0.10
    min_notional: float = 10.0


def position_notional(
    cfg: SizingConfig,
    bankroll: float | np.ndarray,
    fair_prob: np.ndarray,
    price: np.ndarray,
    side: np.ndarray,
    liquidity: np.ndarray | None = None,
) -> np.ndarray:
    """Dollar notional to deploy per signal, after every cap."""
    bankroll = np.asarray(bankroll, dtype=float)

    if cfg.method == "kelly":
        f = kelly_fraction(fair_prob, price, side) * cfg.kelly_multiplier
    elif cfg.method == "fixed_fraction":
        f = np.full_like(np.asarray(price, dtype=float), cfg.fixed_fraction)
    else:
        f = np.asarray(cfg.fixed_notional, dtype=float) / np.where(bankroll > 0, bankroll, 1.0)

    f = np.minimum(f, cfg.max_fraction)
    notional = f * bankroll

    if cfg.max_notional is not None:
        notional = np.minimum(notional, cfg.max_notional)
    if liquidity is not None:
        liq_cap = np.asarray(liquidity, dtype=float) * cfg.max_liquidity_share
        notional = np.minimum(notional, np.where(liq_cap > 0, liq_cap, notional))

    return np.where(notional >= cfg.min_notional, notional, 0.0)
